"""Full-run audit: raw archives, independent AP, bank retraining and seed-0 model checks.
Requires the complete default experiment grid. Run python -m src.audit_full.
"""
import hashlib
import io
import json
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit, softmax
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits

from .data import ROOT
(ROOT/"results").mkdir(parents=True, exist_ok=True)
from src.dam import fit_dam

TASKS = {
    'rec_vs_sci': [('rec.autos','sci.crypt'),('rec.motorcycles','sci.electronics'),('rec.sport.baseball','sci.med'),('rec.sport.hockey','sci.space')],
    'comp_vs_rec': [('comp.graphics','rec.autos'),('comp.os.ms-windows.misc','rec.motorcycles'),('comp.sys.ibm.pc.hardware','rec.sport.baseball'),('comp.sys.mac.hardware','rec.sport.hockey')],
    'sci_vs_comp': [('sci.crypt','comp.graphics'),('sci.electronics','comp.os.ms-windows.misc'),('sci.med','comp.sys.ibm.pc.hardware'),('sci.space','comp.sys.mac.hardware')],
}
DEGREES = [0,1.1,1.2,1.3,1.4,1.5]
report = {'raw_data': {}, 'saved_results': {}, 'source_banks': {}, 'saved_models': {}, 'refits': {}}

def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def log(message):
    print(message, flush=True)
    (ROOT/'results/independent_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def equal_sparse(A,B):
    assert A.shape==B.shape
    difference=A-B
    error=float(np.max(np.abs(difference.data),initial=0))
    assert error < 1e-14, error
    return error

def independent_ap(y, score):
    # Average precision at every distinct score, handling ties as one group.
    order=np.argsort(-score,kind='stable')
    positives=(np.asarray(y)[order]==1).astype(int)
    sorted_scores=np.asarray(score)[order]
    endpoints=np.r_[np.flatnonzero(sorted_scores[:-1]!=sorted_scores[1:]),len(order)-1]
    hits=np.cumsum(positives)[endpoints]
    increments=np.diff(np.r_[0,hits])
    return float(100*np.sum((hits/(endpoints+1))*increments)/positives.sum())

def independent_split(y,m,seed):
    rng=np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.permutation(np.flatnonzero(y==1))[:m],rng.permutation(np.flatnonzero(y==-1))[:m]]))

def audit_runs(base, datasets, email=False):
    values=defaultdict(list); seen=set(); max_error=0.; max_gradient=0.; files=0
    for folder in (['runs'] if email else ['runs','extended_runs','ensemble_runs']):
        paths=sorted((base/folder).glob('*.json'))
        assert len(paths)==(10 if email else 210)
        for path in paths:
            item=load_json(path)
            task='email_spam' if email else item['task']
            seed=int(path.stem[-2:]) if email else item['seed']
            m=10 if email else item['m']
            X,y,domains=datasets[task]; yt=y[domains[-1]]
            observed=np.asarray(item['observed_indices' if email else 'labeled_target_local_indices'],dtype=int)
            test=np.asarray(item['test_indices' if email else 'test_target_local_indices'],dtype=int)
            np.testing.assert_array_equal(observed,independent_split(yt,m,seed))
            np.testing.assert_array_equal(test,np.setdiff1d(np.arange(len(yt)),observed))
            assert len(observed)==2*m and np.sum(yt[observed]==1)==m
            with np.load(path.with_suffix('.npz')) as pred:
                np.testing.assert_array_equal(pred['test_indices'],test)
                np.testing.assert_array_equal(pred['y_test'],yt[test])
                for row in item['rows']:
                    method=row['method']; identity=(task,m,seed,method)
                    assert identity not in seen
                    seen.add(identity)
                    assert row['seed']==seed and row['m']==m and row['task']==task
                    assert row['n_test']==len(test)
                    score=pred[method]
                    assert score.shape==(len(test),) and np.isfinite(score).all()
                    ap=independent_ap(yt[test],score)
                    max_error=max(max_error,abs(ap-row['ap_percent']))
                    values[(task,m,method)].append(ap)
                    if 'gradient_inf' in row:
                        assert row['converged'] and row['gradient_inf']<1e-4
                        max_gradient=max(max_gradient,row['gradient_inf'])
            files+=1
    summary=load_json(base/'summary.json')
    assert len(values)==len(summary)
    for row in summary:
        key=('email_spam',10,row['method']) if email else (row['task'],row['m'],row['method'])
        array=np.asarray(values[key])
        assert len(array)==row['n_runs']==10
        assert abs(array.mean()-row['mean_ap_percent'])<1e-10
        assert abs(array.std(ddof=1)-row['std_ap_percent'])<1e-10
    assert max_error<1e-10
    report['saved_results'][base.name]={'files':files,'method_evaluations':len(seen),'summary_rows':len(summary),'manual_ap_max_abs_error_pp':max_error,'recorded_max_gradient_inf':max_gradient}

def model_stationarity(X,y,domains,bank,base,email=False):
    Xt=X[domains[-1]]; yt=y[domains[-1]]; Xs=X[np.concatenate(domains[:3])]
    observed=independent_split(yt,10,0)
    test=np.setdiff1d(np.arange(len(yt)),observed)
    weights=softmax(-(100 if email else 10000)*bank['mmd2']**2)
    virtual=bank['scores']@weights
    retained=np.setdiff1d(np.flatnonzero(np.abs(virtual)>.3),observed)
    positions=np.r_[observed,retained]
    targets=np.r_[yt[observed],virtual[retained]]
    task='email_spam' if email else base.name
    for method in ['FastDAM','UniverDAM']:
        filename=base/f'{method}_seed0_model.npz' if email else ROOT/f'results/20ng_matlab/models/{task}_{method}_m10_seed0.npz'
        saved=np.load(filename)
        predfile=base/'runs/seed00.npz' if email else ROOT/f'results/20ng_matlab/runs/{task}_m10_seed00.npz'
        pred=np.load(predfile)
        w=saved['coef']; b=float(saved['intercept'])
        Z=Xt[positions]; labels=targets
        if method=='UniverDAM':
            Z=sparse.vstack([Z,Xs],format='csr'); labels=np.r_[labels,np.zeros(len(domains[0])+len(domains[1])+len(domains[2]))]
        resid=Z@w+b-labels
        gradloss=np.sign(resid)*np.minimum(np.maximum(np.abs(resid)-.1,0),1.)
        gradient=float(max(np.max(np.abs(w+Z.T@gradloss)),abs(gradloss.sum())))
        prediction_error=float(np.max(np.abs(Xt[test]@w+b-pred[method])))
        assert gradient<1e-4 and prediction_error<1e-12
        report['saved_models'][f'{base.name}/{method}']={'independently_recomputed_gradient_inf':gradient,'prediction_error':prediction_error,'retained_unlabeled':len(retained)}
    return Xt,yt,Xs,observed,test,weights

def main():
    # Reconstruct the release rather than trusting processed files or prepare().
    archive=ROOT/'data/20ng_matlab/raw/20news-bydate-matlab.tgz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='086fdc442ae2c269c53c6383f641ff92d55cb2b77da19d2cf3562473fc73dd19'
    with tarfile.open(archive) as tf:
        raw=[]; categories=[]
        for split in ['train','test']:
            lines=np.loadtxt(tf.extractfile(f'20news-bydate/matlab/{split}.data'),dtype=np.int64)
            cat=np.loadtxt(tf.extractfile(f'20news-bydate/matlab/{split}.label'),dtype=int)
            raw.append(sparse.coo_matrix((lines[:,2],(lines[:,0]-1,lines[:,1]-1)),shape=(len(cat),61188)).tocsr())
            categories.append(cat)
        mapping=dict((row.split()[0],int(row.split()[1])) for row in tf.extractfile('20news-bydate/matlab/train.map').read().decode().splitlines())
    full=sparse.vstack(raw,format='csr'); category=np.concatenate(categories)
    equal_sparse(full,sparse.load_npz(ROOT/'data/20ng_matlab/processed/all_wordcounts.npz'))
    assert full.shape==(18774,61188)
    datasets={}
    for task,pairs in TASKS.items():
        chosen=np.flatnonzero(np.isin(category,[mapping[c] for p in pairs for c in p]))
        local=full[chosen]; cats=category[chosen]
        y=np.where(np.isin(cats,[mapping[p[0]] for p in pairs]),1,-1)
        domains=[np.flatnonzero(np.isin(cats,[mapping[c] for c in pair])) for pair in pairs]
        assert len(np.unique(np.concatenate(domains)))==len(chosen)
        equal_sparse(local,sparse.load_npz(ROOT/f'data/20ng_matlab/processed/{task}_counts.npz'))
        indices=np.load(ROOT/f'data/20ng_matlab/processed/{task}_indices.npz')
        np.testing.assert_array_equal(indices['original_row'],chosen)
        np.testing.assert_array_equal(indices['y'],y)
        for i,d in enumerate(domains): np.testing.assert_array_equal(indices[f'domain_{i}'],d)
        datasets[task]=(normalize(local.astype(float),norm='l2'),y,domains)
    report['raw_data']['20ng']={'shape':list(full.shape),'processed_values_exact':True,'category_labels_and_indices_exact':True,'task_sizes':{task:[len(d) for d in value[2]] for task,value in datasets.items()}}
    audit_runs(ROOT/'results/20ng_matlab',datasets)
    log('20NG raw reconstruction and all saved predictions/APs/summaries passed')
    for task,(X,y,domains) in datasets.items():
        bank=np.load(ROOT/f'results/20ng_matlab/cache/{task}_source_bank.npz')
        Xt=X[domains[3]]; Ktt=(Xt@Xt.T).toarray(); errors=[]; distances=[]; linear=[]
        for source_id,ix in enumerate(domains[:3]):
            S=X[ix]; Kss=(S@S.T).toarray(); Kts=(Xt@S.T).toarray()
            for ki,degree in enumerate(DEGREES):
                a=Kss if degree==0 else (1+Kss)**degree
                b=Kts if degree==0 else (1+Kts)**degree
                c=Ktt if degree==0 else (1+Ktt)**degree
                model=SVC(C=1,kernel='precomputed',tol=1e-5,cache_size=256).fit(a,y[ix])
                score=model.decision_function(b)
                assert model.fit_status_==0
                errors.append(float(np.max(np.abs(score-bank['scores'][:,source_id*6+ki]))))
                distances.append(max(float(a.mean()+c.mean()-2*b.mean()),0.))
                if degree==0: linear.append(expit(score))
        assert max(errors)<1e-10
        np.testing.assert_allclose(distances,bank['mmd2'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(np.mean(linear,axis=0),bank['svm_s'],atol=1e-12,rtol=0)
        report['source_banks'][task]={'refitted_models':18,'max_prediction_error':max(errors),'mmd_max_error':float(np.max(np.abs(distances-bank['mmd2'])))}
        Xt,yt,Xs,observed,test,weights=model_stationarity(X,y,domains,bank,ROOT/task)
        if task!='rec_vs_sci':
            stored=np.load(ROOT/f'results/20ng_matlab/runs/{task}_m10_seed00.npz')
            for method in ['FastDAM','UniverDAM']:
                model=fit_dam(Xt,observed,yt[observed],bank['scores'],weights,Xs if method=='UniverDAM' else None)
                error=float(np.max(np.abs(model.decision_function(Xt[test])-stored[method])))
                assert error<1e-10
                report['refits'][f'{task}/{method}']={'max_prediction_error':error}
        log(f'{task}: all 18 source scores exactly rebuilt; saved target models independently stationary')
    # Independent text parser for the email releases.
    rows=[]; cols=[]; vals=[]; labels=[]; rowno=0; identities=[]
    with zipfile.ZipFile(ROOT/'data/email_spam/raw/data_task_a.zip') as original, zipfile.ZipFile(ROOT/'data/email_spam/raw/evaluation_data_labeled.zip') as labeled:
        assert original.testzip() is None and labeled.testzip() is None
        blocks=[]
        for i in range(3):
            data=labeled.read(f'task_a_lab/task_a_u0{i}_eval_lab.tf')
            unlabeled=original.read(f'data_task_a/task_a_u0{i}_eval.tf')
            assert [x.split(maxsplit=1)[1] for x in data.splitlines()]==[x.split(maxsplit=1)[1] for x in unlabeled.splitlines()]
            blocks.append(data)
        blocks.append(original.read('data_task_a/task_a_labeled_train.tf'))
        for block in blocks:
            for line in block.splitlines():
                tokens=line.split(); labels.append(-int(tokens[0]))
                for token in tokens[1:]:
                    feature,count=token.split(b':'); rows.append(rowno); cols.append(int(feature)-1); vals.append(float(count))
                rowno+=1
    emailraw=sparse.coo_matrix((vals,(rows,cols)),shape=(rowno,max(cols)+1)).tocsr()
    X=normalize(emailraw,norm='l2').tocsr(); y=np.asarray(labels)
    domains=[np.arange(0,2500),np.arange(2500,5000),np.arange(5000,7500),np.arange(7500,11500)]
    equal_sparse(emailraw,sparse.load_npz(ROOT/'data/email_spam/processed/X_wordcounts.npz'))
    equal_sparse(X,sparse.load_npz(ROOT/'data/email_spam/processed/X_L2.npz'))
    indices=np.load(ROOT/'data/email_spam/processed/indices.npz')
    np.testing.assert_array_equal(indices['y'],y)
    for i,d in enumerate(domains): np.testing.assert_array_equal(indices[f'domain_{i}'],d)
    report['raw_data']['email']={'shape':list(X.shape),'processed_values_exact':True,'source_rows_match_original_unlabeled':True,'labels_correctly_negated':True,'zip_crc_ok':True}
    for folder,sign in [('email_spam',1),('email_spam_challenge_labels',-1)]:
        base=ROOT/'results'/folder; ys=sign*y
        audit_runs(base,{'email_spam':(X,ys,domains)},email=True)
        protocol=load_json(base/'protocol.json')
        bank={'scores':np.load(base/'cache/fr_seed00.npz')['scores'],'mmd2':np.asarray(protocol['mmd_squared'])}
        Xt,yt,Xs,observed,test,weights=model_stationarity(X,ys,domains,bank,base,email=True)
        stored=np.load(base/'runs/seed00.npz')
        for method in ['FastDAM','UniverDAM']:
            model=fit_dam(Xt,observed,yt[observed],bank['scores'],weights,Xs if method=='UniverDAM' else None)
            error=float(np.max(np.abs(model.decision_function(Xt[test])-stored[method])))
            assert error<1e-10
            report['refits'][f'{folder}/{method}']={'max_prediction_error':error}
        Ktt=(Xt@Xt.T).toarray(); errors=[]; distances=[]
        for source_id,ix in enumerate(domains[:3]):
            S=X[ix]; ns=len(ix); Kss=(S@S.T).toarray(); Kts=(Xt@S.T).toarray()
            for ki,degree in enumerate(DEGREES):
                a=Kss if degree==0 else (1+Kss)**degree
                b=Kts if degree==0 else (1+Kts)**degree
                c=Ktt if degree==0 else (1+Ktt)**degree
                train=np.block([[2*a,b[observed].T],[b[observed],2*c[np.ix_(observed,observed)]]])
                cross=np.hstack([b,2*c[:,observed]])
                model=SVC(C=1,kernel='precomputed',tol=1e-5,cache_size=256).fit(train,np.r_[ys[ix],yt[observed]])
                assert model.fit_status_==0
                score=model.decision_function(cross)
                errors.append(float(np.max(np.abs(score-bank['scores'][:,source_id*6+ki]))))
                distances.append(max(float(a.mean()+c.mean()-2*b.mean()),0.))
        assert max(errors)<1e-10
        np.testing.assert_allclose(distances,bank['mmd2'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(weights,protocol['weights'],atol=1e-14,rtol=0)
        report['source_banks'][folder]={'seed':0,'refitted_models':18,'max_prediction_error':max(errors),'mmd_max_error':float(np.max(np.abs(distances-bank['mmd2'])))}
        log(f'{folder}: AP/summary, all seed-0 FR scores and both target refits passed')
    report['status']='passed'
    log('Independent audit checks completed')

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
