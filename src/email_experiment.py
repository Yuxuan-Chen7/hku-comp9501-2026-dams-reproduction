"""DAM Table VII: full ECML/PKDD Task A, 10 seeds, seven methods.

The public 4000-email set is target; U00/U01/U02 are labeled sources.
18 base classifiers use feature replication in each kernel feature space.
No hidden target labels are supplied to a classifier fit or model selection.
"""
import argparse
import csv
import json
import time
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from .data import ROOT, sample_target
from .email_data import load
from .dam import domain_weights, fit_dam
from .cache_contract import ensure_context, update_completed_seeds

DEGREES=[0,1.1,1.2,1.3,1.4,1.5]
PAPER={'SVM_T':(83.01,3.08),'SVM_S':(68.62,.09),'MCC_SVM':(76.89,2.75),
       'FR':(84.13,3.46),'Multi_KMM':(74.10,1.36),'FastDAM':(85.36,2.48),'UniverDAM':(87.05,2.94)}


def gram(A,B):return (A@B.T).toarray()


def feature_replication_kernel(Kss,Kts,Ktt,observed):
    """phi_s=[phi,phi,0], phi_t=[phi,0,phi]."""
    ns=Kss.shape[0]; nl=len(observed)
    train=np.empty((ns+nl,ns+nl),dtype=float)
    train[:ns,:ns]=2*Kss
    train[:ns,ns:]=Kts[observed].T
    train[ns:,:ns]=Kts[observed]
    train[ns:,ns:]=2*Ktt[np.ix_(observed,observed)]
    target=np.c_[Kts,2*Ktt[:,observed]]
    return train,target


def run(seeds, positive_class='nonspam'):
    folder=ROOT/('results/email_spam' if positive_class=='nonspam' else 'results/email_spam_challenge_labels')
    folder.mkdir(parents=True,exist_ok=True)
    runs=folder/'runs';runs.mkdir(exist_ok=True)
    cache=folder/'cache';cache.mkdir(exist_ok=True)
    X,y,domains=load()
    if positive_class=='spam': y=-y
    ensure_context(folder, 'email_'+positive_class, X, y, domains)
    Xt=X[domains[3]];yt=y[domains[3]]
    sources=[X[ix] for ix in domains[:3]]; ys=[y[ix] for ix in domains[:3]]
    Xs=sparse.vstack(sources,format='csr'); ysource=np.concatenate(ys)
    Ktt=gram(Xt,Xt)
    Kss=[gram(s,s) for s in sources];Kts=[gram(Xt,s) for s in sources]
    observed_by_seed={seed:sample_target(yt,10,seed) for seed in seeds}
    # Fixed source SVM baseline; source labels only.
    svm_scores=[]
    for kss,kts,y_s in zip(Kss,Kts,ys):
        model=SVC(C=1,kernel='precomputed',tol=1e-5).fit(kss,y_s)
        svm_scores.append(expit(model.decision_function(kts)))
    svm_s=np.mean(svm_scores,axis=0)
    # Build each seed's FR bank. The 18 models all see the SAME 20 target labels.
    banks={seed:np.empty((len(yt),18)) for seed in seeds if not (cache/f'fr_seed{seed:02d}.npz').exists()}
    distances=[]
    for source_id,(kss,kts,y_s) in enumerate(zip(Kss,Kts,ys)):
        for ki,degree in enumerate(DEGREES):
            a=kss if degree==0 else (kss+1)**degree
            b=kts if degree==0 else (kts+1)**degree
            c=Ktt if degree==0 else (Ktt+1)**degree
            distances.append(max(float(a.mean()+c.mean()-2*b.mean()),0.))
            for seed,scores in banks.items():
                observed=observed_by_seed[seed]
                train,test=feature_replication_kernel(a,b,c,observed)
                model=SVC(C=1,kernel='precomputed',tol=1e-5,cache_size=256).fit(train,np.r_[y_s,yt[observed]])
                scores[:,source_id*6+ki]=model.decision_function(test)
                if model.fit_status_!=0: raise RuntimeError('FR SVM failed to converge')
                del train,test,model
            print('Email FR bank source',source_id+1,'kernel',degree or 'linear','done',flush=True)
            del a,b,c
    for seed,scores in banks.items(): np.savez_compressed(cache/f'fr_seed{seed:02d}.npz',scores=scores)
    del banks
    weights=domain_weights(distances,beta=100)
    (folder/'protocol.json').write_text(json.dumps({
        'requested_seeds_last_invocation':list(seeds),'m_per_class':10,'C':1,'epsilon':.1,
        'lambda_L':1,'lambda_D':1,'lambda_U':1,'beta':100,
        'mmd_convention':'paper_literal','target_kernel':'linear',
        'base_classifiers':'18 FR: 3 sources x (linear, polynomial degrees 1.1 through 1.5)',
        'FR_interpretation':'feature replication in kernel feature space: same-domain 2*K, cross-domain K',
        'preprocessing':'Original term counts with L2 normalization; no tuning data, no feature selection',
        'mmd_squared':distances,'weights':weights.tolist(),
        'evaluation':f'Noninterpolated AP, {positive_class} positive, all 3980 unlabeled target samples',
        'positive_class':positive_class,
        'label_convention_note':'Paper text uses nonspam positive; challenge source files use spam positive. Both conventions are reported without selecting the better test result.',
        'device':'CPU, scikit-learn LIBSVM and SciPy L-BFGS'},indent=2),encoding='utf-8')
    # Multi-KMM's paper-described mean translation (alpha=1), not generic KMM QP.
    means=np.asarray([np.asarray(s.mean(axis=0)).ravel() for s in sources])
    shifts=np.asarray(Xt.mean(axis=0)).ravel()[None,:]-means
    boundaries=np.r_[0,np.cumsum([s.shape[0] for s in sources])]
    ss= Xs@shifts.T; ts=Xt@shifts.T;sg=shifts@shifts.T
    shifted_ss=gram(Xs,Xs);shifted_ts=gram(Xt,Xs)
    for i in range(3):
        lo,hi=boundaries[i:i+2]
        shifted_ss[:,lo:hi]+=ss[:,i,None]
        shifted_ss[lo:hi,:]+=ss[:,i][None,:]
        shifted_ts[:,lo:hi]+=ts[:,i,None]
        for j in range(3):
            l2,h2=boundaries[j:j+2];shifted_ss[lo:hi,l2:h2]+=sg[i,j]
    for seed in seeds:
        dest=runs/f'seed{seed:02d}.json'
        if dest.exists() and dest.with_suffix(".npz").exists():continue
        observed=observed_by_seed[seed];test=np.setdiff1d(np.arange(len(yt)),observed)
        scores=np.load(cache/f'fr_seed{seed:02d}.npz',allow_pickle=False)['scores']
        preds={'SVM_S':svm_s[test],'FR':np.mean(expit(scores[:,[0,6,12]]),axis=1)[test]}
        model=SVC(C=1,kernel='linear',tol=1e-5).fit(Xt[observed],yt[observed])
        preds['SVM_T']=model.decision_function(Xt[test])
        preds['MCC_SVM']=(3*preds['SVM_S']+expit(preds['SVM_T']))/4
        ns=Xs.shape[0];nl=len(observed)
        train=np.empty((ns+nl,ns+nl))
        train[:ns,:ns]=shifted_ss;train[:ns,ns:]=shifted_ts[observed].T
        train[ns:,:ns]=shifted_ts[observed];train[ns:,ns:]=Ktt[np.ix_(observed,observed)]
        model=SVC(C=1,kernel='precomputed',tol=1e-5,cache_size=512).fit(train,np.r_[ysource,yt[observed]])
        preds['Multi_KMM']=model.decision_function(np.c_[shifted_ts[test],Ktt[np.ix_(test,observed)]])
        assert model.fit_status_==0
        del train,model
        diagnostics={}
        for method,universum in [('FastDAM',False),('UniverDAM',True)]:
            begin=time.perf_counter()
            model=fit_dam(Xt,observed,yt[observed],scores,weights,Xs if universum else None)
            preds[method]=model.decision_function(Xt[test])
            diagnostics[method]={'fit_predict_seconds':time.perf_counter()-begin,
                                 'gradient_inf':model.gradient_inf_,'converged':model.converged_,
                                 'retained_unlabeled':len(model.retained_unlabeled_),
                                 'n_support':len(model.support_)}
            if seed==0:
                np.savez_compressed(folder/f'{method}_seed0_model.npz',coef=model.coef_,intercept=model.intercept_,observed=observed)
        rows=[{'task':'email_spam','seed':seed,'m':10,'method':method,
               'ap_percent':float(100*average_precision_score(yt[test]==1,preds[method])),
               'n_test':len(test),**diagnostics.get(method,{})} for method in PAPER]
        np.savez_compressed(dest.with_suffix('.npz'),test_indices=test,y_test=yt[test],**preds)
        dest.write_text(json.dumps({'observed_indices':observed.tolist(),'test_indices':test.tolist(),'rows':rows},indent=2),encoding='utf-8')
        print('Email seed',seed,', '.join(f"{r['method']}={r['ap_percent']:.2f}" for r in rows),flush=True)
    summarize(folder)
    update_completed_seeds(folder, seeds)


def summarize(folder):
    rows=[];max_error=0
    for path in sorted((folder/'runs').glob('seed*.json')):
        payload=json.loads(path.read_text(encoding='utf-8'));pred=np.load(path.with_suffix('.npz'),allow_pickle=False)
        assert len(payload['observed_indices'])==20
        assert not np.intersect1d(payload['observed_indices'],payload['test_indices']).size
        assert len(pred['y_test'])==3980
        for row in payload['rows']:
            recomputed=100*average_precision_score(pred['y_test']==1,pred[row['method']])
            max_error=max(max_error,abs(recomputed-row['ap_percent']))
        rows.extend(payload['rows'])
    summary=[]
    for method,(pmean,psd) in PAPER.items():
        vals=[r['ap_percent'] for r in rows if r['method']==method]
        summary.append({'method':method,'n_runs':len(vals),'mean_ap_percent':float(np.mean(vals)),
                        'std_ap_percent':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.,
                        'paper_mean':pmean,'paper_std':psd,'difference_pp':float(np.mean(vals))-pmean})
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (folder/'summary.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    (folder/'verification.json').write_text(json.dumps({'runs':len(rows)//7,'method_evaluations':len(rows),
        'max_ap_recompute_error':max_error,
        'scope':'Saved-score consistency only; this check does not independently establish absence of label leakage',
        'max_dam_gradient_inf':max(r.get('gradient_inf',0) for r in rows)},indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    parser.add_argument('--positive-class',choices=['nonspam','spam'],default='nonspam')
    args=parser.parse_args()
    with threadpool_limits(limits=2):run(args.seeds,args.positive_class)
