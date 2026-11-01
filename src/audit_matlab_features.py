"""Prepare the original 18,774-document, 61,188-feature count-matrix release."""
import csv
import json
import tarfile
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .data import ROOT,TASKS,sample_target


def prepare():
    base=ROOT/'data/20ng_matlab'
    out=base/'processed';out.mkdir(parents=True,exist_ok=True)
    with tarfile.open(base/'raw/20news-bydate-matlab.tgz','r:gz') as tf:
        arrays=[np.loadtxt(tf.extractfile(f'20news-bydate/matlab/{s}.data'),dtype=np.int32) for s in ['train','test']]
        labels=[np.loadtxt(tf.extractfile(f'20news-bydate/matlab/{s}.label'),dtype=np.int32) for s in ['train','test']]
        mapping={line.split()[0]:int(line.split()[1]) for line in tf.extractfile('20news-bydate/matlab/train.map').read().decode().splitlines()}
    nf=max(a[:,1].max() for a in arrays)
    matrices=[sparse.csr_matrix((a[:,2].astype(float),(a[:,0]-1,a[:,1]-1)),shape=(len(y),nf)) for a,y in zip(arrays,labels)]
    X=sparse.vstack(matrices,format='csr');category=np.concatenate(labels)
    assert X.shape==(18774,61188)
    sparse.save_npz(out/'all_wordcounts.npz',X)
    records=[]
    for task,pairs in TASKS.items():
        allcats=[mapping[c] for p in pairs for c in p]
        chosen=np.flatnonzero(np.isin(category,allcats))
        c=category[chosen];local=X[chosen]
        domains=[np.flatnonzero(np.isin(c,[mapping[p[0]],mapping[p[1]]])) for p in pairs]
        y=np.where(np.isin(c,[mapping[p[0]] for p in pairs]),1,-1)
        sparse.save_npz(out/f'{task}_counts.npz',local)
        np.savez_compressed(out/f'{task}_indices.npz',y=y,original_row=chosen,**{f'domain_{i}':ix for i,ix in enumerate(domains)})
        records.append({'task':task,'domain_counts':[len(d) for d in domains],'target':pairs[3],
                        'features':local.shape[1],'nnz':local.nnz})
    (out/'metadata.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    return records


def load_task(name):
    base=ROOT/'data/20ng_matlab/processed'
    X=normalize(sparse.load_npz(base/f'{name}_counts.npz'),norm='l2').tocsr()
    ix=np.load(base/f'{name}_indices.npz',allow_pickle=False)
    return X,ix['y'],[ix[f'domain_{i}'] for i in range(4)]


if __name__=='__main__':
    prepare()
