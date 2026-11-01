"""ECML/PKDD 2006 Task A, reversed transfer direction in DAM Table IV.

The challenge labels are +1=spam. DAM instead defines +1=non-spam,
so all labels are explicitly negated. Tuning files use a different word-ID
mapping and MUST NOT be pooled with the evaluation/training files.
"""
import io
import json
import zipfile
import numpy as np
from scipy import sparse
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import normalize
from .data import ROOT


def prepare():
    raw = ROOT/'data/email_spam/raw'
    out = ROOT/'data/email_spam/processed'
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(raw/'data_task_a.zip') as original, zipfile.ZipFile(raw/'evaluation_data_labeled.zip') as labeled:
        user_data=[]
        for i in range(3):
            label_bytes=labeled.read(f'task_a_lab/task_a_u0{i}_eval_lab.tf')
            unlabel_bytes=original.read(f'data_task_a/task_a_u0{i}_eval.tf')
            lab_lines=label_bytes.splitlines(); unl_lines=unlabel_bytes.splitlines()
            assert len(lab_lines)==len(unl_lines)==2500
            # Exact feature and row identity; only the released label differs.
            assert all(a.split(maxsplit=1)[1]==b.split(maxsplit=1)[1] for a,b in zip(lab_lines,unl_lines))
            user_data.append(label_bytes)
        public=original.read('data_task_a/task_a_labeled_train.tf')
    parts=load_svmlight_files([io.BytesIO(b) for b in user_data+[public]],zero_based=False)
    matrices=parts[::2]; labels=[-np.asarray(y,dtype=np.int8) for y in parts[1::2]]
    assert [X.shape[0] for X in matrices]==[2500,2500,2500,4000]
    assert all(set(np.unique(y))=={-1,1} and np.sum(y==1)==len(y)//2 for y in labels)
    Xraw=sparse.vstack(matrices,format='csr')
    X=normalize(Xraw,norm='l2').tocsr()
    y=np.concatenate(labels)
    domains=[np.arange(0,2500),np.arange(2500,5000),np.arange(5000,7500),np.arange(7500,11500)]
    sparse.save_npz(out/'X_wordcounts.npz',Xraw)
    sparse.save_npz(out/'X_L2.npz',X)
    np.savez_compressed(out/'indices.npz',y=y,**{f'domain_{i}':ix for i,ix in enumerate(domains)})
    meta={'dataset':'ECML/PKDD 2006 Task A, post-March-9 corrected release',
          'source_domains':['U00','U01','U02'],'target_domain':'public labeled train set',
          'domain_sizes':[2500,2500,2500,4000],'features':X.shape[1],'nnz':X.nnz,
          'positive_label':'non-spam (+1); original challenge labels negated',
          'preprocessing':'Original anonymous term frequencies, shared feature IDs, L2 row normalization, no IDF or feature selection. Paper does not specify normalization; retained as explicit local choice.',
          'tuning_data_used':False,'all_source_feature_rows_match_original_unlabeled_release':True}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)


def load():
    path=ROOT/'data/email_spam/processed'
    X=sparse.load_npz(path/'X_L2.npz')
    a=np.load(path/'indices.npz',allow_pickle=False)
    return X,a['y'],[a[f'domain_{i}'] for i in range(4)]


if __name__=='__main__':prepare()
