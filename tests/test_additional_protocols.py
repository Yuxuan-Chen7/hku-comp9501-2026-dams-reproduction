"""Independent checks for label semantics and FR feature-space construction."""
import pytest
import io
import json
import zipfile
import numpy as np
from sklearn.datasets import load_svmlight_file
from src.data import ROOT
from src.email_data import load
from src.email_experiment import feature_replication_kernel


def test_email_source_release_row_identity_and_label_polarity():
    if not (ROOT/'data/email_spam/processed/indices.npz').exists():
        pytest.skip('Run reproduce.py --stage prepare first for dataset integration checks')
    _,y,domains=load()
    paths=ROOT/'data/email_spam/raw'
    with zipfile.ZipFile(paths/'evaluation_data_labeled.zip') as z:
        for i in range(3):
            _,original=load_svmlight_file(io.BytesIO(z.read(f'task_a_lab/task_a_u0{i}_eval_lab.tf')))
            np.testing.assert_array_equal(y[domains[i]],-original)
    with zipfile.ZipFile(paths/'data_task_a.zip') as z:
        _,original=load_svmlight_file(io.BytesIO(z.read('data_task_a/task_a_labeled_train.tf')))
        np.testing.assert_array_equal(y[domains[3]],-original)
    assert [len(d) for d in domains]==[2500,2500,2500,4000]
    assert all(np.sum(y[d]==1)==len(d)//2 for d in domains)


def test_fr_kernel_matches_explicit_augmented_features():
    rng=np.random.default_rng(12)
    A=rng.normal(size=(7,5));T=rng.normal(size=(9,5));observed=np.array([1,4])
    train,test=feature_replication_kernel(A@A.T,T@A.T,T@T.T,observed)
    source=np.c_[A,A,np.zeros_like(A)]
    target=np.c_[T,np.zeros_like(T),T]
    fit=np.r_[source,target[observed]]
    np.testing.assert_allclose(train,fit@fit.T,atol=1e-12)
    np.testing.assert_allclose(test,target@fit.T,atol=1e-12)


def test_matlab_release_domain_counts():
    path=ROOT/'data/20ng_matlab/processed/metadata.json'
    if not path.exists():
        pytest.skip('Run reproduce.py --stage prepare first for dataset integration checks')
    rows=json.loads(path.read_text(encoding='utf-8'))
    expected={'rec_vs_sci':[1976,1977,1978,1982],
              'comp_vs_rec':[1957,1956,1970,1955],
              'sci_vs_comp':[1959,1947,1966,1943]}
    for row in rows:
        assert row['domain_counts']==expected[row['task']]
        assert row['features']==61188
