import json
import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import average_precision_score
from src.cache_contract import ensure_context, update_completed_seeds
from src.verify_results import manual_ap


def test_cache_rejects_changed_data_or_source(tmp_path, monkeypatch):
    X = sparse.csr_matrix([[1., 0.], [0., 2.]])
    y = np.array([1, -1])
    domains = [np.array([0]), np.array([1])]
    ensure_context(tmp_path, 'test', X, y, domains)
    ensure_context(tmp_path, 'test', X, y, domains)
    with pytest.raises(RuntimeError, match='Cache context changed'):
        ensure_context(tmp_path, 'test', X*2, y, domains)
    monkeypatch.setattr('src.cache_contract.code_hash', lambda: 'changed-source')
    with pytest.raises(RuntimeError, match='Cache context changed'):
        ensure_context(tmp_path, 'test', X, y, domains)


def test_cache_rejects_unversioned_predictions(tmp_path):
    (tmp_path/'runs').mkdir()
    (tmp_path/'runs/test_m10_seed00.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Unversioned'):
        ensure_context(tmp_path, 'test', sparse.eye(2), np.array([1, -1]), [np.arange(2)])


def test_partial_resume_records_all_completed_seeds(tmp_path):
    (tmp_path/'runs').mkdir()
    (tmp_path/'protocol.json').write_text('{}')
    for seed in [0, 1, 2]:
        path = tmp_path/'runs'/f'seed{seed:02d}.json'
        path.write_text(json.dumps({'rows': [{'seed': seed}]}))
        np.savez(path.with_suffix('.npz'), dummy=[0])
    update_completed_seeds(tmp_path, [0])
    protocol = json.loads((tmp_path/'protocol.json').read_text())
    assert protocol['seeds'] == [0, 1, 2]
    assert protocol['requested_seeds_last_invocation'] == [0]


def test_manual_ap_handles_tied_scores():
    y = np.array([1, -1, 1, 1, -1, -1])
    scores = np.array([.8, .8, .5, .1, .1, .1])
    assert abs(manual_ap(y, scores)-100*average_precision_score(y == 1, scores)) < 1e-12
