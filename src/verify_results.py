"""Independent AP arithmetic and saved-result consistency, for partial or full runs.

This verifies predictions against prepared labels. It does NOT retrain models
or independently establish correct preprocessing/no leakage. audit_full adds
raw-archive reconstruction and representative fresh model fits.
"""
import json
from collections import defaultdict
import numpy as np
from .data import load_task, sample_target
from .email_data import load as load_email


def manual_ap(y, scores):
    order = np.argsort(-scores, kind='stable')
    positives = (np.asarray(y)[order] == 1).astype(int)
    ordered = np.asarray(scores)[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(order)-1]
    hits = np.cumsum(positives)[ends]
    return float(100*np.sum(hits/(ends+1)*np.diff(np.r_[0, hits]))/positives.sum())


def verify(root, datasets):
    names = []
    if '20ng' in datasets:
        names.append('20ng_matlab')
    if 'email' in datasets:
        names += ['email_spam', 'email_spam_challenge_labels']
    report = {'scope': 'Saved-score, split and summary consistency; not independent model retraining', 'datasets': {}}
    for name in names:
        base = root/'results'/name
        email = name.startswith('email')
        paths = sorted((base/'runs').glob('*.json'))
        if not email:
            paths += sorted((base/'extended_runs').glob('*.json')) + sorted((base/'ensemble_runs').glob('*.json'))
        if not paths:
            raise RuntimeError(f'No results found in {base}')
        values = defaultdict(list)
        seen = set()
        loaded = {}
        max_error = 0.
        for path in paths:
            payload = json.loads(path.read_text(encoding='utf-8'))
            rows = payload['rows']
            task, m, seed = (rows[0][k] for k in ['task', 'm', 'seed'])
            if task not in loaded:
                _, y, domains = load_email() if email else load_task(task)
                loaded[task] = (-y if name.endswith('challenge_labels') else y)[domains[3]]
            yt = loaded[task]
            observed = np.asarray(payload['observed_indices' if email else 'labeled_target_local_indices'])
            np.testing.assert_array_equal(observed, sample_target(yt, m, seed))
            test = np.setdiff1d(np.arange(len(yt)), observed)
            np.testing.assert_array_equal(test, payload['test_indices' if email else 'test_target_local_indices'])
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions['test_indices'], test)
                np.testing.assert_array_equal(predictions['y_test'], yt[test])
                for row in rows:
                    identity = (task, m, seed, row['method'])
                    if identity in seen:
                        raise AssertionError(f'Duplicate evaluation: {identity}')
                    seen.add(identity)
                    score = predictions[row['method']]
                    assert score.shape == (len(test),) and np.isfinite(score).all()
                    ap = manual_ap(yt[test], score)
                    max_error = max(max_error, abs(ap-row['ap_percent']))
                    values[(task, m, row['method'])].append(ap)
        assert max_error < 1e-10, max_error
        summary = json.loads((base/'summary.json').read_text(encoding='utf-8'))
        assert len(summary) == len(values)
        for row in summary:
            key = ('email_spam', 10, row['method']) if email else (row['task'], row['m'], row['method'])
            v = values[key]
            assert row['n_runs'] == len(v)
            np.testing.assert_allclose(row['mean_ap_percent'], np.mean(v), atol=1e-10, rtol=0)
            np.testing.assert_allclose(row['std_ap_percent'], np.std(v, ddof=1) if len(v)>1 else 0, atol=1e-10, rtol=0)
        if email:
            protocol = json.loads((base/'protocol.json').read_text(encoding='utf-8'))
            assert protocol['seeds'] == sorted({x[2] for x in seen})
        report['datasets'][name] = {'run_files': len(paths), 'method_evaluations': len(seen),
                                   'summary_rows': len(summary), 'max_ap_error_pp': max_error}
    report['status'] = 'passed'
    (root/'results/saved_result_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
