"""Reject stale scientific caches instead of silently mixing experiments."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np


def array_hash(arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        h.update(path.name.encode())
        h.update(path.read_text(encoding='utf-8').replace('\r\n', '\n').encode())
    return h.hexdigest()


def ensure_context(folder, name, X, y, domains):
    """Bind every task's caches/runs to data, code and numerical dependencies.

    Budgets and seeds are in per-run filenames and may be extended. All other
    numerical parameters live in the hashed source. A mismatch fails closed;
    choose a new --work-dir to start a different configuration.
    """
    X = X.tocsr()
    context = {
        'schema': 1, 'name': name, 'source_sha256': code_hash(),
        'data_sha256': array_hash([np.asarray(X.shape), X.data, X.indices,
                                  X.indptr, y, *domains]),
        'versions': {p: importlib.metadata.version(p)
                     for p in ['numpy', 'scipy', 'scikit-learn']},
    }
    manifest = Path(folder)/'contracts'/f'{name}.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous != context:
            raise RuntimeError(f'Cache context changed for {name}. Use a new --work-dir; '
                               'old and new data/code/package versions must not be mixed.')
        return
    patterns = ['**/seed*.json', '**/fr_seed*.npz'] if name.startswith('email_') else [f'**/{name}_*.json', f'**/{name}_*.npz']
    if any(list(Path(folder).glob(pattern)) for pattern in patterns):
        raise RuntimeError(f'Unversioned caches exist for {name}. Use a new --work-dir.')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(context, indent=2), encoding='utf-8')


def update_completed_seeds(folder, requested):
    """Record what the summary includes, even after a partial resume."""
    path = Path(folder)/'protocol.json'
    protocol = json.loads(path.read_text(encoding='utf-8'))
    seeds = set()
    for result in (Path(folder)/'runs').glob('seed*.json'):
        if not result.with_suffix('.npz').exists():
            raise RuntimeError(f'Incomplete result pair: {result.name}')
        rows = json.loads(result.read_text(encoding='utf-8'))['rows']
        seeds.update(row['seed'] for row in rows)
    protocol['seeds'] = sorted(seeds)
    protocol['completed_seeds'] = sorted(seeds)
    protocol['requested_seeds_last_invocation'] = list(requested)
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
