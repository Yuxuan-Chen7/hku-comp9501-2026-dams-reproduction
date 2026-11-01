"""Paper Table III splits and a configurable local experiment directory."""
import os
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('DAMS_WORKDIR', REPO_ROOT / 'work')).resolve()

TASKS = {
    "rec_vs_sci": [("rec.autos", "sci.crypt"),
                   ("rec.motorcycles", "sci.electronics"),
                   ("rec.sport.baseball", "sci.med"),
                   ("rec.sport.hockey", "sci.space")],
    "comp_vs_rec": [("comp.graphics", "rec.autos"),
                    ("comp.os.ms-windows.misc", "rec.motorcycles"),
                    ("comp.sys.ibm.pc.hardware", "rec.sport.baseball"),
                    ("comp.sys.mac.hardware", "rec.sport.hockey")],
    "sci_vs_comp": [("sci.crypt", "comp.graphics"),
                    ("sci.electronics", "comp.os.ms-windows.misc"),
                    ("sci.med", "comp.sys.ibm.pc.hardware"),
                    ("sci.space", "comp.sys.mac.hardware")],
}

def sample_target(y, m, seed):
    rng = np.random.default_rng(seed)
    if m < 0 or m > min(np.sum(y == 1), np.sum(y == -1)):
        raise ValueError("Invalid number of labels per class")
    indices = np.r_[rng.permutation(np.flatnonzero(y == 1))[:m],
                    rng.permutation(np.flatnonzero(y == -1))[:m]]
    return np.sort(indices)

def load_task(name):
    from .audit_matlab_features import load_task as load_original_counts
    return load_original_counts(name)
