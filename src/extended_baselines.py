"""FR and the paper's mean-shift implementation of Multi-KMM (alpha=1).

These implement the exact feature-replication kernel and the explicitly
described mean-shifting baseline; no generic kernel-mean-matching QP is claimed.
"""
import argparse
import json
import time
import numpy as np
from scipy.special import expit
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from .data import ROOT, TASKS, load_task, sample_target
from .cache_contract import ensure_context
from .experiment import dense_kernel, record_row, summarize
from . import experiment

RESULTS = ROOT / "results/20ng_matlab"


def run(args):
    folder = RESULTS / "extended_runs"
    folder.mkdir(exist_ok=True, parents=True)
    for task in args.tasks:
        X, y, domains = load_task(task)
        ensure_context(RESULTS, task, X, y, domains)
        Xt, yt = X[domains[3]], y[domains[3]]
        sources = [X[ix] for ix in domains[:3]]
        ys = [y[ix] for ix in domains[:3]]
        Xs = X[np.concatenate(domains[:3])]
        ysource = np.concatenate(ys)
        nsource = Xs.shape[0]
        Ktt = dense_kernel(Xt, Xt)
        Kss_list = [dense_kernel(s, s) for s in sources]
        Kts_list = [dense_kernel(Xt, s) for s in sources]
        # Center each source and translate its mean to the target mean.
        means = np.asarray([np.asarray(s.mean(axis=0)).ravel() for s in sources])
        shifts = np.asarray(Xt.mean(axis=0)).ravel()[None, :] - means
        n_per = [len(ix) for ix in domains[:3]]
        boundaries = np.r_[0, np.cumsum(n_per)]
        source_shift = Xs @ shifts.T
        target_shift = Xt @ shifts.T
        shift_gram = shifts @ shifts.T
        shifted_Kss = dense_kernel(Xs, Xs)
        shifted_Kts = dense_kernel(Xt, Xs)
        for s in range(3):
            lo, hi = boundaries[s:s+2]
            shifted_Kss[:, lo:hi] += source_shift[:, s, None]
            shifted_Kss[lo:hi, :] += source_shift[:, s][None, :]
            shifted_Kts[:, lo:hi] += target_shift[:, s, None]
            for t in range(3):
                l2, h2 = boundaries[t:t+2]
                shifted_Kss[lo:hi, l2:h2] += shift_gram[s, t]
        print(task, "FR and shifted kernels ready", flush=True)
        no_labels_cache = None
        for m in args.m:
            for seed in args.seeds:
                file = folder / f"{task}_m{m:02d}_seed{seed:02d}.json"
                if file.exists() and file.with_suffix(".npz").exists() and not args.force:
                    continue
                observed = sample_target(yt, m, seed)
                test = np.setdiff1d(np.arange(len(yt)), observed)
                rows, predictions = [], {}
                nl = len(observed)
                if nl:
                    start = time.perf_counter()
                    fr_scores = []
                    for Kss, Kts, y_s in zip(Kss_list, Kts_list, ys):
                        ns = len(y_s)
                        # phi_s(x)=[x,x,0], phi_t(x)=[x,0,x].
                        Ktrain = np.empty((ns+nl, ns+nl))
                        Ktrain[:ns, :ns] = 2*Kss
                        Ktrain[:ns, ns:] = Kts[observed].T
                        Ktrain[ns:, :ns] = Kts[observed]
                        Ktrain[ns:, ns:] = 2*Ktt[np.ix_(observed, observed)]
                        model = SVC(C=1, kernel="precomputed", tol=1e-5).fit(Ktrain, np.r_[y_s, yt[observed]])
                        Ktest = np.c_[Kts[test], 2*Ktt[np.ix_(test, observed)]]
                        fr_scores.append(expit(model.decision_function(Ktest)))
                    fr_time = time.perf_counter()-start
                    predictions["FR"] = np.mean(fr_scores, axis=0)
                    rows.append(record_row(task, m, seed, "FR", yt[test], predictions["FR"], fr_time, 0,
                                           timing_note="Combined fit and prediction timing."))
                start = time.perf_counter()
                if nl == 0 and no_labels_cache is not None:
                    predictions["Multi_KMM"] = no_labels_cache.copy()
                    mk_time = 0.0
                else:
                    Ktrain = np.empty((nsource+nl, nsource+nl))
                    Ktrain[:nsource, :nsource] = shifted_Kss
                    Ktrain[:nsource, nsource:] = shifted_Kts[observed].T
                    Ktrain[nsource:, :nsource] = shifted_Kts[observed]
                    Ktrain[nsource:, nsource:] = Ktt[np.ix_(observed, observed)]
                    mk = SVC(C=1, kernel="precomputed", tol=1e-5, cache_size=512).fit(Ktrain, np.r_[ysource, yt[observed]])
                    Ktest = np.c_[shifted_Kts[test], Ktt[np.ix_(test, observed)]]
                    predictions["Multi_KMM"] = mk.decision_function(Ktest)
                    mk_time = time.perf_counter()-start
                    if nl == 0:
                        no_labels_cache = predictions["Multi_KMM"].copy()
                rows.append(record_row(task, m, seed, "Multi_KMM", yt[test], predictions["Multi_KMM"], mk_time, 0,
                                       timing_note="Paper mean shift alpha=1. Combined fit and prediction; m=0 cached after first repeat."))
                file.write_text(json.dumps({"task": task, "m": m, "seed": seed,
                                           "labeled_target_local_indices": observed.tolist(),
                                           "test_target_local_indices": test.tolist(), "rows": rows}, indent=2), encoding="utf-8")
                np.savez_compressed(file.with_suffix(".npz"), test_indices=test,
                                    y_test=yt[test], **predictions)
                print(f"{task} m={m} seed={seed}: " + ", ".join(f"{r['method']}={r['ap_percent']:.2f}" for r in rows), flush=True)
    summarize()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--m", nargs="+", type=int, default=[0,2,4,6,10,15,20])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--force", action="store_true")
    p.add_argument("--data-version", choices=["matlab"], default="matlab")
    args = p.parse_args()
    if args.data_version == "matlab":
        from .audit_matlab_features import load_task
        RESULTS = ROOT / "results/20ng_matlab"
        experiment.RESULTS = RESULTS
    with threadpool_limits(limits=2):
        run(args)
