"""Run Table III's three tasks with recorded splits and honest local metrics.

python -m src.experiment --m 0 2 4 6 10 15 20 --seeds 0 1 2 3 4 5 6 7 8 9
python -m src.experiment --m 10 --seeds 0 1 2 3 4 5 6 7 8 9 --ablations
"""
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from .data import ROOT, TASKS, load_task, sample_target
from .cache_contract import ensure_context
from .dam import fit_dam, domain_weights

RESULTS = ROOT / "results/20ng_matlab"


def dense_kernel(A, B):
    return (A @ B.T).toarray()


def source_bank(name, X, y, domains):
    ensure_context(RESULTS, name, X, y, domains)
    cache = RESULTS / "cache"
    cache.mkdir(exist_ok=True, parents=True)
    filename = cache / f"{name}_source_bank.npz"
    if filename.exists():
        with np.load(filename, allow_pickle=False) as v:
            return {k: v[k] for k in v.files}
    started = time.perf_counter()
    Xt = X[domains[3]]
    Ktt = dense_kernel(Xt, Xt)
    scores, distances, source_ids, kernels, svm_s = [], [], [], [], []
    for source_id, ix in enumerate(domains[:3]):
        Xs, ys = X[ix], y[ix]
        Kss, Kts = dense_kernel(Xs, Xs), dense_kernel(Xt, Xs)
        for degree in [0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            if degree == 0:
                train, test = Kss, Kts
                mmd2 = Kss.mean() + Ktt.mean() - 2 * Kts.mean()
            else:
                train, test = (Kss + 1) ** degree, (Kts + 1) ** degree
                mmd2 = train.mean() + ((Ktt + 1) ** degree).mean() - 2 * test.mean()
            model = SVC(kernel="precomputed", C=1, tol=1e-5, cache_size=256)
            model.fit(train, ys)
            predicted = model.decision_function(test)
            scores.append(predicted)
            distances.append(max(mmd2, 0.0))
            kernels.append(degree)
            source_ids.append(source_id)
            if degree == 0:
                svm_s.append(expit(predicted))
            print(f"{name}: source {source_id+1}/3, kernel {degree or 'linear'}", flush=True)
    result = {"scores": np.asarray(scores).T, "mmd2": np.array(distances),
              "kernel_degree": np.array(kernels), "source_id": np.array(source_ids),
              "svm_s": np.mean(svm_s, axis=0),
              "base_training_seconds": np.array(time.perf_counter()-started)}
    np.savez_compressed(filename, **result)
    return result


def record_row(task, m, seed, method, y_test, scores, fit_seconds, pred_seconds,
               model=None, **extra):
    row = {"task": task, "m": m, "seed": seed, "method": method,
           "ap_percent": 100 * float(average_precision_score(y_test == 1, scores)),
           "fit_seconds": fit_seconds, "predict_seconds": pred_seconds,
           "n_test": len(y_test), **extra}
    if model is not None:
        row.update({"gradient_inf": model.gradient_inf_, "iterations": model.iterations_,
                    "n_fit": model.n_train_, "n_support": len(model.support_),
                    "n_target_support": model.n_target_support_,
                    "n_source_support": model.n_source_support_,
                    "n_retained_unlabeled": len(model.retained_unlabeled_),
                    "objective": model.objective_, "converged": model.converged_})
    return row


def run(args):
    results = RESULTS
    results.mkdir(exist_ok=True, parents=True)
    run_dir = results / ("ablation_runs" if args.ablations else "runs")
    run_dir.mkdir(exist_ok=True)
    for task in args.tasks:
        X, y, domains = load_task(task)
        ensure_context(RESULTS, task, X, y, domains)
        target_ix = domains[3]
        Xt, yt = X[target_ix], y[target_ix]
        Xs = X[np.concatenate(domains[:3])]
        bank = source_bank(task, X, y, domains)
        weights = domain_weights(bank["mmd2"])
        audit = {"base_training_seconds": float(bank["base_training_seconds"]),
                 "default_weight_convention": "paper_literal",
                 "base_classifiers": [{"source": int(s)+1, "degree": float(d),
                                       "mmd_squared": float(m), "weight": float(w)}
                                      for s, d, m, w in zip(bank["source_id"], bank["kernel_degree"], bank["mmd2"], weights)],
                 "source_weight_sums": [float(weights[bank["source_id"] == i].sum()) for i in range(3)]}
        (results / f"{task}_source_weights.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        for m in args.m:
            for seed in args.seeds:
                file = run_dir / f"{task}_m{m:02d}_seed{seed:02d}.json"
                if file.exists() and file.with_suffix(".npz").exists() and not args.force:
                    continue
                observed = sample_target(yt, m, seed)
                test_ix = np.setdiff1d(np.arange(len(yt)), observed)
                rows, predictions = [], {}
                if not args.ablations:
                    predictions["SVM_S"] = bank["svm_s"][test_ix]
                    rows.append(record_row(task, m, seed, "SVM_S", yt[test_ix], predictions["SVM_S"],
                                           float(bank["base_training_seconds"]), 0,
                                           timing_note="Source bank includes 18 classifiers; SVM_S uses only the 3 linear models. Prediction precomputed."))
                    if m > 0:
                        begin = time.perf_counter()
                        svm_t = SVC(kernel="linear", C=1, tol=1e-5).fit(Xt[observed], yt[observed])
                        fit_sec = time.perf_counter() - begin
                        begin = time.perf_counter()
                        target_score = svm_t.decision_function(Xt[test_ix])
                        predict_sec = time.perf_counter()-begin
                        predictions["SVM_T"] = target_score
                        # Equal fusion of the four SVM probabilities, paper Eq. (3), gamma=1/4.
                        predictions["MCC_SVM"] = (3 * bank["svm_s"][test_ix] + expit(target_score))/4
                        rows.append(record_row(task, m, seed, "SVM_T", yt[test_ix], target_score, fit_sec, predict_sec))
                        rows.append(record_row(task, m, seed, "MCC_SVM", yt[test_ix], predictions["MCC_SVM"], fit_sec, predict_sec,
                                               timing_note="Incremental target classifier only; reuses source bank."))
                variants = [("FastDAM", False, {}), ("UniverDAM", True, {})]
                if args.ablations:
                    variants = [("FastDAM_uniform", False, {"weights": np.ones(18)/18}),
                                ("FastDAM_no_filter", False, {"threshold": 0.0}),
                                ("FastDAM_standard_mmd", False, {"weights": domain_weights(bank["mmd2"], convention="standard_mmd")}),
                                ("UniverDAM_uniform", True, {"weights": np.ones(18)/18}),
                                ("UniverDAM_no_filter", True, {"threshold": 0.0}),
                                ("UniverDAM_standard_mmd", True, {"weights": domain_weights(bank["mmd2"], convention="standard_mmd")}),
                                ("UniverDAM_lambdaU0.1", True, {"lambda_U": .1}),
                                ("UniverDAM_lambdaU10", True, {"lambda_U": 10})]
                for method, universum, options in variants:
                    opts = dict(options)
                    w = opts.pop("weights", weights)
                    start = time.perf_counter()
                    model = fit_dam(Xt, observed, yt[observed], bank["scores"], w,
                                    Xs if universum else None, **opts)
                    fit_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    score = model.decision_function(Xt[test_ix])
                    predict_seconds = time.perf_counter() - start
                    predictions[method] = score
                    rows.append(record_row(task, m, seed, method, yt[test_ix], score,
                                           fit_seconds, predict_seconds, model))
                    if not args.ablations and m == 10 and seed == 0:
                        modeldir = results / "models"
                        modeldir.mkdir(exist_ok=True)
                        np.savez_compressed(modeldir / f"{task}_{method}_m10_seed0.npz",
                                            coef=model.coef_, intercept=model.intercept_,
                                            labeled_target_indices=observed,
                                            dual_coef=model.dual_coef_, support=model.support_)
                payload = {"task": task, "m": m, "seed": seed,
                           "labeled_target_local_indices": observed.tolist(),
                           "labeled_document_global_indices": target_ix[observed].tolist(),
                           "test_target_local_indices": test_ix.tolist(), "rows": rows}
                file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                np.savez_compressed(file.with_suffix(".npz"), test_indices=test_ix,
                                    y_test=yt[test_ix], **predictions)
                print(f"{task} m={m:2d} seed={seed}: " + ", ".join(f"{r['method']}={r['ap_percent']:.2f}" for r in rows), flush=True)
    summarize()


def summarize():
    rows = []
    for folder in ["runs", "extended_runs", "ablation_runs", "ensemble_runs"]:
        for file in sorted((RESULTS / folder).glob("*.json")):
            rows.extend(json.loads(file.read_text(encoding="utf-8"))["rows"])
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with (RESULTS / "all_runs.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["m"], row["method"]), []).append(row)
    summary = []
    for (task, m, method), items in sorted(groups.items()):
        values = np.array([i["ap_percent"] for i in items])
        summary.append({"task": task, "m": m, "method": method, "n_runs": len(items),
                        "mean_ap_percent": float(values.mean()),
                        "std_ap_percent": float(values.std(ddof=1)) if len(items)>1 else 0.0,
                        "mean_fit_seconds": float(np.mean([i["fit_seconds"] for i in items])),
                        "mean_predict_seconds": float(np.mean([i["predict_seconds"] for i in items]))})
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary:
        with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=summary[0].keys())
            writer.writeheader(); writer.writerows(summary)
    print(f"Saved {len(rows)} method evaluations and {len(summary)} summary rows.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--m", nargs="+", type=int, default=[0,2,4,6,10,15,20])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--data-version", choices=["matlab"], default="matlab")
    args = parser.parse_args()
    if args.data_version == "matlab":
        from .audit_matlab_features import load_task
        RESULTS = ROOT / "results/20ng_matlab"
    RESULTS.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "processor": platform.processor(), "utc": datetime.now(timezone.utc).isoformat(),
                   "command": sys.argv, "blas_thread_limit": 2}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (RESULTS / f"environment_{stamp}.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    with threadpool_limits(limits=2):
        summarize() if args.summarize else run(args)
