"""Additional controls: use the identical 18 base models without adaptation."""
import argparse
import json
import numpy as np
from .data import ROOT, TASKS, load_task, sample_target
from .dam import domain_weights
from .cache_contract import ensure_context
from .experiment import record_row, summarize
from . import experiment


def main(data_version='matlab', tasks=None, budgets=None, seeds=None):
    global load_task
    results=ROOT/'results/20ng_matlab'
    if data_version=='matlab':
        from .audit_matlab_features import load_task
        results=ROOT/'results/20ng_matlab'
    experiment.RESULTS=results
    folder=results/'ensemble_runs'
    folder.mkdir(exist_ok=True)
    for task in (list(TASKS) if tasks is None else tasks):
        X,y,domains=load_task(task)
        ensure_context(results,task,X,y,domains)
        yt=y[domains[3]]
        with np.load(results/'cache'/f'{task}_source_bank.npz',allow_pickle=False) as b:
            scores={'Weighted_ensemble18':b['scores']@domain_weights(b['mmd2']),
                    'Uniform_ensemble18':b['scores'].mean(axis=1)}
        for m in ([0,2,4,6,10,15,20] if budgets is None else budgets):
            for seed in (range(10) if seeds is None else seeds):
                labeled=sample_target(yt,m,seed)
                test=np.setdiff1d(np.arange(len(yt)),labeled)
                preds={method:value[test] for method,value in scores.items()}
                rows=[record_row(task,m,seed,method,yt[test],value,0,0,
                                 timing_note='Additional control. Cached base predictions, no adaptation.')
                      for method,value in preds.items()]
                file=folder/f'{task}_m{m:02d}_seed{seed:02d}.json'
                file.write_text(json.dumps({'task':task,'m':m,'seed':seed,
                                 'labeled_target_local_indices':labeled.tolist(),
                                 'test_target_local_indices':test.tolist(),'rows':rows},indent=2),encoding='utf-8')
                np.savez_compressed(file.with_suffix('.npz'),test_indices=test,y_test=yt[test],**preds)
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-version',choices=['matlab'],default='matlab')
    main(parser.parse_args().data_version)
