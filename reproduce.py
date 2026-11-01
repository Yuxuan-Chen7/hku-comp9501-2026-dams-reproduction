"""Portable entry point; see README for full and smoke-run commands."""
import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', default='work')
    p.add_argument('--datasets', nargs='+', choices=['20ng', 'email'], default=['20ng', 'email'])
    p.add_argument('--stage', choices=['all', 'prepare', 'run', 'verify'], default='all')
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--m', nargs='+', type=int, default=[0, 2, 4, 6, 10, 15, 20], help='20NG labels per class; email always uses 10')
    p.add_argument('--tasks', nargs='+', choices=['rec_vs_sci', 'comp_vs_rec', 'sci_vs_comp'])
    p.add_argument('--archive-dir', type=Path, help='Optional existing data directory containing 20ng_matlab/raw and email_spam/raw')
    args = p.parse_args()
    if any(x < 0 for x in args.seeds + args.m):
        p.error('Seeds and label budgets must be nonnegative')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.m)) != len(args.m):
        p.error('Duplicate seeds or label budgets are not allowed')
    if any(m > 900 for m in args.m):
        p.error('Label budget exceeds the supported target-class size')
    root = Path(args.work_dir).resolve()
    os.environ['DAMS_WORKDIR'] = str(root)  # Must precede src imports.
    root.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits
    from src.data import TASKS
    from src import acquire_additional_data as acquisition
    if args.stage in ['all', 'prepare']:
        if args.archive_dir:
            for ds in args.datasets:
                names = [('20ng_matlab', '20news-bydate-matlab.tgz')] if ds == '20ng' else [
                    ('email_spam', 'data_task_a.zip'), ('email_spam', 'evaluation_data_labeled.zip')]
                for folder, name in names:
                    target = root/'data'/folder/'raw'/name
                    if not target.exists():
                        source = args.archive_dir/folder/'raw'/name
                        if not source.is_file():
                            raise FileNotFoundError(source)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
        if '20ng' in args.datasets:
            acquisition.acquire_matlab()
            from src.audit_matlab_features import prepare
            prepare()
        if 'email' in args.datasets:
            acquisition.acquire_email()
            from src.email_data import prepare
            prepare()
    with threadpool_limits(limits=2):
        if args.stage in ['all', 'run']:
            if '20ng' in args.datasets:
                from src import experiment, extended_baselines, ensemble_baselines
                settings = SimpleNamespace(tasks=args.tasks or list(TASKS), m=args.m,
                                           seeds=args.seeds, ablations=False, force=False)
                experiment.run(settings)
                extended_baselines.run(settings)
                ensemble_baselines.main('matlab', settings.tasks, args.m, args.seeds)
            if 'email' in args.datasets:
                from src.email_experiment import run
                for positive in ['nonspam', 'spam']:
                    run(args.seeds, positive)
        if args.stage in ['all', 'verify']:
            from src.verify_results import verify
            report = verify(root, args.datasets)
            print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
