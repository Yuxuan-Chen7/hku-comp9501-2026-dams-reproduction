# DAMs paper reproduction

Independent Python reproduction of **Domain Adaptation from Multiple Sources: A Domain-Dependent Regularization Approach**, Lixin Duan, Dong Xu and Ivor W. Tsang, *IEEE Transactions on Neural Networks and Learning Systems*, 23(3):504–518, 2012. [Paper DOI](https://doi.org/10.1109/TNNLS.2011.2178556).

Course project by **Group 20 (presentation order 16)**: **Chen Yuxuan, Deng Zihan, Han Yue, Jia Feiyu**.

This is an independent implementation, not the authors' original code. It implements **FastDAM**, **UniverDAM**, five paper baselines and two additional ensemble controls. The 20 Newsgroups core results closely match the paper at 10 labeled target examples per class. Email Spam preserves the main FastDAM/UniverDAM ordering but does not exactly reproduce the paper's numerical results. TRECVID has **not** been reproduced because the required original feature data could not be obtained.

## What is included

| Experiment | Completed local runs | Status |
| --- | --- | --- |
| 20 Newsgroups, original count-matrix release | 3 tasks × 7 label budgets × 10 seeds; 1,800 method evaluations | Completed |
| ECML/PKDD 2006 Email Spam | 10 seeds × 7 methods × 2 positive-label conventions; 140 method evaluations | Completed, with unresolved numerical gaps |
| TRECVID 2005 video classification | None | Not reproduced |

`reference_results/` contains frozen aggregate metrics from the completed experiments (1,940 method evaluations), together with their independent audit report. New runs write to a separate work directory and never overwrite those reference tables. Dataset archives, processed document features, per-run predictions, trained models and course materials are not committed. The scripts download and prepare the required public text datasets locally and generate per-run results when executed.

## Install and run

Use **Python 3.12**. All computation uses the **CPU**: NumPy/SciPy and scikit-learn LIBSVM. No CUDA or GPU is required. The experiment driver limits BLAS to two threads. Dense baseline kernels need several GB of RAM; one 7,500 × 7,500 float64 matrix alone occupies about 450 MB.

Windows PowerShell:

```powershell
git clone https://github.com/Yuxuan-Chen7/dams-reproduction.git
cd dams-reproduction
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m pytest -q

# Smoke run: one 20NG task, m=10, one seed; downloads the original count archive.
& $py reproduce.py --work-dir work-smoke --datasets 20ng --tasks rec_vs_sci --m 10 --seeds 0

# Full default experiment grid, including both Email Spam label conventions.
& $py reproduce.py --work-dir work

# Recalculate AP and verify saved splits and summaries without retraining.
& $py reproduce.py --work-dir work --stage verify
```

Linux/macOS setup (the experimental commands are identical after replacing `& $py` above with `python`):

```bash
git clone https://github.com/Yuxuan-Chen7/dams-reproduction.git
cd dams-reproduction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python reproduce.py --work-dir work-smoke --datasets 20ng --tasks rec_vs_sci --m 10 --seeds 0
```

The release was tested on Windows with Python 3.12.14; the Linux/macOS commands are provided for portability but were not executed on those systems. `requirements-reference.txt` records the exact direct package versions used for the reported results. `requirements.txt` permits compatible newer versions; numerical agreement across environments is not guaranteed.

Useful options:

```powershell
# Download/check archives and prepare data only.
& $py reproduce.py --work-dir work --stage prepare

# Only Email Spam, both conventions, all ten seeds.
& $py reproduce.py --work-dir work --datasets email

# Offline/reuse: path to an existing data directory, not to an archive file.
& $py reproduce.py --work-dir work-new --archive-dir "path\to\existing\data"

# Continue training from already prepared data.
& $py reproduce.py --work-dir work --stage run
```

The default 20NG grid is `m = 0, 2, 4, 6, 10, 15, 20`, with seeds `0–9`. Email always uses 10 labels per class. `--seeds 0` is a smoke test, not a reproduction of a ten-run mean. All methods within a task/budget/seed share the same split. At `m=0`, there is no label sampling, so repeated seed results are identical; SVM-T, MCC-SVM and FR are unavailable at that budget.

The archive directory must contain:

```text
data/
  20ng_matlab/raw/20news-bydate-matlab.tgz
  email_spam/raw/data_task_a.zip
  email_spam/raw/evaluation_data_labeled.zip
```

Historical download servers may be unavailable. An unsuccessful download is an error, not an empty or substituted experiment. The offline option accepts only the same checksum-verified releases. See [data sources and hashes](docs/data_sources.md).

## Results versus the paper

The metric is **noninterpolated average precision (AP), in percent**, not classification accuracy. Local tables report ten-run means and sample standard deviations (`ddof=1`). Differences are **percentage points (pp)**, calculated from unrounded local means against the paper's printed two-decimal values. Seeds used by the authors were not available.

### 20 Newsgroups: 10 labeled target examples per class

| Task | Method | Paper AP | Local AP (mean ± SD) | Local − paper (pp) |
| --- | --- | ---: | ---: | ---: |
| rec_vs_sci | FastDAM | 98.01 | 98.016 ± 0.142 | +0.006 |
| rec_vs_sci | UniverDAM | 99.71 | 99.707 ± 0.019 | -0.003 |
| comp_vs_rec | FastDAM | 99.06 | 99.073 ± 0.052 | +0.013 |
| comp_vs_rec | UniverDAM | 99.82 | 99.822 ± 0.009 | +0.002 |
| sci_vs_comp | FastDAM | 95.28 | 95.243 ± 0.218 | -0.037 |
| sci_vs_comp | UniverDAM | 98.59 | 98.583 ± 0.052 | -0.007 |

For this **m=10 comparison only**, the maximum absolute difference is approximately **0.037 pp for FastDAM** and **0.0072 pp for UniverDAM**. This close agreement does not apply to every method or budget: for example, sci-versus-comp FastDAM at `m=20` is 95.710% locally versus 95.87% in the paper. See the [full m=10 paper comparison, including baselines](reference_results/20ng_matlab/paper_comparison_m10.csv) and all budgets in the [20NG summary CSV](reference_results/20ng_matlab/summary.csv).

The original 18,774-document, 61,188-feature count-matrix release is used, with L2 row normalization. Three source domains and one target domain follow the visual columns of the paper's Table III. Labels selected for supervision are excluded from AP evaluation; the remaining target features participate in adaptation. These are **transductive** results, not evaluation on completely unseen future documents.

### Email Spam: 10 labeled target examples per class

The paper text defines non-spam as the positive class, while the challenge data encode spam as positive. AP is sensitive to this choice. Both conventions are reported, rather than choosing whichever produces the best score or closest match. The sampler draws positives before negatives, so the same seed across the two conventions need not select identical documents; this is not a controlled paired estimate of the label-polarity effect alone.

| Method | Paper AP | Local AP, non-spam positive | Local AP, spam positive | Spam-positive − paper (pp) |
| --- | ---: | ---: | ---: | ---: |
| SVM_T | 83.01 | 86.209 | 83.258 | +0.248 |
| SVM_S | 68.62 | 75.782 | 68.582 | -0.038 |
| MCC_SVM | 76.89 | 82.064 | 77.064 | +0.174 |
| FR | 84.13 | 89.631 | 86.637 | +2.507 |
| Multi_KMM | 74.10 | 81.262 | 73.775 | -0.325 |
| FastDAM | 85.36 | 89.500 | 86.997 | +1.637 |
| UniverDAM | 87.05 | 92.435 | 90.272 | +3.222 |

The spam-positive comparison brings SVM-T, SVM-S, MCC-SVM and Multi-KMM within roughly 0.33 pp of the paper, but leaves **FR +2.51 pp, FastDAM +1.64 pp and UniverDAM +3.22 pp**. This is **partial agreement, not exact numerical reproduction**. Non-spam-positive FastDAM/UniverDAM are 89.500%/92.435%, respectively; the table above makes that distinction explicit. Full standard deviations are in [non-spam-positive results](reference_results/email_spam/summary.csv) and [spam-positive results](reference_results/email_spam_challenge_labels/summary.csv).

### Why some local results are higher

The improvement is **not evidence of a GPU advantage**: this code runs on the CPU. Faster hardware mainly changes elapsed time; it does not inherently improve AP for the same deterministic optimization problem and data. Solver tolerances and floating-point libraries can cause small differences, but they do not by themselves explain the observed protocol-dependent gaps.

An earlier exploratory 20NG run used raw news messages with a newly constructed vocabulary and produced higher scores. Switching to the original count-matrix release brought the core `m=10` results close to the paper. That is direct evidence that the data representation mattered; the historical exploratory results are not mixed into this repository's reference tables.

For Email Spam, positive-label semantics demonstrably change AP. Remaining uncertainties include the paper's unspecified feature normalization, exact feature-replication details and unavailable original random splits/training code. This implementation fixes L2 normalization and FR in kernel feature space. Their individual contributions to the remaining gap have not been isolated. We do not claim to have explained away that gap, reproduced the authors' significance tests, or established an improvement over the original method.

## Implementation and protocol choices

- **FastDAM:** continuous virtual targets from weighted source scores; target-only support pool; retain unlabeled target points with absolute virtual score greater than 0.3. Evaluation still covers every unlabeled target point, including those filtered out of fitting.
- **UniverDAM:** add source examples as Universum points with virtual target zero. Their class labels are not supplied as Universum supervision.
- **Optimization:** exact elimination of the latent decision values yields a convex epsilon-Huber primal for the linear target kernel, solved with L-BFGS. This is not ordinary ridge regression. An independent precomputed-kernel SVR implementation checks the linear objective.
- **Source models:** 18 models = 3 source domains × {linear, polynomial degrees 1.1, 1.2, 1.3, 1.4, 1.5}. They are source-only SVMs for 20NG and FR models using the same 20 observed target labels for Email Spam.
- **Parameters:** `C=1`, `epsilon=0.1`, `lambda_L=lambda_D=lambda_U=1`; `beta=10,000` for 20NG and `100` for email. The default weight convention follows Eq. (30) literally, `exp(-beta*(MMD²)²)`, because its printed DIST definition is already squared. The alternative `exp(-beta*MMD²)` is available in `domain_weights`; it is not selected by target test AP.
- **Baselines:** SVM-T, SVM-S, MCC-SVM, FR and the paper-described mean-shift Multi-KMM (`alpha=1`). Multi-KMM here is not a general-purpose kernel-mean-matching quadratic program. Two extra controls directly combine the identical 18 source models, without target adaptation.
- **Email domains:** U00/U01/U02 are the three 2,500-example labeled sources; the public 4,000-example training set is the target. Source labels come from the released evaluation labels and are checked against the original feature rows. The challenge's tuning files are excluded because their word-ID mapping differs.

The linear prediction vector is explicitly formed for both DAM variants. Consequently this repository does **not** validate the paper's general-kernel support-vector prediction-speed claims or compare elapsed times with its 2012 machine. Some baseline timing fields combine fitting and prediction or reuse a bank; do not rank algorithms using those fields without checking their scope.

## Validation, resume and limitations

`python -m pytest -q` runs numerical objective/gradient tests, split and feature-replication tests, tie-aware AP checks and cache-resume regression tests. Two dataset integration tests are skipped until data are prepared. To run them against a nondefault directory in PowerShell, set `$env:DAMS_WORKDIR = "absolute\path\to\work"` before pytest; use `DAMS_WORKDIR=/path/to/work python -m pytest -q` in Bash.

The default driver performs saved-score verification after training. It independently recalculates AP, validates splits against prepared labels and checks every summary mean/SD. **This alone does not independently prove correct preprocessing or absence of training leakage.**

After the **complete default grid for both datasets**, a stronger audit reconstructs features/labels directly from raw archives, recalculates AP with separate arithmetic, retrains all 54 20NG source models and 36 seed-0 email FR models, and checks representative DAM coefficients/predictions:

```powershell
$env:DAMS_WORKDIR = (Resolve-Path work).Path
& $py -m src.audit_full
```

For the frozen reference run, all 1,940 saved AP values were checked; the maximum discrepancy was below `4.3e-14` pp. See [independent audit results](reference_results/independent_audit.json). This is an implementation audit, not certification that all unspecified author choices have been recovered. The full audit requires the default grid and seed 0; it is deliberately stricter than the smoke-run verifier.

The public release also received a fresh seed-0 run of all three 20NG tasks and both email conventions, with outputs compared to the frozen runs. See [release validation](docs/release_validation.json) for the measured outcome and environment.

Each run directory binds its caches to the numerical data, source code and NumPy/SciPy/scikit-learn versions. A change causes an explicit error; select a new `--work-dir` rather than mixing configurations. Adding seeds or budgets with unchanged code/data is supported. Summaries include all completed runs in that work directory; email protocol files record both the requested seeds for the latest invocation and all completed seeds. Use a different directory when you need a separate subset comparison. Do not run concurrent writers into the same work directory.

Output layout:

```text
work/
  data/                         # Downloaded archives and prepared features
  results/20ng_matlab/           # Runs, banks, models, summaries, cache contracts
  results/email_spam/            # Non-spam positive
  results/email_spam_challenge_labels/  # Spam positive
  results/saved_result_verification.json
reference_results/              # Frozen metrics; never used as training input
```

This is a reproduction of the two available text experiments, not the complete paper. TRECVID, the original machine timings, original author splits and full parameter sweeps remain outside the completed scope. External datasets retain their original rights and are fetched from their source/archive locations.

## Citation

```bibtex
@article{duan2012domain,
  title={Domain Adaptation from Multiple Sources: A Domain-Dependent Regularization Approach},
  author={Duan, Lixin and Xu, Dong and Tsang, Ivor W.},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={23}, number={3}, pages={504--518}, year={2012},
  doi={10.1109/TNNLS.2011.2178556}
}
```
