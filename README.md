# Comparative Analysis of Classical ML Classifiers for Spam Detection

Final-year dissertation (Shuvo Roy Ovi, 25004376). An empirical, reproducible comparison of
four classical classifiers - Multinomial Naive Bayes, Logistic Regression, Linear SVM, and
XGBoost - for SMS spam detection on the UCI/Kaggle SMS Spam Collection.

This repository holds the code, data and results. The dissertation document itself is not
published here: it is kept local until it has been marked, so that a similarity check cannot
match it against a public copy by the same author.

## Requirements

- **Python 3.12** exactly. Newer Python (e.g. 3.14) does **not** work here - `xgboost` and
  `shap` don't ship stable wheels for it yet, and pip will fail or silently pull broken builds.
- `git`, to clone the repo.
- No GPU needed - everything runs CPU-only.

---

## Setup - Ubuntu / Linux

**1. Install Python 3.12** (skip if `python3.12 --version` already prints `3.12.x`):

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv
```

If your Ubuntu release doesn't have `python3.12` in its default repos (older LTS versions),
add the deadsnakes PPA first:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv
```

**2. Clone the repo and create the virtual environment:**

```bash
git clone git@github.com:oviroy/sms-spam-classifier-dissertation.git
cd sms-spam-classifier-dissertation

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

**3. Run the notebooks** (see [Reproduce](#reproduce) below) - same commands as Windows from
here, since Jupyter itself is cross-platform.

**Every future terminal session**, before running anything, re-activate the venv:

```bash
cd sms-spam-classifier-dissertation
source .venv/bin/activate
```

---

## Setup - Windows

**1. Install Python 3.12:**

- Download the installer from [python.org/downloads](https://www.python.org/downloads/) -
  pick a **3.12.x** release (not 3.13/3.14).
- Run it and **check "Add python.exe to PATH"** on the first screen before clicking Install.
- Verify in a new PowerShell/Command Prompt window:
  ```powershell
  py -3.12 --version
  ```

**2. Clone the repo and create the virtual environment** (PowerShell):

```powershell
git clone git@github.com:oviroy/sms-spam-classifier-dissertation.git
cd sms-spam-classifier-dissertation

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

If `.venv\Scripts\Activate.ps1` fails with a script-execution error (PowerShell blocks
unsigned scripts by default), either:
- run PowerShell **as Administrator** once and execute
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry activation, **or**
- use Command Prompt (`cmd.exe`) instead of PowerShell, where activation is:
  ```cmd
  .venv\Scripts\activate.bat
  ```

**Every future terminal session**, before running anything, re-activate the venv:

```powershell
cd sms-spam-classifier-dissertation
.venv\Scripts\Activate.ps1
```

---

## Reproduce

Notebooks read `data/spam.csv`, reuse the single persisted train/test split
(`results/split_indices.npz`, created by notebook 01 and reused by every later notebook), and
write every metric and figure to `results/`. **Run them in order** - later notebooks depend on
results/artifacts written by earlier ones (e.g. 03 depends on `top_models.json` from 02):

```bash
# with the venv activated (source .venv/bin/activate  OR  .venv\Scripts\Activate.ps1)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_representations_baselines.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_preprocessing_ablation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_hybrid_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_significance_interpretability_efficiency.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_dataset_artifacts.ipynb
```

Or open them interactively and run all cells top-to-bottom:

```bash
jupyter notebook
```

**Expect notebooks 02 and 04 to take a few minutes each** (`GridSearchCV` over CPU, with
XGBoost the slow part). Notebooks 01, 03 and 05 finish in well under a minute each. This
project deliberately forces `OMP_NUM_THREADS=1` inside `src/config.py` (works identically on
both OSes since it's set from Python, not the shell) to avoid an XGBoost/OpenMP deadlock - you
don't need to set anything yourself.

---

## Building the report

The report and the scripts that build it live outside this repository, for the reason given
above. Everything the report quotes is reproduced here: run the notebooks in order and compare
`results/` against the committed copies.

---

## Layout

```
data/spam.csv     dataset (label + message)
src/              reusable helpers imported by the notebooks
notebooks/        01_eda through 06_dataset_artifacts
results/          metrics CSVs + figures (single source of truth for the report)
```

### What each notebook produces

| Notebook | Writes |
|---|---|
| 01 EDA | `eda_summary.csv`, `split_indices.npz`, class-balance / length / token figures |
| 02 Core comparison | `master_metrics.csv`, `top_models.json`, F1 + FPR heatmaps, ROC curves |
| 03 Preprocessing ablation | `ablation.csv`, `fig_ablation.png` |
| 04 Hybrid features | `hybrid_comparison.csv`, `fig_hybrid_lift.png` |
| 05 Significance / interpretability / efficiency | `significance.csv`, `efficiency.csv`, `hybrid_stat_coefficients.csv`, coefficient / importance / SHAP / efficiency figures |
| 06 Dataset artifacts | `artifact_impact.csv`, `near_duplicates.csv`, `fig_artifact_impact.png` |

## Reproducibility guarantees

- Fixed `RANDOM_SEED = 42` (`src/config.py`).
- One stratified 80/20 split, created once and persisted to `results/split_indices.npz`.
  Notebooks reload it; none of them re-splits.
- Version-pinned dependencies in `requirements.txt`.
- No hand-typed metrics - every reported number comes from a file in `results/`.
- Notebooks 03, 04 and 06 each contain an **anchor row** that re-runs the previous phase's exact
  pipeline and asserts it reproduces that phase's metrics to full floating-point precision, so
  a change that would silently alter results fails loudly instead.
- Results were **frozen** after notebook 05. Notebook 06 was added afterwards, in response to
  external review, and is strictly additive: it writes only new files and its anchor gate proves
  the four models still reproduce their frozen metrics to 1e-12 before it measures anything.
- Every metric file regenerates identically from a fresh clone and a newly built virtualenv.
  `results/efficiency.csv` is the sole exception: it records wall-clock timings, which vary by
  a few percent per run by nature.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install` fails on `xgboost` or `shap` | You're not on Python 3.12 - check `python --version` inside the activated venv. |
| `ModuleNotFoundError` when running a notebook | The venv isn't activated in the terminal Jupyter was launched from, or the kernel picked isn't the venv's - in Jupyter, switch **Kernel → Change Kernel** to the one matching `.venv`. |
| A model fit hangs with 0% CPU (XGBoost) | Should not happen - `OMP_NUM_THREADS=1` is forced in `src/config.py`. If it still hangs, confirm you didn't skip importing `src.config` before other imports in a custom cell. |
| `LookupError` about NLTK stopwords | Run `python -m nltk.downloader stopwords` again inside the activated venv. |
| PowerShell won't run `Activate.ps1` | See the execution-policy fix in the Windows setup section above, or use `cmd.exe` with `activate.bat`. |
