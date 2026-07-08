# Comparative Analysis of Classical ML Classifiers for Spam Detection

Final-year dissertation (Shuvo Roy Ovi, 25004376). An empirical, reproducible comparison of
four classical classifiers — Multinomial Naive Bayes, Logistic Regression, Linear SVM, and
XGBoost — for SMS spam detection on the UCI/Kaggle SMS Spam Collection.

See `PLAN.md` for the full phase-by-phase plan and `CLAUDE.md` for the working rules.

## Setup

Requires **Python 3.12** (not the system default). One-time setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## Reproduce

Notebooks read `data/spam.csv`, reuse the single persisted train/test split, and write every
metric and figure to `results/`. Run them in order:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
# ... 02 → 05 as they are built
```

Or open them interactively with `jupyter notebook`.

## Layout

```
data/spam.csv   dataset (label + message)
src/            reusable helpers imported by the notebooks
notebooks/      01_eda → 05_significance_interpretability_efficiency
results/        metrics CSVs + figures (single source of truth for the report)
report/         dissertation drafts
```

## Reproducibility guarantees

- Fixed `RANDOM_SEED = 42` (`src/config.py`).
- One stratified 80/20 split, created once and persisted to `results/split_indices.npz`.
- Version-pinned dependencies in `requirements.txt`.
- No hand-typed metrics — every reported number comes from a file in `results/`.
