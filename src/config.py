"""Central configuration — imported everywhere so settings are defined once.

Reproducibility rule: the random seed, the data/results paths, and the
cross-validation definition all live here. Never hard-code these elsewhere.
"""

import os

# XGBoost's OpenMP threading deadlocks on this machine (a fit hangs indefinitely
# at ~0% CPU). Forcing a single OpenMP thread eliminates the hang. This must be
# set BEFORE numpy/sklearn/xgboost load their native libraries; because every
# notebook and src module imports this config first, doing it here makes the whole
# project hang-free and reproducible. (Set only if the user hasn't overridden it.)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

from sklearn.model_selection import StratifiedKFold

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42

# --- Paths (all derived from the repo root = parent of this file's dir) -----
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "spam.csv"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
SPLIT_PATH = RESULTS_DIR / "split_indices.npz"

# --- Class labels (positive class = spam) ----------------------------------
HAM = 0
SPAM = 1

# --- Train/test split ------------------------------------------------------
TEST_SIZE = 0.20

# Make sure the results directory exists on import so notebooks can write to it.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_cv(n_splits: int = 5) -> StratifiedKFold:
    """Return the project's stratified k-fold cross-validator.

    Stratified because the dataset is imbalanced (~13% spam) — each fold keeps
    the same spam/ham ratio. Seeded so folds are identical across runs.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
