"""The four classifiers compared in the study, with small tuning grids.

``get_models`` returns a dict mapping ``name -> (estimator, param_grid)``. The
estimators are unfitted; ``param_grid`` keys are bare hyperparameter names (the
evaluation harness prefixes them with ``clf__`` for use inside a Pipeline).

Class imbalance (~13% spam) is handled at the model level:
- Logistic Regression & Linear SVM use ``class_weight='balanced'``.
- XGBoost uses ``scale_pos_weight`` (pass the ham:spam ratio of the *training*
  set, computed after the split).

Grids are deliberately small - the study runs on CPU only.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .config import RANDOM_SEED


def get_models(scale_pos_weight: float = 1.0) -> dict:
    """Return ``{name: (estimator, param_grid)}`` for the four classifiers.

    Parameters
    ----------
    scale_pos_weight : float
        XGBoost's imbalance correction = n_negative / n_positive on the training
        set. Compute it from ``y_train`` after the split and pass it in.
    """
    return {
        "MultinomialNB": (
            MultinomialNB(),
            {"alpha": [0.01, 0.1, 0.5, 1.0]},
        ),
        "LogisticRegression": (
            LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
            {"C": [0.1, 1, 10]},
        ),
        "LinearSVM": (
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=5000,
                random_state=RANDOM_SEED,
            ),
            {"C": [0.1, 1, 10]},
        ),
        "XGBoost": (
            XGBClassifier(
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                n_jobs=1,  # single-threaded by design (see OMP_NUM_THREADS note in config.py)
                random_state=RANDOM_SEED,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 6],
                "learning_rate": [0.1, 0.3],
            },
        ),
    }
