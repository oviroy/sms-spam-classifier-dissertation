"""Evaluation harness: metrics, the leakage-free grid runner, and timing.

Positive class = spam (label 1). The metric the supervisor cares most about is
**FPR = FP / (FP + TN)** — legitimate messages wrongly flagged as spam.

Key design point (no leakage): each configuration is a ``Pipeline([vectorizer,
classifier])`` fed to ``GridSearchCV``. The vectorizer is therefore fitted on the
training folds only, never on validation/test data.
"""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_scores(estimator, X) -> np.ndarray:
    """Return a continuous score for the positive class (for AUC/ROC).

    Uses ``predict_proba[:, 1]`` when available (MNB, LR, XGBoost); falls back to
    ``decision_function`` for models without probabilities (Linear SVM).
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def metrics(y_true, y_pred, y_score=None) -> dict:
    """Full metric suite for the positive (spam) class.

    Returns accuracy, precision, recall, F1, AUC-ROC, FPR, and the raw confusion
    matrix counts. FPR = FP / (FP + TN).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
        "auc": roc_auc_score(y_true, y_score) if y_score is not None else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return out


def run_master_grid(
    models: dict,
    representations: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    cv,
    scoring: str = "f1",
    n_jobs: int = 1,
    verbose: bool = True,
):
    """Run every (representation x model) config and return results + fitted models.

    Parameters
    ----------
    models : dict
        ``{name: (estimator, param_grid)}`` from ``models.get_models``.
    representations : dict
        ``{name: vectorizer_factory}`` — each factory returns a fresh vectorizer.
    cv : cross-validator
        The stratified k-fold from ``config.get_cv``.
    n_jobs : int
        Parallelism for ``GridSearchCV``. Default **1** (sequential): XGBoost uses
        its own internal threads, and joblib/loky is not involved, which avoids a
        known deadlock between GridSearchCV's process pool and XGBoost's OpenMP
        threads. The data here is small, so sequential is only a few minutes.
    verbose : bool
        Print each config with its wall-clock time so a hang is visible immediately.

    Returns
    -------
    (pd.DataFrame, dict)
        A tidy results frame (one row per config) and ``{(model, rep): best_estimator}``.
    """
    rows = []
    fitted = {}
    total = len(representations) * len(models)
    i = 0
    for rep_name, make_vec in representations.items():
        for model_name, (estimator, grid) in models.items():
            i += 1
            t0 = time.perf_counter()
            pipe = Pipeline([("vec", make_vec()), ("clf", clone(estimator))])
            param_grid = {f"clf__{k}": v for k, v in grid.items()}
            gs = GridSearchCV(
                pipe, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, refit=True
            )
            gs.fit(X_train, y_train)

            best = gs.best_estimator_
            y_pred = best.predict(X_test)
            y_score = get_scores(best, X_test)
            m = metrics(y_test, y_pred, y_score)
            row = {
                "model": model_name,
                "representation": rep_name,
                **m,
                "cv_best_f1": gs.best_score_,
                "best_params": {
                    k.replace("clf__", ""): v for k, v in gs.best_params_.items()
                },
            }
            rows.append(row)
            fitted[(model_name, rep_name)] = best
            if verbose:
                dt = time.perf_counter() - t0
                print(
                    f"[{i:2d}/{total}] {rep_name:14s} {model_name:20s} "
                    f"F1={row['f1']:.4f} FPR={row['fpr']:.4f} ({dt:.1f}s)",
                    flush=True,
                )
    return pd.DataFrame(rows), fitted


def time_and_size(estimator, X_train, y_train, X_test, n_repeat: int = 3) -> dict:
    """Measure fit time, per-message inference latency (ms), and pickled size (bytes).

    Used by the Phase-5 efficiency analysis; kept here so timing logic is centralised.
    """
    est = clone(estimator)
    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        est.predict(X_test)
    infer_ms = (time.perf_counter() - t0) / (n_repeat * len(X_test)) * 1000.0

    size = len(pickle.dumps(est))
    return {
        "fit_time_s": fit_time,
        "infer_latency_ms": infer_ms,
        "model_size_bytes": size,
    }
