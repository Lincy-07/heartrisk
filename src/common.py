"""Shared helpers: data loading, the single train/test split, metrics, paths.

Every script imports from here so that ALL methods (manual, tuned, FLAML,
AutoGluon) see exactly the same rows. SEED is fixed for reproducibility.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score,
                             brier_score_loss)

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "framingham.csv")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
MODEL_DIR = os.path.join(ROOT, "model")
TARGET = "TenYearCHD"
# Fixed AutoML time budget in SECONDS, identical for every tool.
# Assignment suggests 15-30 min -> default 900 s. Override: AUTOML_BUDGET=1800
BUDGET = int(os.environ.get("AUTOML_BUDGET", 900))
N_JOBS = int(os.environ.get("N_JOBS", 2))

for d in (RES, FIG, MODEL_DIR):
    os.makedirs(d, exist_ok=True)


def load_raw():
    return pd.read_csv(DATA)


def get_split():
    """80/20 stratified split. Test set is touched ONLY for final reporting."""
    df = load_raw()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def evaluate(y_true, proba, threshold=0.5):
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
    }


def best_f1_threshold(y_true, proba):
    """Threshold maximising F1. Must be chosen on TRAIN/validation data only."""
    ts = np.linspace(0.05, 0.60, 56)
    f = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))])


def save_json(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2)


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t
