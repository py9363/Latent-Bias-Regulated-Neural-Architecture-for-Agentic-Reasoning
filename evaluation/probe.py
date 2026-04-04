"""
Linear probe to predict sensitive attribute s from hidden representations h(x).
Reusable across all baselines.
Returns R(theta) = probe accuracy.
"""
import os
import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROBE_C, PROBE_MAX_ITER


def _ensure_numpy(h: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(h, torch.Tensor):
        return h.detach().cpu().numpy()
    return np.asarray(h)


def run_probe(
    hidden_states: Union[torch.Tensor, np.ndarray],
    sensitive_attributes: Union[list, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = PROBE_C,
    max_iter: int = PROBE_MAX_ITER,
    n_splits: Optional[int] = None,
) -> dict:
    """
    Train logistic regression to predict sensitive attribute from hidden states.

    Steps:
    1. Freeze backbone (not applicable here; we receive pre-extracted h).
    2. Train logistic regression on (h, s).
    3. Report accuracy, chance baseline, ROC-AUC.

    Returns dict with:
      - accuracy: float (probe accuracy = R(theta))
      - chance_baseline: float (majority class frequency)
      - roc_auc: float (or None if binary with single class in split)
    """
    X = _ensure_numpy(hidden_states)
    y = np.asarray(sensitive_attributes).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("hidden_states and sensitive_attributes must have same length")
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    # Chance baseline = max class frequency
    unique, counts = np.unique(y, return_counts=True)
    chance_baseline = float(counts.max() / len(y))

    # Scale features for stable convergence (fit on train only, transform train & test)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, solver="lbfgs")),
    ])

    if n_splits and n_splits > 1:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        accs, rocs = [], []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            accs.append(accuracy_score(y_te, pred))
            if len(np.unique(y_te)) > 1 and len(np.unique(y_tr)) > 1:
                try:
                    proba = clf.predict_proba(X_te)
                    rocs.append(roc_auc_score(y_te, proba[:, 1]))
                except Exception:
                    rocs.append(np.nan)
        accuracy = float(np.mean(accs))
        roc_auc = float(np.nanmean(rocs)) if rocs else None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracy = float(accuracy_score(y_test, pred))
        roc_auc = None
        if len(np.unique(y_test)) > 1:
            try:
                proba = clf.predict_proba(X_test)
                roc_auc = float(roc_auc_score(y_test, proba[:, 1]))
            except Exception:
                pass

    return {
        "accuracy": accuracy,
        "chance_baseline": chance_baseline,
        "roc_auc": roc_auc,
    }


def R_theta(
    representations_path: Optional[str] = None,
    hidden_states: Optional[Union[torch.Tensor, np.ndarray]] = None,
    sensitive_attributes: Optional[Union[list, np.ndarray]] = None,
    **probe_kwargs,
) -> float:
    """
    Compute recoverability R(theta) = probe accuracy from saved representations or arrays.
    Reusable across all baselines: pass either representations_path or (hidden_states, sensitive_attributes).
    """
    if representations_path and os.path.isfile(representations_path):
        try:
            data = torch.load(representations_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(representations_path, map_location="cpu")
        h = data["hidden_states"]
        s = data["sensitive_attributes"]
    elif hidden_states is not None and sensitive_attributes is not None:
        h = hidden_states
        s = sensitive_attributes
    else:
        raise ValueError("Provide either representations_path or (hidden_states, sensitive_attributes)")
    result = run_probe(h, s, **probe_kwargs)
    return result["accuracy"]
