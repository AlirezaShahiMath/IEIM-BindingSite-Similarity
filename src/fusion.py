#!/usr/bin/env python
"""Fusion utilities for binding-site similarity scores."""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .similarity import minmax_normalize, rank_normalize


def weighted_sum(score_dict: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    out = None
    for name, w in weights.items():
        arr = np.asarray(score_dict[name], dtype=float)
        out = w * arr if out is None else out + w * arr
    return out


def convex_weight_grid(names: Sequence[str], step: float = 0.05):
    """Generate convex weight dictionaries for the given method names."""
    units = int(round(1.0 / step))
    k = len(names)
    for counts in itertools.product(range(units + 1), repeat=k):
        if sum(counts) != units:
            continue
        yield {name: c / units for name, c in zip(names, counts)}


def best_grid_fusion(score_dict: Dict[str, np.ndarray], y: np.ndarray, step: float = 0.05):
    names = list(score_dict.keys())
    best_auc = -np.inf
    best_weights = None
    best_scores = None
    for weights in convex_weight_grid(names, step=step):
        fused = weighted_sum(score_dict, weights)
        auc = roc_auc_score(y, fused)
        if auc > best_auc:
            best_auc = auc
            best_weights = weights
            best_scores = fused
    return float(best_auc), best_weights, best_scores


def equal_weight_fusion(score_dict: Dict[str, np.ndarray]) -> np.ndarray:
    names = list(score_dict.keys())
    return np.mean([score_dict[n] for n in names], axis=0)


def rank_fusion(score_dict: Dict[str, np.ndarray]) -> np.ndarray:
    return np.mean([rank_normalize(score_dict[n]) for n in score_dict], axis=0)


def cv_weighted_fusion(score_dict: Dict[str, np.ndarray], y: np.ndarray, step: float = 0.05, n_splits: int = 5, random_state: int = 42):
    """Cross-validated weighted fusion.

    Scores are min-max normalized separately within this function before fusion.
    For strict external validation, replace this with train-fitted normalization.
    """
    names = list(score_dict.keys())
    X = {n: np.asarray(score_dict[n], dtype=float) for n in names}
    y = np.asarray(y, dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_aucs = []
    fold_weights = []

    for train_idx, test_idx in skf.split(np.zeros_like(y), y):
        train_scores = {n: minmax_normalize(X[n][train_idx]) for n in names}
        test_scores = {n: minmax_normalize(X[n][test_idx]) for n in names}
        best_auc, best_w, _ = best_grid_fusion(train_scores, y[train_idx], step=step)
        fused_test = weighted_sum(test_scores, best_w)
        fold_aucs.append(float(roc_auc_score(y[test_idx], fused_test)))
        fold_weights.append(best_w)

    mean_weights = {n: float(np.mean([w[n] for w in fold_weights])) for n in names}
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), mean_weights, fold_aucs
