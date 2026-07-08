#!/usr/bin/env python
"""Similarity utilities for IEIM feature matrices."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


def safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def compute_similarity_scores(X: np.ndarray) -> dict:
    """Return pairwise similarity matrices.

    Distances are negated so that larger values always mean more similar.
    """
    X = safe_nan_to_num(X)
    return {
        "cosine": cosine_similarity(X),
        "euclidean": -cdist(X, X, metric="euclidean"),
        "manhattan": -cdist(X, X, metric="cityblock"),
        "size_only": -np.abs(np.linalg.norm(X, axis=1)[:, None] - np.linalg.norm(X, axis=1)[None, :]),
    }


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Convert scores to min-max normalized ranks in [0, 1].

    Higher scores receive higher ranks. If there is only one score, return 0.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= 1:
        return np.zeros_like(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    return (ranks - 1) / (n - 1)
