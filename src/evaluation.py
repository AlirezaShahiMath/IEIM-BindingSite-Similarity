#!/usr/bin/env python
"""Evaluation utilities for binding-site similarity benchmarks."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def canonical_pairs(ids: Sequence[str], labels: Dict[str, str]):
    """Build unique unordered non-self pairs and same-class labels."""
    pairs = []
    y = []
    for a, b in itertools.combinations(ids, 2):
        if a not in labels or b not in labels:
            continue
        pairs.append((a, b))
        y.append(1 if labels[a] == labels[b] else 0)
    return pairs, np.asarray(y, dtype=int)


def scores_from_matrix(pairs, id_to_idx, sim_matrix):
    return np.asarray([sim_matrix[id_to_idx[a], id_to_idx[b]] for a, b in pairs], dtype=float)


def auc_from_scores(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))


def auc_for_similarity_matrix(ids, labels, sim_matrix):
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    pairs, y = canonical_pairs(ids, labels)
    scores = scores_from_matrix(pairs, id_to_idx, sim_matrix)
    return auc_from_scores(y, scores), pairs, y, scores
