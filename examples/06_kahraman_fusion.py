#!/usr/bin/env python
"""Small template for Kahraman fusion analysis.

This script expects precomputed pair-level scores in a CSV file. Required columns:
label, ieim, cgs, pm
Optional column: dzd
"""

import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.similarity import minmax_normalize
from src.fusion import best_grid_fusion, cv_weighted_fusion, equal_weight_fusion, rank_fusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_scores_csv", required=True)
    parser.add_argument("--use_dzd", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.pair_scores_csv)
    y = df["label"].astype(int).values

    names = ["ieim", "cgs", "pm"]
    if args.use_dzd:
        names.append("dzd")

    scores = {n: minmax_normalize(df[n].values) for n in names}

    print("Individual AUCs")
    for n in names:
        print(f"{n:8s}: {roc_auc_score(y, scores[n]):.4f}")

    auc, weights, _ = best_grid_fusion(scores, y, step=0.05)
    print(f"Grid-best fusion: {auc:.4f} | {weights}")

    cv_mean, cv_std, cv_weights, folds = cv_weighted_fusion(scores, y, step=0.05)
    print(f"CV fusion: {cv_mean:.4f} ± {cv_std:.4f} | {cv_weights}")

    print(f"Rank fusion: {roc_auc_score(y, rank_fusion(scores)):.4f}")
    print(f"Equal fusion: {roc_auc_score(y, equal_weight_fusion(scores)):.4f}")


if __name__ == "__main__":
    main()
