#!/usr/bin/env python
"""Evaluate IEIM on the Kahraman benchmark using canonical same-class pairs."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.similarity import compute_similarity_scores
from src.evaluation import auc_for_similarity_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--id_col", default="structure_id")
    parser.add_argument("--label_col", default="ligand_class")
    args = parser.parse_args()

    data = np.load(args.features_npz, allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    X = data["features"]

    label_df = pd.read_csv(args.label_csv)
    labels = dict(zip(label_df[args.id_col].astype(str), label_df[args.label_col].astype(str)))

    sims = compute_similarity_scores(X)
    for name, mat in sims.items():
        auc, pairs, y, scores = auc_for_similarity_matrix(ids, labels, mat)
        print(f"{name:12s} AUC = {auc:.4f} | pairs={len(y)} pos={int(y.sum())} neg={int(len(y)-y.sum())}")


if __name__ == "__main__":
    main()
