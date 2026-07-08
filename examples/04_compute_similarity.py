#!/usr/bin/env python
"""Compute IEIM pairwise similarity matrices."""

import argparse
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.similarity import compute_similarity_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    data = np.load(args.features_npz, allow_pickle=True)
    X = data["features"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sims = compute_similarity_scores(X)
    for name, mat in sims.items():
        np.save(out_dir / f"{name}.npy", mat)
        print(f"Saved: {out_dir / f'{name}.npy'}")


if __name__ == "__main__":
    main()
