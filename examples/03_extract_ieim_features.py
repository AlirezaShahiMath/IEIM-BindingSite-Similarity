#!/usr/bin/env python
"""Extract IEIM features from prepared pocket files.

The expected pocket file is:
<pocket_dir>/<structure_id>/<structure_id>_EIM_pocket.pdb
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.eim_pocket_global_surface import EIM_Pocket_Global_Surface
from src.eim_pocket_local_surface import EIM_Pocket_Local_Surface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket_dir", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--id_col", default="structure_id")
    parser.add_argument("--label_col", default="ligand_class")
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--mode", choices=["global", "local", "combined"], default="combined")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata_csv)
    ids = meta[args.id_col].astype(str).tolist()
    labels = meta[args.label_col].astype(str).tolist()

    rows = []
    features = []
    kept_ids = []
    kept_labels = []

    for sid, label in tqdm(list(zip(ids, labels)), desc="Extracting IEIM"):
        try:
            feat_parts = []
            if args.mode in ["global", "combined"]:
                g = EIM_Pocket_Global_Surface(args.pocket_dir, sid).get_features()
                if g is None:
                    raise ValueError("missing global features")
                feat_parts.append(g.flatten())
            if args.mode in ["local", "combined"]:
                l = EIM_Pocket_Local_Surface(args.pocket_dir, sid).get_features()
                if l is None:
                    raise ValueError("missing local features")
                feat_parts.append(l.flatten())

            feat = np.nan_to_num(np.concatenate(feat_parts))
            features.append(feat)
            kept_ids.append(sid)
            kept_labels.append(label)
            row = {args.id_col: sid, args.label_col: label}
            row.update({f"f_{i}": v for i, v in enumerate(feat)})
            rows.append(row)
        except Exception as exc:
            print(f"Skipped {sid}: {exc}")

    X = np.asarray(features, dtype=float)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(f"{out_prefix}_features.csv", index=False)
    np.savez_compressed(
        f"{out_prefix}_features.npz",
        features=X,
        ids=np.asarray(kept_ids),
        labels=np.asarray(kept_labels),
        mode=args.mode,
    )

    print(f"Saved: {out_prefix}_features.csv")
    print(f"Saved: {out_prefix}_features.npz")
    print(f"Structures processed: {len(kept_ids)}")
    print(f"Feature dimension: {X.shape[1] if len(X) else 0}")


if __name__ == "__main__":
    main()
