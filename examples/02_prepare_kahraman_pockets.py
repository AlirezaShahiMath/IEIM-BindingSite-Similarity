#!/usr/bin/env python
"""Prepare ligand-proximity IEIM pockets for the Kahraman benchmark."""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pocket_extraction import extract_kahraman_pocket_from_pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Folder containing raw Kahraman PDB files")
    parser.add_argument("--out_dir", required=True, help="Output folder for IEIM pockets")
    parser.add_argument("--cutoff", type=float, default=6.5)
    parser.add_argument("--summary_csv", default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pdb_files = sorted(raw_dir.glob("*.pdb"))
    for pdb_file in tqdm(pdb_files, desc="Extracting Kahraman pockets"):
        sid = pdb_file.stem
        out_pdb = out_dir / sid / f"{sid}_EIM_pocket.pdb"
        try:
            ligand_name, n_res = extract_kahraman_pocket_from_pdb(pdb_file, out_pdb, cutoff=args.cutoff)
            rows.append({"structure_id": sid, "primary_ligand": ligand_name, "n_pocket_residues": n_res, "status": "ok"})
        except Exception as exc:
            rows.append({"structure_id": sid, "primary_ligand": "", "n_pocket_residues": 0, "status": f"failed: {exc}"})

    summary = pd.DataFrame(rows)
    summary_csv = args.summary_csv or str(out_dir / "pocket_extraction_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
