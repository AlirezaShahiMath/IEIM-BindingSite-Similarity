#!/usr/bin/env python
"""Prepare IEIM pocket files for PDBbind.

This script copies curated PDBbind *_pocket.pdb files into the IEIM naming
format expected by the feature extractor: *_EIM_pocket.pdb.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pocket_extraction import copy_pdbbind_curated_pocket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbbind_dir", required=True, help="Directory containing PDBbind complex folders")
    args = parser.parse_args()

    pdbbind_dir = Path(args.pdbbind_dir)
    folders = sorted([p for p in pdbbind_dir.iterdir() if p.is_dir()])

    copied = 0
    failed = 0
    for folder in tqdm(folders, desc="Copying curated pockets"):
        ok = copy_pdbbind_curated_pocket(pdbbind_dir, folder.name)
        copied += int(ok)
        failed += int(not ok)

    print(f"Copied: {copied}")
    print(f"Failed/missing: {failed}")


if __name__ == "__main__":
    main()
