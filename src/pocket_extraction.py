#!/usr/bin/env python
"""
Pocket extraction utilities for IEIM.

IEIM descriptors must be computed only from protein atoms inside the selected
pocket. Ligand atoms can be used here only to define benchmark-aligned pockets.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

COMMON_NONLIGAND_HET = {
    "HOH", "WAT", "DOD",
    "NA", "K", "CL", "CA", "MG", "MN", "ZN", "FE", "CU", "CO", "NI",
    "SO4", "PO4", "NO3", "ACT", "EDO", "GOL",
}

PROTEIN_ELEMENTS = {"C", "N", "O", "S"}


def _pdb_element(line: str) -> str:
    """Return PDB element symbol, with fallback to atom-name parsing."""
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if elem:
        return elem.upper().capitalize()
    atom = line[12:16].strip()
    if not atom:
        return ""
    # Remove leading digit from atom names such as 1HG.
    atom = atom.lstrip("0123456789")
    return atom[:2].strip().upper().capitalize() if len(atom) >= 2 and atom[1].islower() else atom[0].upper()


def _xyz_from_pdb_line(line: str) -> Tuple[float, float, float]:
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _residue_key(line: str) -> Tuple[str, str, str, str]:
    return (line[17:20].strip(), line[21:22].strip(), line[22:26].strip(), line[26:27].strip())


def read_protein_atoms(pdb_file: str | Path, allowed_elements: Iterable[str] = PROTEIN_ELEMENTS):
    """Read protein ATOM records from a PDB file."""
    allowed = {e.upper().capitalize() for e in allowed_elements}
    rows = []
    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            elem = _pdb_element(line)
            if elem not in allowed:
                continue
            rows.append({
                "line": line,
                "element": elem,
                "xyz": np.array(_xyz_from_pdb_line(line), dtype=float),
                "residue": _residue_key(line),
            })
    return rows


def read_het_ligand_groups(
    pdb_file: str | Path,
    exclude_resnames: Iterable[str] = COMMON_NONLIGAND_HET,
):
    """Read heavy-atom HETATM groups from a PDB file.

    Returns a dictionary keyed by (resname, chain, resseq, icode).
    """
    exclude = {x.upper() for x in exclude_resnames}
    groups: Dict[Tuple[str, str, str, str], List[np.ndarray]] = {}
    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip().upper()
            elem = _pdb_element(line)
            if resname in exclude or elem.upper() == "H":
                continue
            key = _residue_key(line)
            groups.setdefault(key, []).append(np.array(_xyz_from_pdb_line(line), dtype=float))
    return groups


def choose_primary_ligand_group(groups: Dict[Tuple[str, str, str, str], List[np.ndarray]]):
    """Choose the largest HETATM heavy-atom group as the primary ligand."""
    if not groups:
        return None, None
    key = max(groups, key=lambda k: len(groups[k]))
    return key, np.vstack(groups[key])


def extract_ligand_proximity_pocket(
    protein_pdb: str | Path,
    out_pdb: str | Path,
    ligand_xyz: np.ndarray,
    cutoff: float = 6.5,
    allowed_elements: Iterable[str] = PROTEIN_ELEMENTS,
) -> int:
    """Extract all protein residues within cutoff Å of any ligand heavy atom.

    Returns the number of selected protein residues.
    """
    protein_atoms = read_protein_atoms(protein_pdb, allowed_elements=allowed_elements)
    if len(protein_atoms) == 0:
        raise ValueError(f"No protein atoms found in {protein_pdb}")
    if ligand_xyz is None or len(ligand_xyz) == 0:
        raise ValueError("No ligand coordinates supplied")

    atom_xyz = np.vstack([r["xyz"] for r in protein_atoms])
    distances = cdist(atom_xyz, ligand_xyz)
    selected_atom_mask = np.any(distances <= cutoff, axis=1)
    selected_residues = {protein_atoms[i]["residue"] for i, keep in enumerate(selected_atom_mask) if keep}

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w", encoding="utf-8") as out:
        for row in protein_atoms:
            if row["residue"] in selected_residues:
                out.write(row["line"])
        out.write("END\n")
    return len(selected_residues)


def extract_kahraman_pocket_from_pdb(
    full_pdb: str | Path,
    out_pdb: str | Path,
    cutoff: float = 6.5,
) -> Tuple[str, int]:
    """Extract a Kahraman-style ligand-proximity pocket from a full PDB file."""
    groups = read_het_ligand_groups(full_pdb)
    lig_key, lig_xyz = choose_primary_ligand_group(groups)
    if lig_key is None:
        raise ValueError(f"No ligand-like HETATM group found in {full_pdb}")
    n_res = extract_ligand_proximity_pocket(full_pdb, out_pdb, lig_xyz, cutoff=cutoff)
    ligand_name = lig_key[0]
    return ligand_name, n_res


def copy_pdbbind_curated_pocket(pdbbind_dir: str | Path, pdbid: str) -> bool:
    """Copy PDBbind curated pocket file to IEIM expected name.

    Expected input:  <pdbbind_dir>/<pdbid>/<pdbid>_pocket.pdb
    Expected output: <pdbbind_dir>/<pdbid>/<pdbid>_EIM_pocket.pdb
    """
    folder = Path(pdbbind_dir) / pdbid
    src = folder / f"{pdbid}_pocket.pdb"
    dst = folder / f"{pdbid}_EIM_pocket.pdb"
    if not src.exists():
        return False
    shutil.copyfile(src, dst)
    return True
