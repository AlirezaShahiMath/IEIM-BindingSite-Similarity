# Kahraman benchmark data

Raw Kahraman structure files are not included in this repository.

Expected local structure:

```text
data/kahraman/raw_structures/
├── 1abcA.pdb
├── 2xyzB.pdb
└── ...
```

Prepared pocket files are saved as:

```text
data/kahraman/pockets/
├── 1abcA/
│   └── 1abcA_EIM_pocket.pdb
└── ...
```

For this benchmark, pockets are defined as all protein residues with at least
one heavy atom within 6.5 Å of any bound-ligand heavy atom. Ligand atoms are used
only for pocket definition and label assignment, not for IEIM feature computation.

Use the template `kahraman_groups_template.csv` to create your final label file:

```text
data/kahraman/kahraman_groups.csv
```
