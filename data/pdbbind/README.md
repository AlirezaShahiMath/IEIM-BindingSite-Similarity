# PDBbind v2016 data

Raw PDBbind files are not included in this repository.

Place your local PDBbind refined-set folders here, for example:

```text
data/pdbbind/refined-set/
├── 1abc/
│   ├── 1abc_protein.pdb
│   ├── 1abc_ligand.sdf
│   ├── 1abc_pocket.pdb
│   └── 1abc_EIM_pocket.pdb
└── ...
```

For IEIM, the descriptor is computed only from protein atoms in the pocket file.
Ligand atoms are not used in feature computation.

If curated PDBbind pocket files are available, prepare IEIM pocket names using:

```bash
python examples/01_prepare_pdbbind_pockets.py --pdbbind_dir data/pdbbind/refined-set
```
