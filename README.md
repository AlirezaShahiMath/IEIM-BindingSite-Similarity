# IEIM: Intra-Pocket Element Interaction Manifolds for Binding-Site Similarity

**IEIM** is a pocket-only extension of the Element Interaction Manifold (EIM) framework for ligand-independent binding-site similarity. In IEIM, the descriptor is computed only from protein atoms inside a selected binding pocket. Ligand information may be used only for benchmark-aligned pocket selection or pocket definition, not for feature computation.

This repository supports IEIM feature extraction, pocket-only similarity calculation, ROC--AUC evaluation, and fusion with baseline binding-site comparison methods.

---

## Main idea

Classical EIM uses protein--ligand atom interactions. IEIM instead uses **intra-pocket protein atom interactions**. Atoms inside the pocket serve as both source and environment points, allowing the descriptor to encode intrinsic binding-site geometry without explicit protein--ligand interaction terms.

IEIM represents each pocket using curvature-based statistics computed across element-pair manifolds. The default supported protein atom elements are:

```text
C, N, O, S
```

The full IEIM descriptor combines:

- global pocket surface features
- local pocket surface features
- mean curvature
- Gaussian curvature
- minimum and maximum principal curvature statistics

---

## Repository structure

```text
IEIM-BindingSite-Similarity/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── eim_pocket_global_surface.py
│   ├── eim_pocket_local_surface.py
│   ├── element_interactive_density.py
│   ├── element_interactive_curvature.py
│   ├── numba_utils_methods.py
│   ├── pocket_extraction.py
│   ├── similarity.py
│   ├── evaluation.py
│   └── fusion.py
│
├── examples/
│   ├── 01_prepare_pdbbind_pockets.py
│   ├── 02_prepare_kahraman_pockets.py
│   ├── 03_extract_ieim_features.py
│   ├── 04_compute_similarity.py
│   ├── 05_kahraman_auc.py
│   └── 06_kahraman_fusion.py
│
├── data/
│   ├── pdbbind/
│   │   └── README.md
│   └── kahraman/
│       ├── README.md
│       └── kahraman_groups_template.csv
│
├── results/
│   ├── README.md
│   ├── kahraman_summary.csv
│   └── pdbbind_summary.csv
│
├── figures/
│   └── README.md
│
└── docs/
    └── method_description.md
```

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python >= 3.9
```

---

## Data organization

Raw datasets are **not included** in this repository.

### PDBbind v2016

Expected local structure:

```text
data/pdbbind/refined-set/
├── 1abc/
│   ├── 1abc_protein.pdb
│   ├── 1abc_ligand.sdf
│   ├── 1abc_pocket.pdb        # optional curated PDBbind pocket
│   └── 1abc_EIM_pocket.pdb    # generated IEIM pocket file
└── ...
```

For PDBbind, candidate pockets can be detected from protein structure using Fpocket. For benchmark evaluation, the candidate nearest to the bound-ligand centroid can be selected. If the curated PDBbind `_pocket.pdb` files are available, they can also be copied into the IEIM naming format.

### Kahraman benchmark

Expected local structure:

```text
data/kahraman/raw_structures/
├── 1abcA.pdb
├── 2xyzB.pdb
└── ...
```

Prepared IEIM pockets are saved as:

```text
data/kahraman/pockets/
├── 1abcA/
│   └── 1abcA_EIM_pocket.pdb
└── ...
```

For Kahraman, pockets are defined as protein residues with at least one heavy atom within 6.5 Å of any bound-ligand heavy atom. Ligand atoms are used only to define the pocket and assign ligand-class labels; they are not used in IEIM descriptor computation.

---

## Basic workflow

### 1. Prepare PDBbind pockets

```bash
python examples/01_prepare_pdbbind_pockets.py \
  --pdbbind_dir data/pdbbind/refined-set
```

### 2. Prepare Kahraman pockets

```bash
python examples/02_prepare_kahraman_pockets.py \
  --raw_dir data/kahraman/raw_structures \
  --out_dir data/kahraman/pockets \
  --cutoff 6.5
```

### 3. Extract IEIM features

```bash
python examples/03_extract_ieim_features.py \
  --pocket_dir data/kahraman/pockets \
  --metadata_csv data/kahraman/kahraman_groups.csv \
  --id_col structure_id \
  --label_col ligand_class \
  --out_prefix results/kahraman_ieim
```

This generates:

```text
results/kahraman_ieim_features.csv
results/kahraman_ieim_features.npz
```

### 4. Compute similarity scores

```bash
python examples/04_compute_similarity.py \
  --features_npz results/kahraman_ieim_features.npz \
  --out_dir results/kahraman_similarity
```

This generates cosine, Euclidean, Manhattan, and size-only similarity files.

### 5. Evaluate Kahraman AUC

```bash
python examples/05_kahraman_auc.py \
  --features_npz results/kahraman_ieim_features.npz \
  --label_csv data/kahraman/kahraman_groups.csv \
  --id_col structure_id \
  --label_col ligand_class
```

---

## Similarity metrics

IEIM can be evaluated with different similarity metrics:

| Metric | Interpretation |
|---|---|
| Cosine similarity | Size-normalized shape pattern |
| Raw Euclidean distance | Magnitude-sensitive, size-aware geometry |
| Raw Manhattan distance | Magnitude-sensitive geometry |
| L2-norm difference | Size / feature-magnitude control |

On the Kahraman benchmark, Euclidean IEIM performs much better than cosine IEIM, but the size-only control shows that most of this gain is driven by pocket-size or feature-magnitude effects. Therefore, Euclidean IEIM should be interpreted as a size-aware similarity score, not as pure shape-normalized recognition.

---

## Fusion strategies

The repository includes utility functions for:

- equal-weight fusion
- weighted grid-search fusion
- cross-validated weighted fusion
- rank fusion
- selected 2-way, 3-way, and 4-way combinations

In the final Kahraman analysis, the strongest fusion combined:

- IEIM Euclidean similarity
- CGS-BSite residue-level chemical/geometric matching
- pocket-restricted 3DZD surface shape

PocketMatch was useful as an individual baseline but received little or no weight in the strongest Euclidean fusion.

---

## Reported benchmark summaries

Summary CSV files are provided in `results/` for manuscript reporting. These files contain the final AUC values used in the IEIM binding-site similarity analysis.

---

## Notes

- IEIM feature extraction can be computationally expensive.
- For large-scale PDBbind analysis, parallel or cluster execution is recommended.
- Keep raw PDBbind and Kahraman datasets outside GitHub.
- Avoid committing large `.npz`, `.npy`, `.pdb`, `.sdf`, or `.mol2` files.

---

## Citation

Citation information will be added when the manuscript is available.

---

## License

Use the license that is appropriate for your source code and dependencies. If this repository reuses GPL-licensed EIM source code, keep the repository GPL-compatible unless you have permission to relicense the code.

