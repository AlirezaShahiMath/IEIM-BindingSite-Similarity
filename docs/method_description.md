# IEIM method description

## Definition

IEIM stands for **Intra-Pocket Element Interaction Manifold**. It is a pocket-only
version of the Element Interaction Manifold framework.

In ligand-aware EIM, descriptors are computed from protein--ligand atom-pair
interactions. In IEIM, descriptors are computed only from protein atoms inside a
selected binding pocket. The atoms in the pocket serve as both source and
environment points.

## Ligand independence

Ligand-independent means that ligand atoms are not used during descriptor
computation. Ligand information may still be used to define or select a pocket
for benchmark alignment.

Examples:

- PDBbind: Fpocket can detect candidate pockets from the protein structure, and
the candidate nearest to the bound ligand can be selected for evaluation.
- Kahraman: pockets are defined as protein residues within 6.5 Å of the bound
ligand heavy atoms.

In both cases, IEIM features are computed only from protein atoms inside the
selected pocket.

## Feature families

IEIM computes element-pair features using C, N, O, and S protein atoms.
For each element pair, it computes statistics of:

- surface area
- volume
- mean curvature
- Gaussian curvature
- minimum principal curvature
- maximum principal curvature

The default full descriptor combines global and local pocket surface features.

## Interpretation of similarity metrics

- Cosine similarity removes feature-vector magnitude and emphasizes
  size-normalized geometric pattern.
- Euclidean distance on raw IEIM features is magnitude-sensitive and captures
  both geometric pattern and pocket-size or feature-magnitude effects.
- Size-only control uses only the difference in feature-vector L2 norms.

The Kahraman benchmark shows that Euclidean IEIM performance is largely
size-aware, whereas cosine IEIM better reflects shape-normalized similarity.
