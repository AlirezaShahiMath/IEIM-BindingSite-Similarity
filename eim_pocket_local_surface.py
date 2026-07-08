#!/usr/bin/env python

"""
Element Interactive Manifold - Pocket-Only Local Surface (Project 2)

Key fix:
    Robust removal of self-interactions using coordinate comparison
    instead of atom indices.

This ensures no atom contributes to its own local neighborhood.
"""

import os
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

import numba_utils_methods as nmb
from element_interactive_density import ElementInteractiveDensity
from element_interactive_curvature import *


class EIM_Pocket_Local_Surface():
    def __init__(self, path, pdbid,
                 kernel_type='exponential',
                 kernel_tau=1.0,
                 kernel_power=2.0,
                 cutoff=7.0,
                 isovalue=0.25):

        self.path = path
        self.pdbid = pdbid
        self.cutoff = cutoff
        self.isovalue = isovalue

        self.kernel_type = kernel_type
        self.kernel_tau = kernel_tau
        self.kernel_power = kernel_power

        self.pocket_atom_type = ['C', 'N', 'O', 'S']

        self.atom_type_radii = {
            'C': 1.7,
            'N': 1.55,
            'O': 1.52,
            'S': 1.8,
        }

        self.num_stat_measure = 6
        self.mesh_size = 0.5

        self.kernel = KernelFunction(self.kernel_type)

    # --------------------------------------------------------------------------
    def pocket_pdb_to_df(self, pdb_file):

        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_file)

        df_atom   = ppdb.df.get('ATOM', pd.DataFrame())
        df_hetatm = ppdb.df.get('HETATM', pd.DataFrame())
        ppdb_all  = pd.concat([df_atom, df_hetatm], ignore_index=True)

        if ppdb_all.empty:
            return pd.DataFrame(columns=['ATOM_INDEX', 'ATOM_ELEMENT', 'X', 'Y', 'Z'])

        ppdb_df = ppdb_all[
            ppdb_all['element_symbol'].isin(self.pocket_atom_type)
        ]

        return pd.DataFrame({
            'ATOM_INDEX':   ppdb_df['atom_number'].values,
            'ATOM_ELEMENT': ppdb_df['element_symbol'].values,
            'X':            ppdb_df['x_coord'].values,
            'Y':            ppdb_df['y_coord'].values,
            'Z':            ppdb_df['z_coord'].values,
        })

    # --------------------------------------------------------------------------
    def get_stats(self, x):
        x = np.asarray(x).astype(float)
        return np.round(np.array([
            np.sum(x),
            np.mean(x),
            np.median(x),
            np.std(x),
            np.min(x),
            np.max(x)
        ]), 5)

    # --------------------------------------------------------------------------
    def get_features(self):

        n_types = len(self.pocket_atom_type)
        num_pairwise_features = 6 * self.num_stat_measure
        num_features = n_types * n_types * num_pairwise_features

        folder = str(self.pdbid)
        pocket_path = os.path.join(self.path, folder, f"{folder}_EIM_pocket.pdb")

        if not os.path.exists(pocket_path):
            return None

        pocket_df = self.pocket_pdb_to_df(pocket_path)
        if pocket_df.empty:
            return None

        pair_wise_features = np.zeros((n_types * n_types, num_pairwise_features))
        atomic_pair_count = 0

        for src_type in self.pocket_atom_type:

            src_df = pocket_df[pocket_df['ATOM_ELEMENT'] == src_type]
            src_xyz = src_df[['X', 'Y', 'Z']].values
            src_vdW = self.atom_type_radii[src_type]

            for nbr_type in self.pocket_atom_type:

                nbr_df = pocket_df[pocket_df['ATOM_ELEMENT'] == nbr_type]
                nbr_xyz = nbr_df[['X', 'Y', 'Z']].values
                nbr_vdW = self.atom_type_radii[nbr_type]

                if len(src_xyz) == 0 or len(nbr_xyz) == 0:
                    atomic_pair_count += 1
                    continue

                atomic_surface_area = []
                atomic_surface_vol  = []
                curvature_dfs       = []

                for src_atom_idx in range(len(src_xyz)):

                    center = src_xyz[src_atom_idx]
                    center_2d = center.reshape(1, 3)

                    # ----------------------------------------------------------
                    # DISTANCE FILTER
                    # ----------------------------------------------------------
                    dmat = cdist(center_2d, nbr_xyz)
                    p_idxs = np.where(dmat < self.cutoff)[1]

                    # ----------------------------------------------------------
                    # FIXED SELF-REMOVAL (IMPORTANT)
                    # ----------------------------------------------------------
                    if src_type == nbr_type:

                        nbr_subset = nbr_xyz[p_idxs]

                        keep_mask = []
                        for pt in nbr_subset:
                            is_same = np.all(np.isclose(pt, center, atol=1e-6))
                            keep_mask.append(not is_same)

                        p_idxs = p_idxs[np.array(keep_mask)]

                    if len(p_idxs) == 0:
                        atomic_surface_area.append(0.0)
                        atomic_surface_vol.append(0.0)
                        continue

                    nbr_in_ball = nbr_xyz[p_idxs]

                    # ----------------------------------------------------------
                    # LOCAL GRID
                    # ----------------------------------------------------------
                    bf = 2.0 + self.cutoff
                    mins = center - bf
                    maxs = center + bf

                    h = self.mesh_size
                    x = np.arange(mins[0], maxs[0], h)
                    y = np.arange(mins[1], maxs[1], h)
                    z = np.arange(mins[2], maxs[2], h)

                    nx, ny, nz = len(x), len(y), len(z)

                    atoms_in_ball = np.vstack([nbr_in_ball, center_2d])

                    # ----------------------------------------------------------
                    # DENSITY
                    # ----------------------------------------------------------
                    eid = ElementInteractiveDensity(
                        kernel_type=self.kernel_type,
                        kernel_tau=self.kernel_tau,
                        kernel_power=self.kernel_power,
                        ligand_vdW=src_vdW,
                        protein_vdW=nbr_vdW
                    )

                    rho = eid.main(nx, ny, nz, x, y, z, atoms_in_ball)
                    rho_max = rho.max()
                    rho_bar = rho / rho_max if rho_max != 0 else rho

                    # ----------------------------------------------------------
                    # SURFACE / VOLUME
                    # ----------------------------------------------------------
                    f = rho_bar - self.isovalue

                    N_x, N_y, N_z = nmb.normal_vector_components(nx, ny, nz, h, f)

                    eisa, eisv = nmb.surface_area_and_volume(
                        nx, ny, nz,
                        N_x, N_y, N_z,
                        x, y, z, h, f, self.isovalue
                    )

                    atomic_surface_area.append(eisa)
                    atomic_surface_vol.append(eisv)

                    # ----------------------------------------------------------
                    # CURVATURE
                    # ----------------------------------------------------------
                    eic = ElementInteractiveCurvature(
                        self.kernel,
                        ligand_vdW=src_vdW,
                        protein_vdW=nbr_vdW
                    )

                    curv_df = eic.evaluate_all_vectorized(
                        center_2d,
                        nbr_in_ball,
                        tau_val=self.kernel_tau,
                        kappa_val=self.kernel_power
                    )

                    curvature_dfs.append(curv_df)

                # --------------------------------------------------------------
                # AGGREGATION
                # --------------------------------------------------------------
                sa_stats  = self.get_stats(atomic_surface_area)
                vol_stats = self.get_stats(atomic_surface_vol)

                if curvature_dfs:
                    all_curv = pd.concat(curvature_dfs, axis=0)

                    curv_feats = np.concatenate([
                        self.get_stats(all_curv['H'].values),
                        self.get_stats(all_curv['K'].values),
                        self.get_stats(all_curv['kappa_min'].values),
                        self.get_stats(all_curv['kappa_max'].values),
                    ])
                else:
                    curv_feats = np.zeros(4 * self.num_stat_measure)

                features = np.concatenate([
                    sa_stats,
                    vol_stats,
                    curv_feats
                ])

                pair_wise_features[atomic_pair_count] = features
                atomic_pair_count += 1

        return pair_wise_features.reshape(1, num_features)