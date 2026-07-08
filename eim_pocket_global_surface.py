#!/usr/bin/env python

"""
Element Interactive Manifold - Pocket-Only Global Surface (Project 2)

Key fix:
    Proper removal of self-interactions when src_type == nbr_type
    using coordinate-aware masking instead of index mismatch.

This avoids incorrect inclusion of identical atoms when the same
atom appears in both src_xyz and nbr_xyz arrays.
"""

import os
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

import numba_utils_methods as nmb
from element_interactive_density import ElementInteractiveDensity
from element_interactive_curvature import *


class EIM_Pocket_Global_Surface():
    def __init__(self, path, pdbid,
                 kernel_type='exponential',
                 kernel_tau=1.0,
                 kernel_power=2.0,
                 cutoff=12.0):

        self.path = path
        self.pdbid = pdbid
        self.cutoff = cutoff

        self.kernel_type = kernel_type
        self.kernel_tau = kernel_tau
        self.kernel_power = kernel_power

        # 4 × 4 element pairs
        self.pocket_atom_type = ['C', 'N', 'O', 'S']

        self.atom_type_radii = {
            'C': 1.7,
            'N': 1.55,
            'O': 1.52,
            'S': 1.8,
        }

        self.num_stat_measure = 6
        self.mesh_size = 0.5
        self.isovalue_list = np.arange(0.05, 0.8, 0.05)

        self.kernel = KernelFunction(self.kernel_type)

    # --------------------------------------------------------------------------
    # PDB READER
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
    # STATS
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
    # MAIN FEATURE FUNCTION
    # --------------------------------------------------------------------------
    def get_features(self):

        n_types = len(self.pocket_atom_type)
        num_pairwise_features = 6 * self.num_stat_measure  # 36
        num_features = n_types * n_types * num_pairwise_features  # 576

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

                # --------------------------------------------------------------
                # DISTANCE FILTER
                # --------------------------------------------------------------
                dmat = cdist(src_xyz, nbr_xyz)
                mask = dmat < self.cutoff

                p_index = np.unique(np.where(mask)[1])

                # --------------------------------------------------------------
                # FIXED SELF-INTERACTION REMOVAL (IMPORTANT)
                # --------------------------------------------------------------
                if src_type == nbr_type:

                    nbr_subset = nbr_xyz[p_index]

                    # remove points that match ANY source atom
                    keep_mask = []

                    for pt in nbr_subset:
                        is_same = np.any(np.all(np.isclose(src_xyz, pt, atol=1e-6), axis=1))
                        keep_mask.append(not is_same)

                    p_index = p_index[np.array(keep_mask)]

                if len(p_index) == 0:
                    atomic_pair_count += 1
                    continue

                # --------------------------------------------------------------
                # BUILD INTERACTION CLOUD
                # --------------------------------------------------------------
                ei_xyz = np.concatenate((src_xyz, nbr_xyz[p_index]))

                # bounding box
                bf = 2.0
                mins = ei_xyz.min(axis=0) - bf
                maxs = ei_xyz.max(axis=0) + bf

                h = self.mesh_size
                x = np.arange(mins[0], maxs[0], h)
                y = np.arange(mins[1], maxs[1], h)
                z = np.arange(mins[2], maxs[2], h)

                nx, ny, nz = len(x), len(y), len(z)

                # --------------------------------------------------------------
                # DENSITY
                # --------------------------------------------------------------
                eid = ElementInteractiveDensity(
                    kernel_type=self.kernel_type,
                    kernel_tau=self.kernel_tau,
                    kernel_power=self.kernel_power,
                    ligand_vdW=src_vdW,
                    protein_vdW=nbr_vdW
                )

                rho = eid.main(nx, ny, nz, x, y, z, ei_xyz)
                rho_max = rho.max()
                rho_bar = rho / rho_max if rho_max != 0 else rho

                # --------------------------------------------------------------
                # SURFACE + VOLUME
                # --------------------------------------------------------------
                area_feats = np.zeros(len(self.isovalue_list))
                vol_feats  = np.zeros(len(self.isovalue_list))

                for k, iso in enumerate(self.isovalue_list):
                    f = rho_bar - iso

                    N_x, N_y, N_z = nmb.normal_vector_components(nx, ny, nz, h, f)

                    eisa, eisv = nmb.surface_area_and_volume(
                        nx, ny, nz,
                        N_x, N_y, N_z,
                        x, y, z, h, f, iso
                    )

                    area_feats[k] = eisa
                    vol_feats[k]  = eisv

                # --------------------------------------------------------------
                # CURVATURE
                # --------------------------------------------------------------
                eic = ElementInteractiveCurvature(
                    self.kernel,
                    ligand_vdW=src_vdW,
                    protein_vdW=nbr_vdW
                )

                curv_df = eic.evaluate_all_vectorized(
                    src_xyz,
                    nbr_xyz[p_index],
                    tau_val=self.kernel_tau,
                    kappa_val=self.kernel_power
                )

                # --------------------------------------------------------------
                # FEATURE VECTOR
                # --------------------------------------------------------------
                features = np.concatenate([
                    self.get_stats(area_feats),
                    self.get_stats(vol_feats),
                    self.get_stats(curv_df['H'].values),
                    self.get_stats(curv_df['K'].values),
                    self.get_stats(curv_df['kappa_min'].values),
                    self.get_stats(curv_df['kappa_max'].values),
                ])

                pair_wise_features[atomic_pair_count] = features
                atomic_pair_count += 1

        return pair_wise_features.reshape(1, num_features)