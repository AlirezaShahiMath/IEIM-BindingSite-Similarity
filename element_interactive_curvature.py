#!/usr/bin/env python

"""
Introduction:
    EIC: element interactive curvature

Author:
    Masud Rana (mrana10@kennesaw.edu)
    
Date last modified:
    July 25, 2025

"""

import sympy as sp
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class KernelFunction:
    def __init__(self, kernel_type='exponential'):
        # Symbols for symbolic differentiation
        self.dx, self.dy, self.dz = sp.symbols('dx dy dz', real=True)
        self.eta = sp.Symbol('eta', real=True)
        self.kappa = sp.Symbol('kappa', real=True)

        self.r = sp.sqrt(self.dx**2 + self.dy**2 + self.dz**2)

        if kernel_type[0].lower() == 'e':
            self.kernel_expr = sp.exp(-(self.r / self.eta) ** self.kappa)
        elif kernel_type[0].lower() == 'l':
            self.kernel_expr = 1 / (1 + self.r / self.eta) ** self.kappa
        else:
            raise ValueError("Unknown kernel type")

        # Derivatives
        self.grad_expr = [sp.diff(self.kernel_expr, var) for var in (self.dx, self.dy, self.dz)]
        self.hess_expr = [
            sp.diff(self.grad_expr[0], self.dx),  # d²φ/dx²
            sp.diff(self.grad_expr[1], self.dy),  # d²φ/dy²
            sp.diff(self.grad_expr[2], self.dz),  # d²φ/dz²
            sp.diff(self.grad_expr[0], self.dy),  # d²φ/dxdy
            sp.diff(self.grad_expr[0], self.dz),  # d²φ/dxdz
            sp.diff(self.grad_expr[1], self.dz),  # d²φ/dydz
        ]

        # Lambdify all
        vars_all = (self.dx, self.dy, self.dz, self.eta, self.kappa)
        self.phi_fn = sp.lambdify(vars_all, self.kernel_expr, 'numpy')
        self.grad_fn = sp.lambdify(vars_all, self.grad_expr, 'numpy')
        self.hess_fn = sp.lambdify(vars_all, self.hess_expr, 'numpy')


class ElementInteractiveCurvature:
    def __init__(self, kernel, ligand_vdW=1.7, protein_vdW=1.4):
        self.ligand_vdW = ligand_vdW
        self.protein_vdW = protein_vdW
        self.kernel = kernel

    def compute_curvatures(self, d):
        rho_x, rho_y, rho_z = d["rho_x"], d["rho_y"], d["rho_z"]
        rho_xx, rho_yy, rho_zz = d["rho_xx"], d["rho_yy"], d["rho_zz"]
        rho_xy, rho_xz, rho_yz = d["rho_xy"], d["rho_xz"], d["rho_yz"]

        g = rho_x**2 + rho_y**2 + rho_z**2

        if g == 0:
            return {"H": 0, "K": 0, "kappa_min": 0, "kappa_max": 0}

        # Mean curvature H
        H = (1 / (2 * g**1.5)) * (
            2 * rho_x * rho_y * rho_xy +
            2 * rho_x * rho_z * rho_xz +
            2 * rho_y * rho_z * rho_yz -
            (rho_y**2 + rho_z**2) * rho_xx -
            (rho_x**2 + rho_z**2) * rho_yy -
            (rho_x**2 + rho_y**2) * rho_zz
        )

        # Gaussian curvature K
        K = (1 / g**2) * (
            rho_z**2 * rho_xx * rho_yy +
            rho_x**2 * rho_yy * rho_zz +
            rho_y**2 * rho_xx * rho_zz +
            2 * rho_x * rho_y * rho_xz * rho_yz +
            2 * rho_y * rho_z * rho_xy * rho_xz +
            2 * rho_x * rho_z * rho_xy * rho_yz -
            2 * rho_x * rho_y * rho_xy * rho_zz -
            2 * rho_y * rho_z * rho_yz * rho_xx -
            2 * rho_x * rho_z * rho_xz * rho_yy -
            rho_x**2 * rho_yz**2 -
            rho_y**2 * rho_xz**2 -
            rho_z**2 * rho_xy**2
        )

        sqrt_term = np.sqrt(max(H**2 - K, 0.0))
        return {
            "H": H,
            "K": K,
            "kappa_min": H - sqrt_term,
            "kappa_max": H + sqrt_term
        }

    def evaluate_all_vectorized(self, ligand_atoms, protein_atoms, tau_val, kappa_val):
        ligand_atoms = np.asarray(ligand_atoms)
        protein_atoms = np.asarray(protein_atoms)

        eta_val = tau_val * (self.ligand_vdW + self.protein_vdW)

        # Difference matrices (N_ligand, N_protein)
        dx = ligand_atoms[:, 0][:, None] - protein_atoms[:, 0]
        dy = ligand_atoms[:, 1][:, None] - protein_atoms[:, 1]
        dz = ligand_atoms[:, 2][:, None] - protein_atoms[:, 2]

        # Evaluate phi, gradients, and Hessians
        phi = self.kernel.phi_fn(dx, dy, dz, eta_val, kappa_val)
        grad = self.kernel.grad_fn(dx, dy, dz, eta_val, kappa_val)
        hess = self.kernel.hess_fn(dx, dy, dz, eta_val, kappa_val)

        # Sum contributions from protein atoms (axis=1)
        rho = np.sum(phi, axis=1)
        rho_x = np.sum(grad[0], axis=1)
        rho_y = np.sum(grad[1], axis=1)
        rho_z = np.sum(grad[2], axis=1)
        rho_xx = np.sum(hess[0], axis=1)
        rho_yy = np.sum(hess[1], axis=1)
        rho_zz = np.sum(hess[2], axis=1)
        rho_xy = np.sum(hess[3], axis=1)
        rho_xz = np.sum(hess[4], axis=1)
        rho_yz = np.sum(hess[5], axis=1)

        # Evaluate curvature per point
        data = []
        for i in range(len(rho)):
            d = {
                "rho": rho[i],
                "rho_x": rho_x[i], "rho_y": rho_y[i], "rho_z": rho_z[i],
                "rho_xx": rho_xx[i], "rho_yy": rho_yy[i], "rho_zz": rho_zz[i],
                "rho_xy": rho_xy[i], "rho_xz": rho_xz[i], "rho_yz": rho_yz[i],
            }
            curv = self.compute_curvatures(d)
            row = {
                "x": ligand_atoms[i, 0],
                "y": ligand_atoms[i, 1],
                "z": ligand_atoms[i, 2],
                 **curv
            }
            data.append(row)
    
        return pd.DataFrame(data)
