"""Elasticity post-processing: stress and strain recovery.

Computes element-averaged strain and stress tensors from displacement
solutions on 2D meshes (quads and triangles). Supports composite
materials with per-element Lame parameters.

Functions
---------
strain_from_displacement_2d
    Compute element-averaged strain tensor components.
stress_from_strain_2d
    Compute stress tensor from strain using plane-stress Hooke's law.
von_mises_stress_2d
    Compute element-averaged von Mises stress from a displacement solution.
"""

import numpy as np


def _shape_function_derivatives_quad(xi: float, eta: float) -> tuple[Any, ...]:
    """Bilinear quad shape function derivatives at a reference point.

    Node ordering: counter-clockwise from bottom-left.
    Node 0: (-1,-1), Node 1: (1,-1), Node 2: (1,1), Node 3: (-1,1).

    Parameters
    ----------
    xi, eta : float
        Reference coordinates in [-1, 1].

    Returns
    -------
    dN_dxi : np.ndarray, shape (4,)
    dN_deta : np.ndarray, shape (4,)
    """
    dN_dxi = (
        np.array(
            [
                -(1 - eta),
                (1 - eta),
                (1 + eta),
                -(1 + eta),
            ]
        )
        / 4.0
    )
    dN_deta = (
        np.array(
            [
                -(1 - xi),
                -(1 + xi),
                (1 + xi),
                (1 - xi),
            ]
        )
        / 4.0
    )
    return dN_dxi, dN_deta


def _shape_function_derivatives_tri() -> tuple[Any, ...]:
    """Linear triangle shape function derivatives (constant).

    Node ordering: standard counter-clockwise.

    Returns
    -------
    dN_dxi : np.ndarray, shape (3,)
    dN_deta : np.ndarray, shape (3,)
    """
    dN_dxi = np.array([-1.0, 1.0, 0.0])
    dN_deta = np.array([-1.0, 0.0, 1.0])
    return dN_dxi, dN_deta


def strain_from_displacement_2d(
    coordx: np.ndarray,
    coordy: np.ndarray,
    connectivity: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> tuple[Any, ...]:
    """Compute element-averaged strain tensor from nodal displacements.

    Uses isoparametric mapping with shape function derivatives evaluated
    at the element center (xi=eta=0 for quads, xi=eta=1/3 for triangles).

    Parameters
    ----------
    coordx : np.ndarray, shape (nnodes,)
        Nodal x-coordinates.
    coordy : np.ndarray, shape (nnodes,)
        Nodal y-coordinates.
    connectivity : np.ndarray, shape (nelems, nodes_per_elem)
        Element connectivity (0-based). Supports 3-node triangles and
        4-node quads.
    ux : np.ndarray, shape (nnodes,)
        Nodal x-displacement.
    uy : np.ndarray, shape (nnodes,)
        Nodal y-displacement.

    Returns
    -------
    exx : np.ndarray, shape (nelems,)
        Normal strain in x.
    eyy : np.ndarray, shape (nelems,)
        Normal strain in y.
    exy : np.ndarray, shape (nelems,)
        Engineering shear strain (gamma_xy / 2).
    """
    nelems = connectivity.shape[0]
    nodes_per_elem = connectivity.shape[1]

    if nodes_per_elem == 4:
        dN_dxi, dN_deta = _shape_function_derivatives_quad(0.0, 0.0)
    elif nodes_per_elem == 3:
        dN_dxi, dN_deta = _shape_function_derivatives_tri()
    else:
        raise ValueError(
            f"Unsupported element type with {nodes_per_elem} nodes. "
            "Only 3-node triangles and 4-node quads are supported."
        )

    exx = np.empty(nelems)
    eyy = np.empty(nelems)
    exy = np.empty(nelems)

    for ie in range(nelems):
        nodes = connectivity[ie]
        xe, ye = coordx[nodes], coordy[nodes]
        ue_x, ue_y = ux[nodes], uy[nodes]

        # Jacobian of isoparametric mapping
        J11 = dN_dxi @ xe
        J12 = dN_dxi @ ye
        J21 = dN_deta @ xe
        J22 = dN_deta @ ye
        detJ = J11 * J22 - J12 * J21

        # Shape function derivatives in physical coordinates
        dN_dx = (J22 * dN_dxi - J12 * dN_deta) / detJ
        dN_dy = (-J21 * dN_dxi + J11 * dN_deta) / detJ

        exx[ie] = dN_dx @ ue_x
        eyy[ie] = dN_dy @ ue_y
        exy[ie] = 0.5 * (dN_dy @ ue_x + dN_dx @ ue_y)

    return exx, eyy, exy


def stress_from_strain_2d(
    exx: np.ndarray,
    eyy: np.ndarray,
    exy: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
) -> tuple[Any, ...]:
    """Compute plane-stress Cauchy stress from strain and Lame parameters.

    sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon

    Parameters
    ----------
    exx, eyy, exy : np.ndarray, shape (nelems,)
        Strain tensor components (exy is the tensor shear, not engineering).
    lam : np.ndarray, shape (nelems,)
        First Lame parameter per element.
    mu : np.ndarray, shape (nelems,)
        Shear modulus per element.

    Returns
    -------
    sxx : np.ndarray, shape (nelems,)
        Normal stress in x.
    syy : np.ndarray, shape (nelems,)
        Normal stress in y.
    sxy : np.ndarray, shape (nelems,)
        Shear stress.
    """
    trace_eps = exx + eyy
    sxx = lam * trace_eps + 2.0 * mu * exx
    syy = lam * trace_eps + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return sxx, syy, sxy


def von_mises_stress_2d(
    coordx: np.ndarray,
    coordy: np.ndarray,
    connectivity: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Compute element-averaged von Mises stress from nodal displacements.

    Combines strain recovery, Hooke's law, and the von Mises criterion
    for plane stress:

        sigma_vm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)

    Parameters
    ----------
    coordx : np.ndarray, shape (nnodes,)
        Nodal x-coordinates.
    coordy : np.ndarray, shape (nnodes,)
        Nodal y-coordinates.
    connectivity : np.ndarray, shape (nelems, nodes_per_elem)
        Element connectivity (0-based). Supports triangles and quads.
    ux : np.ndarray, shape (nnodes,)
        Nodal x-displacement.
    uy : np.ndarray, shape (nnodes,)
        Nodal y-displacement.
    lam : np.ndarray, shape (nelems,)
        First Lame parameter per element.
    mu : np.ndarray, shape (nelems,)
        Shear modulus per element.

    Returns
    -------
    vm : np.ndarray, shape (nelems,)
        Element-averaged von Mises stress.

    Examples
    --------
    >>> from skfem.models.elasticity import lame_parameters
    >>> lam_val, mu_val = lame_parameters(1e4, 0.3)
    >>> lam_arr = np.full(nelems, lam_val)
    >>> mu_arr = np.full(nelems, mu_val)
    >>> vm = von_mises_stress_2d(
    ...     coordx, coordy, conn, ux, uy, lam_arr, mu_arr,
    ... )
    """
    exx, eyy, exy = strain_from_displacement_2d(
        coordx,
        coordy,
        connectivity,
        ux,
        uy,
    )
    sxx, syy, sxy = stress_from_strain_2d(exx, eyy, exy, lam, mu)
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
