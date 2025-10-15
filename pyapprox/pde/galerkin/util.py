from typing import List

import numpy as np
from skfem import (
    MeshLine,
    MeshQuad,
    Mesh2D,
    Mesh3D,
    MeshLine1DG,
    ElementLineP1,
    ElementLineP2,
    ElementQuad1,
    ElementQuad2,
)
from skfem.helpers import dot
from skfem.mesh import Mesh
from skfem.element import Element


def _get_mesh(
    bounds: List[float], nrefine: int, periodic: bool = False, nx: int = 3
) -> Mesh:
    nphys_vars = len(bounds) // 2
    if nphys_vars > 2:
        raise ValueError("Only 1D and 2D meshes supported")

    if nphys_vars == 1:
        if not periodic:
            mesh = (
                MeshLine.init_tensor(np.linspace(*bounds, nx))
                .refined(nrefine)
                .with_boundaries(
                    {
                        "left": lambda x: x[0] == bounds[0],
                        "right": lambda x: x[0] == bounds[1],
                    }
                )
            )
            return mesh
        # can use refine with periodic mesh
        mesh = MeshLine1DG.init_tensor(
            np.linspace(*bounds, 2 ** (nrefine + 1) + 1), periodic=[0]
        )
        return mesh

    if periodic:
        raise ValueError("Periodic not yet wrapped for 2d mesh")
    mesh = (
        MeshQuad.init_tensor(
            np.linspace(*bounds[:2], nx), np.linspace(*bounds[2:], nx)
        )
        .refined(nrefine)
        .with_boundaries(
            {
                "left": lambda x: x[0] == bounds[0],
                "right": lambda x: x[0] == bounds[1],
                "bottom": lambda x: x[1] == bounds[2],
                "top": lambda x: x[1] == bounds[3],
            }
        )
    )
    return mesh


def _get_element(mesh: Mesh, order: int) -> Element:

    if order > 2:
        raise ValueError(f"order {order} not supported")
    if isinstance(mesh, Mesh3D):
        raise ValueError("Only 1D and 2D meshes supported")

    nphys_vars = 2 if isinstance(mesh, Mesh2D) else 1

    if nphys_vars == 1:
        if order == 1:
            return ElementLineP1()
        return ElementLineP2()
    if order == 1:
        return ElementQuad1()
    return ElementQuad2()


def _vector_fun_to_skfem_vector_fun(vel_forc_fun, x):
    vals = vel_forc_fun(x)
    return np.stack([vals[:, ii] for ii in range(vals.shape[1])])


def _vector_forcing(v, w):
    return dot(w["forc"], v)


def _forcing(v, w):
    return w["forc"] * v


def _robin(u, v, w):
    return w["alpha"] * u * v


def _robin_prev_sol(v, w):
    return w["alpha"] * w["u_prev"] * v
