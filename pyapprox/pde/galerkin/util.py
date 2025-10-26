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
    Basis,
    Functional,
)
from skfem.helpers import dot
from skfem.mesh import Mesh
from skfem.element import Element


def get_mesh(
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


def get_element(mesh: Mesh, order: int) -> Element:

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


def get_subdomain_basis(
    mesh: Mesh, element: Element, subdomain_name: str
) -> Basis:
    if mesh.subdomains is None or subdomain_name not in mesh.subdomains:
        raise AttributeError(f"subdomain '{subdomain_name}' not found")
    subdomain_basis = Basis(mesh, element, elements=subdomain_name)
    return subdomain_basis


def _integrate(w) -> np.ndarray:
    return w.y


def integrate(basis: Basis, values: np.ndarray) -> float:
    return Functional(_integrate).assemble(basis, y=values)

    def integrate_on_subdomain(
        self, values: np.ndarray, subdomain_name: str
    ) -> float:
        subdomain_basis = self._physics.subdomain_basis(subdomain_name)
        return Functional(self._integrate).assemble(subdomain_basis, y=values)


def vector_forcing_linearform(v, w):
    return dot(w["forc"], v)


def forcing_linearform(v, w):
    return w["forc"] * v
