r"""The Galerkin verification adapter's exact values at a basis's DOFs.

``exact_dof_values`` on the unit square, two quadrilaterals per side: for
the vector P2 basis and :math:`(u, v) = (x + t, 2 y)` at :math:`t =
0.5`, each DOF equals its own component's value at its location (the
DOFs are interleaved per node, so both components are checked); for the
scalar P1 basis and :math:`u = x y`, each DOF is :math:`x y`. A row count
other than the basis's components is refused.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.models.galerkin.verification import (
    Lagrange,
    StructuredBox,
    VectorLagrange,
    exact_dof_values,
)
from pyapprox.pde.verification.conditions import Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence

_SQUARE = Box([(0.0, 1.0)] * 2)
_DIRICHLET = {side: Dirichlet() for side in _SQUARE.boundaries()}


def test_vector_basis_values_by_component() -> None:
    problem = ManufacturedProblem(
        {"u": "x + t", "v": "2*y"},
        [Divergence("u", ["-u_x", "-u_y"]), Divergence("v", ["-v_x", "-v_y"])],
        _SQUARE,
        _DIRICHLET,
    )
    (mesh,) = StructuredBox([2]).meshes(_SQUARE)
    basis = VectorLagrange(2).build(mesh)
    values = exact_dof_values(problem, basis, ("u", "v"), time=0.5)
    coords = np.asarray(basis.dof_coordinates())
    components = np.asarray(basis.dof_components(np.arange(basis.ndofs())))
    assert values.shape == (basis.ndofs(),)
    assert set(components.tolist()) == {0, 1}
    expected = np.where(components == 0, coords[0] + 0.5, 2.0 * coords[1])
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-14)


def test_scalar_basis_values() -> None:
    problem = ManufacturedProblem(
        {"u": "x*y"}, [Divergence("u", ["-u_x", "-u_y"])], _SQUARE, _DIRICHLET
    )
    (mesh,) = StructuredBox([2]).meshes(_SQUARE)
    basis = Lagrange(1).build(mesh)
    coords = np.asarray(basis.dof_coordinates())
    np.testing.assert_allclose(
        exact_dof_values(problem, basis, ("u",)),
        coords[0] * coords[1],
        rtol=0.0,
        atol=1e-14,
    )


def test_refuses_a_row_count_other_than_the_components() -> None:
    problem = ManufacturedProblem(
        {"u": "x*y"}, [Divergence("u", ["-u_x", "-u_y"])], _SQUARE, _DIRICHLET
    )
    (mesh,) = StructuredBox([2]).meshes(_SQUARE)
    with pytest.raises(ValueError, match="one row per component"):
        exact_dof_values(problem, VectorLagrange(1).build(mesh), ("u",))
