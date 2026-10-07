"""CubicHermiteBasis: DOF layout, boundary selection, interpolation.

The DOFs are ``[w_0, w'_0, w_1, w'_1, ...]``; a cubic is in the space,
so interpolating its values and slopes and evaluating between nodes
must reproduce it exactly.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.galerkin.basis import CubicHermiteBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.protocols.basis import (
    ComponentDofsBasisProtocol,
    GalerkinBasisProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

_NX, _LENGTH = 4, 2.0


def _basis(bkd: Backend[Array]) -> CubicHermiteBasis[Array]:
    return CubicHermiteBasis(StructuredMesh1D(_NX, (0.0, _LENGTH), bkd))


def _cubic(x: np.ndarray) -> np.ndarray:
    return x**3 - 2.0 * x**2 + 0.5


def _cubic_slope(x: np.ndarray) -> np.ndarray:
    return 3.0 * x**2 - 4.0 * x


def _value_and_slope(coords: np.ndarray) -> np.ndarray:
    return np.stack([_cubic(coords[0]), _cubic_slope(coords[0])])


class TestCubicHermiteBasis:
    def test_satisfies_protocols(self, numpy_bkd: NumpyBkd) -> None:
        basis = _basis(numpy_bkd)
        assert isinstance(basis, GalerkinBasisProtocol)
        assert isinstance(basis, ComponentDofsBasisProtocol)
        assert basis.ndofs() == 2 * (_NX + 1)
        assert basis.degree() == 3
        assert basis.ncomponents() == 2

    def test_dof_layout(self, bkd: Backend[Array]) -> None:
        """Each node appears twice in the DOF coordinates (value, slope)."""
        basis = _basis(bkd)
        nodes = np.linspace(0.0, _LENGTH, _NX + 1)
        bkd.assert_allclose(
            basis.dof_coordinates(),
            bkd.asarray(np.repeat(nodes, 2)[None, :]),
            rtol=1e-14,
        )

    def test_boundary_dofs_by_component(self, bkd: Backend[Array]) -> None:
        basis = _basis(bkd)
        last = 2 * _NX
        cases = (
            ("left", None, [0, 1]),
            ("left", (0,), [0]),
            ("left", (1,), [1]),
            ("right", None, [last, last + 1]),
            ("right", (0,), [last]),
        )
        for name, components, expected in cases:
            bkd.assert_allclose(
                basis.get_dofs(name, components=components),
                bkd.asarray(np.array(expected), dtype=bkd.int64_dtype()),
            )

    def test_interpolated_cubic_is_exact(self, bkd: Backend[Array]) -> None:
        basis = _basis(bkd)
        coeffs = basis.interpolate(_value_and_slope)
        nodes = np.linspace(0.0, _LENGTH, _NX + 1)
        bkd.assert_allclose(
            basis.values(coeffs), bkd.asarray(_cubic(nodes)), rtol=1e-14
        )
        bkd.assert_allclose(
            basis.slopes(coeffs), bkd.asarray(_cubic_slope(nodes)),
            rtol=1e-14, atol=1e-14,
        )
        points = np.array([[0.13, 0.9, 1.37, _LENGTH]])
        bkd.assert_allclose(
            basis.evaluate(coeffs, bkd.asarray(points)),
            bkd.asarray(_cubic(points[0])),
            rtol=1e-12,
        )

    def test_dirichlet_values_select_their_component(
        self, bkd: Backend[Array]
    ) -> None:
        """A (w, w') boundary function gives each DOF its own row."""
        basis = _basis(bkd)
        bc = DirichletBC(basis, "right", _value_and_slope, bkd)
        bkd.assert_allclose(
            bc.boundary_values(0.0),
            bkd.asarray([_cubic(_LENGTH), _cubic_slope(_LENGTH)]),
            rtol=1e-14,
        )
        pinned = DirichletBC(basis, "right", 0.0, bkd, components=(0,))
        bkd.assert_allclose(
            pinned.constrained_dofs(),
            bkd.asarray(np.array([2 * _NX]), dtype=bkd.int64_dtype()),
        )

    def test_invalid_arguments_raise(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match="1D mesh"):
            CubicHermiteBasis(
                StructuredMesh2D(2, 2, [[0.0, 1.0], [0.0, 1.0]], bkd)
            )
        basis = _basis(bkd)
        with pytest.raises(ValueError, match="out of range"):
            basis.get_dofs("left", components=(2,))
        with pytest.raises(ValueError, match="rows"):
            basis.interpolate(lambda coords: _cubic(coords[0]))
