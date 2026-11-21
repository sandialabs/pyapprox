"""PeriodicRows: value and derivative matching on periodic pairs of sides.

2D Chebyshev mesh, 5 x 5 nodes on [-1, 1]^2 (the reference square), x
pair sides 0 and 1, y pair sides 2 and 3. The old path
(``PeriodicBC.apply_to_*`` in pair order, from zero) is the reference.
"""

import math
import pickle

import numpy as np
import pytest

from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import ClosureRowsProtocol
from pyapprox.pde.collocation.boundary.periodic import PeriodicBC
from pyapprox.pde.collocation.boundary.periodic_rows import (
    PeriodicPart,
    PeriodicRows,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.util.backends.protocols import Array, Backend

_N1D = 5
_NPTS = _N1D * _N1D


def _mesh(bkd: Backend[Array]) -> TransformedMesh2D[Array]:
    return TransformedMesh2D(_N1D, _N1D, bkd)


def _pair(bkd: Backend[Array], dim: int, component: int = 0) -> PeriodicPart[Array]:
    mesh = _mesh(bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    return PeriodicPart(
        mesh.boundary_indices(2 * dim),
        mesh.boundary_indices(2 * dim + 1),
        basis.derivative_matrix(1, dim),
        component,
    )


def _torus(bkd: Backend[Array]) -> PeriodicRows[Array]:
    return PeriodicRows(
        bkd,
        StateLayout.uniform(_mesh(bkd).reference_points().shape[1], 1),
        _mesh(bkd).reference_points(),
        [_pair(bkd, 0), _pair(bkd, 1)],
    )


def _state(bkd: Backend[Array], n: int = _NPTS) -> Array:
    return bkd.asarray(np.random.default_rng(3).standard_normal(n))


class TestPeriodicRows:
    def test_equals_old_periodic_rows(self, bkd: Backend[Array]) -> None:
        """c and c_u equal the old PeriodicBC rows of the x pair then the
        y pair, corners included, to 1e-14."""
        mesh = _mesh(bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        state = _state(bkd)
        old_residual, old_jacobian = bkd.zeros((_NPTS,)), bkd.zeros((_NPTS, _NPTS))
        for dim in (0, 1):
            old = PeriodicBC(
                bkd,
                mesh.boundary_indices(2 * dim),
                mesh.boundary_indices(2 * dim + 1),
                basis.derivative_matrix(1, dim),
            )
            old_residual = old.apply_to_residual(old_residual, state, 0.0)
            old_jacobian = old.apply_to_jacobian(old_jacobian, state, 0.0)
        rows = _torus(bkd)
        assert isinstance(rows, ClosureRowsProtocol)
        bkd.assert_allclose(
            rows.residual(state, 0.0), old_residual, rtol=1e-14, atol=1e-13
        )
        bkd.assert_allclose(
            rows.jacobian(state, 0.0), old_jacobian, rtol=1e-14, atol=1e-13
        )

    def test_periodic_state_closes_the_rows(self, bkd: Backend[Array]) -> None:
        """u = (1 - x^2)^2 + (1 - y^2)^2 + x^2 y^2 (1 - x^2)^2 (1 - y^2)^2
        is even in x and y with zero normal derivatives on every side, so
        both relations hold to round-off; u + x does not satisfy them."""
        mesh = _mesh(bkd)
        x, y = mesh.points()[0], mesh.points()[1]
        periodic = (1 - x**2) ** 2 + (1 - y**2) ** 2 + (1 - x**2) ** 2 * (1 - y**2) ** 2
        rows = _torus(bkd)
        assert bkd.to_float(bkd.max(bkd.abs(rows.residual(periodic, 0.0)))) < 1e-12
        broken = periodic + x
        assert bkd.to_float(bkd.max(bkd.abs(rows.residual(broken, 0.0)))) > 1e-2

    def test_residual_is_the_jacobian_applied(self, bkd: Backend[Array]) -> None:
        """The rows are linear: c(u) = c_u u, and c_u does not depend on u."""
        rows = _torus(bkd)
        state = _state(bkd)
        jacobian = rows.jacobian(state, 0.0)
        bkd.assert_allclose(rows.residual(state, 0.0), jacobian @ state, rtol=1e-14)
        bkd.assert_allclose(
            rows.jacobian(bkd.zeros((_NPTS,)), 0.0), jacobian, rtol=0.0, atol=0.0
        )

    def test_component_offset(self, bkd: Backend[Array]) -> None:
        """A pair on component 1 of a two-component state replaces rows
        offset by npts and reads only component 1's columns."""
        mesh = _mesh(bkd)
        rows = PeriodicRows(
            bkd,
            StateLayout.uniform(mesh.reference_points().shape[1], 2),
            mesh.reference_points(),
            [_pair(bkd, 0, 1)],
        )
        single = PeriodicRows(
            bkd,
            StateLayout.uniform(mesh.reference_points().shape[1], 1),
            mesh.reference_points(),
            [_pair(bkd, 0)],
        )
        replaced = rows.rows().indices()
        assert all(i >= _NPTS for i in replaced)
        assert [i - _NPTS for i in replaced] == list(single.rows().indices())
        jacobian = rows.jacobian(bkd.zeros((2 * _NPTS,)), 0.0)
        bkd.assert_allclose(
            jacobian[_NPTS:, _NPTS:],
            single.jacobian(bkd.zeros((_NPTS,)), 0.0),
            rtol=0.0,
            atol=0.0,
        )
        bkd.assert_allclose(jacobian[:, :_NPTS], bkd.zeros((2 * _NPTS, _NPTS)))

    def test_declared_derivatives(self, bkd: Backend[Array]) -> None:
        bundle = _torus(bkd).derivatives()
        assert bundle.param_jacobian is STATED_ZERO
        assert bundle.state_curvature is STATED_ZERO
        assert bundle.mixed is STATED_ZERO
        assert bundle.param_curvature is STATED_ZERO
        assert _torus(bkd).is_time_invariant()

    def test_polar_theta_pairing_accepted(self, bkd: Backend[Array]) -> None:
        """The annulus' theta sides pair node by node in reference
        coordinates."""
        transform = PolarTransform(
            r_bounds=(1.0, 2.0), theta_bounds=(0.0, 2 * math.pi), bkd=bkd
        )
        mesh = TransformedMesh2D(_N1D, _N1D, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        part = PeriodicPart(
            mesh.boundary_indices(2),
            mesh.boundary_indices(3),
            basis.derivative_matrix(1, 1),
        )
        PeriodicRows(
            bkd,
            StateLayout.uniform(mesh.reference_points().shape[1], 1),
            mesh.reference_points(),
            [part],
        )

    def test_refusals(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        reference = mesh.reference_points()
        left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
        derivative = ChebyshevBasis2D(mesh, bkd).derivative_matrix(1, 0)
        with pytest.raises(ValueError, match="at least one part"):
            PeriodicRows(bkd, StateLayout.uniform(reference.shape[1], 1), reference, [])
        with pytest.raises(ValueError, match="differ in length"):
            PeriodicRows(
                bkd,
                StateLayout.uniform(reference.shape[1], 1),
                reference,
                [PeriodicPart(left, right[1:], derivative)],
            )
        with pytest.raises(ValueError, match="share nodes"):
            PeriodicRows(
                bkd,
                StateLayout.uniform(reference.shape[1], 1),
                reference,
                [PeriodicPart(left, left, derivative)],
            )
        with pytest.raises(ValueError, match="periodic reference coordinate"):
            PeriodicRows(
                bkd,
                StateLayout.uniform(reference.shape[1], 1),
                reference,
                [PeriodicPart(left, bkd.flip(right), derivative)],
            )
        with pytest.raises(ValueError, match="periodic reference coordinate"):
            # the right side shifted by one node: the pairs differ in both
            # x and y
            shifted = bkd.hstack([right[1:], right[:1]])
            PeriodicRows(
                bkd,
                StateLayout.uniform(reference.shape[1], 1),
                reference,
                [PeriodicPart(left, shifted, derivative)],
            )
        with pytest.raises(ValueError, match="derivative matrix"):
            PeriodicRows(
                bkd,
                StateLayout.uniform(reference.shape[1], 1),
                reference,
                [PeriodicPart(left, right, derivative[:-1])],
            )
        with pytest.raises(ValueError, match="state must have shape"):
            _torus(bkd).residual(bkd.zeros((_NPTS + 1,)), 0.0)

    def test_pickle_round_trip(self, bkd: Backend[Array]) -> None:
        rows = _torus(bkd)
        loaded = pickle.loads(pickle.dumps(rows))
        state = _state(bkd)
        bkd.assert_allclose(
            loaded.residual(state, 0.0), rows.residual(state, 0.0), rtol=1e-14
        )
