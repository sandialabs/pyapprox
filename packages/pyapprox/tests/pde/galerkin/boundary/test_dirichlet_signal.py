"""Galerkin DirichletBC as a selection of DOFs times a BoundarySignal."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyapprox.pde.boundary import (
    BoundarySignal,
    DirichletConstraintSet,
    DofSignal,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import CallableDirichletBC, DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.util.backends.protocols import Array, Backend


def _ramp(x: NDArray[Any], t: float) -> NDArray[Any]:
    return (1.0 + x[1]) * t**2


def _ramp_dot(x: NDArray[Any], t: float) -> NDArray[Any]:
    return 2.0 * (1.0 + x[1]) * t


def _ramp_ddot(x: NDArray[Any], t: float) -> NDArray[Any]:
    return 2.0 * (1.0 + x[1])


def _steady(x: NDArray[Any]) -> NDArray[Any]:
    return 1.0 + x[1]


_DOF_SCALE = np.array([1.0, 3.0])


def _dof_ramp(t: float) -> NDArray[Any]:
    return _DOF_SCALE * t**2


def _dof_ramp_dot(t: float) -> NDArray[Any]:
    return 2.0 * _DOF_SCALE * t


def _dof_ramp_ddot(t: float) -> NDArray[Any]:
    return 2.0 * _DOF_SCALE


def _basis(bkd: Backend[Array]) -> LagrangeBasis[Array]:
    mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0, 1], [0, 1]], bkd=bkd)
    return LagrangeBasis(mesh, degree=1)


def _left_y(basis: LagrangeBasis[Array], bkd: Backend[Array]) -> Array:
    """y coordinates of the left-boundary DOFs, in DOF order."""
    coords = basis.dof_coordinates()
    return coords[1, basis.get_dofs("left")]


class TestDirichletSignal:
    def test_orders_round_trip_through_constraint_set(
        self, bkd: Backend[Array]
    ) -> None:
        basis = _basis(bkd)
        signal = BoundarySignal(
            TimeDependent(_ramp),
            [TimeDependent(_ramp_dot), TimeDependent(_ramp_ddot)],
        )
        bc = DirichletBC(basis, "left", signal, bkd)
        cs = DirichletConstraintSet([bc], basis.ndofs(), bkd)
        assert not cs.is_time_invariant()
        # The left DOFs are the set's only DOFs; order both ways to compare.
        order = bkd.argsort(basis.get_dofs("left"))
        y = _left_y(basis, bkd)[order]
        first = cs.values_derivative(1)
        second = cs.values_derivative(2)
        assert first is not None and second is not None
        bkd.assert_allclose(first(0.5), 2.0 * (1.0 + y) * 0.5)
        bkd.assert_allclose(second(0.5), 2.0 * (1.0 + y))
        assert cs.values_derivative(3) is None

    def test_declared_steady_callable_is_time_invariant(
        self, bkd: Backend[Array]
    ) -> None:
        basis = _basis(bkd)
        bc = DirichletBC(basis, "left", TimeIndependent(_steady), bkd)
        assert bc.is_time_invariant()
        derivative = bc.constrained_values_derivative(1)
        assert derivative is not None
        bkd.assert_allclose(
            derivative(3.0), bkd.zeros((len(basis.get_dofs("left")),))
        )

    def test_declared_time_dependent_without_signal_has_no_derivative(
        self, bkd: Backend[Array]
    ) -> None:
        """Only a signal carries derivatives; none is ever fabricated."""
        bc = DirichletBC(_basis(bkd), "left", TimeDependent(_ramp), bkd)
        assert not bc.is_time_invariant()
        assert bc.constrained_values_derivative(1) is None

    def test_bare_time_callable_is_rejected(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match="ambiguous"):
            DirichletBC(_basis(bkd), "left", _ramp, bkd)

    def test_callable_dirichlet_orders(self, bkd: Backend[Array]) -> None:
        bc = CallableDirichletBC(
            [0, 2],
            DofSignal(_dof_ramp, [_dof_ramp_dot, _dof_ramp_ddot]),
            bkd,
        )
        cs = DirichletConstraintSet([bc], 4, bkd)
        first = cs.values_derivative(1)
        second = cs.values_derivative(2)
        assert first is not None and second is not None
        bkd.assert_allclose(first(0.5), bkd.asarray([1.0, 3.0]))
        bkd.assert_allclose(second(0.5), bkd.asarray([2.0, 6.0]))
        assert cs.values_derivative(3) is None
        restored = pickle.loads(pickle.dumps(bc))
        restored_second = restored.constrained_values_derivative(2)
        assert restored_second is not None
        bkd.assert_allclose(restored_second(0.5), bkd.asarray([2.0, 6.0]))

    def test_callable_dirichlet_plain_function_has_no_derivative(
        self, bkd: Backend[Array]
    ) -> None:
        bc = CallableDirichletBC([0, 2], _dof_ramp, bkd)
        bkd.assert_allclose(
            bc.constrained_values(0.5), bkd.asarray([0.25, 0.75])
        )
        assert bc.constrained_values_derivative(1) is None
        with pytest.raises(ValueError, match="order"):
            bc.constrained_values_derivative(0)

    def test_pickle_round_trip(self, bkd: Backend[Array]) -> None:
        basis = _basis(bkd)
        bc = DirichletBC(
            basis,
            "left",
            BoundarySignal(TimeDependent(_ramp), [TimeDependent(_ramp_dot)]),
            bkd,
        )
        restored = pickle.loads(pickle.dumps(bc))
        bkd.assert_allclose(
            restored.constrained_values(0.7), bc.constrained_values(0.7)
        )
        derivative = bc.constrained_values_derivative(1)
        restored_derivative = restored.constrained_values_derivative(1)
        assert derivative is not None and restored_derivative is not None
        bkd.assert_allclose(restored_derivative(0.7), derivative(0.7))
