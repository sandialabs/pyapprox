"""Snapshot collection and homogenization by a lift.

Steady setup: -(kappa u')' = 1 on [0, 1], u(0) = 1, u(1) = 2, P1 on 10
cells, kappa = exp(W p) with three smooth modes. Transient setup: the
same operator as a heat equation, u(0, t) = 0 and a moving right end
u(1, t) = 0.5 sin(4 t), from u(x, 0) = 0.3 sin(pi x), backward Euler
over [0, 0.02] with dt = 0.005.

Checks are derivation-free: a collected steady snapshot must make the
raw residual vanish on every unconstrained row and carry the boundary
data on the constrained ones; a transient snapshot's right end must
equal the boundary formula at its own time; a homogenized snapshot must
vanish on the constrained DOFs and be unchanged elsewhere. The
wrong-model controls give the homogenizer data from a different system,
or times paired with the wrong states, and require it to refuse.
"""

import math
from typing import Any, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin import (
    GalerkinParameterizedResidual,
    GalerkinTransientForwardModel,
)
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.rom import (
    AlgebraicLift,
    SnapshotSet,
    collect_steady_snapshots,
    collect_transient_snapshots,
    concatenate_snapshot_sets,
    homogenize,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.adjoint_checks import NumpyArray

_Points = NDArray[np.floating[Any]]

_SAMPLES = np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, 0.0], [0.2, 0.1, -0.4]])


def _unit_forcing(coords: _Points) -> _Points:
    return np.ones(coords.shape[1])


def _heat_forcing(coords: _Points, time: float) -> _Points:
    forcing: _Points = (1.0 + coords[0]) * (1.0 + math.sin(5.0 * time))
    return forcing


def _right_value(coords: _Points, time: float) -> _Points:
    return np.full(coords.shape[1], 0.5 * np.sin(4.0 * time))


def _right_rate(coords: _Points, time: float) -> _Points:
    return np.full(coords.shape[1], 2.0 * np.cos(4.0 * time))


def _log_kle_map(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]
) -> TransformedFieldMap[NumpyArray]:
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = np.stack(
        [0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1) for k in range(3)],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _steady_problem(
    bkd: NumpyBkd, right_value: float = 2.0
) -> Tuple[
    GalerkinSystem[NumpyArray],
    AdvectionDiffusionParameterization[NumpyArray],
    LagrangeBasis[NumpyArray],
]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        forcing=TimeIndependent(_unit_forcing),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 1.0, bkd),
            DirichletBC(basis, "right", right_value, bkd),
        ],
    )
    param_obj = AdvectionDiffusionParameterization(
        physics, diffusivity_map=_log_kle_map(bkd, basis), bkd=bkd
    )
    return system, param_obj, basis


def _transient_problem(
    bkd: NumpyBkd,
) -> Tuple[
    GalerkinTransientForwardModel[NumpyArray],
    GalerkinSystem[NumpyArray],
    LagrangeBasis[NumpyArray],
]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        forcing=TimeDependent(_heat_forcing),
    )
    right = BoundarySignal(TimeDependent(_right_value), [TimeDependent(_right_rate)])
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", right, bkd),
        ],
    )
    param_obj = AdvectionDiffusionParameterization(
        physics, diffusivity_map=_log_kle_map(bkd, basis), bkd=bkd
    )
    x = bkd.to_numpy(basis.dof_coordinates())[0]
    config: TimeIntegrationConfig[NumpyArray] = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.02,
        deltat=0.005,
        newton_tol=1e-13,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )
    model = GalerkinTransientForwardModel(
        system,
        param_obj,
        bkd.asarray(0.3 * np.sin(np.pi * x)),
        config,
        bkd,
    )
    return model, system, basis


def _steady_snapshots(
    bkd: NumpyBkd,
) -> Tuple[
    SnapshotSet[NumpyArray],
    GalerkinSystem[NumpyArray],
    AdvectionDiffusionParameterization[NumpyArray],
    LagrangeBasis[NumpyArray],
]:
    system, param_obj, basis = _steady_problem(bkd)
    state_eq = GalerkinParameterizedStateEquation(system.steady(), param_obj, bkd)
    snapshots = collect_steady_snapshots(
        state_eq, bkd.asarray(_SAMPLES), bkd.zeros((system.nstates(), 1))
    )
    return snapshots, system, param_obj, basis


class TestSnapshotSet:
    def test_counts(self, bkd: Backend[Array]) -> None:
        snapshots = SnapshotSet(bkd.zeros((5, 4)), bkd.zeros((2, 4)), bkd.zeros((4,)))
        assert (
            snapshots.nstates(),
            snapshots.nparams(),
            snapshots.nsnapshots(),
        ) == (5, 2, 4)
        assert not snapshots.is_steady()
        assert SnapshotSet(bkd.zeros((5, 4)), bkd.zeros((2, 4))).is_steady()

    def test_rejects_mismatched_columns(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="columns"):
            SnapshotSet(bkd.zeros((5, 4)), bkd.zeros((2, 3)))
        with pytest.raises(ValueError, match="entries"):
            SnapshotSet(bkd.zeros((5, 4)), bkd.zeros((2, 4)), bkd.zeros((3,)))
        with pytest.raises(ValueError, match="2D"):
            SnapshotSet(bkd.zeros((5,)), bkd.zeros((2, 4)))

    def test_concatenate(self, bkd: Backend[Array]) -> None:
        first = SnapshotSet(
            bkd.ones((3, 2)), bkd.zeros((1, 2)), bkd.asarray([0.0, 1.0])
        )
        second = SnapshotSet(
            2.0 * bkd.ones((3, 1)), bkd.ones((1, 1)), bkd.asarray([2.0])
        )
        joined = concatenate_snapshot_sets([first, second], bkd)
        bkd.assert_allclose(joined.states[0], bkd.asarray([1.0, 1.0, 2.0]))
        bkd.assert_allclose(joined.params[0], bkd.asarray([0.0, 0.0, 1.0]))
        assert joined.times is not None
        bkd.assert_allclose(joined.times, bkd.asarray([0.0, 1.0, 2.0]))

    def test_concatenate_rejects_steady_with_transient(
        self, bkd: Backend[Array]
    ) -> None:
        steady = SnapshotSet(bkd.ones((3, 1)), bkd.zeros((1, 1)))
        transient = SnapshotSet(bkd.ones((3, 1)), bkd.zeros((1, 1)), bkd.zeros((1,)))
        with pytest.raises(ValueError, match="steady and transient"):
            concatenate_snapshot_sets([steady, transient], bkd)


class TestSteadyCollection:
    def test_snapshots_solve_the_raw_residual(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        snapshots, system, param_obj, basis = _steady_snapshots(bkd)
        assert snapshots.is_steady()
        bkd.assert_allclose(snapshots.params, bkd.asarray(_SAMPLES))
        residual = GalerkinParameterizedResidual(system.spatial_operator(), param_obj)
        constraint_set = system.constraint_set()
        for ii in range(snapshots.nsnapshots()):
            state = snapshots.states[:, ii]
            raw = residual.residual(state, 0.0, snapshots.params[:, ii])
            unconstrained = constraint_set.zero_entries(raw)
            bkd.assert_allclose(
                unconstrained, bkd.zeros(unconstrained.shape), atol=1e-11
            )
            bkd.assert_allclose(state[basis.get_dofs("left")], bkd.asarray([1.0]))
            bkd.assert_allclose(state[basis.get_dofs("right")], bkd.asarray([2.0]))

    def test_independent_of_sample_order(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        snapshots, system, param_obj, _ = _steady_snapshots(bkd)
        state_eq = GalerkinParameterizedStateEquation(system.steady(), param_obj, bkd)
        reversed_snapshots = collect_steady_snapshots(
            state_eq,
            bkd.asarray(_SAMPLES[:, ::-1].copy()),
            bkd.zeros((system.nstates(), 1)),
        )
        bkd.assert_allclose(
            reversed_snapshots.states,
            bkd.flip(snapshots.states, axis=(1,)),
            rtol=1e-12,
        )

    def test_rejects_wrong_sample_shape(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _steady_problem(bkd)
        state_eq = GalerkinParameterizedStateEquation(system.steady(), param_obj, bkd)
        with pytest.raises(ValueError, match="samples"):
            collect_steady_snapshots(
                state_eq, bkd.zeros((2, 1)), bkd.zeros((system.nstates(), 1))
            )


class TestTransientCollection:
    def test_columns_pair_states_params_times(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        model, _, basis = _transient_problem(bkd)
        samples = bkd.asarray(_SAMPLES[:, :2].copy())
        snapshots = collect_transient_snapshots(model, samples)
        assert snapshots.times is not None
        ntimes = 5
        assert snapshots.nsnapshots() == 2 * ntimes
        right = basis.get_dofs("right")
        for ii in range(snapshots.nsnapshots()):
            bkd.assert_allclose(snapshots.params[:, ii], samples[:, ii // ntimes])
            time = bkd.to_float(snapshots.times[ii])
            bkd.assert_allclose(
                snapshots.states[right, ii],
                bkd.asarray([0.5 * np.sin(4.0 * time)]),
                atol=1e-12,
            )


class TestAlgebraicLift:
    def test_carries_data_and_is_zero_elsewhere(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, _, basis = _steady_problem(bkd)
        lift = AlgebraicLift(system.constraint_set(), system.nstates())
        values = lift.lift(0.0)
        expected = np.zeros(system.nstates())
        expected[bkd.to_numpy(basis.get_dofs("left"))] = 1.0
        expected[bkd.to_numpy(basis.get_dofs("right"))] = 2.0
        bkd.assert_allclose(values, bkd.asarray(expected))
        assert lift.is_time_invariant()


class TestHomogenize:
    def test_steady(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        snapshots, system, _, basis = _steady_snapshots(bkd)
        constraint_set = system.constraint_set()
        lift = AlgebraicLift(constraint_set, system.nstates())
        homogenized = homogenize(snapshots, lift, constraint_set)
        ends = bkd.concatenate([basis.get_dofs("left"), basis.get_dofs("right")])
        bkd.assert_allclose(homogenized.states[ends, :], bkd.zeros((2, 3)), atol=1e-12)
        interior = np.setdiff1d(np.arange(system.nstates()), bkd.to_numpy(ends))
        bkd.assert_allclose(
            homogenized.states[interior, :], snapshots.states[interior, :]
        )
        bkd.assert_allclose(homogenized.params, snapshots.params)

    def test_transient_uses_each_snapshots_time(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        model, system, basis = _transient_problem(bkd)
        snapshots = collect_transient_snapshots(
            model, bkd.asarray(_SAMPLES[:, :1].copy())
        )
        constraint_set = system.constraint_set()
        lift = AlgebraicLift(constraint_set, system.nstates())
        homogenized = homogenize(snapshots, lift, constraint_set)
        right = basis.get_dofs("right")
        bkd.assert_allclose(
            homogenized.states[right, :],
            bkd.zeros((1, snapshots.nsnapshots())),
            atol=1e-12,
        )

    def test_rejects_snapshots_of_a_different_system(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        snapshots, _, _, _ = _steady_snapshots(bkd)
        other_system, _, _ = _steady_problem(bkd, right_value=3.0)
        constraint_set = other_system.constraint_set()
        lift = AlgebraicLift(constraint_set, other_system.nstates())
        with pytest.raises(ValueError, match="constrained DOF"):
            homogenize(snapshots, lift, constraint_set)

    def test_rejects_times_paired_with_wrong_states(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        model, system, _ = _transient_problem(bkd)
        snapshots = collect_transient_snapshots(
            model, bkd.asarray(_SAMPLES[:, :1].copy())
        )
        assert snapshots.times is not None
        shuffled = SnapshotSet(
            snapshots.states,
            snapshots.params,
            bkd.flip(snapshots.times, axis=(0,)),
        )
        constraint_set = system.constraint_set()
        lift = AlgebraicLift(constraint_set, system.nstates())
        with pytest.raises(ValueError, match="constrained DOF"):
            homogenize(shuffled, lift, constraint_set)

    def test_steady_snapshots_need_time_invariant_data(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        _, system, _ = _transient_problem(bkd)
        constraint_set = system.constraint_set()
        lift = AlgebraicLift(constraint_set, system.nstates())
        steady = SnapshotSet(bkd.zeros((system.nstates(), 1)), bkd.zeros((3, 1)))
        with pytest.raises(ValueError, match="time-invariant"):
            homogenize(steady, lift, constraint_set)
