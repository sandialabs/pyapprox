r"""Pinned rows in differentiated form, over a system with a mass.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5),
parameters (1, 0.1, 0.075, 1.5), dt 0.05 to 0.5, Newton tolerance 1e-13;
the prey row :math:`R = \{0\}` constrained to :math:`g(t) = 10 + \sin 3t`,
rate :math:`\dot g = 3 \cos 3t` (``PinnedPrey``).

- The layer's right-hand side carries :math:`\dot g` on the prey row and
  :math:`f` on the other; with the symmetric treatment the other row
  carries :math:`f_1 - M_{10} \dot g`. Its Jacobian and parameter
  Jacobian are :math:`f`'s with the prey row zeroed, the Jacobian at
  :math:`f`'s level; its second derivatives are :math:`f`'s at the
  adjoint with the prey entry zeroed. The masses are ``[[1, 0], [0.5,
  1]]`` (coupled) and the identity (symmetric).
- A Jacobian's snapshots mapped by a fixed transform keep its level.
- Pinned at the step end over the layer, for every tableau and multistep
  method and both treatments: stored states hold the data (1e-12), and
  the stepper checker passes at a step's window; for backward Euler,
  Heun, SDIRK3 and BDF2 the trajectory's gradient and Hessian-vector
  product pass finite differences (smallest error below 1e-7).
- Split systems: with :math:`f = f_E + f_I` (two copies of the model),
  the rate on :math:`f_E` and the rows of :math:`f_I` zeroed, summed,
  equal the layer over :math:`f` in value, Jacobian and parameter
  Jacobian (1e-14), for both treatments, and so do backward Euler and
  Heun trajectories pinned at the step end (1e-12).
- Refusals: a declaration with no rate, a non-declaration, a
  non-treatment.
"""

from typing import Callable, Generic, List, Optional

import numpy as np
import pytest

from pyapprox.ode.differentiated_rows import (
    CoupledRows,
    RowRatesTreatmentProtocol,
    SymmetricRows,
    ZeroedRowsRightHandSide,
    differentiated_rows,
)
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.multistep_methods import bdf2
from pyapprox.ode.pinned_rows import PinnedRowsProtocol, RowPinning
from pyapprox.ode.residual_derivatives import (
    Jacobian,
    backend_spaces,
    map_jacobian,
)
from pyapprox.ode.stepper_derivatives_check import StepperBundleChecker
from pyapprox.ode.tableau import alexander_sdirk3, backward_euler, heun
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    BlockVector,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearizedOperator,
    ScaledSum,
    VectorSpaceProtocol,
)
from tests._helpers.ode.runge_kutta_cases import (
    LV_PARAM,
    backend_sample,
    lotka_volterra_with_mass,
)
from tests._helpers.ode.summed_rhs import SummedRightHandSide
from tests._helpers.ode.trajectory_cases import (
    ALL_IDS,
    ALL_METHODS,
    STEPS,
    Method,
    PinnedPrey,
    Stepper,
    TrajectoryProblem,
    make_stepper,
    min_gradient_fd_error,
    min_hvp_fd_error,
)
from tests._helpers.ode_models import LotkaVolterra

_System = MassFormSystem[Array, Array, DenseFormat[Array]]
_TIME = 0.3
_POINT = LinearizationPoint(0, ("test",))
_TREATMENTS: List[RowRatesTreatmentProtocol] = [CoupledRows(), SymmetricRows()]
_TREATMENT_IDS = ["coupled", "symmetric"]
_SOME: List[Callable[[], Method]] = [backward_euler, heun, alexander_sdirk3, bdf2]
_SOME_IDS = ["backward Euler", "Heun", "SDIRK3", "BDF2"]


class _Layered(Generic[Array]):
    """The model's system, and the layer over it."""

    def __init__(
        self, bkd: Backend[Array], treatment: RowRatesTreatmentProtocol
    ) -> None:
        self.model = lotka_volterra_with_mass(bkd)
        self.model.set_param(bkd.asarray(LV_PARAM))
        self.raw: _System[Array] = dense_mass_form_system(self.model)
        self.prey: PinnedPrey[Array] = PinnedPrey(bkd)
        self.system: _System[Array] = differentiated_rows(
            self.raw, self.prey, treatment
        )


class _Layer(Generic[Array]):
    """A builder of the layer over a model's system, for the trajectory
    problem."""

    def __init__(
        self, bkd: Backend[Array], treatment: RowRatesTreatmentProtocol
    ) -> None:
        self._prey: PinnedPrey[Array] = PinnedPrey(bkd)
        self._treatment = treatment

    def __call__(self, model: LotkaVolterra[Array]) -> _System[Array]:
        return differentiated_rows(
            dense_mass_form_system(model), self._prey, self._treatment
        )


class TestTheLayer:
    @pytest.mark.parametrize("treatment", _TREATMENTS, ids=_TREATMENT_IDS)
    def test_the_rows_carry_the_rate(
        self, bkd: Backend[Array], treatment: RowRatesTreatmentProtocol
    ) -> None:
        case: _Layered[Array] = _Layered(bkd, treatment)
        state = bkd.asarray([10.0, 5.0])
        raw = case.model(state, _TIME)
        rate = 3.0 * float(np.cos(3.0 * _TIME))
        other = bkd.to_float(raw[1])
        if isinstance(treatment, SymmetricRows):
            other -= 0.5 * rate
        bkd.assert_allclose(
            case.system.rhs(state, _TIME), bkd.asarray([rate, other]), rtol=1e-14
        )

    def test_the_derivatives_zero_the_rows(self, bkd: Backend[Array]) -> None:
        case: _Layered[Array] = _Layered(bkd, CoupledRows())
        state = bkd.asarray([10.0, 5.0])
        mine, raw = case.system.rhs.derivatives(), case.raw.rhs.derivatives()
        assert mine.jacobian is not None and raw.jacobian is not None
        assert repr(mine.jacobian) == repr(raw.jacobian)
        direction = bkd.asarray([[0.7], [-1.3]])
        expected = raw.jacobian(state, _TIME, _POINT).apply(direction)
        bkd.assert_allclose(
            mine.jacobian(state, _TIME, _POINT).apply(direction),
            bkd.asarray([[0.0], [bkd.to_float(expected[1, 0])]]),
        )
        assert mine.param is not None and raw.param is not None
        pdir = bkd.asarray([[0.3], [-0.2], [0.5], [0.1]])
        expected = raw.param(state, _TIME).apply(pdir)
        bkd.assert_allclose(
            mine.param(state, _TIME).apply(pdir),
            bkd.asarray([[0.0], [bkd.to_float(expected[1, 0])]]),
        )
        adj, zeroed = bkd.asarray([0.9, -0.4]), bkd.asarray([0.0, -0.4])
        assert mine.state_state is not None and raw.state_state is not None
        assert mine.state_param is not None and raw.state_param is not None
        assert mine.param_param is not None and raw.param_param is not None
        for layered, original, vec in [
            (mine.state_state, raw.state_state, direction),
            (mine.state_param, raw.state_param, pdir),
            (mine.param_param, raw.param_param, pdir),
        ]:
            bkd.assert_allclose(
                layered(state, _TIME, adj).apply(vec),
                original(state, _TIME, zeroed).apply(vec),
            )
        assert mine.initial_state is raw.initial_state

    @pytest.mark.parametrize(
        "treatment, expected",
        [
            (CoupledRows(), [[1.0, 0.0], [0.5, 1.0]]),
            (SymmetricRows(), [[1.0, 0.0], [0.0, 1.0]]),
        ],
        ids=_TREATMENT_IDS,
    )
    def test_the_mass(
        self,
        bkd: Backend[Array],
        treatment: RowRatesTreatmentProtocol,
        expected: List[List[float]],
    ) -> None:
        case: _Layered[Array] = _Layered(bkd, treatment)
        bkd.assert_allclose(case.system.mass.apply(bkd.eye(2)), bkd.asarray(expected))


class _Negated(Generic[Array]):
    def __init__(self, space: VectorSpaceProtocol[Array]) -> None:
        self._space = space

    def __call__(
        self, snapshot: JacobianSnapshotProtocol[Array, DenseFormat[Array]]
    ) -> JacobianSnapshotProtocol[Array, DenseFormat[Array]]:
        return LinearizedOperator(
            ScaledSum([(-1.0, snapshot)], self._space), snapshot.linearization()
        )


class _Snapshot(Generic[Array]):
    def __init__(self, op: BackendDenseOperator[Array]) -> None:
        self._op = op

    def general(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        return LinearizedOperator(self._op, point)

    def state_free(
        self, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        return LinearizedOperator(self._op, point)

    def constant(
        self, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        return LinearizedOperator(self._op, point)


def test_a_mapped_jacobian_keeps_its_level(bkd: Backend[Array]) -> None:
    states, _ = backend_spaces(bkd, 2, 1)
    matrix = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
    source = _Snapshot(BackendDenseOperator(matrix, states.descriptor(), bkd))
    negate: _Negated[Array] = _Negated(states)
    state, direction = bkd.asarray([1.0, 2.0]), bkd.asarray([[1.0], [-1.0]])
    for jacobian in (
        Jacobian.general(source.general),
        Jacobian.state_free(source.state_free),
        Jacobian.constant(source.constant),
    ):
        mapped = map_jacobian(jacobian, negate)
        assert repr(mapped) == repr(jacobian)
        snapshot = mapped(state, _TIME, _POINT)
        assert snapshot.linearization() == _POINT
        bkd.assert_allclose(snapshot.apply(direction), -bkd.dot(matrix, direction))


class _Pinned(Generic[Array]):
    def __init__(
        self,
        bkd: Backend[Array],
        method: Method,
        treatment: RowRatesTreatmentProtocol,
    ) -> None:
        case: _Layered[Array] = _Layered(bkd, treatment)
        self.case, self.param = case, bkd.asarray(LV_PARAM)
        self.stepper: Stepper[Array] = make_stepper(
            method, case.system, RowPinning(case.prey)
        )
        self.store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(self.stepper).solve(bkd.asarray([[10.0], [5.0]]), STEPS, self.store)


class TestPinnedAtTheStepEnd:
    @pytest.mark.parametrize("treatment", _TREATMENTS, ids=_TREATMENT_IDS)
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_states_hold_the_data_and_the_checker_passes(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        treatment: RowRatesTreatmentProtocol,
    ) -> None:
        pinned: _Pinned[Array] = _Pinned(bkd, make(), treatment)
        store, prey = pinned.store, pinned.case.prey
        for n in range(1, store.ntimes()):
            value = bkd.to_float(store.state(n)[0, 0])
            assert abs(value - prey.data(store.time(n))) <= 1e-12
        stepper, sample = pinned.stepper, backend_sample(bkd)
        space = stepper.residual_space()
        adj = BlockVector(
            tuple(
                sample(space.component(i).descriptor(), 99 + i)
                for i in range(space.nblocks())
            )
        )
        checker = StepperBundleChecker(
            stepper, sample, set_param=pinned.case.model.set_param
        )
        window = store.window(3, min(stepper.depth(), 4))
        assert checker.check(window, adj, pinned.param).failures() == []

    @pytest.mark.parametrize("treatment", _TREATMENTS, ids=_TREATMENT_IDS)
    @pytest.mark.parametrize("make", _SOME, ids=_SOME_IDS)
    def test_gradient_and_hvp(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        treatment: RowRatesTreatmentProtocol,
    ) -> None:
        problem: TrajectoryProblem[Array] = TrajectoryProblem(
            bkd,
            make(),
            system=_Layer(bkd, treatment),
            row_pinning=RowPinning(PinnedPrey(bkd)),
        )
        assert min_gradient_fd_error(bkd, problem) < 1e-7
        assert min_hvp_fd_error(bkd, problem) < 1e-7


class _Split(Generic[Array]):
    r""":math:`f = f_E + f_I`, two copies of the model, neutralized per
    partition and as a whole."""

    def __init__(
        self, bkd: Backend[Array], treatment: RowRatesTreatmentProtocol
    ) -> None:
        self.fast, self.slow = (
            lotka_volterra_with_mass(bkd),
            lotka_volterra_with_mass(bkd),
        )
        for model in (self.fast, self.slow):
            model.set_param(bkd.asarray(LV_PARAM))
        base: _System[Array] = dense_mass_form_system(self.fast)
        self.prey: PinnedPrey[Array] = PinnedPrey(bkd)
        rows, form, space = self.prey.rows(), base.form, base.space
        rates = self.prey.rates()
        assert rates is not None
        version = self.fast.parameters_version
        self.per_partition: _System[Array] = MassFormSystem(
            rhs=SummedRightHandSide(
                [
                    treatment.rated(self.fast, rows, rates, form, base.mass, space),
                    ZeroedRowsRightHandSide(self.slow, rows, form),
                ],
                space,
                version,
            ),
            mass=treatment.mass(base.mass, rows),
            space=space,
            solver=base.solver,
            form=form,
        )
        whole: _System[Array] = MassFormSystem(
            rhs=SummedRightHandSide([self.fast, self.slow], space, version),
            mass=base.mass,
            space=space,
            solver=base.solver,
            form=form,
        )
        self.whole: _System[Array] = differentiated_rows(whole, self.prey, treatment)

    def trajectory(self, system: _System[Array], method: Method) -> Array:
        stepper: Stepper[Array] = make_stepper(method, system, RowPinning(self.prey))
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        bkd = self.fast.bkd()
        Integrator(stepper).solve(bkd.asarray([[10.0], [5.0]]), STEPS, store)
        return bkd.hstack([store.state(n) for n in range(store.ntimes())])


class TestSplitSystems:
    @pytest.mark.parametrize("treatment", _TREATMENTS, ids=_TREATMENT_IDS)
    def test_partitions_sum_to_the_whole(
        self, bkd: Backend[Array], treatment: RowRatesTreatmentProtocol
    ) -> None:
        split: _Split[Array] = _Split(bkd, treatment)
        state = bkd.asarray([10.0, 5.0])
        parts, whole = split.per_partition.rhs, split.whole.rhs
        bkd.assert_allclose(parts(state, _TIME), whole(state, _TIME), rtol=1e-14)
        mine, theirs = parts.derivatives(), whole.derivatives()
        assert mine.jacobian is not None and theirs.jacobian is not None
        assert mine.param is not None and theirs.param is not None
        direction = bkd.asarray([[0.7], [-1.3]])
        pdir = bkd.asarray([[0.3], [-0.2], [0.5], [0.1]])
        bkd.assert_allclose(
            mine.jacobian(state, _TIME, _POINT).apply(direction),
            theirs.jacobian(state, _TIME, _POINT).apply(direction),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            mine.param(state, _TIME).apply(pdir),
            theirs.param(state, _TIME).apply(pdir),
            rtol=1e-14,
        )

    @pytest.mark.parametrize("treatment", _TREATMENTS, ids=_TREATMENT_IDS)
    @pytest.mark.parametrize("make", [backward_euler, heun], ids=["BE", "Heun"])
    def test_trajectories_agree(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        treatment: RowRatesTreatmentProtocol,
    ) -> None:
        split: _Split[Array] = _Split(bkd, treatment)
        bkd.assert_allclose(
            split.trajectory(split.per_partition, make()),
            split.trajectory(split.whole, make()),
            rtol=1e-12,
        )


class _NoRate(Generic[Array]):
    """The prey's rows and data, without their rate."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._prey: PinnedPrey[Array] = PinnedPrey(bkd)

    def rows(self) -> BackendRowSet[Array]:
        return self._prey.rows()

    def values(self, time: float) -> Array:
        return self._prey.values(time)

    def rates(self) -> Optional[Callable[[float], Array]]:
        return None


class TestRefusals:
    def test_it_refuses(self, bkd: Backend[Array]) -> None:
        model = lotka_volterra_with_mass(bkd)
        system: _System[Array] = dense_mass_form_system(model)
        no_rate: PinnedRowsProtocol[Array] = _NoRate(bkd)
        with pytest.raises(ValueError, match="supplies no rate"):
            differentiated_rows(system, no_rate)
        with pytest.raises(TypeError, match="PinnedRowsProtocol"):
            differentiated_rows(system, object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="RowRatesTreatmentProtocol"):
            differentiated_rows(  # type: ignore[misc]
                system, PinnedPrey(bkd), object()  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="RightHandSideProtocol"):
            ZeroedRowsRightHandSide(
                object(), PinnedPrey(bkd).rows(), system.form  # type: ignore[arg-type]
            )
