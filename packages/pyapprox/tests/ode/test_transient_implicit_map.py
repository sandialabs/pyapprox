r"""The transient implicit map: a functional of a trajectory as a map of
the parameters, in the stack's types.

Lotka-Volterra with a mass from (10, 5), parameters (1, 0.1, 0.075, 1.5),
ten steps of 0.05, under backward Euler, Heun and BDF2 (Newton to
1e-13):

- Against the sweeps run directly on the same solve
  (``TrajectoryProblem``), for the Tikhonov-augmented MSE as a fixed
  functional: the value, the tangent along a random direction, the
  gradient and the HVP agree to 1e-12 (the same arithmetic, so this
  checks the wiring: parameters set before the solve, setups cleared,
  the initial state taken, the sweeps on the solve's stepper).
- For the integral of :math:`q = \cos(t)\, y^\top y / 100 + (s^\top
  p)(\mathbf 1^\top y)` by the method, a source whose functional depends
  on the recorded trajectory (BDF2's recursion weights): the point's
  value equals the value of a solve keeping no trajectory to 1e-14, the
  gradient passes central differences of the value and the HVP central
  differences of the gradient (DerivativeChecker, minimum ratio below
  1e-6).
- A value of two entries (two MSEs observing different stored states,
  stacked, under BDF2): each entry's value, and the gradient under unit
  weights each entry's gradient, to 1e-12.
- Refusals: arguments that do not satisfy their protocols, and an
  earlier point's derivatives after the next solve has moved the
  parameters.
"""

import dataclasses
from typing import Any, Callable, Generic, List, Optional, Tuple, Union

import pytest

from pyapprox.implicitfunction.functional_terms import (
    ParameterTermDerivatives,
    ParameterTermProtocol,
)
from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ArrayParameters,
    ImplicitJacobianProtocol,
    ImplicitMapDerivatives,
    ImplicitMapFunction,
    TangentColumns,
)
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.tikhonov_term import TikhonovTerm
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.multistep_methods import bdf2
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import RowPinningProtocol
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.stepping_methods import (
    LinearMultistep,
    RungeKutta,
    SteppingMethodProtocol,
)
from pyapprox.ode.tableau import backward_euler, heun
from pyapprox.ode.time_integral import TimeIntegral
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol
from pyapprox.ode.trajectory_terms import (
    StepTermDerivatives,
    StepTermProtocol,
    TrajectoryFunctional,
)
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TrajectoryFunctionalSourceProtocol,
    TransientImplicitMap,
    TransientPoint,
    ValueSinkProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.runge_kutta_cases import LV_PARAM, lotka_volterra_with_mass
from tests._helpers.ode.trajectory_cases import (
    START,
    STEPS,
    EachTruncated,
    EnergyIntegrand,
    Method,
    ReducedLotkaVolterra,
    TrajectoryProblem,
    native_tikhonov_mse,
    random_direction,
)
from tests._helpers.ode_models import LotkaVolterra

_Method = Union[RungeKutta[Array, Array], LinearMultistep[Array, Array]]


def _newton() -> Newton[Array, Array]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=30)


def _backward_euler() -> _Method[Array]:
    return RungeKutta(backward_euler(), _newton())


def _heun() -> _Method[Array]:
    return RungeKutta(heun())


def _bdf2() -> _Method[Array]:
    return LinearMultistep(bdf2(), _newton())


_METHODS = [
    (_backward_euler, backward_euler),
    (_heun, heun),
    (_bdf2, bdf2),
]
_IDS = ["backward Euler", "Heun", "BDF2"]


class _Discretization(Generic[Array]):
    """Lotka-Volterra's dense mass-form system: nothing pinned, the
    initial state taken as given."""

    def __init__(self, bkd: Backend[Array], model: LotkaVolterra[Array]) -> None:
        self._bkd = bkd
        self._system = dense_mass_form_system(model)

    def system(self) -> MassFormSystem[Array, Array, DenseFormat[Array]]:
        return self._system

    def pinning(self) -> Optional[RowPinningProtocol[Array]]:
        return None

    def initial_state(self, state: Array, time: float) -> Array:
        return self._bkd.copy(state)


class _Parameters(Generic[Array]):
    """Sets the model's parameters and those of every term that reads
    them."""

    def __init__(self, *readers: Callable[[Array], None]) -> None:
        self._readers = readers

    def nparams(self) -> int:
        return 4

    def set_param(self, param: Array) -> None:
        for reader in self._readers:
            reader(param)


def _map(
    bkd: Backend[Array],
    method: SteppingMethodProtocol[Array, Array],
    model: LotkaVolterra[Array],
    functional: TrajectoryFunctionalSourceProtocol[Array, Array],
    *readers: Callable[[Array], None],
) -> TransientImplicitMap[Array, Array, DenseFormat[Array]]:
    discretization = _Discretization(bkd, model)
    return TransientImplicitMap(
        discretization,
        _Parameters(model.set_param, *readers),
        method,
        RecomputeAlways(discretization.system().solver),
        STEPS,
        bkd.asarray(START),
        functional,
    )


def _integral(
    bkd: Backend[Array],
    method: SteppingMethodProtocol[Array, Array],
    model: LotkaVolterra[Array],
    integrand: EnergyIntegrand[Array],
) -> TimeIntegral[Array, Array]:
    system = dense_mass_form_system(model)
    stepper = method.stepper(system, RecomputeAlways(system.solver))
    return TimeIntegral(integrand, stepper, bkd)


@pytest.mark.parametrize("make, tableau_or_formula", _METHODS, ids=_IDS)
class TestAgainstTheSweeps:
    def test_value_tangent_gradient_and_hvp(
        self,
        bkd: Backend[Array],
        make: Callable[[], _Method[Array]],
        tableau_or_formula: Callable[[], Method],
    ) -> None:
        oracle = TrajectoryProblem(bkd, tableau_or_formula())
        model = lotka_volterra_with_mass(bkd)
        functional, tikhonov = native_tikhonov_mse(bkd)
        implicit = _map(
            bkd,
            make(),
            model,
            FixedTrajectoryFunctional(functional),
            tikhonov.set_param,
        )
        param = bkd.asarray(LV_PARAM)
        direction = random_direction(bkd)
        bkd.assert_allclose(implicit.value(param), oracle.value(param), rtol=1e-12)
        point = implicit.point(param)
        bkd.assert_allclose(point.value(), oracle.value(param), rtol=1e-12)
        bkd.assert_allclose(
            point.tangent(direction), oracle.tangent(param, direction), rtol=1e-12
        )
        bkd.assert_allclose(point.gradient(), oracle.gradient(param).T, rtol=1e-12)
        bkd.assert_allclose(
            point.hvp(direction), oracle.hvp(param, direction), rtol=1e-12
        )


@pytest.mark.parametrize("make, tableau_or_formula", _METHODS, ids=_IDS)
class TestATimeIntegral:
    def test_derivatives_pass_finite_differences(
        self,
        bkd: Backend[Array],
        make: Callable[[], _Method[Array]],
        tableau_or_formula: Callable[[], Method],
    ) -> None:
        method = make()
        model = lotka_volterra_with_mass(bkd)
        integrand = EnergyIntegrand(bkd)
        implicit = _map(
            bkd,
            method,
            model,
            _integral(bkd, method, model, integrand),
            integrand.set_param,
        )
        param = bkd.asarray(LV_PARAM)
        bkd.assert_allclose(
            implicit.point(param).value(), implicit.value(param), rtol=1e-14
        )

        def values(samples: Array) -> Array:
            return bkd.hstack(
                [implicit.value(samples[:, k : k + 1]) for k in range(samples.shape[1])]
            )

        def gradient(sample: Array) -> Array:
            return implicit.point(sample).gradient()

        def gradients(samples: Array) -> Array:
            return bkd.hstack(
                [gradient(samples[:, k : k + 1]) for k in range(samples.shape[1])]
            )

        def hvp(sample: Array, direction: Array) -> Array:
            return implicit.point(sample).hvp(direction)

        first = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1,
                nvars=4,
                fun=values,
                jacobian=lambda sample: gradient(sample).T,
                bkd=bkd,
            )
        ).check_derivatives(param, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(first)) < 1e-6
        second = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=4, nvars=4, fun=gradients, jvp=hvp, bkd=bkd
            )
        ).check_derivatives(param, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(second)) < 1e-6


class TestTwoEntries:
    """Two MSEs observing different stored states, stacked: the value is
    both, and the gradient under unit weights each one's, to 1e-12."""

    def test_unit_weights_give_each_entrys_gradient(self, bkd: Backend[Array]) -> None:
        method: _Method[Array] = _bdf2()
        model = lotka_volterra_with_mass(bkd)
        first, first_tikhonov = native_tikhonov_mse(bkd)
        second, second_tikhonov = native_tikhonov_mse(
            bkd, prey=(1, 4, 9), predator=(6, 10)
        )
        readers = (first_tikhonov.set_param, second_tikhonov.set_param)
        param = bkd.asarray(LV_PARAM)
        stacked = FixedTrajectoryFunctional(
            TrajectoryFunctional.stack(bkd, [first, second])
        )
        point = _map(bkd, method, model, stacked, *readers).point(param)
        assert point.value().shape == (2, 1)
        gradients = []
        for entry in range(2):
            weights = bkd.copy(bkd.zeros((2, 1)))
            weights[entry, 0] = 1.0
            gradients.append(point.gradient(weights))
        for entry, alone in enumerate((first, second)):
            # A model of its own, so the stacked point stays valid.
            single = _map(
                bkd,
                method,
                lotka_volterra_with_mass(bkd),
                FixedTrajectoryFunctional(alone),
                *readers,
            ).point(param)
            bkd.assert_allclose(point.value()[entry], single.value()[0], rtol=1e-12)
            bkd.assert_allclose(gradients[entry], single.gradient(), rtol=1e-12)


class _ParameterTermOfOrder(Generic[Array]):
    """A parameter term keeping its value and its derivatives up to
    ``order`` (0: none, 1: the first)."""

    def __init__(self, inner: ParameterTermProtocol[Array, Array], order: int) -> None:
        self._inner = inner
        wrt_param = inner.derivatives().wrt_param
        if wrt_param is None:
            raise ValueError("the inner term has its first derivative")
        self._derivatives: ParameterTermDerivatives[Array, Array] = (
            ParameterTermDerivatives.none()
            if order == 0
            else ParameterTermDerivatives.first_order(wrt_param)
        )

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def value(self) -> Array:
        return self._inner.value()

    def derivatives(self) -> ParameterTermDerivatives[Array, Array]:
        return self._derivatives


class _StepTermOfOrder(Generic[Array]):
    """A step term keeping its value and its derivatives up to ``order``
    (0: none, 1: the first)."""

    def __init__(self, inner: StepTermProtocol[Array, Array], order: int) -> None:
        self._inner = inner
        bundle = inner.derivatives()
        self._derivatives: StepTermDerivatives[Array, Array] = (
            StepTermDerivatives.none()
            if order == 0
            else StepTermDerivatives(
                bundle.wrt_unknown, bundle.wrt_lag, bundle.wrt_param
            )
        )

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def value(self, step: int, window: StepWindow[Array]) -> Array:
        return self._inner.value(step, window)

    def derivatives(self) -> StepTermDerivatives[Array, Array]:
        return self._derivatives


class _StepTermsOfOrder(Generic[Array]):
    """A functional source whose step terms keep their derivatives up to
    ``order``."""

    def __init__(
        self, inner: TrajectoryFunctionalSourceProtocol[Array, Array], order: int
    ) -> None:
        self._inner, self._order = inner, order

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def sink(self) -> ValueSinkProtocol[Array, Array]:
        return self._inner.sink()

    def trajectory_functional(
        self, store: TrajectoryStoreProtocol[Array]
    ) -> TrajectoryFunctional[Array, Array]:
        functional = self._inner.trajectory_functional(store)
        return dataclasses.replace(
            functional,
            step_terms=tuple(
                _StepTermOfOrder(term, self._order) for term in functional.step_terms
            ),
        )


def _orders(bundle: ImplicitMapDerivatives[Any, Array, Array]) -> Tuple[bool, ...]:
    return (
        bundle.tangent is not None,
        bundle.gradient is not None,
        bundle.hvp is not None,
    )


_NONE, _FIRST, _SECOND = (False,) * 3, (True, True, False), (True,) * 3


class TestDeclaredDerivatives:
    """The map's bundle is the weakest of its parts', fixed when it is
    built; each declared field is the point's own derivative."""

    def test_the_stepper_and_initial_state_cut_the_map(
        self, bkd: Backend[Array]
    ) -> None:
        for order, expected in ((0, _NONE), (1, _FIRST), (2, _SECOND)):
            model = ReducedLotkaVolterra(bkd, order)
            functional, tikhonov = native_tikhonov_mse(bkd)
            implicit = _map(
                bkd,
                _bdf2(),
                model,
                FixedTrajectoryFunctional(functional),
                tikhonov.set_param,
            )
            assert _orders(implicit.derivatives()) == expected

    def test_the_declared_fields_are_the_points(self, bkd: Backend[Array]) -> None:
        functional, tikhonov = native_tikhonov_mse(bkd)
        implicit = _map(
            bkd,
            _bdf2(),
            lotka_volterra_with_mass(bkd),
            FixedTrajectoryFunctional(functional),
            tikhonov.set_param,
        )
        bundle = implicit.derivatives()
        assert (
            bundle.tangent is not None
            and bundle.gradient is not None
            and bundle.hvp is not None
        )
        point = implicit.point(bkd.asarray(LV_PARAM))
        direction = random_direction(bkd)
        bkd.assert_allclose(
            bundle.tangent(point, direction), point.tangent(direction), rtol=1e-14
        )
        bkd.assert_allclose(bundle.gradient(point, None), point.gradient(), rtol=1e-14)
        bkd.assert_allclose(
            bundle.hvp(point, direction, None), point.hvp(direction), rtol=1e-14
        )

    def test_each_kind_of_term_cuts_the_map(self, bkd: Backend[Array]) -> None:
        endpoint = EndpointTerm(StateEntries(bkd, 2, [0]))
        for order, expected in ((0, _NONE), (1, _FIRST)):
            tikhonov = TikhonovTerm(bkd, 0.3)
            sources: List[TrajectoryFunctionalSourceProtocol[Array, Array]] = [
                FixedTrajectoryFunctional(
                    TrajectoryFunctional.from_terms(
                        bkd, 1, state_terms=(EachTruncated(endpoint, order),)
                    )
                ),
                FixedTrajectoryFunctional(
                    TrajectoryFunctional.from_terms(
                        bkd,
                        1,
                        parameter_terms=(_ParameterTermOfOrder(tikhonov, order),),
                    )
                ),
            ]
            method: _Method[Array] = _bdf2()
            model = lotka_volterra_with_mass(bkd)
            integrand = EnergyIntegrand(bkd)
            sources.append(
                _StepTermsOfOrder(_integral(bkd, method, model, integrand), order)
            )
            for source in sources:
                implicit = _map(
                    bkd, method, model, source, tikhonov.set_param, integrand.set_param
                )
                assert _orders(implicit.derivatives()) == expected, source

    def test_the_bundle_refuses_hvp_alone(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="needs tangent and gradient"):
            refused: ImplicitMapDerivatives[
                TransientPoint[Array, Array], Array, Array
            ] = ImplicitMapDerivatives(hvp=TransientPoint.hvp)
            del refused
        with pytest.raises(TypeError, match="must be callable"):
            ImplicitMapDerivatives(tangent=1.0)  # type: ignore[arg-type]


def _two_entries(
    bkd: Backend[Array], model: Optional[LotkaVolterra[Array]] = None
) -> TransientImplicitMap[Array, Array, DenseFormat[Array]]:
    """BDF2 on Lotka-Volterra (``model``, default with the mass) with two
    stacked Tikhonov MSEs observing different stored states."""
    first, first_tikhonov = native_tikhonov_mse(bkd)
    second, second_tikhonov = native_tikhonov_mse(bkd, prey=(1, 4, 9), predator=(6, 10))
    return _map(
        bkd,
        _bdf2(),
        lotka_volterra_with_mass(bkd) if model is None else model,
        FixedTrajectoryFunctional(TrajectoryFunctional.stack(bkd, [first, second])),
        first_tikhonov.set_param,
        second_tikhonov.set_param,
    )


class _TangentOnly(Generic[Array]):
    """A map declaring only its tangent, as one whose solver had no
    transpose would."""

    def __init__(
        self, inner: TransientImplicitMap[Array, Array, DenseFormat[Array]]
    ) -> None:
        self._inner = inner
        self._derivatives: ImplicitMapDerivatives[
            TransientPoint[Array, Array], Array, Array
        ] = ImplicitMapDerivatives(tangent=inner.derivatives().tangent)

    def nparams(self) -> int:
        return self._inner.nparams()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def derivatives(
        self,
    ) -> ImplicitMapDerivatives[TransientPoint[Array, Array], Array, Array]:
        return self._derivatives

    def value(self, param: Array) -> Array:
        return self._inner.value(param)

    def point(self, param: Array) -> TransientPoint[Array, Array]:
        return self._inner.point(param)


_WEIGHTS = [[0.7], [-0.4]]


class TestAsAFunction:
    """:class:`ImplicitMapFunction` over the map: values per sample
    column; the Jacobian by both strategies and the weighted HVP pass
    central differences (minimum ratio below 1e-6) for a value of two
    entries; the strategies agree to 1e-10 and the Jacobian's rows are the
    point's gradients."""

    @pytest.mark.parametrize(
        "strategy", [AdjointRows(), TangentColumns()], ids=["rows", "columns"]
    )
    def test_jacobian_and_whvp_pass_finite_differences(
        self, bkd: Backend[Array], strategy: ImplicitJacobianProtocol
    ) -> None:
        function = ImplicitMapFunction(
            _two_entries(bkd), ArrayParameters(bkd), strategy
        )
        derivatives = function.derivatives()
        assert derivatives.jacobian is not None and derivatives.whvp is not None
        assert derivatives.hvp is None
        errors = DerivativeChecker(function).check_derivatives(
            bkd.asarray(LV_PARAM),
            relative=True,
            central=True,
            weights=bkd.asarray(_WEIGHTS),
        )
        assert len(errors) == 2
        for error in errors:
            assert bkd.to_float(bkd.min(error)) < 1e-6

    def test_values_and_the_strategies_agree(self, bkd: Backend[Array]) -> None:
        implicit = _two_entries(bkd)
        rows = ImplicitMapFunction(implicit, ArrayParameters(bkd), AdjointRows())
        columns = ImplicitMapFunction(implicit, ArrayParameters(bkd), TangentColumns())
        param = bkd.asarray(LV_PARAM)
        samples = bkd.hstack([param, 1.05 * param, 0.95 * param])
        values = rows(samples)
        assert values.shape == (2, 3)
        for ii in range(3):
            bkd.assert_allclose(
                values[:, ii : ii + 1], implicit.value(samples[:, ii : ii + 1])
            )
        by_rows, by_columns = (
            rows.derivatives().jacobian,
            columns.derivatives().jacobian,
        )
        assert by_rows is not None and by_columns is not None
        jacobian = by_rows(param)
        assert jacobian.shape == (2, 4)
        bkd.assert_allclose(by_columns(param), jacobian, rtol=1e-10)
        point = implicit.point(param)
        bkd.assert_allclose(
            jacobian[1:2, :], point.gradient(bkd.asarray([[0.0], [1.0]])).T, rtol=1e-12
        )

    def test_the_default_strategy(self, bkd: Backend[Array]) -> None:
        """Rows with no more entries than parameters; columns when the map
        has no gradient, and rows refused there."""
        implicit = _two_entries(bkd)
        assert isinstance(
            ImplicitMapFunction(implicit, ArrayParameters(bkd))._strategy, AdjointRows
        )
        tangent_only = _TangentOnly(implicit)
        function = ImplicitMapFunction(tangent_only, ArrayParameters(bkd))
        assert isinstance(function._strategy, TangentColumns)
        derivatives = function.derivatives()
        assert derivatives.jacobian is not None and derivatives.whvp is None
        with pytest.raises(TypeError, match="does not declare"):
            ImplicitMapFunction(tangent_only, ArrayParameters(bkd), AdjointRows())

    def test_capability_follows_the_map(self, bkd: Backend[Array]) -> None:
        """A first-order map gives the Jacobian alone, a map without
        derivatives none; a single entry resolves the plain HVP."""
        first = ImplicitMapFunction(
            _two_entries(bkd, ReducedLotkaVolterra(bkd, 1)), ArrayParameters(bkd)
        ).derivatives()
        assert first.jacobian is not None and first.whvp is None
        endpoint = EndpointTerm(StateEntries(bkd, 2, [0]))
        none = ImplicitMapFunction(
            _map(
                bkd,
                _bdf2(),
                lotka_volterra_with_mass(bkd),
                FixedTrajectoryFunctional(
                    TrajectoryFunctional.from_terms(
                        bkd, 1, state_terms=(EachTruncated(endpoint, 0),)
                    )
                ),
            ),
            ArrayParameters(bkd),
        ).derivatives()
        assert none.jacobian is None and none.whvp is None
        functional, tikhonov = native_tikhonov_mse(bkd)
        implicit = _map(
            bkd,
            _bdf2(),
            lotka_volterra_with_mass(bkd),
            FixedTrajectoryFunctional(functional),
            tikhonov.set_param,
        )
        function = ImplicitMapFunction(implicit, ArrayParameters(bkd))
        hvp = function.derivatives().resolved_hvp(1, bkd)
        assert hvp is not None
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        bkd.assert_allclose(
            hvp(param, direction), implicit.point(param).hvp(direction), rtol=1e-12
        )

    def test_refusals(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        implicit = _two_entries(bkd)
        function = ImplicitMapFunction(implicit, ArrayParameters(bkd))
        param = bkd.asarray(LV_PARAM)
        with pytest.raises(ValueError):
            function(param[:, 0])
        with pytest.raises(ValueError):
            function(param[:3])
        whvp = function.derivatives().whvp
        assert whvp is not None
        with pytest.raises(ValueError, match="weights must have shape"):
            whvp(param, param, bkd.ones((3, 1)))
        with pytest.raises(TypeError, match="implicit"):
            ImplicitMapFunction(object(), ArrayParameters(bkd))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="parameters"):
            ImplicitMapFunction(implicit, object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ImplicitJacobianProtocol"):
            ImplicitMapFunction(
                implicit,
                ArrayParameters(bkd),
                object(),  # type: ignore[arg-type]
            )


class TestRefusals:
    def test_refuses_arguments_of_the_wrong_kind(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        model = lotka_volterra_with_mass(bkd)
        functional, _ = native_tikhonov_mse(bkd)
        discretization = _Discretization(bkd, model)
        arguments = [
            discretization,
            _Parameters(model.set_param),
            _backward_euler(),
            RecomputeAlways(discretization.system().solver),
            STEPS,
            bkd.asarray(START),
            FixedTrajectoryFunctional(functional),
        ]
        for position, name in (
            (0, "discretization"),
            (1, "parameters"),
            (2, "method"),
            (4, "steps"),
            (6, "functional"),
        ):
            wrong = list(arguments)
            wrong[position] = object()
            with pytest.raises(TypeError, match=name):
                TransientImplicitMap(*wrong)  # type: ignore[arg-type]

    def test_a_point_is_refused_after_the_next_solve(self, numpy_bkd: NumpyBkd) -> None:
        """The next solve moves the parameters the sweeps read, so an
        earlier point's derivatives are refused, not silently wrong."""
        bkd = numpy_bkd
        model = lotka_volterra_with_mass(bkd)
        functional, tikhonov = native_tikhonov_mse(bkd)
        implicit = _map(
            bkd,
            _backward_euler(),
            model,
            FixedTrajectoryFunctional(functional),
            tikhonov.set_param,
        )
        param = bkd.asarray(LV_PARAM)
        point = implicit.point(param)
        implicit.value(1.1 * param)
        with pytest.raises(ValueError, match="solve again"):
            point.gradient()

    def test_fixed_functional_refuses_another_kind(self) -> None:
        with pytest.raises(TypeError, match="TrajectoryFunctional"):
            FixedTrajectoryFunctional(object())  # type: ignore[arg-type]
