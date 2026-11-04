r"""The derivatives of a functional integrating over time by the
stepper's own method.

Lotka-Volterra with a mass from (10, 5), parameters (1, 0.1, 0.075, 1.5),
ten steps of 0.05, and :math:`q(y, t; p) = \cos(t)\, y^\top y / 100 +
(s^\top p)(\mathbf 1^\top y)`, which reads the state nonlinearly, the
time and the parameters.

- The adjoint gradient passes central finite differences, the tangent
  sweep's :math:`dQ/dp \cdot v` equals the gradient's inner product with
  :math:`v`, and the Hessian-vector product passes central differences
  of the gradient, for every tableau (stages read from the unknown, a
  trivial first stage from :math:`y_n`; implicit midpoint included, which
  the old path gave no HVP) and every multistep method (BDF's recursion
  weighs each step by the transposed recursion's response).
- Sums of terms: the integral plus the Tikhonov-augmented MSE (a state
  term) and the integral twice (two step terms) give the sums of their
  gradients and Hessian-vector products, to 1e-12.
- Stacks: the integral over the MSE, a value of two entries: one tangent
  sweep gives both entries' directional derivatives, and the adjoint
  sweeps with unit weights each entry's gradient and Hessian-vector
  product, to 1e-12 (Heun, BDF2); weights are required for two entries
  and checked for shape, and functionals of different sizes do not add.
- Binding the integral to a trajectory reads no stored state.
- The sensitivities :math:`\theta_n = \partial z_N/\partial b_n` (``b_n``
  step :math:`n`'s right-hand side) equal the last row of
  :math:`L^{-1}`, :math:`L` the lower-triangular matrix whose row
  :math:`n` holds step :math:`n`'s :math:`\alpha`, for BDF2 on unequal
  steps (startup included), and are 1 for Runge-Kutta and Adams methods.
- Against torch autograd through the forward solve, the gradient agrees
  to 1e-13 and the Hessian-vector product to 1e-12 (double backward), so
  the sink keeps the graph intact.
- Refusals: an integrand without gradients; a parameter gradient without
  the state gradient; a Hessian-vector product of an integral whose
  integrand has no second derivatives.
"""

from typing import Callable, Generic

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.accumulating_sink import MultiSink
from pyapprox.ode.functional_sink import FunctionalSink
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import (
    adams_bashforth2,
    bdf2,
)
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import backward_euler, classical_rk4, heun
from pyapprox.ode.time_integral import (
    IntegrandDerivatives,
    TimeIntegral,
)
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
)
from pyapprox.ode.trajectory_terms import BackendScalarGradient, TrajectoryFunctional
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.ode.runge_kutta_cases import LV_PARAM, lotka_volterra_with_mass
from tests._helpers.ode.trajectory_cases import (
    ALL_IDS,
    ALL_METHODS,
    START,
    STEPS,
    EnergyIntegrand,
    Method,
    Stepper,
    make_stepper,
    native_tikhonov_mse,
    random_direction,
)


class _Problem(Generic[Array]):
    r""":math:`p \mapsto Q(y(p), p)` with the sweeps' gradient, tangent
    and Hessian-vector product; :math:`Q` is the integral, or with
    ``misfit`` the integral plus the Tikhonov-augmented MSE."""

    def __init__(
        self,
        bkd: Backend[Array],
        method: Method,
        misfit: bool = False,
        second_order: bool = True,
    ) -> None:
        self._bkd = bkd
        self._model = lotka_volterra_with_mass(bkd)
        self._stepper: Stepper[Array] = make_stepper(
            method, dense_mass_form_system(self._model)
        )
        self._integrand = EnergyIntegrand(bkd, second_order)
        self._integral: TimeIntegral[Array, Array] = TimeIntegral(
            self._integrand, self._stepper, bkd
        )
        self._misfit, self._tikhonov = native_tikhonov_mse(bkd)
        self._with_misfit = misfit
        self._sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            self._stepper
        )

    def solve(self, param: Array) -> tuple[Array, InMemoryTrajectoryStore[Array]]:
        self._model.set_param(param)
        self._integrand.set_param(param)
        self._tikhonov.set_param(param)
        integral = self._integral.sink()
        misfit: FunctionalSink[Array, Array] = FunctionalSink(
            self._misfit.zero, self._misfit.parameter_terms, self._misfit.state_terms
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        sinks: list[TrajectorySinkProtocol[Array]] = [integral, store]
        if self._with_misfit:
            sinks.append(misfit)
        Integrator(self._stepper).solve(
            self._bkd.asarray(START), STEPS, MultiSink(sinks)
        )
        value = integral.value()
        if self._with_misfit:
            value = value + misfit.value()
        return value, store

    def functional(
        self, store: InMemoryTrajectoryStore[Array], param: Array
    ) -> TrajectoryFunctional[Array, Array]:
        integral = self._integral.trajectory_functional(store)
        if not self._with_misfit:
            return integral
        return integral + self._misfit

    def value(self, samples: Array) -> Array:
        return self._bkd.hstack(
            [self.solve(samples[:, k : k + 1])[0] for k in range(samples.shape[1])]
        )

    def gradient(self, sample: Array) -> Array:
        _, store = self.solve(sample)
        return self._sweeps.gradient(store, self.functional(store, sample)).T

    def gradients(self, samples: Array) -> Array:
        return self._bkd.hstack(
            [self.gradient(samples[:, k : k + 1]).T for k in range(samples.shape[1])]
        )

    def tangent(self, sample: Array, direction: Array) -> Array:
        _, store = self.solve(sample)
        return self._sweeps.tangent(store, self.functional(store, sample), direction)

    def hvp(self, sample: Array, direction: Array) -> Array:
        _, store = self.solve(sample)
        return self._sweeps.hvp(store, self.functional(store, sample), direction)

    def sweeps(self) -> TrajectoryDerivatives[Array, Array]:
        return self._sweeps

    def stepper(self) -> Stepper[Array]:
        return self._stepper

    def integral(self) -> TimeIntegral[Array, Array]:
        return self._integral

    def misfit(self) -> TrajectoryFunctional[Array, Array]:
        """The Tikhonov-augmented MSE as native terms, read at the
        parameters of the last solve."""
        return self._misfit


class TestDerivatives:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_gradient_passes_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem = _Problem(bkd, make())
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=4, fun=problem.value, jacobian=problem.gradient, bkd=bkd
        )
        errors = DerivativeChecker(wrapper).check_derivatives(
            bkd.asarray(LV_PARAM), relative=True, central=True
        )[0]
        assert bkd.to_float(bkd.min(errors)) < 1e-7

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_tangent_is_the_gradient_along_the_direction(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem = _Problem(bkd, make())
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        bkd.assert_allclose(
            problem.tangent(param, direction),
            bkd.dot(problem.gradient(param), direction),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_hvp_passes_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem = _Problem(bkd, make())
        wrapper = FunctionWithJVPFromCallable(
            nqoi=4, nvars=4, fun=problem.gradients, jvp=problem.hvp, bkd=bkd
        )
        errors = DerivativeChecker(wrapper).check_derivatives(
            bkd.asarray(LV_PARAM), relative=True, central=True
        )[0]
        assert bkd.to_float(bkd.min(errors)) < 1e-7


class TestSums:
    @pytest.mark.parametrize(
        "make", [backward_euler, heun, bdf2], ids=["backward Euler", "Heun", "BDF2"]
    )
    def test_a_step_term_and_a_state_term(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        both = _Problem(bkd, make(), misfit=True)
        integral = _Problem(bkd, make())
        gradient, hvp = both.gradient(param), both.hvp(param, direction)
        # The misfit alone, on a trajectory solved after both's last solve.
        _, store = both.solve(param)
        misfit = both.misfit()
        sweeps = both.sweeps()
        bkd.assert_allclose(
            gradient,
            integral.gradient(param) + sweeps.gradient(store, misfit).T,
            rtol=1e-12,
        )
        bkd.assert_allclose(
            hvp,
            integral.hvp(param, direction) + sweeps.hvp(store, misfit, direction),
            rtol=1e-12,
        )

    def test_two_step_terms(self, bkd: Backend[Array]) -> None:
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        problem = _Problem(bkd, bdf2())
        _, store = problem.solve(param)
        once = problem.integral().trajectory_functional(store)
        twice = once + problem.integral().trajectory_functional(store)
        sweeps = problem.sweeps()
        bkd.assert_allclose(
            sweeps.gradient(store, twice),
            2.0 * sweeps.gradient(store, once),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            sweeps.hvp(store, twice, direction),
            2.0 * sweeps.hvp(store, once, direction),
            rtol=1e-12,
        )


class TestStack:
    r"""The integral stacked over the Tikhonov-augmented MSE, a value of
    two entries: one tangent sweep gives both entries' derivatives, and an
    adjoint sweep with a unit weight gives one entry's gradient."""

    def _parts(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> tuple[
        _Problem[Array],
        InMemoryTrajectoryStore[Array],
        TrajectoryFunctional[Array, Array],
        TrajectoryFunctional[Array, Array],
        TrajectoryFunctional[Array, Array],
    ]:
        problem = _Problem(bkd, make())
        param = bkd.asarray(LV_PARAM)
        _, store = problem.solve(param)
        integral = problem.integral().trajectory_functional(store)
        misfit = problem.misfit()
        stacked = TrajectoryFunctional.stack(bkd, [integral, misfit])
        return problem, store, integral, misfit, stacked

    @pytest.mark.parametrize("make", [heun, bdf2], ids=["Heun", "BDF2"])
    def test_one_tangent_gives_every_entry(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem, store, integral, misfit, stacked = self._parts(bkd, make)
        sweeps, direction = problem.sweeps(), random_direction(bkd)
        assert stacked.nqoi() == 2
        bkd.assert_allclose(
            sweeps.tangent(store, stacked, direction),
            bkd.vstack(
                [
                    sweeps.tangent(store, integral, direction),
                    sweeps.tangent(store, misfit, direction),
                ]
            ),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("make", [heun, bdf2], ids=["Heun", "BDF2"])
    def test_unit_weights_give_each_entrys_gradient_and_hvp(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem, store, integral, misfit, stacked = self._parts(bkd, make)
        sweeps, direction = problem.sweeps(), random_direction(bkd)
        for weights, part in (
            (bkd.asarray([[1.0], [0.0]]), integral),
            (bkd.asarray([[0.0], [1.0]]), misfit),
        ):
            bkd.assert_allclose(
                sweeps.gradient(store, stacked, weights=weights),
                sweeps.gradient(store, part),
                rtol=1e-12,
                atol=1e-15,
            )
            bkd.assert_allclose(
                sweeps.hvp(store, stacked, direction, weights=weights),
                sweeps.hvp(store, part, direction),
                rtol=1e-12,
                atol=1e-15,
            )

    def test_weights_are_needed_and_checked(self, bkd: Backend[Array]) -> None:
        problem, store, _, _, stacked = self._parts(bkd, heun)
        with pytest.raises(ValueError, match="2 entries needs its weights"):
            problem.sweeps().gradient(store, stacked)
        with pytest.raises(ValueError, match=r"shape \(2, 1\)"):
            problem.sweeps().gradient(store, stacked, weights=bkd.ones((1, 1)))
        with pytest.raises(ValueError, match="cannot add"):
            stacked + problem.integral().trajectory_functional(store)


class _CountingStore(InMemoryTrajectoryStore[Array]):
    """An in-memory store counting the reads of its states, as a store
    recomputing them from checkpoints would pay for each."""

    def __init__(self) -> None:
        super().__init__()
        self.state_reads = 0

    def state(self, index: int) -> Array:
        self.state_reads += 1
        return super().state(index)

    def window(self, step: int, depth: int = 1) -> StepWindow[Array]:
        self.state_reads += depth
        return super().window(step, depth)


class TestStoreReads:
    @pytest.mark.parametrize(
        "make", [heun, bdf2], ids=["Heun", "BDF2"]
    )
    def test_binding_reads_no_state(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        """The recursion's weights come from the step sizes alone, so
        binding the integral to a trajectory reads none of its states."""
        problem = _Problem(bkd, make())
        problem.solve(bkd.asarray(LV_PARAM))
        store: _CountingStore[Array] = _CountingStore()
        Integrator(problem.stepper()).solve(bkd.asarray(START), STEPS, store)
        problem.integral().trajectory_functional(store)
        assert store.state_reads == 0


class TestAgainstAutograd:
    @pytest.mark.parametrize(
        "make",
        [backward_euler, heun, classical_rk4, bdf2, adams_bashforth2],
        ids=["backward Euler", "Heun", "RK4", "BDF2", "Adams-Bashforth 2"],
    )
    def test_gradient_and_hvp(
        self, torch_bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        torch = pytest.importorskip("torch")
        bkd = torch_bkd
        problem = _Problem(bkd, make())
        param = torch.tensor(LV_PARAM, requires_grad=True)
        direction = random_direction(bkd)
        value, _ = problem.solve(param)
        (auto_gradient,) = torch.autograd.grad(value[0, 0], param, create_graph=True)
        (auto_hvp,) = torch.autograd.grad((auto_gradient * direction).sum(), param)
        fixed = param.detach()
        bkd.assert_allclose(
            problem.gradient(fixed).T, auto_gradient.detach(), rtol=1e-13
        )
        bkd.assert_allclose(problem.hvp(fixed, direction), auto_hvp, rtol=1e-12)


class TestRefusals:
    def test_an_integrand_only_evaluated(self, bkd: Backend[Array]) -> None:
        """An integrand giving only its value integrates (through the sink)
        and binds to a trajectory; a sweep refuses it, naming the term."""
        problem = _Problem(bkd, backward_euler())
        _, store = problem.solve(bkd.asarray(LV_PARAM))
        stepper = make_stepper(
            backward_euler(), dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        integral: TimeIntegral[Array, Array] = TimeIntegral(
            _ValueOnly(bkd), stepper, bkd
        )
        sink = integral.sink()
        Integrator(stepper).solve(bkd.asarray(START), STEPS, sink)
        assert sink.value().shape == (1, 1)
        functional = integral.trajectory_functional(store)
        with pytest.raises(TypeError, match="RecursionWeighted.*only evaluated"):
            problem.sweeps().gradient(store, functional)

    def test_a_parameter_derivative_needs_the_state_derivative(self) -> None:
        with pytest.raises(ValueError, match="needs the state derivative"):
            IntegrandDerivatives(None, _ValueOnly(NumpyBkd()).linearization)

    def test_an_hvp_needs_the_integrands_second_derivatives(
        self, bkd: Backend[Array]
    ) -> None:
        param = bkd.asarray(LV_PARAM)
        problem = _Problem(bkd, backward_euler(), second_order=False)
        problem.gradient(param)
        with pytest.raises(TypeError, match="RecursionWeighted.*no curvature"):
            problem.hvp(param, random_direction(bkd))


class _ValueOnly(Generic[Array]):
    """An integrand with no derivatives."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nqoi(self) -> int:
        return 1

    def value(self, state: Array, time: float) -> Array:
        return self._bkd.dot(state.T, state)

    def linearization(
        self, state: Array, time: float
    ) -> BackendScalarGradient[Array]:
        """The state derivative it does not declare, for the refusal."""
        return BackendScalarGradient(2.0 * state, self._bkd)

    def derivatives(self) -> IntegrandDerivatives[Array, Array]:
        return IntegrandDerivatives.none()
