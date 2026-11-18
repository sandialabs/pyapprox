"""Integration tests for the 1D elastic bar forward UQ problems."""

import numpy as np
import pytest

from pyapprox.implicitfunction.functional_terms import StateTermProtocol
from pyapprox.implicitfunction.observation_terms import ObservedValue
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols import (
    FunctionProtocol,
)
from pyapprox.pde.collocation.functionals.observations import point_observation
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox_benchmarks.pde.elastic_bar import (
    AverageDisplacement,
    AverageStress,
    BarConstitutiveProtocol,
    BarQoIContext,
    BarQoIProtocol,
    LinearElastic,
    NeoHookean,
    StrainEnergy,
    TipDisplacement,
    build_elastic_bar_1d,
)
from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem

LINEAR = LinearElastic()
HYPERELASTIC = NeoHookean(poisson_ratio=0.3)


def _make_problem(
    bkd: Backend[Array],
    constitutive: BarConstitutiveProtocol = LINEAR,
    qoi: BarQoIProtocol = TipDisplacement(),
    npts: int = 20,
    num_kle_terms: int = 2,
) -> ForwardUQProblem[SteadyImplicitFunction[Array, DenseFormat[Array]], Array]:
    """Helper for creating problems with small defaults for fast tests."""
    return build_elastic_bar_1d(
        bkd,
        constitutive=constitutive,
        qoi=qoi,
        npts=npts,
        length=1.0,
        traction=1.0,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
        correlation_length=0.3,
    )


class _MidpointDisplacement:
    """A QoI the benchmark module does not define: u(L/2)."""

    def label(self) -> str:
        return "midpoint_displacement"

    def term(self, context: BarQoIContext[Array]) -> StateTermProtocol[Array, Array]:
        return ObservedValue(
            point_observation(context.basis, [context.length / 2], context.bkd)
        )


def _check_jacobian(
    bkd: Backend[Array], fwd: FunctionProtocol[Array], num_kle_terms: int = 2
) -> None:
    """Helper: run DerivativeChecker on a forward model.

    The checker resolves the model's Derivatives bundle itself; public
    derivative methods are never harvested.
    """
    checker = DerivativeChecker(fwd)
    np.random.seed(42)
    sample = bkd.array([0.1, -0.1][:num_kle_terms])[:, None]
    errors = checker.check_derivatives(sample, relative=True)[0]
    ratio = bkd.to_float(bkd.min(errors) / bkd.max(errors))
    assert ratio <= 2e-5


class TestElasticBar1D:
    # --- Evaluation tests ---

    def test_linear_tip_displacement_evaluate(self, bkd: Backend[Array]) -> None:
        """Linear bar, tip displacement: shape (1,1), positive value."""
        prob = _make_problem(bkd, LINEAR, TipDisplacement())
        fwd = prob.function()
        sample = bkd.zeros((2, 1))
        result = fwd(sample)
        assert result.shape == (1, 1)
        assert bkd.to_float(result[0, 0]) > 0.0

    def test_linear_average_displacement_evaluate(self, bkd: Backend[Array]) -> None:
        """Linear bar, average displacement: shape (1,1), positive value."""
        prob = _make_problem(bkd, LINEAR, AverageDisplacement())
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert bkd.to_float(result[0, 0]) > 0.0

    def test_linear_average_stress_evaluate(self, bkd: Backend[Array]) -> None:
        """Linear bar, average stress: shape (1,1), positive value."""
        prob = _make_problem(bkd, LINEAR, AverageStress())
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert bkd.to_float(result[0, 0]) > 0.0

    def test_linear_strain_energy_evaluate(self, bkd: Backend[Array]) -> None:
        """Linear bar, strain energy: shape (1,1), positive value."""
        prob = _make_problem(bkd, LINEAR, StrainEnergy())
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert bkd.to_float(result[0, 0]) > 0.0

    def test_hyperelastic_tip_displacement_evaluate(self, bkd: Backend[Array]) -> None:
        """Hyperelastic bar, tip displacement: shape (1,1), positive value."""
        prob = _make_problem(bkd, HYPERELASTIC, TipDisplacement())
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert bkd.to_float(result[0, 0]) > 0.0

    def test_hyperelastic_strain_energy_evaluate(self, bkd: Backend[Array]) -> None:
        """Hyperelastic bar, strain energy: shape (1,1), positive value."""
        prob = _make_problem(bkd, HYPERELASTIC, StrainEnergy())
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert bkd.to_float(result[0, 0]) > 0.0

    # --- Jacobian tests ---

    def test_linear_tip_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on linear bar, tip displacement."""
        prob = _make_problem(bkd, LINEAR, TipDisplacement())
        _check_jacobian(bkd, prob.function())

    def test_hyperelastic_tip_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on hyperelastic bar, tip displacement."""
        prob = _make_problem(bkd, HYPERELASTIC, TipDisplacement())
        _check_jacobian(bkd, prob.function())

    def test_linear_strain_energy_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on linear bar, strain energy."""
        prob = _make_problem(bkd, LINEAR, StrainEnergy())
        _check_jacobian(bkd, prob.function())

    def test_hyperelastic_strain_energy_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on hyperelastic bar, strain energy."""
        prob = _make_problem(bkd, HYPERELASTIC, StrainEnergy())
        _check_jacobian(bkd, prob.function())

    def test_linear_average_stress_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on linear bar, average stress."""
        prob = _make_problem(bkd, LINEAR, AverageStress())
        _check_jacobian(bkd, prob.function())

    def test_hyperelastic_average_stress_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on hyperelastic bar, average stress."""
        prob = _make_problem(bkd, HYPERELASTIC, AverageStress())
        _check_jacobian(bkd, prob.function())

    # --- All QoIs produce scalar ---

    def test_all_qois_produce_scalar(self, bkd: Backend[Array]) -> None:
        """All 4 QoIs return nqoi=1 and shape (1,1)."""
        sample = bkd.zeros((2, 1))
        qois: list[BarQoIProtocol] = [
            TipDisplacement(),
            AverageDisplacement(),
            AverageStress(),
            StrainEnergy(),
        ]
        for qoi in qois:
            prob = _make_problem(bkd, LINEAR, qoi)
            fwd = prob.function()
            assert fwd.nqoi() == 1, f"Failed for qoi={qoi}"
            result = fwd(sample)
            assert result.shape == (1, 1), f"Failed for qoi={qoi}"

    # --- Prior ---

    def test_prior_samples_shape(self, bkd: Backend[Array]) -> None:
        """Prior generates samples of correct shape."""
        prob = _make_problem(bkd, LINEAR, TipDisplacement())
        np.random.seed(42)
        samples = prob.prior().rvs(5)
        assert samples.shape == (2, 5)

    # --- Protocol compliance ---

    def test_function_protocol_compliance(self, bkd: Backend[Array]) -> None:
        """Forward model is Function-shaped and declares a jacobian."""
        prob = _make_problem(bkd, LINEAR, TipDisplacement())
        fwd = prob.function()
        assert isinstance(fwd, FunctionProtocol)
        assert fwd.derivatives().jacobian is not None

    # --- Convergence tests ---

    def test_convergence_linear(self, bkd: Backend[Array]) -> None:
        """Linear bar: coarse and fine mesh tip displacement agree."""
        sample = bkd.zeros((2, 1))
        prob_coarse = _make_problem(bkd, LINEAR, TipDisplacement(), npts=15)
        prob_fine = _make_problem(bkd, LINEAR, TipDisplacement(), npts=40)
        val_coarse = prob_coarse.function()(sample)
        val_fine = prob_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-3)

    def test_convergence_hyperelastic(self, bkd: Backend[Array]) -> None:
        """Hyperelastic bar: coarse and fine mesh tip displacement agree."""
        sample = bkd.zeros((2, 1))
        prob_coarse = _make_problem(
            bkd,
            HYPERELASTIC,
            TipDisplacement(),
            npts=15,
        )
        prob_fine = _make_problem(
            bkd,
            HYPERELASTIC,
            TipDisplacement(),
            npts=40,
        )
        val_coarse = prob_coarse.function()(sample)
        val_fine = prob_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-3)

    # --- Error handling ---

    def test_invalid_constitutive_raises(self, bkd: Backend[Array]) -> None:
        """A constitutive law outside the protocol raises TypeError."""
        with pytest.raises(TypeError, match="BarConstitutiveProtocol"):
            build_elastic_bar_1d(bkd, constitutive="rubber")  # type: ignore[arg-type]

    def test_invalid_qoi_raises(self, bkd: Backend[Array]) -> None:
        """A QoI outside the protocol raises TypeError."""
        with pytest.raises(TypeError, match="BarQoIProtocol"):
            build_elastic_bar_1d(bkd, qoi="max_stress")  # type: ignore[arg-type]

    def test_qoi_defined_outside_the_module(self, bkd: Backend[Array]) -> None:
        """A QoI the module does not define needs no change to it: u(L/2)
        lies strictly between u(0) = 0 and the tip displacement, and its
        Jacobian passes the derivative check."""
        sample = bkd.zeros((2, 1))
        prob = _make_problem(bkd, LINEAR, _MidpointDisplacement())
        assert prob.name() == "elastic_bar_1d_linear_midpoint_displacement"
        mid = bkd.to_float(prob.function()(sample)[0, 0])
        tip = bkd.to_float(_make_problem(bkd, LINEAR).function()(sample)[0, 0])
        assert 0.0 < mid < tip
        _check_jacobian(bkd, prob.function())

    # --- Multi-sample evaluation ---

    def test_batch_evaluation(self, bkd: Backend[Array]) -> None:
        """Forward model handles batch evaluation."""
        prob = _make_problem(bkd, LINEAR, TipDisplacement())
        fwd = prob.function()
        np.random.seed(42)
        samples = bkd.array([[0.1, -0.1, 0.2], [0.05, 0.0, -0.15]])
        result = fwd(samples)
        assert result.shape == (1, 3)


class TestElasticBarQoIMaterial:
    """Checks that hold only if the stress and energy QoIs read the
    modulus the state was solved with (the KLE field), not a fixed
    reference modulus."""

    @pytest.mark.parametrize("constitutive", [LINEAR, HYPERELASTIC])
    def test_average_stress_independent_of_material(
        self, bkd: Backend[Array], constitutive: BarConstitutiveProtocol
    ) -> None:
        """The bar is statically determinate: its stress follows from the
        traction and the body force alone, so the average stress is the
        same at any KLE sample."""
        sample = bkd.asarray([[0.8], [-0.6]])
        fwd = _make_problem(bkd, constitutive, AverageStress()).function()
        q_ref = fwd(bkd.zeros((2, 1)))
        bkd.assert_allclose(fwd(sample), q_ref, rtol=1e-8)
        jacobian = fwd.derivatives().jacobian
        assert jacobian is not None
        grad = jacobian(sample)
        assert bkd.to_float(bkd.max(bkd.abs(grad))) <= 1e-8 * abs(
            bkd.to_float(q_ref[0, 0])
        )

    def test_linear_strain_energy_clapeyron(self, bkd: Backend[Array]) -> None:
        """Clapeyron's theorem for the linear bar: twice the strain energy
        is the work of the loads, T u(L) + integral f u dx (f = 1,
        L = 1, T = 1), read from the displacement QoIs."""
        sample = bkd.asarray([[0.8], [-0.6]])
        energy = _make_problem(bkd, LINEAR, StrainEnergy()).function()(sample)
        tip = _make_problem(bkd, LINEAR, TipDisplacement()).function()(sample)
        mean_u = _make_problem(bkd, LINEAR, AverageDisplacement()).function()(
            sample
        )
        bkd.assert_allclose(2.0 * energy, tip + mean_u, rtol=1e-8)
