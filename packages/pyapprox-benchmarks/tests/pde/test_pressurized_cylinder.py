"""Integration tests for the 2D pressurized cylinder forward UQ problems."""

import math
from typing import Any, Dict, Generic, List

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.field_maps.lame import FixedPoissonRatioLameMap
from pyapprox.pde.models.collocation.steady_function import (
    collocation_steady_function,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox_benchmarks.pde.pressurized_cylinder import (
    AverageHoopStress,
    CylinderConstitutiveProtocol,
    CylinderQoIContext,
    CylinderQoIProtocol,
    LinearElastic,
    NeoHookean,
    OuterRadialDisplacement,
    StrainEnergy,
    build_pressurized_cylinder_2d,
)
from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem
from tests._helpers.markers import slow_test, slower_test, slowest_test

_Problem = ForwardUQProblem[SteadyImplicitFunction[Array, DenseFormat[Array]], Array]
_Function = SteadyImplicitFunction[Array, DenseFormat[Array]]

_QOIS: List[CylinderQoIProtocol] = [
    OuterRadialDisplacement(),
    AverageHoopStress(0.25),
    StrainEnergy(),
]


def _make_problem(
    bkd: Backend[Array],
    qoi: CylinderQoIProtocol = OuterRadialDisplacement(),
    npts_r: int = 10,
    npts_theta: int = 10,
    num_kle_terms: int = 2,
) -> _Problem[Array]:
    """Helper for creating problems with small defaults for fast tests."""
    return build_pressurized_cylinder_2d(
        bkd,
        LinearElastic(),
        qoi,
        npts_r=npts_r,
        npts_theta=npts_theta,
        r_inner=1.0,
        r_outer=2.0,
        poisson_ratio=0.3,
        inner_pressure=1.0,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
    )


def _check_jacobian(
    bkd: Backend[Array], fwd: _Function[Array], num_kle_terms: int = 2
) -> None:
    """Helper: run DerivativeChecker on a forward model."""
    # Read the capability from the bundle rather than reaching for a
    # method. Absence is None there, so a model without an analytical
    # jacobian reports that instead of raising AttributeError.
    assert fwd.derivatives().jacobian is not None
    checker = DerivativeChecker(fwd)
    np.random.seed(42)
    sample = bkd.array([0.1, -0.1][:num_kle_terms])[:, None]
    errors = checker.check_derivatives(sample, relative=True)[0]
    assert checker.error_ratio(errors) <= 1e-5


@slow_test
class TestPressurizedCylinder2D:
    _class_bkd: NumpyBkd
    _cached_probs: Dict[str, _Problem[NDArray[Any]]]

    @classmethod
    def setup_class(cls) -> None:
        cls._class_bkd = NumpyBkd()
        bkd = cls._class_bkd
        cls._cached_probs = {}
        for qoi in _QOIS:
            cls._cached_probs[qoi.label()] = _make_problem(bkd, qoi)

    # --- Evaluation tests ---

    def test_outer_displacement_evaluate(self) -> None:
        """Outer radial displacement: shape (1,1), positive value."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        sample = bkd.zeros((2, 1))
        result = fwd(sample)
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_average_hoop_stress_evaluate(self) -> None:
        """Average hoop stress: shape (1,1), positive value."""
        bkd = self._class_bkd
        prob = self._cached_probs["average_hoop_stress"]
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_strain_energy_evaluate(self) -> None:
        """Strain energy: shape (1,1), positive value."""
        bkd = self._class_bkd
        prob = self._cached_probs["strain_energy"]
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    # --- Jacobian tests ---

    def test_outer_displacement_jacobian(self) -> None:
        """DerivativeChecker on outer radial displacement."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        _check_jacobian(bkd, prob.function())

    def test_average_hoop_stress_jacobian(self) -> None:
        """DerivativeChecker on average hoop stress."""
        bkd = self._class_bkd
        prob = self._cached_probs["average_hoop_stress"]
        _check_jacobian(bkd, prob.function())

    def test_strain_energy_jacobian(self) -> None:
        """DerivativeChecker on strain energy."""
        bkd = self._class_bkd
        prob = self._cached_probs["strain_energy"]
        _check_jacobian(bkd, prob.function())

    # --- All QoIs produce scalar ---

    def test_all_qois_produce_scalar(self) -> None:
        """All 3 QoIs return nqoi=1 and shape (1,1)."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        for qoi in self._cached_probs:
            prob = self._cached_probs[qoi]
            fwd = prob.function()
            assert fwd.nqoi() == 1, f"Failed for qoi={qoi}"
            result = fwd(sample)
            assert result.shape == (1, 1), f"Failed for qoi={qoi}"

    # --- Prior ---

    def test_prior_samples_shape(self) -> None:
        """Prior generates samples of correct shape."""
        prob = self._cached_probs["outer_radial_displacement"]
        np.random.seed(42)
        samples = prob.prior().rvs(5)
        assert samples.shape == (2, 5)

    # --- Protocol compliance ---

    def test_function_protocol_compliance(self) -> None:
        """Forward model is Function-shaped with a jacobian."""
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        assert isinstance(fwd, FunctionProtocol)
        assert callable(fwd.derivatives().jacobian)

    # --- Convergence (non-default params, builds fresh) ---

    def test_convergence(self) -> None:
        """Coarse and fine mesh outer displacement agree."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        prob_coarse = _make_problem(
            bkd,
            OuterRadialDisplacement(),
            npts_r=8,
            npts_theta=8,
        )
        prob_fine = _make_problem(
            bkd,
            OuterRadialDisplacement(),
            npts_r=14,
            npts_theta=14,
        )
        val_coarse = prob_coarse.function()(sample)
        val_fine = prob_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-2)

    # --- Error handling ---

    def test_invalid_qoi_raises(self) -> None:
        """A QoI not satisfying CylinderQoIProtocol raises TypeError."""
        bkd = self._class_bkd
        with pytest.raises(TypeError, match="CylinderQoIProtocol"):
            build_pressurized_cylinder_2d(bkd, qoi="max_stress")  # type: ignore[arg-type]

    def test_invalid_constitutive_raises(self) -> None:
        """A law not satisfying CylinderConstitutiveProtocol raises
        TypeError."""
        bkd = self._class_bkd
        with pytest.raises(TypeError, match="CylinderConstitutiveProtocol"):
            build_pressurized_cylinder_2d(bkd, constitutive="linear")  # type: ignore[arg-type]

    def test_problem_names(self) -> None:
        """The name carries the law's and the QoI's labels."""
        assert set(self._cached_probs) == {
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        }
        prob = self._cached_probs["average_hoop_stress"]
        assert prob.name() == "pressurized_cylinder_2d_linear_average_hoop_stress"

    # --- Multi-sample evaluation ---

    def test_batch_evaluation(self) -> None:
        """Forward model handles batch evaluation."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        samples = bkd.array([[0.1, -0.1, 0.2], [0.05, 0.0, -0.15]])
        result = fwd(samples)
        assert result.shape == (1, 3)


# ======================================================================
# Hyperelastic tests
# ======================================================================


def _make_hyperelastic_problem(
    bkd: Backend[Array],
    qoi: CylinderQoIProtocol = OuterRadialDisplacement(),
    npts_r: int = 10,
    npts_theta: int = 10,
    num_kle_terms: int = 2,
    inner_pressure: float = 1.0,
) -> _Problem[Array]:
    """Helper for creating hyperelastic problems with small defaults."""
    return build_pressurized_cylinder_2d(
        bkd,
        NeoHookean(),
        qoi,
        npts_r=npts_r,
        npts_theta=npts_theta,
        r_inner=1.0,
        r_outer=2.0,
        poisson_ratio=0.3,
        inner_pressure=inner_pressure,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
    )


def _check_hyperelastic_jacobian(
    bkd: Backend[Array], fwd: _Function[Array], num_kle_terms: int = 2
) -> None:
    """Helper: run DerivativeChecker on a hyperelastic forward model."""
    assert fwd.derivatives().jacobian is not None
    checker = DerivativeChecker(fwd)
    np.random.seed(42)
    sample = bkd.array([0.1, -0.1][:num_kle_terms])[:, None]
    errors = checker.check_derivatives(sample, relative=True)[0]
    assert checker.error_ratio(errors) <= 2e-5


class TestHyperelasticPressurizedCylinder2D:
    _class_bkd: NumpyBkd
    _cached_probs: Dict[str, _Problem[NDArray[Any]]]

    @classmethod
    def setup_class(cls) -> None:
        cls._class_bkd = NumpyBkd()
        bkd = cls._class_bkd
        cls._cached_probs = {}
        for qoi in _QOIS:
            cls._cached_probs[qoi.label()] = _make_hyperelastic_problem(bkd, qoi)

    # --- Evaluation ---

    @slow_test
    def test_hyperelastic_evaluate(self) -> None:
        """Hyperelastic: evaluate at theta=0, check shape (1,1)."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        sample = bkd.zeros((2, 1))
        result = fwd(sample)
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    # --- Jacobian ---

    @slower_test
    def test_hyperelastic_jacobian(self) -> None:
        """DerivativeChecker on hyperelastic outer radial displacement."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        _check_hyperelastic_jacobian(bkd, prob.function())

    # --- Linear vs hyperelastic at low pressure (non-default params) ---

    @slow_test
    def test_linear_vs_hyperelastic(self) -> None:
        """At low pressure, linear and hyperelastic produce similar QoI."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        prob_hyper = _make_hyperelastic_problem(
            bkd,
            OuterRadialDisplacement(),
            npts_r=10,
            npts_theta=10,
            inner_pressure=1e-3,
        )
        prob_linear_low = build_pressurized_cylinder_2d(
            bkd,
            LinearElastic(),
            OuterRadialDisplacement(),
            npts_r=10,
            npts_theta=10,
            r_inner=1.0,
            r_outer=2.0,
            poisson_ratio=0.3,
            inner_pressure=1e-3,
            num_kle_terms=2,
            sigma=0.3,
        )
        val_linear = prob_linear_low.function()(sample)
        val_hyper = prob_hyper.function()(sample)
        bkd.assert_allclose(val_hyper, val_linear, rtol=1e-2)

    # --- All three QoIs ---

    @slowest_test
    def test_all_three_qois_hyperelastic(self) -> None:
        """Each QoI produces scalar (1,1) with working Jacobian."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        for qoi in self._cached_probs:
            prob = self._cached_probs[qoi]
            fwd = prob.function()
            assert fwd.nqoi() == 1, f"Failed for qoi={qoi}"
            result = fwd(sample)
            assert result.shape == (1, 1), f"Failed for qoi={qoi}"
            _check_hyperelastic_jacobian(bkd, fwd)

    # --- Convergence (non-default params) ---

    @slow_test
    def test_convergence_reference_hyperelastic(self) -> None:
        """Finer mesh resolution produces closer values (convergence)."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        prob_coarse = _make_hyperelastic_problem(
            bkd,
            OuterRadialDisplacement(),
            npts_r=8,
            npts_theta=8,
        )
        prob_fine = _make_hyperelastic_problem(
            bkd,
            OuterRadialDisplacement(),
            npts_r=14,
            npts_theta=14,
        )
        val_coarse = prob_coarse.function()(sample)
        val_fine = prob_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-2)


# ======================================================================
# Stress QoIs read the material the state was solved with
# ======================================================================


class _UniformLogModulus(Generic[Array]):
    """A one-parameter field map: E = exp(p_0) at every node."""

    def __init__(self, bkd: Backend[Array], npts: int) -> None:
        self._bkd = bkd
        self._npts = npts

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def __call__(self, params_1d: Array) -> Array:
        return self._bkd.exp(params_1d[0]) * self._bkd.ones((self._npts,))

    def jacobian(self, params_1d: Array) -> Array:
        return self(params_1d)[:, None]


def _uniform_modulus_model(
    bkd: Backend[Array],
    qoi: CylinderQoIProtocol,
    constitutive: CylinderConstitutiveProtocol,
    pressure: float,
    npts: int = 8,
) -> _Function[Array]:
    """The benchmark's model and QoI, with a uniform modulus exp(p_0)."""
    transform = PolarTransform((1.0, 2.0), (0.0, math.pi / 2.0), bkd)
    mesh = TransformedMesh2D(npts, npts, bkd, transform)
    field_map = _UniformLogModulus(bkd, mesh.npts())
    configured = constitutive.configure(mesh, 0.3, pressure, field_map)
    parameters = ParameterState(configured.parameterization)
    basis = configured.basis
    curvilinear_basis = transform.unit_curvilinear_basis(mesh.reference_points())
    context = CylinderQoIContext(
        mesh,
        basis,
        parameters,
        FixedPoissonRatioLameMap(field_map, 0.3, basis.npts(), bkd),
        constitutive.hoop_stress_density(basis, curvilinear_basis),
        constitutive.strain_energy_density(basis, curvilinear_basis),
        bkd,
    )
    functional = StateFunctional.from_terms(
        bkd, 1, state_terms=(qoi.term(context),)
    )
    return collocation_steady_function(configured.system, parameters, functional)


class TestStressQoIMaterial:
    """Scaling laws that hold only if a stress QoI reads the modulus the
    state was solved with. A QoI reading a fixed material breaks each:
    its stress then scales as 1/E."""

    def test_linear_hoop_stress_independent_of_modulus(self) -> None:
        """Traction loading fixes the linear stress whatever the uniform
        modulus: Q(E=3) = Q(E=1), and dQ/dp_0 = 0."""
        bkd = NumpyBkd()
        fwd = _uniform_modulus_model(bkd, AverageHoopStress(), LinearElastic(), 1.0)
        q1 = fwd(bkd.zeros((1, 1)))
        q3 = fwd(bkd.full((1, 1), math.log(3.0)))
        bkd.assert_allclose(q3, q1, rtol=1e-8)
        jacobian = fwd.derivatives().jacobian
        assert jacobian is not None
        grad = jacobian(bkd.full((1, 1), math.log(3.0)))
        assert abs(bkd.to_float(grad[0, 0])) <= 1e-8 * abs(bkd.to_float(q1[0, 0]))

    def test_linear_strain_energy_inverse_in_modulus(self) -> None:
        """Displacement scales as 1/E at fixed traction, so the strain
        energy does too: Q(E=3) = Q(E=1)/3, and dQ/dp_0 = -Q."""
        bkd = NumpyBkd()
        fwd = _uniform_modulus_model(bkd, StrainEnergy(), LinearElastic(), 1.0)
        p3 = bkd.full((1, 1), math.log(3.0))
        q1 = fwd(bkd.zeros((1, 1)))
        q3 = fwd(p3)
        bkd.assert_allclose(q3, q1 / 3.0, rtol=1e-8)
        jacobian = fwd.derivatives().jacobian
        assert jacobian is not None
        bkd.assert_allclose(jacobian(p3), -q3, rtol=1e-6)

    @pytest.mark.parametrize("qoi", [AverageHoopStress(), StrainEnergy()])
    def test_neo_hookean_scaling(self, qoi: CylinderQoIProtocol) -> None:
        """Neo-Hookean stress and energy are linear in (mu, lamda) at
        fixed F, so scaling the modulus and the pressure by c leaves the
        displacement unchanged and scales the QoI by c."""
        bkd = NumpyBkd()
        c = 3.0
        q_base = _uniform_modulus_model(bkd, qoi, NeoHookean(), 0.2)(
            bkd.zeros((1, 1))
        )
        q_scaled = _uniform_modulus_model(bkd, qoi, NeoHookean(), c * 0.2)(
            bkd.full((1, 1), math.log(c))
        )
        bkd.assert_allclose(q_scaled, c * q_base, rtol=1e-8)
