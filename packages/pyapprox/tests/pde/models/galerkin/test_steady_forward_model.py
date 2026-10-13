"""GalerkinSteadyForwardModel: values, capability tiers, validation.

Setup: -(kappa u')' (+ optional u^2 reaction) = 1 on [0, 1],
u(0) = u(1) = 0, P1 on 10 cells, kappa = exp(W p) with three smooth
modes. The scalar QoI is the flux out of x = 0
(EssentialBoundaryFluxFunctional); the default QoI is every state.
"""

from typing import Any, List, Literal, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
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
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    EssentialBoundaryFluxFunctional,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
)
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
    GalerkinSteadyForwardModel,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NumpyArray

_NPARAMS = 3
_PARAM = np.array([[0.4], [-0.3], [0.2]])


def _unit_forcing(x: Any) -> Any:
    return np.ones(x.shape[1])


def _square(x: Any, u: Any) -> Any:
    return u**2


def _square_deriv(x: Any, u: Any) -> Any:
    return 2.0 * u


def _square_deriv2(x: Any, u: Any) -> Any:
    return np.full_like(u, 2.0)


def _problem(
    bkd: NumpyBkd, reaction: Literal["none", "full", "no_curvature"]
) -> Tuple[
    GalerkinSystem[NumpyArray],
    AdvectionDiffusionParameterization[NumpyArray],
    LagrangeBasis[NumpyArray],
]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    reaction_law = {
        "none": None,
        "full": CallableReaction(_square, _square_deriv, _square_deriv2),
        # No second derivative: the physics declares no state curvature.
        "no_curvature": CallableReaction(_square, _square_deriv),
    }[reaction]
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        reaction=reaction_law,
        forcing=TimeIndependent(_unit_forcing),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
        ],
    )
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = np.stack(
        [
            0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1)
            for k in range(_NPARAMS)
        ],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    param_obj = AdvectionDiffusionParameterization(
        physics,
        diffusivity_map=TransformedFieldMap(
            kle, exp, exp, bkd, transform_deriv2=exp
        ),
        bkd=bkd,
    )
    return system, param_obj, basis


def _left_flux(
    bkd: NumpyBkd,
    system: GalerkinSystem[NumpyArray],
    param_obj: AdvectionDiffusionParameterization[NumpyArray],
    basis: LagrangeBasis[NumpyArray],
) -> EssentialBoundaryFluxFunctional[NumpyArray]:
    left = int(bkd.to_numpy(basis.get_dofs("left"))[0])
    return EssentialBoundaryFluxFunctional(
        GalerkinParameterizedResidual(system.spatial_operator(), param_obj),
        system.steady(),
        bkd.asarray(np.array([left]), dtype=bkd.int64_dtype()),
        bkd.ones((1, 1)),
    )


class _ValueOnlyFunctional:
    """A steady QoI with no derivatives: the sum of the state."""

    def __init__(self, bkd: NumpyBkd, nstates: int) -> None:
        self._bkd = bkd
        self._nstates = nstates

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return self._nstates

    def nparams(self) -> int:
        return _NPARAMS

    def nunique_params(self) -> int:
        return 0

    def time_dependence(self) -> Literal["steady"]:
        return "steady"

    def derivatives(self) -> SteadyFunctionalDerivatives[Any]:
        return SteadyFunctionalDerivatives.none()

    def __call__(self, state: Any, param: Any) -> Any:
        return self._bkd.reshape(self._bkd.sum(state), (1, 1))


def _derivatives_pass_fd(
    bkd: NumpyBkd, model: GalerkinSteadyForwardModel[NumpyArray]
) -> List[bool]:
    """Finite-difference checks of every declared derivative.

    Each model evaluation goes through a Newton solve, whose results are
    already at roundoff (a tighter tolerance changes nothing), and the
    roundoff floor sits near an error ratio of 1e-6, so whether the usual
    ``ratio <= 1e-6`` or a V-shape fit passes depends on the random
    direction (worst ratio over ten seeds: 3.9e-6). Each derivative must
    instead satisfy both:

    - error ratio at most 5e-6 over steps 1e-13 to 1e-1, the bound used
      for noise-limited checks elsewhere;
    - first-order convergence over steps 1e-1 to 1e-5, the truncation
      regime, which is clean for every direction; a wrong derivative
      does not converge there.

    The components are checked without repeated solves in
    test_essential_boundary_flux.
    """
    checker = DerivativeChecker(model)
    fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
    passed = []
    for errors in checker.check_derivatives(
        bkd.asarray(_PARAM), fd_eps=fd_eps, relative=True
    ):
        ratio = bkd.to_float(checker.error_ratio(errors))
        # fd_eps descends, so the first five steps are 1e-1 to 1e-5.
        order = bkd.to_float(
            checker.convergence_order(errors[:5], fd_eps[:5])
        )
        passed.append(ratio <= 5e-6 and abs(order - 1.0) < 0.1)
    return passed


class TestValues:
    def test_satisfies_function_protocol(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, "none")
        model = GalerkinSteadyForwardModel(
            system.steady(), param_obj, bkd.zeros((system.nstates(),)), bkd
        )
        assert isinstance(model, FunctionProtocol)

    def test_values_match_a_manual_solve(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, "full")
        functional = _left_flux(bkd, system, param_obj, basis)
        model = GalerkinSteadyForwardModel(
            system.steady(),
            param_obj,
            bkd.zeros((system.nstates(),)),
            bkd,
            functional=functional,
        )
        samples = bkd.asarray(np.hstack([_PARAM, -_PARAM, 0 * _PARAM]))
        state_eq = GalerkinParameterizedStateEquation(
            system.steady(), param_obj, bkd
        )
        expected = bkd.hstack(
            [
                functional(
                    state_eq.solve(
                        bkd.zeros((system.nstates(), 1)),
                        samples[:, ii : ii + 1],
                    ),
                    samples[:, ii : ii + 1],
                )
                for ii in range(3)
            ]
        )
        bkd.assert_allclose(model(samples), expected, rtol=1e-12)

    def test_default_qoi_is_every_state(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, "none")
        model = GalerkinSteadyForwardModel(
            system.steady(), param_obj, bkd.zeros((system.nstates(),)), bkd
        )
        assert model.nqoi() == system.nstates()
        assert model(bkd.asarray(_PARAM)).shape == (system.nstates(), 1)


class TestCapability:
    def test_scalar_qoi_gives_gradient_and_hvp(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, basis = _problem(bkd, "full")
        model = GalerkinSteadyForwardModel(
            system.steady(),
            param_obj,
            bkd.zeros((system.nstates(),)),
            bkd,
            functional=_left_flux(bkd, system, param_obj, basis),
        )
        derivs = model.derivatives()
        assert derivs.jacobian is not None and derivs.hvp is not None
        assert all(_derivatives_pass_fd(bkd, model))

    def test_vector_qoi_gives_forward_sensitivity_jacobian(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, "full")
        model = GalerkinSteadyForwardModel(
            system.steady(), param_obj, bkd.zeros((system.nstates(),)), bkd
        )
        derivs = model.derivatives()
        assert derivs.jacobian is not None and derivs.hvp is None
        assert all(_derivatives_pass_fd(bkd, model))

    def test_value_only_functional_gives_no_derivatives(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, "none")
        model = GalerkinSteadyForwardModel(
            system.steady(),
            param_obj,
            bkd.zeros((system.nstates(),)),
            bkd,
            functional=_ValueOnlyFunctional(bkd, system.nstates()),
        )
        assert model.derivatives().jacobian is None
        assert model(bkd.asarray(_PARAM)).shape == (1, 1)

    def test_missing_state_curvature_warns_and_downgrades(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A weighted sum of states has HVPs regardless of the physics, so
        the physics' missing curvature alone costs the HVP. (The flux
        functional would lose its own HVPs too, since they come from the
        same curvature, so it would not be a second-order functional.)"""
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, "no_curvature")
        weights = bkd.copy(bkd.zeros((system.nstates(), 1)))
        weights[system.nstates() // 2] = 1.0
        with pytest.warns(UserWarning, match="state curvature"):
            model = GalerkinSteadyForwardModel(
                system.steady(),
                param_obj,
                bkd.zeros((system.nstates(),)),
                bkd,
                functional=WeightedSumFunctional(weights, _NPARAMS, bkd),
            )
        derivs = model.derivatives()
        assert derivs.jacobian is not None and derivs.hvp is None


class TestValidation:
    def test_rejects_a_transient_functional(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, "none")
        with pytest.raises(TypeError, match="transient"):
            GalerkinSteadyForwardModel(
                system.steady(),
                param_obj,
                bkd.zeros((system.nstates(),)),
                bkd,
                functional=EndpointFunctional(  # type: ignore[arg-type]
                    0, system.nstates(), _NPARAMS, bkd
                ),
            )

    def test_rejects_a_non_functional(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, "none")
        with pytest.raises(TypeError, match="SteadyFunctionalProtocol"):
            GalerkinSteadyForwardModel(
                system.steady(),
                param_obj,
                bkd.zeros((system.nstates(),)),
                bkd,
                functional=object(),  # type: ignore[arg-type]
            )

    def test_rejects_wrong_init_state_shape(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, param_obj, _ = _problem(bkd, "none")
        with pytest.raises(ValueError, match="init_state"):
            GalerkinSteadyForwardModel(
                system.steady(),
                param_obj,
                bkd.zeros((system.nstates(), 1)),
                bkd,
            )

    def test_rejects_unowned_parameterization(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        system, _, _ = _problem(bkd, "none")
        _, other_param, _ = _problem(bkd, "none")
        with pytest.raises(ValueError, match="does not hold"):
            GalerkinSteadyForwardModel(
                system.steady(),
                other_param,
                bkd.zeros((system.nstates(),)),
                bkd,
            )
