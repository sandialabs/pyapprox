"""Worked transient gradient + HVP example through the public model API.

The capstone of the galerkin adjoint stack: a 2D transient
advection-diffusion problem with a bilaplacian (Whittle-Matern)
log-diffusivity KLE, driven entirely through
``GalerkinTransientForwardModel`` (``__call__`` ->
``derivatives().jacobian`` -> ``derivatives().hvp``) with a
mass-weighted subdomain-average QoI. Component tiers below this
(assemblies, engine terms, steppers, BC wrappers) each have their own
FD suites; this file validates the composed public surface:

- gradient and HVP vs DerivativeChecker FD for each parameterized
  coefficient (diffusivity/forcing/velocity/composite), BE and CN;
- the HVP symmetry identity <Hu, v> = <Hv, u> (FD-noise immune);
- the scalar-adjoint row vs the shared tangent-linear sensitivity
  matrix (adjoint and forward propagation must agree exactly);
- a Tikhonov-augmented QoI (nonzero direct dQ/dp and d2Q/dp2 —
  otherwise those functional pathways are untested) evaluated at p = 0
  where the parameterized forcing field VANISHES at the test point
  (the configuration where wrongly-Zeroed derivative paths surface);
- the quasilinear kappa(u)*a(x) physics (genuine mixed
  state-parameter curvature, invisible to gradient-only tests).

FD calibration: one-sided FD stacked on Newton-tolerance transient
solves floors near 1e-7 relative, so the raw 1e-6 ratio target is not
met by ANY transient FD check in the repo; gradients assert
``error_ratio <= 2e-5`` (the transient collocation precedent) and HVPs
assert the V-bottom plus a loose ratio, with the symmetry identity
carrying the tight validation.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.functionals.weighted_endpoint import (
    WeightedEndpointFunctional,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldVelocity,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.kle_factory import (
    create_spde_lognormal_kle_field_map,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction,
    QuasilinearDiffusion,
)
from pyapprox.pde.models.galerkin.transient import (
    GalerkinTransientForwardModel,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.galerkin_quasilinear_diffusion import (
    create_quasilinear_diffusivity_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd

from tests._helpers.adjoint_checks import NumpyArray

_N_KLE_MODES = 4
_N_FORCING_MODES = 2


def _kappa(u):
    return 1.0 + u**2


def _kappa_deriv(u):
    return 2.0 * u


def _kappa_second_deriv(u):
    return np.full_like(u, 2.0)


def _time_config(method: str) -> TimeIntegrationConfig[NumpyArray]:
    return TimeIntegrationConfig(
        method=method,
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-12,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )


def _build_physics(bkd: NumpyBkd, cubic_reaction: bool = False):
    """2D unit-square ADR: nodal diffusivity/forcing, constant (1, 0)
    velocity, homogeneous Dirichlet. The optional CUBIC reaction is the
    only state-nonlinear term, and unlike the component tiers'
    quadratic variant its second derivative R'' = 6u is
    state-dependent — per-step staleness in the second-adjoint RHS
    would surface here and nowhere else."""
    mesh = StructuredMesh2D(
        nx=6, ny=6, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
    )
    basis = LagrangeBasis(mesh, degree=1)
    vel_basis = VectorLagrangeBasis(mesh, degree=1)
    vel_dofs = np.zeros(vel_basis.ndofs())
    vel_dofs[0::2] = 1.0  # constant velocity (1, 0)
    reaction = (
        CallableReaction(
            lambda x, u: u**3,
            lambda x, u: 3.0 * u**2,
            lambda x, u: 6.0 * u,
        )
        if cubic_reaction
        else None
    )
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(
            basis, dofs=0.1 * np.ones(basis.ndofs())
        ),
        bkd=bkd,
        velocity=NodalFieldVelocity(vel_basis, vel_dofs),
        reaction=reaction,
        forcing=NodalFieldForcing(basis, dofs=np.ones(basis.ndofs())),
        boundary_conditions=[
            DirichletBC(basis, name, 0.0, bkd)
            for name in ("left", "right", "bottom", "top")
        ],
    )
    return physics, basis, vel_basis


def _diffusivity_map(bkd: NumpyBkd, basis):
    """Bilaplacian-prior lognormal KLE (correlation length ~0.5)."""
    nnodes = basis.ndofs()
    return create_spde_lognormal_kle_field_map(
        basis,
        bkd.asarray(np.log(0.1) * np.ones(nnodes)),
        bkd,
        n_modes=_N_KLE_MODES,
        gamma=1.0,
        delta=4.0,
        sigma=0.3,
    )


def _forcing_map(bkd: NumpyBkd, basis):
    """Linear KLE with zero mean: the forcing field VANISHES at p = 0."""
    coords = bkd.to_numpy(basis.dof_coordinates())
    modes = np.stack(
        [
            np.sin((k + 1) * np.pi * coords[0]) * np.sin(np.pi * coords[1])
            for k in range(_N_FORCING_MODES)
        ],
        axis=1,
    )
    return MeshKLEFieldMap(
        bkd,
        bkd.asarray(np.zeros(coords.shape[1])),
        bkd.asarray(modes),
    )


def _velocity_map(bkd: NumpyBkd, vel_basis):
    """Linear KLE over the interleaved velocity DOFs around (1, 0)."""
    nvel = vel_basis.ndofs()
    mean = np.zeros(nvel)
    mean[0::2] = 1.0
    rng = np.random.default_rng(41)
    modes = 0.3 * rng.normal(0.0, 1.0, (nvel, _N_FORCING_MODES))
    return MeshKLEFieldMap(bkd, bkd.asarray(mean), bkd.asarray(modes))


def _build_parameterization(bkd: NumpyBkd, case: str):
    physics, basis, vel_basis = _build_physics(
        bkd, cubic_reaction=case.startswith("cubic")
    )
    maps = {}
    if case in ("diffusivity", "composite", "cubic-exp"):
        maps["diffusivity_map"] = _diffusivity_map(bkd, basis)
    if case in ("forcing", "composite", "cubic-linear"):
        maps["forcing_map"] = _forcing_map(bkd, basis)
    if case == "velocity":
        maps["velocity_map"] = _velocity_map(bkd, vel_basis)
    param_obj = AdvectionDiffusionParameterization(physics, bkd=bkd, **maps)
    return physics, param_obj


def _subdomain_average_weights(bkd: NumpyBkd, physics) -> NumpyArray:
    """Mass-weighted subdomain average: c = M @ indicator(x > 0.5)."""
    coords = bkd.to_numpy(physics.basis().dof_coordinates())
    indicator = (coords[0] > 0.5).astype(float)
    mass = physics.mass_matrix()
    cvec = np.asarray(mass @ indicator).flatten()
    return bkd.asarray(cvec.reshape(-1, 1))


def _gaussian_bump_ic(bkd: NumpyBkd, physics) -> NumpyArray:
    coords = bkd.to_numpy(physics.basis().dof_coordinates())
    vals = np.exp(
        -50.0 * ((coords[0] - 0.4) ** 2 + (coords[1] - 0.5) ** 2)
    )
    return bkd.asarray(vals)


def _make_model(
    bkd: NumpyBkd, case: str, method: str, functional=None
) -> GalerkinTransientForwardModel[NumpyArray]:
    physics, param_obj = _build_parameterization(bkd, case)
    if functional is None:
        functional = WeightedEndpointFunctional(
            _subdomain_average_weights(bkd, physics),
            param_obj.nparams(),
            bkd,
        )
    return GalerkinTransientForwardModel(
        physics,
        param_obj,
        _gaussian_bump_ic(bkd, physics),
        _time_config(method),
        bkd,
        functional=functional,
    )


def _sample_and_direction(nparams: int):
    rng = np.random.default_rng(7)
    sample = 0.4 * rng.normal(0.0, 1.0, (nparams, 1))
    direction = rng.normal(0.0, 1.0, (nparams, 1))
    return sample, direction


def _check_gradient_and_hvp(
    bkd: NumpyBkd, model, expect_zero_hessian: bool = False
) -> None:
    """DerivativeChecker on the model itself + HVP symmetry identity.

    ``expect_zero_hessian`` covers the linear-in-parameters cases
    (linear map onto a linearly-entering coefficient, linear QoI):
    Q(p) is affine, so the machinery must return an EXACTLY zero HVP
    (relative FD ratios are meaningless around zero).
    """
    sample_np, direction_np = _sample_and_direction(model.nvars())
    sample = bkd.asarray(sample_np)
    direction = bkd.asarray(direction_np)
    hvp_fn = model.derivatives().hvp
    assert hvp_fn is not None

    if expect_zero_hessian:
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            sample, direction=direction, relative=True
        )
        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 2e-5
        hvp_val = hvp_fn(sample, direction)
        bkd.assert_allclose(
            hvp_val, bkd.zeros(hvp_val.shape), atol=1e-14, rtol=0.0
        )
        return

    checker = DerivativeChecker(model)
    errors = checker.check_derivatives(
        sample, direction=direction, relative=True
    )
    jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
    assert jac_ratio <= 2e-5
    hvp_min = float(bkd.to_numpy(bkd.min(errors[1])))
    assert hvp_min <= 1e-6
    hvp_ratio = float(bkd.to_numpy(checker.error_ratio(errors[1])))
    assert hvp_ratio <= 1e-5

    other = bkd.asarray(
        np.random.default_rng(11).normal(0.0, 1.0, (model.nvars(), 1))
    )
    h_dir = bkd.flatten(hvp_fn(sample, direction))
    h_other = bkd.flatten(hvp_fn(sample, other))
    bkd.assert_allclose(
        bkd.sum(h_dir * bkd.flatten(other)),
        bkd.sum(h_other * bkd.flatten(direction)),
        rtol=1e-12,
    )


class _TikhonovWeightedEndpoint(WeightedEndpointFunctional[NumpyArray]):
    """Q = c^T y(T) + (alpha/2) ||p||^2: nonzero direct parameter
    derivatives (every plain-endpoint case has dQ/dp == 0, leaving the
    functional param pathways untested)."""

    def __init__(self, weights, nparams, alpha, bkd):
        super().__init__(weights, nparams, bkd)
        self._alpha = alpha

    def __call__(self, sol, param):
        base = super().__call__(sol, param)
        return base + self._bkd.reshape(
            0.5 * self._alpha * self._bkd.sum(param[:, 0] * param[:, 0]),
            (1, 1),
        )

    def param_jacobian(self, sol, param):
        return self._alpha * param.T

    def param_state_hvp(self, sol, param, time_idx, wvec):
        return self._bkd.zeros((self.nparams(), 1))

    def param_param_hvp(self, sol, param, vvec):
        return self._alpha * vvec


class TestTransientAdjointWorkedExample:
    @pytest.mark.parametrize(
        "case",
        [
            # One-nonlinearity-at-a-time diagonal: each case activates
            # exactly one second-order pathway (plus the composite).
            "diffusivity",  # param_param via exp-map curvature
            "forcing",  # fully linear: Hessian exactly zero
            "velocity",  # linear map through the advection term
            "composite",  # diffusivity+forcing block interactions
            "cubic-linear",  # cubic reaction + linear map: state_state
            "cubic-exp",  # cubic reaction + exp map: cross terms
        ],
    )
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_gradient_and_hvp_match_fd(
        self, numpy_bkd: NumpyBkd, method: str, case: str
    ) -> None:
        bkd = numpy_bkd
        model = _make_model(bkd, case, method)
        assert model.derivatives().hvp is not None
        # The forcing case is linear end to end (linear map, linearly
        # entering coefficient, linear QoI): Q(p) is affine and the
        # exact Hessian is zero. The cubic-linear case shares the
        # linear forcing map but its Hessian is NONZERO: the state
        # enters Q through the cubic reaction.
        _check_gradient_and_hvp(
            bkd, model, expect_zero_hessian=(case == "forcing")
        )

    def test_scalar_adjoint_matches_forward_sensitivity_row(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Adjoint gradient row == c^T W_T with W_T from the shared
        tangent-linear sweep (backward and forward propagation of the
        same linearization must agree to solver precision)."""
        bkd = numpy_bkd
        # ONE physics + parameterization shared by both models: separate
        # KLE constructions are distinct parameterizations (eigensolver
        # sign/ordering indeterminacy), which would make the comparison
        # meaningless.
        physics, param_obj = _build_parameterization(bkd, "diffusivity")
        ic = _gaussian_bump_ic(bkd, physics)
        config = _time_config("backward_euler")
        scalar_model = GalerkinTransientForwardModel(
            physics,
            param_obj,
            ic,
            config,
            bkd,
            functional=WeightedEndpointFunctional(
                _subdomain_average_weights(bkd, physics),
                param_obj.nparams(),
                bkd,
            ),
        )
        vector_model = GalerkinTransientForwardModel(
            physics, param_obj, ic, config, bkd
        )
        sample_np, _ = _sample_and_direction(scalar_model.nvars())
        sample = bkd.asarray(sample_np)
        scalar_jac = scalar_model.derivatives().jacobian
        vector_jac = vector_model.derivatives().jacobian
        assert scalar_jac is not None
        assert vector_jac is not None
        grad = bkd.to_numpy(scalar_jac(sample))
        w_final = bkd.to_numpy(vector_jac(sample))
        weights = bkd.to_numpy(
            _subdomain_average_weights(bkd, scalar_model.physics())
        )[:, 0]
        bkd.assert_allclose(
            bkd.asarray(grad.flatten()),
            bkd.asarray(weights @ w_final),
            rtol=1e-10,
        )

    def test_tikhonov_functional_at_zero_forcing_field(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Tikhonov-augmented QoI at p = 0, where the parameterized
        forcing field vanishes identically at the test point: FD
        perturbs p, the field leaves zero, and any wrongly-Zeroed or
        value/structure-conflating derivative path disagrees
        immediately. The direct dQ/dp and d2Q/dp2 terms are nonzero."""
        bkd = numpy_bkd
        physics, param_obj = _build_parameterization(bkd, "forcing")
        functional = _TikhonovWeightedEndpoint(
            _subdomain_average_weights(bkd, physics),
            param_obj.nparams(),
            0.7,
            bkd,
        )
        model = GalerkinTransientForwardModel(
            physics,
            param_obj,
            _gaussian_bump_ic(bkd, physics),
            _time_config("backward_euler"),
            bkd,
            functional=functional,
        )
        sample = bkd.zeros((model.nvars(), 1))
        direction = bkd.asarray(
            np.random.default_rng(13).normal(0.0, 1.0, (model.nvars(), 1))
        )
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            sample, direction=direction, relative=True
        )
        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 2e-5
        hvp_min = float(bkd.to_numpy(bkd.min(errors[1])))
        assert hvp_min <= 1e-6

    def test_quasilinear_gradient_and_hvp(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Quasilinear kappa(u)*a(x) diffusion through the model: the
        genuinely state-dependent mixed curvature is exercised end to
        end (a plausibly-wrong Hessian is invisible to gradient-only
        tests)."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5, ny=5, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = QuasilinearDiffusion(
            basis=basis,
            diffusivity=NodalFieldDiffusion(
                basis, dofs=0.1 * np.ones(basis.ndofs())
            ),
            bkd=bkd,
            kappa=_kappa,
            kappa_deriv=_kappa_deriv,
            kappa_second_deriv=_kappa_second_deriv,
            boundary_conditions=[
                DirichletBC(basis, name, 0.0, bkd)
                for name in ("left", "right", "bottom", "top")
            ],
        )
        param_obj = create_quasilinear_diffusivity_parameterization(
            physics, _diffusivity_map(bkd, basis), bkd
        )
        functional = WeightedEndpointFunctional(
            _subdomain_average_weights(bkd, physics),
            param_obj.nparams(),
            bkd,
        )
        model = GalerkinTransientForwardModel(
            physics,
            param_obj,
            _gaussian_bump_ic(bkd, physics),
            _time_config("backward_euler"),
            bkd,
            functional=functional,
        )
        assert model.derivatives().hvp is not None
        _check_gradient_and_hvp(bkd, model)
