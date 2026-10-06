"""Galerkin probe and outflux observations, alone and through a model.

- Probes reproduce fields in the element space exactly (linear on P1,
  quadratic on P2), at interior, edge and vertex points.
- The outflux is ``u * (beta . n)`` at the given DOFs, in their order.
- Through ``GalerkinTransientForwardModel`` with a KLE diffusivity, the
  tangent-linear vector Jacobian passes DerivativeChecker and the
  adjoint method reproduces it. Probes on the Dirichlet boundary are
  included: Dirichlet values do not depend on the parameters, so both
  methods must agree there too.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.operator.qoi_jacobian import adjoint_jacobian
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldVelocity,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.kle_factory import (
    create_spde_lognormal_kle_field_map,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.observations import (
    nodal_outflux_functional,
    probe_observation_functional,
)
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.models.galerkin.transient import (
    GalerkinTransientForwardModel,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

# Interior, on an edge, and on a vertex (incl. the domain corner).
_POINTS = np.array(
    [[0.31, 0.5, 0.25, 1.0, 0.0], [0.47, 0.13, 0.5, 1.0, 0.5]]
)


def _field(coords: np.ndarray, degree: int) -> np.ndarray:
    x, y = coords
    if degree == 1:
        return 0.3 + 1.2 * x - 0.7 * y
    return 0.3 + 1.2 * x - 0.7 * y + 0.9 * x * y - 0.4 * x**2 + 0.6 * y**2


class TestProbesAndOutflux:
    @pytest.mark.parametrize("degree", [1, 2])
    def test_probes_exact_in_element_space(
        self, bkd: Backend[Array], degree: int
    ) -> None:
        mesh = StructuredMesh2D(
            4, 4, [[0.0, 1.0], [0.0, 1.0]], bkd, element_type="tri"
        )
        basis = LagrangeBasis(mesh, degree=degree)
        coeffs = bkd.asarray(
            _field(bkd.to_numpy(basis.dof_coordinates()), degree)
        )
        times = [0, 2]
        sol = bkd.stack([coeffs, 2.0 * coeffs, -coeffs], axis=1)
        func = probe_observation_functional(
            basis, bkd.asarray(_POINTS), times, nparams=1
        )
        exact = _field(_POINTS, degree)
        expected = np.concatenate([exact, -exact])[:, None]
        bkd.assert_allclose(
            func(sol, bkd.zeros((1, 1))), bkd.asarray(expected),
            rtol=1e-12, atol=1e-13,
        )

    def test_outflux_values(self, bkd: Backend[Array]) -> None:
        dofs = bkd.asarray(np.array([7, 2, 5]), dtype=bkd.int64_dtype())
        normal_velocity = bkd.asarray(np.array([1.0, 0.5, -2.0]))
        sol = bkd.asarray(np.arange(30.0).reshape(10, 3))
        func = nodal_outflux_functional(
            dofs, normal_velocity, [1, 2], 10, 2, bkd
        )
        sol_np = bkd.to_numpy(sol)
        vn = np.array([1.0, 0.5, -2.0])
        expected = np.concatenate(
            [sol_np[[7, 2, 5], 1] * vn, sol_np[[7, 2, 5], 2] * vn]
        )
        bkd.assert_allclose(
            func(sol, bkd.zeros((2, 1))), bkd.asarray(expected[:, None]),
            rtol=1e-14,
        )

    def test_outflux_shape_mismatch_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match="nbdofs"):
            nodal_outflux_functional(
                bkd.asarray(np.array([1, 2]), dtype=bkd.int64_dtype()),
                bkd.ones((3,)),
                [1],
                10,
                2,
                bkd,
            )


def _models(bkd: NumpyBkd, which: str) -> tuple[Any, Any]:
    """Transient ADR on the unit square: KLE log-diffusivity, velocity
    (1, 0), Dirichlet u = 0 on the left only (outflow elsewhere), a
    Gaussian bump initial state, backward Euler with dt = 0.02 to
    T = 0.1. Returns the tangent-linear and the adjoint model."""
    mesh = StructuredMesh2D(
        6, 6, [[0.0, 1.0], [0.0, 1.0]], bkd, element_type="tri"
    )
    basis = LagrangeBasis(mesh, degree=1)
    vel_basis = VectorLagrangeBasis(mesh, degree=1)
    vel_dofs = np.zeros(vel_basis.ndofs())
    vel_dofs[0::2] = 1.0
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(
            basis, dofs=0.1 * np.ones(basis.ndofs())
        ),
        bkd=bkd,
        velocity=NodalFieldVelocity(vel_basis, vel_dofs),
    )
    system = compose_galerkin_system(
        physics, [DirichletBC(basis, "left", 0.0, bkd)]
    )
    diffusivity_map = create_spde_lognormal_kle_field_map(
        basis,
        bkd.asarray(np.log(0.1) * np.ones(basis.ndofs())),
        bkd,
        n_modes=4,
        gamma=1.0,
        delta=4.0,
        sigma=0.3,
    )
    parameterization = AdvectionDiffusionParameterization(
        physics, bkd=bkd, diffusivity_map=diffusivity_map
    )
    nparams = parameterization.nparams()
    times = [2, 5]
    if which == "probes":
        # (0, 0.5) and (0, 0.2) lie on the Dirichlet boundary.
        points = bkd.asarray(
            np.array([[0.0, 0.0, 0.42, 0.8], [0.5, 0.2, 0.55, 0.3]])
        )
        functional = probe_observation_functional(
            basis, points, times, nparams
        )
    else:
        coords = bkd.to_numpy(basis.dof_coordinates())
        right = bkd.to_numpy(basis.get_dofs("right"))
        right = right[np.argsort(coords[1, right])]
        functional = nodal_outflux_functional(
            bkd.asarray(right, dtype=bkd.int64_dtype()),
            bkd.ones((right.shape[0],)),
            times,
            basis.ndofs(),
            nparams,
            bkd,
        )
    coords = bkd.to_numpy(basis.dof_coordinates())
    init_state = bkd.asarray(
        np.exp(-50.0 * ((coords[0] - 0.4) ** 2 + (coords[1] - 0.5) ** 2))
    )
    config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-12,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )
    tangent = GalerkinTransientForwardModel(
        system, parameterization, init_state, config, bkd,
        functional=functional,
    )
    adjoint = GalerkinTransientForwardModel(
        system, parameterization, init_state, config, bkd,
        functional=functional, jacobian_method=adjoint_jacobian,
    )
    return tangent, adjoint


class TestObservationJacobians:
    @pytest.mark.parametrize("which", ["probes", "outflux"])
    def test_tangent_linear_matches_fd(
        self, numpy_bkd: NumpyBkd, which: str
    ) -> None:
        bkd = numpy_bkd
        tangent, _ = _models(bkd, which)
        rng = np.random.default_rng(7)
        sample = bkd.asarray(0.4 * rng.normal(size=(tangent.nvars(), 1)))
        checker = DerivativeChecker(tangent)
        errors = checker.check_derivatives(
            sample,
            relative=True,
            weights=bkd.asarray(rng.normal(size=(tangent.nqoi(), 1))),
        )
        # Transient FD floors near 1e-7 (see test_transient_adjoint.py).
        assert float(bkd.to_numpy(checker.error_ratio(errors[0]))) <= 2e-5

    @pytest.mark.parametrize("which", ["probes", "outflux"])
    def test_adjoint_matches_tangent_linear(
        self, numpy_bkd: NumpyBkd, which: str
    ) -> None:
        bkd = numpy_bkd
        tangent, adjoint = _models(bkd, which)
        sample = bkd.asarray(
            0.4 * np.random.default_rng(7).normal(size=(tangent.nvars(), 1))
        )
        jac_tangent = tangent.derivatives().jacobian(sample)
        assert jac_tangent.shape == (tangent.nqoi(), tangent.nvars())
        bkd.assert_allclose(
            adjoint.derivatives().jacobian(sample),
            jac_tangent,
            rtol=1e-9,
            atol=1e-12,
        )
