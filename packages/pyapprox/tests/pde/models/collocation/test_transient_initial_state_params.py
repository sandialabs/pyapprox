"""Gradients when a parameter sets the initial state (collocation).

The forward model overwrites the initial state only on essential rows, so on
a natural (Robin) boundary node ``y_0 = u_0(p)`` depends on ``p``, and under
Crank--Nicolson ``F(y_0)`` carries that dependence into the first step's
interior rows. The gradient therefore needs ``dy_0/dp`` on the Robin rows.

The check needs no derivation: the adjoint gradient is compared with finite
differences of the model rebuilt with ``init_state = u_0(p)`` for every
perturbation.
"""

import dataclasses
import math
from typing import Any, Tuple

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import gradient_robin_bc
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.models.collocation.transient import TransientForwardModel
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.backends.protocols import Array, Backend


class _InitialStateParameterization:
    """A diffusion parameterization whose parameters also set the initial
    state, ``u_0(p) = u_base + p_0 * phi``, so ``dy_0/dp = [phi, 0]``."""

    def __init__(self, inner: Any, ic_jacobian: Array) -> None:
        self._inner = inner
        self._ic_jacobian = ic_jacobian

    def nparams(self) -> int:
        return int(self._inner.nparams())

    def physics(self) -> object:
        return self._inner.physics()

    def targets(self) -> Tuple[object, ...]:
        return tuple(self._inner.targets())

    def owned_coefficients(self) -> Tuple[str, ...]:
        return tuple(self._inner.owned_coefficients())

    def apply(self, params_1d: Array) -> None:
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return dataclasses.replace(
            self._inner.param_derivatives(),
            initial_param_jacobian=self._initial_param_jacobian,
        )

    def _initial_param_jacobian(self, params_1d: Array) -> Array:
        return self._ic_jacobian


def _robin_problem(bkd: Backend[Array], method: str, npts: int = 15) -> Any:
    """Transient diffusion with Robin conditions on both ends (E is empty)."""
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()
    physics = AdvectionDiffusionReaction(basis, bkd, diffusion=2.0)
    D = basis.derivative_matrix()
    physics.set_boundary_conditions(
        [
            gradient_robin_bc(
                bkd, mesh.boundary_indices(0), mesh.boundary_normals(0),
                [D], 1.0, 1.0, 0.0,
            ),
            gradient_robin_bc(
                bkd, mesh.boundary_indices(1), mesh.boundary_normals(1),
                [D], 2.0, 1.0, 0.0,
            ),
        ]
    )
    field_map = BasisExpansion(bkd, 2.0, [bkd.ones((npts,)), nodes])
    inner = create_diffusion_parameterization(physics, bkd, field_map)

    u_base = bkd.sin(math.pi * nodes)
    phi = 2.0 + nodes  # nonzero at both Robin nodes x = -1 and x = 1
    ic_jacobian = bkd.stack([phi, bkd.zeros((npts,))], axis=1)
    param = _InitialStateParameterization(inner, ic_jacobian)

    def initial_state(params_1d: Array) -> Array:
        return u_base + params_1d[0] * phi

    time_config = TimeIntegrationConfig(
        method=method,
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-12,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )
    return physics, param, initial_state, time_config


def _model(bkd: Backend[Array], method: str, params_1d: Array) -> Any:
    physics, param, initial_state, time_config = _robin_problem(bkd, method)
    nstates = physics.nstates()
    functional = EndpointFunctional(nstates // 2, nstates, param.nparams(), bkd)
    return TransientForwardModel(
        physics,
        bkd,
        initial_state(params_1d),
        time_config,
        functional=functional,
        parameterization=param,
    )


class TestInitialStateParameterGradient:
    """Adjoint gradient with a parameter-dependent initial state."""

    @pytest.mark.parametrize(
        "method", ["crank_nicolson", "backward_euler", "implicit_midpoint"]
    )
    def test_adjoint_gradient_matches_finite_differences(
        self, bkd: Backend[Array], method: str
    ) -> None:
        base = bkd.array([0.3, 0.1])

        def fun(samples: Array) -> Array:
            cols = [
                _model(bkd, method, samples[:, ii])(samples[:, ii : ii + 1])
                for ii in range(samples.shape[1])
            ]
            return bkd.hstack(cols)

        def jacobian(sample: Array) -> Array:
            model = _model(bkd, method, sample[:, 0])
            return model.derivatives().jacobian(sample)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=2, fun=fun, jacobian=jacobian, bkd=bkd
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            base[:, None], direction=None, relative=True
        )[0]
        assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 1e-5
