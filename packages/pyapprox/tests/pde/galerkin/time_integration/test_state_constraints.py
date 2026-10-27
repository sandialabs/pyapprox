"""Galerkin state constraints: the essential pin.

1D advection-diffusion, P1 on 8 cells, with the time-varying Dirichlet
value g(t) = (1 + x) t^2 on the left. The pin sets the essential entries
to g(t) and leaves the rest; it is a projection, and since it is affine
its tangent is exact for any perturbation, which the checks use instead
of finite differences.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.state_constraints import (
    EssentialPin,
)
from pyapprox.util.backends.numpy import NumpyBkd

_Np = NDArray[Any]
_NPARAMS = 2


def _ramp(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * t**2
    return value


def _ramp_dot(x: _Np, t: float) -> _Np:
    value: _Np = 2.0 * (1.0 + x[0]) * t
    return value


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = np.sin(np.pi * x[0]) * (1.0 + t)
    return value


def _system(bkd: NumpyBkd) -> GalerkinSystem[_Np]:
    mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    bc = DirichletBC(
        basis,
        "left",
        BoundarySignal(TimeDependent(_ramp), [TimeDependent(_ramp_dot)]),
        bkd,
    )
    physics = AdvectionDiffusionReaction(
        basis=basis, diffusivity=0.1, bkd=bkd, forcing=TimeDependent(_forcing)
    )
    return compose_galerkin_system(physics, [bc])


def _vectors(n: int, ncols: int, seed: int) -> _Np:
    vectors: _Np = np.random.default_rng(seed).normal(size=(n, ncols))
    return vectors


class TestEssentialPin:
    def test_sets_the_essential_entries_and_keeps_the_rest(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system = _system(numpy_bkd)
        pin = EssentialPin(system, _NPARAMS)
        assert isinstance(pin, StateConstraintProtocol)
        dofs = np.asarray(system.constraint_set().dofs())
        interior = np.setdiff1d(np.arange(system.nstates()), dofs)
        state = _vectors(system.nstates(), 1, 0)[:, 0]
        for time in (0.3, 0.8):
            pinned = pin.constrain(state, time)
            numpy_bkd.assert_allclose(
                pinned[dofs], system.constraint_set().values(time), rtol=0.0
            )
            numpy_bkd.assert_allclose(pinned[interior], state[interior], rtol=0.0)
            numpy_bkd.assert_allclose(pin.constrain(pinned, time), pinned, rtol=0.0)

    def test_tangent_is_exact_and_its_transpose_adjoint(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Affine, so Pi(z + d) - Pi(z) = Pi' d for any block d; and
        <Pi' x, y> = <x, Pi'^T y>."""
        system = _system(numpy_bkd)
        pin = EssentialPin(system, _NPARAMS)
        tangent = pin.derivatives().tangent
        assert tangent is not None
        n = system.nstates()
        state, direction = _vectors(n, 1, 1)[:, 0], _vectors(n, 3, 2)
        op = tangent(state, 0.5)
        for jj in range(3):
            difference = pin.constrain(state + direction[:, jj], 0.5) - pin.constrain(
                state, 0.5
            )
            numpy_bkd.assert_allclose(
                op.apply(direction)[:, jj], difference, rtol=1e-14, atol=1e-14
            )
        x, y = _vectors(n, 1, 3), _vectors(n, 1, 4)
        numpy_bkd.assert_allclose(
            np.sum(op.apply(x) * y), np.sum(x * op.apply_transpose(y)), rtol=1e-13
        )

    def test_parameter_and_second_derivatives_are_stated_zeros(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system = _system(numpy_bkd)
        derivs = EssentialPin(system, _NPARAMS).derivatives()
        state, adj = np.zeros(system.nstates()), np.ones(system.nstates())
        param, state_state = derivs.param, derivs.state_state
        state_param, param_param = derivs.state_param, derivs.param_param
        assert param is not None and state_state is not None
        assert state_param is not None and param_param is not None
        assert param(state, 0.0).properties().zero
        for second in (state_state, state_param, param_param):
            assert second(state, 0.0, adj).properties().zero
