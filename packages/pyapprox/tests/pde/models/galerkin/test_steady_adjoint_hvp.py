r"""Steady adjoint gradient and HVP for Galerkin ADR with a log-KLE
diffusivity, on the steady system path.

Setup (``tests/_helpers/galerkin_steady.steady_adr``): :math:`-(\kappa
u')' (+ u^2) = 1` on [0, 1], u = 0 at both ends, P1 on 10 cells,
:math:`\kappa = \exp(W p)` with three modes, at p = (0.4, -0.3, 0.2). The
exp map makes every parameter-facing second derivative nonzero; the
:math:`u^2` reaction also makes the state-state one nonzero.

- The residual's state and parameter Jacobians and its four
  adjoint-weighted second derivatives pass the bundle checker (finite
  differences and dot tests) at state linspace(0.1, 0.9) with a random
  adjoint (seed 5).
- The solved state at the first free node has a gradient and HVP that
  pass central finite differences below 1e-6, and an HVP satisfying
  :math:`\langle Hv, u\rangle = \langle Hu, v\rangle` to 1e-12, which
  breaks for contraction or orientation bugs no finite difference can
  resolve.
- A diffusivity map with curvature but no declared HVP gives a residual
  bundle without second derivatives (absent, not a stated zero).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NoHVPQuadraticFieldMap
from tests._helpers.galerkin_steady import ADR_NPARAMS, ADR_PARAM, steady_adr
from tests._helpers.galerkin_steady_checks import (
    first_free_row,
    observed_steady_function,
    residual_bundle_report,
)

_Np = NDArray[Any]


def _unit_forcing(coords: _Np) -> _Np:
    return np.ones(coords.shape[1])


class TestSteadyADRLogKLEAdjointHVP:
    @pytest.mark.parametrize("nonlinear_reaction", [False, True])
    def test_residual_derivatives_pass_the_bundle_checker(
        self, numpy_bkd: NumpyBkd, nonlinear_reaction: bool
    ) -> None:
        """Every derivative the residual declares, the second-order
        blocks included, passes finite differences and dot tests."""
        bkd = numpy_bkd
        system, parameterization, _ = steady_adr(bkd, nonlinear_reaction)
        nstates = system.nstates()
        report = residual_bundle_report(
            system,
            parameterization,
            bkd.asarray(np.linspace(0.1, 0.9, nstates)),
            bkd.asarray(np.random.default_rng(5).normal(size=nstates)),
            bkd.asarray(ADR_PARAM[:, 0]),
        )
        assert report.failures() == []

    @pytest.mark.parametrize("nonlinear_reaction", [False, True])
    def test_solved_gradient_and_hvp_match_fd(
        self, numpy_bkd: NumpyBkd, nonlinear_reaction: bool
    ) -> None:
        """The solved state's gradient and HVP pass central finite
        differences, and the HVP is symmetric."""
        bkd = numpy_bkd
        system, parameterization, _ = steady_adr(bkd, nonlinear_reaction)
        function = observed_steady_function(
            system, parameterization, [first_free_row(system, bkd)], bkd
        )
        param = bkd.asarray(ADR_PARAM)
        errors = DerivativeChecker(function).check_derivatives(
            param, relative=True, central=True
        )
        assert len(errors) == 2
        assert max(bkd.to_float(bkd.min(e)) for e in errors) < 1e-6
        hvp = function.derivatives().hvp
        assert hvp is not None
        vvec = bkd.asarray(np.array([[0.5], [0.7], [-0.6]]))
        uvec = bkd.asarray(np.array([[-0.2], [0.9], [0.3]]))
        h_v, h_u = hvp(param, vvec), hvp(param, uvec)
        bkd.assert_allclose(
            bkd.sum(h_v * uvec)[None], bkd.sum(h_u * vvec)[None], rtol=1e-12
        )

    def test_first_order_bundle_withholds_second_derivatives(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A diffusivity map with curvature but no declared HVP gives a
        residual with parameter Jacobians and no second derivatives."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
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
            [np.sin((k + 1) * np.pi * coords) for k in range(ADR_NPARAMS)],
            axis=1,
        )
        # Curvature without a declared hvp: the honest first-order case
        # (linear maps declare hvp = 0 exactly and stay second order).
        no_hvp_map = NoHVPQuadraticFieldMap(
            bkd,
            bkd.asarray(np.ones(coords.shape[0])),
            bkd.asarray(modes),
        )
        parameterization = AdvectionDiffusionParameterization(
            physics, diffusivity_map=no_hvp_map, bkd=bkd
        )
        residual = galerkin_steady_residual(
            skfem_discretization(system), ParameterState(parameterization)
        )
        bundle = residual.derivatives()
        assert bundle.param is not None
        assert bundle.state_state is None and bundle.param_param is None
