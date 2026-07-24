"""Steady adjoint gradient + HVP for galerkin ADR with log-KLE diffusivity.

Component-wise validation (ImplicitFunctionDerivativeChecker: state and
parameter Jacobians, all four state-equation HVP blocks, functional
derivatives, and the assembled adjoint gradient and HVP) of
GalerkinStateEquationWithHVPAdapter with parameters the coefficients of a
lognormal KLE of the diffusivity field. The exp map makes every
parameter-facing HVP nonzero; the cubic-reaction variant additionally
makes state_state_hvp nonzero.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Tuple

import numpy as np
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.models.galerkin.steady import (
    GalerkinStateEquationWithHVPAdapter,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd

from tests._helpers.adjoint_checks import (
    NoHVPQuadraticFieldMap,
    NumpyArray,
)

_NPARAMS = 3


def _lognormal_kle_map(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]
) -> TransformedFieldMap[NumpyArray]:
    """kappa = exp(W theta): hand-built smooth modes on the DOF nodes."""
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
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _build_state_equation(
    bkd: NumpyBkd, nonlinear_reaction: bool
) -> Tuple[
    GalerkinStateEquationWithHVPAdapter[NumpyArray],
    AdvectionDiffusionReaction[NumpyArray],
]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    reaction = (
        CallableReaction(
            lambda x, u: u**2,
            lambda x, u: 2.0 * u,
            lambda x, u: np.full_like(u, 2.0),
        )
        if nonlinear_reaction
        else None
    )
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        reaction=reaction,
        forcing=lambda x: np.ones(x.shape[1]),
        boundary_conditions=[
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
        ],
    )
    param_obj = AdvectionDiffusionParameterization(
        physics, diffusivity_map=_lognormal_kle_map(bkd, basis), bkd=bkd
    )
    state_eq = GalerkinStateEquationWithHVPAdapter(physics, param_obj, bkd)
    return state_eq, physics


class TestSteadyADRLogKLEAdjointHVP:
    @pytest.mark.parametrize("nonlinear_reaction", [False, True])
    def test_all_derivative_components_match_fd(
        self, numpy_bkd: NumpyBkd, nonlinear_reaction: bool
    ) -> None:
        bkd = numpy_bkd
        state_eq, physics = _build_state_equation(bkd, nonlinear_reaction)
        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        weights = bkd.zeros((nstates, 1))
        weights = bkd.copy(weights)
        weights[state_idx] = 1.0
        functional = WeightedSumFunctional(weights, _NPARAMS, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)

        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        init_state = bkd.zeros((nstates, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        # Noise-limited checks (one-sided FD of PDE-scale quantities
        # floors near 1e-6): assembled gradient (4), state-equation
        # param_param (5) and state_param (8). A genuine convention bug
        # plateaus at ~1, five orders above these bounds.
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        checker.check_derivatives(init_state, param, tols)

        # FD-noise-immune symmetry identity <Hu, v> = <Hv, u>:
        # breaks for contraction/orientation bugs.
        vvec = bkd.asarray(np.array([[0.5], [0.7], [-0.6]]))
        uvec = bkd.asarray(np.array([[-0.2], [0.9], [0.3]]))
        h_v = adjoint_op.hvp(init_state, param, vvec)
        h_u = adjoint_op.hvp(init_state, param, uvec)
        bkd.assert_allclose(
            bkd.sum(h_v * uvec), bkd.sum(h_u * vvec), rtol=1e-12
        )

    def test_first_order_bundle_raises(self, numpy_bkd: NumpyBkd) -> None:
        """A parameterization without HVPs is rejected at construction."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=lambda x: np.ones(x.shape[1]),
            boundary_conditions=[
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", 0.0, bkd),
            ],
        )
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        modes = np.stack(
            [np.sin((k + 1) * np.pi * coords) for k in range(_NPARAMS)],
            axis=1,
        )
        # Curvature without a declared hvp: the honest first-order case
        # (linear maps declare hvp = 0 exactly and stay second order).
        no_hvp_map = NoHVPQuadraticFieldMap(
            bkd,
            bkd.asarray(np.ones(coords.shape[0])),
            bkd.asarray(modes),
        )
        param_obj = AdvectionDiffusionParameterization(
            physics, diffusivity_map=no_hvp_map, bkd=bkd
        )
        with pytest.raises(TypeError, match="second-order"):
            GalerkinStateEquationWithHVPAdapter(physics, param_obj, bkd)
