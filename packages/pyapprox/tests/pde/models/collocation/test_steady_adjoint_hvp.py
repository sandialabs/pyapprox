"""Steady adjoint gradient + HVP for collocation diffusion with log-KLE.

Component-wise validation (ImplicitFunctionDerivativeChecker: state and
parameter Jacobians, all four state-equation HVP blocks, functional
derivatives, and the assembled adjoint gradient and HVP) of
CollocationStateEquationWithHVPAdapter with parameters the coefficients
of a lognormal KLE of the diffusion field. The collocation mixed tensor
is NOT symmetric in (residual, state) indices (non-symmetric
differentiation matrices + Dirichlet row replacement), so this exercises
the checker's transposed state_param FD reference on a genuinely
non-symmetric case.
"""

from typing import Tuple

import numpy as np
import pytest
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.scalar import ScalarAmplitude
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.models.collocation import create_collocation_model
from pyapprox.pde.models.collocation.steady import (
    CollocationStateEquationWithHVPAdapter,
)
from pyapprox.pde.parameterizations.composite import (
    CompositeParameterization,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.forcing import ForcingParameterization
from pyapprox.pde.parameterizations.reaction import ReactionParameterization
from pyapprox.util.backends.numpy import NumpyBkd

from tests._helpers.adjoint_checks import (
    NoHVPQuadraticFieldMap,
    NumpyArray,
)

_NPARAMS = 3


def _build_problem(
    bkd: NumpyBkd, npts: int = 20
) -> Tuple[
    AdvectionDiffusionReaction[NumpyArray],
    ChebyshevBasis1D[NumpyArray],
    NumpyArray,
]:
    """Diffusion problem on (-1, 1) with zero Dirichlet ends."""
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
    nodes = basis.nodes()

    def forcing(t: float) -> NumpyArray:
        return (np.pi**2) * bkd.sin(np.pi * nodes)

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=1.0, forcing=forcing
    )
    physics.set_boundary_conditions(
        [
            zero_dirichlet_bc(bkd, mesh_obj.boundary_indices(0)),
            zero_dirichlet_bc(bkd, mesh_obj.boundary_indices(1)),
        ]
    )
    return physics, basis, nodes


def _lognormal_kle_map(
    bkd: NumpyBkd, nodes: NumpyArray
) -> TransformedFieldMap[NumpyArray]:
    """D = exp(W theta): hand-built smooth modes on the nodes."""
    coords = bkd.to_numpy(nodes)
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


class TestCollocationSteadyLogKLEAdjointHVP:
    def test_all_derivative_components_match_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        physics, basis, nodes = _build_problem(bkd)
        field_map = _lognormal_kle_map(bkd, nodes)
        param_obj = create_diffusion_parameterization(
            physics, bkd, basis, field_map
        )
        model = create_collocation_model(
            physics, bkd, parameterization=param_obj
        )
        state_eq = CollocationStateEquationWithHVPAdapter(
            model, bkd, parameterization=param_obj
        )

        npts = physics.npts()
        weights = bkd.zeros((npts, 1))
        weights = bkd.copy(weights)
        weights[npts // 2] = 1.0
        functional = WeightedSumFunctional(weights, _NPARAMS, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)

        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        init_state = bkd.zeros((npts, 1))
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

    def test_composite_multi_term_hvp(self, numpy_bkd: NumpyBkd) -> None:
        """Composite diffusion+forcing+reaction stays exactly second
        order (linear maps declare hvp = 0), and the block-assembled
        HVPs pass every component check."""
        bkd = numpy_bkd
        physics, basis, nodes = _build_problem(bkd)
        npts = physics.npts()
        dp = create_diffusion_parameterization(
            physics, bkd, basis, _lognormal_kle_map(bkd, nodes)
        )
        fp = ForcingParameterization(
            physics, ScalarAmplitude(bkd, bkd.sin(np.pi * nodes)), bkd
        )
        rp = ReactionParameterization(
            physics, BasisExpansion(bkd, -0.5, [bkd.ones((npts,))]), bkd
        )
        comp = CompositeParameterization([dp, fp, rp], bkd)
        nparams = comp.nparams()
        assert nparams == _NPARAMS + 2

        model = create_collocation_model(
            physics, bkd, parameterization=comp
        )
        state_eq = CollocationStateEquationWithHVPAdapter(
            model, bkd, parameterization=comp
        )
        weights = bkd.zeros((npts, 1))
        weights = bkd.copy(weights)
        weights[npts // 2] = 1.0
        functional = WeightedSumFunctional(weights, nparams, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)

        param = bkd.asarray(
            np.array([[0.4], [-0.3], [0.2], [1.0], [0.3]])
        )
        init_state = bkd.zeros((npts, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        checker.check_derivatives(init_state, param, tols)

        vvec = bkd.asarray(
            np.array([[0.5], [0.7], [-0.6], [0.4], [-0.8]])
        )
        uvec = bkd.asarray(
            np.array([[-0.2], [0.9], [0.3], [-0.5], [0.6]])
        )
        h_v = adjoint_op.hvp(init_state, param, vvec)
        h_u = adjoint_op.hvp(init_state, param, uvec)
        bkd.assert_allclose(
            bkd.sum(h_v * uvec), bkd.sum(h_u * vvec), rtol=1e-12
        )

    def test_first_order_map_raises(self, numpy_bkd: NumpyBkd) -> None:
        """A map with undeclared curvature yields a first-order bundle:
        HVP tier construction is rejected. (Linear maps declare
        hvp = 0 exactly and stay second order.)"""
        bkd = numpy_bkd
        physics, basis, nodes = _build_problem(bkd, npts=10)
        modes = bkd.stack([bkd.ones((10,)), nodes], axis=1)
        no_hvp_map = NoHVPQuadraticFieldMap(
            bkd, bkd.full((10,), 2.0), modes
        )
        param_obj = create_diffusion_parameterization(
            physics, bkd, basis, no_hvp_map
        )
        model = create_collocation_model(
            physics, bkd, parameterization=param_obj
        )
        with pytest.raises(TypeError, match="second-order"):
            CollocationStateEquationWithHVPAdapter(
                model, bkd, parameterization=param_obj
            )
