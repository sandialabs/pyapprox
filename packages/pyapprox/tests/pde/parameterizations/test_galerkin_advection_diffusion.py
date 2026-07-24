"""Facade validation for AdvectionDiffusionParameterization.

Single-term parity with the hand-rolled diffusivity oracle
(supersession evidence), the all-four-field composite through the full
14-check suite, and the eager construction-time raises.
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
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldLinearReaction,
    NodalFieldVelocity,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.models.galerkin.steady import (
    GalerkinStateEquationWithHVPAdapter,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.galerkin_diffusivity import (
    AffineDiffusivityFieldParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd

from tests._helpers.adjoint_checks import NumpyArray


def _exp_kle_map(
    bkd: NumpyBkd, coords: np.ndarray, nmodes: int, amplitude: float
) -> TransformedFieldMap[NumpyArray]:
    modes = np.stack(
        [
            amplitude * np.sin((k + 1) * np.pi * coords) / (k + 1)
            for k in range(nmodes)
        ],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _build_full_physics(
    bkd: NumpyBkd,
) -> Tuple[
    AdvectionDiffusionReaction[NumpyArray],
    LagrangeBasis[NumpyArray],
    VectorLagrangeBasis[NumpyArray],
]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    vel_basis = VectorLagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        velocity=NodalFieldVelocity(
            vel_basis, np.zeros(vel_basis.ndofs())
        ),
        reaction=NodalFieldLinearReaction(basis),
        forcing=NodalFieldForcing(basis),
        boundary_conditions=[
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
        ],
    )
    return physics, basis, vel_basis


class TestAdvectionDiffusionParameterization:
    def test_single_term_parity_with_oracle(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """diffusivity_map-only facade == the hand-rolled oracle."""
        bkd = numpy_bkd
        physics, basis, _ = _build_full_physics(bkd)
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        field_map = _exp_kle_map(bkd, coords, 3, 0.4)
        facade = AdvectionDiffusionParameterization(
            physics, diffusivity_map=field_map, bkd=bkd
        )
        oracle = AffineDiffusivityFieldParameterization(
            physics, field_map, bkd
        )
        assert facade.nparams() == oracle.nparams() == 3

        f_derivs = facade.param_derivatives()
        o_derivs = oracle.param_derivatives()
        assert f_derivs.param_jacobian is not None
        assert o_derivs.param_jacobian is not None
        assert f_derivs.param_param_hvp is not None
        assert o_derivs.param_param_hvp is not None
        assert f_derivs.state_param_hvp is not None
        assert o_derivs.state_param_hvp is not None
        assert f_derivs.param_state_hvp is not None
        assert o_derivs.param_state_hvp is not None

        nstates = physics.nstates()
        rng = np.random.default_rng(17)
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        wvec = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        params = bkd.asarray(np.array([0.4, -0.3, 0.2]))
        vvec = bkd.asarray(np.array([0.5, 0.7, -0.6]))
        facade.apply(params)
        bkd.assert_allclose(
            f_derivs.param_jacobian(state, 0.0, params),
            o_derivs.param_jacobian(state, 0.0, params),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            f_derivs.param_param_hvp(state, 0.0, params, adj, vvec),
            o_derivs.param_param_hvp(state, 0.0, params, adj, vvec),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            f_derivs.state_param_hvp(state, 0.0, params, adj, vvec),
            o_derivs.state_param_hvp(state, 0.0, params, adj, vvec),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            f_derivs.param_state_hvp(state, 0.0, params, adj, wvec),
            o_derivs.param_state_hvp(state, 0.0, params, adj, wvec),
            rtol=1e-12,
        )

    def test_all_fields_pass_component_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """diffusivity+forcing+reaction+velocity facade (contiguous
        parameter slices, composite block assembly) passes the full
        14-check suite."""
        bkd = numpy_bkd
        physics, basis, vel_basis = _build_full_physics(bkd)
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        vel_coords = np.linspace(0.0, 1.0, vel_basis.ndofs())
        facade = AdvectionDiffusionParameterization(
            physics,
            diffusivity_map=_exp_kle_map(bkd, coords, 3, 0.4),
            forcing_map=_exp_kle_map(bkd, coords, 2, 0.6),
            reaction_map=_exp_kle_map(bkd, coords, 2, 0.3),
            velocity_map=_exp_kle_map(bkd, vel_coords, 2, 0.4),
            bkd=bkd,
        )
        nparams = facade.nparams()
        assert nparams == 9

        state_eq = GalerkinStateEquationWithHVPAdapter(
            physics, facade, bkd
        )
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
        functional = WeightedSumFunctional(weights, nparams, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        rng = np.random.default_rng(19)
        param = bkd.asarray(rng.normal(0.0, 0.3, (nparams, 1)))
        init_state = bkd.zeros((nstates, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        # Noise-limited checks (established calibration; the
        # multi-field residual's block dynamic range nudges the
        # param-jacobian FD floor just past 1e-6 as well).
        tols[1] = 5e-6
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        tols[13] = 5e-6
        checker.check_derivatives(init_state, param, tols)

    def test_map_for_non_nodal_coefficient_raises(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Eager raise when a map targets a coefficient the physics
        does not hold in the differentiable representation."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            bkd=bkd,
            forcing=lambda x: np.ones(x.shape[1]),
            boundary_conditions=[DirichletBC(basis, "left", 0.0, bkd)],
        )
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        field_map = _exp_kle_map(bkd, coords, 2, 0.4)
        with pytest.raises(TypeError, match="NodalFieldDiffusion"):
            AdvectionDiffusionParameterization(
                physics, diffusivity_map=field_map, bkd=bkd
            )
        with pytest.raises(TypeError, match="NodalFieldForcing"):
            AdvectionDiffusionParameterization(
                physics, forcing_map=field_map, bkd=bkd
            )

    def test_no_maps_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        physics, _, _ = _build_full_physics(bkd)
        with pytest.raises(TypeError, match="at least one"):
            AdvectionDiffusionParameterization(physics, bkd=bkd)
