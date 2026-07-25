"""Engine validation for the quasilinear-diffusion parameterization.

The exemplar for the engine's CALLABLE mixed second-derivative slots:
the field-carrying term a(x)*kappa(u)*grad(u) is nonlinear in the
state, so state_param_hvp/param_state_hvp route through the physics's
mixed assemblies rather than the linearity identities. The steady
14-check component suite exercises every block against FD.
"""

import pickle

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

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
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction,
    QuasilinearDiffusion,
)
from pyapprox.pde.models.galerkin.steady import (
    GalerkinStateEquationWithHVPAdapter,
)
from pyapprox.pde.parameterizations.galerkin_quasilinear_diffusion import (
    create_quasilinear_diffusivity_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd

_NPARAMS = 3


def _kappa(u):
    return 1.0 + u**2


def _kappa_deriv(u):
    return 2.0 * u


def _kappa_second_deriv(u):
    return np.full_like(u, 2.0)


def _forcing(x):
    return np.ones(x.shape[1])


def _build_physics_and_map(bkd):
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = QuasilinearDiffusion(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        kappa=_kappa,
        kappa_deriv=_kappa_deriv,
        kappa_second_deriv=_kappa_second_deriv,
        forcing=_forcing,
        boundary_conditions=[
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
    field_map = TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)
    return physics, field_map


class TestQuasilinearDiffusivityParameterization:
    def test_all_derivative_components_match_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Steady 14-check component suite: the callable mixed slots
        (nonzero because kappa depends on u) and the kappa''-driven
        state_state_hvp are all FD-validated."""
        bkd = numpy_bkd
        physics, field_map = _build_physics_and_map(bkd)
        param_obj = create_quasilinear_diffusivity_parameterization(
            physics, field_map, bkd
        )
        state_eq = GalerkinStateEquationWithHVPAdapter(
            physics, param_obj, bkd
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
        functional = WeightedSumFunctional(weights, _NPARAMS, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        init_state = bkd.zeros((nstates, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        # The four state-equation HVP blocks (5-8) are noise-limited
        # here (measured ratios 1.0e-6..1.8e-6, V-shaped sweeps); each
        # is tightly FD-validated at the physics tier and cross-checked
        # by the shared-tensor and symmetry identities.
        tols[5] = 5e-6
        tols[6] = 5e-6
        tols[7] = 5e-6
        tols[8] = 5e-6
        checker.check_derivatives(init_state, param, tols)

        # FD-noise-immune symmetry identity <Hu, v> = <Hv, u>.
        vvec = bkd.asarray(np.array([[0.5], [0.7], [-0.6]]))
        uvec = bkd.asarray(np.array([[-0.2], [0.9], [0.3]]))
        h_v = adjoint_op.hvp(init_state, param, vvec)
        h_u = adjoint_op.hvp(init_state, param, uvec)
        bkd.assert_allclose(
            bkd.sum(h_v * uvec), bkd.sum(h_u * vvec), rtol=1e-12
        )

    def test_wrong_physics_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        adr = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=_forcing,
            boundary_conditions=[DirichletBC(basis, "left", 0.0, bkd)],
        )
        _, field_map = _build_physics_and_map(bkd)
        with pytest.raises(TypeError, match="QuasilinearDiffusion"):
            create_quasilinear_diffusivity_parameterization(
                adr,  # type: ignore[arg-type]
                field_map,
                bkd,
            )

    def test_pickle_round_trip(self, numpy_bkd: NumpyBkd) -> None:
        """The term (with its physics and field map) survives pickling
        and the clone produces identical derivatives."""
        bkd = numpy_bkd
        physics, field_map = _build_physics_and_map(bkd)
        param_obj = create_quasilinear_diffusivity_parameterization(
            physics, field_map, bkd
        )
        clone = pickle.loads(pickle.dumps(param_obj))
        assert clone.nparams() == param_obj.nparams()

        rng = np.random.default_rng(31)
        nstates = physics.nstates()
        state = bkd.asarray(rng.normal(0.0, 0.5, nstates))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        params = bkd.asarray(np.array([0.4, -0.3, 0.2]))
        vvec = bkd.asarray(np.array([0.5, 0.7, -0.6]))
        param_obj.apply(params)
        clone.apply(params)
        bkd.assert_allclose(
            param_obj.param_jacobian(state, 0.0, params),
            clone.param_jacobian(state, 0.0, params),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            param_obj.state_param_hvp(state, 0.0, params, adj, vvec),
            clone.state_param_hvp(state, 0.0, params, adj, vvec),
            rtol=1e-14,
        )
