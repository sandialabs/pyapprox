"""Facade validation for CollocationElasticityParameterization.

Second-order validation without an oracle: the 14-check FD suite on
the pressurized-cylinder problem (a genuinely curved polar domain
with traction BCs), exact HVP identities, pickle round-trip, and the
affine-equivalence identity (QoI and gradient invariant under an
affine domain rescale with matched forcing).
"""

import math
import pickle

import numpy as np
import pytest
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.boundary.robin import traction_neumann_bc
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import (
    AffineTransform2D,
    PolarTransform,
)
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.models.collocation import create_collocation_model
from pyapprox.pde.models.collocation.steady import (
    CollocationStateEquationWithHVPAdapter,
    SteadyForwardModel,
)
from pyapprox.pde.parameterizations.collocation_elasticity import (
    CollocationElasticityParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd

_NMODES = 3
_NU = 0.3
_E_MEAN = 10.0


def _exp_kle_map(bkd, coords_np, nmodes, amplitude, log_mean=0.0):
    scale = np.max(np.abs(coords_np)) or 1.0
    modes = np.stack(
        [
            amplitude
            * np.sin((k + 1) * math.pi * coords_np / scale)
            / (k + 1)
            for k in range(nmodes)
        ],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd,
        bkd.asarray(np.full(coords_np.shape[0], log_mean)),
        bkd.asarray(modes),
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _build_pressurized_cylinder(bkd, npts_1d=6):
    """The pressurized-cylinder zoo problem: quarter annulus (polar),
    stress-free outer arc, pressure at the inner arc, symmetry
    Dirichlet+traction edges — traction rows use the mean-material
    constants, so no coefficient-dependent BC rows (the HVP tier
    applies)."""
    transform = PolarTransform((1.0, 2.0), (0.0, math.pi / 2.0), bkd)
    mesh = TransformedMesh2D(npts_1d, npts_1d, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    npts = basis.npts()

    dmu = 1.0 / (2.0 * (1.0 + _NU))
    dlam = _NU / ((1.0 + _NU) * (1.0 - 2.0 * _NU))
    mu_init = _E_MEAN * dmu
    lamda_init = _E_MEAN * dlam
    physics = LinearElasticityPhysics(
        basis, bkd, lamda=lamda_init, mu=mu_init
    )

    d_matrices = [
        basis.derivative_matrix(1, 0),
        basis.derivative_matrix(1, 1),
    ]
    bcs = []
    outer_idx = mesh.boundary_indices(1)
    outer_normals = mesh.boundary_normals(1)
    for comp in (0, 1):
        bcs.append(
            traction_neumann_bc(
                bkd, outer_idx, outer_normals, d_matrices,
                lamda_init, mu_init, npts, comp, values=0.0,
            )
        )
    inner_idx = mesh.boundary_indices(0)
    inner_normals = mesh.boundary_normals(0)
    for comp in (0, 1):
        pressure_vals = -0.5 * inner_normals[:, comp]
        bcs.append(
            traction_neumann_bc(
                bkd, inner_idx, inner_normals, d_matrices,
                lamda_init, mu_init, npts, comp, values=pressure_vals,
            )
        )
    bottom_idx = mesh.boundary_indices(2)
    top_idx = mesh.boundary_indices(3)
    bcs.append(zero_dirichlet_bc(bkd, bottom_idx + npts))
    bcs.append(zero_dirichlet_bc(bkd, top_idx))
    physics.set_boundary_conditions(bcs)

    pts = bkd.to_numpy(mesh.points())
    e_map = _exp_kle_map(
        bkd, pts[0] + pts[1], _NMODES, 0.3, log_mean=math.log(_E_MEAN)
    )
    facade = CollocationElasticityParameterization(
        physics,
        youngs_modulus_map=e_map,
        poisson_ratio=_NU,
        bkd=bkd,
    )
    return physics, basis, facade


class TestSecondOrderCapability:
    def test_bundle_is_second_order(self, bkd):
        """The facade provides the full second-order bundle."""
        _, _, facade = _build_pressurized_cylinder(bkd)
        derivs = facade.param_derivatives()
        assert derivs.param_jacobian is not None
        assert derivs.initial_param_jacobian is not None
        assert derivs.param_param_hvp is not None
        assert derivs.state_param_hvp is not None
        assert derivs.param_state_hvp is not None
        assert derivs.bc_flux_param_sensitivity is not None

    def test_hvp_identities(self, bkd):
        """Exact identities on the polar domain: param-param symmetry
        and the mixed-tensor cross identity."""
        physics, basis, facade = _build_pressurized_cylinder(bkd)
        nstates = physics.nstates()
        rng = np.random.default_rng(21)
        state = bkd.asarray(rng.normal(0.0, 0.1, nstates))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        params = bkd.asarray(rng.normal(0.0, 0.3, _NMODES))
        v1 = bkd.asarray(rng.normal(0.0, 1.0, _NMODES))
        v2 = bkd.asarray(rng.normal(0.0, 1.0, _NMODES))
        wstate = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        facade.apply(params)
        derivs = facade.param_derivatives()
        assert derivs.param_param_hvp is not None
        assert derivs.state_param_hvp is not None
        assert derivs.param_state_hvp is not None

        h12 = derivs.param_param_hvp(state, 0.0, params, adj, v1) @ v2
        h21 = derivs.param_param_hvp(state, 0.0, params, adj, v2) @ v1
        bkd.assert_allclose(
            bkd.asarray([h12]), bkd.asarray([h21]), rtol=1e-12
        )

        lhs = derivs.state_param_hvp(state, 0.0, params, adj, v1) @ wstate
        rhs = derivs.param_state_hvp(state, 0.0, params, adj, wstate) @ v1
        bkd.assert_allclose(
            bkd.asarray([lhs]), bkd.asarray([rhs]), rtol=1e-12
        )

    def test_pressurized_cylinder_14_check(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Full component FD suite at the HVP tier on the polar-domain
        zoo problem — second-order validation with no oracle."""
        bkd = numpy_bkd
        physics, basis, facade = _build_pressurized_cylinder(bkd)
        model = create_collocation_model(
            physics, bkd, parameterization=facade
        )
        state_eq = CollocationStateEquationWithHVPAdapter(
            model, bkd, parameterization=facade
        )
        nstates = physics.nstates()
        npts = basis.npts()
        constrained = set(
            int(i)
            for i in bkd.to_numpy(
                bkd.concatenate(
                    [
                        physics.boundary_conditions()[k].boundary_indices()
                        for k in range(len(physics.boundary_conditions()))
                    ]
                )
            )
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        weights = bkd.copy(bkd.zeros((nstates, 1)))
        weights[state_idx] = 1.0
        functional = WeightedSumFunctional(weights, _NMODES, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        rng = np.random.default_rng(23)
        param = bkd.asarray(rng.normal(0.0, 0.3, (_NMODES, 1)))
        init_state = bkd.zeros((nstates, 1))
        tols = bkd.copy(checker.get_derivative_tolerances(1e-6))
        # Noise-limited checks (established calibration on PDE-scale
        # quantities): assembled gradient (4), state-equation
        # param_param (5) and state_param (8).
        tols[4] = 5e-6
        tols[5] = 5e-6
        tols[8] = 5e-6
        checker.check_derivatives(init_state, param, tols)


class TestFacadeConstruction:
    def test_wrong_physics_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(TypeError, match="LinearElasticityPhysics"):
            CollocationElasticityParameterization(
                object(),
                youngs_modulus_map=object(),
                poisson_ratio=0.3,
                bkd=bkd,
            )

    def test_pickle_round_trip(self, numpy_bkd: NumpyBkd) -> None:
        """The facade (physics, traction BCs, stacked adapters, field
        maps) survives pickling with identical derivatives."""
        bkd = numpy_bkd
        physics, basis, facade = _build_pressurized_cylinder(bkd)
        clone = pickle.loads(pickle.dumps(facade))
        assert clone.nparams() == facade.nparams()
        nstates = physics.nstates()
        rng = np.random.default_rng(29)
        state = bkd.asarray(rng.normal(0.0, 0.1, nstates))
        adj = bkd.asarray(rng.normal(0.0, 1.0, nstates))
        params = bkd.asarray(rng.normal(0.0, 0.3, _NMODES))
        vvec = bkd.asarray(rng.normal(0.0, 1.0, _NMODES))
        facade.apply(params)
        clone.apply(params)
        fd = facade.param_derivatives()
        cd = clone.param_derivatives()
        assert fd.param_jacobian is not None
        assert cd.param_jacobian is not None
        bkd.assert_allclose(
            cd.param_jacobian(state, 0.0, params),
            fd.param_jacobian(state, 0.0, params),
            rtol=1e-14,
        )
        assert fd.param_param_hvp is not None
        assert cd.param_param_hvp is not None
        bkd.assert_allclose(
            cd.param_param_hvp(state, 0.0, params, adj, vvec),
            fd.param_param_hvp(state, 0.0, params, adj, vvec),
            rtol=1e-14,
        )


class TestAffineEquivalence:
    """QoI and gradient are invariant under an affine domain rescale.

    With displacement :math:`u_s(y) = u(y/s)` on the rescaled domain,
    stresses scale as :math:`1/s` and their divergence as
    :math:`1/s^2`, so matching the forcing with a :math:`1/s^2` factor
    makes the two problems share nodal solutions exactly (same
    polynomial space under the affine map). Nodal QoIs and their
    parameter gradients must then agree to machine precision — a pure
    test of the affine metric factors in the derivative matrices.
    """

    def _build(self, bkd, scale):
        npts_1d = 6
        if scale == 1.0:
            mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        else:
            transform = AffineTransform2D(
                (-scale, scale, -scale, scale), bkd
            )
            mesh = TransformedMesh2D(npts_1d, npts_1d, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        nstates = 2 * npts

        rng = np.random.default_rng(31)
        forcing_np = rng.normal(0.0, 1.0, nstates)
        forcing = bkd.asarray(forcing_np / scale**2)
        physics = LinearElasticityPhysics(
            basis,
            bkd,
            lamda=1.0,
            mu=1.0,
            forcing=_ConstantForcing(forcing),
        )
        bcs = []
        for bndry in range(4):
            idx = mesh.boundary_indices(bndry)
            bcs.append(zero_dirichlet_bc(bkd, idx))
            bcs.append(zero_dirichlet_bc(bkd, idx + npts))
        physics.set_boundary_conditions(bcs)

        # Nodal KLE modes: identical arrays on both domains (the nodes
        # correspond under the affine map).
        modes = rng.normal(0.0, 0.2, (npts, _NMODES))
        kle = MeshKLEFieldMap(
            bkd,
            bkd.asarray(np.full(npts, math.log(2.0))),
            bkd.asarray(modes),
        )
        exp = _ExpTransform(bkd)
        e_map = TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)
        facade = CollocationElasticityParameterization(
            physics, youngs_modulus_map=e_map, poisson_ratio=_NU, bkd=bkd
        )
        weights = bkd.copy(bkd.zeros((nstates, 1)))
        weights[nstates // 2 + npts_1d + 1] = 1.0
        functional = WeightedSumFunctional(weights, _NMODES, bkd)
        forward = SteadyForwardModel(
            physics,
            bkd,
            bkd.zeros((nstates,)),
            functional=functional,
            parameterization=facade,
        )
        return forward

    def test_qoi_and_gradient_invariant(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        samples = bkd.asarray(
            np.random.default_rng(37).normal(0.0, 0.3, (_NMODES, 1))
        )
        forward_ref = self._build(bkd, 1.0)
        forward_scaled = self._build(bkd, 1.7)
        qoi_ref = forward_ref(samples)
        qoi_scaled = forward_scaled(samples)
        bkd.assert_allclose(qoi_scaled, qoi_ref, rtol=1e-11)

        derivs_ref: Derivatives = forward_ref.derivatives()
        derivs_scaled: Derivatives = forward_scaled.derivatives()
        assert derivs_ref.jacobian is not None
        assert derivs_scaled.jacobian is not None
        bkd.assert_allclose(
            derivs_scaled.jacobian(samples),
            derivs_ref.jacobian(samples),
            rtol=1e-10,
        )


class _ConstantForcing:
    """Picklable time-constant forcing field."""

    def __init__(self, values):
        self._values = values

    def __call__(self, time):
        return self._values
