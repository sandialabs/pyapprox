r"""Facade validation for CollocationElasticityParameterization.

Second-order validation without an oracle, on the pressurized cylinder
(a quarter annulus :math:`1 \le r \le 2`, :math:`0 \le \theta \le \pi/2`,
6 x 6 polar nodes): stress-free outer arc, pressure 0.5 on the inner arc,
symmetry edges (one displacement component zero, the other's traction
zero); :math:`E = \exp(\log 10 + \text{3 KLE modes})`, :math:`\nu = 0.3`.
The traction rows read the parameterized material.

- the facade's bundle is second order, with exact HVP identities;
- the closed right-hand side's bundle passes ``ODEResidualBundleChecker``
  at a random state and parameter (state, parameter and second
  derivatives, traction rows included);
- the steady function's Jacobian (and HVP, when the closed bundle has
  second derivatives) pass central finite differences;
- pickling keeps the derivatives;
- QoI and gradient are invariant under an affine rescale of a square
  with matched forcing.
"""

import math
import pickle
from typing import Generic, List, Tuple

import numpy as np
import pytest

from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import RobinPart, RobinRows
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import TractionNormalOperator
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import AffineTransform2D, PolarTransform
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import TransformedFieldMap, _ExpTransform
from pyapprox.pde.models.collocation.closed_system import (
    ClosedRightHandSide,
    CollocationSystem,
)
from pyapprox.pde.models.collocation.steady_function import (
    collocation_steady_function,
)
from pyapprox.pde.parameterizations.collocation_elasticity import (
    CollocationElasticityParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat

_NMODES = 3
_NU = 0.3
_E_MEAN = 10.0
_NPTS_1D = 6


def _exp_kle_map(
    bkd: Backend[Array], coords: Array, amplitude: float, log_mean: float
) -> TransformedFieldMap[Array]:
    """:math:`E = \\exp(\\text{log\\_mean} + \\sum_k \\theta_k \\phi_k)`,
    :math:`\\phi_k = a \\sin((k+1) \\pi c / \\max|c|) / (k+1)`."""
    scale = bkd.to_float(bkd.max(bkd.abs(coords)))
    modes = bkd.stack(
        [
            amplitude * bkd.sin((k + 1) * math.pi * coords / scale) / (k + 1)
            for k in range(_NMODES)
        ],
        axis=1,
    )
    kle = MeshKLEFieldMap(bkd, bkd.full(tuple(coords.shape), log_mean), modes)
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _lame(young: float) -> Tuple[float, float]:
    """(lambda, mu) of ``young`` at the Poisson ratio."""
    return (
        young * _NU / ((1.0 + _NU) * (1.0 - 2.0 * _NU)),
        young / (2.0 * (1.0 + _NU)),
    )


class _Cylinder(Generic[Array]):
    """The pressurized cylinder's physics, rows and facade. Corner nodes:
    Dirichlet rows beat traction rows, and among tractions the later part
    wins."""

    def __init__(self, bkd: Backend[Array]) -> None:
        transform = PolarTransform((1.0, 2.0), (0.0, math.pi / 2.0), bkd)
        mesh = TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd, transform)
        lamda, mu = _lame(_E_MEAN)
        self.physics = LinearElasticityPhysics(
            ChebyshevBasis2D(mesh, bkd), bkd, lamda=lamda, mu=mu
        )
        npts = mesh.npts()
        parts: List[RobinPart[Array]] = []

        def traction(side: int, component: int, values: Array) -> None:
            indices, normals = mesh.boundary_indices(side), mesh.boundary_normals(side)
            operator = TractionNormalOperator(
                bkd, indices, normals, self.physics, component
            )
            parts.append(
                RobinPart(indices, operator, 0.0, 1.0, NodalField(values), component)
            )

        zeros = bkd.zeros((_NPTS_1D,))
        traction(1, 0, zeros)
        traction(1, 1, zeros)
        traction(2, 0, zeros)
        traction(3, 1, zeros)
        inner_normals = mesh.boundary_normals(0)
        traction(0, 0, -0.5 * inner_normals[:, 0])
        traction(0, 1, -0.5 * inner_normals[:, 1])
        essential = DirichletRows(
            bkd,
            npts,
            2,
            [
                DirichletPart(mesh.boundary_indices(2), 0.0, component=1),
                DirichletPart(mesh.boundary_indices(3), 0.0, component=0),
            ],
        )
        self.system = CollocationSystem(
            self.physics, essential, [RobinRows(bkd, npts, 2, parts)]
        )
        points = mesh.points()
        self.facade = CollocationElasticityParameterization(
            self.physics,
            youngs_modulus_map=_exp_kle_map(
                bkd, points[0] + points[1], 0.3, math.log(_E_MEAN)
            ),
            poisson_ratio=_NU,
            bkd=bkd,
        )
        # An interior node, x component: no boundary row.
        self.node = 2 * _NPTS_1D + 2


def _random(bkd: Backend[Array], seed: int, size: int, scale: float) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(0.0, scale, size))


class TestSecondOrderCapability:
    def test_bundle_is_second_order(self, bkd: Backend[Array]) -> None:
        """The facade provides the full second-order bundle."""
        derivs = _Cylinder(bkd).facade.param_derivatives()
        assert derivs.param_jacobian is not None
        assert derivs.initial_param_jacobian is not None
        assert derivs.param_param_hvp is not None
        assert derivs.state_param_hvp is not None
        assert derivs.param_state_hvp is not None
        assert derivs.bc_flux_param_sensitivity is not None

    def test_hvp_identities(self, bkd: Backend[Array]) -> None:
        """Exact identities on the polar domain: param-param symmetry
        and the mixed-tensor cross identity."""
        cylinder = _Cylinder(bkd)
        facade, nstates = cylinder.facade, cylinder.physics.nstates()
        state = _random(bkd, 1, nstates, 0.1)
        adj = _random(bkd, 2, nstates, 1.0)
        params = _random(bkd, 3, _NMODES, 0.3)
        v1, v2 = _random(bkd, 4, _NMODES, 1.0), _random(bkd, 5, _NMODES, 1.0)
        wstate = _random(bkd, 6, nstates, 1.0)
        facade.apply(params)
        derivs = facade.param_derivatives()
        assert derivs.param_param_hvp is not None
        assert derivs.state_param_hvp is not None
        assert derivs.param_state_hvp is not None
        h12 = bkd.sum(derivs.param_param_hvp(state, 0.0, params, adj, v1) * v2)
        h21 = bkd.sum(derivs.param_param_hvp(state, 0.0, params, adj, v2) * v1)
        bkd.assert_allclose(h12, h21, rtol=1e-12)
        lhs = bkd.sum(derivs.state_param_hvp(state, 0.0, params, adj, v1) * wstate)
        rhs = bkd.sum(derivs.param_state_hvp(state, 0.0, params, adj, wstate) * v1)
        bkd.assert_allclose(lhs, rhs, rtol=1e-12)


class TestCylinderDerivatives:
    def test_closed_bundle(self, bkd: Backend[Array]) -> None:
        """State, parameter and (where declared) second derivatives of
        the closed right-hand side, traction rows included, against
        finite differences."""
        cylinder = _Cylinder(bkd)
        rhs = ClosedRightHandSide(
            cylinder.system,
            ParameterState(cylinder.facade),
            initial_state=STATED_ZERO,
        )
        assert rhs.derivatives().param is not None
        nstates = cylinder.physics.nstates()
        report = ODEResidualBundleChecker(rhs).check(
            _random(bkd, 7, nstates, 0.1), 0.0, param=_random(bkd, 8, _NMODES, 0.3)
        )
        assert report.failures() == []

    def test_steady_function(self, numpy_bkd: Backend[Array]) -> None:
        """The steady displacement at an interior node: the Jacobian, and
        the HVP when present, pass central finite differences (smallest
        relative error below 1e-6)."""
        bkd = numpy_bkd
        cylinder = _Cylinder(bkd)
        nstates = cylinder.physics.nstates()
        functional: StateFunctional[Array, Array] = StateFunctional.from_terms(
            bkd,
            1,
            state_terms=(ObservedValue(StateEntries(bkd, nstates, [cylinder.node])),),
        )
        function = collocation_steady_function(
            cylinder.system, ParameterState(cylinder.facade), functional
        )
        has_hvp = function.derivatives().hvp is not None
        param = bkd.reshape(_random(bkd, 9, _NMODES, 0.3), (_NMODES, 1))
        errors = DerivativeChecker(function).check_derivatives(
            param, relative=True, central=True
        )
        assert len(errors) == (2 if has_hvp else 1)
        for error in errors:
            assert bkd.to_float(bkd.min(error)) < 1e-6


class TestFacadeConstruction:
    def test_wrong_physics_raises(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="LinearElasticityPhysics"):
            CollocationElasticityParameterization(
                object(),  # type: ignore[arg-type]
                youngs_modulus_map=object(),  # type: ignore[arg-type]
                poisson_ratio=0.3,
                bkd=numpy_bkd,
            )

    def test_pickle_round_trip(self, numpy_bkd: Backend[Array]) -> None:
        """The facade (physics, stacked adapters, field maps) survives
        pickling with identical derivatives."""
        bkd = numpy_bkd
        cylinder = _Cylinder(bkd)
        facade, nstates = cylinder.facade, cylinder.physics.nstates()
        clone = pickle.loads(pickle.dumps(facade))
        assert clone.nparams() == facade.nparams()
        state = _random(bkd, 10, nstates, 0.1)
        adj = _random(bkd, 11, nstates, 1.0)
        params = _random(bkd, 12, _NMODES, 0.3)
        vvec = _random(bkd, 13, _NMODES, 1.0)
        facade.apply(params)
        clone.apply(params)
        fd, cd = facade.param_derivatives(), clone.param_derivatives()
        assert fd.param_jacobian is not None and cd.param_jacobian is not None
        bkd.assert_allclose(
            cd.param_jacobian(state, 0.0, params).matrix(),
            fd.param_jacobian(state, 0.0, params).matrix(),
            rtol=1e-14,
        )
        assert fd.param_param_hvp is not None and cd.param_param_hvp is not None
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

    def _function(
        self, bkd: Backend[Array], scale: float
    ) -> SteadyImplicitFunction[Array, DenseFormat[Array]]:
        transform = AffineTransform2D((-scale, scale, -scale, scale), bkd)
        mesh = TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd, transform)
        npts = mesh.npts()
        nstates = 2 * npts
        rng = np.random.default_rng(31)
        forcing = bkd.asarray(rng.normal(0.0, 1.0, nstates) / scale**2)
        physics = LinearElasticityPhysics(
            ChebyshevBasis2D(mesh, bkd),
            bkd,
            lamda=1.0,
            mu=1.0,
            forcing=NodalField(forcing),
        )
        essential = DirichletRows(
            bkd,
            npts,
            2,
            [
                DirichletPart(mesh.boundary_indices(side), 0.0, component=component)
                for side in range(4)
                for component in (0, 1)
            ],
        )
        # Nodal KLE modes: identical arrays on both domains (the nodes
        # correspond under the affine map).
        modes = bkd.asarray(rng.normal(0.0, 0.2, (npts, _NMODES)))
        kle = MeshKLEFieldMap(bkd, bkd.full((npts,), math.log(2.0)), modes)
        exp = _ExpTransform(bkd)
        facade = CollocationElasticityParameterization(
            physics,
            youngs_modulus_map=TransformedFieldMap(
                kle, exp, exp, bkd, transform_deriv2=exp
            ),
            poisson_ratio=_NU,
            bkd=bkd,
        )
        functional: StateFunctional[Array, Array] = StateFunctional.from_terms(
            bkd,
            1,
            state_terms=(
                ObservedValue(StateEntries(bkd, nstates, [npts + _NPTS_1D + 1])),
            ),
        )
        return collocation_steady_function(
            CollocationSystem(physics, essential), ParameterState(facade), functional
        )

    def test_qoi_and_gradient_invariant(self, numpy_bkd: Backend[Array]) -> None:
        bkd = numpy_bkd
        samples = bkd.reshape(_random(bkd, 37, _NMODES, 0.3), (_NMODES, 1))
        reference, scaled = self._function(bkd, 1.0), self._function(bkd, 1.7)
        bkd.assert_allclose(scaled(samples), reference(samples), rtol=1e-11)
        reference_jacobian = reference.derivatives().jacobian
        scaled_jacobian = scaled.derivatives().jacobian
        assert reference_jacobian is not None and scaled_jacobian is not None
        bkd.assert_allclose(
            scaled_jacobian(samples), reference_jacobian(samples), rtol=1e-10
        )
