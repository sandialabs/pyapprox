"""Facade validation for CollocationHyperelasticityParameterization.

First-order bundle honesty (jacobians and bc-flux present, HVP fields
None, direct second-order engine calls raise), DerivativeChecker FD of
the parameter jacobian in 1D and 2D, and pickle round-trip.
"""

import math
import pickle
from typing import Callable

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.physics.hyperelasticity import (
    create_hyperelasticity,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.parameterizations.collocation_hyperelasticity import (
    CollocationHyperelasticityParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

_NMODES = 3
_NU = 0.3


def _exp_kle_map(bkd, coords_np, nmodes):
    scale = np.max(np.abs(coords_np)) or 1.0
    modes = np.stack(
        [
            0.3 * np.sin((k + 1) * math.pi * coords_np / scale) / (k + 1)
            for k in range(nmodes)
        ],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd,
        bkd.asarray(np.full(coords_np.shape[0], math.log(3.0))),
        bkd.asarray(modes),
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _build(bkd, ndim):
    """Hyperelastic physics + facade with an admissible smooth state
    (small displacement keeps det(F) > 0)."""
    if ndim == 1:
        basis = ChebyshevBasis1D(TransformedMesh1D(10, bkd), bkd)
        coords = bkd.to_numpy(basis.nodes())
        state = bkd.asarray(0.02 * np.sin(math.pi * coords))
    else:
        basis = ChebyshevBasis2D(TransformedMesh2D(5, 5, bkd), bkd)
        pts = bkd.to_numpy(basis.mesh().points())
        coords = pts[0] + pts[1]
        state = bkd.asarray(
            np.concatenate(
                [
                    0.02 * np.sin(math.pi * pts[0]) * np.cos(pts[1]),
                    0.02 * np.cos(pts[0]) * np.sin(math.pi * pts[1]),
                ]
            )
        )
    physics = create_hyperelasticity(basis, bkd, mu=1.0, lamda=1.5)
    facade = CollocationHyperelasticityParameterization(
        physics,
        youngs_modulus_map=_exp_kle_map(bkd, coords, _NMODES),
        poisson_ratio=_NU,
        bkd=bkd,
    )
    return physics, facade, state


class _ParamFunctionWrapper:
    """Adapts params -> vector function + jacobian for DerivativeChecker."""

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        nqoi: int,
        fun: Callable[[Array], Array],
        jac: Callable[[Array], Array],
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._nqoi = nqoi
        self._fun = fun
        self._jac = jac

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, sample: Array) -> Array:
        return self._fun(sample[:, 0])[:, None]

    def jacobian(self, sample: Array) -> Array:
        return self._jac(sample[:, 0])

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


class TestStagedFirstOrder:
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_bundle_is_first_order_with_bc_flux(self, bkd, ndim):
        """The bundle advertises exactly the staged capability."""
        _, facade, _ = _build(bkd, ndim)
        derivs = facade.param_derivatives()
        assert derivs.param_jacobian is not None
        assert derivs.initial_param_jacobian is not None
        assert derivs.bc_flux_param_sensitivity is not None
        assert derivs.param_param_hvp is None
        assert derivs.state_param_hvp is None
        assert derivs.param_state_hvp is None

    def test_direct_second_order_calls_raise(self, numpy_bkd: NumpyBkd):
        """The engine term's mixed slots fail loudly rather than
        returning silently-wrong values."""
        bkd = numpy_bkd
        _, facade, state = _build(bkd, 1)
        rng = np.random.default_rng(43)
        params = bkd.asarray(rng.normal(0.0, 0.3, _NMODES))
        adj = bkd.asarray(rng.normal(0.0, 1.0, state.shape[0]))
        vvec = bkd.asarray(rng.normal(0.0, 1.0, _NMODES))
        term = facade._inner
        with pytest.raises(RuntimeError, match="unavailable"):
            term.param_param_hvp(state, 0.0, params, adj, vvec)
        with pytest.raises(NotImplementedError, match="tangent"):
            term.state_param_hvp(state, 0.0, params, adj, vvec)
        with pytest.raises(NotImplementedError, match="tangent"):
            term.param_state_hvp(state, 0.0, params, adj, state)

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_param_jacobian_derivative_checker(
        self, numpy_bkd: NumpyBkd, ndim
    ):
        bkd = numpy_bkd
        physics, facade, state = _build(bkd, ndim)
        nstates = physics.nstates()
        rng = np.random.default_rng(47)
        params0 = bkd.asarray(rng.normal(0.0, 0.3, _NMODES))
        param_jac = facade.param_derivatives().param_jacobian
        assert param_jac is not None

        def fun(p):
            facade.apply(p)
            return physics.residual(state, 0.0)

        def jac(p):
            facade.apply(p)
            return param_jac(state, 0.0, p)

        checker = DerivativeChecker(
            _ParamFunctionWrapper(bkd, _NMODES, nstates, fun, jac)
        )
        errors = checker.check_derivatives(params0[:, None], verbosity=0)
        # Measured V-shape (2D decays to 4.9e-8 then roundoff climbs):
        # the nonlinear PK1 chain raises the FD floor, nudging the
        # clean ratio marginally past 1e-6 — noise, not a plateau.
        assert bkd.to_float(bkd.min(errors[0])) <= 1e-7
        assert checker.error_ratio(errors[0]) <= 5e-6

    def test_wrong_physics_raises(self, numpy_bkd: NumpyBkd):
        bkd = numpy_bkd
        with pytest.raises(TypeError, match="HyperelasticityPhysics"):
            CollocationHyperelasticityParameterization(
                object(),
                youngs_modulus_map=object(),
                poisson_ratio=_NU,
                bkd=bkd,
            )

    def test_pickle_round_trip(self, numpy_bkd: NumpyBkd):
        bkd = numpy_bkd
        physics, facade, state = _build(bkd, 2)
        clone = pickle.loads(pickle.dumps(facade))
        assert clone.nparams() == facade.nparams()
        rng = np.random.default_rng(53)
        params = bkd.asarray(rng.normal(0.0, 0.3, _NMODES))
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
        npts = physics.npts()
        bc_indices = bkd.array([0, 3], dtype=int)
        raw = rng.normal(0.0, 1.0, (2, 2))
        raw /= np.linalg.norm(raw, axis=1)[:, None]
        normals = bkd.asarray(raw)
        assert fd.bc_flux_param_sensitivity is not None
        assert cd.bc_flux_param_sensitivity is not None
        bkd.assert_allclose(
            cd.bc_flux_param_sensitivity(
                state, 0.0, params, bc_indices, normals
            ),
            fd.bc_flux_param_sensitivity(
                state, 0.0, params, bc_indices, normals
            ),
            rtol=1e-14,
        )
