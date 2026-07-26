"""Facade validation for CollocationAdvectionDiffusionParameterization.

Construction raises, slot validation, pickle round-trip, the
mixed-composite bc-flux block assembly, and the transformed-domain
(polar) FD checks for the parameter jacobian, the HVP identities, and
the boundary-flux sensitivity. (The transition-scoped oracle-parity
suite validated the facade against the deleted interim classes at
rtol 1e-12 before their removal.)
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
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.field_term import (
    FieldStateJacobianAdapter,
    StateJacobianAdapter,
    _FieldParameterizationTerm,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

_NMODES = 3


def _exp_kle_map(bkd, coords_np, nmodes, amplitude):
    """Positive log-KLE field map with usable second derivatives."""
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
        bkd.asarray(np.zeros(coords_np.shape[0])),
        bkd.asarray(modes),
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _build_1d(bkd, npts=12):
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    physics = AdvectionDiffusionReaction(basis, bkd)
    coords = bkd.to_numpy(basis.nodes())
    return basis, physics, coords


def _rng_arrays(bkd, npts, nparams, seed=7):
    rng = np.random.default_rng(seed)
    state = bkd.asarray(rng.normal(0.0, 1.0, npts))
    adj = bkd.asarray(rng.normal(0.0, 1.0, npts))
    params = bkd.asarray(rng.normal(0.0, 0.4, nparams))
    vvec = bkd.asarray(rng.normal(0.0, 1.0, nparams))
    wvec = bkd.asarray(rng.normal(0.0, 1.0, npts))
    return state, adj, params, vvec, wvec


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


class TestMixedCompositeBCFlux:
    def test_mixed_composite_bc_flux_blocks(self, bkd):
        """Diffusion block equals B @ G'; reaction columns are zero."""
        basis, physics, coords = _build_1d(bkd)
        npts = basis.npts()
        diff_map = _exp_kle_map(bkd, coords, _NMODES, 0.4)
        react_map = _exp_kle_map(bkd, coords, 2, 0.3)
        facade = CollocationAdvectionDiffusionParameterization(
            physics,
            diffusion_map=diff_map,
            reaction_map=react_map,
            bkd=bkd,
        )
        nparams = facade.nparams()
        assert nparams == _NMODES + 2
        state, _, params, _, _ = _rng_arrays(bkd, npts, nparams)
        facade.apply(params)
        bc_indices = bkd.array([0, npts - 1], dtype=int)
        normals = bkd.asarray(np.array([[-1.0], [1.0]]))
        bc_flux_fn = facade.param_derivatives().bc_flux_param_sensitivity
        assert bc_flux_fn is not None
        result = bc_flux_fn(state, 0.0, params, bc_indices, normals)
        bmat = physics.boundary_flux_diffusion_jacobian(
            state, 0.0, bc_indices, normals
        )
        expected_diff = bmat @ diff_map.jacobian(params[:_NMODES])
        bkd.assert_allclose(
            result[:, :_NMODES], expected_diff, rtol=1e-12
        )
        bkd.assert_allclose(
            result[:, _NMODES:], bkd.zeros((2, 2)), atol=1e-15
        )


class TestFacadeConstruction:
    def test_no_maps_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        _, physics, _ = _build_1d(bkd)
        with pytest.raises(TypeError, match="at least one"):
            CollocationAdvectionDiffusionParameterization(physics, bkd=bkd)

    def test_wrong_physics_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        _, _, coords = _build_1d(bkd)
        field_map = _exp_kle_map(bkd, coords, 2, 0.4)
        with pytest.raises(TypeError, match="AdvectionDiffusionReaction"):
            CollocationAdvectionDiffusionParameterization(
                object(), diffusion_map=field_map, bkd=bkd
            )

    def test_bc_flux_slot_must_be_callable(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        basis, physics, coords = _build_1d(bkd)
        field_map = _exp_kle_map(bkd, coords, 2, 0.4)
        with pytest.raises(TypeError, match="bc_flux_field_jacobian"):
            _FieldParameterizationTerm.linear_field_state(
                setter=physics.set_diffusion,
                physics=physics,
                field_jacobian=StateJacobianAdapter(
                    physics.residual_diffusion_jacobian
                ),
                field_state_jacobian=FieldStateJacobianAdapter(
                    physics.residual_diffusion_state_jacobian
                ),
                field_map=field_map,
                bkd=bkd,
                nstates=physics.nstates(),
                nfield_dofs=physics.npts(),
                bc_flux_field_jacobian=42,  # type: ignore[arg-type]
            )

    def test_pickle_round_trip(self, numpy_bkd: NumpyBkd) -> None:
        """The facade (physics, field maps, setter adapters) survives
        pickling and the clone produces identical derivatives."""
        bkd = numpy_bkd
        basis, physics, coords = _build_1d(bkd)
        npts = basis.npts()
        facade = CollocationAdvectionDiffusionParameterization(
            physics,
            diffusion_map=_exp_kle_map(bkd, coords, _NMODES, 0.4),
            forcing_map=_exp_kle_map(bkd, coords, 2, 0.5),
            bkd=bkd,
        )
        clone = pickle.loads(pickle.dumps(facade))
        assert clone.nparams() == facade.nparams()
        state, adj, params, vvec, _ = _rng_arrays(
            bkd, npts, facade.nparams()
        )
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
        bc_indices = bkd.array([0, npts - 1], dtype=int)
        normals = bkd.asarray(np.array([[-1.0], [1.0]]))
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


class TestTransformedDomainDerivatives:
    """Parameter derivatives on a genuinely curved (polar) domain.

    Interior assemblies inherit transform correctness from the
    residual-level transform tests structurally; these checks close
    the seams that do not inherit it: FD ground truth for the chain
    rule, exact HVP identities, and the boundary-flux sensitivity with
    genuine curved-boundary normals.
    """

    def _build_polar(self, bkd, npts_1d=12):
        transform = PolarTransform(
            r_bounds=(1.0, 2.0),
            theta_bounds=(-math.pi / 2, math.pi / 2),
            bkd=bkd,
        )
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        physics = AdvectionDiffusionReaction(basis, bkd)
        pts = bkd.to_numpy(mesh.points())
        # Smooth 1D coordinate for the KLE modes: use x + y (bounded,
        # smooth on the annulus patch).
        coords = pts[0] + pts[1]
        field_map = _exp_kle_map(bkd, coords, _NMODES, 0.3)
        facade = CollocationAdvectionDiffusionParameterization(
            physics, diffusion_map=field_map, bkd=bkd
        )
        return basis, physics, facade, field_map, pts

    def test_param_jacobian_derivative_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        basis, physics, facade, _, _ = self._build_polar(bkd)
        npts = basis.npts()
        rng = np.random.default_rng(3)
        state = bkd.asarray(rng.normal(0.0, 1.0, npts))
        params0 = bkd.asarray(rng.normal(0.0, 0.3, _NMODES))
        derivs = facade.param_derivatives()
        assert derivs.param_jacobian is not None
        param_jac = derivs.param_jacobian

        def fun(p):
            facade.apply(p)
            return physics.residual(state, 0.0)

        def jac(p):
            facade.apply(p)
            return param_jac(state, 0.0, p)

        checker = DerivativeChecker(
            _ParamFunctionWrapper(bkd, _NMODES, npts, fun, jac)
        )
        errors = checker.check_derivatives(params0[:, None], verbosity=0)
        # Measured V-shape (decay to 1.2e-8 then roundoff climb): the
        # polar derivative-matrix norms raise the FD floor, nudging
        # the clean ratio marginally past 1e-6 — noise, not a
        # systematic plateau.
        assert bkd.to_float(bkd.min(errors[0])) <= 1e-7
        assert checker.error_ratio(errors[0]) <= 5e-6

    def test_hvp_identities(self, numpy_bkd: NumpyBkd) -> None:
        """Exact (FD-noise-free) identities on the polar domain:
        param-param symmetry <H v, w> == <H w, v> and the mixed-tensor
        cross identity <state_param_hvp(v), w> == <param_state_hvp(w), v>.
        """
        bkd = numpy_bkd
        basis, physics, facade, _, _ = self._build_polar(bkd)
        npts = basis.npts()
        rng = np.random.default_rng(5)
        state = bkd.asarray(rng.normal(0.0, 1.0, npts))
        adj = bkd.asarray(rng.normal(0.0, 1.0, npts))
        params = bkd.asarray(rng.normal(0.0, 0.3, _NMODES))
        v1 = bkd.asarray(rng.normal(0.0, 1.0, _NMODES))
        v2 = bkd.asarray(rng.normal(0.0, 1.0, _NMODES))
        wstate = bkd.asarray(rng.normal(0.0, 1.0, npts))
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

    def test_bc_flux_sensitivity_derivative_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """FD of the outer-arc normal flux w.r.t. parameters, with
        genuine curved-boundary normals n = (x, y)/r."""
        bkd = numpy_bkd
        basis, physics, facade, _, pts = self._build_polar(bkd)
        npts = basis.npts()
        radius = np.sqrt(pts[0] ** 2 + pts[1] ** 2)
        outer = np.where(np.abs(radius - 2.0) < 1e-10)[0]
        assert outer.shape[0] > 0
        normals_np = np.stack(
            [pts[0][outer] / radius[outer], pts[1][outer] / radius[outer]],
            axis=1,
        )
        bc_indices = bkd.array(list(outer), dtype=int)
        normals = bkd.asarray(normals_np)
        rng = np.random.default_rng(9)
        state = bkd.asarray(rng.normal(0.0, 1.0, npts))
        params0 = bkd.asarray(rng.normal(0.0, 0.3, _NMODES))
        derivs = facade.param_derivatives()
        assert derivs.bc_flux_param_sensitivity is not None
        bc_flux_fn = derivs.bc_flux_param_sensitivity

        def fun(p):
            facade.apply(p)
            # residual() refreshes the physics's cached diffusion
            # array, which compute_flux reads directly.
            physics.residual(state, 0.0)
            flux = physics.compute_flux(state)
            result = bkd.zeros((bc_indices.shape[0],))
            for dim in range(2):
                result = (
                    result + flux[dim][bc_indices] * normals[:, dim]
                )
            return result

        def jac(p):
            facade.apply(p)
            return bc_flux_fn(state, 0.0, p, bc_indices, normals)

        checker = DerivativeChecker(
            _ParamFunctionWrapper(
                bkd, _NMODES, bc_indices.shape[0], fun, jac
            )
        )
        errors = checker.check_derivatives(params0[:, None], verbosity=0)
        # Measured V-shape (decay to 1.4e-8 then roundoff climb):
        # noise-limited ratio, same calibration as the interior
        # param-jacobian check above.
        assert bkd.to_float(bkd.min(errors[0])) <= 1e-7
        assert checker.error_ratio(errors[0]) <= 5e-6
