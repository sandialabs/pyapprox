"""Engine validation for the quasilinear-diffusion parameterization.

The exemplar for the engine's CALLABLE mixed second-derivative slots:
the field-carrying term a(x)*kappa(u)*grad(u) is nonlinear in the
state, so state_param_hvp/param_state_hvp route through the physics's
mixed assemblies rather than the linearity identities.

Setup: -(a(x) kappa(u) u')' = 1 on [0, 1], kappa(u) = 1 + u^2, u = 0 at
both ends, P1 on 10 cells, a = exp(W p) with three modes
0.4 sin((k+1) pi x)/(k+1), at p = (0.4, -0.3, 0.2). The residual's
derivatives pass the bundle checker at state linspace(0.1, 0.9) with a
random adjoint (seed 5); the solved state at the first free node, on the
steady system path, has a gradient and HVP passing central finite
differences below 1e-6 and an HVP symmetric to 1e-12.
"""

import pickle
from typing import Any, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeIndependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction,
    QuasilinearDiffusion,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.parameterizations.galerkin_quasilinear_diffusion import (
    create_quasilinear_diffusivity_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.galerkin_steady_checks import (
    first_free_row,
    observed_steady_function,
    residual_bundle_report,
)

_NPARAMS = 3
_Np = NDArray[Any]


def _kappa(u: _Np) -> _Np:
    return 1.0 + u**2


def _kappa_deriv(u: _Np) -> _Np:
    return 2.0 * u


def _kappa_second_deriv(u: _Np) -> _Np:
    return np.full_like(u, 2.0)


def _forcing(x: _Np) -> _Np:
    return np.ones(x.shape[1])


def _build_physics_and_map(
    bkd: NumpyBkd,
) -> Tuple[QuasilinearDiffusion[_Np], GalerkinSystem[_Np], TransformedFieldMap[_Np]]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = QuasilinearDiffusion(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        kappa=_kappa,
        kappa_deriv=_kappa_deriv,
        kappa_second_deriv=_kappa_second_deriv,
        forcing=TimeIndependent(_forcing),
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
        [0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1) for k in range(_NPARAMS)],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    field_map = TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)
    return physics, system, field_map


class TestQuasilinearDiffusivityParameterization:
    def test_residual_derivatives_pass_the_bundle_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The residual's Jacobians and four adjoint-weighted second
        derivatives, including the callable mixed slots (nonzero because
        kappa depends on u) and the kappa''-driven state-state term, pass
        finite differences and dot tests."""
        bkd = numpy_bkd
        physics, system, field_map = _build_physics_and_map(bkd)
        param_obj = create_quasilinear_diffusivity_parameterization(
            physics, field_map, bkd
        )
        nstates = physics.nstates()
        report = residual_bundle_report(
            system,
            param_obj,
            bkd.asarray(np.linspace(0.1, 0.9, nstates)),
            bkd.asarray(np.random.default_rng(5).normal(size=nstates)),
            bkd.asarray(np.array([0.4, -0.3, 0.2])),
        )
        assert report.failures() == []

    def test_solved_gradient_and_hvp_match_fd(self, numpy_bkd: NumpyBkd) -> None:
        """The solved state at the first free node, on the steady system
        path, has a gradient and HVP that pass central finite
        differences, and an HVP satisfying <Hv, u> = <Hu, v>, which no
        finite-difference noise reaches."""
        bkd = numpy_bkd
        physics, system, field_map = _build_physics_and_map(bkd)
        param_obj = create_quasilinear_diffusivity_parameterization(
            physics, field_map, bkd
        )
        function = observed_steady_function(
            system, param_obj, [first_free_row(system, bkd)], bkd
        )
        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
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

    def test_wrong_physics_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        adr = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=TimeIndependent(_forcing),
        )
        _, _, field_map = _build_physics_and_map(bkd)
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
        physics, _, field_map = _build_physics_and_map(bkd)
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
            param_obj.param_jacobian(state, 0.0, params).matrix(),
            clone.param_jacobian(state, 0.0, params).matrix(),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            param_obj.state_param_hvp(state, 0.0, params, adj, vvec),
            clone.state_param_hvp(state, 0.0, params, adj, vvec),
            rtol=1e-14,
        )
