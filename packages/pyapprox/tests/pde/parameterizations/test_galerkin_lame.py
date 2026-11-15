"""Tests for the engine-backed galerkin Lame (E, nu) parameterization."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from scipy.sparse import issparse

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from tests._helpers.galerkin_steady_checks import (
    first_free_row,
    observed_steady_function,
    residual_bundle_report,
)


def _to_dense(mat):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def _body_force_impl(x):
    """Module-level body force (local defs would break pickle tests)."""
    f = np.zeros_like(x)
    f[0, :] = 1.0
    f[1, :] = -2.0
    return f


# Declared time-independent rather than left bare. TimeIndependent is a
# module-level class wrapping a module-level function, so this stays
# picklable --- which the pickle tests below rely on.
_body_force = TimeIndependent(_body_force_impl)


def _make_system(physics, bkd, with_bcs=True):
    """Compose ``physics`` with zero Dirichlet BCs on all four sides, or
    with no BCs when ``with_bcs`` is False."""
    basis = physics.basis()
    if with_bcs:
        bc_list = [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
            DirichletBC(basis, "bottom", 0.0, bkd),
            DirichletBC(basis, "top", 0.0, bkd),
        ]
    else:
        bc_list = []
    return compose_galerkin_system(physics, bc_list)


def _make_physics(bkd, E=1.0, nu=0.3):
    """Create a 2D uniform CompositeLinearElasticity."""
    mesh = StructuredMesh2D(
        nx=5,
        ny=5,
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        bkd=bkd,
    )
    basis = VectorLagrangeBasis(mesh, degree=1)

    return CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=E,
        poisson_ratio=nu,
        body_force=_body_force,
        bkd=bkd,
    )


def _make_multi_material_physics(bkd):
    """Create a 2-material CompositeLinearElasticity."""
    mesh = StructuredMesh2D(
        nx=10,
        ny=5,
        bounds=[[0.0, 2.0], [0.0, 1.0]],
        bkd=bkd,
    )
    basis = VectorLagrangeBasis(mesh, degree=1)
    nelems = basis.skfem_basis().mesh.nelements

    left_elems = np.arange(nelems // 2)
    right_elems = np.arange(nelems // 2, nelems)

    return CompositeLinearElasticity(
        basis=basis,
        material_map={
            "left": (1.0, 0.3),
            "right": (5.0, 0.2),
        },
        element_materials={
            "left": left_elems,
            "right": right_elems,
        },
        bkd=bkd,
        body_force=_body_force,
    )


class TestGalerkinLameParameterizationFactory:

    def test_protocol_conformance(self, numpy_bkd) -> None:
        """The factory product satisfies ParameterizationProtocol."""
        physics = _make_physics(numpy_bkd)
        param = create_galerkin_lame_parameterization(physics, numpy_bkd)
        assert isinstance(param, ParameterizationProtocol)

    def test_nparams(self, numpy_bkd) -> None:
        """nparams returns 2 * nmaterials."""
        physics = _make_physics(numpy_bkd)
        param = create_galerkin_lame_parameterization(physics, numpy_bkd)
        assert param.nparams() == 2

        physics_multi = _make_multi_material_physics(numpy_bkd)
        param_multi = create_galerkin_lame_parameterization(
            physics_multi, numpy_bkd
        )
        assert param_multi.nparams() == 4

    def test_apply_changes_stiffness(self, numpy_bkd) -> None:
        """apply() modifies the physics stiffness matrix."""
        physics = _make_physics(numpy_bkd, E=1.0, nu=0.3)
        param = create_galerkin_lame_parameterization(physics, numpy_bkd)

        K1 = _to_dense(physics.stiffness_matrix()).copy()

        param.apply(numpy_bkd.asarray(np.array([2.0, 0.25])))
        K2 = _to_dense(physics.stiffness_matrix())

        diff = np.linalg.norm(K2 - K1)
        assert diff > 1e-10

    def test_apply_poisson_validation(self, numpy_bkd) -> None:
        """apply() raises for invalid Poisson ratio."""
        physics = _make_physics(numpy_bkd)
        param = create_galerkin_lame_parameterization(physics, numpy_bkd)
        with pytest.raises(ValueError):
            param.apply(numpy_bkd.asarray(np.array([1.0, 0.5])))

    def test_param_jacobian_fd_validation(self, numpy_bkd: NumpyBkd) -> None:
        """FD validation of param_jacobian via DerivativeChecker."""
        bkd = numpy_bkd
        E0, nu0 = 2.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0)
        operator = _make_system(physics, bkd, with_bcs=False).spatial_operator()
        param = create_galerkin_lame_parameterization(physics, bkd)
        param_jac = param.param_derivatives().param_jacobian
        assert param_jac is not None
        nstates = physics.nstates()

        rng = np.random.RandomState(42)
        u = bkd.asarray(rng.randn(nstates))

        def residual_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                param.apply(p)
                res = operator.spatial_residual(u, 0.0)
                results.append(bkd.reshape(res, (nstates, 1)))
            param.apply(bkd.asarray(np.array([E0, nu0])))
            return bkd.hstack(results)

        def jacobian_of_params(params: Array) -> Array:
            p = params[:, 0]
            param.apply(p)
            pj = param_jac(u, 0.0, p)
            param.apply(bkd.asarray(np.array([E0, nu0])))
            return pj

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=2,
            fun=residual_of_params,
            jacobian=jacobian_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray(np.array([[E0], [nu0]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6

    def test_param_jacobian_multi_material_fd(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """FD validation for multi-material param_jacobian."""
        bkd = numpy_bkd
        physics = _make_multi_material_physics(bkd)
        operator = _make_system(physics, bkd, with_bcs=False).spatial_operator()
        param = create_galerkin_lame_parameterization(physics, bkd)
        param_jac = param.param_derivatives().param_jacobian
        assert param_jac is not None
        nstates = physics.nstates()

        rng = np.random.RandomState(42)
        u = bkd.asarray(rng.randn(nstates))
        p0 = np.array([1.0, 0.3, 5.0, 0.2])

        def residual_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                param.apply(p)
                res = operator.spatial_residual(u, 0.0)
                results.append(bkd.reshape(res, (nstates, 1)))
            param.apply(bkd.asarray(p0))
            return bkd.hstack(results)

        def jacobian_of_params(params: Array) -> Array:
            p = params[:, 0]
            param.apply(p)
            pj = param_jac(u, 0.0, p)
            param.apply(bkd.asarray(p0))
            return pj

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=4,
            fun=residual_of_params,
            jacobian=jacobian_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray(p0.reshape(-1, 1))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6

    def test_initial_param_jacobian_is_zero(self, numpy_bkd) -> None:
        """initial_param_jacobian returns all zeros."""
        bkd = numpy_bkd
        physics = _make_physics(bkd)
        param = create_galerkin_lame_parameterization(physics, bkd)
        initial_param_jacobian = (
            param.param_derivatives().initial_param_jacobian
        )
        assert initial_param_jacobian is not None
        p = bkd.asarray(np.array([1.0, 0.3]))
        ipj_np = bkd.to_numpy(initial_param_jacobian(p))
        bkd.assert_allclose(
            ipj_np, np.zeros_like(ipj_np), rtol=0.0, atol=0.0
        )

    def test_pickle_round_trip(self, numpy_bkd: NumpyBkd) -> None:
        """The factory product (with its physics) survives pickling and
        the clone produces identical derivatives."""
        import pickle

        bkd = numpy_bkd
        physics = _make_multi_material_physics(bkd)
        param_obj = create_galerkin_lame_parameterization(physics, bkd)
        clone = pickle.loads(pickle.dumps(param_obj))
        assert clone.nparams() == param_obj.nparams()

        rng = np.random.RandomState(42)
        u = bkd.asarray(rng.randn(physics.nstates()))
        p0 = bkd.asarray(np.array([1.0, 0.3, 5.0, 0.2]))
        param_obj.apply(p0)
        clone.apply(p0)
        orig_jac = param_obj.param_derivatives().param_jacobian
        clone_jac = clone.param_derivatives().param_jacobian
        assert orig_jac is not None
        assert clone_jac is not None
        bkd.assert_allclose(
            orig_jac(u, 0.0, p0), clone_jac(u, 0.0, p0), rtol=1e-14
        )

    def test_residual_derivatives_pass_the_bundle_checker(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Two materials at (E, nu) = (1, 0.3), (5, 0.2): the (E, nu) ->
        Lame map's curvature makes the parameter-parameter term nonzero,
        and the mixed terms come from the engine's FromLinearity slots
        over the typed Lame assemblies; every declared derivative passes
        finite differences and dot tests (state linspace(0.1, 0.9),
        random adjoint, seed 5)."""
        bkd = numpy_bkd
        physics = _make_multi_material_physics(bkd)
        system = _make_system(physics, bkd)
        param_obj = create_galerkin_lame_parameterization(physics, bkd)
        nstates = physics.nstates()
        report = residual_bundle_report(
            system,
            param_obj,
            bkd.asarray(np.linspace(0.1, 0.9, nstates)),
            bkd.asarray(np.random.default_rng(5).normal(size=nstates)),
            bkd.asarray(np.array([1.0, 0.3, 5.0, 0.2])),
        )
        assert report.failures() == []

    def test_solved_gradient_and_hvp_match_fd(self, numpy_bkd: NumpyBkd) -> None:
        """The solved displacement at the first free DOF, on the steady
        system path, has an adjoint gradient and HVP passing central
        finite differences below 1e-6, and an HVP satisfying
        <Hv, u> = <Hu, v> to 1e-12, which breaks for contraction or
        orientation bugs."""
        bkd = numpy_bkd
        physics = _make_multi_material_physics(bkd)
        system = _make_system(physics, bkd)
        param_obj = create_galerkin_lame_parameterization(physics, bkd)
        function = observed_steady_function(
            system, param_obj, [first_free_row(system, bkd)], bkd
        )
        param = bkd.asarray(np.array([[1.0], [0.3], [5.0], [0.2]]))
        errors = DerivativeChecker(function).check_derivatives(
            param, relative=True, central=True
        )
        assert len(errors) == 2
        assert max(bkd.to_float(bkd.min(e)) for e in errors) < 1e-6
        hvp = function.derivatives().hvp
        assert hvp is not None
        vvec = bkd.asarray(np.array([[0.5], [0.7], [-0.6], [0.2]]))
        uvec = bkd.asarray(np.array([[-0.2], [0.9], [0.3], [-0.4]]))
        h_v, h_u = hvp(param, vvec), hvp(param, uvec)
        bkd.assert_allclose(
            bkd.sum(h_v * uvec)[None], bkd.sum(h_u * vvec)[None], rtol=1e-12
        )
