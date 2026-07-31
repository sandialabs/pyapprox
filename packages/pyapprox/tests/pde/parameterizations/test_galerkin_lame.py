"""Tests for the engine-backed galerkin Lame (E, nu) parameterization."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.models.galerkin.steady import (
    GalerkinStateEquationWithHVPAdapter,
)
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array
from scipy.sparse import issparse


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


def _make_physics(bkd, E=1.0, nu=0.3, with_bcs=True):
    """Create a 2D uniform CompositeLinearElasticity."""
    mesh = StructuredMesh2D(
        nx=5,
        ny=5,
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        bkd=bkd,
    )
    basis = VectorLagrangeBasis(mesh, degree=1)

    if with_bcs:
        bc_list = [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
            DirichletBC(basis, "bottom", 0.0, bkd),
            DirichletBC(basis, "top", 0.0, bkd),
        ]
    else:
        bc_list = []

    return CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=E,
        poisson_ratio=nu,
        body_force=_body_force,
        boundary_conditions=bc_list,
        bkd=bkd,
    )


def _make_multi_material_physics(bkd, with_bcs=True):
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

    if with_bcs:
        bc_list = [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
            DirichletBC(basis, "bottom", 0.0, bkd),
            DirichletBC(basis, "top", 0.0, bkd),
        ]
    else:
        bc_list = []

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
        boundary_conditions=bc_list,
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

    def test_param_jacobian_fd_validation(self, numpy_bkd) -> None:
        """FD validation of param_jacobian via DerivativeChecker."""
        bkd = numpy_bkd
        E0, nu0 = 2.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0, with_bcs=False)
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
                res = physics.spatial_residual(u, 0.0)
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

    def test_param_jacobian_multi_material_fd(self, numpy_bkd) -> None:
        """FD validation for multi-material param_jacobian."""
        bkd = numpy_bkd
        physics = _make_multi_material_physics(bkd, with_bcs=False)
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
                res = physics.spatial_residual(u, 0.0)
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
        np.testing.assert_array_equal(ipj_np, 0.0)

    def test_pickle_round_trip(self, numpy_bkd) -> None:
        """The factory product (with its physics) survives pickling and
        the clone produces identical derivatives."""
        import pickle

        bkd = numpy_bkd
        physics = _make_multi_material_physics(bkd, with_bcs=True)
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

    def test_all_derivative_components_match_fd(self, numpy_bkd) -> None:
        """Steady 14-check component suite through the HVP adapter.

        The (E, nu) -> Lame map curvature makes ``param_param_hvp``
        genuinely nonzero; the mixed HVP blocks come from the engine's
        FromLinearity slots over the typed Lame assemblies.
        """
        bkd = numpy_bkd
        physics = _make_multi_material_physics(bkd, with_bcs=True)
        param_obj = create_galerkin_lame_parameterization(physics, bkd)
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
        functional = WeightedSumFunctional(weights, 4, bkd)

        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        param = bkd.asarray(np.array([[1.0], [0.3], [5.0], [0.2]]))
        init_state = bkd.zeros((nstates, 1))
        # 2.3e-6 base is 2x the worst ratio observed across the CI
        # matrix (1.13e-6, consistent on ubuntu and macos —
        # dependency-version numerics, not hardware noise)
        tols = bkd.copy(checker.get_derivative_tolerances(2.3e-6))
        checker.check_derivatives(init_state, param, tols)

        # FD-noise-immune symmetry identity <Hu, v> = <Hv, u>:
        # breaks for contraction/orientation bugs.
        vvec = bkd.asarray(np.array([[0.5], [0.7], [-0.6], [0.2]]))
        uvec = bkd.asarray(np.array([[-0.2], [0.9], [0.3], [-0.4]]))
        h_v = adjoint_op.hvp(init_state, param, vvec)
        h_u = adjoint_op.hvp(init_state, param, uvec)
        bkd.assert_allclose(
            bkd.sum(h_v * uvec), bkd.sum(h_u * vvec), rtol=1e-12
        )

    def test_adjoint_gradient_steady_integration(self, numpy_bkd) -> None:
        """Full steady-state solve + adjoint gradient through parameterization.

        QoI: Q(u(p)) = c^T u(p) where u(p) solves K(p)*u = b.
        Adjoint gradient: dQ/dp = (dF/dp)^T lambda, with J^T lambda = -c.
        """
        bkd = numpy_bkd
        E0, nu0 = 1.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0, with_bcs=True)
        param = create_galerkin_lame_parameterization(physics, bkd)
        param_jac = param.param_derivatives().param_jacobian
        assert param_jac is not None

        np.random.seed(42)
        n = physics.nstates()
        c_np = np.random.randn(n)

        def qoi_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                param.apply(p)
                r = SteadyStateSolver(
                    physics,
                    tol=1e-12,
                    max_iter=5,
                    line_search=False,
                ).solve_linear()
                u_np = bkd.to_numpy(r.solution)
                Q = c_np @ u_np
                results.append(Q)
            param.apply(bkd.asarray(np.array([E0, nu0])))
            return bkd.reshape(bkd.asarray(np.array(results)), (1, nsamples))

        def adjoint_gradient(params: Array) -> Array:
            p = params[:, 0]
            param.apply(p)
            r = SteadyStateSolver(
                physics,
                tol=1e-12,
                max_iter=5,
                line_search=False,
            ).solve_linear()
            u_sol = r.solution

            # Solve adjoint: J^T lambda = -c
            J_np = _to_dense(physics.jacobian(u_sol, 0.0))
            lam_np = np.linalg.solve(J_np.T, -c_np)

            # Use parameterization param_jacobian (raw, no BC enforcement)
            dF_dp_raw = bkd.to_numpy(param_jac(u_sol, 0.0, p))

            # Apply BC enforcement for steady-state
            dF_dp = bkd.to_numpy(
                physics._apply_dirichlet_to_param_jacobian(
                    bkd.asarray(dF_dp_raw), u_sol, 0.0
                )
            )

            grad = dF_dp.T @ lam_np
            param.apply(bkd.asarray(np.array([E0, nu0])))
            return bkd.reshape(bkd.asarray(grad), (1, 2))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=qoi_of_params,
            jacobian=adjoint_gradient,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray(np.array([[E0], [nu0]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6
