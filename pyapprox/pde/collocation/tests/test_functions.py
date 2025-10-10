import unittest
from functools import partial

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.pde.collocation.basis import (
    ChebyshevCollocationBasis1D,
    ChebyshevCollocationBasis2D,
    ChebyshevCollocationBasis3D,
)
from pyapprox.pde.collocation.functions import (
    ScalarSolution,
    ScalarFunction,
    VectorSolutionComponent,
    nabla,
    div,
    VectorOperator,
    VectorSolution,
    MatrixOperator,
    ZeroJac,
    DiagJac,
    CollocationSubdomainFunction,
)
from pyapprox.pde.collocation.sparsejac import DenseJac
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform1D,
    ScaleAndTranslationTransform2D,
    ScaleAndTranslationTransform3D,
)
from pyapprox.pde.collocation.mesh import (
    ChebyshevCollocationMesh1D,
    ChebyshevCollocationMesh2D,
    ChebyshevCollocationMesh3D,
)


class TestOperators:
    def setUp(self):
        np.random.seed(1)

    def test_sparse_jacobian(self):
        bkd = self.get_backend()
        nrows, ninput_funs = 3, 2
        shape = (nrows, nrows * ninput_funs)
        dense_jac_array = bkd.asarray(np.random.normal(0, 1, shape))
        mat = bkd.asarray(np.random.normal(0, 1, (shape[0], shape[0])))
        dense_jac = DenseJac(bkd, shape, dense_jac_array)
        diag_jac_array = bkd.asarray(
            np.random.normal(0, 1, (nrows, ninput_funs))
        )
        diag_jac = DiagJac(bkd, shape, diag_jac_array)
        zero_jac = ZeroJac(bkd, shape)
        vec = bkd.arange(1, nrows + 1)

        # operations on zero jacobian
        assert isinstance(zero_jac * 2.0, ZeroJac)
        assert isinstance(zero_jac * vec, ZeroJac)
        assert isinstance(zero_jac - 2.0 * zero_jac, ZeroJac)
        assert isinstance(zero_jac + zero_jac, ZeroJac)
        assert isinstance(zero_jac / 2.0, ZeroJac)
        assert isinstance(zero_jac / vec, ZeroJac)
        assert isinstance(zero_jac.rdot(mat), ZeroJac)
        assert isinstance(-zero_jac, ZeroJac)

        # operations on diagonal jacobian
        assert isinstance(diag_jac * 2.0, DiagJac)
        assert bkd.allclose(
            (diag_jac * 2.0).get_jacobian(),
            bkd.hstack(
                [bkd.diag(2 * diag_jac_array[:, ii]) for ii in range(2)]
            ),
        )
        assert isinstance(diag_jac * vec, DiagJac)
        assert bkd.allclose(
            (diag_jac * vec).get_jacobian(),
            bkd.hstack(
                [bkd.diag(vec * diag_jac_array[:, ii]) for ii in range(2)]
            ),
        )
        assert isinstance(diag_jac - 2.0 * diag_jac, DiagJac)
        assert bkd.allclose(
            (diag_jac - 2.0 * diag_jac).get_jacobian(),
            bkd.hstack([bkd.diag(-diag_jac_array[:, ii]) for ii in range(2)]),
        )
        assert isinstance(diag_jac + diag_jac, DiagJac)
        assert bkd.allclose(
            (diag_jac + diag_jac).get_jacobian(),
            bkd.hstack(
                [bkd.diag(2 * diag_jac_array[:, ii]) for ii in range(2)]
            ),
        )
        assert isinstance(diag_jac / 2.0, DiagJac)
        assert bkd.allclose(
            (diag_jac / 2.0).get_jacobian(),
            bkd.hstack(
                [bkd.diag(diag_jac_array[:, ii] / 2) for ii in range(2)]
            ),
        )
        assert isinstance(diag_jac / vec, DiagJac)
        assert bkd.allclose(
            (diag_jac / vec).get_jacobian(),
            bkd.hstack(
                [bkd.diag(diag_jac_array[:, ii] / vec) for ii in range(2)]
            ),
        )
        assert isinstance(diag_jac.rdot(mat), DenseJac)
        assert bkd.allclose(
            diag_jac.rdot(mat).get_jacobian(), mat @ diag_jac.get_jacobian()
        )
        assert isinstance(-diag_jac, DiagJac)
        assert bkd.allclose(
            -diag_jac.get_jacobian(),
            bkd.hstack([bkd.diag(-diag_jac_array[:, ii]) for ii in range(2)]),
        )

        # operations on dense jacobian
        assert isinstance(dense_jac * 2.0, DenseJac)
        assert bkd.allclose(
            (dense_jac * 2.0).get_jacobian(), 2.0 * dense_jac_array
        )
        assert isinstance(dense_jac * vec, DenseJac)
        assert bkd.allclose(
            (dense_jac * vec).get_jacobian(), dense_jac_array * vec[:, None]
        )
        assert isinstance(dense_jac - 2.0 * dense_jac, DenseJac)
        assert bkd.allclose(
            (dense_jac - 2.0 * dense_jac).get_jacobian(), -dense_jac_array
        )
        assert isinstance(dense_jac + dense_jac, DenseJac)
        assert bkd.allclose(
            (dense_jac + dense_jac).get_jacobian(), 2.0 * dense_jac_array
        )
        assert isinstance(dense_jac / 2.0, DenseJac)
        assert bkd.allclose(
            (dense_jac / 2.0).get_jacobian(), dense_jac_array / 2.0
        )
        assert isinstance(dense_jac / vec, DenseJac)
        assert bkd.allclose(
            (dense_jac / vec).get_jacobian(), dense_jac_array / vec[:, None]
        )
        assert isinstance(dense_jac.rdot(mat), DenseJac)
        assert bkd.allclose(
            dense_jac.rdot(mat).get_jacobian(), mat @ dense_jac_array
        )
        assert isinstance(-dense_jac, DenseJac)
        assert bkd.allclose(-dense_jac.get_jacobian(), -dense_jac_array)

        # Combining zero and diagonal jacobians
        assert isinstance(diag_jac - 2.0 * zero_jac, DiagJac)
        assert bkd.allclose(
            (diag_jac - 2.0 * zero_jac).get_jacobian(),
            bkd.hstack([bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]),
        )
        assert isinstance(diag_jac + zero_jac, DiagJac)
        assert bkd.allclose(
            (diag_jac + zero_jac).get_jacobian(),
            bkd.hstack([bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]),
        )
        assert isinstance(zero_jac - 2.0 * diag_jac, DiagJac)
        assert bkd.allclose(
            (zero_jac - 2.0 * diag_jac).get_jacobian(),
            bkd.hstack(
                [bkd.diag(-2.0 * diag_jac_array[:, ii]) for ii in range(2)]
            ),
        )
        assert isinstance(zero_jac + diag_jac, DiagJac)
        assert bkd.allclose(
            (zero_jac + diag_jac).get_jacobian(),
            bkd.hstack([bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]),
        )

        # Combining zero and dense jacobians
        assert isinstance(dense_jac - 2.0 * zero_jac, DenseJac)
        assert bkd.allclose(
            (dense_jac - 2.0 * zero_jac).get_jacobian(), dense_jac_array
        )
        assert isinstance(dense_jac + zero_jac, DenseJac)
        assert bkd.allclose(
            (dense_jac + zero_jac).get_jacobian(), dense_jac_array
        )
        assert isinstance(zero_jac - 2.0 * dense_jac, DenseJac)
        assert bkd.allclose(
            (zero_jac - 2.0 * dense_jac).get_jacobian(), -2 * dense_jac_array
        )
        assert isinstance(zero_jac + dense_jac, DenseJac)
        assert bkd.allclose(
            (zero_jac + dense_jac).get_jacobian(), dense_jac_array
        )

        # Combining diag and dense jacobians
        assert isinstance(diag_jac - 2.0 * dense_jac, DenseJac)
        assert bkd.allclose(
            (diag_jac - 2.0 * dense_jac).get_jacobian(),
            bkd.hstack([bkd.diag(diag_jac_array[:, ii]) for ii in range(2)])
            - 2 * dense_jac_array,
        )
        assert isinstance(diag_jac + dense_jac, DenseJac)
        assert bkd.allclose(
            (diag_jac + dense_jac).get_jacobian(),
            bkd.hstack([bkd.diag(diag_jac_array[:, ii]) for ii in range(2)])
            + dense_jac_array,
        )
        assert isinstance(dense_jac - 2.0 * diag_jac, DenseJac)
        assert bkd.allclose(
            (dense_jac - 2.0 * diag_jac).get_jacobian(),
            bkd.hstack(
                [bkd.diag(-2.0 * diag_jac_array[:, ii]) for ii in range(2)]
            )
            + dense_jac_array,
        )
        assert isinstance(dense_jac + diag_jac, DenseJac)
        assert bkd.allclose(
            (dense_jac + diag_jac).get_jacobian(),
            bkd.hstack([bkd.diag(diag_jac_array[:, ii]) for ii in range(2)])
            + dense_jac_array,
        )

    def _check_operations_on_operators_with_single_input_function(self, basis):
        bkd = self.get_backend()

        def tfun0(xx):
            return bkd.sum((1 + xx) ** 3, axis=0)

        def tfun0_deriv(ii, xx):
            # derivative of fun_0 in 0th phys_var
            return 3 * (1 + xx[ii]) ** 2

        tfun1 = partial(tfun0_deriv, 0)

        def tfun0_hess(ii, xx):
            # derivative  of fun_0 in 0th phys_var
            return 6 * (1 + xx[ii])

        def tfun2(xx):
            return bkd.sum((1 + xx) ** 3, axis=0)

        def tsumfun(xx):
            return tfun1(xx) + tfun2(xx)

        def tsubfun(xx):
            return tfun1(xx) - tfun2(xx)

        def tprodfun(xx):
            return tfun1(xx) * tfun2(xx)

        def tpowfun(xx):
            return tfun1(xx) ** 3

        def tsqrtfun(xx):
            return tfun1(xx) ** 0.5

        def tdivfun(xx):
            return tfun1(xx) / tfun2(xx)

        fun_values0 = tfun0(basis.mesh().mesh_pts())
        sol0 = ScalarSolution(basis, fun_values0)

        fun_values1 = tfun1(basis.mesh().mesh_pts())
        sol1 = sol0.deriv(0)
        assert bkd.allclose(sol1.get_values(), fun_values1)
        fun_values2 = tfun2(basis.mesh().mesh_pts())
        # fun is independent of the solution
        fun2 = ScalarFunction(basis, fun_values2)
        sumfun = sol1 + fun2
        assert bkd.allclose(
            sumfun.get_values(), tsumfun(basis.mesh().mesh_pts())
        )
        subfun = sol1 - fun2
        assert bkd.allclose(
            subfun.get_values(), tsubfun(basis.mesh().mesh_pts())
        )
        fun_sub_const = sol1 - 1.0
        assert bkd.allclose(
            fun_sub_const.get_values(), tfun1(basis.mesh().mesh_pts()) - 1
        )
        const_sub_fun = 1.0 - sol1
        assert bkd.allclose(
            const_sub_fun.get_values(), 1 - tfun1(basis.mesh().mesh_pts())
        )
        prodfun = sol1 * fun2
        assert bkd.allclose(
            prodfun.get_values(), tprodfun(basis.mesh().mesh_pts())
        )
        const_prod_fun = 2.0 * sol1
        assert bkd.allclose(
            const_prod_fun.get_values(), 2 * sol1(basis.mesh().mesh_pts())
        )
        powfun = sol1**3
        assert bkd.allclose(
            powfun.get_values(), tpowfun(basis.mesh().mesh_pts())
        )
        sqrtfun = sol1**0.5
        assert bkd.allclose(
            sqrtfun.get_values(), tsqrtfun(basis.mesh().mesh_pts())
        )
        divfun = sol1 / fun2
        assert bkd.allclose(
            divfun.get_values(), tdivfun(basis.mesh().mesh_pts())
        )
        const_div_fun = 2.0 / sol1
        assert bkd.allclose(
            const_div_fun.get_values(), 2 / sol1(basis.mesh().mesh_pts())
        )
        fun_div_const = sol1 / 2.0
        assert bkd.allclose(
            fun_div_const.get_values(), sol1(basis.mesh().mesh_pts()) / 2
        )

        assert bkd.allclose(
            nabla(sol0).get_values(),
            bkd.stack(
                [
                    tfun0_deriv(ii, basis.mesh().mesh_pts())
                    for ii in range(sol0.nphys_vars())
                ],
                axis=0,
            ),
        )

        assert bkd.allclose(
            div(nabla(sol0)).get_values(),
            sum(
                [
                    tfun0_hess(ii, basis.mesh().mesh_pts())
                    for ii in range(sol0.nphys_vars())
                ]
            ),
        )

        # check plots run without calling plt.show
        ax = sol1.get_plot_axis()[1]
        sol1.plot(ax)

        assert isinstance((fun2**2).sparse_jacobian(), ZeroJac)
        assert isinstance((sol0**2).sparse_jacobian(), DiagJac)
        assert isinstance((sol0 * fun2).sparse_jacobian(), DiagJac)
        assert isinstance((sol0 + fun2).sparse_jacobian(), DiagJac)
        assert isinstance((sol0 - fun2).sparse_jacobian(), DiagJac)
        assert isinstance((fun2 - sol0).sparse_jacobian(), DiagJac)
        assert isinstance((1.0 - sol0).sparse_jacobian(), DiagJac)
        assert isinstance((sol0 - 1.0).sparse_jacobian(), DiagJac)
        assert isinstance((1.0 - fun2).sparse_jacobian(), ZeroJac)
        assert isinstance((fun2 - 1.0).sparse_jacobian(), ZeroJac)
        assert isinstance((fun2 / sol0).sparse_jacobian(), DiagJac)
        assert isinstance((1.0 / sol0).sparse_jacobian(), DiagJac)
        assert isinstance((sol0 / 2.0).sparse_jacobian(), DiagJac)
        assert isinstance((1.0 / fun2).sparse_jacobian(), ZeroJac)
        assert isinstance((fun2 / 2.0).sparse_jacobian(), ZeroJac)
        assert isinstance(sol0.deriv(0).sparse_jacobian(), DenseJac)
        assert isinstance(fun2.deriv(0).sparse_jacobian(), ZeroJac)

        if not bkd.jacobian_implemented():
            return

        assert bkd.allclose(
            (sol1 + sol1).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    ScalarFunction(basis, v).deriv(0)
                    + ScalarFunction(basis, v).deriv(0)
                ).get_values(),
                fun_values0,
            ),
        )
        assert bkd.allclose(
            (sol1**3).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    ScalarFunction(basis, v).deriv(0) ** 3
                ).get_values(),
                fun_values0,
            ),
        )

        assert bkd.allclose(
            (sol1**0.5).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    ScalarFunction(basis, v).deriv(0) ** 0.5
                ).get_values(),
                fun_values0,
            ),
        )

        assert bkd.allclose(
            (sol1 * sol1).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    ScalarFunction(basis, v).deriv(0)
                    * ScalarFunction(basis, v).deriv(0)
                ).get_values(),
                fun_values0,
            ),
        )

        assert bkd.allclose(
            (2.0 * sol1 - sol1).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    2.0 * ScalarFunction(basis, v).deriv(0)
                    - ScalarFunction(basis, v).deriv(0)
                ).get_values(),
                fun_values0,
            ),
        )

        assert bkd.allclose(
            (sol1 / fun2).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    ScalarFunction(basis, v).deriv(0) / fun2
                ).get_values(),
                fun_values0,
            ),
        )

        assert bkd.allclose(
            (1.0 / sol1).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    1.0 / ScalarFunction(basis, v).deriv(0)
                ).get_values(),
                fun_values0,
            ),
        )

        assert bkd.allclose(
            (sol1 / 2.0).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    ScalarFunction(basis, v).deriv(0) / 2.0
                ).get_values(),
                fun_values0,
            ),
        )

        assert bkd.allclose(
            nabla(sol0).get_jacobian(),
            bkd.jacobian(
                lambda v: (nabla(ScalarFunction(basis, v))).get_values(),
                fun_values0,
            )[:, None],
        )

        assert bkd.allclose(
            div(nabla(sol0)).get_jacobian(),
            bkd.jacobian(
                lambda v: (div(nabla(ScalarFunction(basis, v)))).get_values(),
                fun_values0,
            ),
        )

    def test_operations_on_operators_with_single_input_function(self):
        bkd = self.get_backend()
        bounds = [0, 1]
        nterms = [4]
        transform = ScaleAndTranslationTransform1D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh1D(nterms, transform)
        basis = ChebyshevCollocationBasis1D(mesh)
        self._check_operations_on_operators_with_single_input_function(basis)

        bounds = [0, 1, 0, 1]
        nterms = [4, 4]
        transform = ScaleAndTranslationTransform2D([-1, 1, -1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh2D(nterms, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        self._check_operations_on_operators_with_single_input_function(basis)

        bounds = [0, 1, 0, 1, 0, 1]
        nterms = [4, 4, 4]
        transform = ScaleAndTranslationTransform3D(
            [-1, 1, -1, 1, -1, 1], bounds, bkd
        )
        mesh = ChebyshevCollocationMesh3D(nterms, transform)
        basis = ChebyshevCollocationBasis3D(mesh)
        self._check_operations_on_operators_with_single_input_function(basis)

    def test_operations_on_operators_with_multiple_input_functions(self):
        bkd = self.get_backend()
        bounds = [0, 1]
        nterms = [4]
        transform = ScaleAndTranslationTransform1D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh1D(nterms, transform)
        basis = ChebyshevCollocationBasis1D(mesh)

        def tfun0(xx):
            return bkd.sum((1 + xx) ** 3, axis=0)

        def tfun1(xx):
            return 3 * (1 + xx[0]) ** 2

        def tfun2(xx):
            return bkd.sum((1 + xx) ** 2, axis=0)

        def tfun3(xx):
            return bkd.sum((2 + xx), axis=0)

        fun_values0 = tfun0(basis.mesh().mesh_pts())
        fun_values1 = tfun1(basis.mesh().mesh_pts())
        fun_values2 = tfun2(basis.mesh().mesh_pts())
        fun_values3 = tfun3(basis.mesh().mesh_pts())
        sol0 = VectorSolutionComponent(basis, 3, 0, fun_values0)
        fun1 = sol0.deriv(0)
        sol2 = VectorSolutionComponent(basis, 3, 1, fun_values2)
        sol3 = VectorSolutionComponent(basis, 3, 2, fun_values3)
        fun4 = VectorSolutionComponent(basis, 3, 2, fun_values3).deriv(0)

        assert bkd.allclose(
            (fun1 * sol2 / fun4).get_values(), fun_values1 * fun_values2
        )

        vec_sol1 = VectorSolution(basis, 3, 3)
        vec_sol1.set_components([sol0, sol2, sol3])
        vec_sol1.get_flattened_values()

        vec_sol2 = VectorSolution(basis, 3, 3)
        vec_sol2.set_flattened_values(vec_sol1.get_flattened_values())
        assert bkd.allclose(vec_sol2.get_values(), vec_sol1.get_values())

        vec_sol2.set_values(vec_sol1.get_values())
        assert bkd.allclose(vec_sol2.get_values(), vec_sol1.get_values())

        vec_op = VectorOperator(basis, 2, 2)
        vec_op.set_components([fun1, fun4])

        mat_op = MatrixOperator(basis, 2, 2, 2)
        mat_op.set_components([[fun1, fun4], [2.0 * fun1, 2.0 * fun4]])

        inner_prod = vec_op.T @ vec_op
        assert bkd.allclose(
            inner_prod.get_values(), vec_op.sqnorm().get_values()
        )
        matmul = mat_op @ vec_op
        assert bkd.allclose(
            matmul.get_values()[0], vec_op.sqnorm().get_values()
        )
        assert bkd.allclose(
            matmul.get_values()[1], 2.0 * vec_op.sqnorm().get_values()
        )

        if not bkd.jacobian_implemented():
            return

        assert bkd.allclose(
            (fun1 * sol2 / fun4).get_jacobian(),
            # (fun1 / fun4).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    VectorSolutionComponent(
                        basis, 3, 0, v[: basis.mesh().nmesh_pts()]
                    ).deriv(0)
                    * VectorSolutionComponent(
                        basis,
                        3,
                        1,
                        v[
                            basis.mesh().nmesh_pts() : 2
                            * basis.mesh().nmesh_pts()
                        ],
                    )
                    / VectorSolutionComponent(
                        basis,
                        3,
                        2,
                        v[
                            2
                            * basis.mesh().nmesh_pts() : 3
                            * basis.mesh().nmesh_pts()
                        ],
                    ).deriv(0)
                ).get_values(),
                bkd.hstack((fun_values0, fun_values2, fun_values3)),
            ),
        )

        assert bkd.allclose(
            div(nabla(sol0 * sol2)).get_jacobian(),
            bkd.jacobian(
                lambda v: (
                    div(
                        nabla(
                            ScalarFunction(
                                basis, v[: basis.mesh().nmesh_pts()]
                            )
                            * ScalarFunction(
                                basis,
                                v[
                                    basis.mesh().nmesh_pts() : 2
                                    * basis.mesh().nmesh_pts()
                                ],
                            )
                        )
                    )
                ).get_values(),
                bkd.hstack((fun_values0, fun_values2, fun_values3)),
            ),
        )

    def test_integrate(self):
        bkd = self.get_backend()

        def tfun0(xx):
            return bkd.sum((1 + xx) ** 3, axis=0)

        bounds = [0, 1]
        nterms = [10]
        transform = ScaleAndTranslationTransform1D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh1D(nterms, transform)
        basis = ChebyshevCollocationBasis1D(mesh)
        fun_values0 = tfun0(basis.mesh().mesh_pts())
        sol0 = ScalarSolution(basis, fun_values0)
        print(sol0.integrate())
        assert bkd.allclose(sol0.integrate(), bkd.array([15 / 4]), atol=1e-15)

        bounds = [0, 1, 0, 1]
        nterms = [4, 4]
        transform = ScaleAndTranslationTransform2D([-1, 1, -1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh2D(nterms, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        fun_values0 = tfun0(basis.mesh().mesh_pts())
        sol0 = ScalarSolution(basis, fun_values0)
        assert bkd.allclose(sol0.integrate(), bkd.array([15 / 2]), atol=1e-15)

    def test_subdomain_function(self):
        bkd = self.get_backend()

        def tfun0(xx):
            return bkd.sum((1 + xx) ** 3, axis=0)

        # check 1D mesh
        bounds = [0, 1]
        nterms = [10]
        transform = ScaleAndTranslationTransform1D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh1D(nterms, transform)
        basis = ChebyshevCollocationBasis1D(mesh)
        fun_values0 = tfun0(basis.mesh().mesh_pts())
        sol0 = ScalarSolution(basis, fun_values0)
        subdomain_fun = CollocationSubdomainFunction([0, 1], sol0)
        assert bkd.allclose(
            bkd.min(
                subdomain_fun._subdomain_fun.basis().mesh().mesh_pts(), axis=1
            ),
            bkd.array([0.5]),
        )
        assert bkd.allclose(
            bkd.max(
                subdomain_fun._subdomain_fun.basis().mesh().mesh_pts(), axis=1
            ),
            bkd.array([1.0]),
        )
        pts = bkd.asarray(np.random.uniform(0.5, 1.0, (1, 10)))
        assert bkd.allclose(subdomain_fun(pts), sol0(pts))
        assert bkd.allclose(
            subdomain_fun.integrate(), bkd.asarray(175.0 / 64.0)
        )

        # check 1D mesh with higher resolution subdomain
        subdomain_fun = CollocationSubdomainFunction([0, 1], sol0, [20])
        assert bkd.allclose(
            bkd.min(
                subdomain_fun._subdomain_fun.basis().mesh().mesh_pts(), axis=1
            ),
            bkd.array([0.5]),
        )
        assert bkd.allclose(
            bkd.max(
                subdomain_fun._subdomain_fun.basis().mesh().mesh_pts(), axis=1
            ),
            bkd.array([1.0]),
        )
        pts = bkd.asarray(np.random.uniform(0.5, 1.0, (1, 10)))
        assert bkd.allclose(subdomain_fun(pts), sol0(pts))
        assert bkd.allclose(
            subdomain_fun.integrate(), bkd.asarray(175.0 / 64.0)
        )
        assert subdomain_fun._subdomain_fun.basis().mesh()._npts_1d[0] == 20

        # check 2D mesh
        bounds = [0, 1, 0, 1]
        nterms = [4, 4]
        transform = ScaleAndTranslationTransform2D([-1, 1, -1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh2D(nterms, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        fun_values0 = tfun0(basis.mesh().mesh_pts())
        sol0 = ScalarSolution(basis, fun_values0)
        # create subdomain on [0.5, 1.0, 0.5, 1.0]
        # map [-1, 1, -1, 1] -> [0, 1, 0, 1] (latter still in orth domain
        # then map this half of domain to user domain
        subdomain_fun = CollocationSubdomainFunction([0, 1, 0, 1], sol0)
        assert bkd.allclose(
            bkd.min(
                subdomain_fun._subdomain_fun.basis().mesh().mesh_pts(), axis=1
            ),
            bkd.array([0.5, 0.5]),
        )
        assert bkd.allclose(
            bkd.max(
                subdomain_fun._subdomain_fun.basis().mesh().mesh_pts(), axis=1
            ),
            bkd.array([1.0, 1.0]),
        )
        pts = bkd.asarray(np.random.uniform(0.5, 1.0, (2, 10)))
        assert bkd.allclose(subdomain_fun(pts), sol0(pts))
        assert bkd.allclose(
            subdomain_fun.integrate(), bkd.asarray(175.0 / 64.0)
        )


class TestNumpyOperators(TestOperators, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchOperators(TestOperators, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
