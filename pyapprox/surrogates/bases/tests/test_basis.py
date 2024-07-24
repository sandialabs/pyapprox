import unittest

import numpy as np

from pyapprox.surrogates.orthopoly.poly import LegendrePolynomial1D
from pyapprox.surrogates.bases.basis import (
    MonomialBasis,
    OrthonormalPolynomialBasis,
)
from pyapprox.surrogates.bases.basisexp import (
    MonomialExpansion,
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.bases.linearsystemsolvers import (
    LstSqSolver,
    OMPSolver,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.interp.indexing import sort_indices_lexiographically
from pyapprox.util.sys_utilities import package_available


if package_available("jax"):
    from pyapprox.util.linearalgebra.jaxlinalg import JaxLinAlgMixin


class TestMonomialBasis:
    def setUp(self):
        np.random.seed(1)

    def _check_basis(self, nvars, nterms_1d):
        bkd = self.get_backend()
        basis = MonomialBasis(backend=self.get_backend())
        basis.set_indices(
            bkd._la_cartesian_product(
                [bkd._la_arange(nterms_1d, dtype=int)] * nvars
            )
        )
        samples = bkd._la_array(np.random.uniform(-1, 1, (nvars, 4)))
        basis_mat = basis(samples)
        for ii, index in enumerate(basis._indices.T):
            assert np.allclose(
                basis_mat[:, ii], bkd._la_prod(samples.T**index, axis=1)
            )

    def test_basis(self):
        test_cases = [[1, 4], [2, 4], [3, 4]]
        for test_case in test_cases:
            self._check_basis(*test_case)

    def _check_jacobian(self, nvars, nterms_1d):
        bkd = self.get_backend()
        basis = MonomialBasis(backend=bkd)
        indices = bkd._la_cartesian_product(
            [bkd._la_arange(nterms_1d, dtype=int)] * nvars
        )
        basis.set_indices(indices)
        samples = bkd._la_array(np.random.uniform(-1, 1, (nvars, 4)))
        # samples = bkd._la_array([[-1, 1], [1, 0.5], [0.5, 0.5], [0.5, 1]]).T
        jac = basis.jacobian(samples)
        # assert False
        derivs = bkd._la_stack(
            [samples * 0, samples * 0 + 1]
            + [ii * samples ** (ii - 1) for ii in range(2, nterms_1d)]
        )
        for ii in range(indices.shape[1]):
            for dd in range(nvars):
                index = bkd._la_copy(indices[:, ii : ii + 1])
                # evaluate basis that has constant in direction of derivative
                index = bkd._la_up(index, dd, 0)
                basis.set_indices(index)
                deriv_dd = derivs[indices[dd, ii], dd, :] * bkd._la_prod(
                    basis(samples), axis=1
                )
                assert bkd._la_allclose(deriv_dd, jac[:, ii, dd])

    def test_jacobian(self):
        test_cases = [[1, 4], [2, 4], [3, 4]]
        for test_case in test_cases:
            # print(test_case)
            self._check_jacobian(*test_case)

    def _check_fit_monomial_expansion(self, nvars, solver, nqoi):
        bkd = self.get_backend()
        nterms_1d = 3
        basis = MonomialBasis(backend=bkd)
        indices = bkd._la_cartesian_product(
            [bkd._la_arange(nterms_1d, dtype=int)] * nvars
        )
        basis.set_indices(indices)
        basisexp = MonomialExpansion(basis, solver=solver, nqoi=nqoi)
        ntrain_samples = 2 * basis.nterms()
        train_samples = bkd._la_cos(
            bkd._la_array(np.random.uniform(0, np.pi, (nvars, ntrain_samples)))
        )

        # Attempt to recover coefficients of additive function
        def fun(samples):
            values = bkd._la_sum(samples**2 + samples, axis=0)[:, None] + 1.0
            # Create 2 QoI
            return bkd._la_hstack([(ii + 1) * values for ii in range(nqoi)])

        train_values = fun(train_samples)
        basisexp.fit(train_samples, train_values)
        coef = basisexp.get_coefficients()
        nonzero_indices = bkd._la_hstack(
            (
                bkd._la_where(
                    bkd._la_count_nonzero(basis._indices, axis=0) == 0
                )[0],
                bkd._la_where(
                    bkd._la_count_nonzero(basis._indices, axis=0) == 1
                )[0],
            )
        )
        true_coef = bkd._la_full((basis.nterms(), basisexp.nqoi()), 0)
        for ii in range(nqoi):
            # true_coef[nonzero_indices, ii] = ii+1
            true_coef = bkd._la_up(true_coef, (nonzero_indices, ii), ii + 1)
        assert bkd._la_allclose(coef, true_coef)
        samples = bkd._la_atleast2d(np.random.uniform(-1, 1, (nvars, 1000)))
        assert bkd._la_allclose(basisexp(samples), fun(samples))

    def test_fit_monomial_expansion(self):
        bkd = self.get_backend()
        test_cases = [
            [1, LstSqSolver(backend=bkd), 2],
            [2, LstSqSolver(backend=bkd), 2],
            [3, LstSqSolver(backend=bkd), 2],
            [1, OMPSolver(max_nonzeros=3, backend=bkd), 1],
            [2, OMPSolver(max_nonzeros=6, backend=bkd), 1],
            [3, OMPSolver(max_nonzeros=9, backend=bkd), 1],
        ]
        for test_case in test_cases:
            self._check_fit_monomial_expansion(*test_case)

    def test_orthonormal_polynomial_basis(self):
        bkd = self.get_backend()
        nvars, degree = 2, 2
        bases_1d = [LegendrePolynomial1D(backend=bkd)] * nvars
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_indices(
            bkd._la_array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T
        )
        samples = bkd._la_array(np.random.uniform(-1, 1, (nvars, 101)))
        basis_mat = basis(samples)
        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(nvars):
            x = samples[dd, :]
            exact_basis_vals_1d.append(
                bkd._la_stack(
                    [1 + 0.0 * x, x, 0.5 * (3.0 * x**2 - 1)], axis=0
                ).T
            )
            exact_basis_derivs_1d.append(
                bkd._la_stack([0.0 * x, 1.0 + 0.0 * x, 3.0 * x], axis=0).T
            )
            exact_basis_vals_1d[-1] /= bkd._la_sqrt(
                1.0 / (2 * bkd._la_arange(degree + 1) + 1)
            )
            exact_basis_derivs_1d[-1] /= bkd._la_sqrt(
                1.0 / (2 * bkd._la_arange(degree + 1) + 1)
            )

        exact_basis_mat = bkd._la_stack(
            [
                exact_basis_vals_1d[0][:, 0],
                exact_basis_vals_1d[0][:, 1],
                exact_basis_vals_1d[1][:, 1],
                exact_basis_vals_1d[0][:, 2],
                exact_basis_vals_1d[0][:, 1] * exact_basis_vals_1d[1][:, 1],
                exact_basis_vals_1d[1][:, 2],
            ],
            axis=0,
        ).T

        assert bkd._la_allclose(basis_mat, exact_basis_mat)

    def _check_multiply_expansion(self, bexp1, bexp2, nqoi):
        bkd = self.get_backend()
        coef1 = bkd._la_arange(bexp1.nterms() * nqoi).reshape(
            (bexp1.nterms(), nqoi)
        )
        coef2 = bkd._la_arange(bexp2.nterms() * nqoi).reshape(
            (bexp2.nterms(), nqoi)
        )
        bexp1.set_coefficients(coef1)
        bexp2.set_coefficients(coef2)

        bexp3 = bexp1 * bexp2
        samples = bkd._la_array(np.random.uniform(-1, 1, (bexp1.nvars(), 101)))
        assert bkd._la_allclose(
            bexp3(samples), bexp1(samples) * bexp2(samples)
        )

        for order in range(4):
            bexp = bexp1**order
            assert bkd._la_allclose(bexp(samples), bexp1(samples) ** order)

    def _check_multiply_monomial_expansion(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        basis1 = MonomialBasis(backend=bkd)
        indices1 = bkd._la_cartesian_product(
            [bkd._la_arange(nterms_1d, dtype=int)] * nvars
        )
        basis2 = MonomialBasis(backend=bkd)
        indices2 = bkd._la_cartesian_product(
            [bkd._la_arange(nterms_1d + 1, dtype=int)] * nvars
        )
        basis1.set_indices(indices1)
        basis2.set_indices(indices2)
        bexp1 = MonomialExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = MonomialExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_multiply_expansion(bexp1, bexp2, nqoi)

    def test_multiply_monomial_expansion(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_multiply_monomial_expansion(*test_case)

    def _check_multiply_pce(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        polys_1d = [LegendrePolynomial1D(bkd)] * nvars
        basis1 = OrthonormalPolynomialBasis(polys_1d)
        indices1 = bkd._la_cartesian_product(
            [bkd._la_arange(nterms_1d, dtype=int)] * nvars
        )
        basis2 = OrthonormalPolynomialBasis(polys_1d)
        indices2 = bkd._la_cartesian_product(
            [bkd._la_arange(nterms_1d + 1, dtype=int)] * nvars
        )
        basis1.set_indices(indices1)
        basis2.set_indices(indices2)
        bexp1 = PolynomialChaosExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = PolynomialChaosExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_multiply_expansion(bexp1, bexp2, nqoi)

    def test_multiply_pce(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_multiply_pce(*test_case)

    def test_marginalize_pce(self):
        nvars, nqoi, nterms_1d = 4, 2, 3
        bkd = self.get_backend()
        polys_1d = [LegendrePolynomial1D(bkd)] * nvars
        basis = OrthonormalPolynomialBasis(polys_1d)
        indices = bkd._la_cartesian_product(
            [bkd._la_arange(nterms_1d, dtype=int)] * nvars
        )
        basis.set_indices(indices)
        pce = PolynomialChaosExpansion(basis, solver=None, nqoi=nqoi)
        coef = bkd._la_arange(pce.nterms() * nqoi).reshape(
            (pce.nterms(), nqoi)
        )
        pce.set_coefficients(coef)
        inactive_idx = bkd._la_arange(nvars, dtype=int)[::2]
        mpce = pce.marginalize(inactive_idx)
        assert mpce.nterms() == (nterms_1d) ** (nvars - len(inactive_idx)) - 1
        assert bkd._la_allclose(
            sort_indices_lexiographically(mpce.basis.get_indices()),
            # delete first index which corresponds to constant term
            sort_indices_lexiographically(
                bkd._la_cartesian_product(
                    [bkd._la_arange(nterms_1d, dtype=int)] * mpce.nvars()
                )
            )[:, 1:],
        )
        indices = bkd._la_all(
            pce.basis.get_indices()[inactive_idx] == 0, axis=0
        )
        # delete first index which corresponds to constant term
        # indices[bkd._la_all(pce.basis.get_indices() == 0, axis=0)] = False
        indices = bkd._la_up(
            indices, bkd._la_all(pce.basis.get_indices() == 0, axis=0), False
        )
        assert bkd._la_allclose(
            mpce.get_coefficients(), pce.get_coefficients()[indices]
        )


class TestNumpyMonomialBasis(TestMonomialBasis, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchMonomialBasis(TestMonomialBasis, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()


class TestJaxMonomialBasis(TestMonomialBasis, unittest.TestCase):
    def setUp(self):
        if not package_available("jax"):
            self.skipTest("jax not available")
        TestMonomialBasis.setUp(self)

    def get_backend(self):
        return JaxLinAlgMixin()


if __name__ == "__main__":
    unittest.main()
