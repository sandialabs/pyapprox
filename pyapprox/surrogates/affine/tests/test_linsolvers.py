import unittest

import numpy as np
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.affine.linearsystemsolvers import (
    OMPSolver,
    QuantileRegressionSolver,
    EntropicLoss,
    EntropicRegressionSolver,
    ConservativeEntropicRegressionSolver,
    ConservativeLstSqSolver,
    ConservativeQuantileRegressionSolver,
)
from pyapprox.optimization.risk import EntropicRisk


class TestLinearSolvers:
    def setUp(self):
        np.random.seed(1)

    def test_omp_solver(self):
        # Test dense least squares solution is recovered
        # when OMP is allowed to add all columns
        bkd = self.get_backend()
        ncols = 10
        Amat = bkd.asarray(np.random.uniform(-1, 1, (101, ncols)))
        assert np.linalg.matrix_rank(Amat) == ncols
        coefs = bkd.asarray(np.random.normal(0, 1, (ncols, 1)))
        Bmat = Amat @ coefs
        solver = OMPSolver(max_nonzeros=np.inf, rtol=0, backend=bkd)
        omp_coefs = solver.solve(Amat, Bmat)
        assert bkd.allclose(omp_coefs, coefs)
        assert solver._termination_flag == 1

        # Test exit when columns are not independent
        solver = OMPSolver(max_nonzeros=np.inf, rtol=0, backend=bkd)
        Cmat = bkd.hstack((Amat, Amat[:, 2:3]))
        omp_coefs = solver.solve(Cmat, Bmat)
        assert bkd.allclose(omp_coefs[:-1], coefs)
        assert bkd.allclose(
            omp_coefs[-1, 0],
            bkd.zeros(
                2,
            ),
        )
        assert solver._termination_flag == 2

        # Test recovery of sparse coefficients
        solver = OMPSolver(max_nonzeros=np.inf, rtol=1e-14, backend=bkd)
        sparse_coefs = bkd.copy(coefs)
        ninactive_cols = 4
        inactive_indices = bkd.asarray(
            np.random.permutation(np.arange(ncols)), dtype=int
        )[:ninactive_cols]
        sparse_coefs[inactive_indices] = 0.0
        Bmat = Amat @ sparse_coefs
        omp_coefs = solver.solve(Amat, Bmat)
        assert bkd.allclose(omp_coefs, sparse_coefs)
        assert solver._termination_flag == 0

    def test_quantile_regression(self):
        bkd = self.get_backend()

        quantile = 0.5
        solver = QuantileRegressionSolver(quantile, backend=bkd)
        basis_mat = bkd.array([1, 1, 1, 1, 1])[:, None]
        values = bkd.array([2, 4, 6, 8, 10])[:, None]
        # Residuals are
        # [2, 4, 6, 8, 10] - [1, 1, 1, 1, 1] * 6 = [-4, -2, 0, -2, -4]
        # median residual is zero
        coef = solver.solve(basis_mat, values)
        assert bkd.allclose(coef, bkd.full((1,), 6))

        quantile = 0.8
        solver = QuantileRegressionSolver(quantile, backend=bkd)
        basis_mat = bkd.array([1, 1, 1, 1, 1])[:, None]
        values = bkd.array([2, 4, 6, 8, 10])[:, None]
        # Residuals are
        # [2, 4, 6, 8, 10] - [1, 1, 1, 1, 1] * 8 = [-6, -4, -2, 0, 2]
        # median residual is zero
        coef = solver.solve(basis_mat, values)
        assert bkd.allclose(coef, bkd.full((1,), 8))

    def test_entropic_regression(self):
        bkd = self.get_backend()
        nsamples, nvars = 10000, 1
        basis_mat = bkd.asarray(np.random.normal(0, 1, (nsamples, nvars)))
        true_coefs = bkd.ones((nvars, 1))
        train_values = basis_mat @ true_coefs
        noise_std = 0.1
        noise = bkd.asarray(np.random.normal(0, noise_std, train_values.shape))
        train_values += noise
        loss = EntropicLoss(basis_mat, train_values, backend=bkd)

        coefs = bkd.asarray(np.random.normal(0, 1, (nvars, 1)))
        errors = loss.check_apply_jacobian(coefs)
        assert errors.min() / errors.max() < 1e-6
        errors = loss.check_apply_hessian(coefs)
        assert errors.min() / errors.max() < 1e-6

        solver = EntropicRegressionSolver(backend=bkd)
        coef = solver.solve(basis_mat, train_values)
        residual = train_values - basis_mat @ coef
        # we are approximating a constant
        # thus the residual stat should be the same as the stat
        # applied to the noise
        # The stat of the entropic risk quadrangle is also
        # the risk measure
        stat = EntropicRisk(1.0, backend=bkd)
        stat.set_samples(residual.T)
        residual_stat = stat()
        stat.set_samples(noise.T)
        assert bkd.allclose(residual_stat, stat(), rtol=1e-2)

    def _check_conservative_surrogate(self, solver):
        bkd = self.get_backend()
        nsamples, nvars = 10, 3
        basis_mat = bkd.asarray(np.random.normal(0, 1, (nsamples, nvars)))
        true_coefs = bkd.ones((nvars, 1))
        train_values = basis_mat @ true_coefs
        noise_std = 0.1

        ntrials = 5
        for ii in range(ntrials):
            noise = bkd.asarray(
                np.random.normal(0, noise_std, train_values.shape)
            )
            train_values += noise
            coef = solver.solve(basis_mat, train_values)
            solver._risk_measure.set_samples((basis_mat @ coef).T)
            risk_value = solver._risk_measure()
            solver._risk_measure.set_samples(train_values.T)
            data_risk_value = solver._risk_measure()
            print(risk_value, data_risk_value)
            assert risk_value >= data_risk_value

    def test_conservative_surrogates(self):
        bkd = self.get_backend()
        test_cases = [
            ConservativeEntropicRegressionSolver(backend=bkd),
            ConservativeLstSqSolver(strength=1.0, backend=bkd),
            ConservativeQuantileRegressionSolver(quantile=0.5, backend=bkd),
        ]
        for test_case in test_cases:
            self._check_conservative_surrogate(test_case)


class TestNumpyLinearSolvers(TestLinearSolvers, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchLinearSolvers(TestLinearSolvers, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
