import unittest

import numpy as np
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.affine.linearsystemsolvers import (
    LstSqSolver,
    OMPSolver,
    QuantileRegressionSolver,
    EntropicLoss,
    EntropicRegressionSolver,
    ConservativeLstSqSolver,
    ConservativeQuantileRegressionSolver,
    QuantileRegressionCVXOPTSolver,
    FSDRegressionSolver,
    SSDRegressionSolver,
)
from pyapprox.surrogates.affine.basisexp import (
    MonomialExpansion,
    MultiIndexBasis,
)
from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.optimization.risk import EntropicRisk
from pyapprox.util.sys_utilities import package_available
from pyapprox.optimization.minimize import (
    SmoothLogBasedLeftHeavisideFunction,
    SmoothLogBasedMaxFunction,
)


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

    def test_lstsq_solver(self):
        bkd = self.get_backend()
        solver = LstSqSolver(backend=bkd)
        basis_mat = bkd.array([1, 1, 1, 1, 1])[:, None]
        values = bkd.array([2, 4, 6, 8, 10])[:, None]
        coef = solver.solve(basis_mat, values)
        # Assert statistic of risk quadrangle is zero. I.e 0.8 qunatile
        # is zero. Must use method="inverted_cdf" avaiable in numpy
        assert bkd.allclose(bkd.mean(values - basis_mat @ coef), bkd.zeros(1))

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
        coef = solver.solve(basis_mat, values)
        # Assert statistic of risk quadrangle is zero. I.e 0.8 qunatile
        # is zero. Must use method="inverted_cdf" avaiable in numpy
        assert bkd.allclose(
            bkd.asarray(
                np.quantile(
                    (values - basis_mat @ coef),
                    quantile,
                    method="inverted_cdf",
                )
            ),
            bkd.zeros(1),
        )
        assert bkd.allclose(coef, bkd.full((1,), 8))

        if not package_available("cvxopt"):
            return

        quantile = 0.8
        solver = QuantileRegressionCVXOPTSolver(quantile, backend=bkd)
        solver.set_options(solver.default_options())
        basis_mat = bkd.array([1, 1, 1, 1, 1])[:, None]
        values = bkd.array([2, 4, 6, 8, 10])[:, None]
        # Residuals are
        # [2, 4, 6, 8, 10] - [1, 1, 1, 1, 1] * 8 = [-6, -4, -2, 0, 2]
        # 0.8 quantile residual is zero
        coef = solver.solve(basis_mat, values)
        assert bkd.allclose(coef, bkd.full((1,), 8))

    def test_entropic_regression(self):
        bkd = self.get_backend()
        nsamples, nvars = 100, 3
        basis_mat = bkd.asarray(np.random.normal(0, 1, (nsamples, nvars)))
        # must make first column (basis) a constant
        basis_mat[:, 0] = 1.0
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
        solver.set_optimizer(solver.default_optimizer(gtol=1e-12))
        coef = solver.solve(basis_mat, train_values)
        # Assert statistic of risk quadrangle is zero. I.e 0.8 qunatile
        # is zero.
        residual = train_values - basis_mat @ coef
        # The stat of the entropic risk quadrangle is also
        # the risk measure
        stat = EntropicRisk(1.0, backend=bkd)
        stat.set_samples(residual.T)
        assert bkd.allclose(stat(), bkd.zeros((1,)))

    def _check_conservative_surrogate(self, solver):
        bkd = self.get_backend()
        nsamples, nvars = 100, 2
        basis_mat = bkd.asarray(np.random.normal(0, 1, (nsamples, nvars)))
        # must set first column to ones to mimic constant term
        basis_mat[:, 0] = 1.0
        true_coefs = bkd.ones((nvars, 1))
        noiseless_train_values = basis_mat @ true_coefs
        noise_std = 0.1
        ntrials = 500
        for ii in range(ntrials):
            noise = bkd.asarray(
                np.random.normal(0, noise_std, noiseless_train_values.shape)
            )
            train_values = noiseless_train_values + noise
            coef = solver.solve(basis_mat, train_values)
            solver.risk_measure().set_samples((basis_mat @ coef).T)
            risk_value = solver.risk_measure()()
            solver.risk_measure().set_samples(train_values.T)
            data_risk_value = solver.risk_measure()()
            assert risk_value >= data_risk_value, solver

    def test_conservative_surrogates(self):
        bkd = self.get_backend()
        test_cases = [
            ConservativeLstSqSolver(strength=1.0, backend=bkd),
            ConservativeQuantileRegressionSolver(quantile=0.5, backend=bkd),
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_conservative_surrogate(test_case)

    def _setup_linear_regression_problem(self, nsamples, degree):
        bkd = self.get_backend()

        # Setup surrogate to train
        basis = MultiIndexBasis([Monomial1D(backend=bkd)])
        basis.set_tensor_product_indices([degree + 1])
        bexp = MonomialExpansion(
            basis, solver=LstSqSolver(backend=bkd), nqoi=1
        )

        # generate training data
        train_samples = bkd.linspace(1e-3, 2 + 1e-3, nsamples)[None, :]
        train_values = bkd.exp(train_samples).T
        probabilities = bkd.full(train_values.shape, 1.0 / nsamples)

        # Compute conservative least squares solution to use as
        # an initial guess for the stochastic dominance
        bexp.fit(train_samples, train_values)
        shift = (train_values - bexp(train_samples)).max()
        coef = bexp.get_coefficients()
        coef[0] += shift
        return bexp, probabilities, train_samples, train_values

    def _check_first_order_stochastic_dominance_gradients(
        self, nsamples, degree
    ):
        bkd = self.get_backend()
        bexp, probabilities, train_samples, train_values = (
            self._setup_linear_regression_problem(nsamples, degree)
        )
        solver = FSDRegressionSolver(
            train_samples.shape[1],
            SmoothLogBasedLeftHeavisideFunction(2, eps=5e-1, backend=bkd),
        )
        solver.set_optimizer(solver.default_optimizer())
        # set training data so we can check gradients of objective and
        # constraints. This is usually set by bexp.fit()
        bexp._set_training_data(train_samples, train_values)
        solver.set_surrogate(bexp)
        solver._setup_optimizer()
        errors = solver._optimizer._objective.check_apply_jacobian(
            bexp.get_coefficients()
        )
        assert errors.min() / errors.max() < 1e-6

        errors = solver._optimizer._objective.check_apply_hessian(
            bexp.get_coefficients()
        )
        assert errors.min() / errors.max() < 1e-6

        errors = solver._optimizer._raw_constraints[0].check_apply_jacobian(
            bexp.get_coefficients(), disp=True
        )
        assert errors.min() / errors.max() < 1e-6

        errors = solver._optimizer._raw_constraints[0].check_apply_hessian(
            bexp.get_coefficients(),
            weights=bkd.array(
                np.random.uniform(0, 1, (train_samples.shape[1], 1))
            ),
        )
        assert errors.min() / errors.max() < 1e-6

    def test_first_order_stochastic_dominance_gradients(self):
        self._check_first_order_stochastic_dominance_gradients(10, 1)
        self._check_first_order_stochastic_dominance_gradients(10, 2)

    def test_first_order_stochastic_dominance_regression(self):
        bkd = self.get_backend()
        nsamples = 10
        degree = 3
        bexp, probabilities, train_samples, train_values = (
            self._setup_linear_regression_problem(nsamples, degree)
        )
        std_vals = bkd.std(train_values)
        train_values = (
            train_values / std_vals
        )  # hack to be consistent with old test
        solver = FSDRegressionSolver(
            train_samples.shape[1],
            SmoothLogBasedLeftHeavisideFunction(
                2, eps=5e-2, shift=5e-2, backend=bkd
            ),
        )
        solver.set_iterate(bexp.get_coefficients())
        bexp.set_solver(solver)
        solver.set_optimizer(
            solver.default_optimizer(verbosity=0, maxiter=1000)
        )
        bexp.fit(train_samples, train_values)
        coef = bexp.get_coefficients() * std_vals

        # Run regression test
        # TODO replace with a unit test, e.g. that surrogate is
        # conservative with respect to data for all error measures
        ref_coef = np.array(
            [0.9854551809, 1.2536694273, -0.0148611493, 0.4947820513]
        )[:, None]
        assert np.allclose(coef, ref_coef)

    def test_second_order_stochastic_dominance_regression(self):
        bkd = self.get_backend()
        nsamples = 10
        degree = 3
        bexp, probabilities, train_samples, train_values = (
            self._setup_linear_regression_problem(nsamples, degree)
        )
        std_vals = bkd.std(train_values)
        train_values = (
            train_values / std_vals
        )  # hack to be consistent with old test
        solver = SSDRegressionSolver(
            train_samples.shape[1],
            SmoothLogBasedMaxFunction(2, eps=5e-2, shift=0, backend=bkd),
        )
        solver.set_iterate(bexp.get_coefficients())
        bexp.set_solver(solver)
        solver.set_optimizer(
            solver.default_optimizer(verbosity=0, maxiter=1000)
        )
        bexp.fit(train_samples, train_values)x
        coef = bexp.get_coefficients() * std_vals

        # TODO check answer consistent with old code and
        # copy the old unit test
        raise NotImplementedError


class TestNumpyLinearSolvers(TestLinearSolvers, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchLinearSolvers(TestLinearSolvers, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
