import unittest

from scipy import stats
import numpy as np

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.integrate import integrate
from pyapprox.surrogates.polychaos.gpc import get_polynomial_from_variable
from pyapprox.surrogates.interp.indexing import (
    tensor_product_indices)

from pyapprox.sciml.linearoplearning import HilbertSchmidtLinearOperator
from pyapprox.sciml.kernels import (
    HilbertSchmidtKernel, PCEHilbertSchmidtBasis1D)
from pyapprox.sciml.util._torch_wrappers import asarray


class TestLinearOperatorLearning(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    @staticmethod
    def _eval_1d_kernel_in_function_form(kernel, samples):
        return np.array([kernel(sample[:1, None], sample[1:2, None])[0, 0]
                         for sample in samples.T])[:, None]

    def test_recover_hilbert_schmidt_coeffs_using_function_approximation(self):
        degree = 2
        marginal_variable = stats.uniform(-1, 2)
        basis = PCEHilbertSchmidtBasis1D(marginal_variable, degree)
        kernel = HilbertSchmidtKernel(basis, 0, [-np.inf, np.inf])
        A = np.random.normal(
            0, 1, (basis.nterms(), basis.nterms()))
        kernel.hyp_list.set_active_opt_params(asarray((A@A.T).flatten()))

        # recover coefficients using least squares for function approximation
        # by treating kernel as a two-dimensional scalar valued function
        variable_2d = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)]*2)
        poly = get_polynomial_from_variable(variable_2d)
        poly.set_indices(
            tensor_product_indices([degree]*variable_2d.num_vars()))
        quad_samples = integrate(
            "tensorproduct", variable_2d,
            levels=[degree+10]*variable_2d.num_vars())[0].copy()
        kernel_vals = self._eval_1d_kernel_in_function_form(
            kernel, quad_samples)
        coef = np.linalg.lstsq(
            poly.basis_matrix(quad_samples), kernel_vals, rcond=None)[0]
        kernel_coef = kernel._get_weights()
        coef = coef.reshape(kernel_coef.shape)
        assert np.allclose(coef, kernel_coef)

    @staticmethod
    def _generate_random_functions(coefs, basis, xx):
        basis_mat = basis(xx)
        return basis_mat @ coefs

    @staticmethod
    def _generate_output_functions(
            kernel, in_quadrule, in_fun_values, out_points):
        quad_x, quad_w = in_quadrule
        Kmat = kernel(out_points, quad_x)
        # keep below to show what eisum is doing
        # nout_dof = out_points.shape[1]
        # nsamples = in_fun_values.shape[1]
        # values = np.empty((nout_dof, nsamples))
        # for ii in range(nsamples):
        #     values[:, ii] = (Kmat * in_fun_values[:, ii]) @ quad_w[:, 0]
        values = np.einsum("ij,jk->ik", Kmat, quad_w*in_fun_values)
        return values

    def test_gaussian_measure_over_1D_functions(self):
        kernel_degree = 2
        marginal_variable = stats.uniform(-1, 2)
        basis = PCEHilbertSchmidtBasis1D(marginal_variable, kernel_degree)
        linearop = HilbertSchmidtLinearOperator(basis)
        kernel = HilbertSchmidtKernel(basis, 0, [-np.inf, np.inf])
        A = np.random.normal(
            0, 1, (basis.nterms(), basis.nterms()))
        kernel.hyp_list.set_active_opt_params(asarray((A@A.T).flatten()))

        # generate training functions as random draws from Gaussian
        # measure on polynomial functions
        # use Monte Carlo
        # nsamples = 100
        # train_coefs = np.random.normal(
        #     0, 1, (kernel._inbasis_nterms, nsamples))
        # out_weights = np.full((nsamples, 1), 1/nsamples)
        # Use quadrature
        coef_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*(kernel_degree+1))
        train_coefs, out_weights = integrate(
            "tensorproduct", coef_variable,
            levels=[kernel_degree+3]*coef_variable.num_vars())

        train_in_values = self._generate_random_functions(
            train_coefs, basis, basis.quadrature_rule()[0])
        train_out_values = self._generate_output_functions(
            kernel, basis.quadrature_rule(), train_in_values,
            basis.quadrature_rule()[0])

        basis_mat = linearop._basis_matrix(
            basis.quadrature_rule()[0], train_in_values)
        gram_mat = linearop._gram_matrix(basis_mat, out_weights)
        np.set_printoptions(linewidth=1000)
        assert np.allclose(gram_mat, np.eye(gram_mat.shape[0]))

        linearop._set_coefficients(kernel._get_weights().flatten()[:, None])

        linearop.fit(train_in_values, train_out_values, out_weights)
        # print(linearop._coef[:, 0])
        # print(kernel._coef.flatten())
        assert np.allclose(
            linearop._hyp_list.get_values(), kernel._get_weights().flatten())

        plot_xx = np.linspace(-1, 1, 101)[None, :]
        # check approximation on training funciton
        # idx = [10]
        # in_coef = train_coefs[:, idx]
        # check approximation at unseen function
        in_coef = np.random.normal(0, 1, (kernel_degree+1, 1))

        infun_values = self._generate_random_functions(
            in_coef, basis, basis.quadrature_rule()[0])
        plot_out_values = self._generate_output_functions(
            kernel, basis.quadrature_rule(), infun_values, plot_xx)
        assert np.allclose(linearop(infun_values, plot_xx), plot_out_values)

        import matplotlib.pyplot as plt
        plt.plot(plot_xx[0], plot_out_values, label="Exact")
        plt.plot(plot_xx[0], linearop(infun_values, plot_xx), '--',
                 label="Approx")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
