import unittest
import numpy as np
from scipy import stats
from functools import partial

from pyapprox.variables import get_probability_masses, float_rv_discrete,\
    is_bounded_continuous_variable, transform_scale_parameters
from pyapprox.univariate_polynomials.recursion_factory import \
    get_recursion_coefficients_from_variable
from pyapprox.univariate_polynomials.numeric_orthonormal_recursions import \
    ortho_polynomial_grammian_bounded_continuous_variable
from pyapprox.univariate_polynomials.orthonormal_polynomials import \
    evaluate_orthonormal_polynomial_1d
from pyapprox.utilities import \
    integrate_using_univariate_gauss_legendre_quadrature_unbounded


class TestRecursionFactory(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_get_recursion_coefficients_from_variable_discrete(self):
        degree = 4
        N = 10
        scipy_discrete_var_names = [
            n for n in stats._discrete_distns._distn_names]
        discrete_var_names = [
            "binom", "bernoulli", "nbinom", "geom", "hypergeom", "logser",
            "poisson", "planck", "boltzmann", "randint", "zipf",
            "dlaplace", "skellam", "yulesimon"]
        # valid shape parameters for each distribution in names
        # there is a one to one correspondence between entries
        discrete_var_shapes = [
            {"n": 10, "p": 0.5}, {"p": 0.5}, {"n": 10, "p": 0.5}, {"p": 0.5},
            {"M": 20, "n": 7, "N": 12}, {"p": 0.5}, {"mu": 1}, {"lambda_": 1},
            {"lambda_": 2, "N": 10}, {"low": 0, "high": 10}, {"a": 2},
            {"a": 1}, {"mu1": 1, "mu2": 3}, {"alpha": 1}]

        for name in scipy_discrete_var_names:
            assert name in discrete_var_names

        # do not support :
        #    yulesimon as there is a bug when interval is called
        #       from a frozen variable
        #    bernoulli which only has two masses
        #    zipf unusual distribution and difficult to compute basis
        #    crystallball is discontinuous and requires special integrator
        #        this can be developed if needed
        unsupported_discrete_var_names = ["bernoulli", "yulesimon", "zipf"]
        for name in unsupported_discrete_var_names:
            ii = discrete_var_names.index(name)
            del discrete_var_names[ii]
            del discrete_var_shapes[ii]

        for name, shapes in zip(discrete_var_names, discrete_var_shapes):
            # print(name)
            var = getattr(stats, name)(**shapes)
            xk, pk = get_probability_masses(var, 1e-15)
            loc, scale = transform_scale_parameters(var)
            xk = (xk-loc)/scale
            ab = get_recursion_coefficients_from_variable(
                var, degree+1, {"orthonormality_tol": 3e-14,
                                "truncated_probability_tol": 1e-15,
                                "numeric": False})
            basis_mat = evaluate_orthonormal_polynomial_1d(xk, degree, ab)
            gram_mat = (basis_mat*pk[:, None]).T.dot(basis_mat)
            assert np.allclose(gram_mat, np.eye(basis_mat.shape[1]), atol=2e-8)

        # custom discrete variables
        xk1, pk1 = np.arange(N), np.ones(N)/N
        xk2, pk2 = np.arange(N)**2, np.ones(N)/N
        custom_vars = [
            float_rv_discrete(name="discrete_chebyshev", values=(xk1, pk1))(),
            float_rv_discrete(name="float_rv_discrete", values=(xk2, pk2))()]
        for var in custom_vars:
            xk, pk = get_probability_masses(var, 1e-15)
            loc, scale = transform_scale_parameters(var)
            xk = (xk-loc)/scale
            ab = get_recursion_coefficients_from_variable(
                var, degree+1, {"orthonormality_tol": 1e-14,
                                "truncated_probability_tol": 1e-15})
            basis_mat = evaluate_orthonormal_polynomial_1d(xk, degree, ab)
            gram_mat = (basis_mat*pk[:, None]).T.dot(basis_mat)
            assert np.allclose(gram_mat, np.eye(basis_mat.shape[1]), atol=2e-8)

    def test_get_recursion_coefficients_from_variable_continuous(self):
        scipy_continuous_var_names = [
            n for n in stats._continuous_distns._distn_names]
        continuous_var_names = [
            "ksone", "kstwobign", "norm", "alpha", "anglit", "arcsine", "beta",
            "betaprime", "bradford", "burr", "burr12", "fisk", "cauchy", "chi",
            "chi2", "cosine", "dgamma", "dweibull", "expon", "exponnorm",
            "exponweib", "exponpow", "fatiguelife", "foldcauchy", "f",
            "foldnorm", "weibull_min", "weibull_max", "frechet_r", "frechet_l",
            "genlogistic", "genpareto", "genexpon", "genextreme", "gamma",
            "erlang", "gengamma", "genhalflogistic", "gompertz", "gumbel_r",
            "gumbel_l", "halfcauchy", "halflogistic", "halfnorm", "hypsecant",
            "gausshyper", "invgamma", "invgauss", "norminvgauss", "invweibull",
            "johnsonsb", "johnsonsu", "laplace", "levy", "levy_l",
            "levy_stable", "logistic", "loggamma", "loglaplace", "lognorm",
            "gilbrat", "maxwell", "mielke", "kappa4", "kappa3", "moyal",
            "nakagami", "ncx2", "ncf", "t", "nct", "pareto", "lomax",
            "pearson3", "powerlaw", "powerlognorm", "powernorm", "rdist",
            "rayleigh", "reciprocal", "rice", "recipinvgauss", "semicircular",
            "skewnorm", "trapz", "triang", "truncexpon", "truncnorm",
            "tukeylambda", "uniform", "vonmises", "vonmises_line", "wald",
            "wrapcauchy", "gennorm", "halfgennorm", "crystalball", "argus"]

        continuous_var_shapes = [
            {"n": int(1e3)}, {}, {}, {"a": 1}, {}, {}, {"a": 2, "b": 3},
            {"a": 2, "b": 3}, {"c": 2}, {"c": 2, "d": 1},
            {"c": 2, "d": 1}, {"c": 3}, {}, {"df": 10}, {"df": 10},
            {}, {"a": 3}, {"c": 3}, {}, {"K": 2}, {"a": 2, "c": 3}, {"b": 3},
            {"c": 3}, {"c": 3}, {"dfn": 1, "dfd": 1}, {"c": 1}, {"c": 1},
            {"c": 1}, {"c": 1}, {"c": 1}, {"c": 1}, {"c": 1},
            {"a": 2, "b": 3, "c": 1}, {"c": 1}, {"a": 2}, {"a": 2},
            {"a": 2, "c": 1}, {"c": 1}, {"c": 1}, {}, {}, {}, {}, {}, {},
            {"a": 2, "b": 3, "c": 1, "z": 1}, {"a": 1}, {"mu": 1},
            {"a": 2, "b": 1}, {"c": 1}, {"a": 2, "b": 1}, {"a": 2, "b": 1},
            {}, {}, {}, {"alpha": 1, "beta": 1}, {}, {"c": 1}, {"c": 1},
            {"s": 1}, {}, {}, {"k": 1, "s": 1}, {"h": 1, "k": 1}, {"a": 1}, {},
            {"nu": 1}, {"df": 10, "nc": 1}, {"dfn": 10, "dfd": 10, "nc": 1},
            {"df": 10}, {"df": 10, "nc": 1}, {"b": 2}, {"c": 2}, {"skew": 2},
            {"a": 1}, {"c": 2, "s": 1}, {"c": 2}, {"c": 2}, {},
            {"a": 2, "b": 3}, {"b": 2}, {"mu": 2}, {}, {"a": 1},
            {"c": 0, "d": 1}, {"c": 1}, {"b": 2}, {"a": 2, "b": 3}, {"lam": 2},
            {}, {"kappa": 2}, {"kappa": 2}, {}, {"c": 0.5}, {"beta": 2},
            {"beta": 2}, {"beta": 2, "m": 2}, {"chi": 1}]

        for name in scipy_continuous_var_names:
            assert name in continuous_var_names

        def custom_integrate_fun(interval_size, lb, ub, integrand):
            # this funciton works well for smooth unbounded variables
            # but scipy.integrate.quad works well for non smooth
            # variables
            val = \
                integrate_using_univariate_gauss_legendre_quadrature_unbounded(
                    integrand, lb, ub, 50, interval_size=interval_size,
                    verbose=0)
            return val

        # do not support :
        #    levy_stable as there is a bug when interval is called
        #       from a frozen variable
        #    vonmises a circular distribution
        unsupported_continuous_var_names = ["levy_stable", "vonmises"]
        # The following are too difficult without better integration algorithm
        unsupported_continuous_var_names += [
            "f", "levy", "levy_l", "loglaplace", "ncf", "crystalball"]

        # The following variables have fat tails and cause
        # scipy.integrate.quad to fail. Use custom integrator for these
        fat_tail_continuous_var_names = [
            "alpha", "betaprime", "burr", "burr12", "fisk", "cauchy",
            "foldcauchy", "f", "genpareto", "halfcauchy", "invgamma",
            "invweibull", "levy", "levy_l", "loglaplace", "mielke",
            "kappa3", "ncf", "pareto", "lomax", "pearson3"]
        for name in unsupported_continuous_var_names:
            ii = continuous_var_names.index(name)
            del continuous_var_names[ii]
            del continuous_var_shapes[ii]

        # start at a midpoint in the list
        # name = "argus"
        # index = continuous_var_names.index(name)
        # shapes = continuous_var_shapes[index]
        # continuous_var_names = continuous_var_names[index:]
        # continuous_var_shapes = continuous_var_shapes[index:]

        degree = 2
        for name, shapes in zip(
                continuous_var_names, continuous_var_shapes):
            print(name, shapes)
            if name == "levy_l":
                loc = -2
            else:
                loc = 2
            var = getattr(stats, name)(**shapes, loc=loc, scale=3)
            # print(var, var.interval(1))
            tol = 1e-8
            if name not in fat_tail_continuous_var_names:
                quad_opts = {"epsrel": tol, "epsabs": tol, "limit": 100}
                integrate_fun = None
            else:
                interval_size = abs(np.diff(var.interval(0.99)))
                integrate_fun = partial(custom_integrate_fun, interval_size)
                quad_opts = {"integrate_fun": integrate_fun}
            opts = {"numeric": True, "quad_options": quad_opts}
            ab = get_recursion_coefficients_from_variable(var, degree+1, opts)
            gram_mat = ortho_polynomial_grammian_bounded_continuous_variable(
                var, ab, degree, tol, integrate_fun)
            assert np.allclose(
                gram_mat, np.eye(gram_mat.shape[1]), atol=tol*300)
            # plot in log space shows fat tails
            # lb, ub = var.interval(0.999)
            # xx = np.linspace(lb, ub)
            # print(var.interval(1))
            # import matplotlib.pyplot as plt
            # plt.semilogy(xx, var.pdf(xx))
            # plt.show()

if __name__ == "__main__":
    recursion_factory_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestRecursionFactory)
    unittest.TextTestRunner(verbosity=2).run(recursion_factory_test_suite)