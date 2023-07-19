import unittest
import numpy as np
from scipy import stats
from warnings import warn

from pyapprox.variables.marginals import (
    get_distribution_info,
    float_rv_discrete, variables_equivalent, get_pdf
)


class TestMarginals(unittest.TestCase):
    def test_get_distribution_params(self):
        name, scales, shapes = get_distribution_info(
            stats.beta(a=1, b=2, loc=0, scale=1))
        assert name == 'beta'
        assert shapes == {'a': 1, 'b': 2}
        assert scales == {'loc': 0, 'scale': 1}

        rv = stats.beta(a=1, b=2, loc=3, scale=4)
        pdf = get_pdf(rv)
        xx = rv.rvs(100)
        assert np.allclose(pdf(xx), rv.pdf(xx))

        name, scales, shapes = get_distribution_info(
            stats.beta(1, 2, loc=0, scale=1))
        assert name == 'beta'
        assert shapes == {'a': 1, 'b': 2}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(
            stats.beta(1, 2, 0, scale=1))
        assert name == 'beta'
        assert shapes == {'a': 1, 'b': 2}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.beta(1, 2, 0, 1))
        assert name == 'beta'
        assert shapes == {'a': 1, 'b': 2}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.norm(0, 1))
        assert name == 'norm'
        assert shapes == dict()
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.norm(0, scale=1))
        assert name == 'norm'
        assert shapes == dict()
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.norm(
            loc=0, scale=1))
        assert name == 'norm'
        assert shapes == dict()
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(
            stats.gamma(a=1, loc=0, scale=1))
        assert name == 'gamma'
        assert shapes == {'a': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(
            stats.gamma(1, loc=0, scale=1))
        assert name == 'gamma'
        assert shapes == {'a': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(
            stats.gamma(1, 0, scale=1))
        assert name == 'gamma'
        assert shapes == {'a': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.gamma(1, 0, 1))
        assert name == 'gamma'
        assert shapes == {'a': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.gamma(1))
        assert name == 'gamma'
        assert shapes == {'a': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.gamma(1, loc=0))
        assert name == 'gamma'
        assert shapes == {'a': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.gamma(1, scale=1))
        assert name == 'gamma'
        assert shapes == {'a': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(
            stats.binom(n=1, p=1, loc=0))
        assert name == 'binom'
        assert shapes == {'n': 1, 'p': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(
            stats.binom(1, p=1, loc=0))
        assert name == 'binom'
        assert shapes == {'n': 1, 'p': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.binom(1, 1, loc=0))
        assert name == 'binom'
        assert shapes == {'n': 1, 'p': 1}
        assert scales == {'loc': 0, 'scale': 1}

        name, scales, shapes = get_distribution_info(stats.binom(1, 1, 0))
        assert name == 'binom'
        assert shapes == {'n': 1, 'p': 1}
        assert scales == {'loc': 0, 'scale': 1}

    def test_get_pdf(self):
        rv = stats.beta(a=1, b=2, loc=3, scale=4)
        pdf = get_pdf(rv)
        xx = rv.rvs(100)
        assert np.allclose(pdf(xx), rv.pdf(xx))

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
            if name not in continuous_var_names:
                warn(f"variable {name} is not tested", UserWarning)

        unsupported_continuous_var_names = ["ncf"]

        unsupported_continuous_var_names += [
            n for n in continuous_var_names
            if n not in scipy_continuous_var_names]

        for name in unsupported_continuous_var_names:
            ii = continuous_var_names.index(name)
            del continuous_var_names[ii]
            del continuous_var_shapes[ii]

        for name, shapes in zip(
                continuous_var_names, continuous_var_shapes):
            if name == "levy_l":
                loc = -2
            else:
                loc = 2
            print(name, shapes)
            var = getattr(stats, name)(**shapes, loc=loc, scale=3)
            pdf = get_pdf(var)
            xx = var.rvs(100)
            assert np.allclose(pdf(xx), var.pdf(xx))

    def test_float_discrete_variable(self):
        nmasses1 = 10
        mass_locations1 = np.geomspace(1.0, 32.0, num=nmasses1)
        masses1 = np.ones(nmasses1, dtype=float)/nmasses1
        var1 = float_rv_discrete(
            name='var1', values=(mass_locations1, masses1))()

        for power in [1, 2, 3]:
            assert np.allclose(
                var1.moment(power), (mass_locations1**power).dot(masses1))

        np.random.seed(1)
        num_samples = int(1e6)
        samples = var1.rvs(size=(1, num_samples))
        assert np.allclose(samples.mean(), var1.moment(1), atol=1e-2)

        # import matplotlib.pyplot as plt
        # xx = np.linspace(0,33,301)
        # plt.plot(mass_locations1,np.cumsum(masses1),'rss')
        # plt.plot(xx,var1.cdf(xx),'-'); plt.show()
        assert np.allclose(np.cumsum(masses1), var1.cdf(mass_locations1))

        # import matplotlib.pyplot as plt
        # yy = np.linspace(0,1,51)
        # plt.plot(mass_locations1,np.cumsum(masses1),'rs')
        # plt.plot(var1.ppf(yy),yy,'-o',ms=2); plt.show()
        xx = mass_locations1
        assert np.allclose(xx, var1.ppf(var1.cdf(xx)))

        xx = mass_locations1
        assert np.allclose(xx, var1.ppf(var1.cdf(xx+1e-1)))

    def test_float_rv_discrete_pdf(self):
        nmasses1 = 10
        mass_locations1 = np.geomspace(1.0, 32.0, num=nmasses1)
        masses1 = np.ones(nmasses1, dtype=float)/nmasses1
        var1 = float_rv_discrete(
            name='var1', values=(mass_locations1, masses1))()

        xk = var1.dist.xk.copy()
        II = np.random.permutation(xk.shape[0])[:3]
        xk[II] = -np.arange(1, 4)
        print(xk)
        pdf_vals = var1.pmf(xk)
        assert np.allclose(pdf_vals[II], np.zeros_like(II, dtype=float))
        print(pdf_vals, II, var1.dist.pk)
        assert np.allclose(
            np.delete(pdf_vals, II), np.delete(var1.dist.pk, II))

    def test_variables_equivalent(self):
        nmasses = 10
        xk = np.array(range(nmasses), dtype='float')
        pk = np.ones(nmasses)/nmasses
        xk2 = np.array(range(nmasses), dtype='float')
        # pk2 = np.ones(nmasses)/(nmasses)
        pk2 = np.geomspace(1.0, 512.0, num=nmasses)
        pk2 /= pk2.sum()
        var1 = float_rv_discrete(
            name='float_rv_discrete', values=(xk, pk))()
        var2 = float_rv_discrete(
            name='float_rv_discrete', values=(xk2, pk2))()
        assert not variables_equivalent(var1, var2)


if __name__ == "__main__":
    marginals_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMarginals)
    unittest.TextTestRunner(verbosity=2).run(marginals_test_suite)
