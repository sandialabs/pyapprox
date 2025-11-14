import unittest
from functools import partial

import numpy as np
from scipy import stats

from pyapprox.variables.marginals import (
    BetaMarginal,
    GammaMarginal,
    GaussianMarginal,
    ContinuousScipyMarginal,
    CustomDiscreteMarginal,
    DiscreteScipyMarginal,
    gaussian_pdf,
    gaussian_pdf_derivative,
    EmpiricalCDF,
    beta_pdf_on_ab,
    pdf_derivative_under_affine_map,
    DiscreteChebyshevMarginal,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.interface.finitedifference import ForwardFiniteDifference


class TestContinuousMarginals:
    def setUp(self):
        np.random.seed(1)

    def _check_continuous_scipy_marginal(self, scipy_rv):
        bkd = self.get_backend()
        marginal = ContinuousScipyMarginal(scipy_rv, bkd)
        nsamples = 3
        samples = scipy_rv.rvs(nsamples)
        usamples = bkd.linspace(0, 1, 5)
        assert bkd.allclose(
            bkd.asarray(marginal.median()), bkd.asarray([scipy_rv.median()])
        )

        if np.isfinite(scipy_rv.var()):
            # some variables do not have finite mean
            assert bkd.allclose(
                bkd.asarray(marginal.mean()), bkd.asarray([scipy_rv.mean()])
            )
        if np.isfinite(scipy_rv.var()):
            # some variables do not have finite variance
            assert bkd.allclose(
                bkd.asarray(marginal.std()), bkd.asarray([scipy_rv.std()])
            )
            assert bkd.allclose(
                bkd.asarray(marginal.var()), bkd.asarray([scipy_rv.var()])
            )
        assert bkd.allclose(
            marginal.pdf(samples), bkd.asarray(scipy_rv.pdf(samples))
        )
        assert bkd.allclose(
            marginal.logpdf(samples), bkd.asarray(scipy_rv.logpdf(samples))
        )
        assert bkd.allclose(
            marginal.cdf(samples), bkd.asarray(scipy_rv.cdf(samples))
        )
        assert bkd.allclose(
            marginal.ppf(usamples), bkd.asarray(scipy_rv.ppf(usamples))
        )
        # tests plots run
        # axs = plt.subplots(1, 3, figsize=(3 * 8, 6))[1]
        # marginal.plot_pdf(axs[0])
        # marginal.plot_cdf(axs[1])
        # marginal.plot_ppf(axs[2])
        # plt.close()

    def test_continuous_scipy_marginals(self):
        scipy_rvs = []

        scipy_continuous_marginal_names = [
            n for n in stats._continuous_distns._distn_names
        ]
        # fmt: off
        continuous_marginal_names = [
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
            "gibrat", "maxwell", "mielke", "kappa4", "kappa3", "moyal",
            "nakagami", "ncx2", "ncf", "t", "nct", "pareto", "lomax",
            "pearson3", "powerlaw",
            "powerlognorm", "powernorm", "rdist",
            "rayleigh", "reciprocal", "rice", "recipinvgauss", "semicircular",
            "skewnorm", "trapezoid", "triang", "truncexpon", "truncnorm",
            "tukeylambda", "uniform", "vonmises", "vonmises_line", "wald",
            "wrapcauchy", "gennorm", "halfgennorm", "crystalball", "argus"]

        continuous_marginal_shapes = [
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

        # for name in scipy_continuous_marginal_names:
        #     if name not in continuous_marginal_names:
        #         warn(f"variable {name} is not tested", UserWarning, stacklevel=2)

        unsupported_continuous_marginal_names = ["ncf"]

        unsupported_continuous_marginal_names += [
            n
            for n in continuous_marginal_names
            if n not in scipy_continuous_marginal_names
        ]

        for name in unsupported_continuous_marginal_names:
            ii = continuous_marginal_names.index(name)
            del continuous_marginal_names[ii]
            del continuous_marginal_shapes[ii]

        loc = 2
        scipy_rvs = []
        for name, shapes in zip(
            continuous_marginal_names, continuous_marginal_shapes
        ):
            scipy_rvs.append(getattr(stats, name)(**shapes, loc=loc, scale=3))

        for scipy_rv in scipy_rvs:
            self._check_continuous_scipy_marginal(scipy_rv)

    def test_variable_equivalence(self):
        bkd = self.get_backend()
        # continuous scipy marginal different bounds
        marginal1 = ContinuousScipyMarginal(
            stats.beta(1, 2, 0, 1), backend=bkd
        )
        marginal2 = ContinuousScipyMarginal(
            stats.beta(1, 2, 0, 2), backend=bkd
        )
        assert marginal1 == marginal1
        assert marginal1 != marginal2
        assert marginal1.is_bounded()

        # continuous scipy marginal different shapes
        marginal1 = ContinuousScipyMarginal(
            stats.beta(3, 2, 0, 1), backend=bkd
        )
        marginal2 = ContinuousScipyMarginal(
            stats.beta(1, 2, 0, 1), backend=bkd
        )
        assert marginal1 == marginal1
        assert marginal1 != marginal2

    def test_gaussian(self):
        bkd = self.get_backend()
        mean, stdev = 1.0, 2.0
        scipy_rv = stats.norm(mean, stdev)
        marginal = GaussianMarginal(mean, stdev, backend=bkd)
        assert not marginal.is_bounded()
        samples = marginal.rvs(5)
        assert bkd.allclose(
            bkd.asarray([marginal.mean()]), bkd.asarray([scipy_rv.mean()])
        )
        assert bkd.allclose(
            bkd.asarray(marginal.median()), bkd.asarray([scipy_rv.median()])
        )
        assert bkd.allclose(
            bkd.asarray([marginal.std()]), bkd.asarray([scipy_rv.std()])
        )
        assert bkd.allclose(
            bkd.asarray([marginal.var()]), bkd.asarray([scipy_rv.var()])
        )
        assert bkd.allclose(
            bkd.asarray(marginal.pdf(samples)),
            bkd.asarray(scipy_rv.pdf(samples)),
        )
        assert bkd.allclose(
            marginal.pdf(samples), bkd.asarray(scipy_rv.pdf(samples))
        )
        assert bkd.allclose(
            marginal.pdf(samples), gaussian_pdf(mean, stdev**2, samples, bkd)
        )
        # cdf vals differ to scipy for noninteger shape params
        assert bkd.allclose(
            marginal.cdf(samples), bkd.asarray(scipy_rv.cdf(samples))
        )
        usamples = bkd.linspace(0, 1, 50)
        assert bkd.allclose(
            marginal.ppf(usamples), bkd.asarray(scipy_rv.ppf(usamples))
        )
        assert bkd.allclose(
            bkd.mean(marginal.rvs(int(1e6))),
            bkd.asarray([scipy_rv.mean()]),
            rtol=1e-2,
        )

        assert bkd.allclose(
            marginal.pdf_jacobian(samples),
            gaussian_pdf_derivative(mean, stdev**2, samples, bkd),
        )

        marginal2 = GaussianMarginal(mean + 1, stdev, backend=bkd)
        assert marginal != marginal2

        marginal2 = GaussianMarginal(mean, stdev + 1, backend=bkd)
        assert marginal != marginal2

        marginal2 = GaussianMarginal(mean, stdev, backend=bkd)
        assert marginal == marginal2

        assert marginal.kl_divergence_implemented()

    def test_beta_variable(self):
        import warnings

        warnings.simplefilter("error")
        bkd = self.get_backend()
        # bounds = [-1, 2]
        bounds = [0.0, 1.0]
        # astat, bstat = bkd.asarray([2.0, 3.0])
        astat, bstat = bkd.asarray([1.1, 3.1])
        marginal = BetaMarginal(astat, bstat, *bounds, backend=bkd)
        samples = bkd.linspace(*bounds, 5)
        scipy_rv = stats.beta(astat, bstat, bounds[0], bounds[1] - bounds[0])
        assert bkd.allclose(marginal.mean(), bkd.asarray([scipy_rv.mean()]))
        assert bkd.allclose(
            bkd.asarray(marginal.median()), bkd.asarray([scipy_rv.median()])
        )
        assert bkd.allclose(marginal.std(), bkd.asarray([scipy_rv.std()]))
        assert bkd.allclose(marginal.var(), bkd.asarray([scipy_rv.var()]))
        assert bkd.allclose(
            marginal.pdf(samples), bkd.asarray(scipy_rv.pdf(samples))
        )
        # cdf vals differ to scipy for noninteger shape params
        assert bkd.allclose(
            marginal.cdf(samples), bkd.asarray(scipy_rv.cdf(samples))
        )
        usamples = bkd.linspace(0, 1, 51)
        assert bkd.allclose(
            marginal.ppf(usamples), bkd.asarray(scipy_rv.ppf(usamples))
        )
        assert bkd.allclose(
            bkd.mean(marginal.rvs(int(1e6))),
            bkd.asarray([scipy_rv.mean()]),
            rtol=1e-2,
        )

        astat2, bstat2 = bkd.asarray([3.0, 3.0])
        rv2 = BetaMarginal(astat2, bstat2, *bounds, backend=bkd)
        scale = bounds[1] - bounds[0]
        quadx, quadw = np.polynomial.legendre.leggauss(100)
        quadx_01 = bkd.asarray((quadx + 1) / 2)
        quadw_01 = bkd.asarray(quadw / 2)
        quadx = quadx_01 * scale + bounds[0]
        quadw = quadw_01 * scale
        kl_div = (
            marginal.pdf(quadx)
            * (bkd.log(marginal.pdf(quadx)) - bkd.log(rv2.pdf(quadx)))
        ) @ quadw
        assert marginal.kl_divergence_implemented()
        assert bkd.allclose(marginal.kl_divergence(rv2), kl_div)

        astat3, bstat3 = bkd.asarray([3.0, 3.0])
        bounds = [0, 2]
        rv3 = BetaMarginal(astat3, bstat3, *bounds, backend=bkd)
        samples3 = rv3.rvs(3)

        def pdf_wrap(shapes):
            rv3.set_shapes(*shapes)
            return rv3.pdf(samples3)[None, :]

        def pdf_jacobian_wrap(shapes):
            rv3.set_shapes(*shapes)
            return rv3.pdf_shape_jacobian(samples3)

        pdf_model = ModelFromSingleSampleCallable(
            samples3.shape[0],
            2,
            pdf_wrap,
            jacobian=pdf_jacobian_wrap,
            sample_ndim=1,
            backend=bkd,
        )
        shapes = bkd.array([astat3, bstat3])
        errors = pdf_model.check_apply_jacobian(shapes[:, None])
        assert errors.min() / errors.max() < 1e-6

        usamples3 = bkd.asarray(np.random.uniform(0, 1, 3))

        def ppf_wrap(param):
            rv3.set_shapes(*param)
            return rv3.ppf(usamples3)[None, :]

        def ppf_jacobian_wrap(param):
            rv3.set_shapes(*param)
            return rv3.ppf_shape_jacobian(usamples3)

        ppf_model = ModelFromSingleSampleCallable(
            usamples3.shape[0],
            2,
            ppf_wrap,
            jacobian=ppf_jacobian_wrap,
            sample_ndim=1,
            backend=bkd,
        )

        param = bkd.array([2.0, 3.0])
        errors = ppf_model.check_apply_jacobian(param[:, None])
        assert errors.min() / errors.max() < 2e-6

        # must use new marginal because rv3 astat and bstat will
        # have been changed by check_apply_jacobian
        rv4 = BetaMarginal(astat3, bstat3, *[0, 2], backend=bkd)
        bkd.assert_allclose(
            beta_pdf_on_ab(astat3, bstat3, *[0, 2], samples, bkd=bkd),
            rv4.pdf(samples),
        )

        self.assertNotEqual(rv3, rv4)
        rv5 = BetaMarginal(astat3, bstat3, *[0, 2], backend=bkd)
        self.assertEqual(rv5, rv4)

        assert rv4.is_bounded()

        nsamples = 10000
        # _rvs calles differentiable (with autograd) sampling
        # where as rvs calls scipy
        bkd.assert_allclose(
            bkd.mean(rv4._rvs(nsamples)), rv4.mean(), atol=1e-3
        )

        rv6 = BetaMarginal(1, 1, *[0, 2], backend=bkd)
        alpha = 0.5
        bkd.assert_allclose(rv6.interval(alpha), bkd.array([0.5, 1.5]))

        rv7 = BetaMarginal(2, 2, *[0, 1], backend=bkd)
        rv8 = BetaMarginal(2, 2, *[0, 2], backend=bkd)
        bkd.assert_allclose(
            pdf_derivative_under_affine_map(
                lambda x: rv7.pdf_jacobian(x)[0], 0, 2, samples
            ),
            rv8.pdf_jacobian(samples)[0],
        )

    def test_gamma_variable(self):
        bkd = self.get_backend()
        shape, rate = bkd.array([2, 3])
        marginal = GammaMarginal(shape, rate, backend=bkd)
        scipy_rv = marginal._scipy_rv
        nsamples = 10
        samples = bkd.asarray(scipy_rv.rvs(nsamples))
        assert bkd.allclose(marginal.mean(), bkd.asarray([scipy_rv.mean()]))
        assert bkd.allclose(marginal.var(), bkd.asarray([scipy_rv.var()]))
        assert bkd.allclose(marginal.std(), bkd.asarray([scipy_rv.std()]))
        assert bkd.allclose(
            marginal.cdf(samples), bkd.asarray(scipy_rv.cdf(samples))
        )
        assert bkd.allclose(
            bkd.asarray(marginal.median()), bkd.asarray([scipy_rv.median()])
        )
        usamples = bkd.linspace(0, 1 - 1e-6, 51)
        assert bkd.allclose(
            marginal.ppf(usamples), bkd.asarray(scipy_rv.ppf(usamples))
        )
        assert bkd.allclose(
            bkd.mean(marginal.rvs(int(1e6))),
            bkd.asarray([scipy_rv.mean()]),
            rtol=1e-2,
        )

        nsamples = 1000000
        samples = marginal.rvs(nsamples)
        other_shapes = bkd.array([3, 4])
        other = GammaMarginal(*other_shapes, backend=bkd)
        kl_divergence = (
            bkd.log(marginal.pdf(samples)) - bkd.log(other.pdf(samples))
        ).mean()
        assert marginal.kl_divergence_implemented()
        assert bkd.allclose(
            kl_divergence, marginal.kl_divergence(other), rtol=1e-2
        )

        self.assertNotEqual(marginal, other)
        self.assertEqual(marginal, marginal)
        assert not marginal.is_bounded()

        samples = marginal.rvs(3)
        bkd.assert_allclose(
            marginal._gammainc(samples), marginal._bkd_gammainc(samples)
        )

        if not bkd.jacobian_implemented():
            return

        # Torch gradient of cdf
        def fun(shapes):
            marginal = GammaMarginal(*bkd.asarray(shapes[:, 0]), backend=bkd)
            return marginal.ppf(usamples)[:5]

        x0 = bkd.array([3.0, 4.0])[:, None]

        model = ModelFromSingleSampleCallable(
            5, 2, fun, values_ndim=1, backend=bkd
        )
        fd = ForwardFiniteDifference(model)
        assert bkd.allclose(
            bkd.jacobian(lambda x: fun(x[:, None]), x0[:, 0]), fd.jacobian(x0)
        )


class TestDiscreteMarginals:

    def setUp(self):
        """Set up test cases with various discrete distributions."""
        np.random.seed(1)
        self.distributions = [
            ("hypergeom", stats.hypergeom(20, 7, 12)),
            ("binom", stats.binom(10, 0.5)),
            ("nbinom", stats.nbinom(10, 0.5)),
            ("geom", stats.geom(0.5)),
            ("logser", stats.logser(0.5)),
            ("poisson", stats.poisson(5)),
            ("planck", stats.planck(0.5)),
            ("zipf", stats.zipf(2)),
            ("dlaplace", stats.dlaplace(0.5)),
            ("skellam", stats.skellam(5, 10)),
            ("boltzmann", stats.boltzmann(1, 10)),
            ("randint", stats.randint(0, 10)),
        ]

    def test_check_marginal(self):
        """Test the `_check_marginal` method."""
        for name, dist in self.distributions:
            marginal = DiscreteScipyMarginal(dist, backend=self.get_backend())
            self.assertEqual(marginal._scipy_rv.dist.name, name)

    def test_pdf(self):
        """Test the `_pdf` method."""
        for name, dist in self.distributions:
            marginal = DiscreteScipyMarginal(dist, backend=self.get_backend())
            samples = self.get_backend().arange(0, 10, dtype=float)
            pdf_values = marginal._pdf(samples)
            expected_values = dist.pmf(samples)
            self.get_backend().assert_allclose(
                pdf_values, expected_values, rtol=1e-5
            )

    def test_discrete_chebyshev(self):
        bkd = self.get_backend()
        nmasses = 3
        rv = DiscreteChebyshevMarginal(nmasses, bkd)
        samples = self.get_backend().arange(0, nmasses, dtype=float)
        bkd.assert_allclose(rv.probability_masses()[0], samples)
        bkd.assert_allclose(
            rv.probability_masses()[1], bkd.full((nmasses,), 1 / nmasses)
        )

    def test_is_bounded(self):
        """Test the `is_bounded` method."""
        for name, dist in self.distributions:
            marginal = DiscreteScipyMarginal(dist, backend=self.get_backend())
            interval = dist.interval(1)
            self.assertEqual(
                marginal.is_bounded(),
                np.isfinite(interval[0]) and np.isfinite(interval[1]),
            )

    def test_probability_masses(self):
        """Test the `_probability_masses` method."""
        for name, dist in self.distributions:
            marginal = DiscreteScipyMarginal(dist, backend=self.get_backend())
            xk, pk = marginal.probability_masses()
            expected_pk = self.get_backend().asarray(dist.pmf(xk))
            self.get_backend().assert_allclose(pk, expected_pk, rtol=1e-5)

    def test_transform_scale_parameters(self):
        """Test the `_transform_scale_parameters` method."""
        for name, dist in self.distributions:
            marginal = DiscreteScipyMarginal(dist, backend=self.get_backend())
            loc, scale = marginal._transform_scale_parameters()
            xk, _ = marginal._probability_masses()
            expected_loc = (xk.min() + xk.max()) / 2
            expected_scale = xk.max() - expected_loc
            self.assertAlmostEqual(loc, expected_loc, places=5)
            self.assertAlmostEqual(scale, expected_scale, places=5)

    def test_shapes_equal(self):
        """Test the `_shapes_equal` method."""
        for name, dist in self.distributions:
            marginal1 = DiscreteScipyMarginal(dist, backend=self.get_backend())
            marginal2 = DiscreteScipyMarginal(dist, backend=self.get_backend())
            self.assertTrue(marginal1._shapes_equal(marginal2))

    def _check_discrete_scipy_marginal(self, scipy_rv):
        bkd = self.get_backend()
        marginal = DiscreteScipyMarginal(scipy_rv, bkd)
        nsamples = 3
        samples = scipy_rv.rvs(nsamples)
        usamples = bkd.linspace(0, 1, 5)
        assert bkd.allclose(
            bkd.asarray(marginal.median()), bkd.asarray([scipy_rv.median()])
        )
        assert bkd.allclose(
            bkd.asarray(marginal.mean()), bkd.asarray([scipy_rv.mean()])
        )
        assert bkd.allclose(
            bkd.asarray(marginal.std()), bkd.asarray([scipy_rv.std()])
        )
        assert bkd.allclose(
            bkd.asarray(marginal.var()), bkd.asarray([scipy_rv.var()])
        )
        assert bkd.allclose(
            marginal.pdf(samples), bkd.asarray(scipy_rv.pmf(samples))
        )
        assert bkd.allclose(
            marginal.cdf(samples), bkd.asarray(scipy_rv.cdf(samples))
        )
        assert bkd.allclose(
            marginal.ppf(usamples), bkd.asarray(scipy_rv.ppf(usamples))
        )
        # tests plots run
        # axs = plt.subplots(1, 3, figsize=(3 * 8, 6))[1]
        # marginal.plot_pdf(axs[0])
        # marginal.plot_cdf(axs[1])
        # marginal.plot_ppf(axs[2])
        # plt.close()

    def test_discrete_scipy_marginals(self):
        # fmt: off
        discrete_marginal_names = [
            "binom", "bernoulli", "nbinom", "geom", "hypergeom", "logser",
            "poisson", "planck", "boltzmann", "randint", "zipf",
            "dlaplace", "yulesimon"]
        # valid shape parameters for each distribution in names
        # there is a one to one correspondence between entries
        discrete_marginal_shapes = [
            {"n": 10, "p": 0.5}, {"p": 0.5}, {"n": 10, "p": 0.5}, {"p": 0.5},
            {"M": 20, "n": 7, "N": 12}, {"p": 0.5}, {"mu": 1}, {"lambda_": 1},
            {"lambda_": 2, "N": 10}, {"low": 0, "high": 10}, {"a": 2},
            {"a": 1}, {"alpha": 1}]
        # fmt: on

        # do not support :
        #    yulesimon as there is a bug when interval is called
        #       from a frozen variable
        #    bernoulli which only has two masses
        #    zipf unusual distribution and difficult to compute basis
        unsupported_discrete_marginal_names = [
            "bernoulli",
            "yulesimon",
            "zipf",
        ]
        for name in unsupported_discrete_marginal_names:
            ii = discrete_marginal_names.index(name)
            del discrete_marginal_names[ii]
            del discrete_marginal_shapes[ii]

        for name, shapes in zip(
            discrete_marginal_names, discrete_marginal_shapes
        ):
            scipy_rv = getattr(stats, name)(**shapes)
            self._check_discrete_scipy_marginal(scipy_rv)

    def test_integer_discrete_variable(self):
        bkd = self.get_backend()
        nmasses = 10
        xk = bkd.asarray(np.geomspace(1.0, 32.0, num=nmasses))
        pk = bkd.arange(1.0, 1 + nmasses, dtype=float)
        pk /= bkd.sum(pk)
        marginal = CustomDiscreteMarginal(xk, pk, backend=bkd)
        scipy_rv = stats.rv_discrete(
            values=(bkd.to_numpy(xk), bkd.to_numpy(pk))
        )
        for power in [1, 2, 3]:
            assert bkd.allclose(
                marginal.moment(power), bkd.asarray([scipy_rv.moment(power)])
            )

        nsamples = int(1e6)
        samples = bkd.asarray(marginal.rvs(nsamples))
        assert bkd.allclose(samples.mean(), marginal.moment(1), rtol=1e-2)
        assert bkd.allclose(
            bkd.var(samples, ddof=1), bkd.asarray([scipy_rv.var()]), rtol=1e-2
        )
        assert bkd.allclose(bkd.cumsum(pk), marginal.cdf(xk))
        assert bkd.allclose(marginal.mean(), bkd.asarray([scipy_rv.mean()]))
        assert bkd.allclose(
            marginal.median(), bkd.asarray([scipy_rv.median()])
        )
        assert bkd.allclose(marginal.std(), bkd.asarray([scipy_rv.std()]))
        assert bkd.allclose(marginal.var(), bkd.asarray([scipy_rv.var()]))
        usamples = bkd.linspace(0.01, 1, 50)
        assert bkd.allclose(
            marginal.ppf(usamples), bkd.asarray(scipy_rv.ppf(usamples))
        )

    def test_variable_equivalence(self):
        bkd = self.get_backend()
        # bounded discrete scipy marginal different shapes
        marginal1 = DiscreteScipyMarginal(
            stats.hypergeom(20, 7, 12), backend=bkd
        )
        marginal2 = DiscreteScipyMarginal(
            stats.hypergeom(20, 6, 12), backend=bkd
        )
        assert marginal1 == marginal1
        assert marginal1 != marginal2
        assert marginal1.is_bounded()

        # unbounded discrete scipy marginal different shapes
        marginal1 = DiscreteScipyMarginal(stats.geom(0.5), backend=bkd)
        marginal2 = DiscreteScipyMarginal(stats.geom(0.6), backend=bkd)
        assert not marginal1.is_bounded()
        assert marginal1 == marginal1
        assert marginal1 != marginal2

        # custom discrete variable
        nmasses = 10
        xk1 = bkd.arange(nmasses, dtype=float)
        pk1 = bkd.ones(nmasses) / nmasses
        xk2 = bkd.arange(nmasses, dtype=float)
        pk2 = bkd.asarray(np.geomspace(1.0, 512.0, num=nmasses))
        pk2 /= pk2.sum()
        marginal1 = CustomDiscreteMarginal(xk1, pk1, backend=bkd)
        marginal2 = CustomDiscreteMarginal(xk2, pk2, backend=bkd)
        assert marginal1 == marginal1
        assert marginal1 != marginal2
        assert marginal1.is_bounded()

    def test_empirical_cdf(self):
        bkd = self.get_backend()
        n, p = 10, 0.5  # Parameters for the binomial distribution
        samples = bkd.asarray(
            range(n + 1)
        )  # Possible outcomes of the binomial distribution
        rv = stats.binom(n, p)
        weights = bkd.asarray(rv.pmf(samples))  # PMF weights
        ecdf = EmpiricalCDF(samples, weights=weights, backend=bkd)
        expected_sorted_samples = bkd.sort(samples)
        expected_ecdf = bkd.asarray(rv.cdf(expected_sorted_samples))
        bkd.assert_allclose(
            ecdf._sorted_samples, expected_sorted_samples, atol=1e-12
        )
        bkd.assert_allclose(ecdf._ecdf, expected_ecdf, atol=1e-12)
        bkd.assert_allclose(
            ecdf(expected_sorted_samples), expected_ecdf, atol=1e-12
        )

    def test_custom_discrete_marginal(self):
        bkd = self.get_backend()
        n, p = 10, 0.5  # Parameters for the binomial distribution
        xk = bkd.asarray(
            range(n + 1)
        )  # Possible outcomes of the binomial distribution
        rv = stats.binom(n, p)
        pk = bkd.asarray(rv.pmf(xk))  # PMF weights
        marginal = CustomDiscreteMarginal(xk, pk, backend=bkd)
        expected_ecdf = bkd.asarray(rv.cdf(xk))
        bkd.assert_allclose(marginal.pdf(xk), pk, atol=1e-12)
        bkd.assert_allclose(marginal.cdf(xk), expected_ecdf, atol=1e-12)
        bkd.assert_allclose(marginal.probability_masses()[0], xk, atol=1e-12)
        bkd.assert_allclose(marginal.probability_masses()[1], pk, atol=1e-12)


class TestNumpyContinuousMarginals(TestContinuousMarginals, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchContinuousMarginals(TestContinuousMarginals, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


class TestNumpyDiscreteMarginals(TestDiscreteMarginals, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchDiscreteMarginals(TestDiscreteMarginals, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
