import unittest
from scipy import stats
import numpy as np

from pyapprox.variables.marginals import (
    CustomDiscreteMarginal,
    ContinuousScipyMarginal,
)
from pyapprox.variables.transforms import (
    AffineTransform,
    RosenblattTransform,
    NatafTransform,
    define_iid_random_variable_transformation,
    ComposeTransforms,
    UniformMarginalTransformation,
    DenseGaussianTransform,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.tests.test_rosenblatt_transformation import (
    rosenblatt_example_2d,
)
from pyapprox.variables._nataf import (
    gaussian_copula_compute_x_correlation_from_z_correlation,
    generate_x_samples_using_gaussian_copula,
)
from pyapprox.util.utilities import correlation_to_covariance
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


class TestVariableTransforms:
    def setUp(self):
        np.random.seed(1)

    def test_define_mixed_tensor_product_random_variable(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable
        type the distribution parameters ARE NOT the same
        """
        bkd = self.get_backend()
        marginals = [
            stats.uniform(-1, 2),
            stats.beta(1, 1, -1, 2),
            stats.norm(-1, np.sqrt(4)),
            stats.uniform(),
            stats.uniform(-1, 2),
            stats.beta(2, 1, -2, 3),
        ]
        var_trans = AffineTransform(
            IndependentMarginalsVariable(marginals, backend=bkd)
        )

        # first sample is on left boundary of all bounded variables
        # and one standard deviation to left of mean for gaussian variable
        # second sample is on right boundary of all bounded variables
        # and one standard deviation to right of mean for gaussian variable
        true_user_samples = bkd.asarray(
            [[-1, -1, -3, 0, -1, -2], [1, 1, 1, 1, 1, 1]]
        ).T

        canonical_samples = var_trans.map_to_canonical(true_user_samples)
        true_canonical_samples = bkd.ones(true_user_samples.shape)
        true_canonical_samples[:, 0] = -1
        assert bkd.allclose(true_canonical_samples, canonical_samples)

        user_samples = var_trans.map_from_canonical(canonical_samples)
        assert bkd.allclose(user_samples, true_user_samples)

    def test_define_mixed_tensor_product_random_variable_contin_discrete(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable
        type the distribution parameters ARE NOT the same
        """
        bkd = self.get_backend()
        # parameters of binomial distribution
        ntrials = 10
        prob_success = 0.5
        marginals = [
            stats.uniform(),
            stats.norm(-1, np.sqrt(4)),
            stats.norm(-1, np.sqrt(4)),
            stats.binom(ntrials, prob_success),
            stats.norm(-1, np.sqrt(4)),
            stats.uniform(0, 1),
            stats.uniform(0, 1),
            stats.binom(ntrials, prob_success),
        ]
        var_trans = AffineTransform(
            IndependentMarginalsVariable(marginals, backend=bkd)
        )

        # first sample is on left boundary of all bounded variables
        # and one standard deviation to left of mean for gaussian variables
        # second sample is on right boundary of all bounded variables
        # and one standard deviation to right of mean for gaussian variable
        true_user_samples = bkd.asarray(
            [[0, -3, -3, 0, -3, 0, 0, 0], [1, 1, 1, ntrials, 1, 1, 1, 10]]
        ).T

        canonical_samples = var_trans.map_to_canonical(true_user_samples)
        true_canonical_samples = bkd.ones(true_user_samples.shape)
        true_canonical_samples[:, 0] = -1
        true_canonical_samples[5, 0] = -1
        true_canonical_samples[3, :] = bkd.asarray([-1, 1])
        true_canonical_samples[7, :] = bkd.asarray([-1, 1])
        assert bkd.allclose(true_canonical_samples, canonical_samples)

        user_samples = var_trans.map_from_canonical(canonical_samples)
        assert bkd.allclose(user_samples, true_user_samples)

    def test_rosenblatt_transformation(self):
        # todo port rosenblatt transform to use backend
        bkd = NumpyMixin
        true_samples, true_canonical_samples, joint_density, limits = (
            rosenblatt_example_2d(nsamples=10)
        )

        nvars = 2
        opts = {"limits": limits, "nquad_samples_1d": 100}
        var_trans = RosenblattTransform(
            lambda x: joint_density.pdf(x)[:, 0], nvars, opts, backend=bkd
        )

        samples = var_trans.map_from_canonical(true_canonical_samples)
        assert bkd.allclose(true_samples, samples)

        canonical_samples = var_trans.map_to_canonical(samples)
        assert bkd.allclose(true_canonical_samples, canonical_samples)

    def test_transformation_composition_I(self):
        bkd = NumpyMixin
        # todo port rosenblatt transform to use backend
        np.random.seed(2)
        true_samples, true_canonical_samples, joint_density, limits = (
            rosenblatt_example_2d(nsamples=10)
        )

        #  rosenblatt_example_2d is defined on [0,1] remap to [-1,1]
        true_canonical_samples = true_canonical_samples * 2 - 1

        nvars = 2
        opts = {"limits": limits, "nquad_samples_1d": 100}
        var_trans_1 = RosenblattTransform(
            lambda x: joint_density.pdf(x)[:, 0], nvars, opts, backend=bkd
        )
        var_trans_2 = define_iid_random_variable_transformation(
            stats.uniform(0, 1), nvars, backend=bkd
        )
        var_trans = ComposeTransforms([var_trans_1, var_trans_2])

        samples = var_trans.map_from_canonical(true_canonical_samples)
        assert bkd.allclose(true_samples, samples)

        canonical_samples = var_trans.map_to_canonical(samples)
        assert bkd.allclose(true_canonical_samples, canonical_samples)

    def test_transformation_composition_II(self):
        bkd = self.get_backend()
        nvars = 2
        alpha_stat = 5
        beta_stat = 2
        scipy_rv = stats.beta(a=alpha_stat, b=beta_stat)
        beta = ContinuousScipyMarginal(scipy_rv, backend=bkd)
        x_marginals = [beta] * nvars

        z_correlation = -0.9 * bkd.ones((nvars, nvars))
        for ii in range(nvars):
            z_correlation[ii, ii] = 1.0

        x_correlation = (
            gaussian_copula_compute_x_correlation_from_z_correlation(
                x_marginals, z_correlation
            )
        )
        x_covariance = correlation_to_covariance(
            x_correlation, bkd.array([m.std() for m in x_marginals])
        )

        var_trans_1 = NatafTransform(x_marginals, x_covariance)

        # rosenblatt maps to [0,1] but polynomials of bounded variables
        # are in [-1,1] so add second transformation for this second mapping
        normal = ContinuousScipyMarginal(stats.norm(0, 1), backend=bkd)

        std_normal_marginal_cdfs = [normal.cdf] * nvars
        std_normal_marginal_inv_cdfs = [normal.ppf] * nvars
        var_trans_2 = UniformMarginalTransformation(
            std_normal_marginal_cdfs, std_normal_marginal_inv_cdfs, backend=bkd
        )
        var_trans = ComposeTransforms([var_trans_1, var_trans_2])

        nsamples = 1000
        true_samples, true_canonical_samples = (
            generate_x_samples_using_gaussian_copula(
                nvars, z_correlation, x_marginals, nsamples
            )
        )
        true_canonical_samples = bkd.asarray(
            stats.norm.cdf(true_canonical_samples)
        )

        samples = var_trans.map_from_canonical(true_canonical_samples)
        assert bkd.allclose(samples, true_samples)

        canonical_samples = var_trans.map_to_canonical(samples)
        assert bkd.allclose(canonical_samples, true_canonical_samples)

    def test_pickle_rosenblatt_transformation(self):
        # todo port rosenblatt transform to use backend
        # bkd = self.get_backend()
        bkd = NumpyMixin
        import pickle
        import os

        true_samples, true_canonical_samples, joint_density, limits = (
            rosenblatt_example_2d(nsamples=10)
        )

        nvars = 2
        opts = {"limits": limits, "nquad_samples_1d": 100}
        var_trans = RosenblattTransform(
            joint_density, nvars, opts, backend=bkd
        )

        filename = "rv_trans.pkl"
        with open(filename, "wb") as f:
            pickle.dump(var_trans, f)

        with open(filename, "rb") as f:
            pickle.load(f)

        os.remove(filename)

    def test_pickle_affine_random_variable_transformation(self):
        bkd = self.get_backend()
        import pickle
        import os

        nvars = 2
        alpha_stat = 2
        beta_stat = 10
        var_trans = define_iid_random_variable_transformation(
            stats.beta(alpha_stat, beta_stat, 0, 1), nvars, backend=bkd
        )

        filename = "rv_trans.pkl"
        with open(filename, "wb") as f:
            pickle.dump(var_trans, f)

        with open(filename, "rb") as f:
            pickle.load(f)

        os.remove(filename)

    def test_map_rv_discrete(self):
        bkd = self.get_backend()
        nvars = 2

        mass_locs = bkd.arange(5, 501, step=50)
        nmasses = mass_locs.shape[0]
        mass_probs = bkd.ones(nmasses, dtype=float) / float(nmasses)
        marginals = [
            CustomDiscreteMarginal(mass_locs, mass_probs, backend=bkd)
        ] * nvars
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        var_trans = AffineTransform(variable)
        print(marginals[0]._bkd, variable._bkd)

        samples = bkd.vstack(
            [mass_locs[None, :], mass_locs[0] * bkd.ones((1, nmasses))]
        )
        canonical_samples = var_trans.map_to_canonical(samples)

        assert canonical_samples[0].min() == -1
        assert canonical_samples[0].max() == 1

        recovered_samples = var_trans.map_from_canonical(canonical_samples)
        assert bkd.allclose(recovered_samples, samples)

    def test_identity_map_subset(self):
        bkd = self.get_backend()
        nvars = 3
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(0, 1), nvars, backend=bkd
        )
        var_trans.set_identity_maps([1])

        samples = bkd.asarray(np.random.uniform(0, 1, (nvars, 4)))
        canonical_samples = var_trans.map_to_canonical(samples)
        assert bkd.allclose(canonical_samples[1, :], samples[1, :])

        assert bkd.allclose(
            var_trans.map_from_canonical(canonical_samples), samples
        )

        marginals = [
            stats.uniform(-1, 2),
            stats.beta(1, 1, -1, 2),
            stats.norm(-1, np.sqrt(4)),
            stats.uniform(),
            stats.uniform(-1, 2),
            stats.beta(2, 1, -2, 3),
        ]
        var_trans = AffineTransform(
            IndependentMarginalsVariable(marginals, backend=bkd)
        )
        var_trans.set_identity_maps([4, 2])

        samples = var_trans.variable().rvs(10)
        canonical_samples = var_trans.map_to_canonical(samples)
        assert bkd.allclose(canonical_samples[[2, 4], :], samples[[2, 4], :])

        assert bkd.allclose(
            var_trans.map_from_canonical(canonical_samples), samples
        )

    def test_map_derivatives(self):
        bkd = self.get_backend()
        nvars = 2
        nsamples = 10
        x = bkd.asarray(np.random.uniform(0, 1, (nvars, nsamples)))
        # vals = bkd.sum(x**2, axis=0)[:, None]
        grad = bkd.vstack([2 * x[ii : ii + 1, :] for ii in range(nvars)])
        var_trans = AffineTransform(
            IndependentMarginalsVariable(
                [stats.uniform(0, 1), stats.uniform(2, 2)], backend=bkd
            )
        )
        canonical_derivs = var_trans.map_derivatives_to_canonical_space(grad)
        for ii in range(nvars):
            lb, ub = var_trans.variable().marginals()[ii].interval(1)
            assert bkd.allclose(
                canonical_derivs[ii, :], (ub - lb) * grad[ii, :] / 2
            )
        recovered_derivs = var_trans.map_derivatives_from_canonical_space(
            canonical_derivs
        )
        assert bkd.allclose(recovered_derivs, grad)

    def test_dense_gaussian_transform(self):
        bkd = self.get_backend()
        nvars = 2
        mean = bkd.asarray(np.random.uniform(0, 1, (nvars, 1)))
        A = bkd.asarray(np.random.normal(0, 1, (nvars, nvars)))
        cov = A.T @ A
        scipy_rv = stats.multivariate_normal(mean=mean[:, 0], cov=cov)
        nsamples = int(1e5)
        samples = bkd.asarray(scipy_rv.rvs(nsamples).T)
        trans = DenseGaussianTransform(mean, cov, bkd)
        canonical_samples = trans.map_to_canonical(samples)
        assert bkd.allclose(
            bkd.mean(canonical_samples, axis=1), bkd.zeros((nvars,)), atol=1e-2
        )
        assert bkd.allclose(
            bkd.cov(canonical_samples, rowvar=True, ddof=1),
            bkd.eye(nvars),
            atol=1e-2,
        )
        recovered_samples = trans.map_from_canonical(canonical_samples)
        assert bkd.allclose(recovered_samples, samples)


class TestNumpyVariableTransforms(TestVariableTransforms, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchVariableTransforms(TestVariableTransforms, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
