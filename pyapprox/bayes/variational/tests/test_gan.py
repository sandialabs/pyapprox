import unittest

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.bayes.variational.gan import (
    LogisticGenerativeAdvesarialModel,
    GenerativeAdvesarialGradientDescent,
)
from pyapprox.bayes.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    DenseMatrixLaplaceApproximationForPrediction,
)


class TestGAN:
    def setUp(self):
        np.random.seed(1)

    def test_conditional_gan(self):
        bkd = self.get_backend()
        nvars = 100

        # set prior to be standard normal
        L = np.eye(nvars)
        mu = np.zeros((nvars, 1))

        # define a function to sample from the prior
        def sample_prior(nvars, nsamples):
            return bkd.asarray(
                mu + L @ np.random.normal(0, 1, (nvars, nsamples))
            )

        # define the noiselsess observational model
        nobs = 2
        Amat = bkd.asarray(np.random.uniform(0, 1, (nobs, nvars)) + 1)
        print(Amat)

        def obs_model(samples):
            return Amat @ samples

        # define the qoi model
        nqoi = 1
        Qmat = bkd.asarray(np.random.uniform(0, 1, (nqoi, nvars)) + 1)
        Qvec = bkd.asarray(np.random.uniform(0, 1, (nqoi, 1)) + 1)

        def qoi_model(samples):
            return Qmat @ samples + Qvec

        # define the noise model
        noise_std = 0.4

        def generate_obs(samples):
            vals = obs_model(samples)
            noise = bkd.asarray(np.random.normal(0, noise_std, vals.shape))
            return vals + noise

        # Generate real samples
        ntrain_samples = 10000
        real_train_prior_samples = sample_prior(nvars, ntrain_samples)
        real_train_samples = qoi_model(real_train_prior_samples)
        real_train_conditional_samples = generate_obs(real_train_prior_samples)

        gen_model = LogisticGenerativeAdvesarialModel(
            [2] * (nqoi + nobs), [2] * (nqoi + nobs)
        )
        gen_model.set_optimizer(GenerativeAdvesarialGradientDescent())
        gen_model.fit(real_train_samples, real_train_conditional_samples)

        # Generate fake samples using GAN and compare to new real samples
        ntest_samples = 10000
        ndata_realizations = 2
        ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        prior_hessian = np.linalg.inv(L @ L.T)
        noise_cov_inv = np.diag(np.full((nobs,), 1 / noise_std**2))
        priorpush_mean, priorpush_cov = (
            push_forward_gaussian_though_linear_model(
                bkd.to_numpy(Qmat), bkd.to_numpy(Qvec), mu, L @ L.T
            )
        )
        priorpush_rv = stats.norm(
            priorpush_mean[0], np.sqrt(priorpush_cov[0, 0])
        )
        xx = np.linspace(*priorpush_rv.interval(1 - 1e-3), 101).T
        ax.fill_between(xx[0], 0, priorpush_rv.pdf(xx[0]), label="Prior PDF")
        for ii in range(ndata_realizations):
            # Generate realization of the observations for testing
            true_sample = sample_prior(nvars, 1)
            obs = generate_obs(true_sample)
            # todo implement la_tile
            conditional_vars = bkd.tile(obs, (1, ntest_samples))
            fake_test_samples = bkd.to_numpy(
                generator.rvs(ntest_samples, conditional_vars, detach=True)
            )

            # Get analytical mean and covariance of posterior
            post_mean, post_cov = (
                laplace_posterior_approximation_for_linear_models(
                    bkd.to_numpy(Amat), mu, prior_hessian, noise_cov_inv, obs
                )
            )
            postpush_mean, postpush_cov = (
                push_forward_gaussian_though_linear_model(
                    bkd.to_numpy(Qmat), bkd.to_numpy(Qvec), post_mean, post_cov
                )
            )

            postpush_rv = stats.norm(
                postpush_mean[0, 0], np.sqrt(postpush_cov[0, 0])
            )
            print(
                "Post Push. Mean",
                postpush_mean,
                fake_test_samples.mean(axis=1),
            )
            print(
                "Post Push. Covariance",
                postpush_cov,
                np.cov(fake_test_samples, ddof=1),
            )
            # Take tranpose below because interval returns two arrays such
            # that when passed to linspace creates an array wigh shape (N, 1)
            ax.plot(
                xx[0],
                postpush_rv.pdf(xx[0]),
                ls="-",
                label=f"Post PDF {ii}",
                lw=3,
            )
            gen_kde = stats.gaussian_kde(fake_test_samples)
            # ax.hist(fake_test_samples[0], bins=30, density=True,
            #     label='Fake samples')
            ax.plot(
                xx[0], gen_kde(xx), ls="--", label=f"Fake Post PDF {ii}", lw=3
            )
        ax.legend()
        plt.savefig("wgan-pf.pdf", transparent=True)
        plt.show()


class TestTorchMOStats(TestGAN, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
