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
    GaussianPushForward,
)
from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian


class TestGAN:
    def setUp(self):
        np.random.seed(1)

    def test_conditional_gan(self):
        bkd = self.get_backend()
        nvars = 100

        # set prior to be standard normal
        L = bkd.eye(nvars)
        mu = bkd.zeros((nvars, 1))

        prior = DenseCholeskyMultivariateGaussian(mu, L, backend=bkd)

        # define the noiselsess observational model
        nobs = 2
        obs_mat = bkd.asarray(np.random.uniform(0, 1, (nobs, nvars)) + 1)

        def obs_model(samples):
            return obs_mat @ samples

        # define the qoi model
        nqoi = 1
        pred_mat = bkd.asarray(np.random.uniform(0, 1, (nqoi, nvars)) + 1)
        pred_vec = 0 * bkd.asarray(np.random.uniform(0, 1, (nqoi, 1)) + 1)

        def qoi_model(samples):
            return pred_mat @ samples + pred_vec

        # define the noise model
        noise_std = 0.4

        def generate_obs(samples):
            vals = obs_model(samples)
            noise = bkd.asarray(np.random.normal(0, noise_std, vals.shape))
            return vals + noise

        # Generate real samples
        ntrain_samples = 100  # 00
        real_train_prior_samples = prior.rvs(ntrain_samples)
        real_train_samples = qoi_model(real_train_prior_samples)
        real_train_conditional_samples = generate_obs(real_train_prior_samples)

        gen_model = LogisticGenerativeAdvesarialModel(
            nqoi, nobs, [2] * (nqoi + nobs), [2] * (nqoi + nobs)
        )
        optimizer = GenerativeAdvesarialGradientDescent(epochs=2)
        optimizer.set_verbosity(3)
        gen_model.set_optimizer(optimizer)
        gen_model.fit(real_train_samples, real_train_conditional_samples)

        # Generate fake samples using GAN and compare to new real samples
        ntest_samples = 10000
        ndata_realizations = 2
        noise_cov = bkd.diag(bkd.full((nobs,), noise_std**2))
        prior_pushforward = GaussianPushForward(
            pred_mat, mu, L @ L.T, pred_vec, backend=bkd
        )
        priorpush_rv = stats.norm(
            prior_pushforward.mean()[0],
            bkd.sqrt(prior_pushforward.covariance()[0, 0]),
        )
        lb, ub = priorpush_rv.interval(1 - 1e-3)
        lb, ub = lb[0], ub[0]

        ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        xx = bkd.linspace(lb, ub, 101)[None, :]
        ax.fill_between(
            xx[0], 0, priorpush_rv.pdf(xx[0]), label="Prior PDF", alpha=0.5
        )

        posterior = DenseMatrixLaplacePosteriorApproximation(
            obs_mat,
            prior.mean(),
            prior.covariance(),
            noise_cov,
            backend=bkd,
        )
        posterior_pushfoward = DenseMatrixLaplaceApproximationForPrediction(
            obs_mat,
            pred_mat,
            prior.mean(),
            prior.covariance(),
            noise_cov,
            backend=bkd,
        )
        for ii in range(ndata_realizations):
            # Generate realization of the observations for testing
            true_sample = prior.rvs(1)
            obs = generate_obs(true_sample)
            conditional_vars = bkd.tile(obs, (1, ntest_samples))
            fake_test_samples = gen_model._gen.rvs(
                ntest_samples, conditional_vars
            )

            # Get analytical mean and covariance of posterior
            print(obs, "obs")
            posterior.compute(obs)
            posterior_pushfoward.compute(obs)
            postpush_rv = stats.norm(
                posterior_pushfoward.mean()[0, 0],
                bkd.sqrt(posterior_pushfoward.covariance()[0, 0]),
            )
            print(
                "Post Push. Mean",
                posterior_pushfoward.mean(),
                fake_test_samples.mean(axis=1),
            )
            print(
                "Post Push. Covariance",
                posterior_pushfoward.covariance(),
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
            print(fake_test_samples.shape)
            assert False
            # ax.hist(
            #    fake_test_samples[0],
            #    bins=30,
            #    density=True,
            #    label="Fake samples",
            # )
            # gen_kde = stats.gaussian_kde(fake_test_samples[0])
            # ax.plot(
            #     xx[0], gen_kde(xx), ls="--", label=f"Fake Post PDF {ii}", lw=3
            # )
        ax.legend()
        # plt.savefig("wgan-pf.pdf", transparent=True)
        plt.show()


class TestTorchMOStats(TestGAN, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
