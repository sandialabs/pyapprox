from abc import abstractmethod

import numpy as np

from pyapprox.util.linalg import (
    cholesky_inverse, log_determinant_from_cholesky_factor,
    diag_of_mat_mat_product)
from pyapprox.interface.model import Model


class LogLikelihood(Model):
    def __init__(self):
        super().__init__()
        self._obs = None
        self._nobs = None

    def _get_nobs(self):
        return self._nobs

    def set_observations(self, obs):
        """
        Parameters
        ----------
        obs : np.ndarray (nobs, 1)
            The observations
        """
        if (obs.ndim != 2 or obs.shape[0] != self._get_nobs()):
            raise ValueError("obs has the wrong shape {0}".format(
                obs.shape))
        self._obs = obs

    @abstractmethod
    def _loglike(self, many_pred_obs):
        raise NotImplementedError

    def __call__(self, many_pred_obs):
        if many_pred_obs.shape[0] != self._get_nobs():
            raise ValueError("many_pred_obs has the wrong shape {0}".format(
                many_pred_obs.shape)+" nobs is {0}".format(
                    self._get_nobs()))
        return self._loglike(many_pred_obs)


class GaussianLogLikelihood(LogLikelihood):
    def __init__(self, noise_cov, tile_obs=True):
        """
        Correlated Gaussian noise. Takes pred_obs as argument to call
        """
        super().__init__()
        self._loglike_consts = self._setup(noise_cov)
        self._tile_obs = tile_obs

    def _setup(self, noise_cov):
        if (noise_cov.ndim != 2 or noise_cov.shape[0] != noise_cov.shape[1]):
            raise ValueError("noise_cov has the wrong shape {0}".format(
                noise_cov.shape))
        self._nobs = noise_cov.shape[0]
        self._noise_cov = noise_cov
        self._noise_chol = np.linalg.cholesky(self._noise_cov)
        self._noise_chol_inv = cholesky_inverse(self._noise_chol)
        self._log_noise_cov_det = log_determinant_from_cholesky_factor(
            self._noise_chol)
        self._last_L_inv_res = None
        self.set_design_weights(np.ones((self._nobs, 1)))
        return self._nobs*np.log(2*np.pi) + self._log_noise_cov_det

    def set_design_weights(self, design_weights):
        self._design_weights = design_weights
        self._weighted_noise_chol_inv = self._noise_chol_inv*np.sqrt(
            self._design_weights)

    def _make_noisy(self, noiseless_obs, noise):
        if noiseless_obs.shape != noise.shape:
            msg = "shapes of noiseless_obs {0} and obs {1} must match".format(
                noiseless_obs.shape, noise.shape)
            raise ValueError(msg)
        return noiseless_obs + noise

    def _sample_noise(self, nsamples):
        # create samples (nsamples, nobs) then take transpose
        # to ensure same noise is used for ith sample
        # regardless of size of nsam
        normal_samples = np.random.normal(
            0, 1, (nsamples, self._get_nobs())).T
        return self._noise_chol @ normal_samples

    def _loglike_many(self, many_obs, many_pred_obs):
        residual = (many_obs-many_pred_obs)
        L_inv_res = self._weighted_noise_chol_inv@residual
        vals = diag_of_mat_mat_product(
            L_inv_res.T, L_inv_res)
        return (-0.5*(vals + self._loglike_consts))[:, None]

    def _parse_obs(self, obs, many_pred_obs):
        if self._tile_obs:
            # stack vals for each obs vertically
            return (
                np.repeat(self._obs, many_pred_obs.shape[1], axis=1),
                np.tile(many_pred_obs, (1, self._obs.shape[1])))
        if many_pred_obs.shape != self._obs.shape:
            msg = "many_pred_obs shape does not match self._obs shape"
            raise ValueError(msg)
        return (self._obs, many_pred_obs)

    def _loglike(self, many_pred_obs):
        return self._loglike_many(
            *self._parse_obs(self._obs, many_pred_obs))

    def noise_covariance(self):
        return self._noise_cov


class IndependentGaussianLogLikelihood(
        GaussianLogLikelihood):
    def __init__(self, noise_cov_diag, tile_obs=True):
        """
        Independent but not necessarily identically distributed Gaussian noise.
        Evaluating this is faster than GaussianLikelihood if noise covariance
        is a diagonal matrix
        """
        super().__init__(noise_cov_diag, tile_obs)

    def set_design_weights(self, design_weights):
        self._design_weights = design_weights
        self._weighted_noise_cov_inv_diag = (
            self._noise_cov_inv_diag*self._design_weights)

    def _setup(self, noise_cov_diag):
        if (noise_cov_diag.ndim != 2 or noise_cov_diag.shape[1] != 1):
            raise ValueError("noise_cov_diag has the wrong shape {0}".format(
                noise_cov_diag.shape))
        self._nobs = noise_cov_diag.shape[0]
        self._noise_cov_diag = noise_cov_diag
        self._noise_std_diag = np.sqrt(self._noise_cov_diag)
        self._log_noise_cov_det = np.log(self._noise_cov_diag).sum()
        self._noise_cov_inv_diag = 1/(self._noise_cov_diag)
        self._r_noise_cov_inv = None
        self.set_design_weights(np.ones((self._nobs, 1)))
        return self._nobs*np.log(2*np.pi) + self._log_noise_cov_det

    def _loglike_many(self, many_obs, many_pred_obs):
        residual = (many_obs-many_pred_obs)
        vals = diag_of_mat_mat_product(
            residual.T, self._weighted_noise_cov_inv_diag*residual)
        return (-0.5*(vals + self._loglike_consts))[:, None]

    def _sample_noise(self, nsamples):
        # create samples (nsamples, nobs) then take transpose
        # to ensure same noise is used for ith sample
        # regardless of size of nsam
        normal_samples = np.random.normal(
            0, 1, (nsamples, self._get_nobs())).T
        return self._noise_std_diag*normal_samples

    def noise_covariance(self):
        return np.diag(self._noise_cov_diag[:, 0])

    def __repr__(self):
        return "{0}(nobs={1}, sigma={2})".format(
            self.__class__.__name__, self._get_nobs(),
            self._noise_std_diag[:, 0])


class SingleObsIndependentGaussianLogLikelihood(
        IndependentGaussianLogLikelihood):

    def _loglike(self, many_pred_obs):
        return self._loglike_many(
            np.repeat(self._obs, many_pred_obs.shape[1], axis=1),
            many_pred_obs)

    def set_observations(self, obs):
        """
        Parameters
        ----------
        obs : np.ndarray (nobs, 1)
            The observations
        """
        if (obs.ndim != 2 or obs.shape[1] != 1 or
                obs.shape[0] != self._get_nobs()):
            raise ValueError("obs has shape {0}".format(
                obs.shape)+f" but nobs={self._get_nobs()}")
        self._obs = obs


class ModelBasedLogLikelihood(LogLikelihood):
    def __init__(self, model, loglike):
        super().__init__()
        self._model = model
        self._loglike = loglike
        self._last_sample = None
        self._last_pred_obs = None

    def _get_nobs(self):
        return self._loglike._get_nobs()

    def set_observations(self, obs):
        # if (obs.ndim != 2 or obs.shape[1] != 1 or
        #         obs.shape[0] != self._model._nobs):
        #     raise ValueError("obs has the wrong shape {0}".format(
        #         obs.shape))
        self._model._obs = obs

    def _loglike(self, many_pred_obs):
        return self._loglike._loglike(many_pred_obs)

    def __call__(self, samples):
        many_pred_obs = self._model(samples).T
        self._last_sample = samples[:, -1:]
        self._last_pred_obs = many_pred_obs[:, -1:]
        return self._loglike(many_pred_obs)

    def rvs(self, sample):
        """Draw a realization from the likelihood"""
        if sample.ndim != 2 or sample.shape[1] != 1:
            raise ValueError("samples has the wrong shape")
        noiseless_obs = self._model(sample).T
        noise = self._loglike._sample_noise(1)
        return self._loglike._make_noisy(noiseless_obs, noise)

    def __repr__(self):
        return "{0}(nobs={1}, model={2})".format(
            self.__class__.__name__, self._get_nobs(), self._model)


class ModelBasedGaussianLogLikelihood(ModelBasedLogLikelihood):
    def __init__(self, model, loglike):
        super().__init__(model, loglike)
        self._jacobian_implemented = model._jacobian_implemented

    def _jacobian(self, sample):
        if self._last_sample is None or not np.allclose(
                sample, self._last_sample):
            self._last_sample = sample.copy()
            self._last_pred_obs = self._model(sample).T
        residual = (self._loglike._obs - self._last_pred_obs)
        L_inv_res = self._loglike._weighted_noise_chol_inv@residual
        return np.linalg.multi_dot(
            (L_inv_res.T, self._loglike._weighted_noise_chol_inv,
             self._model.jacobian(sample)))


class ModelBasedIndependentGaussianLogLikelihood(ModelBasedLogLikelihood):
    def __init__(self, model, loglike):
        super().__init__(model, loglike)
        self._jacobian_implemented = model._jacobian_implemented

    def _jacobian(self, sample):
        if self._last_sample is None or not np.allclose(
                sample, self._last_sample):
            self._last_sample = sample.copy()
            self._last_pred_obs = self._model(sample).T
        residual = (self._loglike._obs - self._last_pred_obs)
        tmp = self._loglike._weighted_noise_cov_inv_diag*residual
        return tmp.T @ self._model.jacobian(sample)


# TODO when complete move classes (and tests in test_likelihood.py) to
# expdesign module
class OEDGaussianLogLikelihood(Model):
    def __init__(self, loglike, many_pred_obs, pred_weights):
        super().__init__()
        if not isinstance(loglike, IndependentGaussianLogLikelihood):
            raise ValueError(
                "loglike must be IndependentGaussianLogLikelihood")
        self._loglike = loglike
        self._many_pred_obs = many_pred_obs
        if pred_weights.shape[0] != many_pred_obs.shape[1]:
            raise ValueError("pred_weights and many_pred_obs are inconsistent")
        self._pred_weights = pred_weights
        self._jacobian_implemented = True

    def __call__(self, design_weights):
        self._loglike.set_design_weights(design_weights)
        return self._loglike(self._many_pred_obs)

    def _jacobian(self, design_weights):
        # stack jacobians for each obs vertically
        # todo could be just done once when objected is created
        obs, many_pred_obs = self._loglike._parse_obs(
            self._loglike._obs, self._many_pred_obs)
        residual = (obs-many_pred_obs)
        jac = (residual.T**2)*(self._loglike._noise_cov_inv_diag[:, 0]*(-0.5))
        return jac

    def __repr__(self):
        return "{0}(loglike={1}, M={2}, N={3})".format(
            self.__class__.__name__, self._loglike.__class__.__name__,
            self._loglike._obs.shape[1], self._many_pred_obs.shape[1])


class Evidence(Model):
    def __init__(self, loglike):
        super().__init__()
        if not isinstance(loglike, OEDGaussianLogLikelihood):
            raise ValueError(
                "loglike must be OEDGaussianLogLikelihood")
        if not loglike._loglike._tile_obs:
            raise ValueError(
                "loglike._loglike._tile_obs is False. Must be True")

        self._loglike = loglike
        self._jacobian_implemented = True

    def _reshape_vals(self, vals):
        # unflatten vals
        return vals.reshape((
            self._loglike._many_pred_obs.shape[1],
            self._loglike._loglike._obs.shape[1]), order='F')

    def _reshape_jacobian(self, jac):
        # unflatten jacobian
        return jac.reshape(
            self._loglike._many_pred_obs.shape[1],
            self._loglike._loglike._obs.shape[1], jac.shape[1],
            order='F')

    def __call__(self, design_weights):
        like_vals = self._reshape_vals(np.exp(self._loglike(design_weights)))
        return (self._loglike._pred_weights*like_vals).sum(axis=0)[:, None]

    def _jacobian(self, design_weights):
        like_vals = self._reshape_vals(np.exp(self._loglike(design_weights)))
        like_jac = self._reshape_jacobian(
            self._loglike.jacobian(design_weights))
        jac = np.sum(
            (self._loglike._pred_weights*like_vals)[..., None]*like_jac,
            axis=0)
        return jac

    def __repr__(self):
        return "{0}(loglike={1})".format(
            self.__class__.__name__, self._loglike)


class LogEvidence(Evidence):
    def __call__(self, design_weights):
        return np.log(super().__call__(design_weights))

    def _jacobian(self, design_weights):
        like_vals = np.exp(self._reshape_vals(self._loglike(design_weights)))
        weighted_like_vals = self._loglike._pred_weights*like_vals
        like_jac = self._reshape_jacobian(
            self._loglike.jacobian(design_weights))
        jac = 1/np.sum(weighted_like_vals, axis=0)[:, None]*np.sum(
            weighted_like_vals[..., None]*like_jac, axis=0)
        return jac


class KLOEDObjective(Model):
    #TODO this is currently only useful for GaussianLikelihood. Generalize
    #to allow any loglike
    def __init__(self, noise_cov_diag, outer_pred_obs,
                 outer_pred_weights, noise_samples,
                 inner_pred_obs, inner_pred_weights):
        super().__init__()

        self._outer_loglike = IndependentGaussianLogLikelihood(
            noise_cov_diag, tile_obs=False)
        outer_obs = self._outer_loglike._make_noisy(
            outer_pred_obs, noise_samples)
        self._outer_loglike.set_observations(outer_obs)
        self._outer_oed_loglike = OEDGaussianLogLikelihood(
            self._outer_loglike, outer_pred_obs, outer_pred_weights)

        self._inner_loglike = IndependentGaussianLogLikelihood(noise_cov_diag)
        self._inner_loglike.set_observations(outer_obs)
        self._inner_oed_loglike = OEDGaussianLogLikelihood(
            self._inner_loglike, inner_pred_obs, inner_pred_weights)
        self._log_evidence = LogEvidence(self._inner_oed_loglike)

        self._jacobian_implemented = True

    def __call__(self, design_weights):
        log_evidences = self._log_evidence(design_weights)
        outer_log_like_vals = self._outer_oed_loglike(design_weights)
        if log_evidences.shape != outer_log_like_vals.shape:
            msg = "log_evidences and outer_log_like_vals.shape do not match"
            raise ValueError(msg)
        if log_evidences.ndim != 2:
            raise ValueError("log_evidences must be a 2d array")
        outer_weights = self._outer_oed_loglike._pred_weights
        vals = (outer_weights*(outer_log_like_vals-log_evidences)).sum(
            axis=0)[:, None]
        return vals

    def _reshape_jacobian(self, jac):
        # unflatten jacobian
        return jac.reshape(
            self._outer_loglike._obs.shape[1], jac.shape[1])

    def jacobian(self, design_weights):
        jac_log_evidences = self._log_evidence.jacobian(design_weights)
        jac_outer_log_like = self._outer_oed_loglike.jacobian(design_weights)
        jac_outer_log_like = self._reshape_jacobian(jac_outer_log_like)
        outer_weights = self._outer_oed_loglike._pred_weights
        jac = (outer_weights*(jac_outer_log_like-jac_log_evidences)).sum(
            axis=0)
        return jac
