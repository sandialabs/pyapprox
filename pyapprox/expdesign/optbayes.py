import numpy as np

from pyapprox.interface.model import Model
from pyapprox.bayes.likelihood import (
    IndependentExponentialLogLikelihood, IndependentGaussianLogLikelihood)
from pyapprox.optimization.pya_minimize import Constraint


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

        self._prev_design_weights = None
        self._like_vals = None
        self._like_jac = None
        self._weighted_like_vals = None
        self._weighted_like_vals_prod_jac = None
        self._evidence_jac = None

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
        if (self._prev_design_weights is None or
            not np.allclose(design_weights, self._prev_design_weights,
                            atol=1e-15, rtol=1e-15)):
            self._prev_design_weights = design_weights.copy()
            self._like_vals = self._reshape_vals(
                np.exp(self._loglike(design_weights)))
            self._weighted_like_vals = (
                self._loglike._pred_weights*self._like_vals)
        return (self._weighted_like_vals).sum(axis=0)[:, None]

    def _jacobian(self, design_weights):
        if not np.allclose(design_weights, self._prev_design_weights,
                           atol=1e-15, rtol=1e-15):
            # recompute necessary data
            self(design_weights)
        self._like_jac = self._reshape_jacobian(
            self._loglike.jacobian(design_weights))
        self._weighted_like_vals_prod_jac = (
            self._weighted_like_vals[..., None] * self._like_jac)
        self._evidence_jac = np.sum(self._weighted_like_vals_prod_jac, axis=0)
        return self._evidence_jac

    def __repr__(self):
        return "{0}(loglike={1})".format(
            self.__class__.__name__, self._loglike)


class LogEvidence(Evidence):
    def __call__(self, design_weights):
        evidence = super().__call__(design_weights)
        return np.log(evidence)

    def _jacobian(self, design_weights):
        evidence = super().__call__(design_weights)
        jac = super()._jacobian(design_weights)
        jac = 1/evidence*jac
        return jac


from pyapprox.optimization.pya_minimize import SampleAverageMean
class NoiseStatistic():
    def __init__(self, stat):
        self._stat = stat
    
    def __call__(self, outer_vals, outer_weights):
        return self._stat.__call__(outer_vals, outer_weights)

    def jacobian(self, outer_vals, outer_jacs, outer_weights):
        return self._stat.jacobian(
            outer_vals, outer_jacs[..., None], outer_weights).T


# class EntropicNoiseStatistic(SampleAverageEntropicRisk):
#     def __call__(self, outer_vals, outer_weights):
#         return np.log((np.exp(outer_weights)*outer_vals)).sum(axis=0)[:, None]

#     def jacobian(self, outer_vals, outer_jacs, outer_weights):
#         risk = self(outer_vals, outer_weights)
#         return ((outer_weights*outer_vals)*outer_jacs).sum(axis=0)/risk


class KLOEDObjective(Model):
    def __init__(self, noise_cov_diag, outer_pred_obs,
                 outer_pred_weights, noise_samples,
                 inner_pred_obs, inner_pred_weights,
                 noise_stat=NoiseStatistic(SampleAverageMean())):
        super().__init__()

        self._noise_stat = noise_stat
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
        # apply hessian reduces optimization iteration count but increases
        # run time because cost of each iteration increases so turn off
        # self._apply_hessian_implemented = True

    def __call__(self, design_weights):
        log_evidences = self._log_evidence(design_weights)
        outer_log_like_vals = self._outer_oed_loglike(design_weights)
        outer_weights = self._outer_oed_loglike._pred_weights
        vals = self._noise_stat(
            outer_log_like_vals-log_evidences, outer_weights)
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -vals

    def _reshape_jacobian(self, jac):
        # unflatten jacobian
        return jac.reshape(
            self._outer_loglike._obs.shape[1], jac.shape[1])

    def _jacobian(self, design_weights):
        log_evidences = self._log_evidence(design_weights)
        outer_log_like_vals = self._outer_oed_loglike(design_weights)
        jac_log_evidences = self._log_evidence.jacobian(design_weights)
        jac_outer_log_like = self._outer_oed_loglike.jacobian(design_weights)
        jac_outer_log_like = self._reshape_jacobian(jac_outer_log_like)
        outer_weights = self._outer_oed_loglike._pred_weights
        jac = self._noise_stat.jacobian(
            outer_log_like_vals-log_evidences,
            jac_outer_log_like-jac_log_evidences, outer_weights)
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -jac

    def _hvp1(self, outer_weights, evidence, vec):
        # g'(f(x))\nabla^2 f^\top \dot v
        # this assumes that hessian of log-likelihood is zero
        # thus this function can only be used with log likelihoods that are
        # linear in the weights
        tmp = np.sum((
            self._log_evidence._weighted_like_vals_prod_jac*(
                self._log_evidence._like_jac@vec)), axis=0)
        hvp1 = np.sum((outer_weights/evidence)*tmp, axis=0)
        return hvp1

    def _hvp2(self, outer_weights, evidence, vec):
        # g''(f(x))\nabla f^\top nabla f \dot v
        evidence_jac = self._log_evidence._evidence_jac
        hvp2 = np.sum(
            (outer_weights/evidence**2)*evidence_jac*(evidence_jac@vec),
            axis=0)
        return hvp2

    def _apply_hessian(self, design_weights, vec):
        if not isinstance(self._noise_stat._stat, SampleAverageMean):
            msg = "apply hessian only supported for MeanNoiseStatistic"
            raise ValueError(msg)
        evidence = super(LogEvidence, self._log_evidence).__call__(
            design_weights)
        outer_weights = self._outer_oed_loglike._pred_weights
        hvp1 = self._hvp1(outer_weights, evidence, vec)
        hvp2 = self._hvp2(outer_weights, evidence, vec)
        hvp = hvp1-hvp2
        return hvp[:, None]


class PredictionOEDDeviation(Model):
    def __init__(self, loglike, qoi_vals, qoi_weights):
        self._qoi_vals = qoi_vals
        self._qoi_weights = qoi_weights
        self._loglike = loglike

    def __call__(self):
        raise NotImplementedError


class OEDStandardDeviation(PredictionOEDDeviation):
    def _first_momement(self, like_vals):
        return (self._qoi_vals*like_vals).sum(axis=1)

    def _second_momement(self, like_vals):
        return (self._qoi_vals**2*like_vals).sum(axis=1)

    def __call__(self, like_vals):
        return (self._second_moment(like_vals)/evidences-self._first_moment(like_vals)/evidences**2)

    def _first_momement_jac(self, like_vals):
        return (self._qoi_vals*like_vals).sum(axis=1)

    def _second_momement_jac(self, like_vals):
        return (self._qoi_vals**2*like_vals).sum(axis=1)

    def _jacobian(self, like_vals):
        return (self._second_moment(like_vals)/evidences-self._first_moment(like_vals)/evidences**2)


class PredictionOEDObjective(KLOEDObjective):
    def __init__(self, noise_cov_diag, outer_pred_obs,
                 outer_pred_weights, noise_samples,
                 inner_pred_obs, inner_pred_weights,
                 noise_stat=NoiseStatistic(SampleAverageMean())):
        super().__init__(
            noise_cov_diag, outer_pred_obs, outer_pred_weights, noise_samples,
            inner_pred_obs, inner_pred_weights)

    def __call__(self, design_weights):
        evidences = self._log_evidence._evidence(design_weights)
        deviations = 

class WeightsConstraintModel(Model):
    def __init__(self):
        super().__init__()
        self._jacobian_implemented = True

    def __call__(self, weights):
        assert np.all(weights >= 0)
        return weights.sum(axis=0)[:, None]

    def _jacobian(self, weights):
        assert np.all(weights >= 0)
        return np.ones((1, weights.shape[0]))


class WeightsConstraint(Constraint):
    def __init__(self, nobs, keep_feasible=False):
        model = WeightsConstraintModel()
        bounds = np.array([[nobs, nobs]])
        super().__init__(model, bounds, keep_feasible)


class SparseOEDObjective(Model):
    def __init__(self, objective, penalty):
        super().__init__()
        self._objective = objective
        self._penalty = penalty
        self._jacobian_implemented = self._objective._jacobian_implemented

    def _l1norm(self, weights):
        # assumes weights are positive
        return np.sum(weights)

    def _l1norm_jac(self, weights):
        # assumes weights are positive
        return np.ones((1, weights.shape[0]))

    def __call__(self, weights):
        assert np.all(weights >= 0)
        # neagtive penalty because we are minimizing negative KL divergence
        return self._objective(weights) - self._penalty*self._l1norm(weights)

    def _jacobian(self, weights):
        assert np.all(weights >= 0)
        # neagtive penalty because we are minimizing negative KL divergence
        return self._objective.jacobian(
            weights) - self._penalty*self._l1norm_jac(weights)


class DOptimalLinearModelObjective(Model):
    def __init__(self, model, noise_cov, prior_cov):
        """
        Compute the d-optimality criterion for a linear model
        f(x) = Amat.dot(x)

        F = A.T@A/sigma^2
        G = F*prior_cov
        obj(w) = 1/2 log(Det(G+I)))

        References
        ----------
        Alen Alexanderian and Arvind K. Saibaba
        Efficient D-Optimal Design of Experiments for
        Infinite-Dimensional Bayesian Linear Inverse Problems
        SIAM Journal on Scientific Computing 2018 40:5, A2956-A2985
        https://doi.org/10.1137/17M115712X
        """
        super().__init__()
        if not np.isscalar(noise_cov):
            raise ValueError("noise_cov must be a scalar")
        if not np.isscalar(prior_cov):
            raise ValueError("prior_cov must be a scalar")
        self._model = model
        self._noise_cov = noise_cov
        self._prior_cov = prior_cov
        self._jacobian_implemented = True
        self._hessian_implemented = True

    def __call__(self, weights):
        Amat = self._model._jac_matrix
        nvars = Amat.shape[1]
        hess_misfit = Amat.T.dot(weights*Amat)*self._prior_cov/self._noise_cov
        ident = np.eye(nvars)
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -np.array(
            [0.5*np.linalg.slogdet(hess_misfit+ident)[1]])[:, None]

    def _Y(self, weights):
        Amat = self._model._jac_matrix
        nvars = Amat.shape[1]
        hess_misfit = Amat.T.dot(weights*Amat)*self._prior_cov/self._noise_cov
        ident = np.eye(nvars)
        Y = hess_misfit+ident
        return Y

    def _jacobian(self, weights):
        Y = self._Y(weights)
        inv_Y = np.linalg.inv(Y)
        jac_log_det_Y = np.array(
            [np.trace(inv_Y@row[:, None]@row[None, :])
             for row in self._model._jac_matrix])*(
                     self._prior_cov/self._noise_cov)
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -0.5*jac_log_det_Y

    def _Y_inv_dYdw(self, inv_Y, ii):
        rowii = self._model._jac_matrix[ii]
        return inv_Y@rowii[:, None]@rowii[None, :]

    def _hessian(self, weights):
        Y = self._Y(weights)
        inv_Y = np.linalg.inv(Y)
        det_Y = np.linalg.det(Y)
        jac_det_Y = (np.array(
            [np.trace(inv_Y@row[:, None]@row[None, :])
             for row in self._model._jac_matrix])*(
                     det_Y*self._prior_cov/self._noise_cov))[None, :]
        hess_det_Y = np.empty((weights.shape[0], weights.shape[0]))
        const = det_Y*(self._prior_cov/self._noise_cov)**2
        for ii in range(weights.shape[0]):
            for jj in range(ii, weights.shape[0]):
                hess_det_Y[ii, jj] = (
                    const *
                    np.trace(self._Y_inv_dYdw(inv_Y, ii)) *
                    np.trace(self._Y_inv_dYdw(inv_Y, jj)) -
                    const *
                    np.trace(self._Y_inv_dYdw(inv_Y, ii) @
                             self._Y_inv_dYdw(inv_Y, jj))
                )
                hess_det_Y[jj, ii] = hess_det_Y[ii, jj]

        hess_log_det_Y = (hess_det_Y/det_Y-jac_det_Y.T@jac_det_Y/det_Y**2)
        return -0.5*hess_log_det_Y
