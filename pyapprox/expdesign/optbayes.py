from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.interface.model import Model
from pyapprox.bayes.likelihood import (
    # IndependentExponentialLogLikelihood,
    IndependentGaussianLogLikelihood,
)
from pyapprox.optimization.minimize import Constraint, SampleAverageMean


class OEDIndependentGaussianLogLikelihood(IndependentGaussianLogLikelihood):
    def __init__(
        self,
        noise_cov_diag: Array,
        many_pred_obs: Array,
        pred_weights: Array,
        tile_obs: bool,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(backend=backend)
        self._many_pred_obs = many_pred_obs
        if pred_weights.shape[0] != many_pred_obs.shape[1]:
            raise ValueError("pred_weights and many_pred_obs are inconsistent")
        self._pred_weights = pred_weights
        self._setup(noise_cov_diag)
        self._set_tile_obs(tile_obs)

    def nqoi(self) -> int:
        if self._tile_obs:
            return self._obs.shape[1] * self._many_pred_obs.shape[1]
        return self._many_pred_obs.shape[1]

    def nvars(self) -> int:
        return self._noise_cov_inv_diag.shape[0]

    def jacobian_implemented(self) -> bool:
        return True

    def _values(self, design_weights: Array) -> Array:
        self.set_design_weights(design_weights)
        return self._loglike(self._many_pred_obs).T

    def _jacobian(self, design_weights: Array) -> Array:
        # stack jacobians for each obs vertically

        # todo next line can be done just done once when objected is created
        # alsocan reduce cost of values if make this happen in
        # self._loglike class
        obs, many_pred_obs = self._parse_obs(self._obs, self._many_pred_obs)
        residual = obs - many_pred_obs
        jac = (residual.T**2) * (
            self._noise_cov_inv_diag[:, 0] * (-0.5)
        ) + 0.5 / design_weights.T
        # second term on rhs is gradient of determinant of weighted
        # noise covariance
        return jac

    def __repr__(self) -> str:
        return "{0}(M={1}, N={2})".format(
            self.__class__.__name__,
            self._obs.shape[1],
            self._many_pred_obs.shape[1],
        )


class Evidence(Model):
    def __init__(
        self,
        loglike: OEDIndependentGaussianLogLikelihood,
    ):
        super().__init__(backend=loglike._bkd)
        if not isinstance(loglike, OEDIndependentGaussianLogLikelihood):
            raise ValueError(
                "loglike must be OEDIndependentGaussianLogLikelihood"
            )

        self._loglike = loglike

        self._prev_design_weights = None
        self._like_vals = None
        self._like_jac = None
        self._weighted_like_vals = None
        self._weighted_like_vals_prod_jac = None
        self._evidence_jac = None

    def nvars(self) -> int:
        return self._loglike.nvars()

    def nqoi(self) -> int:
        return self._loglike._obs.shape[1]

    def jacobian_implemented(self) -> bool:
        return True

    def _reshape_vals(self, vals: Array) -> Array:
        # unflatten vals
        return self._bkd.reshape(
            vals,
            (
                self._loglike._obs.shape[1],
                self._loglike._many_pred_obs.shape[1],
            ),
        ).T

    def _reshape_jacobian(self, jac: Array) -> Array:
        # unflatten jacobian
        return self._bkd.reshape_fortran(
            jac,
            (
                self._loglike._many_pred_obs.shape[1],
                self._loglike._obs.shape[1],
                jac.shape[1],
            ),
        )

    def _values(self, design_weights: Array) -> Array:
        # if self._prev_design_weights is None or not self._bkd.allclose(
        #     design_weights, self._prev_design_weights, atol=1e-15, rtol=1e-15
        # ):
        if True:
            self._prev_design_weights = self._bkd.copy(design_weights)
            self._like_vals = self._reshape_vals(
                self._bkd.exp(self._loglike(design_weights))
            )
            self._weighted_like_vals = (
                self._loglike._pred_weights * self._like_vals
            )
        return (self._weighted_like_vals).sum(axis=0)[None, :]

    def _jacobian(self, design_weights: Array) -> Array:
        if not self._bkd.allclose(
            design_weights, self._prev_design_weights, atol=1e-15, rtol=1e-15
        ):
            # recompute necessary data
            self(design_weights)
        self._like_jac = self._reshape_jacobian(
            self._loglike.jacobian(design_weights)
        )
        self._weighted_like_vals_prod_jac = (
            self._weighted_like_vals[..., None] * self._like_jac
        )
        self._evidence_jac = self._bkd.sum(
            self._weighted_like_vals_prod_jac, axis=0
        )
        return self._evidence_jac

    def __repr__(self) -> str:
        return "{0}(loglike={1})".format(
            self.__class__.__name__, self._loglike
        )


class LogEvidence(Evidence):
    def __call__(self, design_weights: Array) -> Array:
        evidence = super().__call__(design_weights)
        return self._bkd.log(evidence)

    def _jacobian(self, design_weights: Array) -> Array:
        evidence = super().__call__(design_weights)
        jac = super()._jacobian(design_weights)
        jac = 1 / evidence.T * jac
        return jac


class NoiseStatistic:
    def __init__(self, stat):
        self._stat = stat

    def __call__(self, outer_vals, outer_weights):
        return self._stat.__call__(outer_vals, outer_weights)

    def jacobian(self, outer_vals, outer_jacs, outer_weights):
        return self._stat.jacobian(
            outer_vals, outer_jacs[..., None], outer_weights
        ).T


# class EntropicNoiseStatistic(SampleAverageEntropicRisk):
#     def __call__(self, outer_vals, outer_weights):
#         return self._bkd.log((self._bkd.exp(outer_weights)*outer_vals)).sum(axis=0)[:, None]

#     def jacobian(self, outer_vals, outer_jacs, outer_weights):
#         risk = self(outer_vals, outer_weights)
#         return ((outer_weights*outer_vals)*outer_jacs).sum(axis=0)/risk


class KLOEDObjective(Model):
    def __init__(
        self,
        noise_cov_diag: Array,
        outer_pred_obs: Array,
        outer_pred_weights: Array,
        noise_samples: Array,
        inner_pred_obs: Array,
        inner_pred_weights: Array,
        noise_stat: NoiseStatistic = None,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(backend=backend)
        if noise_stat is None:
            noise_stat = NoiseStatistic(SampleAverageMean(self._bkd))

        self._noise_stat = noise_stat
        self._outer_oed_loglike = OEDIndependentGaussianLogLikelihood(
            noise_cov_diag,
            outer_pred_obs,
            outer_pred_weights,
            tile_obs=False,
            backend=self._bkd,
        )
        outer_obs = self._outer_oed_loglike._make_noisy(
            outer_pred_obs, noise_samples
        )
        self._outer_oed_loglike.set_observations(outer_obs)

        self._inner_oed_loglike = OEDIndependentGaussianLogLikelihood(
            noise_cov_diag,
            inner_pred_obs,
            inner_pred_weights,
            tile_obs=True,
            backend=self._bkd,
        )
        self._inner_oed_loglike.set_observations(outer_obs)
        self._log_evidence = LogEvidence(self._inner_oed_loglike)

    def nvars(self) -> int:
        return self._outer_oed_loglike.nvars()

    def nqoi(self) -> int:
        return 1

    def jacobian_implemented(self) -> bool:
        return True

    # apply hessian reduces optimization iteration count but increases
    # run time because cost of each iteration increases so do not activate

    def _values(self, design_weights: Array) -> Array:
        log_evidences = self._log_evidence(design_weights)
        outer_log_like_vals = self._outer_oed_loglike(design_weights)
        outer_weights = self._outer_oed_loglike._pred_weights
        vals = self._noise_stat(
            (outer_log_like_vals - log_evidences).T, outer_weights
        )
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -vals

    def _reshape_jacobian(self, jac: Array) -> Array:
        # unflatten jacobian
        return jac.reshape(self._outer_oed_loglike._obs.shape[1], jac.shape[1])

    def _jacobian(self, design_weights: Array) -> Array:
        log_evidences = self._log_evidence(design_weights)
        outer_log_like_vals = self._outer_oed_loglike(design_weights)
        jac_log_evidences = self._log_evidence.jacobian(design_weights)
        jac_outer_log_like = self._outer_oed_loglike.jacobian(design_weights)
        jac_outer_log_like = self._reshape_jacobian(jac_outer_log_like)
        outer_weights = self._outer_oed_loglike._pred_weights
        jac = self._noise_stat.jacobian(
            outer_log_like_vals - log_evidences,
            jac_outer_log_like - jac_log_evidences,
            outer_weights,
        )
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -jac

    def _hvp1(
        self, outer_weights: Array, evidence: Array, vec: Array
    ) -> Array:
        # g'(f(x))\nabla^2 f^\top \dot v
        # this assumes that hessian of log-likelihood is zero
        # thus this function can only be used with log likelihoods that are
        # linear in the weights
        tmp = self._bkd.sum(
            (
                self._log_evidence._weighted_like_vals_prod_jac
                * (self._log_evidence._like_jac @ vec)
            ),
            axis=0,
        )
        hvp1 = self._bkd.sum((outer_weights / evidence) * tmp, axis=0)
        return hvp1

    def _hvp2(
        self, outer_weights: Array, evidence: Array, vec: Array
    ) -> Array:
        # g''(f(x))\nabla f^\top nabla f \dot v
        evidence_jac = self._log_evidence._evidence_jac
        hvp2 = self._bkd.sum(
            (outer_weights / evidence**2)
            * evidence_jac
            * (evidence_jac @ vec),
            axis=0,
        )
        return hvp2

    def _apply_hessian(self, design_weights: Array, vec: Array) -> Array:
        if not isinstance(self._noise_stat._stat, SampleAverageMean):
            msg = "apply hessian only supported for MeanNoiseStatistic"
            raise ValueError(msg)
        evidence = (
            super(LogEvidence, self._log_evidence).__call__(design_weights).T
        )

        outer_weights = self._outer_oed_loglike._pred_weights
        hvp1 = self._hvp1(outer_weights, evidence, vec)
        hvp2 = self._hvp2(outer_weights, evidence, vec)
        hvp = hvp1 - hvp2
        return hvp[:, None]


class PredictionOEDDeviation(Model):
    def __init__(
        self,
        loglike: OEDIndependentGaussianLogLikelihood,
        qoi_vals: Array,
        qoi_weights: Array,
    ):
        self._qoi_vals = qoi_vals
        self._qoi_weights = qoi_weights
        self._loglike = loglike
        super().__init__(backend=loglike._bkd)

    def __call__(self):
        raise NotImplementedError


class OEDStandardDeviation(PredictionOEDDeviation):
    def _first_momement(self, like_vals: Array) -> Array:
        return (self._qoi_vals * like_vals).sum(axis=1)

    def _second_momement(self, like_vals: Array) -> Array:
        return (self._qoi_vals**2 * like_vals).sum(axis=1)

    def __call__(self, like_vals: Array) -> Array:
        return (
            self._second_moment(like_vals) / evidences
            - self._first_moment(like_vals) / evidences**2
        )

    def _first_momement_jac(self, like_vals: Array) -> Array:
        return (self._qoi_vals * like_vals).sum(axis=1)

    def _second_momement_jac(self, like_vals: Array) -> Array:
        return (self._qoi_vals**2 * like_vals).sum(axis=1)

    def _jacobian(self, like_vals: Array) -> Array:
        return (
            self._second_moment(like_vals) / evidences
            - self._first_moment(like_vals) / evidences**2
        )


class PredictionOEDObjective(KLOEDObjective):
    def __init__(
        self,
        noise_cov_diag: Array,
        outer_pred_obs: Array,
        outer_pred_weights: Array,
        noise_samples: Array,
        inner_pred_obs: Array,
        inner_pred_weights: Array,
        noise_stat: NoiseStatistic = NoiseStatistic(SampleAverageMean()),
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(
            noise_cov_diag,
            outer_pred_obs,
            outer_pred_weights,
            noise_samples,
            inner_pred_obs,
            inner_pred_weights,
            backend=backend,
        )

    def __call__(self, design_weights: Array) -> Array:
        evidences = self._log_evidence._evidence(design_weights)
        deviations = None


class WeightsConstraintModel(Model):
    def jacobian_implemented(self) -> bool:
        return True

    def __call__(self, weights: Array) -> Array:
        assert self._bkd.all(weights >= 0)
        return weights.sum(axis=0)[:, None]

    def _jacobian(self, weights: Array) -> Array:
        assert self._bkd.all(weights >= 0)
        return self._bkd.ones((1, weights.shape[0]))


class WeightsConstraint(Constraint):
    def __init__(
        self,
        nobs: int,
        keep_feasible: bool = False,
        backend: BackendMixin = NumpyMixin,
    ):
        model = WeightsConstraintModel()
        bounds = self._bkd.array([[nobs, nobs]])
        super().__init__(model, bounds, keep_feasible)


class SparseOEDObjective(Model):
    def __init__(self, objective: Model, penalty: float):
        super().__init__(backend=objective._bkd)
        self._objective = objective
        self._penalty = penalty

    def jacobian_implemented(self) -> bool:
        return self._objective.jacobian_implemented()

    def _l1norm(self, weights: Array) -> float:
        # assumes weights are positive
        return self._bkd.sum(weights)

    def _l1norm_jac(self, weights: Array) -> Array:
        # assumes weights are positive
        return self._bkd.ones((1, weights.shape[0]))

    def __call__(self, weights: Array) -> Array:
        assert self._bkd.all(weights >= 0)
        # neagtive penalty because we are minimizing negative KL divergence
        return self._objective(weights) - self._penalty * self._l1norm(weights)

    def _jacobian(self, weights: Array) -> Array:
        assert self._bkd.all(weights >= 0)
        # neagtive penalty because we are minimizing negative KL divergence
        return self._objective.jacobian(
            weights
        ) - self._penalty * self._l1norm_jac(weights)


class DOptimalLinearModelObjective(Model):
    def __init__(self, model: Model, noise_cov: Array, prior_cov: Array):
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
        super().__init__(backend=model._bkd)
        if not self._bkd.isscalar(noise_cov):
            raise ValueError("noise_cov must be a scalar")
        if not self._bkd.isscalar(prior_cov):
            raise ValueError("prior_cov must be a scalar")
        self._model = model
        self._noise_cov = noise_cov
        self._prior_cov = prior_cov

    def jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def __call__(self, weights: Array) -> Array:
        Amat = self._model._jac_matrix
        nvars = Amat.shape[1]
        hess_misfit = (
            Amat.T.dot(weights * Amat) * self._prior_cov / self._noise_cov
        )
        ident = self._bkd.eye(nvars)
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -self._bkd.array(
            [0.5 * self._bkd.linalg.slogdet(hess_misfit + ident)[1]]
        )[:, None]

    def _Y(self, weights: Array) -> Array:
        Amat = self._model._jac_matrix
        nvars = Amat.shape[1]
        hess_misfit = (
            Amat.T.dot(weights * Amat) * self._prior_cov / self._noise_cov
        )
        ident = self._bkd.eye(nvars)
        Y = hess_misfit + ident
        return Y

    def _jacobian(self, weights: Array) -> Array:
        Y = self._Y(weights)
        inv_Y = self._bkd.linalg.inv(Y)
        jac_log_det_Y = self._bkd.array(
            [
                self._bkd.trace(inv_Y @ row[:, None] @ row[None, :])
                for row in self._model._jac_matrix
            ]
        ) * (self._prior_cov / self._noise_cov)
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -0.5 * jac_log_det_Y

    def _Y_inv_dYdw(self, inv_Y: Array, ii: int) -> Array:
        rowii = self._model._jac_matrix[ii]
        return inv_Y @ rowii[:, None] @ rowii[None, :]

    def _hessian(self, weights: Array) -> Array:
        Y = self._Y(weights)
        inv_Y = self._bkd.linalg.inv(Y)
        det_Y = self._bkd.linalg.det(Y)
        jac_det_Y = (
            self._bkd.array(
                [
                    self._bkd.trace(inv_Y @ row[:, None] @ row[None, :])
                    for row in self._model._jac_matrix
                ]
            )
            * (det_Y * self._prior_cov / self._noise_cov)
        )[None, :]
        hess_det_Y = self._bkd.empty((weights.shape[0], weights.shape[0]))
        const = det_Y * (self._prior_cov / self._noise_cov) ** 2
        for ii in range(weights.shape[0]):
            for jj in range(ii, weights.shape[0]):
                hess_det_Y[ii, jj] = const * self._bkd.trace(
                    self._Y_inv_dYdw(inv_Y, ii)
                ) * self._bkd.trace(
                    self._Y_inv_dYdw(inv_Y, jj)
                ) - const * self._bkd.trace(
                    self._Y_inv_dYdw(inv_Y, ii) @ self._Y_inv_dYdw(inv_Y, jj)
                )
                hess_det_Y[jj, ii] = hess_det_Y[ii, jj]

        hess_log_det_Y = (
            hess_det_Y / det_Y - jac_det_Y.T @ jac_det_Y / det_Y**2
        )
        return -0.5 * hess_log_det_Y
