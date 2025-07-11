from abc import ABC, abstractmethod
from typing import List

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.interface.model import Model
from pyapprox.bayes.likelihood import (
    LogLikelihood,
    IndependentGaussianLogLikelihood,
)
from pyapprox.optimization.minimize import (
    Constraint,
    SampleAverageMean,
    ConstrainedOptimizer,
    LinearConstraint,
    OptimizationResult,
)
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer


class OEDOuterLoopLogLikelihoodMixin(ABC):
    """
    Wrap Likelihood function so that it is a function of the design weights
    and not a function of the likelihood shapes (or parameters of a model
    that predict the shapes)
    """

    def set_observations_and_shapes(self, obs: Array, shapes: Array):
        # Unlike likelihoods from pyapprox.bayes.likelihood, which take
        # obs and shapes with different shapes, obs and shapes passed to
        # OED likelihoods require obs and shapes with the same shape
        if obs.shape != shapes.shape:
            raise ValueError(f"{obs.shape=} does not match {shapes.shape=}")
        self.set_observations(obs)
        self._shapes = shapes

    def nqoi(self) -> int:
        return self._obs.shape[1]

    def nvars(self) -> int:
        return self._obs.shape[0]

    @abstractmethod
    def _values(self, design_weights: Array) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        if self._obs is not None:
            return "{0}(nobs={1}, nsamples={2})".format(
                self.__class__.__name__,
                self.nobs(),
                self._obs.shape[1],
            )
        return "{0}".format(self.__class__.__name__)

    def rvs_from_shapes(self, shapes: Array) -> Array:
        """
        Generate observations based on the
        likelihood distribution's shapes,
        i.e. predictions of models at some samples.

        This function is useful for OED

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs : Array (nsamples, nobs)
            Generated observations.
        """
        return super()._rvs(shapes)


class IndependentGaussianOEDOuterLoopLogLikelihood(
    OEDOuterLoopLogLikelihoodMixin, IndependentGaussianLogLikelihood
):
    def set_observations_and_shapes(self, obs: Array, shapes: Array):
        super().set_observations_and_shapes(obs, shapes)
        self._residuals = self._obs - shapes

    def _values(self, design_weights: Array) -> Array:
        self.set_design_weights(design_weights)
        # OED assumes all observations have one experiment so that dimension
        # is removed. But
        # loglikelihoods do not. So expand obs to have
        # shape (nobs, nexperiments, nsamples), where nexperiments=1
        return self._loglike_from_residuals(self._residuals[:, None, :]).T

    def jacobian_implemented(self) -> bool:
        return True

    def _jacobian(self, design_weights: Array) -> Array:
        # stack jacobians for each obs vertically

        # todo next line can be done just done once when objected is created
        # alsocan reduce cost of values if make this happen in
        # self._loglike class
        jac = (self._residuals.T**2) * (
            self._noise_cov_inv_diag[:, 0] * (-0.5)
        ) + 0.5 / design_weights.T
        # second term on rhs is gradient of determinant of weighted
        # noise covariance
        return jac


class OEDInnerLoopLogLikelihoodMixin:
    def set_observations_and_shapes(self, obs: Array, shapes: Array):
        # Unlike likelihoods from pyapprox.bayes.likelihood, which take
        # obs and shapes with different shapes, obs and shapes passed to
        # OED likelihoods require obs and shapes with the same shape
        if obs.ndim != 2:
            raise ValueError("obs must be a 2D array (nobs, nouterloop)")
        if shapes.ndim != 2:
            raise ValueError("shapes must be a 2D array (nobs, ninnerloop)")
        if obs.shape[0] != shapes.shape[0]:
            raise ValueError(
                "The number of rows of obs and shapes are inconsistent"
            )
        self._obs = obs
        self._shapes = shapes

    def nqoi(self) -> int:
        return self._obs.shape[1] * self._shapes.shape[1]

    def nvars(self) -> int:
        return self._obs.shape[0]

    @abstractmethod
    def _values(self, design_weights: Array) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        if self._obs is not None:
            return "{0}(nobs={1}, nouterloop={2}, nsamples={3})".format(
                self.__class__.__name__,
                self.nobs(),
                self._obs.shape[1],
                self._shapes.shape[1],
            )
        return "{0}".format(self.__class__.__name__)

    @abstractmethod
    def _setup_outerloop_loglike(self) -> OEDOuterLoopLogLikelihoodMixin:
        """
        Setup the log likelihood function for the outerloop which
        operates on obs and shapes of different size to innterloop likelihood.

        This function is required to ensure that the user supplies
        consistent innerloop and outerloop log likelihoods
        """
        raise NotImplementedError

    def outerloop_loglike(self) -> OEDOuterLoopLogLikelihoodMixin:
        """
        Return the log likelihood function for the outerloop which
        operates on obs and shapes of different size to innterloop likelihood.

        This function is required to ensure that the user supplies
        consistent innerloop and outerloop log likelihoods
        """
        if not hasattr(self, "_outerloop_loglike"):
            self._outerloop_loglike = self._setup_outerloop_loglike()
        return self._outerloop_loglike


class IndependentGaussianOEDInnerLoopLogLikelihood(
    OEDInnerLoopLogLikelihoodMixin, IndependentGaussianLogLikelihood
):
    def set_observations_and_shapes(self, obs: Array, shapes: Array):
        super().set_observations_and_shapes(obs, shapes)
        self._residuals = self._obs[..., None] - shapes[:, None, :]

    def _noise_cov_sqrt_inv_apply(self, vecs: Array) -> Array:
        # vecs is a 3D tensor (nobs, ninner_samples, nouterloop_samples)
        return self._bkd.sqrt(self._wnoise_cov_inv_diag)[..., None] * vecs

    def _values(self, design_weights: Array) -> Array:
        # IndependentGaussianLogLikelihood should not know about OED need
        # to compute log-liklihood values for multiple observations
        # (not to be confused with experiments). So implement custom function
        # here
        self.set_design_weights(design_weights)
        L_inv_res = self._noise_cov_sqrt_inv_apply(self._residuals)
        # compute diagonals for each outerloop obs
        # need to compute transpose of L_inv_res[ii] for each row ii
        # so do this by chainging first string argument of einsum.
        # e.g. for 2D A, A einsum("jk,kj->j", A.T, A)
        # is equal toeinsum("jk,jk->j", A, A)
        vals = self._bkd.einsum("ijk,ijk->jk", L_inv_res, L_inv_res).flatten()
        # the flatten stacks the jacobian (ninnerloop)
        # for each outerloop sample
        return -0.5 * vals[None, :] + self._loglike_const

    def jacobian_implemented(self) -> bool:
        return True

    def _jacobian(self, design_weights: Array) -> Array:
        # stack jacobians for each obs vertically

        # todo next line can be done just done once when objected is created
        # alsocan reduce cost of values if make this happen in
        # self._loglike class
        # Compute jacobian of quadratic component of likelihood
        jac = -0.5 * self._noise_cov_inv_diag[..., None] * self._residuals**2
        # reorder axes so 3D jacobian tensor has the correct shape
        jac = self._bkd.swapaxes(self._bkd.swapaxes(jac, 0, 1), 1, 2)
        # reshape to be 2D
        jac = self._bkd.reshape(jac, (self.nqoi(), self.nobs()))
        # add gradient of determinant of weighted
        # noise covariance
        jac += 0.5 / design_weights.T
        return jac

    def _setup_outerloop_loglike(self) -> OEDOuterLoopLogLikelihoodMixin:
        return IndependentGaussianOEDOuterLoopLogLikelihood(
            self._noise_cov_diag, backend=self._bkd
        )


class Evidence(Model):
    """
    Compute evidence for different realizations of the observational data
    simultaneously.
    """

    def __init__(
        self,
        loglike: OEDInnerLoopLogLikelihoodMixin,
        quad_weights: Array = None,
    ):
        super().__init__(backend=loglike._bkd)
        if not isinstance(loglike, OEDInnerLoopLogLikelihoodMixin):
            raise ValueError("loglike must be OEDLogLikelihoodMixin")

        self._loglike = loglike
        self._quad_weights = quad_weights

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
                self._loglike._shapes.shape[1],
            ),
        ).T

    def _values(self, design_weights: Array) -> Array:
        self._prev_design_weights = self._bkd.copy(design_weights)
        like_vals = self._reshape_vals(
            self._bkd.exp(self._loglike(design_weights))
        )
        self._like_vals = like_vals
        if self._quad_weights is None:
            quad_weights = 1.0 / self._like_vals.shape[0]
        else:
            quad_weights = self._quad_weights
        self._weighted_like_vals = quad_weights * self._like_vals
        return (self._weighted_like_vals).sum(axis=0)[None, :]

    def _jacobian(self, design_weights: Array) -> Array:
        if not self._bkd.allclose(
            design_weights, self._prev_design_weights, atol=1e-15, rtol=1e-15
        ):
            # recompute necessary data
            self(design_weights)

        self._like_jac = self._bkd.reshape(
            self._loglike.jacobian(design_weights),
            (
                self._loglike._obs.shape[1],
                self._loglike._shapes.shape[1],
                self._loglike.nvars(),
            ),
        )
        self._weighted_like_vals_prod_jac = (
            self._weighted_like_vals.T[..., None] * self._like_jac
        )
        self._evidence_jac = self._bkd.sum(
            self._weighted_like_vals_prod_jac, axis=1
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

    def __repr__(self) -> str:
        return "{0}({1})".format(self.__class__.__name__, self._stat)


# class EntropicNoiseStatistic(SampleAverageEntropicRisk):
#     def __call__(self, outer_vals, outer_weights):
#         return self._bkd.log((self._bkd.exp(outer_weights)*outer_vals)).sum(axis=0)[:, None]

#     def jacobian(self, outer_vals, outer_jacs, outer_weights):
#         risk = self(outer_vals, outer_weights)
#         return ((outer_weights*outer_vals)*outer_jacs).sum(axis=0)/risk


class BayesianOEDObjective(Model):
    def nqoi(self) -> int:
        return 1


class KLOEDObjective(BayesianOEDObjective):
    def __init__(
        self,
        innerloop_loglike: OEDInnerLoopLogLikelihoodMixin,
        outerloop_shapes: Array,
        outerloop_quad_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_shapes: Array,
        innerloop_quad_weights: Array,
        noise_stat: NoiseStatistic = None,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(backend=backend)
        if noise_stat is None:
            noise_stat = NoiseStatistic(SampleAverageMean(self._bkd))
        self._noise_stat = noise_stat

        if not isinstance(innerloop_loglike, OEDInnerLoopLogLikelihoodMixin):
            raise ValueError(
                "innerloop_loglike must be a OEDInnerLoopLogLikelihoodMixin"
            )
        self._innerloop_loglike = innerloop_loglike

        self._outerloop_loglike = innerloop_loglike.outerloop_loglike()
        obs = self._outerloop_loglike._rvs_from_likelihood_samples(
            outerloop_shapes,
            outerloop_quad_samples[-outerloop_shapes.shape[0] :],
        )
        self._outerloop_loglike.set_observations_and_shapes(
            obs, outerloop_shapes
        )
        self._innerloop_loglike.set_observations_and_shapes(
            obs, innerloop_shapes
        )
        self._log_evidence = LogEvidence(
            self._innerloop_loglike, innerloop_quad_weights
        )
        self._set_quadrature_weights(
            outerloop_quad_weights, innerloop_quad_weights
        )

    def _set_quadrature_weights(
        self, outerloop_quad_weights: Array, innerloop_quad_weights: Array
    ):
        nouterloop_samples = self._outerloop_loglike._shapes.shape[1]
        if outerloop_quad_weights is None:
            outerloop_quad_weights = self._bkd.full(
                (nouterloop_samples, 1),
                1.0 / nouterloop_samples,
            )

        ninnerloop_samples = self._innerloop_loglike._shapes.shape[1]
        if innerloop_quad_weights is None:
            innerloop_quad_weights = self._bkd.full(
                (ninnerloop_samples, 1),
                1.0 / ninnerloop_samples,
            )

        if outerloop_quad_weights.shape != (nouterloop_samples, 1):
            raise ValueError(
                "outerloop_quad_weights and outerloop_shapes are inconsistent"
            )
        self._outerloop_quad_weights = outerloop_quad_weights

        if innerloop_quad_weights.shape != (ninnerloop_samples, 1):
            raise ValueError(
                "innerloop_quad_weights and innerloop_shapes are inconsistent"
            )
        self._innerloop_quad_weights = innerloop_quad_weights

    def jacobian_implemented(self) -> bool:
        return True

    def nvars(self) -> int:
        return self._outerloop_loglike.nvars()

    # apply hessian reduces optimization iteration count but increases
    # run time because cost of each iteration increases so do not activate
    # TODO use adjoints to derive more efficient HVP
    def apply_hessian_implemented(self) -> bool:
        return False

    def _values(self, design_weights: Array) -> Array:
        log_evidences = self._log_evidence(design_weights)
        outer_log_like_vals = self._outerloop_loglike(design_weights)
        vals = self._noise_stat(
            (outer_log_like_vals - log_evidences).T,
            self._outerloop_quad_weights,
        )
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -vals

    def _reshape_jacobian(self, jac: Array) -> Array:
        # unflatten jacobian
        return jac.reshape(self._outerloop_loglike._obs.shape[1], jac.shape[1])

    def _jacobian(self, design_weights: Array) -> Array:
        log_evidences = self._log_evidence(design_weights)
        outer_log_like_vals = self._outerloop_loglike(design_weights)
        jac_log_evidences = self._log_evidence.jacobian(design_weights)
        jac_outer_log_like = self._outerloop_loglike.jacobian(design_weights)
        jac_outer_log_like = self._reshape_jacobian(jac_outer_log_like)
        outer_weights = self._outerloop_quad_weights
        jac = self._noise_stat.jacobian(
            (outer_log_like_vals - log_evidences).T,
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
            axis=1,
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

        outer_weights = self._outerloop_quad_weights
        hvp1 = self._hvp1(outer_weights, evidence, vec)
        hvp2 = self._hvp2(outer_weights, evidence, vec)
        hvp = hvp1 - hvp2
        return hvp[:, None]


class PredictionOEDDeviation(Model):
    def __init__(
        self,
        loglike: OEDOuterLoopLogLikelihoodMixin,
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
    raise NotImplementedError

    def __call__(self, design_weights: Array) -> Array:
        evidences = self._log_evidence._evidence(design_weights)
        deviations = None


class DOptimalLinearModelObjective(BayesianOEDObjective):
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
        if noise_cov.ndim != 0:
            raise ValueError("noise_cov must be a scalar")
        if prior_cov.ndim != 0:
            raise ValueError("prior_cov must be a scalar")
        self._model = model
        self._noise_cov = noise_cov
        self._prior_cov = prior_cov

    def nvars(self) -> int:
        return self._model.nqoi()

    def jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def _values(self, weights: Array) -> Array:
        Amat = self._model._matrix
        nvars = Amat.shape[1]
        hess_misfit = (
            Amat.T @ (weights * Amat) * self._prior_cov / self._noise_cov
        )
        ident = self._bkd.eye(nvars)
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -self._bkd.array(
            [0.5 * self._bkd.slogdet(hess_misfit + ident)[1]]
        )[:, None]

    def _Y(self, weights: Array) -> Array:
        Amat = self._model._matrix
        nvars = Amat.shape[1]
        hess_misfit = (
            Amat.T @ (weights * Amat) * self._prior_cov / self._noise_cov
        )
        ident = self._bkd.eye(nvars)
        Y = hess_misfit + ident
        return Y

    def _jacobian(self, weights: Array) -> Array:
        Y = self._Y(weights)
        inv_Y = self._bkd.inv(Y)
        jac_log_det_Y = self._bkd.array(
            [
                self._bkd.trace(inv_Y @ row[:, None] @ row[None, :])
                for row in self._model._matrix
            ]
        ) * (self._prior_cov / self._noise_cov)
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return (-0.5 * jac_log_det_Y)[None, :]

    def _Y_inv_dYdw(self, inv_Y: Array, ii: int) -> Array:
        rowii = self._model._matrix[ii]
        return inv_Y @ rowii[:, None] @ rowii[None, :]

    def _hessian(self, weights: Array) -> Array:
        Y = self._Y(weights)
        inv_Y = self._bkd.inv(Y)
        det_Y = self._bkd.det(Y)
        jac_det_Y = (
            self._bkd.array(
                [
                    self._bkd.trace(inv_Y @ row[:, None] @ row[None, :])
                    for row in self._model._matrix
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
        return (-0.5 * hess_log_det_Y)[None, ...]


class BayesianOED:
    def __init__(self, objective: BayesianOEDObjective):
        if not isinstance(objective, BayesianOEDObjective):
            raise ValueError(
                "objective must be an instance of BayesianOEDObjective"
            )
        self._objective = objective
        self._bkd = objective._bkd

    def default_optimizer(
        self,
        ncandidates: int = 1,
        verbosity: int = 0,
        gtol: float = 1e-8,
        maxiter: int = 1000,
        method: str = "trust-constr",
    ) -> ConstrainedOptimizer:
        local_optimizer = ScipyConstrainedOptimizer()
        local_optimizer.set_options(
            gtol=gtol,
            maxiter=maxiter,
            method=method,
        )
        local_optimizer.set_verbosity(verbosity)
        return local_optimizer

    def _constraints(self) -> List[Constraint]:
        # ensure sum of weights == 1
        return [
            LinearConstraint(
                self._bkd.ones((1, self._objective.nvars())),
                1.0,
                1.0,
                keep_feasible=True,
            )
        ]

    def set_optimizer(self, optimizer: ConstrainedOptimizer):
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                "optimizer must be an instance of ConstrainedOptimizer"
            )
        self._optimizer = optimizer
        self._optimizer.set_objective_function(self._objective)
        self._optimizer.set_constraints(self._constraints())
        self._optimizer.set_bounds(
            self._bkd.stack(
                (
                    self._bkd.zeros(self._objective.nvars()),
                    self._bkd.ones(self._objective.nvars()),
                ),
                axis=1,
            )
        )

    def compute(self, iterate: Array = None) -> Array:
        iterate = self._bkd.full(
            (self._objective.nvars(), 1), 1 / self._objective.nvars()
        )
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        self._res = self._optimizer.minimize(iterate)
        return self._res.x

    def optimization_result(self) -> OptimizationResult:
        return self._res
