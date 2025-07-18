from abc import ABC, abstractmethod
from typing import List

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.interface.model import Model, SingleSampleModel
from pyapprox.bayes.likelihood import IndependentGaussianLogLikelihood
from pyapprox.optimization.minimize import (
    Constraint,
    SampleAverageMean,
    ConstrainedOptimizer,
    LinearConstraint,
    OptimizationResult,
    SampleAverageStat,
)
from pyapprox.variables.gaussian import MultivariateGaussian
from pyapprox.bayes.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    GaussianPushForward,
    _compute_expected_kl_divergence,
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

    def shapes(self) -> Array:
        """
        Return the shapes, e.g. mean of Gaussian, used to compute the
        likelihood

        Returns
        -------
        shapes : Array (nobs, nouterloop_samples)
        """
        return self._shapes

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
            raise ValueError(
                "obs must be a 2D array (nobs, nouterloop_samples)"
            )
        if shapes.ndim != 2:
            raise ValueError(
                "shapes must be a 2D array (nobs, ninnerloop_samples)"
            )
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

    def shapes(self) -> Array:
        """
        Return the shapes, e.g. mean of Gaussian, used to compute the
        likelihood

        Returns
        -------
        shapes : Array (nobs, ninnerloop_samples)
        """
        return self._shapes

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
        self.set_quadrature_weights(quad_weights)

        self._prev_design_weights = None
        self._like_vals = None
        self._like_jac = None
        self._quad_weighted_like_vals = None
        self._quad_weighted_like_vals_prod_jac = None
        self._evidence_jac = None

    def set_quadrature_weights(self, quad_weights: Array):
        if quad_weights is None:
            quad_weights = 1.0 / self._loglike.nobs()
        if quad_weights.shape != (self._loglike._shapes.shape[1], 1):
            raise ValueError(
                f"quad_weights has shape {quad_weights.shape} "
                f"but should have shape {(self._loglike.nobs(), 1)}"
            )

        self._quad_weights = quad_weights

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
        self._quad_weighted_like_vals = self._quad_weights * self._like_vals
        # k is needed to ensure that quadweights is applied to each column
        # in like_vals via broadcast
        # o: outer, i: inner
        return self._bkd.einsum("ok,oi->i", self._quad_weights, like_vals)[
            None, :
        ]
        # return (self._quad_weighted_like_vals).sum(axis=0)[None, :]

    def _quad_weighted_likelihood_jacobian(
        self, design_weights: Array
    ) -> Array:
        # jacobian of exp(loglike) with respect to design weights
        self._like_jac = self._bkd.reshape(
            self._loglike.jacobian(design_weights),
            (
                self._loglike._obs.shape[1],
                self._loglike._shapes.shape[1],
                self._loglike.nvars(),
            ),
        )
        self._quad_weighted_like_vals_prod_jac = (
            self._quad_weighted_like_vals.T[..., None] * self._like_jac
        )
        return self._quad_weighted_like_vals_prod_jac

    def _jacobian(self, design_weights: Array) -> Array:
        like_jac = self._bkd.reshape(
            self._loglike.jacobian(design_weights),
            (
                self._loglike._obs.shape[1],
                self._loglike._shapes.shape[1],
                self._loglike.nvars(),
            ),
        )
        like_vals = self._reshape_vals(
            self._bkd.exp(self._loglike(design_weights))
        )
        quad_weighted_like_vals = self._quad_weights * like_vals
        # return self._bkd.sum(
        #     quad_weighted_like_vals.T[..., None] * like_jac, axis=1
        # )
        # o:outer, i: inner, d: derivatives
        self._evidence_jac = self._bkd.einsum(
            "io, oik -> ok", quad_weighted_like_vals, like_jac
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


class BayesianOEDObjective(SingleSampleModel):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(backend=backend)

    def set_design_weights_map(self, design_weights_map: Array):
        if design_weights_map.shape != (self.nobs(),):
            raise ValueError(
                f"design_weights must have shape {(self.nobs(),)}"
            )
        self._design_weights_map = self._bkd.asarray(
            design_weights_map, dtype=int
        )
        self._nunique_design_weights = self._bkd.unique(
            design_weights_map
        ).shape[0]
        self._design_weights_map_jacobian = self._bkd.zeros(
            (self.nobs(), self._nunique_design_weights)
        )
        for ii in range(self.nobs()):
            self._design_weights_map_jacobian[
                ii, self._design_weights_map[ii]
            ] = 1.0

    def nvars(self) -> int:
        if not hasattr(self, "_design_weights_map"):
            return self.nobs()
        return self._nunique_design_weights

    def nqoi(self) -> int:
        return 1

    @abstractmethod
    def nobs(self) -> int:
        raise NotImplementedError

    def _expand_design_weights(self, design_weights: Array) -> Array:
        if not hasattr(self, "_design_weights_map"):
            return design_weights
        return design_weights[self._design_weights_map]

    def _evaluate(self, design_weights: Array) -> Array:
        return self._evaluate_from_expanded_design_weights(
            self._expand_design_weights(design_weights)
        )

    def _jacobian(self, design_weights: Array) -> Array:
        jac = self._jacobian_from_expanded_design_weights(
            self._expand_design_weights(design_weights)
        )
        if not hasattr(self, "_design_weights_map"):
            return jac
        return jac @ self._design_weights_map_jacobian


class KLOEDObjective(BayesianOEDObjective):
    def __init__(
        self,
        innerloop_loglike: OEDInnerLoopLogLikelihoodMixin,
        outerloop_shapes: Array,
        outerloop_quad_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_shapes: Array,
        innerloop_quad_weights: Array,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(backend=backend)
        # set default noise statistic
        noise_stat = NoiseStatistic(SampleAverageMean(self._bkd))
        self.set_noise_statistic(noise_stat)

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
        self._innerloop_quad_weights = innerloop_quad_weights
        self._setup_evidence()
        self._set_quadrature_weights(
            outerloop_quad_weights, innerloop_quad_weights
        )

    def set_noise_statistic(self, noise_stat: NoiseStatistic):
        if not isinstance(noise_stat, NoiseStatistic):
            raise ValueError(
                "noise_stat must be an instance of NoiseStatistic"
            )
        self._noise_stat = noise_stat

    def _setup_evidence(self):
        self._log_evidence = LogEvidence(
            self._innerloop_loglike, self._innerloop_quad_weights
        )

    def nobs(self) -> int:
        return self._outerloop_loglike.nobs()

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

    def apply_hessian_implemented(self) -> bool:
        return True

    def _evaluate_from_expanded_design_weights(
        self, design_weights: Array
    ) -> Array:
        log_evidences = self._log_evidence(design_weights)
        outer_loglike_vals = self._outerloop_loglike(design_weights)
        vals = self._noise_stat(
            (outer_loglike_vals - log_evidences).T,
            self._outerloop_quad_weights,
        )
        # return negative because we want to maximize KL divergence
        # which is equivalent to minimizing the negative KL divergence
        return -vals

    def _reshape_jacobian(self, jac: Array) -> Array:
        # unflatten jacobian
        return jac.reshape(self._outerloop_loglike._obs.shape[1], jac.shape[1])

    def _jacobian_from_expanded_design_weights(
        self, design_weights: Array
    ) -> Array:
        log_evidences = self._log_evidence(design_weights)
        outer_loglike_vals = self._outerloop_loglike(design_weights)
        jac_log_evidences = self._log_evidence.jacobian(design_weights)
        jac_outer_loglike = self._outerloop_loglike.jacobian(design_weights)
        jac_outer_loglike = self._reshape_jacobian(jac_outer_loglike)
        outer_weights = self._outerloop_quad_weights
        jac = self._noise_stat.jacobian(
            (outer_loglike_vals - log_evidences).T,
            jac_outer_loglike - jac_log_evidences,
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
        # tmp = self._bkd.sum(
        #     (
        #         self._log_evidence._quad_weighted_like_vals_prod_jac
        #         * (self._log_evidence._like_jac @ vec)
        #     ),
        #     axis=1,
        # )
        # o:outer, i: inner, q: qoi, d:derivatives
        tmp = self._bkd.einsum(
            "oid,oi->od",
            self._log_evidence._quad_weighted_like_vals_prod_jac,
            (self._log_evidence._like_jac @ vec[:, 0]),
        )
        hvp1 = self._bkd.sum((outer_weights / evidence) * tmp, axis=0)
        return hvp1

    def _hvp2(
        self, outer_weights: Array, evidence: Array, vec: Array
    ) -> Array:
        # g''(f(x))\nabla f^\top nabla f \dot v
        evidence_jac = self._log_evidence._evidence_jac
        # o:outer, i: inner, q: qoi, d:derivatives
        hvp2 = self._bkd.sum(
            (outer_weights / evidence**2)
            * evidence_jac
            * (evidence_jac @ vec),
            axis=0,
        )
        return hvp2

    def _apply_hessian(self, design_weights: Array, vec: Array) -> Array:
        design_weights = self._expand_design_weights(design_weights)
        if not isinstance(self._noise_stat._stat, SampleAverageMean):
            msg = "apply hessian only supported for MeanNoiseStatistic"
            raise ValueError(msg)
        evidence = (
            super(LogEvidence, self._log_evidence).__call__(design_weights).T
        )

        outer_weights = self._outerloop_quad_weights
        if hasattr(self, "_design_weights_map"):
            vec = self._design_weights_map_jacobian @ vec

        self._log_evidence._quad_weighted_likelihood_jacobian(design_weights)
        hvp1 = self._hvp1(outer_weights, evidence, vec)
        hvp2 = self._hvp2(outer_weights, evidence, vec)
        hvp = hvp1 - hvp2
        if hasattr(self, "_design_weights_map"):
            hvp = self._design_weights_map_jacobian.T @ hvp
        return hvp[:, None]


class PredictionOEDDeviationMeasure(SingleSampleModel):
    def __init__(self, npred: int, backend: BackendMixin = NumpyMixin):
        super().__init__(backend=backend)
        self._npred = npred

    def set_loglikelihood(
        self,
        innerloop_loglike: OEDInnerLoopLogLikelihoodMixin,
    ):
        if not isinstance(innerloop_loglike, OEDInnerLoopLogLikelihoodMixin):
            raise ValueError(
                "loglike must be an instance of OEDInnerLoopLogLikelihoodMixin"
            )
        if not innerloop_loglike._bkd.bkd_equal(
            self._bkd, innerloop_loglike._bkd
        ):
            raise ValueError("backends are inconsistent")
        self._innerloop_loglike = innerloop_loglike
        self._outerloop_loglike = self._innerloop_loglike._outerloop_loglike
        self._nouterloop_samples = self._outerloop_loglike._shapes.shape[1]
        self._ninnerloop_samples = self._innerloop_loglike._shapes.shape[1]

    def set_data(self, qoi_vals: Array, qoi_weights: Array):
        """
        Parameters
        ----------
        qoi_vals : Array (ninner_samples, npred)
            The qoi values at the inner loop samples
        qoi_weights : Array (npred, 1)
            The quadrature weights over the prediction space
        """
        if qoi_vals.shape != (self._ninnerloop_samples, self.npred()):
            raise ValueError(
                "qoi_vals must have shape "
                "{0} but had shape {1}".format(
                    (self._ninnerloop_samples, self.npred()), qoi_vals.shape
                )
            )

        self._qoi_vals = qoi_vals
        if qoi_weights.shape != (self.npred(), 1):
            raise ValueError(
                "qoi_weights must have shape "
                "{0} but had shape {1}".format(
                    (self.npred(), 1), qoi_weights.shape
                )
            )
        self._qoi_weights = qoi_weights
        self._weighted_qoi_vals = qoi_weights * qoi_vals[:, 0]
        self._npred = self._qoi_vals.shape[1]

    def nqoi(self) -> int:
        # models can only return 2D array so we must flatten deviation measures
        # for each qoi and each outerloop observation into a 1D
        return self._nouterloop_samples * self.npred()

    def npred(self) -> int:
        # return the number of predicted quantities of interest.
        # this is different to nqoi, which just dictates the amount of columns
        # returned by __call__
        return self._npred

    def nvars(self) -> int:
        return self._innerloop_loglike.nvars()

    def set_evidence(self, evidence: Evidence):
        if not isinstance(evidence, Evidence):
            raise ValueError("evidence must be an instance of Evidence")
        self._evidence = evidence


class OEDStandardDeviationMeasure(PredictionOEDDeviationMeasure):
    def _first_moment(self, quad_weighted_like_vals: Array) -> Array:
        # after reshape the 3D arrays will have shape
        # (npred, ninnerloop_samples, nouterloop_samples)
        # return (
        #    self._qoi_vals.T[..., None] * quad_weighted_like_vals[None, ...]
        # ).sum(axis=1)
        # o:outer, i: inner, q: qoi
        return self._bkd.einsum(
            "iq,io->qo", self._qoi_vals, quad_weighted_like_vals
        )

    def _second_moment(self, quad_weighted_like_vals: Array) -> Array:
        # return (
        #     self._qoi_vals.T[..., None] ** 2
        #     * quad_weighted_like_vals[None, ...]
        # ).sum(axis=1)
        # o:outer, i: inner, q: qoi,
        return self._bkd.einsum(
            "iq,io->qo", self._qoi_vals**2, quad_weighted_like_vals
        )

    def _evaluate(self, design_weights: Array) -> Array:
        evidences = self._evidence(design_weights).T
        return self._bkd.sqrt(
            self._second_moment(self._evidence._quad_weighted_like_vals)
            / evidences[:, 0]
            - self._first_moment(self._evidence._quad_weighted_like_vals) ** 2
            / evidences[:, 0] ** 2
        ).flatten()[None, :]

    def _first_moment_jac(self, quad_weighted_like_vals_jac: Array) -> Array:
        # after reshape 4D arrays will have
        # (npred, nouterloop_samples, ninnerloop_samples, ndesign))
        # return (
        #     self._qoi_vals.T[:, None, :, None]
        #     * quad_weighted_like_vals_jac[None, ...]
        # ).sum(axis=2)
        # o:outer, i: inner, q: qoi, d: derivatives
        return self._bkd.einsum(
            "iq,oid->qod",
            self._qoi_vals,
            quad_weighted_like_vals_jac,
        )

    def _second_moment_jac(self, quad_weighted_like_vals_jac: Array) -> Array:
        # return (
        #     self._qoi_vals.T[:, None, :, None] ** 2
        #     * quad_weighted_like_vals_jac[None, ...]
        # ).sum(axis=2)
        # o:outer, i: inner, q: qoi, d: derivatives
        return self._bkd.einsum(
            "iq,oid->qod",
            self._qoi_vals**2,
            quad_weighted_like_vals_jac,
        )

    def _jacobian(self, design_weights: Array) -> Array:
        values = self._values(design_weights)
        evidences = self._evidence(design_weights).T
        evidences_jac = self._evidence.jacobian(design_weights)
        like_jac = self._evidence._quad_weighted_likelihood_jacobian(
            design_weights
        )
        first_mom = self._first_moment(self._evidence._quad_weighted_like_vals)
        first_mom_jac = self._first_moment_jac(like_jac)
        second_mom = self._second_moment(
            self._evidence._quad_weighted_like_vals
        )
        second_mom_jac = self._second_moment_jac(like_jac)
        variance_jac = (
            second_mom_jac / evidences
            - second_mom[..., None] * evidences_jac[None, :] / evidences**2
            - 2.0 * first_mom[..., None] * first_mom_jac / evidences**2
            + 2.0
            * first_mom[..., None] ** 2
            * evidences_jac[None, :]
            / evidences**3
        )
        variance_jac = self._bkd.reshape(
            variance_jac,
            (self._npred * self._nouterloop_samples, self.nvars()),
        )
        sqrt_jac = variance_jac / (2.0 * values[..., None])
        # fun = lambda w: self._evaluate(w[:, None])[0]
        # auto_jac = self._bkd.jacobian(fun, design_weights[:, 0]).reshape(
        #     (self._npred * self._nouterloop_samples, self.nvars())
        # )
        # print(auto_jac)
        # print(sqrt_jac)
        # print(auto_jac - sqrt_jac)
        # assert self._bkd.allclose(auto_jac, sqrt_jac)
        return sqrt_jac


class OEDEntropicDeviationMeasure(PredictionOEDDeviationMeasure):
    def __init__(
        self, npred: int, alpha: float, backend: BackendMixin = NumpyMixin
    ):
        super().__init__(npred, backend=backend)
        self.set_alpha(alpha)

    def set_alpha(self, alpha: float):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self._alpha = alpha

    def _weighted_exp_values(self):
        return (
            self._bkd.exp(self._alpha * self._qoi_vals.T[..., None])
            * self._evidence._quad_weighted_like_vals[None, ...]
        )

    def _first_moment(self, quad_weighted_like_vals: Array) -> Array:
        # after reshape the 3D arrays will have shape
        # (npred, ninnerloop_samples, nouterloop_samples)
        # return (
        #     self._qoi_vals.T[..., None] * quad_weighted_like_vals[None, ...]
        # ).sum(axis=1)
        # o:outer, i: inner, q: qoi
        return self._bkd.einsum(
            "iq,io->qo", self._qoi_vals, quad_weighted_like_vals
        )

    def _evaluate(self, design_weights: Array) -> Array:
        evidences = self._evidence(design_weights).T
        # risk1 = (
        #    self._bkd.log(
        #        self._weighted_exp_values().sum(axis=1) / evidences[:, 0]
        #    )
        #    / self._alpha
        # )
        # o:outer, i: inner, q: qoi, d: derivatives
        risk = (
            self._bkd.log(
                self._bkd.einsum(
                    "iq,io->qo",
                    self._bkd.exp(self._alpha * self._qoi_vals),
                    self._evidence._quad_weighted_like_vals,
                )
                / evidences[:, 0]
            )
            / self._alpha
        )
        mean = (
            self._first_moment(self._evidence._quad_weighted_like_vals)
            / evidences[:, 0]
        )
        return (risk - mean).flatten()[None, :]

    def _first_moment_jac(self, quad_weighted_like_vals_jac: Array) -> Array:
        # after reshape 4D arrays will have
        # (npred, nouterloop_samples, ninnerloop_samples, ndesign))
        # return (
        #     self._qoi_vals.T[:, None, :, None]
        #     * quad_weighted_like_vals_jac[None, ...]
        # ).sum(axis=2)
        # o:outer, i: inner, q: qoi, d: derivatives
        return self._bkd.einsum(
            "iq,oid->qod", self._qoi_vals, quad_weighted_like_vals_jac
        )

    def _jacobian(self, design_weights: Array) -> Array:
        # must call evidence first so weighted_exp_values are correct
        evidences = self._evidence(design_weights).T
        evidences_jac = self._evidence.jacobian(design_weights)
        quad_weighted_like_vals_jac = (
            self._evidence._quad_weighted_likelihood_jacobian(design_weights)
        )
        # term1 = (
        #     self._alpha
        #     * self._bkd.exp(self._alpha * self._qoi_vals.T)[:, None, :, None]
        #     * quad_weighted_like_vals_jac[None, ...]
        # ).sum(axis=2) / evidences

        # o:outer, i: inner, q: qoi, d: derivatives
        term1 = (
            self._alpha
            * self._bkd.einsum(
                "iq,oid->qod",
                self._bkd.exp(self._alpha * self._qoi_vals),
                quad_weighted_like_vals_jac,
            )
            / evidences
        )

        # o:outer, i: inner, q: qoi
        exp_values_mean = self._bkd.einsum(
            "iq,io->qo",
            self._bkd.exp(self._alpha * self._qoi_vals),
            self._evidence._quad_weighted_like_vals,
        )
        term2 = (
            exp_values_mean[..., None] * evidences_jac[None, :] / evidences**2
        )
        risk_jac = (term1 - term2) / (
            self._alpha * (exp_values_mean[..., None] / evidences)
        )
        first_mom = self._first_moment(self._evidence._quad_weighted_like_vals)
        first_mom_jac = self._first_moment_jac(quad_weighted_like_vals_jac)
        mean_jac = (
            first_mom_jac / evidences
            - first_mom[..., None] * evidences_jac[None, :] / evidences**2
        )
        return risk_jac - mean_jac


class PredictionOEDObjective(KLOEDObjective):
    def apply_hessian_implemented(self) -> bool:
        return False

    def set_qoi_quadrature_weights(self, qoi_quad_weights: Array):
        if not hasattr(self, "_deviation_measure"):
            raise ValueError("Must call set_deviation_measure")
        if qoi_quad_weights.shape != (self._deviation_measure.npred(), 1):
            raise ValueError(
                "qoi_quad_weights must have shape "
                "{0} but had shape {1}".format(
                    (self._deviation_measure.npred(), 1),
                    qoi_quad_weights.shape,
                )
            )
        self._qoi_quad_weights = qoi_quad_weights

    def set_deviation_measure(
        self, deviation_measure: PredictionOEDDeviationMeasure
    ):
        if not isinstance(deviation_measure, PredictionOEDDeviationMeasure):
            raise ValueError(
                "deviation_measure must be an instance of "
                "PredictionOEDDeviationMeasure"
            )
        self._deviation_measure = deviation_measure

    def set_risk_measure(self, risk_measure):
        if not isinstance(risk_measure, SampleAverageStat):
            raise ValueError(
                "risk_measure must be an instance of SampleAverageStat"
            )
        self._risk_measure = risk_measure

    def _setup_evidence(self):
        self._evidence = Evidence(
            self._innerloop_loglike, self._innerloop_quad_weights
        )

    def _evaluate_from_expanded_design_weights(
        self, design_weights: Array
    ) -> Array:
        self._deviation_measure.set_evidence(self._evidence)
        # resuse likelihood values computed to estimate evidences.
        # however techinically we could use a different quadrature rule
        deviations = self._deviation_measure(design_weights).reshape(
            self._deviation_measure._npred,
            self._deviation_measure._nouterloop_samples,
        )
        if not hasattr(self, "_qoi_quad_weights"):
            self.set_qoi_quadrature_weights(
                self._bkd.full(
                    (deviations.shape[0], 1), 1.0 / deviations.shape[0]
                )
            )
        risk_measures = self._risk_measure(
            deviations, self._qoi_quad_weights
        ).T
        return self._noise_stat(risk_measures, self._outerloop_quad_weights)

    def _jacobian_from_expanded_design_weights(
        self, design_weights: Array
    ) -> Array:
        self._deviation_measure.set_evidence(self._evidence)

        deviation_jac = self._deviation_measure._jacobian(
            design_weights
        ).reshape(
            (
                self._deviation_measure._npred,
                self._deviation_measure._nouterloop_samples,
                self.nvars(),
            )
        )
        deviations = self._deviation_measure(design_weights).reshape(
            self._deviation_measure._npred,
            self._deviation_measure._nouterloop_samples,
        )
        risk_values = self._risk_measure(deviations, self._qoi_quad_weights).T
        risk_values_jac = self._risk_measure.jacobian(
            deviations, deviation_jac, self._qoi_quad_weights
        )
        return self._noise_stat.jacobian(
            risk_values, risk_values_jac, self._outerloop_quad_weights
        )

    def __repr__(self) -> str:
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            self._deviation_measure,
            self._risk_measure,
            self._noise_stat,
        )


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

    def nobs(self) -> int:
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


class BayesianOED(ABC):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        self._bkd = backend

    def _set_objective_function(self, objective: BayesianOEDObjective):
        if not isinstance(objective, BayesianOEDObjective):
            raise ValueError(
                "objective must be an instance of BayesianOEDObjective"
            )
        if not objective._bkd.bkd_equal(self._bkd, objective._bkd):
            raise ValueError("backends are inconsistent")
        self._objective = objective

    def objective(self) -> BayesianOEDObjective:
        return self._objective

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
        if not hasattr(self, "_objective"):
            raise ValueError("must call set_data")
        iterate = self._bkd.full(
            (self._objective.nvars(), 1), 1 / self._objective.nvars()
        )
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        self._res = self._optimizer.minimize(iterate)
        return self._res.x

    def optimization_result(self) -> OptimizationResult:
        return self._res

    @abstractmethod
    def set_data(*args):
        raise NotImplementedError


class KLBayesianOED(BayesianOED):
    def __init__(
        self,
        innerloop_loglike: OEDInnerLoopLogLikelihoodMixin,
    ):
        super().__init__(innerloop_loglike._bkd)
        self._innerloop_loglike = innerloop_loglike

    def set_data(
        self,
        outerloop_shapes: Array,
        outerloop_quad_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_shapes: Array,
        innerloop_quad_weights: Array,
    ):
        self._set_objective_function(
            KLOEDObjective(
                self._innerloop_loglike,
                outerloop_shapes,
                outerloop_quad_samples,
                outerloop_quad_weights,
                innerloop_shapes,
                innerloop_quad_weights,
                backend=self._bkd,
            )
        )


class BayesianOEDForPrediction(BayesianOED):
    def __init__(
        self,
        innerloop_loglike: OEDInnerLoopLogLikelihoodMixin,
        deviation_measure: PredictionOEDDeviationMeasure,
        risk_measure: SampleAverageStat,
        noise_stat: NoiseStatistic,
    ):
        super().__init__(innerloop_loglike._bkd)
        self._innerloop_loglike = innerloop_loglike
        self._deviation_measure = deviation_measure
        self._risk_measure = risk_measure
        self._noise_stat = noise_stat

    def set_data(
        self,
        outerloop_shapes: Array,
        outerloop_quad_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_shapes: Array,
        innerloop_quad_weights: Array,
        qoi_vals: Array,
        pred_quad_weights: Array,
    ):
        objective = PredictionOEDObjective(
            self._innerloop_loglike,
            outerloop_shapes,
            outerloop_quad_samples,
            outerloop_quad_weights,
            innerloop_shapes,
            innerloop_quad_weights,
            backend=self._bkd,
        )
        objective.set_noise_statistic(self._noise_stat)
        self._deviation_measure.set_data(qoi_vals, pred_quad_weights)
        objective.set_deviation_measure(self._deviation_measure)
        objective.set_risk_measure(self._risk_measure)
        self._set_objective_function(objective)


class ConjugateGaussianPriorOEDForLinearPredictionUtility(ABC):
    """
    Compute the expected divergence or deviation of the pushforward of the posterior,
    arising from a conugate Gaussian prior, through a linear prediction model of a
    scalar quantity
    of interest (QoI).
    """

    def __init__(self, prior: MultivariateGaussian, qoi_mat: Array):
        self._bkd = prior._bkd
        self._prior = prior
        self._prior_cov_inv = self._bkd.inv(self._prior.covariance())
        self._qoi_mat = qoi_mat
        self._prior_pushforward = GaussianPushForward(
            self._qoi_mat,
            self._prior.mean(),
            self._prior.covariance(),
            backend=self._bkd,
        )

    def set_observation_matrix(self, obs_mat: Array):
        if obs_mat.shape[1] != self._prior.nvars():
            raise ValueError("obs matrix has the wrong number of columns")
        self._obs_mat = obs_mat

    def set_noise_covariance(self, noise_covariance: Array):
        self._noise_cov = noise_covariance
        self._noise_cov_inv = self._bkd.inv(self._noise_cov)
        self._compute()

    def _compute_expected_posterior_stats(self):
        if not hasattr(self, "_obs_mat"):
            raise ValueError("must call set_observation_matrix()")
        self._posterior = DenseMatrixLaplacePosteriorApproximation(
            self._obs_mat,
            self._prior.mean(),
            self._prior.covariance(),
            self._noise_cov,
            backend=self._bkd,
        )
        # value of obs does not matter for expected stats
        dummy_obs = self._bkd.ones((self._obs_mat.shape[0], 1))
        self._posterior.compute(dummy_obs)
        self._nu_vec = self._posterior._nu_vec
        self._Cmat = self._posterior._Cmat

    @abstractmethod
    def _compute_utility(self) -> float:
        raise NotImplementedError

    def _compute(self):
        self._compute_expected_posterior_stats()
        self._post_pushforward = GaussianPushForward(
            self._qoi_mat,
            self._posterior.posterior_mean(),
            self._posterior.posterior_covariance(),
            backend=self._bkd,
        )
        self._utility = self._compute_utility()

    def value(self) -> float:
        """Return the utility"""
        if not hasattr(self, "_utility"):
            raise ValueError("must call set_noise_covariance()")
        return self._utility


class ConjugateGaussianPriorOEDForLinearPredictionKLDivergence(
    ConjugateGaussianPriorOEDForLinearPredictionUtility
):
    def _compute_utility(self) -> float:
        return _compute_expected_kl_divergence(
            self._prior_pushforward.mean(),
            self._prior_pushforward.covariance(),
            self._post_pushforward.covariance(),
            self._qoi_mat @ self._nu_vec,
            self._qoi_mat @ self._Cmat @ self._qoi_mat.T,
            self._bkd,
        )


class ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation(
    ConjugateGaussianPriorOEDForLinearPredictionUtility
):
    def _compute_utility(self) -> float:
        return self._bkd.sqrt(self._post_pushforward.covariance()[0, 0])


class ConjugateGaussianPriorOEDForLogNormalPredictionStandardDeviation(
    ConjugateGaussianPriorOEDForLinearPredictionUtility
):
    def _lognormal_mean(self, mu: float, sigma: float):
        return self._bkd.exp(mu + sigma**2 / 2.0)

    def _compute_utility(self) -> float:
        tau_hat = self._qoi_mat @ self._nu_vec
        sigma_hat_sq = self._bkd.multidot(
            (self._qoi_mat, self._Cmat, self._qoi_mat.T)
        )
        tmp = self._bkd.exp(self._post_pushforward.covariance()[0, 0])
        factor = (tmp - 1.0) * tmp
        # We know the variance F \exp(X) is a lognormal distribution using
        # where X is a normal with
        # mean 2\tau_hat and standard deviation 2*\sqrt(\sigma_hat_sq)
        # Then standard deviation = F^{1/2}Y^{1/2}=F^{1/2}\exp(X)^{1/2}=F^{1/2}exp(X/2) = F^{1/2}\exp(Z) where Z is a normal with mean tau_hat and stdev sqrt(sigma_hat_sq)
        return self._bkd.sqrt(factor) * self._lognormal_mean(
            tau_hat, self._bkd.sqrt(sigma_hat_sq)
        )


class ConjugateGaussianPriorOEDForLogNormalPredictionKLDivergence(
    ConjugateGaussianPriorOEDForLinearPredictionKLDivergence
):
    # The KL of two lognormals is the same as the KL of the
    # two associated Normals
    pass


# TODO Consider using lognormal noise for OED
# if Y is lognormal (mu, sigma) -- mu and sigma are mean and standard deviation of gaussian --, then cY is lognormal (mu+log(c),sigma)
# So, we can sample from the likelihood by making model m(x) predict log mean of noise eps ~ lognormal(0, sigma) then y=exp(m(x))*eps which is lognormal ((x),sigma)
