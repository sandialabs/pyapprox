"""
This module implements Bayesian Optimal Experimental Design (OED) algorithms, including methods
for inferring model parameters from data and conditioning model predictions on observational data.
These algorithms are designed to optimize experimental designs by maximizing the expected
information gain (EIG) and improving the accuracy and robustness of model-based predictions.

Purpose
-------
The algorithms in this module enable researchers to compute experimental configurations that target:
-  Inference of model parameters from experimental data.
- Conditioning model predictions on observational data to improve decision-making.

Features
--------
- Bayesian OED algorithms for parameter inference.
- Bayesian OED algorithms for conditioning predictions on observations.

Usage
-----
This module is intended for researchers and practitioners working on Bayesian OED problems who
require robust algorithms for optimizing experimental designs and analyzing model-based predictions.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import itertools

from scipy import stats

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
    SampleSmoothedConditionalValueAtRisk,
)
from pyapprox.variables.joint import (
    JointVariable,
    IndependentGroupsVariable,
    IndependentMarginalsVariable,
)
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.expdesign.sequences import HaltonSequence
from pyapprox.surrogates.affine.basis import (
    setup_tensor_product_gauss_quadrature_rule,
)


class OEDOuterLoopLogLikelihoodMixin(ABC):
    """
    Wrap Likelihood function so that it is a function of the design weights
    and not a function of the likelihood shapes (or parameters of a model
    that predict the shapes)
    """

    def set_shapes(self, shapes: Array):
        self._shapes = shapes

    def set_artificial_observations(self, obs: Array):
        # Unlike likelihoods from pyapprox.bayes.likelihood, which take
        # obs and shapes with different shapes, obs and shapes passed to
        # OED likelihoods require obs and shapes with the same shape
        if obs.shape != self._shapes.shape:
            raise ValueError(
                f"{obs.shape=} does not match {self._shapes.shape=}"
            )
        self.set_observations(obs)

    def shapes(self) -> Array:
        """
        Return the shapes, e.g. mean of Gaussian, used to compute the
        likelihood

        Returns
        -------
        shapes : Array (nobs, nouterloop_samples)
        """
        if not hasattr(self, "_shapes"):
            raise AttributeError("must call set_shapes()")
        return self._shapes

    def nqoi(self) -> int:
        if not hasattr(self, "_obs"):
            raise AttributeError("must call set_observations")
        return self._obs.shape[1]

    def nvars(self) -> int:
        return self.shapes().shape[0]

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

    @abstractmethod
    def joint_prior_data_variable(self) -> JointVariable:
        """
        Set up the joint distribution over the prior and the
        latent space of the data likelihood. E.g. for gaussian noise with
        standard deviation sigma sample from the standard normal
        The mapping to the true data space will be taken care of
        by oed classes

        Returns
        -------
        variable: JointVariable
            The joint distribution over the prior and data.
        """
        raise NotImplementedError

    def set_latent_likelihood_samples(self, latent_samples: Array):
        """
        Set the latent samples used to modify the noiseless observations.
        E.g. when using additive Gaussian noise with standard deviation sigma,
        the obs = model_vals + sigma * latent_samples, where the latent
        samples are from the standard normal distribution.
        """
        self._latent_likelihood_samples = latent_samples


class IndependentGaussianOEDOuterLoopLogLikelihood(
    OEDOuterLoopLogLikelihoodMixin, IndependentGaussianLogLikelihood
):
    def set_artificial_observations(self, obs: Array):
        super().set_artificial_observations(obs)
        self._residuals = self._obs - self._shapes

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
        jac = (
            -0.5 * self._residuals.T**2 * self._noise_cov_inv_diag[:, 0]
            + 0.5 / design_weights.T
        )

        # second term on rhs above is gradient of determinant of weighted
        # noise covariance

        # Compute component of gradient due to using the reparameterization
        # trick to compute the outerloop observations.
        # "_latent_likelihood_samples" will always be true if computing
        # an oed objective, but we allow it not to be set for computing
        # gradient of likelihood when not using reparameterization trick.
        # This later is only used for testing.
        if hasattr(self, "_latent_likelihood_samples"):
            jac += 0.5 * (
                self._residuals.T
                * self._latent_likelihood_samples.T
                * self._bkd.sqrt(self._noise_cov_inv_diag[:, 0])
                / self._bkd.sqrt(design_weights)[:, 0]
            )
        return jac

    def joint_prior_data_variable(
        self, prior: JointVariable
    ) -> IndependentGroupsVariable:
        """
        Set up the joint distribution over the prior and the
        latent space of the data likelihood. E.g. for gaussian noise with
        standard deviation sigma sample from the standard normal
        The mapping to the true data space will be taken care of
        by oed classes

        Parameters
        ----------
        prior: IndependentGroupsVariable
            The prior variable.

        Returns
        -------
        variable: JointVariable
            The joint distribution over the prior and data.
        """
        latent_data_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1) for ii in range(self.nobs())],
            backend=self._bkd,
        )
        return IndependentGroupsVariable([prior, latent_data_variable])


class OEDInnerLoopLogLikelihoodMixin:
    def set_shapes(self, shapes: Array):
        if shapes.ndim != 2:
            raise ValueError(
                "shapes must be a 2D array (nobs, ninnerloop_samples)"
            )
        self._shapes = shapes

    def set_artificial_observations(self, obs: Array):
        # Unlike likelihoods from pyapprox.bayes.likelihood, which take
        # obs and shapes with different shapes, obs and shapes passed to
        # OED likelihoods require obs and shapes with the same shape
        if obs.ndim != 2:
            raise ValueError(
                "obs must be a 2D array (nobs, nouterloop_samples)"
            )
        if obs.shape[0] != self._shapes.shape[0]:
            raise ValueError(
                "The number of rows of obs and shapes are inconsistent"
            )
        self._obs = obs

    def nqoi(self) -> int:
        return self._obs.shape[1] * self._shapes.shape[1]

    def nvars(self) -> int:
        return self._shapes.shape[0]

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

    def set_latent_likelihood_samples(self, latent_samples: Array):
        """
        Set the latent samples used to modify the noiseless observations.
        E.g. when using additive Gaussian noise with standard deviation sigma,
        the obs = model_vals + sigma * latent_samples, where the latent
        samples are from the standard normal distribution.
        """
        self._latent_likelihood_samples = latent_samples


class IndependentGaussianOEDInnerLoopLogLikelihood(
    OEDInnerLoopLogLikelihoodMixin, IndependentGaussianLogLikelihood
):
    def set_artificial_observations(self, obs: Array):
        super().set_artificial_observations(obs)
        self._residuals = self._obs[..., None] - self._shapes[:, None, :]

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
        # Compute jacobian of quadratic component of likelihood
        jac = -0.5 * self._noise_cov_inv_diag[..., None] * self._residuals**2

        # Compute component of gradient due to using the reparameterization
        # trick to compute the outerloop observations.
        # "_latent_likelihood_samples" will always be true if computing
        # an oed objective, but we allow it not to be set for computing
        # gradient of likelihood when not using reparameterization trick.
        # This later is only used for testing.
        if hasattr(self, "_latent_likelihood_samples"):
            jac += 0.5 * (
                self._bkd.sqrt(self._noise_cov_inv_diag[..., None])
                * self._residuals
                * self._latent_likelihood_samples[..., None]
                / self._bkd.sqrt(design_weights)[..., None]
            )

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
            raise TypeError("loglike must be OEDLogLikelihoodMixin")

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

    def effective_sample_size(self, design_weights: Array) -> float:
        """Compute effective sample size for each outerloop observation"""
        like_vals = self._reshape_vals(
            self._bkd.exp(self._loglike(design_weights))
        )
        if self._bkd.any(self._quad_weights != self._quad_weights[0]):
            raise ValueError("Only intended for MC sampling")
        ess = self._bkd.sum(like_vals, axis=0) ** 2 / self._bkd.sum(
            like_vals**2, axis=0
        )
        return ess


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
            raise TypeError(
                "innerloop_loglike must be a OEDInnerLoopLogLikelihoodMixin"
            )
        self._innerloop_loglike = innerloop_loglike
        self._outerloop_loglike = innerloop_loglike.outerloop_loglike()
        self._outerloop_shapes = outerloop_shapes
        self._outerloop_quad_samples = outerloop_quad_samples
        self._innerloop_shapes = innerloop_shapes
        self._set_quadrature_weights(
            self._outerloop_shapes.shape[1],
            outerloop_quad_weights,
            self._innerloop_shapes.shape[1],
            innerloop_quad_weights,
        )

    def _set_expanded_design_weights(self, design_weights: Array):
        """
        Updates the outerloop observations based on the design weights.
        """
        self._outerloop_loglike.set_design_weights(design_weights)
        latent_likelihood_samples = self._outerloop_quad_samples[
            -self._outerloop_shapes.shape[0] :
        ]
        obs = self._outerloop_loglike._rvs_from_likelihood_samples(
            self._outerloop_shapes, latent_likelihood_samples
        )
        self._outerloop_loglike.set_artificial_observations(obs)
        self._innerloop_loglike.set_artificial_observations(obs)
        self._outerloop_loglike.set_latent_likelihood_samples(
            latent_likelihood_samples
        )
        self._innerloop_loglike.set_latent_likelihood_samples(
            latent_likelihood_samples
        )
        self._setup_evidence()

    def _evaluate(self, design_weights: Array) -> Array:
        self._set_expanded_design_weights(
            self._expand_design_weights(design_weights)
        )
        return super()._evaluate(design_weights)

    def _jacobian(self, design_weights: Array) -> Array:
        self._set_expanded_design_weights(
            self._expand_design_weights(design_weights)
        )
        return super()._jacobian(design_weights)

    def set_noise_statistic(self, noise_stat: NoiseStatistic):
        if not isinstance(noise_stat, NoiseStatistic):
            raise TypeError("noise_stat must be an instance of NoiseStatistic")
        self._noise_stat = noise_stat

    def _setup_evidence(self):
        self._log_evidence = LogEvidence(
            self._innerloop_loglike, self._innerloop_quad_weights
        )

    def nobs(self) -> int:
        return self._outerloop_loglike.nobs()

    def _set_quadrature_weights(
        self,
        nouterloop_samples: int,
        outerloop_quad_weights: Array,
        ninnerloop_samples: int,
        innerloop_quad_weights: Array,
    ):
        if outerloop_quad_weights is None:
            outerloop_quad_weights = self._bkd.full(
                (nouterloop_samples, 1),
                1.0 / nouterloop_samples,
            )

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

    # Hessian reduces number of iterations, but currently
    # results in slower optimization because hvp is not fast enough
    def apply_hessian_implemented(self) -> bool:
        return False

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


class PredictionOEDDeviationMeasure(SingleSampleModel):
    def __init__(self, npred: int, backend: BackendMixin = NumpyMixin):
        super().__init__(backend=backend)
        self._npred = npred

    def set_loglikelihood(
        self,
        innerloop_loglike: OEDInnerLoopLogLikelihoodMixin,
    ):
        if not isinstance(innerloop_loglike, OEDInnerLoopLogLikelihoodMixin):
            raise TypeError(
                "loglike must be an instance of OEDInnerLoopLogLikelihoodMixin"
            )
        if not innerloop_loglike._bkd.bkd_equal(
            self._bkd, innerloop_loglike._bkd
        ):
            raise TypeError("backends are inconsistent")
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
            raise TypeError(
                f"evidence {evidence} must be an instance of Evidence"
            )
        self._evidence = evidence


class OEDPredictionFirstMomentMixin:
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


class OEDStandardDeviationMeasure(
    PredictionOEDDeviationMeasure, OEDPredictionFirstMomentMixin
):
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
        variance = (
            self._second_moment(self._evidence._quad_weighted_like_vals)
            / evidences[:, 0]
            - self._first_moment(self._evidence._quad_weighted_like_vals) ** 2
            / evidences[:, 0] ** 2
        )
        # avoid small values just below zero
        variance = self._bkd.maximum(
            variance, self._bkd.full(variance.shape, 1e-16)
        )
        return self._bkd.sqrt(variance).flatten()[None, :]

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
        # the commented lines below are less numerically stable than their
        # uncommented equivalent versions when evidences are small
        second_mom_jac = self._second_moment_jac(like_jac)
        variance_jac = (
            second_mom_jac / evidences
            # - second_mom[..., None] * evidences_jac[None, :] / evidences**2
            - (
                (second_mom[..., None] * evidences_jac[None, :])
                / evidences
                / evidences
            )
            - 2.0 * first_mom[..., None] * first_mom_jac / evidences**2
            + 2.0
            # * first_mom[..., None] ** 2
            # * evidences_jac[None, :]
            # / evidences**3
            * (first_mom[..., None] ** 2 / evidences**2)
            * evidences_jac[None, :]
            / evidences
        )
        variance_jac = self._bkd.reshape(
            variance_jac,
            (self._npred * self._nouterloop_samples, self.nvars()),
        )
        sqrt_jac = variance_jac / (2.0 * values[..., None])
        return sqrt_jac


class OEDEntropicDeviationMeasure(
    PredictionOEDDeviationMeasure, OEDPredictionFirstMomentMixin
):
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

    def _evaluate(self, design_weights: Array) -> Array:
        evidences = self._evidence(design_weights).T
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


class OEDAVaRDeviationMeasure(
    PredictionOEDDeviationMeasure, OEDPredictionFirstMomentMixin
):
    def __init__(
        self,
        npred: int,
        alpha: float,
        eps: float,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(npred, backend=backend)
        self._eps = eps
        self.set_alpha(alpha)

    def set_alpha(self, alpha: float):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self._alpha = alpha
        self._smoothed_avar = SampleSmoothedConditionalValueAtRisk(
            self._alpha, self._eps, backend=self._bkd
        )

    def _evaluate_option1(self, design_weights: Array) -> Array:
        evidences = self._evidence(design_weights).T
        # o:outer, i: inner, q: qoi, d: derivatives
        # self._evidence._quad_weighted_like_vals.shape = (i,o)
        # self._qoi_vals.shape = (i,q)
        # evidences.shape = (o,1)
        # return (q,o).flatten()
        mean = (
            self._first_moment(self._evidence._quad_weighted_like_vals)
            / evidences[:, 0]
        )
        vals = []
        for qq in range(self._qoi_vals.shape[1]):
            outer_vals = []
            for oo in range(evidences.shape[0]):
                # mean needs to be included here and not subtracted at end
                # otherwise accuracy of avardev is limited.
                avardev = self._smoothed_avar(
                    # self._qoi_vals[:, qq : qq + 1] - mean[qq, oo],
                    self._qoi_vals[:, qq : qq + 1],
                    self._evidence._quad_weighted_like_vals[:, oo : oo + 1]
                    / evidences[oo : oo + 1],
                )
                outer_vals.append(avardev)
            vals.append(self._bkd.asarray(outer_vals))
        # avardev = self._bkd.stack(vals, axis=0)
        avardev = self._bkd.stack(vals, axis=0) - mean
        return avardev.flatten()[None, :]

    def _evaluate_option2(self, design_weights: Array) -> Array:
        evidences = self._evidence(design_weights).T
        # o:outer, i: inner, q: qoi, d: derivatives
        # self._evidence._quad_weighted_like_vals.shape = (i,o)
        # self._qoi_vals.shape = (i,q)
        # evidences.shape = (o,1)
        # return (q,o).flatten()
        mean = (
            self._first_moment(self._evidence._quad_weighted_like_vals)
            / evidences[:, 0]
        )
        vals = []
        for qq in range(self._qoi_vals.shape[1]):
            outer_vals = []
            for oo in range(evidences.shape[0]):
                # mean needs to be included here and not subtracted at end
                # otherwise accuracy of avardev is limited.
                avardev = (
                    self._smoothed_avar(
                        (self._qoi_vals[:, qq : qq + 1] - mean[qq, oo])
                        * self._evidence._like_vals[:, oo : oo + 1],  #
                        # / evidences[oo : oo + 1],
                        self._evidence._quad_weights,
                    )
                    / evidences[oo : oo + 1]
                )
                outer_vals.append(avardev)
            vals.append(self._bkd.asarray(outer_vals))
        avardev = self._bkd.stack(vals, axis=0)
        return avardev.flatten()[None, :]

    def _evaluate_option3(self, design_weights: Array) -> Array:
        evidences = self._evidence(design_weights).T
        # o:outer, i: inner, q: qoi, d: derivatives
        # self._evidence._quad_weighted_like_vals.shape = (i,o)
        # self._qoi_vals.shape = (i,q)
        # evidences.shape = (o,1)
        # return (q,o).flatten()
        mean = (
            self._first_moment(self._evidence._quad_weighted_like_vals)
            / evidences[:, 0]
        )
        vals = []
        for qq in range(self._qoi_vals.shape[1]):
            outer_vals = []
            for oo in range(evidences.shape[0]):
                avardev = (
                    self._smoothed_avar(
                        self._qoi_vals[:, qq : qq + 1]
                        * self._evidence._like_vals[:, oo : oo + 1],
                        # / evidences[oo : oo + 1],
                        self._evidence._quad_weights,
                    )
                    / evidences[oo : oo + 1]
                )
                outer_vals.append(avardev)
            vals.append(self._bkd.asarray(outer_vals))
        # print(evidences)
        avardev = self._bkd.stack(vals, axis=0) - mean
        return avardev.flatten()[None, :]

    def _evaluate(self, design_weights: Array) -> Array:
        # print()
        # print(self._evaluate_option1(design_weights))
        # print(self._evaluate_option2(design_weights))
        # print(self._evaluate_option3(design_weights))
        return self._evaluate_option1(design_weights)

    def _jacobian(self, design_weights: Array) -> Array:
        raise NotImplementedError
        evidences = self._evidence(design_weights).T
        evidences_jac = self._evidence.jacobian(design_weights)
        quad_weighted_like_vals_jac = (
            self._evidence._quad_weighted_likelihood_jacobian(design_weights)
        )

        first_mom = self._first_moment(self._evidence._quad_weighted_like_vals)
        # mean = first_mom / evidences[:, 0]
        first_mom_jac = self._first_moment_jac(quad_weighted_like_vals_jac)
        mean_jac = (
            first_mom_jac / evidences
            - first_mom[..., None] * evidences_jac[None, :] / evidences**2
        )
        like_jac = self._evidence._like_jac
        print(like_jac.shape, "lj")
        jacs = []
        for qq in range(self._qoi_vals.shape[1]):
            outer_vals = []
            for oo in range(evidences.shape[0]):
                # mean needs to be included here and not subtracted at end
                # otherwise accuracy of avardev is limited.
                # avar_jac = self._smoothed_avar.jacobian(
                #     self._qoi_vals[:, qq : qq + 1],
                #     like_jac[oo],
                #     self._evidence._quad_weighted_like_vals[:, oo : oo + 1]
                #     / evidences[oo : oo + 1],
                # )
                avar_jac = self._smoothed_avar.jacobian(
                    self._qoi_vals[:, qq : qq + 1],
                    quad_weighted_like_vals_jac[oo] / evidences[oo : oo + 1],
                    self._evidence._quad_weights,
                )
                print((avar_jac - mean_jac[qq : qq + 1, oo]).shape)
                outer_vals.append(avar_jac - mean_jac[qq : qq + 1, oo])
            jacs.append(self._bkd.vstack(outer_vals))
        print(mean_jac.shape, self._bkd.stack(jacs, axis=0).shape)
        return self._bkd.stack(jacs, axis=0)


class PredictionOEDObjective(KLOEDObjective):
    def apply_hessian_implemented(self) -> bool:
        return False

    def set_qoi_quadrature_weights(self, qoi_quad_weights: Array):
        if not hasattr(self, "_deviation_measure"):
            raise AttributeError("Must call set_deviation_measure")
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
            raise TypeError(
                "deviation_measure must be an instance of "
                "PredictionOEDDeviationMeasure"
            )
        self._deviation_measure = deviation_measure

    def set_risk_measure(self, risk_measure):
        if not isinstance(risk_measure, SampleAverageStat):
            raise TypeError(
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
            raise TypeError("noise_cov must be a scalar")
        if prior_cov.ndim != 0:
            raise TypeError("prior_cov must be a scalar")
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
    def __init__(
        self,
        innerloop_loglike: OEDInnerLoopLogLikelihoodMixin,
    ):
        self._innerloop_loglike = innerloop_loglike
        self._bkd = innerloop_loglike._bkd

    def _set_objective_function(self, objective: BayesianOEDObjective):
        if not isinstance(objective, BayesianOEDObjective):
            raise TypeError(
                "objective must be an instance of BayesianOEDObjective"
            )
        if not objective._bkd.bkd_equal(self._bkd, objective._bkd):
            raise TypeError("backends are inconsistent")
        self._objective = objective

    def objective(self) -> BayesianOEDObjective:
        return self._objective

    @abstractmethod
    def set_data(*args):
        raise NotImplementedError


class RelaxedBayesianOED(BayesianOED):

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
            raise TypeError(
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
            raise AttributeError("must call set_data")
        iterate = self._bkd.full(
            (self._objective.nvars(), 1), 1 / self._objective.nvars()
        )
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        self._res = self._optimizer.minimize(iterate)
        return self._res.x

    def optimization_result(self) -> OptimizationResult:
        return self._res

    def get_innerloop_loglike(self) -> OEDInnerLoopLogLikelihoodMixin:
        return self._innerloop_loglike

    def get_outerloop_loglike(self) -> OEDOuterLoopLogLikelihoodMixin:
        return self._innerloop_loglike.outerloop_loglike()

    def monte_carlo_quadrature_data(
        self, prior: JointVariable, nsamples: int
    ) -> Tuple[Array, Array]:
        """
        Return the quadrature samples over the joint prior and data space,
        needed to evaluate
        the model to obtain the simulations, and the quadrature
        weights needed to compute the OED utility.

        Parameters
        ----------
        prior: JointVariable
            The prior over the uncertain model parameters.
        nsamples : int
            The number of samples needed.
        """
        outerloop_loglike = self._innerloop_loglike.outerloop_loglike()
        prior_data_variable = outerloop_loglike.joint_prior_data_variable(
            self._problem.get_prior()
        )
        prior_data_variable.rvs(nsamples), self._bkd.full(
            (nsamples, 1), 1.0 / nsamples
        )


class KLBayesianOEDMixin:
    def set_data(
        self,
        outerloop_shapes: Array,
        outerloop_quad_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_shapes: Array,
        innerloop_quad_weights: Array,
    ):
        """
        Set the data needed for computing the OED.

        Parameters
        ----------
        outerloop_shapes : Array (nobs, nouterloop_samples)
            Samples for the outer loop quadrature.
        outerloop_quad_samples : Array (nvars+nobs, nouterloop_samples)
            Weights for the outer loop quadrature.
        outerloop_quad_weights : Array (nouterloop_samples, 1)
            Weights for the outer loop quadrature.
        innerloop_shapes : Array (nobs, ninnerloop_samples)
            Samples for the inner loop quadrature.
        innerloop_quad_weights : Array (ninnerloop_samples, 1)
            Weights for the inner loop quadrature.
        """

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

    def set_data_from_model(
        self,
        obs_model: Model,
        prior: JointVariable,
        outerloop_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_samples: Array,
        innerloop_quad_weights: Array,
    ):
        """
        Set the data needed for computing the OED running the models
        to generate the simulation data.

        Parameters
        ----------
        obs_model: Model
            The observation model used to predict likely observations.
        prior: JointVariable
            The prior of the uncertain model parameters.
        innerloop_loglike : IndependentGaussianOEDInnerLoopLogLikelihood
            Inner loop log-likelihood object.
        outerloop_samples : Array (nvars+nobs, nouterloop_samples)
            Samples for the outer loop quadrature.
        outerloop_quad_weights : Array (nouterloop_samples, 1)
            Weights for the outer loop quadrature.
        innerloop_samples : Array (nvars, ninnerloop_samples)
            Samples for the inner loop quadrature.
        innerloop_quad_weights : Array (ninnerloop_samples, 1)
            Weights for the inner loop quadrature.
        """
        outerloop_loglike = self._innerloop_loglike.outerloop_loglike()
        outerloop_shapes_samples = outerloop_samples[: prior.nvars()]
        outerloop_shapes = obs_model(outerloop_shapes_samples).T
        print(outerloop_shapes.shape, obs_model)
        outerloop_loglike.set_shapes(outerloop_shapes)

        innerloop_shapes = obs_model(innerloop_samples).T
        self._innerloop_loglike.set_shapes(innerloop_shapes)

        self.set_data(
            self._innerloop_loglike.outerloop_loglike().shapes(),
            outerloop_samples,
            outerloop_quad_weights,
            self._innerloop_loglike.shapes(),
            innerloop_quad_weights,
        )


class KLBayesianOED(KLBayesianOEDMixin, RelaxedBayesianOED):
    pass


class BruteForceKLBayesianOED(KLBayesianOEDMixin, BayesianOED):
    def _evaluate_designs(self, design_indices: Array):
        utility_vals = []
        for index in design_indices:
            eps = 1e-14  # need a small value to avoid low rank matrix
            design_weights = self._bkd.full((self._objective.nvars(), 1), eps)
            design_weights[index, 0] = 1.0
            utility_vals.append(self._objective(design_weights)[:, 0])
        return self._bkd.hstack(utility_vals)

    def compute(self, nobs: int) -> Array:
        if nobs > self._objective.nvars():
            raise ValueError(
                "nobs must not be greater than number of candidates"
            )
        if not hasattr(self, "_objective"):
            raise AttributeError("must call set_data")
        self._design_indices = self._bkd.asarray(
            list(
                itertools.combinations(
                    self._bkd.arange(self._objective.nvars()), nobs
                )
            )
        )
        self._utility_vals = self._evaluate_designs(self._design_indices)
        selected_index = self._design_indices[
            self._bkd.argmin(self._utility_vals)
        ]
        return selected_index


class BayesianOEDForPrediction(RelaxedBayesianOED):
    def set_data_from_model(
        self,
        obs_model: Model,
        qoi_model: Model,
        prior: JointVariable,
        outerloop_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_samples: Array,
        innerloop_quad_weights: Array,
        pred_quad_weights: Array,
        deviation_measure: PredictionOEDDeviationMeasure,
        risk_measure: SampleAverageStat,
        noise_stat: NoiseStatistic,
    ):
        """
        Set the data needed for computing the OED running the models
        to generate the simulation data.

        Parameters
        ----------
        obs_model: Model
            The observation model used to predict likely observations.
        qoi_model: Model
            The prediction model used to predict the QoI.
        prior: JointVariable
            The prior of the uncertain model parameters.
        innerloop_loglike : IndependentGaussianOEDInnerLoopLogLikelihood
            Inner loop log-likelihood object.
        outerloop_samples : Array (nvars+nobs, nouterloop_samples)
            Samples for the outer loop quadrature.
        outerloop_quad_weights : Array (nouterloop_samples, 1)
            Weights for the outer loop quadrature.
        innerloop_samples : Array (nvars, ninnerloop_samples)
            Samples for the inner loop quadrature.
        innerloop_quad_weights : Array (ninnerloop_samples, 1)
            Weights for the inner loop quadrature.
        pred_quad_weights : Array (nqoi, 1)
            The QoI weights used to compute the weighted sum (integrate)
            the QoI
        deviation_measure: PredictionOEDDeviationMeasure
            The deviation measure used to compute the reduction in uncertainty
            for each innerloop posterior
        risk_measure: SampleAverageStat
            The risk measure applied over the prediction space
        noise_stat: NoiseStatistic
           The risk measure applied over all realizations of the data
        """
        outerloop_loglike = self._innerloop_loglike.outerloop_loglike()
        outerloop_shapes_samples = outerloop_samples[: prior.nvars()]
        outerloop_shapes = obs_model(outerloop_shapes_samples).T
        outerloop_loglike.set_shapes(outerloop_shapes)

        innerloop_shapes = obs_model(innerloop_samples).T
        self._innerloop_loglike.set_shapes(innerloop_shapes)
        qoi_vals = qoi_model(innerloop_samples)

        self.set_data(
            self._innerloop_loglike.outerloop_loglike().shapes(),
            outerloop_samples,
            outerloop_quad_weights,
            self._innerloop_loglike.shapes(),
            innerloop_quad_weights,
            qoi_vals,
            pred_quad_weights,
            deviation_measure,
            risk_measure,
            noise_stat,
        )

    def set_data(
        self,
        outerloop_shapes: Array,
        outerloop_quad_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_shapes: Array,
        innerloop_quad_weights: Array,
        qoi_vals: Array,
        pred_quad_weights: Array,
        deviation_measure: PredictionOEDDeviationMeasure,
        risk_measure: SampleAverageStat,
        noise_stat: NoiseStatistic,
    ):
        """
        Set the data needed for computing the OED.

        Parameters
        ----------
        outerloop_shapes : Array (nobs, nouterloop_samples)
            Samples for the outer loop quadrature.
        outerloop_quad_samples : Array (nvars+nobs, nouterloop_samples)
            Weights for the outer loop quadrature.
        outerloop_quad_weights : Array (nouterloop_samples, 1)
            Weights for the outer loop quadrature.
        innerloop_shapes : Array (nobs, ninnerloop_samples)
            Samples for the inner loop quadrature.
        innerloop_quad_weights : Array (ninnerloop_samples, 1)
            Weights for the inner loop quadrature.
        qoi_vals : Array(ninner_samples, nqoi)
            The values of the QoI at the innerloop samples
        pred_quad_weights : Array (nqoi, 1)
            The QoI weights used to compute the weighted sum (integrate)
            the QoI
        deviation_measure: PredictionOEDDeviationMeasure
            The deviation measure used to compute the reduction in uncertainty
            for each innerloop posterior
        risk_measure: SampleAverageStat
            The risk measure applied over the prediction space
        noise_stat: NoiseStatistic
           The risk measure applied over all realizations of the data
        """

        objective = PredictionOEDObjective(
            self._innerloop_loglike,
            outerloop_shapes,
            outerloop_quad_samples,
            outerloop_quad_weights,
            innerloop_shapes,
            innerloop_quad_weights,
            backend=self._bkd,
        )
        self._deviation_measure = deviation_measure
        self._deviation_measure.set_loglikelihood(self._innerloop_loglike)
        self._risk_measure = risk_measure
        self._noise_stat = noise_stat

        objective.set_noise_statistic(self._noise_stat)
        self._deviation_measure.set_data(qoi_vals, pred_quad_weights)
        objective.set_deviation_measure(self._deviation_measure)
        objective.set_risk_measure(self._risk_measure)
        self._set_objective_function(objective)


class BayesianOEDDataGenerator:
    def __init__(self, backend: BackendMixin):
        self._bkd = backend

    def _setup_quadrature_data(
        self, quadtype: str, variable: JointVariable, nsamples: int
    ) -> Tuple[Array, Array]:
        """
        Helper function to set up quadrature data for integration using
        different types of quadrature rules.

        Parameters
        ----------
        quadtype : str
            Type of quadrature method (e.g., "MC", "Halton","gauss").
        prior_data_variable : JointVariable
            Variable that defines the integration measure.
        nsamples : int
            Number of samples for quadrature.

        Returns
        -------
        Tuple[Array, Array]
            Quadrature samples and weights.
        """
        if quadtype == "MC":
            return variable.rvs(nsamples), self._bkd.full(
                (nsamples, 1), 1.0 / nsamples
            )

        if quadtype == "Halton":
            sequence = HaltonSequence(
                variable.nvars(),
                1,
                variable,
                variable._bkd,
                unbounded_eps=1e-4,
                increment_start_index=True,
            )
            return sequence.rvs(nsamples), self._bkd.full(
                (nsamples, 1), 1.0 / nsamples
            )

        if quadtype == "gauss":
            quad_rule = setup_tensor_product_gauss_quadrature_rule(variable)
            level1d = int(nsamples ** (1 / variable.nvars()))
            return quad_rule([level1d] * variable.nvars())

        raise TypeError(f"{quadtype} not supported")

    def prepare_simulation_inputs(
        self,
        oed: BayesianOED,
        prior: JointVariable,
        outerloop_quadtype: str,
        nouterloop_samples: int,
        innerloop_quadtype: str,
        ninnerloop_samples: int,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Set up data for Bayesian OED.

        Parameters
        ----------
        outerloop_quadtype : str
            Type of quadrature method for the outer loop.
        nouterloop_samples : int
            Number of samples for the outer loop quadrature.
        innerloop_quadtype : str
            Type of quadrature method for the inner loop.
        ninnerloop_samples : int
            Number of samples for the inner loop quadrature.

        Returns
        -------
        outerloop_samples : Array (nvars, nouterloop_samples)
            Samples for the outer loop quadrature.
        outerloop_quad_weights : Array (nouterloop_samples, 1)
            Weights for the outer loop quadrature.
        innerloop_samples : Array (nvars, ninnerloop_samples)
            Samples for the inner loop quadrature.
        innerloop_quad_weights : Array (ninnerloop_samples, 1)
            Weights for the inner loop quadrature.
        innerloop_loglike : IndependentGaussianOEDInnerLoopLogLikelihood
            Inner loop log-likelihood object.
        """
        # Generate outer loop quadrature data
        outerloop_loglike = oed._innerloop_loglike.outerloop_loglike()
        prior_data_variable = outerloop_loglike.joint_prior_data_variable(
            prior
        )
        outerloop_samples, outerloop_quad_weights = (
            self._setup_quadrature_data(
                outerloop_quadtype, prior_data_variable, nouterloop_samples
            )
        )

        # Generate inner loop quadrature data
        innerloop_samples, innerloop_quad_weights = (
            self._setup_quadrature_data(
                innerloop_quadtype, prior, ninnerloop_samples
            )
        )
        return (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )


# TODO Consider using lognormal noise for OED
# if Y is lognormal (mu, sigma) -- mu and sigma are mean and standard deviation of gaussian --, then cY is lognormal (mu+log(c),sigma)
# So, we can sample from the likelihood by making model m(x) predict log mean of noise eps ~ lognormal(0, sigma) then y=exp(m(x))*eps which is lognormal ((x),sigma)
