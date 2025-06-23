"""
Variational inference for Bayesian inverse problems.

This module implements the ELBO (Evidence Lower BOund) objective for
variational inference in Bayesian inverse problems
"""

from abc import ABC, abstractmethod
import warnings
from typing import Tuple, Union, Optional

import numpy as np

from pyapprox.variables.joint import JointVariable
from pyapprox.interface.model import SingleSampleModel
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.optimization.risk import (
    CholeskyBasedGaussianExactKLDivergence,
    IndependentGaussianExactKLDivergence,
)
from pyapprox.expdesign.sequences import LowDiscrepancySequence
from pyapprox.bayes.likelihood import ModelBasedLogLikelihoodMixin
from pyapprox.optimization.minimize import (
    MultiStartOptimizer,
    OptimizerIterateGenerator,
    RandomUniformOptimzerIterateGenerator,
)
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
    CholeskyHyperParameter,
)
from pyapprox.variables.marginals import (
    BetaMarginal,
    UniformMarginal,
    GaussianMarginal,
)
from pyapprox.variables.gaussian import (
    DenseCholeskyMultivariateGaussian,
    IndependentMultivariateGaussian,
)
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    DirichletVariable,
)
from pyapprox.surrogates.affine.basis import QuadratureRule

# TODO implement diagonal plus low rank Gaussian covariance based divergence see
# https://proceedings.neurips.cc/paper_files/paper/2020/file/310cc7ca5a76a446f85c1a0d641ba96d-Paper.pdf
# TODO: Implement KLDivergence when second gaussian is IID standard Normal


class LatentVariableGenerator(ABC):
    """
    Base class for all latent variable generators.

    Parameters
    ----------
    nvars : int
        Number of latent variables.
    backend : BackendMixin
        Backend to use for computations.

    Attributes
    ----------
    _bkd : BackendMixin
        Backend to use for computations.
    _nvars : int
        Number of latent variables.
    """

    def __init__(self, nvars: int, backend: BackendMixin):
        """
        Initialize the latent variable generator.

        Parameters
        ----------
        nvars : int
            Number of latent variables.
        backend : BackendMixin
            Backend to use for computations.
        """
        self._bkd = backend
        self._nvars = nvars

    @abstractmethod
    def _samples_weights(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Generate samples and their corresponding weights defined on the latent space.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Generated samples.
        weights : Array
            Corresponding weights.
        """
        raise NotImplementedError

    @abstractmethod
    def _rvs(self, nsamples: int) -> Array:
        """
        Generate random realizations from the variational posterior.

        Parameters
        ----------
        nsamples : int
            Number of random realizations to generate.

        Returns
        -------
        rvs : Array
            Generated random reaizations.
        """
        raise NotImplementedError

    def _check_samples_weights(
        self, nsamples: int, samples: Array, weights: Array
    ):
        """
        Check if the generated samples and weights are valid.

        Parameters
        ----------
        nsamples : int
            Number of samples.
        samples : Array
            Generated samples.
        weights : Array
            Corresponding weights.
        """
        if samples.shape != (self._nvars, nsamples):
            raise RuntimeError(
                "samples has the wrong shape. Was {0} should be {1}".format(
                    samples.shape, (self._nvars, nsamples)
                )
            )
        if weights.shape != (nsamples, 1):
            raise RuntimeError("samples has the wrong shape")

    def __call__(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Generate samples and their corresponding weights defined on the latent space
        used to compute the expected loss during training.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Generated samples.
        weights : Array
            Corresponding weights.
        """
        samples, weights = self._samples_weights(nsamples)
        self._check_samples_weights(nsamples, samples, weights)
        return samples, weights

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random realizations from the variational posterior.

        Parameters
        ----------
        nsamples : int
            Number of random realizations to generate.

        Returns
        -------
        rvs : Array
            Generated random reaizations.
        """
        samples, weights = self._rvs(nsamples)
        if samples.shape != (self._nvars, nsamples):
            raise RuntimeError("samples has the wrong shape")
        return samples

    def __repr__(self) -> str:
        """
        Return a string representation of the object.
        """
        return "{0}".format(self.__class__.__name__)

    def nvars(self) -> int:
        """
        Get the number of latent variables.

        Returns
        -------
        nvars : int
            Number of latent variables.
        """
        return self._nvars


class MonteCarloIndependentLatentVariableGenerator(LatentVariableGenerator):
    """
    Latent variable generator that uses Monte Carlo quadrature when
    computing the expected loss during training.

    Parameters
    ----------
    variable : IndependentMarginalsVariable
        Variable to generate latent variables for.
    """

    def __init__(self, variable: IndependentMarginalsVariable):
        """
        Initialize the latent variable generator.

        Parameters
        ----------
        variable : IndependentMarginalsVariable
            Latent space variable.
        """
        if not isinstance(variable, IndependentMarginalsVariable):
            raise ValueError(
                "variable must be an IndependentMarginalsVariable"
            )
        self._variable = variable
        super().__init__(variable.nvars(), variable._bkd)
        self._samples = self._bkd.zeros((self.nvars(), 0))

    def _samples_weights(self, nsamples: int) -> Tuple[Array, Array]:
        if self._samples.shape[1] < nsamples:
            self._samples = self._bkd.hstack(
                (self._samples, self._variable.rvs(nsamples))
            )
        return self._samples[:, :nsamples], self._bkd.full(
            (nsamples, 1), 1 / nsamples
        )

    def _rvs(self, nsamples: int) -> Array:
        return self._variable.rvs(nsamples)


class LowDiscrepanySequenceIndependentLatentVariableGenerator(
    LatentVariableGenerator
):
    """
    Latent variable generator that uses low discrepancy sequences for
    quadrature when computing the expected loss during training.

    Parameters
        ----------
        sequence : LowDiscrepancySequence
            Sequence sampling the latent variable.
    """

    def __init__(self, sequence: LowDiscrepancySequence):
        """
        Initialize the latent variable generator.

        Parameters
        ----------
        sequence : LowDiscrepancySequence
            Sequence sampling the latent variable.
        """
        if not isinstance(sequence, LowDiscrepancySequence):
            raise ValueError("sequence must be a LowDiscrepancySequence")
        self._seq = sequence
        super().__init__(sequence.nvars(), sequence._bkd)
        self._samples = self._bkd.zeros((self.nvars(), 0))

    def _samples_weights(self, nsamples: int) -> Tuple[Array, Array]:
        if self._samples.shape[1] < nsamples:
            self._samples = self._seq.rvs(nsamples)
        return self._samples[:, :nsamples], self._bkd.full(
            (nsamples, 1), 1 / nsamples
        )

    def _rvs(self, nsamples: int) -> Array:
        return self._seq._variable.rvs(nsamples)


class QuadratureRuleLatentVariableGenerator(LatentVariableGenerator):
    """
    Latent variable generator that uses QuadratureRule for
    quadrature when computing the expected loss during training.

    Parameters
    ----------
    quad_rule : QuadratureRule
        Quadrature rule for the latent variable space.
    """

    def __init__(self, quad_rule: QuadratureRule):
        """
        Initialize the latent variable generator.

        Parameters
        ----------
        quad_rule : QuadratureRule
            Quadrature rule for the latent variable space.
        """
        if not isinstance(quad_rule, QuadratureRule):
            raise ValueError("quad_rule must be a QaudratureRule")
        self._quad_rule = quad_rule
        super().__init__(quad_rule.nvars(), quad_rule._bkd)

    def _check_samples_weights(
        self, nsamples: int, samples: Array, weights: Array
    ):
        if not hasattr(nsamples, "shape"):
            nsamples = self._bkd.array([nsamples] * self.nvars())
        super()._check_samples_weights(
            self._bkd.prod(nsamples), samples, weights
        )

    def _samples_weights(self, nsamples: int) -> Tuple[Array, Array]:
        # assumes same number of samples used for each dimension
        return self._quad_rule([nsamples] * self.nvars())

    def _rvs(self, nsamples: int) -> Array:
        return self._quad_rule._variable.rvs(nsamples)


class VariationalPosterior(ABC):
    """
    Abstract base class for variational posteriors.

    Parameters
    ----------
    latent_generator : LatentVariableGenerator
        Generator of the latent variables used to sample the variational
        posterior
    nlatent_samples : int
        Number of samples used to compute the ELBO used to train the
        variational posterior
    """

    def __init__(
        self,
        latent_generator: LatentVariableGenerator,
        nlatent_samples: int,
    ):
        """
        Initialize the latent variational posterior.

        Parameters
        ----------
        latent_generator : LatentVariableGenerator
            Generator of the latent variables used to compute the variational
            posterior
        nlatent_samples : int
            Number of samples used to compute the variational posterior
        """
        if not isinstance(latent_generator, LatentVariableGenerator):
            raise ValueError(
                "latent_generator must be a LatentVariableGenerator"
            )
        self._bkd = latent_generator._bkd
        self._nvars = latent_generator.nvars()
        self._latent_generator = latent_generator
        self._latent_samples, self._latent_weights = self._latent_generator(
            nlatent_samples
        )

    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters used to train the variational
        posterior.
        """
        return self._hyp_list

    def __repr__(self) -> str:
        """Return a string representation of the variational posterior."""
        return "{0}({1}, nvars={2}, bkd={3})".format(
            self.__class__.__name__,
            self._hyp_list._short_repr(),
            self._nvars,
            self._bkd.__name__,
        )

    @abstractmethod
    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        """
        Map latent samples to samples from the variational posterior.

        Parameters
        ----------
        latent_samples : Array
            Samples from the latent distribution

        Returns
        -------
        samples : Array
            Samples from the variational posterior
        """
        raise NotImplementedError

    def rvs(self, nsamples: int) -> Array:
        """
        Return random samples from the variational posterior.

        Parameters
        ----------
        nsamples : int
            Number of samples to return

        Returns
        -------
        samples : Array
            Random samples from the variational posterior
        """
        return self._map_from_latent_samples(
            self._latent_generator.rvs(nsamples)
        )

    def _ppf_shape_wrapper(self, latent_samples: Array, param: Array) -> Array:
        self._hyp_list.set_active_opt_params(param)
        self.update()
        return self._map_from_latent_samples(latent_samples).T

    def _ppf_shape_jacobian(
        self, latent_samples: Array, param: Array
    ) -> Array:
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError(
                f"{self}: backend does not support autodifferntiation and a"
                "custom implementation is not implemented"
            )
        return self._bkd.jacobian(
            lambda p: self._ppf_shape_wrapper(latent_samples, p), param[:, 0]
        )


class CholeskyGaussianVariationalPosterior(VariationalPosterior):
    """
    Variational posterior for a multivariate Gaussian distribution with
    Cholesky factorization of the covariance matrix.

    Parameters
    ----------
    prior : DenseCholeskyMultivariateGaussian
        Prior multivariate Gaussian distribution
    nlatent_samples : int
        Number of samples used to compute the variational posterior
    flattened_cholesky_values : Array
        Flattened Cholesky factor of the prior covariance matrix
    mean_values : Array, optional
        Mean of the variational distribution. If None, it is set to zero.
    cholesky_bounds : Union[Tuple[float, float], Array], optional
        Bounds for the Cholesky factor of the variational distribution.
        Default is (-inf, inf)
    mean_bounds : Union[Tuple[float, float], Array], optional
        Bounds for the mean of the variational distribution. Default is
        (-inf, inf)
    latent_generator : LatentVariableGenerator, optional
        Generator of the latent variables used to compute the variational
        posterior. If None, it is set to a Monte Carlo independent latent
        variable generator with a standard multivariate Gaussian
        distribution.
    backend : BackendMixin, optional
        Backend to use for computations. Default is NumpyMixin
    """

    def __init__(
        self,
        prior: DenseCholeskyMultivariateGaussian,
        nlatent_samples: int,
        flattened_cholesky_values: Array,  # only entries on and below diagonal
        mean_values: Array = None,
        cholesky_bounds: Union[Tuple[float, float], Array] = (-np.inf, np.inf),
        mean_bounds: Union[Tuple[float, float], Array] = (-np.inf, np.inf),
        latent_generator: LatentVariableGenerator = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the variational distribution.

        Parameters
        ----------
        prior : DenseCholeskyMultivariateGaussian
            Prior multivariate Gaussian distribution
        nlatent_samples : int
            Number of samples used to compute the variational posterior
        flattened_cholesky_values : Array
            Flattened Cholesky factor of the prior covariance matrix
        mean_values : Array, optional
            Mean of the variational distribution. If None, it is set to zero.
        cholesky_bounds : Union[Tuple[float, float], Array], optional
            Bounds for the Cholesky factor of the variational distribution.
            Default is (-inf, inf)
        mean_bounds : Union[Tuple[float, float], Array], optional
            Bounds for the mean of the variational distribution. Default is
            (-inf, inf)
        latent_generator : LatentVariableGenerator, optional
            Generator of the latent variables used to compute the variational
            posterior. If None, it is set to a Monte Carlo independent latent
            variable generator with a standard multivariate Gaussian
            distribution.
        backend : BackendMixin, optional
            Backend to use for computations. Default is NumpyMixin
        """
        nvars = prior.nvars()
        if latent_generator is None:
            latent_variable = IndependentMarginalsVariable(
                [GaussianMarginal(0.0, 1.0, backend)] * nvars, backend=backend
            )
            latent_generator = MonteCarloIndependentLatentVariableGenerator(
                latent_variable
            )
        super().__init__(latent_generator, nlatent_samples)
        if mean_values is None:
            mean_values = backend.zeros((nvars))
        self._mean = HyperParameter(
            "mean",
            nvars,
            mean_values,
            mean_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._cholesky = CholeskyHyperParameter(
            "cholesky",
            nvars,
            flattened_cholesky_values,
            cholesky_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._mean, self._cholesky])
        self._setup_divergence(prior)

    def mean(self) -> Array:
        """
        Return the mean of the variational distribution.

        Returns
        -------
        mean : Array
            Mean of the variational distribution
        """
        return self._mean.get_values()[:, None]

    def covariance(self) -> Array:
        """
        Return the covariance matrix of the variational distribution.

        Returns
        -------
        cov: Array
            Covariance matrix of the variational distribution
        """
        chol = self._cholesky.get_cholesky_factor()
        return chol @ chol.T

    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        chol = self._cholesky.get_cholesky_factor()
        return self.mean() + chol @ latent_samples

    def update(self):
        """
        Update the variational distribution by computing the Cholesky factor
        and the mean. Must be called after any hyperparameters are updated.
        """
        mean1 = self._mean.get_values()[:, None]
        chol1 = self._cholesky.get_cholesky_factor()
        self._divergence.set_left_distribution(mean1, chol1)

    def _setup_divergence(self, prior: IndependentMultivariateGaussian):
        """
        Set up the divergence for the variational distribution.

        Parameters
        ----------
        prior : IndependentMultivariateGaussian
            Prior multivariate Gaussian distribution
        """
        self._divergence = CholeskyBasedGaussianExactKLDivergence(
            prior.nvars(), prior._bkd
        )
        self._divergence.set_right_distribution(
            prior.mean(), prior.covariance()
        )

    def _ppf_shape_jacobian(
        self, latent_samples: Array, param: Array
    ) -> Array:
        nvars = latent_samples.shape[0]
        jac = self._bkd.zeros(
            (
                latent_samples.shape[1],
                nvars,
                self._hyp_list.nvars(),
            )
        )
        jac[:, :nvars, :nvars] = self._bkd.eye(nvars)
        cnt = 0
        for ii in range(nvars):
            jac[:, ii, nvars + cnt : nvars + cnt + ii + 1] = latent_samples[
                : ii + 1
            ].T
            cnt += ii + 1
        # import torch

        # torch.set_printoptions(linewidth=1000)
        # print(jac)
        # print(super()._ppf_shape_jacobian(latent_samples, param))
        # assert self._bkd.allclose(
        #     jac, super()._ppf_shape_jacobian(latent_samples, param)
        # )
        return jac


class IndependentGaussianVariationalPosterior(VariationalPosterior):
    """
    Variational posterior for a multivariate Gaussian distribution with
    independent marginals.

    Parameters
    ----------
    prior : IndependentMultivariateGaussian
        Prior multivariate Gaussian distribution
    nlatent_samples : int
        Number of samples used to compute the variational posterior
    std_diag_values : Array
        Standard deviations of the variational distribution
    mean_values : Array, optional
        Mean of the variational distribution. If None, it is set to zero.
    std_diag_bounds : Union[Tuple[float, float], Array], optional
        Bounds for the standard deviations of the variational distribution.
        Default is (0, inf)
    mean_bounds : Union[Tuple[float, float], Array], optional
        Bounds for the mean of the variational distribution. Default is
        (-inf, inf)
    latent_generator : LatentVariableGenerator, optional
        Generator of the latent variables used to compute the variational
        posterior. If None, it is set to a Monte Carlo independent latent
        variable generator with a standard multivariate Gaussian
        distribution.
    backend : BackendMixin, optional
        Backend to use for computations. Default is NumpyMixin
    """

    def __init__(
        self,
        prior: IndependentMultivariateGaussian,
        nlatent_samples: int,
        std_diag_values: Array,
        mean_values: Array = None,
        std_diag_bounds: Union[Tuple[float, float], Array] = (0, np.inf),
        mean_bounds: Union[Tuple[float, float], Array] = (-np.inf, np.inf),
        latent_generator: LatentVariableGenerator = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the variational posterior.

        Parameters
        ----------
        prior : IndependentMultivariateGaussian
            Prior multivariate Gaussian distribution
        nlatent_samples : int
            Number of samples used to compute the variational posterior
        std_diag_values : Array
            Standard deviations of the variational distribution
        mean_values : Array, optional
            Mean of the variational distribution. If None, it is set to zero.
        std_diag_bounds : Union[Tuple[float, float], Array], optional
            Bounds for the standard deviations of the variational distribution.
            Default is (0, inf)
        mean_bounds : Union[Tuple[float, float], Array], optional
            Bounds for the mean of the variational distribution. Default is
            (-inf, inf)
        latent_generator : LatentVariableGenerator, optional
            Generator of the latent variables used to compute the variational
            posterior. If None, it is set to a Monte Carlo independent latent
            variable generator with a standard multivariate Gaussian
            distribution.
        backend : BackendMixin, optional
            Backend to use for computations. Default is NumpyMixin
        """
        nvars = prior.nvars()
        if latent_generator is None:
            latent_variable = IndependentMarginalsVariable(
                [GaussianMarginal(0.0, 1.0, backend)] * nvars, backend=backend
            )
            latent_generator = MonteCarloIndependentLatentVariableGenerator(
                latent_variable
            )
        super().__init__(latent_generator, nlatent_samples)
        if mean_values is None:
            mean_values = backend.zeros((nvars))
        self._mean = HyperParameter(
            "mean",
            nvars,
            mean_values,
            mean_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._std_diag = HyperParameter(
            "std_diag",
            nvars,
            std_diag_values,
            std_diag_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._mean, self._std_diag])
        self._setup_divergence(prior)

    def mean(self) -> Array:
        """
        Return the mean of the variational distribution.

        Returns
        -------
        mean : Array
            Mean of the variational distribution
        """
        return self._mean.get_values()[:, None]

    def covariance(self) -> Array:
        """
        Return the covariance matrix of the variational distribution.

        Returns
        -------
        cov : Array
            Covariance matrix of the variational distribution
        """
        return self._bkd.diag(self._std_diag.get_values() ** 2)

    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        return (
            self.mean() + self._std_diag.get_values()[:, None] * latent_samples
        )

    def _ppf_shape_jacobian(
        self, latent_samples: Array, param: Array
    ) -> Array:
        jac = self._bkd.empty(
            (
                latent_samples.shape[1],
                latent_samples.shape[0],
                self._hyp_list.nvars(),
            )
        )
        jac[:, :, 0] = 1.0
        jac[:, :, 1] = latent_samples.T
        return jac

    def _setup_divergence(self, prior: IndependentMultivariateGaussian):
        """
        Set up the divergence for the variational distribution.

        Parameters
        ----------
        prior : IndependentMultivariateGaussian
            Prior multivariate Gaussian distribution
        """
        self._divergence = IndependentGaussianExactKLDivergence(
            prior.nvars(), prior._bkd
        )
        self._divergence.set_right_distribution(
            prior.mean(), prior.covariance_diagonal()[:, None]
        )

    def update(self):
        """
        Update the variational distribution by computing the mean and the
        standard deviations. Must be called after any hyperparameters are
        updated.
        """
        mean1 = self._mean.get_values()[:, None]
        diag1 = self._std_diag.get_values()[:, None] ** 2
        self._divergence.set_left_distribution(mean1, diag1)


class IndependentBetaVariationalPosterior(VariationalPosterior):
    """
    Variational posterior for a multivariate beta distribution with
    independent marginals.

    Parameters
    ----------
    prior : IndependentMarginalsVariable
        Prior multivariate beta distribution
    nlatent_samples : int
        Number of samples used to compute the variational posterior
    ashape_values : Array
        Shape parameters of the variational distribution
    bshape_values : Array
        Shape parameters of the variational distribution
    bounds : Array
        Bounds of the variational distribution
    ashape_bounds : Union[Tuple[float, float], Array], optional
        Bounds for the shape parameters of the variational distribution.
        Default is (1, inf)
    bshape_bounds : Union[Tuple[float, float], Array], optional
        Bounds for the shape parameters of the variational distribution.
        Default is (1, inf)
    latent_generator : LatentVariableGenerator, optional
        Generator of the latent variables used to compute the variational
        posterior. If None, it is set to a Halton sequence independent latent
        variable generator.
    backend : BackendMixin, optional
        Backend to use for computations. Default is NumpyMixin
    """

    def __init__(
        self,
        prior: IndependentMarginalsVariable,
        nlatent_samples: int,
        ashape_values: Array,
        bshape_values: Array,
        bounds: Array,
        ashape_bounds: Union[Tuple[float, float], Array] = (1, np.inf),
        bshape_bounds: Union[Tuple[float, float], Array] = (1, np.inf),
        latent_generator: LatentVariableGenerator = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Intialize the variational posterior.

        Parameters
        ----------
        prior : IndependentMarginalsVariable
            Prior multivariate beta distribution
        nlatent_samples : int
            Number of samples used to compute the variational posterior
        ashape_values : Array
            Shape parameters of the variational distribution
        bshape_values : Array
            Shape parameters of the variational distribution
        bounds : Array
            Bounds of the variational distribution
        ashape_bounds : Union[Tuple[float, float], Array], optional
            Bounds for the shape parameters of the variational distribution.
            Default is (1, inf)
        bshape_bounds : Union[Tuple[float, float], Array], optional
            Bounds for the shape parameters of the variational distribution.
            Default is (1, inf)
        latent_generator : LatentVariableGenerator, optional
            Generator of the latent variables used to compute the variational
            posterior. If None, it is set to a Halton sequence independent
            latent variable generator.
        backend : BackendMixin, optional
            Backend to use for computations. Default is NumpyMixin
        """
        nvars = prior.nvars()
        if latent_generator is None:
            latent_variable = IndependentMarginalsVariable(
                [UniformMarginal(0, 1, backend)] * nvars, backend=backend
            )
            # latent_generator = MonteCarloIndependentLatentVariableGenerator(
            #     latent_variable
            # )
            from pyapprox.expdesign.sequences import HaltonSequence

            sequence = HaltonSequence(
                nvars, start_idx=1, variable=latent_variable, bkd=backend
            )
            latent_generator = (
                LowDiscrepanySequenceIndependentLatentVariableGenerator(
                    sequence
                )
            )
        super().__init__(latent_generator, nlatent_samples)
        self._ashapes = HyperParameter(
            "ashapes",
            nvars,
            ashape_values,
            ashape_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._bshapes = HyperParameter(
            "bshapes",
            nvars,
            bshape_values,
            bshape_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._ashapes, self._bshapes])
        if bounds.shape != (nvars, 2):
            raise ValueError("Bounds has the wrong shape")
        # nquad samples effects accuracy of variational inference gradients
        marginals = [
            BetaMarginal(
                ashape_values[ii],
                bshape_values[ii],
                *bounds[ii],
                nquad_samples=1001,
                backend=self._bkd,
            )
            for ii in range(nvars)
        ]
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )
        self._prior = prior

    def update(self):
        """
        Update the variational distribution.
        Must be called after any hyperparameters are updated.
        """
        ashapes = self._ashapes.get_values()
        bshapes = self._bshapes.get_values()
        for ii, marginal in enumerate(self.marginals()):
            marginal.set_shapes(ashapes[ii], bshapes[ii])

    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        return self._variable.ppf(latent_samples)

    def _ppf_shape_jacobian(
        self, latent_samples: Array, param: Array
    ) -> Array:
        """
        Compute the Jacobian of the PPF with respect to the shape
        parameters of the posterior used to generate samples mapped
        from the latent space to the model space.

        Parameters
        ----------
        latent_samples : Array
            Latent samples.

        Returns
        -------
        jac : Array
            Shape Jacobian of the model variables.

        """
        self._hyp_list.set_active_opt_params(param[:, 0])
        self.update()
        # len(jacs) = nvars
        # jacs[i] ~ (nsamples, nparams_1d)
        # jacs[i] = [[ai1, bi1], [ai2, bi2] ...] for ith variable
        # but hyperparams stored a(i)1, a(i+1)2, ..., b(i)1, b(i)2 ...
        # so reorder
        nvars = latent_samples.shape[0]
        jacs = self._variable.ppf_shape_jacobian(latent_samples, False)
        jac = self._bkd.zeros(
            (
                latent_samples.shape[1],
                nvars,
                self._hyp_list.nvars(),
            )
        )
        for ii in range(nvars):
            jac[:, ii, ii] = jacs[:, ii, 0]  # self._ashapes
            jac[:, ii, nvars + ii] = jacs[:, ii, 1]  # self._bshapes
        return jac

    def marginals(self):
        """Get the marginal distributions.

        Returns
        -------
        marginals : List[Marginal]
            Marginal distributions.
        """
        return self._variable.marginals()

    def _divergence(self):
        """
        Compute the divergence between the variational posterior and the prior.

        Returns
        -------
        dkld_div : Array
            Divergence between the variational posterior and the prior.

        """
        return self._bkd.atleast2d(self._variable.kl_divergence(self._prior))


class DirichletVariationalPosterior(VariationalPosterior):
    def __init__(
        self,
        prior: DirichletVariable,
        nlatent_samples: int,
        ashape_values: Array,
        ashape_bounds: Union[Tuple[float, float], Array] = (1, np.inf),
        latent_generator: LatentVariableGenerator = None,
        backend: BackendMixin = NumpyMixin,
    ):
        nvars = prior.nvars()
        if latent_generator is None:
            latent_variable = IndependentMarginalsVariable(
                [UniformMarginal(0, 1, backend)] * nvars, backend=backend
            )
            # self._latent_generator = MonteCarloIndependentLatentVariableGenerator(
            #     latent_variable
            # )
            from pyapprox.expdesign.sequences import HaltonSequence

            sequence = HaltonSequence(
                nvars, start_idx=1, variable=latent_variable, bkd=backend
            )
            latent_generator = (
                LowDiscrepanySequenceIndependentLatentVariableGenerator(
                    sequence
                )
            )
        if latent_generator.nvars() != ashape_values.shape[0]:
            raise ValueError(
                "latent generator and shape_values are inconsistent"
            )
        super().__init__(latent_generator, nlatent_samples)
        self._ashapes = HyperParameter(
            "ashapes",
            nvars,
            ashape_values,
            ashape_bounds,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._ashapes])
        # nquad samples effects accuracy of variational inference gradients
        self._variable = DirichletVariable(
            self._ashapes.get_values(), backend=self._bkd
        )
        self._prior = prior

    def update(self):
        """
        Update the variational distribution.
        Must be called after any hyperparameters are updated.
        """
        self._variable.set_shapes(self._ashapes.get_values())

    def _map_from_latent_samples(self, latent_samples: Array) -> Array:
        # map uniform samples to gamma samples
        gamma_samples = self._bkd.stack(
            [
                marginal.ppf(latent_samples[ii])
                for ii, marginal in enumerate(
                    self._variable._gamma_variable.marginals()
                )
            ],
            axis=0,
        )
        # map gamma samples to dirichlet samples
        return gamma_samples / self._bkd.sum(gamma_samples, axis=0)[None, :]

    def _divergence(self):
        """
        Compute the divergence between the variational posterior and the prior.

        Returns
        -------
        dkld_div : Array
            Divergence between the variational posterior and the prior.

        """
        return self._bkd.atleast2d(self._variable.kl_divergence(self._prior))


class NegELBO(SingleSampleModel):
    """
    Negative Evidence Lower Bound objective used to train the variational posterior.
    """

    def __init__(
        self,
        loglike: ModelBasedLogLikelihoodMixin,
        posterior: VariationalPosterior,
        nsamples: int = 1000,
        bkd_jacobian_supported: bool = False,
    ):
        """
        Initialize the negative evidence lower bound.

        Parameters
        ----------
        loglike : ModelBasedLogLikelihoodMixin
            Log likelihood model.
        posterior : VariationalPosterior
            Variational posterior.
        nsamples : int, optional
            Number of samples. Default is 1000.
        bkd_jacobian_supported : boolean, optional
            If True, the code will use auto differntiation through
            the likelihood (and the model it calls) and the posterior
            KL divergence and function mapping latent space samples.
            If False, the code will call .jacobian of these functions
            Use True with caution.
        """
        super().__init__(posterior._bkd)
        if not isinstance(posterior, VariationalPosterior):
            raise ValueError(
                "posterior must be an instance of VariationalPosterior"
            )
        self._posterior = posterior
        self._loglike = loglike
        self._hyp_list = (
            self._posterior.hyp_list()  # + self._loglike.hyplist()
        )
        self._bkd_jacobian_supported = bkd_jacobian_supported
        print(self.jacobian_implemented(), self.apply_jacobian_implemented())

    def nvars(self) -> int:
        return self._hyp_list.nactive_vars()

    def nqoi(self) -> int:
        return 1

    def jacobian_implemented(self) -> bool:
        if not self._bkd.jacobian_implemented():
            return False
        return (
            self._bkd_jacobian_supported
            or self._loglike.jacobian_implemented()
        )

    def apply_jacobian_implemented(self) -> bool:
        if not self._bkd.jvp_implemented():
            return False
        return (
            self._bkd_jacobian_supported
            or self._loglike.apply_jacobian_implemented()
        )

    def _loglike_jacobian_wrt_samples(self) -> Array:
        # compute grad of loglike dldx with respect to model variables x

        # After profiling the code. This function is slowing it
        # down by orders of magnitude compared to
        # self._bkd.jacobian(
        #     lambda x: self._values(x[:, None])[:, 0], param[:, 0]
        # )
        # called in self._jacobian -
        # but this requires autograd can be used for all components
        samples = self._posterior._map_from_latent_samples(
            self._posterior._latent_samples
        )
        # stacking is slow but not as slow as unvectorized computation
        # of self._loglike.jacobian
        return self._bkd.stack(
            [
                self._loglike.jacobian(sample[:, None])[0]
                for sample in samples.T
            ]
        )

    def _loglike_apply_jacobian_wrt_samples(self, vecs: Array) -> Array:
        # compute grad of loglike dldx @ v with respect to model variables x

        samples = self._posterior._map_from_latent_samples(
            self._posterior._latent_samples
        )
        return self._bkd.stack(
            [
                self._loglike.apply_jacobian(sample[:, None], vecs[ii])[0]
                for ii, sample in enumerate(samples.T)
            ]
        )

    def _loglike_jacobian_wrt_param(
        self, jac_values_at_samples: Array, dxdp: Array
    ) -> Array:
        # compute gradient of loglikelihood with respect to parameters
        # using the chain rule dldp = dldx dxdp
        weights = self._posterior._latent_weights
        return (
            weights[:, 0]
            @ self._bkd.einsum("ij, ijk->ik", jac_values_at_samples, dxdp)[
                None, :
            ]
        )

    def _jacobian(self, param: Array) -> Array:
        # The following requires posterior and likelihood
        # to both be implemented with autograd. This does not allow
        # for user provided model jacobians
        if self._bkd_jacobian_supported:
            return self._bkd.jacobian(
                lambda x: self._values(x[:, None])[:, 0], param[:, 0]
            )

        # update posterior parameters
        self._hyp_list.set_active_opt_params(param[:, 0])
        self._posterior.update()

        # compute grad of loglike dldx with respect to model variables x
        jac_values_at_samples = self._loglike_jacobian_wrt_samples()

        # compute grad of model variables dxdp with respect to parameters p
        # of variational distribution used to map latent samples to model
        # variables
        dxdp = self._reparameterized_samples_jacobian(param)

        # compute gradient of loglikelihood with respect to parameters
        # using the chain rule dldp = dldx dxdp
        dldp = self._loglike_jacobian_wrt_param(jac_values_at_samples, dxdp)

        return -dldp + self._bkd.jacobian(
            lambda x: self._posterior_divergence(x[:, None])[:, 0], param[:, 0]
        )

    def _apply_jacobian(self, param: Array, vec: Array) -> Array:
        if self._bkd_jacobian_supported:
            return self._bkd.jvp(
                lambda x: self._values(x[:, None])[:, 0],
                param[:, 0],
                vec[:, 0],
            )

        # update posterior parameters
        self._hyp_list.set_active_opt_params(param[:, 0])
        self._posterior.update()

        # compute grad of model variables dxdp with respect to parameters p
        # of variational distribution used to map latent samples to model
        # variables
        dxdp = self._reparameterized_samples_jacobian(param)

        weights = self._posterior._latent_weights
        jvp = -weights.T @ self._loglike_apply_jacobian_wrt_samples(
            dxdp @ vec
        ) + self._bkd.jvp(
            lambda x: self._posterior_divergence(x[:, None])[:, 0],
            param[:, 0],
            vec[:, 0],
        )

        return jvp

    def apply_jacobian_implemented(self) -> bool:
        return True

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _posterior_divergence(self, params: Array) -> Array:
        # for hp in self._hyp_list._hyper_params:
        #     hp.detach()
        self._hyp_list.set_active_opt_params(params[:, 0])
        self._posterior.update()
        return self._posterior._divergence()

    def _reparameterized_samples(self, params: Array) -> Array:
        # for hp in self._hyp_list._hyper_params:
        #     hp.detach()
        self._hyp_list.set_active_opt_params(params[:, 0])
        self._posterior.update()
        return self._posterior._map_from_latent_samples(
            self._posterior._latent_samples
        )

    def _reparameterized_samples_jacobian(self, params: Array) -> Array:
        return self._posterior._ppf_shape_jacobian(
            self._posterior._latent_samples, params
        )

    def _evaluate(self, params: Array) -> Array:
        self._hyp_list.set_active_opt_params(params[:, 0])
        self._posterior.update()
        samples = self._posterior._map_from_latent_samples(
            self._posterior._latent_samples
        )
        # TODO make this a sample average objective
        weights = self._posterior._latent_weights
        expected_loglike = self._bkd.atleast2d(
            self._loglike(samples)[:, 0] @ weights[:, 0]
        )
        return -(expected_loglike - self._posterior._divergence())
        # return -expected_loglike


class VariationalInverseProblem:
    """
    A class for formulating and solving Bayesian Variational Inference using
    ELBO optimization.
    """

    def __init__(
        self,
        prior: JointVariable,
        loglike: ModelBasedLogLikelihoodMixin,
        posterior: VariationalPosterior,
        bkd_jacobian_supported: bool = False,
    ):
        """
        Initialize the variational inverse problem

        Parameters
        ----------
        prior : JointVariable
            The prior distribution over the model parameters.
        loglike : ModelBasedLogLikelihoodMixin
            The model-based log-likelihood function.
        posterior : VariationalPosterior
            The variational posterior distribution.
        bkd_jacobian_supported : boolean, optional
        If True, the code will use auto differntiation through
            the likelihood (and the model it calls) and the posterior
            KL divergence and function mapping latent space samples.
            If False, the code will call .jacobian of these functions
            Use True with caution
        """
        self._bkd = posterior._bkd
        self._prior = prior
        self._neg_elbo = NegELBO(
            loglike, posterior, bkd_jacobian_supported=bkd_jacobian_supported
        )
        self._optimizer = None

    def fit(self, iterate: Optional[Array] = None):
        """
        Minimize the ELBO using the optimization algorithm.

        Parameters
        ----------
        iterate : Array, optional
            The initial iterate (default is None).
        """
        if self._neg_elbo.hyp_list().nactive_vars() == 0:
            warnings.warn("No active parameters so fit was not called")
            return
        if self._optimizer is None:
            self.set_optimizer(self.default_optimizer())
        if iterate is None:
            # iterate = self._optimizer._initial_interate_gen()
            iterate = self._neg_elbo.hyp_list().get_active_opt_params()[
                :, None
            ]
        res = self._optimizer.minimize(iterate)
        active_opt_params = res.x[:, 0]
        self._neg_elbo.hyp_list().set_active_opt_params(active_opt_params)

    def set_optimizer(self, optimizer: MultiStartOptimizer):
        """
        Set the optimization algorithm.

        Parameters
        ----------
        optimizer : MultiStartOptimizer
            The optimization algorithm.
        """
        if not isinstance(optimizer, MultiStartOptimizer):
            raise ValueError(
                "optimizer {0} must be instance of MultiStartOptimizer".format(
                    optimizer
                )
            )
        self._optimizer = optimizer
        self._optimizer.set_objective_function(self._neg_elbo)
        self._optimizer.set_bounds(
            self._neg_elbo.hyp_list().get_active_opt_bounds()
        )

    def _default_iterator_gen(self):
        """
        Generate the default iterate generator.

        Returns
        -------
        iterate_gen : OptimizerIterateGenerator
            The default iterate generator.
        """
        iterate_gen = RandomUniformOptimzerIterateGenerator(
            self._neg_elbo.hyp_list().nactive_vars(), backend=self._bkd
        )
        iterate_gen.set_bounds(
            self._bkd.to_numpy(
                self._neg_elbo.hyp_list().get_active_opt_bounds()
            )
        )
        return iterate_gen

    def default_optimizer(
        self,
        ncandidates: int = 1,
        verbosity: int = 0,
        gtol: float = 1e-8,
        maxiter: int = 1000,
        iterate_gen: OptimizerIterateGenerator = None,
        local_method="L-BFGS-B",
    ) -> MultiStartOptimizer:
        """
        Get the default optimization algorithm.

        Parameters
        ----------
        ncandidates : int, optional
            The number of candidates (default is 1).
        verbosity : int, optional
            The verbosity level (default is 0).
        gtol : float, optional
            The gradient tolerance (default is 1e-8).
        maxiter : int, optional
            The maximum number of iterations (default is 1000).
        iterate_gen : OptimizerIterateGenerator, optional
            The iterate generator (default is None).
        local_method : str, optional
            The local optimization method (default is "L-BFGS-B").

        Returns
        -------
        optimizer : MultiStartOptimizer
            The default optimization algorithm.
        """
        local_optimizer = ScipyConstrainedOptimizer()
        local_optimizer.set_options(
            gtol=gtol,
            maxiter=maxiter,
            method=local_method,
        )
        local_optimizer.set_verbosity(verbosity - 1)
        ms_optimizer = MultiStartOptimizer(
            local_optimizer, ncandidates=ncandidates
        )
        if iterate_gen is None:
            iterate_gen = self._default_iterator_gen()
        if not isinstance(iterate_gen, OptimizerIterateGenerator):
            raise ValueError(
                "iterate_gen must be an instance of OptimizerIterateGenerator"
            )
        ms_optimizer.set_initial_iterate_generator(iterate_gen)
        ms_optimizer.set_verbosity(verbosity)
        return ms_optimizer

    def __repr__(self) -> str:
        """Get the string representation of the object."""
        return "{0}({1}, {2}, {3})".format(
            self.__class__.__name__,
            self._prior,
            self._neg_elbo._loglike,
            self._neg_elbo._posterior,
        )


# TODO: read Variational Gaussian Copula Inference.
# http://proceedings.mlr.press/v51/han16.pdf
