"""
This module contains benchmarks and tools for testing and analyzing Bayesian Optimal Experimental
Design (OED) algorithms. It provides functionality to evaluate the performance of Bayesian OED
methods and tools to assist in the analysis of diagnostics such as bias, variance, mean squared
error (MSE), and posterior marginals.

Purpose
-------
The benchmarks implemented in this module are designed to test the accuracy, efficiency, and
robustness of Bayesian OED algorithms. The tools provided enable detailed analysis of the
expected information gain (EIG) and posterior distributions, facilitating comparisons between
different experimental designs and algorithms.

Features
--------
- Benchmarks for Bayesian OED algorithms.
- Tools for computing diagnostics such as bias, variance, and MSE.
- Visualization utilities for posterior marginals and diagnostics.

Usage
-----
This module is intended for researchers and practitioners working on Bayesian OED problems who
require benchmarks and analysis tools to evaluate their algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
from scipy.stats import norm
import matplotlib


from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.variables.marginals import GaussianMarginal
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    JointVariable,
)
from pyapprox.variables.gaussian import (
    MultivariateGaussian,
    DenseCholeskyMultivariateGaussian,
)
from pyapprox.interface.model import Model, DenseMatrixLinearModel
from pyapprox.bayes.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    GaussianPushForward,
    _compute_expected_kl_divergence,
)
from pyapprox.expdesign.optbayes import (
    BayesianOED,
    KLBayesianOED,
    BayesianOEDForPrediction,
    IndependentGaussianOEDInnerLoopLogLikelihood,
    NoiseStatistic,
    PredictionOEDDeviationMeasure,
    SampleAverageStat,
    BayesianOEDDataGenerator,
)


class Linear1DRegressionModel(DenseMatrixLinearModel):
    """
    Linear regression model for 1D data.

    Parameters
    ----------
    design : Array
        Design matrix for the regression model.
    degree : int
        Maximum degree of the polynomial basis.
    min_degree : int, optional
        Minimum degree of the polynomial basis (default is 0).
    backend : BackendMixin, optional
        Backend implementation for numerical operations (default is NumpyMixin).
    """

    def __init__(
        self,
        design,
        degree: int,
        min_degree: int = 0,
        backend: BackendMixin = NumpyMixin,
    ):
        assert degree >= min_degree
        self._design = design
        self._degree = degree
        super().__init__(
            self._design.T
            ** backend.arange(min_degree, self._degree + 1)[None, :],
            backend=backend,
        )


class ExponentialQoIModel(Model):
    """
    Exponential of a linear regression model.

    Parameters
    ----------
    regression_model: Linear1DRegressionModel
        The linear regression model.
    """

    def __init__(self, regression_model: Linear1DRegressionModel):
        super().__init__(regression_model._bkd)
        self._regression_model = regression_model

    def _values(self, samples: Array) -> Array:
        return self._bkd.exp(self._regression_model(samples))

    def nqoi(self) -> int:
        return self._regression_model.nqoi()

    def nvars(self) -> int:
        return self._regression_model.nvars()

    def matrix(self) -> Array:
        return self._regression_model.matrix()


class LinearGaussianBayesianOEDBenchmark:
    """
    Defines a Bayesian Optimal Experimental Design (OED) benchmark consisting
    of a linear observation model, a Gaussian prior and independent Gaussian
    noise.

    Parameters
    ----------
    nobs : int
        Number of observations.
    min_degree : int
        Minimum degree of the polynomial basis.
    degree : int
        Maximum degree of the polynomial basis.
    noise_std : float
        Standard deviation of the noise.
    prior_std : float
        Standard deviation of the prior distribution.
    backend : BackendMixin
        Backend implementation for numerical operations.
    """

    def __init__(
        self,
        nobs: int,
        min_degree: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        backend: BackendMixin,
    ):
        self._nobs = nobs
        self._min_degree = min_degree
        self._degree = degree
        self._noise_std = noise_std
        self._prior_std = prior_std
        self._bkd = backend

        # Initialize problem components
        self._design = self._setup_design_locations()
        self._noise_cov_diag = self._setup_noise_covariance_diag()
        self._prior = self._setup_prior()
        self._obs_model = self._setup_observation_model()

    def _setup_design_locations(self) -> Array:
        """
        Set up the design locations of the experiments.

        Returns
        -------
        design: Array (1, nobs)
            Design matrix for the experimental conditions.
        """
        design = self._bkd.linspace(-1, 1, self._nobs - 2)[None, :]
        design = self._bkd.sort(
            self._bkd.hstack(
                (
                    design[0],
                    self._bkd.asarray([-1 / np.sqrt(5), 1 / np.sqrt(5)]),
                )
            )
        )[None, :]
        return design

    def _setup_noise_covariance_diag(self) -> Array:
        """
        Set up the noise covariance matrix.

        Returns
        -------
        noise_cov_diag: Array (nobs,)
            Noise covariance matrix (diagonal).
        """
        return self._bkd.full((self._nobs,), self._noise_std**2)

    def _setup_prior(self) -> JointVariable:
        """
        Set up the prior distribution.

        Returns
        -------
        prior: JointVariable
            Prior distribution for the model parameters.
        """
        return IndependentMarginalsVariable(
            [norm(0, self._prior_std)] * (self._degree - self._min_degree + 1),
            backend=self._bkd,
        )

    def _setup_observation_model(self) -> Model:
        """
        Set up the observation model.

        Returns
        -------
        obs_model: Model
            Observation model for the experimental design.
        """
        return Linear1DRegressionModel(
            self._design,
            self._degree,
            min_degree=self._min_degree,
            backend=self._bkd,
        )

    def get_prior(self) -> JointVariable:
        """
        Return the prior distribution.

        Returns
        -------
        prior: JointVariable
            Prior distribution for the model parameters.
        """
        return self._prior

    def get_noise_covariance_diag(self) -> Array:
        """
        Return the noise covariance matrix.

        Returns
        -------
        noise_cov_diag: Array (nobs,)
            Noise covariance matrix (diagonal).
        """
        return self._noise_cov_diag

    def get_observation_model(self) -> Model:
        """
        Return the observation model.

        Returns
        -------
        obs_model: Model
            Observation model for the experimental design.
        """
        return self._obs_model

    def get_design_matrix(self) -> Array:
        """
        Return the design matrix.

        Returns
        -------
        design : Array (nobs, nvars)
            Design matrix for the experimental conditions.
        """
        return self._design

    def _noise_covariance_diag(self, design_weights: Array) -> Array:
        """
        Compute the noise covariance matrix adjusted by design weights.

        Parameters
        ----------
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        weigted_noise_cov_diag : Array (nobs,)
            Adjusted noise covariance matrix (diagonal).
        """
        noise_cov_diag = self.get_noise_covariance_diag()
        noise_cov_diag_inv = 1.0 / noise_cov_diag
        return 1.0 / (noise_cov_diag_inv * design_weights[:, 0])

    def exact_expected_information_gain(self, design_weights: Array) -> float:
        """
        Compute the exact expected information gain using properties of the
        Gauss conugate posterior.

        Parameters
        ----------
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        eig: float
            Exact expected information gain.
        """
        obs_model = self.get_observation_model()
        prior_variable = self.get_prior()
        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_model.matrix(),
            prior_variable.mean(),
            prior_variable.covariance(),
            self._bkd.diag(self._noise_covariance_diag(design_weights)),
            backend=self._bkd,
        )
        # expected_kl_divergence does not depend on specific value of obs_idx
        # so just make values up
        dummy_obs = self._bkd.zeros(
            (self.get_noise_covariance_diag().shape[0], 1)
        )
        laplace.compute(dummy_obs)
        return laplace.expected_kl_divergence()


class LinearGaussianBayesianOEDForPredictionBenchmark(
    LinearGaussianBayesianOEDBenchmark
):
    """
    Defines a Bayesian Optimal Experimental Design (OED) benchmark consisting
    of a linear observation model, a Gaussian prior and independent Gaussian
    noise.

    Parameters
    ----------
    nobs : int
        Number of observations.
    min_degree : int
        Minimum degree of the polynomial basis.
    degree : int
        Maximum degree of the polynomial basis.
    noise_std : float
        Standard deviation of the noise.
    prior_std : float
        Standard deviation of the prior distribution.
    backend : BackendMixin
        Backend implementation for numerical operations.
    """

    def __init__(
        self,
        nobs: int,
        min_degree: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        nqoi,
        backend: BackendMixin,
    ):
        super().__init__(
            nobs, min_degree, degree, noise_std, prior_std, backend
        )
        self._nqoi = nqoi
        self._qoi_design = self._setup_qoi_locations()
        self._qoi_model = self._setup_qoi_model()
        self._qoi_quad_weights = self._bkd.full((nqoi, 1), 1.0 / nqoi)

    def _setup_qoi_locations(self) -> Array:
        """
        Set up the QoI locations.

        Returns
        -------
        design: Array (1, nqoi)
            Design matrix associated with the QoI.
        """
        design = self._bkd.linspace(-2 / 3, 2 / 3, self._nqoi)[None, :]
        return design

    def _setup_qoi_model(self) -> Model:
        """
        Set up the QoI model.

        Returns
        -------
        qoi_model: Model
            QoI model for the experimental design.
        """
        return Linear1DRegressionModel(
            self._qoi_design,
            self._degree,
            min_degree=self._min_degree,
            backend=self._bkd,
        )

    def get_qoi_model(self) -> Model:
        """
        Return the QoI model.

        Returns
        -------
        qoi_model: Model
            QoI model for the experimental design.
        """
        return self._qoi_model

    def get_qoi_locations(self) -> Array:
        """
        Return the prediction locations associated with the QoI.

        Returns
        -------
        design : Array (nobs, nvars)
            locations for the QoI.
        """
        return self._qoi_design

    def get_qoi_quad_weights(self) -> Array:
        """
        Return the weigthings applied to the QoI.

        Returns
        -------
        qoi_weights : Array (nqoi, 1)
            locations for the QoI.
        """
        return self._qoi_quad_weights


class BayesianOEDDiagnostics(ABC):
    """
    Computes diagnostics, such as the Mean Squared Error (MSE) and effective
    sample size,  for Bayesian Optimal Experimental Design (OED).

    Parameters
    ----------
    problem : LinearGaussianBayesianOEDBenchmark
        Instance of the LinearGaussianBayesianOEDBenchmark class defining the
        experimental design problem.
    """

    def __init__(self, problem: LinearGaussianBayesianOEDBenchmark):
        self._problem = problem
        self._bkd = self._problem._bkd
        self._data_gen = BayesianOEDDataGenerator(self._bkd)

    def data_generator(self) -> BayesianOEDDataGenerator:
        return self._data_gen

    def setup_oed(self) -> BayesianOED:
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            self._problem.get_noise_covariance_diag()[:, None],
            backend=self._bkd,
        )
        return self._setup_OED(innerloop_loglike)

    @abstractmethod
    def _setup_OED(
        self, innerloop_loglike: IndependentGaussianOEDInnerLoopLogLikelihood
    ) -> BayesianOED:
        raise NotImplementedError

    @abstractmethod
    def compute_utility(
        self,
        innerloop_loglike: IndependentGaussianOEDInnerLoopLogLikelihood,
        outerloop_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_samples: Array,
        innerloop_quad_weights: Array,
        design_weights: Array,
    ) -> float:
        """
        Approximate the utility of a design.

        Parameters
        ----------
        innerloop_loglike : IndependentGaussianOEDInnerLoopLogLikelihood
            Inner loop log-likelihood object.
        outerloop_samples : Array (nvars, nouterloop_samples)
            Samples for the outer loop quadrature.
        outerloop_quad_weights : Array (nouterloop_samples, 1)
            Weights for the outer loop quadrature.
        innerloop_samples : Array (nvars, ninnerloop_samples)
            Samples for the inner loop quadrature.
        innerloop_quad_weights : Array (ninnerloop_samples, 1)
            Weights for the inner loop quadrature.
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        utility: float
            The utility
        """
        raise NotImplementedError

    @abstractmethod
    def exact_utility(self, design_weights: Array) -> float:
        """
        Compute the exact utility.

        Parameters
        ----------
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        utility: float
        """
        raise NotImplementedError

    def compute_effective_sample_size(
        self,
        outerloop_quadtype: str,
        nouterloop_samples: int,
        innerloop_quadtype: str,
        ninnerloop_samples: int,
        design_weights: Array,
    ) -> float:
        """
        Compute the effective sample size (ESS) of the innerloop samples
        used to compute the evidence.

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
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        ess: float
            Effective sample size.
        """
        kl_oed = self.setup_oed()
        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = self._data_gen.prepare_simulation_inputs(
            kl_oed,
            self._problem.get_prior(),
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )
        kl_oed.set_data_from_model(
            self._problem.get_observation_model(),
            self._problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )
        kl_oed.objective()._set_expanded_design_weights(design_weights)
        return kl_oed.objective()._log_evidence.effective_sample_size(
            design_weights
        )

    def plot_ess_histogram(
        self, ax: matplotlib.axes.Axes, ess: Array, ninnerloop_samples: int
    ) -> matplotlib.container.BarContainer:
        """
        Plot a histogram of effective sample sizes (ESS).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axis object for plotting.
        ess : Array (nouterloop_samples,)
            Vector of effective sample sizes.
        ninnerloop_samples : int
            Maximum number of inner loop samples (used as x-axis limit).

        Returns
        -------
        im = matplotlib.container.BarContainer
            Histogram plot object.
        """
        # Plot histogram
        im = ax.hist(
            ess, bins=30, color="skyblue", edgecolor="black", alpha=0.7
        )

        # Customize the plot
        ax.set_xlabel("Effective Sample Size (ESS)")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Effective Sample Sizes")
        ax.set_xlim(
            0, ninnerloop_samples
        )  # Set x-axis limit to maximum inner loop samples
        ax.grid(True)
        return im

    def compute_mse(
        self,
        outerloop_quadtype: str,
        nouterloop_samples: int,
        innerloop_quadtype: str,
        ninnerloop_samples: int,
        nrealizations: int,
        design_weights: Array,
    ) -> float:
        """
        Compute the mean squared error (MSE) of the OED objective.

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
        nrealizations : int
            Number of realizations for computing MSE.
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        bias : float
            Bias of the expected information gain.
        variance : float
            Variance of the expected information gain.
        mse: float
            Mean squared error of the OED objective.
        """
        exact_utility = self.exact_utility(design_weights)
        if (
            outerloop_quadtype == "gauss" or innerloop_quadtype == "gauss"
        ) and nrealizations != 1:
            raise ValueError(
                "The variance of the estimator cannot be computed when using "
                "Gauss quadratureso must set nrealization to 1"
            )

        utility_values = []
        # start incrementing halton sequence start index from 1
        # everytime new number of ninner and nouter samples are given
        # not really necessary but ensures that the samples
        # used for one study are a subset of the ones used for the next.
        if hasattr(self._data_gen, "_inner_halton_seq"):
            delattr(self._data_gen, "_inner_halton_seq")
        if hasattr(self._data_gen, "_outer_halton_seq"):
            delattr(self._data_gen, "_outer_halton_seq")
        for realization in range(nrealizations):
            oed = self.setup_oed()
            (
                outerloop_samples,
                outerloop_quad_weights,
                innerloop_samples,
                innerloop_quad_weights,
            ) = self._data_gen.prepare_simulation_inputs(
                oed,
                self._problem.get_prior(),
                outerloop_quadtype,
                nouterloop_samples,
                innerloop_quadtype,
                ninnerloop_samples,
            )

            # Compute expected information gain and Laplace approximation
            utility = self.compute_utility(
                oed._innerloop_loglike,
                outerloop_samples,
                outerloop_quad_weights,
                innerloop_samples,
                innerloop_quad_weights,
                design_weights,
            )
            utility_values.append(utility)

        # Convert to array for easier computation
        utility_values = self._bkd.asarray(utility_values)

        # Compute bias, variance, and MSE
        bias = self._bkd.mean(utility_values) - exact_utility
        variance = self._bkd.var(utility_values)
        mse = bias**2 + variance
        return bias, variance, mse

    def compute_mse_for_sample_combinations(
        self,
        outerloop_sample_counts: List[int],
        innerloop_sample_counts: List[int],
        nrealizations: int,
        design_weights: Array,
        outerloop_quadtype: str = "MC",
        innerloop_quadtype: str = "MC",
    ) -> Dict[str, List[Array]]:
        """
        Plot the MSE for different combinations of outer loop and inner loop
        samples.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Matplotlib axis object for plotting.
        outerloop_sample_counts : list
            List of outer loop sample counts to test.
        innerloop_sample_counts : list
            List of inner loop sample counts to test.
        nrealizations : int
            Number of realizations for computing MSE.
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        values: Dict[str, List[Array]]
            Dictionary containing lists of arrays of values for each axis:
            - "sqbias": List[Array] squared-bias values for each
              ninnerloop_samples.
            - "variance":  List[Array] variance values for each
              ninnerloop_samples.
            - "mse":  List[Array] mse values for each
              ninnerloop_samples.
        """
        # Store lines for each axis
        values = {"sqbias": [], "variance": [], "mse": []}
        for ninnerloop_samples in innerloop_sample_counts:
            bias_values = []
            variance_values = []
            mse_values = []
            for nouterloop_samples in outerloop_sample_counts:
                bias, variance, mse = self.compute_mse(
                    outerloop_quadtype=outerloop_quadtype,
                    nouterloop_samples=nouterloop_samples,
                    innerloop_quadtype=innerloop_quadtype,
                    ninnerloop_samples=ninnerloop_samples,
                    nrealizations=nrealizations,
                    design_weights=design_weights,
                )
                bias_values.append(bias)
                variance_values.append(variance)
                mse_values.append(mse)
                print(
                    "MSE for ({0},{1}): {2}".format(
                        nouterloop_samples, ninnerloop_samples, mse
                    )
                )

            # Store bias, variance, mse values
            values["sqbias"].append(self._bkd.hstack(bias_values) ** 2)
            values["variance"].append(self._bkd.hstack(variance_values))
            values["mse"].append(self._bkd.hstack(mse_values))
        return values

    def plot_mse_vs_outerloop_samples(
        self,
        axes: matplotlib.axes.Axes,
        outerloop_sample_counts: List[int],
        innerloop_sample_counts: List[int],
        values: Dict[str, List[Array]],
    ) -> List[matplotlib.lines.Line2D]:
        """
        Plot the MSE vs the number of outer loop samples for different numbers
        of inner loop samples.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Matplotlib axis object for plotting.
        outerloop_sample_counts : list
            List of outer loop sample counts to test.
        innerloop_sample_counts : list
            List of inner loop sample counts to test.
        values: Dict[str, List[Array]]
            Dictionary containing lists of arrays of values for each axis:
            - "sqbias": List[Array] squared-bias values for each
              ninnerloop_samples.
            - "variance":  List[Array] variance values for each
              ninnerloop_samples.
            - "mse":  List[Array] mse values for each
              ninnerloop_samples.

        Returns
        -------
        lines : Dict[str, List[matplotlib.lines.Line2D]]
            Dictionary containing lists of line objects for each axis:
            - "sqbias": Lines for the squared-bias plot.
            - "variance": Lines for the variance plot.
            - "mse": Lines for the MSE plot.
        """

        lines = {"sqbias": [], "variance": [], "mse": []}
        for ii, ninnerloop_samples in enumerate(innerloop_sample_counts):

            # Plot squared-bias
            bias_line = axes[0].loglog(
                outerloop_sample_counts,
                values["sqbias"][ii],
                label=f"Inner Loop Samples: {ninnerloop_samples}",
                marker="o",
            )
            lines["sqbias"].extend(bias_line)

            # Plot variance
            variance_line = axes[1].loglog(
                outerloop_sample_counts,
                values["variance"][ii],
                label=f"Inner Loop Samples: {ninnerloop_samples}",
                marker="o",
            )
            lines["variance"].extend(variance_line)

            # Plot MSE
            mse_line = axes[2].loglog(
                outerloop_sample_counts,
                values["mse"][ii],
                label=f"Inner Loop Samples: {ninnerloop_samples}",
                marker="o",
            )
            lines["mse"].extend(mse_line)

        # Customize bias plot
        axes[0].set_xlabel("Outer Loop Samples")
        axes[0].set_ylabel("Squared-Bias")
        axes[0].set_title("Squared-Bias of Expected Information Gain")
        axes[0].legend()
        axes[0].grid(True)

        # Customize variance plot
        axes[1].set_xlabel("Outer Loop Samples")
        axes[1].set_ylabel("Variance")
        axes[1].set_title("Variance of Expected Information Gain")
        axes[1].legend()
        axes[1].grid(True)

        # Customize MSE plot
        axes[2].set_xlabel("Outer Loop Samples")
        axes[2].set_ylabel("Mean Squared Error (MSE)")
        axes[2].set_title("MSE of Expected Information Gain")
        axes[2].legend()
        axes[2].grid(True)
        return lines

    def plot_mse_vs_innerloop_samples(
        self,
        axes: matplotlib.axes.Axes,
        outerloop_sample_counts: List[int],
        innerloop_sample_counts: List[int],
        values: Dict[str, List[Array]],
    ) -> List[matplotlib.lines.Line2D]:
        """
        Plot the MSE vs the number of inner loop samples for different numbers
        of outer loop samples.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Matplotlib axis object for plotting.
        outerloop_sample_counts : list
            List of outer loop sample counts to test.
        innerloop_sample_counts : list
            List of inner loop sample counts to test.
        values: Dict[str, List[Array]]
            Dictionary containing lists of arrays of values for each axis:
            - "sqbias": List[Array] squared-bias values for each
              ninnerloop_samples.
            - "variance":  List[Array] variance values for each
              ninnerloop_samples.
            - "mse":  List[Array] mse values for each
              ninnerloop_samples.

        Returns
        -------
        lines : Dict[str, List[matplotlib.lines.Line2D]]
            Dictionary containing lists of line objects for each axis:
            - "sqbias": Lines for the squared-bias plot.
            - "variance": Lines for the variance plot.
            - "mse": Lines for the MSE plot.
        """
        lines = {"sqbias": [], "variance": [], "mse": []}
        sqbias = self._bkd.vstack(values["sqbias"])
        variance = self._bkd.vstack(values["variance"])
        mse = self._bkd.vstack(values["mse"])
        for ii, nouterloop_samples in enumerate(outerloop_sample_counts):

            # Plot squared-bias
            bias_line = axes[0].loglog(
                innerloop_sample_counts,
                sqbias[:, ii],
                label=f"Outer Loop Samples: {nouterloop_samples}",
                marker="o",
            )
            lines["sqbias"].extend(bias_line)

            # Plot variance
            variance_line = axes[1].loglog(
                innerloop_sample_counts,
                variance[:, ii],
                label=f"Outer Loop Samples: {nouterloop_samples}",
                marker="o",
            )
            lines["variance"].extend(variance_line)

            # Plot MSE
            mse_line = axes[2].loglog(
                innerloop_sample_counts,
                mse[:, ii],
                label=f"Outer Loop Samples: {nouterloop_samples}",
                marker="o",
            )
            lines["mse"].extend(mse_line)

        # Customize bias plot
        axes[0].set_xlabel("Inner Loop Samples")
        axes[0].set_ylabel("Squared-Bias")
        axes[0].set_title("Squared-Bias of Expected Information Gain")
        axes[0].legend()
        axes[0].grid(True)

        # Customize variance plot
        axes[1].set_xlabel("Inner Loop Samples")
        axes[1].set_ylabel("Variance")
        axes[1].set_title("Variance of Expected Information Gain")
        axes[1].legend()
        axes[1].grid(True)

        # Customize MSE plot
        axes[2].set_xlabel("Inner Loop Samples")
        axes[2].set_ylabel("Mean Squared Error (MSE)")
        axes[2].set_title("MSE of Expected Information Gain")
        axes[2].legend()
        axes[2].grid(True)
        return lines

    def compute_convergence_rate(
        self, sample_counts: List[int], values: List[float]
    ) -> float:
        """
        Compute the rate of convergence for a given set of sample counts and
        values.

        Parameters
        ----------
        sample_counts : List[int]
            List of sample counts (e.g., outer loop sample counts).
        values : List[float]
            List of values corresponding to the sample counts (e.g., MSE values).

        Returns
        -------
        float
            Rate of convergence computed as the slope of the log-log plot.
        """
        log_sample_counts = np.log(np.array(sample_counts))
        log_values = np.log(values)
        # Linear fit on log-log scale
        slope = np.polyfit(log_sample_counts, log_values, 1)[0]
        return -slope  # Negative slope represents the rate of convergence

    def add_convergence_rate_to_mse_plot(
        self,
        ax: matplotlib.axes.Axes,
        sample_counts: List[int],
        values: List[float],
        shift: float = 1.0,
        label: str = "",
    ):
        convergence_rate = self.compute_convergence_rate(sample_counts, values)
        starting_value = values[0] + shift
        line = ax.loglog(
            sample_counts,
            self._bkd.hstack(
                [
                    starting_value
                    * (count / sample_counts[0]) ** (-convergence_rate)
                    for count in sample_counts
                ]
            ),
            label=f"{label} Convergence Rate: {convergence_rate:.2f}",
            ls="--",
        )
        ax.legend()
        return line


class BayesianKLOEDDiagnostics(BayesianOEDDiagnostics):
    def _setup_OED(
        self, innerloop_loglike: IndependentGaussianOEDInnerLoopLogLikelihood
    ) -> BayesianOED:
        return KLBayesianOED(innerloop_loglike)

    def compute_utility(
        self,
        innerloop_loglike: IndependentGaussianOEDInnerLoopLogLikelihood,
        outerloop_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_samples: Array,
        innerloop_quad_weights: Array,
        design_weights: Array,
    ) -> float:
        """
        Compute the expected information gain using KL divergence.

        Parameters
        ----------
        innerloop_loglike : IndependentGaussianOEDInnerLoopLogLikelihood
            Inner loop log-likelihood object.
        outerloop_samples : Array (nvars, nouterloop_samples)
            Samples for the outer loop quadrature.
        outerloop_quad_weights : Array (nouterloop_samples, 1)
            Weights for the outer loop quadrature.
        innerloop_samples : Array (nvars, ninnerloop_samples)
            Samples for the inner loop quadrature.
        innerloop_quad_weights : Array (ninnerloop_samples, 1)
            Weights for the inner loop quadrature.
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        eig: float
            Expected information gain.
        """
        kl_oed = self.setup_oed()
        kl_oed.set_data_from_model(
            self._problem.get_observation_model(),
            self._problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )
        return -kl_oed.objective()(design_weights)

    def exact_utility(self, design_weights: Array) -> float:
        """
        Compute the exact expected information gain using KL divergence.

        Parameters
        ----------
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        eig: float
            Exact expected information gain.
        """
        return self._problem.exact_expected_information_gain(design_weights)


class ConjugateGaussianPriorOEDForLinearPredictionUtility(ABC):
    """
    Compute the expected divergence or deviation of the pushforward of the
    posterior, arising from a conugate Gaussian prior, through a
    linear prediction model of a scalar quantity of interest (QoI).
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


class ConjugateGaussianPriorOEDForLinearPredictionEntropicDeviation(
    ConjugateGaussianPriorOEDForLinearPredictionUtility
):
    def __init__(
        self, prior: MultivariateGaussian, qoi_mat: Array, lamda: float = 1.0
    ):
        self._lamda = lamda
        super().__init__(prior, qoi_mat)

    def _compute_utility(self) -> float:
        return self._lamda * self._post_pushforward.covariance()[0, 0] / 2.0


class ConjugateGaussianPriorOEDForLinearPredictionAVaRDeviation(
    ConjugateGaussianPriorOEDForLinearPredictionUtility
):
    def __init__(
        self, prior: MultivariateGaussian, qoi_mat: Array, beta: float = 0.5
    ):
        super().__init__(prior, qoi_mat)
        self._beta = beta
        self._std_normal = GaussianMarginal(0, 1, backend=self._bkd)

    def _compute_utility(self) -> float:
        return (
            self._bkd.sqrt(self._post_pushforward.covariance()[0, 0])
            * self._std_normal.pdf(
                self._std_normal.ppf(self._bkd.array([self._beta]))
            )
            / (1.0 - self._beta)
        )


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


class BayesianOEDForPredictionDiagnostics(BayesianOEDDiagnostics):
    def __init__(
        self,
        problem: LinearGaussianBayesianOEDBenchmark,
        utility_cls: ConjugateGaussianPriorOEDForLinearPredictionUtility,
        deviation_measure: PredictionOEDDeviationMeasure,
        risk_measure: SampleAverageStat,
        noise_stat: NoiseStatistic,
    ):
        self._utility_cls = utility_cls
        self._deviation_measure = deviation_measure
        self._risk_measure = risk_measure
        self._noise_stat = noise_stat
        super().__init__(problem)

    def _setup_OED(
        self, innerloop_loglike: IndependentGaussianOEDInnerLoopLogLikelihood
    ) -> BayesianOED:
        return BayesianOEDForPrediction(innerloop_loglike)

    def exact_utility(self, design_weights: Array) -> float:
        """
        Compute the exact utility.

        Parameters
        ----------
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        utility: float
            Exact utility of the design
        """
        prior = DenseCholeskyMultivariateGaussian(
            self._problem.get_prior().mean(),
            self._problem.get_prior().covariance(),
            backend=self._bkd,
        )
        utility = self._utility_cls(
            prior, self._problem.get_qoi_model().matrix()
        )
        utility.set_observation_matrix(
            self._problem.get_observation_model().matrix()
        )
        utility.set_noise_covariance(
            self._bkd.diag(
                self._problem.get_noise_covariance_diag()
                / design_weights[:, 0]
            )
        )
        return utility.value()

    def compute_utility(
        self,
        innerloop_loglike: IndependentGaussianOEDInnerLoopLogLikelihood,
        outerloop_samples: Array,
        outerloop_quad_weights: Array,
        innerloop_samples: Array,
        innerloop_quad_weights: Array,
        design_weights: Array,
    ) -> float:
        """
        Compute the expected information gain using KL divergence.

        Parameters
        ----------
        innerloop_loglike : IndependentGaussianOEDInnerLoopLogLikelihood
            Inner loop log-likelihood object.
        outerloop_samples : Array (nvars, nouterloop_samples)
            Samples for the outer loop quadrature.
        outerloop_quad_weights : Array (nouterloop_samples, 1)
            Weights for the outer loop quadrature.
        innerloop_samples : Array (nvars, ninnerloop_samples)
            Samples for the inner loop quadrature.
        innerloop_quad_weights : Array (ninnerloop_samples, 1)
            Weights for the inner loop quadrature.
        design_weights : Array (nobs, 1)
            Weights for the experimental design.

        Returns
        -------
        eig: float
            Expected information gain.
        """
        oed = self.setup_oed()
        oed.set_data_from_model(
            self._problem.get_observation_model(),
            self._problem.get_qoi_model(),
            self._problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
            self._problem.get_qoi_quad_weights(),
            self._deviation_measure,
            self._risk_measure,
            self._noise_stat,
        )
        return oed.objective()(design_weights)


class NonLinearGaussianBayesianOEDForPredictionBenchmark(
    LinearGaussianBayesianOEDForPredictionBenchmark
):
    def _setup_qoi_model(self) -> Model:
        """
        Set up the QoI model.

        Returns
        -------
        qoi_model: Model
            QoI model for the experimental design.
        """
        regression_model = Linear1DRegressionModel(
            self._qoi_design,
            self._degree,
            min_degree=self._min_degree,
            backend=self._bkd,
        )
        return ExponentialQoIModel(regression_model)
