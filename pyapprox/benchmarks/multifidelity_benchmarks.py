from typing import List, Tuple
import math

from scipy import stats

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.interface.model import ModelFromVectorizedCallable, Model
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.benchmarks.base import (
    ACVBenchmark,
    MultiIndexModelBenchmark,
    MultiIndexModelEnsemble,
)


class PolynomialModelEnsembleBenchmark(ACVBenchmark):
    r"""
    Polynomial model ensemble benchmark.

    This class implements an ensemble of 5 univariate polynomial models of the
    form:

    .. math::
        f_\alpha(\rv) = \rv^{5-\alpha}, \quad \alpha=0,\ldots,4

    where :math:`\rv \sim \mathcal{U}[0, 1]`.

    The benchmark provides the mean and covariance of the outputs of each model
    fidelity, as well as their costs.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    nmodels : int, optional
        Number of models in the ensemble. Default is 5.

    References
    ----------
    .. [GGEJJCP2020] `A generalized approximate control variate framework for
       multifidelity uncertainty quantification, Journal of Computational
       Physics, 408:109257, 2020.
       <https://doi.org/10.1016/j.jcp.2020.109257>`_
    """

    def __init__(self, backend: BackendMixin, nmodels: int = 5):
        """
        Initialize the polynomial model ensemble benchmark.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.
        nmodels : int, optional
            Number of models in the ensemble. Default is 5.
        """
        self._nmodels = nmodels
        super().__init__(backend)

    def nmodels(self) -> int:
        """
        Return the number of models in the ensemble.

        Returns
        -------
        nmodels : int
            Number of models in the ensemble.
        """
        return self._nmodels

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this benchmark, it is always 1.
        """
        return 1

    def costs(self) -> Array:
        """
        Return the cost of each model.

        Returns
        -------
        costs : Array
            Array of shape (nmodels,) containing the cost of each model.
        """
        return self._bkd.logspace(0, -self.nmodels(), self.nmodels())

    def m0(self, samples):
        """
        Evaluate the highest fidelity model (degree 5 polynomial).

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        m0 : Array
            Array of shape (nsamples,) containing the model evaluations.
        """
        return samples.T**5

    def m1(self, samples):
        """
        Evaluate the second highest fidelity model (degree 4 polynomial).

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        m1 : Array
            Array of shape (nsamples,) containing the model evaluations.
        """
        return samples.T**4

    def m2(self, samples):
        """
        Evaluate the third highest fidelity model (degree 3 polynomial).

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        m2 : Array
            Array of shape (nsamples,) containing the model evaluations.
        """
        return samples.T**3

    def m3(self, samples):
        """
        Evaluate the fourth highest fidelity model (degree 2 polynomial).

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        m3 : Array
            Array of shape (nsamples,) containing the model evaluations.
        """
        return samples.T**2

    def m4(self, samples):
        """
        Evaluate the lowest fidelity model (degree 1 polynomial).

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        m4 : Array
            Array of shape (nsamples,) containing the model evaluations.
        """
        return samples.T**1

    def _set_models(self):
        """
        Define the polynomial models included in the benchmark.
        """
        funs = [self.m0, self.m1, self.m2, self.m3, self.m4][: self.nmodels()]
        self._models = [
            ModelFromVectorizedCallable(
                self.nqoi(), self.nvars(), fun, backend=self._bkd
            )
            for fun in funs
        ]

    def _mean(self):
        """
        Compute the mean of the QoI for each model.

        Returns
        -------
        mean : Array (nmodels, nqoi)
            Array containing the mean of each model fidelity.
        """
        return 1 / self._bkd.arange(6, 1, -1)[: self.nmodels()].reshape(
            self.nmodels(), self.nqoi()
        )

    def _set_prior(self):
        """
        Define the prior distribution for the input variable.
        """
        marginals = [stats.uniform(0, 1)]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )


class TunableModelEnsembleBenchmark(ACVBenchmark):
    """
    Tunable model ensemble benchmark.

    This class implements an ensemble of three tunable models with varying fidelity levels.
    The models are parameterized by coefficients and angles, allowing control over their
    variance and covariance. The benchmark provides methods to compute the mean, covariance,
    kurtosis, and covariance of variances for the models.

    Parameters
    ----------
    theta1 : float
        Angle controlling the second model's fidelity.
    backend : BackendMixin
        Backend for numerical computations.
    shifts : List[float], optional
        Shifts applied to the second and third models. Default is [0, 0].
    """

    def __init__(
        self,
        theta1: float,
        backend: BackendMixin,
        shifts: List[float] = None,
    ):
        """
        Initialize the tunable model ensemble benchmark.

        Parameters
        ----------
        theta1 : float
            Angle controlling the second model's fidelity.
        backend : BackendMixin
            Backend for numerical computations.
        shifts : List[float], optional
            Shifts applied to the second and third models. Default is [0, 0].

        Notes
        -----
        The choice of coefficients (`A0`, `A1`, `A2`) ensures unit variance for each model.
        """
        self._theta1 = backend.array(theta1)
        if shifts is None:
            shifts = [0.0, 0.0]
        self._shifts = shifts
        super().__init__(backend)

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this benchmark, it is always 1.
        """
        return 1

    def nmodels(self) -> int:
        """
        Return the number of models in the ensemble.

        Returns
        -------
        nmodels : int
            Number of models in the ensemble. For this benchmark, it is always 3.
        """
        return 3

    def m0(self, samples):
        """
        Evaluate the highest fidelity model.

        Parameters
        ----------
        samples : Array
            Array of shape (2, nsamples) containing the input samples.

        Returns
        -------
        m0 : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        assert samples.shape[0] == 2
        x, y = samples[0, :], samples[1, :]
        return (
            self.A0
            * (
                self._bkd.cos(self._theta0) * x**5
                + self._bkd.sin(self._theta0) * y**5
            )
        )[:, None]

    def m1(self, samples):
        """
        Evaluate the second highest fidelity model.

        Parameters
        ----------
        samples : Array
            Array of shape (2, nsamples) containing the input samples.

        Returns
        -------
        m1 : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        assert samples.shape[0] == 2
        x, y = samples[0, :], samples[1, :]
        return (
            self.A1
            * (
                self._bkd.cos(self._theta1) * x**3
                + self._bkd.sin(self._theta1) * y**3
            )
            + self._shifts[0]
        )[:, None]

    def m2(self, samples):
        """
        Evaluate the lowest fidelity model.

        Parameters
        ----------
        samples : Array
            Array of shape (2, nsamples) containing the input samples.

        Returns
        -------
        m2 : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        assert samples.shape[0] == 2
        x, y = samples[0, :], samples[1, :]
        return (
            self.A2
            * (
                self._bkd.cos(self._theta2) * x
                + self._bkd.sin(self._theta2) * y
            )
            + self._shifts[1]
        )[:, None]

    def _set_models(self):
        """
        Define the tunable models included in the benchmark.

        Returns
        -------
        None
        """
        self.A0 = math.sqrt(11)
        self.A1 = math.sqrt(7)
        self.A2 = math.sqrt(3)
        self._theta0 = self._bkd.array(math.pi / 2)
        self._theta2 = self._bkd.array(math.pi / 6)
        assert self._theta0 > self._theta1 and self._theta1 > self._theta2
        funs = [self.m0, self.m1, self.m2]
        self._models = [
            ModelFromVectorizedCallable(
                self.nqoi(), self.nvars(), fun, backend=self._bkd
            )
            for fun in funs
        ]

    def _set_prior(self):
        """
        Define the prior distribution for the input variables.

        Returns
        -------
        None
        """
        marginals = [stats.uniform(-1, 2), stats.uniform(-1, 2)]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _covariance(self):
        """
        Compute the covariance matrix between the models.

        Returns
        -------
        covariance : Array (nmodels, nmodels)
            Covariance matrix between the models.
        """
        cov = self._bkd.eye(self.nmodels())
        cov[0, 1] = (
            self.A0
            * self.A1
            / 9
            * (
                self._bkd.sin(self._theta0) * self._bkd.sin(self._theta1)
                + self._bkd.cos(self._theta0) * self._bkd.cos(self._theta1)
            )
        )
        cov[1, 0] = cov[0, 1]
        cov[0, 2] = (
            self.A0
            * self.A2
            / 7
            * (
                self._bkd.sin(self._theta0) * self._bkd.sin(self._theta2)
                + self._bkd.cos(self._theta0) * self._bkd.cos(self._theta2)
            )
        )
        cov[2, 0] = cov[0, 2]
        cov[1, 2] = (
            self.A1
            * self.A2
            / 5
            * (
                self._bkd.sin(self._theta1) * self._bkd.sin(self._theta2)
                + self._bkd.cos(self._theta1) * self._bkd.cos(self._theta2)
            )
        )
        cov[2, 1] = cov[1, 2]
        return cov

    def _mean(self):
        """
        Compute the mean of the QoI for each model.

        Returns
        -------
        mean : Array (nmodels, 1)
            Array containing the mean of each model fidelity.
        """
        return self._bkd.array([0.0, self._shifts[0], self._shifts[1]])[
            :, None
        ]

    def get_kurtosis(self):
        """
        Compute the kurtosis of the QoI for each model.

        Returns
        -------
        kurtosis : Array (nmodels,)
            Array containing the kurtosis of each model fidelity.
        """
        return self._bkd.array(
            [
                (self.A0**4 * (213.0 + 29.0 * self._bkd.cos(4 * self._theta0)))
                / 5082,
                (self.A1**4 * (93.0 + 5.0 * self._bkd.cos(4 * self._theta1)))
                / 1274.0,
                -(1.0 / 30.0)
                * self.A2**4
                * (-7.0 + self._bkd.cos(4.0 * self._theta2)),
            ]
        )

    def costs(self) -> Array:
        """
        Return the nominal costs of each model for a single sample.

        Returns
        -------
        costs : Array (nmodels,)
            Array containing the model costs.
        """
        return self._bkd.array([1.0, 0.01, 0.001])


class ShortColumnModelEnsembleBenchmark(ACVBenchmark):
    r"""
    Short column model ensemble benchmark.

    This class implements an ensemble of 5 models.

    The high-fidelity model is:

    .. math::   f^{(1)}(z) = 1 - \frac{4z_4}{z_1 z_2^2 z_3} - \left( \frac{z_5}{z_1 z_2 z_3} \right)^2,

    where :math:`z = [z_1, \dots, z_5]^T \in D \text{ with } D = [5, 15] \times [15, 25] \times \mathbb{R}^+ \times \mathbb{R} \times\mathbb{R}`


    The benchmark provides the mean and covariance of the outputs of each model
    fidelity, as well as their costs.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    nmodels : int, optional
        Number of models in the ensemble. Default is 5.

    References
    ----------
    .. [Peherstorfer2016] `Peherstorfer, B., Willcox, K., & Gunzburger, M. Optimal Model Management for Multifidelity Monte Carlo Estimation. SIAM Journal on Scientific Computing, 38(5), A3163-A3194, 2016. <https://doi.org/10.1137/15M1046472>`_

    """

    def __init__(self, backend: BackendMixin, nmodels: int = 5):
        """
        Initialize the short column model ensemble benchmark.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.
        nmodels : int, optional
            Number of models in the ensemble. Default is 5.
        """
        self._nmodels = nmodels
        super().__init__(backend)

    def nmodels(self) -> int:
        """
        Return the number of models in the ensemble.

        Returns
        -------
        nmodels : int
            Number of models in the ensemble.
        """
        return self._nmodels

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this benchmark, it is always 1.
        """
        return 1

    def costs(self) -> Array:
        """
        Return the cost of each model.

        Returns
        -------
        costs : Array
            Array of shape (nmodels,) containing the cost of each model.
        """
        return self._bkd.logspace(0, -self.nmodels(), self.nmodels())

    def _extract_variables(
        self, samples: Array
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Extract the input variables from the samples.

        Parameters
        ----------
        samples : Array
            Array of shape (5, nsamples) containing the input samples.

        Returns
        -------
        b : Array
            Width of the column.
        h : Array
            Height of the column.
        P : Array
            Axial force applied to the column.
        M : Array
            Bending moment applied to the column.
        Y : Array
            Yield stress of the column material. If `self._apply_lognormal` is True,
            the yield stress is transformed using the exponential function.
        """
        assert samples.shape[0] == 5
        b = samples[0, :]
        h = samples[1, :]
        P = samples[2, :]
        M = samples[3, :]
        Y = samples[4, :]
        self._apply_lognormal = False
        if self._apply_lognormal:
            Y = self._bkd.exp(Y)
        return b, h, P, M, Y

    def m0(self, samples: Array) -> Array:
        """
        Evaluate the highest fidelity model.

        Parameters
        ----------
        samples : Array
            Array of shape (5, nsamples) containing the input samples.

        Returns
        -------
        m0 : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - 4 * M / (b * (h**2) * Y) - (P / (b * h * Y)) ** 2)[:, None]

    def m1(self, samples: Array) -> Array:
        """
        Evaluate the second fidelity model.

        Parameters
        ----------
        samples : Array
            Array of shape (5, nsamples) containing the input samples.

        Returns
        -------
        m1 : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        b, h, P, M, Y = self._extract_variables(samples)
        return (
            1
            - 3.8 * M / (b * (h**2) * Y)
            - ((P * (1 + (M - 2000) / 4000)) / (b * h * Y)) ** 2
        )[:, None]

    def m2(self, samples: Array) -> Array:
        """
        Evaluate the third fidelity model.

        Parameters
        ----------
        samples : Array
            Array of shape (5, nsamples) containing the input samples.

        Returns
        -------
        m2 : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - M / (b * (h**2) * Y) - (P / (b * h * Y)) ** 2)[:, None]

    def m3(self, samples: Array) -> Array:
        """
        Evaluate the fourth fidelity model.

        Parameters
        ----------
        samples : Array
            Array of shape (5, nsamples) containing the input samples.

        Returns
        -------
        m3 : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - M / (b * (h**2) * Y) - (P * (1 + M) / (b * h * Y)) ** 2)[
            :, None
        ]

    def m4(self, samples: Array) -> Array:
        """
        Evaluate the lowest fidelity model.

        Parameters
        ----------
        samples : Array
            Array of shape (5, nsamples) containing the input samples.

        Returns
        -------
        m4 : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        b, h, P, M, Y = self._extract_variables(samples)
        return (1 - M / (b * (h**2) * Y) - (P * (1 + M) / (h * Y)) ** 2)[
            :, None
        ]

    def _set_models(self):
        """
        Define the polynomial models included in the benchmark.
        """
        funs = [self.m0, self.m1, self.m2, self.m3, self.m4][: self.nmodels()]
        self._models = [
            ModelFromVectorizedCallable(
                self.nqoi(), self.nvars(), fun, backend=self._bkd
            )
            for fun in funs
        ]

    def _set_prior(self):
        """
        Define the prior distribution for the input variable.
        """
        marginals = [
            stats.uniform(5, 10),
            stats.uniform(15, 10),
            stats.norm(500, 100),
            stats.norm(2000, 400),
            stats.lognorm(s=0.5, scale=math.exp(5)),
        ]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_quadrature_samples_weights(self):
        """
        Set the quadrature samples and weights for integration.
        """
        nsamples = int(1e7)
        self._quadx = self.prior().rvs(nsamples)
        self._quadw = self._bkd.full((nsamples,), 1.0 / nsamples)


class MultiOutputModelEnsembleBenchmark(ACVBenchmark):
    """
    Multi-output model ensemble benchmark.

    This class implements a benchmark for testing multifidelity algorithms that
    estimate statistics for vector-valued models of varying fidelity. Each model
    outputs multiple quantities of interest (QoI), and the benchmark provides
    methods to compute the mean, covariance, and costs of the models.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.

    References
    ----------
    .. [DWBG2024] `T. Dixon et al. Covariance Expressions for Multi-Fidelity Sampling with Multi-Output, Multi-Statistic Estimators: Application to Approximate Control Variates. SIAM/ASA Journal on Uncertainty Quantification, 12(3):1005-1049, 2024 <https://doi.org/10.1137/23M1607994>`_
    """

    def nmodels(self) -> int:
        """
        Return the number of models in the ensemble.

        Returns
        -------
        nmodels : int
            Number of models in the ensemble. For this benchmark, it is always 3.
        """
        return 3

    def f0(self, samples: Array) -> Array:
        """
        Evaluate the highest fidelity model.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations.

        Returns
        -------
        values : Array (nsamples, nqoi)
            Model evaluations at the samples.
        """
        return self._bkd.hstack(
            [
                math.sqrt(11) * samples.T**5,
                samples.T**4,
                self._bkd.sin(2 * math.pi * samples.T),
            ]
        )

    def f1(self, samples: Array) -> Array:
        """
        Evaluate the second fidelity model.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations.

        Returns
        -------
        values : Array (nsamples, nqoi)
            Model evaluations at the samples.
        """
        return self._bkd.hstack(
            [
                math.sqrt(7) * samples.T**3,
                math.sqrt(7) * samples.T**2,
                self._bkd.cos(2 * math.pi * samples.T + math.pi / 2),
            ]
        )

    def f2(self, samples: Array) -> Array:
        """
        Evaluate the lowest fidelity model.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations.

        Returns
        -------
        values : Array (nsamples, nqoi)
            Model evaluations at the samples.
        """
        return self._bkd.hstack(
            [
                math.sqrt(3) / 2 * samples.T**2,
                math.sqrt(3) / 2 * samples.T,
                self._bkd.cos(2 * math.pi * samples.T + math.pi / 4),
            ]
        )

    def _set_prior(self):
        """
        Define the prior distribution for the input variable.
        """
        self._prior = IndependentMarginalsVariable(
            [stats.uniform(0, 1)], backend=self._bkd
        )

    def _set_models(self):
        """
        Define the models included in the benchmark.
        """
        self._models = [
            ModelFromVectorizedCallable(
                self.nqoi(), self.nvars(), fun, backend=self._bkd
            )
            for fun in [self.f0, self.f1, self.f2]
        ]

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this benchmark, it is always 3.
        """
        return 3

    def costs(self) -> Array:
        """
        Return the nominal costs of each model for a single sample.

        Returns
        -------
        costs : Array (nmodels,)
            Array containing the model costs.
        """
        return self._bkd.array([1.0, 0.01, 0.001])

    def _uniform_means(self):
        """
        Compute the uniform means of the QoI for each model.

        Returns
        -------
        uniform_means : Array (nmodels, nqoi)
            Array containing the uniform means of the QoI for each model.
        """
        return self._bkd.array(
            [
                [math.sqrt(11) / 6, 1 / 5, 0.0],
                [math.sqrt(7) / 4, math.sqrt(7) / 3, 0.0],
                [1 / (2 * math.sqrt(3)), math.sqrt(3) / 4, 0.0],
            ]
        )

    def _uniform_covariance_matrices(self):
        """
        Compute the uniform covariance matrices for the models.

        Returns
        -------
        cov11, cov22, cov33, cov12, cov13, cov23 : Tuple[Array]
            Covariance matrices for the models.
        """
        # Compute diagonal blocks
        c13 = (
            -math.sqrt(11)
            * (15 - 10 * math.pi**2 + 2 * math.pi**4)
            / (4 * math.pi**5)
        )
        c23 = (3 - math.pi**2) / (2 * math.pi**3)
        cov11 = self._bkd.array(
            [
                [25 / 36, math.sqrt(11) / 15.0, c13],
                [math.sqrt(11) / 15.0, 16 / 225, c23],
                [c13, c23, 1 / 2],
            ]
        )
        c13 = math.sqrt(7) * (-3 + 2 * math.pi**2) / (4 * math.pi**3)
        c23 = math.sqrt(7) / (2 * math.pi)
        cov22 = self._bkd.array(
            [[9 / 16, 7 / 12, c13], [7 / 12, 28 / 45, c23], [c13, c23, 1 / 2]]
        )
        c13 = math.sqrt(3 / 2) * (1 + math.pi) / (4 * math.pi**2)
        c23 = math.sqrt(3 / 2) / (4 * math.pi)
        cov33 = self._bkd.array(
            [[1 / 15, 1 / 16, c13], [1 / 16, 1 / 16, c23], [c13, c23, 1 / 2]]
        )
        # Compute off-diagonal block covariance between models
        c13 = (
            math.sqrt(11)
            * (15 - 10 * math.pi**2 + 2 * math.pi**4)
            / (4 * math.pi**5)
        )
        c31 = math.sqrt(7) * (3 - 2 * math.pi**2) / (4 * math.pi**3)
        cov12 = self._bkd.array(
            [
                [5 * math.sqrt(77) / 72, 5 * math.sqrt(77) / 72, c13],
                [
                    3 * math.sqrt(7) / 40,
                    8 / (15 * math.sqrt(7)),
                    (-3 + math.pi**2) / (2 * math.pi**3),
                ],
                [c31, -math.sqrt(7) / (2 * math.pi), -1 / 2],
            ]
        )
        c13 = (
            math.sqrt(11 / 2)
            * (
                15
                + math.pi
                * (-15 + math.pi * (-10 + math.pi * (5 + 2 * math.pi)))
            )
            / (4 * math.pi**5)
        )
        c23 = (-3 + (-1 + math.pi) * math.pi * (3 + math.pi)) / (
            2 * math.sqrt(2) * math.pi**4
        )
        cov13 = self._bkd.array(
            [
                [5 * math.sqrt(11 / 3) / 48, 5 * math.sqrt(11 / 3) / 56, c13],
                [4 / (35 * math.sqrt(3)), 1 / (10 * math.sqrt(3)), c23],
                [
                    -math.sqrt(3) / (4 * math.pi),
                    -math.sqrt(3) / (4 * math.pi),
                    -1 / (2 * math.sqrt(2)),
                ],
            ]
        )
        c13 = (
            math.sqrt(7 / 2)
            * (-3 + 3 * math.pi + 2 * math.pi**2)
            / (4 * math.pi**3)
        )
        c23 = math.sqrt(7 / 2) * (1 + math.pi) / (2 * math.pi**2)
        cov23 = self._bkd.array(
            [
                [math.sqrt(7 / 3) / 8, 3 * math.sqrt(21) / 80, c13],
                [2 * math.sqrt(7 / 3) / 15, math.sqrt(7 / 3) / 8, c23],
                [
                    math.sqrt(3) / (4 * math.pi),
                    math.sqrt(3) / (4 * math.pi),
                    1 / (2 * math.sqrt(2)),
                ],
            ]
        )
        return cov11, cov22, cov33, cov12, cov13, cov23


class PSDMultiOutputModelEnsembleBenchmark(ACVBenchmark):
    """
    Positive Semi-Definite Multi-output model ensemble benchmark which is a
    modification of the MultiOutputModelEnsembleBenchmark.


    This class implements a

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def nmodels(self) -> int:
        """
        Return the number of models in the ensemble.

        Returns
        -------
        nmodels : int
            Number of models in the ensemble. For this benchmark, it is always 3.
        """
        return 3

    def f0(self, samples: Array) -> Array:
        """
        Evaluate the highest fidelity model.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations.

        Returns
        -------
        values : Array (nsamples, nqoi)
            Model evaluations at the samples.
        """
        eps = 1  # 1e-2
        return self._bkd.hstack(
            [
                math.sqrt(11) * samples.T**5,
                samples.T**4 + eps * self._bkd.cos(2.2 * math.pi * samples.T),
                self._bkd.sin(2 * math.pi * samples.T),
            ]
        )

    def f1(self, samples: Array) -> Array:
        """
        Evaluate the second fidelity model.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations.

        Returns
        -------
        values : Array (nsamples, nqoi)
            Model evaluations at the samples.
        """
        eps = 1e-1  # in [0, 1]
        # Making closer to 0 makes mean estimation with all three QoI ill-conditioned
        return self._bkd.hstack(
            [
                math.sqrt(7) * samples.T**3,
                math.sqrt(7) * samples.T**2,
                self._bkd.cos((2 + eps) * math.pi * samples.T + math.pi / 2),
            ]
        )

    def f2(self, samples: Array) -> Array:
        """
        Evaluate the lowest fidelity model.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations.

        Returns
        -------
        values : Array (nsamples, nqoi)
            Model evaluations at the samples.
        """
        # Making eps smaller will make variance estimation with QoI 0 and w more ill-conditioned
        eps = 1e-2
        return self._bkd.hstack(
            [
                math.sqrt(3) / 2 * samples.T**2 + samples.T,
                math.sqrt(3) / 2 * samples.T
                + self._bkd.cos(math.pi * samples.T * 2.0 + 2.1) * eps,
                self._bkd.cos(2 * math.pi * samples.T + math.pi / 4),
            ]
        )

    def _set_prior(self):
        """
        Define the prior distribution for the input variable.

        Returns
        -------
        None
        """
        self._prior = IndependentMarginalsVariable(
            [stats.uniform(0, 1)], backend=self._bkd
        )

    def _set_models(self):
        """
        Define the models included in the benchmark.

        Returns
        -------
        None
        """
        self._models = [
            ModelFromVectorizedCallable(
                self.nqoi(), self._prior.nvars(), fun, backend=self._bkd
            )
            for fun in [self.f0, self.f1, self.f2]
        ]

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this benchmark, it is always 3.
        """
        return 3

    def costs(self) -> Array:
        """
        Return the nominal costs of each model for a single sample.

        Returns
        -------
        costs : Array (nmodels,)
            Array containing the model costs.
        """
        return self._bkd.array([1.0, 0.01, 0.001])


class MultiIndexCosineModel(Model):
    r"""
    Multi-index cosine model.

    This class implements a model that evaluates the sum of cosine functions
    with phase shifts applied to the input samples. The model is parameterized
    by a shifts :math:`\delta_i`, which define the phase of each cosine
    function.

    .. math::

        f(\rv) = \cos\left(\pi \frac{\rv + 1}{2} + \delta\right)

    Parameters
    ----------
    shifts : Array
        Array containing the phase shifts :math:`\delta_i` for the cosine
        functions.
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, shifts: Array, backend: BackendMixin):
        """
        Initialize the multi-index cosine model.

        Parameters
        ----------
        shifts : Array
            Array containing the phase shifts for the cosine functions.
        backend : BackendMixin
            Backend for numerical computations.
        """
        self._shifts = shifts
        self._nrefinement_vars = len(shifts)
        super().__init__(backend)

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the model for given samples.

        The model computes the sum of cosine functions with phase shifts applied
        to the input samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        _values : Array
            Array of shape (nsamples, 1) containing the model evaluations.
        """
        vals = 0
        for shift in self._shifts:
            vals += self._bkd.cos(math.pi * (samples[0] + 1) / 2 + shift)[
                :, None
            ]
        return vals

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this model, it is always 1.
        """
        return 1

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars : int
            Number of uncertain variables. For this model, it is always 1.
        """
        return 1

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns
        -------
        repr : str
            String representation of the class, including the phase shifts.
        """
        return "{0}(shifts={1})".format(self.__class__.__name__, self._shifts)


class MultiLevelCosineModelEnsemble(MultiIndexModelEnsemble):
    r"""
    Multi-level cosine model ensemble.

    This class implements an ensemble of cosine models organized in a multi-level
    hierarchy. Each model is parameterized by phase shifts, which are determined
    based on the model's index in the hierarchy.

    The ith model is given by:

    .. math::

        f_i(\rv) = \sum_{i=1}^N \cos\left(\pi \frac{\rv + 1}{2} + \delta_i\right)

    where :math:`\rv` is the input variable and :math:`\delta_i` are the phase shifts.

    Parameters
    ----------
    index_bounds : List[int]
        Bounds for the indices defining the hierarchy levels.
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(
        self,
        index_bounds: List[int],
        backend: BackendMixin,
    ):
        """
        Initialize the multi-level cosine model ensemble.

        Parameters
        ----------
        index_bounds : List[int]
            Bounds for the indices defining the hierarchy levels.
        backend : BackendMixin
            Backend for numerical computations.
        """
        super().__init__(index_bounds, backend)

    def _model_id_to_shifts(self, model_id: Array) -> Array:
        """
        Map a model ID to its corresponding phase shifts.

        Parameters
        ----------
        model_id : Array
            Array containing the model ID.

        Returns
        -------
        shifts : Array
            Array containing the phase shifts for the model.
        """
        possible_shifts = self._bkd.array([0.25, 0.125, 0.0])
        shifts = possible_shifts[model_id]
        return shifts

    def setup_model(self, model_id: Array) -> Model:
        """
        Create a cosine model for the given model ID.

        Parameters
        ----------
        model_id : Array
            Array containing the model ID.

        Returns
        -------
        model : Model
            The cosine model corresponding to the given model ID.
        """
        return MultiIndexCosineModel(
            self._model_id_to_shifts(model_id), self._bkd
        )


class MultiLevelCosineBenchmark(MultiIndexModelBenchmark):
    r"""
    Multi-level cosine benchmark.

    This class implements a benchmark for analyzing models organized in a multi-level
    hierarchy, where each model is parameterized by phase shifts and evaluated using
    cosine functions. The benchmark uses a single uncertain variable with a uniform
    prior distribution.

    The ith model is given by:

    .. math::

        f_i(\rv) = \sum_{i=1}^N \cos\left(\pi \frac{\rv + 1}{2} + \delta_i\right)

    where :math:`\rv` is the input variable and :math:`\delta_i` are the phase shifts.

    The prior distribution is defined as:

    .. math::
        z \sim \mathcal{U}[-1, 2]

    where :math:`z` is the input variable.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this benchmark, it is always 1.
        """
        return 1

    def _set_prior(self):
        """
        Define the prior distribution for the input variable.

        The prior distribution is uniform over the interval [-1, 2].

        Returns
        -------
        None
        """
        self._prior = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)], backend=self._bkd
        )

    def _set_models(self):
        """
        Define the models included in the benchmark.

        The models are organized in a multi-level hierarchy and evaluated using
        cosine functions.

        Returns
        -------
        None
        """
        self._models = MultiLevelCosineModelEnsemble([2], self._bkd)

    def _validate_models(self):
        """
        Validate that the models are set correctly.

        Raises
        ------
        ValueError
            If the models are not set or are invalid.
        """
        if not hasattr(self, "_models") or self._models is None:
            raise ValueError(
                "The models are not set. Ensure `_set_models` is implemented "
                "correctly."
            )
        if not isinstance(self._models, MultiIndexModelEnsemble):
            raise ValueError(
                "The models must be a MultiIndexModelEnsemble instance."
            )

    def nmodels(self) -> int:
        """
        Return the number of models in the hierarchy.

        Returns
        -------
        nmodels : int
            Number of models in the hierarchy.
            For this benchmark, it is always 3.
        """
        return 3

    def nrefinement_vars(self) -> int:
        """
        Return the number of refinement variables.

        Returns
        -------
        nrefinement_vars : int
            Number of refinement variables. For this benchmark, it is always 1.
        """
        return 1
