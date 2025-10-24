import math
from typing import List, Union

import numpy as np
from scipy import stats
from scipy.optimize import rosen, rosen_der, rosen_hess_prod, LinearConstraint

from pyapprox.interface.model import Model, SingleSampleModel
from pyapprox.interface.wrappers import (
    create_active_set_variable_model,
    ChangeModelSignWrapper,
)
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.variables.joint import (
    JointVariable,
    DesignVariable,
    IndependentMarginalsVariable,
)
from pyapprox.benchmarks.base import (
    SingleModelBenchmark,
    OptimizationBenchmark,
    ConstrainedOptimizationBenchmark,
    ConstrainedUncertainOptimizationBenchmark,
)
from pyapprox.optimization.minimize import (
    SampleAverageConstraint,
    SampleAverageMeanPlusStdev,
    ConstraintFromModel,
    Constraint,
)
from pyapprox.surrogates.affine.basis import (
    FixedGaussianTensorProductQuadratureRuleFromVariable,
)


class IshigamiModel(Model):
    r"""
    Ishigami function

    .. math:: f(z) = \sin(z_1)+a\sin^2(z_2) + bz_3^4\sin(z_0)
    """

    def __init__(
        self,
        backend: BackendMixin,
        a: float = 7,
        b: float = 0.1,
    ):
        self._a = a
        self._b = b
        super().__init__(backend)

    def jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return 1

    def _values(self, samples: Array) -> Array:
        return (
            self._bkd.sin(samples[0, :])
            + self._a * self._bkd.sin(samples[1, :]) ** 2
            + self._b * samples[2, :] ** 4 * self._bkd.sin(samples[0, :])
        )[:, None]

    def nvars(self) -> int:
        return 3

    def _jacobian(self, sample: Array) -> Array:
        assert sample.shape[0] == self.nvars()
        return self._bkd.stack(
            (
                self._bkd.cos(sample[0, :])
                + self._b * sample[2, :] ** 4 * self._bkd.cos(sample[0, :]),
                2
                * self._a
                * self._bkd.sin(sample[1, :])
                * self._bkd.cos(sample[1, :]),
                4 * self._b * sample[2, :] ** 3 * self._bkd.sin(sample[0, :]),
            ),
            axis=1,
        )

    def _hessian(self, sample: Array) -> Array:
        assert sample.shape[0] == self.nvars()
        hess = self._bkd.empty((self.nvars(), self.nvars()))
        hess[0, 0] = -self._bkd.sin(sample[0, 0]) - self._b * sample[
            2, 0
        ] ** 4 * self._bkd.sin(sample[0, 0])
        hess[1, 1] = (
            2
            * self._a
            * (
                self._bkd.cos(sample[1, 0]) ** 2
                - self._bkd.sin(sample[1, 0]) ** 2
            )
        )
        hess[2, 2] = (
            12 * self._b * sample[2, 0] ** 2 * self._bkd.sin(sample[0, 0])
        )
        hess[0, 1], hess[1, 0] = 0, 0
        hess[0, 2] = (
            4 * self._b * sample[2, 0] ** 3 * self._bkd.cos(sample[0, 0])
        )
        hess[2, 0] = hess[0, 2]
        hess[1, 2], hess[2, 1] = 0, 0
        return hess[None, ...]


class IshigamiBenchmark(SingleModelBenchmark):
    r"""
     Ishigami function benchmark.

     The Ishigami function is a well-known benchmark function for sensitivity
     analysis. It is defined as:

     .. math:: f(z) = \sin(z_1) + a \sin^2(z_2) + b z_3^4 \sin(z_0)

     The function exhibits strong nonlinearity and interaction effects, making
     it suitable for testing sensitivity analysis methods.

     Parameters
     ----------
     a :float
         Coefficient for the second term. Default is 7.
     b : float
         Coefficient for the third term. Default is 0.1.
    backend :BackendMixin
         Backend for numerical computations

     References
     ----------
     .. [Ishigami1990] T. Ishigami and T. Homma, "An importance quantification
        technique in uncertainty analysis for computer models," Proceedings.
        First International Symposium on Uncertainty Modeling and Analysis,
        College Park, MD, USA, 1990, pp. 398-403.
        https://doi.org/10.1109/ISUMA.1990.151285
    """

    def __init__(
        self,
        backend: BackendMixin,
        a: float = 7,
        b: float = 0.1,
    ):
        """
        Initialize the Ishigami benchmark.

        Parameters
        ----------
        a :float
            Coefficient for the second term. Default is 7.
        b : float
            Coefficient for the third term. Default is 0.1.
        backend :BackendMixin
            Backend for numerical computations
        """
        self._a = a
        self._b = b
        super().__init__(backend)

    def _set_model(self):
        """
        Set the Ishigami model.

        The model is initialized with the coefficients `a` and `b` and the
        specified backend.
        """
        self._model = IshigamiModel(self._bkd, self._a, self._b)

    def _set_prior(self) -> JointVariable:
        r"""
        Set the prior distribution.

        The prior distribution is defined as:

        .. math:: p_i(X_i) ~ U[-\pi, \pi], i = 1, \ldots, 3
        """
        marginals = [stats.uniform(-np.pi, 2 * np.pi)] * 3
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def mean(self) -> Array:
        """
        Compute the mean of the Ishigami function.

        Returns
        -------
        mean: Array
            The mean value of the Ishigami function.
        """
        return self._bkd.atleast1d(self._bkd.asarray(self._a / 2))

    def variance(self) -> Array:
        """
        Compute the variance of the Ishigami function.

        Returns
        -------
        variance: Array
            The variance of the Ishigami function.
        """
        return self._bkd.atleast1d(
            self._bkd.asarray(
                self._a**2 / 8
                + self._b * np.pi**4 / 5
                + self._b**2 * np.pi**8 / 18
                + 0.5
            )
        )

    def _unnormalized_sobol_indices(self):
        """
        Compute the unnormalized Sobol indices.

        Returns
        -------
        indices: Tuple
            Unnormalized Sobol indices for main and interaction effects.
        """
        zero = self._bkd.zeros((1,))[0]
        one = self._bkd.ones((1,))[0]
        D_1 = self._b * np.pi**4 / 5 + self._b**2 * np.pi**8 / 50 + 0.5 * one
        D_2 = self._a**2 / 8 * one
        D_3 = zero
        D_12 = zero
        D_13 = self._b**2 * np.pi**8 / 18 - self._b**2 * np.pi**8 / 50 * one
        D_23 = zero
        D_123 = zero
        return D_1, D_2, D_3, D_12, D_13, D_23, D_123

    def main_effects(self) -> Array:
        """
        Compute the main effects of the Ishigami function.

        Returns:
            Array: Main effects for each quantity of interest (QoI).
        """
        return (
            self._bkd.hstack(self._unnormalized_sobol_indices()[:3])
            / self.variance()
        )[:, None]

    def total_effects(self) -> Array:
        """
        Compute the total effects of the Ishigami function.

        Returns
        -------
        total_effects: Array
            Total effects for each quantity of interest (QoI).
        """
        D_1, D_2, D_3, D_12, D_13, D_23, D_123 = (
            self._unnormalized_sobol_indices()
        )
        total_effects1 = (
            self._bkd.hstack(
                [
                    D_1 + D_12 + D_13 + D_123,
                    D_2 + D_12 + D_23 + D_123,
                    D_3 + D_13 + D_23 + D_123,
                ]
            )
            / self.variance()
        )
        total_effects = (
            1
            - self._bkd.hstack(
                [D_2 + D_3 + D_23, D_1 + D_3 + D_13, D_1 + D_2 + D_12]
            )
            / self.variance()
        )
        assert np.allclose(total_effects1, total_effects)
        return total_effects[:, None]

    def sobol_indices(self) -> Array:
        """
        Compute the Sobol indices of the Ishigami function.

        Returns
        -------
        indices: Array
            Sobol indices for each quantity of interest (QoI).

        """
        sobol_indices = (
            self._bkd.hstack(self._unnormalized_sobol_indices())
            / self.variance()
        )
        return sobol_indices[:, None]

    def sobol_interaction_indices(self) -> Array:
        """
        Compute the Sobol interaction indices.

        Returns
        -------
        indcices: Array
            Sobol interaction indices for each quantity of interest (QoI).
        """
        sobol_interaction_indices = self._bkd.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=int,
        ).T
        return sobol_interaction_indices


def get_oakley_function_data(bkd: BackendMixin = NumpyMixin) -> Array:
    r"""
    Get the data :math:`a_1,a_2,a_3` and :math:`M` of the Oakley function

    .. math:: f(z) = a_1^Tz + a_2^T\sin(z) + a_3^T\cos(z) + z^TMz

    Returns
    -------
    a1 : Array (15)
       The vector :math:`a_1` of the Oakley function

    a2 : Array (15)
       The vector :math:`a_2` of the Oakley function

    a3 : Array (15)
       The vector :math:`a_3` of the Oakley function

    M : Array (15,15)
       The non-symmetric matrix :math:`M` of the Oakley function

    """
    a1 = bkd.array(
        [
            0.0118,
            0.0456,
            0.2297,
            0.0393,
            0.1177,
            0.3865,
            0.3897,
            0.6061,
            0.6159,
            0.4005,
            1.0741,
            1.1474,
            0.7880,
            1.1242,
            1.1982,
        ]
    )
    a2 = bkd.array(
        [
            0.4341,
            0.0887,
            0.0512,
            0.3233,
            0.1489,
            1.0360,
            0.9892,
            0.9672,
            0.8977,
            0.8083,
            1.8426,
            2.4712,
            2.3946,
            2.0045,
            2.2621,
        ]
    )
    a3 = bkd.array(
        [
            0.1044,
            0.2057,
            0.0774,
            0.2730,
            0.1253,
            0.7526,
            0.8570,
            1.0331,
            0.8388,
            0.7970,
            2.2145,
            2.0382,
            2.4004,
            2.0541,
            1.9845,
        ]
    )
    M = bkd.array(
        [
            [
                -2.2482886e-002,
                -1.8501666e-001,
                1.3418263e-001,
                3.6867264e-001,
                1.7172785e-001,
                1.3651143e-001,
                -4.4034404e-001,
                -8.1422854e-002,
                7.1321025e-001,
                -4.4361072e-001,
                5.0383394e-001,
                -2.4101458e-002,
                -4.5939684e-002,
                2.1666181e-001,
                5.5887417e-002,
            ],
            [
                2.5659630e-001,
                5.3792287e-002,
                2.5800381e-001,
                2.3795905e-001,
                -5.9125756e-001,
                -8.1627077e-002,
                -2.8749073e-001,
                4.1581639e-001,
                4.9752241e-001,
                8.3893165e-002,
                -1.1056683e-001,
                3.3222351e-002,
                -1.3979497e-001,
                -3.1020556e-002,
                -2.2318721e-001,
            ],
            [
                -5.5999811e-002,
                1.9542252e-001,
                9.5529005e-002,
                -2.8626530e-001,
                -1.4441303e-001,
                2.2369356e-001,
                1.4527412e-001,
                2.8998481e-001,
                2.3105010e-001,
                -3.1929879e-001,
                -2.9039128e-001,
                -2.0956898e-001,
                4.3139047e-001,
                2.4429152e-002,
                4.4904409e-002,
            ],
            [
                6.6448103e-001,
                4.3069872e-001,
                2.9924645e-001,
                -1.6202441e-001,
                -3.1479544e-001,
                -3.9026802e-001,
                1.7679822e-001,
                5.7952663e-002,
                1.7230342e-001,
                1.3466011e-001,
                -3.5275240e-001,
                2.5146896e-001,
                -1.8810529e-002,
                3.6482392e-001,
                -3.2504618e-001,
            ],
            [
                -1.2127800e-001,
                1.2463327e-001,
                1.0656519e-001,
                4.6562296e-002,
                -2.1678617e-001,
                1.9492172e-001,
                -6.5521126e-002,
                2.4404669e-002,
                -9.6828860e-002,
                1.9366196e-001,
                3.3354757e-001,
                3.1295994e-001,
                -8.3615456e-002,
                -2.5342082e-001,
                3.7325717e-001,
            ],
            [
                -2.8376230e-001,
                -3.2820154e-001,
                -1.0496068e-001,
                -2.2073452e-001,
                -1.3708154e-001,
                -1.4426375e-001,
                -1.1503319e-001,
                2.2424151e-001,
                -3.0395022e-002,
                -5.1505615e-001,
                1.7254978e-002,
                3.8957118e-002,
                3.6069184e-001,
                3.0902452e-001,
                5.0030193e-002,
            ],
            [
                -7.7875893e-002,
                3.7456560e-003,
                8.8685604e-001,
                -2.6590028e-001,
                -7.9325357e-002,
                -4.2734919e-002,
                -1.8653782e-001,
                -3.5604718e-001,
                -1.7497421e-001,
                8.8699956e-002,
                4.0025886e-001,
                -5.5979693e-002,
                1.3724479e-001,
                2.1485613e-001,
                -1.1265799e-002,
            ],
            [
                -9.2294730e-002,
                5.9209563e-001,
                3.1338285e-002,
                -3.3080861e-002,
                -2.4308858e-001,
                -9.9798547e-002,
                3.4460195e-002,
                9.5119813e-002,
                -3.3801620e-001,
                6.3860024e-003,
                -6.1207299e-001,
                8.1325416e-002,
                8.8683114e-001,
                1.4254905e-001,
                1.4776204e-001,
            ],
            [
                -1.3189434e-001,
                5.2878496e-001,
                1.2652391e-001,
                4.5113625e-002,
                5.8373514e-001,
                3.7291503e-001,
                1.1395325e-001,
                -2.9479222e-001,
                -5.7014085e-001,
                4.6291592e-001,
                -9.4050179e-002,
                1.3959097e-001,
                -3.8607402e-001,
                -4.4897060e-001,
                -1.4602419e-001,
            ],
            [
                5.8107658e-002,
                -3.2289338e-001,
                9.3139162e-002,
                7.2427234e-002,
                -5.6919401e-001,
                5.2554237e-001,
                2.3656926e-001,
                -1.1782016e-002,
                7.1820601e-002,
                7.8277291e-002,
                -1.3355752e-001,
                2.2722721e-001,
                1.4369455e-001,
                -4.5198935e-001,
                -5.5574794e-001,
            ],
            [
                6.6145875e-001,
                3.4633299e-001,
                1.4098019e-001,
                5.1882591e-001,
                -2.8019898e-001,
                -1.6032260e-001,
                -6.8413337e-002,
                -2.0428242e-001,
                6.9672173e-002,
                2.3112577e-001,
                -4.4368579e-002,
                -1.6455425e-001,
                2.1620977e-001,
                4.2702105e-003,
                -8.7399014e-002,
            ],
            [
                3.1599556e-001,
                -2.7551859e-002,
                1.3434254e-001,
                1.3497371e-001,
                5.4005680e-002,
                -1.7374789e-001,
                1.7525393e-001,
                6.0258929e-002,
                -1.7914162e-001,
                -3.1056619e-001,
                -2.5358691e-001,
                2.5847535e-002,
                -4.3006001e-001,
                -6.2266361e-001,
                -3.3996882e-002,
            ],
            [
                -2.9038151e-001,
                3.4101270e-002,
                3.4903413e-002,
                -1.2121764e-001,
                2.6030714e-002,
                -3.3546274e-001,
                -4.1424111e-001,
                5.3248380e-002,
                -2.7099455e-001,
                -2.6251302e-002,
                4.1024137e-001,
                2.6636349e-001,
                1.5582891e-001,
                -1.8666254e-001,
                1.9895831e-002,
            ],
            [
                -2.4388652e-001,
                -4.4098852e-001,
                1.2618825e-002,
                2.4945112e-001,
                7.1101888e-002,
                2.4623792e-001,
                1.7484502e-001,
                8.5286769e-003,
                2.5147070e-001,
                -1.4659862e-001,
                -8.4625150e-002,
                3.6931333e-001,
                -2.9955293e-001,
                1.1044360e-001,
                -7.5690139e-001,
            ],
            [
                4.1494323e-002,
                -2.5980564e-001,
                4.6402128e-001,
                -3.6112127e-001,
                -9.4980789e-001,
                -1.6504063e-001,
                3.0943325e-003,
                5.2792942e-002,
                2.2523648e-001,
                3.8390366e-001,
                4.5562427e-001,
                -1.8631744e-001,
                8.2333995e-003,
                1.6670803e-001,
                1.6045688e-001,
            ],
        ]
    )
    return a1, a2, a3, M


class OakleyModel(Model):
    r"""
    Oakley sensitivity benchmark model.

    This class implements the Oakley benchmark model, which is used for
    probabilistic sensitivity analysis. The model is defined as:

    .. math:: f(z) = a_1^Tz + a_2^T\sin(z) + a_3^T\cos(z) + z^TMz

    where :math:`z` consists of 15 I.I.D. standard Normal variables, and the
    data :math:`a_1, a_2, a_3` and :math:`M` are defined in the function
    :py:func:`pyapprox.benchmarks.algebraic.get_oakley_function_data`.

    Attributes:
        _bkd (BackendMixin): Backend used for numerical computations.

    Methods:
        nqoi: Return the number of quantities of interest (QoI).
        nvars: Return the number of uncertain variables.
        _values: Evaluate the model for given samples.
    """

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For the Oakley model, this is always 1.
        """
        return 1

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars: int
            Number of uncertain variables. For the Oakley model, this is
            always 15.
        """
        return 15

    def _values(self, samples: Array) -> Array:
        r"""
        Evaluate the Oakley model for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        vals: Array
            Array of shape (nsamples, 1) containing the model
            evaluations for each sample.
        """
        a1, a2, a3, M = get_oakley_function_data(self._bkd)
        term1 = a1 @ samples
        term2 = a2 @ self._bkd.sin(samples)
        term3 = a3 @ self._bkd.cos(samples)
        term4 = ((samples.T @ M) * samples.T).sum(axis=1)
        vals = term1 + term2 + term3 + term4
        return vals[:, None]


class OakleyBenchmark(SingleModelBenchmark):
    r"""
    Oakley sensitivity benchmark.

    The Oakley benchmark is a probabilistic sensitivity analysis function that
    models interactions between uncertain variables. It is defined as:

    .. math:: f(z) = a_1^Tz + a_2^T\sin(z) + a_3^T\cos(z) + z^TMz

    where :math:`z` consists of 15 I.I.D. standard Normal variables, and the
    data :math:`a_1, a_2, a_3` and :math:`M` are defined in the function
    :py:func:`pyapprox.benchmarks.algebraic.get_oakley_function_data`.

    This benchmark is widely used for testing sensitivity analysis methods due
    to its complexity and inclusion of nonlinear and interaction effects.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations

    References
    ----------
    .. [OakelyOJRSB2004] Oakley, J.E. and O'Hagan, A. (2004), Probabilistic
       sensitivity analysis of complex models: a Bayesian approach. Journal of
       the Royal Statistical Society: Series B (Statistical Methodology), 66:
       751-769. https://doi.org/10.1111/j.1467-9868.2004.05304.x
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the Oakley benchmark.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations
        """
        super().__init__(backend)
        self._set_statistics()

    def _variance_linear_combination_of_indendent_variables(
        self, coef: Array, variances: Array
    ) -> Array:
        """
        Compute the variance of a linear combination of independent variables.

        Parameters
        ----------
        coef : Array
            Coefficients of the linear combination.
        variances : Array
            Variances of the independent variables.

        Returns
        -------
        var: Array
            Variance of the linear combination.
        """
        assert coef.shape[0] == variances.shape[0]
        return self._bkd.sum(coef**2 * variances)

    def _set_model(self):
        """
        Set the Oakley model.

        The model is initialized with the specified backend.
        """
        self._model = OakleyModel(self._bkd)

    def _set_prior(self):
        """
        Set the prior distribution.

        The prior distribution is defined as 15 independent standard Normal
        variables.
        """
        marginals = [stats.norm()] * 15
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_statistics(self):
        """
        Compute and set the mean, variance, and main effects of the model.

        This method calculates the mean, variance, and main effects of the
        Oakley function based on its definition and the provided coefficients
        and matrix.
        """
        e = math.exp(1)
        a1, a2, a3, M = get_oakley_function_data(self._bkd)
        nvars = M.shape[0]

        # Compute mean
        term1_mean, term2_mean = 0, 0
        term3_mean = self._bkd.sum(a3 / math.sqrt(e))
        term4_mean = self._bkd.trace(M)
        self._mean = term1_mean + term2_mean + term3_mean + term4_mean

        # Compute variance
        term1_var = self._variance_linear_combination_of_indendent_variables(
            a1, self._bkd.ones(a1.shape[0])
        )
        variances_1d = self._bkd.ones(a2.shape[0]) * (0.5 * (1 - 1 / e**2))
        term2_var = self._variance_linear_combination_of_indendent_variables(
            a2, variances_1d
        )
        variances_1d = self._bkd.ones(a3.shape[0]) * (
            0.5 * (1 + 1 / e**2) - 1.0 / e
        )
        term3_var = self._variance_linear_combination_of_indendent_variables(
            a3, variances_1d
        )
        A = 0.5 * (M.T + M)  # Ensure symmetry of M
        term4_var = 2 * self._bkd.trace(A @ A)

        cov_xsinx = 1 / math.sqrt(e)
        covar13, covar14, covar23, covar24 = 0, 0, 0, 0
        covar12 = self._bkd.sum(a1 * a2 * cov_xsinx)
        covar34 = self._bkd.sum(-1 / math.sqrt(e) * a3 * self._bkd.diag(M))

        self._variance = term1_var + term2_var + term3_var + term4_var
        self._variance += 2 * (
            covar12 + covar13 + covar14 + covar23 + covar24 + covar34
        )

        # Compute main effects
        self._main_effects = self._bkd.empty((nvars, 1))
        for ii in range(nvars):
            var1 = a1[ii] ** 2
            var2 = a2[ii] ** 2 * (0.5 * (1 - 1 / e**2))
            var3 = a3[ii] ** 2 * (0.5 * (1 + 1 / e**2) - 1.0 / e)
            var4 = 2 * M[ii, ii] ** 2
            cov12 = cov_xsinx * a1[ii] * a2[ii]
            cov34 = -1 / math.sqrt(e) * a3[ii] * M[ii, ii]
            self._main_effects[ii] = (
                var1 + var2 + var3 + var4 + 2 * cov12 + 2 * cov34
            )
        self._main_effects /= self._variance

    def mean(self) -> Array:
        """
        Return the mean of the Oakley function.

        Returns
        -------
        mean : Array
            Mean value of the Oakley function.
        """
        return self._bkd.atleast1d(self._bkd.asarray(self._mean))

    def variance(self) -> Array:
        """
        Return the variance of the Oakley function.

        Returns
        -------
        var: Array
            Variance of the Oakley function.
        """
        return self._bkd.atleast1d(self._bkd.asarray(self._variance))

    def main_effects(self) -> Array:
        """
        Return the main effects of the Oakley function.

        Returns
        -------
        mean_effects: Array
            Main effects for each variable.
        """
        return self._main_effects


class SobolGModel(Model):
    r"""
    Sobol-G function benchmark model.

    This class implements the Sobol-G function, which is widely used for
    sensitivity analysis. The function is defined as:

    .. math:: f(z) = \prod_{i=1}^d\frac{\lvert 4z_i-2\rvert+a_i}{1+a_i},
              \quad a_i=\frac{i-2}{2}

    The coefficients :math:`a_i` control the sensitivity of each variable and
    limit the range of the outputs.

    Parameters
    ----------
    nvars : int
        Number of uncertain variables.
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, nvars: int, backend: BackendMixin):
        """
        Initialize the Sobol-G model.

        Parameters
        ----------
        nvars :int
            Number of uncertain variables.
        backend : BackendMixin
            Backend for numerical computations.
        """
        self._nvars = nvars
        super().__init__(backend)
        self._acoefs = (self._bkd.arange(1, self._nvars + 1) - 2) / 2

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
            nqoi: int
            Number of QoI. For the Sobol-G model, this is always 1.
        """
        return 1

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
            nvars: int
            Number of uncertain variables.
        """
        return self._nvars

    def _values(self, samples: Array) -> Array:
        r"""
        Evaluate the Sobol-G model for given samples.

        The coefficients :math:`a_i` control the sensitivity of each variable
        and limit the range of the outputs. Specifically, the outputs are
        bounded as follows:

        .. math::

            1 - \frac{1}{1+a_i} \leq \frac{\lvert 4z_i-2\rvert+a_i}{1+a_i}
            \leq 1 - \frac{1}{1+a_i}

        Parameters:
        ----------
        samples : Array:
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        vals : Array
            Array of shape (nsamples, 1) containing the model evaluations
            for each sample.
        """
        vals = self._bkd.prod(
            (self._bkd.abs(4 * samples - 2) + self._acoefs[:, None])
            / (1 + self._acoefs[:, None]),
            axis=0,
        )[:, None]
        return vals


class SobolGBenchmark(SingleModelBenchmark):
    r"""
    Sobol-G function benchmark.

    The Sobol-G function is a widely used benchmark for sensitivity analysis.
    It is defined as:

    .. math:: f(z) = \prod_{i=1}^d\frac{\lvert 4z_i-2\rvert+a_i}{1+a_i},
              \quad a_i=\frac{i-2}{2}

    where :math:`z` consists of `d` independent uniform random variables on
    [0, 1], and :math:`a_i` are coefficients that control the nonlinearity of
    the function.

    Parameters
    ----------
    nvars : int
        Number of uncertain variables. Default is 5.
    backend : BackendMixin
        Backend for numerical computations.

    References
    ----------
    .. [Saltelli1995] Saltelli, A., & Sobol, I. M. About the use of rank
       transformation in sensitivity analysis of model output. Reliability
       Engineering & System Safety, 50(3), 225-239, 1995.
       https://doi.org/10.1016/0951-8320(95)00099-2
    """

    def __init__(self, backend: BackendMixin, nvars: int = 5):
        """
        Initialize the Sobol-G benchmark.

        Parameters
        ----------
        nvars : int
            Number of uncertain variables. Default is 5.
        backend : BackendMixin
            Backend for numerical computations
        """
        self._nvars = nvars
        super().__init__(backend)
        self._acoefs = (
            self._bkd.arange(
                1, self.nvars() + 1, dtype=self._bkd.double_type()
            )
            - 2
        ) / 2
        self._set_statistics()

    def sobol_interaction_indices(self) -> Array:
        """
        Return the Sobol interaction indices.

        The interaction indices represent the combinations of variables that
        contribute to the interaction effects.

        Returns
        -------
        indices: Array
            Interaction indices as a binary matrix.
        """
        indices = []
        for compressed_index in self._interaction_indices:
            index = self._bkd.zeros((self.nvars(),), dtype=int)
            index[compressed_index] = 1
            indices.append(index)
        return self._bkd.stack(indices, axis=1)

    def _set_statistics(self):
        """
        Compute and set the mean, variance, main effects, total effects, and
        Sobol indices.

        This method calculates the statistical properties of the Sobol-G
        function based on its definition.
        """
        self._interaction_indices = [[ii] for ii in range(self.nvars())]
        # Get all single and pairwise interactions
        for ii in range(self.nvars()):
            for jj in range(ii + 1, self.nvars()):
                self._interaction_indices.append([ii, jj])
        zero = self._acoefs[0] * 0.0  # Necessary for torch compatibility
        self._mean = zero + 1.0
        unnormalized_main_effects = 1 / (3 * (1 + self._acoefs) ** 2)
        self._variance = self._bkd.prod(unnormalized_main_effects + 1) - 1
        self._main_effects = unnormalized_main_effects / self._variance
        self._total_effects = self._bkd.tile(
            self._bkd.prod(unnormalized_main_effects + 1),
            (self.nvars(),),
        )
        self._total_effects *= unnormalized_main_effects / (
            unnormalized_main_effects + 1
        )
        self._total_effects /= self._variance

        self._sobol_indices = self._bkd.array(
            [
                unnormalized_main_effects[index].prod() / self._variance
                for index in self._interaction_indices
            ]
        )[:, None]

    def _set_model(self):
        """
        Set the Sobol-G model.

        The model is initialized with the number of variables and the specified
        backend.
        """
        self._model = SobolGModel(self._nvars, self._bkd)

    def _set_prior(self):
        """
        Set the prior distribution.

        The prior distribution is defined as `nvars` independent uniform random
        variables on [0, 1].
        """
        marginals = [stats.uniform(0, 1)] * self._nvars
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def mean(self) -> Array:
        """
        Return the mean of the Sobol-G function.

        Returns
        -------
        mean : Array
            Mean value of the Sobol-G function.
        """
        return self._bkd.atleast1d(self._bkd.asarray(self._mean))

    def variance(self) -> Array:
        """
        Return the variance of the Sobol-G function.

        Returns
        -------
        var: Array
            Variance of the Sobol-G function.
        """
        return self._bkd.atleast1d(self._bkd.asarray(self._variance))

    def main_effects(self) -> Array:
        """
        Return the main effects of the Sobol-G function.

        Returns
        -------
        main_effects: Array
            Main effects for each variable.
        """
        return self._main_effects[:, None]

    def total_effects(self) -> Array:
        """
        Return the total effects of the Sobol-G function.

        Returns
        -------
        total_effects: Array
            Total effects for each variable.
        """
        return self._total_effects[:, None]

    def sobol_indices(self) -> Array:
        """
        Return the Sobol indices of the Sobol-G function.

        Returns
        -------
        indices: Array
            Sobol indices for each interaction.
        """
        return self._sobol_indices


class RosenbrockModel(Model):
    r"""
    The Rosenbrock function model.

    The Rosenbrock function is a classic benchmark function for optimization
    problems. It is defined as:

    .. math::
        f(z) = \sum_{i=1}^{d/2}\left[100(z_{2i-1}^{2}-z_{2i})^{2}+(z_{2i-1}-1)^{2}\right]

    Parameters
    ----------
    nvars : int
        Number of uncertain variables (dimensionality of the input).
    backend : BackendMixin
        Backend for numerical computations.


    Notes
    -----
    This method uses the `rosen` function from `scipy.optimize`. Note that
    backpropagation will not work with this function when using PyTorch
    because `rosen` is a NumPy function.
    """

    def __init__(self, nvars: int, backend: BackendMixin):
        """
        Initialize the Rosenbrock model.

        Parameters
        ----------
        nvars : int
            Number of uncertain variables (dimensionality of the input).
        backend : BackendMixin
            Backend for numerical computations.
        """
        super().__init__(backend)
        self._nvars = nvars

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        flag : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def apply_hessian_implemented(self) -> bool:
        """
        Check if the Hessian application is implemented.

        Returns
        -------
        flag : bbool
            True if the Hessian application is implemented, False otherwise.
        """
        return True

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For the Rosenbrock model, this is always 1.
        """
        return 1

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars: int
            Number of uncertain variables.
        """
        return self._nvars

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the Rosenbrock function for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        vals : Array
            Array of shape (nsamples, 1) containing the function evaluations
            for each sample.
        """
        return self._bkd.asarray(rosen(self._bkd.to_numpy(samples)))[:, None]

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the Rosenbrock function at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac: Array
            Array of shape (nvars, 1) containing the Jacobian at the sample.
        """
        return self._bkd.asarray(rosen_der(self._bkd.to_numpy(sample))).T

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Hessian of the Rosenbrock function to a vector.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.
        vec : Array
            Array of shape (nvars, 1) containing the vector to which the
            Hessian is applied.

        Returns
        -------
        hvp: Array
            Array of shape (nvars, 1) containing the result of the Hessian
            application.
        """
        return self._bkd.asarray(
            rosen_hess_prod(
                self._bkd.to_numpy(sample[:, 0]), self._bkd.to_numpy(vec[:, 0])
            )
        )[:, None]


class RosenbrockUnconstrainedOptimizationBenchmark(OptimizationBenchmark):
    r"""
    The Rosenbrock function benchmark.

    The Rosenbrock function is a classic benchmark for optimization problems,
    particularly for testing unconstrained optimization algorithms. It is
    defined as:

    .. math::
        f(z) = \sum_{i=1}^{d/2}\left[100(z_{2i-1}^{2}-z_{2i})^{2}+(z_{2i-1}-1)^{2}\right]

    Parameters
    ----------
    nvars : int
        Number of uncertain variables (dimensionality of the input).
    backend : BackendMixin
        Backend for numerical computations

    References
    ----------
    .. [DixonSzego1990] Dixon, L. C. W.; Mills, D. J. "Effect of Rounding
       Errors on the Variable Metric Method". Journal of Optimization Theory
       and Applications. 80: 175–179. 1994.
       https://doi.org/10.1007%2FBF02196600
    """

    def __init__(
        self,
        nvars: int,
        backend: BackendMixin,
    ):
        """
        Initialize the Rosenbrock benchmark.

        Parameters
        ----------
        nvars : int
            Number of uncertain variables (dimensionality of the input).
        backend : BackendMixin
            Backend for numerical computations.
        """
        self._nvars = nvars
        super().__init__(backend)

    def _set_objective(self):
        """
        Set the objective function for the optimization problem.

        The objective is the Rosenbrock function model.
        """
        self._objective = RosenbrockModel(self._nvars, self._bkd)

    def design_variable(self) -> DesignVariable:
        """
        Define the design variable for the optimization problem.

        The design variable is bounded in [-2, 2] for each dimension.

        Returns
        -------
        var: DesignVariable
            The design variable with bounds.
        """
        design_bounds = self._bkd.full((self._nvars, 2), -2.0)
        design_bounds[:, 1] = 2.0
        return DesignVariable(design_bounds)

    def prior(self) -> IndependentMarginalsVariable:
        """
        Define the prior distribution for the optimization problem.

        The prior distribution is uniform in [-2, 2] for each dimension.

        Returns
        -------
        prior: IndependentMarginalsVariable
            The prior distribution.
        """
        marginals = [stats.uniform(-2, 4)] * self._nvars
        return IndependentMarginalsVariable(marginals, backend=self._bkd)

    def mean(self) -> Array:
        """
        Compute the mean of the Rosenbrock function.

        The mean is computed for uniform variables in [-2, 2]^d.

        Returns
        -------
        mean : Array
            The mean value of the Rosenbrock function.

        Notes
        -----
        The mean is computed using symbolic integration for exact results.
        """
        assert self._nvars % 2 == 0
        import sympy as sp

        lb, ub = -2, 2
        x, y = sp.Symbol("x"), sp.Symbol("y")
        exact_mean = (
            self._nvars
            / 2
            * float(
                sp.integrate(
                    100 * (y - x**2) ** 2 + (1 - x) ** 2,
                    (x, lb, ub),
                    (y, lb, ub),
                )
            )
            / (4**self._nvars)
        )
        return self._bkd.atleast1d(self._bkd.asarray(exact_mean))

    def optimal_iterate(self) -> Array:
        """
        Return the optimal iterate for the Rosenbrock function.

        The optimal iterate is a vector of ones.

        Returns
        -------
        iterate: Array
            The optimal iterate.
        """
        return self._bkd.ones((self._nvars, 1))

    def init_iterate(self) -> Array:
        """
        Return the initial iterate for the Rosenbrock function.

        The initial iterate is a vector with all entries equal to  1.5.

        Returns
        -------
        iterate : Array
            The initial iterate.
        """
        return self._bkd.ones((self._nvars, 1)) + 0.5


class RosenbrockConstraint(Constraint):
    """
    Rosenbrock constraint.

    This class implements constraints for optimization problems involving the
    Rosenbrock function. The constraints are defined as:

    .. math::
        g_1(x) = -((x_1 - 1)^3) + x_2 - 1 \\
        g_2(x) = 2 - x_1 - x_2

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the Rosenbrock constraint.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.
        """
        bounds = backend.array([[0.0, np.inf], [0.0, np.inf]])
        super().__init__(bounds, True, backend)

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        flag: bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def hessian_implemented(self) -> bool:
        """
        Check if the Hessian is implemented.

        Returns
        -------
        flag : bool
            True if the Hessian is implemented, False otherwise.
        """
        return True

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars: int
            Number of uncertain variables. For this constraint, it is always 2.
        """
        return 2

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi: int
            Number of QoI. For this constraint, it is always 2.
        """
        return 2

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the constraint values for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        values : Array
            Array of shape (nsamples, nqoi) containing the constraint values
            for each sample.
        """
        return self._bkd.stack(
            [
                -((samples[0] - 1) ** 3) + samples[1] - 1,
                2 - samples[0] - samples[1],
            ],
            axis=1,
        )

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the constraints at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac : Array
            Array of shape (nqoi, nvars) containing the Jacobian matrix at the
            sample.
        """
        return self._bkd.stack(
            (
                self._bkd.hstack(
                    [-3 * (sample[0] - 1) ** 2, self._bkd.ones((1,))]
                ),
                self._bkd.full((self.nvars(),), -1),
            ),
            axis=0,
        )

    def _hessian(self, sample: Array) -> Array:
        """
        Compute the Hessian of the constraints at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        hess : Array
            Array of shape (nqoi, nvars, nvars) containing the Hessian matrices
            for each constraint at the sample.
        """
        zero = self._bkd.zeros((1,))
        hess = self._bkd.zeros((self.nqoi(), self.nvars(), self.nvars()))
        hess[0] = self._bkd.stack(
            (
                self._bkd.hstack((-6 * (sample[0] - 1), zero)),
                self._bkd.zeros((self.nvars(),)),
            ),
            axis=0,
        )
        return hess


class RosenbrockConstrainedOptimizationBenchmark(
    ConstrainedOptimizationBenchmark
):
    r"""
    Rosenbrock constrained optimization benchmark.

    This class implements a constrained optimization benchmark based on the
    Rosenbrock function.

    The objective is defined as:

    .. math::
        f(z) = \sum_{i=1}^{d/2}\left[100(z_{2i-1}^{2}-z_{2i})^{2}+(z_{2i-1}-1)^{2}\right]

    The constraints are defined as:

    .. math::
        g_1(x) = -((x_1 - 1)^3) + x_2 - 1 \\
        g_2(x) = 2 - x_1 - x_2
    
    The design variable is bounded in [-2, 2] for each dimension.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the Rosenbrock constrained optimization benchmark.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations
        """
        self._nvars = 2
        super().__init__(backend)

    def optimal_iterate(self) -> Array:
        """
        Return the optimal iterate for the Rosenbrock function.

        Returns
        -------
        optimal_iterate : Array
            The optimal iterate is a vector of ones.
        """
        return self._bkd.ones((self._nvars, 1))

    def init_iterate(self) -> Array:
        """
        Return the initial iterate for the Rosenbrock function.

        The initial iterate is chosen to lie on the constraint boundary, with
        a small offset for numerical stability.

        Returns
        -------
        init_iterate : Array
            The initial iterate is a vector near the constraint boundary.
        """
        eps = 0.2  # Initial iterate on constraint
        # Shift x-coordinate due to issues with scipy trust-constr solver
        return self._bkd.array([1 - eps - 1e-16, 1 + eps])[:, None]

    def _set_objective(self):
        """
        Set the objective function for the optimization problem.

        The objective is the Rosenbrock function model.
        """
        self._objective = RosenbrockModel(2, self._bkd)

    def design_variable(self) -> DesignVariable:
        """
        Define the design variable for the optimization problem.

        The design variable is bounded in [-2, 2] for each dimension.

        Returns
        -------
        design_variable : DesignVariable
            The design variable with bounds.
        """
        design_bounds = self._bkd.full((self._nvars, 2), -2.0)
        design_bounds[:, 1] = 2.0
        return DesignVariable(design_bounds)

    def prior(self) -> IndependentMarginalsVariable:
        """
        Define the prior distribution for the optimization problem.

        The prior distribution is uniform in [-2, 2] for each dimension.

        Returns
        -------
        prior : IndependentMarginalsVariable
            The prior distribution.
        """
        marginals = [stats.uniform(-2, 4)] * self._nvars
        return IndependentMarginalsVariable(marginals, backend=self._bkd)

    def _set_constraints(self):
        """
        Set the constraints for the optimization problem.

        The constraints are defined by the `RosenbrockConstraint` class.
        """
        self._constraints = [RosenbrockConstraint(self._bkd)]


class CantileverBeamModel(SingleSampleModel):
    """
    Cantilever beam model.

    This class implements a cantilever beam model with symbolic expressions for
    the function, Jacobian, and Hessian. The model evaluates quantities of
    interest (QoI) related to the beam's cross-sectional area, stress, and
    deflection.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the cantilever beam model.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.
        """
        import sympy as sp

        super().__init__(backend)
        symbs = sp.symbols(["X", "Y", "E", "R", "w", "t"])
        X, Y, E, R, w, t = symbs
        L, D0 = 100, 2.2535
        sp_fun = [
            w * t,
            1 - 6 * L / (w * t) * (X / w + Y / t) / R,
            1
            - 4 * L**3 / (E * w * t) * sp.sqrt(X**2 / w**4 + Y**2 / t**4) / D0,
        ]
        sp_grad = [[fun.diff(x) for x in symbs] for fun in sp_fun]
        sp_hess = [
            [[fun.diff(_x).diff(_y) for _x in symbs] for _y in symbs]
            for fun in sp_fun
        ]
        self._lam_fun = sp.lambdify(symbs, sp_fun, "numpy")
        self._lam_jac = sp.lambdify(symbs, sp_grad, "numpy")
        self._lam_hess = sp.lambdify(symbs, sp_hess, "numpy")

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def apply_jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian application is implemented.

        Returns
        -------
        apply_jacobian_implemented : bool
            True if the Jacobian application is implemented, False otherwise.
        """
        return True

    def hessian_implemented(self) -> bool:
        """
        Check if the Hessian is implemented.

        Returns
        -------
        hessian_implemented : bool
            True if the Hessian is implemented, False otherwise.
        """
        return True

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this model, it is 2.
        """
        return 2

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars : int
            Number of uncertain variables. For this model, it is 6.
        """
        return 6

    def _evaluate_sp_lambda(self, sp_lambda: callable, sample: Array) -> Array:
        """
        Evaluate a symbolic lambda function for a given sample.

        Parameters
        ----------
        sp_lambda : callable
            Symbolic lambda function to evaluate.
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        vals : Array
            Array of shape (nqoi, 1) containing the evaluation result.
        """
        assert sample.ndim == 2 and sample.shape[1] == 1
        vals = self._bkd.atleast2d(
            self._bkd.asarray(sp_lambda(*self._bkd.to_numpy(sample[:, 0])))
        )
        return vals

    def _evaluate(self, sample: Array):
        """
        Evaluate the cantilever beam model for a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        vals : Array
            Array of shape (nqoi, 1) containing the model evaluations.
        """
        return self._evaluate_sp_lambda(self._lam_fun, sample)

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the cantilever beam model at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac : Array
            Array of shape (nqoi, nvars) containing the Jacobian matrix.
        """
        return self._evaluate_sp_lambda(self._lam_jac, sample)

    def _apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Jacobian of the cantilever beam model to a vector.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.
        vec : Array
            Array of shape (nvars, 1) containing the vector to which the
            Jacobian is applied.

        Returns
        -------
        jvp : Array
            Array of shape (nqoi, 1) containing the result of the Jacobian
            application.
        """
        return self.jacobian(sample) @ vec

    def _hessian(self, sample: Array) -> Array:
        """
        Compute the Hessian of the cantilever beam model at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        hess : Array
            Array of shape (nqoi, nvars, nvars) containing the Hessian matrices.
        """
        return self._evaluate_sp_lambda(self._lam_hess, sample)


class CantileverBeamObjectiveModel(CantileverBeamModel):
    """
    Cantilever beam objective model.

    This class implements the objective model for the cantilever beam problem,
    which evaluates the first quantity of interest (QoI) related to the beam's
    cross-sectional area.
    """

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this model, it is 1.
        """
        return 1

    def _evaluate(self, sample: Array) -> Array:
        """
        Evaluate the objective model for a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        vals : Array
            Array of shape (1, 1) containing the evaluation result.
        """
        return super()._evaluate(sample)[:, :1]

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the objective model at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac : Array
            Array of shape (1, nvars) containing the Jacobian matrix.
        """
        return super()._jacobian(sample)[:1, :]

    def _hessian(self, sample: Array) -> Array:
        """
        Compute the Hessian of the objective model at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        hess : Array
            Array of shape (1, nvars, nvars) containing the Hessian matrix.
        """
        return self._bkd.copy(super()._hessian(sample)[:1, ...])


class CantileverBeamConstraintsModel(CantileverBeamModel):
    r"""
    Cantilever beam constraints model.

    This class implements the constraints model for the cantilever beam
    problem, which evaluates the second and third quantities of interest (QoI)
    related to stress and deflection.


    Specifically, the stress constraint

    .. math:: 6L\left(\frac{X}{tw^2}+\frac{Y}{t^2w}\right) < R

    and the displacement constraint

    .. math:: \frac{4L^3}{Ewt}\sqrt{\left(\frac{Y}{t}\right)^2+\left(\frac{X}{w}\right)^2} < D
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the cantilever beam constraints model.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        super().__init__(backend)

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this model, it is 2.
        """
        return 2

    def _evaluate(self, sample: Array) -> Array:
        """
        Evaluate the constraints model for a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        vals : Array
            Array of shape (2, 1) containing the evaluation result.
        """
        return super()._evaluate(sample)[:, 1:]

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the constraints model at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac : Array
            Array of shape (2, nvars) containing the Jacobian matrix.
        """
        return super()._jacobian(sample)[1:, :]

    def _hessian(self, sample: Array) -> Array:
        """
        Compute the Hessian of the constraints model at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        hess : Array
            Array of shape (2, nvars, nvars) containing the Hessian matrices.
        """
        return self._bkd.copy(super()._hessian(sample)[1:, ...])


class CantileverBeamDeterminsticOptimizationBenchmark(
    ConstrainedUncertainOptimizationBenchmark
):
    r"""
    Cantilever beam deterministic optimization benchmark.

    This class implements a deterministic optimization benchmark for the
    cantilever beam problem, including constraints and design variables.

    The optimization problem minimizes the objective function

    .. math:: wt

    Subject to a stress constraint

    .. math:: 6L\left(\frac{X}{tw^2}+\frac{Y}{t^2w}\right) < R

    and a displacement constraint

    .. math:: \frac{4L^3}{Ewt}\sqrt{\left(\frac{Y}{t}\right)^2+\left(\frac{X}{w}\right)^2} < D
    """

    def _set_objective(self):
        """
        Set the objective function for the optimization problem.
        """
        self._objective = create_active_set_variable_model(
            CantileverBeamObjectiveModel(self._bkd),
            self.variable().nvars() + self.design_variable().nvars(),
            self._nominal_values,
            self._design_var_indices,
        )

    def _set_constraints(self):
        """
        Set the constraints for the optimization problem.
        """
        constraint_model = create_active_set_variable_model(
            CantileverBeamConstraintsModel(self._bkd),
            self.variable().nvars() + self.design_variable().nvars(),
            self._nominal_values,
            self._design_var_indices,
        )
        self._constraints = [
            ConstraintFromModel(constraint_model, self.constraint_bounds())
        ]

    def prior(self) -> JointVariable:
        """
        Define the prior distribution for the optimization problem.

        The marginal distribution of the independent random variables are

        .. table:: Uncertainties
           :align: center

           =============== ========= =======================
           Uncertainty     Symbol    Prior
           =============== ========= =======================
           Yield stress    :math:`R` :math:`N(40000,2000)`
           Young's modulus :math:`E` :math:`N(2.9e7,1.45e6)`
           Horizontal load :math:`X` :math:`N(500,100)`
           Vertical Load   :math:`Y` :math:`N(1000,100)`
           =============== ========= =======================

        Returns
        -------
        prior : JointVariable
            The prior distribution.
        """
        X = stats.norm(loc=500, scale=np.sqrt(100) ** 2)
        Y = stats.norm(loc=1000, scale=np.sqrt(100) ** 2)
        E = stats.norm(loc=2.9e7, scale=np.sqrt(1.45e6) ** 2)
        R = stats.norm(loc=40000, scale=np.sqrt(2000) ** 2)
        self._prior = IndependentMarginalsVariable(
            [X, Y, E, R], backend=self._bkd
        )
        self._nominal_values = self._prior.mean()
        return self._prior

    def design_variable(self) -> DesignVariable:
        """
        Define the design variable for the optimization problem.

        Returns
        -------
        design_variable : DesignVariable
            The design variable with bounds.
        """
        design_bounds = self._bkd.stack(
            [self._bkd.ones((2,)), self._bkd.full((2,), 4.0)], axis=1
        )
        self._design_variable = DesignVariable(design_bounds)
        self._design_var_indices = self._bkd.array([4, 5], dtype=int)
        return self._design_variable

    def constraint_bounds(self) -> Array:
        """
        Define the bounds for the constraints.

        Returns
        -------
        constraint_bounds : Array
            Array of shape (2, 2) containing the constraint bounds.
        """
        return self._bkd.hstack(
            [self._bkd.zeros((2, 1)), self._bkd.full((2, 1), np.inf)]
        )

    def design_var_indices(self) -> Array:
        """
        Return the indices of the design variables.

        Returns
        -------
        design_var_indices : Array
            Array of shape (2,) containing the indices of the design variables.
        """
        return self._design_var_indices

    def init_iterate(self) -> Array:
        """
        Return the initial iterate for the optimization problem.

        Returns
        -------
        init_iterate : Array
            Array of shape (2, 1) containing the initial iterate.
        """
        return self._bkd.array([3.0, 3.0])[:, None]

    def optimal_iterate(self) -> Array:
        """
        Return the optimal iterate for the optimization problem.

        Returns
        -------
        optimal_iterate : Array
            Array of shape (2, 1) containing the optimal iterate.
        """
        return self._bkd.array([2.35, 3.33])[:, None]


class CantileverBeamUncertainOptimizationBenchmark(
    CantileverBeamDeterminsticOptimizationBenchmark
):
    """
    Cantilever beam uncertain optimization benchmark.

    This class extends the deterministic optimization benchmark for the
    cantilever beam problem to include uncertainty in the constraints. The
    constraints are evaluated using a quadrature rule and statistical metrics
    such as the mean and standard deviation.
    """

    def _set_constraints(self):
        """
        Set the constraints for the optimization problem.

        The constraints are evaluated using a quadrature rule and statistical
        metrics such as the mean and standard deviation.

        Returns
        -------
        None
        """
        # TODO: Change weights to create unbiased estimators of mean and variance
        quad_rule = FixedGaussianTensorProductQuadratureRuleFromVariable(
            self.variable(),
            [5 for ii in range(self.variable().nvars())],
        )
        samples, weights = quad_rule()
        constraint_model = ChangeModelSignWrapper(
            CantileverBeamConstraintsModel(self._bkd)
        )
        stat = SampleAverageMeanPlusStdev(3, backend=self._bkd)
        self._constraints = [
            SampleAverageConstraint(
                constraint_model,
                samples,
                weights,
                stat,
                self.constraint_bounds(),
                self.variable().nvars() + self.design_variable().nvars(),
                self._design_var_indices,
                backend=self._bkd,
            )
        ]

    def constraint_bounds(self) -> Array:
        """
        Define the bounds for the constraints.

        The bounds are defined as [-∞, 0] for each constraint.

        Returns
        -------
        constraint_bounds : Array
            Array of shape (2, 2) containing the constraint bounds.
        """
        return self._bkd.hstack(
            [self._bkd.full((2, 1), -np.inf), self._bkd.zeros((2, 1))]
        )

    def optimal_iterate(self) -> Array:
        """
        Raise a `NotImplementedError` as the optimal iterate is not defined.

        Returns
        -------
        optimal_iterate : Array
            Raises `NotImplementedError`.
        """
        raise NotImplementedError(
            "Optimal design vars from literature are not accurate enough"
        )


class PistonModel(Model):
    r"""
    Predict cycle time of a piston in a cylinder.

    This class models the cycle time of a piston in a cylinder based on various
    physical parameters. The cycle time is computed using the following formula:

    .. math::
        Z = P_0 V_0 / T_0 T_a \\
        A = P_0 S + 19.62 M - k V_0 / S \\
        V = S / (2k) (\sqrt{A^2 + 4kZ} - A) \\
        C = 2\pi \sqrt{M / (k + S^2 Z / V^2)}

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the piston model.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations
        """
        super().__init__(backend)

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this model, it is 1.
        """
        return 1

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars : int
            Number of uncertain variables. For this model, it is 7.
        """
        return 7

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the cycle time for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        vals : Array
            Array of shape (nsamples, 1) containing the cycle time for each
            sample.
        """
        M, S, V_0, k, P_0, T_a, T_0 = samples
        Z = P_0 * V_0 / T_0 * T_a
        A = P_0 * S + 19.62 * M - k * V_0 / S
        V = S / (2.0 * k) * (self._bkd.sqrt(A**2 + 4.0 * k * Z) - A)
        C = 2.0 * np.pi * self._bkd.sqrt(M / (k + S**2 * Z / V**2))
        return C[:, None]

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the gradient of the cycle time for a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac : Array
            Array of shape (nvars, 1) containing the gradient of the cycle time.
        """
        M, S, V_0, k, P_0, T_a, T_0 = sample

        Z = P_0 * V_0 / T_0 * T_a
        A = P_0 * S + 19.62 * M - k * V_0 / S
        V = S / (2.0 * k) * (self._bkd.sqrt(A**2 + 4.0 * k * Z) - A)

        tmp0 = S**2 * P_0 * V_0
        tmp1 = k + tmp0 * T_a / (T_0 * V**2)
        tmp2 = (A**2 + 4 * k * P_0 * V_0 * T_a / T_0) ** (-0.5)
        tmp3 = np.pi * M**0.5 * tmp1 ** (-1.5)

        grad_M = np.pi * (M * tmp1) ** (
            -0.5
        ) + 2 * tmp3 * S**3 * P_0 * V_0 * T_a / (2 * k * T_0 * V**3) * (
            tmp2 * A * 19.62 - 19.62
        )
        grad_S = -tmp3 * (
            2 * S * P_0 * V_0 * T_a / (T_0 * V**2)
            - 2
            * tmp0
            * T_a
            / (T_0 * V**3)
            * (
                V / S
                + S
                / (2 * k)
                * (tmp2 * A * (P_0 + k * V_0 / S**2) - P_0 - k * V_0 / S**2)
            )
        )
        grad_V_0 = -tmp3 * (
            S**2 * P_0 * T_a / (T_0 * V**2)
            - 2
            * S**3
            * P_0
            * V_0
            * T_a
            / (2 * k * T_0 * V**3)
            * (tmp2 / 2 * (4 * k * P_0 * T_a / T_0 - 2 * A * k / S) + k / S)
        )
        grad_k = -tmp3 * (
            1
            - 2
            * tmp0
            * T_a
            / (T_0 * V**3)
            * (
                -V / k
                + S
                / (2 * k)
                * (
                    tmp2 / 2 * (4 * P_0 * V_0 * T_a / T_0 - 2 * A * V_0 / S)
                    + V_0 / S
                )
            )
        )
        grad_P_0 = -tmp3 * (
            S**2 * V_0 * T_a / (T_0 * V**2)
            - 2
            * S**3
            * P_0
            * V_0
            * T_a
            / (2 * k * T_0 * V**3)
            * (tmp2 / 2 * (4 * k * V_0 * T_a / T_0 + 2 * A * S) - S)
        )
        grad_T_a = -tmp3 * (
            tmp0 / (T_0 * V**2)
            - 2
            * S**3
            * P_0
            * V_0
            * T_a
            / (2 * k * T_0 * V**3)
            * (tmp2 / 2 * 4 * k * P_0 * V_0 / T_0)
        )
        grad_T_0 = tmp3 * (
            tmp0 * T_a / (T_0**2 * V**2)
            + 2
            * S**3
            * P_0
            * V_0
            * T_a
            / (2 * k * T_0 * V**3)
            * (-tmp2 / 2 * P_0 * 4 * k * V_0 * T_a / T_0**2)
        )
        return self._bkd.vstack(
            (grad_M, grad_S, grad_V_0, grad_k, grad_P_0, grad_T_a, grad_T_0)
        ).T


class PistonBenchmark(SingleModelBenchmark):
    r"""
    Piston benchmark.

    This class models the cycle time of a piston in a cylinder based on various
    physical parameters. The `PistonModel` evaluates the cycle time
    using the following formula:

    .. math::
        Z = P_0 V_0 / T_0 T_a \\
        A = P_0 S + 19.62 M - k V_0 / S \\
        V = S / (2k) (\sqrt{A^2 + 4kZ} - A) \\
        C = 2\pi \sqrt{M / (k + S^2 Z / V^2)}

    The prior distribution for the piston parameters is defined as follows:

    ============== ========= ======================= ==============
    Parameter      Symbol    Prior                   Units
    ============== ========= ======================= ==============
    Piston mass    :math:`M` :math:`U(30,60)`        kg
    Surface area   :math:`S` :math:`U(0.005,0.02)`   m²
    Initial volume :math:`V_0` :math:`U(0.002,0.01)` m³
    Spring coeff.  :math:`k` :math:`U(1000,5000)`    N/m
    Atmos. pressure :math:`P_0` :math:`U(90000,110000)` N/m²
    Ambient temp.  :math:`T_a` :math:`U(290,296)`    K
    Gas temp.      :math:`T_0` :math:`U(340,360)`    K
    ============== ========= ======================= ==============
    """

    def _set_model(self):
        """
        Set the piston model as the benchmark model.
        """
        self._model = PistonModel(self._bkd)

    def _set_prior(self):
        """
        Define the prior distribution for the piston parameters.

        The prior distribution includes the following parameters:
        - M: Piston mass (kg)
        - S: Surface area (m²)
        - V_0: Initial gas volume (m³)
        - k: Spring coefficient (N/m)
        - P_0: Atmospheric pressure (N/m²)
        - T_a: Ambient temperature (K)
        - T_0: Filling gas temperature (K)
        """
        M = stats.uniform(loc=30.0, scale=30.0)
        S = stats.uniform(loc=0.005, scale=0.015)
        V_0 = stats.uniform(loc=0.002, scale=0.008)
        k = stats.uniform(loc=1000.0, scale=4000.0)
        P_0 = stats.uniform(loc=90000.0, scale=20000.0)
        T_a = stats.uniform(loc=290.0, scale=6.0)
        T_0 = stats.uniform(loc=340.0, scale=20.0)

        self._prior = IndependentMarginalsVariable(
            [M, S, V_0, k, P_0, T_a, T_0], backend=self._bkd
        )


class WingWeightModel(Model):
    r"""
    Weight of a light aircraft wing.

    This class models the weight of a light aircraft wing based on various
    physical parameters.

    The weight is:

    .. math::
        f(z) = 0.036 \cdot S_w^{0.758} \cdot W_{fw}^{0.0035} \cdot A^{0.6}
        \cdot \cos(\Lambda)^{-0.9} \cdot q^{0.006} \cdot \lambda^{0.04} \cdot 100^{-0.3}
        \cdot tc^{-0.3} \cdot N_z^{0.49} \cdot W_{dg}^{0.49} + S_w \cdot W_p

    where :math:`z=[S_w, W_{fw}, A, \Lambda, q, \lambda, tc, N_z, W_{dg}, W_p]`

    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the wing weight model.

        Parameters
        ----------
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        super().__init__(backend)

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this model, it is 1.
        """
        return 1

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars : int
            Number of uncertain variables. For this model, it is 10.
        """
        return 10

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the wing weight for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        _values : Array
            Array of shape (nsamples, 1) containing the wing weight for each
            sample.
        """
        S_w, W_fw, A, Lamda, q, lamda, tc, N_z, W_dg, W_p = samples
        Lamda *= np.pi / 180.0
        vals = (
            0.036
            * (S_w**0.758)
            * (W_fw**0.0035)
            * (A**0.6)
            * (self._bkd.cos(Lamda) ** -0.9)
            * (q**0.006)
            * (lamda**0.04)
            * (100**-0.3)
            * (tc**-0.3)
            * (N_z**0.49)
            * (W_dg**0.49)
        ) + S_w * W_p
        return vals[:, None]

    def _jacobian(self, samples: Array) -> Array:
        """
        Compute the gradient of the wing weight for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        _jacobian : Array
            Array of shape (nsamples, nvars) containing the gradient of the wing
            weight for each sample.
        """
        S_w, W_fw, A, Lamda, q, lamda, tc, N_z, W_dg, W_p = samples
        Lamda *= np.pi / 180.0
        vals = self(samples)[:, 0] - S_w * W_p
        nvars, nsamples = samples.shape
        grad = self._bkd.empty((nvars, nsamples))
        grad[0] = 0.758 * vals / S_w + W_p
        grad[1] = 0.0035 * vals / W_fw
        grad[2] = 0.6 * vals / A
        grad[3] = (
            (0.9 * vals * self._bkd.sin(Lamda) / self._bkd.cos(Lamda))
            * np.pi
            / 180
        )
        grad[4] = 0.006 * vals / q
        grad[5] = 0.04 * vals / lamda
        grad[6] = -0.3 * vals / tc
        grad[7] = 0.49 * vals / N_z
        grad[8] = 0.49 * vals / W_dg
        grad[9] = S_w
        return grad.T


class WingWeightBenchmark(SingleModelBenchmark):
    r"""
    Wing weight benchmark.

    This class models the weight of a light aircraft wing based on various
    physical parameters. It uses the `WingWeightModel` to evaluate the wing
    weight.

    The weight is:

    .. math::
        f(z) = 0.036 \cdot S_w^{0.758} \cdot W_{fw}^{0.0035} \cdot A^{0.6}
        \cdot \cos(\Lambda)^{-0.9} \cdot q^{0.006} \cdot \lambda^{0.04} \cdot 100^{-0.3}
        \cdot tc^{-0.3} \cdot N_z^{0.49} \cdot W_{dg}^{0.49} + S_w \cdot W_p

    where :math:`z=[S_w, W_{fw}, A, \Lambda, q, \lambda, tc, N_z, W_{dg}, W_p]`


    The variables are independent with the marginals:

    ============== ========= =======================
    Symbol        Prior
    ============== ========= =======================
    :math:`S_w`   :math:`U(150,200)`
    :math:`W_{fw}` :math:`U(220,300)`
    :math:`A`     :math:`U(6,10)`
    :math:`\Lambda` :math:`U(-10,10)`
    :math:`q`     :math:`U(16,45)`
    :math:`\lambda` :math:`U(0.5,1.0)`
    :math:`tc`    :math:`U(0.08,0.18)`
    :math:`N_z`   :math:`U(2.5,6.0)`
    :math:`W_{dg}` :math:`U(1700,2500)`
    :math:`W_p`   :math:`U(0.025,0.08)`
    ============== ========= =======================
    """

    def _set_model(self):
        """
        Set the wing weight model as the benchmark model.
        """
        self._model = WingWeightModel(self._bkd)

    def _set_prior(self):
        """
        Define the prior distribution for the wing weight parameters.

        Returns
        -------
        None
        """
        marginals = [
            stats.uniform(150, 50),
            stats.uniform(220, 80),
            stats.uniform(6, 4),
            stats.uniform(-10, 20),
            stats.uniform(16, 29),
            stats.uniform(0.5, 0.5),
            stats.uniform(0.08, 0.1),
            stats.uniform(2.5, 3.5),
            stats.uniform(1700, 800),
            stats.uniform(0.025, 0.055),
        ]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )


class EvtushenkoObjective(Model):
    r"""
    Objective of the constrained optimization benchmark from Evtushenko.

    The objective function is defined as:

    .. math::
        f(z) = (z_1 + 3z_2 + z_3)^2 + 4(z_1 - z_2)^2

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the objective model.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.
        """
        super().__init__(backend)

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def apply_hessian_implemented(self) -> bool:
        """
        Check if the Hessian application is implemented.

        Returns
        -------
        apply_hessian_implemented : bool
            True if the Hessian application is implemented, False otherwise.
        """
        return True

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this model, it is 1.
        """
        return 1

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars : int
            Number of uncertain variables. For this model, it is 3.
        """
        return 3

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the objective function for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        vals : Array
            Array of shape (nsamples, 1) containing the objective function
            evaluations.
        """
        return (
            (samples[0] + 3 * samples[1] + samples[2]) ** 2
            + 4 * (samples[0] - samples[1]) ** 2
        )[:, None]

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the gradient of the objective function at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac : Array
            Array of shape (1, nvars) containing the gradient of the objective
            function.
        """
        return self._bkd.stack(
            (
                2 * (sample[0] + 3 * sample[1] + sample[2])
                + 8 * (sample[0] - sample[1]),
                6 * (sample[0] + 3 * sample[1] + sample[2])
                - 8 * (sample[0] - sample[1]),
                2 * (sample[0] + 3 * sample[1] + sample[2]),
            ),
            axis=1,
        )

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Hessian of the objective function to a vector.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.
        vec : Array
            Array of shape (nvars, 1) containing the vector to which the
            Hessian is applied.

        Returns
        -------
        hvp : Array
            Array of shape (nvars, 1) containing the result of the Hessian
            application.
        """
        return self._bkd.stack(
            (
                10 * vec[0] - 2 * vec[1] + 2 * vec[2],
                -2 * vec[0] + 26 * vec[1] + 6 * vec[2],
                2 * vec[0] + 6 * vec[1] + 2 * vec[2],
            ),
            axis=0,
        )


class EvtushenkoNonLinearConstraint(Constraint):
    r"""
    Nonlinear constraint of the constrained optimization benchmark from
    Evtushenko.

    The constraint is defined as:

    .. math::
        c(z) = 6z_2 + 4z_3 - z_1^3 - 3

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the nonlinear constraint.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.
        """
        super().__init__(backend.array([[0.0, np.inf]]), True, backend)

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def apply_weighted_hessian_implemented(self) -> bool:
        """
        Check if the weighted Hessian application is implemented.

        Returns
        -------
        apply_weighted_hessian_implemented : bool
            True if the weighted Hessian application is implemented, False
            otherwise.
        """
        return True

    def weighted_hessian_implemented(self) -> bool:
        """
        Check if the weighted Hessian is implemented.

        Returns
        -------
        weighted_hessian_implemented : bool
            True if the weighted Hessian is implemented, False otherwise.
        """
        return True

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this constraint, it is 1.
        """
        return 1

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars : int
            Number of uncertain variables. For this constraint, it is 3.
        """
        return 3

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the constraint for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        vals : Array
            Array of shape (nsamples, 1) containing the constraint evaluations.
        """
        return (6 * samples[1] + 4 * samples[2] - samples[0] ** 3 - 3)[:, None]

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the gradient of the constraint at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac : Array
            Array of shape (1, nvars) containing the gradient of the constraint.
        """
        return self._bkd.stack(
            (
                -3.0 * sample[0] ** 2,
                self._bkd.array([6.0]),
                self._bkd.array([4.0]),
            ),
            axis=1,
        )

    def _apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        Apply the weighted Hessian of the constraint to a vector.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.
        vec : Array
            Array of shape (nvars, 1) containing the vector to which the
            weighted Hessian is applied.
        weights : Array
            Array of shape (1, 1) containing the weights.

        Returns
        -------
        hvp: Array
            Array of shape (nvars, 1) containing the result of the weighted
            Hessian application.
        """
        return self._bkd.hstack(
            [-6 * sample[0] * vec[0] * weights[0], self._bkd.zeros((2,))]
        )[:, None]

    def _weighted_hessian(self, sample: Array, weights: Array) -> Array:
        """
        Compute the weighted Hessian of the constraint.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.
        weights : Array
            Array of shape (1, 1) containing the weights.

        Returns
        -------
        weighted_hess : Array
            Array of shape (nvars, nvars) containing the weighted Hessian matrix.
        """
        hess = self._bkd.zeros((3, 3))
        hess[0, 0] = -6 * sample[0, 0] * weights[0, 0]
        return hess


class EvtushenkoConstrainedOptimizationBenchmark(
    ConstrainedOptimizationBenchmark
):
    """
    Evtushenko constrained optimization benchmark.

    This class implements the constrained optimization benchmark from
    Evtushenko, including both nonlinear and linear constraints.

    The objective function is defined as:

    .. math::
        f(z) = (z_1 + 3z_2 + z_3)^2 + 4(z_1 - z_2)^2

    The nonlinear constraint is defined as:

    .. math::
        c(z) = 6z_2 + 4z_3 - z_1^3 - 3

    The linear constraint is defined as:

    .. math::
        k(z) = z_1 + z_2 + z_3  = 1
    """

    def _set_objective(self):
        """
        Set the objective function for the optimization problem.

        Returns
        -------
        None
        """
        self._objective = EvtushenkoObjective(self._bkd)

    def _set_constraints(self):
        """
        Set the constraints for the optimization problem.

        Returns
        -------
        None
        """
        nonlinear_con = EvtushenkoNonLinearConstraint(self._bkd)
        linear_con = LinearConstraint(
            self._bkd.ones((1, 3)), 1, 1, keep_feasible=True
        )
        self._constraints = [nonlinear_con, linear_con]

    def design_variable(self) -> DesignVariable:
        """
        Define the design variable for the optimization problem.

        Returns
        -------
        design_variable : DesignVariable
            The design variable with bounds.
        """
        design_bounds = self._bkd.stack(
            [self._bkd.zeros((3,)), self._bkd.full((3,), np.inf)], axis=1
        )
        return DesignVariable(design_bounds)

    def init_iterate(self) -> Array:
        """
        Return the initial iterate for the optimization problem.

        Returns
        -------
        init_iterate : Array
            Array of shape (3, 1) containing the initial iterate.
        """
        return self._bkd.array([0.1, 0.7, 0.2])[:, None]

    def optimal_iterate(self) -> Array:
        """
        Return the optimal iterate for the optimization problem.

        Returns
        -------
        optimal_iterate : Array
            Array of shape (3, 1) containing the optimal iterate.
        """
        return self._bkd.array([0.0, 0.0, 1.0])[:, None]


class MichaelisMentenModel(SingleSampleModel):
    r"""
    Michaelis-Menten model.

    This class implements the Michaelis-Menten model, which is widely used in
    enzyme kinetics to describe the rate of enzymatic reactions. The model is
    defined as:

    .. math::
        v = \frac{\theta_1 \cdot [S]}{\theta_2 + [S]}

    where:
    - :math:`v` is the reaction rate.
    - :math:`\theta_1` is the maximum reaction rate.
    - :math:`\theta_2` is the Michaelis constant.
    - :math:`[S]` is the substrate concentration.

    Parameters
    ----------
    mesh : Array
        2D array with one row specifying the substrate concentrations :math:`[S]`.
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, mesh: Array, backend: BackendMixin):
        """
        Initialize the Michaelis-Menten model.

        Parameters
        ----------
        mesh : Array
            2D array with one row specifying the substrate concentrations.
        backend : BackendMixin
            Backend for numerical computations.

        Raises
        ------
        ValueError
            If `mesh` is not a 2D array with one row.
        """
        super().__init__(backend)
        if mesh.ndim != 2 or mesh.shape[0] != 1:
            raise ValueError("mesh must be 2D array with one row")
        self._mesh = mesh

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI, which corresponds to the number of substrate
            concentrations in the mesh.
        """
        return self._mesh.shape[1]

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars : int
            Number of uncertain variables. For this model, it is 2
        """
        return 2

    def _evaluate(self, sample: Array) -> Array:
        """
        Evaluate the Michaelis-Menten model for a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample

        Returns
        -------
        vals : Array
            Array of shape (1, nqoi) containing the reaction rates for each
            substrate concentration.
        """
        theta_1, theta_2 = sample
        return (theta_1 * self._mesh[0] / (theta_2 + self._mesh[0]))[None, :]

    def _jacobian(self, sample: Array) -> Array:
        r"""
        Compute the Jacobian of the Michaelis-Menten model at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample

        Returns
        -------
        jac : Array
            Array of shape (nqoi, nvars) containing the Jacobian matrix.
        """
        theta_1, theta_2 = sample
        return self._bkd.stack(
            (
                self._mesh[0] / (theta_2 + self._mesh[0]),
                -theta_1 * self._mesh[0] / (theta_2 + self._mesh[0]) ** 2,
            ),
            axis=1,
        )
