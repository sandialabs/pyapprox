import math
from typing import List, Union

import numpy as np
from scipy import stats
from scipy.optimize import rosen, rosen_der, rosen_hess_prod, LinearConstraint

from pyapprox.interface.model import (
    Model,
    ActiveSetVariableModel,
    SingleSampleModel,
    ChangeModelSignWrapper,
)
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
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
from pyapprox.util.utilities import evaluate_quadratic_form
from pyapprox.optimization.pya_minimize import (
    SampleAverageConstraint,
    SampleAverageMeanPlusStdev,
    ConstraintFromModel,
    Constraint,
)


class IshigamiModel(Model):
    r"""
    Ishigami function

    .. math:: f(z) = \sin(z_1)+a\sin^2(z_2) + bz_3^4\sin(z_0)
    """

    def __init__(
        self,
        a: float = 7,
        b: float = 0.1,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._a = a
        self._b = b
        super().__init__(backend)
        self._jacobian_implemented = True
        self._hessian_implemented = True

    def nqoi(self) -> int:
        return 1

    def _values(self, samples: Array) -> Array:
        return (
            self._bkd.sin(samples[0, :])
            + self._a * self._bkd.sin(samples[1, :]) ** 2
            + self._b * samples[2, :] ** 4 * self._bkd.sin(samples[0, :])
        )[:, None]

    def nvars(self):
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
        hess[0, 0] = -self._bkd.sin(sample[0, :]) - self._b * sample[
            2, :
        ] ** 4 * self._bkd.sin(sample[0, :])
        hess[1, 1] = (
            2
            * self._a
            * (
                self._bkd.cos(sample[1, :]) ** 2
                - self._bkd.sin(sample[1, :]) ** 2
            )
        )
        hess[2, 2] = (
            12 * self._b * sample[2, :] ** 2 * self._bkd.sin(sample[0, :])
        )
        hess[0, 1], hess[1, 0] = 0, 0
        hess[0, 2] = (
            4 * self._b * sample[2, :] ** 3 * self._bkd.cos(sample[0, :])
        )
        hess[2, 0] = hess[0, 2]
        hess[1, 2], hess[2, 1] = 0, 0
        return hess[None, ...]


class IshigamiBenchmark(SingleModelBenchmark):
    r"""
    Ishigami function benchmark

    .. math:: f(z) = \sin(z_1)+a\sin^2(z_2) + bz_3^4\sin(z_0)

    References
    ----------
    .. [Ishigami1990] `T. Ishigami and T. Homma, "An importance quantification technique in uncertainty analysis for computer models," [1990] Proceedings. First International Symposium on Uncertainty Modeling and Analysis, College Park, MD, USA, 1990, pp. 398-403 <https://doi.org/10.1109/ISUMA.1990.151285>`_
    """

    def __init__(
        self,
        a: float = 7,
        b: float = 0.1,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._a = a
        self._b = b
        super().__init__(backend)

    def _set_model(self):
        self._model = IshigamiModel(self._a, self._b, self._bkd)

    def _set_variable(self) -> JointVariable:
        """
        p_i(X_i) ~ U[-pi,pi], i=1,...3
        """
        marginals = [stats.uniform(-np.pi, 2 * np.pi)] * 3
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def mean(self) -> float:
        return self._a / 2 * self._bkd.ones((1,))[0]

    def variance(self) -> float:
        return (
            self._a**2 / 8
            + self._b * np.pi**4 / 5
            + self._b**2 * np.pi**8 / 18
            + 0.5
        ) * self._bkd.ones((1,))[0]

    def _unnormalized_sobol_indices(self):
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
        return (
            self._bkd.hstack(self._unnormalized_sobol_indices()[:3])
            / self.variance()
        )

    def total_effects(self) -> Array:
        D_1, D_2, D_3, D_12, D_13, D_23, D_123 = (
            self._unnormalized_sobol_indices()
        )
        # the following two ways of calulating the total effects are equivalent
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
        return total_effects

    def sobol_indices(self) -> Array:
        sobol_indices = (
            self._bkd.hstack(self._unnormalized_sobol_indices())
            / self.variance()
        )
        sobol_interaction_indices = [
            [0],
            [1],
            [2],
            [0, 1],
            [0, 2],
            [1, 2],
            [0, 1, 2],
        ]
        return sobol_indices[:, None], sobol_interaction_indices


def get_oakley_function_data(bkd=NumpyLinAlgMixin) -> Array:
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

    Examples
    --------

    >>> from pyapprox.benchmarks.sensitivity_benchmarks import get_oakley_function_data
    >>> a1,a2,a3,M=get_oakley_function_data()
    >>> print(a1)
    [0.0118 0.0456 0.2297 0.0393 0.1177 0.3865 0.3897 0.6061 0.6159 0.4005
     1.0741 1.1474 0.788  1.1242 1.1982]
    >>> print(a2)
    [0.4341 0.0887 0.0512 0.3233 0.1489 1.036  0.9892 0.9672 0.8977 0.8083
     1.8426 2.4712 2.3946 2.0045 2.2621]
    >>> print(a3)
    [0.1044 0.2057 0.0774 0.273  0.1253 0.7526 0.857  1.0331 0.8388 0.797
     2.2145 2.0382 2.4004 2.0541 1.9845]
    >>> print(M)
    [[-0.02248289 -0.18501666  0.13418263  0.36867264  0.17172785  0.13651143
      -0.44034404 -0.08142285  0.71321025 -0.44361072  0.50383394 -0.02410146
      -0.04593968  0.21666181  0.05588742]
     [ 0.2565963   0.05379229  0.25800381  0.23795905 -0.59125756 -0.08162708
      -0.28749073  0.41581639  0.49752241  0.08389317 -0.11056683  0.03322235
      -0.13979497 -0.03102056 -0.22318721]
     [-0.05599981  0.19542252  0.09552901 -0.2862653  -0.14441303  0.22369356
       0.14527412  0.28998481  0.2310501  -0.31929879 -0.29039128 -0.20956898
       0.43139047  0.02442915  0.04490441]
     [ 0.66448103  0.43069872  0.29924645 -0.16202441 -0.31479544 -0.39026802
       0.17679822  0.05795266  0.17230342  0.13466011 -0.3527524   0.25146896
      -0.01881053  0.36482392 -0.32504618]
     [-0.121278    0.12463327  0.10656519  0.0465623  -0.21678617  0.19492172
      -0.06552113  0.02440467 -0.09682886  0.19366196  0.33354757  0.31295994
      -0.08361546 -0.25342082  0.37325717]
     [-0.2837623  -0.32820154 -0.10496068 -0.22073452 -0.13708154 -0.14426375
      -0.11503319  0.22424151 -0.03039502 -0.51505615  0.01725498  0.03895712
       0.36069184  0.30902452  0.05003019]
     [-0.07787589  0.00374566  0.88685604 -0.26590028 -0.07932536 -0.04273492
      -0.18653782 -0.35604718 -0.17497421  0.08869996  0.40025886 -0.05597969
       0.13724479  0.21485613 -0.0112658 ]
     [-0.09229473  0.59209563  0.03133829 -0.03308086 -0.24308858 -0.09979855
       0.03446019  0.09511981 -0.3380162   0.006386   -0.61207299  0.08132542
       0.88683114  0.14254905  0.14776204]
     [-0.13189434  0.52878496  0.12652391  0.04511362  0.58373514  0.37291503
       0.11395325 -0.29479222 -0.57014085  0.46291592 -0.09405018  0.13959097
      -0.38607402 -0.4489706  -0.14602419]
     [ 0.05810766 -0.32289338  0.09313916  0.07242723 -0.56919401  0.52554237
       0.23656926 -0.01178202  0.0718206   0.07827729 -0.13355752  0.22722721
       0.14369455 -0.45198935 -0.55574794]
     [ 0.66145875  0.34633299  0.14098019  0.51882591 -0.28019898 -0.1603226
      -0.06841334 -0.20428242  0.06967217  0.23112577 -0.04436858 -0.16455425
       0.21620977  0.00427021 -0.08739901]
     [ 0.31599556 -0.02755186  0.13434254  0.13497371  0.05400568 -0.17374789
       0.17525393  0.06025893 -0.17914162 -0.31056619 -0.25358691  0.02584754
      -0.43006001 -0.62266361 -0.03399688]
     [-0.29038151  0.03410127  0.03490341 -0.12121764  0.02603071 -0.33546274
      -0.41424111  0.05324838 -0.27099455 -0.0262513   0.41024137  0.26636349
       0.15582891 -0.18666254  0.01989583]
     [-0.24388652 -0.44098852  0.01261883  0.24945112  0.07110189  0.24623792
       0.17484502  0.00852868  0.2514707  -0.14659862 -0.08462515  0.36931333
      -0.29955293  0.1104436  -0.75690139]
     [ 0.04149432 -0.25980564  0.46402128 -0.36112127 -0.94980789 -0.16504063
       0.00309433  0.05279294  0.22523648  0.38390366  0.45562427 -0.18631744
       0.0082334   0.16670803  0.16045688]]
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
    Oakely sensivity benchmark model

    .. math:: f(z) = a_1^Tz + a_2^T\sin(z) + a_3^T\cos(z) + z^TMz

    where :math:`z` consists of 15 I.I.D. standard Normal variables and
    the data :math:`a_1,a_2,a_3` and :math:`~M` are defined in the function
    :py:func:`~pyapprox.benchmarks.sensitivity_benchmarks.get_oakley_function_data`.
    """

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        15

    def _values(self, samples: Array) -> Array:
        a1, a2, a3, M = get_oakley_function_data(self._bkd)
        term1, term2 = a1 @ (samples), a2 @ (self._bkd.sin(samples))
        term3 = a3 @ self._bkd.cos(samples)
        term4 = evaluate_quadratic_form(M, samples, self._bkd)
        vals = term1 + term2 + term3 + term4
        return vals[:, None]


class OakleyBenchmark(SingleModelBenchmark):
    r"""
    Oakely sensivity benchmark

    .. math:: f(z) = a_1^Tz + a_2^T\sin(z) + a_3^T\cos(z) + z^TMz

    where :math:`z` consists of 15 I.I.D. standard Normal variables and
    the data :math:`a_1,a_2,a_3` and :math:`~M` are defined in the function.

    References
    ----------
    .. [OakelyOJRSB2004] `Oakley, J.E. and O'Hagan, A. (2004), Probabilistic sensitivity analysis of complex models: a Bayesian approach. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 66: 751-769. <https://doi.org/10.1111/j.1467-9868.2004.05304.x>`_
    """

    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._set_statistics()

    def _variance_linear_combination_of_indendent_variables(
        self, coef: Array, variances: Array
    ) -> Array:
        assert coef.shape[0] == variances.shape[0]
        return self._bkd.sum(coef**2 * variances)

    def _set_model(self):
        self._model = OakleyModel(self._bkd)

    def _set_variable(self):
        marginals = [stats.norm()] * 15
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_statistics(self):
        e = math.exp(1)
        a1, a2, a3, M = get_oakley_function_data(self._bkd)
        nvars = M.shape[0]

        term1_mean, term2_mean = 0, 0
        term3_mean, term4_mean = self._bkd.sum(
            a3 / math.sqrt(e)
        ), self._bkd.trace(M)
        self._mean = term1_mean + term2_mean + term3_mean + term4_mean

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
        A = 0.5 * (M.T + M)  # needed because M is not symmetric
        term4_var = 2 * self._bkd.trace(A @ A)

        cov_xsinx = 1 / math.sqrt(e)
        covar13, covar14, covar23, covar24 = 0, 0, 0, 0
        covar12 = self._bkd.sum(a1 * a2 * cov_xsinx)
        covar34 = self._bkd.sum(-1 / math.sqrt(e) * a3 * self._bkd.diag(M))

        self._variance = term1_var + term2_var + term3_var + term4_var
        self._variance += 2 * (
            covar12 + covar13 + covar14 + covar23 + covar24 + covar34
        )
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

    def mean(self) -> float:
        return self._mean

    def variance(self) -> float:
        return self._variance

    def main_effects(self) -> Array:
        return self._main_effects


class SobolGModel(Model):
    r"""
    Sobol-G function benchmark model

    .. math:: f(z) = \prod_{i=1}^d\frac{\lvert 4z_i-2\rvert+a_i}{1+a_i}, \quad a_i=\frac{i-2}{2}
    """

    def __init__(self, nvars, backend=NumpyLinAlgMixin):
        self._nvars = nvars
        super().__init__(backend)
        self._acoefs = (self._bkd.arange(1, self._nvars + 1) - 2) / 2

    def nqoi(self) -> int:
        return 1

    def _values(self, samples: Array) -> Array:
        """
        The coefficients control the sensitivity of each variable. Specifically
        they limit the range of the outputs, i.e.
        1-1/(1+a_i) <= (abs(4*x-2)+a_i)/(a_i+1) <= 1-1/(1+a_i)
        """
        vals = self._bkd.prod(
            (self._bkd.abs(4 * samples - 2) + self._acoefs[:, None])
            / (1 + self._acoefs[:, None]),
            axis=0,
        )[:, None]
        return vals


class SobolGBenchmark(SingleModelBenchmark):
    r"""
    Sobol-G function benchmark

    .. math:: f(z) = \prod_{i=1}^d\frac{\lvert 4z_i-2\rvert+a_i}{1+a_i}, \quad a_i=\frac{i-2}{2}

     References
    ----------
    .. [Saltelli1995] `Saltelli, A., & Sobol, I. M. About the use of rank transformation in sensitivity analysis of model output. Reliability Engineering & System Safety, 50(3), 225-239, 1995. <https://doi.org/10.1016/0951-8320(95)00099-2>`_
    """

    def __init__(self, nvars=5, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._nvars = nvars
        super().__init__(backend)
        self._acoefs = (
            self._bkd.arange(
                1, self.nvars() + 1, dtype=self._bkd.double_type()
            )
            - 2
        ) / 2
        self._set_statistics()

    def _set_statistics(self):
        """
        See article: Variance based sensitivity analysis of model output.
        Design and estimator for the total sensitivity index
        """
        self._interaction_indices = [[ii] for ii in range(self.nvars())]
        # get all single and pairwise interactions
        for ii in range(self.nvars()):
            for jj in range(ii + 1, self.nvars()):
                self._interaction_indices.append([ii, jj])
        zero = self._acoefs[0] * 0.0  # necessary for torch
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
        # if interaction_terms is None:
        #     return mean, variance, main_effects, total_effects

        self._sobol_indices = self._bkd.array(
            [
                unnormalized_main_effects[index].prod() / self._variance
                for index in self._interaction_indices
            ]
        )

    def _set_model(self):
        self._model = SobolGModel(self._nvars, self._bkd)

    def _set_variable(self):
        marginals = [stats.uniform(0, 1)] * self._nvars
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def mean(self) -> float:
        return self._mean

    def variance(self) -> float:
        return self._variance

    def main_effects(self) -> Array:
        return self._main_effects

    def total_effects(self) -> Array:
        return self._total_effects

    def sobol_indices(self) -> Array:
        return self._sobol_indices


class RosenbrockModel(Model):
    r"""The Rosenbrock function

    .. math:: f(z) = \sum_{i=1}^{d/2}\left[100(z_{2i-1}^{2}-z_{2i})^{2}+(z_{2i-1}-1)^{2}\right]
    """

    def __init__(self, nvars: int, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._jacobian_implemented = True
        self._apply_hessian_implemented = True

    def nqoi(self) -> int:
        return 1

    def _values(self, samples: Array) -> Array:
        # note torch back prop wont work this function because rosen is a
        # numpy function
        return self._bkd.asarray(rosen(self._bkd.to_numpy(samples)))[:, None]

    def nvars(self):
        return self._nvars

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.asarray(rosen_der(self._bkd.to_numpy(sample))).T

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return self._bkd.asarray(
            rosen_hess_prod(
                self._bkd.to_numpy(sample[:, 0]), self._bkd.to_numpy(vec[:, 0])
            )
        )[:, None]


class RosenbrockUnconstrainedOptimizationBenchmark(OptimizationBenchmark):
    r"""The Rosenbrock function benchmark

    .. math:: f(z) = \sum_{i=1}^{d/2}\left[100(z_{2i-1}^{2}-z_{2i})^{2}+(z_{2i-1}-1)^{2}\right]

    References
    ----------
    .. [DixonSzego1990] `Dixon, L. C. W.; Mills, D. J. "Effect of Rounding Errors on the Variable Metric Method". Journal of Optimization Theory and Applications. 80: 175–179. 1994 <https://doi.org/10.1007%2FBF02196600>`_
    """

    def __init__(
        self,
        nvars: int,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._nvars = nvars
        super().__init__(backend)

    def objective(self) -> Model:
        return RosenbrockModel(self._nvars, self._bkd)

    def design_variable(self) -> DesignVariable:
        design_bounds = self._bkd.full((self._nvars, 2), -2.0)
        design_bounds[:, 1] = 2.0
        return DesignVariable(design_bounds)

    def variable(self) -> IndependentMarginalsVariable:
        marginals = [stats.uniform(-2, 4)] * self._nvars
        return IndependentMarginalsVariable(marginals, backend=self._bkd)

    def mean(self) -> float:
        """
        Mean of rosenbrock function with uniform variables in [-2,2]^d
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
        return exact_mean * self._bkd.ones((1,))[0]

    def optimal_iterate(self) -> Array:
        return self._bkd.ones((self._nvars, 1))

    def init_iterate(self) -> Array:
        return self._bkd.ones((self._nvars, 1)) + 0.5


class RosenbrockConstraint(Constraint):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend.array([[0.0, np.inf]]), True, backend)
        self._jacobian_implemented = True
        self._hessian_implemented = True

    def nqoi(self) -> int:
        return 2

    def _values(self, samples: Array) -> Array:
        return self._bkd.stack(
            [
                -((samples[0] - 1) ** 3) + samples[1] - 1,
                2 - samples[0] - samples[1],
            ],
            axis=1,
        )

    def _jacobian(self, sample: Array) -> Array:

        return self._bkd.stack(
            (
                self._bkd.hstack(
                    [-3 * (sample[0] - 1) ** 2, self._bkd.ones((1,))]
                ),
                self._bkd.full((2,), -1),
            ),
            axis=0,
        )

    def _hessian(self, sample: Array) -> Array:
        zero = self._bkd.zeros((1,))
        hess = self._bkd.zeros((self.nqoi(), 2, 2))
        hess[0] = self._bkd.stack(
            (
                self._bkd.hstack((-6 * (sample[0] - 1), zero)),
                self._bkd.zeros((2,)),
            ),
            axis=0,
        )
        return hess


class RosenbrockConstrainedOptimizationBenchmark(
    ConstrainedOptimizationBenchmark
):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._nvars = 2
        super().__init__(backend)

    def optimal_iterate(self) -> Array:
        return self._bkd.ones((self._nvars, 1))

    def init_iterate(self) -> Array:
        eps = 0.2 # initial iterate on constraint
        # need to shift x coord because of issues with scipy trust-constr solver
        return self._bkd.array([1-eps-1e-16, 1+eps])[:, None]

    def objective(self) -> Model:
        return RosenbrockModel(2, self._bkd)

    def design_variable(self) -> DesignVariable:
        design_bounds = self._bkd.full((self._nvars, 2), -2.0)
        design_bounds[:, 1] = 2.0
        return DesignVariable(design_bounds)

    def constraints(self) -> List[Union[Constraint, LinearConstraint]]:
        return [RosenbrockConstraint(self._bkd)]


class CantileverBeamModel(SingleSampleModel):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        import sympy as sp

        super().__init__(backend)
        self._jacobian_implemented = True
        self._hessian_implemented = True
        self._apply_jacobian_implemented = True
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

    def nqoi(self):
        return 2

    def _evaluate_sp_lambda(self, sp_lambda, sample):
        assert sample.ndim == 2 and sample.shape[1] == 1
        vals = self._bkd.atleast2d(sp_lambda(*sample[:, 0]))
        return vals

    def _evaluate(self, sample):
        return self._evaluate_sp_lambda(self._lam_fun, sample)

    def _jacobian(self, sample):
        return self._evaluate_sp_lambda(self._lam_jac, sample)

    def _apply_jacobian(self, sample, vec):
        return self.jacobian(sample) @ vec

    def _hessian(self, sample):
        return self._evaluate_sp_lambda(self._lam_hess, sample)


class CantileverBeamObjectiveModel(CantileverBeamModel):
    def nqoi(self) -> int:
        return 1

    def _evaluate(self, sample):
        return super()._evaluate(sample)[:, :1]

    def _jacobian(self, sample):
        return super()._jacobian(sample)[:1, :]

    def _hessian(self, sample):
        return self._bkd.copy(super()._hessian(sample)[:1, ...])


class CantileverBeamConstraintsModel(CantileverBeamModel):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)

    def nqoi(self) -> int:
        return 2

    def _evaluate(self, sample):
        return super()._evaluate(sample)[:, 1:]

    def _jacobian(self, sample):
        return super()._jacobian(sample)[1:, :]

    def _hessian(self, sample):
        return self._bkd.copy(super()._hessian(sample)[1:, ...])


class CantileverBeamDeterminsticOptimizationBenchmark(
    ConstrainedUncertainOptimizationBenchmark
):
    def objective(self) -> Model:
        return ActiveSetVariableModel(
            CantileverBeamObjectiveModel(self._bkd),
            self.variable().num_vars() + self.design_variable().nvars(),
            self._nominal_values,
            self._design_var_indices,
        )

    def constraints(self) -> List[Union[Constraint, LinearConstraint]]:
        constraint_model = ActiveSetVariableModel(
            CantileverBeamConstraintsModel(self._bkd),
            self.variable().num_vars() + self.design_variable().nvars(),
            self._nominal_values,
            self._design_var_indices,
        )
        return [
            ConstraintFromModel(constraint_model, self.constraint_bounds())
        ]

    def variable(self) -> JointVariable:
        # traditional parameterization
        X = stats.norm(loc=500, scale=np.sqrt(100) ** 2)
        Y = stats.norm(loc=1000, scale=np.sqrt(100) ** 2)
        E = stats.norm(loc=2.9e7, scale=np.sqrt(1.45e6) ** 2)
        R = stats.norm(loc=40000, scale=np.sqrt(2000) ** 2)
        self._variable = IndependentMarginalsVariable(
            [X, Y, E, R], backend=self._bkd
        )
        self._nominal_values = self._variable.get_statistics("mean")
        return self._variable

    def design_variable(self) -> DesignVariable:
        design_bounds = self._bkd.stack(
            [self._bkd.ones((2,)), self._bkd.full((2,), 4.0)], axis=1
        )
        self._design_variable = DesignVariable(design_bounds)
        self._design_var_indices = self._bkd.array([4, 5], dtype=int)
        return self._design_variable

    def constraint_bounds(self) -> Array:
        return self._bkd.hstack(
            [self._bkd.zeros((2, 1)), self._bkd.full((2, 1), np.inf)]
        )

    def design_var_indices(self) -> Array:
        return self._design_var_indices

    def init_iterate(self) -> Array:
        return self._bkd.array([3.0, 3.0])[:, None]

    def optimal_iterate(self) -> Array:
        # Optimal design vars from literature
        return self._bkd.array([2.35, 3.33])[:, None]


class CantileverBeamUncertainOptimizationBenchmark(
    CantileverBeamDeterminsticOptimizationBenchmark
):
    def constraints(self) -> List[Union[Constraint, LinearConstraint]]:
        # TODO change weights to create unbiased estimators of mean and variance
        from pyapprox.surrogates.bases.basis import (
            FixedTensorProductQuadratureRule,
        )
        from pyapprox.surrogates.bases.orthopoly import GaussQuadratureRule

        quad_rule = FixedTensorProductQuadratureRule(
            self.variable().num_vars(),
            [
                GaussQuadratureRule(marginal, backend=self._bkd)
                for marginal in self.variable().marginals()
            ],
            [5 for ii in range(self.variable().num_vars())],
        )
        samples, weights = quad_rule()
        constraint_model = ChangeModelSignWrapper(
            CantileverBeamConstraintsModel(self._bkd)
        )
        stat = SampleAverageMeanPlusStdev(3)
        return [
            SampleAverageConstraint(
                constraint_model,
                samples,
                weights,
                stat,
                self.constraint_bounds(),
                self.variable().num_vars() + self.design_variable().nvars(),
                self._design_var_indices,
            )
        ]

    def constraint_bounds(self) -> Array:
        return self._bkd.hstack(
            [self._bkd.full((2, 1), -np.inf), self._bkd.zeros((2, 1))]
        )

    def optimal_iterate(self) -> Array:
        # Optimal design vars from literature are not accurate enough
        raise NotImplementedError


class PistonModel(Model):
    """Predict cycle time of a piston in a cylinder"""

    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._jacobian_implemented = True

    def nqoi(self) -> int:
        return 1

    def _values(self, samples: Array) -> Array:
        M, S, V_0, k, P_0, T_a, T_0 = samples
        Z = P_0 * V_0 / T_0 * T_a
        A = P_0 * S + 19.62 * M - k * V_0 / S
        V = S / (2.0 * k) * (self._bkd.sqrt(A**2 + 4.0 * k * Z) - A)
        C = 2.0 * np.pi * self._bkd.sqrt(M / (k + S**2 * Z / V**2))
        return C[:, None]

    def _jacobian(self, sample: Array) -> Array:
        """Gradient of cycle time of a piston in a cylinder"""
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
    def _set_model(self):
        self._model = PistonModel(self._bkd)

    def _set_variable(self):
        """
        M piston mass (kg)
        S surface area (m^2)
        V_0 initial gas volume (m^3)
        k spring coefficient (N/m)
        P_0 atmospheric pressure (N/m^2)
        T_a ambient temperature (K)
        T_0 filling gas temperature (K)
        """
        M = stats.uniform(loc=30.0, scale=30.0)
        S = stats.uniform(loc=0.005, scale=0.015)
        V_0 = stats.uniform(loc=0.002, scale=0.008)
        k = stats.uniform(loc=1000.0, scale=4000.0)
        P_0 = stats.uniform(loc=90000.0, scale=20000.0)
        T_a = stats.uniform(loc=290.0, scale=6.0)
        T_0 = stats.uniform(loc=340.0, scale=20.0)

        self._variable = IndependentMarginalsVariable(
            [M, S, V_0, k, P_0, T_a, T_0], backend=self._bkd
        )


class WingWeightModel(Model):
    """Weight of a light aircraft wing"""

    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._jacobian_implemented = True

    def nqoi(self) -> int:
        return 1

    def _values(self, samples: Array) -> Array:
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
    def _set_model(self):
        self._model = WingWeightModel(self._bkd)

    def _set_variable(self):
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
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )


class EvtushenkoObjective(Model):
    r"""Objective of the constrained optimization benchmark from Evtushenko.

    The objective is
    .. math:: f(z) = (z_1+3z_2+z_3)^2 +4(z_1-z_2)^2
    """

    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._jacobian_implemented = True
        self._apply_hessian_implemented = True

    def nqoi(self) -> int:
        return 1

    def _values(self, samples: Array) -> Array:
        return (
            (samples[0] + 3 * samples[1] + samples[2]) ** 2
            + 4 * (samples[0] - samples[1]) ** 2
        )[:, None]

    def nvars(self):
        return 3

    def _jacobian(self, sample: Array) -> Array:
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
    The nonlinear Constraint of the constrained optimization benchmark from
    Evtushenko.

    The constraints are
    .. math:: c(z) = (z_1+3z_2+z_3)^2 +4(z_1-z_2)^2
    """

    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend.array([[0.0, np.inf]]), True, backend)
        self._jacobian_implemented = True
        self._apply_weighted_hessian_implemented = True

    def nqoi(self) -> int:
        return 1

    def _values(self, samples: Array) -> Array:
        return (6 * samples[1] + 4 * samples[2] - samples[0] ** 3 - 3)[:, None]

    def _jacobian(self, sample: Array) -> Array:
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
        return self._bkd.hstack(
            [-6 * sample[0] * vec[0] * weights[0], 0.0, 0.0]
        )[:, None]


class EvtushenkoConstrainedOptimizationBenchmark(
    ConstrainedOptimizationBenchmark
):
    def objective(self) -> Model:
        return EvtushenkoObjective(self._bkd)

    def constraints(self) -> List[Union[Constraint, LinearConstraint]]:
        nonlinear_con = EvtushenkoNonLinearConstraint(self._bkd)
        linear_con = LinearConstraint(
            self._bkd.ones((1, 3)), 1, 1, keep_feasible=True
        )
        return [nonlinear_con, linear_con]

    def design_variable(self) -> DesignVariable:
        design_bounds = self._bkd.stack(
            [self._bkd.zeros((3,)), self._bkd.full((3,), np.inf)], axis=1
        )
        return DesignVariable(design_bounds)

    def init_iterate(self) -> Array:
        return self._bkd.array([0.1, 0.7, 0.2])[:, None]

    def optimal_iterate(self) -> Array:
        return self._bkd.array([0.0, 0.0, 1.0])[:, None]
