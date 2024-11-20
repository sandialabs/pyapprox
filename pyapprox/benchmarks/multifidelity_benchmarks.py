from abc import ABC, abstractmethod
from functools import partial
from typing import List
import math

from scipy import stats

from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    JointVariable,
)
from pyapprox.interface.model import ModelFromVectorizedCallable, Model
from pyapprox.surrogates.bases.orthopoly import GaussQuadratureRule
from pyapprox.surrogates.bases.basis import FixedTensorProductQuadratureRule
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class MultiModelBenchmark(ABC):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._bkd = backend
        self._set_variable()
        self._set_models()

    def nvars(self) -> int:
        """Return the number of uncertain variables."""
        return self._variable.num_vars()

    @abstractmethod
    def nmodels(self) -> int:
        """Return the number of models."""
        raise NotImplementedError

    def variable(self) -> JointVariable:
        """Return the random variable parameterizing the model uncertainty."""
        return self._variable

    @abstractmethod
    def nqoi(self) -> int:
        """Return the number of quantities of interest (QoI) of all models."""
        raise NotImplementedError

    @abstractmethod
    def _set_models(self):
        raise NotImplementedError

    def models(self) -> List[Model]:
        return self._models


class ACVBenchmark(MultiModelBenchmark):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._set_quadrature_samples_weights()
        self._flatten_funs()

    @abstractmethod
    def costs(self) -> Array:
        """Return the cost of each model."""
        raise NotImplementedError

    def _set_quadrature_samples_weights(self):
        univariate_quad_rules = [
            GaussQuadratureRule(variable)
            for variable in self.variable().marginals()
        ]
        quad_rule = FixedTensorProductQuadratureRule(
            self.nvars(), univariate_quad_rules, [21] * self.nvars()
        )
        self._quadx, self._quadw = quad_rule()

    def _flat_fun_wrapper(self, ii, jj, xx) -> Array:
        if xx.ndim != 2 or xx.shape[0] != self.nvars():
            raise RuntimeError("xx has the wrong shape")
        return self._models[ii](xx)[:, jj]

    def _flatten_funs(self):
        self._flat_funs = []
        for ii in range(self.nmodels()):
            for jj in range(self.nqoi()):
                self._flat_funs.append(partial(self._flat_fun_wrapper, ii, jj))

    def _mean(self) -> Array:
        # overide this function if you know the means exactly
        means = self._bkd.array(
            [f(self._quadx).dot(self._quadw) for f in self._flat_funs]
        )
        return self._bkd.reshape(means, (self.nmodels(), self.nqoi()))

    def mean(self) -> Array:
        """
        Return the means of the QoI of each model.

        Returns
        -------
        means : Array (nmodels, nqoi)
            The means of each model
        """
        means = self._mean()
        if (
            means.ndim != 2
            or means.shape[0] != self.nmodels()
            or means.shape[1] != self.nqoi()
        ):
            raise RuntimeError("_mean() not implemented correctly")
        return means

    def _covariance(self) -> Array:
        # overide this function if you know the means exactly
        means = [f(self._quadx).dot(self._quadw) for f in self._flat_funs]

        cov = self._bkd.empty(
            (self.nmodels() * self.nqoi(), self.nmodels() * self.nqoi())
        )
        ii = 0
        for fi, mi in zip(self._flat_funs, means):
            jj = 0
            for fj, mj in zip(self._flat_funs, means):
                cov[ii, jj] = (
                    ((fi(self._quadx) - mi) * (fj(self._quadx) - mj))
                    .dot(self._quadw)
                    .item()
                )
                jj += 1
            ii += 1
        return cov

    def covariance(self) -> Array:
        """
        The covariance between the qoi of each model

        Returns
        -------
        cov = Array (nmodels*nqoi, nmodels*nqoi)
            The covariance treating functions concatinating the qoi
            of each model f0, f1, f2
        """
        cov = self._covariance()
        if (
            cov.ndim != 2
            or cov.shape[0] != self.nmodels() * self.nqoi()
            or cov.shape[1] != self.nmodels() * self.nqoi()
        ):
            raise RuntimeError("_covariance() not implemented correctly")
        return cov

    def _V_fun_entry(self, jj, kk, ll, means, flat_covs, xx):
        idx1 = jj * self.nqoi() + kk
        idx2 = jj * self.nqoi() + ll
        return (self._flat_funs[idx1](xx) - means[idx1]) * (
            self._flat_funs[idx2](xx) - means[idx2]
        ) - flat_covs[jj][kk * self.nqoi() + ll]

    def _V_fun(self, jj1, kk1, ll1, jj2, kk2, ll2, means, flat_covs, xx):
        return self._V_fun_entry(
            jj1, kk1, ll1, means, flat_covs, xx
        ) * self._V_fun_entry(jj2, kk2, ll2, means, flat_covs, xx)

    def _B_fun(self, ii, jj, kk, ll, means, flat_covs, xx):
        return (self._flat_funs[ii](xx) - means[ii]) * self._V_fun_entry(
            jj, kk, ll, means, flat_covs, xx
        )

    def _flat_covs(self) -> List[Array]:
        cov = self.covariance()
        # store covariance only between the QoI of a model with QoI of the same
        # model
        flat_covs = []
        for ii in range(self.nmodels()):
            flat_covs.append([])
            for jj in range(self.nqoi()):
                for kk in range(self.nqoi()):
                    flat_covs[ii].append(
                        cov[ii * self.nqoi() + jj][ii * self.nqoi() + kk]
                    )
        return flat_covs

    def covariance_of_centered_values_kronker_product(self) -> Array:
        r"""
        The W matrix used to compute the covariance between the
        Kroneker product of centered (mean is subtracted off) values.

        Returns
        -------
        res : Array (nmodels*nqoi**2, nmodels*nqoi**2)
            The covariance :math:`Cov[(f_i-\mathbb{E}[f_i])^{\otimes^2}, (f_j-\mathbb{E}[f_j])^{\otimes^2}]`
        """
        means = self.mean().flatten()
        flat_covs = self._flat_covs()

        est_cov = self._bkd.empty(
            (
                self.nmodels() * self.nqoi() ** 2,
                self.nmodels() * self.nqoi() ** 2,
            )
        )
        cnt1 = 0
        for jj1 in range(self.nmodels()):
            for kk1 in range(self.nqoi()):
                for ll1 in range(self.nqoi()):
                    cnt2 = 0
                    for jj2 in range(self.nmodels()):
                        for kk2 in range(self.nqoi()):
                            for ll2 in range(self.nqoi()):
                                quad_cov = self._V_fun(
                                    jj1,
                                    kk1,
                                    ll1,
                                    jj2,
                                    kk2,
                                    ll2,
                                    means,
                                    flat_covs,
                                    self._quadx,
                                ).dot(self._quadw)
                                est_cov[cnt1, cnt2] = quad_cov.item()
                                cnt2 += 1
                    cnt1 += 1
        return self._bkd.array(est_cov)

    def covariance_of_mean_and_variance_estimators(self) -> Array:
        r"""
        The B matrix used to compute the covariance between mean and variance
        estimators.

        Returns
        -------
        res : Array (nmodels*nqoi, nmodels*nqoi**2)
            The covariance :math:`Cov[f_i, (f_j-\mathbb{E}[f_j])^{\otimes^2}]`
        """
        means = self.mean().flatten()
        flat_covs = self._flat_covs()
        est_cov = self._bkd.empty(
            (self.nmodels() * self.nqoi(), self.nmodels() * self.nqoi() ** 2)
        )
        for ii in range(len(self._flat_funs)):
            cnt = 0
            for jj in range(self.nmodels()):
                for kk in range(self.nqoi()):
                    for ll in range(self.nqoi()):
                        quad_cov = self._B_fun(
                            ii, jj, kk, ll, means, flat_covs, self._quadx
                        ).dot(self._quadw)
                        est_cov[ii, cnt] = quad_cov.item()
                        cnt += 1
        return self._bkd.array(est_cov)

    def __repr__(self):
        return "{0}(nmodels={1}, nqoi={2})".format(
            self.__class__.__name__, self.nmodels(), self.nqoi()
        )

    def __call__(self, samples):
        # samples must include model id in last row
        return self._model_ensemble(samples)


class PolynomialModelEnsemble(ACVBenchmark):
    r"""
    Return an ensemble of 5 univariate models of the form

    .. math:: f_\alpha(\rv)=\rv^{5-\alpha}, \quad \alpha=0,\ldots,4

    where :math:`z\sim\mathcal{U}[0, 1]`

    Returns
    -------
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`~pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    means : np.ndarray (nmodels)
        The mean of each model fidelity

    covariance : np.ndarray (nmodels)
        The covariance between the outputs of each model fidelity

    References
    ----------
    .. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification,  Journal of Computational Physics,  408:109257, 2020. <https://doi.org/10.1016/j.jcp.2020.109257>`_
    """

    def __init__(self, nmodels: int = 5, backend=NumpyLinAlgMixin):
        self._nmodels = nmodels
        super().__init__(backend)

    def nmodels(self) -> int:
        return self._nmodels

    def nqoi(self) -> int:
        return 1

    def costs(self) -> Array:
        return self._bkd.logspace(0, -self.nmodels(), self.nmodels())

    def m0(self, samples):
        return samples.T**5

    def m1(self, samples):
        return samples.T**4

    def m2(self, samples):
        return samples.T**3

    def m3(self, samples):
        return samples.T**2

    def m4(self, samples):
        return samples.T**1

    def _set_models(self):
        funs = [self.m0, self.m1, self.m2, self.m3, self.m4][: self.nmodels()]
        self._models = [
            ModelFromVectorizedCallable(self.nqoi(), fun, backend=self._bkd)
            for fun in funs
        ]

    def _mean(self):
        return 1 / self._bkd.arange(6, 1, -1)[: self.nmodels()].reshape(
            self.nmodels(), self.nqoi()
        )

    def _set_variable(self):
        univariate_variables = [stats.uniform(0, 1)]
        self._variable = IndependentMarginalsVariable(univariate_variables)


class TunableModelEnsemble(ACVBenchmark):

    def __init__(
        self,
        theta1: float,
        shifts: List[float] = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        """
        Parameters
        ----------
        theta1 : float
            Angle controling
        Notes
        -----
        The choice of A0, A1, A2 here results in unit variance for each model
        """
        self._theta1 = theta1
        if shifts is None:
            shifts = [0, 0]
        self._shifts = shifts
        super().__init__(backend)

    def nqoi(self) -> int:
        return 1

    def nmodels(self) -> int:
        return 3

    def m0(self, samples):
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
        self.A0 = math.sqrt(11)
        self.A1 = math.sqrt(7)
        self.A2 = math.sqrt(3)
        self._theta0 = math.pi / 2
        self._theta2 = math.pi / 6
        assert self._theta0 > self._theta1 and self._theta1 > self._theta2
        funs = [self.m0, self.m1, self.m2]
        self._models = [
            ModelFromVectorizedCallable(self.nqoi(), fun, backend=self._bkd)
            for fun in funs
        ]

    def _set_variable(self):
        univariate_variables = [stats.uniform(-1, 2), stats.uniform(-1, 2)]
        self._variable = IndependentMarginalsVariable(univariate_variables)

    def _covariance(self):
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
        return self._bkd.array([0, self._shifts[0], self._shifts[1]])[:, None]

    def get_kurtosis(self):
        return self._bkd.array(
            [
                (self.A0**4 * (213 + 29 * self._bkd.cos(4 * self._theta0)))
                / 5082,
                (self.A1**4 * (93 + 5 * self._bkd.cos(4 * self._theta1)))
                / 1274,
                -(1 / 30)
                * self.A2**4
                * (-7 + self._bkd.cos(4 * self._theta2)),
            ]
        )

    def _covariance_of_variances(
        self,
        nsamples,
        E_f0_sq_f1_sq,
        E_f0_sq,
        E_f1_sq,
        E_f1,
        E_f0_sq_f1,
        E_f0,
        E_f0_f1_sq,
        E_f0_f1,
    ):
        E_u20_u21 = (
            E_f0_sq_f1_sq
            - E_f0_sq * E_f1_sq
            - 2 * E_f1 * E_f0_sq_f1
            + 2 * E_f1**2 * E_f0_sq
            - 2 * E_f0 * E_f0_f1_sq
            + 2 * E_f0**2 * E_f1_sq
            + 4 * E_f0 * E_f1 * E_f0_f1
            - 4 * E_f0**2 * E_f1**2
        )
        return E_u20_u21 / nsamples + 1 / (nsamples * (nsamples - 1)) * (
            E_f0_f1**2 - 2 * E_f0_f1 * E_f0 * E_f1 + (E_f0 * E_f1) ** 2
        )

    def covariance_of_variances(self, nsamples):
        t0, t1, t2 = self._theta0, self._theta1, self._theta2
        s1, s2 = self._shifts
        E_f0_sq_f1 = self.A0**2 * s1 / 11
        E_f0_f1_sq = 2 * self.A0 * self.A1 * s1 * self._bkd.cos(t1 - t0) / 9
        E_f0_sq_f1_sq = (
            self.A0**2
            * (
                7614 * self.A1**2
                + 19278 * s1**2
                + 3739 * self.A1**2 * self._bkd.cos(2 * (t1 - t0))
                + 1121 * self.A1**2 * self._bkd.cos(2 * (t1 + t0))
            )
        ) / 212058

        E_f0_sq_f2 = self.A0**2 * s2 / 11
        E_f0_f2_sq = 2 * self.A0 * self.A2 * s2 * self._bkd.cos(t2 - t0) / 7
        E_f0_sq_f2_sq = (
            self.A0**2
            * (
                98 * (23 * self.A2**2 + 39 * s2**2)
                + 919 * self.A2**2 * self._bkd.cos(2 * (t2 - t0))
                + 61 * self.A2**2 * self._bkd.cos(2 * (t2 + t0))
            )
        ) / 42042

        E_f1_sq_f2 = (
            (self.A1**2 * s2) / 7
            + s1**2 * s2
            + 2 / 5 * self.A1 * self.A2 * s1 * self._bkd.cos(t1 - t2)
        )
        E_f1_f2_sq = (
            (self.A2**2 * s1) / 3
            + s1 * s2**2
            + 2 / 5 * self.A1 * self.A2 * s2 * self._bkd.cos(t1 - t2)
        )
        E_f1_sq_f2_sq = (
            (5 * self.A1**2 * self.A2**2) / 63
            + (self.A2**2 * s1**2) / 3
            + (self.A1**2 * s2**2) / 7
            + s1**2 * s2**2
            + 4 / 5 * self.A1 * self.A2 * s1 * s2 * self._bkd.cos(t1 - t2)
            + (
                113
                * (self.A1**2 * self.A2**2 * self._bkd.cos(2 * (t1 - t2)))
                / 3150
                - (13 * self.A1**2 * self.A2**2 * self._bkd.cos(2 * (t1 + t2)))
                / 3150
            )
        )

        E_f0, E_f1, E_f2 = self.mean()
        cov = self.covariance()
        E_f0_sq = cov[0, 0] + E_f0**2
        E_f1_sq = cov[0, 0] + E_f1**2
        E_f2_sq = cov[0, 0] + E_f2**2

        E_f0_f1 = cov[0, 1] + E_f0 * E_f1
        E_f0_f2 = cov[0, 2] + E_f0 * E_f2
        E_f1_f2 = cov[1, 2] + E_f1 * E_f2

        Cmat = self._bkd.zeros((3, 3))
        Cmat[0, 1] = self._covariance_of_variances(
            nsamples,
            E_f0_sq_f1_sq,
            E_f0_sq,
            E_f1_sq,
            E_f1,
            E_f0_sq_f1,
            E_f0,
            E_f0_f1_sq,
            E_f0_f1,
        )
        Cmat[1, 0] = Cmat[0, 1]

        Cmat[0, 2] = self._covariance_of_variances(
            nsamples,
            E_f0_sq_f2_sq,
            E_f0_sq,
            E_f2_sq,
            E_f2,
            E_f0_sq_f2,
            E_f0,
            E_f0_f2_sq,
            E_f0_f2,
        )
        Cmat[2, 0] = Cmat[0, 2]

        Cmat[1, 2] = self._covariance_of_variances(
            nsamples,
            E_f1_sq_f2_sq,
            E_f1_sq,
            E_f2_sq,
            E_f2,
            E_f1_sq_f2,
            E_f1,
            E_f1_f2_sq,
            E_f1_f2,
        )
        Cmat[2, 1] = Cmat[1, 2]

        variances = self._bkd.diag(cov)
        kurtosis = self.get_kurtosis()
        C_mat_diag = (
            kurtosis - (nsamples - 3) / (nsamples - 1) * variances**2
        ) / nsamples
        for ii in range(3):
            Cmat[ii, ii] = C_mat_diag[ii]

        return Cmat

    def costs(self) -> Array:
        """
        The nominal costs of each model for a single sample

        Returns
        -------
        values : Array (nmodels)
            Model costs
        """
        return self._bkd.array([1.0, 0.01, 0.001])


class ShortColumnModelEnsemble(ACVBenchmark):
    def __init__(self, nmodels: int = 5, backend=NumpyLinAlgMixin):
        self._nmodels = nmodels
        super().__init__(backend)

    def nmodels(self) -> int:
        return self._nmodels

    def nqoi(self) -> int:
        return 1

    def costs(self) -> Array:
        return self._bkd.logspace(0, -self.nmodels(), self.nmodels())

    def m0(self, samples):
        return samples.T**5

    def m1(self, samples):
        return samples.T**4

    def m2(self, samples):
        return samples.T**3

    def m3(self, samples):
        return samples.T**2

    def m4(self, samples):
        return samples.T**1

    def _set_models(self):
        funs = [self.m0, self.m1, self.m2, self.m3, self.m4][: self.nmodels()]
        self._models = [
            ModelFromVectorizedCallable(self.nqoi(), fun, backend=self._bkd)
            for fun in funs
        ]

    def _mean(self):
        return 1 / self._bkd.arange(6, 1, -1)[: self.nmodels()].reshape(
            self.nmodels(), self.nqoi()
        )

    def _set_variable(self):
        univariate_variables = [stats.uniform(0, 1)]
        self._variable = IndependentMarginalsVariable(univariate_variables)


class MultiOutputModelEnsemble(ACVBenchmark):
    """
    Benchmark for testing multifidelity algorithms that estimate statistics
    for vector valued models of varying fidelity.
    """

    def nmodels(self) -> int:
        return 3

    def f0(self, samples: Array) -> Array:
        """
        Highest fidelity model

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations

        Returns
        -------
        values : Array (nsamples, qoi)
            Model evaluations at the samples
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
        A low fidelity model

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations

        Returns
        -------
        values : Array (nsamples, qoi)
            Model evaluations at the samples
        """
        return self._bkd.hstack(
            [
                math.sqrt(7) * samples.T**3,
                math.sqrt(7) * samples.T**2,  # alexs paper
                # math.sqrt(6.9)*samples.T**2, # test
                self._bkd.cos(2 * math.pi * samples.T + math.pi / 2),
            ]
        )

    def f2(self, samples: Array) -> Array:
        """
        A low fidelity model

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Samples realizations

        Returns
        -------
        values : Array (nsamples, qoi)
            Model evaluations at the samples
        """
        return self._bkd.hstack(
            [
                math.sqrt(3) / 2 * samples.T**2,
                math.sqrt(3) / 2 * samples.T,
                self._bkd.cos(2 * math.pi * samples.T + math.pi / 4),
            ]
        )

    def _set_variable(self):
        self._variable = IndependentMarginalsVariable([stats.uniform(0, 1)])

    def _set_models(self):
        self._models = [
            ModelFromVectorizedCallable(self.nqoi(), fun, backend=self._bkd)
            for fun in [self.f0, self.f1, self.f2]
        ]

    def nqoi(self) -> int:
        return 3

    def costs(self) -> Array:
        """
        The nominal costs of each model for a single sample

        Returns
        -------
        values : Array (nmodels)
            Model costs
        """
        return self._bkd.array([1.0, 0.01, 0.001])

    def _uniform_means(self):
        return self._bkd.array(
            [
                [math.sqrt(11) / 6, 1 / 5, 0.0],
                [math.sqrt(7) / 4, math.sqrt(7) / 3, 0.0],
                [1 / (2 * math.sqrt(3)), math.sqrt(3) / 4, 0.0],
            ]
        )

    def _uniform_covariance_matrices(self):
        # compute diagonal blocks
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
        # compute off digonal block covariance between model 0 and mode 1
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

        # compute off digonal block covariance between model 0 and mode 2
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

        # compute off digonal block covariance between model 1 and mode 2
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
