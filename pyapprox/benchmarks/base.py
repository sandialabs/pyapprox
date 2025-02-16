from abc import ABC, abstractmethod
from functools import partial
from typing import List, Union

from pyapprox.interface.model import Model
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.variables.joint import JointVariable, DesignVariable
from pyapprox.surrogates.bases.orthopoly import GaussQuadratureRule
from pyapprox.surrogates.bases.basis import (
    FixedGaussianTensorProductQuadratureRuleFromVariable,
)
from scipy.optimize import LinearConstraint
from pyapprox.optimization.pya_minimize import Constraint


class SingleModelBenchmark(ABC):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._bkd = backend
        self._set_variable()
        self._set_model()

    def nvars(self) -> int:
        """Return the number of uncertain variables."""
        return self._variable.nvars()

    def variable(self) -> JointVariable:
        """Return the random variable parameterizing the model uncertainty."""
        return self._variable

    def nqoi(self) -> int:
        """Return the number of quantities of interest (QoI) of all models."""
        return self._model.nqoi()

    @abstractmethod
    def _set_model(self):
        raise NotImplementedError

    @abstractmethod
    def _set_variable(self):
        raise NotImplementedError

    def model(self) -> Model:
        """Return the model"""
        return self._model

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class MultiModelBenchmark(ABC):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._bkd = backend
        self._set_variable()
        self._set_models()

    def nvars(self) -> int:
        """Return the number of uncertain variables."""
        return self._variable.nvars()

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
        quad_rule = FixedGaussianTensorProductQuadratureRuleFromVariable(
            self.variable(), [21] * self.nvars()
        )
        self._quadx, self._quadw = quad_rule()
        self._quadw = self._quadw[:, 0]

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
        means = [f(self._quadx) @ self._quadw for f in self._flat_funs]

        cov = self._bkd.empty(
            (self.nmodels() * self.nqoi(), self.nmodels() * self.nqoi())
        )
        ii = 0
        for fi, mi in zip(self._flat_funs, means):
            jj = 0
            for fj, mj in zip(self._flat_funs, means):
                cov[ii, jj] = (
                    (fi(self._quadx) - mi) * (fj(self._quadx) - mj)
                ) @ (self._quadw)
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

    def __call__(self, samples: Array) -> Array:
        # samples must include model id in last row
        return self._model_ensemble(samples)


class OptimizationBenchmark(ABC):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._bkd = backend
        # must use set_objective here to make sure model
        # is only created once and not everytime self.objective is called
        # so that internal model variables persist, e.g. _work_tracker
        self._set_objective()

    @abstractmethod
    def _set_objective(self):
        raise NotImplementedError

    def objective(self) -> Model:
        return self._objective

    @abstractmethod
    def design_variable(self) -> DesignVariable:
        raise NotImplementedError


class ConstrainedOptimizationBenchmark(OptimizationBenchmark):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        # must use set_constraints here to make sure model
        # is only created once and not everytime self.objective is called
        # so that internal model variables persist, e.g. _work_tracker
        self._set_constraints()

    @abstractmethod
    def _set_constraints(self):
        raise NotImplementedError

    def constraints(self) -> List[Union[Constraint, LinearConstraint]]:
        return self._constraints

    @abstractmethod
    def optimal_iterate(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def init_iterate(self) -> Array:
        raise NotImplementedError


class ConstrainedUncertainOptimizationBenchmark(
    ConstrainedOptimizationBenchmark
):
    @abstractmethod
    def variable(self) -> JointVariable:
        raise NotImplementedError

    @abstractmethod
    def design_var_indices(self) -> Array:
        """
        The position of the design variables in a combined
        uncertain + design variable array
        """
        raise NotImplementedError


class SingleModelBayesianInferenceBenchmark(SingleModelBenchmark):
    @abstractmethod
    def negloglike(self) -> Model:
        raise NotImplementedError


# TODO make bencmhar base class then derive different types of benchmarks from
# that class to better reuse code
class OperatorBenchmark(ABC):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._bkd = backend
        # must use set_model here to make sure model
        # is only created once and not everytime self.objective is called
        # so that internal model variables persist, e.g. _work_tracker
        self._set_model()

    @abstractmethod
    def variable(self) -> JointVariable:
        """Return the random variable parameterizing the model uncertainty."""
        raise NotImplementedError

    @abstractmethod
    def _set_model(self) -> Model:
        raise NotImplementedError

    def model(self) -> Model:
        """Return the model"""
        return self._model

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)
