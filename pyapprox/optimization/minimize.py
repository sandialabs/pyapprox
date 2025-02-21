from typing import List
from abc import ABC, abstractmethod

import numpy as np

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin

from pyapprox.interface.model import (
    Model,
    ActiveSetVariableModel,
    SingleSampleModel,
)
from scipy.optimize import LinearConstraint


class OptimizationResult(dict):
    """
    The optimization result returned by optimizers. must contain at least
    the iterate and objective function value at the minima,
    which can be accessed via res.x and res.fun, respectively.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())

    def __repr__(self):
        return self.__class__.__name__ + (
            "(\n\t x[:, 0]={0}, \n\t fun={1}, \n\t attr={2})".format(
                self.x[:, 0], self.fun, list(self.keys())
            )
        )


class Constraint(Model):
    def __init__(
        self,
        bounds=None,
        keep_feasible=False,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        if bounds is not None:
            self.set_bounds(bounds)
        self._keep_feasible = keep_feasible

    def hessian_implemented(self) -> bool:
        return False

    def set_bounds(self, bounds: Array):
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must be 2D np.ndarray with two columns")
        self._bounds = bounds


class ConstraintFromModel(Constraint):
    def __init__(
        self, model, bounds=None, keep_feasible=False, backend=NumpyLinAlgMixin
    ):
        super().__init__(bounds, keep_feasible, backend)
        if not isinstance(model, Model):
            raise ValueError(
                "constraint must be an instance of {0}".format(
                    "pyapprox.interface.model.Model"
                )
            )
        self._model = model
        self._update_attributes()

    def _update_attributes(self):
        for attr in [
            "apply_jacobian_implemented",
            "jacobian_implemented",
            "hessian_implemented",
            "apply_hessian_implemented",
            "jacobian",
            "apply_jacobian",
            "apply_hessian",
            "hessian",
        ]:
            setattr(self, attr, getattr(self._model, attr))

    def nvars(self) -> int:
        return self._model.nvars()

    def _check_sample(self, sample: Array):
        if sample.shape[1] > 1:
            raise ValueError(
                "Constraint can only be evaluated at one sample "
                f"but {sample.shape=}"
            )

    def _values(self, sample: Array) -> Array:
        return self._model(sample)

    def nqoi(self) -> int:
        return self._model.nqoi()

    def __repr__(self) -> str:
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class OptimizerIterateGenerator(ABC):
    def __init__(self, backend):
        self._bkd = backend

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class RandomUniformOptimzerIterateGenerator(OptimizerIterateGenerator):
    def __init__(self, nvars, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._bounds = None
        self._nvars = nvars
        self._numeric_upper_bound = 100

    def set_bounds(self, bounds):
        bounds = self._bkd.atleast1d(bounds)
        if bounds.shape[0] == 2 and bounds.ndim != 2:
            bounds = self._bkd.reshape(
                self._bkd.repeat(bounds, self._nvars), (self._nvars, 2)
            )
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds has the wrong shape")
        self._bounds = bounds

    def set_numeric_upper_bound(self, ub):
        self._numeric_upper_bound = ub

    def __call__(self):
        if self._bounds is None:
            raise RuntimeError(
                "must call set_bounds to generate random initial guess"
            )
        # convert bounds to numpy to use numpy random number generator
        bounds = self._bkd.to_numpy(self._bounds)
        bounds[bounds == -np.inf] = -self._numeric_upper_bound
        bounds[bounds == np.inf] = self._numeric_upper_bound
        return self._bkd.asarray(
            np.random.uniform(bounds[:, 0], bounds[:, 1])
        )[:, None]


# TODO Some optimizer classes here are duplicated in
# pyapprox.bases.optimsers. Merge and remove duplicate code
class Optimizer(ABC):
    def __init__(self, opts={}):
        self._bkd = None
        self._verbosity = 0
        self.set_options(**opts)

    def set_options(self, **opts):
        self._opts = opts

    @abstractmethod
    def _minimize(self, iterate: Array) -> OptimizationResult:
        raise NotImplementedError

    def minimize(self, iterate: Array) -> OptimizationResult:
        """
        Minimize the objective function.

        Parameters
        ----------
        iterate : array
             The initial guess used to start the optimizer

        Returns
        -------
        res : :py:class:`~pyapprox.sciml.OptimizationResult`
             The optimization result.
        """
        if iterate.ndim != 2 or iterate.shape[1] != 1:
            raise ValueError("iterate must be a 2D array with one column.")
        result = self._minimize(iterate)
        if not isinstance(result, OptimizationResult):
            raise RuntimeError(
                "{0}.minimize did not return OptimizationResult".format(self)
            )
        return result

    def set_verbosity(self, verbosity: int):
        """
        Set the verbosity.

        Parameters
        ----------
        verbosity_flag : int, default 0
            0 = no output
            1 = final iteration
            2 = each iteration
            3 = each iteration, plus details
        """
        self._verbosity = verbosity

    def __repr__(self) -> str:
        return "{0}(verbosity={1})".format(
            self.__class__.__name__, self._verbosity
        )


class OptimizerWithObjective(Optimizer):
    def __init__(self, objective=None, bounds=None, opts={}):
        super().__init__(opts)
        self._objective = None
        self._bounds = None

        if objective is not None:
            self.set_objective_function(objective)
        if bounds is not None:
            self.set_bounds(bounds)
        self.set_options(**opts)

    def set_options(self, **opts):
        self._opts = opts

    def minimize(self, iterate):
        """
        Minimize the objective function.

        Parameters
        ----------
        iterate : array
             The initial guess used to start the optimizer

        Returns
        -------
        res : :py:class:`~pyapprox.sciml.OptimizationResult`
             The optimization result.
        """
        if self._objective is None:
            raise RuntimeError("Must call set_objective_function")
        return super().minimize(iterate)

    def set_bounds(self, bounds: Array):
        """
        Set the bounds of the design variables.

        Parameters
        ----------
        bounds : array (ndesign_vars, 2)
            The upper and lower bounds of each design variable
        """
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds has the wrong shape")
        self._bounds = bounds

    def set_objective_function(self, objective):
        """
        Set the objective function.

        Parameters
        ----------
        objective_fun : callable
            Function that returns both the function value and gradient at an
            iterate with signature

            `objective_fun(x) -> (val, grad)`

            where `x` and `val` are 1D arrays with shape (ndesign_vars,) and
            `val` is a float.
        """
        if not isinstance(objective, Model) and objective.nqoi() == 1:
            raise ValueError(
                "objective must be an instance of {0}".format(
                    "pyapprox.interface.model.Model with nqoi() == 1"
                )
            )
        self._bkd = objective._bkd
        self._objective = objective

    def _is_iterate_within_bounds(self, iterate):
        if self._bounds is None:
            return True
        # convert bounds to numpy to use np.logical
        bounds = self._bkd.to_numpy(self._bounds)
        iterate = self._bkd.to_numpy(iterate)
        return np.logical_and(
            iterate >= bounds[:, 0], iterate <= bounds[:, 1]
        ).all()


class MultiStartOptimizer(OptimizerWithObjective):
    def __init__(self, optimizer, ncandidates=1):
        """
        Find the smallest local optima associated with a set of
        initial guesses.

        Parameters
        ----------
        optimizer : :py:class:`~pyapprox.sciml.Optimizer`
            Optimizer to find each local minima

        ncandidates : int
            Number of initial guesses used to comptue local optima
        """
        self._ncandidates = ncandidates
        self._optimizer = optimizer
        self._bounds = None
        self._initial_interate_gen = None
        super().__init__(objective=optimizer._objective)

    def set_bounds(self, bounds):
        super().set_bounds(bounds)
        self._optimizer.set_bounds(bounds)

    def set_verbosity(self, verbosity):
        self._verbosity = verbosity
        self._optimizer.set_verbosity(verbosity)

    def set_initial_iterate_generator(self, gen):
        if not isinstance(gen, OptimizerIterateGenerator):
            raise ValueError("gen is not an OptimizerIterateGenerator.")
        self._initial_interate_gen = gen

    def set_objective_function(self, objective):
        self._optimizer.set_objective_function(objective)
        super().set_objective_function(objective)

    def _minimize(self, x0_global, **kwargs):
        if self._initial_interate_gen is None:
            raise ValueError("Must call set_initial_iterate_generator")
        best_res = self._optimizer.minimize(x0_global)
        sucess = best_res.success
        if self._verbosity > 1:
            print("it {0}: best objective {1}".format(0, best_res.fun))
        for ii in range(1, self._ncandidates):
            iterate = self._initial_interate_gen()
            res = self._optimizer.minimize(iterate, **kwargs)
            if res.success:
                sucess = True
            if res.fun < best_res.fun:
                best_res = res
            if self._verbosity > 1:
                print("it {0}: objective {1}".format(ii + 1, res.fun))
                print(
                    "it {0}: best objective {1}".format(ii + 1, best_res.fun)
                )
        if self._verbosity > 0:
            print("{0}\n\t {1}".format(self, best_res))
        if not sucess:
            raise RuntimeError("All optimizations failed")
        return best_res

    def __repr__(self):
        return "{0}(optimizer={1}, ncandidates={2})".format(
            self.__class__.__name__, self._optimizer, self._ncandidates
        )


class ConstrainedOptimizer(OptimizerWithObjective):
    def __init__(self, objective=None, constraints=[], bounds=None, opts={}):
        super().__init__(objective, bounds, opts)
        self._raw_constraints = None
        # self._constraints = constraints
        self.set_constraints(constraints)

    def set_constraints(self, constraints: List[Constraint]):
        self._raw_constraints = constraints
        for con in constraints:
            if not isinstance(con, Constraint) and not isinstance(
                con, LinearConstraint
            ):
                raise ValueError(
                    "constraint must be an instance of {0} or {1}".format(
                        "pyapprox.optimize.pya_minimize.Constraint",
                        "scipy.optimize.LinearConstraint",
                    )
                )
        self._constraints = self._convert_constraints(constraints)

    def _convert_constraints(self, constraints):
        return constraints


class ConstrainedMultiStartOptimizer(MultiStartOptimizer):
    def __init__(self, optimizer, ncandidates=1):
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                "optimizer must be an instance of ConstrainedOptimizer"
            )
        super().__init__(optimizer, ncandidates)

    def set_constraints(self, constraints):
        self._optimizer.set_constraints(constraints)


class ConstraintPenalizedObjective(Model):
    def __init__(
        self, unconstrained_objective: Model, constraints: List[Constraint]
    ):
        super().__init__(unconstrained_objective._bkd)
        self._unconstrained_objective = unconstrained_objective
        self._constraints = constraints
        self._penalty = None

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._unconstrained_objective.nvars()

    def set_penalty(self, penalty):
        self._penalty = penalty

    def _values(self, samples: Array) -> Array:
        cons_term = 0
        for con in self._constraints:
            con_vals = con(samples)
            for con_val in con_vals.T:
                if con_val < 0:
                    # if constraint violated add a penalty
                    cons_term += -con_val * self._penalty
        return self._unconstrained_objective(samples) + cons_term


class ChainedOptimizer(Optimizer):
    def __init__(self, optimizer1: Optimizer, optimizer2: Optimizer):
        super().__init__()
        if not isinstance(optimizer1, Optimizer):
            raise ValueError(
                "optimizer1 {0} must be an instance of Optimizer".format(
                    optimizer1
                )
            )
        if not isinstance(optimizer2, Optimizer):
            raise ValueError(
                "optimizer2 {0} must be an instance of Optimizer".format(
                    optimizer2
                )
            )
        self._optimizer1 = optimizer1
        self._optimizer2 = optimizer2

    def _minimize(self, iterate: Array) -> OptimizationResult:
        result1 = self._optimizer1.minimize(iterate)
        result2 = self._optimizer2.minimize(result1.x)
        return result2

    def set_bounds(self, bounds: Array):
        self._optimizer1.set_bounds(bounds)
        self._optimizer2.set_bounds(bounds)

    def set_objective_function(self, objective: Model):
        self._optimizer1.set_objective_function(objective)
        self._optimizer2.set_objective_function(objective)

    def set_constraints(self, constraints: List[Constraint]):
        self._optimizer1.set_constraints(constraints)
        self._optimizer2.set_constraints(constraints)

    def set_verbosity(self, verbosity: int):
        self._verbosity = verbosity
        self._optimizer1.set_verbosity(verbosity)
        self._optimizer2.set_verbosity(verbosity)


# TODO consider merging with multifidelity.stat
class SampleAverageStat(ABC):
    def __init__(self, backend=NumpyLinAlgMixin):
        self._bkd = backend

    def hessian_implemented(self) -> bool:
        return False

    @abstractmethod
    def __call__(self, values, weights):
        """
        Compute the sample average statistic.

        Parameters
        ----------
        values: array (nsamples, nqoi)
           function values at each sample

        weights: array(nsamples, 1)
            qudrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, 1)
            Estimate of the statistic
        """
        raise NotImplementedError

    def jacobian(self, values, jac_values, weights):
        """
        Compute the sample average jacobian.

        Parameters
        ----------
        values: array (nsamples, nqoi)
           function values at each sample

        jac_values: array (nsamples, nqoi, nvars)
           function values at each sample

        weights: array(nsamples, 1)
            qudrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, nvars)
            Estimate of the statistic jacobian
        """
        raise NotImplementedError

    def apply_jacobian(self, values, jv_values, weights):
        """
        Compute the sample average jacobian dot product with
        a vector.

        Parameters
        ----------
        values: array (nsamples, nqoi)
           function values at each sample

        jv_values : array (nsamples, nqoi, 1)
            values of the jacobian vector product (jvp) at each sample

        weights: array(nsamples, 1)
            qudrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, 1)
            Estimate of the statistic jacobian vector product
        """
        raise NotImplementedError

    def hessian(self, values, jac_values, hess_values, weights):
        """
        Compute the sample average hessian.

        Parameters
        ----------
        values: array (nsamples, nqoi)
           function values at each sample

        jac_values: array (nsamples, nqoi, nvars)
           function values at each sample

        jac_values: array (nsamples, nqoi, nvars, nvars)
           function values at each sample

        weights: array(nsamples, 1)
            qudrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, nvars, nvars)
            Estimate of the statistic hessian
        """
        raise NotImplementedError

    def apply_hessian(self, values, jv_values, hv_values, weights, lagrange):
        """
        Compute the sample average weighted combination of the
        Qoi, dot product with a vector.

        Parameters
        ----------
        values: array (nsamples, nqoi)
           function values at each sample

        jv_values : array (nsamples, nqoi, 1)
            values of the jacobian vector product (jvp) at each sample

        hv_values : array (nsamples, nqoi, 1)
            values of the hessian vector product (hvp) at each sample

        weights: array(nsamples, 1)
            qudrature weight for each sample

        lagrange: array (nqoi, 1)
            The weights defining the combination of QoI.

        Returns
        -------
        estimate: array (nqoi, 1)
            Estimate of the statistic jacobian vector product
        """
        raise NotImplementedError

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)


class SampleAverageMean(SampleAverageStat):
    def __init__(self, backend=NumpyLinAlgMixin):
        super().__init__(backend)

    def hessian_implemented(self) -> bool:
        return True

    def __call__(self, values, weights):
        # values.shape (nsamples, ncontraints)
        return (values.T @ weights).T

    def jacobian(self, values, jac_values, weights):
        # jac_values.shape (nsamples, ncontraints, ndesign_vars)
        return self._bkd.einsum("ijk,i->jk", jac_values, weights[:, 0])

    def apply_jacobian(self, values, jv_values, weights):
        # jac_values.shape (nsamples, ncontraints)
        return (jv_values[..., 0].T @ weights[:, 0])[:, None]

    def hessian(self, values, jac_values, hess_values, weights):
        return self._bkd.einsum("ijkl,i->jkl", hess_values, weights[:, 0])


class SampleAverageVariance(SampleAverageStat):
    def __init__(self, backend=NumpyLinAlgMixin):
        super().__init__(backend=backend)
        self._mean_stat = SampleAverageMean(backend=backend)

    def hessian_implemented(self) -> bool:
        return True

    def _diff(self, values, weights):
        mean = self._mean_stat(values, weights).T
        return (values - mean[:, 0]).T

    def __call__(self, values, weights):
        # values.shape (nsamples, ncontraints)
        return (self._diff(values, weights) ** 2 @ weights).T

    def jacobian(self, values, jac_values, weights):
        # jac_values.shape (nsamples, ncontraints, ndesign_vars)
        mean_jac = self._mean_stat.jacobian(values, jac_values, weights)[
            None, :
        ]
        tmp = jac_values - mean_jac
        tmp = 2 * self._diff(values, weights).T[..., None] * tmp
        return self._bkd.einsum("ijk,i->jk", tmp, weights[:, 0])

    def apply_jacobian(self, values, jv_values, weights):
        mean_jv = self._mean_stat.apply_jacobian(values, jv_values, weights)
        tmp = jv_values - mean_jv
        tmp = 2 * self._diff(values, weights).T * tmp[..., 0]
        return self._bkd.einsum("ij,i->j", tmp, weights[:, 0])

    def hessian(self, values, jac_values, hess_values, weights):
        mean_jac = self._mean_stat.jacobian(values, jac_values, weights)[
            None, :
        ]
        mean_hess = self._mean_stat.hessian(
            values, jac_values, hess_values, weights
        )[None, :]
        tmp_jac = jac_values - mean_jac
        tmp_hess = hess_values - mean_hess
        tmp1 = 2 * self._diff(values, weights).T[..., None, None] * tmp_hess
        tmp2 = 2 * self._bkd.einsum("ijk, ijl->ijkl", tmp_jac, tmp_jac)
        return self._bkd.einsum("ijkl,i->jkl", tmp1 + tmp2, weights[:, 0])


class SampleAverageStdev(SampleAverageVariance):
    def __call__(self, samples, weights):
        return self._bkd.sqrt(super().__call__(samples, weights))

    def jacobian(self, values, jac_values, weights):
        variance_jac = super().jacobian(values, jac_values, weights)
        # d/dx y^{1/2} = 0.5y^{-1/2}
        tmp = 1 / (2 * self._bkd.sqrt(super().__call__(values, weights).T))
        return tmp * variance_jac

    def apply_jacobian(self, values, jv_values, weights):
        variance_jv = super().apply_jacobian(values, jv_values, weights)
        # d/dx y^{1/2} = 0.5y^{-1/2}
        tmp = 1 / (2 * self._bkd.sqrt(super().__call__(values, weights).T))
        return tmp * variance_jv[:, None]

    def hessian(self, values, jac_values, hess_values, weights):
        variance_jac = super().jacobian(values, jac_values, weights)
        variance_hess = super().hessian(
            values, jac_values, hess_values, weights
        )
        # f:R^n->R, g(R->R) h(x) = g(f(x))
        # d^2h(x)/dx^2 g'(f(x))\nabla^2 f(x)+g''(f(x))\nabla f(x)\nabla f(x)^T
        # g(x)=sqrt(x), g'(x) = 1/(2x^{1/2}), g''(x)=-1/(4x^{3/2})
        tmp0 = self._bkd.sqrt(super().__call__(values, weights).T)
        tmp1 = (
            1.0
            / (4.0 * tmp0[..., None] ** 3.0)
            * self._bkd.einsum("ij,ik->ijk", variance_jac, variance_jac)
        )
        tmp2 = 1.0 / (2.0 * tmp0[..., None]) * variance_hess
        return tmp2 - tmp1


class SampleAverageMeanPlusStdev(SampleAverageStat):
    def __init__(self, safety_factor):
        super().__init__()
        self._mean_stat = SampleAverageMean()
        self._stdev_stat = SampleAverageStdev()
        self._safety_factor = safety_factor

    def hessian_implemented(self) -> bool:
        return True

    def __call__(self, values, weights):
        return self._mean_stat(
            values, weights
        ) + self._safety_factor * self._stdev_stat(values, weights)

    def jacobian(self, values, jac_values, weights):
        return self._mean_stat.jacobian(
            values, jac_values, weights
        ) + self._safety_factor * self._stdev_stat.jacobian(
            values, jac_values, weights
        )

    def apply_jacobian(self, values, jv_values, weights):
        return self._mean_stat.apply_jacobian(
            values, jv_values, weights
        ) + self._safety_factor * self._stdev_stat.apply_jacobian(
            values, jv_values, weights
        )

    def hessian(self, values, jac_values, hess_values, weights):
        return self._mean_stat.hessian(
            values, jac_values, hess_values, weights
        ) + self._safety_factor * self._stdev_stat.hessian(
            values, jac_values, hess_values, weights
        )


class SampleAverageEntropicRisk(SampleAverageStat):
    def __init__(self, alpha, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._alpha = alpha

    def hessian_implemented(self) -> bool:
        return True

    def __call__(self, values, weights):
        # values (nsamples, noutputs)
        return (
            self._bkd.log(self._bkd.exp(self._alpha * values.T) @ weights).T
            / self._alpha
        )

    def jacobian(self, values, jac_values, weights):
        # jac_values (nsamples, noutputs, nvars)
        exp_values = self._bkd.exp(self._alpha * values)
        tmp = exp_values.T @ weights
        # h = g(f(x))
        # dh/dx = g'(f(x))\nabla f(x)
        # g(y) = log(y)/alpha, g'(y) = 1/(alpha*y)
        jac = (
            1
            / tmp
            * self._bkd.einsum(
                "ijk,i->jk",
                (self._alpha * exp_values[..., None] * jac_values),
                weights[:, 0],
            )
        )
        return jac / self._alpha

    def apply_jacobian(self, values, jv_values, weights):
        exp_values = self._bkd.exp(self._alpha * values)
        tmp = exp_values.T @ weights
        jv = (
            1
            / tmp
            * self._bkd.einsum(
                "ij,i->j",
                (self._alpha * exp_values * jv_values[..., 0]),
                weights[:, 0],
            )[:, None]
        )
        return jv / self._alpha

    def hessian(self, values, jac_values, hess_values, weights):
        exp_values = self._bkd.exp(self._alpha * values)
        exp_jac = self._alpha * self._bkd.einsum(
            "ijk,i->jk", (exp_values[..., None] * jac_values), weights[:, 0]
        )
        exp_jac_outprod = self._bkd.einsum("ij,ik->ijk", exp_jac, exp_jac)
        jac_values_outprod = self._bkd.einsum(
            "ijk,ijl->ijkl", jac_values, jac_values
        )
        exp_hess = self._alpha * self._bkd.einsum(
            "ijkl,i->jkl",
            (exp_values[..., None, None] * hess_values),
            weights[:, 0],
        ) + self._alpha**2 * self._bkd.einsum(
            "ijkl,i->jkl",
            (exp_values[..., None, None] * jac_values_outprod),
            weights[:, 0],
        )
        tmp0 = exp_values.T @ weights
        tmp1 = 1.0 / tmp0[..., None] * exp_hess
        tmp2 = 1.0 / tmp0[..., None] ** 2 * exp_jac_outprod
        return (tmp1 - tmp2) / self._alpha


class SmoothLogBasedMaxFunction:
    def __init__(self, eps, threshold=None, backend=NumpyLinAlgMixin):
        self._bkd = backend
        self._eps = eps
        if threshold is None:
            threshold = 1e2
        self._thresh = threshold

    def jacobian_implemented(self) -> bool:
        return True

    def _check_samples(self, samples):
        if samples.ndim != 2:
            raise ValueError("samples must be a 2D array")

    def __call__(self, samples):
        self._check_samples(samples)
        x = samples
        x_div_eps = x / self._eps
        # avoid overflow
        vals = self._bkd.zeros(x.shape)
        II = self._bkd.where(
            (x_div_eps < self._thresh) & (x_div_eps > -self._thresh)
        )
        vals[II] = x[II] + self._eps * self._bkd.log(
            1 + self._bkd.exp(-x_div_eps[II])
        )
        J = self._bkd.where(x_div_eps >= self._thresh)
        vals[J] = x[J]
        return vals

    def jacobians(self, samples):
        # samples (noutputs, nsamples)
        # jac_values (nsamples, noutputs, noutputs)
        # but only return (nsamples, noutputs) because jac for each sample
        # is just a diagonal matrix
        self._check_samples(samples)
        x = samples
        x_div_eps = x / self._eps
        # Avoid overflow.
        II = self._bkd.where(
            (x_div_eps < self._thresh) & (x_div_eps > -self._thresh)
        )
        jac = self._bkd.zeros((x_div_eps.shape))
        jac[II] = 1.0 / (1 + self._bkd.exp(-x_div_eps[II]))
        jac[x_div_eps >= self._thresh] = 1.0
        return jac[..., None]


class SampleAverageConditionalValueAtRisk(SampleAverageStat):
    def __init__(self, alpha, eps=1e-2, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        alpha = self._bkd.atleast1d(alpha)
        self._alpha = alpha
        self._max = SmoothLogBasedMaxFunction(eps, backend=self._bkd)
        self._t = None

    def set_value_at_risk(self, t):
        t = self._bkd.atleast1d(t)
        if t.shape[0] != self._alpha.shape[0]:
            msg = "VaR shape {0} and alpha shape {1} are inconsitent".format(
                t.shape, self._alpha.shape
            )
            raise ValueError(msg)
        if t.ndim != 1:
            raise ValueError("t must be a 1D array")
        self._t = t

    def __call__(self, values, weights):
        if values.shape[1] != self._t.shape[0]:
            raise ValueError("must specify a VaR for each QoI")
        return self._t + (self._max(values - self._t).T @ weights).T / (
            1 - self._alpha
        )

    def jacobian(self, values, jac_values, weights):
        # grad withe respect to parameters of x
        max_jac = self._max.jacobians(values - self._t)
        param_jac = self._bkd.einsum(
            "ijk,i->jk", (max_jac * jac_values), weights[:, 0]
        ) / (1 - self._alpha[:, None])
        t_jac = 1 - self._bkd.einsum(
            "ij,i->j", max_jac[..., 0], weights[:, 0]
        ) / (1 - self._alpha)
        return self._bkd.hstack((param_jac, self._bkd.diag(t_jac)))


class SampleAverageConstraint(ConstraintFromModel):
    def __init__(
        self,
        model,
        samples,
        weights,
        stat,
        design_bounds,
        nvars,
        design_indices,
        keep_feasible=False,
        backend=NumpyLinAlgMixin,
    ):
        self._stat = stat
        super().__init__(model, design_bounds, keep_feasible, backend=backend)
        if samples.ndim != 2 or weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError("shapes of samples and/or weights are incorrect")
        if samples.shape[1] != weights.shape[0]:
            raise ValueError("samples and weights are inconsistent")
        self._weights = weights
        self._samples = samples
        self._nvars = nvars
        self._design_indices = design_indices
        self._random_indices = self._bkd.delete(
            self._bkd.arange(nvars, dtype=int), design_indices
        )
        # warning self._joint_samples must be recomputed if self._samples
        # is changed.
        self._joint_samples = (
            ActiveSetVariableModel._expand_samples_from_indices(
                self._samples,
                self._random_indices,
                self._design_indices,
                self._bkd.zeros((design_indices.shape[0], 1)),
            )
        )

    def nvars(self) -> int:
        # optimizers obtain nvars from here so must be size
        # of design variables
        return self._design_indices.shape[0]

    def _update_attributes(self):
        for attr in [
            "apply_jacobian_implemented",
            "jacobian_implemented",
        ]:
            setattr(self, attr, getattr(self._model, attr))

    def hessian_implemented(self) -> bool:
        return (
            self._model.hessian_implemented()
            and self._stat.hessian_implemented()
        )

    def _random_samples_at_design_sample(self, design_sample):
        # this is slow so only update design samples as self._samples is
        # always fixed
        # return ActiveSetVariableModel._expand_samples_from_indices(
        #     self._samples, self._random_indices, self._design_indices,
        #     design_sample)
        self._joint_samples[self._design_indices, :] = self._bkd.asarray(
            np.repeat(design_sample, self._samples.shape[1], axis=1)
        )
        return self._joint_samples

    def _values(self, design_sample):
        self._check_sample(design_sample)
        samples = self._random_samples_at_design_sample(design_sample)
        values = self._model(samples)
        return self._stat(values, self._weights)

    def _jacobian(self, design_sample):
        samples = self._random_samples_at_design_sample(design_sample)
        # todo take advantage of model prallelism to compute
        # multiple jacobians. Also how to take advantage of
        # adjoint methods that compute model values to then
        # compute jacobian
        # TODO: reuse values if design sample is the same as used to last call
        # to _values
        values = self._model(samples)
        jac_values = self._bkd.array(
            [
                self._model.jacobian(sample[:, None])[:, self._design_indices]
                for sample in samples.T
            ]
        )
        return self._stat.jacobian(values, jac_values, self._weights)

    def _apply_jacobian(self, design_sample, vec):
        samples = self._random_samples_at_design_sample(design_sample)
        # todo take advantage of model prallelism to compute
        # multiple apply_jacs
        expanded_vec = self._bkd.zeros((self._nvars, 1))
        expanded_vec[self._design_indices] = vec
        values = self._model(samples)
        jv_values = self._bkd.array(
            [
                self._model._apply_jacobian(sample[:, None], expanded_vec)
                for sample in samples.T
            ]
        )
        return self._stat.apply_jacobian(values, jv_values, self._weights)

    def _hessian(self, design_sample):
        # TODO: reuse values if design sample is the same as used to last call
        # to _values same for jac_values
        samples = self._random_samples_at_design_sample(design_sample)
        values = self._model(samples)
        jac_values = self._bkd.array(
            [
                self._model.jacobian(sample[:, None])[:, self._design_indices]
                for sample in samples.T
            ]
        )
        idx = np.ix_(self._design_indices, self._design_indices)
        hess_values = self._bkd.array(
            [
                self._model.hessian(sample[:, None])[..., idx[0], idx[1]]
                for sample in samples.T
            ]
        )
        return self._stat.hessian(
            values, jac_values, hess_values, self._weights
        )

    def __repr__(self):
        return "{0}(model={1}, stat={2})".format(
            self.__class__.__name__, self._model, self._stat
        )


class CVaRSampleAverageConstraint(SampleAverageConstraint):
    def __init__(
        self,
        model,
        samples,
        weights,
        stat,
        design_bounds,
        nvars,
        design_indices,
        keep_feasible=False,
    ):
        if not isinstance(stat, SampleAverageConditionalValueAtRisk):
            msg = "stat not instance of SampleAverageConditionalValueAtRisk"
            raise ValueError(msg)
        self._nconstraints = stat._alpha.shape[0]
        super().__init__(
            model,
            samples,
            weights,
            stat,
            design_bounds,
            nvars,
            design_indices,
            keep_feasible,
        )

    def nvars(self) -> int:
        # optimizers obtain nvars from here so must be size
        # of design variables
        return self._design_indices.shape[0] + self._nconstraints

    def apply_jacobian_implemented(self) -> bool:
        # even if model has apply jacobian sample_average_constraint does not
        # so turn off. If it is True then use of jacobian to compute
        # jacobian apply will fail due to passing around VaR
        return False

    def __call__(self, design_sample):
        # have to ovewrite call instead of just defining values
        # to avoid error check that will not work here
        # assumes avar variable t is at the end of design_sample
        self._stat.set_value_at_risk(design_sample[-self._nconstraints :, 0])
        return super().__call__(design_sample[: -self._nconstraints])

    def _jacobian(self, design_sample):
        self._stat.set_value_at_risk(design_sample[-self._nconstraints :, 0])
        jac = super()._jacobian(design_sample[: -self._nconstraints])
        return jac


class ObjectiveWithCVaRConstraints(Model):
    """
    When optimizing for CVaR additional variables t are introduced.
    This class wraps a function that does not take variables t
    and returns a jacobian that includes derivatives with respect to the
    variables t (they will be zero).

    Assumes samples consist of vstack(random_vars, t)
    """

    def __init__(
        self, model: Model, ncvar_constraints: int, ndesign_vars: int
    ):
        super().__init__()
        self._model = model
        self._ndesign_vars = ndesign_vars
        if model.nqoi() != 1:
            raise ValueError("objective can only have one QoI")
        self._ncvar_constraints = ncvar_constraints
        for attr in [
            "apply_jacobian_implemented",
            "jacobian_implemented",
        ]:
            setattr(self, attr, getattr(self._model, attr))
        # until sampleaveragecvar.hessian is implemented turn
        # off objective hessian
        # self._hessian_implemented = self._model.hessian_implemented()

    def nqoi(self) -> int:
        return self._model.nqoi()

    def nvars(self) -> int:
        return self._ndesign_vars + self._ncvar_constraints

    def _values(self, design_samples):
        return self._model(design_samples[: -self._ncvar_constraints])

    def _apply_jacobian(self, design_sample, vec):
        return self._model.apply_jacobian(
            design_sample[: -self._ncvar_constraints],
            vec[: -self._ncvar_constraints],
        )

    def _jacobian(self, design_sample):
        jac = self._model.jacobian(design_sample[: -self._ncvar_constraints])
        return self._bkd.hstack(
            (jac, self._bkd.zeros((jac.shape[0], self._ncvar_constraints)))
        )

    def _hessian(self, design_sample):
        model_hess = self._model.hessian(
            design_sample[: -self._ncvar_constraints]
        )
        nvars = model_hess.shape[-1]
        hess = self._bkd.zeros(
            (
                self.nqoi(),
                nvars + self._ncvar_constraints,
                nvars + self._ncvar_constraints,
            )
        )
        idx = np.ix_(self._bkd.arange(nvars), self._bkd.arange(nvars))
        hess[:, idx[0], idx[1]] = model_hess
        # hess[:, :nvars, :nvars] = model_hess
        return hess


def approx_jacobian(
    func, x, epsilon=np.sqrt(np.finfo(float).eps), bkd=NumpyLinAlgMixin
):
    x0 = bkd.asarray(x)
    assert x0.ndim == 1 or x0.shape[1] == 1
    f0 = bkd.atleast2d(func(x0))
    assert f0.shape[0] == 1
    f0 = f0[0, :]
    jac = bkd.zeros([len(f0), len(x0)])
    dx = bkd.zeros(x0.shape)
    for ii in range(len(x0)):
        dx[ii] = epsilon
        f1 = bkd.atleast2d(func(x0 + dx))
        assert f1.shape[0] == 1
        f1 = f1[0, :]
        jac[:, ii] = (f1 - f0) / epsilon
        dx[ii] = 0.0
    return jac


def approx_hessian(
    jac_fun, x, epsilon=np.sqrt(np.finfo(float).eps), bkd=NumpyLinAlgMixin
):
    return approx_jacobian(lambda y: jac_fun(y).T, x, epsilon, bkd=bkd)


class MiniMaxObjective(SingleSampleModel):
    def __init__(
        self, nmodel_vars: int, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        self._nmodel_vars = nmodel_vars
        super().__init__(backend)

    def nslack(self) -> int:
        return 1

    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._nmodel_vars + self.nslack()

    def _evaluate(self, sample: Array) -> Array:
        return sample[:1]

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.hstack(
            (self._bkd.ones((1,)), self._bkd.zeros((sample.shape[0] - 1,)))
        )[None, :]

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return self._bkd.zeros((sample.shape[0],))


class AVaRObjective(SingleSampleModel):
    def __init__(
        self, nmodel_vars: int, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        self._nmodel_vars = nmodel_vars
        super().__init__(backend)

    def set_beta(self, beta: float):
        self._beta = beta

    def set_quadrature_weights(self, quadw: Array):
        if quadw.ndim != 1:
            raise ValueError("quadw has the wrong shape")
        self._quadw = quadw

    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return 1

    def nslack(self) -> int:
        return 1 + self._quadw.shape[0]

    def nvars(self) -> int:
        return self.nslack() + self._nmodel_vars

    def _evaluate(self, sample: Array) -> Array:
        t_slack = sample[:1]
        gamma_slack = sample[1 : self.nslack()]
        return t_slack + 1.0 / (1.0 - self._beta) * self._quadw @ gamma_slack

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.hstack(
            (
                self._bkd.ones((1,)),
                self._quadw / (1.0 - self._beta),
                self._bkd.zeros((sample.shape[0] - self.nslack(),)),
            ),
        )[None, :]

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return self._bkd.zeros((sample.shape[0],))


class SlackBasedConstraintFromModel(Constraint):
    def __init__(self, model: Model, keep_feasible: bool = False):
        bounds = model._bkd.stack(
            (
                model._bkd.zeros((model.nqoi(),)),
                model._bkd.full((model.nqoi(),), np.inf),
            ),
            axis=1,
        )
        super().__init__(bounds, keep_feasible, model._bkd)
        if not isinstance(model, Model):
            raise ValueError(
                "constraint must be an instance of {0}".format(
                    "pyapprox.interface.model.Model"
                )
            )
        self._model = model

        for attr in [
            "jacobian_implemented",
            "hessian_implemented",
            "weighted_hessian_implemented",
            "apply_weighted_hessian_implemented",
        ]:
            setattr(self, attr, getattr(self._model, attr))

    @abstractmethod
    def nslack(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _values(self, sample: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _jacobian(self, sample: Array) -> Array:
        raise NotImplementedError

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        model_hvp = self._model.apply_hessian(
            sample[self.nslack() :], vec[self.nslack() :]
        )
        return self._bkd.hstack((vec[: self.nslack()] * 0, -model_hvp))

    def apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        model_whvp = self._model.apply_weighted_hessian(
            sample[self.nslack() :], vec[self.nslack() :], weights
        )
        return self._bkd.vstack(
            (self._bkd.zeros((self.nslack(), 1)), -model_whvp)
        )

    def _hessian(self, sample: Array) -> Array:
        model_hess = self._model.hessian(sample[self.nslack() :])
        hess = self._bkd.zeros((self.nqoi(), sample.shape[0], sample.shape[0]))
        hess[:, self.nslack() :, self.nslack() :] = -model_hess
        return hess

    def _weighted_hessian(self, sample: Array, weights: Array) -> Array:
        model_whess = self._model.weighted_hessian(
            sample[self.nslack() :], weights
        )
        whess = self._bkd.zeros((sample.shape[0], sample.shape[0]))
        whess[self.nslack() :, self.nslack() :] = -model_whess
        return whess

    def nqoi(self) -> int:
        return self._model.nqoi()

    def nvars(self) -> int:
        return self._model.nvars() + self.nslack()

    def __repr__(self):
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class AVaRConstraintFromModel(SlackBasedConstraintFromModel):
    def nslack(self) -> int:
        return 1 + self._model.nqoi()

    def _values(self, sample: Array) -> Array:
        return (
            sample[:1].T
            + sample[1 : self.nslack()].T
            - self._model(sample[self.nslack() :])
        )

    def _jacobian(self, sample: Array) -> Array:
        model_jac = self._model.jacobian(sample[self.nslack() :])
        jac = self._bkd.hstack(
            (
                self._bkd.ones((model_jac.shape[0], 1)),
                self._bkd.eye(model_jac.shape[0]),
                -model_jac,
            )
        )
        return jac


class MiniMaxConstraintFromModel(SlackBasedConstraintFromModel):
    def nslack(self) -> int:
        return 1

    def _values(self, sample: Array) -> Array:
        return sample[0] - self._model(sample[1:])

    def _jacobian(self, sample: Array) -> Array:
        model_jac = self._model.jacobian(sample[1:])
        return self._bkd.hstack(
            (self._bkd.ones((model_jac.shape[0], 1)), -model_jac)
        )


class SlackBasedAdjustedConstraint(Constraint):
    def __init__(self, constraint: Constraint):
        if not isinstance(constraint, Constraint):
            raise ValueError("constraint must be an instance of Constraint")
        self._constraint = constraint
        super().__init__(
            constraint._bounds, constraint._keep_feasible, constraint._bkd
        )
        for attr in [
            "jacobian_implemented",
            "hessian_implemented",
            "apply_hessian_implemented",
            "weighted_hessian_implemented",
            "apply_weighted_hessian_implemented",
        ]:
            setattr(self, attr, getattr(self._constraint, attr))

    def _values(self, sample: Array) -> Array:
        return self._constraint(sample[self.nslack() :])

    def _jacobian(self, sample: Array) -> Array:
        con_jac = self._constraint.jacobian(sample[self.nslack() :])
        jac = self._bkd.hstack(
            (
                self._bkd.zeros((con_jac.shape[0], self.nslack())),
                con_jac,
            ),
        )
        return jac

    @abstractmethod
    def nslack(self) -> int:
        raise NotImplementedError


class MiniMaxAdjustedConstraint(SlackBasedAdjustedConstraint):
    def nslack(self) -> int:
        return 1


class AVaRAdjustedConstraint(SlackBasedAdjustedConstraint):
    def nslack(self) -> int:
        return 1


class SlackBasedOptimizer:
    """
    Use slack variables to solve a minimax problem with gradient
    based optimizers
    """

    def __init__(
        self,
        optimizer: ConstrainedOptimizer,
        nslack: int,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                "optimizer must be an instance of ConstrainedOptimizer"
            )
        self._bkd = backend
        self._optimizer = optimizer
        self._nslack = nslack
        self.set_slack_bounds(
            self._bkd.tile(self._bkd.array([-np.inf, np.inf]), (nslack, 1))
        )

    def nslack(self) -> int:
        return self._nslack

    def set_slack_bounds(self, slack_bounds: Array):
        if slack_bounds.shape != (self.nslack(), 2):
            raise ValueError("slack_bounds has the wrong shape")
        self._slack_bounds = slack_bounds

    @abstractmethod
    def _set_objective(self):
        raise NotImplementedError

    @abstractmethod
    def _convert_objective_function(self, model: Model):
        raise NotImplementedError

    def set_objective_function(self, model: Model):
        """
        Set the model that returns the value of the objective at each
        element in the set the maximum (of the min max) is taken over.
        """
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model")
        # create slack based constraints. Model has no knowledge of slack
        # variable so wrapper is created.
        self._constraint_from_objective = self._convert_objective_function(
            model
        )
        self._set_objective()

    def _adjust_nonlinear_constraint(self, con: Constraint):
        return MiniMaxAdjustedConstraint(con)

    def _adjust_constraints(
        self, constraints: List[Constraint]
    ) -> List[Constraint]:
        # adjust constraints to take in slack variable
        adjusted_constraints = []
        for con in constraints:
            if isinstance(con, LinearConstraint):
                A = self._bkd.hstack(
                    (
                        self._bkd.zeros((con.A.shape[0], self.nslack())),
                        self._bkd.asarray(con.A),
                    )
                )
                adjusted_constraints.append(
                    LinearConstraint(A, con.lb, con.ub, con.keep_feasible)
                )
            else:
                adjusted_constraints.append(
                    self._adjust_nonlinear_constraint(con)
                )
        return adjusted_constraints

    def set_constraints(self, constraints: List[Constraint]):
        if not hasattr(self, "_constraint_from_objective"):
            raise RuntimeError(
                "Must first call set_constraint_from_objective_model"
            )
        self._optimizer.set_constraints(
            [self._constraint_from_objective]
            + self._adjust_constraints(constraints)
        )

    def __repr__(self):
        return "{0}(optimizer={1})".format(
            self.__class__.__name__, self._optimizer
        )

    def minimize(self, iterate: Array):
        if not hasattr(self, "_bounds_set"):
            raise ValueError("must call set_bounds")
        res = self._optimizer.minimize(iterate)
        res.slack = res.x[: self.nslack()]
        res.x = res.x[self.nslack() :]
        return res

    def set_bounds(self, bounds: Array):
        # convert bounds to include bounds over slack variable
        self._bounds_set = True
        if bounds is not None:
            self._optimizer.set_bounds(
                self._bkd.vstack((self._slack_bounds, bounds))
            )
        else:
            self._optimizer.set_bounds(self._bkd.vstack(self._slack_bounds))


class MiniMaxOptimizer(SlackBasedOptimizer):
    """
    MinMax optimization with only one slack variable.
    The slack variable replaces the objective. Cannot be used with constraints
    That require additional slack variables
    """

    def __init__(
        self,
        optimizer: ConstrainedOptimizer,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(optimizer, 1, backend)

    def _convert_objective_function(
        self, model: Model
    ) -> MiniMaxConstraintFromModel:
        return MiniMaxConstraintFromModel(model, keep_feasible=False)

    def _set_objective(self):
        objective = MiniMaxObjective(
            self._constraint_from_objective._model.nvars(),
            backend=self._bkd,
        )
        self._optimizer.set_objective_function(objective)


class AVaRSlackBasedOptimizer(SlackBasedOptimizer):
    """
    AVaR optimization with only slack variables arising from replacing the
    objective. Cannot be used with constraints that require additional
    slack variables.
    """

    def __init__(
        self,
        optimizer: ConstrainedOptimizer,
        beta: float,
        quadrature_weights: Array,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._beta = beta
        self.set_quadrature_weights(quadrature_weights)
        super().__init__(optimizer, 1 + self._quadw.shape[0], backend)
        self.set_slack_bounds(
            self._bkd.vstack(
                (
                    self._bkd.array([[-np.inf, np.inf]]),
                    self._bkd.tile(
                        self._bkd.array([0, np.inf]), (self.nslack() - 1, 1)
                    ),
                ),
            )
        )

    def set_quadrature_weights(self, quadw: Array):
        if quadw.ndim != 1:
            raise ValueError("quadw has the wrong shape")
        self._quadw = quadw

    def _convert_objective_function(
        self, model: Model
    ) -> AVaRConstraintFromModel:
        return AVaRConstraintFromModel(model, keep_feasible=False)

    def _set_objective(self):
        objective = AVaRObjective(
            self._constraint_from_objective._model.nvars(), backend=self._bkd
        )
        objective.set_beta(self._beta)
        objective.set_quadrature_weights(self._quadw)
        self._optimizer.set_objective_function(objective)


class _AVaRDummyModel(Model):
    """
    Model with no parameters to be used to compute AVaR from a set of samples.
    Only to be used by EmpiricalAVaRSlackBasedOptimizer
    """

    def __init__(
        self, samples: Array, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        super().__init__(backend)
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError("samples must be 2D row vector")
        self._samples = samples

    def jacobian_implemented(self) -> bool:
        return True

    def weighted_hessian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return self._samples.shape[1]

    def nvars(self) -> int:
        return 0

    def _values(self, samples) -> Array:
        return self._samples

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.zeros((self.nqoi(), self.nvars()))

    def _weighted_hessian(self, sample: Array, weights) -> Array:
        return self._bkd.zeros((self.nvars(), self.nvars()))


class EmpiricalAVaRSlackBasedOptimizer(AVaRSlackBasedOptimizer):
    """
    Compute AVaR from a set of samples using the optimization based formulation
    Only intended for testing and tutorials. If one wants to solve with
    optimization one should solve the equivalent linear program.
    The use of nonlinear optimization here is just to check important
    components of non-linear constrained avar minimization without adding
    the complexity of a model
    """

    def __init__(
        self,
        optimizer: ConstrainedOptimizer,
        beta: float,
        samples: Array,
        quadrature_weights: Array,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(optimizer, beta, quadrature_weights, backend=backend)
        self.set_objective_function(_AVaRDummyModel(samples))
        self.set_constraints([])
