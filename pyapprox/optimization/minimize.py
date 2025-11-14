from typing import List, Tuple
from abc import ABC, abstractmethod
import textwrap

import numpy as np

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin

from pyapprox.interface.model import (
    Model,
    SingleSampleModel,
    ScalarElementwiseFunction,
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

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + (
                textwrap.indent(
                    "\nx[:, 0]={0}, \nfun={1}, \nattr={2})".format(
                        self.x[:, 0],
                        self.fun,
                        "\n".join(
                            [
                                f"{key}: {item}"
                                for key, item in self.items()
                                if key not in ["fun", "x"]
                            ]
                        ),
                    ),
                    "\t",
                )
            )
            + "\n)"
        )


class Constraint(Model):
    def __init__(
        self,
        bounds: Array = None,
        keep_feasible: bool = False,
        backend: BackendMixin = NumpyMixin,
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

    def get_bounds(self) -> Array:
        return self._bounds


class ConstraintFromModel(Constraint):
    def __init__(
        self, model: Model, bounds: Array = None, keep_feasible: bool = False
    ):
        super().__init__(bounds, keep_feasible, model._bkd)
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
    def __init__(self, backend: BackendMixin):
        self._bkd = backend

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class RandomUniformOptimzerIterateGenerator(OptimizerIterateGenerator):
    def __init__(self, nvars, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)
        self._bounds = None
        self._nvars = nvars
        self._numeric_upper_bound = 100

    def set_bounds(self, bounds):
        bounds = self._bkd.asarray(bounds)
        if bounds.shape[0] == 2 and bounds.ndim != 2:
            bounds = self._bkd.reshape(
                self._bkd.tile(bounds, (self._nvars,)), (self._nvars, 2)
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

    def objective(self) -> Model:
        if self._objective is None:
            raise RuntimeError("set_objective_function ahs not been called")
        return self._objective


class MultiStartOptimizer(OptimizerWithObjective):
    def __init__(
        self,
        optimizer: Optimizer,
        ncandidates: int = 1,
        exit_hard: bool = True,
    ):
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
        self._exit_hard = exit_hard
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
            print("it {0}: best objective {1}".format(1, best_res.fun))
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
            print("{0}\n".format(self) + textwrap.indent(str(best_res), "\t"))
        if not sucess and self._exit_hard:
            raise RuntimeError("All optimizations failed")
        return best_res

    def __repr__(self) -> str:
        return "{0}(optimizer={1}, ncandidates={2})".format(
            self.__class__.__name__, self._optimizer, self._ncandidates
        )


class ConstrainedOptimizer(OptimizerWithObjective):
    def __init__(
        self,
        objective=None,
        constraints=[],
        bounds: Array = None,
        opts: dict = {},
    ):
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
    def __init__(
        self,
        optimizer,
        ncandidates: int = 1,
        exit_hard: bool = True,
    ):
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                "optimizer must be an instance of ConstrainedOptimizer"
            )
        super().__init__(optimizer, ncandidates, exit_hard=exit_hard)

    def set_constraints(self, constraints):
        self._optimizer.set_constraints(constraints)


class ConstraintPenalizedObjective(Model):
    def __init__(
        self,
        unconstrained_objective: Model,
        constraints: List[Constraint],
        enforce_hard: bool = True,
    ):
        super().__init__(unconstrained_objective._bkd)
        self._unconstrained_objective = unconstrained_objective
        self._constraints = constraints
        self._penalty = None
        self._enforce_hard = enforce_hard

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
                    if self._enforce_hard:
                        return (np.inf + con_val * 0)[:, None]
                    # if constraint violated add a penalty
                    else:
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

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__, self._optimizer1, self._optimizer2
        )

    def objective(self) -> Model:
        return self._optimizer1.objective()  # does not matter which one


def approx_jacobian(
    func: callable,
    x: Array,
    epsilon: float = np.sqrt(np.finfo(float).eps),
    bkd: BackendMixin = NumpyMixin,
    forward: bool = True,
) -> Array:
    x0 = bkd.asarray(x)
    assert x0.ndim == 1 or x0.shape[1] == 1
    f0 = bkd.atleast2d(func(x0))
    assert f0.shape[0] == 1
    f0 = f0[0, :]
    jac = bkd.zeros([len(f0), len(x0)])
    dx = bkd.zeros(x0.shape)
    for ii in range(len(x0)):
        dx[ii] = epsilon
        if forward:
            xdx = x0 + dx
        else:
            xdx = x0 - dx
        f1 = bkd.atleast2d(func(xdx))
        assert f1.shape[0] == 1
        f1 = f1[0, :]
        jac[:, ii] = (f1 - f0) / epsilon
        dx[ii] = 0.0
    return jac


def approx_hessian(
    jac_fun: callable,
    x: Array,
    epsilon: float = np.sqrt(np.finfo(float).eps),
    bkd: BackendMixin = NumpyMixin,
) -> Array:
    return approx_jacobian(lambda y: jac_fun(y), x, epsilon, bkd=bkd)


class MiniMaxObjective(SingleSampleModel):
    def __init__(self, nmodel_vars: int, backend: BackendMixin = NumpyMixin):
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
        return self._bkd.zeros((sample.shape[0], vec.shape[1]))


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

    def __repr__(self) -> str:
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


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


class SlackBasedOptimizer:
    """
    Use slack variables to solve a minimax problem with gradient
    based optimizers
    """

    def __init__(
        self,
        optimizer: ConstrainedOptimizer,
        nslack: int,
        backend: BackendMixin = NumpyMixin,
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

    def __repr__(self) -> str:
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
            self._optimizer.set_bounds(self._slack_bounds)


class MiniMaxOptimizer(SlackBasedOptimizer):
    """
    MinMax optimization with only one slack variable.
    The slack variable replaces the objective. Cannot be used with constraints
    That require additional slack variables
    """

    def __init__(
        self,
        optimizer: ConstrainedOptimizer,
        backend: BackendMixin = NumpyMixin,
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


class ADAMOptimizer(OptimizerWithObjective):
    """
    ADAM Optimization Algorithm

    This class implements the ADAM optimization algorithm, which is a popular
    stochastic gradient descent (SGD) variant that adapts the learning rate for
    each parameter based on the magnitude of the gradient.

    Attributes
    ----------
    _learning_rate : float
        The initial learning rate for the optimizer.
    _beta1 : float
        The decay rate for the first moment estimates.
    _beta2 : float
        The decay rate for the second moment estimates.
    _epsilon : float
        A small value added to the denominator for numerical stability.
    _m : Array
        The first moment estimate for the current parameter.
    _v : Array
        The second moment estimate for the current parameter.
    _iter : int
        The current iteration number.
    _maxiters : int
        The maximum number of iterations.
    _gtol: float
        The desired accuracy of the gradient.
    _clip : tuple[float, float]
        Values to clip the gradient from below and above.
    _store: bool
        Flag determining if objective values are stored at each iteration.
    _history: list
        List containing the objective values at each iteration
    """

    def set_options(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        maxiters: int = 100,
        gtol: float = 1e-4,
        clip: Tuple[float, float] = None,
        store: bool = False,
    ):
        """
        Set the options of the ADAM optimizer.

        Parameters
        ----------
        learning_rate : float, optional
            The initial learning rate for the optimizer (default: 0.001).
        beta1 : float, optional
            The decay rate for the first moment estimates (default: 0.9).
        beta2 : float, optional
            The decay rate for the second moment estimates (default: 0.999).
        epsilon : float, optional
            A small value added to the denominator for numerical
            stability (default: 1e-8).
        maxiters : int, optional
           The maximum number of iterations (default: 100).
        gtol: float, optional
           The desired accuracy of the gradient (default: 1e-4).
        clip : tuple[float, float] optional
            Values to clip the gradient from below and above (default: None).
        store: bool, optional
            If true store the value of the objective at each
            iteration (default: False)
        """
        self._learning_rate: float = learning_rate
        self._beta1: float = beta1
        self._beta2: float = beta2
        self._epsilon: float = epsilon
        self._first_mom: Array = None
        self._second_mom: Array = None
        self._iter: int = 0
        self._maxiters: int = maxiters
        self._gtol: float = gtol
        self._clip: Tuple[float, float] = clip
        self._store = store
        self._history = []

    def _minimize(self, iterate: Array) -> OptimizationResult:
        while self._iter < self._maxiters:
            obj = self._objective(iterate)
            if self._store:
                self._history.append(obj[0, 0])
            gradient = self._objective.jacobian(iterate)
            iterate = self.update(iterate, gradient)
            gnorm = self._bkd.norm(gradient)
            if gnorm < self._gtol:
                message = "gtol reached"
                break
        if self._iter >= self._maxiters:
            message = "maxiters reached"
        return OptimizationResult(
            {"fun": obj, "gnorm": gnorm, "x": iterate, "message": message}
        )

    def update(self, iterate: Array, jacobian: Array) -> Array:
        """
        Update the parameters using the ADAM optimization algorithm.

        Parameters
        ----------
        iterate : Array
            The current parameters.
        jacobian : Array
            The jacobian of the loss function with respect to the parameters.
        """
        if self._first_mom is None:
            self._first_mom = self._bkd.zeros(iterate.shape)
            self._second_mom = self._bkd.zeros(iterate.shape)

        if jacobian.shape[0] != 1:
            raise ValueError("jacobian must be a 2D array with one row")
        gradient = jacobian.T
        self._iter += 1
        self._first_mom = self._beta1 * self._first_mom + (1 - self._beta1) * (
            gradient
            if self._clip is None
            else self._bkd.clip(gradient, self._clip[0], self._clip[1])
        )
        self._second_mom = self._beta2 * self._second_mom + (
            1 - self._beta2
        ) * (
            gradient**2
            if self._clip is None
            else self._bkd.clip(gradient, self._clip[0], self._clip[1]) ** 2
        )

        # correct for bias
        mhat: Array = self._first_mom / (1 - self._beta1**self._iter)
        vhat: Array = self._second_mom / (1 - self._beta2**self._iter)

        # update iterate
        new_iterate = iterate - (
            self._learning_rate * mhat / (self._bkd.sqrt(vhat) + self._epsilon)
        )
        return new_iterate

    def zero_grad(self):
        """
        Reset the gradient estimates to zero.
        """
        self._first_mom = None
        self._second_mom = None

    def get_history(self) -> Array:
        if not self._store:
            raise ValueError("No history available. Store was not set to true")
        return self._bkd.asarray(self._history)


class StochasticGradientDescentOptimizer(OptimizerWithObjective):
    """
    Gradinet Descent Optimizer

    This class implements a basic gradient descent algorithm for optimizing
    objective functions.
    It uses a specified backend (e.g., NumPy or PyTorch) to compute gradients
    and update parameters.

    Attributes
    ----------
    _iter : int
        The current iteration number.
    _learn_rate : float
        Learning rate for gradient descent.
    _maxiters : int
        The maximum number of iterations.
    _gtol: float
        The desired accuracy of the gradient.
    _clip : tuple[float, float]
            Values to clip the gradient from below and above (default: None).
    _store: bool
        Flag determining if objective values are stored at each iteration.
    _history: list
        List containing the objective values at each iteration
    """

    def set_options(
        self,
        learning_rate: float = 0.001,
        maxiters: int = 100,
        gtol: float = 1e-4,
        clip: Tuple[float, float] = None,
        store: bool = False,
    ):
        """
        Set the options of the Stochastic Gradient Descent optimizer.

        Parameters
        ----------
        learning_rate : float, optional
            The initial learning rate for the optimizer (default: 0.001).
            maxiters : int, optional
           The maximum number of iterations (default: 100).
        gtol: float, optional
           The desired accuracy of the gradient (default: 1e-4).
        clip : tuple[float, float] optional
            Values to clip the gradient from below and above (default: None).
        store: bool, optional
            If true store the value of the objective at each
            iteration (default: False)
        """
        self._learning_rate: float = learning_rate
        self._iter: int = 0
        self._maxiters: int = maxiters
        self._gtol: float = gtol
        self._clip: Tuple[float, float] = clip
        self._store = store
        self._history = []

    def _step_from_objective(
        self, objective: Model, iterate: Array
    ) -> Tuple[float, Array]:
        """
        Compute a single gradient descent step from an objective function.

        Parameters
        ----------
        objective : Callable[[Array], Array]
            Objective function to optimize.
        iterate : Array
            Current iterate.

        Returns
        -------
        val, grad, iterate : float, Array
            Value of the objective function, its gradient, and updated iterate.
        """
        val = objective(iterate)
        grad = objective.jacobian(iterate).T
        new_iterate = iterate - self._learning_rate * grad
        return val, grad, new_iterate

    def _step(self, iterate: Array) -> Tuple[float, Array]:
        """
        Compute a single gradient descent step using the internal objective
        function.

        Parameters
        ----------
        iterate : Array
            Current iterate.

        Returns
        -------
        val, iterate : float, Array
            Value of the objective function and updated iterate.
        """
        self._val, self._grad, new_iterate = self._step_from_objective(
            self._objective, iterate
        )
        if self._store:
            self._history.append(self._val[0, 0])
        self._iter += 1
        return new_iterate

    def _prepare_result(
        self, iterate: Array, message: str
    ) -> OptimizationResult:
        """
        Prepare the optimization result.

        Parameters
        ----------
        iterate : Array
            Current iterate.

        Returns
        -------
        result : OptimizationResult
            Optimization result.
        """
        result = OptimizationResult()
        result["x"] = iterate
        result["fun"] = self._val
        result["gnorm"] = self._bkd.norm(self._grad)
        result["message"] = message
        return result

    def _minimize(self, iterate: Array) -> OptimizationResult:
        """
        Perform gradient descent optimization.

        Args:
            iterate (Array): Initial iterate.

        Returns:
            OptimizationResult: Optimization result.
        """
        self._iter = 0
        while self._iter < self._maxiters:
            iterate = self._step(iterate)
            gnorm = self._bkd.norm(self._grad)
            if gnorm < self._gtol:
                message = "gtol reached"
                break
            if self._verbosity > 1 and self._iter % 10 == 0:
                message = "it: {0}, val: {1}, gnorm: {2}".format(
                    self._iter, self._val, gnorm
                )
                print(message)
        if self._iter >= self._maxiters:
            message = "maxiters reached"
        return self._prepare_result(iterate, message)

    def get_history(self) -> Array:
        if not self._store:
            raise ValueError("No history available. Store was not set to true")
        return self._bkd.asarray(self._history)


class ChainRuleArrays:
    def __init__(
        self, compress_dx_dp: bool, compress_du_dx: bool, backend: BackendMixin
    ):
        self._compress_dx_dp = compress_dx_dp
        self._compress_du_dx = compress_du_dx
        self._bkd = backend

    def _validate_shapes(
        self,
        p_shape: Tuple[int, int],
        x_shape: Tuple[int, int],
        u_shape: Tuple[int, int],
        dx_dp: Array,
        du_dx: Array,
    ):
        """
        Validate the shapes of p, x, u, dx_dp, and du_dx.

        Parameters
        ----------
        p_shape: Tuple[int, int]
            Shape of input tensor (shape: N x n_p)
        x_shape: Tuple[int, int]
            Shape of intermediate tensor (shape: N x n_i)
        u_shape: Tuple[int, int]
            Shape of Output tensor (shape: N x n_o)
        dx_dp: Array
            Jacobian of x with respect to p (shape: N x n_i x n_p)
        du_dx: Array
            Jacobian of u with respect to x (shape: N x n_o x n_i)
        """
        N, n_p = p_shape
        _, n_i = x_shape
        _, n_o = u_shape

        if x_shape[0] != N or u_shape[0] != N:
            raise ValueError(
                "Batch size mismatch between p, x, and u."
                f"shapes were {p_shape=}, {x_shape=}, {u_shape}"
            )

        if dx_dp.shape != (N, n_i, n_p):
            raise ValueError(
                f"Shape of dx_dp is {dx_dp.shape}, expected "
                f"({N}, {n_i}, {n_p})."
            )
        if du_dx.shape != (N, n_o, n_i):
            raise ValueError(
                f"Shape of du_dx is {du_dx.shape}, expected "
                f"({N}, {n_o}, {n_i})."
            )

    def _compress_jacobian(self, jac: Array, name: str) -> Array:
        """
        Compress batch Jacobians by summing over the redundant batch dimension.

        Parameters
        ----------
        jac: Array
            Jacobian tensor with redundant batch dimension (e.g., N x n_o x N x n_i)

        Returns
        -------
        compressed_jac: Array
            Compressed Jacobian tensor (e.g., N x n_o x n_i)
        """
        if jac.ndim != 4 or jac.shape[0] != jac.shape[2]:
            raise ValueError(
                f"Shape of {name} is {jac.shape}, expected "
                f"({jac.shape[0]}, n_i, {jac.shape[0]}, n_p)."
            )
        return self._bkd.sum(jac, axis=2)

    def _uncompress_jacobian(self, jac_compressed: Array) -> Array:
        """
        Expand a compressed Jacobian (N x n_o x n_p) into an uncompressed
        Jacobian (N x n_o x N x n_p) using NumPy.

        Parameters
        ----------
        jac_compressed: Array
            Compressed Jacobian (shape: N x n_o x n_p)

        Returns
        -------
        jac_uncompressed: Array
            Uncompressed Jacobian (shape: N x n_o x N x n_p)
        """
        N, n_o, n_p = jac_compressed.shape
        jac_uncompressed = self._bkd.zeros((N, n_o, N, n_p))

        # Populate the diagonal blocks
        for n in range(N):
            jac_uncompressed[n, :, n, :] = jac_compressed[n]

        return jac_uncompressed

    def _apply_chain_rule(self, dx_dp: Array, du_dx: Array) -> Array:
        """
        Apply the chain rule using einsum.

        Parameters
        ----------
        dx_dp: Array
            Jacobian of x with respect to p (shape: N x n_i x n_p)
        du_dx: Array
            Jacobian of u with respect to x (shape: N x n_o x n_i)

        Returns
        -------
        du_dp: Array
            Derivative of u with respect to p (shape: N x n_o x n_p)
        """
        return self._bkd.einsum(
            "noi,nip->nop", du_dx, dx_dp
        )  # Shape: N x n_o x n_p

    def set_arrays(
        self,
        x_shape: Tuple[int, int],
        u_shape: Tuple[int, int],
        dx_dp: Array,
        du_dx: Array,
    ):
        """
        Set the data needed to compute the chain rule.

        Parameters
        ----------
        x_shape: Tuple[int, int]
            Shape of precomputed intermediate tensor (shape: N x n_i)
        u_shape: Tuple[int, int]
            Shape of precomputed output tensor (shape: N x n_o)
        dx_dp: Array
            Precomputed Jacobian of x with respect to p (shape: N x n_i x n_p)
        du_dx: Array
            Precomputed Jacobian of u with respect to x (shape: N x n_o x n_i)
        """
        self._x_shape = x_shape
        self._u_shape = u_shape
        if self._compress_du_dx:
            du_dx = self._compress_jacobian(du_dx, "du_dx")
        if self._compress_dx_dp:
            dx_dp = self._compress_jacobian(dx_dp, "dx_dp")

        self._dx_dp = dx_dp  # Shape: N x n_i x n_p
        self._du_dx = du_dx  # Shape: N x n_o x n_i

    def __call__(self, p_shape: Tuple[int, int]) -> Array:
        """
        Compute the derivative of u(x(p)) with respect to p using the chain rule.

        Parameters
        ----------
        p_shape: Tuple[int, int]
            Shape of the design parameters (shape: N x n_p)

        Returns:
        -------
        du_dp: Array
            Derivative of u with respect to p (shape: N x n_o x n_p)
        """
        # Check arrays have been setUp
        if not hasattr(self, "_u_shape"):
            raise RuntimeError("must call set_arrays")

        # Validate shapes
        self._validate_shapes(
            p_shape, self._x_shape, self._u_shape, self._dx_dp, self._du_dx
        )

        # Apply chain rule
        return self._apply_chain_rule(self._dx_dp, self._du_dx)


class ChainRuleFunctions(ChainRuleArrays):
    def __init__(
        self,
        x_function,
        u_function,
        x_jac,
        u_jac,
        compress_dx_dp: bool,
        compress_du_dx: bool,
        backend: BackendMixin,
    ):
        """
        Initialize the ChainRuleFunctions class.

        Parameters:
        ----------
        x_function: callable
            Function x(p), maps p to intermediate x (shape: N x n_i)
        u_function: callable
            Function u(x), maps x to output u (shape: N x n_o)
        x_jac: callable
            Function to compute the Jacobian of x with respect to p
            (shape: N x n_i x n_p)
        u_jac: callable
            Function to compute the Jacobian of u with respect to x
            (shape: N x n_o x n_i)
        compress_dx_dp: bool
            If true compress dx_dp from 4D to 3D tensor
        compress_du_dx: bool
            If true compress du_dx from 4D to 3D tensor
        backend: BackendMixin
            The backend used to peform Array manipulations
        """
        super().__init__(compress_dx_dp, compress_du_dx, backend)
        self._x_function = x_function
        self._u_function = u_function
        self._x_jac = x_jac
        self._u_jac = u_jac

    def __call__(self, p: Array) -> Array:
        """
        Compute the derivative of u(x(p)) with respect to p using the chain
        rule.

        Parameters:
        ----------
        p: Array
            The design parameters (shape: N x n_p)

        Returns:
        -------
        du_dp: Array
            Derivative of u with respect to p (shape: N x n_o x n_p)
        """
        # Compute x(p) and u(x)
        x = self._x_function(p)  # Shape: N x n_i
        u = self._u_function(x)  # Shape: N x n_o

        # Compute Jacobians
        dx_dp = self._x_jac(p)
        du_dx = self._u_jac(x)
        self.set_arrays(x.shape, u.shape, dx_dp, du_dx)
        return super().__call__(p.shape)


class SmoothLogBasedMaxFunction(ScalarElementwiseFunction):
    def __init__(
        self,
        ndim: int,
        eps: float,
        backend: BackendMixin,
        threshold: float = 1e2,
        shift: float = 0.0,
    ):
        self._eps = eps
        self._thresh = threshold
        self._shift = shift
        super().__init__(ndim, backend)

    def jacobian_implemented(self) -> bool:
        return True

    def _check_samples(self, samples: Array):
        if samples.ndim != 2:
            raise ValueError("samples must be a 2D array")

    def _values(self, samples: Array) -> Array:
        x = samples + self._shift
        x_div_eps = x / self._eps
        # avoid overflow
        vals = self._bkd.zeros(x.shape)
        II = self._bkd.where(
            (x_div_eps < self._thresh) & (x_div_eps > -self._thresh)
        )
        vals[II] = x[II] + self._eps * self._bkd.log(
            1 + self._bkd.exp(-x_div_eps[II] - self._shift / self._eps)
        )
        J = self._bkd.where(x_div_eps >= self._thresh)
        vals[J] = x[J]
        return vals

    def first_derivative_implemented(self) -> bool:
        return True

    def _first_derivative(self, samples: Array) -> Array:
        # samples (noutputs, nsamples)
        # jac_values (nsamples, noutputs, noutputs)
        # but only return (nsamples, noutputs) because jac for each sample
        # is just a diagonal matrix
        x = samples + self._shift
        x_div_eps = x / self._eps
        # Avoid overflow.
        II = self._bkd.where(
            (x_div_eps < self._thresh) & (x_div_eps > -self._thresh)
        )
        jac = self._bkd.zeros((x_div_eps.shape))
        jac[II] = 1.0 / (
            1 + self._bkd.exp(-x_div_eps[II] - self._shift / self._eps)
        )
        jac[x_div_eps >= self._thresh] = 1.0
        return jac

    def second_derivative_implemented(self) -> bool:
        return True

    def _second_derivative(self, samples: Array) -> Array:
        x = samples + self._shift
        x_div_eps = x / self._eps
        # Avoid overflow.
        II = self._bkd.where(
            (x_div_eps < self._thresh) & (x_div_eps > -self._thresh)
        )
        vals = self._bkd.zeros(x.shape)
        vals[II] = self._bkd.exp(x_div_eps[II] + self._shift / self._eps) / (
            self._eps
            * (self._bkd.exp(x_div_eps[II] + self._shift / self._eps) + 1) ** 2
        )
        return vals

    def third_derivative_implemented(self) -> bool:
        return True

    def _third_derivative(self, samples: Array) -> Array:
        x = samples + self._shift
        x_div_eps = x / self._eps
        # Avoid overflow.
        II = self._bkd.where(
            (x_div_eps < self._thresh) & (x_div_eps > -self._thresh)
        )
        vals = self._bkd.zeros(x.shape)
        vals[II] = self._bkd.exp(x_div_eps[II] + self._shift / self._eps) / (
            self._eps**2
            * (1 + self._bkd.exp(x_div_eps[II] + self._shift / self._eps)) ** 2
        )
        vals[II] -= (
            2
            * self._bkd.exp(x_div_eps[II] + self._shift / self._eps) ** 2
            / (
                self._eps**2
                * (1 + self._bkd.exp(x_div_eps[II] + self._shift / self._eps))
                ** 3
            )
        )
        return vals


# create base class mixin for checking if argument is correct
# when passing to FSD. This will allow for other smooth heaviside
# functions which I have yet to convert to this API
class SmoothLeftHeavisideFunction:
    pass


class SmoothLogBasedRightHeavisideFunction(ScalarElementwiseFunction):
    r"""
    Smooth left heaviside function :math:`1_{(-\infty, 0]}`

    The smooth log-based right heaviside function is the first derivative
    of the smooth log-based max function m(x)=smoothmax(0, x).
    Thus, the smooth log based left heaviside function is the derivative of
    m(-x).
    """

    def __init__(
        self,
        ndim: int,
        eps: float,
        backend: BackendMixin,
        threshold: float = 1e2,
        shift: float = 0,
    ):
        super().__init__(ndim, backend)
        self._shift = shift

        self._max_function = SmoothLogBasedMaxFunction(
            ndim, eps, self._bkd, threshold, self._shift
        )

    def _values(self, samples: Array) -> Array:
        return self._max_function.first_derivative(samples)

    def first_derivative_implemented(self) -> bool:
        return True

    def _first_derivative(self, samples: Array) -> Array:
        return self._max_function.second_derivative(samples)

    def second_derivative_implemented(self) -> bool:
        return True

    def _second_derivative(self, samples: Array) -> Array:
        return self._max_function.third_derivative(samples)


class SmoothLogBasedLeftHeavisideFunction(
    SmoothLogBasedRightHeavisideFunction, SmoothLeftHeavisideFunction
):
    def __init__(
        self,
        ndim: int,
        eps: float,
        backend: BackendMixin,
        threshold: float = 1e2,
        shift: float = 0,
    ):
        super().__init__(ndim, eps, backend, threshold, shift)

    def _values(self, samples: Array) -> Array:
        return super()._values(-samples)

    def first_derivative_implemented(self) -> bool:
        return True

    def _first_derivative(self, samples: Array) -> Array:
        return -super()._first_derivative(-samples)

    def second_derivative_implemented(self) -> bool:
        return True

    def _second_derivative(self, samples: Array) -> Array:
        return super()._second_derivative(-samples)


class SmoothQuarticBasedLeftHeavisideFunction(
    ScalarElementwiseFunction, SmoothLeftHeavisideFunction
):
    def __init__(
        self,
        ndim: int,
        eps: float,
        backend: BackendMixin,
    ):
        super().__init__(ndim, backend)
        self._eps = eps

    def _values(self, samples: Array):
        shift = 0
        x = samples + shift

        # Create an array of ones with the same shape as x
        vals = self._bkd.ones(x.shape)

        # Compute masks for the conditions
        mask_neg = (x < 0) & (x > -self._eps)  # Elements where -eps < x < 0
        mask_zero = x >= 0  # Elements where x >= 0

        # Apply the quartic function for elements satisfying -eps < x < 0
        vals[mask_neg] = (
            6 * (-x[mask_neg] / self._eps) ** 2
            - 8 * (-x[mask_neg] / self._eps) ** 3
            + 3 * (-x[mask_neg] / self._eps) ** 4
        )

        # Set values to 0 for elements where x >= 0
        vals[mask_zero] = 0

        return vals

    def first_derivative_implemented(self) -> bool:
        return True

    def _first_derivative(self, samples: Array) -> Array:
        shift = 0
        x = samples + shift
        # Create an array of zeros with the same shape as x
        vals = self._bkd.zeros(x.shape)

        # Compute mask for the condition (-eps < x < 0)
        mask_neg = (x < 0) & (x > -self._eps)

        # Apply the derivative formula for elements satisfying -eps < x < 0
        vals[mask_neg] = (
            12 * x[mask_neg] * (self._eps + x[mask_neg]) ** 2 / self._eps**4
        )

        return vals

    def second_derivative_implemented(self) -> bool:
        return True

    def _second_derivative(self, samples: Array) -> Array:
        shift = 0
        x = samples + shift

        # Create an array of zeros with the same shape as x
        vals = self._bkd.zeros(x.shape)

        # Compute mask for the condition (-eps < x < 0)
        mask_neg = (x < -np.finfo(float).eps) & (x > -self._eps)

        # Apply the second derivative formula for elements satisfying -eps < x < 0
        vals[mask_neg] = (
            12
            * (
                self._eps**2
                + 4 * self._eps * x[mask_neg]
                + 3 * x[mask_neg] ** 2
            )
            / self._eps**4
        )
        return vals


class SmoothQuinticBasedLeftHeavisideFunction(
    ScalarElementwiseFunction, SmoothLeftHeavisideFunction
):
    def __init__(
        self,
        ndim: int,
        eps: float,
        backend: BackendMixin,
    ):
        super().__init__(ndim, backend)
        self._eps = eps

    def _values(self, samples: Array):
        shift = 0
        x = samples + shift

        # Create an array of ones with the same shape as x
        vals = self._bkd.ones(x.shape)

        # Coefficients
        c3, c4, c5 = 10, -15, 6

        # Compute masks for the conditions
        mask_neg = (x < 0) & (
            x > -self._eps
        )  # Elements where -self._eps < x < 0
        mask_zero = x >= 0  # Elements where x >= 0

        # Apply the quintic function for elements satisfying -self._eps < x < 0
        xe = -x[mask_neg] / self._eps
        vals[mask_neg] = c3 * xe**3 + c4 * xe**4 + c5 * xe**5

        # Set values to 0 for elements where x >= 0
        vals[mask_zero] = 0

        return vals

    def first_derivative_implemented(self) -> bool:
        return True

    def _first_derivative(self, samples: Array) -> Array:
        shift = 0
        x = samples + shift

        # Create an array of zeros with the same shape as x
        vals = self._bkd.zeros(x.shape)

        # Coefficients
        c3, c4, c5 = 10, -15, 6

        # Compute mask for the condition (-self._eps < x < 0)
        mask_neg = (x < 0) & (x > -self._eps)

        # Apply the first derivative formula for elements satisfying -self._eps < x < 0
        xe = -x[mask_neg] / self._eps
        vals[mask_neg] = (
            -(3 * c3 * xe**2 + 4 * c4 * xe**3 + 5 * c5 * xe**4) / self._eps
        )

        return vals

    def second_derivative_implemented(self) -> bool:
        return True

    def _second_derivative(self, samples: Array) -> Array:
        shift = 0
        x = samples + shift

        # Create an array of zeros with the same shape as x
        vals = self._bkd.zeros(x.shape)

        # Coefficients of spline enforcing all derivatives of step function at 0 and eps
        c3, c4, c5 = 10, -15, 6

        # Compute mask for the condition (-self._eps < x < 0)
        mask_neg = (x < 0) & (x > -self._eps)

        # Apply the second derivative formula for elements satisfying -self._eps < x < 0
        xe = -x[mask_neg] / self._eps
        vals[mask_neg] = (
            6 * c3 * xe + 12 * c4 * xe**2 + 20 * c5 * xe**3
        ) / self._eps**2
        return vals
