import numpy as np
from scipy.optimize import (
    Bounds,
    NonlinearConstraint,
    LinearConstraint,
    minimize as scipy_minimize,
)
import scipy

from pyapprox.optimize.minimize import (
    OptimizationResult,
    ConstrainedOptimizer,
    ConstraintPenalizedObjective,
)
from pyapprox.interface.model import ScipyModelWrapper


class ScipyOptimizationResult(OptimizationResult):
    def __init__(self, scipy_result, bkd):
        """
        Parameters
        ----------
        scipy_result : :py:class:`scipy.optimize.OptimizeResult`
            The result returned by scipy.minimize
        """
        super().__init__()
        for key, item in scipy_result.items():
            if isinstance(item, np.ndarray):
                self[key] = bkd.asarray(item)
            else:
                self[key] = item


class ScipyConstrainedOptimizer(ConstrainedOptimizer):
    def _convert_constraints(self, constraints):
        scipy_constraints = []
        for _con in constraints:
            if isinstance(_con, LinearConstraint):
                scipy_constraints.append(_con)
                continue
            con = ScipyModelWrapper(_con)
            jac = con.jac if con.jacobian_implemented() else "2-point"
            if (
                con.weighted_hessian_implemented()
                or con.hessian_implemented()
                or con.apply_weighted_hessian_implemented()
            ):
                # model implementation of weighted_hess can use
                # direct implementation of weighted hessian (first condition)
                # or compute it from 3D hessian tensor (2nd condition)
                hess = con.weighted_hess
            else:
                hess = scipy.optimize._hessian_update_strategy.BFGS()
            scipy_con = NonlinearConstraint(
                con, *_con._bounds.T, jac, hess, _con._keep_feasible
            )
            scipy_constraints.append(scipy_con)
        return scipy_constraints

    def _get_bounds(self, nvars):
        if self._bounds is None:
            return Bounds(
                np.full((nvars,), -np.inf),
                np.full((nvars,), np.inf),
                keep_feasible=True,
            )
        return Bounds(
            self._bounds[:, 0], self._bounds[:, 1], keep_feasible=True
        )

    def _minimize(self, init_guess):
        opts = self._opts.copy()
        nvars = init_guess.shape[0]
        bounds = self._get_bounds(nvars)

        objective = ScipyModelWrapper(self._objective)
        jac = objective.jac if objective.jacobian_implemented() else None
        if (
            objective.apply_hessian_implemented()
            or objective.hessian_implemented()
        ):
            hessp = objective.hessp
        else:
            hessp = None

        method = opts.pop("method", "trust-constr")
        if method == "L-BFGS-B":
            if self._verbosity < 3:
                opts["iprint"] = self._verbosity - 1
            else:
                opts["iprint"] = 200
            if len(self._raw_constraints) > 0:
                raise ValueError(
                    "{0} cannot be used with constraints".format(method)
                )
        elif method == "trust-constr":
            opts["verbose"] = self._verbosity
        elif method == "slsqp":
            if self._verbosity > 0:
                opts["disp"] = True
                opts["iprint"] = self._verbosity
        else:
            raise ValueError("method {0} is not supported".format(method))

        scipy_result = scipy_minimize(
            objective,
            init_guess[:, 0],
            method=method,
            jac=jac,
            hessp=hessp,
            bounds=bounds,
            constraints=self._constraints,
            options=opts,
        )
        # use copy to avoid warning:
        # The given NumPy array is not writable ...
        scipy_result.x = np.copy(scipy_result.x[:, None])
        result = ScipyOptimizationResult(scipy_result, self._bkd)
        if self._verbosity > 1:
            print(result)
        return result


class ScipyConstrainedNelderMeadOptimizer(ScipyConstrainedOptimizer):
    def __init__(self, objective=None, constraints=[], bounds=None, opts={}):
        super().__init__(objective, constraints, bounds, opts)
        self._penalty = None

    def _minimize(self, iterate):
        opts = self._opts.copy()
        nvars = iterate.shape[0]
        bounds = self._get_bounds(nvars)
        constrained_objective = ConstraintPenalizedObjective(
            self._objective, self._raw_constraints
        )
        constrained_objective.set_penalty(opts.pop("penalty", 1e8))
        objective = ScipyModelWrapper(constrained_objective)

        scipy_result = scipy_minimize(
            objective,
            iterate[:, 0],
            method="Nelder-Mead",
            bounds=bounds,
            options=self._opts,
        )
        scipy_result.x = scipy_result.x[:, None]
        result = ScipyOptimizationResult(scipy_result, self._bkd)
        if self._verbosity > 1:
            print(result)
        return result
