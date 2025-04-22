from typing import List

import numpy as np
from scipy.optimize import LinearConstraint

from pyrol import (
    Objective,
    Bounds as ROLBounds,
    Vector,
    getCout,
    ParameterList,
    Problem,
    Solver,
    LinearOperator,
    Constraint as ROLConstraint,
    LinearConstraint as ROLLinearConstraint,
)

from pyrol.vectors import NumPyVector
from pyapprox.interface.model import Model
from pyapprox.optimization.minimize import (
    ConstrainedOptimizer,
    OptimizationResult,
    Constraint,
)
from pyapprox.util.backends.template import Array, BackendMixin


class ROLOptimizationResult(OptimizationResult):
    def __init__(self, x, val, bkd: BackendMixin):
        """
        Parameters
        ----------
        scipy_result : :py:class:`scipy.optimize.OptimizeResult`
            The result returned by scipy.minimize
        """
        super().__init__()
        self["x"] = bkd.asarray(x)[:, None]
        self["fun"] = val
        self["success"] = True


class ROLObjectiveWrapper(Objective):
    def __init__(self, model: Model):
        self._bkd = model._bkd
        if not issubclass(model.__class__, Model):
            raise ValueError("model must be derived from Model")
        self._model = model
        for attr in [
            "jacobian_implemented",
            "hessian_implemented",
            "apply_hessian_implemented",
        ]:
            setattr(self, attr, getattr(self._model, attr))
        super().__init__()

    def value(self, x: Vector, tol: float) -> NumPyVector:
        return self._bkd.to_numpy(
            self._model(self._bkd.asarray(x.array)[:, None])
        )[0, 0]

    def gradient(self, g: Vector, x: Vector, tol: float):
        jac = self._model.jacobian(self._bkd.asarray(x.array)[:, None])
        g[:] = self._bkd.to_numpy(jac[0, :])
        return g

    def hessVec(self, hv: Vector, v: Vector, x: Vector, tol: float):
        hvp = self._model.apply_hessian(
            self._bkd.asarray(x.array)[:, None],
            self._bkd.asarray(v.array)[:, None],
        )
        hv[:] = self._bkd.to_numpy(hvp[:, 0])


class ROLNonLinearConstraintWrapper(ROLConstraint):
    """
    Authors:
    -------
    J.D. Jakeman and A. Javeed
    """

    def __init__(self, constraint: Constraint):
        self._bkd = constraint._bkd
        if not issubclass(constraint.__class__, Constraint):
            raise ValueError("constraint must be derived from Constraint")
        self._constraint = constraint
        super().__init__()

    def value(self, c: Vector, x: Vector, tol: float):
        vals = self._constraint(self._bkd.asarray(x.array)[:, None])
        c[:] = vals[0, :]

    def applyJacobian(self, jv: Vector, v: Vector, x: Vector, tol: float):
        # indexing access x.array
        jvp = self._constraint.apply_jacobian(
            self._bkd.asarray(x.array)[:, None],
            self._bkd.asarray(v.array)[:, None],
        )
        jv[:] = jvp[:, 0]

    def applyAdjointJacobian(
        self, jv: Vector, v: Vector, x: Vector, tol: float
    ):
        jac = self._constraint.jacobian(self._bkd.asarray(x.array)[:, None])
        jv[:] = jac.T @ v[:]

    def applyAdjointHessian(
        self,
        hv: Vector,
        u: Vector,
        v: Vector,
        x: Vector,
        tol: float,
    ):
        hvp = self._constraint.apply_weighted_hessian(
            self._bkd.asarray(x.array)[:, None],
            self._bkd.asarray(v.array)[:, None],
            self._bkd.asarray(u.array)[:, None],
        )
        hv[:] = hvp[:, 0]

    def nres(self) -> int:
        return self._constraint.nqoi()


class ROLLinearOperatorWrapper(LinearOperator):
    def __init__(self, Amat: Array, backend: BackendMixin):
        self._bkd = backend
        self._Amat = self._bkd.to_numpy(Amat)

    def apply(self, hv: Vector, v: Vector, tol: float):
        hv[:] = self._Amat @ v[:]

    def applyAdjoint(self, hv: Vector, v: Vector, tol: float):
        hv[:] = self._Amat.T @ v[:]


class ROLLinearConstraintWrapper(ROLLinearConstraint):
    def __init__(self, constraint: LinearConstraint):
        self._bkd = constraint._bkd
        if not issubclass(constraint.__class__, LinearConstraint):
            raise ValueError(
                "constraint must be derived from LinearConstraint"
            )
        self._constraint = constraint
        linop = ROLLinearOperatorWrapper(self._constraint.A)
        b = NumPyVector(
            self._bkd.zeros(
                (self._constraint.A.shape[0]),
            )
        )
        self._nres = self._constraint.A.shape[0]
        super().__init__(linop, b)

    def nres(self) -> int:
        return self._nres


class ROLConstrainedOptimizer(ConstrainedOptimizer):
    def xdefault_parameters(self) -> ParameterList:
        parameters = ParameterList()
        parameters["General"] = ParameterList()
        parameters["General"]["Output Level"] = 1
        parameters["General"]["Polyhedral Projection"] = ParameterList()
        parameters["General"]["Polyhedral Projection"]["Type"] = "Dai-Fletcher"
        parameters["General"]["Polyhedral Projection"]["Iteration Limit"] = 100
        parameters["General"]["Polyhedral Projection"][
            "Absolute Tolerance"
        ] = 1e-14
        parameters["General"]["Polyhedral Projection"][
            "Relative Tolerance"
        ] = 1e-14
        parameters["General"]["Krylov"] = ParameterList()
        parameters["General"]["Krylov"]["Iteration Limit"] = 400
        parameters["General"]["Secant"] = ParameterList()
        parameters["General"]["Secant"]["Type"] = "Limited-Memory BFGS"
        parameters["General"]["Secant"]["Use as Hessian"] = False
        parameters["General"]["Secant"]["Maximum Storage"] = 10
        parameters["Status Test"] = ParameterList()
        parameters["Status Test"]["Iteration Limit"] = 2000
        parameters["Status Test"]["Use Relative Tolerances"] = True
        parameters["Status Test"]["Gradient Tolerance"] = 1e-11
        parameters["Status Test"]["Step Tolerance"] = 1e-14
        parameters["Step"] = ParameterList()
        parameters["Step"]["Type"] = "Trust Region"
        parameters["Step"]["Line Search"] = ParameterList()
        parameters["Step"]["Line Search"]["Descent Method"] = ParameterList()
        parameters["Step"]["Line Search"]["Descent Method"][
            "Type"
        ] = "Quasi-Newton"
        parameters["Step"]["Augmented Lagrangian"] = ParameterList()
        parameters["Step"]["Augmented Lagrangian"][
            "Use Default Initial Penalty Parameter"
        ] = False
        parameters["Step"]["Augmented Lagrangian"][
            "Initial Penalty Parameter"
        ] = 1e5
        parameters["Step"]["Trust Region"] = ParameterList()
        parameters["Step"]["Trust Region"][
            "Subproblem Solver"
        ] = "Truncated CG"
        parameters["Step"]["Trust Region"]["Subproblem Model"] = "Lin-More"
        parameters["Step"]["Trust Region"]["Initial Radius"] = 1.0
        return parameters

    def default_parameters(self) -> ParameterList:
        params = ParameterList()
        params["General"] = ParameterList()
        params["General"]["Output Level"] = 1
        params["Step"] = ParameterList()
        params["Step"]["Trust Region"] = ParameterList()
        params["Step"]["Trust Region"]["Subproblem Solver"] = "Truncated CG"
        return params

    def _set_linear_constraint(self, problem: Problem, _con: Constraint):
        con = ROLLinearConstraintWrapper(_con)
        emul = NumPyVector(self._bkd.zeros(con.nres()))
        if self._bkd.all(_con._bounds[:, 0] == _con._bounds[:, 1]):
            # equality constraints
            problem.addLinearConstraint(
                f"EqLinearConstraint_{self._neqlincons}", con, emul
            )
            self._neqlincons += 1
        else:
            bounds = ROLBounds(
                NumPyVector(self._bkd.to_numpy(_con._bounds[:, 0])),
                NumPyVector(self._bkd.to_numpy(_con._bounds[:, 1])),
            )
            problem.addLinearConstraint(
                f"IneqLinearConstraint_{self._nineqlincons}", con, emul, bounds
            )
            self._nineqlincons += 1

    def _set_nonlinear_constraint(self, problem: Problem, _con: Constraint):
        con = ROLNonLinearConstraintWrapper(_con)
        emul = NumPyVector(np.zeros(con.nres()))
        if self._bkd.all(_con._bounds[:, 0] == _con._bounds[:, 1]):
            # equality constraints
            problem.addConstraint(
                f"EqNonLinearConstraint_{self._neqnonlincons}", con, emul
            )
            self._neqnonlincons += 1
        else:
            bounds = ROLBounds(
                NumPyVector(self._bkd.to_numpy(_con._bounds[:, 0])),
                NumPyVector(self._bkd.to_numpy(_con._bounds[:, 1])),
            )
            problem.addConstraint(
                f"IneqNonLinearConstraint_{self._nineqnonlincons}",
                con,
                emul,
                bounds,
            )
            self._nineqnonlincons += 1

    def _set_constraints(
        self, problem: Problem, constraints: List[Constraint]
    ):
        self._neqlincons = 0
        self._nineqlincons = 0
        self._neqnonlincons = 0
        self._nineqnonlincons = 0
        for _con in constraints:
            if isinstance(_con, LinearConstraint):
                self._set_linear_constraint(problem, _con)
            else:
                self._set_nonlinear_constraint(problem, _con)

    def _minimize(self, init_guess: Vector):
        tol = 0
        x0 = NumPyVector(self._bkd.to_numpy(init_guess[:, 0]))
        objective = ROLObjectiveWrapper(self._objective)
        problem = Problem(objective, x0, x0.dual())
        if self._verbosity > 0:
            stream = getCout()
        else:
            # figure out stream that supresses output
            stream = getCout()

        if self._bkd.any(self._bounds[:, 0] != -np.inf) and self._bkd.any(
            self._bounds[:, 1] != np.inf
        ):
            problem.addBoundConstraint(
                ROLBounds(
                    NumPyVector(self._bkd.to_numpy(self._bounds[:, 0])),
                    NumPyVector(self._bkd.to_numpy(self._bounds[:, 1])),
                )
            )

        self._set_constraints(problem, self._constraints)

        # print attributes of problem
        problem.finalize(False, self._verbosity > 0, stream)

        # use ROL's native check derivatives
        # problem.check(self._verbosity > 0, stream)

        params = self.default_parameters()
        solver = Solver(problem, params)
        if self._verbosity == 0:
            solver.solve()
        else:
            solver.solve(stream)

        # todo try to avoid computing objective value again below
        result = ROLOptimizationResult(
            x0.array, objective.value(x0, tol), self._bkd
        )
        return result
