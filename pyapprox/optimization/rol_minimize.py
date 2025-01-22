from pyapprox.util.linearalgebra.linalgbase import Array

from pyrol import (
    Objective, Bounds, LinearOperator, Problem, Vector, getCout, ParameterList,
    Problem, Solver
)

from pyrol.vectors import NumPyVector
from pyrol.unsupported import Constraint_SimOpt, Objective_SimOpt
from pyapprox.interface.model import Model
from pyapprox.optimization.pya_minimize import (
    ConstrainedOptimizer, OptimizationResult
)


class ROLOptimizationResult(OptimizationResult):
    def __init__(self, x, val, bkd):
        """
        Parameters
        ----------
        scipy_result : :py:class:`scipy.optimize.OptimizeResult`
            The result returned by scipy.minimize
        """
        super().__init__()
        self["x"] = x[:, None]
        self["fun"] = val


class ROLObjectiveWrapper(Objective):
    def __init__(self, model: Model):
        self._bkd = model._bkd
        if not issubclass(model.__class__, Model):
            raise ValueError("model must be derived from Model")
        self._model = model
        for attr in [
            "_jacobian_implemented",
            "_hessian_implemented",
            "_apply_hessian_implemented",
        ]:
            setattr(self, attr, self._model.__dict__[attr])
        super().__init__()

    def value(self, x: Array, tol: float) -> Array:
        return self._model(x[:, None])

    def gradient(self, g: Array, x: Array, tol: float) -> Array:
        jac = self._model.jacobian(x[:, None])
        g[:] = jac[0, :]
        return g

    def hessVec(self, hv: Array, v: Array, x: Array, tol: float) -> Array:
        hvp = self._model._apply_hessian(x[:, None], v[:, None])
        hv[:] = hvp[:, 0]


class ROLConstrainedOptimizer(ConstrainedOptimizer):
    def xdefault_parameters(self) -> ParameterList:
        parameters = ParameterList()
        parameters['General'] = ParameterList()
        parameters['General']['Output Level'] = 1
        parameters['General']['Polyhedral Projection'] = ParameterList()
        parameters['General']['Polyhedral Projection']['Type'] = 'Dai-Fletcher'
        parameters['General']['Polyhedral Projection']['Iteration Limit'] = 100
        parameters['General']['Polyhedral Projection']['Absolute Tolerance'] = 1e-14
        parameters['General']['Polyhedral Projection']['Relative Tolerance'] = 1e-14
        parameters['General']['Krylov'] = ParameterList()
        parameters['General']['Krylov']['Iteration Limit'] = 400
        parameters['General']['Secant'] = ParameterList()
        parameters['General']['Secant']['Type'] = 'Limited-Memory BFGS'
        parameters['General']['Secant']['Use as Hessian'] = False
        parameters['General']['Secant']['Maximum Storage'] = 10
        parameters['Status Test'] = ParameterList()
        parameters['Status Test']['Iteration Limit'] = 2000
        parameters['Status Test']['Use Relative Tolerances'] = True
        parameters['Status Test']['Gradient Tolerance'] = 1e-11
        parameters['Status Test']['Step Tolerance'] = 1e-14
        parameters['Step'] = ParameterList()
        parameters['Step']['Type'] = 'Trust Region'
        parameters['Step']['Line Search'] = ParameterList()
        parameters['Step']['Line Search']['Descent Method'] = ParameterList()
        parameters['Step']['Line Search']['Descent Method']['Type'] = 'Quasi-Newton'
        parameters['Step']['Augmented Lagrangian'] = ParameterList()
        parameters['Step']['Augmented Lagrangian']['Use Default Initial Penalty Parameter'] = False
        parameters['Step']['Augmented Lagrangian']['Initial Penalty Parameter'] = 1e5
        parameters['Step']['Trust Region'] = ParameterList()
        parameters['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
        parameters['Step']['Trust Region']['Subproblem Model'] = 'Lin-More'
        parameters['Step']['Trust Region']['Initial Radius'] = 1.0
        return parameters

    def default_parameters(self) -> ParameterList:
        params = ParameterList()
        params['General'] =  ParameterList()
        params['General']['Output Level'] = 1
        params['Step'] = ParameterList()
        params['Step']['Trust Region'] = ParameterList()
        params['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
        return params

    def _minimize(self, init_guess: Array):
        tol = 0
        x0 = NumPyVector(init_guess[:, 0])
        objective = ROLObjectiveWrapper(self._objective)
        print(objective.value(x0, tol))
        problem = Problem(objective, x0, x0.dual())
        if self._verbosity > 0:
            stream = getCout()
        else:
            # figure out stream that supresses output
            stream = getCout()

        # use ROL's native check derivatives
        # problem.check(True, stream)

        params = self.default_parameters()
        solver = Solver(problem, params)
        solver.solve(stream)
        # todo try to avoid computing objective value again below
        result = ROLOptimizationResult(
            x0.array, objective.value(x0, tol), self._bkd
        )
        return result
