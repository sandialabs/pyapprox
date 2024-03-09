import numpy as np
from scipy.optimize import minimize as scipy_minimize
import scipy

from pyapprox.util.sys_utilities import package_available
if package_available("ROL"):
    has_ROL = True
    from pyapprox.optimization._rol_minimize import rol_minimize
else:
    has_ROL = False


def pyapprox_minimize(fun, x0, args=(), method='rol-trust-constr', jac=None,
                      hess=None, hessp=None, bounds=None, constraints=(),
                      tol=None, callback=None, options={}, x_grad=None):
    options = options.copy()
    if x_grad is not None and 'rol' not in method:
        # Fix this limitation
        msg = f"Method {method} does not currently support gradient checking"
        # raise Exception(msg)
        print(msg)

    if 'rol' in method and has_ROL:
        if callback is not None:
            raise ValueError(f'Method {method} cannot use callbacks')
        if args != ():
            raise ValueError(f'Method {method} cannot use args')
        rol_methods = {'rol-trust-constr': None}
        if method in rol_methods:
            rol_method = rol_methods[method]
        else:
            raise ValueError(f"Method {method} not found")
        return rol_minimize(
            fun, x0, rol_method, jac, hess, hessp, bounds, constraints, tol,
            options, x_grad)

    x0 = x0.squeeze()  # scipy only takes 1D np.ndarrays
    x0 = np.atleast_1d(x0)  # change scalars to np.ndarrays
    assert x0.ndim <= 1
    if method == 'rol-trust-constr' and not has_ROL:
        print('ROL requested by not available switching to scipy.minimize')
        method = 'trust-constr'
    
    if method == 'trust-constr':
        if 'ctol' in options:
            del options['ctol']
        return scipy_minimize(
            fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol,
            callback, options)
    elif method == 'slsqp':
        hess, hessp = None, None
        if 'ctol' in options:
            del options['ctol']
        if 'gtol' in options:
            ftol = options['gtol']
            del options['gtol']
        options['ftol'] = ftol
        if 'verbose' in options:
            verbose = options['verbose']
            options['disp'] = verbose
            del options['verbose']
        return scipy_minimize(
            fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol,
            callback, options)

    raise ValueError(f"Method {method} was not found")

from abc import ABC, abstractmethod
from pyapprox.interface.model import Model, ScipyModelWrapper
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint


class Constraint(Model):
    def __init__(self, model, bounds, keep_feasible=False):
        if not isinstance(model, Model):
            raise ValueError("objective must be an instance of {0}".format(
                "pyapprox.interface.model.Model"))
        self._model = model
        if (not isinstance(bounds, np.ndarray) or bounds.ndim != 2
                or bounds.shape[1] != 2):
            raise ValueError("bounds must be 2D np.ndarray with two columns")
        self._bounds = bounds
        self._keep_feasible = keep_feasible
        for attr in ["_apply_jacobian_implemented", "_jacobian_implemented",
                     "_jacobian_implemented", "_hessian_implemented",
                     "_apply_hessian_implemented", "jacobian",
                     "apply_jacobian", "apply_hessian", "hessian"]:
            setattr(self, attr, getattr(self._model, attr))

    def __call__(self, sample):
        return self._model(sample)

    def __repr__(self):
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class Optimizer(ABC):
    def __init__(self, objective, bounds=None, opts={}):
        if not isinstance(objective, Model):
            raise ValueError("objective must be an instance of {0}".format(
                "pyapprox.interface.model.Model"))
        self._objective = objective
        if bounds is not None and not isinstance(bounds, Bounds):
            raise ValueError(
                "bounds must be an instance of scipy.minimize.Bounds")
        self._bounds = bounds
        self._opts = self._parse_options(opts)

    def _parse_options(self, opts):
        return opts

    @abstractmethod
    def minimize(self, init_guess):
        raise NotImplementedError


class ConstrainedOptimizer(Optimizer):
    def __init__(self, objective, constraints=[], bounds=None, opts={}):
        super().__init__(objective, bounds, opts)
        for con in constraints:
            if (not isinstance(con, Constraint) and
                    not isinstance(con, LinearConstraint)):
                raise ValueError(
                    "constraint must be an instance of {0} or {1}".format(
                        "pyapprox.optimize.pya_minimize.Constraint",
                        "scipy.optimize.LinearConstraint"))
        self._constraints = self._convert_constraints(constraints)

    def _convert_constraints(self, constraints):
        return constraints


class ScipyConstrainedOptimizer(ConstrainedOptimizer):
    def _convert_constraints(self, constraints):
        scipy_constraints = []
        for _con in constraints:
            if isinstance(_con, LinearConstraint):
                scipy_constraints.append(_con)
                continue
            con = ScipyModelWrapper(_con)
            jac = con.jac if con._jacobian_implemented else "2-point"
            hess = scipy.optimize._hessian_update_strategy.BFGS()
            scipy_con = NonlinearConstraint(
                con, *_con._bounds.T, jac, hess, _con._keep_feasible)
            scipy_constraints.append(scipy_con)
        return scipy_constraints

    def minimize(self, init_guess):
        opts = self._opts.copy()
        nvars = init_guess.shape[0]
        if self._bounds is None:
            bounds = Bounds(
                np.full((nvars,), -np.inf), np.full((nvars,), np.inf))
        else:
            bounds = self._bounds
        objective = ScipyModelWrapper(self._objective)
        jac = (objective.jac if objective._jacobian_implemented else None)
        hessp = (objective.hessp
                 if objective._apply_hessian_implemented else None)
        if hessp is None and self._objective._apply_hessian_implemented:
            hess = self._objective.hessian
        else:
            hess = None
        result = scipy_minimize(
            objective, init_guess[:, 0],
            method=opts.pop("method", "trust-constr"),
            jac=jac, hess=hess, hessp=hessp, bounds=bounds,
            constraints=self._constraints, options=self._opts)
        return result
