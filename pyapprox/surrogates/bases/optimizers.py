from abc import ABC, abstractmethod

import numpy as np
import scipy
import torch.optim

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


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
            "(\n\t x={0}, \n\t fun={1}, \n\t attr={2})".format(
                self.x, self.fun, list(self.keys())))


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
                self[key] = bkd._la_asarray(item)
            else:
                self[key] = item


class Optimizer(ABC):
    def __init__(self, backend):
        """
        Abstract base Optimizer class.
        """
        self._bkd = backend
        self._bounds = None
        self._objective_fun = None
        self._verbosity = 0
        self._numeric_upper_bound =100

    def set_numeric_upper_bound(self, ub):
        """Set the value used when computing initial guesses to replace np.inf."""
        self._numeric_upper_bound = ub

    def set_objective_function(self, objective_fun):
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
        self._objective_fun = objective_fun

    def set_bounds(self, bounds):
        """
        Set the bounds of the design variables.

        Parameters
        ----------
        bounds : array (ndesign_vars, 2)
            The upper and lower bounds of each design variable
        """
        self._bounds = bounds

    def set_verbosity(self, verbosity):
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

    def set_tolerance(self, tol):
        """
        Set the tolerance that will be passed to the optimizer.

        Parameters
        ----------
        tol : float
            Tolerance (see specific optimizer documentation for details)
        """
        self._tol = tol

    def _get_random_optimizer_initial_guess(self):
        if self._bounds is None:
            raise RuntimeError("must call set_bounds")
        # convert bounds to numpy to use numpy random number generator
        bounds = self._bkd._la_to_numpy(self._bounds)
        bounds[bounds == -np.inf] = -self._numeric_upper_bound
        bounds[bounds == np.inf] = self._numeric_upper_bound
        return self._bkd._la_asarray(
            np.random.uniform(bounds[:, 0], bounds[:, 1]))

    def _is_iterate_within_bounds(self, iterate):
        # convert bounds to np.logical
        bounds = self._bkd._la_to_numpy(self._bounds)
        iterate = self._bkd._la_to_numpy(iterate)
        return np.logical_and(
            iterate >= bounds[:, 0],
            iterate <= bounds[:, 1]).all()

    @abstractmethod
    def _optimize(self, iterate):
        raise NotImplementedError
    
    def optimize(self, iterate):
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
        if self._bounds is None:
            raise RuntimeError("Must call set_bounds")
        if self._objective_fun is None:
            raise RuntimeError("Must call set_objective_function")
        return self._optimize(iterate)

    def __repr__(self):
        return "{0}(verbosity={1})".format(self.__class__.__name__, self._verbosity)


class ScipyLBFGSB(Optimizer):
    def __init__(self, backend=NumpyLinAlgMixin()):
        """
        Use Scipy's L-BGFGS-B to optimize an objective function
        """
        super().__init__(backend=backend)
        self._options = {}

    def _np_objective_fun_wrapper(self, iterate):
        val, grad = self._objective_fun(self._bkd._la_asarray(iterate))
        return val, self._bkd._la_to_numpy(grad)

    def set_options(self, **options):
        """
        Parameters
        ----------kwargs : **kwargs
            Arguments to Scipy's minimize(method=L-BGFGS-B).
            See Scipy's documentation.
        """
        if "iprint" in options:
            raise ValueError("iprint is set by set_verbosity")
        self._options = options

    def _optimize(self, iterate):
        """
        Parameters
        ----------
        iterate : array
            Initial iterate for optimizer

        """
        if not self._is_iterate_within_bounds(iterate):
            raise ValueError('Initial iterate is not within bounds')

        if self._verbosity < 3:
            self._options['iprint'] = self._verbosity-1
        else:
            self._options['iprint'] = 200

        scipy_res = scipy.optimize.minimize(
            self._np_objective_fun_wrapper, self._bkd._la_to_numpy(iterate), method='L-BFGS-B',
            jac=True, bounds=self._bkd._la_to_numpy(self._bounds), options=self._options)

        result = ScipyOptimizationResult(scipy_res, self._bkd)
        
        if self._verbosity > 0:
            print(result)

        return result


class MultiStartOptimizer(Optimizer):
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
        super().__init__(backend=optimizer._bkd)
        self._ncandidates = ncandidates
        self._optimizer = optimizer
        self._bounds = None

    def set_bounds(self, bounds):
        self._bounds = bounds
        self._optimizer.set_bounds(bounds)

    def set_numeric_upper_bound(self, ub):
        self._numeric_upper_bound = ub
        self._optimizer.set_numeric_upper_bound(ub)
        
    def set_objective_function(self, objective_fun):
        self._objective_fun = objective_fun
        self._optimizer.set_objective_function(objective_fun)

    def _optimize(self, x0_global, **kwargs):
        best_res = self._optimizer.optimize(x0_global)
        if self._verbosity > 1:
                print("it {1}: best objective {1}".format(0, best_res.fun))
        for ii in range(1, self._ncandidates):
            print( self._optimizer._get_random_optimizer_initial_guess())
            res = self._optimizer.optimize(
                self._optimizer._get_random_optimizer_initial_guess(), **kwargs)
            if res.fun < best_res.fun:
                best_res = res
            if self._verbosity > 1:
                print("it {0}: best objective {1}".format(ii+1, best_res.fun))
        if self._verbosity > 0:
            print("{0}\n\t {1}".format(self, best_res))
        return best_res

    def __repr__(self):
        return "{0}(optimizer={1}, ncandidates={2})".format(
            self.__class__.__name__, self._optimizer, self._ncandidates
        )
        
        
    
