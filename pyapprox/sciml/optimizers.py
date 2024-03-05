from abc import ABC, abstractmethod

import numpy as np
import scipy
import torch.optim

from pyapprox.sciml.util._torch_wrappers import array, asarray, to_numpy, inf


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
    def __init__(self, scipy_result):
        """
        Parameters
        ----------
        scipy_result : :py:class:`scipy.optimize.OptimizeResult`
            The result returned by scipy.minimize
        """
        super().__init__()
        for key, item in scipy_result.items():
            if isinstance(item, np.ndarray):
                self[key] = asarray(item)
            else:
                self[key] = item


class Optimizer(ABC):
    def __init__(self):
        """
        Abstract base Optimizer class.
        """
        self._bounds = None
        self._objective_fun = None
        self._verbosity = 0
        self._tol = 1e-5

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
        # convert bounds to numpy to use numpy random number generator
        bounds = to_numpy(self._bounds)
        return asarray(
            np.random.uniform(bounds[:, 0], bounds[:, 1]))

    def _is_iterate_within_bounds(self, iterate: array):
        # convert bounds to np.logical
        bounds = to_numpy(self._bounds)
        iterate = to_numpy(iterate)
        return np.logical_and(
            iterate >= bounds[:, 0],
            iterate <= bounds[:, 1]).all()

    @abstractmethod
    def optimize(self, iterate: array, num_candidates=1):
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
        raise NotImplementedError


class LBFGSB(Optimizer):
    def __init__(self):
        """
        Use Scipy's L-BGFGS-B to optimize an objective function
        """
        super().__init__()

    def optimize(self, iterate: array, **kwargs):
        """
        Parameters
        ----------
        iterate : array
            Initial iterate for optimizer

        kwargs : **kwargs
            Arguments to Scipy's minimize(method=L-BGFGS-B).
            See Scipy's documentation.
        """
        if not self._is_iterate_within_bounds(iterate):
            raise ValueError('Initial iterate is not within bounds')

        if self._verbosity < 3:
            kwargs['options'] = {'iprint': self._verbosity-1}
        else:
            kwargs['options'] = {'iprint': 200}

        kwargs['tol'] = self._tol

        scipy_res = scipy.optimize.minimize(
            self._objective_fun, to_numpy(iterate), method='L-BFGS-B',
            jac=True, bounds=to_numpy(self._bounds), **kwargs)

        if self._verbosity > 0:
            print(ScipyOptimizationResult(scipy_res))

        return ScipyOptimizationResult(scipy_res)


class Adam(Optimizer):
    def __init__(self, epochs=20, lr=1e-3, batches=1):
        '''
        Use the Adam optimizer
        '''
        super().__init__()
        self._epochs = epochs
        self._lr = lr
        self._batches = batches

    def optimize(self, iterate: array, **kwargs):
        """
        Parameters
        ----------
        iterate : array
            Initial iterate for optimizer

        epochs : int, default 20
            Number of epochs to run optimizer

        lr : float, default 1e-3
            Learning rate

        kwargs : **kwargs
            Arguments to torch.optim.Adam(); see PyTorch documentation.
        """
        adam = torch.optim.Adam([iterate], lr=self._lr, **kwargs)
        fmin = inf
        for ii in range(self._epochs):
            for jj in range(self._batches):
                adam.zero_grad()
                fc, gc = self._objective_fun(
                    iterate, batches=self._batches, batch_index=jj)
                if fc < fmin:
                    fmin = fc
                    xmin = iterate.detach()
                iterate.grad = asarray(gc)
                adam.step()

        res = OptimizationResult({'x': xmin, 'fun': fmin})
        if self._verbosity > 0:
            print(res)

        return res


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
        super().__init__(self)
        self._ncandidates = 1
        self._optimizer = optimizer

    def optimize(self, x0_global: array, num_candidates=1, **kwargs):
        res = self._local_optimize(x0_global)
        xopt, fopt = res.x, res.fun
        for ii in range(1, num_candidates):
            res = self._optimizer(
                self._get_random_optimizer_initial_guess(), **kwargs)
            if res.fun < fopt:
                xopt, fopt = res.x, res.fun
        return asarray(xopt)
