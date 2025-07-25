"""
Linear System Solvers for Regression Problems

This module provides classes for solving linear systems in regression problems. It includes
an abstract base class (`LinearSystemSolver`) that defines the interface for solving linear
systems, as well as specific implementations such as least squares regression (`LstSqSolver`).
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from scipy.optimize import linprog
from pyapprox.optimization.risk import (
    AverageValueAtRisk,
    SafetyMarginRiskMeasure,
    EntropicRisk,
    RiskMeasure,
)
from pyapprox.interface.model import SingleSampleModel
from pyapprox.optimization.scipy import (
    ConstrainedOptimizer,
    ScipyConstrainedOptimizer,
)
from pyapprox.util.sys_utilities import package_available


if package_available("cvxopt"):
    from cvxopt import matrix, solvers, spmatrix, sparse


class LinearSystemSolver(ABC):
    """
    Abstract base class for solving linear systems.

    The `LinearSystemSolver` class provides an interface for optimizing the coefficients
    of a linear system. Subclasses must implement the `solve` method to provide specific
    optimization techniques.
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the LinearSystemSolver.

        Parameters
        ----------
        backend : BackendMixin, optional
            Backend for numerical operations (default is `NumpyMixin`).
        """
        if backend is None:
            backend = NumpyMixin
        self._bkd = backend

    @abstractmethod
    def solve(self, basis_mat: Array, values: Array) -> Array:
        r"""
        Find the optimal coefficients :math:`x` such that
        :math:`Ax \approx B`.

        Parameters
        ----------
        basis : ~pyapprox.surrogates.affines._basis.Basis
            The basis of the expansion

        basis_mat : array (nsamples, nterms)
            The matrix A.

        values : array (nsamples, nqoi)
            The matrix B.

        Returns
        -------
        coef : array (nterms, nqoi)
            The matrix x.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class LstSqSolver(LinearSystemSolver):
    """
    Optimize the coefficients of a linear system using linear least squares.
    """

    def solve(self, basis_mat: Array, values: Array) -> Array:
        """Return the least squares solution."""
        return self._bkd.lstsq(basis_mat, values)


class OMPSolver(LinearSystemSolver):
    """
    Orthogonal Matching Pursuit (OMP) Solver for sparse linear systems.

    The OMPSolver class implements the Orthogonal Matching Pursuit algorithm to solve
    sparse linear systems by iteratively selecting basis functions that minimize the residual.
    It supports termination based on relative residual norm or maximum sparsity.
    """

    def __init__(
        self,
        verbosity: int = 0,
        rtol: float = 1e-3,
        max_nonzeros: int = 10,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the OMPSolver.

        Parameters
        ----------
        verbosity : int, optional
            Level of verbosity for logging (default is 0).
        rtol : float, optional
            Relative tolerance for termination based on residual norm (default is 1e-3).
        max_nonzeros : int, optional
            Maximum number of non-zero coefficients in the solution (default is 10).
        backend : BackendMixin, optional
            Backend for numerical operations (default is `NumpyMixin`).
        """
        super().__init__(backend=backend)
        self._verbosity = verbosity
        self._rtol = rtol
        self.set_max_nonzeros(max_nonzeros)

        self._Amat = None
        self._bvec = None
        self._active_indices = None
        self._cholfactor = None
        self._termination_flag = None

    def set_max_nonzeros(self, max_nonzeros: int):
        """
        Set the maximum number of non-zero coefficients in the solution.

        Parameters
        ----------
        max_nonzeros : int
            Maximum number of non-zero coefficients.
        """
        self._max_nonzeros = max_nonzeros

    def _terminate(
        self,
        residnorm: float,
        bnorm: float,
        nactive_indices: int,
        max_nonzeros: int,
    ) -> bool:
        if residnorm / bnorm < self._rtol:
            self._termination_flag = 0
            return True

        if nactive_indices >= max_nonzeros:
            self._termination_flag = 1
            return True

        return False

    def _update_coef_naive(self) -> Array:
        sparse_coef = self._bkd.lstsq(
            self._Amat[:, self._active_indices], self._bvec
        )
        return sparse_coef

    def _update_coef(self) -> Array:
        Amat_sparse = self._Amat[:, self._active_indices]
        col = self._Amat[:, self._active_indices[-1]][:, None]
        cholfactor, passed = self._bkd.update_cholesky_factorization(
            self._cholfactor,
            self._bkd.dot(Amat_sparse[:, :-1].T, col),
            self._bkd.dot(col.T, col),
        )
        if not passed:
            return None
        self._cholfactor = cholfactor
        return self._bkd.cholesky_solve(
            self._cholfactor, self._bkd.dot(Amat_sparse.T, self._bvec)
        )

    def _termination_message(self, flag: int) -> str:
        messages = {
            0: "relative residual norm is below tolerance",
            1: "maximum number of basis functions added",
            2: "columns are not independent",
        }
        return messages[flag]

    def _print_termination_message(self, flag: int):
        if self._verbosity > 0:
            print(
                "{0}\n\tTerminating: {1}".format(
                    self, self._termination_message(flag)
                )
            )

    def solve(self, basis_mat: Array, values: Array) -> Array:
        """
        Solve the sparse linear system using the OMP algorithm.

        Parameters
        ----------
        basis_mat : Array
            Matrix of basis functions (shape: [n_samples, n_features]).
        values : Array
            Vector of target values (shape: [n_samples, 1]).

        Returns
        -------
        coef : Array
            Sparse coefficient vector (shape: [n_features, 1]).

        Raises
        ------
        ValueError
            If `values` is not a 1D vector or if the number of rows in `basis_mat`
            does not match the number of rows in `values`.
        """
        if values.shape[1] != 1:
            raise ValueError("{0} can only be used for 1D bvec".format(self))

        if basis_mat.shape[0] != values.shape[0]:
            raise ValueError(
                "rows of basis_mat {0} not equal to rows of values {1}".format(
                    basis_mat.shape[0], values[0]
                )
            )

        self._Amat = basis_mat
        self._bvec = values
        self._active_indices = self._bkd.empty((0), dtype=int)
        self._cholfactor = None

        correlation = self._bkd.dot(self._Amat.T, self._bvec)
        nindices = self._Amat.shape[1]
        inactive_indices_mask = self._bkd.asarray(
            [True] * nindices, dtype=bool
        )
        bnorm = self._bkd.norm(self._bvec)

        if self._max_nonzeros > nindices:
            max_nonzeros = nindices
        else:
            max_nonzeros = self._max_nonzeros

        resid = self._bkd.copy(self._bvec)
        if self._verbosity > 1:
            print(("sparsity".center(8), "index".center(5), "||r||".center(9)))
        while True:
            residnorm = self._bkd.norm(resid)
            if self._verbosity > 1:
                if self._active_indices.shape[0] > 0:
                    print(
                        (
                            repr(self._active_indices.shape[0]).center(8),
                            repr(self._active_indices[-1]).center(5),
                            format(residnorm, "1.3e").center(9),
                        )
                    )

            if self._terminate(
                residnorm, bnorm, self._active_indices.shape[0], max_nonzeros
            ):
                break

            inactive_indices = self._bkd.arange(nindices, dtype=int)[
                inactive_indices_mask
            ]
            best_inactive_index = self._bkd.argmax(
                self._bkd.abs(correlation[inactive_indices, 0])
            )
            best_index = inactive_indices[best_inactive_index]
            self._active_indices = self._bkd.hstack(
                (
                    self._active_indices,
                    self._bkd.array([best_index], dtype=int),
                )
            )
            # inactive_indices_mask[best_index] = False
            inactive_indices_mask = self._bkd.up(
                inactive_indices_mask, best_index, False
            )
            result = self._update_coef()
            if result is None:
                # cholesky failed
                # use last sparse_coef
                self._termination_flag = 2
                self._active_indices = self._active_indices[:-1]
                break
            sparse_coef = result
            resid = self._bvec - self._bkd.dot(
                self._Amat[:, self._active_indices], sparse_coef
            )
            correlation = self._bkd.dot(self._Amat.T, resid)

        self._print_termination_message(self._termination_flag)
        coef = self._bkd.full((self._Amat.shape[1], 1), 0.0)
        # coef[self._active_indices] = sparse_coef
        coef = self._bkd.up(coef, self._active_indices, sparse_coef)
        return coef

    def __repr__(self):
        return "{0}(verbosity={1}, tol={2}, max_nz={3})".format(
            self.__class__.__name__,
            self._verbosity,
            self._rtol,
            self._max_nonzeros,
        )


class QuantileRegressionSolver(LinearSystemSolver):
    """
    Solver for quantile regression using linear programming.

    The `QuantileRegressionSolver` class solves quantile regression problems by formulating
    them as linear programs.
    It minimizes the weighted sum of positive and negative residuals
    to estimate conditional quantiles of a response variable given predictor variables.
    """

    def __init__(self, quantile: float, backend: BackendMixin = NumpyMixin):
        """
        Initialize the QuantileRegressionSolver.

        Parameters
        ----------
        quantile : float
            The quantile level (0 <= quantile <= 1).
        backend : BackendMixin, optional
            Backend for numerical operations (default is `NumpyMixin`).
        """
        super().__init__(backend)
        self.set_quantile(quantile)

    def set_quantile(self, quantile: float):
        if quantile < 0 or quantile > 1:
            raise ValueError("quantile must be in [0, 1]")
        self._quantile = quantile

    def solve(self, basis_mat: Array, values: Array) -> Array:
        """
        Solve the quantile regression problem using linear programming.

        Parameters
        ----------
        basis_mat : Array
            Matrix of basis functions (shape: [n_samples, n_features]).
        values : Array
            Vector of target values (shape: [n_samples, 1]).

        Returns
        -------
        coef : Array
            Coefficient vector (shape: [n_features, 1]).

        Raises
        ------
        ValueError
            If `values` is not a 1D vector or if the number of rows in `basis_mat`
            does not match the number of rows in `values`.
        """
        if values.shape[1] != 1:
            raise ValueError("{0} can only be used for 1D bvec".format(self))

        if basis_mat.shape[0] != values.shape[0]:
            raise ValueError(
                "rows of basis_mat {0} not equal to rows of values {1}".format(
                    basis_mat.shape[0], values[0]
                )
            )
        # minimize c.T @ x
        # subject to Gx <= h
        #            Ax = b
        nsamples, nbasis = basis_mat.shape
        # c.T @ x = q * \sum_n u_n + (1-q) * \sum_n v_n
        cvec = self._bkd.hstack(
            (
                self._bkd.zeros(nbasis),
                self._bkd.full((nsamples,), self._quantile),
                self._bkd.full((nsamples,), (1.0 - self._quantile)),
            )
        )
        Ident = self._bkd.eye(nsamples)
        # Equality constraints
        # B @ x + u - v = y
        Amat = self._bkd.hstack([basis_mat, Ident, -Ident])
        bvec = values
        bounds = (
            [(None, None) for ii in range(nbasis)]  # coefficient bounds
            + [(0, None) for ii in range(nsamples)]  # u slack bounds
            + [(0, None) for ii in range(nsamples)]  # vslack bounds
        )
        result = linprog(
            cvec,
            A_ub=None,
            b_ub=None,
            A_eq=Amat,
            b_eq=bvec,
            bounds=bounds,
            method="highs",
            # options={"tol": 1e-14},
        )

        # from pyapprox.optimization.quantile_regression import (
        #     quantile_regression,
        #     solve_quantile_regression,
        # )

        # quantile_coef = quantile_regression(
        #     basis_mat, values[:, 0], self._quantile
        # )
        # adjusted_quantile_coef = solve_quantile_regression(
        #     self._quantile, basis_mat, values
        # )
        # # print(quantile_coef[:, 0], "q")
        # # print(result.x[:nbasis])
        # print(adjusted_quantile_coef[:, 0], "a")

        return self._bkd.asarray(result.x[:nbasis, None])


class QuantileRegressionCVXOPTSolver(QuantileRegressionSolver):
    """
    Solver for quantile regression using CVXOPT.

    The `QuantileRegressionCVXOPTSolver` class extends `QuantileRegressionSolver` to solve
    quantile regression problems using the CVXOPT library. It provides additional options
    for controlling the solver's behavior.
    """

    def set_options(self, opts: dict):
        """
        Set solver options for CVXOPT.

        Parameters
        ----------
        opts : dict
            Dictionary of solver options. Supported keys are:
            - "abstol": Absolute tolerance for convergence.
            - "reltol": Relative tolerance for convergence.
            - "feastol": Feasibility tolerance.
            - "show_progress": Whether to display progress during optimization.
            - "maxiter": Maximum number of iterations.

        Raises
        ------
        ValueError
            If an unsupported option key is provided.
        """
        for key in opts.keys():
            if key not in [
                "abstol",
                "reltol",
                "feastol",
                "show_progress",
                "maxiter",
            ]:
                raise ValueError(f"Option {key} not supported")
        self._opts = opts

    def default_options(self) -> dict:
        """
        Get the default solver options for CVXOPT.

        Returns
        -------
        opts : dict
            Dictionary of default solver options.
        """
        return {
            "maxiter": 10000,
            "abstol": 1e-8,
            "reltol": 1e-8,
            "feastol": 1e-8,
            "show_progress": False,
        }

    def solve(
        self, basis_mat: Array, values: Array, return_slack: bool = False
    ) -> Array:
        if not hasattr(self, "_opts"):
            self.set_options(self.default_options())
        nsamples, nbasis = basis_mat.shape
        # Define objective
        c_arr = np.hstack(
            (
                np.zeros(nbasis),  # coefficients
                self._quantile * np.ones(nsamples),  # u slack variables
                (1 - self._quantile) * np.ones(nsamples),  # v slack variables
            )
        )[:, None]
        c = matrix(c_arr)

        # Set equality constraints
        Isamp = np.identity(nsamples)
        A = sparse(
            [
                [matrix(self._bkd.to_numpy(basis_mat))],
                [matrix(Isamp)],
                [matrix(-Isamp)],
            ]
        )
        b = matrix(self._bkd.to_numpy(values))

        # use inequality constraints to set bounds on slack variables
        # of form G @ x <= h
        # want slack variables to be positive so bounds are -u <= 0, -v <=0
        G = spmatrix(
            -1.0,
            nbasis + np.arange(2 * nsamples),
            nbasis + np.arange(2 * nsamples),
        )
        h = matrix(np.zeros(nbasis + 2 * nsamples))

        solvers.options["max_iters"] = self._opts["maxiter"]
        solvers.options["abstol"] = self._opts["abstol"]
        solvers.options["reltol"] = self._opts["reltol"]
        solvers.options["feastol"] = self._opts["feastol"]
        if not self._opts["show_progress"]:
            # useful only for GLPK
            # cvxopt 1.1.8
            solvers.options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}
        solvers.options["show_progress"] = self._opts["show_progress"]

        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver="glpk")["x"]
        sol = self._bkd.asarray(np.array(sol[:]))
        if return_slack:
            return sol
        coef = sol[:nbasis]
        return coef


class RiskConservativeMixin:
    """
    Mixin to make a risk-aware surrogate conservative.

    The `RiskConservativeMixin` modifies surrogate models to ensure that their predictions
    are conservative with respect to a risk measure. The risk measure associated with the
    surrogate must satisfy the following properties:
    - Positively homogeneous
    - Translation equivariant
    - Convex
    """

    def solve(self, basis_mat: Array, values: Array) -> Array:
        """
        Solve the surrogate model while ensuring conservativeness.

        Parameters
        ----------
        basis_mat : Array
            Matrix of basis functions (shape: [n_samples, n_features]).
        values : Array
            Vector of target values (shape: [n_samples, 1]).

        Returns
        -------
        coef : Array
            Coefficient vector (shape: [n_features, 1]).

        """
        coef = super().solve(basis_mat, values)
        # set constant coeficient to zero before computing residuals
        coef[0] = 0.0
        residuals = values - basis_mat @ coef
        self._risk_measure.set_samples(residuals.T)
        coef[0, 0] = self._risk_measure()
        return coef

    def risk_measure(self) -> RiskMeasure:
        """
        Get the risk measure associated with the surrogate.

        Returns
        -------
        risk_measure : RiskMeasure
            The risk measure object.
        """
        return self._risk_measure

    def __repr__(self) -> str:
        return "{0}({1})".format(self.__class__.__name__, self._risk_measure)


class ConservativeQuantileRegressionSolver(
    RiskConservativeMixin, QuantileRegressionSolver
):
    """
    Conservative quantile regression solver.

    The `ConservativeQuantileRegressionSolver` combines quantile regression with the
    `RiskConservativeMixin` to ensure that the surrogate model is conservative with
    respect to the Average Value at Risk (AVaR) measure.
    "
    """

    def set_quantile(self, quantile: float):
        """
        Set the quantile level and initialize the  Average Value at Risk (AVaR) for the specified quantile.

        Parameters
        ----------
        quantile : float
            The quantile level (0 <= quantile <= 1).
        """
        super().set_quantile(quantile)
        self._risk_measure = AverageValueAtRisk(
            self._quantile, return_all=False, backend=self._bkd
        )


class ConservativeQuantileRegressionCVXOPTSolver(
    RiskConservativeMixin, QuantileRegressionCVXOPTSolver
):
    """
    Conservative quantile regression solver using CVXOPT.

    The `ConservativeQuantileRegressionCVXOPTSolver` combines quantile regression with
    the `RiskConservativeMixin` to ensure that the surrogate model is conservative with
    respect to the Average Value at Risk (AVaR) measure. It uses CVXOPT for solving
    the regression problem.
    """

    def set_quantile(self, quantile: float):
        """
        Set the quantile level and initialize the  Average Value at Risk (AVaR) for the specified quantile.

        Parameters
        ----------
        quantile : float
            The quantile level (0 <= quantile <= 1).
        """
        super().set_quantile(quantile)
        self._risk_measure = AverageValueAtRisk(
            self._quantile, return_all=False, backend=self._bkd
        )


class ConservativeLstSqSolver(RiskConservativeMixin, LstSqSolver):
    """
    Conservative least squares solver.

    The `ConservativeLstSqSolver` combines least squares regression with the
    `RiskConservativeMixin` to ensure that the surrogate model is conservative with
    respect to the Safety Margin Risk Measure.
    """

    def __init__(self, strength: float, backend: BackendMixin = NumpyMixin):
        """
        Initialize the ConservativeLstSqSolver.

        Parameters
        ----------
        strength : float
            Strength parameter for the Safety Margin Risk Measure.
        backend : BackendMixin, optional
            Backend for numerical operations (default is `NumpyMixin`).
        """
        super().__init__(backend)
        self.set_strength(strength)

    def set_strength(self, strength: float):
        """
        Set the strength parameter for the risk measure, i.e. the scalar that multiplies the standard deviation of the random variable.

        Parameters
        ----------
        strength : float
            Strength parameter for the Safety Margin Risk Measure.
        """
        self._strength = strength
        self._risk_measure = SafetyMarginRiskMeasure(
            self._strength, backend=self._bkd
        )


class EntropicLoss(SingleSampleModel):
    """
    Entropic loss function for regression problems.
    """

    def __init__(
        self,
        basis_mat: Array,
        train_values: Array,
        weights: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
                """
        Initialize the EntropicLoss.

        Parameters
        ----------
        basis_mat : Array
            Matrix of basis functions (shape: [n_samples, n_features]).
        train_values : Array
            Vector of observed values (shape: [n_samples, 1]).
        weights : Array, optional
            Quadrature weights for the samples (shape: [n_samples, 1]).
            If not provided, MC weights are used.
        backend : BackendMixin, optional
            Backend for numerical operations (default is `NumpyMixin`).

        Raises
        ------
        ValueError
            If `weights` has the wrong shape.
        """
        super().__init__(backend)
        if weights is None:
            weights = self._bkd.full(
                (basis_mat.shape[0], 1), 1 / basis_mat.shape[0]
            )
        if weights.shape != (basis_mat.shape[0], 1):
            raise ValueError("weights has the wrong shape")
        self._train_values = train_values
        self._weights = weights
        self._basis_mat = basis_mat

    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._basis_mat.shape[1]

    def _evaluate(self, coefs: Array) -> Array:
        pred_values = self._basis_mat @ coefs
        residuals = self._train_values - pred_values
        return (self._bkd.exp(residuals) - residuals - 1.0).T @ self._weights

    def _jacobian(self, coefs: Array) -> Array:
        pred_values = self._basis_mat @ coefs
        residuals = self._train_values - pred_values
        return self._bkd.einsum(
            "i,ij->j",
            (self._weights * (1.0 - self._bkd.exp(residuals)))[:, 0],
            self._basis_mat,
        )[None, :]

    def _apply_hessian(self, coefs: Array, vec: Array) -> Array:
        pred_values = self._basis_mat @ coefs
        residuals = self._train_values - pred_values
        return self._basis_mat.T @ (
            self._weights * self._bkd.exp(residuals) * (self._basis_mat @ vec)
        )


class EntropicRegressionSolver(LinearSystemSolver):
    """
    ""
    Solver for regression problems using the Entropic Risk Quadrangle.

    The `EntropicRegressionSolver` minimizes the error measure associated with the
    Entropic Risk Quadrangle. Unlike other risk measures, the Entropic risk measure
    is not positively homogeneous,  i.e. R[t*X] != r*R[X], meaning it cannot be used with `RiskConservativeMixin`
    to conservatively estimate risk.
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the EntropicRegressionSolver.

        Parameters
        ----------
        backend : BackendMixin, optional
            Backend for numerical operations (default is `NumpyMixin`).
        """
        super().__init__(backend)
        # todo allows for varying strength values
        # must also update EntropicLoss
        self.set_strength(1.0)

    def set_strength(self, strength: float):
        """
        Set the strength parameter for the Entropic risk measure.

        Parameters
        ----------
        strength : float
            Strength parameter for the Entropic risk measure.
        """
        self._strength = strength

    def solve(self, basis_mat: Array, values: Array) -> Array:
        """
        Solve the regression problem using the Entropic loss function.

        Parameters
        ----------
        basis_mat : Array
            Matrix of basis functions (shape: [n_samples, n_features]).
        values : Array
            Vector of observed values (shape: [n_samples, 1]).

        Returns
        -------
        coef : Array
            Coefficient vector (shape: [n_features, 1]).
        """
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        loss = EntropicLoss(basis_mat, values, backend=self._bkd)
        self._optimizer.set_objective_function(loss)
        iterate = self._bkd.ones((basis_mat.shape[1], 1))
        result = self._optimizer.minimize(iterate)
        return result.x

    def default_optimizer(
        self,
        verbosity: int = 0,
        gtol: float = 1e-8,
        maxiter: int = 1000,
        method: str = "trust-constr",
    ) -> ScipyConstrainedOptimizer:
        """
        Get the default optimizer for minimizing the Entropic loss function.

        Parameters
        ----------
        verbosity : int, optional
            Verbosity level for logging (default is 0).
        gtol : float, optional
            Gradient tolerance for convergence (default is 1e-8).
        maxiter : int, optional
            Maximum number of iterations (default is 1000).
        method : str, optional
            Optimization method (default is "trust-constr").

        Returns
        -------
        optimizer : ScipyConstrainedOptimizer
            Default optimizer for minimizing the Entropic loss function.
        """
        local_optimizer = ScipyConstrainedOptimizer()
        local_optimizer.set_verbosity(verbosity)
        local_optimizer.set_options(
            gtol=gtol,
            maxiter=maxiter,
            method=method,
        )
        return local_optimizer

    def set_optimizer(self, optimizer: ConstrainedOptimizer):
        """
        Set a custom optimizer for minimizing the Entropic loss function.

        Parameters
        ----------
        optimizer : ConstrainedOptimizer
            Custom optimizer for minimizing the Entropic loss function.

        Raises
        ------
        ValueError
            If `optimizer` is not an instance of `ConstrainedOptimizer`.
        """
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                f"optimizer {optimizer} must be instance of "
                "ConstrainedOptimizer"
            )
        self._optimizer = optimizer
