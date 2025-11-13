"""
Linear System Solvers for Regression Problems

This module provides classes for solving linear systems in regression problems. It includes
an abstract base class (`LinearSystemSolver`) that defines the interface for solving linear
systems, as well as specific implementations such as least squares regression (`LstSqSolver`).
"""

from typing import Union, List
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from scipy.optimize import linprog, LinearConstraint
from pyapprox.optimization.risk import (
    AverageValueAtRisk,
    SafetyMarginRiskMeasure,
    RiskMeasure,
)
from pyapprox.interface.model import SingleSampleModel
from pyapprox.surrogates.regressor import Regressor
from pyapprox.optimization.scipy import (
    ConstrainedOptimizer,
    ScipyConstrainedOptimizer,
)
from pyapprox.util.sys_utilities import package_available
from pyapprox.optimization.minimize import (
    SmoothLeftHeavisideFunction,
    SmoothLogBasedMaxFunction,
    Constraint,
)


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
    def _solve(self, basis_mat: Array, values: Array) -> Array:
        raise NotImplementedError

    def _check_inputs(self, basis_mat: Array, values: Array):
        if values.ndim != 2:
            raise ValueError("values must be a 2D array")

        if basis_mat.shape[0] != values.shape[0]:
            raise ValueError(
                "rows of basis_mat {0} not equal to rows of values {1}".format(
                    basis_mat.shape[0], values[0]
                )
            )

    def _check_result(self, basis_mat: Array, values: Array, coef: Array):
        if coef.shape != (basis_mat.shape[1], values.shape[1]):
            raise ValueError(
                "coef shape was {0} but must be {1}".format(
                    coef.shape, (basis_mat.shape[1], values.shape[1])
                )
            )

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
        self._check_inputs(basis_mat, values)
        if not hasattr(self, "_sqrt_weights"):
            coef = self._solve(basis_mat, values)
        else:
            coef = self._weighted_solve(basis_mat, values)
        self._check_result(basis_mat, values, coef)
        return coef

    def set_weights(self, weights: Array):
        if weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError(
                "weights must be a 2D array with one column but has "
                f"shape {weights.shape}"
            )
        self._sqrt_weights = self._bkd.sqrt(weights)

    def _weighted_solve(self, basis_mat: Array, values: Array) -> Array:
        if self._sqrt_weights.shape != (values.shape[0], 1):
            raise ValueError(
                "weights has shape {1} but must have shape {0}".format(
                    self._sqrt_weights.shape,
                    (values.shape[1], 1),
                )
            )
        # print("cond", self._bkd.cond(self._sqrt_weights * basis_mat))
        # tmp = self._sqrt_weights * basis_mat
        # print(tmp.T @ tmp, tmp.shape)
        return self._solve(
            self._sqrt_weights * basis_mat, self._sqrt_weights * values
        )

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def set_surrogate(self, surrogate: Regressor):
        self._surrogate = surrogate


class SingleQoiLinearSolverMixin:
    def _check_inputs(self, basis_mat: Array, values: Array):
        super()._check_inputs(basis_mat, values)
        if values.shape[1] != 1:
            raise ValueError(
                "{0} can only be used for bvec with 1 column".format(self)
            )


class LstSqSolver(LinearSystemSolver):
    """
    Optimize the coefficients of a linear system using linear least squares.
    """

    def _solve(self, basis_mat: Array, values: Array) -> Array:
        """Return the least squares solution."""
        return self._bkd.lstsq(basis_mat, values)


class OMPSolver(SingleQoiLinearSolverMixin, LinearSystemSolver):
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

    def _solve(self, basis_mat: Array, values: Array) -> Array:
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


class QuantileRegressionSolver(SingleQoiLinearSolverMixin, LinearSystemSolver):
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

    def _solve(self, basis_mat: Array, values: Array) -> Array:
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
            self._bkd.to_numpy(cvec),
            A_ub=None,
            b_ub=None,
            A_eq=self._bkd.to_numpy(Amat),
            b_eq=self._bkd.to_numpy(bvec),
            bounds=bounds,
            method="highs",
            # options={"tol": 1e-14},
        )
        return self._bkd.asarray(result.x[:nbasis])[:, None]


class BasisPursuitRegressionSolver(
    SingleQoiLinearSolverMixin, LinearSystemSolver
):
    def _solve(self, basis_mat: Array, values: Array) -> Array:
        basis_mat = self._bkd.to_numpy(basis_mat)
        values = self._bkd.to_numpy(values)
        nunknowns = basis_mat.shape[1]
        nslack_variables = nunknowns
        c = np.zeros(nunknowns + nslack_variables)
        c[nunknowns:] = 1.0

        II = sp.identity(nunknowns)
        tmp = np.array([[1, -1], [-1, -1]])
        A_ub = sp.kron(tmp, II)
        b_ub = np.zeros(nunknowns + nslack_variables)

        A_eq = sp.lil_matrix((basis_mat.shape[0], c.shape[0]), dtype=float)
        A_eq[:, : basis_mat.shape[1]] = basis_mat
        b_eq = values

        bounds = [(-np.inf, np.inf)] * nunknowns + [
            (0, np.inf)
        ] * nslack_variables

        if not hasattr(self, "_options"):
            options = {}
        else:
            options = self._options
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            options=options,
        )
        return self._bkd.asarray(res.x[:nunknowns, None])

    def set_options(self, options: dict):
        self._options = options


class CVXOPTOptionsMixin:
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

    def _set_cvxopt_options(self):
        solvers.options["max_iters"] = self._opts.get("maxiter", 10000)
        solvers.options["abstol"] = self._opts.get("abstol", 1e-8)
        solvers.options["reltol"] = self._opts.get("reltol", 1e-8)
        solvers.options["feastol"] = self._opts.get("feastol", 1e-8)
        if not self._opts.get("show_progress", False):
            # useful only for GLPK
            # cvxopt 1.1.8
            solvers.options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}
        solvers.options["show_progress"] = self._opts.get(
            "show_progress", False
        )


class QuantileRegressionCVXOPTSolver(
    CVXOPTOptionsMixin, QuantileRegressionSolver
):
    """
    Solver for quantile regression using CVXOPT.

    The `QuantileRegressionCVXOPTSolver` class extends `QuantileRegressionSolver` to solve
    quantile regression problems using the CVXOPT library. It provides additional options
    for controlling the solver's behavior.
    """

    def _solve(
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
        b = matrix(self._bkd.to_numpy(values[:, 0]))

        # use inequality constraints to set bounds on slack variables
        # of form G @ x <= h
        # want slack variables to be positive so bounds are -u <= 0, -v <=0
        G = spmatrix(
            -1.0,
            nbasis + np.arange(2 * nsamples),
            nbasis + np.arange(2 * nsamples),
        )
        h = matrix(np.zeros(nbasis + 2 * nsamples))
        self._set_cvxopt_options()
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

    def _solve(self, basis_mat: Array, values: Array) -> Array:
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
        coef = super()._solve(basis_mat, values)
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


class OptimizerMixin:
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


class EntropicRegressionSolver(LinearSystemSolver, OptimizerMixin):
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

    def _solve(self, basis_mat: Array, values: Array) -> Array:
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


class FSDObjectiveNew(SingleSampleModel):
    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        return True

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._surrogate.nterms()

    def set_regression_solver(self, solver: "FSDOptProblem"):
        if solver._surrogate.nqoi() != 1:
            raise ValueError("surrogate must have only one QoI")
        self._surrogate = solver._surrogate
        self._train_samples = self._surrogate._ctrain_samples
        self._train_values = self._surrogate._ctrain_values
        self._probabilities = solver._probabilities

    def _evaluate(self, coef: Array) -> Array:
        # todo replace the following with more general
        # set unknowns because this function can otherwise be
        # applied to nonlinear surrogates without change
        self._surrogate.set_coefficients(coef)
        surrogate_values = self._surrogate(self._train_samples)
        val = 0.5 * self._bkd.sum(
            self._probabilities * (self._train_values - surrogate_values) ** 2
        )
        return self._bkd.atleast2d(val)

    def _jacobian(self, coef: Array) -> Array:
        self._surrogate.set_coefficients(coef)
        surrogate_values = self._surrogate(self._train_samples)
        surrogate_jac = self._surrogate.hyperparam_jacobian(coef)
        jac = -surrogate_jac.T @ (
            self._probabilities * (self._train_values - surrogate_values)
        )
        return jac.T

    def _apply_hessian(self, coef: Array, vec: Array) -> Array:
        # only works if surrogate.hvp returns zero
        surrogate_jac = self._surrogate.hyperparam_jacobian(coef)
        return surrogate_jac.T @ (self._probabilities * (surrogate_jac @ vec))


class StochasticDominanceConstraint(Constraint):
    def set_regression_solver(
        self, solver: "StochasticDominanceRegressionSolver"
    ):
        if solver._surrogate.nqoi() != 1:
            raise ValueError("surrogate must have only one QoI")
        self._surrogate = solver._surrogate
        self._train_samples = self._surrogate._ctrain_samples
        self._train_values = self._surrogate._ctrain_values
        self._probabilities = solver._probabilities
        self._constraint_indices = solver._constraint_indices
        self._smooth_function = solver._smooth_function

    def nqoi(self) -> int:
        return self._probabilities.shape[0]

    def nvars(self) -> int:
        return self._surrogate._hyp_list.nactive_vars()

    def _values(self, coef: Array) -> Array:
        self._surrogate.set_coefficients(coef)
        surrogate_values = self._surrogate(self._train_samples)
        tmp1 = self._smooth_function(
            surrogate_values - surrogate_values[self._constraint_indices].T
        )
        tmp2 = self._smooth_function(
            (self._train_values - surrogate_values[self._constraint_indices].T)
        )
        val = self._probabilities.T @ (tmp1 - tmp2)
        return val

    def jacobian_implemented(self) -> bool:
        return True

    def _jacobian(self, coef: Array) -> Array:
        r"""
        Compute the Jacobian of the constraints. The nth row of the Jacobian is
        the derivative of the nth constraint :math:`c_n(x)`.
        Let :math:`h(z)` be the smooth heaviside function and :math:`f(x)` the
        function approximation evaluated
        at the training samples and coeficients :math:`x`, then

        .. math::

           \frac{\partial c_n}{\partial x} =
           \sum_{m=1}^M h^\prime(f(x_m)-f(x_n))
              \left(\nabla_x f(x_m)-\nabla_x f(x_n))\right) -
              h^\prime(y_m-f(x_n))\left(-\nabla_x f(x_n))\right)

        Parameters
        ----------
        coef : Array (ncoef)
            The unknowns

        Returns
        -------
        jac : Array (nconstraints, ncoef)
            The Jacobian of the constraints
        """
        self._surrogate.set_coefficients(coef)
        surrogate_values = self._surrogate(self._train_samples)
        surrogate_jac = self._surrogate.hyperparam_jacobian(coef)
        hder1 = (
            self._smooth_function.first_derivative(
                (
                    surrogate_values.T
                    - surrogate_values[self._constraint_indices]
                )
            )
            * self._probabilities
        )
        fder1 = (
            surrogate_jac[None, :, :]
            - surrogate_jac[self._constraint_indices, None, :]
        )
        # con_jac = self._bkd.sum(hder1[:, :, None] * fder1, axis=1)
        # c: nconstraints, d: ncoefs, n: nsamples
        con_jac = self._bkd.einsum("cn,cnd->cd", hder1, fder1)
        hder2 = (
            self._smooth_function.first_derivative(
                (
                    self._train_values.T
                    - surrogate_values[self._constraint_indices]
                )
            )
            * self._probabilities
        )
        fder2 = (
            0 * surrogate_jac[None, :, :]
            - surrogate_jac[self._constraint_indices, None, :]
        )
        # con_jac -= self._bkd.sum(hder2[:, :, None] * fder2, axis=1)
        con_jac -= self._bkd.einsum("cn,cnd->cd", hder2, fder2)
        return con_jac

    def weighted_hessian_implemented(self) -> bool:
        return True

    def _weighted_hessian(self, coef: Array, lmult: Array) -> Array:
        r"""
        Compute the Hessian of the constraints applied to the Lagrange
        multipliers.

        We need to compute

        .. math:: d^2/dx^2 f(g(x))=g'(x)^2 f''(g(x))+g''(x)f'(g(x))

        and assume that  :math:`g''(x)=0 \forall x`. I.e. only linear
        approximations g(x) are implemented

        Parameters
        ----------
        coef : Array (ncoef)
            The unknowns

        lmult : Array (nconstraints)
            vector of N Lagrange multipliers with

        Returns
        -------
        hess : Arrat (ncoef, ncoef)
            The weighted sum of the individual constraint Hessians

            .. math:: \sum_{n=1}^N H_n(x)
        """
        self._surrogate.set_coefficients(coef)
        surrogate_values = self._surrogate(self._train_samples)
        surrogate_jac = self._surrogate.hyperparam_jacobian(coef)
        hder1 = (
            self._smooth_function.second_derivative(
                (
                    surrogate_values.T
                    - surrogate_values[self._constraint_indices]
                )
            )
            * self._probabilities
        )
        hder2 = (
            self._smooth_function.second_derivative(
                (
                    self._train_values.T
                    - surrogate_values[self._constraint_indices]
                )
            )
            * self._probabilities
        )
        # Todo fder1 and fder2 can be stored when computing Jacobian and
        # reused
        fder1 = (
            surrogate_jac[None, :, :]
            - surrogate_jac[self._constraint_indices, None, :]
        )
        fder2 = (
            0 * surrogate_jac[None, :, :]
            - surrogate_jac[self._constraint_indices, None, :]
        )
        ncoef = coef.shape[0]
        hessian = self._bkd.zeros((ncoef, ncoef))
        # c: nconstraints, d: ncoefs, n: nsamples
        hessian = self._bkd.einsum(
            "c, cn, cnd, cnf -> df",
            lmult[:, 0],
            hder1,
            fder1,
            fder1,
        ) - self._bkd.einsum(
            "c, cn, cnd, cnf -> df",
            lmult[:, 0],
            hder2,
            fder2,
            fder2,
        )
        # for ii in range(lmult.shape[0]):
        #     hessian += (
        #         lmult[ii]
        #         * (hder1[ii, :, None] * fder1[ii, :]).T
        #         @ (fder1[ii, :])
        #     )
        #     hessian -= (
        #         lmult[ii]
        #         * (hder2[ii, :, None] * fder2[ii, :]).T
        #         @ (fder2[ii, :])
        #     )
        return hessian


class StochasticDominanceRegressionSolver(LinearSystemSolver, OptimizerMixin):
    def __init__(
        self,
        ntrain_samples: int,
        smooth_function: Union[
            SmoothLeftHeavisideFunction, SmoothLogBasedMaxFunction
        ],
    ):
        """
        Initialize the stochastic domina solver.
        """
        self._set_smooth_function(smooth_function)
        self._ntrain_samples = ntrain_samples
        super().__init__(self._smooth_function._bkd)

    def set_probabilities(self, probabilities: Array):
        if probabilities.shape != (self._ntrain_samples, 1):
            raise ValueError("probabilities has the wrong shape")
        self._probabilities = probabilities

    def set_constraint_indices(self, indices: Array):
        if indices.shape != (self._ntrain_samples,):
            raise ValueError(
                "indices must be a 1D array with shape {(self._ntrain.shape[1],)}"
            )
        self._constraint_indices = indices

    def _setup_optimizer(self):
        if not hasattr(self, "_constraint_indices"):
            self.set_constraint_indices(self._bkd.arange(self._ntrain_samples))
        if not hasattr(self, "_probabilities"):
            self.set_probabilities(
                self._bkd.full(
                    (self._ntrain_samples, 1), 1 / self._ntrain_samples
                )
            )

        constraint = StochasticDominanceConstraint(
            self._constraint_bounds(), keep_feasible=True, backend=self._bkd
        )
        constraint.set_regression_solver(self)
        objective = FSDObjectiveNew(backend=self._bkd)
        objective.set_regression_solver(self)
        self._optimizer.set_objective_function(objective)
        self._optimizer.set_constraints([constraint])
        iterate_bounds = self._bkd.stack(
            (
                self._bkd.full((self._surrogate.nvars(),), -np.inf),
                self._bkd.full((self._surrogate.nvars(),), np.inf),
            ),
            axis=1,
        )
        self._optimizer.set_bounds(iterate_bounds)

    def set_iterate(self, iterate: Array):
        self._iterate = iterate

    def _default_iterate(self, basis_mat: Array, values: Array) -> Array:
        # Compute conservative least squares solution to use as
        # an initial guess for the stochastic dominance that statisfies
        # the constraint
        lstq_solver = LstSqSolver(backend=self._bkd)
        coef = lstq_solver.solve(basis_mat, values)
        shift = self._bkd.max(values - basis_mat @ coef)
        coef[0] += shift
        return coef

    def _solve(self, basis_mat: Array, values: Array) -> Array:
        if not hasattr(self, "_surrogate"):
            raise RuntimeError("must call set_surrogate()")
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        self._setup_optimizer()
        if not hasattr(self, "_iterate"):
            self.set_iterate(self._default_iterate(basis_mat, values))
        if not hasattr(self, "_constraint_indices"):
            self.set_constraint_indices(
                self._bkd.arrays((self._train_samples.shape[1],))
            )
        result = self._optimizer.minimize(self._iterate)
        return result.x


class FSDRegressionSolver(StochasticDominanceRegressionSolver):
    def __init__(
        self,
        ntrain_samples: int,
        smooth_heaviside_function: SmoothLeftHeavisideFunction,
    ):
        """
        Initialize the First-order Stochastic Dominance (FSD) solver.
        """
        super().__init__(ntrain_samples, smooth_heaviside_function)

    def _set_smooth_function(
        self, smooth_heaviside_function: SmoothLeftHeavisideFunction
    ):
        if not isinstance(
            smooth_heaviside_function, SmoothLeftHeavisideFunction
        ):
            raise ValueError(
                "smooth_heaviside_function must be an instance of "
                "SmoothHeavisideFunction"
            )
        self._smooth_function = smooth_heaviside_function

    def _constraint_bounds(self):
        nconstraints = self._constraint_indices.shape[0]
        return self._bkd.stack(
            (
                self._bkd.full((nconstraints,), -np.inf),
                self._bkd.zeros((nconstraints,)),
            ),
            axis=1,
        )


class SSDRegressionSolver(FSDRegressionSolver):
    def __init__(
        self,
        ntrain_samples: int,
        smooth_max_function: SmoothLogBasedMaxFunction,
    ):
        """
        Initialize the Second-order Stochastic Dominance (SSD) solver.

        # Conceptually the FSD and SSD constraints are similar.
        # But the smooth function and bounds are different.
        FSD bounds are (-oo, 0) and SSD are (0, oo)
        """
        super().__init__(ntrain_samples, smooth_max_function)

    def _set_smooth_function(
        self, smooth_max_function: SmoothLogBasedMaxFunction
    ):
        if not isinstance(smooth_max_function, SmoothLogBasedMaxFunction):
            raise ValueError(
                "smooth_max_function must be an instance of "
                "SmoothLogBasedMaxFunction"
            )
        self._smooth_function = smooth_max_function

    def _constraint_bounds(self):
        nconstraints = self._constraint_indices.shape[0]
        return self._bkd.stack(
            (
                self._bkd.zeros((nconstraints,)),
                self._bkd.full((nconstraints,), np.inf),
            ),
            axis=1,
        )


class BasisPursuitDenoisingCVXRegressionSolver(
    CVXOPTOptionsMixin, LinearSystemSolver
):
    def __init__(self, penalty: float, backend: BackendMixin = NumpyMixin):
        if not package_available("cvxopt"):
            raise ImportError("cvxopt is not installed")
        self._penalty = penalty
        super().__init__(backend)

    def _solve(self, basis_mat: Array, values: Array) -> Array:
        """
        Solve the Basis Pursuit Denosining (BPDN) problem using a CVX
        quadratic program solver
        """
        nsamples, nbasis = basis_mat.shape
        Gram = matrix(self._bkd.to_numpy(basis_mat.T @ basis_mat))
        diag_zeros = spmatrix([], [], [], (nbasis, nbasis))
        offdiag_zeros = spmatrix([], [], [], (nbasis, nbasis))
        # assume design variables are [coef, slack]
        # cvx opt solves  (1/2)x.T@ P @ x + q.T@x subject to constraints
        # so no need to multiply gram by 0.5
        Pmat = sparse([[Gram, offdiag_zeros], [offdiag_zeros, diag_zeros]])
        lam = self._bkd.full((nbasis,), self._penalty)
        qvec = matrix(
            self._bkd.to_numpy(
                self._bkd.hstack((-basis_mat.T @ values[:, 0], lam))
            )
        )
        # Set inequality constraints including bounds on slack variables
        # of form G @ x <= h. That is, we want slack variables to satisfy
        # -x_i <= u_i
        # x_i <= u_i
        # u_i >= 0

        # -u_i <= 0
        vals = [-1.0 for ii in range(nbasis)]
        rows = [ii for ii in range(nbasis)]
        cols = [nbasis + ii for ii in range(nbasis)]
        # -x_i -u_i <= 0
        for ii in range(nbasis):
            rows += [nbasis + ii, nbasis + ii]
            cols += [ii, nbasis + ii]
            vals += [-1.0, -1.0]
        #  x_i -u_i <= 0
        for ii in range(nbasis):
            rows += [2 * nbasis + ii, 2 * nbasis + ii]
            cols += [ii, nbasis + ii]
            vals += [1.0, -1.0]
        cols = np.asarray(cols)
        vals = np.asarray(vals)
        Gmat = spmatrix(vals, rows, cols)
        # from cvxopt import printing

        # printing.options["width"] = 100
        # print(Pmat, "P")
        # print(qvec, "q")
        # print(Gmat, "G")

        hvec = matrix(np.zeros(3 * nbasis))
        self._set_cvxopt_options()
        x0 = matrix(np.full((2 * basis_mat.shape[1], 1), 1.0))

        result = solvers.qp(
            Pmat,
            qvec,
            Gmat,
            hvec,
            initvals=x0,
        )
        # print(0.5 * x0.T * Pmat * x0 + qvec.T * x0, "o")
        # print(qvec.T * x0, "s")
        # c = result["x"]
        # print(0.5 * c.T * Pmat * c + qvec.T * c, qvec.T * c, "opt")
        self._sol = self._bkd.asarray(np.array(result["x"]))
        return self._sol[:nbasis]


class BasisPursuitDenoisingObjective(SingleSampleModel):
    def __init__(
        self,
        basis_mat: Array,
        rhs_values: Array,
        penalty: float,
        backend: BackendMixin,
    ):
        super().__init__(backend)
        self._penalty = penalty
        self._basis_mat = self._bkd.to_numpy(basis_mat)
        self._rhs_values = self._bkd.to_numpy(rhs_values)
        self._nbasis = self._basis_mat.shape[1]
        self._precompute()

    def _precompute(self):
        Gram = self._basis_mat.T @ self._basis_mat
        diag_zeros = sp.csr_matrix((self._nbasis, self._nbasis))
        self._Pmat = sp.block_diag([Gram, diag_zeros])
        lam = self._bkd.full((self._nbasis,), self._penalty)
        self._qvec = np.hstack(
            (-self._basis_mat.T @ self._rhs_values[:, 0], lam)
        )[:, None]

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._nbasis * 2

    def _evaluate(self, iterate: Array) -> Array:
        iterate = self._bkd.to_numpy(iterate)
        return self._bkd.asarray(
            0.5 * iterate.T @ (self._Pmat @ iterate) + self._qvec.T @ iterate
        )

    def _jacobian(self, iterate: Array) -> Array:
        iterate = self._bkd.to_numpy(iterate)
        return self._bkd.asarray(self._Pmat @ iterate + self._qvec).T

    def _apply_hessian(self, iterate: Array, vec: Array) -> Array:
        return self._bkd.asarray(self._Pmat @ vec)

    def jacobian_implemented(self) -> bool:
        return True

    def apply_hessian_implemented(self) -> bool:
        return True


class BasisPursuitDenoisingRegressionSolver(
    LinearSystemSolver, OptimizerMixin
):
    """
    Use general purpose nonlinear optimizer to solve BPDN problem.
    It is more computationally efficient to use a quadratic program solver
    e.g that wrapped but BasisPursuitDenoisingCVXRegressionSolver.
    This implementation only exists incase cvx is not installed.
    """

    def __init__(self, penalty: float, backend: BackendMixin):
        self._penalty = penalty
        super().__init__(backend)

    def set_iterate(self, iterate: Array):
        self._iterate = iterate

    def _get_constraints(self, nbasis: int) -> List[LinearConstraint]:
        # iterate [x, u]. x: coefficients, u: slack variables

        # # -u_i <= 0
        vals = [-1.0 for ii in range(nbasis)]
        rows = [ii for ii in range(nbasis)]
        cols = [nbasis + ii for ii in range(nbasis)]
        # -x_i -u_i <= 0
        for ii in range(nbasis):
            rows += [nbasis + ii, nbasis + ii]
            cols += [ii, nbasis + ii]
            vals += [-1.0, -1.0]
        #  x_i -u_i <= 0
        for ii in range(nbasis):
            rows += [2 * nbasis + ii, 2 * nbasis + ii]
            cols += [ii, nbasis + ii]
            vals += [1.0, -1.0]
        cols = np.asarray(cols)
        vals = np.asarray(vals)
        Gmat = sp.coo_matrix((vals, (rows, cols)))
        # Convert to CSR format for efficient operations
        Gmat = Gmat.tocsr()
        Gmat = Gmat.toarray()
        # setting keep_feasible to True stops the optimizer from converging
        return [LinearConstraint(Gmat, -np.inf, 0.0, keep_feasible=False)]

    def _setup_optimizer(self, basis_mat: Array, rhs_values: Array):
        objective = BasisPursuitDenoisingObjective(
            basis_mat, rhs_values, self._penalty, self._bkd
        )
        self._optimizer.set_objective_function(objective)
        self._optimizer.set_constraints(
            self._get_constraints(basis_mat.shape[1])
        )
        iterate_bounds = self._bkd.stack(
            (
                self._bkd.full((basis_mat.shape[1] * 2,), -np.inf),
                self._bkd.full((basis_mat.shape[1] * 2,), np.inf),
            ),
            axis=1,
        )
        # Enforce slack variables are positive is done by constraints
        # iterate_bounds[basis_mat.shape[1] :, 0] = 0
        self._optimizer.set_bounds(iterate_bounds)

    def _solve(self, basis_mat: Array, rhs_values: Array) -> Array:
        """
        Solve the Basis Pursuit Denosining (BPDN) problem using a general purpose
        nonlinear optimizer
        """
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        if not hasattr(self, "_iterate"):
            iterate = self._bkd.full((basis_mat.shape[1] * 2, 1), -1.0)
            iterate[basis_mat.shape[1] :] = 1.0
            self.set_iterate(iterate)
        self._setup_optimizer(basis_mat, rhs_values)
        result = self._optimizer.minimize(self._iterate)
        self._sol = result.x
        return self._sol[: basis_mat.shape[1]]


class LinearlyConstrainedLstSqSolver(LinearSystemSolver):
    r"""
    Solve the linearly constrained least squares problem:

    .. math::

        \vec{x} = (\mat{A}^\top \mat{A})^{-1} \left( \mat{A}^\top \vec{y}
        - \mat{C}^\top \left( \mat{C} (\mat{A}^\top \mat{A})^{-1} \mat{C}^\top \right)^{-1}
        \left( \mat{C} (\mat{A}^\top \mat{A})^{-1} \mat{A}^\top \vec{y} - \vec{d} \right) \right)
    """

    def __init__(
        self,
        constraint_mat: Array,
        constraint_vec: Array,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Parameters
        ----------
        constraint_mat : Array
            The constraint matrix of size `(nconstraints, ncoef)` representing the linear constraints.

        constraint_vec : Array
            The constraint vector of size `(nconstraints,)` specifying the values of the constraints.

        backend : BackendMixin, optional
            Backend for numerical operations (default is `NumpyMixin`).
        """

        if constraint_vec.shape != (constraint_mat.shape[0], 1):
            raise ValueError("constraint_vec has the wrong shape")
        self._Cmat = constraint_mat
        self._dvec = constraint_vec
        super().__init__(backend)

    def _solve(self, basis_mat: Array, values: Array) -> Array:
        if basis_mat.shape[1] != self._Cmat.shape[1]:
            raise ValueError(
                "basis_mat and constraint_mat have inconsistent shapes"
            )

        A = basis_mat
        y = values
        C = self._Cmat
        d = self._dvec

        # Precompute A.T @ A and its inverse
        ATA = A.T @ A
        ATA_inv = self._bkd.inv(ATA)

        # Precompute A.T @ y
        ATy = A.T @ y

        # Precompute C @ ATA_inv
        C_ATA_inv = C @ ATA_inv

        # Precompute C @ ATA_inv @ C.T
        C_ATA_inv_CT = C_ATA_inv @ C.T

        # Compute (C @ ATA_inv @ C.T)^(-1)
        C_ATA_inv_CT_inv = self._bkd.inv(C_ATA_inv_CT)

        # Precompute C @ ATA_inv @ ATy
        C_ATA_inv_ATy = C_ATA_inv @ ATy

        # Compute the inner term
        inner_term = C.T @ C_ATA_inv_CT_inv @ (C_ATA_inv_ATy - d)

        # Compute the final solution
        x = ATA_inv @ (ATy - inner_term)

        return x
