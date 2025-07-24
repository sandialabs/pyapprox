from __future__ import annotations
from abc import ABC, abstractmethod

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from scipy.optimize import linprog
from pyapprox.optimization.risk import (
    AverageValueAtRisk,
    SafetyMarginRiskMeasure,
    EntropicRisk,
)
from pyapprox.interface.model import SingleSampleModel
from pyapprox.optimization.scipy import (
    ConstrainedOptimizer,
    ScipyConstrainedOptimizer,
)


class LinearSystemSolver(ABC):
    """Optimize the coefficients of a linear system."""

    def __init__(self, backend: BackendMixin = NumpyMixin):
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

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class LstSqSolver(LinearSystemSolver):
    """
    Optimize the coefficients of a linear system using linear least squares.
    """

    def solve(self, basis_mat: Array, values: Array) -> Array:
        """Return the least squares solution."""
        return self._bkd.lstsq(basis_mat, values)


class OMPSolver(LinearSystemSolver):
    def __init__(
        self,
        verbosity=0,
        rtol=1e-3,
        max_nonzeros=10,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(backend=backend)
        self._verbosity = verbosity
        self._rtol = rtol
        self.set_max_nonzeros(max_nonzeros)

        self._Amat = None
        self._bvec = None
        self._active_indices = None
        self._cholfactor = None
        self._termination_flag = None

    def set_max_nonzeros(self, max_nonzeros):
        self._max_nonzeros = max_nonzeros

    def _terminate(self, residnorm, bnorm, nactive_indices, max_nonzeros):
        if residnorm / bnorm < self._rtol:
            self._termination_flag = 0
            return True

        if nactive_indices >= max_nonzeros:
            self._termination_flag = 1
            return True

        return False

    def _update_coef_naive(self):
        sparse_coef = self._bkd.lstsq(
            self._Amat[:, self._active_indices], self._bvec
        )
        return sparse_coef

    def _update_coef(self):
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

    def _termination_message(self, flag):
        messages = {
            0: "relative residual norm is below tolerance",
            1: "maximum number of basis functions added",
            2: "columns are not independent",
        }
        return messages[flag]

    def _print_termination_message(self, flag):
        if self._verbosity > 0:
            print(
                "{0}\n\tTerminating: {1}".format(
                    self, self._termination_message(flag)
                )
            )

    def solve(self, basis_mat: Array, values: Array) -> Array:
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
    def __init__(self, quantile: float, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)
        self.set_quantile(quantile)

    def set_quantile(self, quantile: float):
        if quantile < 0 or quantile > 1:
            raise ValueError("quantile must be in [0, 1]")
        self._quantile = quantile

    def solve(self, basis_mat: Array, values: Array) -> Array:
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
            cvec, A_ub=None, b_ub=None, A_eq=Amat, b_eq=bvec, bounds=bounds
        )
        return self._bkd.asarray(result.x[:nbasis, None])


class RiskConservativeMixin:

    def solve(self, basis_mat: Array, values: Array) -> Array:
        coef = super().solve(basis_mat, values)
        residuals = values - basis_mat @ coef
        self._risk_measure.set_samples(residuals.T)
        print(self._risk_measure(), self)
        coef[0, 0] = self._risk_measure()
        return coef


class ConservativeQuantileRegressionSolver(
    RiskConservativeMixin, QuantileRegressionSolver
):
    def set_quantile(self, quantile: float):
        super().set_quantile(quantile)
        self._risk_measure = AverageValueAtRisk(
            self._quantile, return_all=False, backend=self._bkd
        )


class ConservativeLstSqSolver(RiskConservativeMixin, LstSqSolver):
    def __init__(self, strength: float, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)
        self.set_strength(strength)

    def set_strength(self, strength: float):
        self._strength = strength
        self._risk_measure = SafetyMarginRiskMeasure(
            self._strength, backend=self._bkd
        )


class EntropicLoss(SingleSampleModel):
    def __init__(
        self,
        basis_mat: Array,
        train_values: Array,
        weights: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
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
        return (self._bkd.exp(residuals) - residuals).T @ self._weights - 1.0

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
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)
        # todo allows for varying strength values
        # must also update EntropicLoss
        self.set_strength(1.0)

    def set_strength(self, strength: float):
        self._strength = strength

    def solve(self, basis_mat: Array, values: Array) -> Array:
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
        local_optimizer = ScipyConstrainedOptimizer()
        local_optimizer.set_verbosity(verbosity)
        local_optimizer.set_options(
            gtol=gtol,
            maxiter=maxiter,
            method=method,
        )
        return local_optimizer

    def set_optimizer(self, optimizer: ConstrainedOptimizer):
        if not isinstance(optimizer, ConstrainedOptimizer):
            raise ValueError(
                f"optimizer {optimizer} must be instance of "
                "ConstrainedOptimizer"
            )
        self._optimizer = optimizer


class ConservativeEntropicRegressionSolver(
    RiskConservativeMixin, EntropicRegressionSolver
):
    def set_strength(self, strength: float):
        super().set_strength(strength)

        self._risk_measure = EntropicRisk(strength, backend=self._bkd)
