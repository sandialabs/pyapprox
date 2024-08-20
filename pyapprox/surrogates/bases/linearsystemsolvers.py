from __future__ import annotations
from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.numpylinalg import (
    LinAlgMixin,
    NumpyLinAlgMixin,
)


class LinearSystemSolver(ABC):
    """Optimize the coefficients of a linear system."""

    def __init__(self, backend: LinAlgMixin = None):
        if backend is None:
            backend = NumpyLinAlgMixin
        self._bkd = backend

    @abstractmethod
    def solve(self, basis_mat, values):
        r"""
        Find the optimal coefficients :math:`x` such that
        :math:`Ax \approx B`.

        Parameters
        ----------
        basis : ~pyapprox.surrogates.basess._basis.Basis
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

    def solve(self, Amat, Bmat):
        """Return the least squares solution."""
        return self._bkd.lstsq(Amat, Bmat)


class OMPSolver(LinearSystemSolver):
    def __init__(
        self,
        verbosity=0,
        rtol=1e-3,
        max_nonzeros=10,
        backend: LinAlgMixin = None,
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

    def solve(self, Amat, bvec):
        if bvec.shape[1] != 1:
            raise ValueError("{0} can only be used for 1D bvec".format(self))

        if Amat.shape[0] != bvec.shape[0]:
            raise ValueError(
                "rows of Amat {0} not equal to rows of bvec {1}".format(
                    Amat.shape[0], bvec[0]
                )
            )

        self._Amat = Amat
        self._bvec = bvec
        self._active_indices = self._bkd.empty((0), dtype=int)
        self._cholfactor = None

        correlation = self._bkd.dot(self._Amat.T, self._bvec)
        nindices = self._Amat.shape[1]
        inactive_indices_mask = self._bkd.atleast1d(
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
