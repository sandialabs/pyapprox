from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.numpylinalg import (
    LinAlgMixin, NumpyLinAlgMixin)


class LinearSystemSolver(ABC):
    """Optimize the coefficients of a linear system."""

    def __init__(self, backend: LinAlgMixin = None):
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._backend = backend
        self._backend._la_set_attributes(self)

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
        return self._la_lstsq(Amat, Bmat)


class OMPSolver(LinearSystemSolver):
    def __init__(self, verbosity=0, rtol=1e-3, max_nonzeros=10,
                 backend: LinAlgMixin = None):
        super().__init__(backend=backend)
        self._verbosity = verbosity
        self._rtol = rtol
        self._max_nonzeros = max_nonzeros

        self._Amat = None
        self._bvec = None
        self._active_indices = None
        self._cholfactor = None

    def _terminate(self, residnorm, bnorm, nactive_indices):
        if residnorm/bnorm < self._rtol:
            if self._verbosity > 0:
                print('Terminating: relative residual norm is below tolerance')
            return True

        if nactive_indices >= self._max_nonzeros:
            if self._verbosity > 0:
                print('Terminating: maximum number of basis functions added')
            return False

    def _update_coef_naive(self):
        sparse_coef = self._la_lstsq(
            self._Amat[:, self._active_indices], self._bvec)
        return sparse_coef

    def _update_coef(self):
        Amat_sparse = self._Amat[:, self._active_indices]
        col = self._Amat[:, self._active_indices[-1]][:, None]
        self._cholfactor = self._la_update_cholesky_factorization(
            self._cholfactor,  self._la_dot(Amat_sparse[:, :-1].T,  col),
            self._la_dot(col.T, col))
        sparse_coef = self._la_cholesky_solve(
            self._cholfactor, self._la_dot(Amat_sparse.T, self._bvec))
        return sparse_coef

    def solve(self, Amat, bvec):
        if bvec.shape[1] != 1:
            raise ValueError("{0} can only be used for 1D bvec".format(self))

        if Amat.shape[0] != bvec.shape[0]:
            raise ValueError(
                "rows of Amat {0} not equal to rows of bvec {1}".format(
                    Amat.shape[0], bvec[0]))

        self._Amat = Amat
        self._bvec = bvec
        self._active_indices = self._la_empty((0), dtype=int)

        correlation = self._la_dot(self._Amat.T, self._bvec)
        nindices = self._Amat.shape[1]
        inactive_indices_mask = self._la_atleast1d([True]*nindices, dtype=bool)
        bnorm = self._la_norm(self._bvec)

        if self._max_nonzeros > nindices:
            raise ValueError("max_nonzeros {0} > Amat.shape[1] {1}".format(
                self._max_nonzeros, nindices))

        resid = self._la_copy(self._bvec)
        if self._verbosity > 0:
            print(('sparsity'.center(8), 'index'.center(5), '||r||'.center(9)))
        while True:
            residnorm = self._la_norm(resid)
            if self. _verbosity > 0:
                if self._active_indices.shape[0] > 0:
                    print((repr(self._active_indices.shape[0]).center(8), repr(
                        self._active_indices[-1]).center(5),
                           format(residnorm, '1.3e').center(9)))

            if self._terminate(
                    residnorm, bnorm, self._active_indices.shape[0]):
                break

            inactive_indices = self._la_arange(
                nindices, dtype=int)[inactive_indices_mask]
            best_inactive_index = self._la_argmax(
                self._la_abs(correlation[inactive_indices, 0]))
            best_index = inactive_indices[best_inactive_index]
            self._active_indices = self._la_hstack(
                (self._active_indices,
                 self._la_array([best_index], dtype=int)))
            inactive_indices_mask[best_index] = False
            # sparse_coef = self._update_coef_naive()
            sparse_coef = self._update_coef()
            resid = self._bvec - self._la_dot(
                self._Amat[:, self._active_indices], sparse_coef)
            correlation = self._la_dot(self._Amat.T, resid)

        coef = self._la_full((self._Amat.shape[1], 1), 0.)
        coef[self._active_indices] = sparse_coef
        return coef

    def __repr__(self):
        return "{0}(verbosity={1}, tol={2}, max_nz={3})".format(
            self.__class__.__name__, self._verbosity, self._rtol,
            self._max_nonzeros)
