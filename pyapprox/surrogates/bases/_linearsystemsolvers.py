from abc import ABC, abstractmethod

class LinearSystemSolver(ABC):
    """Optimize the coefficients of a linear system."""
    
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
    def __init__(self, verbosity=0, rtol=1e-3, max_nonzeros=10):
        self._verbosity = verbosity
        self._rtol = rtol
        self._max_nonzeros = max_nonzeros

    def _terminate(self, residnorm, bnorm, nactive_indices):
        if residnorm/bnorm < self._rtol:
            if self._verbosity > 0:
                print('Terminating: relative residual norm is below tolerance')
            return True

        if nactive_indices >= self._max_nonzeros:
            if self._verbosity > 0:
                print('Terminating: maximum number of basis functions added')
            return False
    
    def solve(self, Amat, bvec):
        if bvec.shape[1] != 1:
            raise ValueError("{0} can only be used for 1D bvec".format(self))

        nindices = Amat.shape[1]
        inactive_indices_mask = self._la_atleast1d([True]*nindices, dtype=bool)
        bnorm = self._la_norm(bvec)

        if self._max_nonzeros > nindices:
            raise ValueError("max_nonzeros {0} > Amat.shape[1] {1}".format(
                self._max_nonzeros, nindices))

        active_indices = self._la_empty((0), dtype=int)
        resid = self._la_copy(bvec)
        guess = self._la_full((nindices, 1), 0.)
        if self._verbosity > 0:
            print(('sparsity'.center(8), 'index'.center(5), '||r||'.center(9)))
        while True:
            residnorm = self._la_norm(resid)
            if self. _verbosity > 0:
                if active_indices.shape[0] > 0:
                    print((repr(active_indices.shape[0]).center(8), repr(
                        active_indices[-1]).center(5),
                           format(residnorm, '1.3e').center(9)))
                    
            if self._terminate(residnorm, bnorm, active_indices.shape[0]):
                break

            grad = -Amat.T @ bvec

            ninactive_indices = nindices-active_indices.shape[0]
            inactive_indices = self._la_arange(nindices)[inactive_indices_mask]
            best_inactive_index = self._la_argmax(
                self._la_abs(grad[inactive_indices, 0]))
            best_index = inactive_indices[best_inactive_index]
            active_indices = self._la_hstack((active_indices, [best_index]))
            inactive_indices_mask[best_index] = False
            sparse_guess = self._la_lstsq(Amat[:, active_indices], bvec)
            guess = self._la_full((nindices, 1), 0.)
            guess[active_indices] = sparse_guess
            resid = bvec-Amat @ guess

        print(guess)
        return guess
            
    def __repr__(self):
        return "{0}(verbosity={1}, tol={2}, max_nz={3})".format(
            self.__class__.__name__, self._verbosity, self._rtol,
            self._max_nonzeros)

