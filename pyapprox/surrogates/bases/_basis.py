from abc import ABC, abstractmethod


class Basis(ABC):
    """
    The base class for any multivariate Basis.
    """
    
    @abstractmethod
    def nterms():
        """
        Return the number of basis functions.
        """
        raise NotImplementedError()

    @abstractmethod
    def nvars(self):
        """
        Return the number of inputs to the basis.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, samples):
        """
        Evaluate a multivariate basis at a set of samples.

        Parameters
        ----------
        samples : array (nvars, nsamples)
            Samples at which to evaluate the basis

        Return
        ------
        basis_matrix : array (nsamples, nterms)
            The values of the basis at the samples
        """
        raise NotImplementedError()

    def jacobian(self, samples):
        """
        Compute the Jacobians of multivariate basis at a set of samples.

        Parameters
        ----------
        samples : array (nvars, nsamples)
            Samples at which to evaluate the basis Jacobian

        Return
        ------
        jac : array (nsamples, nterms, nvars)
            The Jacobian of the basis at each sample
        """
        raise NotImplementedError("Basis jacobian not implemented")

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class MonomialBasis(Basis):
    """Multivariate monomial basis."""
    
    def __init__(self):
        self._indices = None

    def set_indices(self, indices):
        """
        Set the multivariate indices of the basis functions.
        
        Parameters
        ----------
        indices : array (nvars, nterms)
            Multivariate indices specifying the basis functions
        """
        if indices.ndim != 2:
            raise ValueError("indices must have two dimensions")
        self._indices = self._la_atleast2d(indices).astype(int)

    def nterms(self):
        return self._indices.shape[1]

    def nvars(self):
        return self._indices.shape[0]

    def _univariate_monomial_basis_matrix(self, max_level, samples):
        assert samples.ndim == 1
        basis_matrix = samples[:, None]**self._la_arange(max_level+1)[None, :]
        return basis_matrix

    def _basis_vals_1d(self, samples):
        return [
            self._univariate_monomial_basis_matrix(
                self._indices[dd, :].max(), samples[dd, :])
            for dd in range(self.nvars())]
    
    def __call__(self, samples):
        nsamples = samples.shape[1]
        basis_vals_1d = self._basis_vals_1d(samples)
        basis_matrix = self._la_empty((nsamples, self.nterms()))
        basis_matrix[:nsamples, :] = basis_vals_1d[0][:, self._indices[0, :]]
        for dd in range(1, self.nvars()):
            basis_matrix[:nsamples, :] *= (
                basis_vals_1d[dd][:, self._indices[dd, :]])
        return basis_matrix

    def jacobian(self, samples):
        nsamples = samples.shape[1]
        basis_vals_1d = self._basis_vals_1d(samples)
        jac = self._la_empty((nsamples, self.nterms(), self.nvars()))
        for ii in range(self.nterms()):
            index = self._indices[:, ii]
            for jj in range(self.nvars()):
                # derivative in jj direction
                basis_vals = basis_vals_1d[jj][:, max(0, index[jj]-1)]*index[jj]
                # basis values in other directions
                for dd in range(self.nvars()):
                    if dd != jj:
                        basis_vals *= basis_vals_1d[dd][:, index[dd]]
                jac[:, ii, jj] = basis_vals
        return jac

    def __repr__(self):
        return "{0}(nvars={1}, nterms={2})".format(
            self.__class__.__name__, self.nvars(), self.nterms())
