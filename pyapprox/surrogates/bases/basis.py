from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.numpylinalg import (
    LinAlgMixin,
    NumpyLinAlgMixin,
)
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices_itertools,
)


class Basis(ABC):
    """
    The base class for any multivariate Basis.
    """

    def __init__(self, backend: LinAlgMixin):
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._bkd = backend
        self._jacobian_implemented = False

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


class MultiIndexBasis(Basis):
    """Multivariate basis defined by multi-indices."""

    def __init__(self, indices=None, backend: LinAlgMixin = None):
        super().__init__(backend)
        self._indices = None
        if indices is not None:
            self.set_indices(indices)
        self._jacobian_implemented = True

    def set_hyperbolic_indices(self, nvars, nterms, pnorm):
        indices = self._bkd._la_asarray(
            compute_hyperbolic_indices_itertools(nvars, nterms, pnorm),
            dtype=int,
        )
        self.set_indices(indices)

    def set_tensor_product_indices(self, nterms):
        self.set_indices(
            self._bkd._la_cartesian_product(
                [self._bkd._la_arange(nt) for nt in nterms]
            )
        )

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
        self._indices = self._bkd._la_array(indices, dtype=int)

    def get_indices(self):
        """Return the indices defining the basis terms."""
        return self._indices

    def nterms(self):
        if self._indices is None:
            raise ValueError("indices have not been set")
        return self._indices.shape[1]

    def nvars(self):
        if self._indices is None:
            raise ValueError("indices have not been set")
        return self._indices.shape[0]

    @abstractmethod
    def _basis_vals_1d(self, samples):
        raise NotImplementedError

    def _basis_derivs_1d(self, samples):
        raise NotImplementedError

    def __call__(self, samples):
        if samples.shape[0] != self.nvars():
            raise ValueError("samples must have nrows={0}".format(
                self.nvars()))
        basis_vals_1d = self._basis_vals_1d(samples)
        basis_matrix = basis_vals_1d[0][:, self._indices[0, :]]
        for dd in range(1, self.nvars()):
            basis_matrix *= basis_vals_1d[dd][:, self._indices[dd, :]]
        return basis_matrix

    def jacobian(self, samples):
        basis_vals_1d = self._basis_vals_1d(samples)
        deriv_vals_1d = self._basis_derivs_1d(samples)
        jac = []
        for ii in range(self.nterms()):
            inner_jac = []
            index = self._indices[:, ii]
            for jj in range(self.nvars()):
                # derivative in jj direction
                basis_vals = self._bkd._la_copy(
                    deriv_vals_1d[jj][:, index[jj]]
                )
                # basis values in other directions
                for dd in range(self.nvars()):
                    if dd != jj:
                        basis_vals *= basis_vals_1d[dd][:, index[dd]]
                inner_jac.append(basis_vals)
            jac.append(self._bkd._la_stack(inner_jac, axis=0))
        jac = self._bkd._la_moveaxis(self._bkd._la_stack(jac, axis=0), -1, 0)
        return jac

    def __repr__(self):
        return "{0}(nvars={1}, nterms={2})".format(
            self.__class__.__name__, self.nvars(), self.nterms()
        )


class MonomialBasis(MultiIndexBasis):
    """Multivariate monomial basis."""

    def _univariate_monomial_basis_matrix(self, max_level, samples):
        assert samples.ndim == 1
        basis_matrix = (
            samples[:, None] ** self._bkd._la_arange(max_level + 1)[None, :]
        )
        return basis_matrix

    def _basis_vals_1d(self, samples):
        return [
            self._univariate_monomial_basis_matrix(
                self._indices[dd, :].max(), samples[dd, :]
            )
            for dd in range(self.nvars())
        ]

    def _basis_derivs_1d(self, samples):
        basis_vals_1d = self._basis_vals_1d(samples)
        deriv_vals_1d = []
        for jj in range(len(basis_vals_1d)):
            derivs = [self._bkd._la_full((basis_vals_1d[jj].shape[0], 1), 0.0)]
            if basis_vals_1d[jj].shape[0] > 1:
                derivs.append(
                    self._bkd._la_full((basis_vals_1d[jj].shape[0], 1), 1.0)
                )
            if basis_vals_1d[jj].shape[0] > 2:
                consts = self._bkd._la_arange(2, len(basis_vals_1d[jj]))
                derivs.append(basis_vals_1d[jj][:, 1:-1] * consts)
            deriv_vals_1d.append(self._bkd._la_hstack(derivs))
        return deriv_vals_1d


class OrthonormalPolynomialBasis(MultiIndexBasis):
    """Multivariate orthogonal polynomial basis."""

    def __init__(self, polys_1d, indices=None):
        """
        Parameters
        ----------
        """
        super().__init__(indices, polys_1d[0]._bkd)
        self._polys_1d = polys_1d

        self._max_degree = None
        self._recursion_coefs = None
        self._basis_type_var_indices = None

    def nvars(self):
        # use polys_1d so do not have to set indices to determine nvars
        # like is done for base class
        return len(self._polys_1d)

    def _basis_vals_1d(self, samples):
        return [
            poly(samples[dd, :], self._indices[dd, :].max())
            for dd, poly in enumerate(self._polys_1d)
        ]

    def _basis_derivs_1d(self, samples):
        return [
            poly.derivatives(samples[dd, :], self._indices[dd, :].max(), 1)
            for dd, poly in enumerate(self._polys_1d)
        ]

    def _update_recursion_coefficients(self, ncoefs_per_poly):
        for ii, poly in enumerate(self._polys_1d):
            poly.set_recursion_coefficients(ncoefs_per_poly[ii])

    def set_indices(self, indices):
        if indices.shape[0] != len(self._polys_1d):
            raise ValueError(
                "indices.shape[0] {0} doesnt match len(polys_1d) {1}".format(
                    indices.shape[0], len(self._polys_1d)
                )
            )
        super().set_indices(indices)
        self._update_recursion_coefficients(
            self._bkd._la_max(self._indices, axis=1) + 1
        )
