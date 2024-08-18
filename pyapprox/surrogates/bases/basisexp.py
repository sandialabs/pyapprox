import copy

import numpy as np

from pyapprox.surrogates.bases.basis import (
    Basis, OrthonormalPolynomialBasis, MultiIndexBasis,
    TensorProductInterpolatingBasis)
from pyapprox.surrogates.bases.linearsystemsolvers import (
    LinearSystemSolver)
from pyapprox.util.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform)
from pyapprox.surrogates.interp.manipulate_polynomials import add_polynomials
from pyapprox.surrogates.polychaos.gpc import (
    multiply_multivariate_orthonormal_polynomial_expansions)
from pyapprox.surrogates.regressor import Regressor
from pyapprox.surrogates.bases.univariate import Monomial1D


class BasisExpansion(Regressor):
    """The base class for any linear basis expansion for multiple
       quantities of interest (QoI)."""

    def __init__(self, basis: Basis, solver: LinearSystemSolver = None,
                 nqoi=1, coef_bounds=None):
        # todo make model accept backend and pass in through with call to super
        super().__init__()
        self._nqoi = int(nqoi)
        self.set_basis(basis, coef_bounds)
        if solver is not None and (type(basis._bkd) is not type(solver._bkd)):
            raise ValueError("Basis and solver must have the same backend.")
        self._jacobian_implemented = basis._jacobian_implemented
        self._solver = solver

    def set_basis(self, basis, coef_bounds=None):
        self.basis = basis
        self._bkd = self.basis._bkd
        init_coef = self._bkd._la_full((self.basis.nterms()*self.nqoi(), ), 0.)
        self._transform = IdentityHyperParameterTransform(backend=self._bkd)
        self._coef = HyperParameter(
            "coef", self.basis.nterms()*self._nqoi, init_coef,
            self._parse_coef_bounds(coef_bounds), None, backend=self._bkd)
        self.hyp_list = HyperParameterList([self._coef])

    def _parse_coef_bounds(self, coef_bounds):
        if coef_bounds is None:
            return [-self._bkd._la_inf(), self._bkd._la_inf()]
        return coef_bounds

    def nqoi(self):
        """
        Return the number of quantities of interest (QoI).
        """
        return self._nqoi

    def nterms(self):
        """
        Return the number of terms in the expansion.
        """
        return self.basis.nterms()

    def nvars(self):
        """
        Return the number of inputs to the basis.
        """
        return self.basis.nvars()

    def set_coefficients(self, coef):
        """
        Set the basis coefficients.

        Parameters
        ----------
        coef : array (nterms, nqoi)
            The basis coefficients for each quantity of interest (QoI)
        """
        if coef.ndim != 2 or coef.shape != (self.basis.nterms(), self.nqoi()):
            raise ValueError(
                "coef shape {0} is must be {1}".format(
                    coef.shape, (self.basis.nterms(), self.nqoi())))
        self._coef.set_values(coef.flatten())

    def get_coefficients(self):
        """
        Get the basis coefficients.

        Returns
        -------
        coef : array (nterms, nqoi)
            The basis coefficients for each quantity of interest (QoI)
        """
        return self._coef.get_values().reshape(
            self.basis.nterms(), self.nqoi())

    def __call__(self, samples):
        """
        Evaluate the expansion at a set of samples.

        ----------
        samples : array (nsamples, nqoi)
            The samples used to evaluate the expansion.

        Returns
        -------
            The values of the expansion for each QoI and sample
        """
        return self.basis(samples) @ self.get_coefficients()

    def __repr__(self):
        return "{0}(basis={1}, nqoi={2})".format(
            self.__class__.__name__, self.basis, self.nqoi())

    def _fit(self, iterate):
        """Fit the expansion by finding the optimal coefficients. """
        if iterate is not None:
            raise ValueError("iterate will be ignored set to None")
        if self._ctrain_values.shape[1] != self.nqoi():
            raise ValueError(
                "Number of cols {0} in values does not match nqoi {1}".format(
                    self._ctrain_values.shape[1], self.nqoi()))
        coef = self._solver.solve(
            self.basis(self._ctrain_samples), self._ctrain_values
        )
        self.set_coefficients(coef)


class MonomialExpansion(BasisExpansion):
    def _parse_basis(self, basis):
        if not isinstance(basis, MultiIndexBasis):
            raise ValueError("basis must be a MultiIndexBasis")
        for basis_1d in basis._bases_1d:
            if not isinstance(basis_1d, Monomial1D):
                raise ValueError(
                    "each 1d basis bust be instance of Monomial1D"
                )
        return basis

    def _group_like_terms(self, coeffs, indices):
        if coeffs.ndim == 1:
            coeffs = coeffs[:, None]

        unique_indices, repeated_idx = self._bkd_la_unique(
            indices, axis=1, return_inverse=True)

        nunique_indices = unique_indices.shape[1]
        unique_coeff = self._bkd._la_full(
            (nunique_indices, coeffs.shape[1]), 0.)
        for ii in range(repeated_idx.shape[0]):
            unique_coeff[repeated_idx[ii]] += coeffs[ii]
        return unique_indices, unique_coeff

    def _add_polynomials(self, indices_list, coeffs_list):
        """
        Add many polynomials together.

        Example:
            p1 = x1**2+x2+x3, p2 = x2**2+2*x3
            p3 = p1+p2

           return the degrees of each term in the the polynomial

           p3 = x1**2+x2+3*x3+x2**2

           [2, 1, 1, 2]

           and the coefficients of each of these terms

           [1., 1., 3., 1.]


        Parameters
        ----------
        indices_list : list [array (num_vars,num_indices_i)]
            List of polynomial indices. indices_i may be different for each
            polynomial

        coeffs_list : list [array (num_indices_i,num_qoi)]
            List of polynomial coefficients. indices_i may be different for
            each polynomial. num_qoi must be the same for each list element.

        Returns
        -------
        indices: array (num_vars,num_terms)
            the polynomial indices of the polynomial obtained from
            summing the polynomials. This will be the union of the indices
            of the input polynomials

        coeffs: array (num_terms,num_qoi)
            the polynomial coefficients of the polynomial obtained from
            summing the polynomials
        """
        num_polynomials = len(indices_list)
        assert num_polynomials == len(coeffs_list)
        all_coeffs = self._bkd._la_vstack(coeffs_list)
        all_indices = self._bkd._la_hstack(indices_list)
        return self._group_like_terms(all_coeffs, all_indices)

    def __add__(self, other):
        indices_list = [self.indices, other.indices]
        coefs_list = [self.coefficients, other.coefficients]
        indices, coefs = add_polynomials(indices_list, coefs_list)
        poly = copy.deepcopy(self)
        poly.basis.set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def __sub__(self, other):
        indices_list = [self.indices, other.indices]
        coefs_list = [self.coefficients, -other.coefficients]
        indices, coefs = add_polynomials(indices_list, coefs_list)
        poly = copy.deepcopy(self)
        poly.basis.set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def _multiply_monomials(self, indices1, coefs1, indices2, coefs2):
        nvars = indices1.shape[0]
        nindices1 = indices1.shape[1]
        nindices2 = indices2.shape[1]
        nqoi = coefs1.shape[1]
        assert nindices1 == coefs1.shape[0]
        assert nindices2 == coefs2.shape[0]
        assert nvars == indices2.shape[0]
        assert nqoi == coefs2.shape[1]

        indices_dict = dict()
        indices, coefs = [], []
        kk = 0
        for ii in range(nindices1):
            index1 = indices1[:, ii]
            coef1 = coefs1[ii]
            for jj in range(nindices2):
                index = index1+indices2[:, jj]
                # hash_array does not work with jit
                # key = hash_array(index)
                # so use a polynomial hash
                key = 0
                for dd in range(index.shape[0]):
                    key = 31*key + int(index[dd])
                coef = coef1*coefs2[jj]
                if key in indices_dict:
                    coefs[indices_dict[key]] += coef
                else:
                    indices_dict[key] = kk
                    indices.append(index)
                    coefs.append(coef)
                    kk += 1
        indices = self._bkd._la_stack(indices, axis=1)
        coefs = self._bkd._la_stack(coefs, axis=0)
        return indices, coefs

    def __mul__(self, other):
        if self.nterms() > other.nterms():
            poly1 = self
            poly2 = other
        else:
            poly1 = other
            poly2 = self
        indices, coefs = self._multiply_monomials(
            poly1.basis.get_indices(), poly1.get_coefficients(),
            poly2.basis.get_indices(), poly2.get_coefficients())
        poly = copy.deepcopy(self)
        poly.basis.set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def __pow__(self, order):
        poly = copy.deepcopy(self)
        if order == 0:
            poly.basis.set_indices(
                self._bkd._la_full([self.nvars(), 1], 0., dtype=int))
            poly.set_coefficients(
                self._bkd._la_full([1, self.nqoi()], 1.))
            return poly

        poly = copy.deepcopy(self)
        for ii in range(2, order+1):
            poly = poly*self
        return poly


class PolynomialChaosExpansion(MonomialExpansion):
    def _parse_basis(self, basis):
        if not isinstance(basis, MultiIndexBasis):
            raise ValueError("basis must be a MultiIndexBasis")
        if not isinstance(basis, OrthonormalPolynomialBasis):
            raise ValueError("basis must be an OrthonormalPolynomialBasis")
        return basis

    def mean(self):
        """
        Compute the mean of the polynomial chaos expansion

        Returns
        -------
        mean : array (nqoi)
            The mean of each quantitity of interest
        """
        return self.coefficients[0, :]

    def variance(self):
        """
        Compute the variance of the polynomial chaos expansion

        Returns
        -------
        var : array (nqoi)
            The variance of each quantitity of interest
        """

        var = self._bkd._la_sum(self.coefficients[1:, :]**2, axis=0)
        return var

    def covariance(self):
        """
        Compute the covariance between each quantity of interest of the
        polynomial chaos expansion

        Returns
        -------
        covar : array (nqoi)
            The covariance between each quantitity of interest
        """
        covar = self.coefficients[1:, :].T @ self.coefficients[1:, :]
        return covar

    def _compute_product_coeffs_1d(
            self, poly, max_degrees1, max_degrees2):
        product_coefs_1d = []
        for ii, poly in enumerate(self.basis._bases_1d):
            max_degree1 = max_degrees1[ii]
            max_degree2 = max_degrees2[ii]
            assert max_degree1 >= max_degree2
            max_degree = max_degree1+max_degree2
            nquad_points = max_degree+1

            poly.set_nterms(nquad_points)
            x_quad, w_quad = poly.gauss_quadrature_rule(nquad_points)
            w_quad = w_quad

            # evaluate the orthonormal basis at the quadrature points. This can
            # be computed once for all degrees up to the maximum degree
            ortho_basis_matrix = poly(x_quad)

            # compute coefficients of orthonormal basis using pseudo
            # spectral projection
            product_coefs_1d.append([])
            for d1 in range(max_degree1+1):
                for d2 in range(min(d1+1, max_degree2+1)):
                    product_vals = (ortho_basis_matrix[:, d1] *
                                    ortho_basis_matrix[:, d2])
                    coefs = (w_quad.T @ (
                        product_vals[:, None] *
                        ortho_basis_matrix[:, :d1+d2+1])).T
                    product_coefs_1d[-1].append(coefs)
        return product_coefs_1d

    def __mul__(self, other):
        if self.basis.nterms() > other.nterms():
            poly1 = self
            poly2 = other
        else:
            poly1 = other
            poly2 = self
        poly1 = copy.deepcopy(poly1)
        poly2 = copy.deepcopy(poly2)
        max_degrees1 = self._bkd._la_max(poly1.basis.get_indices(), axis=1)
        max_degrees2 = self._bkd._la_max(poly2.basis.get_indices(), axis=1)
        product_coefs_1d = self._compute_product_coeffs_1d(
            poly1, max_degrees1, max_degrees2)

        indices, coefs = \
            multiply_multivariate_orthonormal_polynomial_expansions(
                product_coefs_1d, poly1.basis.get_indices(),
                poly1.get_coefficients(),
                poly2.basis.get_indices(), poly2.get_coefficients(),
                backend=self._bkd)

        poly = copy.deepcopy(self)
        poly.basis.set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def marginalize(self, inactive_idx, center=True):
        inactive_idx = self._bkd._la_array(inactive_idx, dtype=int)
        if self.basis.get_indices() is None:
            raise ValueError("PCE cannot be marginalizd as no indices are set")
        if self.get_coefficients() is None:
            raise ValueError(
                "PCE cannot be marginalizd as no coeffocients are set")
        active_idx = self._bkd._la_array(
            np.setdiff1d(np.arange(self.nvars()),
                         self._bkd._la_to_numpy(inactive_idx)), dtype=int)
        marginalized_polys_1d = [
            copy.deepcopy(self.basis._bases_1d[ii]) for ii in active_idx]
        marginalized_basis = OrthonormalPolynomialBasis(marginalized_polys_1d)
        marginalized_array_indices = []
        for ii, index in enumerate(self.basis.get_indices().T):
            if ((self._bkd._la_sum(index) == 0 and center is False) or
                    self._bkd._la_any(index[active_idx]) and
                    (not self._bkd._la_any(index[inactive_idx] > 0))):
                marginalized_array_indices.append(ii)
        marginalized_basis.set_indices(
            self.basis.get_indices()[
                np.ix_(self._bkd._la_to_numpy(active_idx),
                       np.array(marginalized_array_indices))])
        marginalized_pce = PolynomialChaosExpansion(
            marginalized_basis, solver=self._solver, nqoi=self.nqoi())
        marginalized_pce.set_coefficients(
            self._bkd._la_copy(
                self.get_coefficients()[marginalized_array_indices, :]))
        return marginalized_pce


class TensorProductInterpolant:
    # TODO make tensor product interpolant derive from regressor
    # or create InterpolantBases class that has fit that just takes values
    def __init__(self, basis: TensorProductInterpolatingBasis):
        if not isinstance(basis, TensorProductInterpolatingBasis):
            raise ValueError(
                "{0} is not an instance of {1}".format(
                    basis, "TensorProductInterpolatingBasis"
                )
            )
        self._basis = basis
        self._bkd = basis._bkd

    def fit(self, values):
        # fit does not have samples like most surrogates because the samples
        # are predetermined by self._basis
        if values.shape[0] != self._basis.nterms():
            raise ValueError("nodes_1d and values are inconsistent")
        if values.ndim != 2:
            raise ValueError("values must be a 2d array")
        self._values = values

    def __call__(self, samples):
        basis_mat = self._basis(samples)
        return basis_mat @ self._values

    def __repr__(self):
        return "{0}(basis={1})".format(self.__class__.__name__, self._basis)

    def integrate(self):
        quad_weights = self._basis.quadrature_rule()[1]
        return (self._values.T @ quad_weights)[:, 0]
