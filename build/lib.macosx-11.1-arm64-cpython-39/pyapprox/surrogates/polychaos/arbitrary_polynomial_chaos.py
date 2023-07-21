from functools import partial
import numpy as np

from pyapprox.surrogates.interp.sparse_grid import (
    get_sparse_grid_samples_and_weights
)
from pyapprox.surrogates.interp.adaptive_sparse_grid import (
    CombinationSparseGrid,
    max_level_admissibility_function, variance_refinement_indicator,
    get_sparse_grid_univariate_leja_quadrature_rules_economical
)

from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, evaluate_multivariate_orthonormal_polynomial
)
from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.surrogates.interp.indexing import set_difference
from pyapprox.surrogates.interp.monomial import monomial_basis_matrix


class APC(PolynomialChaosExpansion):
    """
    A polynomial chaos expansion for dependent random variables.
    """
    def __init__(self, compute_moment_matrix_function=None, moments=None,
                 compute_grammian_function=None):
        """
        Constructor.

        Parameters
        ----------
        compute_moment_matrix_function : callable
            A function to compute ``np.sqrt(np.diag(w)).dot(B)`` where ``w``
            are the positive weights of the quadrature rule and B is the
            basis matrix evaluated at the quadrature points.
            It must have the signature

            ``compute_moment_matrix_function(basis_mat_fun) -> np.ndarray (
              nsamples, nbasis)``

            where basis_mat_fun is a function with the signature
            ``basis_mat_fun(samples) -> np.ndarray(nsamples, nbasis)``
             where samples : np.ndarray (nvars, nsamples)

        moments : np.ndarray (nbasis, nbasis)
            The symmetric matrix containing the inner product of each
            polynomial basis with every polynomial basis
            (including itself - diagonal entries)

        compute_grammian_function : callable
            A function to compute the inner products of all basis combinations
            with the signature

            ``compute_moment_matrix_function(basis_mat) -> np.ndarray (
              nbasis, nbasis)``

            where basis_mat : np.ndarray (nsamples, nbasis)
        """
        super(APC, self).__init__()
        self.compute_moment_matrix_function = compute_moment_matrix_function
        self.compute_grammian_function = compute_grammian_function
        self.moment_matrix_cond = None
        self.moments = moments
        self.R_inv = None

    def compute_rotation(self):
        if self.moments is not None:
            if type(self.moments) != np.ndarray:
                raise Exception('moments was given but was not a np.ndarray')
            assert self.compute_moment_matrix_function is None
            self.R_inv = compute_rotation_from_moments_linear_system(
                self.moments)
        elif self.compute_moment_matrix_function is not None:
            if not callable(self.compute_moment_matrix_function):
                msg = 'compute_moment_matrix_function was given but was '
                msg += 'not a callable function'
                raise Exception(msg)
            assert self.moments is None
            assert self.compute_grammian_function is None

            moment_matrix = self.compute_moment_matrix_function(
                self.unrotated_basis_matrix)
            self.R_inv = compute_rotation_qr(moment_matrix)
            self.moment_matrix_cond = np.linalg.cond(moment_matrix)
        elif self.compute_grammian_function is not None:
            if not callable(self.compute_grammian_function):
                msg = 'compute_grammian_function was given but was '
                msg += 'not a callable function'
                raise Exception(msg)
            assert self.moments is None
            assert self.compute_moment_matrix_function is None

            grammian = self.compute_grammian_function(
                self.unrotated_basis_matrix, self.indices)
            # cholesky requires moment_matrix function to return grammian
            # A'*A not basis matrix A
            assert grammian.shape[0] == grammian.shape[1]
            self.R_inv = compute_rotation_cholesky(grammian)
            self.moment_matrix_cond = np.linalg.cond(grammian)
        else:
            raise Exception

    def unrotated_canonical_basis_matrix(self, canonical_samples):
        """
        Cannot just call super(APCE,self).canonical_basis because I was
        running into inheritance problems.
        """
        deriv_order = 0
        if self.recursion_coeffs is not None:
            unrotated_basis_matrix = \
                evaluate_multivariate_orthonormal_polynomial(
                    canonical_samples, self.indices, self.recursion_coeffs,
                    deriv_order, self.basis_type_index_map)
        else:
            unrotated_basis_matrix = monomial_basis_matrix(
                self.indices, canonical_samples)
        return unrotated_basis_matrix

    def unrotated_basis_matrix(self, samples):
        assert samples.ndim == 2
        assert samples.shape[0] == self.num_vars()
        canonical_samples = self.var_trans.map_to_canonical(
            samples)
        matrix = self.unrotated_canonical_basis_matrix(canonical_samples)
        return matrix

    def canonical_basis_matrix(self, canonical_samples, opts=dict()):
        deriv_order = opts.get('deriv_order', 0)
        assert deriv_order == 0
        unrotated_basis_mat = self.unrotated_canonical_basis_matrix(
            canonical_samples)

        if self.R_inv is not None:
            # basis_matrix = np.dot(unrotated_basis_mat, self.R_inv)
            basis_matrix = unrotated_basis_mat.dot(
                self.R_inv[:self.num_terms(), :self.num_terms()])
        else:
            basis_matrix = unrotated_basis_mat
        return basis_matrix

    def basis_matrix(self, samples, opts=dict()):
        assert samples.ndim == 2
        assert samples.shape[0] == self.num_vars()
        canonical_samples = self.var_trans.map_to_canonical(
            samples)
        return self.canonical_basis_matrix(canonical_samples, opts)

    # def basis_matrix(self,samples):
    #     if self.compute_moment_matrix_function is not None:
    #         return np.dot(self.unrotated_basis_matrix(samples),self.R_inv)
    #     else:
    #         return self.unrotated_basis_matrix(samples)

    # def unrotated_basis_matrix(self,samples):
    #     return super(APC,self).basis_matrix(samples)

    def set_indices(self, indices):
        # need to perform check before base class updated self.indices
        # if (self.compute_moment_matrix_function is not None and
        #         (self.indices is None or
        #              self.indices.shape[1]!=indices.shape[1] or
        #         not np.allclose(indices,self.indices))):
        #     update_rotation=True
        # else:
        #     update_rotation=False

        # TODO eventually use following. need to distinguish between
        # indices for pce and indices used for computing basis rotation

        update_indices = False
        if self.indices is None:
            update_indices = True
        else:
            # check indices is subset of self.indices
            if indices.shape[1] > self.indices.shape[1]:
                update_indices = True
            else:
                # check indices is subset of self.indices
                update_indices = set_difference(
                    self.indices, indices).shape[1] != 0

        if update_indices:
            super(APC, self).set_indices(indices)
            if (self.compute_moment_matrix_function is not None or
                self.moments is not None or
                    self.compute_grammian_function is not None):
                self.compute_rotation()

    # TODO: __call__ take advantage of fact that
    # basis_matrix.dot(R_inv.dot(coeff))
    # (m x n)*((n x n)*(n*1)) = n**2+m*n
    # is faster than
    # (basis_matrix.dot(R_inv)).dot(coeff)
    # ((m x n)*(n x n))*(n*1) = m*n**2+m*n
    # faster way can also leverage subsitution solve
    # instead of directly inverting R to get R_inv


def compute_moment_matrix_from_samples(basis_matrix_func, samples):
    return basis_matrix_func(samples)/np.sqrt(samples.shape[1])


def compute_moment_matrix_using_tensor_product_quadrature(
        basis_matrix_func, num_samples, num_vars,
        univariate_quadrature_rule, density_function=None):
    """
    Parameters
    ----------
    num_samples : integer
       The number of samples in the 1D quadrature rule

    univariate_quadrature_rule : tuple (x,w)
       x : np.ndarray (num_samples) the quadrature points in the user space
       w : np.ndarray (num_samples) the quadrature weights

    density_function : callable
       v = density_function(x)
       A probability density function. If not None then quadrature rule
       should be for lebesque measure and weights will be multiplied by the
       value of the density at the quarature points
    """
    samples, weights = get_tensor_product_quadrature_rule(
        num_samples, num_vars, univariate_quadrature_rule, None,
        density_function)
    basis_matrix = basis_matrix_func(samples)
    moment_matrix = np.dot(np.diag(np.sqrt(weights)), basis_matrix)
    return moment_matrix


def compute_coefficients_of_unrotated_basis(coefficients, R_inv):
    r"""
    Given pce coefficients a such that p(z)=\Phi(Z)*a
    where phi is the multivariate orthgonal Gram-Schmidt basis compute
    coefficients of tensor-product basis \psi where \Phi(Z)=\Psi(Z)*R_inv
    and \Phi(Z),\Psi(Z) are vandermonde matrices evaluate at the samples Z.
    """
    num_terms = coefficients.shape[0]
    unrotated_basis_coefficients = np.zeros_like(coefficients)
    for ii in range(num_terms):
        unrotated_basis_coefficients[ii, :] =\
            np.sum(coefficients[ii:, :]*(R_inv[ii, ii:])[:, np.newaxis],
                   axis=0)
        # for kk in range(pce.coefficients.shape[1]):
        # for jj in range(ii,num_terms):
        #    unrotated_basis_coefficients[ii,kk]+=\
        #        pce.coefficients[jj,kk]*pce.R_inv[ii,jj]
    return unrotated_basis_coefficients


class FPC(APC):
    def __init__(self, compute_moment_matrix_function):
        super(FPC, self).__init__(compute_moment_matrix_function)

    def configure(self, opts):
        super(FPC, self).configure(opts)
        self.truncation_tol = opts['truncation_tol']

    def compute_rotation(self):
        moment_matrix = self.compute_moment_matrix_function(
            self.unrotated_basis_matrix)
        assert moment_matrix.shape[0] >= moment_matrix.shape[1]
        grammian = np.dot(moment_matrix.T, moment_matrix)
        U_factor, S_factor, V_factor = np.linalg.svd(grammian)
        II = np.where(S_factor > self.truncation_tol)[0]
        if (II.shape[0] < grammian.shape[0]):
            raise Exception('frame approximation produced low-rank basis')
        truncated_singular_vectors = (U_factor[:, II]/np.sqrt(S_factor[II]))
        # num_truncated_bases = truncated_singular_vectors.shape[1]
        self.R_inv = truncated_singular_vectors
        self.moment_matrix_cond = S_factor[II].max()/S_factor[II].min()

    def num_terms(self):
        # truncated svd creates basis with num_terms <= num_indices
        return self.R_inv.shape[1]


def compute_rotation_from_moments_linear_system(poly_moments):
    num_terms = poly_moments.shape[0]
    R_inv = np.zeros((num_terms, num_terms), dtype=float)
    R_inv[0, 0] = 1.
    for kk in range(1, num_terms):
        moment_matrix = np.zeros((kk+1, kk+1), dtype=float)
        for ii in range(kk):
            for jj in range(kk+1):
                moment_matrix[ii, jj] = poly_moments[ii, jj]
        moment_matrix[-1, -1] = 1.

        rhs = np.zeros(kk+1)
        rhs[-1] = 1.
        rotated_basis_coefs = np.linalg.solve(moment_matrix, rhs)
        R_inv[:kk+1, kk] = rotated_basis_coefs

        # orthonormalize
        l2_norm = 0.
        for ii in range(kk+1):
            for jj in range(kk+1):
                l2_norm += R_inv[ii, kk]*R_inv[jj, kk]*poly_moments[ii, jj]
        R_inv[:kk+1, kk] /= np.sqrt(l2_norm)

    return R_inv


def compute_rotation_from_moments_gram_schmidt(poly_moments):
    num_terms = poly_moments.shape[0]
    R_inv = np.zeros((num_terms, num_terms), dtype=float)
    R_inv[0, 0] = 1.
    for kk in range(1, num_terms):
        R_inv[kk, kk] = 1.
        for ii in range(kk):
            # compute <e_k,phi_i>
            numerator = 0.
            for jj in range(ii+1):
                numerator += poly_moments[kk, jj]*R_inv[jj, ii]
            # basis_norms are one because of Step ***
            ratio = numerator  # /basis_norms[ii]
            # compute contribution to r_ik
            for jj in range(ii+1):
                R_inv[jj, kk] += -ratio*R_inv[jj, ii]

        # compute <phi_k,phi_k>
        basis_norm = 0
        for ii in range(kk+1):
            for jj in range(kk+1):
                basis_norm += R_inv[ii, kk]*R_inv[jj, kk]*poly_moments[ii, jj]
        R_inv[:kk+1, kk] /= np.sqrt(basis_norm)  # Step ***

    return R_inv


def compute_rotation_qr(moment_matrix):
    assert moment_matrix.shape[0] >= moment_matrix.shape[1]
    Q_factor, R_factor = np.linalg.qr(moment_matrix)
    for ii in range(R_factor.shape[0]):
        if R_factor[ii, ii] < 0.:
            R_factor[ii, :] *= -1.
            Q_factor[:, ii] *= -1.
    R_inv = np.linalg.inv(R_factor)
    return R_inv


def compute_rotation_cholesky(grammian_matrix):
    assert grammian_matrix.shape[0] == grammian_matrix.shape[1]
    L_factor = np.linalg.cholesky(grammian_matrix)
    R_inv = np.linalg.inv(L_factor.T)
    return R_inv


def compute_polynomial_moments_using_tensor_product_quadrature(
        basis_matrix_function, num_samples, num_vars,
        univariate_quadrature_rule, density_function=None):
    """
    Compute the moments of a polynomial basis using tensor product quadrature

    Parameters
    ----------
    num_samples : integer
       The number of samples in the 1D quadrature rule

    univariate_quadrature_rule : tuple (x,w)
       x : np.ndarray (num_samples) the quadrature points in the user space
       w : np.ndarray (num_samples) the quadrature weights

    density_function : callable
       v = density_function(x)
       A probability density function. If not None then quadrature rule
       should be for lebesque measure and weights will be multiplied by the
       value of the density at the quarature points

    Returns
    -------
    poly_moments : np.ndarray (num_terms, num_terms)
       The symmetric matrix containing the inner product of each polynomial
       basis with every polynomial basis (including itself - diagonal entries)
    """
    samples, weights = get_tensor_product_quadrature_rule(
        num_samples, num_vars, univariate_quadrature_rule, None,
        density_function)
    basis_matrix = basis_matrix_function(samples)

    poly_moments = np.empty((basis_matrix.shape[1], basis_matrix.shape[1]),
                            dtype=float)
    for ii in range(basis_matrix.shape[1]):
        for jj in range(ii, basis_matrix.shape[1]):
            poly_moments[ii, jj] = np.dot(
                basis_matrix[:, ii]*basis_matrix[:, jj], weights)
            poly_moments[jj, ii] = poly_moments[ii, jj]
    return poly_moments


def compute_grammian_matrix_using_combination_sparse_grid(
        basis_matrix_function, dummy_indices, var_trans, max_num_samples,
        error_tol=0,
        density_function=None, quad_rule_opts=None):
    num_vars = var_trans.num_vars()
    sparse_grid = CombinationSparseGrid(num_vars)
    admissibility_function = partial(
        max_level_admissibility_function, np.inf, [np.inf]*num_vars,
        max_num_samples, error_tol, verbose=True)
    if quad_rule_opts is None:
        quad_rules, growth_rules, unique_quadrule_indices = \
            get_sparse_grid_univariate_leja_quadrature_rules_economical(
                var_trans)
    else:
        quad_rules = quad_rule_opts['quad_rules']
        growth_rules = quad_rule_opts['growth_rules']
        unique_quadrule_indices = quad_rule_opts.get(
            'unique_quadrule_indices')
    if density_function is None:
        def density_function(samples): return np.ones(samples.shape[1])

    def function(samples):
        # need to make sure that basis_matrix_function takes
        # points in user domain and not canonical domain
        basis_matrix = basis_matrix_function(samples)
        pdf_vals = density_function(samples)
        vals = []
        for ii in range(basis_matrix.shape[1]):
            for jj in range(ii, basis_matrix.shape[1]):
                vals.append(
                    basis_matrix[:, ii]*basis_matrix[:, jj]*pdf_vals)
        return np.asarray(vals).T

    sparse_grid.setup(function, None,
                      partial(variance_refinement_indicator),
                      admissibility_function, growth_rules, quad_rules,
                      var_trans,
                      unique_quadrule_indices=unique_quadrule_indices)
    sparse_grid.build()
    # todo allow level to be passed in per dimension so I can base it on
    # sparse_grid.subspace_indices.max(axis=1)
    samples, weights = get_sparse_grid_samples_and_weights(
        num_vars, sparse_grid.subspace_indices.max(),
        sparse_grid.univariate_quad_rule,
        sparse_grid.univariate_growth_rule, sparse_grid.subspace_indices)[:2]
    samples = var_trans.map_from_canonical(samples)
    weights *= density_function(samples)
    basis_matrix = basis_matrix_function(samples)
    moment_matrix = np.dot(basis_matrix.T*weights, basis_matrix)
    return moment_matrix
