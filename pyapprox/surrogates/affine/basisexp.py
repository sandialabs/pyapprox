import copy
from typing import Tuple, List, Union

import numpy as np

from pyapprox.util.backends.template import Array
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
    GaussQuadratureRule,
    OrthonormalPolynomial1D,
)
from pyapprox.util.linalg import (
    flattened_rectangular_lower_triangular_matrix_index,
)
from pyapprox.surrogates.univariate.lagrange import (
    UnivariateLagrangeBasis,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.marginals import (
    Marginal,
    ScipyMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.basis import (
    Basis,
    OrthonormalPolynomialBasis,
    MultiIndexBasis,
    TensorProductInterpolatingBasis,
    TrigonometricBasis,
    FourierBasis,
    TensorProductQuadratureRule,
)
from pyapprox.surrogates.affine.linearsystemsolvers import (
    LinearSystemSolver,
    LstSqSolver,
)
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
    IdentityHyperParameterTransform,
)
from pyapprox.surrogates.regressor import Regressor, Surrogate
from pyapprox.surrogates.univariate.base import (
    Monomial1D,
    UnivariateBasis,
)


class BasisExpansion(Regressor):
    """The base class for any linear basis expansion for multiple
    quantities of interest (QoI)."""

    def __init__(
        self,
        basis: Basis,
        solver: LinearSystemSolver = None,
        nqoi=1,
        coef_bounds=None,
        fixed: bool = False,
    ):
        super().__init__(basis._bkd)
        self._nqoi = int(nqoi)
        self.set_basis(basis, coef_bounds, fixed)
        if solver is not None:
            self.set_solver(solver)

    def set_solver(self, solver: LinearSystemSolver):
        if not self._bkd.bkd_equal(solver._bkd, self._bkd):
            raise ValueError("solver must have the same backend.")
        self._solver = solver

    def hyperparam_jacobian(self, active_opt_params: Array) -> Array:
        return self._basis(self._train_samples)

    def jacobian_implemented(self) -> bool:
        return self._basis.jacobian_implemented()

    def hessian_implemented(self) -> bool:
        return self._basis.hessian_implemented()

    def apply_hessian_implemented(self) -> bool:
        return self._basis.hessian_implemented()

    def apply_weighted_hessian_implemented(self) -> bool:
        return self._basis.hessian_implemented()

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def basis(self) -> Basis:
        return self._basis

    def set_basis(
        self, basis: Basis, coef_bounds: Array = None, fixed: bool = False
    ):
        self._basis = basis
        self._bkd = self._basis._bkd
        init_coef = self._bkd.full((self._basis.nterms() * self.nqoi(),), 0.0)
        self.set_coefficient_bounds(init_coef, coef_bounds, fixed)

    def _parse_coef_bounds(self, coef_bounds: Array):
        if coef_bounds is None:
            return [-self._bkd.inf(), self._bkd.inf()]
        return coef_bounds

    def set_coefficient_bounds(
        self, init_coef: Array, coef_bounds: Array, fixed: bool = False
    ):
        self._transform = IdentityHyperParameterTransform(backend=self._bkd)
        self._coef = HyperParameter(
            "coef",
            self._basis.nterms() * self._nqoi,
            init_coef,
            self._parse_coef_bounds(coef_bounds),
            None,
            fixed,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._coef])

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).
        """
        return self._nqoi

    def nterms(self) -> int:
        """
        Return the number of terms in the expansion.
        """
        return self._basis.nterms()

    def nvars(self) -> int:
        """
        Return the number of inputs to the basis.
        """
        return self._basis.nvars()

    def set_coefficients(self, coef: Array):
        """
        Set the basis coefficients.

        Parameters
        ----------
        coef : array (nterms, nqoi)
            The basis coefficients for each quantity of interest (QoI)
        """
        if coef.ndim != 2 or coef.shape != (self._basis.nterms(), self.nqoi()):
            raise ValueError(
                "coef shape is {0} but must be {1}".format(
                    coef.shape, (self._basis.nterms(), self.nqoi())
                )
            )
        self._coef.set_values(coef.flatten())

    def get_coefficients(self) -> Array:
        """
        Get the basis coefficients.

        Returns
        -------
        coef : array (nterms, nqoi)
            The basis coefficients for each quantity of interest (QoI)
        """
        return self._coef.get_values().reshape(
            self._basis.nterms(), self.nqoi()
        )

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the expansion at a set of samples.

        ----------
        samples : array (nsamples, nqoi)
            The samples used to evaluate the expansion.

        Returns
        -------
            The values of the expansion for each QoI and sample
        """
        return self._basis(samples) @ self.get_coefficients()

    def _many_jacobian(self, samples: Array) -> Array:
        # jacobian shape (nsamples, nqoi, nvars)
        return self._bkd.einsum(
            "ijk, jl->ilk",
            self._basis.jacobian(samples),
            self.get_coefficients(),
        )

    def _jacobian(self, sample: Array) -> Array:
        return self._many_jacobian(sample)[0]

    def _many_hessian(self, samples):
        hess = self._basis.hessian(samples)
        # hess shape is (nsamples, nterms, nvars, nvars)
        # coef shape is (nterms, nqoi)
        return self._bkd.einsum(
            "ijkl, jm->imkl", hess, self.get_coefficients()
        )

    def _hessian(self, sample: Array) -> Array:
        return self._many_hessian(sample)[0]

    def _apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        Compute the weighted_Hessian-vector product without the sample index
        using Einstein summation.

        Returns
        -------
        hvp : Array (nqoi, nvars, nvecs)
            Hessian-vector product.
        """
        # hess : Array
        #     Hessian of the basis functions.
        #     Shape: (1, nterms, nvars, nvars)
        # coef : Array
        #     Coefficients associated with the basis terms for each QoI.
        #     Shape: (nterms, nqoi)
        # vec : Array
        #     Vector to multiply with the Hessian.
        #     Shape: (nvars, nvecs)
        # weights : Array
        #     Vector to sum the rows of the Hessian.
        #     Shape: (nvars, 1)

        # "jkl": Represents the indices of the hess tensor:
        #     j: Basis term index (nterms).
        #     k: First variable index (nvars)
        #     l: Second variable index (nvars).
        # "jm": Represents the indices of the coef tensor:
        #     j: Basis term index (nterms).
        #     m: QoI index (nqoi).
        # "l": Represents the indices of the vector tensor:
        #     l: Second variable index (nvars).
        # "mk": Specifies the output tensor:
        #     m: QoI index (nqoi).
        #     k: First variable index (nvars).

        hess = self._basis.hessian(sample)
        coef = self.get_coefficients()
        # Perform the Einstein summation to compute the Hessian-vector product
        # use optimize=True to find the best contraction
        return self._bkd.einsum(
            "jkl, jm, m, ln -> kn", hess[0], coef, weights[:, 0], vec
        )

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return self._apply_weighted_hessian(
            sample, vec, self._bkd.ones((1, 1))
        )

    def __repr__(self):
        return "{0}(basis={1}, nqoi={2})".format(
            self.__class__.__name__, self._basis, self.nqoi()
        )

    def _fit(self, iterate: Array):
        """Fit the expansion by finding the optimal coefficients."""
        if iterate is not None:
            raise ValueError("iterate will be ignored set to None")
        if self._ctrain_values.shape[1] != self.nqoi():
            raise ValueError(
                "Number of cols {0} in values does not match nqoi {1}".format(
                    self._ctrain_values.shape[1], self.nqoi()
                )
            )
        self._solver.set_surrogate(self)
        coef = self._solver.solve(
            self._basis(self._ctrain_samples), self._ctrain_values
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

    def _group_like_terms(
        self, coeffs: Array, indices: Array
    ) -> Tuple[Array, Array]:
        if coeffs.ndim == 1:
            coeffs = coeffs[:, None]

        unique_indices, repeated_idx = self._bkd.unique(
            indices, axis=1, return_inverse=True
        )

        nunique_indices = unique_indices.shape[1]
        unique_coeff = self._bkd.full((nunique_indices, coeffs.shape[1]), 0.0)
        for ii in range(repeated_idx.shape[0]):
            unique_coeff[repeated_idx[ii]] += coeffs[ii]
        return unique_indices, unique_coeff

    def _add_polynomials(
        self, indices_list: List[Array], coeffs_list: List[Array]
    ) -> Tuple[Array, Array]:
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
        all_coeffs = self._bkd.vstack(coeffs_list)
        all_indices = self._bkd.hstack(indices_list)
        return self._group_like_terms(all_coeffs, all_indices)

    def _signed_add(
        self, other: "MonomialExpansion", other_sign: float
    ) -> "MonomialExpansion":
        indices_list = [self._basis.get_indices(), other.basis().get_indices()]
        coefs_list = [
            self.get_coefficients(),
            other_sign * other.get_coefficients(),
        ]
        indices, coefs = self._add_polynomials(indices_list, coefs_list)
        poly = copy.deepcopy(self)
        poly.basis().set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def __add__(
        self, other: Union["MonomialExpansion", float]
    ) -> "MonomialExpansion":
        if isinstance(other, float) or isinstance(other, int):
            return self._add_constant(other)
        return self._signed_add(other, 1.0)

    def __radd__(
        self, other: Union["MonomialExpansion", float]
    ) -> "MonomialExpansion":
        return self.__add__(other)

    def _add_constant(self, other: float) -> "MonomialExpansion":
        poly = copy.deepcopy(self)
        coefs = self.get_coefficients()
        coefs[self._constant_basis_coefficient_idx()] += other
        poly.set_coefficients(coefs)
        return poly

    def __sub__(
        self, other: Union["MonomialExpansion", float]
    ) -> "MonomialExpansion":
        if isinstance(other, float) or isinstance(other, int):
            return self._add_constant(other)
        return self._signed_add(other, -1.0)

    def _multiply_monomials(
        self, indices1: Array, coefs1: Array, indices2: Array, coefs2: Array
    ) -> Tuple[Array, Array]:
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
                index = index1 + indices2[:, jj]
                # hash_array does not work with jit
                # key = hash_array(index)
                # so use a polynomial hash
                key = 0
                for dd in range(index.shape[0]):
                    key = 31 * key + int(index[dd])
                coef = coef1 * coefs2[jj]
                if key in indices_dict:
                    coefs[indices_dict[key]] += coef
                else:
                    indices_dict[key] = kk
                    indices.append(index)
                    coefs.append(coef)
                    kk += 1
        indices = self._bkd.stack(indices, axis=1)
        coefs = self._bkd.stack(coefs, axis=0)
        return indices, coefs

    def _mul_constant(self, other: float) -> "MonomialExpansion":
        poly = copy.deepcopy(self)
        poly.set_coefficients(self.get_coefficients() * other)
        return poly

    def _mul(self, other: "MonomialExpansion") -> "MonomialExpansion":
        if self.nterms() > other.nterms():
            poly1 = self
            poly2 = other
        else:
            poly1 = other
            poly2 = self
        indices, coefs = self._multiply_monomials(
            poly1.basis().get_indices(),
            poly1.get_coefficients(),
            poly2.basis().get_indices(),
            poly2.get_coefficients(),
        )
        poly = copy.deepcopy(self)
        poly.basis().set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def __mul__(
        self, other: Union["MonomialExpansion", float]
    ) -> "MonomialExpansion":
        if (
            isinstance(other, float)
            or isinstance(other, int)
            or self._bkd.is_scalar_array(other)
        ):
            return self._mul_constant(other)
        return self._mul(other)

    def __rmul__(
        self, other: Union["MonomialExpansion", float]
    ) -> "MonomialExpansion":
        # if isinstance(other, float) or isinstance(other, int):
        #     return self._mul_constant(other)
        # return self._mul(other)
        return self.__mul__(other)

    def __pow__(self, order: int) -> "MonomialExpansion":
        poly = copy.deepcopy(self)
        if order == 0:
            poly.basis().set_indices(
                self._bkd.full([self.nvars(), 1], 0.0, dtype=int)
            )
            poly.set_coefficients(self._bkd.full([1, self.nqoi()], 1.0))
            return poly

        poly = copy.deepcopy(self)
        for ii in range(2, order + 1):
            poly = poly * self
        return poly

    def _constant_basis_coefficient_idx(self) -> Array:
        return self._bkd.where(
            ~self._bkd.any(self._basis.get_indices(), axis=0)
        )[0]

    def mean_iid_uniform_marginals(self, marginals: List[Marginal]) -> Array:
        for marginal in marginals:
            if (
                not isinstance(marginal, UniformMarginal)
                and not isinstance(marginal, ScipyMarginal)
                and not marginal._name == "uniform"
            ):
                raise ValueError("marginals must all be uniform")
        bounds = self._bkd.stack([m.interval(1) for m in marginals], axis=0)
        indices = self.basis().get_indices()
        L = self._bkd.prod(bounds[:, 1] - bounds[:, 0])
        vals = (
            self._bkd.prod(
                (
                    bounds[:, 1:2] ** (indices + 1.0)
                    - bounds[:, 0:1] ** (indices + 1.0)
                )
                / (indices + 1.0),
                axis=0,
            )
            / L
        )
        integral = self._bkd.sum(
            vals[:, None] * self.get_coefficients(), axis=0
        )
        return integral

    def variance_iid_uniform_marginals(
        self, marginals: List[Marginal]
    ) -> Array:
        poly = self * self
        return (
            poly.mean_iid_uniform_marginals(marginals)
            - self.mean_iid_uniform_marginals(marginals) ** 2
        )

    def mean_iid_gaussian_marginals(self) -> Array:
        indices = self.basis().get_indices()
        vals = self._bkd.prod(
            (2.0) ** (-indices / 2)
            * self._bkd.factorial(indices)
            / self._bkd.factorial(indices / 2),
            axis=0,
        )
        idx = self._bkd.any(indices % 2 == 1, axis=0)
        vals[idx] = 0
        integral = self._bkd.sum(
            vals[:, None] * self.get_coefficients(), axis=0
        )
        return integral


class PolynomialChaosExpansion(MonomialExpansion):
    def _parse_basis(self, basis):
        if not isinstance(basis, MultiIndexBasis):
            raise ValueError("basis must be a MultiIndexBasis")
        if not isinstance(basis, OrthonormalPolynomialBasis):
            raise ValueError("basis must be an OrthonormalPolynomialBasis")
        return basis

    def mean(self) -> Array:
        """
        Compute the mean of the polynomial chaos expansion

        Returns
        -------
        mean : array (nqoi)
            The mean of each quantitity of interest
        """
        return self.get_coefficients()[
            self._constant_basis_coefficient_idx(), :
        ]

    def variance(self) -> Array:
        """
        Compute the variance of the polynomial chaos expansion

        Returns
        -------
        var : array (nqoi)
            The variance of each quantitity of interest
        """
        coefs = self.get_coefficients()
        return self._bkd.sum(coefs**2, axis=0) - self.mean() ** 2

    def covariance(self) -> Array:
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
        self,
        poly: "PolynomialChaosExpansion",
        max_degrees1: Array,
        max_degrees2: Array,
    ) -> List[Array]:
        product_coefs_1d = []
        for ii, poly in enumerate(self._basis._bases_1d):
            max_degree1 = max_degrees1[ii]
            max_degree2 = max_degrees2[ii]
            assert max_degree1 >= max_degree2
            max_degree = max_degree1 + max_degree2
            nquad_points = max_degree + 1

            poly.set_nterms(nquad_points)
            x_quad, w_quad = poly.gauss_quadrature_rule(nquad_points)
            w_quad = w_quad

            # evaluate the orthonormal basis at the quadrature points. This can
            # be computed once for all degrees up to the maximum degree
            ortho_basis_matrix = poly(x_quad)

            # compute coefficients of orthonormal basis using pseudo
            # spectral projection
            product_coefs_1d.append([])
            for d1 in range(max_degree1 + 1):
                for d2 in range(min(d1 + 1, max_degree2 + 1)):
                    product_vals = (
                        ortho_basis_matrix[:, d1] * ortho_basis_matrix[:, d2]
                    )
                    coefs = (
                        w_quad.T
                        @ (
                            product_vals[:, None]
                            * ortho_basis_matrix[:, : d1 + d2 + 1]
                        )
                    ).T
                    product_coefs_1d[-1].append(coefs)
        return product_coefs_1d

    def _mul(
        self, other: "PolynomialChaosExpansion"
    ) -> "PolynomialChaosExpansion":
        if self._basis.nterms() > other.nterms():
            poly1 = self
            poly2 = other
        else:
            poly1 = other
            poly2 = self
        poly1 = copy.deepcopy(poly1)
        poly2 = copy.deepcopy(poly2)
        max_degrees1 = self._bkd.max(poly1.basis().get_indices(), axis=1)
        max_degrees2 = self._bkd.max(poly2.basis().get_indices(), axis=1)
        product_coefs_1d = self._compute_product_coeffs_1d(
            poly1, max_degrees1, max_degrees2
        )

        indices, coefs = (
            self._multiply_multivariate_orthonormal_polynomial_expansions(
                product_coefs_1d,
                poly1.basis().get_indices(),
                poly1.get_coefficients(),
                poly2.basis().get_indices(),
                poly2.get_coefficients(),
            )
        )

        poly = copy.deepcopy(self)
        poly.basis().set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def marginalize(
        self, inactive_idx: Array, center: bool = True
    ) -> "PolynomialChaosExpansion":
        inactive_idx = self._bkd.array(inactive_idx, dtype=int)
        if self._basis.get_indices() is None:
            raise ValueError("PCE cannot be marginalizd as no indices are set")
        if self.get_coefficients() is None:
            raise ValueError(
                "PCE cannot be marginalizd as no coeffocients are set"
            )
        active_idx = self._bkd.array(
            np.setdiff1d(
                np.arange(self.nvars()), self._bkd.to_numpy(inactive_idx)
            ),
            dtype=int,
        )
        marginalized_polys_1d = [
            copy.deepcopy(self._basis._bases_1d[ii]) for ii in active_idx
        ]
        marginalized_basis = OrthonormalPolynomialBasis(marginalized_polys_1d)
        marginalized_array_indices = []
        for ii, index in enumerate(self._basis.get_indices().T):
            if (
                (self._bkd.sum(index) == 0 and center is False)
                or self._bkd.any(index[active_idx])
                and (not self._bkd.any(index[inactive_idx] > 0))
            ):
                marginalized_array_indices.append(ii)

        marginalized_basis.set_indices(
            self._basis.get_indices()[
                np.ix_(
                    self._bkd.to_numpy(active_idx),
                    np.array(marginalized_array_indices),
                )
            ]
        )
        marginalized_pce = PolynomialChaosExpansion(
            marginalized_basis,
            solver=(self._solver if hasattr(self, "_solver") else None),
            nqoi=self.nqoi(),
        )
        marginalized_pce.set_coefficients(
            self._bkd.copy(
                self.get_coefficients()[marginalized_array_indices, :]
            )
        )
        return marginalized_pce

    def _shift_momomial_expansion(
        self, coef: Array, shift: float, scale: float
    ) -> Array:
        shifted_coef = self._bkd.zeros(coef.shape)
        # shifted_coef[0] = coef[0]
        shifted_coef = self._bkd.up(shifted_coef, (0,), coef[0], axis=0)
        nterms = coef.shape[0]
        for ii in range(1, nterms):
            temp = self._bkd.array(
                np.polynomial.polynomial.polypow([1, -shift], ii)
            )[:, None]
            # shifted_coef[:ii+1] += coef[ii]*bkd.flip(temp)/scale**ii
            indices = self._bkd.arange(ii + 1)
            shifted_coef = self._bkd.up(
                shifted_coef,
                indices,
                shifted_coef[: ii + 1]
                + coef[ii] * self._bkd.flip(temp) / scale**ii,
                axis=0,
            )
        return shifted_coef

    def to_monomial_expansion(self) -> MonomialExpansion:
        if self.nvars() > 1:
            raise NotImplementedError("Only supported for nvars==1")

        basis = MultiIndexBasis([Monomial1D(backend=self._bkd)])
        basis.set_indices(self._bkd.arange(self.nterms())[None, :])
        basis_mono_coefs = self._bkd.array(
            self.basis()
            ._bases_1d[0]
            ._convert_orthonormal_polynomials_to_monomials_1d(
                self._bkd.to_numpy(self._basis._bases_1d[0]._rcoefs),
                self.nterms() - 1,
            )
        )
        mono_coefs = basis_mono_coefs.T @ self.get_coefficients()
        mono_coefs = self._shift_momomial_expansion(
            mono_coefs,
            self._basis._bases_1d[0]._trans._loc,
            self._basis._bases_1d[0]._trans._scale,
        )
        mono = MonomialExpansion(basis, nqoi=mono_coefs.shape[1])
        mono.set_coefficients(mono_coefs)
        return mono

    def _compute_multivariate_orthonormal_basis_product(
        self,
        product_coefs_1d: Array,
        poly_index_ii: Array,
        poly_index_jj: Array,
        max_degrees1: Array,
        max_degrees2: Array,
        tol: float = 2 * np.finfo(float).eps,
    ):
        """
        Compute the product of two multivariate orthonormal bases and re-express
        as an expansion using the orthonormal basis.
        """
        nvars = poly_index_ii.shape[0]
        poly_index = poly_index_ii + poly_index_jj
        active_vars = self._bkd.where(poly_index > 0)[0]
        if active_vars.shape[0] > 0:
            coefs_1d = []
            for dd in active_vars:
                pii, pjj = poly_index_ii[dd], poly_index_jj[dd]
                if pii < pjj:
                    tmp = pjj
                    pjj = pii
                    pii = tmp
                kk = flattened_rectangular_lower_triangular_matrix_index(
                    pii, pjj, max_degrees1[dd] + 1, max_degrees2[dd] + 1
                )
                coefs_1d.append(product_coefs_1d[dd][kk][:, 0])
            indices_1d = [
                self._bkd.arange(poly_index[dd] + 1, dtype=int)
                for dd in active_vars
            ]
            product_coefs = self._bkd.outer_product(coefs_1d)[:, None]
            active_product_indices = self._bkd.cartesian_product(indices_1d)
            II = self._bkd.where(self._bkd.abs(product_coefs) > tol)[0]
            active_product_indices = active_product_indices[:, II]
            product_coefs = product_coefs[II]
            product_indices = self._bkd.full(
                (nvars, active_product_indices.shape[1]),
                0.0,
                dtype=int,
            )
            # product_indices[active_vars] = active_product_indices
            product_indices = self._bkd.up(
                product_indices, active_vars, active_product_indices
            )
        else:
            product_coefs = self._bkd.full((1, 1), 1.0)
            product_indices = self._bkd.full([nvars, 1], 0.0, dtype=int)

        return product_indices, product_coefs

    def _multiply_multivariate_orthonormal_polynomial_expansions(
        self,
        product_coefs_1d: Array,
        poly_indices1: Array,
        poly_coefficients1: Array,
        poly_indices2: Array,
        poly_coefficients2: Array,
    ) -> Tuple[Array, Array]:
        num_indices1 = poly_indices1.shape[1]
        num_indices2 = poly_indices2.shape[1]
        assert num_indices2 <= num_indices1
        assert poly_coefficients1.shape[0] == num_indices1
        assert poly_coefficients2.shape[0] == num_indices2

        # following assumes the max degrees were used to create product_coefs_1d
        max_degrees1 = self._bkd.max(poly_indices1, axis=1)
        max_degrees2 = self._bkd.max(poly_indices2, axis=1)
        basis_coefs, basis_indices = [], []
        for ii in range(num_indices1):
            poly_index_ii = poly_indices1[:, ii]
            for jj in range(num_indices2):
                poly_index_jj = poly_indices2[:, jj]
                product_indices, product_coefs = (
                    self._compute_multivariate_orthonormal_basis_product(
                        product_coefs_1d,
                        poly_index_ii,
                        poly_index_jj,
                        max_degrees1,
                        max_degrees2,
                    )
                )
                # print(ii,jj,product_coefs,poly_index_ii,poly_index_jj)
                # TODO for unique polynomials the product_coefs and indices
                # of [0,1,2] is the same as [2,1,0] so perhaps store
                # sorted active indices and look up to reuse computations
                product_coefs_iijj = (
                    product_coefs
                    * poly_coefficients1[ii, :]
                    * poly_coefficients2[jj, :]
                )
                basis_coefs.append(product_coefs_iijj)
                basis_indices.append(product_indices)

                assert basis_coefs[-1].shape[0] == basis_indices[-1].shape[1]

        indices, coefs = self._add_polynomials(basis_indices, basis_coefs)
        return indices, coefs

    def coefficients_from_projection(
        self, quad_samples: Array, values: Array, quad_weights: Array
    ) -> Array:
        return self._bkd.einsum(
            "nm,nq->mq", self._basis(quad_samples), quad_weights * values
        )


def setup_polynomial_chaos_expansion_from_variable(
    variable: IndependentMarginalsVariable,
    nqoi: int,
    solver: LinearSystemSolver = None,
) -> PolynomialChaosExpansion:
    bases_1d = [
        setup_univariate_orthogonal_polynomial_from_marginal(
            marginal, backend=variable._bkd
        )
        for marginal in variable.marginals()
    ]
    basis = OrthonormalPolynomialBasis(bases_1d)
    if solver is None:
        solver = LstSqSolver(backend=variable._bkd)
    return PolynomialChaosExpansion(basis, solver=solver, nqoi=nqoi)


class TensorProductInterpolant(Surrogate):
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
        super().__init__(self._basis._bkd)

    def jacobian_implemented(self) -> bool:
        return self._basis.jacobian_implemented()

    def hessian_implemented(self) -> bool:
        return self._basis.hessian_implemented()

    def _many_jacobian(self, samples: Array) -> Array:
        # jacobian shape (nsamples, nqoi, nvars)
        return self._bkd.einsum(
            "ijk, jl->ilk",
            self._basis.jacobian(samples),
            self._train_values,
        )

    def _jacobian(self, sample: Array) -> Array:
        return self._many_jacobian(sample)[0]

    def _many_hessian(self, samples):
        hess = self._basis.hessian(samples)
        # hess shape is (nsamples, nterms, nvars, nvars)
        # coef shape is (nterms, nqoi)
        return self._bkd.einsum("ijkl, jm->imkl", hess, self._train_values)

    def _hessian(self, sample: Array) -> Array:
        return self._many_hessian(sample)[0]

    def fit(self, train_values: Array):
        # fit does not have samples like most surrogates because the samples
        # are predetermined by self._basis
        if train_values.shape[0] != self._basis.nterms():
            raise ValueError("nodes_1d and train_values are inconsistent")
        if train_values.ndim != 2:
            raise ValueError("train_values must be a 2d array")
        self._train_values = train_values
        self._nqoi = train_values.shape[1]

    def nqoi(self) -> int:
        return self._nqoi

    def nvars(self) -> int:
        return self._basis.nvars()

    def _values(self, samples: Array) -> Array:
        basis_mat = self._basis(samples)
        return basis_mat @ self._train_values

    def __repr__(self):
        return "{0}(basis={1})".format(self.__class__.__name__, self._basis)

    def mean(self) -> Array:
        quad_weights = self._basis.quadrature_rule()[1]
        return (self._train_values.T @ quad_weights)[:, 0]

    def variance(self) -> Array:
        quad_weights = self._basis.quadrature_rule()[1]
        return ((self._train_values**2).T @ quad_weights)[
            :, 0
        ] - self.mean() ** 2

    def get_train_samples(self) -> Array:
        return self._basis.tensor_product_grid()

    def get_train_values(self) -> Array:
        return self._train_values

    def basis(self) -> TensorProductInterpolatingBasis:
        return self._basis


class TrigonometricExpansion(BasisExpansion):
    def set_basis(self, basis, coef_bounds=None, fixed: bool = False):
        if not isinstance(basis, TrigonometricBasis):
            raise ValueError("basis must be an instance of TrigonometricBasis")
        super().set_basis(basis, coef_bounds, fixed)

    def trig_coefficients_from_fourier_coefficients(
        self, fourier_coefs: Array, real_function: bool = True
    ) -> Array:
        r"""
        Derive using eulers formula

        .. math::
             \cos(kx) = 1/2(e^{ikx}+e^{-ikx}), \sin(kx) = 1/(2i)(e^{ikx}-e^{-ikx}) = i/2(e^{-ikx}-e^{ikx})

        so
        .. math::

            p(x) &= a_0/2 + \sum_{k=1}^K a_k \cos(kx) + b_k \sin(kx)\\
                 &= a_0/2 + \sum_{k=1}^K a_k/2(e^{ikx}+e^{-ikx}) + ib_k/2(e^{-ikx}-e^{ikx})\\
                 &= a_0/2 + \sum_{k=1}^K 1/2(a_k-ib_k)e^{ikx} + (1/2)(a_k+ib_k)e^{-ikx}\\
                 &= \sum_{k=-K}^K c_k e^{ikx}

        where :math:`c_{-k} = 1/2(a_k+ib_k), c_k = 1/2(a_k-ib_k)`.
        Conversely :math:`a_k = c_k+c_{-k}, b_k = i(c_k-c_{-k})`
        """
        if fourier_coefs.ndim != 2:
            raise ValueError("fourier_coefs must be a 2d array")
        # a_k =  c_{-k} + c_k
        nterms = fourier_coefs.shape[0]
        Kmax = (nterms - 1) // 2
        cmk = self._bkd.flip(fourier_coefs[:Kmax])
        cos_coefs = cmk + fourier_coefs[Kmax + 1 :]
        # b_k =  i (c_k-c_{-k} )
        sin_coefs = 1j * (fourier_coefs[Kmax + 1 :] - cmk)
        coefs = self._bkd.vstack(
            (fourier_coefs[Kmax : Kmax + 1], cos_coefs, sin_coefs)
        )
        if real_function:
            return self._bkd.real(coefs)
        return coefs

    def quadrature_samples(self) -> Array:
        bounds = self._basis._bases_1d[0]._bounds
        return self._bkd.linspace(*bounds, self.nterms() + 1)[None, :-1]


class FourierExpansion(BasisExpansion):
    def set_basis(
        self,
        basis: FourierBasis,
        coef_bounds: bool = None,
        fixed: bool = False,
    ):
        if not isinstance(basis, FourierBasis):
            raise ValueError("basis must be an instance of FourierBasis")
        super().set_basis(basis, coef_bounds, fixed)

    def fourier_coefficients_from_trig_coefficients(
        self, trig_coefs: Array
    ) -> Array:
        if trig_coefs.ndim != 2:
            raise ValueError("trig_coefs must be a 2d array")
        const_coefs = trig_coefs[:1]
        Kmax = self._basis._bases_1d[0]._Kmax
        cos_coefs = trig_coefs[1 : Kmax + 1]
        sin_coefs = trig_coefs[Kmax + 1 :]
        left_coefs = self._bkd.flip(
            (cos_coefs + 1j * sin_coefs) / 2.0, axis=(0,)
        )
        right_coefs = (cos_coefs - 1j * sin_coefs) / 2.0
        return self._bkd.vstack((left_coefs, const_coefs, right_coefs))

    def quadrature_samples(self) -> Array:
        bounds = self._basis._bases_1d[0]._bounds
        return self._bkd.linspace(
            *bounds, self.nterms() + 1, dtype=self._bkd.complex_dtype()
        )[None, :-1]

    def compute_coefficients(self, values: Array) -> Array:
        quad_samples = self.quadrature_samples()
        return (self._basis(quad_samples).T) @ values / self.nterms()


class TensorProductLagrangeInterpolantToPolynomialChaosExpansionConverter:
    def __init__(self, quad_rule: TensorProductQuadratureRule):
        self._check_quadrature_rule(quad_rule)
        self._quad_rule = quad_rule
        self._bkd = self._quad_rule._bkd
        marginals = [
            quad_rule1d._marginal
            for quad_rule1d in self._quad_rule._univariate_quad_rules
        ]
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _check_interpolant(self, interp: TensorProductInterpolant):
        if not isinstance(interp, TensorProductInterpolant):
            raise ValueError(
                "interp must be an instance of TensorProductInterpolant"
            )

    def _check_quadrature_rule(self, quad_rule: TensorProductQuadratureRule):
        for quad_rule1d in quad_rule._univariate_quad_rules:
            if not isinstance(quad_rule1d, GaussQuadratureRule):
                raise ValueError(
                    "quad_rule must be an instance of GaussQuadratureRule"
                )

    def univariate_lagrange_basis_to_univariate_orthopoly_coefs(
        self, basis: UnivariateLagrangeBasis, quad_rule: GaussQuadratureRule
    ) -> Array:
        if not isinstance(basis, UnivariateLagrangeBasis):
            raise Exception(
                "Basis must be an instance of UnivariateLagrangeBasis"
            )
        # Use spectral projection to compute the coefficients of an orthonormal
        # polynomial that represent each lagrange basis function. The
        # orthonormal polynomial is defined implicitly by the
        # GaussQuadratureRule
        nterms = basis._quad_samples.shape[1]
        nsamples = nterms + 1
        quadx, quadw = quad_rule(nsamples)
        # sets quad_rule._poly internally to have nterms=nsamples
        ortho_basis_mat = quad_rule._poly(quadx)[:, :nterms]
        lagrange_basis_mat = basis(quadx)
        coefs = []
        for ii in range(lagrange_basis_mat.shape[1]):
            coefs_ii = quadw[:, 0] @ (
                lagrange_basis_mat[:, ii : ii + 1] * ortho_basis_mat
            )
            coefs.append(coefs_ii)
        return self._bkd.stack(coefs, axis=1)

    def multivariate_lagrange_basis_to_univariate_orthopoly_coefs(
        self,
        basis: TensorProductInterpolatingBasis,
    ) -> List[Array]:
        coefs_1d = []
        for basis1d, quad_rule1d in zip(
            basis._bases_1d, self._quad_rule._univariate_quad_rules
        ):
            coefs_1d.append(
                self.univariate_lagrange_basis_to_univariate_orthopoly_coefs(
                    basis1d, quad_rule1d
                )
            )
        return coefs_1d

    def polynomial_chaos_expansion_coefficients(
        self,
        basis: TensorProductInterpolatingBasis,
        values: Array,
    ):
        coefs_1d = (
            self.multivariate_lagrange_basis_to_univariate_orthopoly_coefs(
                basis
            )
        )
        basis_index = self._bkd.array(
            [basis1d.nterms() for basis1d in basis._bases_1d], dtype=int
        )
        active_idx = self._bkd.where(basis_index > 0)[0]
        basis_indices = basis.get_indices()
        nqoi = values.shape[1]
        ortho_poly_coefs = self._bkd.zeros((basis.nterms(), nqoi))
        for ii in range(basis.nterms()):
            active_coefs_1d = [
                coefs_1d[idx][:, basis_indices[idx, ii]] for idx in active_idx
            ]
            ortho_basis_coefs = self._bkd.outer_product(active_coefs_1d)
            ortho_poly_coefs += ortho_basis_coefs[:, None] * values[ii]
        return ortho_poly_coefs

    def convert(
        self,
        interp: TensorProductInterpolant,
    ) -> PolynomialChaosExpansion:
        self._check_interpolant(interp)
        poly = setup_polynomial_chaos_expansion_from_variable(
            self._variable, interp.nqoi()
        )
        poly.basis().set_indices(interp.basis().get_indices())
        pce_coef = self.polynomial_chaos_expansion_coefficients(
            interp.basis(), interp.get_train_values()
        )
        poly.set_coefficients(pce_coef)
        return poly

    def variable(self) -> IndependentMarginalsVariable:
        return self._variable


# convenience classes that simplify construction


class HyperbolicBasisExpansionMixin:
    def _setup_basis(
        self, univariate_bases: List[UnivariateBasis], level: int, pnorm: float
    ):
        basis = MultiIndexBasis(univariate_bases)
        basis.set_hyperbolic_indices(level, pnorm)
        return basis


class TensorProductBasisExpansionMixin:
    def _setup_basis(
        self, univariate_bases: List[UnivariateBasis], nterms: List[int]
    ):
        basis = MultiIndexBasis(univariate_bases)
        basis.set_tensor_product_indices(nterms)
        return basis


class HyperbolicMonomialExpansion(
    HyperbolicBasisExpansionMixin, MonomialExpansion
):
    def __init__(
        self,
        univariate_bases: List[Monomial1D],
        level: int,
        pnorm: float = 1.0,
        solver: LinearSystemSolver = None,
        nqoi=1,
        coef_bounds=None,
        fixed: bool = False,
    ):
        basis = self._setup_basis(univariate_bases, level, pnorm)
        super().__init__(basis, solver, nqoi, coef_bounds, fixed)


class HyperbolicPolynomialChaosExpansion(
    HyperbolicBasisExpansionMixin, PolynomialChaosExpansion
):
    def __init__(
        self,
        univariate_bases: List[OrthonormalPolynomial1D],
        level: int,
        pnorm: float = 1.0,
        solver: LinearSystemSolver = None,
        nqoi=1,
        coef_bounds=None,
        fixed: bool = False,
    ):
        basis = self._setup_basis(univariate_bases, level, pnorm)
        super().__init__(basis, solver, nqoi, coef_bounds, fixed)


class TensorProductMonomialExpansion(
    TensorProductBasisExpansionMixin, MonomialExpansion
):
    def __init__(
        self,
        univariate_bases: List[Monomial1D],
        nterms: List[int],
        solver: LinearSystemSolver = None,
        nqoi=1,
        coef_bounds=None,
        fixed: bool = False,
    ):
        basis = self._setup_basis(univariate_bases, nterms)
        super().__init__(basis, solver, nqoi, coef_bounds, fixed)


class TensorProductPolynomialChaosExpansion(
    TensorProductBasisExpansionMixin, PolynomialChaosExpansion
):
    def __init__(
        self,
        univariate_bases: List[OrthonormalPolynomial1D],
        nterms: List[int],
        solver: LinearSystemSolver = None,
        nqoi=1,
        coef_bounds=None,
        fixed: bool = False,
    ):
        basis = self._setup_basis(univariate_bases, nterms)
        super().__init__(basis, solver, nqoi, coef_bounds, fixed)
