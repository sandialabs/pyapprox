"""Arithmetic operations for Polynomial Chaos Expansions.

Provides addition, subtraction, multiplication, and exponentiation of PCE
objects. All operations return new PCE instances without modifying the inputs.

All operations preserve PyTorch autograd computation graphs. The only int()
calls are on integer index arrays (multi-indices used for hashing), never
on autograd-tracked coefficient values.
"""

import copy
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.cartesian import cartesian_product, outer_product

if TYPE_CHECKING:
    from pyapprox.surrogates.affine.expansions.pce import (
        PolynomialChaosExpansion,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_compatible_bases(
    pce1: "PolynomialChaosExpansion[Array]",
    pce2: "PolynomialChaosExpansion[Array]",
) -> None:
    """Validate two PCEs have compatible univariate basis types.

    PCEs can only be combined if they share the same polynomial type
    in each dimension. The index sets may differ.

    Raises
    ------
    ValueError
        If number of variables differs.
    TypeError
        If univariate polynomial types differ in any dimension.
    """
    if pce1.nvars() != pce2.nvars():
        raise ValueError(
            f"Cannot combine PCEs with different nvars: "
            f"{pce1.nvars()} vs {pce2.nvars()}"
        )
    basis1 = pce1.get_basis()
    basis2 = pce2.get_basis()
    for dd in range(pce1.nvars()):
        b1 = basis1.get_univariate_basis(dd)
        b2 = basis2.get_univariate_basis(dd)
        poly_type1 = type(b1.polynomial())
        poly_type2 = type(b2.polynomial())
        if poly_type1 != poly_type2:
            raise TypeError(
                f"Incompatible bases in dimension {dd}: "
                f"{poly_type1.__name__} vs {poly_type2.__name__}"
            )


# ---------------------------------------------------------------------------
# Index grouping (replaces legacy _group_like_terms which used
# bkd.unique(axis=1, return_inverse=True) - not available in typing backend)
# ---------------------------------------------------------------------------


def _group_like_terms(
    bkd: Backend[Array], indices: Array, coeffs: Array
) -> Tuple[Array, Array]:
    """Group and sum coefficients for duplicate multi-indices.

    Uses a polynomial hash over integer index columns to identify duplicates.
    int() is only called on integer index values (not autograd-tracked).

    Parameters
    ----------
    bkd : Backend[Array]
    indices : Array, shape (nvars, nterms)
    coeffs : Array, shape (nterms, nqoi)

    Returns
    -------
    unique_indices : Array, shape (nvars, nunique)
    unique_coeffs : Array, shape (nunique, nqoi)
    """
    if coeffs.ndim == 1:
        coeffs = bkd.reshape(coeffs, (-1, 1))

    index_dict: dict[int, int] = {}
    out_indices: List[Array] = []
    out_coeffs: List[Array] = []
    kk = 0
    for ii in range(indices.shape[1]):
        col = indices[:, ii]
        # Polynomial hash for the multi-index column (integer values only)
        key = 0
        for dd in range(col.shape[0]):
            key = 31 * key + int(col[dd])
        if key in index_dict:
            pos = index_dict[key]
            out_coeffs[pos] = out_coeffs[pos] + coeffs[ii]
        else:
            index_dict[key] = kk
            out_indices.append(col)
            out_coeffs.append(coeffs[ii])
            kk += 1

    unique_indices = bkd.stack(out_indices, axis=1)
    unique_coeffs = bkd.stack(out_coeffs, axis=0)
    return unique_indices, unique_coeffs


def _add_polynomials(
    bkd: Backend[Array],
    indices_list: List[Array],
    coeffs_list: List[Array],
) -> Tuple[Array, Array]:
    """Add multiple polynomials by concatenating and grouping like terms.

    Parameters
    ----------
    bkd : Backend[Array]
    indices_list : list of Array, each shape (nvars, nterms_i)
    coeffs_list : list of Array, each shape (nterms_i, nqoi)

    Returns
    -------
    indices : Array, shape (nvars, nunique)
    coeffs : Array, shape (nunique, nqoi)
    """
    all_indices = bkd.hstack(indices_list)
    all_coeffs = bkd.vstack(coeffs_list)
    return _group_like_terms(bkd, all_indices, all_coeffs)


# ---------------------------------------------------------------------------
# Result PCE construction
# ---------------------------------------------------------------------------


def _create_result_pce(
    source_pce: "PolynomialChaosExpansion[Array]",
    new_indices: Array,
    new_coeffs: Array,
) -> "PolynomialChaosExpansion[Array]":
    """Create a new PCE with updated indices and coefficients.

    Deep-copies the source, updates basis indices and coefficients,
    and resets the cached hyperparameter list.
    """
    result = copy.deepcopy(source_pce)
    result._basis.set_indices(new_indices)
    result._initialize_coefficients()
    result.set_coefficients(new_coeffs)
    result._hyp_list = None
    return result


# ---------------------------------------------------------------------------
# Addition / subtraction
# ---------------------------------------------------------------------------


def add_pce(
    pce1: "PolynomialChaosExpansion[Array]",
    pce2: "PolynomialChaosExpansion[Array]",
    sign: float = 1.0,
) -> "PolynomialChaosExpansion[Array]":
    """Add (or subtract) two PCEs.

    Parameters
    ----------
    pce1, pce2 : PolynomialChaosExpansion
    sign : float
        Scale factor for pce2 coefficients. Use 1.0 for add, -1.0 for sub.

    Returns
    -------
    PolynomialChaosExpansion
        New PCE whose evaluation equals pce1(x) + sign * pce2(x).
    """
    _validate_compatible_bases(pce1, pce2)
    bkd = pce1.bkd()
    indices_list = [pce1.get_indices(), pce2.get_indices()]
    coeffs_list = [
        pce1.get_coefficients(),
        sign * pce2.get_coefficients(),
    ]
    new_indices, new_coeffs = _add_polynomials(bkd, indices_list, coeffs_list)
    return _create_result_pce(pce1, new_indices, new_coeffs)


def add_constant_to_pce(
    pce: "PolynomialChaosExpansion[Array]",
    constant: float,
) -> "PolynomialChaosExpansion[Array]":
    """Add a scalar constant to a PCE.

    The constant is added to the coefficient of the constant basis term
    (the term with all-zero multi-index).

    Returns
    -------
    PolynomialChaosExpansion
        New PCE whose evaluation equals pce(x) + constant.
    """
    result = copy.deepcopy(pce)
    bkd = result.bkd()
    const_idx = result._get_constant_index()
    coef = result.get_coefficients()
    # Build new coefficient array preserving autograd graph
    new_row = coef[const_idx, :] + constant
    new_coef = bkd.concatenate(
        [coef[:const_idx, :], bkd.reshape(new_row, (1, -1)), coef[const_idx + 1 :, :]],
        axis=0,
    )
    result.set_coefficients(new_coef)
    result._hyp_list = None
    return result


def multiply_pce_by_constant(
    pce: "PolynomialChaosExpansion[Array]",
    constant: float,
) -> "PolynomialChaosExpansion[Array]":
    """Multiply a PCE by a scalar constant.

    Returns
    -------
    PolynomialChaosExpansion
        New PCE whose evaluation equals constant * pce(x).
    """
    result = copy.deepcopy(pce)
    result.set_coefficients(pce.get_coefficients() * constant)
    result._hyp_list = None
    return result


# ---------------------------------------------------------------------------
# Triangular indexing helpers (pure Python, ports from legacy util/linalg.py)
# ---------------------------------------------------------------------------


def _nentries_square_triangular(N: int) -> int:
    """Number of entries in lower triangular part of N x N matrix."""
    return N * (N + 1) // 2


def _nentries_rectangular_triangular(M: int, N: int) -> int:
    """Number of entries in lower triangular part of M x N matrix (M >= N)."""
    return _nentries_square_triangular(M) - _nentries_square_triangular(M - N)


def _flattened_lower_tri_index(ii: int, jj: int, M: int, N: int) -> int:
    """Flattened index from (ii, jj) in lower triangular part of M x N matrix.

    The iteration order is:
        for d1 in range(M):
            for d2 in range(min(d1+1, N)):
    """
    if ii == 0:
        return 0
    T = _nentries_rectangular_triangular(ii, min(ii, N))
    return T + jj


# ---------------------------------------------------------------------------
# PCE multiplication via spectral projection
# ---------------------------------------------------------------------------


def _compute_product_coeffs_1d(
    basis: "PolynomialChaosExpansion[Array]",
    max_degrees1: Array,
    max_degrees2: Array,
) -> List[List[Array]]:
    """Precompute 1D orthonormal basis product coefficients via quadrature.

    For each dimension, uses Gaussian quadrature to express the product
    of two 1D basis functions as a sum of basis functions via pseudo-spectral
    projection.

    Parameters
    ----------
    basis : PolynomialChaosExpansion
        Source PCE (used to access 1D bases and backend).
    max_degrees1, max_degrees2 : Array
        Maximum degree per dimension for each operand.
        max_degrees1[dd] >= max_degrees2[dd] for all dd.

    Returns
    -------
    product_coefs_1d : list of list of Array
        product_coefs_1d[dd][kk] gives the projection coefficients for
        the kk-th (d1, d2) pair in dimension dd.
    """
    bkd = basis.bkd()
    orthonormal_basis = basis.get_basis()
    product_coefs_1d: List[List[Array]] = []

    for dd in range(basis.nvars()):
        max_degree1 = int(max_degrees1[dd])
        max_degree2 = int(max_degrees2[dd])
        max_degree = max_degree1 + max_degree2
        nquad_points = max_degree + 1

        basis_1d = orthonormal_basis.get_univariate_basis(dd)
        # Ensure enough terms for the product degree
        if basis_1d.nterms() < nquad_points:
            basis_1d.set_nterms(nquad_points)

        x_quad, w_quad = basis_1d.gauss_quadrature_rule(nquad_points)
        # Flatten weights to 1D
        w_quad = bkd.ravel(w_quad)

        # Evaluate orthonormal basis at quadrature points
        ortho_basis_matrix = basis_1d(x_quad)  # (nquad_points, nterms_1d)

        product_coefs_1d.append([])
        for d1 in range(max_degree1 + 1):
            for d2 in range(min(d1 + 1, max_degree2 + 1)):
                product_vals = ortho_basis_matrix[:, d1] * ortho_basis_matrix[:, d2]
                # Spectral projection: c_k = Σ_j w_j * product(x_j) * ψ_k(x_j)
                coefs = (
                    (w_quad * product_vals)[:, None].T
                    @ ortho_basis_matrix[:, : d1 + d2 + 1]
                ).T  # shape (d1+d2+1, 1)
                product_coefs_1d[-1].append(coefs)

    return product_coefs_1d


def _compute_multivariate_orthonormal_basis_product(
    bkd: Backend[Array],
    product_coefs_1d: List[List[Array]],
    poly_index_ii: Array,
    poly_index_jj: Array,
    max_degrees1: Array,
    max_degrees2: Array,
    tol: float = 2 * np.finfo(float).eps,
) -> Tuple[Array, Array]:
    """Compute product of two multivariate basis functions in orthonormal basis.

    Re-expresses ψ_i(x) * ψ_j(x) as Σ_k c_k ψ_k(x).

    Parameters
    ----------
    bkd : Backend[Array]
    product_coefs_1d : precomputed 1D product coefficients
    poly_index_ii, poly_index_jj : Array, shape (nvars,)
    max_degrees1, max_degrees2 : Array, shape (nvars,)
    tol : float
        Coefficients below this threshold are dropped.

    Returns
    -------
    product_indices : Array, shape (nvars, nproduct_terms)
    product_coefs : Array, shape (nproduct_terms, 1)
    """
    nvars = poly_index_ii.shape[0]
    poly_index = poly_index_ii + poly_index_jj
    active_vars = bkd.where(poly_index > 0)[0]

    if active_vars.shape[0] > 0:
        coefs_1d = []
        for dd in active_vars:
            pii = int(poly_index_ii[dd])
            pjj = int(poly_index_jj[dd])
            if pii < pjj:
                pii, pjj = pjj, pii
            kk = _flattened_lower_tri_index(
                pii,
                pjj,
                int(max_degrees1[dd]) + 1,
                int(max_degrees2[dd]) + 1,
            )
            coefs_1d.append(product_coefs_1d[int(dd)][kk][:, 0])

        indices_1d = [
            bkd.arange(int(poly_index[dd]) + 1, dtype=bkd.int64_dtype())
            for dd in active_vars
        ]

        if len(coefs_1d) >= 2:
            product_coefs = bkd.ravel(outer_product(bkd, coefs_1d))[:, None]
            active_product_indices = cartesian_product(bkd, indices_1d)
        else:
            # Single active variable: no need for outer/cartesian product
            product_coefs = coefs_1d[0][:, None]
            active_product_indices = indices_1d[0][None, :]

        # Filter near-zero coefficients
        II = bkd.where(bkd.abs(bkd.ravel(product_coefs)) > tol)[0]
        active_product_indices = active_product_indices[:, II]
        product_coefs = product_coefs[II]

        # Build full-dimensional index array with zeros for inactive vars
        product_indices = bkd.zeros(
            (nvars, active_product_indices.shape[1]),
            dtype=bkd.int64_dtype(),
        )
        # Assign active variable rows
        rows = []
        for idx_in_active, dd in enumerate(active_vars):
            row = bkd.zeros(
                (1, active_product_indices.shape[1]),
                dtype=bkd.int64_dtype(),
            )
            row = row + active_product_indices[idx_in_active : idx_in_active + 1, :]
            rows.append((int(dd), row))

        for dd_int, row in rows:
            # Build product_indices by concatenating rows in correct order
            pass

        # Simpler approach: build row-by-row
        index_rows = []
        active_set = set(int(dd) for dd in active_vars)
        active_idx_map = {}
        for idx_in_active, dd in enumerate(active_vars):
            active_idx_map[int(dd)] = idx_in_active

        for dd in range(nvars):
            if dd in active_set:
                index_rows.append(
                    active_product_indices[
                        active_idx_map[dd] : active_idx_map[dd] + 1, :
                    ]
                )
            else:
                index_rows.append(
                    bkd.zeros(
                        (1, active_product_indices.shape[1]),
                        dtype=bkd.int64_dtype(),
                    )
                )
        product_indices = bkd.vstack(index_rows)
    else:
        product_coefs = bkd.ones((1, 1))
        product_indices = bkd.zeros((nvars, 1), dtype=bkd.int64_dtype())

    return product_indices, product_coefs


def _multiply_multivariate_orthonormal_polynomial_expansions(
    bkd: Backend[Array],
    product_coefs_1d: List[List[Array]],
    poly_indices1: Array,
    poly_coefficients1: Array,
    poly_indices2: Array,
    poly_coefficients2: Array,
) -> Tuple[Array, Array]:
    """Multiply two multivariate orthonormal polynomial expansions.

    Parameters
    ----------
    bkd : Backend[Array]
    product_coefs_1d : precomputed 1D product coefficients
    poly_indices1 : Array, shape (nvars, nterms1)
    poly_coefficients1 : Array, shape (nterms1, nqoi)
    poly_indices2 : Array, shape (nvars, nterms2)
    poly_coefficients2 : Array, shape (nterms2, nqoi)

    Returns
    -------
    indices : Array, shape (nvars, nresult_terms)
    coeffs : Array, shape (nresult_terms, nqoi)
    """
    num_indices1 = poly_indices1.shape[1]
    num_indices2 = poly_indices2.shape[1]

    max_degrees1 = bkd.max(poly_indices1, axis=1)
    max_degrees2 = bkd.max(poly_indices2, axis=1)

    basis_coefs: List[Array] = []
    basis_indices: List[Array] = []

    for ii in range(num_indices1):
        poly_index_ii = poly_indices1[:, ii]
        for jj in range(num_indices2):
            poly_index_jj = poly_indices2[:, jj]
            product_indices, product_coefs = (
                _compute_multivariate_orthonormal_basis_product(
                    bkd,
                    product_coefs_1d,
                    poly_index_ii,
                    poly_index_jj,
                    max_degrees1,
                    max_degrees2,
                )
            )
            # Scale by both input coefficients (preserves autograd)
            product_coefs_iijj = (
                product_coefs * poly_coefficients1[ii, :] * poly_coefficients2[jj, :]
            )
            basis_coefs.append(product_coefs_iijj)
            basis_indices.append(product_indices)

    return _add_polynomials(bkd, basis_indices, basis_coefs)


def multiply_pce(
    pce1: "PolynomialChaosExpansion[Array]",
    pce2: "PolynomialChaosExpansion[Array]",
) -> "PolynomialChaosExpansion[Array]":
    """Multiply two PCEs using spectral projection.

    Parameters
    ----------
    pce1, pce2 : PolynomialChaosExpansion

    Returns
    -------
    PolynomialChaosExpansion
        New PCE whose evaluation equals pce1(x) * pce2(x).
    """
    _validate_compatible_bases(pce1, pce2)
    bkd = pce1.bkd()

    # Order so poly1 has more terms (optimization for the double loop)
    if pce1.nterms() >= pce2.nterms():
        poly1, poly2 = pce1, pce2
    else:
        poly1, poly2 = pce2, pce1

    # Deep copy only for the quadrature computation (which calls set_nterms
    # and may mutate the 1D bases). The original coefficients are used in
    # the multiplication to preserve autograd graphs.
    poly1_copy = copy.deepcopy(poly1)

    max_degrees1 = bkd.max(poly1.get_indices(), axis=1)
    max_degrees2 = bkd.max(poly2.get_indices(), axis=1)

    product_coefs_1d = _compute_product_coeffs_1d(
        poly1_copy, max_degrees1, max_degrees2
    )

    # Use original (non-copied) coefficients to preserve autograd graph
    new_indices, new_coeffs = _multiply_multivariate_orthonormal_polynomial_expansions(
        bkd,
        product_coefs_1d,
        poly1.get_indices(),
        poly1.get_coefficients(),
        poly2.get_indices(),
        poly2.get_coefficients(),
    )

    return _create_result_pce(pce1, new_indices, new_coeffs)


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


def pce_power(
    pce: "PolynomialChaosExpansion[Array]",
    order: int,
) -> "PolynomialChaosExpansion[Array]":
    """Raise a PCE to a non-negative integer power.

    Parameters
    ----------
    pce : PolynomialChaosExpansion
    order : int
        Non-negative integer exponent.

    Returns
    -------
    PolynomialChaosExpansion
        New PCE whose evaluation equals pce(x) ** order.

    Raises
    ------
    TypeError
        If order is not an integer.
    ValueError
        If order is negative.
    """
    if not isinstance(order, int):
        raise TypeError(f"Power order must be an integer, got {type(order).__name__}")
    if order < 0:
        raise ValueError(f"Power order must be non-negative, got {order}")

    bkd = pce.bkd()

    if order == 0:
        # Return constant 1 polynomial
        result = copy.deepcopy(pce)
        const_indices = bkd.zeros((pce.nvars(), 1), dtype=bkd.int64_dtype())
        result._basis.set_indices(const_indices)
        result._initialize_coefficients()
        ones_coef = bkd.ones((1, pce.nqoi()))
        result.set_coefficients(ones_coef)
        result._hyp_list = None
        return result

    # order >= 1: start with a copy and multiply iteratively
    result = copy.deepcopy(pce)
    for _ in range(2, order + 1):
        result = multiply_pce(result, pce)
    return result
