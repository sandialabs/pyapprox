"""Convert orthonormal polynomial basis to monomial (power) basis.

Given recursion coefficients for an orthonormal polynomial family, compute
the monomial representation of each basis polynomial using the three-term
recurrence relation.
"""

from pyapprox.util.backends.protocols import Array, Backend


def convert_orthonormal_to_monomials_1d(
    rcoefs: Array, bkd: Backend[Array]
) -> Array:
    """Convert orthonormal polynomial basis to monomial coefficients.

    Uses the three-term recurrence to incrementally build the monomial
    representation of each orthonormal basis polynomial:

        psi_k(x) = sum_{j=0}^{k} M[k, j] * x^j

    The recurrence is:
        psi_0(x) = 1 / b_0
        psi_1(x) = (x - a_0) * psi_0(x) / b_1
        psi_n(x) = ((x - a_{n-1}) * psi_{n-1}(x) - b_{n-1} * psi_{n-2}(x)) / b_n

    Parameters
    ----------
    rcoefs : Array
        Recursion coefficients. Shape: (nterms, 2).
        Column 0: alpha (shift) coefficients.
        Column 1: beta (normalization) coefficients.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Monomial coefficient matrix. Shape: (nterms, nterms).
        Row k contains monomial coefficients of psi_k in ascending
        power order: M[k, j] is the coefficient of x^j in psi_k(x).
    """
    nterms = rcoefs.shape[0]
    nmax = nterms - 1

    monomial_coefs = bkd.zeros((nterms, nterms))
    monomial_coefs[0, 0] = 1.0 / rcoefs[0, 1]

    if nmax > 0:
        monomial_coefs[1, :2] = (
            bkd.array([-rcoefs[0, 0], 1.0])
            * monomial_coefs[0, 0]
            / rcoefs[1, 1]
        )

    for jj in range(2, nmax + 1):
        # Contribution from -a_{j-1} * psi_{j-1} - b_{j-1} * psi_{j-2}
        monomial_coefs[jj, :jj] += (
            -rcoefs[jj - 1, 0] * monomial_coefs[jj - 1, :jj]
            - rcoefs[jj - 1, 1] * monomial_coefs[jj - 2, :jj]
        ) / rcoefs[jj, 1]
        # Contribution from x * psi_{j-1} (shifts coefficients up by one power)
        monomial_coefs[jj, 1 : jj + 1] += (
            monomial_coefs[jj - 1, :jj] / rcoefs[jj, 1]
        )

    return monomial_coefs


__all__ = [
    "convert_orthonormal_to_monomials_1d",
]
