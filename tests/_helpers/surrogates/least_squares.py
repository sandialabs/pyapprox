"""A least-squares fit that keeps the expansion's own type.

``LeastSquaresFitter`` types its result's surrogate as
``BasisExpansionProtocol``, which declares no ``hyp_list``, so a fitted
expansion cannot be passed on as a learned function without losing its
type. Tests that fit an expansion and then use it as one (the
dynamical-systems tests) solve the same least-squares problem here and
build the fitted expansion with ``with_params``, which returns the
expansion's own type.
"""

from pyapprox.optimization.linear import LeastSquaresSolver
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.util.backends.protocols import Array


def least_squares_fit(
    expansion: BasisExpansion[Array], samples: Array, values: Array
) -> BasisExpansion[Array]:
    """``expansion`` with the coefficients minimizing
    :math:`\\|\\Phi c - y\\|_2`.

    Parameters
    ----------
    expansion : BasisExpansion
        The expansion to fit; not modified.
    samples : Array
        Inputs. Shape: (nvars, nsamples)
    values : Array
        Targets. Shape: (nqoi, nsamples)

    Returns
    -------
    BasisExpansion
        The fitted copy.
    """
    params = LeastSquaresSolver(expansion.bkd()).solve(
        expansion.basis_matrix(samples), values.T
    )
    return expansion.with_params(params)
