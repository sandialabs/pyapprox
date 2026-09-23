"""Per-subspace moments under a tensor product quadrature rule.

Free functions rather than subspace methods: a subspace owns its rule,
its samples and its values, and these are things computed from that
data rather than part of what a subspace is.

Each takes a subspace whose values are set and returns an array of
shape (nqoi,). All use the public ``get_values`` and
``get_quadrature_weights``, so they work for any subspace regardless of
how its basis was built.
"""

from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array


def _values_and_weights(
    subspace: TensorProductSubspace[Array],
) -> tuple[Array, Array]:
    """Return the subspace's values and quadrature weights.

    Raises
    ------
    ValueError
        If the subspace has no values.
    """
    values = subspace.get_values()
    if values is None:
        raise ValueError(
            "subspace has no values; set them before computing a moment"
        )
    return values, subspace.get_quadrature_weights()


def subspace_mean(subspace: TensorProductSubspace[Array]) -> Array:
    """Return Q_k f, the quadrature estimate of the mean.

    The weights are probability weights for the input measure, so this
    is an expectation rather than a Lebesgue integral.

    Parameters
    ----------
    subspace : TensorProductSubspace[Array]
        Subspace with values set.

    Returns
    -------
    Array
        Mean per quantity of interest, shape (nqoi,).
    """
    values, weights = _values_and_weights(subspace)
    return values @ weights


def subspace_variance(subspace: TensorProductSubspace[Array]) -> Array:
    """Return Var_k f under this subspace's quadrature rule.

    Computed in two passes, as sum_i w_i (f_i - Q_k f)^2. The one-pass
    form E[f^2] - E[f]^2 loses precision by cancellation when the mean
    is large relative to the spread.

    Parameters
    ----------
    subspace : TensorProductSubspace[Array]
        Subspace with values set.

    Returns
    -------
    Array
        Variance per quantity of interest, shape (nqoi,).
    """
    values, weights = _values_and_weights(subspace)
    mean = values @ weights
    # mean is (nqoi,); broadcast it across the sample axis.
    centered = values - mean[:, None]
    return (centered**2) @ weights


def subspace_raw_moment(
    subspace: TensorProductSubspace[Array], order: int
) -> Array:
    """Return Q_k[f^order], the raw moment of the given order.

    Order 1 is ``subspace_mean``. Order 2 is the second moment about
    zero, which combines across subspaces by Smolyak coefficients where
    a central moment does not.

    Parameters
    ----------
    subspace : TensorProductSubspace[Array]
        Subspace with values set.
    order : int
        Moment order, at least 1.

    Returns
    -------
    Array
        Raw moment per quantity of interest, shape (nqoi,).

    Raises
    ------
    ValueError
        If order is less than 1.
    """
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    values, weights = _values_and_weights(subspace)
    return (values**order) @ weights
