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


def _check_moment_shapes(**arrays: Array) -> None:
    """Raise unless every array is 1D of the same length.

    Moments are ``(nqoi,)``. A ``(nqoi, 1)`` slipping in broadcasts to
    ``(nqoi, nqoi)`` instead of failing, and that reaches an error
    metric as a plausible number rather than an exception, so the shape
    is checked where the arrays meet.
    """
    items = list(arrays.items())
    for name, array in items:
        if array.ndim != 1:
            raise ValueError(
                f"{name} must be 1D of shape (nqoi,), got shape "
                f"{tuple(array.shape)}"
            )
    first_name, first = items[0]
    for name, array in items[1:]:
        if array.shape[0] != first.shape[0]:
            raise ValueError(
                f"{name} has nqoi={array.shape[0]} but {first_name} has "
                f"nqoi={first.shape[0]}"
            )


def variance_from_raw_moments(mean: Array, second: Array) -> Array:
    """Return V = M2 - m^2 from the first two raw moments.

    Raw moments combine linearly through the Smolyak coefficients where
    a central moment does not, so a variance over a combination is built
    from them rather than from per-subspace variances.

    Parameters
    ----------
    mean : Array
        First raw moment, shape (nqoi,).
    second : Array
        Second raw moment about zero, shape (nqoi,).

    Returns
    -------
    Array
        Variance, shape (nqoi,). Not guaranteed nonnegative when the
        moments come from a signed rule.

    Raises
    ------
    ValueError
        If the arrays are not 1D of matching length.
    """
    _check_moment_shapes(mean=mean, second=second)
    return second - mean**2


def variance_delta(
    mean: Array, delta_mean: Array, delta_second: Array
) -> Array:
    """Return the change in M2 - m^2 given changes in the raw moments.

    Expanding (m + dm)^2 - m^2 gives

        Delta V = Delta M2 - Delta m (2 m + Delta m)

    which is algebraically V_new - V_old but does not form either. Once
    the grid resolves the target those two are nearly equal and large
    next to their difference, so subtracting them loses most of the
    significant digits of the answer.

    Parameters
    ----------
    mean : Array
        Current first raw moment, shape (nqoi,).
    delta_mean : Array
        Change in the first raw moment, shape (nqoi,).
    delta_second : Array
        Change in the second raw moment, shape (nqoi,).

    Returns
    -------
    Array
        Change in the variance, shape (nqoi,).

    Raises
    ------
    ValueError
        If the arrays are not 1D of matching length.
    """
    _check_moment_shapes(
        mean=mean, delta_mean=delta_mean, delta_second=delta_second
    )
    return delta_second - delta_mean * (2.0 * mean + delta_mean)


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
