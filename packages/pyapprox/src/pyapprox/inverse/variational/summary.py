"""
Summary statistic protocols and implementations for amortized VI.

Provides composable pieces for mapping raw observations to fixed-size
labels used by amortized variational inference:

- **Transform** — per-observation feature extraction (identity, polynomial, etc.)
- **Aggregation** — reduce variable-length features to fixed size
  (mean, flatten, etc.)
- **SummaryStatistic** — composes transform + aggregation via
  ``TransformAggregateSummary``
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class Transform(Protocol[Array]):
    """Protocol for per-observation feature extraction.

    Applies a function ``h`` to each observation independently.

    Methods
    -------
    __call__(observations)
        Map observations to features (applied column-wise).
    nfeatures()
        Number of output feature dimensions.
    """

    def __call__(self, observations: Array) -> Array:
        """Map observations to features.

        Parameters
        ----------
        observations : Array
            Shape ``(nobs_dim, n)`` — n observations as columns.

        Returns
        -------
        Array
            Shape ``(nfeatures, n)``.
        """
        ...

    def nfeatures(self) -> int: ...


@runtime_checkable
class Aggregation(Protocol[Array]):
    """Protocol for reducing variable-length feature columns to fixed size.

    Methods
    -------
    __call__(features)
        Aggregate features across observations.
    nlabel_dims()
        Number of output label dimensions.
    """

    def __call__(self, features: Array) -> Array:
        """Aggregate features across observations.

        Parameters
        ----------
        features : Array
            Shape ``(nfeatures, n_obs)`` — variable n_obs.

        Returns
        -------
        Array
            Shape ``(nlabel_dims, 1)`` — fixed size.
        """
        ...

    def nlabel_dims(self) -> int: ...


@runtime_checkable
class SummaryStatistic(Protocol[Array]):
    """Protocol for mapping raw observations to a fixed-size label.

    Methods
    -------
    __call__(observations)
        Map raw observations to a fixed-size label.
    nlabel_dims()
        Number of output label dimensions.
    """

    def __call__(self, observations: Array) -> Array:
        """Map raw observations to a fixed-size label.

        Parameters
        ----------
        observations : Array
            Shape ``(nobs_dim, n_obs)``.

        Returns
        -------
        Array
            Shape ``(nlabel_dims, 1)``.
        """
        ...

    def nlabel_dims(self) -> int: ...


class IdentityTransform(Generic[Array]):
    """Identity transform: ``h(y) = y``.

    Parameters
    ----------
    nobs_dim : int
        Number of observation dimensions.
    """

    def __init__(self, nobs_dim: int) -> None:
        self._nobs_dim = nobs_dim

    def __call__(self, observations: Array) -> Array:
        return observations

    def nfeatures(self) -> int:
        return self._nobs_dim


class MeanAggregation(Generic[Array]):
    """Compute the column-wise mean of features.

    ``nlabel_dims = nfeatures``. Handles variable ``n_obs``.

    Parameters
    ----------
    nfeatures : int
        Number of feature dimensions (rows of input).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, nfeatures: int, bkd: Backend[Array]) -> None:
        self._nfeatures = nfeatures
        self._bkd = bkd

    def __call__(self, features: Array) -> Array:
        return self._bkd.reshape(
            self._bkd.mean(features, axis=1), (self._nfeatures, 1)
        )

    def nlabel_dims(self) -> int:
        return self._nfeatures


class MeanAndVarianceAggregation(Generic[Array]):
    """Stack column-wise mean and variance of features.

    ``nlabel_dims = 2 * nfeatures``. Handles variable ``n_obs``.

    Parameters
    ----------
    nfeatures : int
        Number of feature dimensions (rows of input).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, nfeatures: int, bkd: Backend[Array]) -> None:
        self._nfeatures = nfeatures
        self._bkd = bkd

    def __call__(self, features: Array) -> Array:
        m = self._bkd.reshape(
            self._bkd.mean(features, axis=1), (self._nfeatures, 1)
        )
        v = self._bkd.reshape(
            self._bkd.var(features, axis=1), (self._nfeatures, 1)
        )
        return self._bkd.concatenate([m, v], axis=0)

    def nlabel_dims(self) -> int:
        return 2 * self._nfeatures


class FlattenAggregation(Generic[Array]):
    """Concatenate all columns into a single vector.

    ``nlabel_dims = nfeatures * n_obs``. Requires fixed ``n_obs``.
    NOT permutation-invariant.

    Parameters
    ----------
    nfeatures : int
        Number of feature dimensions (rows of input).
    n_obs : int
        Fixed number of observations (validated on each call).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, nfeatures: int, n_obs: int, bkd: Backend[Array]) -> None:
        self._nfeatures = nfeatures
        self._n_obs = n_obs
        self._bkd = bkd

    def __call__(self, features: Array) -> Array:
        if features.shape[1] != self._n_obs:
            raise ValueError(
                f"FlattenAggregation expects {self._n_obs} observations, "
                f"got {features.shape[1]}"
            )
        # Flatten column-major: stack columns top-to-bottom
        return self._bkd.reshape(features, (self._nfeatures * self._n_obs, 1))

    def nlabel_dims(self) -> int:
        return self._nfeatures * self._n_obs


class MaxAggregation(Generic[Array]):
    """Element-wise max across observations.

    ``nlabel_dims = nfeatures``. Handles variable ``n_obs``.

    Parameters
    ----------
    nfeatures : int
        Number of feature dimensions (rows of input).
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    Not differentiable everywhere; use with caution in gradient-based
    optimization of learned summaries.
    """

    def __init__(self, nfeatures: int, bkd: Backend[Array]) -> None:
        self._nfeatures = nfeatures
        self._bkd = bkd

    def __call__(self, features: Array) -> Array:
        return self._bkd.reshape(
            self._bkd.max(features, axis=1), (self._nfeatures, 1)
        )

    def nlabel_dims(self) -> int:
        return self._nfeatures


class TransformAggregateSummary(Generic[Array]):
    """Compose a transform and aggregation into a summary statistic.

    This is the standard way to build summary statistics for amortized VI.
    All summary approaches use the same composition pattern.

    Parameters
    ----------
    transform : Transform-like
        Per-observation feature extraction.
    aggregation : Aggregation-like
        Reduction of features to fixed-size vector.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    Mean summary (most common):

    >>> summary = TransformAggregateSummary(
    ...     IdentityTransform(nobs_dim=1),
    ...     MeanAggregation(nfeatures=1, bkd=bkd),
    ...     bkd,
    ... )

    Raw data (fixed observation count):

    >>> summary = TransformAggregateSummary(
    ...     IdentityTransform(nobs_dim=1),
    ...     FlattenAggregation(nfeatures=1, n_obs=5, bkd=bkd),
    ...     bkd,
    ... )
    """

    def __init__(
        self,
        transform: Transform[Array],
        aggregation: Aggregation[Array],
        bkd: Backend[Array],
    ) -> None:
        self._transform = transform
        self._aggregation = aggregation
        self._bkd = bkd

    def __call__(self, observations: Array) -> Array:
        """Map raw observations to a fixed-size label.

        Parameters
        ----------
        observations : Array
            Shape ``(nobs_dim, n_obs)``.

        Returns
        -------
        Array
            Shape ``(nlabel_dims, 1)``.
        """
        features = self._transform(observations)
        return self._aggregation(features)

    def nlabel_dims(self) -> int:
        return self._aggregation.nlabel_dims()
