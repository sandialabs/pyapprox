"""Input builder strategies for Deep GP layers.

Controls how each layer's input is assembled from the original input X
and parent layer outputs. Different builders encode different DGP
architectures (skip-connected, pure composition, etc.).
"""

from typing import Generic, List, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class InputBuilder(Protocol, Generic[Array]):
    """Builds a layer's input from the original X and parent outputs."""

    def build(
        self,
        X: Array,
        parent_samples: List[Array],
        bkd: Backend[Array],
    ) -> Array:
        """Assemble layer input.

        Parameters
        ----------
        X : Array
            Original input features, shape (d_x, N).
        parent_samples : List[Array]
            Parent layer outputs, each shape (d_parent, N).
        bkd : Backend[Array]
            Backend for array operations.

        Returns
        -------
        Array
            Layer input, shape (d_in, N).
        """
        ...

    def input_dim(self, d_x: int, parent_dims: List[int]) -> int:
        """Compute layer input dimensionality.

        Parameters
        ----------
        d_x : int
            Original input dimensionality.
        parent_dims : List[int]
            Output dimensionality of each parent.

        Returns
        -------
        int
            Layer input dimensionality d_in.
        """
        ...


class SkipConnectedBuilder(Generic[Array]):
    """Concatenates [X, *parents]. Default for modern DGPs (Salimbeni 2017)."""

    def build(
        self,
        X: Array,
        parent_samples: List[Array],
        bkd: Backend[Array],
    ) -> Array:
        if not parent_samples:
            return X
        return bkd.vstack([X] + parent_samples)

    def input_dim(self, d_x: int, parent_dims: List[int]) -> int:
        return d_x + sum(parent_dims)


class PureCompositionBuilder(Generic[Array]):
    """Concatenates parent outputs only. Damianou-style DGPs.

    For root layers (no parents), falls back to returning X.
    """

    def build(
        self,
        X: Array,
        parent_samples: List[Array],
        bkd: Backend[Array],
    ) -> Array:
        if not parent_samples:
            return X
        return bkd.vstack(parent_samples)

    def input_dim(self, d_x: int, parent_dims: List[int]) -> int:
        if not parent_dims:
            return d_x
        return sum(parent_dims)


class RootBuilder(Generic[Array]):
    """Returns X only. For root layers with no parents."""

    def build(
        self,
        X: Array,
        parent_samples: List[Array],
        bkd: Backend[Array],
    ) -> Array:
        if parent_samples:
            raise ValueError("RootBuilder used at non-root node")
        return X

    def input_dim(self, d_x: int, parent_dims: List[int]) -> int:
        if parent_dims:
            raise ValueError("RootBuilder cannot have parent dims")
        return d_x
