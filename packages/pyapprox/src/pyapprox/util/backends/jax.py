"""Experimental JAX backend, grown one protocol method at a time.

``JaxBkd`` implements only the ``Backend`` methods whose conformance rows
pass on JAX; the rest are listed in the conformance suite's
``NOT_YET_ON_JAX``, which may only shrink. It does not yet satisfy
``Backend``, is not exported, and joins the general ``bkd`` test fixture
only when that list is empty.

Constructing a ``JaxBkd`` enables 64-bit types process-wide
(``jax_enable_x64``), as ``TorchBkd`` sets torch's default dtype: without
it JAX silently computes in 32 bits.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import ArrayLike
from pyapprox.util.backends.validation import (
    check_index_add_shapes,
    check_to_bool_shape,
)
from pyapprox.util.optional_deps import missing_dependency_message

try:
    import jax
    import jax.numpy as jnp
except ImportError as err:
    raise ImportError(missing_dependency_message("jax", "JaxBkd", "jax")) from err

Axes = Optional[Union[int, Tuple[int, ...]]]


class JaxBkd:
    """JAX arrays behind the ``Backend`` method names (partial)."""

    def __init__(self) -> None:
        jax.enable_x64.set_global(True)

    # ---- dtypes and conversion

    @staticmethod
    def default_dtype() -> Any:
        return jnp.float64

    @staticmethod
    def double_dtype() -> Any:
        return jnp.float64

    @staticmethod
    def complex_dtype() -> Any:
        return jnp.complex128

    @staticmethod
    def int64_dtype() -> Any:
        return jnp.int64

    @staticmethod
    def asarray(array: ArrayLike, dtype: Optional[Any] = None) -> jax.Array:
        return jnp.asarray(array, dtype=dtype)

    @staticmethod
    def to_numpy(array: jax.Array) -> NDArray[Any]:
        return np.asarray(array)

    @staticmethod
    def assert_allclose(
        actual: jax.Array,
        desired: jax.Array,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: Optional[str] = None,
    ) -> None:
        # Strict, as the other backends are: shape and dtype must match.
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(desired),
            rtol,
            atol,
            equal_nan,
            err_msg="" if err_msg is None else err_msg,
            verbose=True,
            strict=True,
        )

    @staticmethod
    def to_bool(array: jax.Array) -> bool:
        check_to_bool_shape(tuple(array.shape))
        return bool(array.reshape(()))

    # ---- creation

    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype: Optional[Any] = None) -> jax.Array:
        return jnp.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape: Tuple[int, ...], dtype: Optional[Any] = None) -> jax.Array:
        return jnp.ones(shape, dtype=dtype)

    @staticmethod
    def full(
        shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> jax.Array:
        return jnp.full(shape, fill_value, dtype=dtype)

    # ---- shape

    @staticmethod
    def reshape(array: jax.Array, newshape: Sequence[int]) -> jax.Array:
        return jnp.reshape(array, tuple(newshape))

    @staticmethod
    def stack(
        arrays: Union[List[jax.Array], Tuple[jax.Array, ...]], axis: int = 0
    ) -> jax.Array:
        return jnp.stack(arrays, axis=axis)

    @staticmethod
    def concatenate(
        arrays: Union[List[jax.Array], Tuple[jax.Array, ...]], axis: int = 0
    ) -> jax.Array:
        return jnp.concatenate(arrays, axis=axis)

    @staticmethod
    def moveaxis(
        array: jax.Array,
        source: Union[int, Tuple[int, ...]],
        destination: Union[int, Tuple[int, ...]],
    ) -> jax.Array:
        return jnp.moveaxis(array, source, destination)

    # ---- elementwise and contraction

    @staticmethod
    def sqrt(array: jax.Array) -> jax.Array:
        return jnp.sqrt(array)

    @staticmethod
    def exp(array: jax.Array) -> jax.Array:
        return jnp.exp(array)

    @staticmethod
    def dot(Amat: jax.Array, Bmat: jax.Array) -> jax.Array:
        return jnp.dot(Amat, Bmat)

    @staticmethod
    def einsum(subscripts: str, *operands: jax.Array) -> jax.Array:
        return jnp.einsum(subscripts, *operands)

    @staticmethod
    def where(
        condition: jax.Array,
        x: Optional[Union[jax.Array, float]] = None,
        y: Optional[Union[jax.Array, float]] = None,
    ) -> Union[jax.Array, Tuple[jax.Array, ...]]:
        if x is None:
            return tuple(jnp.nonzero(condition))
        if y is None:
            raise ValueError("Must provide both x and y or neither.")
        return jnp.where(condition, x, y)

    # ---- reductions

    @staticmethod
    def sum(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        return jnp.sum(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        return jnp.mean(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def prod(
        array: jax.Array, axis: Optional[int] = None, keepdims: bool = False
    ) -> jax.Array:
        return jnp.prod(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def any(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        return jnp.any(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def all(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        return jnp.all(array, axis=axis, keepdims=keepdims)

    # ---- functional updates

    @staticmethod
    def index_update(
        array: jax.Array,
        index: Union[int, slice, jax.Array, Tuple[Union[int, slice, jax.Array], ...]],
        value: Any,
    ) -> jax.Array:
        return array.at[index].set(value)

    @staticmethod
    def index_add(array: jax.Array, index: jax.Array, values: jax.Array) -> jax.Array:
        check_index_add_shapes(tuple(index.shape), tuple(values.shape))
        return array.at[index].add(values)
