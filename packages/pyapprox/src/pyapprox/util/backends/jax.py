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

import functools
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import ArrayLike, KernelResult
from pyapprox.util.backends.validation import (
    check_index_add_shapes,
    check_jit_kernel,
    check_to_bool_shape,
)
from pyapprox.util.optional_deps import missing_dependency_message

try:
    import jax
    import jax.numpy as jnp
except ImportError as err:
    raise ImportError(missing_dependency_message("jax", "JaxBkd", "jax")) from err

Axes = Optional[Union[int, Tuple[int, ...]]]
# JaxBkd.jit's cache key: kernel, static argument names, dynamic.
_JitKey = Tuple[Callable[..., Any], Tuple[str, ...], bool]


class JaxBkd:
    """JAX arrays behind the ``Backend`` method names (partial)."""

    def __init__(self) -> None:
        jax.enable_x64.set_global(True)
        self._jit_cache: Dict[_JitKey, Callable[..., Any]] = {}

    def __getstate__(self) -> Dict[str, Any]:
        # Compiled kernels do not pickle; the cache is rebuilt on use.
        state = dict(self.__dict__)
        state["_jit_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        jax.enable_x64.set_global(True)
        self.__dict__.update(state)

    def jit(
        self,
        kernel: Callable[..., KernelResult],
        static_argnames: Sequence[str] = (),
        dynamic: bool = False,
    ) -> Callable[..., KernelResult]:
        # jax.jit caches on the function object, so the bound kernel is
        # built once per key here. It compiles once per shape whatever
        # dynamic says, and inlines a jit called while tracing.
        check_jit_kernel(kernel)
        key = (kernel, tuple(static_argnames), dynamic)
        compiled = self._jit_cache.get(key)
        if compiled is None:
            bound = functools.partial(kernel, bkd=self)
            compiled = jax.jit(bound, static_argnames=tuple(static_argnames))
            self._jit_cache[key] = compiled
        return compiled

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

    # ---- linear algebra

    @staticmethod
    def qr(
        array: jax.Array, mode: Literal["reduced", "complete"] = "reduced"
    ) -> Tuple[jax.Array, jax.Array]:
        q, r = jnp.linalg.qr(array, mode=mode)
        return q, r

    @staticmethod
    def svd(
        array: jax.Array, full_matrices: bool = True
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        u, s, vh = jnp.linalg.svd(array, full_matrices=full_matrices)
        return u, s, vh

    @staticmethod
    def eigh(array: jax.Array) -> Tuple[jax.Array, jax.Array]:
        values, vectors = jnp.linalg.eigh(array)
        return values, vectors

    @staticmethod
    def eigvalsh(array: jax.Array) -> jax.Array:
        result: jax.Array = jnp.linalg.eigvalsh(array)
        return result

    @staticmethod
    def pinv(matrix: jax.Array, *, rtol: Optional[float] = None) -> jax.Array:
        if rtol is None:
            # JAX dtypes are NumPy dtypes; jnp.finfo is untyped.
            rtol = max(matrix.shape[-2:]) * float(np.finfo(matrix.dtype).eps)
        return jnp.linalg.pinv(matrix, rtol=rtol)

    @staticmethod
    def lstsq(a: jax.Array, b: jax.Array, rcond: Optional[float] = None) -> jax.Array:
        return jnp.linalg.lstsq(a, b, rcond=rcond)[0]

    @staticmethod
    def cond(
        array: jax.Array,
        p: Optional[Union[int, float, Literal["fro", "nuc"]]] = None,
    ) -> jax.Array:
        result: jax.Array = jnp.linalg.cond(array, p)
        return result

    @staticmethod
    def norm(
        array: jax.Array,
        ord: Optional[Union[int, float, Literal["fro", "nuc"]]] = None,
        axis: Optional[Union[int, Tuple[int, int]]] = None,
        keepdims: bool = False,
    ) -> jax.Array:
        result: jax.Array = jnp.linalg.norm(
            array, ord=ord, axis=axis, keepdims=keepdims
        )
        return result

    @staticmethod
    def trace(array: jax.Array) -> jax.Array:
        return jnp.linalg.trace(array)

    @staticmethod
    def multidot(arrays: List[jax.Array]) -> jax.Array:
        return jnp.linalg.multi_dot(arrays)

    @staticmethod
    def kron(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.kron(a, b)

    @staticmethod
    def outer(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.outer(a, b)

    @staticmethod
    def solve_sparse(Amat: Any, bvec: jax.Array) -> jax.Array:
        """Not supported: JAX's sparse direct solver
        (``jax.experimental.sparse.linalg.spsolve``) is experimental and
        GPU-only, and a SciPy fallback would not trace or differentiate.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "JaxBkd.solve_sparse() is not supported: JAX's sparse direct "
            "solver is experimental and GPU-only. Use NumpyBkd for sparse "
            "solves or convert to dense first."
        )
