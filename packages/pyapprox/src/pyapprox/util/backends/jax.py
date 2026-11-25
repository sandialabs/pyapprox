"""Experimental JAX backend.

``JaxBkd`` satisfies ``Backend`` and passes the whole conformance table
in both precisions. It is still experimental: not exported, and not in
the general ``bkd`` test fixture, because much library code writes into
arrays in place, which JAX arrays refuse.

``JaxBkd`` never changes JAX's process-wide configuration. Double
precision needs 64-bit types, which JAX enables only process-wide
(``jax.config.update("jax_enable_x64", True)``, before any array is
made): the caller's choice, so ``JaxBkd(precision=Precision.DOUBLE)``
refuses to start without it rather than turning it on.
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
    overload,
)

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.errors import LinAlgError
from pyapprox.util.backends.precision import Precision
from pyapprox.util.backends.protocols import (
    ArrayLike,
    ArrayProtocol,
    Backend,
    KernelResult,
)
from pyapprox.util.backends.validation import (
    check_dtype_kind,
    check_index_add_shapes,
    check_jit_kernel,
    check_orderable,
    check_ordered_reduction,
    check_scalar_conversion,
    check_to_bool_shape,
    reduction_axes,
)
from pyapprox.util.optional_deps import missing_dependency_message

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg
    import jax.scipy.special
except ImportError as err:
    raise ImportError(missing_dependency_message("jax", "JaxBkd", "jax")) from err

Axes = Optional[Union[int, Tuple[int, ...]]]


def _finite_or_raise(result: jax.Array, reason: str) -> jax.Array:
    """``result``, or ``LinAlgError`` if it is not finite.

    JAX returns NaN or inf where NumPy and torch raise. The check runs
    eagerly only: under ``jax.jit`` the values are unknown while tracing,
    so a compiled solve returns the non-finite result. Eagerly it costs a
    host sync, and a non-finite input raises too.
    """
    if isinstance(result, jax.core.Tracer):
        return result
    if not bool(jnp.all(jnp.isfinite(result))):
        raise LinAlgError(reason)
    return result


# Each floating or complex dtype's (real, complex) pair of its width.
_WIDTHS: Dict[np.dtype[Any], Tuple[Any, Any]] = {
    np.dtype(np.float32): (jnp.float32, jnp.complex64),
    np.dtype(np.complex64): (jnp.float32, jnp.complex64),
    np.dtype(np.float64): (jnp.float64, jnp.complex128),
    np.dtype(np.complex128): (jnp.float64, jnp.complex128),
}


def _width_pair(dtype: Any) -> Tuple[Any, Any]:
    pair = _WIDTHS.get(np.dtype(dtype))
    if pair is None:
        raise ValueError(f"needs a floating or complex dtype, got {dtype}")
    return pair


# JaxBkd.jit's cache key: kernel, static argument names, dynamic.
_JitKey = Tuple[Callable[..., Any], Tuple[str, ...], bool]


def _resolve_device(device: Optional[str]) -> "jax.Device":
    """The JAX device named ``device`` ("cpu", "gpu", "tpu", optionally
    with ":index"), or JAX's default device for ``None``."""
    if device is None:
        return jax.devices()[0]
    platform, _, index = device.partition(":")
    try:
        candidates = jax.devices(platform)
    except RuntimeError as err:
        raise ValueError(f"JAX has no {platform!r} device") from err
    position = int(index) if index else 0
    if position >= len(candidates):
        raise ValueError(
            f"JAX has {len(candidates)} {platform!r} device(s), got index {position}"
        )
    return candidates[position]


def _device_name(device: "jax.Device") -> str:
    # The platform alone names its first device, as torch's "cpu" does.
    return device.platform if device.id == 0 else f"{device.platform}:{device.id}"


class JaxBkd:
    """JAX arrays behind the ``Backend`` method names (partial)."""

    def __init__(
        self,
        *,
        precision: Precision = Precision.DOUBLE,
        device: Optional[str] = None,
    ) -> None:
        """``precision`` sets the real working dtype (float32 or float64)
        and with it the complex one; ``device`` ("cpu", "gpu", "tpu",
        optionally ":index") defaults to JAX's default device.

        Double precision needs ``jax_enable_x64``, which only the caller
        may set (it is process-wide). The index dtype is int64 with it
        and int32 without, read here."""
        if not isinstance(precision, Precision):
            raise TypeError(
                f"precision must be a Precision, got {type(precision).__name__}"
            )
        x64 = bool(jax.config.jax_enable_x64)
        if precision is Precision.DOUBLE and not x64:
            raise ValueError(
                "JAX double precision needs 64-bit types: call "
                'jax.config.update("jax_enable_x64", True) before creating '
                "any array, or use Precision.SINGLE"
            )
        self._precision = precision
        self._device_spec = device
        self._device = _resolve_device(device)
        single = precision is Precision.SINGLE
        self._dtype = jnp.float32 if single else jnp.float64
        self._complex_dtype = jnp.complex64 if single else jnp.complex128
        self._index_dtype = jnp.int64 if x64 else jnp.int32
        self._jit_cache: Dict[_JitKey, Callable[..., Any]] = {}

    def __getstate__(self) -> Dict[str, Any]:
        # Compiled kernels and JAX devices do not pickle; both are rebuilt.
        state = dict(self.__dict__)
        state["_jit_cache"] = {}
        del state["_device"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._device = _resolve_device(self._device_spec)

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

    def precision(self) -> Precision:
        return self._precision

    def device(self) -> str:
        return _device_name(self._device)

    def __repr__(self) -> str:
        return f"JaxBkd(precision={self._precision.value}, device={self.device()})"

    def _identity(self) -> Tuple[Precision, str]:
        return (self._precision, self.device())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, JaxBkd) and self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash((JaxBkd, self._identity()))

    def is_compatible(self, other: object) -> bool:
        # JAX has no compile switch, so compatibility is equality.
        return self == other

    def replace(
        self, *, precision: Optional[Precision] = None, device: Optional[str] = None
    ) -> "JaxBkd":
        return JaxBkd(
            precision=self._precision if precision is None else precision,
            device=self.device() if device is None else device,
        )

    def from_other(self, array: ArrayProtocol, src: Backend[Any]) -> jax.Array:
        if isinstance(array, jax.Array):
            if self.is_compatible(src):
                return array
            # device_put and astype keep the array traceable across
            # precision and device.
            if jnp.issubdtype(array.dtype, jnp.complexfloating):
                array = array.astype(self._complex_dtype)
            elif jnp.issubdtype(array.dtype, jnp.floating):
                array = array.astype(self._dtype)
            placed: jax.Array = jax.device_put(array, self._device)
            return placed
        # Another library: through NumPy, so not differentiable through.
        return self.array(src.to_numpy(array))

    def real_dtype(self) -> Any:
        return self._dtype

    def complex_dtype(self) -> Any:
        return self._complex_dtype

    def index_dtype(self) -> Any:
        return self._index_dtype

    def eps(self) -> float:
        return float(np.finfo(self._dtype).eps)

    @staticmethod
    def result_type(*arrays: jax.Array) -> Any:
        # NumPy's rules on the dtypes, not JAX's weak-type lattice.
        return np.result_type(*(np.dtype(a.dtype) for a in arrays))

    @staticmethod
    def isdtype(array: jax.Array, kind: str) -> bool:
        check_dtype_kind(kind)
        return bool(np.isdtype(np.dtype(array.dtype), kind))

    @staticmethod
    def real_dtype_of(dtype: Any) -> Any:
        return _width_pair(dtype)[0]

    @staticmethod
    def complex_dtype_of(dtype: Any) -> Any:
        return _width_pair(dtype)[1]

    @staticmethod
    def real(array: jax.Array) -> jax.Array:
        return jnp.real(array)

    @staticmethod
    def imag(array: jax.Array) -> jax.Array:
        return jnp.imag(array)

    def default_dtype(self) -> Any:
        return self._dtype

    def double_dtype(self) -> Any:
        return self._dtype

    def int64_dtype(self) -> Any:
        return self._index_dtype

    def _kind_dtype(self, dtype: Any) -> Any:
        """The configured dtype of ``dtype``'s kind: real floating to the
        real dtype, complex to the complex one, integer to the index
        dtype; bool kept."""
        if jnp.issubdtype(dtype, jnp.complexfloating):
            return self._complex_dtype
        if jnp.issubdtype(dtype, jnp.floating):
            return self._dtype
        if jnp.issubdtype(dtype, jnp.integer):
            return self._index_dtype
        return dtype

    def asarray(self, array: ArrayLike, dtype: Optional[Any] = None) -> jax.Array:
        if dtype is None and not hasattr(array, "dtype"):
            # Python data is coerced by kind, read by NumPy at full width
            # so a float is rounded once, to the configured precision.
            host = np.asarray(array)
            return jnp.asarray(
                host, dtype=self._kind_dtype(host.dtype), device=self._device
            )
        return jnp.asarray(array, dtype=dtype, device=self._device)

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

    def array(self, array: ArrayLike, dtype: Optional[Any] = None) -> jax.Array:
        source = array if isinstance(array, jax.Array) else np.asarray(array)
        if dtype is None:
            # Precision is normalized and kind kept: real floating becomes
            # the real working dtype, complex the complex one.
            kind = source.dtype
            if jnp.issubdtype(kind, jnp.complexfloating):
                dtype = self._complex_dtype
            elif jnp.issubdtype(kind, jnp.floating):
                dtype = self._dtype
            elif (
                jnp.issubdtype(kind, jnp.integer)
                and np.dtype(kind).itemsize > np.dtype(self._index_dtype).itemsize
            ):
                # Without 64-bit types JAX has no int64 to keep.
                dtype = self._index_dtype
        return jnp.array(source, dtype=dtype, copy=True, device=self._device)

    @staticmethod
    def array_type() -> type:
        # Annotated: where jax is not installed (CI's lint job) it is Any.
        array_class: type = jax.Array
        return array_class

    @staticmethod
    def is_integer_dtype(array: jax.Array) -> bool:
        return bool(jnp.issubdtype(array.dtype, jnp.integer))

    @staticmethod
    def is_floating_dtype(array: jax.Array) -> bool:
        return bool(jnp.issubdtype(array.dtype, jnp.floating))

    @staticmethod
    def tracks_gradient(array: jax.Array) -> bool:
        """Whether ``array`` is being traced (by ``grad``, ``jit`` or any
        other transformation): a traced value must not be memoized by
        value, as the protocol warns."""
        return isinstance(array, jax.core.Tracer)

    @staticmethod
    def to_float(array: "float | int | jax.Array") -> float:
        if not isinstance(array, jax.Array):
            return float(array)
        check_scalar_conversion("to_float", tuple(array.shape), jnp.iscomplexobj(array))
        return float(array.reshape(()))

    @staticmethod
    def to_int(array: "float | int | jax.Array") -> int:
        if not isinstance(array, jax.Array):
            return int(array)
        check_scalar_conversion("to_int", tuple(array.shape), jnp.iscomplexobj(array))
        return int(array.reshape(()))

    @staticmethod
    def to_bool(array: jax.Array) -> bool:
        check_to_bool_shape(tuple(array.shape))
        return bool(array.reshape(()))

    # ---- creation

    def _or_real(self, dtype: Optional[Any]) -> Any:
        return self._dtype if dtype is None else dtype

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> jax.Array:
        return jnp.zeros(shape, dtype=self._or_real(dtype), device=self._device)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> jax.Array:
        return jnp.ones(shape, dtype=self._or_real(dtype), device=self._device)

    def full(
        self, shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> jax.Array:
        return jnp.full(
            shape, fill_value, dtype=self._or_real(dtype), device=self._device
        )

    def empty(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> jax.Array:
        return jnp.empty(shape, dtype=self._or_real(dtype), device=self._device)

    def eye(
        self, nrows: int, ncols: Optional[int] = None, dtype: Optional[Any] = None
    ) -> jax.Array:
        return jnp.eye(nrows, ncols, dtype=self._or_real(dtype), device=self._device)

    def arange(
        self,
        start: Union[int, float],
        /,
        stop: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        *,
        dtype: Optional[Any] = None,
    ) -> jax.Array:
        if stop is None:
            start, stop = 0, start
        if dtype is None:
            # Integer bounds give the index dtype, float bounds the real one.
            integral = all(
                isinstance(v, (int, np.integer)) for v in (start, stop, step)
            )
            dtype = self._index_dtype if integral else self._dtype
        return jnp.arange(start, stop, step, dtype=dtype, device=self._device)

    def linspace(self, start: float, stop: float, num: int) -> jax.Array:
        return jnp.linspace(start, stop, num, dtype=self._dtype, device=self._device)

    def logspace(self, start: float, stop: float, num: int) -> jax.Array:
        return jnp.logspace(start, stop, num, dtype=self._dtype, device=self._device)

    @staticmethod
    def zeros_like(array: jax.Array, dtype: Optional[Any] = None) -> jax.Array:
        return jnp.zeros_like(array, dtype=dtype)

    @staticmethod
    def ones_like(array: jax.Array, dtype: Optional[Any] = None) -> jax.Array:
        return jnp.ones_like(array, dtype=dtype)

    @staticmethod
    def full_like(a: jax.Array, fill_value: float) -> jax.Array:
        return jnp.full_like(a, fill_value)

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
    def sin(array: jax.Array) -> jax.Array:
        return jnp.sin(array)

    @staticmethod
    def cos(array: jax.Array) -> jax.Array:
        return jnp.cos(array)

    @staticmethod
    def tanh(array: jax.Array) -> jax.Array:
        return jnp.tanh(array)

    @staticmethod
    def sinh(array: jax.Array) -> jax.Array:
        return jnp.sinh(array)

    @staticmethod
    def cosh(array: jax.Array) -> jax.Array:
        return jnp.cosh(array)

    @staticmethod
    def arcsin(array: jax.Array) -> jax.Array:
        return jnp.arcsin(array)

    @staticmethod
    def arccos(array: jax.Array) -> jax.Array:
        return jnp.arccos(array)

    @staticmethod
    def arctan(array: jax.Array) -> jax.Array:
        return jnp.arctan(array)

    @staticmethod
    def arctan2(y: jax.Array, x: jax.Array) -> jax.Array:
        return jnp.arctan2(y, x)

    @staticmethod
    def arcsinh(array: jax.Array) -> jax.Array:
        return jnp.arcsinh(array)

    @staticmethod
    def arccosh(array: jax.Array) -> jax.Array:
        return jnp.arccosh(array)

    @staticmethod
    def arctanh(array: jax.Array) -> jax.Array:
        return jnp.arctanh(array)

    @staticmethod
    def log(array: jax.Array) -> jax.Array:
        return jnp.log(array)

    @staticmethod
    def log1p(array: jax.Array) -> jax.Array:
        return jnp.log1p(array)

    @staticmethod
    def expm1(array: jax.Array) -> jax.Array:
        return jnp.expm1(array)

    @staticmethod
    def erf(array: jax.Array) -> jax.Array:
        result: jax.Array = jax.scipy.special.erf(array)
        return result

    @staticmethod
    def erfinv(array: jax.Array) -> jax.Array:
        result: jax.Array = jax.scipy.special.erfinv(array)
        return result

    @staticmethod
    def ndtr(array: jax.Array) -> jax.Array:
        result: jax.Array = jax.scipy.special.ndtr(array)
        return result

    @staticmethod
    def log_ndtr(array: jax.Array) -> jax.Array:
        result: jax.Array = jax.scipy.special.log_ndtr(array)
        return result

    @staticmethod
    def gammaln(array: jax.Array) -> jax.Array:
        result: jax.Array = jax.scipy.special.gammaln(array)
        return result

    @staticmethod
    def digamma(array: jax.Array) -> jax.Array:
        result: jax.Array = jax.scipy.special.digamma(array)
        return result

    @staticmethod
    def abs(array: jax.Array) -> jax.Array:
        return jnp.abs(array)

    @staticmethod
    def sign(array: jax.Array) -> jax.Array:
        return jnp.sign(array)

    @staticmethod
    def conj(array: jax.Array) -> jax.Array:
        return jnp.conj(array)

    @staticmethod
    def round(array: jax.Array) -> jax.Array:
        return jnp.round(array)

    @staticmethod
    def floor(array: jax.Array) -> jax.Array:
        return jnp.floor(array)

    @staticmethod
    def ceil(array: jax.Array) -> jax.Array:
        return jnp.ceil(array)

    @staticmethod
    def floor_divide(array: jax.Array, divisor: "int | jax.Array") -> jax.Array:
        return jnp.floor_divide(array, divisor)

    @staticmethod
    def maximum(x1: jax.Array, x2: jax.Array) -> jax.Array:
        return jnp.maximum(x1, x2)

    @staticmethod
    def minimum(x1: jax.Array, x2: jax.Array) -> jax.Array:
        return jnp.minimum(x1, x2)

    @staticmethod
    def clip(array: jax.Array, a_min: Any, a_max: Any) -> jax.Array:
        return jnp.clip(array, a_min, a_max)

    @staticmethod
    def isnan(array: jax.Array) -> jax.Array:
        return jnp.isnan(array)

    @staticmethod
    def isinf(array: jax.Array) -> jax.Array:
        return jnp.isinf(array)

    @staticmethod
    def isfinite(array: jax.Array) -> jax.Array:
        return jnp.isfinite(array)

    @staticmethod
    def equal(a: jax.Array, b: "float | int | jax.Array") -> jax.Array:
        return jnp.equal(a, b)

    @staticmethod
    def allclose(
        array1: jax.Array,
        array2: jax.Array,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        return bool(
            jnp.allclose(array1, array2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        )

    @staticmethod
    def cdist(XA: jax.Array, XB: jax.Array, p: float = 2.0) -> jax.Array:
        differences = XA[:, None, :] - XB[None, :, :]
        if p == 1.0:
            return jnp.sum(jnp.abs(differences), axis=-1)
        if p == 2.0:
            return jnp.sqrt(jnp.sum(differences**2, axis=-1))
        raise ValueError("p must be either 1.0 or 2.0")

    @staticmethod
    def dot(Amat: jax.Array, Bmat: jax.Array) -> jax.Array:
        return jnp.dot(Amat, Bmat)

    @staticmethod
    def einsum(subscripts: str, *operands: jax.Array) -> jax.Array:
        return jnp.einsum(subscripts, *operands)

    @overload
    @staticmethod
    def where(condition: jax.Array) -> Tuple[jax.Array, ...]: ...

    @overload
    @staticmethod
    def where(
        condition: jax.Array,
        x: Union[jax.Array, float],
        y: Union[jax.Array, float],
    ) -> jax.Array: ...

    @staticmethod
    def where(
        condition: jax.Array,
        x: Optional[Union[jax.Array, float]] = None,
        y: Optional[Union[jax.Array, float]] = None,
    ) -> Union[jax.Array, Tuple[jax.Array, ...]]:
        if x is None and y is None:
            return tuple(jnp.nonzero(condition))
        if x is not None and y is not None:
            return jnp.where(condition, x, y)
        raise ValueError("Must provide both x and y or neither.")

    # ---- reductions

    @staticmethod
    def sum(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        return jnp.sum(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        return jnp.mean(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def prod(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        return jnp.prod(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def min(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        check_ordered_reduction(
            "min",
            jnp.iscomplexobj(array),
            tuple(array.shape),
            reduction_axes(array.ndim, axis),
        )
        return jnp.min(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(array: jax.Array, axis: Axes = None, keepdims: bool = False) -> jax.Array:
        check_ordered_reduction(
            "max",
            jnp.iscomplexobj(array),
            tuple(array.shape),
            reduction_axes(array.ndim, axis),
        )
        return jnp.max(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def argmin(array: jax.Array, axis: Optional[int] = None) -> jax.Array:
        check_ordered_reduction(
            "argmin",
            jnp.iscomplexobj(array),
            tuple(array.shape),
            reduction_axes(array.ndim, axis),
        )
        return jnp.argmin(array, axis=axis)

    @staticmethod
    def argmax(array: jax.Array, axis: Optional[int] = None) -> jax.Array:
        check_ordered_reduction(
            "argmax",
            jnp.iscomplexobj(array),
            tuple(array.shape),
            reduction_axes(array.ndim, axis),
        )
        return jnp.argmax(array, axis=axis)

    @staticmethod
    def var(
        array: jax.Array, axis: Axes = None, keepdims: bool = False, ddof: int = 0
    ) -> jax.Array:
        return jnp.var(array, axis=axis, keepdims=keepdims, ddof=ddof)

    @staticmethod
    def std(
        array: jax.Array, axis: Axes = None, keepdims: bool = False, ddof: int = 0
    ) -> jax.Array:
        return jnp.std(array, axis=axis, keepdims=keepdims, ddof=ddof)

    @staticmethod
    def logsumexp(
        array: jax.Array, axis: Axes = None, keepdims: bool = False
    ) -> jax.Array:
        result: jax.Array = jax.scipy.special.logsumexp(
            array, axis=axis, keepdims=keepdims
        )
        return result

    @staticmethod
    def cumsum(array: jax.Array, axis: Optional[int] = None) -> jax.Array:
        return jnp.cumsum(array, axis=axis)

    @staticmethod
    def quantile(array: jax.Array, level: float, axis: int = 0) -> jax.Array:
        return jnp.quantile(array, level, axis=axis)

    @staticmethod
    def cov(array: jax.Array, rowvar: bool = True, ddof: int = 1) -> jax.Array:
        # 2-D for one variable too, as the other backends.
        return jnp.atleast_2d(jnp.cov(array, rowvar=rowvar, ddof=ddof))

    @staticmethod
    def diff(array: jax.Array, n: int = 1, axis: int = -1) -> jax.Array:
        return jnp.diff(array, n=n, axis=axis)

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

    # ---- linear algebra: solves and factorizations

    @staticmethod
    def solve(Amat: jax.Array, Bmat: jax.Array) -> jax.Array:
        return _finite_or_raise(jnp.linalg.solve(Amat, Bmat), "singular matrix")

    @staticmethod
    def solve_triangular(
        matrix: jax.Array,
        rhs: jax.Array,
        *,
        lower: bool,
        unit_diagonal: bool = False,
    ) -> jax.Array:
        result: jax.Array = jax.scipy.linalg.solve_triangular(
            matrix, rhs, lower=lower, unit_diagonal=unit_diagonal
        )
        return result

    @staticmethod
    def cholesky(array: jax.Array) -> jax.Array:
        return _finite_or_raise(
            jnp.linalg.cholesky(array), "matrix is not positive definite"
        )

    @staticmethod
    def cholesky_solve(
        matrix: jax.Array, rhs: jax.Array, lower: bool = True
    ) -> jax.Array:
        result: jax.Array = jax.scipy.linalg.cho_solve((matrix, lower), rhs)
        return result

    @staticmethod
    def lu(array: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        p, lower, upper = jax.scipy.linalg.lu(array)
        return p, lower, upper

    def lu_factor(self, array: jax.Array) -> Tuple[jax.Array, jax.Array]:
        lu, pivots = jax.scipy.linalg.lu_factor(array)
        # Pivots in the index dtype; JAX gives LAPACK's int32.
        return lu, pivots.astype(self._index_dtype)

    @staticmethod
    def lu_solve(
        LU: jax.Array, pivots: jax.Array, rhs: jax.Array, transpose: bool = False
    ) -> jax.Array:
        result: jax.Array = jax.scipy.linalg.lu_solve(
            (LU, pivots.astype(jnp.int32)), rhs, trans=1 if transpose else 0
        )
        return result

    @staticmethod
    def inv(matrix: jax.Array) -> jax.Array:
        return _finite_or_raise(jnp.linalg.inv(matrix), "singular matrix")

    @staticmethod
    def det(array: jax.Array) -> jax.Array:
        result: jax.Array = jnp.linalg.det(array)
        return result

    @staticmethod
    def slogdet(array: jax.Array) -> Tuple[jax.Array, jax.Array]:
        sign, logabsdet = jnp.linalg.slogdet(array)
        return sign, logabsdet

    # ---- linear algebra: decompositions, least squares, norms, products

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

    # ---- shape and indexing. JAX arrays are immutable, so whether a
    # result shares memory with its input cannot be observed.

    @staticmethod
    def flatten(array: jax.Array) -> jax.Array:
        return jnp.ravel(array)

    @staticmethod
    def ravel(array: jax.Array) -> jax.Array:
        return jnp.ravel(array)

    @staticmethod
    def squeeze(array: jax.Array) -> jax.Array:
        return jnp.squeeze(array)

    @staticmethod
    def transpose(array: jax.Array, axes: Optional[Sequence[int]] = None) -> jax.Array:
        return jnp.transpose(array, None if axes is None else tuple(axes))

    @staticmethod
    def flip(array: jax.Array, axis: Axes = None) -> jax.Array:
        return jnp.flip(array, axis=axis)

    @staticmethod
    def tile(array: jax.Array, reps: Tuple[int, ...]) -> jax.Array:
        return jnp.tile(array, reps)

    @staticmethod
    def repeat(array: jax.Array, repeats: int, axis: Optional[int] = None) -> jax.Array:
        return jnp.repeat(array, repeats, axis=axis)

    @staticmethod
    def split(
        array: jax.Array, indices_or_sections: jax.Array, axis: int = 0
    ) -> List[jax.Array]:
        # The indices fix the output shapes, so they must be concrete.
        pieces: List[jax.Array] = jnp.split(
            array, np.asarray(indices_or_sections), axis=axis
        )
        return pieces

    @staticmethod
    def hstack(arrays: Union[List[jax.Array], Tuple[jax.Array, ...]]) -> jax.Array:
        return jnp.hstack(arrays)

    @staticmethod
    def vstack(arrays: Union[List[jax.Array], Tuple[jax.Array, ...]]) -> jax.Array:
        return jnp.vstack(arrays)

    @staticmethod
    def block(blocks: List[List[jax.Array]]) -> jax.Array:
        return jnp.block(blocks)

    @staticmethod
    def tril(array: jax.Array, k: int = 0) -> jax.Array:
        return jnp.tril(array, k)

    @staticmethod
    def triu(array: jax.Array, k: int = 0) -> jax.Array:
        return jnp.triu(array, k)

    def tril_indices(self, n: int, k: int = 0) -> Tuple[jax.Array, jax.Array]:
        rows, cols = jnp.tril_indices(n, k)
        return rows.astype(self._index_dtype), cols.astype(self._index_dtype)

    @staticmethod
    def get_diagonal(
        array: jax.Array, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> jax.Array:
        result: jax.Array = jnp.diagonal(array, offset, axis1=axis1, axis2=axis2)
        return result

    @staticmethod
    def diag(array: jax.Array, k: int = 0) -> jax.Array:
        return jnp.diag(array, k)

    @staticmethod
    def nonzero(condition: jax.Array) -> Tuple[jax.Array, ...]:
        return tuple(jnp.nonzero(condition))

    @staticmethod
    def unique(array: jax.Array) -> jax.Array:
        result: jax.Array = jnp.unique(array)
        return result

    @staticmethod
    def sort(array: jax.Array, axis: int = -1) -> jax.Array:
        check_orderable("sort", jnp.iscomplexobj(array))
        return jnp.sort(array, axis=axis, stable=True)

    @staticmethod
    def argsort(array: jax.Array, axis: int = -1) -> jax.Array:
        check_orderable("argsort", jnp.iscomplexobj(array))
        return jnp.argsort(array, axis=axis, stable=True)

    def searchsorted(
        self,
        sorted_array: jax.Array,
        values: jax.Array,
        side: Literal["left", "right"] = "left",
    ) -> jax.Array:
        # In the index dtype; JAX gives int32 even with 64-bit types on.
        return jnp.searchsorted(sorted_array, values, side=side).astype(
            self._index_dtype
        )

    @staticmethod
    def atleast_1d(array: jax.Array) -> jax.Array:
        result: jax.Array = jnp.atleast_1d(array)
        return result

    @staticmethod
    def atleast_2d(array: jax.Array) -> jax.Array:
        result: jax.Array = jnp.atleast_2d(array)
        return result

    @staticmethod
    def meshgrid(
        *arrays: jax.Array, indexing: Literal["xy", "ij"] = "xy"
    ) -> Tuple[jax.Array, ...]:
        return tuple(jnp.meshgrid(*arrays, indexing=indexing))

    @staticmethod
    def copy(array: jax.Array) -> jax.Array:
        return jnp.array(array, copy=True)
