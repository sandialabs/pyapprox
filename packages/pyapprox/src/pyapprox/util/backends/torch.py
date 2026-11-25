import functools
import math
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
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.util.backends.errors import LinAlgError
from pyapprox.util.backends.precision import Precision
from pyapprox.util.backends.protocols import (
    ArrayLike,
    ArrayProtocol,
    Backend,
    IndexLike,
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
from pyapprox.util.openmp import use_torch_openmp
from pyapprox.util.optional_deps import missing_dependency_message

try:
    import torch
except ImportError as err:
    raise ImportError(missing_dependency_message("torch", "TorchBkd", "torch")) from err

# annotated alias: set_default_dtype is untyped in older torch stubs, and
# a type: ignore would be flagged unused under newer (typed) stubs
_set_default_dtype: Callable[["torch.dtype"], None] = torch.set_default_dtype

# torch.linalg.LinAlgError exists at runtime but the stubs do not export
# it; a RuntimeError subclass.
_TorchLinAlgError: type[RuntimeError] = getattr(torch.linalg, "LinAlgError")

# The torch dtype of each NumPy dtype NumPy's promotion can produce from
# torch dtypes (torch has no unsigned widths beyond uint8).
_TORCH_DTYPES: Dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.complex64): torch.complex64,
    np.dtype(np.complex128): torch.complex128,
}
_NUMPY_DTYPES: Dict[torch.dtype, np.dtype[Any]] = {
    t: n for n, t in _TORCH_DTYPES.items()
}

# Each floating or complex dtype's (real, complex) pair of its width.
_WIDTHS: Dict[torch.dtype, Tuple[torch.dtype, torch.dtype]] = {
    torch.float32: (torch.float32, torch.complex64),
    torch.complex64: (torch.float32, torch.complex64),
    torch.float64: (torch.float64, torch.complex128),
    torch.complex128: (torch.float64, torch.complex128),
}


def _variance(
    array: torch.Tensor, dims: Tuple[int, ...], keepdim: bool, ddof: int
) -> torch.Tensor:
    """NumPy's variance over ``dims`` (nonempty): ``mean(|x - mean|**2)``
    with ``ddof`` subtracted from the count, real for complex input.

    Complex input is written out: torch on MPS returns a wrong, complex
    variance of complex data. Real input uses ``torch.var``.
    """
    if not array.is_complex():
        return torch.var(array, dim=dims, keepdim=keepdim, correction=ddof)
    # A loop, not math.prod over a generator, which dynamo cannot trace.
    count = 1
    for dim in dims:
        count *= array.shape[dim]
    deviation = array - torch.mean(array, dim=dims, keepdim=True)
    squares = torch.real(deviation) ** 2 + torch.imag(deviation) ** 2
    return torch.sum(squares, dim=dims, keepdim=keepdim) / (count - ddof)


def _numpy_dtype(dtype: torch.dtype) -> np.dtype[Any]:
    result = _NUMPY_DTYPES.get(dtype)
    if result is None:
        raise ValueError(f"{dtype} has no NumPy equivalent")
    return result


def _width_pair(dtype: Any) -> Tuple[torch.dtype, torch.dtype]:
    pair = _WIDTHS.get(dtype)
    if pair is None:
        raise ValueError(f"needs a floating or complex dtype, got {dtype}")
    return pair


# TorchBkd.jit's cache key: kernel, static argument names, dynamic.
_JitKey = Tuple[Callable[..., Any], Tuple[str, ...], bool]


def _bool_reduction(
    array: torch.Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]],
    keepdims: bool,
    every: bool,
) -> torch.Tensor:
    """``any`` (``every=False``) or ``all`` with NumPy's axis semantics.

    torch.any/all return uint8 for uint8 input and have no keepdim when
    every axis is reduced, and an empty dim tuple does not mean "reduce
    nothing" as it does in NumPy; each case is handled here.
    """
    if array.dtype != torch.bool:
        array = array != 0
    if axis == ():
        return array
    reduce = torch.all if every else torch.any
    if axis is None:
        result = reduce(array)
        return result.reshape((1,) * array.ndim) if keepdims else result
    return reduce(array, dim=axis, keepdim=keepdims)


_Reduce = Callable[[torch.Tensor, Tuple[int, ...], bool], torch.Tensor]


def _numpy_reduction(
    reduce: _Reduce,
    array: torch.Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]],
    keepdims: bool,
) -> torch.Tensor:
    """``reduce(array, dims, keepdim)`` with NumPy's axis semantics.

    Every axis is passed explicitly for ``axis=None``, so ``keepdims``
    holds there too; an empty tuple, which torch reads as every axis,
    reduces each element as a group of one, as NumPy does.
    """
    dims = reduction_axes(array.ndim, axis)
    if not dims:
        return reduce(array.unsqueeze(-1), (-1,), False)
    return reduce(array, dims, keepdims)


def _prod(array: torch.Tensor, dims: Tuple[int, ...], keepdim: bool) -> torch.Tensor:
    # torch.prod reduces one dim at a time.
    for dim in sorted((d % array.ndim for d in dims), reverse=True):
        array = torch.prod(array, dim=dim, keepdim=True)
    return array if keepdim else array.squeeze(dims)


# Implement the PyTorch backend
class TorchBkd(Backend[torch.Tensor]):  # Specify torch.Tensor type
    def __init__(
        self,
        *,
        precision: Precision = Precision.DOUBLE,
        device: Optional[Union[str, "torch.device"]] = None,
        compile: bool = False,
    ) -> None:
        """``precision`` sets the real working dtype (float32 or float64)
        and with it the complex one; ``device`` defaults to the CPU. MPS
        has no float64, so ``device="mps"`` needs ``Precision.SINGLE``.

        ``compile=True`` makes :meth:`jit` compile kernels with
        ``torch.compile``. Off by default: a torch-compiled kernel supports
        first derivatives only, so compiling by default would take
        ``hvp`` away from every generic kernel."""
        if not isinstance(precision, Precision):
            raise TypeError(
                f"precision must be a Precision, got {type(precision).__name__}"
            )
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        if self._device.type == "mps" and precision is Precision.DOUBLE:
            raise ValueError("MPS has no float64: use Precision.SINGLE on mps")
        _set_default_dtype(torch.float64)
        # Every torch.compile path in the library runs with a TorchBkd, so
        # this runs before inductor builds its first kernel.
        use_torch_openmp()
        self._precision = precision
        single = precision is Precision.SINGLE
        self._dtype = torch.float32 if single else torch.float64
        self._complex_dtype = torch.complex64 if single else torch.complex128
        self._compile = compile
        self._jit_cache: Dict[_JitKey, Callable[..., Any]] = {}

    def _placement(self) -> Tuple[str, int]:
        # "mps" and "mps:0" name one device; an unindexed device is index 0.
        index = self._device.index
        return (self._device.type, 0 if index is None else index)

    def __repr__(self) -> str:
        compiled = ", compile=True" if self._compile else ""
        return (
            f"TorchBkd(precision={self._precision.value}, "
            f"device={self._device}{compiled})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            self.is_compatible(other)
            and isinstance(other, TorchBkd)
            and other._compile == self._compile
        )

    def __hash__(self) -> int:
        return hash((TorchBkd, self._precision, self._placement(), self._compile))

    def is_compatible(self, other: object) -> bool:
        return (
            isinstance(other, TorchBkd)
            and other._precision is self._precision
            and other._placement() == self._placement()
        )

    def replace(
        self,
        *,
        precision: Optional[Precision] = None,
        device: Optional[Union[str, "torch.device"]] = None,
        compile: Optional[bool] = None,
    ) -> "TorchBkd":
        return TorchBkd(
            precision=self._precision if precision is None else precision,
            device=self._device if device is None else device,
            compile=self._compile if compile is None else compile,
        )

    def from_other(self, array: ArrayProtocol, src: Backend[Any]) -> torch.Tensor:
        if isinstance(array, torch.Tensor) and self.is_compatible(src):
            return array
        if isinstance(array, torch.Tensor):
            # tensor.to keeps the autograd graph across precision and device.
            if array.is_floating_point():
                dtype = self._dtype
            elif array.is_complex():
                dtype = self._complex_dtype
            else:
                dtype = array.dtype
            return array.to(device=self._device, dtype=dtype)
        # Another library: through NumPy, so not differentiable through.
        return self.array(src.to_numpy(array))

    def __getstate__(self) -> Dict[str, Any]:
        # Compiled kernels do not pickle; the cache is rebuilt on use.
        state = dict(self.__dict__)
        state["_jit_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def jit(
        self,
        kernel: Callable[..., KernelResult],
        static_argnames: Sequence[str] = (),
        dynamic: bool = False,
    ) -> Callable[..., KernelResult]:
        # torch.compile specializes on Python scalars and other non-tensor
        # arguments by itself, so static_argnames needs no translation.
        # fullgraph=True: a graph break in a kernel raises instead of
        # silently running part of it eagerly.
        if torch.compiler.is_compiling():
            # Nested in a kernel being traced: inlined into its graph. The
            # check is skipped because dynamo cannot trace it; the same
            # call is checked whenever it runs uncompiled.
            return functools.partial(kernel, bkd=self)
        check_jit_kernel(kernel)
        bound = functools.partial(kernel, bkd=self)
        if not self._compile:
            return bound
        key = (kernel, tuple(static_argnames), dynamic)
        compiled = self._jit_cache.get(key)
        if compiled is None:
            compiled = torch.compile(bound, fullgraph=True, dynamic=dynamic)
            self._jit_cache[key] = compiled
        return compiled

    def synchronize(self) -> None:
        """Block until all pending operations on this backend's device complete.

        CPU is always synchronous so this is a no-op for device="cpu".
        For CUDA and MPS devices, this waits for all queued GPU work to finish.
        """
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        elif self._device.type == "mps":
            torch.mps.synchronize()

    @staticmethod
    def dot(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        # for typing consistancy make sure dot always returns an array
        return torch.matmul(Amat, Bmat)

    def eye(
        self, nrows: int, ncols: Optional[int] = None, dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.eye(
            nrows,
            ncols if ncols is not None else nrows,
            dtype=dtype if dtype is not None else self._dtype,
            device=self._device,
        )

    @staticmethod
    def inv(matrix: torch.Tensor) -> torch.Tensor:
        try:
            result: torch.Tensor = torch.linalg.inv(matrix)
        except _TorchLinAlgError as err:
            raise LinAlgError(str(err)) from err
        return result

    @staticmethod
    def pinv(matrix: torch.Tensor, *, rtol: Optional[float] = None) -> torch.Tensor:
        if rtol is None:
            rtol = max(matrix.shape[-2:]) * torch.finfo(matrix.dtype).eps
        result: torch.Tensor = torch.linalg.pinv(matrix, rtol=rtol)
        return result

    def array(
        self,
        array: ArrayLike,
        dtype: Optional[Any] = None,
    ) -> torch.Tensor:
        source = (
            array.detach()
            if isinstance(array, torch.Tensor)
            else torch.as_tensor(array)
        )
        if dtype is None:
            # Precision is normalized and kind kept: real floating becomes
            # the real working dtype, complex the complex one.
            if source.is_floating_point():
                dtype = self._dtype
            elif source.is_complex():
                dtype = self._complex_dtype
            else:
                dtype = source.dtype
        # copy=True: as_tensor shares a NumPy array's memory.
        return source.to(dtype=dtype, device=self._device, copy=True)

    def asarray(
        self,
        array: ArrayLike,
        dtype: Optional[Any] = None,
    ) -> torch.Tensor:
        if dtype is None and not isinstance(array, torch.Tensor):
            dtype = self._dtype
        return torch.as_tensor(array, dtype=dtype, device=self._device)

    @staticmethod
    def assert_allclose(
        actual: torch.Tensor,
        desired: torch.Tensor,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: Optional[str] = None,
    ) -> None:
        torch.testing.assert_close(
            actual,
            desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            msg=err_msg,
        )

    def precision(self) -> Precision:
        return self._precision

    def device(self) -> str:
        return str(self._device)

    def real_dtype(self) -> Any:
        return self._dtype

    def complex_dtype(self) -> Any:
        return self._complex_dtype

    def index_dtype(self) -> Any:
        return torch.int64

    def eps(self) -> float:
        return float(torch.finfo(self._dtype).eps)

    def result_type(self, *arrays: torch.Tensor) -> Any:
        # NumPy's rules, which torch's differ from (int with float32 is
        # float64 in NumPy, float32 in torch); dtypes only, no data.
        numpy_dtype = np.result_type(*(_numpy_dtype(a.dtype) for a in arrays))
        return _TORCH_DTYPES[np.dtype(numpy_dtype)]

    def isdtype(self, array: torch.Tensor, kind: str) -> bool:
        check_dtype_kind(kind)
        dtype = array.dtype
        if kind == "bool":
            return dtype == torch.bool
        if kind == "real floating":
            return dtype.is_floating_point
        if kind == "complex floating":
            return dtype.is_complex
        return not (dtype.is_floating_point or dtype.is_complex) and (
            dtype != torch.bool
        )

    def real_dtype_of(self, dtype: Any) -> Any:
        return _width_pair(dtype)[0]

    def complex_dtype_of(self, dtype: Any) -> Any:
        return _width_pair(dtype)[1]

    @staticmethod
    def real(array: torch.Tensor) -> torch.Tensor:
        return torch.real(array)

    @staticmethod
    def imag(array: torch.Tensor) -> torch.Tensor:
        # torch.imag refuses real input; NumPy gives zeros.
        if not array.is_complex():
            return torch.zeros_like(array)
        return torch.imag(array)

    def double_dtype(self) -> Any:
        return self._dtype

    def int64_dtype(self) -> Any:
        return torch.int64

    @staticmethod
    def is_integer_dtype(array: torch.Tensor) -> bool:
        # Boolean is not integer, as in NumPy.
        return (
            not array.is_floating_point()
            and not array.is_complex()
            and array.dtype != torch.bool
        )

    @staticmethod
    def is_floating_dtype(array: torch.Tensor) -> bool:
        return array.is_floating_point()

    @staticmethod
    def tracks_gradient(array: torch.Tensor) -> bool:
        """Whether autograd is recording operations on ``array``.

        ``grad_fn`` covers arrays produced by differentiated
        operations; ``requires_grad`` covers leaves that gradients are
        requested for but which no operation has yet consumed.
        """
        return array.requires_grad or array.grad_fn is not None

    @staticmethod
    def stack(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        axis: int = 0,
    ) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def concatenate(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        axis: int = 0,
    ) -> torch.Tensor:
        return torch.cat(arrays, dim=axis)

    @staticmethod
    def hstack(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        return torch.hstack(arrays)

    @staticmethod
    def vstack(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        return torch.vstack(arrays)

    def linspace(self, start: float, stop: float, num: int) -> torch.Tensor:
        return torch.linspace(start, stop, num, dtype=self._dtype, device=self._device)

    def logspace(self, start: float, stop: float, num: int) -> torch.Tensor:
        return torch.logspace(start, stop, num, dtype=self._dtype, device=self._device)

    @staticmethod
    def to_numpy(array: Union[torch.Tensor, np.ndarray, float, int]) -> NDArray[Any]:
        if isinstance(array, np.ndarray):
            return array
        if not isinstance(array, torch.Tensor):
            return np.asarray(array)
        # resolve_conj: a lazily conjugated tensor (conj) refuses numpy().
        return np.asarray(array.detach().cpu().resolve_conj().numpy())

    @staticmethod
    def to_float(array: "float | int | torch.Tensor") -> float:
        # A Python or NumPy scalar converts as Python converts it.
        if not isinstance(array, torch.Tensor):
            return float(array)
        check_scalar_conversion("to_float", tuple(array.shape), array.is_complex())
        return float(array.item())

    @staticmethod
    def to_int(array: "float | int | torch.Tensor") -> int:
        if not isinstance(array, torch.Tensor):
            return int(array)
        check_scalar_conversion("to_int", tuple(array.shape), array.is_complex())
        return int(array.item())

    @staticmethod
    def flatten(array: torch.Tensor) -> torch.Tensor:
        # One contiguous copy, as NumPy's flatten: torch.flatten returns a
        # view, or the input itself when it is 1-D.
        return array.clone(memory_format=torch.contiguous_format).reshape(-1)

    @staticmethod
    def squeeze(array: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(array)

    @staticmethod
    def ravel(array: torch.Tensor) -> torch.Tensor:
        return torch.ravel(array)

    @staticmethod
    def meshgrid(
        *arrays: torch.Tensor, indexing: Literal["xy", "ij"] = "xy"
    ) -> Tuple[torch.Tensor, ...]:
        return torch.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def reshape(array: torch.Tensor, newshape: Sequence[int]) -> torch.Tensor:
        return torch.reshape(array, newshape)

    @staticmethod
    def transpose(
        array: torch.Tensor,
        axes: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if axes is None:
            # Tensor.T is deprecated beyond two dimensions.
            return array.permute(*reversed(range(array.ndim)))
        return array.permute(*axes)

    @staticmethod
    def sum(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return _numpy_reduction(
            lambda x, dims, keep: torch.sum(x, dim=dims, keepdim=keep),
            array,
            axis,
            keepdims,
        )

    @staticmethod
    def cumsum(array: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        if axis is None:
            return torch.cumsum(array.flatten(), dim=0)
        return torch.cumsum(array, dim=axis)

    @staticmethod
    def sin(array: torch.Tensor) -> torch.Tensor:
        return torch.sin(array)

    @staticmethod
    def cos(array: torch.Tensor) -> torch.Tensor:
        return torch.cos(array)

    @staticmethod
    def arcsin(array: torch.Tensor) -> torch.Tensor:
        return torch.arcsin(array)

    @staticmethod
    def arccos(array: torch.Tensor) -> torch.Tensor:
        return torch.arccos(array)

    @staticmethod
    def arctan(array: torch.Tensor) -> torch.Tensor:
        return torch.arctan(array)

    @staticmethod
    def arctan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.arctan2(y, x)

    @staticmethod
    def sinh(array: torch.Tensor) -> torch.Tensor:
        return torch.sinh(array)

    @staticmethod
    def cosh(array: torch.Tensor) -> torch.Tensor:
        return torch.cosh(array)

    @staticmethod
    def tanh(array: torch.Tensor) -> torch.Tensor:
        return torch.tanh(array)

    @staticmethod
    def arcsinh(array: torch.Tensor) -> torch.Tensor:
        return torch.asinh(array)

    @staticmethod
    def arccosh(array: torch.Tensor) -> torch.Tensor:
        return torch.acosh(array)

    @staticmethod
    def arctanh(array: torch.Tensor) -> torch.Tensor:
        return torch.atanh(array)

    def full(
        self, shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.full(
            shape,
            fill_value,
            dtype=dtype if dtype is not None else self._dtype,
            device=self._device,
        )

    def full_like(self, a: torch.Tensor, fill_value: float) -> torch.Tensor:
        return torch.full_like(a, fill_value)

    def zeros(
        self, shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.zeros(
            shape,
            dtype=dtype if dtype is not None else self._dtype,
            device=self._device,
        )

    def ones(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> torch.Tensor:
        return torch.ones(
            shape,
            dtype=dtype if dtype is not None else self._dtype,
            device=self._device,
        )

    def empty(
        self, shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.empty(
            shape,
            dtype=dtype if dtype is not None else self._dtype,
            device=self._device,
        )

    def arange(
        self,
        start: Union[int, float],
        /,
        stop: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        *,
        dtype: Optional[Any] = None,
    ) -> torch.Tensor:
        if stop is None:
            start, stop = 0, start
        if dtype is None and not all(isinstance(v, int) for v in (start, stop, step)):
            # The configured dtype, not torch's global default.
            dtype = self._dtype
        return torch.arange(start, stop, step, dtype=dtype, device=self._device)

    @staticmethod
    def prod(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return _numpy_reduction(_prod, array, axis, keepdims)

    @staticmethod
    def any(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return _bool_reduction(array, axis, keepdims, every=False)

    @staticmethod
    def all(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return _bool_reduction(array, axis, keepdims, every=True)

    @staticmethod
    def to_bool(array: torch.Tensor) -> bool:
        check_to_bool_shape(tuple(array.shape))
        return bool(array.reshape(()))

    @staticmethod
    def log(array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    @staticmethod
    def exp(array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    @staticmethod
    def expm1(array: torch.Tensor) -> torch.Tensor:
        return torch.expm1(array)

    @staticmethod
    def log1p(array: torch.Tensor) -> torch.Tensor:
        return torch.log1p(array)

    @staticmethod
    def copy(array: torch.Tensor) -> torch.Tensor:
        return array.clone()

    @staticmethod
    def erf(array: torch.Tensor) -> torch.Tensor:
        return torch.erf(array)

    @staticmethod
    def erfinv(array: torch.Tensor) -> torch.Tensor:
        return torch.erfinv(array)

    @staticmethod
    def ndtr(array: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, torch.special.ndtr(array))

    @staticmethod
    def log_ndtr(array: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, torch.special.log_ndtr(array))

    @staticmethod
    def gammaln(array: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(array)

    @staticmethod
    def digamma(array: torch.Tensor) -> torch.Tensor:
        return torch.digamma(array)

    @staticmethod
    def equal(a: torch.Tensor, b: "float | int | torch.Tensor") -> torch.Tensor:
        return torch.eq(a, b)

    @staticmethod
    def isfinite(array: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(array)

    @staticmethod
    def nonzero(condition: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return torch.nonzero(condition, as_tuple=True)

    @staticmethod
    def norm(
        array: torch.Tensor,
        ord: Optional[Union[int, float, Literal["fro", "nuc"]]] = None,
        axis: Optional[Union[int, Tuple[int, int]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return cast(torch.Tensor, torch.linalg.norm(array, ord, axis, keepdims))

    @staticmethod
    def sign(array: torch.Tensor) -> torch.Tensor:
        # NaN stays NaN, as in NumPy; torch.sign gives 0.
        return torch.where(torch.isnan(array), array, torch.sign(array))

    @staticmethod
    def abs(array: torch.Tensor) -> torch.Tensor:
        return torch.abs(array)

    @staticmethod
    def conj(array: torch.Tensor) -> torch.Tensor:
        # A lazy conjugate view: no copy, and autograd follows it.
        return torch.conj(array)

    @staticmethod
    def round(array: torch.Tensor) -> torch.Tensor:
        return torch.round(array)

    @staticmethod
    def floor(array: torch.Tensor) -> torch.Tensor:
        return torch.floor(array)

    @staticmethod
    def floor_divide(
        array: torch.Tensor, divisor: "int | torch.Tensor"
    ) -> torch.Tensor:
        return torch.floor_divide(array, divisor)

    @staticmethod
    def ceil(array: torch.Tensor) -> torch.Tensor:
        return torch.ceil(array)

    @staticmethod
    def sqrt(array: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(array)

    @staticmethod
    def solve(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        try:
            return cast(torch.Tensor, torch.linalg.solve(Amat, Bmat))
        except _TorchLinAlgError as err:
            raise LinAlgError(str(err)) from err

    @staticmethod
    def solve_sparse(Amat: spmatrix, bvec: torch.Tensor) -> torch.Tensor:
        """Solve A @ x = b where A is a sparse matrix.

        Raises
        ------
        NotImplementedError
            Always — sparse solves are not supported with TorchBkd.
            torch.sparse.spsolve requires MPS/CUDA (not available on CPU),
            and falling back to scipy would silently break the autograd
            computation graph.
        """
        raise NotImplementedError(
            "TorchBkd.solve_sparse() is not supported. "
            "torch.sparse.spsolve requires MPS/CUDA and is not available "
            "on CPU. Falling back to scipy would silently break the "
            "autograd computation graph. Use NumpyBkd for sparse solves "
            "or convert to dense with .toarray() first."
        )

    @staticmethod
    def lstsq(
        a: torch.Tensor,
        b: torch.Tensor,
        rcond: Optional[float] = None,
    ) -> torch.Tensor:
        """Solve least squares problem min ||ax - b||_2.

        The default CPU driver (gelsy, pivoted QR) gives NumPy's
        minimum-norm solution when ``a`` is rank deficient. It does not
        compile under ``fullgraph=True`` (its rank output has a
        data-dependent shape); a compiled kernel solves through ``pinv``
        or a QR factorization instead.
        """
        result = torch.linalg.lstsq(a, b, rcond=rcond)
        return cast(torch.Tensor, result.solution)

    @staticmethod
    def flip(
        array: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        return torch.flip(array, dims=reduction_axes(array.ndim, axis))

    @staticmethod
    def sort(array: torch.Tensor, axis: int = -1) -> torch.Tensor:
        check_orderable("sort", array.is_complex())
        return torch.sort(array, dim=axis, stable=True).values

    @staticmethod
    def quantile(array: torch.Tensor, level: float, axis: int = 0) -> torch.Tensor:
        # NumPy's linear interpolation between order statistics, written
        # out: torch.quantile checks its level against the data, which does
        # not compile under fullgraph=True. The level is a Python float,
        # so the two order statistics are fixed when traced.
        ordered = torch.sort(array, dim=axis).values
        position = level * (array.shape[axis] - 1)
        below = math.floor(position)
        above = min(below + 1, array.shape[axis] - 1)
        weight = position - below
        low = ordered.select(axis, below)
        return low + weight * (ordered.select(axis, above) - low)

    @staticmethod
    def argsort(array: torch.Tensor, axis: int = -1) -> torch.Tensor:
        check_orderable("argsort", array.is_complex())
        return torch.argsort(array, dim=axis, stable=True)

    @staticmethod
    def searchsorted(
        sorted_array: torch.Tensor,
        values: torch.Tensor,
        side: Literal["left", "right"] = "left",
    ) -> torch.Tensor:
        if not sorted_array.is_contiguous():
            sorted_array = sorted_array.contiguous()
        return torch.searchsorted(sorted_array, values, side=side)

    @staticmethod
    def clip(array: torch.Tensor, a_min: Any, a_max: Any) -> torch.Tensor:
        return torch.clamp(array, a_min, a_max)

    @staticmethod
    def min(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        check_ordered_reduction(
            "min",
            array.is_complex(),
            tuple(array.shape),
            reduction_axes(array.ndim, axis),
        )
        return _numpy_reduction(
            lambda x, dims, keep: torch.amin(x, dim=dims, keepdim=keep),
            array,
            axis,
            keepdims,
        )

    @staticmethod
    def max(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        check_ordered_reduction(
            "max",
            array.is_complex(),
            tuple(array.shape),
            reduction_axes(array.ndim, axis),
        )
        return _numpy_reduction(
            lambda x, dims, keep: torch.amax(x, dim=dims, keepdim=keep),
            array,
            axis,
            keepdims,
        )

    @staticmethod
    def logsumexp(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return _numpy_reduction(
            lambda x, dims, keep: torch.logsumexp(x, dim=dims, keepdim=keep),
            array,
            axis,
            keepdims,
        )

    @staticmethod
    def einsum(subscripts: str, *operands: torch.Tensor) -> torch.Tensor:
        return torch.einsum(subscripts, *operands)

    @staticmethod
    def moveaxis(
        array: torch.Tensor,
        source: Union[int, tuple[int, ...]],
        destination: Union[int, tuple[int, ...]],
    ) -> torch.Tensor:
        return torch.movedim(array, cast(Any, source), cast(Any, destination))

    @staticmethod
    def diag(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.diag(array, diagonal=k)

    @staticmethod
    def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # torch.outer takes vectors only; NumPy flattens.
        return torch.outer(a.reshape(-1), b.reshape(-1))

    @staticmethod
    def diff(array: torch.Tensor, n: int = 1, axis: int = -1) -> torch.Tensor:
        return torch.diff(array, n=n, dim=axis)

    @staticmethod
    def allclose(
        array1: torch.Tensor,
        array2: torch.Tensor,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        return torch.allclose(array1, array2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    @staticmethod
    def jacobian(
        fun: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor
    ) -> torch.Tensor:
        _jacobian = cast(
            Callable[
                [Callable[[torch.Tensor], torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
            torch.autograd.functional.jacobian,
        )
        result: torch.Tensor = _jacobian(fun, params)
        return result

    @staticmethod
    def hvp(
        fun: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        _hvp = cast(
            Callable[
                [Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor],
            ],
            torch.autograd.functional.hvp,
        )
        result: torch.Tensor = _hvp(fun, params, vec)[1]
        return result

    @staticmethod
    def jvp(
        fun: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        # set create_graph=True so that result is differentiable.
        _jvp = cast(
            Callable[..., Tuple[torch.Tensor, torch.Tensor]],
            torch.autograd.functional.jvp,
        )
        result: torch.Tensor = _jvp(fun, params, vec, create_graph=True)[1]
        return result

    @staticmethod
    def hessian(
        fun: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor
    ) -> torch.Tensor:
        _hessian = cast(
            Callable[
                [Callable[[torch.Tensor], torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
            torch.autograd.functional.hessian,
        )
        result: torch.Tensor = _hessian(fun, params)
        return result

    @staticmethod
    def cholesky(array: torch.Tensor) -> torch.Tensor:
        try:
            result: torch.Tensor = torch.linalg.cholesky(array)
        except _TorchLinAlgError as err:
            raise LinAlgError(str(err)) from err
        return result

    @staticmethod
    def solve_triangular(
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        *,
        lower: bool,
        unit_diagonal: bool = False,
    ) -> torch.Tensor:
        # torch needs a 2-D right-hand side; NumPy's contract takes 1-D too.
        rhs_2d = rhs[:, None] if rhs.ndim == 1 else rhs
        result: torch.Tensor = torch.linalg.solve_triangular(
            matrix, rhs_2d, upper=(not lower), unitriangular=unit_diagonal
        )
        return result[:, 0] if rhs.ndim == 1 else result

    @staticmethod
    def cholesky_solve(
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        lower: bool = True,
    ) -> torch.Tensor:
        rhs_2d = rhs[:, None] if rhs.ndim == 1 else rhs
        result: torch.Tensor = torch.cholesky_solve(rhs_2d, matrix, upper=(not lower))
        return result[:, 0] if rhs.ndim == 1 else result

    @staticmethod
    def eigvalsh(array: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = torch.linalg.eigvalsh(array)
        return result

    @staticmethod
    def cond(
        array: torch.Tensor,
        p: Optional[Union[int, float, Literal["fro", "nuc"]]] = None,
    ) -> torch.Tensor:
        if p is None or p in (2, -2):
            result: torch.Tensor = torch.linalg.cond(array, p)
            return result
        # As NumPy: torch.linalg.cond writes into its inverse in place,
        # which breaks the backward pass for these norms.
        dims = (-2, -1)
        norm: torch.Tensor = torch.linalg.matrix_norm(array, p, dim=dims)
        inv_norm: torch.Tensor = torch.linalg.matrix_norm(
            torch.linalg.inv(array), p, dim=dims
        )
        return norm * inv_norm

    @staticmethod
    def trace(array: torch.Tensor) -> torch.Tensor:
        # torch.trace takes matrices only; numpy.linalg.trace batches.
        return torch.diagonal(array, dim1=-2, dim2=-1).sum(dim=-1)

    @staticmethod
    def atleast_1d(array: torch.Tensor) -> torch.Tensor:
        _atleast_1d = cast(Callable[[torch.Tensor], torch.Tensor], torch.atleast_1d)
        return _atleast_1d(array)

    @staticmethod
    def atleast_2d(array: torch.Tensor) -> torch.Tensor:
        _atleast_2d = cast(Callable[[torch.Tensor], torch.Tensor], torch.atleast_2d)
        return _atleast_2d(array)

    @staticmethod
    def tile(array: torch.Tensor, reps: Tuple[int, ...]) -> torch.Tensor:
        return torch.tile(array, dims=reps)

    @staticmethod
    def isnan(array: torch.Tensor) -> torch.Tensor:
        return torch.isnan(array)

    @staticmethod
    def isinf(array: torch.Tensor) -> torch.Tensor:
        return torch.isinf(array)

    @staticmethod
    def unique(array: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, torch.unique(array))

    @staticmethod
    def argmin(array: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        check_ordered_reduction(
            "argmin",
            array.is_complex(),
            tuple(array.shape),
            reduction_axes(array.ndim, axis),
        )
        return torch.argmin(array, dim=axis)

    @staticmethod
    def argmax(array: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        check_ordered_reduction(
            "argmax",
            array.is_complex(),
            tuple(array.shape),
            reduction_axes(array.ndim, axis),
        )
        return torch.argmax(array, dim=axis)

    @staticmethod
    def get_diagonal(
        array: torch.Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> torch.Tensor:
        return torch.diagonal(array, offset, dim1=axis1, dim2=axis2)

    @staticmethod
    def cdist(XA: torch.Tensor, XB: torch.Tensor, p: float = 2.0) -> torch.Tensor:
        if p == 1.0:
            return torch.cdist(XA, XB, p)
        if p != 2.0:
            raise ValueError("p must be either 1.0 or 2.0")
        # torch defaults to use_mm_for_euclid_dist_if_necessary, which
        # switches to the ||a||^2 + ||b||^2 - 2a.b identity once either
        # input has more than 25 rows. The torch documentation states
        # that the switch happens but gives no rationale for it, and
        # does not mention its two consequences.
        #
        # The identity loses accuracy to cancellation, and because the
        # switch depends on the row count, the same pair of points gets
        # different distances according to how many points accompany
        # them. Measured on 37 points in 2D after scaling by a
        # lengthscale, a 7-row block disagreed with the matching rows of
        # the full call by 2.1e-08 in float64; the direct formula agrees
        # bit for bit. Blockwise kernel evaluation depends on that
        # consistency -- the pivoted Cholesky column operator and the
        # matrix-free matvec operator both assemble a matrix from pieces
        # and use it in place of the whole.
        #
        # Nor is the default faster for the shapes this library uses.
        # Measured on CPU, the direct path ran at 0.6x the default at
        # n=5000 in 2D and 0.5x at n=2000, with a 1.7x penalty only at
        # n=500 in 10D and parity by n=5000. The mm identity needs a
        # high enough dimension for its GEMM to repay the extra norm
        # passes, and spatial coordinates are 1-3D.
        return torch.cdist(XA, XB, p, compute_mode="donot_use_mm_for_euclid_dist")

    @staticmethod
    def tril(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.tril(array, diagonal=k)

    @staticmethod
    def triu(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.triu(array, diagonal=k)

    @staticmethod
    def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.kron(a, b)

    @staticmethod
    def repeat(
        array: torch.Tensor, repeats: int, axis: Optional[int] = None
    ) -> torch.Tensor:
        if axis is None:
            return array.flatten().repeat_interleave(repeats)
        return array.repeat_interleave(repeats, dim=axis)

    @staticmethod
    def slogdet(
        array: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = torch.linalg.slogdet(array)
        return cast(torch.Tensor, result.sign), cast(torch.Tensor, result.logabsdet)

    @staticmethod
    def det(array: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, torch.linalg.det(array))

    @staticmethod
    def mean(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return _numpy_reduction(
            lambda x, dims, keep: torch.mean(x, dim=dims, keepdim=keep),
            array,
            axis,
            keepdims,
        )

    @staticmethod
    def var(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> torch.Tensor:
        # torch.var uses correction (Bessel's correction) instead of ddof
        # correction=0 gives population variance, correction=1 gives sample variance
        return _numpy_reduction(
            lambda x, dims, keep: _variance(x, dims, keep, ddof),
            array,
            axis,
            keepdims,
        )

    @staticmethod
    def std(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> torch.Tensor:
        # torch.std uses correction (Bessel's correction) instead of ddof
        # correction=0 gives population std, correction=1 gives sample std
        return _numpy_reduction(
            lambda x, dims, keep: (
                torch.std(x, dim=dims, keepdim=keep, correction=ddof)
                if not x.is_complex()
                else torch.sqrt(_variance(x, dims, keep, ddof))
            ),
            array,
            axis,
            keepdims,
        )

    @staticmethod
    def cov(
        array: torch.Tensor,
        rowvar: bool = True,
        ddof: int = 1,
    ) -> torch.Tensor:
        # Written out: torch.cov checks its weights against the data, which
        # does not compile under fullgraph=True. Always 2-D, as NumPy's
        # result for one variable is not.
        if array.ndim == 1:
            array = array[None, :]
        elif not rowvar:
            array = array.T
        centered = array - torch.mean(array, dim=1, keepdim=True)
        return (centered @ centered.mH) / (array.shape[1] - ddof)

    @staticmethod
    def eigh(array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = torch.linalg.eigh(array)
        return (
            cast(torch.Tensor, result.eigenvalues),
            cast(torch.Tensor, result.eigenvectors),
        )

    @staticmethod
    def svd(
        array: torch.Tensor, full_matrices: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = torch.linalg.svd(array, full_matrices=full_matrices)
        return (
            cast(torch.Tensor, result.U),
            cast(torch.Tensor, result.S),
            cast(torch.Tensor, result.Vh),
        )

    @staticmethod
    def split(
        array: torch.Tensor,
        indices_or_sections: torch.Tensor,
        axis: int = 0,
    ) -> List[torch.Tensor]:
        # torch.tensor_split takes indices directly, but only from a CPU
        # tensor; on the CPU .cpu() returns the tensor itself.
        indices = indices_or_sections
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu()
        return list(torch.tensor_split(array, indices, dim=axis))

    @overload
    @staticmethod
    def where(condition: torch.Tensor) -> Tuple[torch.Tensor, ...]: ...

    @overload
    @staticmethod
    def where(
        condition: torch.Tensor,
        x: Union[torch.Tensor, float],
        y: Union[torch.Tensor, float],
    ) -> torch.Tensor: ...

    @staticmethod
    def where(
        condition: torch.Tensor,
        x: Union[torch.Tensor, float, None] = None,
        y: Union[torch.Tensor, float, None] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if x is None and y is None:
            # tuple() normalizes the return type across torch stub versions
            # (some type single-arg where() as list, others as tuple)
            return tuple(torch.where(condition))
        if x is not None and y is not None:
            return torch.where(condition, x, y)
        raise ValueError("Must provide both x and y or neither.")

    @staticmethod
    def ones_like(array: torch.Tensor, dtype: Optional[Any] = None) -> torch.Tensor:
        return torch.ones_like(array, dtype=dtype)

    @staticmethod
    def zeros_like(array: torch.Tensor, dtype: Optional[Any] = None) -> torch.Tensor:
        return torch.zeros_like(array, dtype=dtype)

    def default_dtype(self) -> Any:
        return self._dtype

    @staticmethod
    def maximum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x1, x2)

    @staticmethod
    def minimum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x1, x2)

    @staticmethod
    def qr(
        array: torch.Tensor,
        mode: Literal["reduced", "complete"] = "reduced",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = torch.linalg.qr(array, mode=mode)
        return cast(torch.Tensor, result.Q), cast(torch.Tensor, result.R)

    @staticmethod
    def lu(
        array: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LU factorization with partial pivoting.

        Returns P, L, U such that A = P @ L @ U.

        Parameters
        ----------
        array : torch.Tensor
            Matrix to factorize.

        Returns
        -------
        P : torch.Tensor
            Permutation matrix.
        L : torch.Tensor
            Lower triangular matrix with unit diagonal.
        U : torch.Tensor
            Upper triangular matrix.
        """
        result = torch.linalg.lu(array)
        return (
            cast(torch.Tensor, result.P),
            cast(torch.Tensor, result.L),
            cast(torch.Tensor, result.U),
        )

    @staticmethod
    def lu_factor(
        array: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # lu_factor_ex tolerates singular matrices (like DAE mass matrices)
        LU, pivots, _info = torch.linalg.lu_factor_ex(array)
        # The contract is NumPy's: 0-based pivots in the index dtype, not
        # LAPACK's 1-based int32 that torch returns.
        return cast(torch.Tensor, LU), (pivots - 1).to(torch.int64)

    @staticmethod
    def lu_solve(
        LU: torch.Tensor,
        pivots: torch.Tensor,
        rhs: torch.Tensor,
        transpose: bool = False,
    ) -> torch.Tensor:
        rhs_2d = rhs[:, None] if rhs.ndim == 1 else rhs
        # Back from the protocol's 0-based index pivots to LAPACK's.
        lapack_pivots = (pivots + 1).to(torch.int32)
        if transpose:
            # torch's adjoint solve is with A^H; the protocol's is with A^T.
            # They agree for real A; in general A^T x = b iff
            # A^H conj(x) = conj(b). conj is a free view for real tensors.
            result = torch.conj(
                torch.linalg.lu_solve(
                    LU, lapack_pivots, torch.conj(rhs_2d), adjoint=True
                )
            )
        else:
            result = torch.linalg.lu_solve(LU, lapack_pivots, rhs_2d)
        return result[:, 0] if rhs.ndim == 1 else result

    @staticmethod
    def index_update(
        array: torch.Tensor,
        index: IndexLike[torch.Tensor],
        value: Any,
    ) -> torch.Tensor:
        """Return a copy of array with array[index] = value.

        This provides a functional interface for array updates that
        is compatible with autograd frameworks.

        Parameters
        ----------
        array : torch.Tensor
            Tensor to update.
        index : int, slice, index array, or tuple of those
            Any index expression valid in ``array[index] = value``.
        value : Any
            Value to set at index.

        Returns
        -------
        torch.Tensor
            Copy of tensor with updated value.
        """
        result = array.clone()
        result[index] = value
        return result

    @staticmethod
    def index_add(
        array: torch.Tensor, index: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Return a copy of array with ``values[k]`` added at
        ``array[index[k]]`` along the first axis, repeated indices
        accumulating (out-of-place ``Tensor.index_add``)."""
        check_index_add_shapes(tuple(index.shape), tuple(values.shape))
        return array.index_add(0, index, values)

    @staticmethod
    def block(blocks: List[List[torch.Tensor]]) -> torch.Tensor:
        """Create a block matrix from nested list of tensors."""
        rows = [torch.hstack(row) for row in blocks]
        return torch.vstack(rows)

    @staticmethod
    def array_type() -> type:
        """Return the array type class for this backend."""
        return torch.Tensor

    def tril_indices(self, n: int, k: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return indices for lower-triangular part of an (n, n) array."""
        indices = torch.tril_indices(n, n, offset=k, device=self._device)
        return (indices[0], indices[1])

    @staticmethod
    def multidot(arrays: List[torch.Tensor]) -> torch.Tensor:
        """Compute the dot product of two or more arrays in a single call."""
        return cast(torch.Tensor, torch.linalg.multi_dot(arrays))
