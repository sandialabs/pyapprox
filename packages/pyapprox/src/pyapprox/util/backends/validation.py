import inspect
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from pyapprox.util.backends.protocols import Backend


def reduction_axes(
    ndim: int, axis: Optional[Union[int, Tuple[int, ...]]]
) -> Tuple[int, ...]:
    """The axes a reduction over ``axis`` reduces, as NumPy reads it:
    ``None`` is every axis, an int one axis, a tuple those axes (``()``
    none). Negative axes are kept as given."""
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis,)
    return tuple(axis)


def check_orderable(method: str, is_complex: bool) -> None:
    """Check that ``method`` (an ordering: sort, max, argmax, ...) was
    given real values.

    Raises
    ------
    TypeError
        If the input is complex: complex values have no order.
    """
    if is_complex:
        raise TypeError(
            f"{method} needs real values; complex values have no order, so "
            "use abs() or the real part instead"
        )


def check_ordered_reduction(
    method: str,
    is_complex: bool,
    shape: Tuple[int, ...],
    axes: Tuple[int, ...],
) -> None:
    """Check that ``Backend.max``, ``min``, ``argmax`` or ``argmin`` can
    order the values it is given.

    Parameters
    ----------
    method : str
        The method, for the message.
    is_complex : bool
        Whether the input is complex.
    shape : Tuple[int, ...]
        The input's shape.
    axes : Tuple[int, ...]
        The axes reduced over (:func:`reduction_axes`).

    Raises
    ------
    TypeError
        If the input is complex: complex values have no order.
    ValueError
        If an axis reduced over has length zero.
    """
    check_orderable(method, is_complex)
    empty = [a for a in axes if shape[a] == 0]
    if empty:
        raise ValueError(
            f"{method} over an axis of length zero (axes {empty} of shape "
            f"{shape}) has no result"
        )


def check_to_bool_shape(shape: Tuple[int, ...]) -> None:
    """Check that ``Backend.to_bool`` was given a single element.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the array to convert; must hold exactly one element.

    Raises
    ------
    ValueError
        If the array holds no element or more than one.
    """
    size = 1
    for n in shape:
        size *= n
    if size != 1:
        raise ValueError(
            f"to_bool needs an array with one element, got shape {shape}; "
            "reduce it first with any or all"
        )


def check_scalar_conversion(
    method: str, shape: Tuple[int, ...], is_complex: bool
) -> None:
    """Check that ``Backend.to_float`` or ``to_int`` was given one real
    element.

    Raises
    ------
    ValueError
        If the array holds no element or more than one.
    TypeError
        If the array is complex: a real scalar would silently drop the
        imaginary part.
    """
    size = 1
    for n in shape:
        size *= n
    if size != 1:
        raise ValueError(f"{method} needs an array with one element, got shape {shape}")
    if is_complex:
        raise TypeError(
            f"{method} needs a real array; take the real part or abs() first"
        )


def check_index_add_shapes(
    index_shape: Tuple[int, ...], values_shape: Tuple[int, ...]
) -> None:
    """Check the shapes ``Backend.index_add`` accepts.

    Parameters
    ----------
    index_shape : Tuple[int, ...]
        Shape of the row indices; must be ``(m,)``.
    values_shape : Tuple[int, ...]
        Shape of the rows to add; must start with ``m``.

    Raises
    ------
    ValueError
        If the index is not 1-D or the row counts differ.
    """
    if len(index_shape) != 1:
        raise ValueError(f"index must be 1-D, got shape {index_shape}")
    if len(values_shape) == 0 or values_shape[0] != index_shape[0]:
        raise ValueError(
            f"values must have {index_shape[0]} rows, one per index, got "
            f"shape {values_shape}"
        )


def check_jit_kernel(kernel: Callable[..., Any]) -> None:
    """Check that ``Backend.jit`` was given a module-level function or a
    staticmethod.

    Parameters
    ----------
    kernel : Callable[..., Any]
        The function to compile.

    Raises
    ------
    TypeError
        If ``kernel`` is a lambda, a local function, a bound method, a
        ``functools.partial`` or any other callable object. Each is
        usually a new object per call, so it would be compiled again on
        every call, and the values it captures would be compiled in as
        constants, stale once they change.
    """
    qualname = getattr(kernel, "__qualname__", "")
    if (
        not inspect.isfunction(kernel)
        or kernel.__name__ == "<lambda>"
        or "<locals>" in qualname
    ):
        raise TypeError(
            f"jit needs a module-level function or a staticmethod taking "
            f"the backend as the keyword bkd, got {kernel!r}; pass what "
            f"changes as arguments and code-path choices through "
            f"static_argnames instead of capturing them in a lambda, local "
            f"function, bound method or partial"
        )


def validate_backend(obj: Any) -> None:
    """
    Validate that the given object is an instance of the Backend protocol.

    Parameters
    ----------
    obj : Any
        The object to validate.

    Raises
    ------
    TypeError
        If the object is not an instance of Backend.
    """
    if not isinstance(obj, Backend):
        raise TypeError(
            f"Invalid backend type: expected an instance of Backend, "
            f"got {type(obj).__name__}. Object details: {obj}"
        )


def validate_backends(backends: Sequence[Any]) -> None:
    """
    Validate that all backends in the sequence have the same
    class name and are valid backends.

    Parameters
    ----------
    backends : Sequence[Any]
        A sequence of backend objects or classes.

    Raises
    ------
    ValueError
        If the sequence is empty or the backends are inconsistent.
    TypeError
        If any backend is not a valid backend.
    """
    if len(backends) == 0:
        raise ValueError("The sequence of backends cannot be empty.")

    # Validate each backend
    for backend in backends:
        validate_backend(backend)

    # Extract class names for comparison
    class_names = [
        (type(backend).__name__ if not isinstance(backend, type) else backend.__name__)
        for backend in backends
    ]

    # Check if all class names are the same
    if len(set(class_names)) != 1:
        raise ValueError(
            "Inconsistent backends: expected all backends"
            " to have the same class name,"
            f" got {set(class_names)}."
        )
