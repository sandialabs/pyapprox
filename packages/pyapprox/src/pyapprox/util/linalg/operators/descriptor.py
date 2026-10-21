r"""Space descriptors: the runtime description of the space a vector lives in.

A :class:`SpaceDescriptor` is a frozen value naming everything generic
code may know about a vector without looking inside it: which space it belongs
to, its global size, its scalar type, and a version that is bumped
whenever the space is rebuilt (for example a resize or a
redistribution). Two vectors, or a vector and an operator, can be
combined only when their descriptors match.

A space is identified by its name, not only by its size: two spaces of
equal dimension are still different spaces, so a vector of one is never
silently accepted as a vector of the other. An operator between two
spaces records both, as its domain and range.

Vectors themselves are opaque handles owned by a linear-algebra stack,
so a descriptor cannot always be recovered from a vector. Descriptors are
therefore compared where objects that carry one meet (an operator and
the space it is applied in, a solver and the operator it factors), with
:meth:`SpaceDescriptor.require_same`.

Distributed and in-place linear-algebra libraries describe a vector's
space the same way: a runtime map or layout object, of global size,
scalar type and distribution, that is compared before vectors are
combined.
"""

from dataclasses import dataclass, fields

_SUPPORTED_DTYPES = ("float32", "float64", "complex64", "complex128")


class SpaceMismatchError(ValueError):
    """Raised when objects belonging to different spaces are combined."""


def check_column(k: int, ncols: int) -> None:
    """Refuse a column index outside ``[0, ncols)``, negative included.

    Raises
    ------
    IndexError
        If ``k`` does not index a column of a block of ``ncols`` vectors.
    """
    if not 0 <= k < ncols:
        raise IndexError(f"column {k} is outside a block of {ncols} vectors")


@dataclass(frozen=True)
class SpaceDescriptor:
    """The global description of a vector space.

    Parameters
    ----------
    name : str
        Identifies the space. Spaces of equal size but different names
        are different spaces.
    size : int
        Global number of rows of a vector in this space. Not the size of
        any one process's portion.
    dtype : str
        Scalar type name: one of ``"float32"``, ``"float64"``,
        ``"complex64"``, ``"complex128"``. A name rather than a library
        dtype object, so a descriptor means the same to every stack.
    version : int
        Incremented whenever the space is rebuilt. Vectors, operators and
        factorizations built for one version are invalid for another.

    Raises
    ------
    ValueError
        If ``name`` is empty, ``size`` or ``version`` is negative, or
        ``dtype`` is not supported.
    """

    name: str
    size: int
    dtype: str = "float64"
    version: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be nonempty")
        if self.size < 0:
            raise ValueError(f"size must be nonnegative, got {self.size}")
        if self.version < 0:
            raise ValueError(f"version must be nonnegative, got {self.version}")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {_SUPPORTED_DTYPES}, got {self.dtype!r}"
            )

    def require_same(self, other: "SpaceDescriptor", what: str) -> None:
        """Raise unless ``other`` describes the same space as ``self``.

        Parameters
        ----------
        other : SpaceDescriptor
            The descriptor received.
        what : str
            What ``other`` belongs to, named in the error message.

        Raises
        ------
        SpaceMismatchError
            Naming every field that differs, with both values.
        """
        if other == self:
            return
        differences = ", ".join(
            f"{field.name} {getattr(self, field.name)!r} != "
            f"{getattr(other, field.name)!r}"
            for field in fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        )
        raise SpaceMismatchError(
            f"{what} describes a different space (expected vs received: {differences})"
        )
