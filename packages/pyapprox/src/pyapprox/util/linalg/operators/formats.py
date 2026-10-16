r"""Format witnesses that need no linear-algebra library.

An operator's :meth:`format` returns a *format witness*: a value whose
type says how the operator is stored, and which, for a stored format,
gives access to the storage. A direct solver declares the format it
needs in its signature, so pairing it with an operator of another
format is a static type error. Stored formats belong to the stack that
owns their storage; only the storage-free format lives here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixFree:
    """The format of an operator known only through its action.

    It has no storage, so nothing that needs a matrix can be paired with
    an operator of this format.
    """
