"""Resolve a function's Derivatives bundle for derivative checking.

The DerivativeChecker validates whatever a function declares through its
``derivatives()`` accessor. Objects without the accessor are rejected:
capability is declared through the bundle (absence is ``None``), never
discovered by probing public methods.
"""

from __future__ import annotations

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.interface.functions.protocols.objective import ObjectiveProtocol
from pyapprox.util.backends.protocols import Array


def resolve_bundle(obj: FunctionProtocol[Array]) -> Derivatives[Array]:
    """Return ``obj``'s Derivatives bundle.

    Parameters
    ----------
    obj : FunctionProtocol[Array]
        Function whose derivatives are to be checked. Must expose a
        ``derivatives()`` accessor returning a ``Derivatives`` bundle.

    Raises
    ------
    TypeError
        If ``obj`` lacks the base function shape, lacks the
        ``derivatives()`` accessor, or the accessor returns something
        other than a ``Derivatives`` bundle.
    """
    if not isinstance(obj, FunctionProtocol):
        raise TypeError(
            "derivative checking requires an object with bkd/nvars/nqoi/"
            f"__call__ (FunctionProtocol); got {type(obj).__name__}. "
            "Kernels and other non-function objects have their own "
            "derivative accessors and do not belong here."
        )
    if not isinstance(obj, ObjectiveProtocol):
        raise TypeError(
            f"{type(obj).__name__} does not expose a derivatives() "
            "accessor. Capability is declared through the Derivatives "
            "bundle; public derivative methods are never harvested."
        )
    bundle = obj.derivatives()
    if not isinstance(bundle, Derivatives):
        raise TypeError(
            f"{type(obj).__name__}.derivatives() must return a "
            f"Derivatives bundle; got {type(bundle).__name__}"
        )
    return bundle
