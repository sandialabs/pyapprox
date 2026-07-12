"""LEGACY SHIM — MUST BE DELETED IN PHASE 5 of the derivatives refactor.

``as_derivatives(obj)`` gives optimizer consumers a single door to an
object's derivative capabilities during the migration:

- objects already exposing ``derivatives()`` return their bundle (fast
  path), with a migration-era warning when a public derivative method
  exists but is not declared in the bundle;
- legacy objects (capability expressed by attribute presence) get a bundle
  harvested from their attributes.

This is the ONLY module permitted to use ``getattr`` for capability
discovery. Once every producer in the bind() data flow exposes
``derivatives()``, consumers switch to calling it directly and this file
is deleted (flip the legacy branch to raise first, run the full suite to
prove no producer still needs it).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from pyapprox.interface.functions.derivatives import (
    Derivatives,
    InexactSuite,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.interface.functions.protocols.objective import Function
from pyapprox.util.backends.protocols import Array

_CAPABILITY_FIELDS = (
    "jacobian",
    "jacobian_batch",
    "jvp",
    "hvp",
    "whvp",
    "hessian",
    "hessian_batch",
    "hvp_batch",
    "whvp_batch",
)


def as_derivatives(obj: FunctionProtocol[Array]) -> Derivatives[Array]:
    """Return ``obj``'s Derivatives bundle, harvesting legacy attributes.

    Requires the base function shape (``bkd``/``nvars``/``nqoi``/
    ``__call__``) so a wrong-kind object (e.g. a kernel, whose
    ``jacobian(X1, X2)`` has a different signature) fails loudly here
    instead of mis-broadcasting mid-minimize.
    """
    if not isinstance(obj, FunctionProtocol):
        raise TypeError(
            "as_derivatives requires an object with bkd/nvars/nqoi/"
            f"__call__ (FunctionProtocol); got {type(obj).__name__}. "
            "Kernels and other non-function objects have their own "
            "derivative accessors and do not belong here."
        )
    accessor = getattr(obj, "derivatives", None)
    if callable(accessor):
        bundle = accessor()
        if not isinstance(bundle, Derivatives):
            raise TypeError(
                f"{type(obj).__name__}.derivatives() must return a "
                f"Derivatives bundle; got {type(bundle).__name__}"
            )
        _warn_undeclared_methods(obj, bundle)
        return bundle
    return _harvest_legacy(obj)


def _warn_undeclared_methods(obj: object, bundle: Derivatives[Any]) -> None:
    """Migration safety net: a defined-but-undeclared public derivative
    method usually means a producer was only half-migrated. Post-migration
    omission can be deliberate, which is why this warning lives in the
    shim and sunsets with it."""
    for name in _CAPABILITY_FIELDS:
        if getattr(bundle, name) is not None:
            continue
        implementation = getattr(type(obj), name, None)
        if implementation is None or not callable(implementation):
            continue
        if implementation is getattr(Function, name, None):
            # the Function sugar's delegating method, not a capability
            continue
        warnings.warn(
            f"{type(obj).__name__} defines '{name}' but its Derivatives "
            "bundle does not declare it; if the capability is intended, "
            "add it to the bundle",
            UserWarning,
            stacklevel=3,
        )


def _harvest_legacy(obj: FunctionProtocol[Array]) -> Derivatives[Array]:
    inexact: Optional[InexactSuite[Array]] = None
    # lazy import: inexact protocols live above this module's layer
    from pyapprox.optimization.minimize.inexact.protocols import (
        InexactDifferentiable,
        InexactEvaluable,
    )

    if isinstance(obj, InexactEvaluable):
        inexact = InexactSuite(
            value=obj.inexact_value,
            jacobian=obj.inexact_jacobian
            if isinstance(obj, InexactDifferentiable)
            else None,
        )
    return Derivatives(
        jacobian=getattr(obj, "jacobian", None),
        jacobian_batch=getattr(obj, "jacobian_batch", None),
        jvp=getattr(obj, "jvp", None),
        hvp=getattr(obj, "hvp", None),
        whvp=getattr(obj, "whvp", None),
        hessian=getattr(obj, "hessian", None),
        hessian_batch=getattr(obj, "hessian_batch", None),
        hvp_batch=getattr(obj, "hvp_batch", None),
        whvp_batch=getattr(obj, "whvp_batch", None),
        inexact=inexact,
    )
