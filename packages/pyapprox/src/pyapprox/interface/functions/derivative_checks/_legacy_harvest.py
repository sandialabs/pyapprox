"""TEMPORARY EXEMPTION — deleted at the end of the staged migration.

Diagnostic-only public-method harvest for objects that have not yet been
migrated to the ``Derivatives`` bundle (pde/ode/probability/leja/
affine-base/sparsegrids, retired module-by-module in Stage B of the
derivatives refactor). This module is the ONLY file permitted
capability-``getattr``; the repo grep gates exclude it by name, exactly
as they excluded the deleted ``legacy_adapter.py``.

The DerivativeChecker prefers ``derivatives()`` when an object provides
it; this fallback keeps not-yet-migrated modules checkable in the
meantime and WARNS on every harvest so stragglers stay visible. Once the
last Stage B module lands, this file is deleted and the checker requires
``derivatives()``.
"""

from __future__ import annotations

import warnings

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array


def resolve_bundle(obj: FunctionProtocol[Array]) -> Derivatives[Array]:
    """Return ``obj``'s bundle, harvesting public methods as a fallback.

    Objects exposing ``derivatives()`` return their bundle unchanged
    (the required path). Anything else gets a bundle harvested from its
    public derivative-named methods, with a migration warning naming the
    class.
    """
    if not isinstance(obj, FunctionProtocol):
        raise TypeError(
            "derivative checking requires an object with bkd/nvars/nqoi/"
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
        return bundle
    warnings.warn(
        f"{type(obj).__name__} has no derivatives() accessor; harvesting "
        "public derivative methods (legacy path, removed when the staged "
        "migration completes)",
        UserWarning,
        stacklevel=3,
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
    )
