"""Element DOF maps of skfem bases, checked and in state numbering."""

import numpy as np
from numpy.typing import NDArray

try:
    from skfem import Basis
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


def skfem_element_dofs(basis: Basis) -> NDArray[np.int64]:
    """Return each element's DOFs, one element per row.

    skfem stores the map as ``(ndofs_per_element, nelems)`` and is
    untyped, so its type and shape are checked here, once, rather than
    trusted by every caller.

    Parameters
    ----------
    basis : skfem.Basis
        The basis whose map to read.

    Returns
    -------
    NDArray[np.int64]
        Row ``e`` holds the DOFs whose basis functions are supported on
        element ``e``. Shape: (nelems, ndofs_per_element).

    Raises
    ------
    TypeError
        If the basis's map is not a 2-D integer array.
    """
    dofs = basis.element_dofs
    if (
        not isinstance(dofs, np.ndarray)
        or dofs.ndim != 2
        or not np.issubdtype(dofs.dtype, np.integer)
    ):
        raise TypeError(
            "expected the basis's element DOF map as a 2-D integer array, "
            f"got {type(dofs).__name__}"
        )
    result: NDArray[np.int64] = np.ascontiguousarray(dofs.T, dtype=np.int64)
    return result
