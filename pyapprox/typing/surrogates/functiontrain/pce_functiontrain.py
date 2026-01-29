"""PCE-validated FunctionTrain wrapper.

This wrapper validates that a FunctionTrain uses orthonormal PCE
univariate expansions, enabling analytical statistics computation.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.typing.surrogates.functiontrain.pce_core import (
    PCEFunctionTrainCore,
)


class PCEFunctionTrain(Generic[Array]):
    """FunctionTrain wrapper that validates PCE structure.

    Provides access to PCE-specific core operations needed for
    analytical statistics computation.

    Parameters
    ----------
    ft : FunctionTrain[Array]
        FunctionTrain with orthonormal PCE univariate expansions.

    Raises
    ------
    TypeError
        If ft is not a FunctionTrain instance, or if any core's
        univariate expansions aren't compatible PCE.
    ValueError
        If nqoi != 1 or cores have inconsistent structure.

    Warning
    -------
    Assumes all basis expansions use orthonormal polynomials. Results are
    mathematically incorrect for non-orthonormal bases.

    Notes
    -----
    Currently only supports nqoi=1.

    FunctionTrain structures using mixed nterms within a core (e.g., additive
    structure with ConstantExpansion) are not supported. Use uniform PCE
    cores instead.
    """

    def __init__(self, ft: FunctionTrain[Array]) -> None:
        if not isinstance(ft, FunctionTrain):
            raise TypeError(
                f"Expected FunctionTrain, got {type(ft).__name__}"
            )
        if ft.nqoi() != 1:
            raise ValueError(
                f"PCEFunctionTrain only supports nqoi=1, got nqoi={ft.nqoi()}"
            )
        self._ft = ft
        self._bkd = ft.bkd()
        # This validates all cores (raises if incompatible)
        self._pce_cores = [PCEFunctionTrainCore(c) for c in ft.cores()]

    def ft(self) -> FunctionTrain[Array]:
        """Access underlying FunctionTrain."""
        return self._ft

    def nvars(self) -> int:
        """Number of input variables (same as number of cores)."""
        return self._ft.nvars()

    def nqoi(self) -> int:
        """Number of quantities of interest."""
        return self._ft.nqoi()

    def bkd(self) -> Backend[Array]:
        """Computational backend."""
        return self._bkd

    def pce_cores(self) -> List[PCEFunctionTrainCore[Array]]:
        """Access PCE-aware cores for statistics computation."""
        return self._pce_cores

    def __call__(self, samples: Array) -> Array:
        """Evaluate. Delegates to underlying FunctionTrain.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Output values. Shape: (nqoi, nsamples)
        """
        return self._ft(samples)

    def __repr__(self) -> str:
        return (
            f"PCEFunctionTrain(nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"ncores={len(self._pce_cores)})"
        )
