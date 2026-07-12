"""Bundle-driven numpy boundary adapter.

Replaces the four-class ``NumpyFunction*Wrapper`` ladder and its
isinstance factory with ONE adapter constructed from an evaluation object
and its ``Derivatives`` bundle. Capability is captured once at
construction into always-present Optional accessors; converting
numpy <-> backend arrays (and coercing scipy's int8 probe vectors to
double) happens inside the returned closures.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Optional

from numpy.typing import NDArray

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend

NumpyArray = NDArray[Any]
NumpyFn = Callable[[NumpyArray], NumpyArray]
NumpyHVPFn = Callable[[NumpyArray, NumpyArray], NumpyArray]
NumpyWHVPFn = Callable[[NumpyArray, NumpyArray, NumpyArray], NumpyArray]


class NumpyDerivativesAdapter(Generic[Array]):
    """Numpy-facing view of a function and its Derivatives bundle.

    Derivative accessors return Optional numpy-callables (2D sample
    convention, matching the backend shapes documented in
    :mod:`pyapprox.interface.functions.derivatives`); ``None`` means the
    capability is absent and the consumer decides what to do (e.g. hand
    scipy ``jac=None`` for its own finite differencing).

    ``resolved`` second-order forms use the wrapped function's OWN nqoi.
    """

    def __init__(
        self,
        function: FunctionProtocol[Array],
        derivatives: Derivatives[Array],
        sample_ndim: int = 2,
    ) -> None:
        if not isinstance(function, FunctionProtocol):
            raise TypeError(
                "function must satisfy FunctionProtocol, got "
                f"{type(function).__name__}"
            )
        if not isinstance(derivatives, Derivatives):
            raise TypeError(
                "derivatives must be a Derivatives bundle, got "
                f"{type(derivatives).__name__}"
            )
        self._bkd = function.bkd()
        self._function = function
        # PyApprox assumes samples are always 2D but numpy functions, e.g.
        # from scipy, may only use 1D arrays. sample_ndim is the size of
        # the array the numpy function passes to this adapter.
        self._sample_ndim = sample_ndim

        # capture-narrow each capability ONCE; the numpy-facing forms are
        # BOUND METHODS (not closures) so the adapter stays picklable,
        # e.g. for multiprocessing
        self._bundle_jacobian: Optional[Callable[[Array], Array]] = (
            derivatives.jacobian
        )
        self._bundle_hvp: Optional[Callable[[Array, Array], Array]] = (
            derivatives.resolved_hvp(function.nqoi(), self._bkd)
        )
        self._bundle_whvp: Optional[
            Callable[[Array, Array, Array], Array]
        ] = derivatives.resolved_whvp(function.nqoi())

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._function.nvars()

    def nqoi(self) -> int:
        return self._function.nqoi()

    def _convert_samples_from_numpy(self, samples: NumpyArray) -> Array:
        if self._sample_ndim == 2:
            return self._bkd.asarray(samples)
        return self._bkd.asarray(samples[:, None])

    def __call__(self, samples: NumpyArray) -> NumpyArray:
        return self._bkd.to_numpy(
            self._function(self._convert_samples_from_numpy(samples))
        )

    def jacobian(self) -> Optional[NumpyFn]:
        """Numpy jacobian ``(nvars, 1) -> (nqoi, nvars)``, or None."""
        if self._bundle_jacobian is None:
            return None
        return self._numpy_jacobian

    def hvp(self) -> Optional[NumpyHVPFn]:
        """Numpy hvp ``(sample, vec) -> (nvars, 1)`` (resolved), or None."""
        if self._bundle_hvp is None:
            return None
        return self._numpy_hvp

    def whvp(self) -> Optional[NumpyWHVPFn]:
        """Numpy whvp ``(sample, vec, weights) -> (nvars, 1)`` (resolved),
        or None."""
        if self._bundle_whvp is None:
            return None
        return self._numpy_whvp

    def _numpy_jacobian(self, sample: NumpyArray) -> NumpyArray:
        bundle_jacobian = self._bundle_jacobian
        if bundle_jacobian is None:
            raise RuntimeError("jacobian capability is absent")
        return self._bkd.to_numpy(
            bundle_jacobian(self._convert_samples_from_numpy(sample))
        )

    def _numpy_hvp(self, sample: NumpyArray, vec: NumpyArray) -> NumpyArray:
        bundle_hvp = self._bundle_hvp
        if bundle_hvp is None:
            raise RuntimeError("hvp capability is absent")
        # Ensure vec is double - scipy's LinearOperator may probe with int8
        bkd_vec = self._bkd.asarray(vec, dtype=self._bkd.double_dtype())
        return self._bkd.to_numpy(
            bundle_hvp(self._convert_samples_from_numpy(sample), bkd_vec)
        )

    def _numpy_whvp(
        self, sample: NumpyArray, vec: NumpyArray, weights: NumpyArray
    ) -> NumpyArray:
        bundle_whvp = self._bundle_whvp
        if bundle_whvp is None:
            raise RuntimeError("whvp capability is absent")
        # Ensure vec and weights are double - scipy may probe with int8
        bkd_vec = self._bkd.asarray(vec, dtype=self._bkd.double_dtype())
        bkd_weights = self._bkd.asarray(
            weights, dtype=self._bkd.double_dtype()
        )
        return self._bkd.to_numpy(
            bundle_whvp(
                self._convert_samples_from_numpy(sample),
                bkd_vec,
                bkd_weights,
            )
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(function={repr(self._function)})"
