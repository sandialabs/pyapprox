"""GP posterior mean dimension reducer.

Wraps MarginalizedGP.predict_mean() as a DimensionReducerProtocol,
enabling the GP posterior mean to be visualized with PairPlotter and
other dimension-reduction tools.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.gaussianprocess.protocols import (
    PredictiveGPProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.integrals import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.marginalization import (
    MarginalizedGP,
)
from pyapprox.typing.interface.functions.marginalize import ReducedFunction


class GPMeanDimensionReducer(Generic[Array]):
    """Reduces GP posterior mean via probability-measure marginalization.

    Satisfies ``DimensionReducerProtocol``. Mean only (nqoi=1).

    Given a fitted GP with a separable (product) kernel, creates
    ``MarginalizedGP`` instances for each requested subset of kept
    variables, and wraps ``predict_mean()`` in a ``ReducedFunction``
    with output shape ``(1, nsamples)``.

    Parameters
    ----------
    gp : PredictiveGPProtocol[Array]
        Fitted GP with separable (product) kernel.
    integral_calculator : SeparableKernelIntegralCalculator[Array]
        Calculator providing kernel integrals and quadrature.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        gp: PredictiveGPProtocol[Array],
        integral_calculator: SeparableKernelIntegralCalculator[Array],
        bkd: Backend[Array],
    ):
        self._gp = gp
        self._calc = integral_calculator
        self._bkd = bkd
        self._nvars = len(integral_calculator._kernels_1d)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables in the original GP."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of quantities of interest (always 1)."""
        return 1

    def reduce(
        self, keep_indices: List[int]
    ) -> ReducedFunction[Array]:
        """Reduce to the specified variables via GP marginalization.

        Parameters
        ----------
        keep_indices : List[int]
            Indices of variables to keep (0-based).

        Returns
        -------
        ReducedFunction[Array]
            A function with ``nvars = len(keep_indices)`` and ``nqoi = 1``.
        """
        bkd = self._bkd
        marg_gp = MarginalizedGP(self._gp, self._calc, keep_indices)

        def eval_fn(samples: Array) -> Array:
            # samples: (n_keep, nsamples)
            mean = marg_gp.predict_mean(samples)  # (nsamples,)
            return bkd.reshape(mean, (1, -1))  # (1, nsamples)

        return ReducedFunction(len(keep_indices), 1, eval_fn, bkd)
