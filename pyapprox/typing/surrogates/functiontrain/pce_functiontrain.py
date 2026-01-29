"""PCE-validated FunctionTrain wrapper and factory.

This module provides:
- PCEFunctionTrain: wrapper that validates orthonormal PCE structure
- create_uniform_pce_functiontrain: factory for rank-r FT with uniform PCE cores
"""

from typing import Generic, List, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.typing.surrogates.functiontrain.core import FunctionTrainCore
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


def create_uniform_pce_functiontrain(
    univariate_expansion_factory: BasisExpansionProtocol[Array],
    nvars: int,
    ranks: Sequence[int],
    bkd: Backend[Array],
    init_scale: float = 0.1,
) -> FunctionTrain[Array]:
    """Create a FunctionTrain with uniform PCE cores.

    Creates a rank-r FT where each core position (i, j) uses an independent
    copy of the same univariate expansion type. This structure is compatible
    with PCEFunctionTrain for statistics computation.

    Parameters
    ----------
    univariate_expansion_factory : BasisExpansionProtocol[Array]
        A univariate PCE expansion to use as a template. Each core position
        will get a fresh copy.
    nvars : int
        Number of input variables.
    ranks : Sequence[int]
        Interior ranks [r_1, r_2, ..., r_{d-1}]. Length must be nvars - 1.
        Boundary ranks are always 1.
    bkd : Backend[Array]
        Computational backend.
    init_scale : float
        Scale for random initialization of coefficients. Default 0.1.
        Small non-zero values help ALS converge. Use 0.0 for zero init.

    Returns
    -------
    FunctionTrain[Array]
        FT with uniform PCE cores, compatible with PCEFunctionTrain.

    Raises
    ------
    ValueError
        If ranks has wrong length or univariate expansion is not univariate.

    Examples
    --------
    >>> # Create a rank-2 FT for 3 variables
    >>> from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
    >>> from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
    >>> from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
    >>> from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
    >>> from pyapprox.typing.probability import UniformMarginal
    >>>
    >>> marginals = [UniformMarginal(0.0, 1.0, bkd)]
    >>> bases_1d = create_bases_1d(marginals, bkd)
    >>> indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
    >>> basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    >>> pce_template = BasisExpansion(basis, bkd, nqoi=1)
    >>>
    >>> ft = create_uniform_pce_functiontrain(pce_template, nvars=3, ranks=[2, 2], bkd=bkd)
    """
    import numpy as np

    if len(ranks) != nvars - 1:
        raise ValueError(
            f"ranks must have length nvars - 1 = {nvars - 1}, got {len(ranks)}"
        )
    if univariate_expansion_factory.nvars() != 1:
        raise ValueError(
            f"univariate_expansion_factory must have nvars=1, "
            f"got {univariate_expansion_factory.nvars()}"
        )

    # Full ranks including boundaries
    full_ranks = [1] + list(ranks) + [1]
    nterms = univariate_expansion_factory.nterms()
    cores: List[FunctionTrainCore[Array]] = []

    for k in range(nvars):
        r_left = full_ranks[k]
        r_right = full_ranks[k + 1]

        # Build 2D list of independent expansions
        basisexps: List[List[BasisExpansionProtocol[Array]]] = []
        for _ in range(r_left):
            row: List[BasisExpansionProtocol[Array]] = []
            for _ in range(r_right):
                # Create fresh copy with small random coefficients
                if init_scale > 0:
                    init_params = bkd.asarray(
                        np.random.randn(nterms) * init_scale
                    )
                else:
                    init_params = bkd.zeros((nterms,))
                pce = univariate_expansion_factory.with_params(init_params)
                row.append(pce)
            basisexps.append(row)

        core = FunctionTrainCore(basisexps, bkd)
        cores.append(core)

    return FunctionTrain(cores, bkd, nqoi=1)
