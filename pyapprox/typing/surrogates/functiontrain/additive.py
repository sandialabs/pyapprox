"""Additive FunctionTrain factory - creates f(x) = Σ f_i(x_i) structure."""

from typing import List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.typing.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.typing.surrogates.functiontrain.functiontrain import FunctionTrain


class ConstantExpansion:
    """A constant function that returns a fixed value.

    This is used for the structural constants (0 and 1) in the additive
    FunctionTrain tensor train structure.

    Parameters
    ----------
    value : float
        The constant value to return.
    bkd : Backend[Array]
        Computational backend.
    nqoi : int
        Number of quantities of interest.
    """

    def __init__(self, value: float, bkd: Backend[Array], nqoi: int = 1):
        self._value = value
        self._bkd = bkd
        self._nqoi = nqoi
        # Single constant coefficient
        self._coef = bkd.full((1, nqoi), value)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables (always 1 for univariate)."""
        return 1

    def nterms(self) -> int:
        """Return number of terms (always 1 for constant)."""
        return 1

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return self._nqoi

    def nparams(self) -> int:
        """Return number of parameters (0 - constants are fixed)."""
        return 0  # Constants have no trainable parameters

    def get_coefficients(self) -> Array:
        """Return coefficients. Shape: (1, nqoi)."""
        return self._coef

    def set_coefficients(self, coef: Array) -> None:
        """Constants cannot be modified."""
        raise ValueError("ConstantExpansion coefficients are fixed")

    def __call__(self, samples: Array) -> Array:
        """Evaluate constant at samples.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Constant values. Shape: (nqoi, nsamples)
        """
        nsamples = samples.shape[1]
        return self._bkd.full((self._nqoi, nsamples), self._value)

    def basis_matrix(self, samples: Array) -> Array:
        """Return basis matrix (all ones for constant).

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Basis matrix. Shape: (nsamples, 1)
        """
        nsamples = samples.shape[1]
        return self._bkd.ones((nsamples, 1))

    def with_params(self, params: Array) -> "ConstantExpansion":
        """Return same constant (no parameters to change)."""
        return ConstantExpansion(self._value, self._bkd, self._nqoi)

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """Jacobian w.r.t. parameters (empty since no trainable params).

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Empty Jacobian. Shape: (nsamples, nqoi, 0)
        """
        nsamples = samples.shape[1]
        return self._bkd.zeros((nsamples, self._nqoi, 0))


def create_additive_functiontrain(
    univariate_bases: List[BasisExpansionProtocol[Array]],
    bkd: Backend[Array],
    nqoi: int = 1,
) -> FunctionTrain[Array]:
    """Create a FunctionTrain with additive structure.

    Creates f(x) = f_1(x_1) + f_2(x_2) + ... + f_d(x_d)

    The tensor train structure for additive functions:
    - First core: [[f_1, 1]] with ranks (1, 2)
    - Middle cores: [[1, 0], [f_i, 1]] with ranks (2, 2)
    - Last core: [[1], [f_d]] with ranks (2, 1)

    This ensures the contraction gives the sum of univariate functions.

    Parameters
    ----------
    univariate_bases : List[BasisExpansionProtocol]
        List of univariate basis expansions, one per input variable.
        Each should have nvars() == 1.
    bkd : Backend[Array]
        Computational backend.
    nqoi : int
        Number of quantities of interest. Default: 1.

    Returns
    -------
    FunctionTrain
        FunctionTrain with additive structure.

    Raises
    ------
    ValueError
        If fewer than 2 univariate bases provided (need at least 2 for
        additive structure).
    ValueError
        If any univariate basis does not have nvars() == 1.
    """
    nvars = len(univariate_bases)
    if nvars < 2:
        raise ValueError(
            f"Additive FunctionTrain requires at least 2 variables, got {nvars}"
        )

    # Validate univariate functions
    for ii, basis in enumerate(univariate_bases):
        if basis.nvars() != 1:
            raise ValueError(
                f"univariate_bases[{ii}] must have nvars=1, got {basis.nvars()}"
            )

    # Helper to create constant expansions
    def const(value: float) -> ConstantExpansion:
        return ConstantExpansion(value, bkd, nqoi)

    cores: List[FunctionTrainCore[Array]] = []

    # First core: [[f_1, 1]] with ranks (1, 2)
    cores.append(
        FunctionTrainCore(
            [[univariate_bases[0], const(1.0)]],
            bkd,
        )
    )

    # Middle cores: [[1, 0], [f_i, 1]] with ranks (2, 2)
    for ii in range(1, nvars - 1):
        cores.append(
            FunctionTrainCore(
                [
                    [const(1.0), const(0.0)],
                    [univariate_bases[ii], const(1.0)],
                ],
                bkd,
            )
        )

    # Last core: [[1], [f_d]] with ranks (2, 1)
    cores.append(
        FunctionTrainCore(
            [
                [const(1.0)],
                [univariate_bases[nvars - 1]],
            ],
            bkd,
        )
    )

    return FunctionTrain(cores, bkd, nqoi)
