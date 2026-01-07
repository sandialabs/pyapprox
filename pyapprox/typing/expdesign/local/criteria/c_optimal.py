"""
C-optimal design criterion.

C-optimal designs minimize the variance of a linear combination of parameters:
    J(w) = c^T @ Cov(w) @ c = c^T @ M1^{-1} @ M0 @ M1^{-1} @ c

This is equivalent to x^T @ M0 @ x where M1 @ x = c.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.local.protocols import DesignMatricesProtocol
from pyapprox.typing.expdesign.local.adjoint import AdjointModel

from .base import LocalOEDCriterionBase


class COptimalCriterion(LocalOEDCriterionBase[Array], Generic[Array]):
    """
    C-optimal design criterion.

    Minimizes the variance of a linear combination c^T @ theta, where
    theta are the regression parameters. Uses the adjoint method for
    efficient gradient and HVP computation.

    Parameters
    ----------
    design_matrices : DesignMatricesProtocol[Array]
        Object that computes M0, M1 matrices from design weights.
    vec : Array
        Linear combination vector. Shape: (ndesign_vars,)
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The objective is:
        J(w) = c^T @ Cov(w) @ c
             = c^T @ M1(w)^{-1} @ M0(w) @ M1(w)^{-1} @ c
             = x(w)^T @ M0(w) @ x(w)

    where x(w) = M1(w)^{-1} @ c.

    For homoscedastic noise, this requires creating design matrices with
    explicit noise_mult = ones, since the adjoint method needs M0.
    """

    def __init__(
        self,
        design_matrices: DesignMatricesProtocol[Array],
        vec: Array,
        bkd: Backend[Array],
    ) -> None:
        super().__init__(design_matrices, bkd)
        self._vec = vec
        self._adjoint = AdjointModel(design_matrices, bkd)
        self._adjoint.set_vector(vec)

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate C-optimal criterion.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Criterion value. Shape: (1, 1)
        """
        return self._adjoint.value(design_weights)

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of C-optimal criterion.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nvars)
        """
        return self._adjoint.jacobian(design_weights)

    def hvp(self, design_weights: Array, vec: Array) -> Array:
        """
        Hessian-vector product.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)
        vec : Array
            Direction vector. Shape: (nvars, 1)

        Returns
        -------
        Array
            Hessian-vector product. Shape: (nvars, 1)
        """
        return self._adjoint.hvp(design_weights, vec)


class COptimalLeastSquaresCriterion(COptimalCriterion[Array], Generic[Array]):
    """
    C-optimal criterion for least squares regression.

    Alias for COptimalCriterion when used with LeastSquaresDesignMatrices.
    """

    pass


class COptimalQuantileCriterion(COptimalCriterion[Array], Generic[Array]):
    """
    C-optimal criterion for quantile regression.

    Uses the same formula as COptimalCriterion but with different
    design matrices (QuantileDesignMatrices).
    """

    pass
