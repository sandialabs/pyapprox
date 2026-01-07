"""
I-optimal design criterion.

I-optimal designs minimize the integrated prediction variance:
    J(w) = integral phi(x)^T @ Cov(w) @ phi(x) dP(x)
         = trace(Cov(w) @ B)

where B = integral phi(x) @ phi(x)^T dP(x) is the integrated basis matrix.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.local.protocols import DesignMatricesProtocol
from pyapprox.typing.expdesign.local.adjoint import AdjointModel

from .base import LocalOEDCriterionBase


class IOptimalCriterion(LocalOEDCriterionBase[Array], Generic[Array]):
    """
    I-optimal design criterion.

    Minimizes the integrated prediction variance over a prediction domain.
    The integral is represented by prediction factors and optional weights.

    Parameters
    ----------
    design_matrices : DesignMatricesProtocol[Array]
        Object that computes M0, M1 matrices from design weights.
    pred_factors : Array
        Basis function values at prediction points. Shape: (npred_pts, ndesign_vars)
    bkd : Backend[Array]
        Computational backend.
    pred_weights : Array, optional
        Weights for prediction points. Shape: (npred_pts,)
        If None, uses uniform weights 1/npred_pts.

    Notes
    -----
    The objective is:
        J(w) = trace(Cov(w) @ B)
             = trace(M1^{-1} @ M0 @ M1^{-1} @ B)

    where B = sum_j w_j * phi_j @ phi_j^T (integrated prediction matrix).

    This can be written as sum over Cholesky factors:
        J(w) = sum_i L_i^T @ M1^{-1} @ M0 @ M1^{-1} @ L_i

    where B = L @ L^T is the Cholesky decomposition.
    """

    def __init__(
        self,
        design_matrices: DesignMatricesProtocol[Array],
        pred_factors: Array,
        bkd: Backend[Array],
        pred_weights: Optional[Array] = None,
    ) -> None:
        super().__init__(design_matrices, bkd)
        self._pred_factors = pred_factors
        self._setup_prediction_matrix(pred_factors, pred_weights)

    def _setup_prediction_matrix(
        self, pred_factors: Array, pred_weights: Optional[Array]
    ) -> None:
        """Set up the integrated prediction matrix B and its Cholesky factor."""
        npred_pts = pred_factors.shape[0]

        if pred_weights is None:
            pred_weights = self._bkd.full((npred_pts,), 1.0 / npred_pts)

        self._pred_weights = pred_weights

        # B = sum_j w_j * phi_j @ phi_j^T = pred_factors^T @ diag(w) @ pred_factors
        weighted_factors = pred_weights[:, None] * pred_factors
        self._B = pred_factors.T @ weighted_factors

        # Cholesky decomposition: B = L @ L^T
        self._L = self._bkd.cholesky(self._B)

        # Create adjoint models for each column of L
        ndesign_vars = self._design_matrices.ndesign_vars()
        self._adjoints = []
        for i in range(ndesign_vars):
            adjoint = AdjointModel(self._design_matrices, self._bkd)
            # Column of L (transposed Cholesky factor)
            vec = self._L[:, i]
            adjoint.set_vector(vec)
            self._adjoints.append(adjoint)

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate I-optimal criterion.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Criterion value. Shape: (1, 1)
        """
        val = self._bkd.zeros((1, 1))
        for adjoint in self._adjoints:
            val = val + adjoint.value(design_weights)
        return val

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of I-optimal criterion.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nvars)
        """
        jac = self._bkd.zeros((1, self.nvars()))
        for adjoint in self._adjoints:
            jac = jac + adjoint.jacobian(design_weights)
        return jac

    def hvp_implemented(self) -> bool:
        """Whether Hessian-vector product is implemented."""
        return True

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
        hvp_result = self._bkd.zeros((self.nvars(), 1))
        for adjoint in self._adjoints:
            hvp_result = hvp_result + adjoint.hvp(design_weights, vec)
        return hvp_result


class IOptimalLeastSquaresCriterion(IOptimalCriterion[Array], Generic[Array]):
    """
    I-optimal criterion for least squares regression.

    Alias for IOptimalCriterion when used with LeastSquaresDesignMatrices.
    """

    pass
