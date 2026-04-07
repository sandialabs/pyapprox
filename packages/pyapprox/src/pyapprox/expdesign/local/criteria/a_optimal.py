"""
A-optimal design criterion.

A-optimal designs minimize the trace of the covariance matrix:
    J(w) = trace(Cov(w)) = trace(M1^{-1} @ M0 @ M1^{-1})

This is equivalent to sum_i e_i^T @ Cov @ e_i where e_i are unit vectors.
"""

from typing import Generic

from pyapprox.expdesign.local.adjoint import AdjointModel
from pyapprox.expdesign.local.protocols import DesignMatricesProtocol
from pyapprox.util.backends.protocols import Array, Backend

from .base import LocalOEDCriterionBase


class AOptimalCriterion(LocalOEDCriterionBase[Array], Generic[Array]):
    """
    A-optimal design criterion.

    Minimizes the trace of the parameter covariance matrix. This is
    implemented as the sum of C-optimal objectives for each unit vector.

    Parameters
    ----------
    design_matrices : DesignMatricesProtocol[Array]
        Object that computes M0, M1 matrices from design weights.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The objective is:
        J(w) = trace(Cov(w))
             = trace(M1(w)^{-1} @ M0(w) @ M1(w)^{-1})
             = sum_i x_i(w)^T @ M0(w) @ x_i(w)

    where x_i(w) = M1(w)^{-1} @ e_i and e_i is the i-th unit vector.
    """

    def __init__(
        self,
        design_matrices: DesignMatricesProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        super().__init__(design_matrices, bkd)

        # Create adjoint models for each unit vector
        ndesign_vars = design_matrices.ndesign_vars()
        self._adjoints = []
        for i in range(ndesign_vars):
            adjoint = AdjointModel(design_matrices, bkd)
            # Unit vector e_i
            vec = bkd.zeros((ndesign_vars,))
            vec[i] = 1.0
            adjoint.set_vector(vec)
            self._adjoints.append(adjoint)

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate A-optimal criterion.

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
        Jacobian of A-optimal criterion.

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


class AOptimalLeastSquaresCriterion(AOptimalCriterion[Array], Generic[Array]):
    """
    A-optimal criterion for least squares regression.

    Alias for AOptimalCriterion when used with LeastSquaresDesignMatrices.
    """

    pass


class AOptimalQuantileCriterion(AOptimalCriterion[Array], Generic[Array]):
    """
    A-optimal criterion for quantile regression.

    Uses the same formula as AOptimalCriterion but with different
    design matrices (QuantileDesignMatrices).
    """

    pass
