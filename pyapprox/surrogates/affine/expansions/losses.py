"""Loss functions for iterative fitting."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol


class BasisExpansionMSELoss(Generic[Array]):
    """Mean squared error loss for basis expansion fitting.

    L(c) = (1/2n) ||Phi c - y||^2

    Provides analytical jacobian and hvp for efficient optimization.
    Works with any BasisExpansionProtocol.

    Conforms to ObjectiveProtocol for use with ScipyTrustConstrOptimizer.
    The optimizer auto-detects jacobian() and hvp() methods.

    Parameters
    ----------
    expansion : BasisExpansionProtocol[Array]
        Basis expansion defining the model structure.
    samples : Array
        Training samples. Shape: (nvars, nsamples)
    values : Array
        Target values. Shape: (nqoi, nsamples)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        bkd: Backend[Array],
    ):
        self._expansion = expansion
        self._samples = samples
        self._values = values  # (nqoi, nsamples)
        self._bkd = bkd
        self._nsamples = samples.shape[1]
        self._nqoi = expansion.nqoi()
        # Pre-compute basis matrix for efficiency
        self._Phi = expansion.basis_matrix(samples)  # (nsamples, nterms)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        """Number of parameters (nterms * nqoi)."""
        return self._expansion.nterms() * self._nqoi

    def nqoi(self) -> int:
        """Loss is scalar."""
        return 1

    def __call__(self, params: Array) -> Array:
        """Compute loss.

        Parameters
        ----------
        params : Array
            Shape: (nvars, 1)

        Returns
        -------
        Array
            Loss value. Shape: (1, 1)
        """
        params_2d = self._reshape_params(params)  # (nterms, nqoi)
        residual = self._Phi @ params_2d - self._values.T  # (nsamples, nqoi)
        mse = 0.5 * self._bkd.sum(residual ** 2) / self._nsamples
        return self._bkd.reshape(mse, (1, 1))

    def jacobian(self, sample: Array) -> Array:
        """Compute gradient of loss w.r.t. params.

        grad = (1/n) Phi^T (Phi c - y)

        Parameters
        ----------
        sample : Array
            Parameter values. Shape: (nvars, 1)

        Returns
        -------
        Array
            Gradient. Shape: (1, nvars)
        """
        params_2d = self._reshape_params(sample)  # (nterms, nqoi)
        residual = self._Phi @ params_2d - self._values.T  # (nsamples, nqoi)
        grad = self._Phi.T @ residual / self._nsamples  # (nterms, nqoi)
        return self._bkd.reshape(self._bkd.flatten(grad), (1, -1))

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product.

        For MSE, Hessian is constant: H = (1/n) Phi^T Phi
        So hvp(v) = (1/n) Phi^T Phi v

        Parameters
        ----------
        sample : Array
            Parameter values. Shape: (nvars, 1) - not used since Hessian is
            constant
        vec : Array
            Direction vector. Shape: (nvars, 1)

        Returns
        -------
        Array
            Hessian-vector product. Shape: (nvars, 1)
        """
        vec_2d = self._reshape_params(vec)  # (nterms, nqoi)
        # H v = (1/n) Phi^T Phi v
        Phi_v = self._Phi @ vec_2d  # (nsamples, nqoi)
        hvp_result = self._Phi.T @ Phi_v / self._nsamples  # (nterms, nqoi)
        return self._bkd.reshape(self._bkd.flatten(hvp_result), (-1, 1))

    def _reshape_params(self, params: Array) -> Array:
        """Reshape params from flat (nvars,) or (nvars, 1) to (nterms, nqoi)."""
        if params.ndim == 1:
            return self._bkd.reshape(params, (-1, self._nqoi))
        elif params.shape[1] == 1:
            return self._bkd.reshape(params[:, 0], (-1, self._nqoi))
        return params
