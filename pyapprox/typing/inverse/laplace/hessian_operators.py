"""
Hessian operators for Laplace approximation.

These operators compute the action of various Hessian matrices needed
for Laplace approximation without forming the full dense matrix.
"""

from typing import Generic, Callable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.protocols import SqrtCovarianceOperatorProtocol


class ApplyNegLogLikelihoodHessian(Generic[Array]):
    r"""
    Apply the Hessian of the negative log-likelihood to vectors.

    For a log-likelihood :math:`\log p(y|x)`, this operator computes:

    .. math::
        H_{nll}(x) v = -\nabla^2_x \log p(y|x) \cdot v

    for any vector v. This is used in Laplace approximation where the
    posterior precision is approximated by the Hessian of the negative
    log-posterior at the MAP point.

    Parameters
    ----------
    apply_hessian : Callable[[Array, Array], Array]
        Function that applies the Hessian of the log-likelihood to a vector.
        Signature: apply_hessian(sample, vec) -> hessian_vec_product
        where sample has shape (nvars, 1) and vec has shape (nvars, 1).
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # For quadratic log-likelihood, Hessian is constant
    >>> H = np.array([[2.0, 0.5], [0.5, 1.0]])
    >>> def apply_hess(sample, vec):
    ...     return H @ vec
    >>> op = ApplyNegLogLikelihoodHessian(apply_hess, 2, bkd)
    """

    def __init__(
        self,
        apply_hessian: Callable[[Array, Array], Array],
        nvars: int,
        bkd: Backend[Array],
    ):
        self._apply_hessian = apply_hessian
        self._nvars = nvars
        self._bkd = bkd
        self._sample: Array | None = None

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def set_sample(self, sample: Array) -> None:
        """
        Set the sample point at which to evaluate the Hessian.

        Parameters
        ----------
        sample : Array
            The sample point. Shape: (nvars, 1)
        """
        if sample.shape != (self._nvars, 1):
            raise ValueError(
                f"sample has wrong shape {sample.shape}, "
                f"expected ({self._nvars}, 1)"
            )
        self._sample = sample

    def apply(self, vecs: Array) -> Array:
        r"""
        Apply the negative log-likelihood Hessian to vectors.

        Computes :math:`-H \cdot vecs` where H is the Hessian of the
        log-likelihood at the current sample point.

        Parameters
        ----------
        vecs : Array
            Input vectors. Shape: (nvars, nvecs)

        Returns
        -------
        Array
            Hessian-vector products. Shape: (nvars, nvecs)
        """
        if self._sample is None:
            raise RuntimeError("Must call set_sample() first")

        hvps = []
        for ii in range(vecs.shape[1]):
            vec = vecs[:, ii:ii+1]
            # Negate because we want negative log-likelihood Hessian
            hvps.append(-self._apply_hessian(self._sample, vec))
        return self._bkd.hstack(hvps)


class PriorConditionedHessianMatVec(Generic[Array]):
    r"""
    Prior-conditioned misfit Hessian matrix-vector operator.

    For a prior with covariance :math:`C = LL^T` and misfit Hessian H,
    this operator computes:

    .. math::
        L^T \cdot H \cdot L \cdot v

    This conditioning is useful for:
    1. Randomized eigenvalue decomposition (eigenvalues relative to prior)
    2. Low-rank approximations of the posterior covariance
    3. Detecting data-informed directions in parameter space

    The prior-conditioned Hessian has eigenvalues that indicate how much
    the data informs each direction relative to the prior:
    - Eigenvalue >> 1: Data strongly constrains this direction
    - Eigenvalue ~ 0: Data provides little information beyond prior

    Parameters
    ----------
    prior_sqrt : SqrtCovarianceOperatorProtocol
        Square-root covariance operator L for the prior.
    apply_hessian : Callable[[Array], Array]
        Function that applies the misfit Hessian H to a vector.
        Signature: apply_hessian(vec) -> H @ vec

    Notes
    -----
    This operator is symmetric since H is symmetric and we compute
    :math:`L^T H L`, which is symmetric for any symmetric H.
    """

    def __init__(
        self,
        prior_sqrt: SqrtCovarianceOperatorProtocol[Array],
        apply_hessian: Callable[[Array], Array],
    ):
        self._prior_sqrt = prior_sqrt
        self._apply_hessian = apply_hessian
        self._bkd = prior_sqrt.bkd()

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._prior_sqrt.nvars()

    def apply(self, vecs: Array) -> Array:
        r"""
        Apply the prior-conditioned Hessian to vectors.

        Computes :math:`L^T \cdot H \cdot L \cdot vecs`.

        Parameters
        ----------
        vecs : Array
            Input vectors. Shape: (nvars, nvecs)

        Returns
        -------
        Array
            Prior-conditioned Hessian-vector products. Shape: (nvars, nvecs)
        """
        # L @ vecs
        Lv = self._prior_sqrt.apply(vecs)
        # H @ L @ vecs
        HLv = self._apply_hessian(Lv)
        # L^T @ H @ L @ vecs
        return self._prior_sqrt.apply_transpose(HLv)

    def apply_transpose(self, vecs: Array) -> Array:
        """
        Apply the transpose (same as apply for symmetric operator).

        Parameters
        ----------
        vecs : Array
            Input vectors. Shape: (nvars, nvecs)

        Returns
        -------
        Array
            Same as apply() since this operator is symmetric.
        """
        return self.apply(vecs)

    def nrows(self) -> int:
        """Return the number of rows (same as nvars for square operator)."""
        return self.nvars()

    def ncols(self) -> int:
        """Return the number of columns (same as nvars for square operator)."""
        return self.nvars()
