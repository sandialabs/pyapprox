r"""SPDE-based KLE for Matern random fields via bilaplacian prior.

The SPDE approach computes a Karhunen-Loeve Expansion by solving a sparse
generalized eigenvalue problem derived from the stochastic PDE
representation of Matern random fields, achieving O(N) memory with sparse
matrices instead of the O(N^2) dense kernel matrices used by
:class:`MeshKLE` and :class:`GalerkinKLE`.

The bilaplacian prior (:math:`\alpha=2`) corresponds to a Matern kernel
with smoothness :math:`\nu = \alpha - d/2`:

- **1D**: :math:`\nu = 3/2` (once differentiable, identical to the
  standard Matern-3/2 GP kernel).
- **2D**: :math:`\nu = 1` (once differentiable, between the standard
  Matern-1/2 and 3/2 GP kernels).
- **3D**: :math:`\nu = 1/2` (continuous but not differentiable, identical
  to the exponential kernel).

The correlation length is :math:`\ell_c = \sqrt{\gamma/\delta}` in all
dimensions.  Integer :math:`\alpha` avoids fractional differential
operators, which is why :math:`\alpha = 2` is the standard choice for the
SPDE approach.

The eigenvalue problem solved is:

.. math::

    A \phi_k = \mu_k\, M \phi_k

where :math:`A = \gamma K_{\mathrm{stiff}} + \delta M + \xi M_{\partial}`
is the SPDE precision operator and :math:`M` is the FEM mass matrix.
The KLE eigenvalues are :math:`\lambda_k = \gamma^2/(\tau^2 \mu_k^2)`,
where the :math:`\gamma^2` factor arises because
:math:`A = \gamma L_h` (so :math:`A^{-1} = \gamma^{-1} L_h^{-1}`), and
:math:`\tau` is computed analytically from the SPDE-Matern variance
formula:

.. math::

    \sigma^2 = \frac{\Gamma(\nu)}
                    {\Gamma(\nu + d/2)\,(4\pi)^{d/2}\,
                     \kappa^{2\nu}\,\tau^2}

with :math:`\kappa = \sqrt{\delta/\gamma}` and :math:`\nu = \alpha - d/2`.
This ensures the eigenvalues match kernel-based KLE methods mode-by-mode
(up to discretization and boundary effects).

The SPDE, Nystrom, and Galerkin methods all discretize the same continuous
Fredholm eigenvalue problem but differ in how they represent the covariance
operator: Nystrom evaluates the kernel at discrete points with quadrature
weights, Galerkin projects the kernel onto the FE basis via double
integration, and the SPDE replaces the dense kernel matrix with a sparse
differential operator whose Green's function is the Matern covariance.
On a bounded domain the SPDE covariance differs from the stationary kernel
due to boundary conditions, but all three methods converge to the same
eigenvalues in the infinite-domain limit; on any fixed domain Nystrom and
Galerkin agree to :math:`O(h^2)` while SPDE agrees to within a
boundary-dominated error that decreases as the domain grows relative to
the correlation length.

Note on the :math:`A = \gamma L_h` factorization: this holds exactly only
when the Robin coefficient satisfies :math:`\xi/\gamma = \kappa^2`.  For
independently chosen :math:`\xi` the factorization is approximate, but
the boundary mass :math:`M_\partial` is a lower-order correction that does
not affect the interior eigenvalues at leading order.
"""

from typing import Generic, Optional, Union

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


class SPDEMaternKLE(Generic[Array]):
    r"""SPDE-based KLE for Matern random fields.

    Stores pre-computed eigenpairs from the SPDE eigenvalue problem.
    The expansion is:

    .. math::

        f(x) = \mu(x) + \sigma \sum_{k=1}^{K}
               \sqrt{\lambda_k}\,\phi_k(x)\,z_k

    where :math:`\lambda_k` are scaled eigenvalues, :math:`\phi_k` are
    M-orthonormal eigenvectors, and :math:`z_k \sim N(0,1)`.

    The bilaplacian prior (:math:`\alpha=2`) corresponds to a Matern
    kernel with smoothness :math:`\nu = \alpha - d/2`: in 1D this gives
    :math:`\nu = 3/2` (identical to the standard Matern-3/2 GP kernel),
    in 2D :math:`\nu = 1` (between Matern-1/2 and 3/2), and in 3D
    :math:`\nu = 1/2` (identical to the exponential kernel).  The
    correlation length is :math:`\ell_c = \sqrt{\gamma/\delta}` in all
    dimensions.

    Parameters
    ----------
    eigenvalues : Array, shape (nterms,)
        Scaled eigenvalues in descending order.
    eigenvectors : Array, shape (nnodes, nterms)
        M-orthonormal eigenvectors.
    sigma : float
        Target marginal standard deviation.
    mean_field : float or Array
        Mean field.  Scalar is broadcast to all nodes.
    bkd : Backend[Array]
        Computational backend.
    gamma : float
        Diffusion coefficient of the SPDE.
    delta : float
        Reaction coefficient of the SPDE.
    xi : float
        Robin boundary condition coefficient.
    """

    def __init__(
        self,
        eigenvalues: Array,
        eigenvectors: Array,
        sigma: float,
        mean_field: Union[float, Array],
        bkd: Backend[Array],
        gamma: float,
        delta: float,
        xi: float,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        self._bkd = bkd
        self._sigma = sigma
        self._gamma = gamma
        self._delta = delta
        self._xi = xi

        self._eig_vals = eigenvalues
        self._nterms = eigenvalues.shape[0]
        nnodes = eigenvectors.shape[0]

        if np.isscalar(mean_field):
            self._mean_field = bkd.full((nnodes,), 1) * mean_field
        else:
            self._mean_field = mean_field

        self._unweighted_eig_vecs = eigenvectors
        self._sqrt_eig_vals = bkd.sqrt(eigenvalues)
        self._eig_vecs = eigenvectors * self._sqrt_eig_vals * self._sigma

    def __call__(self, coef: Array) -> Array:
        """Evaluate the KLE at given coefficients.

        Parameters
        ----------
        coef : Array, shape (nterms, nsamples)
            Random coefficients for each sample.

        Returns
        -------
        Array, shape (nnodes, nsamples)
            Field values at mesh nodes for each sample.
        """
        if coef.ndim != 2:
            raise ValueError(f"coef.ndim={coef.ndim} but should be 2")
        if coef.shape[0] != self._nterms:
            raise ValueError(
                f"coef.shape[0]={coef.shape[0]} != nterms={self._nterms}"
            )
        return self._mean_field[:, None] + self._eig_vecs @ coef

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nterms(self) -> int:
        """Return the number of KLE terms."""
        return self._nterms

    def nvars(self) -> int:
        """Return the number of KLE terms (alias for nterms)."""
        return self._nterms

    def eigenvectors(self) -> Array:
        """Return unweighted eigenvectors, shape (nnodes, nterms)."""
        return self._unweighted_eig_vecs

    def weighted_eigenvectors(self) -> Array:
        """Return eigenvectors scaled by sqrt(eigenvalues) * sigma.

        Shape (nnodes, nterms).
        """
        return self._eig_vecs

    def eigenvalues(self) -> Array:
        """Return eigenvalues, shape (nterms,)."""
        return self._eig_vals

    def mean_field(self) -> Array:
        """Return the mean field, shape (nnodes,)."""
        return self._mean_field

    def correlation_length(self) -> float:
        r"""Return the correlation length :math:`\ell_c = \sqrt{\gamma/\delta}`."""
        return float(np.sqrt(self._gamma / self._delta))

    def pointwise_variance(self) -> Array:
        r"""Return pointwise variance from the truncated KLE.

        .. math::

            \mathrm{Var}(x) = \sigma^2 \sum_{k=1}^{K}
                              \lambda_k\,\phi_k(x)^2

        Returns
        -------
        Array, shape (nnodes,)
            Pointwise variance at each mesh node.
        """
        # weighted_eigvecs already includes sqrt(lam)*sigma
        # so sum of squares gives sigma^2 * sum(lam_k * phi_k^2)
        return self._bkd.sum(self._eig_vecs ** 2, axis=1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nterms={self._nterms}, "
            f"sigma={self._sigma}, "
            f"corr_len={self.correlation_length():.4f})"
        )
