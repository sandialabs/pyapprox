"""FitzHugh-Nagumo physics for spectral collocation.

Implements the FitzHugh-Nagumo model for excitable media:
    dv/dt = D_v * Laplacian(v) + v*(1-v)*(v-alpha) - w + f_v
    dw/dt = eps * (beta*v - gamma*w) + f_w

where:
    v = voltage/activation variable
    w = recovery variable
    D_v = diffusion coefficient for v (w typically doesn't diffuse)
    alpha = threshold parameter (typically 0.1)
    eps = time scale separation (typically 0.01)
    beta = coupling coefficient (typically 0.5)
    gamma = recovery rate (typically 1.0)
"""

from typing import Callable, Optional

from pyapprox.pde.collocation.physics.reaction_diffusion import (
    FitzHughNagumoReaction,
    TwoSpeciesReactionDiffusionPhysics,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class FitzHughNagumoPhysics(TwoSpeciesReactionDiffusionPhysics[Array]):
    """FitzHugh-Nagumo physics for excitable media.

    Implements the FitzHugh-Nagumo model:
        dv/dt = D_v * Laplacian(v) + v*(1-v)*(v-alpha) - w + f_v
        dw/dt = eps * (beta*v - gamma*w) + f_w

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion_v : float
        Diffusion coefficient for voltage (default: 1e-3).
    alpha : float
        Threshold parameter (default: 0.1).
    eps : float
        Time scale separation (default: 0.01).
    beta : float
        Coupling coefficient (default: 0.5).
    gamma : float
        Recovery rate (default: 1.0).
    forcing_v : Callable[[float], Array], optional
        Forcing term for voltage.
    forcing_w : Callable[[float], Array], optional
        Forcing term for recovery.

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> mesh = TransformedMesh1D(30, bkd)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
    >>> physics = FitzHughNagumoPhysics(basis, bkd, diffusion_v=1e-3)
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        diffusion_v: float = 1e-3,
        alpha: float = 0.1,
        eps: float = 0.01,
        beta: float = 0.5,
        gamma: float = 1.0,
        forcing_v: Optional[Callable[[float], Array]] = None,
        forcing_w: Optional[Callable[[float], Array]] = None,
    ):
        self._alpha = alpha
        self._eps = eps
        self._beta = beta
        self._gamma = gamma

        reaction = FitzHughNagumoReaction(alpha, eps, beta, gamma, bkd)

        super().__init__(
            basis=basis,
            bkd=bkd,
            diffusion0=diffusion_v,
            diffusion1=0.0,  # Recovery variable doesn't diffuse
            reaction=reaction,
            forcing0=forcing_v,
            forcing1=forcing_w,
        )

    def alpha(self) -> float:
        """Return threshold parameter."""
        return self._alpha

    def eps(self) -> float:
        """Return time scale separation."""
        return self._eps

    def beta(self) -> float:
        """Return coupling coefficient."""
        return self._beta

    def gamma(self) -> float:
        """Return recovery rate."""
        return self._gamma

    def set_parameters(
        self,
        alpha: Optional[float] = None,
        eps: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """Update FitzHugh-Nagumo parameters.

        Parameters
        ----------
        alpha : float, optional
            Threshold parameter.
        eps : float, optional
            Time scale separation.
        beta : float, optional
            Coupling coefficient.
        gamma : float, optional
            Recovery rate.
        """
        if alpha is not None:
            self._alpha = alpha
        if eps is not None:
            self._eps = eps
        if beta is not None:
            self._beta = beta
        if gamma is not None:
            self._gamma = gamma

        # Update reaction object
        self._reaction = FitzHughNagumoReaction(
            self._alpha, self._eps, self._beta, self._gamma, self._bkd
        )


def create_fitzhugh_nagumo(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    diffusion_v: float = 1e-3,
    alpha: float = 0.1,
    eps: float = 0.01,
    beta: float = 0.5,
    gamma: float = 1.0,
    forcing_v: Optional[Callable[[float], Array]] = None,
    forcing_w: Optional[Callable[[float], Array]] = None,
) -> FitzHughNagumoPhysics[Array]:
    """Create FitzHugh-Nagumo physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    diffusion_v : float
        Diffusion coefficient for voltage (default: 1e-3).
    alpha : float
        Threshold parameter (default: 0.1).
    eps : float
        Time scale separation (default: 0.01).
    beta : float
        Coupling coefficient (default: 0.5).
    gamma : float
        Recovery rate (default: 1.0).
    forcing_v : Callable, optional
        Forcing for voltage.
    forcing_w : Callable, optional
        Forcing for recovery.

    Returns
    -------
    FitzHughNagumoPhysics
        FitzHugh-Nagumo physics.
    """
    return FitzHughNagumoPhysics(
        basis=basis,
        bkd=bkd,
        diffusion_v=diffusion_v,
        alpha=alpha,
        eps=eps,
        beta=beta,
        gamma=gamma,
        forcing_v=forcing_v,
        forcing_w=forcing_w,
    )
