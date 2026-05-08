"""Helmholtz equation physics for spectral collocation.

Implements the Helmholtz equation:
    -Laplacian(u) + k^2 * u = f

where k is the wave number. This is the standard form of the Helmholtz
equation that arises from separating the wave equation or quantum mechanics.
"""

from typing import Callable, Optional, Union

from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class HelmholtzPhysics(AdvectionDiffusionReaction[Array]):
    """Helmholtz equation physics.

    Implements the Helmholtz equation:
        -Laplacian(u) + k^2 * u = f

    This is equivalent to:
        Laplacian(u) - k^2 * u + f = 0

    Which corresponds to ADR with:
        diffusion = 1 (so Laplacian term appears)
        reaction = -k^2 (negative, because ADR adds +r*u but we want -k^2*u)
        forcing = f

    Note: For the residual = 0 formulation:
        residual = Laplacian(u) - k^2*u + f = 0
    corresponds to:
        -Laplacian(u) + k^2*u = f (standard Helmholtz form)

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    wave_number_sq : float or Array
        Squared wave number k^2. If Array, shape: (npts,).
    forcing : Callable[[float], Array] or Array, optional
        Forcing term f(x). If callable, takes time and returns (npts,) array.

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> mesh = TransformedMesh1D(20, bkd)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
    >>> physics = HelmholtzPhysics(basis, bkd, wave_number_sq=1.0)
    >>> physics.set_boundary_conditions([bc_left, bc_right])
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        wave_number_sq: Union[float, Array],
        forcing: Optional[Callable[[float], Array]] = None,
    ):
        # Helmholtz: -Laplacian(u) + k^2 * u = f
        # Rewrite as: Laplacian(u) - k^2 * u + f = 0
        # ADR residual: D * Laplacian(u) + r * u + f
        # So: D = 1, r = -k^2
        if isinstance(wave_number_sq, (int, float)):
            reaction = -wave_number_sq
        else:
            reaction = -wave_number_sq
        super().__init__(
            basis=basis,
            bkd=bkd,
            diffusion=1.0,
            velocity=None,
            reaction=reaction,
            forcing=forcing,
        )
        self._wave_number_sq = wave_number_sq

    def wave_number_sq(self) -> Union[float, Array]:
        """Return the squared wave number."""
        return self._wave_number_sq


def create_helmholtz(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    wave_number_sq: Union[float, Array],
    forcing: Optional[Callable[[float], Array]] = None,
) -> HelmholtzPhysics[Array]:
    """Create Helmholtz equation physics.

    Solves: -Laplacian(u) - k^2 * u = f

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    wave_number_sq : float or Array
        Squared wave number k^2.
    forcing : Callable or Array, optional
        Source term f(x).

    Returns
    -------
    HelmholtzPhysics
        Helmholtz equation physics.
    """
    return HelmholtzPhysics(
        basis=basis,
        bkd=bkd,
        wave_number_sq=wave_number_sq,
        forcing=forcing,
    )
