r"""Helmholtz equation physics for spectral collocation.

Implements, in the screened sign convention,

.. math:: -\Delta u + k^2 u = f,

which the Galerkin Helmholtz physics shares. Separating the wave equation
gives the opposite sign, :math:`-\Delta u - k^2 u = f`; that form is
reached here with a negative ``wave_number_sq``.
"""

from typing import Optional, Union

from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalField,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class HelmholtzPhysics(AdvectionDiffusionReaction[Array]):
    r"""Helmholtz equation physics, screened sign convention.

    Solves :math:`-\Delta u + k^2 u = f` (see the module docstring for
    the sign). The residual is :math:`\Delta u - k^2 u + f`: ADR with
    diffusion 1, reaction :math:`-k^2` and forcing :math:`f`.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    wave_number_sq : float or Array
        Squared wave number k^2. If Array, shape: (npts,); it becomes the
        reaction ``NodalField`` :math:`-k^2`.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Forcing term f, in the forms ``AdvectionDiffusionReaction`` takes.

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
        forcing: Optional[CoefficientInput[Array]] = None,
    ):
        # ADR residual D lap(u) + r u + f with D = 1, r = -k^2
        reaction: Union[float, NodalField[Array]] = (
            -wave_number_sq
            if isinstance(wave_number_sq, (int, float))
            else NodalField(-wave_number_sq)
        )
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
        """The squared wave number given at construction.

        A parameterization writes the reaction field, not this value.
        """
        return self._wave_number_sq


def create_helmholtz(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    wave_number_sq: Union[float, Array],
    forcing: Optional[CoefficientInput[Array]] = None,
) -> HelmholtzPhysics[Array]:
    r"""Create Helmholtz equation physics.

    Solves :math:`-\Delta u + k^2 u = f`, the screened sign convention.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    wave_number_sq : float or Array
        Squared wave number k^2.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Source term f.

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
