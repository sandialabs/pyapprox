r"""Burgers equation physics for spectral collocation.

Implements the 1D viscous Burgers equation

.. math:: \partial_t u + u\, u_x = (\nu u_x)_x + f

in one of two discrete forms, which agree for the continuous equation
but not at the nodes:

- conservative, :math:`-D_1 (u^2/2 - \nu\, D_1 u) + f`: the flux is
  formed at the nodes and differentiated, so the nodal residual
  integrates exactly to the boundary flux difference;
- non-conservative (advective),
  :math:`-u \circ D_1 u + \nu\, D_2 u + (D_1 \nu) \circ D_1 u + f`.
"""

from typing import Optional

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.physics.base import AbstractScalarPhysics
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalCoefficientProtocol,
    as_nodal_coefficient,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class BurgersPhysics1D(AbstractScalarPhysics[Array]):
    r"""1D viscous Burgers equation physics.

    Implements :math:`\partial_t u + u\, u_x = (\nu u_x)_x + f`; see the
    module docstring for the two discrete forms.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        1D collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    viscosity : float, NodalCoefficientProtocol or callable
        Kinematic viscosity :math:`\nu`, in the forms
        ``as_nodal_coefficient`` takes (an array arrives as
        ``NodalField``).
    forcing : float, NodalCoefficientProtocol or callable, optional
        Forcing term f, in the same forms. Absent means none.
    conservative : bool, optional
        Use the conservative (flux) form. Default: True.

    Raises
    ------
    ValueError
        If the basis is not 1D.

    Examples
    --------
    >>> bkd = NumpyBkd()
    >>> mesh = TransformedMesh1D(30, bkd)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
    >>> physics = BurgersPhysics1D(basis, bkd, viscosity=0.01)
    >>> physics.set_boundary_conditions([bc_left, bc_right])
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        viscosity: CoefficientInput[Array],
        forcing: Optional[CoefficientInput[Array]] = None,
        conservative: bool = True,
    ):
        super().__init__(basis, bkd)

        if basis.ndim() != 1:
            raise ValueError("BurgersPhysics1D requires a 1D basis")

        npts = basis.npts()
        self._viscosity = as_nodal_coefficient(viscosity, npts, bkd)
        self._forcing: Optional[NodalCoefficientProtocol[Array]] = (
            None if forcing is None else as_nodal_coefficient(forcing, npts, bkd)
        )
        self._conservative = conservative

        # Precompute derivative matrices
        self._D1 = basis.derivative_matrix(1, 0)  # First derivative
        self._D2 = basis.derivative_matrix(2, 0)  # Second derivative

    def viscosity_function(self) -> NodalCoefficientProtocol[Array]:
        """The viscosity coefficient."""
        return self._viscosity

    def forcing_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The forcing; None without a forcing term."""
        return self._forcing

    def is_conservative(self) -> bool:
        """Whether the conservative (flux) form is used."""
        return self._conservative

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        if self._forcing is None:
            return self._bkd.zeros((self.npts(),))
        return self._forcing.values(time)

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        For transient problems: du/dt = residual(u, t)

        Parameters
        ----------
        state : Array
            Solution state u. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (npts,)
        """
        nu = self._viscosity.values(time)
        du_dx = self._D1 @ state
        if self._conservative:
            flux = 0.5 * state * state - nu * du_dx
            residual = -(self._D1 @ flux)
        else:
            residual = (
                nu * (self._D2 @ state)
                + (self._D1 @ nu) * du_dx
                - state * du_dx
            )
        return residual + self._get_forcing(time)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/du.

        Parameters
        ----------
        state : Array
            Solution state u. Shape: (npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (npts, npts)
        """
        nu = self._viscosity.values(time)
        D1 = self._D1
        if self._conservative:
            # -D1 (diag(u) - diag(nu) D1), formed by column / row scaling
            return -(D1 * state[None, :]) + D1 @ (nu[:, None] * D1)
        du_dx = D1 @ state
        jacobian = nu[:, None] * self._D2 + (D1 @ nu)[:, None] * D1
        return jacobian - self._bkd.diag(du_dx) - state[:, None] * D1

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        r""":math:`\lambda^\top (\partial^2 f/\partial u^2) w`, exact: the
        residual is quadratic in the state. Shape: (npts,)

        Conservative: :math:`-w \circ D_1^\top \lambda`. Non-conservative:
        :math:`-\lambda \circ D_1 w - D_1^\top (\lambda \circ w)`.
        """
        D1 = self._D1
        if self._conservative:
            return -wvec * (D1.T @ adj_state)
        return -adj_state * (D1 @ wvec) - D1.T @ (adj_state * wvec)

    # -- the interior operator (CollocationInteriorOperatorProtocol)

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`f(u, t)`, no boundary condition applied. Shape: (npts,)"""
        return self.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial f/\partial u`. Shape: (npts, npts)"""
        return self.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Quadratic in the state: exact second derivatives."""
        return StateDerivatives.second_order(self.state_state_hvp)

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether the viscosity is declared time-invariant."""
        return not self._viscosity.is_time_dependent()

    def interior_is_time_invariant(self) -> bool:
        """Whether, besides the viscosity, the forcing is declared
        time-invariant."""
        return self.interior_jacobian_is_time_invariant() and not (
            self._forcing is not None and self._forcing.is_time_dependent()
        )


def create_burgers_1d(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    viscosity: CoefficientInput[Array],
    forcing: Optional[CoefficientInput[Array]] = None,
    conservative: bool = True,
) -> BurgersPhysics1D[Array]:
    r"""Create 1D Burgers equation physics.

    Solves :math:`\partial_t u + u\, u_x = (\nu u_x)_x + f`.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        1D collocation basis.
    bkd : Backend
        Computational backend.
    viscosity : float, NodalCoefficientProtocol or callable
        Kinematic viscosity.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Source term f.
    conservative : bool
        Use conservative form (default: True).

    Returns
    -------
    BurgersPhysics1D
        Burgers equation physics.
    """
    return BurgersPhysics1D(
        basis=basis,
        bkd=bkd,
        viscosity=viscosity,
        forcing=forcing,
        conservative=conservative,
    )
