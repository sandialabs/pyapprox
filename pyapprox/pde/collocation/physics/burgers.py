"""Burgers equation physics for spectral collocation.

Implements the 1D viscous Burgers equation:
    du/dt + u * du/dx = nu * d^2u/dx^2 + f

In conservative form:
    du/dt = -d/dx(u^2/2 - nu * du/dx) + f

In non-conservative form:
    du/dt = nu * d^2u/dx^2 - u * du/dx + f
"""

from typing import Generic, Optional, Callable, Union

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.pde.collocation.physics.base import AbstractScalarPhysics


class BurgersPhysics1D(AbstractScalarPhysics[Array]):
    """1D viscous Burgers equation physics.

    Implements the viscous Burgers equation:
        du/dt + u * du/dx = nu * d^2u/dx^2 + f

    where nu is the kinematic viscosity.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        1D collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    viscosity : float or Array
        Kinematic viscosity nu. If Array, shape: (npts,).
    forcing : Callable[[float], Array] or Array, optional
        Forcing term f(x). If callable, takes time and returns (npts,) array.
    conservative : bool, optional
        If True, use conservative flux form. Default: True.

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
        viscosity: Union[float, Array],
        forcing: Optional[Callable[[float], Array]] = None,
        conservative: bool = True,
    ):
        super().__init__(basis, bkd)

        if basis.ndim() != 1:
            raise ValueError("BurgersPhysics1D requires a 1D basis")

        npts = basis.npts()

        # Store viscosity
        if isinstance(viscosity, (int, float)):
            self._viscosity_value = float(viscosity)
            self._viscosity_array = bkd.full((npts,), float(viscosity))
            self._is_variable_viscosity = False
        else:
            self._viscosity_value = None
            self._viscosity_array = viscosity
            self._is_variable_viscosity = True

        self._forcing_func = forcing
        self._conservative = conservative

        # Precompute derivative matrices
        self._D1 = basis.derivative_matrix(1, 0)  # First derivative
        self._D2 = basis.derivative_matrix(2, 0)  # Second derivative

    def viscosity(self) -> Union[float, Array]:
        """Return viscosity coefficient."""
        if self._viscosity_value is not None:
            return self._viscosity_value
        return self._viscosity_array

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        npts = self.npts()
        if self._forcing_func is None:
            return self._bkd.zeros((npts,))
        if callable(self._forcing_func):
            return self._forcing_func(time)
        return self._forcing_func

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
        bkd = self._bkd
        npts = self.npts()

        # First and second derivatives
        du_dx = self._D1 @ state
        d2u_dx2 = self._D2 @ state

        if self._conservative:
            # Conservative form: -d/dx(u^2/2 - nu * du/dx)
            # = -d/dx(u^2/2) + nu * d^2u/dx^2
            # = -u * du/dx + nu * d^2u/dx^2
            # Note: d/dx(u^2/2) = u * du/dx

            # Flux = u^2/2 - nu * du/dx
            # -div(flux) = -d/dx(u^2/2) + d/dx(nu * du/dx)

            # For constant viscosity:
            # residual = -u * du/dx + nu * d^2u/dx^2
            if not self._is_variable_viscosity:
                residual = -state * du_dx + self._viscosity_value * d2u_dx2
            else:
                # Variable viscosity: d/dx(nu * du/dx) = nu * d^2u/dx^2 + dnu/dx * du/dx
                dnu_dx = self._D1 @ self._viscosity_array
                residual = (
                    -state * du_dx
                    + self._viscosity_array * d2u_dx2
                    + dnu_dx * du_dx
                )
        else:
            # Non-conservative form: nu * d^2u/dx^2 - u * du/dx
            if not self._is_variable_viscosity:
                residual = self._viscosity_value * d2u_dx2 - state * du_dx
            else:
                dnu_dx = self._D1 @ self._viscosity_array
                residual = (
                    self._viscosity_array * d2u_dx2
                    + dnu_dx * du_dx
                    - state * du_dx
                )

        # Add forcing
        residual = residual + self._get_forcing(time)

        return residual

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
        bkd = self._bkd
        npts = self.npts()

        # Compute du/dx for the nonlinear term
        du_dx = self._D1 @ state

        # Jacobian of -u * du/dx w.r.t. u:
        # d/du_j[-u_i * (du/dx)_i]
        # = -delta_{ij} * (du/dx)_i - u_i * D1[i,j]
        # = -diag(du/dx) - diag(u) @ D1

        # Jacobian of viscous term:
        # d/du_j[nu * d^2u/dx^2] = nu * D2 (for constant nu)

        jacobian = bkd.zeros((npts, npts))

        # Viscous term contribution
        if not self._is_variable_viscosity:
            jacobian = jacobian + self._viscosity_value * self._D2
        else:
            # Variable viscosity: d/du[nu * d^2u/dx^2 + dnu/dx * du/dx]
            # = diag(nu) @ D2 + diag(dnu/dx) @ D1
            dnu_dx = self._D1 @ self._viscosity_array
            jacobian = (
                jacobian
                + bkd.diag(self._viscosity_array) @ self._D2
                + bkd.diag(dnu_dx) @ self._D1
            )

        # Nonlinear term: -u * du/dx
        # Jacobian = -diag(du/dx) - diag(u) @ D1
        jacobian = jacobian - bkd.diag(du_dx) - bkd.diag(state) @ self._D1

        return jacobian


def create_burgers_1d(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    viscosity: Union[float, Array],
    forcing: Optional[Callable[[float], Array]] = None,
    conservative: bool = True,
) -> BurgersPhysics1D[Array]:
    """Create 1D Burgers equation physics.

    Solves: du/dt + u * du/dx = nu * d^2u/dx^2 + f

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        1D collocation basis.
    bkd : Backend
        Computational backend.
    viscosity : float or Array
        Kinematic viscosity.
    forcing : Callable or Array, optional
        Source term f(x).
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
