r"""Steady Stokes flow on a staggered collocation pair.

.. math::

    -\nabla \cdot \bigl(2\mu\, \varepsilon(u)\bigr) + \nabla p = f, \qquad
    \nabla \cdot u = g,

with :math:`\varepsilon(u) = \tfrac12 (\nabla u + \nabla u^\top)`. With
constant :math:`\mu` and :math:`\nabla \cdot u = 0` the viscous term is
:math:`-\mu \Delta u`.

The velocity lives on one mesh (Gauss-Lobatto nodes), the pressure on
another (interior Gauss nodes, ``N - 2`` per direction), which makes the
pair inf-sup stable: only the constant pressure is left undetermined. The
momentum rows are collocated at the velocity nodes, reading the pressure
there through its interpolant; the continuity rows at the pressure nodes,
reading the velocity's divergence there through its interpolant.
"""

from typing import Generic, List, Optional, Protocol, Sequence, runtime_checkable

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.physics.coefficients import (
    CheckedValues,
    CoefficientInput,
    NodalCoefficientProtocol,
    as_nodal_coefficient,
    declared_time_invariant,
    optional_nodal_coefficient,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class StokesBasisProtocol(Protocol, Generic[Array]):
    """What the Stokes operator reads from the velocity basis."""

    def bkd(self) -> Backend[Array]: ...

    def ndim(self) -> int: ...

    def npts(self) -> int: ...

    def derivative_matrix(self, order: int, dim: int) -> Array:
        """:math:`\\partial^{order}/\\partial x_{dim}` on the nodes.
        Shape: (npts, npts)"""
        ...


class StokesPhysics(Generic[Array]):
    r"""The interior operator of steady Stokes flow, velocity and pressure
    on different nodes.

    The state stacks the velocity components, each at the velocity nodes,
    then the pressure at the pressure nodes. With :math:`D_j` the velocity
    basis's first derivatives, :math:`T_u` the pressure's interpolant at
    the velocity nodes and :math:`T_p` the velocity's at the pressure
    nodes, the residual (zero at the solution) is

    .. math::

        r_i &= \sum_j D_j \bigl(\mu (D_j u_i + D_i u_j)\bigr)
               - D_i T_u p + f_i, \\
        r_p &= g - T_p \sum_k D_k u_k.

    Its constant pressure is undetermined: a closure row fixes it (one
    continuity row replaced by a mean or point value).

    Parameters
    ----------
    velocity_basis : StokesBasisProtocol
        The 2D or 3D basis on the velocity nodes.
    to_velocity : Array
        :math:`T_u`, pressure nodes to velocity nodes. Shape:
        (nvelocity, npressure)
    to_pressure : Array
        :math:`T_p`, velocity nodes to pressure nodes. Shape:
        (npressure, nvelocity)
    viscosity : float, NodalCoefficientProtocol or callable
        :math:`\mu` at the velocity nodes, in the forms
        ``as_nodal_coefficient`` takes. Must be positive.
    forcing : Sequence of float, NodalCoefficientProtocol or callable, optional
        :math:`f_i` at the velocity nodes, one per component; absent means
        none.
    divergence : float, NodalCoefficientProtocol or callable, optional
        :math:`g` at the pressure nodes; absent means zero.

    Raises
    ------
    TypeError
        If ``velocity_basis`` does not satisfy ``StokesBasisProtocol``.
    ValueError
        If the basis is not 2D or 3D, the transfer shapes do not match the
        basis and each other, or ``forcing`` does not have one entry per
        component.
    """

    def __init__(
        self,
        velocity_basis: StokesBasisProtocol[Array],
        to_velocity: Array,
        to_pressure: Array,
        viscosity: CoefficientInput[Array],
        forcing: Optional[Sequence[CoefficientInput[Array]]] = None,
        divergence: Optional[CoefficientInput[Array]] = None,
    ) -> None:
        if not isinstance(velocity_basis, StokesBasisProtocol):
            raise TypeError(
                "velocity_basis must satisfy StokesBasisProtocol, got "
                f"{type(velocity_basis).__name__}"
            )
        ndim = velocity_basis.ndim()
        if ndim not in (2, 3):
            raise ValueError(f"Stokes flow needs a 2D or 3D basis, got {ndim}D")
        bkd = velocity_basis.bkd()
        nvelocity = velocity_basis.npts()
        npressure = to_velocity.shape[1] if len(to_velocity.shape) == 2 else -1
        if tuple(to_velocity.shape) != (nvelocity, npressure) or npressure < 1:
            raise ValueError(
                f"to_velocity must have shape ({nvelocity}, npressure), got "
                f"{tuple(to_velocity.shape)}"
            )
        if tuple(to_pressure.shape) != (npressure, nvelocity):
            raise ValueError(
                f"to_pressure must have shape ({npressure}, {nvelocity}), got "
                f"{tuple(to_pressure.shape)}"
            )
        if forcing is not None and len(forcing) != ndim:
            raise ValueError(
                f"forcing needs one entry per velocity component ({ndim}), got "
                f"{len(forcing)}"
            )
        self._bkd, self._ndim = bkd, ndim
        self._nvelocity, self._npressure = nvelocity, npressure
        self._mu = as_nodal_coefficient(viscosity, nvelocity, bkd)
        self._checked_mu = CheckedValues(self._mu, self._check_mu)
        self._forcing: List[Optional[NodalCoefficientProtocol[Array]]] = (
            [None] * ndim
            if forcing is None
            else [optional_nodal_coefficient(f, nvelocity, bkd) for f in forcing]
        )
        self._divergence = optional_nodal_coefficient(divergence, npressure, bkd)
        self._D = [velocity_basis.derivative_matrix(1, d) for d in range(ndim)]
        # The pressure gradient at the velocity nodes, D_i T_u, and the
        # velocity divergence at the pressure nodes, T_p D_k: both
        # state-independent.
        self._gradient = [D @ to_velocity for D in self._D]
        self._divergence_blocks = [to_pressure @ D for D in self._D]

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return self._ndim

    def nstates(self) -> int:
        """Velocity components at the velocity nodes, then the pressure."""
        return self._ndim * self._nvelocity + self._npressure

    def layout(self) -> StateLayout:
        """Where each velocity component and the pressure sit in the state."""
        return StateLayout([self._nvelocity] * self._ndim + [self._npressure])

    def viscosity_function(self) -> NodalCoefficientProtocol[Array]:
        """:math:`\\mu` at the velocity nodes."""
        return self._mu

    def _check_mu(self, mu: Array, time: float) -> None:
        mu_min = self._bkd.to_float(self._bkd.min(mu))
        if mu_min <= 0.0:
            raise ValueError(f"viscosity must be positive; found min {mu_min:.2e}")

    def _check_state(self, state: Array) -> None:
        if tuple(state.shape) != (self.nstates(),):
            raise ValueError(
                f"state must have shape ({self.nstates()},), got {tuple(state.shape)}"
            )

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`(r_0, ..., r_{d-1}, r_p)`, no boundary or closure row
        applied. Shape: (nstates,)"""
        self._check_state(state)
        bkd, ndim, n = self._bkd, self._ndim, self._nvelocity
        mu = self._checked_mu.values(time)
        velocity = [state[i * n : (i + 1) * n] for i in range(ndim)]
        pressure = state[ndim * n :]
        gradients = [[D @ u for D in self._D] for u in velocity]
        blocks = []
        for i in range(ndim):
            row = -(self._gradient[i] @ pressure)
            for j in range(ndim):
                row = row + self._D[j] @ (mu * (gradients[i][j] + gradients[j][i]))
            forcing = self._forcing[i]
            if forcing is not None:
                row = row + forcing.values(time)
            blocks.append(row)
        continuity = -self._divergence_blocks[0] @ velocity[0]
        for k in range(1, ndim):
            continuity = continuity - self._divergence_blocks[k] @ velocity[k]
        if self._divergence is not None:
            continuity = continuity + self._divergence.values(time)
        blocks.append(continuity)
        return bkd.concatenate(blocks)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial r/\partial (u, p)`, independent of the state:

        .. math::

            \partial r_i/\partial u_k &= \delta_{ik} \sum_j D_j \mu D_j
            + D_k \mu D_i, &\quad \partial r_i/\partial p &= -D_i T_u, \\
            \partial r_p/\partial u_k &= -T_p D_k, &\quad
            \partial r_p/\partial p &= 0,

        with :math:`\mu` scaling rows. Shape: (nstates, nstates)"""
        self._check_state(state)
        bkd, ndim = self._bkd, self._ndim
        mu = self._checked_mu.values(time)
        mu_D = [mu[:, None] * D for D in self._D]
        rows = []
        for i in range(ndim):
            blocks = []
            for k in range(ndim):
                block = self._D[k] @ mu_D[i]
                if i == k:
                    for j in range(ndim):
                        block = block + self._D[j] @ mu_D[j]
                blocks.append(block)
            blocks.append(-self._gradient[i])
            rows.append(bkd.concatenate(blocks, axis=1))
        continuity = [-block for block in self._divergence_blocks]
        continuity.append(bkd.zeros((self._npressure, self._npressure)))
        rows.append(bkd.concatenate(continuity, axis=1))
        return bkd.concatenate(rows, axis=0)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Linear in the state: the second derivative is zero."""
        return StateDerivatives.linear(self._bkd)

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether the viscosity is declared time-invariant."""
        return declared_time_invariant(self._mu)

    def interior_is_time_invariant(self) -> bool:
        """Whether the viscosity and every forcing are declared
        time-invariant."""
        return declared_time_invariant(self._mu, self._divergence, *self._forcing)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ndim={self._ndim}, nvelocity={self._nvelocity}, "
            f"npressure={self._npressure})"
        )
