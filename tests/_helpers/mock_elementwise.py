r"""A 1D P1 element-wise operator in closed form, with no assembly library.

Galerkin P1 on uniform cells of :math:`[0, 1]` for

.. math:: -(\kappa u')' + c u' [+ u^2] = f,

with :math:`\kappa` constant per cell and :math:`f` nodal. Every element
quantity is a closed-form 2 x 2 formula, so the element-wise protocols
(``pyapprox.pde.elementwise``) are exercised on any backend without skfem,
and a consumer written against them (hyper-reduction, the generic
assembler) is shown not to depend on one assembly library.

The element residual of cell :math:`e = [x_e, x_{e+1}]`, length
:math:`h`, nodal states :math:`(a, b)`:

.. math::
    F_e = \frac{h}{6}\begin{pmatrix}2 & 1\\ 1 & 2\end{pmatrix} f_e
        - \frac{\kappa_e}{h}\begin{pmatrix}1 & -1\\ -1 & 1\end{pmatrix} u_e
        - \frac{c}{2}\begin{pmatrix}-1 & 1\\ -1 & 1\end{pmatrix} u_e
        - \frac{h}{12}\begin{pmatrix}3a^2 + 2ab + b^2\\
                                     a^2 + 2ab + 3b^2\end{pmatrix},

the last term only with the reaction. Advection makes the element
Jacobian non-symmetric, so a transpose action that returned the forward
one is detectable.
"""

from typing import Any, Generic, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.elementwise import (
    ElementSet,
    check_element_block,
    element_jacobian_blocks,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Backend

_Points = NDArray[np.floating[Any]]


class MockP1Operator(Generic[Array]):
    """The whole-mesh operator: element-wise and assembled.

    Satisfies ``ElementwiseOperatorProtocol`` and
    ``SpatialOperatorProtocol``. Its assembled residual and Jacobian are
    the scatter-sums of its own element contributions over every
    element; the coefficients are read at evaluation time, so a
    parameterization sets them here and every restriction sees them.

    Parameters
    ----------
    ncells : int
        Number of uniform cells; ``ncells + 1`` nodes.
    reaction : bool
        Include the :math:`u^2` reaction.
    bkd : Backend
        Computational backend.
    velocity : float
        Constant advection speed :math:`c`.
    """

    def __init__(
        self, ncells: int, reaction: bool, bkd: Backend[Array], velocity: float = 0.0
    ) -> None:
        if ncells < 1:
            raise ValueError(f"ncells must be positive, got {ncells}")
        self._ncells = ncells
        self._h = 1.0 / ncells
        self._reaction = reaction
        self._velocity = velocity
        self._bkd = bkd
        self._kappa = bkd.ones((ncells,))
        self._forcing = bkd.ones((ncells + 1,))
        self._all = MockP1Restricted(self, ElementSet.all(ncells, bkd))
        # Scatter from (element, local node) to global node, (ncells, 2, n).
        gather = np.zeros((ncells, 2, ncells + 1))
        for elem in range(ncells):
            gather[elem, 0, elem] = gather[elem, 1, elem + 1] = 1.0
        self._gather = bkd.asarray(gather)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return self._ncells + 1

    def nelements(self) -> int:
        return self._ncells

    def h(self) -> float:
        return self._h

    def velocity(self) -> float:
        return self._velocity

    def has_reaction(self) -> bool:
        return self._reaction

    def kappa(self) -> Array:
        """Diffusivity per cell. Shape: (ncells,)."""
        return self._kappa

    def forcing(self) -> Array:
        """Forcing per node. Shape: (ncells + 1,)."""
        return self._forcing

    def set_coefficients(self, kappa: Array, forcing: Array) -> None:
        if tuple(kappa.shape) != (self._ncells,):
            raise ValueError(f"kappa must have shape ({self._ncells},)")
        if tuple(forcing.shape) != (self._ncells + 1,):
            raise ValueError(f"forcing must have shape ({self._ncells + 1},)")
        self._kappa = kappa
        self._forcing = forcing

    def restrict(self, elements: ElementSet[Array]) -> "MockP1Restricted[Array]":
        """Restrict to ``elements``; every cell has two nodes, so any set
        is homogeneous."""
        return MockP1Restricted(self, elements)

    def _local_states(self, state: Array) -> Array:
        return self._bkd.einsum("ein,n->ei", self._gather, state)

    def spatial_residual(self, state: Array, time: float) -> Array:
        residuals = self._all.element_residuals(self._local_states(state), time)
        return self._bkd.einsum("ein,ei->n", self._gather, residuals)

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        blocks = element_jacobian_blocks(self._all, self._local_states(state), time)
        return self._bkd.einsum("ein,eij,ejm->nm", self._gather, blocks, self._gather)

    def state_derivatives(self) -> StateDerivatives[Array]:
        # The reaction's curvature is not supplied, which only rules out
        # second-order adjoints.
        if self._reaction:
            return StateDerivatives.none()
        return StateDerivatives.linear(self._bkd)

    def is_time_invariant(self) -> bool:
        return True

    def diffusivity_jacobian(self, state: Array) -> Array:
        """dF/dkappa. Shape: (ncells + 1, ncells)."""
        bkd = self._bkd
        stiffness = bkd.asarray([[1.0, -1.0], [-1.0, 1.0]]) / self._h
        per_cell = -bkd.einsum("ij,ej->ei", stiffness, self._local_states(state))
        return bkd.einsum("ein,ei->ne", self._gather, per_cell)

    def forcing_jacobian(self) -> Array:
        """dF/df, the consistent mass matrix. Shape: (n, n)."""
        bkd = self._bkd
        mass = bkd.asarray([[2.0, 1.0], [1.0, 2.0]]) * (self._h / 6.0)
        return bkd.einsum("ein,ij,ejm->nm", self._gather, mass, self._gather)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ncells={self._ncells}, "
            f"reaction={self._reaction}, velocity={self._velocity})"
        )


class MockP1Restricted(Generic[Array]):
    """The operator on a set of cells; ``RestrictedElementOperatorProtocol``.

    Rows and states coincide (a cell's two nodes).
    """

    def __init__(
        self, parent: MockP1Operator[Array], elements: ElementSet[Array]
    ) -> None:
        bkd = parent.bkd()
        indices = [int(index) for index in bkd.to_numpy(elements.indices())]
        if any(index < 0 or index >= parent.nelements() for index in indices):
            raise ValueError(f"element indices must lie in [0, {parent.nelements()})")
        nodes = np.array([[index, index + 1] for index in indices], dtype=np.int64)
        nodes = nodes.reshape(len(indices), 2)
        global_dofs = np.unique(nodes)
        self._parent = parent
        self._elements = elements
        self._bkd = bkd
        self._cells = elements.indices()
        self._nodes = bkd.asarray(nodes, dtype=bkd.int64_dtype())
        self._global_dofs = bkd.asarray(global_dofs, dtype=bkd.int64_dtype())
        self._local = bkd.asarray(
            np.searchsorted(global_dofs, nodes), dtype=bkd.int64_dtype()
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def elements(self) -> ElementSet[Array]:
        return self._elements

    def global_dofs(self) -> Array:
        return self._global_dofs

    def element_row_dofs(self) -> Array:
        return self._local

    def element_state_dofs(self) -> Array:
        return self._local

    def _linear_blocks(self) -> Array:
        """Stiffness plus advection per cell, (nsel, 2, 2), entering as -B u."""
        bkd, h = self._bkd, self._parent.h()
        stiffness = bkd.asarray([[1.0, -1.0], [-1.0, 1.0]]) / h
        advection = bkd.asarray([[-1.0, 1.0], [-1.0, 1.0]]) * (
            self._parent.velocity() / 2.0
        )
        kappa = self._parent.kappa()[self._cells]
        return kappa[:, None, None] * stiffness[None, :, :] + advection[None, :, :]

    def _reaction_blocks(self, local_states: Array) -> Array:
        """d(reaction)/du per cell, (nsel, 2, 2)."""
        bkd, scale = self._bkd, self._parent.h() / 12.0
        a, b = local_states[:, 0], local_states[:, 1]
        row0 = bkd.stack([6.0 * a + 2.0 * b, 2.0 * a + 2.0 * b], axis=1)
        row1 = bkd.stack([2.0 * a + 2.0 * b, 2.0 * a + 6.0 * b], axis=1)
        return scale * bkd.stack([row0, row1], axis=1)

    def _jacobian_blocks(self, local_states: Array) -> Array:
        blocks = -self._linear_blocks()
        if self._parent.has_reaction():
            blocks = blocks - self._reaction_blocks(local_states)
        return blocks

    def element_residuals(self, local_states: Array, time: float) -> Array:
        """Return :math:`F_e`. Shape: (nselected, 2)."""
        bkd, h = self._bkd, self._parent.h()
        expected = (self._elements.nselected(), 2)
        if tuple(local_states.shape) != expected:
            raise ValueError(
                f"local_states must have shape {expected}, got "
                f"{tuple(local_states.shape)}"
            )
        mass = bkd.asarray([[2.0, 1.0], [1.0, 2.0]]) * (h / 6.0)
        forcing = self._parent.forcing()[self._nodes]
        residual = bkd.einsum("ij,ej->ei", mass, forcing) - bkd.einsum(
            "eij,ej->ei", self._linear_blocks(), local_states
        )
        if self._parent.has_reaction():
            a, b = local_states[:, 0], local_states[:, 1]
            reaction = bkd.stack(
                [3.0 * a**2 + 2.0 * a * b + b**2, a**2 + 2.0 * a * b + 3.0 * b**2],
                axis=1,
            )
            residual = residual - (h / 12.0) * reaction
        return residual

    def element_jacobian_actions(
        self, local_states: Array, time: float, directions: Array
    ) -> Array:
        """Return :math:`J_e d_e`. Shape: (nselected, 2, k)."""
        check_element_block(directions, self._elements.nselected(), 2, "directions")
        return self._bkd.einsum(
            "eij,ejk->eik", self._jacobian_blocks(local_states), directions
        )

    def element_jacobian_transpose_actions(
        self, local_states: Array, time: float, adjoints: Array
    ) -> Array:
        """Return :math:`J_e^T a_e`. Shape: (nselected, 2, k)."""
        check_element_block(adjoints, self._elements.nselected(), 2, "adjoints")
        return self._bkd.einsum(
            "eij,eik->ejk", self._jacobian_blocks(local_states), adjoints
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nselected={self._elements.nselected()})"


class ModalCoefficientOperator(Protocol, Generic[Array]):
    """What :class:`ModalCoefficientParameterization` sets and differentiates.

    ``kappa`` lives on ``ncells`` cells (midpoints) and ``f`` on
    ``ncells + 1`` nodes of a uniform mesh of :math:`[0, 1]`.
    """

    def nstates(self) -> int: ...

    def set_coefficients(self, kappa: Array, forcing: Array) -> None: ...

    def diffusivity_jacobian(self, state: Array) -> Array: ...

    def forcing_jacobian(self) -> Array: ...


def _modes(points: _Points) -> _Points:
    return np.stack([0.2 * np.sin((k + 1) * np.pi * points) for k in range(3)], axis=1)


class ModalCoefficientParameterization(Generic[Array]):
    r"""kappa = 1 + Phi_mid p[:3], f = 1 + Phi_nodes p[3:], affine in p.

    Phi holds the modes :math:`0.2 \sin(k \pi x)`, k = 1..3, at the cell
    midpoints and at the nodes; kappa >= 0.4 on :math:`[-1, 1]^6`.
    """

    def __init__(
        self, operator: ModalCoefficientOperator[Array], bkd: Backend[Array]
    ) -> None:
        nodes = np.linspace(0.0, 1.0, operator.nstates())
        self._mid_modes = bkd.asarray(_modes(0.5 * (nodes[1:] + nodes[:-1])))
        self._node_modes = bkd.asarray(_modes(nodes))
        self._operator = operator
        self._bkd = bkd

    def nparams(self) -> int:
        return 6

    def targets(self) -> Tuple[object, ...]:
        return (self._operator,)

    def owned_coefficients(self) -> Tuple[str, ...]:
        return ("diffusivity", "forcing")

    def apply(self, params_1d: Array) -> None:
        bkd = self._bkd
        self._operator.set_coefficients(
            1.0 + bkd.dot(self._mid_modes, params_1d[:3]),
            1.0 + bkd.dot(self._node_modes, params_1d[3:]),
        )

    def is_affine_in_params(self) -> bool:
        return True

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return ParamDerivatives.first_order(
            self._param_jacobian, self._initial_param_jacobian
        )

    def _param_jacobian(self, state: Array, time: float, params_1d: Array) -> Array:
        bkd = self._bkd
        return bkd.hstack(
            [
                bkd.dot(self._operator.diffusivity_jacobian(state), self._mid_modes),
                bkd.dot(self._operator.forcing_jacobian(), self._node_modes),
            ]
        )

    def _initial_param_jacobian(self, params_1d: Array) -> Array:
        return self._bkd.zeros((self._operator.nstates(), 6))
