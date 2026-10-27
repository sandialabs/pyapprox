r"""State constraints for Galerkin systems with essential boundary data.

The time integrator applies a
:class:`~pyapprox.ode.protocols.StateConstraintProtocol` to every stored
state. For a Galerkin system with essential (Dirichlet) data :math:`g(t)`
on the DOFs :math:`D`:

- :class:`EssentialPin` sets :math:`y_D = g(t)` and leaves the other
  entries unchanged. With the symmetric row treatment the stored states
  are exact on the boundary, and total content is conserved to the
  stepper's order.
- :class:`MassConsistentProjection` also corrects the interior,
  :math:`y_I = z_I + M_{II}^{-1} M_{ID} (z_D - g)`, so that the stored
  step's interior rows hold with the exact boundary increment
  :math:`M_{ID}(g_{n+1} - g_n)` instead of the stepper's quadrature of
  :math:`\int M_{ID} \dot g`: total content :math:`1^\top M u` is then
  conserved exactly by every stepper. The correction is the quadrature
  error, so it costs no order, at one solve with the mass per stored
  state.

The constraints are affine in the state and independent of the
parameters, so their parameter and second derivatives are stated zeros
and the sensitivity sweeps apply only their tangents.
"""

from typing import Generic, Optional, Protocol, Union

from scipy.sparse import spmatrix

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_spaces
from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinODEResidualProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet
from pyapprox.util.linalg.operators import (
    IdentityOperator,
    MatrixFree,
    OperatorProperties,
    RowsZeroedOperator,
    SpaceDescriptor,
    VectorSpaceProtocol,
)


class _PinTangent(Generic[Array]):
    """The pin's tangent, the identity with the essential rows zeroed,
    the same at every state and time."""

    def __init__(self, op: RowsZeroedOperator[Array, MatrixFree]) -> None:
        self._op = op

    def __call__(
        self, state: Array, time: float
    ) -> RowsZeroedOperator[Array, MatrixFree]:
        return self._op


class EssentialPin(Generic[Array]):
    r""":math:`\Pi_t(y)`: ``y`` with its essential entries set to the
    boundary data :math:`g(t)`, the other entries unchanged.

    A projection, affine in the state: its tangent is the identity with
    the essential rows zeroed (so :math:`\Pi'^\top` zeros the essential
    entries of an adjoint), and its parameter and second derivatives are
    stated zeros.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The system whose constraint set gives the essential DOFs and data.
    nparams : int
        The residual's number of parameters, sizing the derivatives'
        parameter space. The boundary data do not depend on them.
    """

    def __init__(
        self, system: GalerkinTransientSystemProtocol[Array], nparams: int
    ) -> None:
        self._constraint_set = system.constraint_set()
        bkd = system.bkd()
        states, params = backend_spaces(bkd, system.nstates(), nparams)
        dofs = [int(dof) for dof in bkd.to_numpy(self._constraint_set.dofs())]
        rows = BackendRowSet(states.descriptor(), dofs, bkd)
        tangent = _PinTangent(RowsZeroedOperator(IdentityOperator(states), rows))
        self._derivs: StateConstraintDerivatives[Array] = (
            StateConstraintDerivatives.second_order(
                states,
                params,
                tangent,
                param=STATED_ZERO,
                state_state=STATED_ZERO,
                state_param=STATED_ZERO,
                param_param=STATED_ZERO,
            )
        )

    def constrain(self, state: Array, time: float) -> Array:
        """``state`` with :math:`y_D = g(t)`."""
        return self._constraint_set.inject(state, time)

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs

    def __repr__(self) -> str:
        return f"{type(self).__name__}(ndofs={self._constraint_set.ndofs()})"


class _InteriorCorrection(Generic[Array]):
    r"""The map :math:`d \mapsto M_{II}^{-1} M_{ID} d` and its transpose,
    on full-length vectors (zero on the essential entries).

    ``mass`` is a neutralized mass with identity essential rows, with or
    without its coupling columns: solving it on a right-hand side that is
    zero on the essential rows gives :math:`M_{II}^{-1}` on the interior
    and zero on the boundary either way. ``raw_mass`` supplies
    :math:`M_{ID}` and :math:`M_{DI} = M_{ID}^\top`.
    """

    def __init__(
        self,
        constraint_set: ConstraintSetProtocol[Array],
        mass: MassMatrixProtocol[Array],
        raw_mass: Union[spmatrix, Array],
        bkd: Backend[Array],
    ) -> None:
        self._constraint_set = constraint_set
        self._mass = mass
        self._raw_mass = raw_mass
        self._bkd = bkd

    def _times_raw(self, vec: Array) -> Array:
        raw_mass = self._raw_mass
        if isinstance(raw_mass, spmatrix):
            return self._bkd.asarray(raw_mass @ self._bkd.to_numpy(vec))
        return raw_mass @ vec

    def _boundary_only(self, vec: Array) -> Array:
        """``vec`` with its interior entries zeroed."""
        return vec - self._constraint_set.zero_entries(vec)

    def apply(self, vec: Array) -> Array:
        r""":math:`(M_{II}^{-1} M_{ID} v_D, 0)` from the essential entries
        of ``vec``."""
        cs = self._constraint_set
        coupled = cs.zero_entries(self._times_raw(self._boundary_only(vec)))
        return cs.zero_entries(self._mass.solve(coupled))

    def apply_transpose(self, vec: Array) -> Array:
        r""":math:`(0, M_{DI} M_{II}^{-\top} v_I)` from the interior entries
        of ``vec``."""
        cs = self._constraint_set
        solved = cs.zero_entries(self._mass.solve_transpose(cs.zero_entries(vec)))
        return self._boundary_only(self._times_raw(solved))


class _MassConsistentTangentOperator(Generic[Array]):
    r""":math:`\Pi' = [[I, M_{II}^{-1} M_{ID}], [0, 0]]`, matrix-free on
    blocks of the state space."""

    def __init__(
        self,
        space: VectorSpaceProtocol[Array],
        constraint_set: ConstraintSetProtocol[Array],
        correction: _InteriorCorrection[Array],
        bkd: Backend[Array],
    ) -> None:
        self._space = space
        self._constraint_set = constraint_set
        self._correction = correction
        self._bkd = bkd

    def domain(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def _columns(self, block: Array, column_map: "_ColumnMap[Array]") -> Array:
        bkd = self._bkd
        return bkd.stack(
            [column_map(block[:, jj]) for jj in range(block.shape[1])], axis=1
        )

    def _apply_column(self, column: Array) -> Array:
        cs = self._constraint_set
        return cs.zero_entries(column) + self._correction.apply(column)

    def _apply_transpose_column(self, column: Array) -> Array:
        cs = self._constraint_set
        return cs.zero_entries(column) + self._correction.apply_transpose(column)

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        return self._columns(x, self._apply_column)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        return self._columns(y, self._apply_transpose_column)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.domain()!r})"


class _ColumnMap(Protocol[Array]):
    def __call__(self, column: Array) -> Array: ...


class _ConstantTangent(Generic[Array]):
    """A tangent that is the same operator at every state and time."""

    def __init__(self, op: _MassConsistentTangentOperator[Array]) -> None:
        self._op = op

    def __call__(
        self, state: Array, time: float
    ) -> _MassConsistentTangentOperator[Array]:
        return self._op


class MassConsistentProjection(Generic[Array]):
    r""":math:`\Pi_t(z) = (z_I + M_{II}^{-1} M_{ID} (z_D - g(t)),\; g(t))`.

    Pins the essential entries like :class:`EssentialPin` and moves the
    interior so that the mass rows of the stored step hold with the exact
    boundary increment (see module). A projection, affine in the state
    and independent of the parameters: its tangent is the constant
    :math:`[[I, M_{II}^{-1} M_{ID}], [0, 0]]` and its parameter and
    second derivatives are stated zeros. A lumped mass has
    :math:`M_{ID} = 0`, so the projection is the pin.

    Parameters
    ----------
    adapter : GalerkinODEResidualProtocol
        The time-integration adapter the stepper uses: its system gives the
        essential DOFs, data and consistent mass, its mass the solves with
        :math:`M_{II}` (either row treatment's), and its lumping whether a
        correction is needed at all.
    nparams : int
        The residual's number of parameters, sizing the derivatives'
        parameter space.
    """

    def __init__(
        self, adapter: GalerkinODEResidualProtocol[Array], nparams: int
    ) -> None:
        system = adapter.system()
        self._pin = EssentialPin(system, nparams)
        self._constraint_set = system.constraint_set()
        bkd = system.bkd()
        states, params = backend_spaces(bkd, system.nstates(), nparams)
        # Decided once: a lumped mass has no coupling to correct.
        self._correction: Optional[_InteriorCorrection[Array]] = (
            None
            if adapter.lumped_mass()
            else _InteriorCorrection(
                self._constraint_set, adapter.mass_matrix(), system.mass_matrix(), bkd
            )
        )
        if self._correction is None:
            self._derivs = self._pin.derivatives()
            return
        tangent = _ConstantTangent(
            _MassConsistentTangentOperator(
                states, self._constraint_set, self._correction, bkd
            )
        )
        self._derivs = StateConstraintDerivatives.second_order(
            states,
            params,
            tangent,
            param=STATED_ZERO,
            state_state=STATED_ZERO,
            state_param=STATED_ZERO,
            param_param=STATED_ZERO,
        )

    def constrain(self, state: Array, time: float) -> Array:
        r"""``state`` with :math:`y_D = g(t)` and the interior corrected."""
        pinned = self._pin.constrain(state, time)
        if self._correction is None:
            return pinned
        return pinned + self._correction.apply(state - pinned)

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ndofs={self._constraint_set.ndofs()}, "
            f"corrects={self._correction is not None})"
        )
