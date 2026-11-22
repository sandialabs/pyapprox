r"""A collocation system: a physics, its essential rows and its closure rows,
and the closed right-hand side it defines.

The closed right-hand side is

.. math::

    \tilde f = P f + \sum_k M_k c_k,

with :math:`f` the interior operator, :math:`c_k` the closure declarations,
:math:`M_k` selecting the rows declaration :math:`k` supplies and :math:`P`
selecting every other row. Shared nodes follow one rule: essential rows
beat closure rows, otherwise the later declaration wins. The essential
rows keep :math:`f` here; the pin that solves the system replaces them.
"""

from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives, PinnedRowsProtocol
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    ODEResidualDerivatives,
    StatedZero,
    backend_state_space,
)
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureParamFn,
    ClosureRowsProtocol,
    ClosureWeightedFn,
    NormalTermParamFn,
)
from pyapprox.pde.collocation.boundary.essential_rows import SHARED_ROW_RTOL
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
    bundle_jacobian_from_dense,
)
from pyapprox.pde.parameterizations.derivatives import (
    BCFluxParamSensitivityFn,
    ParamDerivatives,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.operators import RowSetProtocol


@runtime_checkable
class EssentialRowsProtocol(PinnedRowsProtocol[Array], Protocol[Array]):
    """Essential rows as the system reads them: the shared declaration,
    and whether its data are time-invariant."""

    def is_time_invariant(self) -> bool: ...


class NoEssentialRows(Generic[Array]):
    """The essential declaration of a system without essential rows: no
    rows, so a pin built on it leaves every row in place."""

    def __init__(self, bkd: Backend[Array], nstates: int) -> None:
        self._bkd, self._nstates = bkd, nstates
        self._rows = BackendRowSet(
            backend_state_space(bkd, nstates).descriptor(), [], bkd
        )

    def rows(self) -> RowSetProtocol[Array]:
        return self._rows

    def values(self, time: float) -> Array:
        return self._bkd.zeros((self._nstates, 1))

    def rates(self) -> Optional[Callable[[float], Array]]:
        return None

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        return PinnedRowsParamDerivatives.independent()

    def is_time_invariant(self) -> bool:
        return True


@runtime_checkable
class PairedRowsProtocol(Protocol):
    """Closure rows relating pairs of rows (periodic sides)."""

    def pairs(self) -> List[Tuple[List[int], List[int]]]: ...


class _PeriodicallyConsistentRows(Generic[Array]):
    """The essential rows, refusing values that differ on rows a periodic
    declaration pairs: the essential rows win those nodes and drop the
    periodic relation, so a smooth periodic solution needs the data to
    agree there."""

    def __init__(
        self,
        essential: EssentialRowsProtocol[Array],
        pairs: List[Tuple[int, int]],
        bkd: Backend[Array],
    ) -> None:
        self._essential, self._bkd = essential, bkd
        self._first = bkd.asarray([a for a, _ in pairs], dtype=bkd.int64_dtype())
        self._second = bkd.asarray([b for _, b in pairs], dtype=bkd.int64_dtype())
        self._pairs = pairs

    def rows(self) -> RowSetProtocol[Array]:
        return self._essential.rows()

    def values(self, time: float) -> Array:
        """The essential values. Shape: (nstates, 1)

        Raises
        ------
        ValueError
            If the values on two periodically paired rows differ by more
            than a relative 1e-10.
        """
        bkd = self._bkd
        column = self._essential.values(time)
        first, second = column[self._first, 0], column[self._second, 0]
        scale = bkd.maximum(
            bkd.maximum(bkd.abs(first), bkd.abs(second)), bkd.ones(first.shape)
        )
        mismatch = bkd.to_float(bkd.max(bkd.abs(first - second) / scale))
        if mismatch > SHARED_ROW_RTOL:
            raise ValueError(
                "essential data differ on rows a periodic declaration pairs "
                f"{self._pairs} (relative mismatch {mismatch:.3e}): no smooth "
                "periodic solution takes both values"
            )
        return column

    def rates(self) -> Optional[Callable[[float], Array]]:
        return self._essential.rates()

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        return self._essential.parameter_derivatives()

    def is_time_invariant(self) -> bool:
        return self._essential.is_time_invariant()


class CollocationSystem(Generic[Array]):
    r"""A physics with its essential and closure declarations.

    The declarations come from the arguments only, never from the
    physics. Shared nodes are resolved, not refused: essential rows beat
    closure rows, and among closure declarations the later one wins.

    Parameters
    ----------
    physics : CollocationInteriorOperatorProtocol
        The interior operator.
    essential : EssentialRowsProtocol, optional
        The essential (pinned) rows.
    closures : sequence of ClosureRowsProtocol
        The closure declarations, in the order that settles shared rows.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If a declaration's rows are not rows of the physics' state space.
    """

    def __init__(
        self,
        physics: CollocationInteriorOperatorProtocol[Array],
        essential: Optional[EssentialRowsProtocol[Array]] = None,
        closures: Sequence[ClosureRowsProtocol[Array]] = (),
    ) -> None:
        if not isinstance(physics, CollocationInteriorOperatorProtocol):
            raise TypeError(
                "physics must satisfy CollocationInteriorOperatorProtocol, got "
                f"{type(physics).__name__}"
            )
        if essential is not None and not isinstance(essential, EssentialRowsProtocol):
            raise TypeError(
                "essential must satisfy EssentialRowsProtocol, got "
                f"{type(essential).__name__}"
            )
        for closure in closures:
            if not isinstance(closure, ClosureRowsProtocol):
                raise TypeError(
                    "closures must satisfy ClosureRowsProtocol, got "
                    f"{type(closure).__name__}"
                )
        self._physics = physics
        self._bkd = physics.bkd()
        self._nstates = physics.nstates()
        bkd = self._bkd
        self._closures = list(closures)
        essential_mask = (
            bkd.zeros((self._nstates,))
            if essential is None
            else self._indicator(essential.rows(), "essential")
        )
        # Walk the closures last to first: each takes its rows that no
        # essential row and no later closure has taken
        free = 1.0 - essential_mask
        masks: List[Array] = []
        for k in reversed(range(len(self._closures))):
            indicator = self._indicator(self._closures[k].rows(), f"closure {k}")
            masks.append(free * indicator)
            free = free * (1.0 - indicator)
        self._closure_masks = masks[::-1]
        self._interior_mask = free + essential_mask
        self._essential = self._checked_essential(essential, essential_mask)

    def _indicator(self, rows: RowSetProtocol[Array], name: str) -> Array:
        """1 on the rows of the set, 0 elsewhere. Shape: (nstates,)"""
        size = rows.descriptor().size
        if size != self._nstates:
            raise ValueError(
                f"the {name} declaration's rows are of a space of size {size}, "
                f"not the physics' {self._nstates} states"
            )
        bkd = self._bkd
        return rows.replace_rows(bkd.zeros((size, 1)), bkd.ones((size, 1)))[:, 0]

    def _checked_essential(
        self,
        essential: Optional[EssentialRowsProtocol[Array]],
        essential_mask: Array,
    ) -> Optional[EssentialRowsProtocol[Array]]:
        if essential is None:
            return None
        bkd = self._bkd

        def pinned(row: int) -> bool:
            return bkd.to_float(essential_mask[row]) > 0.5

        pairs = [
            (first, second)
            for closure in self._closures
            if isinstance(closure, PairedRowsProtocol)
            for primary, partner in closure.pairs()
            for first, second in zip(primary, partner)
            if pinned(first) and pinned(second)
        ]
        if not pairs:
            return essential
        return _PeriodicallyConsistentRows(essential, pairs, self._bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return self._nstates

    def physics(self) -> CollocationInteriorOperatorProtocol[Array]:
        return self._physics

    def essential(self) -> Optional[EssentialRowsProtocol[Array]]:
        """The essential rows, checked against periodic pairings."""
        return self._essential

    def closures(self) -> List[ClosureRowsProtocol[Array]]:
        return list(self._closures)

    def closure_masks(self) -> List[Array]:
        """1 on the rows each closure declaration supplies. Each shape:
        (nstates,)"""
        return list(self._closure_masks)

    def interior_mask(self) -> Array:
        """1 on the rows the interior operator supplies (essential rows
        included). Shape: (nstates,)"""
        return self._interior_mask

    def is_time_invariant(self) -> bool:
        """Whether the interior operator and every declaration are
        declared time-invariant."""
        return (
            self._physics.interior_is_time_invariant()
            and (self._essential is None or self._essential.is_time_invariant())
            and all(closure.is_time_invariant() for closure in self._closures)
        )

    def owns(self, target: object) -> bool:
        """Whether ``target`` is the physics or a declaration."""
        return (
            target is self._physics
            or target is self._essential
            or any(target is closure for closure in self._closures)
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(physics={self._physics!r}, "
            f"nclosures={len(self._closures)})"
        )


def _refuse_own_parameters(system: CollocationSystem[Array]) -> None:
    """Refuse declarations whose data hold parameters of their own: the
    system's parameter state must be the only writer of the parameters,
    and boundary data cannot yet be targets of its parameterization."""
    essential = system.essential()
    if essential is not None and not isinstance(
        essential.parameter_derivatives().values, StatedZero
    ):
        raise ValueError(
            "the essential data depend on parameters of their own; a system "
            "with a parameter state does not yet accept parameter-dependent "
            "boundary data"
        )
    for k, closure in enumerate(system.closures()):
        # Without a normal-term sensitivity, a parameter Jacobian can only
        # come from the declaration's own data
        own = closure.derivatives().param_jacobian
        if own is not None and not isinstance(own, StatedZero):
            raise ValueError(
                f"closure {k} holds parameters of its own; a system with a "
                "parameter state does not yet accept parameter-dependent "
                "boundary data"
            )


class _NormalTermSensitivity(Generic[Array]):
    """The parameterization's flux sensitivity at the current parameters."""

    def __init__(
        self, parameters: ParameterState[Array], fn: BCFluxParamSensitivityFn[Array]
    ) -> None:
        self._parameters, self._fn = parameters, fn

    def __call__(self, state: Array, time: float, rows: Array, normals: Array) -> Array:
        return self._fn(state, time, self._parameters.current(), rows, normals)


class _ClosedJacobianMatrix(Generic[Array]):
    def __init__(self, rhs: "ClosedRightHandSide[Array]") -> None:
        self._rhs = rhs

    def jacobian(self, state: Array, time: float) -> Array:
        return self._rhs._jacobian_matrix(state, time)


class _ClosedParam(Generic[Array]):
    def __init__(self, rhs: "ClosedRightHandSide[Array]") -> None:
        self._rhs = rhs

    def __call__(self, state: Array, time: float) -> DerivativeOperator[Array]:
        return self._rhs._param_operator(state, time)


_STATE_STATE, _STATE_PARAM, _PARAM_PARAM = 0, 1, 2
_Weighted = Union[ClosureWeightedFn[Array], StatedZero]


class _ClosedSecond(Generic[Array]):
    def __init__(self, rhs: "ClosedRightHandSide[Array]", kind: int) -> None:
        self._rhs, self._kind = rhs, kind

    def __call__(
        self, state: Array, time: float, adj: Array
    ) -> DerivativeOperator[Array]:
        return self._rhs._second_operator(self._kind, state, time, adj)


class ClosedRightHandSide(Generic[Array]):
    r""":math:`\tilde f = P f + \sum_k M_k c_k` of a collocation system, and
    its derivatives bundle to first order.

    :math:`\tilde f_u = P f_u + \sum_k M_k c_{k,u}`, declared affine when
    :math:`f` is and every closure declaration states zero state
    curvature, and time-invariant when also every declaration is.
    :math:`\tilde f_p = P f_p + \sum_k M_k c_{k,p}`, present when every
    closure declaration supplies its parameter Jacobian; a normal term
    reading a physics coefficient gets :math:`\partial N/\partial p` from
    the parameterization's ``bc_flux_param_sensitivity``. The initial
    state's parameter derivatives are the interior's.

    Parameters
    ----------
    system : CollocationSystem
        The physics and its declarations.
    parameters : ParameterState, optional
        The parameters; none for a system without them.
    initial_state : ParamDerivatives or STATED_ZERO, optional
        As for :class:`CollocationRightHandSide`.
    """

    def __init__(
        self,
        system: CollocationSystem[Array],
        parameters: Optional[ParameterState[Array]] = None,
        initial_state: Optional[Union[ParamDerivatives[Array], StatedZero]] = None,
    ) -> None:
        if not isinstance(system, CollocationSystem):
            raise TypeError(
                f"system must be a CollocationSystem, got {type(system).__name__}"
            )
        self._system = system
        self._bkd = system.bkd()
        self._interior = CollocationRightHandSide(
            system.physics(), parameters, initial_state
        )
        self._parameters = parameters
        sensitivity: Optional[NormalTermParamFn[Array]] = None
        if parameters is not None:
            _refuse_own_parameters(system)
            fn = (
                parameters.parameterization()
                .param_derivatives()
                .bc_flux_param_sensitivity
            )
            if fn is not None:
                sensitivity = _NormalTermSensitivity(parameters, fn)
        self._closure_derivatives = [
            closure.derivatives(sensitivity) for closure in system.closures()
        ]
        self._derivatives = self._build_derivatives()

    def _build_derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        bkd, nstates = self._bkd, self._system.nstates()
        interior = self._system.physics()
        affine = interior.interior_state_derivatives().affine() and all(
            isinstance(d.state_curvature, StatedZero) for d in self._closure_derivatives
        )
        time_invariant = interior.interior_jacobian_is_time_invariant() and all(
            closure.is_time_invariant() for closure in self._system.closures()
        )
        interior_bundle = self._interior.derivatives()
        states_desc = backend_state_space(bkd, nstates).descriptor()
        jacobian = bundle_jacobian_from_dense(
            _ClosedJacobianMatrix(self), states_desc, bkd, affine, time_invariant
        )
        self._param_fns: List[Union[ClosureParamFn[Array], StatedZero]] = []
        for derivatives in self._closure_derivatives:
            if derivatives.param_jacobian is None:
                return ODEResidualDerivatives.state_only(jacobian)
            self._param_fns.append(derivatives.param_jacobian)
        if (
            self._parameters is None
            or interior_bundle.param is None
            or interior_bundle.initial_state is None
            or interior_bundle.param_space is None
        ):
            return ODEResidualDerivatives.state_only(jacobian)
        self._states_desc = states_desc
        self._params_desc = interior_bundle.param_space.descriptor()
        states = backend_state_space(bkd, nstates)
        params = interior_bundle.param_space
        curvatures = self._closure_curvatures()
        if (
            interior_bundle.state_state is None
            or interior_bundle.initial_state_curvature is None
            or curvatures is None
        ):
            return ODEResidualDerivatives.first_order(
                states,
                params,
                jacobian=jacobian,
                param=_ClosedParam(self),
                initial_state=interior_bundle.initial_state,
            )
        self._curvatures = curvatures
        return ODEResidualDerivatives.second_order(
            states,
            params,
            jacobian=jacobian,
            param=_ClosedParam(self),
            initial_state=interior_bundle.initial_state,
            initial_state_curvature=interior_bundle.initial_state_curvature,
            state_state=(STATED_ZERO if affine else _ClosedSecond(self, _STATE_STATE)),
            state_param=_ClosedSecond(self, _STATE_PARAM),
            param_param=_ClosedSecond(self, _PARAM_PARAM),
        )

    def _closure_curvatures(
        self,
    ) -> Optional[List[Tuple[_Weighted[Array], _Weighted[Array], _Weighted[Array]]]]:
        """Each closure's (state, mixed, parameter) curvature, or None when
        a closure lacks one."""
        curvatures = []
        for derivatives in self._closure_derivatives:
            state = derivatives.state_curvature
            mixed, param = derivatives.mixed, derivatives.param_curvature
            if state is None or mixed is None or param is None:
                return None
            curvatures.append((state, mixed, param))
        return curvatures

    def _second_operator(
        self, kind: int, state: Array, time: float, adj: Array
    ) -> DerivativeOperator[Array]:
        """One adjoint-weighted second derivative of f~: the interior's at
        the adjoint on its rows, plus each closure's at the adjoint on the
        rows it supplies."""
        bkd = self._bkd
        bundle = self._interior.derivatives()
        interior_fn = (bundle.state_state, bundle.state_param, bundle.param_param)[kind]
        if interior_fn is None:
            raise RuntimeError("the interior has no second derivatives")
        domain = self._states_desc if kind == _STATE_STATE else self._params_desc
        range_ = self._params_desc if kind == _PARAM_PARAM else self._states_desc
        masked = self._system.interior_mask() * adj
        matrix = interior_fn(state, time, masked).apply(bkd.eye(domain.size))
        for curvature, mask in zip(self._curvatures, self._system.closure_masks()):
            field = curvature[kind]
            if not isinstance(field, StatedZero):
                matrix = matrix + field(state, time, mask * adj)
        return BackendDenseOperator(matrix, domain, bkd, range=range_)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def system(self) -> CollocationSystem[Array]:
        return self._system

    def __call__(self, state: Array, time: float) -> Array:
        r""":math:`\tilde f(u, t; p)`. Shape: (nstates,)"""
        value = self._system.interior_mask() * self._interior(state, time)
        for closure, mask in zip(self._system.closures(), self._system.closure_masks()):
            value = value + mask * closure.residual(state, time)
        return value

    def _jacobian_matrix(self, state: Array, time: float) -> Array:
        interior = self._system.physics().interior_jacobian(state, time)
        matrix = self._system.interior_mask()[:, None] * interior
        for closure, mask in zip(self._system.closures(), self._system.closure_masks()):
            matrix = matrix + mask[:, None] * closure.jacobian(state, time)
        return matrix

    def _param_operator(self, state: Array, time: float) -> DerivativeOperator[Array]:
        bkd = self._bkd
        interior_param = self._interior.derivatives().param
        if interior_param is None:
            raise RuntimeError("the interior has no parameter Jacobian")
        nparams = self._params_desc.size
        dense = interior_param(state, time).apply(bkd.eye(nparams))
        matrix = self._system.interior_mask()[:, None] * dense
        for fn, mask in zip(self._param_fns, self._system.closure_masks()):
            if not isinstance(fn, StatedZero):
                matrix = matrix + mask[:, None] * fn(state, time)
        return BackendDenseOperator(
            matrix, self._params_desc, bkd, range=self._states_desc
        )

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives

    def set_param(self, param: Array) -> None:
        """Set the shared parameters."""
        self._interior.set_param(param)

    def parameters_version(self) -> int:
        return self._interior.parameters_version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(system={self._system!r})"
