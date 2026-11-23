r"""The first-order system of a second-order one, on any stack.

:func:`second_order_system` builds, from the first-order system
:math:`M \dot u = f(u, t; p)` of a stack's restoring force, an optional
velocity-dependent force :math:`g(u, v, t)` and a storage of the two state
blocks, the system of :math:`w = [u; v]`

.. math::

    \begin{bmatrix} I & 0 \\ 0 & M \end{bmatrix} \dot w
    = R(w, t) = \begin{bmatrix} v \\ f(u, t) + g(u, v, t) \end{bmatrix},
    \qquad
    \frac{\partial R}{\partial w}
    = \begin{bmatrix} 0 & I \\ f_u + g_u & g_v \end{bmatrix}.

The force splits additively, so a conservative physics supplies only
:math:`f`, with the contract it already has as a first-order right-hand
side, and damping (linear, nonlinear or gyroscopic) is a separate term
summed in one place.

How the blocks are stored is the caller's (:class:`StateBlocksProtocol`):
the states stay vectors of the restoring system's stack, and the storage
splits and joins them and assembles the block mass and Jacobian in a format
its solver takes. An assembled :math:`2n` operator and an elimination of
one block that solves at size :math:`n` are two storages of the same
reduction; the method stepping it is chosen independently.

The parameter derivatives are lifted from :math:`f`'s bundle: :math:`p`
enters :math:`R` through :math:`f` only (a velocity force declares itself
independent of :math:`p`), so with :math:`\lambda = [\lambda_u;
\lambda_v]`

.. math::

    R_p = \begin{bmatrix} 0 \\ f_p \end{bmatrix},\quad
    \lambda^\top R_{ww} = \begin{bmatrix} \lambda_v^\top f_{uu} & 0 \\
    0 & 0 \end{bmatrix},\quad
    \lambda^\top R_{wp} = \begin{bmatrix} \lambda_v^\top f_{up} \\ 0
    \end{bmatrix},\quad
    \lambda^\top R_{pp} = \lambda_v^\top f_{pp},

the second order needing :math:`g` affine in the state. The initial state
:math:`w_0 = [u_0; v_0]` takes :math:`\partial u_0/\partial p` from
:math:`f`'s bundle and :math:`\partial v_0/\partial p` from the caller
(:class:`InitialVelocityDerivatives`).
"""

from dataclasses import dataclass
from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from pyapprox.ode.mass_form_system import (
    MassFormSystem,
    RightHandSideProtocol,
    parameters_version_reader,
)
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    CurvatureFn,
    DerivativeOperator,
    InitialStateFn,
    Jacobian,
    ODEResidualDerivatives,
    StatedZero,
)
from pyapprox.ode.residual_form import ResidualFormProtocol
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    InnerProductSpaceProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearizedOperator,
    LinearSolverProtocol,
    MatrixFree,
    OperatorProperties,
    ScaledSum,
    SpaceDescriptor,
    VectorSpaceProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)
G = TypeVar("G", bound=FormatProtocol)
A_co = TypeVar("A_co", bound=ArrayProtocol, covariant=True)

Block = Optional[ActionOperatorProtocol[V, F]]
"""One block of a 2 x 2 block operator; None for a zero block."""

Blocks = Tuple[Tuple[Block[V, F], Block[V, F]], Tuple[Block[V, F], Block[V, F]]]
"""The blocks of a 2 x 2 block operator, row by row."""


@runtime_checkable
class StateBlocksProtocol(Protocol[V, A_co, F, G]):
    r"""How the state :math:`w = [u; v]` and the block operators are
    stored, on the restoring system's stack.

    Vectors are the stack's ``V``, single states in the residual's form
    and blocks of columns alike, the displacement first. Operators on one
    block are in format ``F``; assembled ones in ``G``, which the
    storage's solver takes.
    """

    def half_space(self) -> InnerProductSpaceProtocol[V, A_co]:
        """The space of one block, size :math:`n`."""
        ...

    def space(self) -> InnerProductSpaceProtocol[V, A_co]:
        """The space of :math:`w`, size :math:`2n`."""
        ...

    def displacement(self, state: V) -> V:
        """The first block of ``state``."""
        ...

    def velocity(self, state: V) -> V:
        """The second block of ``state``."""
        ...

    def join(self, displacement: V, velocity: V) -> V:
        """The vector with blocks ``displacement`` and ``velocity``."""
        ...

    def identity(self) -> ActionOperatorProtocol[V, F]:
        """The identity on one block."""
        ...

    def assemble(self, blocks: Blocks[V, F]) -> ActionOperatorProtocol[V, G]:
        """The operator with these blocks, a None block zero, declaring
        the properties its blocks imply
        (:func:`~pyapprox.util.linalg.operators.block_properties`)."""
        ...

    def solver(self) -> LinearSolverProtocol[V, G]:
        """Solves with the assembled block mass and stage operators."""
        ...


@dataclass(frozen=True)
class VelocityForceDerivatives(Generic[V, F]):
    r"""The optional derivatives of a velocity force :math:`g(u, v, t)`.

    Build with the named constructors. ``param`` is :data:`STATED_ZERO`
    when :math:`g` does not depend on the parameters, and
    ``state_curvature`` when :math:`g` is affine in :math:`(u, v)`; None
    means not supplied, and the reduced system then has no derivatives of
    that order.

    Attributes
    ----------
    wrt_displacement, wrt_velocity : Jacobian or None
        :math:`g_u` and :math:`g_v`, together or not at all. Each is
        called with the whole state :math:`w = [u; v]`, since either may
        depend on both blocks, and returns an operator on one block.
    param : STATED_ZERO or None
        :math:`g_p`; requires the Jacobians.
    state_curvature : STATED_ZERO or None
        The second state derivatives of :math:`g`; requires ``param`` and
        Jacobians that cannot see the state.
    """

    wrt_displacement: Optional[Jacobian[V, F]] = None
    wrt_velocity: Optional[Jacobian[V, F]] = None
    param: Optional[StatedZero] = None
    state_curvature: Optional[StatedZero] = None

    def __post_init__(self) -> None:
        if (self.wrt_displacement is None) != (self.wrt_velocity is None):
            raise ValueError("wrt_displacement and wrt_velocity come together")
        if self.param is not None and self.wrt_displacement is None:
            raise ValueError("param requires the state Jacobians")
        if self.state_curvature is not None:
            if self.param is None:
                raise ValueError("state_curvature requires param")
            if not (_affine(self.wrt_displacement) and _affine(self.wrt_velocity)):
                raise ValueError(
                    "state_curvature is STATED_ZERO only for g affine in the "
                    "state: both Jacobians must be state-free or constant"
                )

    @staticmethod
    def none() -> "VelocityForceDerivatives[V, F]":
        """No derivatives: explicit stepping only."""
        return VelocityForceDerivatives()

    @staticmethod
    def state_only(
        wrt_displacement: Jacobian[V, F], wrt_velocity: Jacobian[V, F]
    ) -> "VelocityForceDerivatives[V, F]":
        """The state Jacobians; the parameter dependence not declared."""
        return VelocityForceDerivatives(wrt_displacement, wrt_velocity)

    @staticmethod
    def parameter_free(
        wrt_displacement: Jacobian[V, F],
        wrt_velocity: Jacobian[V, F],
        affine: bool,
    ) -> "VelocityForceDerivatives[V, F]":
        """The state Jacobians of a :math:`g` independent of the
        parameters; ``affine`` declares it affine in the state, which the
        Jacobians must show."""
        return VelocityForceDerivatives(
            wrt_displacement,
            wrt_velocity,
            STATED_ZERO,
            STATED_ZERO if affine else None,
        )


def _affine(jacobian: Optional[Jacobian[V, F]]) -> bool:
    return jacobian is not None and jacobian.affine()


@runtime_checkable
class VelocityForceProtocol(Protocol[V, F]):
    r""":math:`g(u, v, t)` and its derivatives bundle; states in the
    residual's form."""

    def __call__(self, displacement: V, velocity: V, time: float) -> V: ...

    def derivatives(self) -> VelocityForceDerivatives[V, F]: ...


@dataclass(frozen=True)
class InitialVelocityDerivatives(Generic[V]):
    r""":math:`\partial v_0/\partial p` and its curvature, for the reduced
    system's initial state.

    Attributes
    ----------
    param_derivative : callable or STATED_ZERO
        ``() ->`` the operator :math:`\partial v_0/\partial p`, parameter
        to state.
    curvature : callable, STATED_ZERO or None
        ``weight ->`` :math:`\sum_i w_i\,\partial^2 v_{0,i}/\partial p^2`,
        parameter to parameter; None if not supplied, and the reduced
        system then has no second derivatives.
    """

    param_derivative: Union[InitialStateFn[V], StatedZero]
    curvature: Optional[Union[CurvatureFn[V], StatedZero]]

    @staticmethod
    def independent() -> "InitialVelocityDerivatives[V]":
        """An initial velocity independent of the parameters."""
        return InitialVelocityDerivatives(STATED_ZERO, STATED_ZERO)


class _Stacked(Generic[V, A, F, G]):
    r"""Parameter to :math:`w`: :math:`x \mapsto [T x; B x]`, a missing
    block zero."""

    def __init__(
        self,
        top: Optional[DerivativeOperator[V]],
        bottom: Optional[DerivativeOperator[V]],
        blocks: StateBlocksProtocol[V, A, F, G],
        domain_space: VectorSpaceProtocol[V],
    ) -> None:
        self._top, self._bottom = top, bottom
        self._blocks, self._domain_space = blocks, domain_space

    def domain(self) -> SpaceDescriptor:
        return self._domain_space.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._blocks.space().descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        half = self._blocks.half_space()
        ncols = self._domain_space.ncols(x)
        top = half.zeros(ncols) if self._top is None else self._top.apply(x)
        bottom = half.zeros(ncols) if self._bottom is None else self._bottom.apply(x)
        return self._blocks.join(top, bottom)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        blocks, space = self._blocks, self._domain_space
        result = space.zeros(blocks.space().ncols(y))
        if self._top is not None:
            top = self._top.apply_transpose(blocks.displacement(y))
            result = space.axpby(1.0, top, 1.0, result)
        if self._bottom is not None:
            bottom = self._bottom.apply_transpose(blocks.velocity(y))
            result = space.axpby(1.0, bottom, 1.0, result)
        return result


class _DisplacementBlock(Generic[V, A, F, G]):
    r""":math:`w \mapsto [B u; 0]`: an operator on the displacement block
    alone."""

    def __init__(
        self, op: DerivativeOperator[V], blocks: StateBlocksProtocol[V, A, F, G]
    ) -> None:
        self._op, self._blocks = op, blocks

    def domain(self) -> SpaceDescriptor:
        return self._blocks.space().descriptor()

    def range(self) -> SpaceDescriptor:
        return self._blocks.space().descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties(symmetric=self._op.properties().symmetric)

    def format(self) -> MatrixFree:
        return MatrixFree()

    def _embed(self, image: V) -> V:
        zero = self._blocks.half_space().zeros_like(image)
        return self._blocks.join(image, zero)

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._embed(self._op.apply(self._blocks.displacement(x)))

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self._embed(self._op.apply_transpose(self._blocks.displacement(y)))


class SecondOrderRightHandSide(Generic[V, A, F, G]):
    r""":math:`R(w, t)` of the reduced system and its derivatives bundle
    (see the module docstring). Built by :func:`second_order_system`.

    Reports the restoring force's parameter version, so a stepper's setups
    see its ``set_param`` calls. Which derivatives the bundle has is fixed
    at construction, as the weakest of its parts'.

    Parameters
    ----------
    restoring : RightHandSideProtocol
        :math:`f`, with its bundle.
    velocity_force : VelocityForceProtocol or None
        :math:`g`; None for an undamped system.
    blocks : StateBlocksProtocol
        The storage of the blocks.
    form : ResidualFormProtocol
        The restoring system's residual form, for the zero state at which
        a Jacobian that cannot see the state is evaluated.
    initial_velocity : InitialVelocityDerivatives
        :math:`\partial v_0/\partial p`.
    """

    def __init__(
        self,
        restoring: RightHandSideProtocol[V, F],
        velocity_force: Optional[VelocityForceProtocol[V, F]],
        blocks: StateBlocksProtocol[V, A, F, G],
        form: ResidualFormProtocol[V, V],
        initial_velocity: InitialVelocityDerivatives[V],
    ) -> None:
        self._restoring, self._velocity_force = restoring, velocity_force
        self._blocks, self._initial_velocity = blocks, initial_velocity
        self._form = form
        self._version = parameters_version_reader(restoring)
        zero = form.to_residual(blocks.half_space().zeros())
        self._zero_state = blocks.join(zero, zero)
        self._param_space = restoring.derivatives().param_space
        self._derivatives = self._bundle()

    def parameters_version(self) -> int:
        """The restoring force's parameter version."""
        return self._version()

    def __call__(self, state: V, time: float) -> V:
        blocks = self._blocks
        displacement, velocity = blocks.displacement(state), blocks.velocity(state)
        force = self._restoring(displacement, time)
        if self._velocity_force is not None:
            form, half = self._form, blocks.half_space()
            damping = self._velocity_force(displacement, velocity, time)
            total = half.axpby(
                1.0, form.from_residual(damping), 1.0, form.from_residual(force)
            )
            force = form.to_residual(total)
        return blocks.join(velocity, force)

    def derivatives(self) -> ODEResidualDerivatives[V, G]:
        """The lifted bundle; its fields read the restoring force's bundle
        when called, so they follow its parameters."""
        return self._derivatives

    def _bundle(self) -> ODEResidualDerivatives[V, G]:
        restoring = self._restoring.derivatives()
        undamped = self._velocity_force is None
        damping: VelocityForceDerivatives[V, F] = (
            VelocityForceDerivatives()
            if self._velocity_force is None
            else self._velocity_force.derivatives()
        )
        if restoring.jacobian is None or not (
            undamped or damping.wrt_displacement is not None
        ):
            return ODEResidualDerivatives.none()
        jacobian = self._jacobian(restoring.jacobian, damping)
        params = self._param_space
        if params is None or not (undamped or damping.param is not None):
            return ODEResidualDerivatives.state_only(jacobian)
        space = self._blocks.space()
        if not (
            restoring.state_state is not None
            and restoring.initial_state_curvature is not None
            and (undamped or damping.state_curvature is not None)
            and self._initial_velocity.curvature is not None
        ):
            return ODEResidualDerivatives.first_order(
                space, params, jacobian, self._param, self._initial_state
            )
        return ODEResidualDerivatives.second_order(
            space,
            params,
            jacobian,
            self._param,
            self._initial_state,
            self._initial_curvature,
            # R is affine exactly when every term is: its curvature is then
            # the stated zero the bundle requires, not f's wrapped.
            STATED_ZERO if jacobian.affine() else self._state_state,
            self._state_param,
            self._param_param,
        )

    def _restoring_bundle(self) -> ODEResidualDerivatives[V, F]:
        return self._restoring.derivatives()

    def _require_params(self) -> VectorSpaceProtocol[V]:
        # Read only by fields the bundle has only with a parameter space.
        if self._param_space is None:
            raise RuntimeError("the restoring force has no parameter space")
        return self._param_space

    # The Jacobian, of the kind its terms allow: constant when every term's
    # is, state-free (R affine) when every term's is, general otherwise.

    def _jacobian(
        self, restoring: Jacobian[V, F], damping: VelocityForceDerivatives[V, F]
    ) -> Jacobian[V, G]:
        terms = [restoring]
        if damping.wrt_displacement is not None and damping.wrt_velocity is not None:
            terms += [damping.wrt_displacement, damping.wrt_velocity]
        if all(term.time_invariant() for term in terms):
            return Jacobian.constant(self._constant_jacobian)
        if all(term.affine() for term in terms):
            return Jacobian.state_free(self._state_free_jacobian)
        return Jacobian.general(self._assembled)

    def _constant_jacobian(
        self, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, G]:
        return self._assembled(self._zero_state, 0.0, point)

    def _state_free_jacobian(
        self, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, G]:
        return self._assembled(self._zero_state, time, point)

    def _assembled(
        self, state: V, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, G]:
        blocks = self._blocks
        jacobian = self._restoring_bundle().jacobian
        if jacobian is None:
            raise RuntimeError("the restoring force's bundle has no jacobian")
        wrt_u: ActionOperatorProtocol[V, F] = jacobian(
            blocks.displacement(state), time, point
        )
        wrt_v: Block[V, F] = None
        if self._velocity_force is not None:
            damping = self._velocity_force.derivatives()
            if damping.wrt_displacement is None or damping.wrt_velocity is None:
                raise RuntimeError("the velocity force's bundle has no Jacobians")
            terms: List[Tuple[float, ActionOperatorProtocol[V, F]]] = [
                (1.0, wrt_u),
                (1.0, damping.wrt_displacement(state, time, point)),
            ]
            wrt_u = ScaledSum(terms, blocks.half_space())
            wrt_v = damping.wrt_velocity(state, time, point)
        op = blocks.assemble(((None, blocks.identity()), (wrt_u, wrt_v)))
        return LinearizedOperator(op, point)

    # The parameter derivatives, lifted from f's (see the module docstring).

    def _param(self, state: V, time: float) -> DerivativeOperator[V]:
        param = self._restoring_bundle().param
        if param is None:
            raise RuntimeError("the restoring force's bundle has no param")
        return _Stacked(
            None,
            param(self._blocks.displacement(state), time),
            self._blocks,
            self._require_params(),
        )

    def _initial_state(self) -> DerivativeOperator[V]:
        initial = self._restoring_bundle().initial_state
        if initial is None:
            raise RuntimeError("the restoring force's bundle has no initial_state")
        velocity = self._initial_velocity.param_derivative
        return _Stacked(
            initial(),
            None if isinstance(velocity, StatedZero) else velocity(),
            self._blocks,
            self._require_params(),
        )

    def _initial_curvature(self, weight: V) -> DerivativeOperator[V]:
        curvature = self._restoring_bundle().initial_state_curvature
        if curvature is None:
            raise RuntimeError(
                "the restoring force's bundle has no initial_state_curvature"
            )
        blocks = self._blocks
        displacement = curvature(blocks.displacement(weight))
        velocity = self._initial_velocity.curvature
        if velocity is None or isinstance(velocity, StatedZero):
            return displacement
        terms: List[Tuple[float, DerivativeOperator[V]]] = [
            (1.0, displacement),
            (1.0, velocity(blocks.velocity(weight))),
        ]
        return _MatrixFreeSum(terms, self._require_params())

    def _state_state(self, state: V, time: float, adj: V) -> DerivativeOperator[V]:
        field = self._restoring_bundle().state_state
        if field is None:
            raise RuntimeError("the restoring force's bundle has no state_state")
        blocks = self._blocks
        return _DisplacementBlock(
            field(blocks.displacement(state), time, blocks.velocity(adj)), blocks
        )

    def _state_param(self, state: V, time: float, adj: V) -> DerivativeOperator[V]:
        field = self._restoring_bundle().state_param
        if field is None:
            raise RuntimeError("the restoring force's bundle has no state_param")
        blocks = self._blocks
        return _Stacked(
            field(blocks.displacement(state), time, blocks.velocity(adj)),
            None,
            blocks,
            self._require_params(),
        )

    def _param_param(self, state: V, time: float, adj: V) -> DerivativeOperator[V]:
        field = self._restoring_bundle().param_param
        if field is None:
            raise RuntimeError("the restoring force's bundle has no param_param")
        blocks = self._blocks
        return field(blocks.displacement(state), time, blocks.velocity(adj))


class _MatrixFreeSum(Generic[V]):
    """The sum of derivative operators of any formats, by action."""

    def __init__(
        self,
        terms: List[Tuple[float, DerivativeOperator[V]]],
        space: VectorSpaceProtocol[V],
    ) -> None:
        self._sum: ScaledSum[V, FormatProtocol] = ScaledSum(terms, space)

    def domain(self) -> SpaceDescriptor:
        return self._sum.domain()

    def range(self) -> SpaceDescriptor:
        return self._sum.range()

    def properties(self) -> OperatorProperties:
        return self._sum.properties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._sum.apply(x)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self._sum.apply_transpose(y)


def second_order_system(
    restoring: MassFormSystem[V, A, F],
    blocks: StateBlocksProtocol[V, A, F, G],
    initial_velocity: InitialVelocityDerivatives[V],
    velocity_force: Optional[VelocityForceProtocol[V, F]] = None,
) -> MassFormSystem[V, A, G]:
    r"""The first-order system of :math:`M \ddot u = f(u, t; p) + g(u,
    \dot u, t)`, in :math:`w = [u; v]`.

    Parameters
    ----------
    restoring : MassFormSystem
        :math:`M \dot u = f(u, t; p)`: its mass is the inertia, its
        right-hand side the restoring force, its form the residual form of
        each block.
    blocks : StateBlocksProtocol
        How the blocks are stored and solved with; its block space must
        be the restoring system's.
    initial_velocity : InitialVelocityDerivatives
        :math:`\partial v_0/\partial p`, read only when :math:`f` has
        parameter derivatives.
    velocity_force : VelocityForceProtocol, optional
        :math:`g`; none for an undamped system.

    Raises
    ------
    TypeError
        If ``blocks`` or ``velocity_force`` does not satisfy its protocol.
    ValueError
        If the block space differs from the restoring system's space.
    """
    if not isinstance(blocks, StateBlocksProtocol):
        raise TypeError(
            f"blocks must satisfy StateBlocksProtocol, got {type(blocks).__name__}"
        )
    if velocity_force is not None and not isinstance(
        velocity_force, VelocityForceProtocol
    ):
        raise TypeError(
            "velocity_force must satisfy VelocityForceProtocol, got "
            f"{type(velocity_force).__name__}"
        )
    restoring.space.descriptor().require_same(
        blocks.half_space().descriptor(), "space of one block"
    )
    # Block-diagonal, so it keeps the mass's properties.
    block_mass = blocks.assemble(((blocks.identity(), None), (None, restoring.mass)))
    rhs = SecondOrderRightHandSide(
        restoring.rhs, velocity_force, blocks, restoring.form, initial_velocity
    )
    return MassFormSystem(
        rhs=rhs,
        mass=block_mass,
        space=blocks.space(),
        solver=blocks.solver(),
        form=restoring.form,
    )
