r"""A functional of a transient solution as a map of the parameters.

:class:`TransientImplicitMap` is :math:`p \mapsto Q(y(p);\, p)`, where
:math:`y(p)` is the trajectory a stepping method computes for the
system a discretization builds at parameters :math:`p`. It works in the
stack's types alone: a parameter vector of the stack in, a value of
``nqoi`` entries out, and the derivatives as stack vectors. It knows
nothing of batches of samples or of backend arrays as inputs; the
:class:`~pyapprox.interface.functions.protocols.function.FunctionProtocol`
boundary is a wrapper's.

A value is computed during the solve by the functional's sink, so no
trajectory is kept. :meth:`TransientImplicitMap.point` records the
trajectory as well, for the tangent and adjoint sweeps of
:mod:`pyapprox.ode.trajectory_derivatives`; the functional the sweeps
differentiate is built from that store, so a time integral's recursion
weights are those of the trajectory solved.

The discretization (Galerkin, collocation) enters through
:class:`TransientDiscretizationProtocol`, the parameters through
:class:`ParameterVectorSetterProtocol`, the functional through
:class:`TrajectoryFunctionalSourceProtocol`; the method, the linear-solver
setups and the steps are injected.
"""

from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.implicitfunction.functional_terms import (
    has_first_derivatives,
    has_second_derivatives,
)
from pyapprox.implicitfunction.implicit_map import ImplicitMapDerivatives
from pyapprox.ode.accumulating_sink import MultiSink
from pyapprox.ode.functional_sink import FunctionalSink, PlainSum
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.pinned_rows import RowPinningProtocol
from pyapprox.ode.protocols.differentiable_stepper import StepperProtocol
from pyapprox.ode.setups import MassShiftStrategy
from pyapprox.ode.step_plan import StepPlanProtocol
from pyapprox.ode.stepper_derivatives import StepperDerivatives, StepWindow
from pyapprox.ode.stepping_methods import SteppingMethodProtocol
from pyapprox.ode.trajectory_derivatives import (
    InitialStateDerivatives,
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
    TrajectoryStoreProtocol,
)
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import FormatProtocol

V = TypeVar("V")
V_contra = TypeVar("V_contra", contravariant=True)
A = TypeVar("A", bound=ArrayProtocol)
A_co = TypeVar("A_co", bound=ArrayProtocol, covariant=True)
F = TypeVar("F", bound=FormatProtocol)


@runtime_checkable
class TransientDiscretizationProtocol(Protocol[V, A, F]):
    """What a transient map reads of a discretization: the system it
    steps, the rows pinned in each step, and how an initial state takes
    its essential data."""

    def system(self) -> MassFormSystem[V, A, F]:
        """The system stepped."""
        ...

    def pinning(self) -> Optional[RowPinningProtocol[V, F]]:
        """The rows pinned inside each step's solves; None for none."""
        ...

    def initial_state(self, state: V, time: float) -> V:
        """``state`` with its essential entries set to their data at
        ``time``, as a new vector."""
        ...


@runtime_checkable
class ParameterVectorSetterProtocol(Protocol[V_contra]):
    """Sets the parameters that the system, and any term of the
    functional that reads them, read: a vector of the stack's parameter
    space."""

    def nparams(self) -> int:
        """The number of parameters."""
        ...

    def set_param(self, param: V_contra) -> None:
        """Set the parameters to ``param``."""
        ...


@runtime_checkable
class ValueSinkProtocol(TrajectorySinkProtocol[V], Protocol[V, A_co]):
    """A sink that computes a value from the trajectory it receives."""

    def value(self) -> A_co:
        """The value of the trajectory received, shape ``(nqoi, 1)``."""
        ...


@runtime_checkable
class TrajectoryFunctionalSourceProtocol(Protocol[V, A]):
    """A functional of a trajectory, as a value sink for a solve and as
    the sweeps' functional of a recorded trajectory; both read the same
    terms. :class:`~pyapprox.ode.time_integral.TimeIntegral` is one."""

    def nqoi(self) -> int:
        """The number of entries of the value."""
        ...

    def sink(self) -> ValueSinkProtocol[V, A]:
        """A fresh sink for one solve."""
        ...

    def trajectory_functional(
        self, store: TrajectoryStoreProtocol[V]
    ) -> TrajectoryFunctional[V, A]:
        """The functional on the trajectory ``store`` recorded. Its terms
        are the same for every store, only their data (a time integral's
        recursion weights) depending on it, so their derivative bundles
        may be read from the functional on an empty store."""
        ...


class FixedTrajectoryFunctional(Generic[V, A]):
    """A :class:`~pyapprox.ode.trajectory_terms.TrajectoryFunctional` as a
    functional source: the same functional for every trajectory, its step
    terms' contributions summed.

    Parameters
    ----------
    functional : TrajectoryFunctional
        The functional.

    Raises
    ------
    TypeError
        If ``functional`` is not a ``TrajectoryFunctional``.
    """

    def __init__(self, functional: TrajectoryFunctional[V, A]) -> None:
        if not isinstance(functional, TrajectoryFunctional):
            raise TypeError(
                "functional must be a TrajectoryFunctional, got "
                f"{type(functional).__name__}"
            )
        self._functional = functional

    def nqoi(self) -> int:
        return self._functional.nqoi()

    def sink(self) -> FunctionalSink[V, A]:
        functional = self._functional
        return FunctionalSink(
            functional.zero,
            functional.parameter_terms,
            functional.state_terms,
            [(term, PlainSum()) for term in functional.step_terms],
        )

    def trajectory_functional(
        self, store: TrajectoryStoreProtocol[V]
    ) -> TrajectoryFunctional[V, A]:
        return self._functional

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nqoi={self.nqoi()})"


class _SummedSink(Generic[V, A]):
    """Passes the trajectory to each of several value sinks, its value
    their sum."""

    def __init__(self, sinks: Sequence[ValueSinkProtocol[V, A]]) -> None:
        self._sinks = tuple(sinks)
        self._forward = MultiSink(self._sinks)

    def append(self, time: float, state: V, unconstrained: Optional[V] = None) -> None:
        self._forward.append(time, state, unconstrained)

    def append_step(
        self, window: StepWindow[V], state: V, unconstrained: Optional[V] = None
    ) -> None:
        self._forward.append_step(window, state, unconstrained)

    def ntimes(self) -> int:
        return self._sinks[0].ntimes()

    def value(self) -> A:
        total = self._sinks[0].value()
        for sink in self._sinks[1:]:
            total = total + sink.value()
        return total


class SummedFunctional(Generic[V, A]):
    r"""The sum :math:`Q = \sum_k Q_k` of functional sources of one
    ``nqoi``, itself a source: a time integral plus a parameter
    regularization, say, which no one source holds.

    During a solve each source's sink receives the trajectory and the
    value is the sum of theirs. For the sweeps the functional on a store
    has every source's terms on that store.

    Parameters
    ----------
    sources : sequence of TrajectoryFunctionalSourceProtocol
        The :math:`Q_k`, at least one.

    Raises
    ------
    TypeError
        If a source does not satisfy ``TrajectoryFunctionalSourceProtocol``.
    ValueError
        If there is no source, or their ``nqoi`` differ.
    """

    def __init__(
        self, sources: Sequence[TrajectoryFunctionalSourceProtocol[V, A]]
    ) -> None:
        if not sources:
            raise ValueError("a SummedFunctional needs at least one source")
        for source in sources:
            if not isinstance(source, TrajectoryFunctionalSourceProtocol):
                raise TypeError(
                    "each source must satisfy TrajectoryFunctionalSourceProtocol, "
                    f"got {type(source).__name__}"
                )
        sizes = {source.nqoi() for source in sources}
        if len(sizes) != 1:
            raise ValueError(f"the sources must share one nqoi, got {sorted(sizes)}")
        self._sources = tuple(sources)

    def nqoi(self) -> int:
        return self._sources[0].nqoi()

    def sink(self) -> _SummedSink[V, A]:
        return _SummedSink([source.sink() for source in self._sources])

    def trajectory_functional(
        self, store: TrajectoryStoreProtocol[V]
    ) -> TrajectoryFunctional[V, A]:
        functionals = [source.trajectory_functional(store) for source in self._sources]
        return TrajectoryFunctional(
            functionals[0].zero,
            tuple(term for f in functionals for term in f.parameter_terms),
            tuple(term for f in functionals for term in f.state_terms),
            tuple(term for f in functionals for term in f.step_terms),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self._sources))})"


class ParameterFanOut(Generic[V]):
    """Sets the parameters on a setter and passes them to every reader: a
    parameter term of the functional, such as
    :class:`~pyapprox.implicitfunction.tikhonov_term.TikhonovTerm`, reads
    the parameters the system is solved at.

    Parameters
    ----------
    setter : ParameterVectorSetterProtocol
        Sets the system's parameters; its ``nparams`` is this one's.
    readers : sequence of callable
        Each called with every parameter vector set, after ``setter``.

    Raises
    ------
    TypeError
        If ``setter`` does not satisfy ``ParameterVectorSetterProtocol``
        or a reader is not callable.
    """

    def __init__(
        self,
        setter: ParameterVectorSetterProtocol[V],
        readers: Sequence[Callable[[V], None]],
    ) -> None:
        if not isinstance(setter, ParameterVectorSetterProtocol):
            raise TypeError(
                "setter must satisfy ParameterVectorSetterProtocol, got "
                f"{type(setter).__name__}"
            )
        for reader in readers:
            if not callable(reader):
                raise TypeError(f"each reader must be callable, got {reader!r}")
        self._setter, self._readers = setter, tuple(readers)

    def nparams(self) -> int:
        return self._setter.nparams()

    def set_param(self, param: V) -> None:
        self._setter.set_param(param)
        for reader in self._readers:
            reader(param)


class TransientPoint(Generic[V, A]):
    """One solved trajectory with its value and derivatives, so a value,
    tangents, gradients and Hessian-vector products at one parameter
    share one solve. Valid until the map's next solve, which moves the
    parameters the sweeps read; the sweeps then refuse it.

    Built by :meth:`TransientImplicitMap.point`.
    """

    def __init__(
        self,
        value: A,
        store: TrajectoryStoreProtocol[V],
        sweeps: TrajectoryDerivatives[V, A],
        functional: TrajectoryFunctional[V, A],
    ) -> None:
        self._value, self._store = value, store
        self._sweeps, self._functional = sweeps, functional

    def value(self) -> A:
        """:math:`Q`, shape ``(nqoi, 1)``."""
        return self._value

    def store(self) -> TrajectoryStoreProtocol[V]:
        """The recorded trajectory."""
        return self._store

    def functional(self) -> TrajectoryFunctional[V, A]:
        """The functional the sweeps differentiate."""
        return self._functional

    def tangent(self, direction: V) -> A:
        r""":math:`dQ/dp\, v` for a parameter vector ``direction``, every
        entry by one forward sweep. Shape ``(nqoi, 1)``."""
        return self._sweeps.tangent(self._store, self._functional, direction)

    def gradient(self, weights: Optional[A] = None) -> V:
        r""":math:`d(\omega^\top Q)/dp`, a parameter vector, by one
        backward sweep; ``weights`` :math:`\omega` of shape ``(nqoi, 1)``,
        required unless ``nqoi`` is 1."""
        return self._sweeps.gradient(self._store, self._functional, weights=weights)

    def hvp(self, direction: V, weights: Optional[A] = None) -> V:
        r""":math:`(d^2 \omega^\top Q/dp^2)\, v`, a parameter vector."""
        return self._sweeps.hvp(
            self._store, self._functional, direction, weights=weights
        )


def _stepper_order(param_space: bool, derivs: StepperDerivatives[V]) -> int:
    """The derivative order of a stepper's bundle the sweeps can use: 1
    with parameter derivatives, 2 with second derivatives too."""
    if not param_space or derivs.param is None:
        return 0
    # The three second-order fields come together.
    return 2 if derivs.param_param is not None else 1


def _initial_order(derivs: InitialStateDerivatives[V]) -> int:
    """The derivative order of an initial state's bundle."""
    if derivs.param_derivative is None:
        return 0
    return 2 if derivs.curvature is not None else 1


def _functional_order(functional: TrajectoryFunctional[V, A]) -> int:
    """The derivative order of a functional: that of its weakest term."""
    families = [term.derivatives() for term in functional.state_terms]
    # A step term's first derivatives come all three or none, and so do
    # its curvatures.
    steps = [term.derivatives() for term in functional.step_terms]
    if not (
        has_first_derivatives(functional.parameter_terms, ())
        and all(family.first is not None for family in families)
        and all(step.wrt_unknown is not None for step in steps)
    ):
        return 0
    if not (
        has_second_derivatives(functional.parameter_terms, ())
        and all(family.second is not None for family in families)
        and all(step.curvature_wrt_unknown is not None for step in steps)
    ):
        return 1
    return 2


class TransientImplicitMap(Generic[V, A, F]):
    r""":math:`p \mapsto Q(y(p);\, p)` for the trajectory :math:`y(p)` of a
    discretization stepped by a method (see the module docstring).

    Each evaluation sets the parameters, clears ``setups``, sets the
    initial state's essential entries to their data at the plan's origin,
    and steps through ``steps``.

    Which derivatives the map has (:meth:`derivatives`) is fixed here, as
    the weakest of its parts' bundles: a stepper built for the
    discretization's system, the initial state, and the functional's
    terms. The steps contribute nothing, so a plan that chooses its steps
    during the solve differentiates as a fixed one does.

    Parameters
    ----------
    discretization : TransientDiscretizationProtocol
        The system stepped and how its initial state is set.
    parameters : ParameterVectorSetterProtocol
        Sets the parameters the system and the functional's terms read.
    method : SteppingMethodProtocol
        The stepping method, with its nonlinear solver when implicit.
    setups : MassShiftStrategy
        Sets up the stepper's linear solves; its solver pairs with the
        system's. Cleared at every evaluation.
    steps : StepPlanProtocol
        The steps, from the time of the initial state.
    init_state : V
        The initial state, before its essential entries are set.
    functional : TrajectoryFunctionalSourceProtocol
        :math:`Q`.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    """

    _derivatives: ImplicitMapDerivatives[TransientPoint[V, A], V, A]

    def __init__(
        self,
        discretization: TransientDiscretizationProtocol[V, A, F],
        parameters: ParameterVectorSetterProtocol[V],
        method: SteppingMethodProtocol[V, A],
        setups: MassShiftStrategy[V, A, F],
        steps: StepPlanProtocol,
        init_state: V,
        functional: TrajectoryFunctionalSourceProtocol[V, A],
    ) -> None:
        for argument, protocol, name in (
            (discretization, TransientDiscretizationProtocol, "discretization"),
            (parameters, ParameterVectorSetterProtocol, "parameters"),
            (method, SteppingMethodProtocol, "method"),
            (steps, StepPlanProtocol, "steps"),
            (functional, TrajectoryFunctionalSourceProtocol, "functional"),
        ):
            if not isinstance(argument, protocol):
                raise TypeError(
                    f"{name} must satisfy {protocol.__name__}, got "
                    f"{type(argument).__name__}"
                )
        self._discretization, self._parameters = discretization, parameters
        self._method, self._setups, self._steps = method, setups, steps
        self._init_state, self._functional = init_state, functional
        # A stepper built only to read its bundle: building one solves
        # nothing, and a bundle's fields do not change with the parameters.
        system = discretization.system()
        stepper = method.stepper(system, setups, discretization.pinning())
        order = min(
            _stepper_order(stepper.param_space() is not None, stepper.derivatives()),
            _initial_order(ResidualInitialState(system.rhs, system.form).derivatives()),
            _functional_order(
                functional.trajectory_functional(InMemoryTrajectoryStore())
            ),
        )
        self._derivatives = (
            ImplicitMapDerivatives.none()
            if order == 0
            else ImplicitMapDerivatives.first_order(
                TransientPoint.tangent, TransientPoint.gradient
            )
            if order == 1
            else ImplicitMapDerivatives.second_order(
                TransientPoint.tangent, TransientPoint.gradient, TransientPoint.hvp
            )
        )

    def derivatives(self) -> ImplicitMapDerivatives[TransientPoint[V, A], V, A]:
        """Which derivatives a point has, as callables over it; fixed when
        the map is built."""
        return self._derivatives

    def nparams(self) -> int:
        """The number of parameters."""
        return self._parameters.nparams()

    def nqoi(self) -> int:
        """The number of entries of :math:`Q`."""
        return self._functional.nqoi()

    def value(self, param: V) -> A:
        """:math:`Q` at ``param``, computed during the solve, keeping no
        trajectory. Shape ``(nqoi, 1)``."""
        sink = self._functional.sink()
        self._solve(param, [sink])
        return sink.value()

    def point(self, param: V) -> TransientPoint[V, A]:
        """The solve at ``param`` with its trajectory recorded, for the
        derivatives."""
        sink = self._functional.sink()
        store: InMemoryTrajectoryStore[V] = InMemoryTrajectoryStore()
        system, stepper = self._solve(param, [sink, store])
        sweeps: TrajectoryDerivatives[V, A] = TrajectoryDerivatives(
            stepper, None, ResidualInitialState(system.rhs, system.form)
        )
        return TransientPoint(
            sink.value(), store, sweeps, self._functional.trajectory_functional(store)
        )

    def _solve(
        self, param: V, sinks: Sequence[TrajectorySinkProtocol[V]]
    ) -> Tuple[MassFormSystem[V, A, F], StepperProtocol[V, A]]:
        """Solve at ``param`` into ``sinks``; the system and the stepper
        the solve used."""
        discretization = self._discretization
        self._parameters.set_param(param)
        self._setups.clear()
        system = discretization.system()
        stepper = self._method.stepper(system, self._setups, discretization.pinning())
        start = discretization.initial_state(self._init_state, self._steps.origin())
        Integrator(stepper).solve(start, self._steps, MultiSink(sinks))
        return system, stepper

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nparams={self.nparams()}, nqoi={self.nqoi()}, "
            f"method={self._method.name()})"
        )
