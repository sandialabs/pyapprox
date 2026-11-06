r"""Pinned rows in differentiated form: rows that evolve by the data's rate.

A system :math:`M \dot y = f(y, t; p)` whose rows :math:`R` are constrained
to data, :math:`y_R = g(t)`, can impose the constraint undifferentiated,
pinning the rows of a block of the step equation
(:mod:`pyapprox.ode.pinned_rows`), or differentiated, replacing each row's
equation by :math:`\dot y_R = \dot g(t)`. A step's blocks that are not
pinned need the differentiated form: the system's own rows :math:`R` are
not valid evolution equations (an assembled Dirichlet row lacks its
boundary flux), and a consistent mass solve would spread them into every
other row. With the step end pinned as well, the stored states hold the
data exactly; with every block pinned, no rate is needed and this layer is
not used.

The differentiated system, built over the system it replaces rows of:

- each right-hand side :math:`f` has the rows :math:`R` of its value
  zeroed, of its Jacobian and parameter Jacobian zeroed, and of the
  adjoint weighting its second derivatives zeroed
  (:class:`ZeroedRowsRightHandSide`); the initial-state derivatives are
  not rows of :math:`f` and pass through;
- exactly one right-hand side carries the rate, its rows :math:`R` set to
  :math:`\dot g(t; p)` (:class:`DifferentiatedRowsRightHandSide`), with
  the derivatives of the zeroed one plus, for data depending on the
  parameters, the rate's parameter derivatives on the rows (the rate
  never sees the state);
- the mass is edited once, by an injected treatment, which also fixes
  where the coupling :math:`M_{ID}\, \dot g` sits: in the mass
  (:class:`CoupledRows`, the mass :math:`[M_{II}\; M_{ID};\, 0\; I]`) or
  in the rate-carrying right-hand side (:class:`SymmetricRows`, the mass
  :math:`[M_{II}\; 0;\, 0\; I]` and :math:`-M_{ID}\, \dot g` on the other
  rows of :math:`f`).

:func:`differentiated_rows` composes them for a system with one right-hand
side; a system split into partitions sharing :math:`M` applies the rate to
one partition's right-hand side and zeroes the rows of the others. The
rows' values are written through the residual form, the derivatives'
rows through the stack's row set.
"""

from dataclasses import replace
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.mass_form_system import MassFormSystem, RightHandSideProtocol
from pyapprox.ode.pinned_rows import (
    PinnedData,
    PinnedRowsParamDerivatives,
    PinnedRowsProtocol,
)
from pyapprox.ode.residual_derivatives import (
    DerivativeOperator,
    ODEResidualDerivatives,
    ParamFn,
    SecondDerivativeFn,
    StatedZero,
    map_jacobian,
)
from pyapprox.ode.residual_form import ResidualFormProtocol
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    InnerProductSpaceProtocol,
    JacobianSnapshotProtocol,
    LinearizedOperator,
    Product,
    RowReplacedOperator,
    RowSetProtocol,
    RowsZeroedOperator,
    ScaledSum,
    SymmetricEliminatedOperator,
    VectorSpaceProtocol,
    ZeroOperator,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)

RatesFn = Callable[[float], V]
r""":math:`t \mapsto \dot g(t)`, a single column whose rows :math:`R` hold
the rate."""


@runtime_checkable
class _Versioned(Protocol):
    def parameters_version(self) -> int: ...


def _unversioned() -> int:
    return 0


# ----------------------------------------------------------------------
# The zeroed rows of a right-hand side's derivatives
# ----------------------------------------------------------------------


class _ZeroedSnapshotRows(Generic[V, F]):
    """A Jacobian snapshot with the rows zeroed, under the same label."""

    def __init__(self, rows: RowSetProtocol[V]) -> None:
        self._rows = rows

    def __call__(
        self, snapshot: JacobianSnapshotProtocol[V, F]
    ) -> JacobianSnapshotProtocol[V, F]:
        return LinearizedOperator(
            RowsZeroedOperator(snapshot, self._rows), snapshot.linearization()
        )


class _ZeroedParam(Generic[V]):
    """A parameter Jacobian with the rows zeroed."""

    def __init__(self, fn: ParamFn[V], rows: RowSetProtocol[V]) -> None:
        self._fn, self._rows = fn, rows

    def __call__(self, state: V, time: float) -> DerivativeOperator[V]:
        return RowsZeroedOperator(self._fn(state, time), self._rows)


class _ZeroedAdjoint(Generic[V]):
    """A second derivative weighted by the adjoint with its rows zeroed:
    the rows carry no curvature."""

    def __init__(
        self,
        fn: SecondDerivativeFn[V],
        rows: RowSetProtocol[V],
        form: ResidualFormProtocol[V, V],
    ) -> None:
        self._fn, self._rows, self._form = fn, rows, form

    def __call__(self, state: V, time: float, adj: V) -> DerivativeOperator[V]:
        zeroed = self._rows.zero_rows(self._form.from_residual(adj))
        return self._fn(state, time, self._form.to_residual(zeroed))


def _zeroed_derivatives(
    bundle: ODEResidualDerivatives[V, F],
    rows: RowSetProtocol[V],
    form: ResidualFormProtocol[V, V],
) -> ODEResidualDerivatives[V, F]:
    """``bundle`` for :math:`f` with the rows zeroed: every field present
    is adjusted, absent ones stay absent, and a stated zero stays one."""
    jacobian = bundle.jacobian
    if jacobian is None:
        raise TypeError("a right-hand side's bundle needs a Jacobian")
    param = bundle.param
    state_state = bundle.state_state
    state_param = bundle.state_param
    param_param = bundle.param_param
    zero_rows: _ZeroedSnapshotRows[V, F] = _ZeroedSnapshotRows(rows)
    return replace(
        bundle,
        jacobian=map_jacobian(jacobian, zero_rows),
        param=None if param is None else _ZeroedParam(param, rows),
        # An affine f's state curvature is the stated zero, kept as
        # declared.
        state_state=(
            state_state
            if state_state is None or jacobian.affine()
            else _ZeroedAdjoint(state_state, rows, form)
        ),
        state_param=(
            None if state_param is None else _ZeroedAdjoint(state_param, rows, form)
        ),
        param_param=(
            None if param_param is None else _ZeroedAdjoint(param_param, rows, form)
        ),
    )


class ZeroedRowsRightHandSide(Generic[V, F]):
    r""":math:`f` with the rows :math:`R` zeroed, value and derivatives:
    a right-hand side that leaves the rows to another partition, or to a
    rate set over it.

    Parameters
    ----------
    rhs : RightHandSideProtocol
        :math:`f`.
    rows : RowSetProtocol
        The rows :math:`R`, of the state space.
    form : ResidualFormProtocol
        Converts the space's vectors to the right-hand side's state form
        and back.

    Raises
    ------
    TypeError
        If ``rhs`` does not satisfy
        :class:`~pyapprox.ode.mass_form_system.RightHandSideProtocol` or
        its bundle has no Jacobian.
    """

    def __init__(
        self,
        rhs: RightHandSideProtocol[V, F],
        rows: RowSetProtocol[V],
        form: ResidualFormProtocol[V, V],
    ) -> None:
        if not isinstance(rhs, RightHandSideProtocol):
            raise TypeError(
                "rhs must satisfy RightHandSideProtocol, got "
                f"{type(rhs).__name__}"
            )
        self._rhs, self._rows, self._form = rhs, rows, form
        self._version: Callable[[], int] = (
            rhs.parameters_version if isinstance(rhs, _Versioned) else _unversioned
        )
        # Which fields are present is fixed for the right-hand side's life.
        self._derivatives = _zeroed_derivatives(rhs.derivatives(), rows, form)

    def __call__(self, state: V, time: float) -> V:
        value = self._form.from_residual(self._rhs(state, time))
        return self._form.to_residual(self._rows.zero_rows(value))

    def derivatives(self) -> ODEResidualDerivatives[V, F]:
        return self._derivatives

    def parameters_version(self) -> int:
        """The wrapped right-hand side's version; zero if it has none."""
        return self._version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._rhs!r}, nrows={self._rows.count()})"


class _RatedParam(Generic[V]):
    r""":math:`P f_p + D(t)`, :math:`D` the rate's parameter Jacobian on
    the rows as the right-hand side places it."""

    def __init__(
        self,
        base: ParamFn[V],
        owner: "DifferentiatedRowsRightHandSide[V, Any]",
        states: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> None:
        self._base, self._owner = base, owner
        self._states, self._params = states, params

    def __call__(self, state: V, time: float) -> DerivativeOperator[V]:
        return ScaledSum(
            [(1.0, self._base(state, time)), (1.0, self._owner._rate_param(time))],
            self._states,
            self._params,
        )


class _RatedCurvature(Generic[V]):
    r""":math:`f_{pp}` at the adjoint with the rows zeroed, plus the rate's
    curvature at the adjoint's weight on the rows."""

    def __init__(
        self,
        base: SecondDerivativeFn[V],
        owner: "DifferentiatedRowsRightHandSide[V, Any]",
        params: VectorSpaceProtocol[V],
    ) -> None:
        self._base, self._owner, self._params = base, owner, params

    def __call__(self, state: V, time: float, adj: V) -> DerivativeOperator[V]:
        terms: List[tuple[float, DerivativeOperator[V]]] = [
            (1.0, self._base(state, time, adj))
        ]
        curvature = self._owner._rate_curvature(time, adj)
        if curvature is not None:
            terms.append((1.0, curvature))
        return ScaledSum(terms, self._params)


class DifferentiatedRowsRightHandSide(Generic[V, F]):
    r""":math:`f` with the rows :math:`R` carrying :math:`\dot g(t; p)`:
    the differentiated constraint's right-hand side. Its derivatives are
    those of :math:`f` with the rows zeroed, plus, for data depending on
    the parameters, the rate's parameter derivatives on the rows (the rate
    never sees the state).

    Parameters
    ----------
    rhs : RightHandSideProtocol
        :math:`f`.
    declaration : PinnedRowsProtocol
        The rows :math:`R`, the rate and its parameter derivatives.
    form : ResidualFormProtocol
        Converts between the space's vectors and the state form.
    space : VectorSpaceProtocol
        The state space.

    Raises
    ------
    ValueError
        If the declaration supplies no rate, or its rate depends on the
        parameters and ``rhs`` has no parameter derivatives.
    """

    def __init__(
        self,
        rhs: RightHandSideProtocol[V, F],
        declaration: PinnedRowsProtocol[V],
        form: ResidualFormProtocol[V, V],
        space: VectorSpaceProtocol[V],
    ) -> None:
        self._rates = required_rates(declaration)
        self._rows = declaration.rows()
        self._zeroed = ZeroedRowsRightHandSide(rhs, self._rows, form)
        self._rhs, self._form, self._space = rhs, form, space
        self._declaration = declaration
        self._data = PinnedData(declaration, space)
        self._derivatives = self._rated_derivatives(
            declaration.parameter_derivatives()
        )

    def _rated_derivatives(
        self, data: PinnedRowsParamDerivatives[V]
    ) -> ODEResidualDerivatives[V, F]:
        """The zeroed bundle, with the rate's parameter derivatives added
        where the declaration has them; fixed for the right-hand side's
        life."""
        zeroed = self._zeroed.derivatives()
        if isinstance(data.rates, StatedZero) and isinstance(
            data.rates_curvature, StatedZero
        ):
            return zeroed
        param, params = zeroed.param, zeroed.param_space
        if param is None or params is None:
            raise ValueError(
                "the declared rate depends on the parameters, but the right-hand "
                "side has no parameter derivatives to add it to"
            )
        if data.rates is None:
            raise ValueError(
                f"{type(self._declaration).__name__} does not declare the rate's "
                "parameter Jacobian, which these derivatives need: declare it, or "
                "that the data do not depend on the parameters "
                "(PinnedRowsParamDerivatives.independent())"
            )
        param_param = zeroed.param_param
        return replace(
            zeroed,
            param=_RatedParam(param, self, self._space, params),
            param_param=(
                None
                if param_param is None
                else _RatedCurvature(param_param, self, params)
            ),
        )

    def _rate_param(self, time: float) -> DerivativeOperator[V]:
        r""":math:`(I - P)\, \partial \dot g/\partial p`, or a zero."""
        op = self._data.rates_param(time)
        return ZeroOperator(self._zeroed_params(), self._space) if op is None else op

    def _zeroed_params(self) -> VectorSpaceProtocol[V]:
        params = self._zeroed.derivatives().param_space
        if params is None:
            raise AssertionError("checked at construction")
        return params

    def _rate_curvature(self, time: float, adj: V) -> Optional[DerivativeOperator[V]]:
        r""":math:`\sum_{i \in R} \lambda_i\, \partial^2 \dot g_i/\partial
        p^2` at the adjoint ``adj`` (in the state form)."""
        return self._data.rates_curvature(time, self._form.from_residual(adj))

    def _rated(self, state: V, time: float) -> V:
        """The value in the space's form, rows :math:`R` the rate."""
        value = self._form.from_residual(self._rhs(state, time))
        return self._rows.replace_rows(value, self._rates(time))

    def __call__(self, state: V, time: float) -> V:
        return self._form.to_residual(self._rated(state, time))

    def derivatives(self) -> ODEResidualDerivatives[V, F]:
        return self._derivatives

    def parameters_version(self) -> int:
        """The wrapped right-hand side's version; zero if it has none."""
        return self._zeroed.parameters_version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._rhs!r}, nrows={self._rows.count()})"


class EliminatedDifferentiatedRowsRightHandSide(
    DifferentiatedRowsRightHandSide[V, F]
):
    r"""As :class:`DifferentiatedRowsRightHandSide`, with the coupling
    :math:`-P M (I - P)\, \dot g(t; p)` added to the other rows: the
    right-hand side of a mass whose coupling columns were eliminated
    (:class:`SymmetricRows`). The coupling's parameter derivatives follow
    the rate's.

    Parameters
    ----------
    rhs, declaration, form, space
        As for :class:`DifferentiatedRowsRightHandSide`.
    mass : ActionOperatorProtocol
        The mass before its rows and columns were edited.
    """

    def __init__(
        self,
        rhs: RightHandSideProtocol[V, F],
        declaration: PinnedRowsProtocol[V],
        form: ResidualFormProtocol[V, V],
        space: VectorSpaceProtocol[V],
        mass: ActionOperatorProtocol[V, F],
    ) -> None:
        self._mass = mass
        super().__init__(rhs, declaration, form, space)

    def __call__(self, state: V, time: float) -> V:
        rated = self._rated(state, time)
        placed = self._rows.replace_rows(self._space.zeros(), self._rates(time))
        coupling = self._rows.zero_rows(self._mass.apply(placed))
        return self._form.to_residual(self._space.axpby(1.0, rated, -1.0, coupling))

    def _rate_param(self, time: float) -> DerivativeOperator[V]:
        r""":math:`(I - P M)\, D(t)`, :math:`D` the rate's parameter
        Jacobian on the rows."""
        rate = super()._rate_param(time)
        coupling = RowsZeroedOperator(Product([self._mass, rate]), self._rows)
        return ScaledSum(
            [(1.0, rate), (-1.0, coupling)], self._space, self._zeroed_params()
        )

    def _rate_curvature(self, time: float, adj: V) -> Optional[DerivativeOperator[V]]:
        r"""The rate's curvature at the weight :math:`\lambda - M^\top P
        \lambda`, restricted to the rows."""
        mu = self._form.from_residual(adj)
        coupled = self._mass.apply_transpose(self._rows.zero_rows(mu))
        weight = self._space.axpby(1.0, mu, -1.0, coupled)
        return self._data.rates_curvature(time, weight)


# ----------------------------------------------------------------------
# Mass treatments
# ----------------------------------------------------------------------


@runtime_checkable
class RowRatesTreatmentProtocol(Protocol):
    """How the mass of a system with differentiated rows treats them, and
    so where the coupling :math:`M_{ID}\\, \\dot g` sits."""

    def mass(
        self, mass: ActionOperatorProtocol[V, F], rows: RowSetProtocol[V]
    ) -> ActionOperatorProtocol[V, F]:
        """The mass with the rows :math:`R` identity rows."""
        ...

    def differentiated(
        self,
        rhs: RightHandSideProtocol[V, F],
        declaration: PinnedRowsProtocol[V],
        form: ResidualFormProtocol[V, V],
        space: VectorSpaceProtocol[V],
        mass: ActionOperatorProtocol[V, F],
    ) -> RightHandSideProtocol[V, F]:
        """The right-hand side carrying the declared rate on its rows,
        consistent with :meth:`mass`; ``mass`` is the unedited one."""
        ...


class CoupledRows:
    r"""The mass :math:`[M_{II}\; M_{ID};\, 0\; I]`: identity rows, the
    coupling columns kept, so the coupling sits in the mass. With the
    step end pinned, the other rows see the data's exact increment and a
    one-step method conserves :math:`1^\top M y` exactly."""

    def mass(
        self, mass: ActionOperatorProtocol[V, F], rows: RowSetProtocol[V]
    ) -> ActionOperatorProtocol[V, F]:
        return RowReplacedOperator(mass, rows)

    def differentiated(
        self,
        rhs: RightHandSideProtocol[V, F],
        declaration: PinnedRowsProtocol[V],
        form: ResidualFormProtocol[V, V],
        space: VectorSpaceProtocol[V],
        mass: ActionOperatorProtocol[V, F],
    ) -> RightHandSideProtocol[V, F]:
        return DifferentiatedRowsRightHandSide(rhs, declaration, form, space)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SymmetricRows:
    r"""The mass :math:`[M_{II}\; 0;\, 0\; I]`, symmetric (and positive
    definite when :math:`M` is), with :math:`-M_{ID}\, \dot g(t)` moved to
    the other rows of the rate-carrying :math:`f`. The stepper integrates
    the coupling like a forcing term, so content is conserved to the
    method's order rather than exactly; a solver may use a symmetric
    factorization."""

    def mass(
        self, mass: ActionOperatorProtocol[V, F], rows: RowSetProtocol[V]
    ) -> ActionOperatorProtocol[V, F]:
        return SymmetricEliminatedOperator(mass, rows)

    def differentiated(
        self,
        rhs: RightHandSideProtocol[V, F],
        declaration: PinnedRowsProtocol[V],
        form: ResidualFormProtocol[V, V],
        space: VectorSpaceProtocol[V],
        mass: ActionOperatorProtocol[V, F],
    ) -> RightHandSideProtocol[V, F]:
        return EliminatedDifferentiatedRowsRightHandSide(
            rhs, declaration, form, space, mass
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def required_rates(declaration: PinnedRowsProtocol[V]) -> RatesFn[V]:
    """The declaration's rates, or refuse: without them the rows of a
    block left unpinned would hold an assembled equation.

    Raises
    ------
    TypeError
        If ``declaration`` does not satisfy
        :class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol`.
    ValueError
        If it supplies no rates.
    """
    if not isinstance(declaration, PinnedRowsProtocol):
        raise TypeError(
            "declaration must satisfy PinnedRowsProtocol, got "
            f"{type(declaration).__name__}"
        )
    rates = declaration.rates()
    if rates is None:
        raise ValueError(
            f"{type(declaration).__name__} supplies no rate of its data, "
            "which differentiated rows need: supply the rate, or pin every "
            "block (AtEveryBlock) on the system itself"
        )
    return rates


def differentiated_rows(
    system: MassFormSystem[V, A, F],
    declaration: PinnedRowsProtocol[V],
    treatment: Optional[RowRatesTreatmentProtocol] = None,
) -> MassFormSystem[V, A, F]:
    r"""``system`` with the declared rows in differentiated form: its
    right-hand side carrying the rate on them, its mass edited by
    ``treatment``; the space, solver and form are ``system``'s.

    Parameters
    ----------
    system : MassFormSystem
        The system whose rows are replaced; its mass is the one
        assembled, rows unedited.
    declaration : PinnedRowsProtocol
        The rows and their data's rate, the declaration a step-end pin
        reads too.
    treatment : RowRatesTreatmentProtocol, optional
        Where the coupling sits; :class:`CoupledRows` by default.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If the declaration supplies no rates, or its rows belong to
        another space.
    """
    required_rates(declaration)
    chosen: RowRatesTreatmentProtocol = (
        CoupledRows() if treatment is None else treatment
    )
    if not isinstance(chosen, RowRatesTreatmentProtocol):
        raise TypeError(
            "treatment must satisfy RowRatesTreatmentProtocol, got "
            f"{type(chosen).__name__}"
        )
    rows = declaration.rows()
    system.space.descriptor().require_same(rows.descriptor(), "the declared rows")
    space: InnerProductSpaceProtocol[V, A] = system.space
    return MassFormSystem(
        rhs=chosen.differentiated(
            system.rhs, declaration, system.form, space, system.mass
        ),
        mass=chosen.mass(system.mass, rows),
        space=space,
        solver=system.solver,
        form=system.form,
    )
