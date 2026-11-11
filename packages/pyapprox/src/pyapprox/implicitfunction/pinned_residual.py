r"""A residual with rows pinned to data: :math:`R = P F + (I - P)(u - g)`.

For a residual :math:`F(u; p)` and rows :math:`R` pinned to data
:math:`g(p)` (a :class:`~pyapprox.implicitfunction.pinned_rows.RowPin`),
:math:`P` the projection zeroing the rows, the pinned residual is

.. math::

    R(u; p) = P F(u; p) + (I - P)(u - g(p)),

so its rows :math:`R` read :math:`u_R - g_R` and the others are
:math:`F`'s. Its derivatives follow from :math:`F`'s and the data's:

- :math:`R_u = P F_u + (I - P)`, at :math:`F_u`'s level (a constant
  Jacobian stays constant);
- :math:`R_p = P F_p - (I - P)\, g_p`;
- :math:`\lambda^\top R_{uu} = (P\lambda)^\top F_{uu}` and
  :math:`\lambda^\top R_{up} = (P\lambda)^\top F_{up}`: the pinned rows
  are linear in :math:`u` and do not couple it to :math:`p`;
- :math:`\lambda^\top R_{pp} = (P\lambda)^\top F_{pp} - \sum_{i \in R}
  \lambda_i\, \partial^2 g_i/\partial p^2`.

Stated zeros are kept, so a consumer may still skip them.
"""

from typing import Generic, List, TypeVar, Union

from pyapprox.implicitfunction.pinned_rows import (
    RowPin,
    require_data_derivative,
    undeclared_data_derivative,
)
from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    ResidualDerivatives,
    ResidualParamFn,
    ResidualSecondDerivativeFn,
    StatedZero,
    is_stated_zero,
    map_jacobian,
)
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    JacobianSnapshotProtocol,
    LinearizedOperator,
    ScaledSum,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


class _OwnRows(Generic[V, F]):
    """A Jacobian snapshot with the pinned rows identity rows, under the
    same label."""

    def __init__(self, pin: RowPin[V]) -> None:
        self._pin = pin

    def __call__(
        self, snapshot: JacobianSnapshotProtocol[V, F]
    ) -> JacobianSnapshotProtocol[V, F]:
        return LinearizedOperator(self._pin.own(snapshot), snapshot.linearization())


class _PinnedParam(Generic[V]):
    r""":math:`P F_p - (I - P)\, g_p`; :math:`P F_p` alone for data
    independent of the parameters."""

    def __init__(
        self,
        fn: ResidualParamFn[V],
        pin: RowPin[V],
        states: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
        data_depend: bool,
    ) -> None:
        self._fn, self._pin = fn, pin
        self._states, self._params = states, params
        self._data_depend = data_depend

    def __call__(self, state: V) -> DerivativeOperator[V]:
        zeroed = self._pin.coupled(self._fn(state))
        if not self._data_depend:
            return zeroed
        data = require_data_derivative(
            self._pin.data_param_jacobian(self._states, self._params),
            "the parameter Jacobian",
            self._pin,
        )
        return ScaledSum([(1.0, zeroed), (-1.0, data)], self._states, self._params)


class _AtPinnedAdjoint(Generic[V]):
    r""":math:`F`'s second derivative at the adjoint with the pinned rows
    zeroed: :math:`(P \lambda)^\top F_{..}`."""

    def __init__(
        self,
        fn: ResidualSecondDerivativeFn[V],
        pin: RowPin[V],
        states: VectorSpaceProtocol[V],
    ) -> None:
        self._fn, self._pin, self._states = fn, pin, states

    def __call__(self, state: V, adj: V) -> DerivativeOperator[V]:
        return self._fn(state, self._pin.adjoint(adj, self._states))


class _PinnedParamCurvature(Generic[V]):
    r""":math:`(P\lambda)^\top F_{pp} - \sum_{i \in R} \lambda_i\,
    \partial^2 g_i/\partial p^2`."""

    def __init__(
        self,
        fn: ResidualSecondDerivativeFn[V],
        pin: RowPin[V],
        states: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> None:
        self._fn, self._pin = fn, pin
        self._states, self._params = states, params

    def __call__(self, state: V, adj: V) -> DerivativeOperator[V]:
        terms: List[tuple[float, DerivativeOperator[V]]] = [
            (1.0, self._fn(state, self._pin.adjoint(adj, self._states)))
        ]
        data = require_data_derivative(
            self._pin.data_param_curvature(adj, self._states, self._params),
            "the second parameter derivatives",
            self._pin,
        )
        if not data.properties().zero:
            terms.append((-1.0, data))
        return ScaledSum(terms, self._params)


class PinnedResidual(Generic[V, F]):
    r""":math:`R = P F + (I - P)(u - g)`: a residual with the pin's rows
    reading :math:`u_R - g_R` (see the module docstring).

    Parameters
    ----------
    residual : ResidualProtocol
        :math:`F`.
    pin : RowPin
        The rows :math:`R` and their data :math:`g(p)`.
    space : VectorSpaceProtocol
        The state space, which the pinned rows belong to.

    Raises
    ------
    TypeError
        If ``residual`` does not satisfy :class:`ResidualProtocol`, ``pin``
        is not a :class:`RowPin`, or ``space`` does not satisfy
        ``VectorSpaceProtocol``.
    ValueError
        If the rows belong to another space, or :math:`F` has parameter
        derivatives and the declaration makes its data's parameter
        Jacobian unavailable. Data without second derivatives give a
        first-order bundle.
    """

    def __init__(
        self,
        residual: ResidualProtocol[V, F],
        pin: RowPin[V],
        space: VectorSpaceProtocol[V],
    ) -> None:
        if not isinstance(residual, ResidualProtocol):
            raise TypeError(
                f"residual must satisfy ResidualProtocol, got {type(residual).__name__}"
            )
        if not isinstance(pin, RowPin):
            raise TypeError(f"pin must be a RowPin, got {type(pin).__name__}")
        if not isinstance(space, VectorSpaceProtocol):
            raise TypeError(
                f"space must satisfy VectorSpaceProtocol, got {type(space).__name__}"
            )
        space.descriptor().require_same(pin.rows().descriptor(), "the pinned rows")
        self._residual, self._pin, self._space = residual, pin, space
        # Which fields are present is fixed for the residual's life.
        self._derivatives = self._pinned(residual.derivatives())

    def _pinned(self, bundle: ResidualDerivatives[V, F]) -> ResidualDerivatives[V, F]:
        jacobian = bundle.jacobian
        if jacobian is None:
            return ResidualDerivatives.none()
        own: _OwnRows[V, F] = _OwnRows(self._pin)
        pinned_jacobian = map_jacobian(jacobian, own)
        param, params = bundle.param, bundle.param_space
        if param is None or params is None:
            return ResidualDerivatives.state_only(pinned_jacobian)
        data = self._pin.parameter_derivatives()
        # Refuses data whose parameter Jacobian is unavailable.
        if data.values is None:
            raise undeclared_data_derivative("the parameter Jacobian", self._pin)
        pinned_param = _PinnedParam(
            param,
            self._pin,
            self._space,
            params,
            not isinstance(data.values, StatedZero),
        )
        state_state, state_param = bundle.state_state, bundle.state_param
        param_param = bundle.param_param
        if (
            state_state is None
            or state_param is None
            or param_param is None
            or data.curvature is None
        ):
            return ResidualDerivatives.first_order(
                params, pinned_jacobian, pinned_param
            )
        pinned_param_param: Union[ResidualSecondDerivativeFn[V], StatedZero] = (
            STATED_ZERO
            if is_stated_zero(param_param) and isinstance(data.curvature, StatedZero)
            else _PinnedParamCurvature(param_param, self._pin, self._space, params)
        )
        return ResidualDerivatives.second_order(
            self._space,
            params,
            pinned_jacobian,
            pinned_param,
            state_state=self._at_pinned_adjoint(state_state),
            state_param=self._at_pinned_adjoint(state_param),
            param_param=pinned_param_param,
        )

    def _at_pinned_adjoint(
        self, fn: ResidualSecondDerivativeFn[V]
    ) -> Union[ResidualSecondDerivativeFn[V], StatedZero]:
        """``fn`` at the adjoint with the rows zeroed; a stated zero stays
        stated."""
        if is_stated_zero(fn):
            return STATED_ZERO
        return _AtPinnedAdjoint(fn, self._pin, self._space)

    def __call__(self, state: V) -> V:
        """:math:`P F(u) + (I - P)(u - g)`."""
        return self._pin.residual(self._residual(state), state, self._space)

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._derivatives

    def parameters_version(self) -> int:
        """:math:`F`'s version: the data read the same parameters."""
        return self._residual.parameters_version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._residual!r}, {self._pin!r})"
