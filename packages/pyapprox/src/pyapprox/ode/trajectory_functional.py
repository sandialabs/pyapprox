r"""A whole-trajectory functional, as the sweeps read it.

The transient functionals take a whole trajectory as one backend array
(:class:`~pyapprox.ode.functionals.protocols.TransientFunctionalProtocol`).
The sweeps of :mod:`pyapprox.ode.trajectory_derivatives` read a
functional's terms (:mod:`pyapprox.ode.trajectory_terms`).
:class:`TransientFunctionalTerms` is the boundary between the two on the
backend stack: it forms the trajectory array once and evaluates the
functional's Jacobians, and its HVPs when it has them, on it once, so it
is built per recorded trajectory. It splits :math:`Q(Y, p)` into a state
term (:math:`\partial Q/\partial y_m` at every stored state, and the
second derivatives through one state) and a parameter term (the value,
the direct :math:`\partial Q/\partial p` and :math:`Q_{pp}`): a
bookkeeping split of one functional, whose sums are its derivatives.

It is transitional: a functional that reads the states one at a time is
a state term itself, needs no trajectory array and replaces it.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Sequence

from pyapprox.ode.functionals.protocols import (
    TransientFunctionalProtocol,
    TransientParamHVPFn,
    TransientStepHVPFn,
)
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol, trajectory_array
from pyapprox.ode.trajectory_terms import (
    BackendScalarGradient,
    ParameterTermDerivatives,
    StateTermDerivatives,
    TrajectoryFunctional,
)
from pyapprox.util.backends.protocols import Array, Backend


class TransientFunctionalTerms(Generic[Array]):
    r"""A scalar transient functional on one recorded trajectory, as a
    state term and a parameter term (see the module docstring).

    The first derivatives are always given; the terms' bundles carry the
    second derivatives when the functional's bundle has its HVPs (which
    index stored states, so its second derivative in the states is block
    diagonal over them, as the Hessian-vector sweep requires). Weights
    scale them, the functional being scalar.

    Parameters
    ----------
    functional : TransientFunctionalProtocol
        :math:`Q(y, p)`, one QoI, every parameter shared with the
        residual; its bundle must have ``state_jacobian`` and
        ``param_jacobian``. A functional integrating over time must
        already hold the scheme's quadrature.
    store : TrajectoryStoreProtocol
        The recorded trajectory.
    param : Array
        The parameters the trajectory was recorded at. Shape:
        ``(nparams, 1)``.

    Raises
    ------
    TypeError
        If ``functional`` does not satisfy its protocol or lacks the
        Jacobians.
    ValueError
        If it has more than one QoI or parameters of its own, or
        ``param`` is not a single column.
    """

    def __init__(
        self,
        functional: TransientFunctionalProtocol[Array],
        store: TrajectoryStoreProtocol[Array],
        param: Array,
    ) -> None:
        if not isinstance(functional, TransientFunctionalProtocol):
            raise TypeError(
                "functional must satisfy TransientFunctionalProtocol, got "
                f"{type(functional).__name__}"
            )
        if functional.nqoi() != 1:
            raise ValueError(
                f"the sweeps take a scalar functional, got nqoi={functional.nqoi()}"
            )
        if functional.nunique_params() != 0:
            raise ValueError(
                "the functional has parameters of its own "
                f"({functional.nunique_params()}); the sweeps differentiate "
                "the residual's parameters only"
            )
        if param.ndim != 2 or param.shape[1] != 1:
            raise ValueError(
                f"param must have shape (nparams, 1), got {tuple(param.shape)}"
            )
        derivs = functional.derivatives()
        if derivs.state_jacobian is None or derivs.param_jacobian is None:
            raise TypeError(
                f"{type(functional).__name__}'s derivatives() has no Jacobians"
            )
        self._bkd, self._param = functional.bkd(), param
        self._trajectory = trajectory_array(store, self._bkd)
        self._value = self._bkd.reshape(functional(self._trajectory, param), (1, 1))
        # dQ/dy at every stored state, (nstates, ntimes), and the direct
        # dQ/dp, (1, nparams).
        self._state_jacobian = derivs.state_jacobian(self._trajectory, param)
        self._param_jacobian = derivs.param_jacobian(self._trajectory, param)
        self._hvps: Optional[_Hvps[Array]] = (
            None
            if derivs.state_state_hvp is None
            or derivs.state_param_hvp is None
            or derivs.param_state_hvp is None
            or derivs.param_param_hvp is None
            else _Hvps(
                derivs.state_state_hvp,
                derivs.state_param_hvp,
                derivs.param_state_hvp,
                derivs.param_param_hvp,
            )
        )
        self._state_term = _TransientStateTerm(self)
        self._parameter_term = _TransientParameterTerm(self)

    def functional(self) -> TrajectoryFunctional[Array, Array]:
        """The two terms, as the sweeps read the functional."""
        return TrajectoryFunctional.from_terms(
            self._bkd,
            1,
            parameter_terms=(self._parameter_term,),
            state_terms=(self._state_term,),
        )

    def state_term(self) -> "_TransientStateTerm[Array]":
        r""":math:`\partial Q/\partial y_m` at every stored state."""
        return self._state_term

    def parameter_term(self) -> "_TransientParameterTerm[Array]":
        r"""The value, the direct :math:`\partial Q/\partial p` and
        :math:`Q_{pp}`."""
        return self._parameter_term

    def ntimes(self) -> int:
        return int(self._trajectory.shape[1])

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def value(self) -> Array:
        """:math:`Q`, shape ``(1, 1)``."""
        return self._value

    def state_gradient(self, index: int) -> Array:
        r""":math:`\partial Q/\partial y_{\mathrm{index}}`, shape
        ``(nstates, 1)``."""
        return self._state_jacobian[:, index : index + 1]

    def param_gradient(self) -> Array:
        r"""The direct :math:`\partial Q/\partial p`, shape
        ``(nparams, 1)``."""
        return self._param_jacobian.T

    def has_hvps(self) -> bool:
        """Whether the functional's bundle has its HVPs."""
        return self._hvps is not None

    def _column(self, x: Array) -> Array:
        """``x`` as a column: the HVPs may return 1-D arrays."""
        return self._bkd.reshape(x, (-1, 1))

    def _checked_hvps(self) -> "_Hvps[Array]":
        if self._hvps is None:
            raise RuntimeError(
                "the functional has no HVPs; check derivatives() before calling"
            )
        return self._hvps

    def state_state(self, index: int, w: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial y_m^2)\, w`, ``(nstates, 1)``."""
        hvp = self._checked_hvps().state_state
        return self._column(hvp(self._trajectory, self._param, index, w))

    def state_param(self, index: int, v: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial y_m \partial p)\, v`,
        ``(nstates, 1)``."""
        hvp = self._checked_hvps().state_param
        return self._column(hvp(self._trajectory, self._param, index, v))

    def param_state(self, index: int, w: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial p\, \partial y_m)\, w`,
        ``(nparams, 1)``."""
        hvp = self._checked_hvps().param_state
        return self._column(hvp(self._trajectory, self._param, index, w))

    def param_param(self, v: Array) -> Array:
        r""":math:`(\partial^2 Q/\partial p^2)\, v`, ``(nparams, 1)``."""
        hvp = self._checked_hvps().param_param
        return self._column(hvp(self._trajectory, self._param, v))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class _TransientStateTerm(Generic[Array]):
    """The state term of :class:`TransientFunctionalTerms`: the value lies
    in the parameter term, so this one's values are zero."""

    def __init__(self, terms: TransientFunctionalTerms[Array]) -> None:
        self._terms = terms
        self._zero = terms.bkd().zeros((1, 1))
        # The state gradients are always there; the parameters enter
        # through the parameter term, so this one has no wrt_param.
        self._derivatives: StateTermDerivatives[Array, Array] = (
            StateTermDerivatives.second_order(
                self._wrt_state,
                None,
                self._state_state,
                self._state_param,
                self._param_state,
                self._param_param,
            )
            if terms.has_hvps()
            else StateTermDerivatives.first_order(self._wrt_state)
        )

    def nqoi(self) -> int:
        return 1

    def indices(self, ntimes: int) -> Sequence[int]:
        return range(ntimes)

    def value(self, index: int, time: float, state: Array) -> Array:
        return self._zero

    def _wrt_state(
        self, index: int, time: float, state: Array
    ) -> BackendScalarGradient[Array]:
        return BackendScalarGradient(
            self._terms.state_gradient(index), self._terms.bkd()
        )

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return self._derivatives

    def _state_state(
        self, index: int, time: float, state: Array, weights: Array, x: Array
    ) -> Array:
        return self._terms.state_state(index, x) * weights

    def _state_param(
        self, index: int, time: float, state: Array, weights: Array, x: Array
    ) -> Array:
        return self._terms.state_param(index, x) * weights

    def _param_state(
        self, index: int, time: float, state: Array, weights: Array, x: Array
    ) -> Array:
        return self._terms.param_state(index, x) * weights

    def _param_param(
        self, index: int, time: float, state: Array, weights: Array, x: Array
    ) -> None:
        # Q_pp is the parameter term's.
        return None


class _TransientParameterTerm(Generic[Array]):
    """The parameter term of :class:`TransientFunctionalTerms`: the value,
    the direct parameter derivative and :math:`Q_{pp}`."""

    def __init__(self, terms: TransientFunctionalTerms[Array]) -> None:
        self._terms = terms
        self._derivatives: ParameterTermDerivatives[Array, Array] = (
            ParameterTermDerivatives.second_order(self._wrt_param, self._param_param)
            if terms.has_hvps()
            else ParameterTermDerivatives.first_order(self._wrt_param)
        )

    def nqoi(self) -> int:
        return 1

    def value(self) -> Array:
        return self._terms.value()

    def _wrt_param(self) -> BackendScalarGradient[Array]:
        return BackendScalarGradient(self._terms.param_gradient(), self._terms.bkd())

    def derivatives(self) -> ParameterTermDerivatives[Array, Array]:
        return self._derivatives

    def _param_param(self, weights: Array, v: Array) -> Array:
        return self._terms.param_param(v) * weights


@dataclass(frozen=True)
class _Hvps(Generic[Array]):
    """A transient functional's four HVPs, known present."""

    state_state: TransientStepHVPFn[Array]
    state_param: TransientStepHVPFn[Array]
    param_state: TransientStepHVPFn[Array]
    param_param: TransientParamHVPFn[Array]


def from_transient_functional(
    functional: TransientFunctionalProtocol[Array],
    store: TrajectoryStoreProtocol[Array],
    param: Array,
) -> TrajectoryFunctional[Array, Array]:
    """A transient functional, which takes a whole trajectory array, as
    the sweeps read it on the trajectory ``store`` recorded: the terms of
    a :class:`TransientFunctionalTerms`; see it for the parameters and the
    errors raised."""
    return TransientFunctionalTerms(functional, store, param).functional()
