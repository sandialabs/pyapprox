r"""A method-based steady functional :math:`Q(u, p)` as a state term of a
:class:`~pyapprox.implicitfunction.state_functional.StateFunctional`.

Transitional: the method-based functionals take the parameters as an
argument and return dense Jacobians; a state term reads the parameters
currently set and gives linearizations. :class:`SteadyFunctionalTerm`
reads the parameters from a supplier and wraps the Jacobians, so those
functionals serve the stack-generic adjoint until they are written as
terms. States and parameters are backend columns.
"""

from typing import Callable, Generic, Optional

from pyapprox.implicitfunction.functional_terms import StateTermDerivatives
from pyapprox.implicitfunction.functionals.derivatives import (
    SteadyFunctionalHVPFn,
    SteadyFunctionalJacobianFn,
)
from pyapprox.implicitfunction.functionals.protocols import SteadyFunctionalProtocol
from pyapprox.util.backends.protocols import Array


class _DenseRows(Generic[Array]):
    r"""The linearization of a value by its dense Jacobian :math:`J`,
    ``(nqoi, n)``: forward :math:`J x`, transposed :math:`J^\top
    \omega`."""

    def __init__(self, matrix: Array) -> None:
        self._matrix = matrix

    def apply(self, x: Array) -> Array:
        return self._matrix @ x

    def apply_transpose(self, weights: Array) -> Array:
        return self._matrix.T @ weights


class SteadyFunctionalTerm(Generic[Array]):
    r"""``functional`` as a state term, at the parameters ``parameters()``
    returns when called.

    Its bundle mirrors the functional's: none, first order, or (for a
    scalar functional, the only kind whose Hessian-vector products the
    method protocol defines) second order, each product scaled by the
    weight :math:`\omega`.

    Parameters
    ----------
    functional : SteadyFunctionalProtocol
        :math:`Q(u, p)`.
    parameters : callable
        Returns the current parameters, a column ``(nparams, 1)``.

    Raises
    ------
    TypeError
        If ``functional`` does not satisfy ``SteadyFunctionalProtocol``.
    """

    def __init__(
        self,
        functional: SteadyFunctionalProtocol[Array],
        parameters: Callable[[], Array],
    ) -> None:
        if not isinstance(functional, SteadyFunctionalProtocol):
            raise TypeError(
                "functional must satisfy SteadyFunctionalProtocol, got "
                f"{type(functional).__name__}"
            )
        self._functional, self._parameters = functional, parameters
        old = functional.derivatives()
        self._state_jacobian: Optional[SteadyFunctionalJacobianFn[Array]] = (
            old.state_jacobian
        )
        self._param_jacobian: Optional[SteadyFunctionalJacobianFn[Array]] = (
            old.param_jacobian
        )
        self._hvps: Optional[
            tuple[
                SteadyFunctionalHVPFn[Array],
                SteadyFunctionalHVPFn[Array],
                SteadyFunctionalHVPFn[Array],
                SteadyFunctionalHVPFn[Array],
            ]
        ] = None
        self._derivs: StateTermDerivatives[Array, Array] = StateTermDerivatives.none()
        if old.state_jacobian is None:
            return
        if (
            functional.nqoi() == 1
            and old.state_state_hvp is not None
            and old.state_param_hvp is not None
            and old.param_state_hvp is not None
            and old.param_param_hvp is not None
        ):
            self._hvps = (
                old.state_state_hvp,
                old.state_param_hvp,
                old.param_state_hvp,
                old.param_param_hvp,
            )
            self._derivs = StateTermDerivatives.second_order(
                self._wrt_state,
                self._wrt_param,
                self._state_state,
                self._state_param,
                self._param_state,
                self._param_param,
            )
        else:
            self._derivs = StateTermDerivatives.first_order(
                self._wrt_state, self._wrt_param
            )

    def nqoi(self) -> int:
        return self._functional.nqoi()

    def value(self, state: Array) -> Array:
        return self._functional(state, self._parameters())

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return self._derivs

    def _wrt_state(self, state: Array) -> _DenseRows[Array]:
        if self._state_jacobian is None:
            raise RuntimeError("the functional has no Jacobians")
        return _DenseRows(self._state_jacobian(state, self._parameters()))

    def _wrt_param(self, state: Array) -> _DenseRows[Array]:
        if self._param_jacobian is None:
            raise RuntimeError("the functional has no Jacobians")
        return _DenseRows(self._param_jacobian(state, self._parameters()))

    def _hvp(self, which: int) -> SteadyFunctionalHVPFn[Array]:
        if self._hvps is None:
            raise RuntimeError("the functional has no Hessian-vector products")
        return self._hvps[which]

    def _state_state(self, state: Array, weights: Array, x: Array) -> Array:
        return weights[0, 0] * self._hvp(0)(state, self._parameters(), x)

    def _state_param(self, state: Array, weights: Array, x: Array) -> Array:
        return weights[0, 0] * self._hvp(1)(state, self._parameters(), x)

    def _param_state(self, state: Array, weights: Array, x: Array) -> Array:
        return weights[0, 0] * self._hvp(2)(state, self._parameters(), x)

    def _param_param(self, state: Array, weights: Array, x: Array) -> Array:
        return weights[0, 0] * self._hvp(3)(state, self._parameters(), x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._functional!r})"
