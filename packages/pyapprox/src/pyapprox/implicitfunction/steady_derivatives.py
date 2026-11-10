r"""The parameter derivatives of a steady functional through the solved
state: the forward (tangent) product, the adjoint gradient and the
second-order adjoint Hessian-vector product.

At a solution :math:`u(p)` of :math:`R(u;\, p) = 0`, the functional's
derivative along a parameter direction :math:`v` is

.. math::

    dQ\, v = Q_u\, \delta u + Q_p v, \qquad R_u\, \delta u = -R_p v,

one solve per direction for every entry (forward: cheap with few
parameters); the weighted functional :math:`q(p) = \omega^\top Q(u(p);\,
p)` has the gradient

.. math::

    \nabla q = Q_p^\top \omega + R_p^\top \lambda, \qquad
    R_u^\top \lambda = -Q_u^\top \omega,

one solve per weighting for every parameter (adjoint: cheap with few
entries), and, with the Lagrangian :math:`L = \omega^\top Q + \lambda^\top R`, the
Hessian-vector product along :math:`v`

.. math::

    R_u\, \delta u = -R_p v, \qquad
    R_u^\top \mu = -(L_{uu}\, \delta u + L_{up}\, v), \qquad
    \nabla^2 q\, v = L_{pu}\, \delta u + L_{pp}\, v + R_p^\top \mu.

Every solve is with :math:`R_u` at the solved state, on the one handle of
the :class:`~pyapprox.implicitfunction.steady_system.SteadyPoint`
(transposed for the adjoints), so a constant :math:`R_u` is set up once
for the solve and all its derivatives, and a state-dependent one once
more, at the solution.
"""

from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

from pyapprox.implicitfunction.functional_terms import (
    LinearizationProtocol,
    StateCurvature,
    StateFirst,
    accumulate,
    has_second_derivatives,
    parameter_first_derivatives,
    parameter_second_derivatives,
    state_first_derivatives,
    state_second_derivatives,
)
from pyapprox.implicitfunction.residual_derivatives import (
    ResidualParamFn,
    ResidualSecondDerivativeFn,
)
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    InnerProductSpaceProtocol,
    VectorSpaceProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


@dataclass(frozen=True)
class SteadyAdjoint(Generic[V, A]):
    r"""The adjoint :math:`\lambda` of a solve for value weights
    :math:`\omega`. Build with :meth:`SteadyDerivatives.adjoint`."""

    point: SteadyPoint[V, A]
    weights: A
    vector: V


class _Linearized(Generic[V, A]):
    r"""What both orders read: the spaces, :math:`R_p`, and the solves
    with :math:`R_u` at a solved point."""

    def __init__(
        self,
        space: InnerProductSpaceProtocol[V, A],
        params: VectorSpaceProtocol[V],
        residual_param: ResidualParamFn[V],
    ) -> None:
        self.space, self.params, self._residual_param = space, params, residual_param

    def param_cotangent(self, state: V, adjoint: V) -> V:
        r""":math:`R_p^\top \lambda`, a new vector."""
        return self.params.copy(self._residual_param(state).apply_transpose(adjoint))

    def state_tangent(self, point: SteadyPoint[V, A], state: V, direction: V) -> V:
        r""":math:`\delta u` solving :math:`R_u\, \delta u = -R_p v`: the
        state's derivative along ``direction``.

        Raises
        ------
        ValueError
            If ``direction`` is not one parameter vector, or the solve
            fails.
        """
        if self.params.ncols(direction) != 1:
            raise ValueError("direction must be one parameter vector")
        rhs = self._residual_param(state).apply(direction)
        return self.solve(point, self.space.scale(-1.0, rhs), False)

    def solve(self, point: SteadyPoint[V, A], rhs: V, transpose: bool) -> V:
        r"""Solve with :math:`R_u` (:math:`R_u^\top` when ``transpose``) on
        ``point``'s handle.

        Raises
        ------
        ValueError
            If the handle has no transpose or the solve fails.
        """
        handle = point.jacobian_handle()
        if transpose:
            transposed = handle.transpose()
            if transposed is None:
                raise ValueError(
                    "the linear solver's handle has no transpose; the adjoint needs one"
                )
            handle = transposed
        x, info = handle.solve(rhs)
        if not info.converged:
            kind = "adjoint" if transpose else "tangent"
            raise ValueError(f"the {kind} linear solve failed: {info.message}")
        return x


class SteadyHessian(Generic[V, A]):
    r"""The second-order adjoint Hessian-vector product of a steady
    functional. Read from :meth:`SteadyDerivatives.hessian`."""

    def __init__(
        self,
        linearized: _Linearized[V, A],
        residual: Tuple[
            ResidualSecondDerivativeFn[V],
            ResidualSecondDerivativeFn[V],
            ResidualSecondDerivativeFn[V],
        ],
        parameter_terms: Tuple[Callable[[A, V], Optional[V]], ...],
        state_terms: Tuple[StateCurvature[V, A], ...],
    ) -> None:
        self._linearized = linearized
        # R's adjoint-weighted second derivatives.
        self._state_state, self._state_param, self._param_param = residual
        # Each term of Q's second derivatives, one entry per term.
        self._parameter_terms, self._state_terms = parameter_terms, state_terms

    def hvp(self, adjoint: SteadyAdjoint[V, A], direction: V) -> V:
        r""":math:`\nabla^2 q\, v` at the adjoint's point and weights: one
        tangent and one transposed solve.

        Raises
        ------
        ValueError
            If ``direction`` is not one parameter vector, or a solve fails.
        """
        lin = self._linearized
        point, omega, lam = adjoint.point, adjoint.weights, adjoint.vector
        state = point.solution()
        tangent = lin.state_tangent(point, state, direction)
        state_param = self._state_param(state, lam)
        # L_uu du + L_up v, a state vector.
        to_state = lin.space.axpby(
            1.0,
            self._state_state(state, lam).apply(tangent),
            1.0,
            state_param.apply(direction),
        )
        for term in self._state_terms:
            to_state = _add(
                to_state, term.state_state(state, omega, tangent), lin.space
            )
            to_state = _add(
                to_state, term.state_param(state, omega, direction), lin.space
            )
        second = lin.solve(point, lin.space.scale(-1.0, to_state, out=to_state), True)
        # L_pu du + L_pp v + R_p^T mu, a parameter vector.
        total = lin.param_cotangent(state, second)
        total = _add(total, state_param.apply_transpose(tangent), lin.params)
        total = _add(total, self._param_param(state, lam).apply(direction), lin.params)
        for term in self._state_terms:
            total = _add(total, term.param_state(state, omega, tangent), lin.params)
            total = _add(total, term.param_param(state, omega, direction), lin.params)
        for param_param in self._parameter_terms:
            total = _add(total, param_param(omega, direction), lin.params)
        return total


def _add(total: V, term: Optional[V], space: VectorSpaceProtocol[V]) -> V:
    """``total + term`` into ``total``, a vector the caller owns."""
    if term is None:
        return total
    return space.axpby(1.0, term, 1.0, total, out=total)


class SteadyDerivatives(Generic[V, A, F]):
    r"""The adjoint gradient of a steady functional, and its Hessian-vector
    product when every bundle carries second derivatives.

    Parameters
    ----------
    system : SteadySystem
        :math:`R(u;\, p) = 0`; its residual's bundle must have the
        parameter Jacobian. The adjoint solves transposed: a solver whose
        handles have no transpose is refused at the first adjoint solve.
    functional : StateFunctional
        :math:`Q(u;\, p)`; every term's bundle must have first
        derivatives.

    Raises
    ------
    TypeError
        If an argument is of the wrong type, the residual's bundle has no
        parameter Jacobian, or a term no first derivatives.
    """

    def __init__(
        self, system: SteadySystem[V, A, F], functional: StateFunctional[V, A]
    ) -> None:
        if not isinstance(system, SteadySystem):
            raise TypeError(
                f"system must be a SteadySystem, got {type(system).__name__}"
            )
        if not isinstance(functional, StateFunctional):
            raise TypeError(
                "functional must be a StateFunctional, got "
                f"{type(functional).__name__}"
            )
        bundle = system.residual_derivatives()
        if bundle.param is None or bundle.param_space is None:
            raise TypeError(
                "the residual's derivatives bundle has no parameter Jacobian; "
                "the adjoint gradient needs one"
            )
        self._functional = functional
        self._linearized: _Linearized[V, A] = _Linearized(
            system.space(), bundle.param_space, bundle.param
        )
        # The first derivatives of Q's terms, one entry per term; every
        # state term reads the one solved state.
        self._parameter_terms: List[Callable[[], LinearizationProtocol[V, A]]] = (
            parameter_first_derivatives(functional.parameter_terms)
        )
        self._state_terms: List[StateFirst[V, A]] = state_first_derivatives(
            functional.state_terms
        )
        # Which fields are present is fixed for the bundles' lives.
        self._hessian: Optional[SteadyHessian[V, A]] = None
        if (
            bundle.state_state is not None
            and bundle.state_param is not None
            and bundle.param_param is not None
            and has_second_derivatives(
                functional.parameter_terms, functional.state_terms
            )
        ):
            self._hessian = SteadyHessian(
                self._linearized,
                (bundle.state_state, bundle.state_param, bundle.param_param),
                tuple(parameter_second_derivatives(functional.parameter_terms)),
                tuple(state_second_derivatives(functional.state_terms)),
            )

    def value(self, point: SteadyPoint[V, A]) -> A:
        """:math:`Q(u)` at a solved point (refused as :meth:`~pyapprox.
        implicitfunction.steady_system.SteadyPoint.solution` refuses)."""
        return self._functional.value(point.solution())

    def adjoint(
        self, point: SteadyPoint[V, A], weights: Optional[A] = None
    ) -> SteadyAdjoint[V, A]:
        r""":math:`\lambda` for the value weights ``weights``
        (:math:`\omega = 1` for a scalar functional): one transposed solve,
        none when no term reads the state.

        Raises
        ------
        ValueError
            If the point is not a solution of the current residual, the
            weights have the wrong shape, or the solve fails.
        """
        lin = self._linearized
        state = point.solution()
        omega = self._functional.weights(weights)
        rhs: Optional[V] = None
        for term in self._state_terms:
            rhs = accumulate(
                rhs, term.wrt_state(state).apply_transpose(omega), lin.space
            )
        if rhs is None:
            return SteadyAdjoint(point, omega, lin.space.zeros(1))
        vector = lin.solve(point, lin.space.scale(-1.0, rhs, out=rhs), True)
        return SteadyAdjoint(point, omega, vector)

    def gradient(self, adjoint: SteadyAdjoint[V, A]) -> V:
        r""":math:`\nabla q = Q_p^\top \omega + R_p^\top \lambda`, a
        parameter vector.

        Raises
        ------
        ValueError
            If the adjoint's point is no longer a solution of the current
            residual.
        """
        lin = self._linearized
        state, omega = adjoint.point.solution(), adjoint.weights
        total = lin.param_cotangent(state, adjoint.vector)
        for wrt_param in self._parameter_terms:
            total = _add(total, wrt_param().apply_transpose(omega), lin.params)
        for term in self._state_terms:
            if term.wrt_param is None:
                continue
            by_param = term.wrt_param(state)
            if by_param is not None:
                total = _add(total, by_param.apply_transpose(omega), lin.params)
        return total

    def tangent(self, point: SteadyPoint[V, A], direction: V) -> A:
        r""":math:`dQ\, v = Q_u\, \delta u + Q_p v` with :math:`R_u\, \delta
        u = -R_p v`: every entry's derivative along ``direction``, a column
        of ``nqoi`` entries, from one solve.

        Raises
        ------
        ValueError
            If the point is not a solution of the current residual,
            ``direction`` is not one parameter vector, or the solve fails.
        """
        lin = self._linearized
        state = point.solution()
        delta = lin.state_tangent(point, state, direction)
        total = self._functional.zero
        for wrt_param in self._parameter_terms:
            total = total + wrt_param().apply(direction)
        for term in self._state_terms:
            total = total + term.wrt_state(state).apply(delta)
            if term.wrt_param is None:
                continue
            by_param = term.wrt_param(state)
            if by_param is not None:
                total = total + by_param.apply(direction)
        return total

    def hessian(self) -> Optional[SteadyHessian[V, A]]:
        """The Hessian-vector products, or None when a bundle has no second
        derivatives."""
        return self._hessian

    def __repr__(self) -> str:
        return f"{type(self).__name__}(hessian={self._hessian is not None})"
