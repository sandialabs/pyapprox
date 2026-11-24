r"""Rayleigh damping over a restoring system's operators.

:class:`RayleighDamping` is the velocity force

.. math::

    g(u, v, t) = -(\alpha M + \beta K)\, v, \qquad K = -f_u,

read from the restoring system's mass and its force's Jacobian whenever it
is evaluated, so :math:`K` follows the parameters the force is set to; a
matrix copied at construction would stop tracking them.

:math:`K` is the force's Jacobian at a reference state :math:`u_r`, the
zero state unless one is given. For a force affine in the state that is
:math:`K` itself. For a nonlinear force a reference state must be given:
:math:`K = -f_u(u_r)` is then the stiffness linearized there (commonly
the initial or a static equilibrium state), and :math:`g` stays linear in
:math:`v` and independent of :math:`u`. Following the tangent
:math:`K(u)` instead would make :math:`g_u = \beta f_{uu}[v]`, a
directional second derivative the force's bundle does not supply, so it
is refused.

The damping owns no parameters: its dependence on them is through
:math:`K(p)` alone. For an affine force :math:`K(p) v = f(0; p) - f(v;
p)`, so the reduction derives :math:`g`'s parameter derivatives from
the force's (``VelocityForceDerivatives.through_stiffness``) and the
reduced system keeps the force's tier. For a nonlinear force damped at a
reference state, :math:`K(u_r; p) v` is no such difference: its
parameter dependence is not supplied, and the reduced system has state
derivatives only.
"""

from typing import Generic, List, Optional, Tuple, TypeVar

from pyapprox.ode.mass_form_system import MassFormSystem, parameters_version_reader
from pyapprox.ode.residual_derivatives import Jacobian, constant_jacobian
from pyapprox.ode.second_order.reduction import VelocityForceDerivatives
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearizedOperator,
    ScaledSum,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


class RayleighDamping(Generic[V, A, F]):
    r""":math:`g = -(\alpha M + \beta K) v` (see the module docstring).

    Satisfies
    :class:`~pyapprox.ode.second_order.reduction.VelocityForceProtocol`.

    Parameters
    ----------
    restoring : MassFormSystem
        The restoring system: its mass :math:`M` and its force's Jacobian
        :math:`f_u = -K`.
    alpha, beta : float
        The mass and stiffness coefficients.
    reference_state : V, optional
        :math:`u_r`, in the restoring system's residual form (as its
        states are); the zero state by default. Required when ``beta`` is
        nonzero and the force is not affine in the state.

    Raises
    ------
    ValueError
        If ``beta`` is nonzero and the force has no Jacobian, or is not
        affine in the state and no reference state is given.
    """

    def __init__(
        self,
        restoring: MassFormSystem[V, A, F],
        alpha: float,
        beta: float,
        reference_state: Optional[V] = None,
    ) -> None:
        jacobian = restoring.rhs.derivatives().jacobian
        if beta != 0.0:
            if jacobian is None:
                raise ValueError(
                    "stiffness-proportional damping needs the restoring force's "
                    "Jacobian, and the force declares none"
                )
            if reference_state is None and not jacobian.affine():
                raise ValueError(
                    "stiffness-proportional damping of a restoring force not "
                    "affine in the state needs a reference_state to linearize it "
                    "at: following the tangent stiffness would need a directional "
                    "second derivative its bundle does not supply"
                )
        self._restoring, self._alpha, self._beta = restoring, alpha, beta
        # The force's Jacobian, read once; None when beta is zero.
        self._stiffness: Optional[Jacobian[V, F]] = None if beta == 0.0 else jacobian
        self._reference = (
            restoring.form.to_residual(restoring.space.zeros())
            if reference_state is None
            else reference_state
        )
        self._version = parameters_version_reader(restoring.rhs)
        zero = ScaledSum([(0.0, restoring.mass)], restoring.space)
        wrt_displacement: Jacobian[V, F] = constant_jacobian(zero)
        wrt_velocity: Jacobian[V, F] = (
            constant_jacobian(ScaledSum([(-alpha, restoring.mass)], restoring.space))
            if beta == 0.0
            else Jacobian.state_free(self._stiffness_jacobian)
            if jacobian is not None and not jacobian.time_invariant()
            else Jacobian.constant(self._constant_stiffness_jacobian)
        )
        parameterized = restoring.rhs.derivatives().param_space is not None
        self._derivatives: VelocityForceDerivatives[V, F]
        if beta == 0.0 or not parameterized:
            self._derivatives = VelocityForceDerivatives.parameter_free(
                wrt_displacement, wrt_velocity, affine=True
            )
        elif jacobian is not None and jacobian.affine():
            # K(p) v = f(v; p) - f(0; p): the reduction derives g_p from f's.
            self._derivatives = VelocityForceDerivatives.through_stiffness(
                wrt_displacement, wrt_velocity, beta
            )
        else:
            # K(u_r; p) v is no difference of f's values: g_p is unsupplied.
            self._derivatives = VelocityForceDerivatives.state_only(
                wrt_displacement, wrt_velocity
            )

    def __call__(self, displacement: V, velocity: V, time: float) -> V:
        system = self._restoring
        column = system.form.from_residual(velocity)
        damping = self._operator(time, LinearizationPoint(self._version(), None))
        return system.form.to_residual(damping.apply(column))

    def derivatives(self) -> VelocityForceDerivatives[V, F]:
        return self._derivatives

    def _operator(
        self, time: float, point: LinearizationPoint
    ) -> ActionOperatorProtocol[V, F]:
        r""":math:`g_v = -\alpha M + \beta f_u(u_r)`."""
        system = self._restoring
        terms: List[Tuple[float, ActionOperatorProtocol[V, F]]] = [
            (-self._alpha, system.mass)
        ]
        if self._stiffness is not None:
            terms.append((self._beta, self._stiffness(self._reference, time, point)))
        return ScaledSum(terms, system.space)

    def _stiffness_jacobian(
        self, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, F]:
        return LinearizedOperator(self._operator(time, point), point)

    def _constant_stiffness_jacobian(
        self, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, F]:
        return LinearizedOperator(self._operator(0.0, point), point)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(alpha={self._alpha}, beta={self._beta}, "
            f"restoring={self._restoring!r})"
        )
