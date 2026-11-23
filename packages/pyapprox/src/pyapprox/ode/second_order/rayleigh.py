r"""Rayleigh damping over a restoring system's operators.

:class:`RayleighDamping` is the velocity force

.. math::

    g(u, v, t) = -(\alpha M + \beta K)\, v, \qquad K = -f_u,

read from the restoring system's mass and its force's Jacobian whenever it
is evaluated, so :math:`K` follows the parameters the force is set to; a
matrix copied at construction would stop tracking them.

With :math:`\beta \ne 0` the force must be affine in the state: for a
nonlinear :math:`f`, :math:`K(u)` is a tangent and :math:`g_u = \beta
f_{uu}[v]` needs a directional second derivative the force's bundle does
not supply. Its parameter dependence, :math:`g_p = -\beta K_p v`, is not
supplied either, so with :math:`\beta \ne 0` and a parameterized force
the reduced system has state derivatives only.
"""

from typing import Generic, List, Tuple, TypeVar

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

    Raises
    ------
    ValueError
        If ``beta`` is nonzero and the force has no Jacobian or is not
        affine in the state.
    """

    def __init__(
        self, restoring: MassFormSystem[V, A, F], alpha: float, beta: float
    ) -> None:
        jacobian = restoring.rhs.derivatives().jacobian
        if beta != 0.0 and (jacobian is None or not jacobian.affine()):
            raise ValueError(
                "stiffness-proportional damping needs a restoring force affine "
                "in the state: for a nonlinear force, g_u needs a directional "
                "second derivative its bundle does not supply"
            )
        self._restoring, self._alpha, self._beta = restoring, alpha, beta
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
        self._derivatives: VelocityForceDerivatives[V, F] = (
            VelocityForceDerivatives.state_only(wrt_displacement, wrt_velocity)
            if beta != 0.0 and parameterized
            else VelocityForceDerivatives.parameter_free(
                wrt_displacement, wrt_velocity, affine=True
            )
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
        r""":math:`g_v = -\alpha M + \beta f_u`."""
        system = self._restoring
        terms: List[Tuple[float, ActionOperatorProtocol[V, F]]] = [
            (-self._alpha, system.mass)
        ]
        if self._beta != 0.0:
            zero = system.form.to_residual(system.space.zeros())
            terms.append((self._beta, system.jacobian(zero, time, point)))
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
