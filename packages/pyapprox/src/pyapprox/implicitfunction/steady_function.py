r"""A steady functional as a function of the parameters.

The map :math:`p \mapsto Q(u(p);\, p)`, where :math:`u(p)` solves
:math:`R(u;\, p) = 0`, is a :class:`SteadyImplicitMap`: it solves with a
:class:`~pyapprox.implicitfunction.steady_system.SteadySystem`,
differentiates with :class:`~pyapprox.implicitfunction.steady_derivatives.
SteadyDerivatives`, and declares what it can differentiate in an
:class:`~pyapprox.implicitfunction.implicit_map.ImplicitMapDerivatives`
bundle over its solved point, :class:`SteadyImplicitPoint`. The residual,
the linear-solver setups and the nonlinear solver are all injected.

:class:`SteadyImplicitFunction` is that map as a
:class:`~pyapprox.interface.functions.protocols.function.FunctionProtocol`
of backend sample columns, by :class:`~pyapprox.implicitfunction.
implicit_map.ImplicitMapFunction`, so anything that takes one (sampling,
sensitivity analysis, optimization, inference) can use it.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.implicitfunction.functional_terms import has_first_derivatives
from pyapprox.implicitfunction.implicit_map import (
    ArrayParameters,
    ImplicitJacobianProtocol,
    ImplicitMapDerivatives,
    ImplicitMapFunction,
)
from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import FormatProtocol, InnerProductSpaceProtocol
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

F = TypeVar("F", bound=FormatProtocol)
Array_contra = TypeVar("Array_contra", contravariant=True)


@runtime_checkable
class ParameterSetterProtocol(Protocol[Array_contra]):
    """Moves the parameters that a residual, and its functional's terms,
    read."""

    def nparams(self) -> int:
        """The number of parameters."""
        ...

    def set_param(self, param: Array_contra) -> None:
        """Apply ``param``, shape ``(nparams,)``."""
        ...


class SteadyImplicitPoint(Generic[Array, F]):
    r"""A steady solve at one parameter, with the derivatives taken there:
    the point of :class:`SteadyImplicitMap`, whose bundle's fields are
    this class's methods.

    Parameters
    ----------
    point : SteadyPoint
        The converged solve.
    derivatives : SteadyDerivatives, optional
        The map's derivatives; None when it has none.
    """

    def __init__(
        self,
        point: SteadyPoint[Array, Array],
        derivatives: Optional[SteadyDerivatives[Array, Array, F]],
    ) -> None:
        self._point, self._derivatives = point, derivatives

    def state(self) -> Array:
        """The solved state. Shape: ``(nstates, 1)``."""
        return self._point.state()

    def _derivatives_or_raise(self) -> SteadyDerivatives[Array, Array, F]:
        if self._derivatives is None:
            raise RuntimeError(
                "the map has no derivatives; check derivatives() before calling"
            )
        return self._derivatives

    def tangent(self, direction: Array) -> Array:
        r""":math:`dQ/dp\, v`, one tangent solve. Shape: ``(nqoi, 1)``."""
        return self._derivatives_or_raise().tangent(self._point, direction)

    def gradient(self, weights: Optional[Array]) -> Array:
        r""":math:`d(\omega^\top Q)/dp`, one adjoint solve; ``weights`` of
        shape ``(nqoi, 1)``, None only for one entry. Shape:
        ``(nparams, 1)``."""
        derivatives = self._derivatives_or_raise()
        return derivatives.gradient(derivatives.adjoint(self._point, weights))

    def hvp(self, direction: Array, weights: Optional[Array]) -> Array:
        r""":math:`(d^2 \omega^\top Q/dp^2)\, v`: an adjoint solve, then a
        tangent and a second adjoint solve. Shape: ``(nparams, 1)``."""
        derivatives = self._derivatives_or_raise()
        hessian = derivatives.hessian()
        if hessian is None:
            raise RuntimeError(
                "the map has no second derivatives; check derivatives() before calling"
            )
        return hessian.hvp(derivatives.adjoint(self._point, weights), direction)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SteadyImplicitMap(Generic[Array, F]):
    r""":math:`p \mapsto Q(u(p);\, p)` with :math:`R(u(p);\, p) = 0`: an
    implicit map (``ImplicitMapProtocol``) whose parameter vectors are
    backend columns ``(nparams, 1)``.

    Each solve sets the parameters, clears ``setups``, and solves from
    ``init_state``. ``derivatives()`` is decided once, here: none when the
    residual has no parameter Jacobian or a term of ``functional`` has no
    first derivatives; otherwise tangents and gradients; and the
    Hessian-vector product, for every ``nqoi``, when the residual and every
    term carry second derivatives.

    States are backend arrays: the state space's vectors are ``Array``.

    Parameters
    ----------
    residual : ResidualProtocol
        :math:`R(u;\, p)`, reading the parameters ``parameters`` sets.
    space : InnerProductSpaceProtocol
        The state space.
    parameters : ParameterSetterProtocol
        Sets the parameters; every ``set_param`` must move ``residual``'s
        parameter version.
    setups : SetupStrategyProtocol
        The linear-solver setups for :math:`R_u`; cleared at every solve.
    nonlinear : NonlinearSolverProtocol
        Solves :math:`R(u;\, p) = 0`.
    functional : StateFunctional
        :math:`Q(u;\, p)`.
    init_state : Array
        The initial guess of every solve. Shape: ``(nstates, 1)``.

    Raises
    ------
    TypeError
        If ``parameters`` or ``functional`` is of the wrong type (the
        others are checked by
        :class:`~pyapprox.implicitfunction.steady_system.SteadySystem`).
    ValueError
        If ``init_state`` does not have shape ``(nstates, 1)``.
    """

    def __init__(
        self,
        residual: ResidualProtocol[Array, F],
        space: InnerProductSpaceProtocol[Array, Array],
        parameters: ParameterSetterProtocol[Array],
        setups: SetupStrategyProtocol[SteadyRequest[Array, F], Array, F],
        nonlinear: NonlinearSolverProtocol[Array, Array],
        functional: StateFunctional[Array, Array],
        init_state: Array,
    ) -> None:
        self._system: SteadySystem[Array, Array, F] = SteadySystem(
            residual, space, setups
        )
        if not isinstance(parameters, ParameterSetterProtocol):
            raise TypeError(
                "parameters must satisfy ParameterSetterProtocol, got "
                f"{type(parameters).__name__}"
            )
        if not isinstance(functional, StateFunctional):
            raise TypeError(
                f"functional must be a StateFunctional, got {type(functional).__name__}"
            )
        nstates = space.descriptor().size
        if tuple(init_state.shape) != (nstates, 1):
            raise ValueError(
                f"init_state must have shape ({nstates}, 1), got "
                f"{tuple(init_state.shape)}"
            )
        self._parameters, self._setups = parameters, setups
        self._nonlinear, self._functional = nonlinear, functional
        self._init_state = init_state
        self._steady: Optional[SteadyDerivatives[Array, Array, F]] = None
        self._derivatives: ImplicitMapDerivatives[
            SteadyImplicitPoint[Array, F], Array, Array
        ] = ImplicitMapDerivatives.none()
        bundle = residual.derivatives()
        if bundle.param is None or not has_first_derivatives(
            functional.parameter_terms, functional.state_terms
        ):
            return
        steady: SteadyDerivatives[Array, Array, F] = SteadyDerivatives(
            residual, space, functional
        )
        self._steady = steady
        self._derivatives = (
            ImplicitMapDerivatives.first_order(
                SteadyImplicitPoint.tangent, SteadyImplicitPoint.gradient
            )
            if steady.hessian() is None
            else ImplicitMapDerivatives.second_order(
                SteadyImplicitPoint.tangent,
                SteadyImplicitPoint.gradient,
                SteadyImplicitPoint.hvp,
            )
        )

    def nparams(self) -> int:
        """The number of parameters."""
        return self._parameters.nparams()

    def nqoi(self) -> int:
        """The number of entries of the functional."""
        return self._functional.nqoi()

    def derivatives(
        self,
    ) -> ImplicitMapDerivatives[SteadyImplicitPoint[Array, F], Array, Array]:
        """Which derivatives a point has, fixed at construction."""
        return self._derivatives

    def value(self, param: Array) -> Array:
        """:math:`Q` at the parameter column ``param``. Shape:
        ``(nqoi, 1)``.

        Raises
        ------
        ValueError
            If setting the parameters does not move the residual's
            parameter version.
        RuntimeError
            If the solve does not converge.
        """
        return self._functional.value(self._solve(param).state())

    def point(self, param: Array) -> SteadyImplicitPoint[Array, F]:
        """The solve at the parameter column ``param``, for the
        derivatives; raises as :meth:`value`."""
        return SteadyImplicitPoint(self._solve(param), self._steady)

    def _solve(self, param: Array) -> SteadyPoint[Array, Array]:
        before = self._system.residual_version()
        self._parameters.set_param(param[:, 0])
        if self._system.residual_version() == before:
            raise ValueError(
                "setting the parameters did not move the residual's parameter "
                "version: the residual does not read these parameters"
            )
        self._setups.clear()
        point = self._system.solve(self._nonlinear, self._init_state)
        if not point.converged():
            raise RuntimeError(
                f"the steady solve did not converge: {point.info().message}"
            )
        return point

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nparams={self.nparams()}, nqoi={self.nqoi()}, "
            f"derivatives={self._derivatives!r})"
        )


class SteadyImplicitFunction(
    ImplicitMapFunction[SteadyImplicitPoint[Array, F], Array, Array]
):
    r""":math:`q(p) = Q(u(p);\, p)` with :math:`R(u(p);\, p) = 0`, as a
    function of the parameters: a :class:`SteadyImplicitMap` through
    :class:`~pyapprox.implicitfunction.implicit_map.ImplicitMapFunction`,
    with the identity parameter-space map.

    Satisfies ``FunctionProtocol``. ``derivatives()`` is decided once,
    here: none when the map has none; otherwise the Jacobian, formed by
    ``jacobian``; and the weighted Hessian-vector product ``whvp`` for
    every ``nqoi`` when the map has second derivatives, with the plain
    ``hvp`` as well for one entry.

    Parameters
    ----------
    residual, space, parameters, setups, nonlinear, functional, init_state
        As :class:`SteadyImplicitMap`.
    bkd : Backend
        The backend of the samples and values.
    jacobian : ImplicitJacobianProtocol, optional
        How the Jacobian is formed. Default:
        :class:`~pyapprox.implicitfunction.implicit_map.AdjointRows` when
        there are no more entries than parameters, else
        :class:`~pyapprox.implicitfunction.implicit_map.TangentColumns`,
        whichever takes fewer solves.

    Raises
    ------
    TypeError
        As :class:`SteadyImplicitMap`, or if ``jacobian`` does not satisfy
        ``ImplicitJacobianProtocol``.
    ValueError
        As :class:`SteadyImplicitMap`.
    """

    def __init__(
        self,
        residual: ResidualProtocol[Array, F],
        space: InnerProductSpaceProtocol[Array, Array],
        parameters: ParameterSetterProtocol[Array],
        setups: SetupStrategyProtocol[SteadyRequest[Array, F], Array, F],
        nonlinear: NonlinearSolverProtocol[Array, Array],
        functional: StateFunctional[Array, Array],
        init_state: Array,
        bkd: Backend[Array],
        jacobian: Optional[ImplicitJacobianProtocol] = None,
    ) -> None:
        implicit: SteadyImplicitMap[Array, F] = SteadyImplicitMap(
            residual, space, parameters, setups, nonlinear, functional, init_state
        )
        super().__init__(implicit, ArrayParameters(bkd), jacobian)
