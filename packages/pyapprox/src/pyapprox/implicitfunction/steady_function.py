r"""A steady functional as a function of the parameters.

The function :math:`q(p) = Q(u(p);\, p)`, where :math:`u(p)` solves
:math:`R(u;\, p) = 0`, maps a batch of parameter samples to their values
and carries its derivatives in a
:class:`~pyapprox.interface.functions.derivatives.Derivatives` bundle, so
anything that takes a :class:`~pyapprox.interface.functions.protocols.
function.FunctionProtocol` (sampling, sensitivity analysis, optimization,
inference) can use it. It solves with a
:class:`~pyapprox.implicitfunction.steady_system.SteadySystem` and
differentiates with :class:`~pyapprox.implicitfunction.steady_derivatives.
SteadyDerivatives`; the residual, the linear-solver setups and the
nonlinear solver are all injected.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.implicitfunction.functional_terms import has_first_derivatives
from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import (
    SteadyDerivatives,
    SteadyHessian,
)
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
    validate_samples,
)
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


class SteadyImplicitFunction(Generic[Array, F]):
    r""":math:`q(p) = Q(u(p);\, p)` with :math:`R(u(p);\, p) = 0`, as a
    function of the parameters.

    Satisfies ``FunctionProtocol``. Each sample sets the parameters,
    clears ``setups``, and solves from ``init_state``.

    ``derivatives()`` is decided once, here:

    - none when the residual has no parameter Jacobian or a term of
      ``functional`` has no first derivatives;
    - otherwise the Jacobian, by one adjoint solve per entry when there
      are no more entries than parameters, else by one tangent solve per
      parameter;
    - for a single entry, also the Hessian-vector product when the
      residual and every term carry second derivatives.

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
        The linear-solver setups for :math:`R_u`; cleared at every new
        sample.
    nonlinear : NonlinearSolverProtocol
        Solves :math:`R(u;\, p) = 0`.
    functional : StateFunctional
        :math:`Q(u;\, p)`.
    init_state : Array
        The initial guess of every solve. Shape: ``(nstates, 1)``.
    bkd : Backend
        The backend of the samples and values.

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
        bkd: Backend[Array],
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
        self._init_state, self._bkd = init_state, bkd
        self._derivatives: Optional[SteadyDerivatives[Array, Array, F]] = None
        self._hessian: Optional[SteadyHessian[Array, Array]] = None
        self._derivs: Derivatives[Array] = Derivatives.none()
        bundle = residual.derivatives()
        if bundle.param is None or not has_first_derivatives(
            functional.parameter_terms, functional.state_terms
        ):
            return
        derivatives: SteadyDerivatives[Array, Array, F] = SteadyDerivatives(
            residual, space, functional
        )
        self._derivatives = derivatives
        self._hessian = derivatives.hessian() if functional.nqoi() == 1 else None
        if self._hessian is not None:
            self._derivs = Derivatives.second_order(
                jacobian=self._jacobian, hvp=self._hvp
            )
            return
        self._derivs = Derivatives.first_order(jacobian=self._jacobian)

    def bkd(self) -> Backend[Array]:
        """The backend of the samples and values."""
        return self._bkd

    def nvars(self) -> int:
        """The number of parameters."""
        return self._parameters.nparams()

    def nqoi(self) -> int:
        """The number of entries of the functional."""
        return self._functional.nqoi()

    def derivatives(self) -> Derivatives[Array]:
        """The derivative bundle, with respect to the parameters."""
        return self._derivs

    def __call__(self, samples: Array) -> Array:
        """The functional at each sample.

        Parameters
        ----------
        samples : Array
            Shape: ``(nvars, nsamples)``.

        Returns
        -------
        Array
            Shape: ``(nqoi, nsamples)``.

        Raises
        ------
        ValueError
            If setting the parameters does not move the residual's
            parameter version.
        RuntimeError
            If a solve does not converge.
        """
        validate_samples(self.nvars(), samples)
        return self._bkd.hstack(
            [
                self._functional.value(self._solve(samples[:, ii : ii + 1]).state())
                for ii in range(samples.shape[1])
            ]
        )

    def _solve(self, sample: Array) -> SteadyPoint[Array, Array]:
        before = self._system.residual_version()
        self._parameters.set_param(sample[:, 0])
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

    def _derivatives_or_raise(self) -> SteadyDerivatives[Array, Array, F]:
        if self._derivatives is None:
            raise RuntimeError(
                "jacobian is unavailable; check derivatives() before calling"
            )
        return self._derivatives

    def _jacobian(self, sample: Array) -> Array:
        """:math:`dq/dp` at one sample. Shape: ``(nqoi, nvars)``."""
        derivatives = self._derivatives_or_raise()
        validate_sample(self.nvars(), sample)
        point = self._solve(sample)
        bkd, nqoi, nvars = self._bkd, self.nqoi(), self.nvars()
        if nqoi <= nvars:
            eye = bkd.eye(nqoi)
            rows = [
                derivatives.gradient(derivatives.adjoint(point, eye[:, ii : ii + 1])).T
                for ii in range(nqoi)
            ]
            return bkd.vstack(rows)
        eye = bkd.eye(nvars)
        return bkd.hstack(
            [derivatives.tangent(point, eye[:, jj : jj + 1]) for jj in range(nvars)]
        )

    def _hvp(self, sample: Array, vec: Array) -> Array:
        r""":math:`\nabla^2 q\, v` at one sample. Shape: ``(nvars, 1)``."""
        hessian = self._hessian
        if hessian is None:
            raise RuntimeError("hvp is unavailable; check derivatives() before calling")
        derivatives = self._derivatives_or_raise()
        validate_sample(self.nvars(), sample)
        validate_sample(self.nvars(), vec)
        point = self._solve(sample)
        return hessian.hvp(derivatives.adjoint(point), vec)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"derivatives={self._derivs!r})"
        )
