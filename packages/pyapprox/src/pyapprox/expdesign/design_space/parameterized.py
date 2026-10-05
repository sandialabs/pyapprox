"""An objective of the observation weights, as a function of other variables.

Given a map ``v -> w`` and an objective ``f(w)``, the composite ``f(w(v))``
has gradient ``J(v)^T grad f(w)``. A linear map ``w = P v`` (grouped
designs) and a nonlinear one are the same code.
"""

from typing import Callable, Generic

from pyapprox.expdesign.protocols.objective import OEDObjectiveProtocol
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols import ObjectiveProtocol
from pyapprox.util.backends.protocols import Array, Backend


class _ChainRuleJacobian(Generic[Array]):
    """``v -> (df/dw)(w(v)) (dw/dv)(v)``, shape (1, nvars).

    A class rather than a closure so the Derivatives bundle, and with it the
    objective, stays picklable.
    """

    def __init__(
        self,
        param: ObjectiveProtocol[Array],
        objective_jac: Callable[[Array], Array],
        param_jac: Callable[[Array], Array],
    ) -> None:
        self._param = param
        self._objective_jac = objective_jac
        self._param_jac = param_jac

    def __call__(self, design_variables: Array) -> Array:
        weights = self._param(design_variables)
        return self._param.bkd().dot(
            self._objective_jac(weights), self._param_jac(design_variables)
        )


class ParameterizedObjective(Generic[Array]):
    """``f(w(v))``, an objective of the design variables ``v``.

    Satisfies ``OEDObjectiveProtocol`` over ``v``. Its Jacobian is
    ``(df/dw)(w(v)) (dw/dv)(v)``, declared when both ``objective`` and
    ``param`` declare a Jacobian; otherwise the bundle is empty and
    optimizers fall back to finite differences of the value.

    Parameters
    ----------
    objective : OEDObjectiveProtocol[Array]
        The objective of the observation weights ``w``.
    param : ObjectiveProtocol[Array]
        The map ``v -> w``, with ``nqoi()`` equal to ``objective.nvars()``,
        and a ``derivatives()`` bundle that may be empty. For example a
        ``GroupedDesign``.
    """

    def __init__(
        self,
        objective: OEDObjectiveProtocol[Array],
        param: ObjectiveProtocol[Array],
    ) -> None:
        if not isinstance(objective, OEDObjectiveProtocol):
            raise TypeError(
                "objective must satisfy OEDObjectiveProtocol, got "
                f"{type(objective).__name__}"
            )
        if not isinstance(param, ObjectiveProtocol):
            raise TypeError(
                f"param must satisfy ObjectiveProtocol, got {type(param).__name__}"
            )
        if param.nqoi() != objective.nvars():
            raise ValueError(
                f"param maps to {param.nqoi()} weights but the objective takes "
                f"{objective.nvars()}"
            )
        self._objective = objective
        self._param = param
        objective_jac = objective.derivatives().jacobian
        param_jac = param.derivatives().jacobian
        if objective_jac is not None and param_jac is not None:
            self._derivatives = Derivatives.first_order(
                jacobian=_ChainRuleJacobian(param, objective_jac, param_jac)
            )
        else:
            self._derivatives = Derivatives.none()

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._objective.bkd()

    def nvars(self) -> int:
        """Number of design variables ``v``."""
        return self._param.nvars()

    def nqoi(self) -> int:
        """Always 1."""
        return 1

    def weights(self, design_variables: Array) -> Array:
        """``w(v)``. Shape: (nvars, n) to (nobs, n)"""
        return self._param(design_variables)

    def __call__(self, design_weights: Array) -> Array:
        """``f(w(v))``. Variables (nvars, n) give values of shape (1, n)."""
        bkd = self.bkd()
        weights = self._param(design_weights)
        values = [
            self._objective(weights[:, ii : ii + 1]) for ii in range(weights.shape[1])
        ]
        return bkd.reshape(bkd.hstack(values), (1, -1))

    def derivatives(self) -> Derivatives[Array]:
        """First-order bundle when both pieces have a Jacobian, else empty."""
        return self._derivatives
