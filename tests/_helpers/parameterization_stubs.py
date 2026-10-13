"""Stand-in parameterizations for physics that have none yet."""

from typing import Any, Tuple

from pyapprox.pde.parameterizations.derivatives import ParamDerivatives


class ApplyOnlyParameterization:
    """Targets a physics and writes nothing, with no derivatives.

    Lets value-only tests (balance identities) build objects that need a
    parameterization bound to the physics.
    """

    def __init__(self, physics: object) -> None:
        self._physics = physics

    def nparams(self) -> int:
        return 1

    def targets(self) -> Tuple[object, ...]:
        return (self._physics,)

    def owned_coefficients(self) -> Tuple[str, ...]:
        return ("none",)

    def apply(self, params_1d: Any) -> None:
        pass

    def param_derivatives(self) -> ParamDerivatives[Any]:
        return ParamDerivatives.none()
