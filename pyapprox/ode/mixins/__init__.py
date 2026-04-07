"""Mixin classes for time stepping residuals.

Provides composable capabilities for time steppers via mixin inheritance.
Each mixin declares abstract methods that concrete steppers must implement.

Mixin Hierarchy (most specific leftmost in MRO)
------------------------------------------------
CoreStepperMixin          -- __init__, set_time, bkd, native_residual, __repr__
ImplicitStepperMixin      -- jacobian (abstract), linsolve
SensitivityMixin          -- is_explicit, has_prev_state_hessian,
                             sensitivity_off_diag_jacobian
QuadratureMixin           -- _get_quadrature_class,
                             quadrature_samples_weights
AdjointMixin              -- adjoint methods, param_jacobian
HVPMixin                  -- state_state_hvp, state_param_hvp,
                             param_state_hvp, param_param_hvp
PrevStepHVPMixin          -- prev_state_state_hvp,
                             prev_state_param_hvp,
                             prev_param_state_hvp
"""

from pyapprox.ode.mixins.adjoint import AdjointMixin
from pyapprox.ode.mixins.core import CoreStepperMixin
from pyapprox.ode.mixins.hvp import HVPMixin, PrevStepHVPMixin
from pyapprox.ode.mixins.implicit import ImplicitStepperMixin
from pyapprox.ode.mixins.quadrature import QuadratureMixin
from pyapprox.ode.mixins.sensitivity import SensitivityMixin

__all__ = [
    "CoreStepperMixin",
    "ImplicitStepperMixin",
    "SensitivityMixin",
    "QuadratureMixin",
    "AdjointMixin",
    "HVPMixin",
    "PrevStepHVPMixin",
]
