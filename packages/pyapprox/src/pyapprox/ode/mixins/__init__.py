"""Mixin classes for time stepping residuals.

Provides composable capabilities for time steppers via mixin inheritance.
Each mixin declares abstract methods that concrete steppers must implement.

Mixin Hierarchy (most specific leftmost in MRO)
------------------------------------------------
CoreStepperMixin          -- __init__, bind, bkd, native_residual, __repr__
ImplicitStepperMixin      -- jacobian (abstract), linsolve
SensitivityMixin          -- is_explicit, has_prev_state_hessian,
                             sensitivity_off_diag_jacobian
AdjointMixin              -- adjoint methods, param_jacobian
HVPMixin                  -- 4 same-step HVP + 3 cross-step prev_* HVP
"""

from pyapprox.ode.mixins.adjoint import AdjointMixin
from pyapprox.ode.mixins.core import CoreStepperMixin
from pyapprox.ode.mixins.hvp import HVPMixin
from pyapprox.ode.mixins.implicit import ImplicitStepperMixin
from pyapprox.ode.mixins.sensitivity import SensitivityMixin

__all__ = [
    "CoreStepperMixin",
    "ImplicitStepperMixin",
    "SensitivityMixin",
    "AdjointMixin",
    "HVPMixin",
]
