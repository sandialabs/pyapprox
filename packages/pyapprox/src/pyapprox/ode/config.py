"""Shared configuration for time integration methods.

Used by both collocation and Galerkin modules.
"""

from dataclasses import dataclass
from typing import Generic, Union

from pyapprox.ode.stepper_table import StepperFactory
from pyapprox.util.backends.protocols import Array


@dataclass
class TimeIntegrationConfig(Generic[Array]):
    """Configuration for time integration.

    All fields are required: time-integration settings are problem
    dependent, so silent defaults would mask configuration mistakes.

    Parameters
    ----------
    method : str or StepperFactory
        Time integration method: a built-in name resolved against the
        closed stepper table, or a StepperFactory callable (typed
        extension path for custom steppers; no registration needed).
        Built-in names:
        - "forward_euler": Explicit first-order
        - "backward_euler": Implicit first-order (A-stable)
        - "crank_nicolson": Implicit second-order
        - "heun": Explicit second-order (RK2)
        - "implicit_midpoint": Implicit second-order (A-stable,
          symplectic; conserves a modified energy for nonlinear
          Hamiltonian systems, where Crank-Nicolson drifts)
    init_time : float
        Initial time.
    final_time : float
        Final time.
    deltat : float
        Time step size.
    newton_tol : float
        Newton solver tolerance for implicit methods.
    newton_maxiter : int
        Newton solver maximum iterations.
    lumped_mass : bool
        Use a lumped (diagonal) mass matrix for explicit Galerkin
        stepping instead of the consistent mass.
    verbosity : int
        Verbosity level.
    """

    method: Union[str, StepperFactory[Array]]
    init_time: float
    final_time: float
    deltat: float
    newton_tol: float
    newton_maxiter: int
    lumped_mass: bool
    verbosity: int
