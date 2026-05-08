"""Shared configuration for time integration methods.

Used by both collocation and Galerkin modules.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TimeIntegrationConfig:
    """Configuration for time integration.

    Parameters
    ----------
    method : str
        Time integration method. One of:
        - "forward_euler": Explicit first-order
        - "backward_euler": Implicit first-order (A-stable)
        - "crank_nicolson": Implicit second-order
        - "heun": Explicit second-order (RK2)
    init_time : float
        Initial time. Default: 0.0
    final_time : float
        Final time.
    deltat : float
        Time step size.
    newton_tol : float
        Newton solver tolerance for implicit methods. Default: 1e-10
    newton_maxiter : int
        Newton solver maximum iterations. Default: 20
    verbosity : int
        Verbosity level. Default: 0
    """

    method: Literal["forward_euler", "backward_euler", "crank_nicolson", "heun"] = (
        "backward_euler"
    )
    init_time: float = 0.0
    final_time: float = 1.0
    deltat: float = 0.01
    newton_tol: float = 1e-10
    newton_maxiter: int = 20
    lumped_mass: bool = False
    verbosity: int = 0
