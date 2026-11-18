"""Zoo of pre-configured forward models for common PDE problems."""

from .obstructed_flow import (
    ParabolicInlet,
    ZeroVelocity,
    build_obstructed_mesh,
    extract_velocity_callable,
    solve_obstructed_stokes,
)

__all__ = [
    "ParabolicInlet",
    "ZeroVelocity",
    "build_obstructed_mesh",
    "extract_velocity_callable",
    "solve_obstructed_stokes",
]
