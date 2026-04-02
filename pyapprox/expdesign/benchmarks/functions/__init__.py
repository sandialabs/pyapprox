"""Pure forward maps for OED benchmarks (FunctionProtocol)."""

from .linear_gaussian import (
    build_exp_qoi_map,
    build_linear_gaussian_inference_problem,
    build_linear_obs_map,
    build_linear_qoi_map,
)

__all__ = [
    "build_linear_obs_map",
    "build_linear_qoi_map",
    "build_exp_qoi_map",
    "build_linear_gaussian_inference_problem",
]
