"""The :mod:`pyapprox.benchmarks` module implements numerous benchmarks from
the modeling literature.
"""

from pyapprox.benchmarks.benchmarks import (
    setup_ishigami_function, setup_sobol_g_function,
    setup_oakley_function, setup_genz_function,
    setup_rosenbrock_function, setup_benchmark, Benchmark,
    setup_piston_benchmark, setup_wing_weight_benchmark,
    setup_polynomial_ensemble, list_benchmarks,
    setup_advection_diffusion_kle_inversion_benchmark,
    setup_multi_index_advection_diffusion_benchmark
)

__all__ = ["setup_ishigami_function", "setup_sobol_g_function",
           "setup_oakley_function", "setup_genz_function",
           "setup_rosenbrock_function", "setup_benchmark", "Benchmark",
           "setup_piston_benchmark", "setup_wing_weight_benchmark",
           "setup_polynomial_ensemble", "list_benchmarks",
           "setup_advection_diffusion_kle_inversion_benchmark",
           "setup_multi_index_advection_diffusion_benchmark"]
