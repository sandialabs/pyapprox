"""The :mod:`pyapprox.benchmarks` module implements numerous benchmarks from
the modeling literature.
"""

from pyapprox.benchmarks.benchmarks import (
    setup_ishigami_function, setup_sobol_g_function,
    setup_oakley_function, setup_genz_function,
    setup_rosenbrock_function, setup_benchmark, Benchmark,
    setup_piston_benchmark, setup_wing_weight_benchmark,
    setup_polynomial_ensemble
)

__all__ = ["setup_ishigami_function", "setup_sobol_g_function",
           "setup_oakley_function", "setup_genz_function",
           "setup_rosenbrock_function", "setup_benchmark", "Benchmark",
           "setup_piston_benchmark", "setup_wing_weight_benchmark",
           "setup_polynomial_ensemble"]
