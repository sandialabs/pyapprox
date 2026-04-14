"""Benchmark suite for the PyApprox typing module.

This module provides a collection of benchmark problems for testing and
validating algorithms in sensitivity analysis, optimization, quadrature,
multifidelity methods, and inverse problems.

Design Principles:
- Functions implement existing protocols from typing.interface.functions.protocols
- No inheritance hierarchies - composition and protocol compliance
- HVP (Hessian-vector product) preferred over full Hessian for efficiency
- Fixed benchmark instances with known ground truth
- Backend-agnostic using Backend[Array] protocol
"""

from pyapprox_benchmarks.benchmark import (
    Benchmark,
    BenchmarkWithPrior,
    BoxDomain,
    ConstrainedBenchmark,
)

# Functions
from pyapprox_benchmarks.functions.algebraic import (
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
    BraninFunction,
    CantileverBeam1DAnalytical,
    CantileverBeam2DAnalytical,
    HomogeneousBeam1DAnalytical,
    IshigamiFunction,
    IshigamiSensitivityIndices,
    RosenbrockFunction,
    SobolGFunction,
    SobolGSensitivityIndices,
)
from pyapprox_benchmarks.functions.genz import (
    CornerPeakFunction,
    GaussianPeakFunction,
    OscillatoryFunction,
    ProductPeakFunction,
)
from pyapprox_benchmarks.functions.multifidelity import (
    PolynomialModelFunction,
)

# ODE
from pyapprox_benchmarks.functions.ode import (
    ODEQoIFunction,
    ODETimeConfig,
)
from pyapprox_benchmarks.ground_truth import (
    InverseGroundTruth,
    MultifidelityGroundTruth,
    ODEGroundTruth,
    OptimizationGroundTruth,
    QuadratureGroundTruth,
    SensitivityGroundTruth,
)

# Instances (also registers with BenchmarkRegistry)
from pyapprox_benchmarks.instances import (
    branin_2d,
    cantilever_beam_1d,
    cantilever_beam_1d_analytical,
    cantilever_beam_2d_analytical,
    cantilever_beam_2d_linear,
    cantilever_beam_2d_neohookean,
    elastic_bar_1d,
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_gaussian_peak_5d,
    genz_oscillatory_2d,
    genz_oscillatory_5d,
    genz_product_peak_2d,
    ishigami_3d,
    rosenbrock_2d,
    rosenbrock_10d,
    sobol_g_4d,
    sobol_g_6d,
)

# ODE builders
from pyapprox_benchmarks.ode import (
    build_chemical_reaction_surface,
    build_coupled_springs_2mass,
    build_hastings_ecology_3species,
    build_lotka_volterra_3species,
)
from pyapprox_benchmarks.protocols import (
    BenchmarkProtocol,
    BenchmarkWithPriorProtocol,
    ConstrainedBenchmarkProtocol,
    ConstraintProtocol,
    DomainProtocol,
    GroundTruthProtocol,
    HasExactEIG,
    KLOEDBenchmarkProtocol,
    PredictionOEDBenchmarkProtocol,
)
from pyapprox_benchmarks.registry import BenchmarkRegistry

__all__ = [
    # Protocols
    "DomainProtocol",
    "GroundTruthProtocol",
    "BenchmarkProtocol",
    "BenchmarkWithPriorProtocol",
    "ConstraintProtocol",
    "ConstrainedBenchmarkProtocol",
    "HasExactEIG",
    # Ground truth
    "SensitivityGroundTruth",
    "OptimizationGroundTruth",
    "QuadratureGroundTruth",
    "MultifidelityGroundTruth",
    "InverseGroundTruth",
    "ODEGroundTruth",
    # Benchmark classes
    "BoxDomain",
    "Benchmark",
    "BenchmarkWithPrior",
    "ConstrainedBenchmark",
    # Registry
    "BenchmarkRegistry",
    # Functions - Algebraic
    "IshigamiFunction",
    "IshigamiSensitivityIndices",
    "RosenbrockFunction",
    "SobolGFunction",
    "SobolGSensitivityIndices",
    "BraninFunction",
    "BRANIN_GLOBAL_MINIMUM",
    "BRANIN_MINIMIZERS",
    "CantileverBeam1DAnalytical",
    "CantileverBeam2DAnalytical",
    "HomogeneousBeam1DAnalytical",
    # Functions - Genz
    "OscillatoryFunction",
    "ProductPeakFunction",
    "CornerPeakFunction",
    "GaussianPeakFunction",
    # Functions - Multifidelity
    "PolynomialModelFunction",
    # Functions - ODE
    "ODETimeConfig",
    "ODEQoIFunction",
    # ODE builders
    "build_lotka_volterra_3species",
    "build_coupled_springs_2mass",
    "build_hastings_ecology_3species",
    "build_chemical_reaction_surface",
    # Instances - Sensitivity
    "ishigami_3d",
    "sobol_g_6d",
    "sobol_g_4d",
    # Instances - Optimization
    "rosenbrock_2d",
    "rosenbrock_10d",
    "branin_2d",
    # Instances - Quadrature
    "genz_oscillatory_2d",
    "genz_product_peak_2d",
    "genz_corner_peak_2d",
    "genz_gaussian_peak_2d",
    "genz_oscillatory_5d",
    "genz_gaussian_peak_5d",
    # Instances - PDE
    "elastic_bar_1d",
    "cantilever_beam_1d",
    "cantilever_beam_2d_linear",
    "cantilever_beam_2d_neohookean",
    "cantilever_beam_1d_analytical",
    "cantilever_beam_2d_analytical",
]
