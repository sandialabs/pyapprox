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

from pyapprox.typing.benchmarks.protocols import (
    DomainProtocol,
    GroundTruthProtocol,
    BenchmarkProtocol,
    BenchmarkWithPriorProtocol,
    ConstraintProtocol,
    ConstrainedBenchmarkProtocol,
)
from pyapprox.typing.benchmarks.ground_truth import (
    SensitivityGroundTruth,
    OptimizationGroundTruth,
    QuadratureGroundTruth,
    MultifidelityGroundTruth,
    InverseGroundTruth,
    ODEGroundTruth,
)
from pyapprox.typing.benchmarks.benchmark import (
    BoxDomain,
    Benchmark,
    BenchmarkWithPrior,
    ConstrainedBenchmark,
)
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry

# Functions
from pyapprox.typing.benchmarks.functions.algebraic import (
    IshigamiFunction,
    IshigamiSensitivityIndices,
    RosenbrockFunction,
    SobolGFunction,
    SobolGSensitivityIndices,
    BraninFunction,
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
    CantileverBeam1DAnalytical,
    CantileverBeam2DAnalytical,
    HomogeneousBeam1DAnalytical,
)
from pyapprox.typing.benchmarks.functions.genz import (
    OscillatoryFunction,
    ProductPeakFunction,
    CornerPeakFunction,
    GaussianPeakFunction,
)
from pyapprox.typing.benchmarks.functions.multifidelity import (
    PolynomialModelFunction,
    PolynomialEnsemble,
)

# ODE Benchmarks
from pyapprox.typing.benchmarks.functions.ode import (
    ODEBenchmark,
    ODETimeConfig,
    ODEQoIFunction,
)

# Instances (also registers with BenchmarkRegistry)
from pyapprox.typing.benchmarks.instances import (
    ishigami_3d,
    sobol_g_6d,
    sobol_g_4d,
    rosenbrock_2d,
    rosenbrock_10d,
    branin_2d,
    genz_oscillatory_2d,
    genz_product_peak_2d,
    genz_corner_peak_2d,
    genz_gaussian_peak_2d,
    genz_oscillatory_5d,
    genz_gaussian_peak_5d,
    polynomial_ensemble_5model,
    polynomial_ensemble_3model,
    multioutput_ensemble_3x3,
    psd_multioutput_ensemble_3x3,
    tunable_ensemble_3model,
    lotka_volterra_3species,
    coupled_springs_2mass,
    hastings_ecology_3species,
    chemical_reaction_surface,
    elastic_bar_1d,
    cantilever_beam_1d,
    cantilever_beam_2d_linear,
    cantilever_beam_2d_neohookean,
    cantilever_beam_1d_analytical,
    cantilever_beam_2d_analytical,
)

__all__ = [
    # Protocols
    "DomainProtocol",
    "GroundTruthProtocol",
    "BenchmarkProtocol",
    "BenchmarkWithPriorProtocol",
    "ConstraintProtocol",
    "ConstrainedBenchmarkProtocol",
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
    "PolynomialEnsemble",
    # Functions - ODE
    "ODEBenchmark",
    "ODETimeConfig",
    "ODEQoIFunction",
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
    # Instances - Multifidelity
    "polynomial_ensemble_5model",
    "polynomial_ensemble_3model",
    "multioutput_ensemble_3x3",
    "psd_multioutput_ensemble_3x3",
    "tunable_ensemble_3model",
    # Instances - ODE
    "lotka_volterra_3species",
    "coupled_springs_2mass",
    "hastings_ecology_3species",
    "chemical_reaction_surface",
    # Instances - PDE
    "elastic_bar_1d",
    "cantilever_beam_1d",
    "cantilever_beam_2d_linear",
    "cantilever_beam_2d_neohookean",
    "cantilever_beam_1d_analytical",
    "cantilever_beam_2d_analytical",
]
