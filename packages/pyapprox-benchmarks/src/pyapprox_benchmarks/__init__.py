"""Benchmark suite for the PyApprox module.

This module provides a collection of benchmark problems for testing and
validating algorithms in sensitivity analysis, optimization, quadrature,
multifidelity methods, and inverse problems.

Design Principles:
- Functions implement existing protocols from pyapprox.interface.functions.protocols
- No inheritance hierarchies - composition and protocol compliance
- HVP (Hessian-vector product) preferred over full Hessian for efficiency
- Fixed benchmark instances with known ground truth
- Backend-agnostic using Backend[Array] protocol
"""

from pyapprox_benchmarks.benchmark import BoxDomain

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

# ODE builders
from pyapprox_benchmarks.ode import (
    build_chemical_reaction_surface,
    build_coupled_springs_2mass,
    build_hastings_ecology_3species,
    build_lotka_volterra_3species,
)

# PDE builders
from pyapprox_benchmarks.pde import (
    build_cantilever_beam_1d,
    build_cantilever_beam_1d_analytical,
    build_cantilever_beam_1d_spde,
    build_cantilever_beam_2d_analytical,
    build_cantilever_beam_2d_linear,
    build_cantilever_beam_2d_linear_spde,
    build_cantilever_beam_2d_neohookean,
    build_cantilever_beam_2d_neohookean_spde,
    build_elastic_bar_1d,
    build_hyperelastic_pressurized_cylinder_2d,
    build_pressurized_cylinder_2d,
)

# Protocols
from pyapprox_benchmarks.protocols import (
    ConstraintProtocol,
    DomainProtocol,
    HasExactEIG,
)

# Sensitivity benchmarks
from pyapprox_benchmarks.sensitivity import (
    IshigamiBenchmark,
    SobolGBenchmark,
)

# Optimization benchmarks
from pyapprox_benchmarks.optimization import (
    BraninBenchmark,
    RosenbrockBenchmark,
)

# Quadrature benchmarks
from pyapprox_benchmarks.quadrature import (
    GenzCornerPeakBenchmark,
    GenzGaussianPeakBenchmark,
    GenzOscillatoryBenchmark,
    GenzProductPeakBenchmark,
)

__all__ = [
    # Protocols
    "DomainProtocol",
    "ConstraintProtocol",
    "HasExactEIG",
    # Domain
    "BoxDomain",
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
    # PDE builders
    "build_elastic_bar_1d",
    "build_cantilever_beam_1d",
    "build_cantilever_beam_1d_analytical",
    "build_cantilever_beam_1d_spde",
    "build_cantilever_beam_2d_analytical",
    "build_cantilever_beam_2d_linear",
    "build_cantilever_beam_2d_linear_spde",
    "build_cantilever_beam_2d_neohookean",
    "build_cantilever_beam_2d_neohookean_spde",
    "build_hyperelastic_pressurized_cylinder_2d",
    "build_pressurized_cylinder_2d",
    # Sensitivity benchmarks
    "IshigamiBenchmark",
    "SobolGBenchmark",
    # Optimization benchmarks
    "BraninBenchmark",
    "RosenbrockBenchmark",
    # Quadrature benchmarks
    "GenzOscillatoryBenchmark",
    "GenzProductPeakBenchmark",
    "GenzCornerPeakBenchmark",
    "GenzGaussianPeakBenchmark",
]
