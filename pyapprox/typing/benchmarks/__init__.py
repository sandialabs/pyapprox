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
    RosenbrockFunction,
)

# Instances (also registers with BenchmarkRegistry)
from pyapprox.typing.benchmarks.instances import (
    ishigami_3d,
    rosenbrock_2d,
    rosenbrock_10d,
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
    # Benchmark classes
    "BoxDomain",
    "Benchmark",
    "BenchmarkWithPrior",
    "ConstrainedBenchmark",
    # Registry
    "BenchmarkRegistry",
    # Functions
    "IshigamiFunction",
    "RosenbrockFunction",
    # Instances
    "ishigami_3d",
    "rosenbrock_2d",
    "rosenbrock_10d",
]
