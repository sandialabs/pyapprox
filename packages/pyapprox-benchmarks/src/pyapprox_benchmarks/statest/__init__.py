"""Multifidelity variance reduction benchmarks and problems."""

from pyapprox_benchmarks.statest.branin_ensemble import (
    BraninEnsembleProblem,
)
from pyapprox_benchmarks.statest.forrester_ensemble import (
    ForresterEnsembleProblem,
)
from pyapprox_benchmarks.statest.multioutput_ensemble import (
    MultiOutputEnsembleBenchmark,
)
from pyapprox_benchmarks.statest.polynomial_ensemble import (
    PolynomialEnsembleBenchmark,
)
from pyapprox_benchmarks.statest.tunable_ensemble import (
    TunableEnsembleBenchmark,
)

__all__ = [
    "BraninEnsembleProblem",
    "ForresterEnsembleProblem",
    "MultiOutputEnsembleBenchmark",
    "PolynomialEnsembleBenchmark",
    "TunableEnsembleBenchmark",
]
