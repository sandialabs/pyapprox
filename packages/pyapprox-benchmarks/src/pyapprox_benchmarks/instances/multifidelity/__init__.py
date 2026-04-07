"""Multifidelity benchmark instances."""

from pyapprox_benchmarks.instances.multifidelity.branin_ensemble import (
    branin_ensemble_3model,
)
from pyapprox_benchmarks.instances.multifidelity.forrester_ensemble import (
    forrester_ensemble_2model,
)
from pyapprox_benchmarks.instances.multifidelity.multioutput_ensemble import (
    multioutput_ensemble_3x3,
    psd_multioutput_ensemble_3x3,
)
from pyapprox_benchmarks.instances.multifidelity.polynomial_ensemble import (
    polynomial_ensemble_3model,
    polynomial_ensemble_5model,
)
from pyapprox_benchmarks.instances.multifidelity.tunable_ensemble import (
    tunable_ensemble_3model,
)

__all__ = [
    "branin_ensemble_3model",
    "forrester_ensemble_2model",
    "polynomial_ensemble_5model",
    "polynomial_ensemble_3model",
    "multioutput_ensemble_3x3",
    "psd_multioutput_ensemble_3x3",
    "tunable_ensemble_3model",
]
