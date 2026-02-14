"""Multifidelity benchmark instances."""

from pyapprox.typing.benchmarks.instances.multifidelity.polynomial_ensemble import (
    polynomial_ensemble_5model,
    polynomial_ensemble_3model,
)
from pyapprox.typing.benchmarks.instances.multifidelity.multioutput_ensemble import (
    multioutput_ensemble_3x3,
    psd_multioutput_ensemble_3x3,
)
from pyapprox.typing.benchmarks.instances.multifidelity.tunable_ensemble import (
    tunable_ensemble_3model,
)

__all__ = [
    "polynomial_ensemble_5model",
    "polynomial_ensemble_3model",
    "multioutput_ensemble_3x3",
    "psd_multioutput_ensemble_3x3",
    "tunable_ensemble_3model",
]
