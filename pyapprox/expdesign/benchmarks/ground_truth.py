"""Ground truth for OED benchmarks.

Provides a frozen dataclass to hold exact utility/EIG callables
that benchmarks compose and delegate to.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class OEDGroundTruth:
    """Ground truth for OED benchmarks.

    Benchmarks compose this and delegate exact_eig/exact_utility calls to it.

    Parameters
    ----------
    exact_eig : callable or None
        Callable mapping weights -> exact EIG value.
    exact_utility : callable or None
        Callable mapping weights -> exact utility value.
    """

    exact_eig: Optional[Callable[..., float]] = None
    exact_utility: Optional[Callable[..., float]] = None
