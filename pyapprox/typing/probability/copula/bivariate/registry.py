"""
Registry for bivariate copula families.

Provides a factory pattern for creating bivariate copulas by name.
New copula families can be registered via `register_bivariate_copula`.
"""

from typing import Any, Callable, Dict, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.copula.bivariate.protocols import (
    BivariateCopulaProtocol,
)
from pyapprox.typing.probability.copula.bivariate.gaussian import (
    BivariateGaussianCopula,
)
from pyapprox.typing.probability.copula.bivariate.clayton import (
    ClaytonCopula,
)
from pyapprox.typing.probability.copula.bivariate.frank import (
    FrankCopula,
)
from pyapprox.typing.probability.copula.bivariate.gumbel import (
    GumbelCopula,
)


# Factory type: takes (bkd, **kwargs) and returns a BivariateCopulaProtocol
_CopulaFactory = Callable[..., Any]

_REGISTRY: Dict[str, _CopulaFactory] = {}


def register_bivariate_copula(name: str, factory: _CopulaFactory) -> None:
    """
    Register a bivariate copula factory.

    Parameters
    ----------
    name : str
        Name for the copula family (e.g., "gaussian", "clayton").
    factory : callable
        Factory function that takes (param, bkd) and returns a copula.
    """
    _REGISTRY[name] = factory


def create_bivariate_copula(
    name: str, bkd: Backend[Array], **kwargs: Any
) -> Any:
    """
    Create a bivariate copula by name.

    Parameters
    ----------
    name : str
        Name of the copula family.
    bkd : Backend[Array]
        Computational backend.
    **kwargs
        Additional keyword arguments passed to the copula constructor.

    Returns
    -------
    BivariateCopulaProtocol
        The constructed copula.

    Raises
    ------
    KeyError
        If the copula name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown bivariate copula: '{name}'. "
            f"Available: {available}"
        )
    return _REGISTRY[name](bkd=bkd, **kwargs)


def list_bivariate_copulas() -> List[str]:
    """Return sorted list of registered copula names."""
    return sorted(_REGISTRY.keys())


# Auto-register built-in families
def _gaussian_factory(
    bkd: Backend[Array], rho: float = 0.5, **kwargs: Any
) -> "BivariateGaussianCopula[Array]":
    return BivariateGaussianCopula(rho, bkd)


def _clayton_factory(
    bkd: Backend[Array], theta: float = 2.0, **kwargs: Any
) -> "ClaytonCopula[Array]":
    return ClaytonCopula(theta, bkd)


def _frank_factory(
    bkd: Backend[Array], theta: float = 5.0, **kwargs: Any
) -> "FrankCopula[Array]":
    return FrankCopula(theta, bkd)


def _gumbel_factory(
    bkd: Backend[Array], theta: float = 2.0, **kwargs: Any
) -> "GumbelCopula[Array]":
    return GumbelCopula(theta, bkd)


register_bivariate_copula("gaussian", _gaussian_factory)
register_bivariate_copula("clayton", _clayton_factory)
register_bivariate_copula("frank", _frank_factory)
register_bivariate_copula("gumbel", _gumbel_factory)
