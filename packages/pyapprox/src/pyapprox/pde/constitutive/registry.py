"""Registry for hyperelastic stress models.

Provides registration and factory functions for stress model plugins.
New stress models can be added without modifying existing code.
"""

from typing import Any, Callable, Dict, List

_STRESS_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_stress_model(name: str, factory: Callable[..., Any]) -> None:
    """Register a stress model factory.

    Parameters
    ----------
    name : str
        Unique name for the stress model (e.g., "neo_hookean").
    factory : Callable[..., Any]
        Factory function that creates the stress model instance.
        Should accept keyword arguments for material parameters.

    Raises
    ------
    ValueError
        If name is already registered.
    """
    if name in _STRESS_MODEL_REGISTRY:
        raise ValueError(f"Stress model '{name}' is already registered.")
    _STRESS_MODEL_REGISTRY[name] = factory


def create_stress_model(name: str, **kwargs: Any) -> Any:
    """Create a stress model instance by name.

    Parameters
    ----------
    name : str
        Registered stress model name.
    **kwargs
        Material parameters passed to the factory.

    Returns
    -------
    StressModelProtocol
        Stress model instance.

    Raises
    ------
    KeyError
        If name is not registered.
    """
    if name not in _STRESS_MODEL_REGISTRY:
        available = ", ".join(sorted(_STRESS_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown stress model: '{name}'. Available: {available}")
    return _STRESS_MODEL_REGISTRY[name](**kwargs)


def list_stress_models() -> List[str]:
    """Return sorted list of registered stress model names."""
    return sorted(_STRESS_MODEL_REGISTRY.keys())
