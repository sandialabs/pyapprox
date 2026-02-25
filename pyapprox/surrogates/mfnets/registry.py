"""Node model factory registry for MFNets.

Provides a registry pattern so that new node model types can be added
without modifying existing code.
"""

from typing import Any, Callable, Dict, List

from pyapprox.util.backends.protocols import Array, Backend

_NODE_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_node_model(name: str, factory: Callable[..., Any]) -> None:
    """Register a node model factory.

    Parameters
    ----------
    name : str
        Name for the model type.
    factory : callable
        Factory function ``(bkd, **kwargs) -> NodeModelProtocol``.
    """
    _NODE_MODEL_REGISTRY[name] = factory


def create_node_model(
    name: str, bkd: Backend[Array], **kwargs: Any
) -> Any:
    """Create a node model by registered name.

    Parameters
    ----------
    name : str
        Registered model name.
    bkd : Backend[Array]
        Computational backend.
    **kwargs
        Additional arguments passed to the factory.

    Returns
    -------
    NodeModelProtocol
        The created model.

    Raises
    ------
    KeyError
        If name is not registered.
    """
    if name not in _NODE_MODEL_REGISTRY:
        available = ", ".join(sorted(_NODE_MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown node model: '{name}'. Available: {available}"
        )
    return _NODE_MODEL_REGISTRY[name](bkd=bkd, **kwargs)


def list_node_models() -> List[str]:
    """Return sorted list of registered model names."""
    return sorted(_NODE_MODEL_REGISTRY.keys())


# --- Built-in factories ---

def _basis_expansion_factory(
    bkd: Backend[Array],
    nvars: int = 1,
    nqoi: int = 1,
    max_level: int = 2,
    **kwargs: Any,
) -> Any:
    """Create a BasisExpansion with monomial basis."""
    from pyapprox.surrogates.affine.univariate import MonomialBasis1D
    from pyapprox.surrogates.affine.indices import (
        compute_hyperbolic_indices,
    )
    from pyapprox.surrogates.affine.basis import MultiIndexBasis
    from pyapprox.surrogates.affine.expansions import BasisExpansion

    bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = MultiIndexBasis.__new__(MultiIndexBasis)
    MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=nqoi)


def _multiplicative_additive_factory(
    bkd: Backend[Array],
    nvars_x: int = 1,
    nqoi: int = 1,
    nscaled_qoi: int = 1,
    scale_level: int = 1,
    delta_level: int = 2,
    **kwargs: Any,
) -> Any:
    """Create a MultiplicativeAdditiveDiscrepancy model."""
    from pyapprox.surrogates.mfnets.discrepancy import (
        MultiplicativeAdditiveDiscrepancy,
    )

    scalings = [
        _basis_expansion_factory(
            bkd, nvars=nvars_x, nqoi=nscaled_qoi, max_level=scale_level
        )
        for _ in range(nqoi)
    ]
    delta = _basis_expansion_factory(
        bkd, nvars=nvars_x, nqoi=nqoi, max_level=delta_level
    )
    return MultiplicativeAdditiveDiscrepancy(
        scalings, delta, nscaled_qoi, bkd
    )


# Auto-register built-ins
register_node_model("basis_expansion", _basis_expansion_factory)
register_node_model(
    "multiplicative_additive", _multiplicative_additive_factory
)
