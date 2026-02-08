"""Multi-Fidelity Network (MFNets) surrogates.

Provides DAG-based multi-fidelity surrogate models where each node owns
a local model and the network combines low-fidelity and high-fidelity
data through a directed acyclic graph.
"""

from pyapprox.typing.surrogates.mfnets.protocols import (
    LinearNodeModelProtocol,
    NodeModelProtocol,
    NodeModelWithParamJacobianProtocol,
)
from pyapprox.typing.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    MFNetNode,
    RootMFNetNode,
)
from pyapprox.typing.surrogates.mfnets.edges import MFNetEdge
from pyapprox.typing.surrogates.mfnets.network import MFNet
from pyapprox.typing.surrogates.mfnets.discrepancy import (
    MultiplicativeAdditiveDiscrepancy,
)
from pyapprox.typing.surrogates.mfnets.losses import (
    MFNetNegLogLikelihoodLoss,
)
from pyapprox.typing.surrogates.mfnets.registry import (
    create_node_model,
    list_node_models,
    register_node_model,
)
from pyapprox.typing.surrogates.mfnets.builders import (
    build_chain_mfnet,
    build_dag_mfnet,
)
from pyapprox.typing.surrogates.mfnets.helpers import (
    generate_synthetic_data,
    randomize_coefficients,
)
from pyapprox.typing.surrogates.mfnets.fitters.composite_fitter import (
    MFNetCompositeFitResult,
    MFNetCompositeFitter,
)

__all__ = [
    "LinearNodeModelProtocol",
    "LeafMFNetNode",
    "MFNet",
    "MFNetCompositeFitResult",
    "MFNetCompositeFitter",
    "MFNetEdge",
    "MFNetNegLogLikelihoodLoss",
    "MFNetNode",
    "MultiplicativeAdditiveDiscrepancy",
    "NodeModelProtocol",
    "NodeModelWithParamJacobianProtocol",
    "RootMFNetNode",
    "build_chain_mfnet",
    "build_dag_mfnet",
    "create_node_model",
    "generate_synthetic_data",
    "list_node_models",
    "randomize_coefficients",
    "register_node_model",
]
