"""Convenience builders for common MFNet topologies.

Provides functions to construct polynomial MFNet DAGs with minimal
boilerplate, handling node classification, model creation, and validation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from pyapprox.surrogates.mfnets.edges import MFNetEdge
from pyapprox.surrogates.mfnets.network import MFNet
from pyapprox.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    MFNetNode,
    RootMFNetNode,
)
from pyapprox.surrogates.mfnets.registry import create_node_model
from pyapprox.util.backends.protocols import Array, Backend


def build_chain_mfnet(
    nvars: int,
    nqoi: int,
    nnodes: int,
    bkd: Backend[Array],
    leaf_level: int = 3,
    scale_level: int = 1,
    delta_level: int = 3,
    noise_std: Union[float, List[float]] = 1e-2,
    fixed_noise_std: bool = True,
    leaf_model_name: str = "basis_expansion",
    interior_model_name: str = "multiplicative_additive",
    leaf_kwargs: Optional[Dict[str, Any]] = None,
    interior_kwargs: Optional[Dict[str, Any]] = None,
) -> MFNet[Array]:
    """Build a chain-topology MFNet: node_0 -> node_1 -> ... -> node_{N-1}.

    Node 0 is a leaf (lowest fidelity), node N-1 is the root (highest
    fidelity), and any intermediate nodes are interior.

    Parameters
    ----------
    nvars : int
        Number of global input variables.
    nqoi : int
        Number of quantities of interest per node.
    nnodes : int
        Number of nodes in the chain (must be >= 2).
    bkd : Backend[Array]
        Computational backend.
    leaf_level : int
        Polynomial level for the leaf node model. Default: 3.
    scale_level : int
        Polynomial level for scaling sub-models in non-leaf nodes. Default: 1.
    delta_level : int
        Polynomial level for delta sub-models in non-leaf nodes. Default: 3.
    noise_std : float or list of float
        Noise standard deviation. If a single float, used for all nodes.
        If a list, must have length ``nnodes``.
    fixed_noise_std : bool
        Whether to fix noise_std during optimization. Default: True.
    leaf_model_name : str
        Registry name for the leaf model. Default: ``"basis_expansion"``.
    interior_model_name : str
        Registry name for non-leaf models. Default:
        ``"multiplicative_additive"``.
    leaf_kwargs : dict, optional
        Extra keyword arguments for the leaf model factory.
    interior_kwargs : dict, optional
        Extra keyword arguments for interior/root model factories.

    Returns
    -------
    MFNet[Array]
        A validated chain MFNet.
    """
    if nnodes < 2:
        raise ValueError(f"nnodes must be >= 2, got {nnodes}")

    # Normalize noise_std to list
    if isinstance(noise_std, (int, float)):
        noise_std_list = [float(noise_std)] * nnodes
    else:
        if len(noise_std) != nnodes:
            raise ValueError(
                f"noise_std list length ({len(noise_std)}) != nnodes ({nnodes})"
            )
        noise_std_list = [float(s) for s in noise_std]

    lkw = leaf_kwargs or {}
    ikw = interior_kwargs or {}

    net = MFNet(nvars=nvars, bkd=bkd)

    # Node 0: leaf
    leaf_model = create_node_model(
        leaf_model_name,
        bkd,
        nvars=nvars,
        nqoi=nqoi,
        max_level=leaf_level,
        **lkw,
    )
    net.add_node(
        LeafMFNetNode(
            0,
            leaf_model,
            noise_std=noise_std_list[0],
            bkd=bkd,
            fixed_noise_std=fixed_noise_std,
        )
    )

    # Interior nodes (1 through nnodes-2)
    for i in range(1, nnodes - 1):
        model = create_node_model(
            interior_model_name,
            bkd,
            nvars_x=nvars,
            nqoi=nqoi,
            nscaled_qoi=nqoi,
            scale_level=scale_level,
            delta_level=delta_level,
            **ikw,
        )
        net.add_node(
            MFNetNode(
                i,
                model,
                noise_std=noise_std_list[i],
                bkd=bkd,
                fixed_noise_std=fixed_noise_std,
            )
        )

    # Node nnodes-1: root
    root_model = create_node_model(
        interior_model_name,
        bkd,
        nvars_x=nvars,
        nqoi=nqoi,
        nscaled_qoi=nqoi,
        scale_level=scale_level,
        delta_level=delta_level,
        **ikw,
    )
    net.add_node(
        RootMFNetNode(
            nnodes - 1,
            root_model,
            noise_std=noise_std_list[nnodes - 1],
            bkd=bkd,
            fixed_noise_std=fixed_noise_std,
        )
    )

    # Chain edges: 0->1, 1->2, ..., (N-2)->(N-1)
    for i in range(nnodes - 1):
        net.add_edge(MFNetEdge(i, i + 1, bkd=bkd))

    net.validate()
    return net


def build_dag_mfnet(
    nvars: int,
    nqoi: int,
    edges: List[Tuple[int, int]],
    bkd: Backend[Array],
    node_configs: Optional[Dict[int, Dict[str, Any]]] = None,
    default_leaf_level: int = 3,
    default_scale_level: int = 1,
    default_delta_level: int = 3,
    default_noise_std: float = 1e-2,
    default_fixed_noise_std: bool = True,
    default_leaf_model_name: str = "basis_expansion",
    default_interior_model_name: str = "multiplicative_additive",
) -> MFNet[Array]:
    """Build a general DAG-topology MFNet from a list of edges.

    Automatically determines which nodes are leaves (no incoming edges),
    roots (no outgoing edges), and interior (both). Creates appropriate
    node types and models.

    Parameters
    ----------
    nvars : int
        Number of global input variables.
    nqoi : int
        Number of quantities of interest per node.
    edges : list of (child_id, parent_id) tuples
        Directed edges from child (low-fidelity) to parent (high-fidelity).
    bkd : Backend[Array]
        Computational backend.
    node_configs : dict, optional
        Per-node configuration overrides. Keys are node IDs, values are dicts
        that may contain: ``"model_name"``, ``"leaf_level"``,
        ``"scale_level"``, ``"delta_level"``, ``"noise_std"``,
        ``"fixed_noise_std"``.
    default_leaf_level : int
        Default polynomial level for leaf node models.
    default_scale_level : int
        Default polynomial level for scaling sub-models.
    default_delta_level : int
        Default polynomial level for delta sub-models.
    default_noise_std : float
        Default noise standard deviation.
    default_fixed_noise_std : bool
        Default for whether to fix noise_std during optimization.
    default_leaf_model_name : str
        Default registry name for leaf models.
    default_interior_model_name : str
        Default registry name for non-leaf models.

    Returns
    -------
    MFNet[Array]
        A validated MFNet with the specified DAG topology.
    """
    if not edges:
        raise ValueError("edges list must be non-empty")

    node_configs = node_configs or {}

    # Collect all node IDs and compute in-degree / out-degree
    all_node_ids: set[int] = set()
    children_of: Dict[int, List[int]] = {}  # parent -> list of children
    parents_of: Dict[int, List[int]] = {}  # child -> list of parents
    for child_id, parent_id in edges:
        all_node_ids.add(child_id)
        all_node_ids.add(parent_id)
        children_of.setdefault(parent_id, []).append(child_id)
        parents_of.setdefault(child_id, []).append(parent_id)

    # Classify nodes
    leaf_ids = set()
    root_ids = set()
    interior_ids = set()
    for nid in all_node_ids:
        has_children = nid in children_of
        has_parents = nid in parents_of
        if not has_children and has_parents:
            leaf_ids.add(nid)
        elif has_children and not has_parents:
            root_ids.add(nid)
        elif has_children and has_parents:
            interior_ids.add(nid)
        else:
            # Isolated node (no edges) — shouldn't happen with valid edges
            leaf_ids.add(nid)

    net = MFNet(nvars=nvars, bkd=bkd)

    # Create nodes in sorted order for determinism
    for nid in sorted(all_node_ids):
        cfg = node_configs.get(nid, {})
        noise = cfg.get("noise_std", default_noise_std)
        fixed = cfg.get("fixed_noise_std", default_fixed_noise_std)

        if nid in leaf_ids:
            level = cfg.get("leaf_level", default_leaf_level)
            model_name = cfg.get("model_name", default_leaf_model_name)
            model = create_node_model(
                model_name,
                bkd,
                nvars=nvars,
                nqoi=nqoi,
                max_level=level,
            )
            net.add_node(
                LeafMFNetNode(
                    nid,
                    model,
                    noise_std=noise,
                    bkd=bkd,
                    fixed_noise_std=fixed,
                )
            )
        else:
            # Interior or root: discrepancy model
            # Compute nscaled_qoi from children
            child_list = children_of.get(nid, [])
            nscaled_qoi = nqoi * len(child_list)

            s_level = cfg.get("scale_level", default_scale_level)
            d_level = cfg.get("delta_level", default_delta_level)
            model_name = cfg.get("model_name", default_interior_model_name)
            model = create_node_model(
                model_name,
                bkd,
                nvars_x=nvars,
                nqoi=nqoi,
                nscaled_qoi=nscaled_qoi,
                scale_level=s_level,
                delta_level=d_level,
            )

            if nid in root_ids:
                net.add_node(
                    RootMFNetNode(
                        nid,
                        model,
                        noise_std=noise,
                        bkd=bkd,
                        fixed_noise_std=fixed,
                    )
                )
            else:
                net.add_node(
                    MFNetNode(
                        nid,
                        model,
                        noise_std=noise,
                        bkd=bkd,
                        fixed_noise_std=fixed,
                    )
                )

    # Add edges
    for child_id, parent_id in edges:
        net.add_edge(MFNetEdge(child_id, parent_id, bkd=bkd))

    net.validate()
    return net
