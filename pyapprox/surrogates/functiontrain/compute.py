"""Optimized FunctionTrain evaluation using cached basis matrices.

Eliminates redundant basis evaluations by computing basis matrices once
per unique Basis object and reusing them across all positions within cores.

Caching uses id(basis_object) as key, which is safe because:
- Basis objects are owned by BasisExpansion objects inside FunctionTrain
- FunctionTrain is held alive throughout the cache lifetime
- No object can be GC'd and have its id reused within scope
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.util.backends.protocols import Array, Backend

BasisCache = Dict[int, Array]


def cache_basis_matrices(
    cores: List[FunctionTrainCore[Array]],
    samples: Array,
    bkd: Backend[Array],
) -> BasisCache:
    """Build a deduplicated cache of basis matrices for all cores.

    Iterates all expansion positions across all cores and evaluates each
    unique Basis object exactly once. Expansions without a get_basis method
    (e.g., ConstantExpansion) are skipped since they don't benefit from
    caching.

    Parameters
    ----------
    cores : List[FunctionTrainCore]
        FT cores.
    samples : Array
        Input samples. Shape: (nvars, nsamples)
    bkd : Backend

    Returns
    -------
    BasisCache
        Dict mapping id(basis) to basis_matrix of shape (nsamples, nterms).
    """
    cache: BasisCache = {}
    for kk, core in enumerate(cores):
        sample_1d = samples[kk : kk + 1]
        r_left, r_right = core.ranks()
        for ii in range(r_left):
            for jj in range(r_right):
                bexp = core.get_basisexp(ii, jj)
                if not hasattr(bexp, "get_basis"):
                    continue
                basis = bexp.get_basis()
                key = id(basis)
                if key not in cache:
                    cache[key] = basis(sample_1d)
    return cache


def core_eval_cached(
    core: FunctionTrainCore[Array],
    sample_1d: Array,
    cache: BasisCache,
    bkd: Backend[Array],
) -> Array:
    """Evaluate core tensor using cached basis matrices.

    Groups expansions by basis identity for batched matmul. Falls back to
    direct evaluation for ConstantExpansion or other types without get_basis.

    Parameters
    ----------
    core : FunctionTrainCore
    sample_1d : Array
        Shape: (1, nsamples)
    cache : BasisCache
        Pre-computed basis matrices from cache_basis_matrices.
    bkd : Backend

    Returns
    -------
    Array
        Shape: (r_left, r_right, nsamples, nqoi)
    """
    r_left, r_right = core.ranks()
    nsamples = sample_1d.shape[1]
    nqoi = core.nqoi()

    result = bkd.zeros((r_left, r_right, nsamples, nqoi))

    # Group expansions by basis identity for batched matmul
    groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    constants: List[Tuple[int, int]] = []

    for ii in range(r_left):
        for jj in range(r_right):
            bexp = core.get_basisexp(ii, jj)
            if hasattr(bexp, "get_basis"):
                key = id(bexp.get_basis())
                groups[key].append((ii, jj))
            else:
                constants.append((ii, jj))

    # Batch matmul per group of expansions sharing the same basis
    for basis_id, members in groups.items():
        basis_matrix = cache[basis_id]  # (nsamples, nterms)
        nterms = basis_matrix.shape[1]

        # Stack coefficients: each is (nterms, nqoi)
        coefs_list = [
            core.get_basisexp(ii, jj).get_coefficients() for ii, jj in members
        ]
        # (group_size, nterms, nqoi)
        coefs_stacked = bkd.stack(coefs_list, axis=0)
        group_size = len(members)

        # Reshape to (nterms, group_size * nqoi) for single matmul
        coefs_flat = bkd.reshape(
            bkd.transpose(coefs_stacked, (1, 0, 2)),
            (nterms, group_size * nqoi),
        )

        # (nsamples, nterms) @ (nterms, group_size*nqoi) -> (nsamples, group*nqoi)
        vals_flat = basis_matrix @ coefs_flat

        # Scatter results into (r_left, r_right, nsamples, nqoi)
        for idx, (ii, jj) in enumerate(members):
            result[ii, jj] = vals_flat[:, idx * nqoi : (idx + 1) * nqoi]

    # Handle constant expansions (no basis to cache)
    for ii, jj in constants:
        bexp = core.get_basisexp(ii, jj)
        result[ii, jj] = bexp(sample_1d).T

    return result


def ft_eval_cached(
    cores: List[FunctionTrainCore[Array]],
    samples: Array,
    cache: BasisCache,
    bkd: Backend[Array],
) -> Array:
    """Evaluate FunctionTrain using cached basis matrices.

    Parameters
    ----------
    cores : List[FunctionTrainCore]
    samples : Array
        Shape: (nvars, nsamples)
    cache : BasisCache
    bkd : Backend

    Returns
    -------
    Array
        Shape: (nqoi, nsamples)
    """
    nvars = len(cores)

    # Evaluate first core
    values = core_eval_cached(cores[0], samples[:1], cache, bkd)

    # Contract with remaining cores
    for kk in range(1, nvars):
        core_val = core_eval_cached(cores[kk], samples[kk : kk + 1], cache, bkd)
        values = bkd.einsum("ijkl, jmkl->imkl", values, core_val)

    return values[0, 0].T


def _forward_backward_sweep(
    cores: List[FunctionTrainCore[Array]],
    samples: Array,
    cache: BasisCache,
    bkd: Backend[Array],
) -> Tuple[List[Array], List[Array]]:
    """Compute forward/backward sweep products using cached basis matrices.

    Parameters
    ----------
    cores : List[FunctionTrainCore]
    samples : Array
        Shape: (nvars, nsamples)
    cache : BasisCache
    bkd : Backend

    Returns
    -------
    left_products : List[Array]
        left_products[k] is the product of cores [0..k].
        Shape: (1, r_{k+1}, nsamples, nqoi)
        Length: nvars - 1
    right_products : List[Array]
        right_products[k] is the product of cores [k+1..nvars-1].
        Shape: (r_{k+1}, 1, nsamples, nqoi)
        Length: nvars (indices 0..nvars-2 are valid, nvars-1 is placeholder)
    """
    nvars = len(cores)

    # Forward sweep: L_k = F_0 * ... * F_k
    left_products: List[Array] = []
    left_products.append(core_eval_cached(cores[0], samples[:1], cache, bkd))
    for kk in range(1, nvars - 1):
        core_val = core_eval_cached(cores[kk], samples[kk : kk + 1], cache, bkd)
        left_products.append(
            bkd.einsum("ijkl, jmkl->imkl", left_products[-1], core_val)
        )

    # Backward sweep: R_k = F_{k+1} * ... * F_{nvars-1}
    right_products: List[Array] = [bkd.zeros((1,))] * nvars
    right_products[nvars - 2] = core_eval_cached(
        cores[nvars - 1], samples[nvars - 1 : nvars], cache, bkd
    )
    for kk in range(nvars - 3, -1, -1):
        core_val = core_eval_cached(cores[kk + 1], samples[kk + 1 : kk + 2], cache, bkd)
        right_products[kk] = bkd.einsum(
            "ijkl, jmkl->imkl", core_val, right_products[kk + 1]
        )

    return left_products, right_products


def _core_jacobian_direct(
    core: FunctionTrainCore[Array],
    sample_1d: Array,
    cache: BasisCache,
    L_weight: Array,
    R_weight: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute weighted core Jacobian via direct scatter.

    Instead of building a sparse 5D tensor (r_left, r_right, nsamples,
    nqoi, nparams) that is ~94% zeros and then contracting with weights,
    directly scatters weight * basis_matrix into the result.

    Parameters
    ----------
    core : FunctionTrainCore
    sample_1d : Array
        Shape: (1, nsamples)
    cache : BasisCache
    L_weight : Array
        Left product weight. Shape: (r_left, nsamples, nqoi)
    R_weight : Array
        Right product weight. Shape: (r_right, nsamples, nqoi)
    bkd : Backend

    Returns
    -------
    Array
        Weighted core Jacobian. Shape: (nsamples, nqoi, core_nparams)
    """
    r_left, r_right = core.ranks()
    nsamples = sample_1d.shape[1]
    nqoi = core.nqoi()
    core_nparams = core.nparams()

    if core_nparams == 0:
        return bkd.zeros((nsamples, nqoi, 0))

    result = bkd.zeros((nsamples, nqoi, core_nparams))

    # Track parameter offset for each (ii, jj) position
    param_idx = 0
    for ii in range(r_left):
        for jj in range(r_right):
            bexp = core.get_basisexp(ii, jj)
            bexp_nparams = bexp.nparams()

            if bexp_nparams == 0:
                continue

            # Combined weight: (nsamples, nqoi)
            weight = L_weight[ii, :, :] * R_weight[jj, :, :]

            # Get basis matrix from cache
            if hasattr(bexp, "get_basis"):
                basis_matrix = cache[id(bexp.get_basis())]
            else:
                basis_matrix = bexp.basis_matrix(sample_1d)

            nterms = bexp.nterms()

            if nqoi == 1:
                # Fast path: weight is (nsamples, 1), basis is (nsamples, nterms)
                # Result slice: (nsamples, 1, nterms)
                result[:, 0, param_idx : param_idx + nterms] = weight * basis_matrix
            else:
                # Multi-QoI: params are (nterms, nqoi).flatten() row-major
                # c_{i,q} at param index i*nqoi + q
                # d(f_q)/d(c_{i,q'}) = phi_i(x) * weight if q==q' else 0
                for qq in range(nqoi):
                    for ll in range(nterms):
                        p = param_idx + ll * nqoi + qq
                        result[:, qq, p] = weight[:, qq] * basis_matrix[:, ll]

            param_idx += bexp_nparams

    return result


def ft_jacobian_wrt_params_cached(
    cores: List[FunctionTrainCore[Array]],
    samples: Array,
    cache: BasisCache,
    bkd: Backend[Array],
) -> Array:
    """Compute FT Jacobian w.r.t. params using cached basis matrices.

    Uses forward-backward sweep with cached core evaluations and direct
    scatter for core Jacobians. No redundant basis evaluations.

    Parameters
    ----------
    cores : List[FunctionTrainCore]
    samples : Array
        Shape: (nvars, nsamples)
    cache : BasisCache
    bkd : Backend

    Returns
    -------
    Array
        Shape: (nsamples, nqoi, nparams)
    """
    nvars = len(cores)
    nsamples = samples.shape[1]
    nqoi = cores[0].nqoi()
    nparams = sum(c.nparams() for c in cores)

    if nparams == 0:
        return bkd.zeros((nsamples, nqoi, 0))

    # Edge case: single variable
    if nvars == 1:
        core = cores[0]
        if not hasattr(core.get_basisexp(0, 0), "get_basis"):
            # All constant -- no params
            return bkd.zeros((nsamples, nqoi, 0))
        L_weight = bkd.ones((1, nsamples, nqoi))
        R_weight = bkd.ones((1, nsamples, nqoi))
        return _core_jacobian_direct(core, samples[0:1], cache, L_weight, R_weight, bkd)

    # Forward-backward sweep
    left_products, right_products = _forward_backward_sweep(cores, samples, cache, bkd)

    # Assemble Jacobian: contribution from each core
    jac_parts = []

    for kk in range(nvars):
        core = cores[kk]
        core_nparams = core.nparams()

        if core_nparams == 0:
            continue

        r_left, r_right = core.ranks()

        # Left weight
        if kk == 0:
            L_weight = bkd.ones((r_left, nsamples, nqoi))
        else:
            L_weight = left_products[kk - 1][0, :, :, :]

        # Right weight
        if kk == nvars - 1:
            R_weight = bkd.ones((r_right, nsamples, nqoi))
        else:
            R_weight = right_products[kk][:, 0, :, :]

        core_contribution = _core_jacobian_direct(
            core, samples[kk : kk + 1], cache, L_weight, R_weight, bkd
        )
        jac_parts.append(core_contribution)

    return bkd.concatenate(jac_parts, axis=2)
