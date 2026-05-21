"""ACV utility functions (leaf module — no ACV-internal imports)."""

from typing import List

from pyapprox.util.backends.protocols import Array, Backend


def _combine_acv_values(
    reorder_allocation_mat: Array,
    npartition_samples: Array,
    acv_values: List[List[Array]],
    bkd: Backend[Array],
) -> List[Array]:
    r"""
    Extract the unique values from the sets
    :math:`f_\alpha(\mathcal{Z}_\alpha), f_\alpha(\mathcal{Z}_\alpha^*)`
    for each model :math:`\alpha=0,\ldots,M`
    """
    nmodels = len(acv_values)
    values_per_model: List[Array] = [acv_values[0][1]]
    for ii in range(1, nmodels):
        lb, ub = 0, 0
        lb2, ub2 = 0, 0
        parts: List[Array] = []
        for jj in range(nmodels):
            found = False
            if bkd.to_int(reorder_allocation_mat[jj, 2 * ii]) == 1:
                ub = lb + bkd.to_int(npartition_samples[jj])
                parts.append(acv_values[ii][0][lb:ub])
                lb = ub
                found = True
            if bkd.to_int(reorder_allocation_mat[jj, 2 * ii + 1]) == 1:
                # there is no need to enter here if sample set has already
                # been added by acv_values[ii][0], hence the use of elseif here
                ub2 = lb2 + bkd.to_int(npartition_samples[jj])
                if not found:
                    parts.append(acv_values[ii][1][lb2:ub2])
                lb2 = ub2
        values_per_model.append(bkd.vstack(parts))
    return values_per_model


def _combine_acv_samples(
    reorder_allocation_mat: Array,
    npartition_samples: Array,
    acv_samples: List[List[Array]],
    bkd: Backend[Array],
) -> List[Array]:
    r"""
    Extract the unique samples from the sets
    :math:`\mathcal{Z}_\alpha, \mathcal{Z}_\alpha^*` for each model
    :math:`\alpha=0,\ldots,M`
    """
    nmodels = len(acv_samples)
    samples_per_model: List[Array] = [acv_samples[0][1]]
    for ii in range(1, nmodels):
        lb, ub = 0, 0
        lb2, ub2 = 0, 0
        parts: List[Array] = []
        for jj in range(nmodels):
            found = False
            if bkd.to_int(reorder_allocation_mat[jj, 2 * ii]) == 1:
                ub = lb + bkd.to_int(npartition_samples[jj])
                parts.append(acv_samples[ii][0][:, lb:ub])
                lb = ub
                found = True
            if bkd.to_int(reorder_allocation_mat[jj, 2 * ii + 1]) == 1:
                ub2 = lb2 + bkd.to_int(npartition_samples[jj])
                if not found:
                    parts.append(acv_samples[ii][1][:, lb2:ub2])
                lb2 = ub2
        samples_per_model.append(bkd.hstack(parts))
    return samples_per_model
