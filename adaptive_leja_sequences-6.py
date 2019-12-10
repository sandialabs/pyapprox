while (not pce.active_subspace_queue.empty() or
       pce.subspace_indices.shape[1]==0):
    pce.refine()
    pce.recompute_active_subspace_priorities()
    if callback is not None:
        callback(pce)