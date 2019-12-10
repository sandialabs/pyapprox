from pyapprox.sparse_grid import plot_sparse_grid_2d
plot_sparse_grid_2d(
    pce.samples,np.ones(pce.samples.shape[1]),
    pce.pce.indices, pce.subspace_indices)

plt.figure()
plt.loglog(num_samples,errors,'o-')
plt.show()