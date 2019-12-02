num_vars = 2; level = 3
quad_rule = gaussian_leja_quadrature_rule
growth_rule = leja_growth_rule
samples, weights, data_structures = get_sparse_grid_samples_and_weights(
num_vars,level,quad_rule,growth_rule)
axs = plot_sparse_grid_2d(samples,weights,poly_indices=data_structures[1],
   subspace_indices=data_structures[2])