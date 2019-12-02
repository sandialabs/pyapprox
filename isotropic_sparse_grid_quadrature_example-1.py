from pyapprox.sparse_grid import *
from pyapprox.visualization import plot_3d_indices
from pyapprox.univariate_quadrature import *
num_vars = 2; level = 3
quad_rule = clenshaw_curtis_in_polynomial_order
growth_rule = clenshaw_curtis_rule_growth
samples, weights, data_structures = get_sparse_grid_samples_and_weights(
num_vars,level,quad_rule,growth_rule)
plot_sparse_grid_2d(samples,weights)
plt.xlabel(r'$z_1$')
plt.ylabel(r'$z_2$')
plt.show()