degrees = [3]*var_trans.num_vars()
indices = tensor_product_indices(degrees)
#indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
poly.set_indices(indices)