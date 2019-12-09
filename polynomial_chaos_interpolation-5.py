#univariate_quadrature_rules = [
#    partial(gauss_jacobi_pts_wts_1D,alpha_poly=0,beta_poly=0),
#    partial(gauss_jacobi_pts_wts_1D,alpha_poly=2,beta_poly=2)]
level=1
univariate_quadrature_rules = [
    partial(clenshaw_curtis_in_polynomial_order,
    return_weights_for_all_levels=False)]*poly.num_vars()
train_samples, train_weights = get_tensor_product_quadrature_rule(
    level,var_trans.num_vars(),univariate_quadrature_rules,
    var_trans.map_from_canonical_space)

train_values = model(train_samples)

basis_matrix = poly.basis_matrix(train_samples)
coef = np.linalg.lstsq(basis_matrix,train_values,rcond=None)[0]
poly.set_coefficients(coef)