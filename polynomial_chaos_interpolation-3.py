var_trans = AffineRandomVariableTransformation(variable)
poly = PolynomialChaosExpansion()
poly_opts = define_poly_options_from_variable_transformation(var_trans)
poly.configure(poly_opts)