basis_matrix = poly.basis_matrix(train_samples)
precond_weights = christoffel_weights(basis_matrix)
precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
precond_train_values = precond_weights[:,np.newaxis]*train_values
coef = np.linalg.lstsq(precond_basis_matrix,precond_train_values,rcond=None)[0]
poly.set_coefficients(coef)