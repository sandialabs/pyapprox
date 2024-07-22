from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.util.hyperparameter._hyperparameter import HyperParameter


class Monomial():
    def __init__(self, nvars, degree, coefs, coef_bounds,
                 transform, name="MonomialCoefficients"):
        self._nvars = nvars
        self.degree = degree
        self.indices = compute_hyperbolic_indices(self.nvars(), self.degree)
        self.nterms = self.indices.shape[1]
        self._coef = HyperParameter(
            name, self.nterms, coefs, coef_bounds, transform, backend=self)
        self.hyp_list = self._HyperParameterList([self._coef])

    def nvars(self):
        return self._nvars

    def _univariate_monomial_basis_matrix(self, max_level, samples):
        assert samples.ndim == 1
        basis_matrix = samples[:, None]**self._la_arange(max_level+1)[None, :]
        return basis_matrix

    def _monomial_basis_matrix(self, indices, samples):
        num_vars, num_indices = indices.shape
        assert samples.shape[0] == num_vars
        num_samples = samples.shape[1]

        deriv_order = 0
        basis_matrix = self._la_empty(
            ((1+deriv_order*num_vars)*num_samples, num_indices))
        basis_vals_1d = [self._univariate_monomial_basis_matrix(
            indices[0, :].max(), samples[0, :])]
        basis_matrix[:num_samples, :] = basis_vals_1d[0][:, indices[0, :]]
        for dd in range(1, num_vars):
            basis_vals_1d.append(self._univariate_monomial_basis_matrix(
                indices[dd, :].max(), samples[dd, :]))
            basis_matrix[:num_samples, :] *= (
                basis_vals_1d[dd][:, indices[dd, :]])
        return basis_matrix

    def basis_matrix(self, samples):
        return self._monomial_basis_matrix(self.indices, samples)

    def __call__(self, samples):
        if self.degree == 0:
            vals = self._la_empty((samples.shape[1], 1))
            vals[:] = self._coef.get_values()
            return vals
        basis_mat = self.basis_matrix(samples)
        vals = basis_mat @ self._coef.get_values()
        return vals[:, None]

    def __repr__(self):
        return "{0}(name={1}, nvars={2}, degree={3}, nterms={4})".format(
            self.__class__.__name__, self._coef.name, self.nvars(),
            self.degree, self.nterms)
