import unittest
import numpy as np

from pyapprox.variables.gaussian import (
    compute_gaussian_pdf_canonical_form_normalization,
    GaussianFactor, condition_gaussian_on_data,
    convert_gaussian_to_canonical_form, condition_gaussian_in_canonical_form,
    convert_gaussian_from_canonical_form,
    marginalize_gaussian_in_canonical_form, get_matrix_partition_indices,
    joint_density_from_linear_conditional_relationship,
    marginal_density_from_linear_conditional_relationship,
    conditional_density_from_linear_conditional_relationship,
    convert_conditional_probability_density_to_canonical_form,
    multiply_gaussian_densities_in_compact_canonical_form,
    multiply_gaussian_densities, expand_scope_of_gaussian
)


class TestGaussian(unittest.TestCase):

    def test_eval_pdf_of_gaussian_in_canonical_form(self):
        num_vars = 2
        covariance = np.array([[1, 0.5], [0.5, 1]])
        precision = np.linalg.inv(covariance)
        mean = np.ones(num_vars)

        shift = precision.dot(mean)
        normalization = compute_gaussian_pdf_canonical_form_normalization(
            mean, shift, precision)
        gaussian_pdf_cannoical_form = GaussianFactor(
            precision, shift, normalization, [0], [num_vars])

        num_samples = 5
        samples = mean[:, np.newaxis] + np.linalg.cholesky(covariance).dot(
            np.random.normal(0, 1, (num_vars, num_samples)))
        pdf_vals = gaussian_pdf_cannoical_form(samples)

        from scipy.stats import multivariate_normal
        true_pdf_vals = multivariate_normal.pdf(samples.T, mean, covariance)
        assert np.allclose(true_pdf_vals, pdf_vals)

    def test_condition_gaussian(self):
        covariance = np.array([[1, 0.5], [0.5, 1]])
        mean = np.array([-1, 1])

        remain_indices, fixed_indices = np.array([0]), np.array([1])
        values = [1.]
        cond_mean, cond_covar = condition_gaussian_on_data(
            mean, covariance, fixed_indices, values)
        mu0, mu1 = mean
        sigma = np.sqrt(np.diag(covariance))
        correlation = covariance[1, 0]/(sigma[0]*sigma[1])
        true_cond_mean = mean[remain_indices] + \
            sigma[remain_indices]/sigma[fixed_indices]*correlation*(
                values-mean[fixed_indices])
        true_cond_covar = (1-correlation**2)*sigma[remain_indices]**2
        assert np.allclose(true_cond_mean, cond_mean)
        assert np.allclose(true_cond_covar, cond_covar)

    def test_condition_gaussian_in_canonical_form(self):
        covariance = np.array([[1, 0.5], [0.5, 1]])
        mean = np.array([-1, 1])

        remain_indices, fixed_indices = np.array([0]), np.array([1])
        values = np.array([1.])

        precision_matrix, shift, normalization = \
            convert_gaussian_to_canonical_form(mean, covariance)
        cond_precision, cond_shift, cond_normalization = \
            condition_gaussian_in_canonical_form(
                fixed_indices, precision_matrix,
                shift, normalization, values, remain_indices)
        cond_mean, cond_covar = convert_gaussian_from_canonical_form(
            cond_precision, cond_shift)

        mu0, mu1 = mean
        sigma = np.sqrt(np.diag(covariance))
        correlation = covariance[1, 0]/(sigma[0]*sigma[1])
        true_cond_mean = mean[remain_indices] + \
            sigma[remain_indices]/sigma[fixed_indices]*correlation*(
                values-mean[fixed_indices])
        true_cond_covar = (1-correlation**2)*sigma[remain_indices]**2
        assert np.allclose(true_cond_mean, cond_mean)
        assert np.allclose(true_cond_covar, cond_covar)

    def test_marginalize_gaussian_in_canonical_form(self):
        covariance = np.array([[1, 0.5], [0.5, 1]])
        mean = np.array([-1, 1])

        remain_indices, fixed_indices = np.array([0]), np.array([1])

        precision_matrix, shift, normalization = \
            convert_gaussian_to_canonical_form(mean, covariance)

        marg_precision, marg_shift, marg_normalization = \
            marginalize_gaussian_in_canonical_form(
                fixed_indices, precision_matrix,
                shift, normalization, remain_indices=None)

        marg_mean, marg_covar = convert_gaussian_from_canonical_form(
            marg_precision, marg_shift)

        true_marg_mean = mean[remain_indices]
        true_marg_covar = np.diag(covariance)[remain_indices]
        assert np.allclose(true_marg_mean, marg_mean)
        assert np.allclose(true_marg_covar, marg_covar)

        var_ids, nvars_per_var = [0, 1], [1, 1]
        factor = GaussianFactor(
            precision_matrix, shift, normalization, var_ids,
            nvars_per_var)
        factor.marginalize(fixed_indices)
        marg_mean, marg_covar = convert_gaussian_from_canonical_form(
            factor.precision_matrix, factor.shift)
        assert np.allclose(true_marg_mean, marg_mean)
        assert np.allclose(true_marg_covar, marg_covar)

    def test_get_square_matrix_partition_indices(self):
        # Test with consecutive montonically increasing all_block_ids on square
        # matrix
        nblocks = 4
        nentries_per_block = [2]*nblocks
        nrows = np.sum(nentries_per_block)
        selected_ids = [0, 2]
        all_ids = np.arange(nblocks)

        keep_rows, keep_rows_reduced = get_matrix_partition_indices(
            selected_ids, all_ids, nentries_per_block)

        assert np.allclose(keep_rows, [0, 1, 4, 5])
        assert np.allclose(keep_rows_reduced, [0, 1, 2, 3])

        # Test with consecutive montonically increasing all_block_ids on
        # rectangular matrix
        nblocks = 4
        nentries_per_block = [2]*nblocks
        nrows = np.sum(nentries_per_block)
        selected_ids = [0, 2]
        all_ids = np.arange(nblocks)

        keep_rows, keep_rows_reduced = get_matrix_partition_indices(
            selected_ids, all_ids, nentries_per_block)

        assert np.allclose(keep_rows, [0, 1, 4, 5])
        assert np.allclose(keep_rows_reduced, [0, 1, 2, 3])

        # Test with consecutive non montonic all_block_ids with some values
        # larger than number of row blocks
        nblocks = 4
        nentries_per_block = [3, 2, 1, 4]  # [2]*nblocks
        # rows [0,1,2, 3,4, 5, 6,7,8,9]
        nrows = np.sum(nentries_per_block)
        selected_ids = [9, 2]
        all_ids = [3, 2, 6, 9]
        keep_rows, keep_rows_reduced = get_matrix_partition_indices(
            selected_ids, all_ids, nentries_per_block)

        true_keep_rows = [3, 4, 6, 7, 8, 9]
        true_keep_rows_reduced = [4, 5, 0, 1, 2, 3]
        assert np.allclose(keep_rows, true_keep_rows)
        assert np.allclose(keep_rows_reduced, true_keep_rows_reduced)

        # Test with consecutive non montonic all_block_ids with some values
        # larger than number of row blocks
        nblocks = 5
        nentries_per_block = [2]*nblocks  # [2,1,3,2,4]
        nblocks = len(nentries_per_block)
        nrows = np.sum(nentries_per_block)
        selected_ids = [9, 2, 6]
        all_ids = [3, 2, 6, 9, 1]
        keep_rows, keep_rows_reduced = get_matrix_partition_indices(
            selected_ids, all_ids, nentries_per_block)

        true_keep_rows = [2, 3, 4, 5, 6, 7]
        true_keep_rows_reduced = [2, 3, 4, 5, 0, 1]
        assert np.allclose(keep_rows, true_keep_rows)
        assert np.allclose(keep_rows_reduced, true_keep_rows_reduced)

        # all ids and vector entry indices
        # |  3  | 2 |   6   |  9  |     1     |
        # | 0 1 | 2 | 3 4 5 | 6 7 | 8 9 10 11 |
        # selected ids and vector indices sorted in order they appear in all_ids
        # | 2 | 6     |  9  |
        # | 2 | 3 4 5 | 6 7 |
        # selected_ids and vector indices ordered as they appear in selected_ids
        # i.e. the reduced matrix
        # | 9   | 2 |   6   |
        # | 0 1 | 2 | 3 4 5 |
        # now selected ids and indices in order as they appear in all_ids
        # | 2 |   6   |  9  |
        # | 2 | 3 4 5 | 0 1 |

        nentries_per_block = [2, 1, 3, 2, 4]
        nblocks = len(nentries_per_block)
        selected_ids = [9, 2, 6]
        all_ids = [3, 2, 6, 9, 1]
        keep_rows, keep_rows_reduced = get_matrix_partition_indices(
            selected_ids, all_ids, nentries_per_block)

        true_keep_rows = [2, 3, 4, 5, 6, 7]
        true_keep_rows_reduced = [2, 3, 4, 5, 0, 1]
        assert np.allclose(keep_rows, true_keep_rows)
        assert np.allclose(keep_rows_reduced, true_keep_rows_reduced)

    def test_joint_density_from_linear_conditional_relationship(self):
        """
        This tests only checks whether conditional denisty can be obtained
        after forming joint density and then conditioning the data again. Thus
        it relies on condition_gaussian_on_data working.
        """
        nvars1, nvars2 = [3, 2]
        Amat = np.random.normal(0., 1., (nvars2, nvars1))
        bvec = np.random.normal(0, 1, (nvars2))
        mean1 = np.random.normal(0, 1, nvars1)
        true_joint_mean = np.concatenate([mean1, Amat.dot(mean1)+bvec])
        temp = np.random.normal(0., 1., (nvars1, nvars1))
        cov1 = temp.T.dot(temp)
        temp = np.random.normal(0., 1., (nvars2, nvars2))
        cov2g1 = temp.T.dot(temp)

        # Compute joint probability P(x1,x2)
        joint_mean, joint_covar = \
            joint_density_from_linear_conditional_relationship(
                mean1, cov1, cov2g1, Amat, bvec)
        assert np.allclose(joint_mean, true_joint_mean)
        assert joint_covar.shape == (nvars1+nvars2, nvars1+nvars2)

        # Compute conditonal probability P(x2|x1)
        fixed_indices = np.arange(nvars1)
        values = np.array([0.25]*nvars1)
        mean_2g1, cov_2g1 = condition_gaussian_on_data(
            joint_mean, joint_covar, fixed_indices, values)

        true_mean_2g1 = Amat.dot(values)+bvec
        true_cov_2g1 = cov2g1

        assert np.allclose(true_mean_2g1, mean_2g1)
        assert np.allclose(true_cov_2g1, cov_2g1)

    def test_marginal_density_from_linear_conditional_relationship(self):
        """
        This tests only checks whether marginal denisty P(x2) can be
        obtained after starting with conditional density P(x2|x1) and
        marginal density P(x1) forming joint density P(x1,x2) and
        then marginalizing to obtain P(x2). Thus it relies on
        joint_density_from_Linear_conditional_relationship working.
        """
        nvars1, nvars2 = [3, 2]
        Amat = np.random.normal(0., 1., (nvars2, nvars1))
        bvec = np.random.normal(0, 1, (nvars2))
        mean1 = np.random.normal(0, 1, nvars1)
        temp = np.random.normal(0., 1., (nvars1, nvars1))
        cov1 = temp.T.dot(temp)
        temp = np.random.normal(0., 1., (nvars2, nvars2))
        cov2g1 = temp.T.dot(temp)

        mean2, cov2 = marginal_density_from_linear_conditional_relationship(
            mean1, cov1, cov2g1, Amat, bvec)

        # Compute joint probability P(x1,x2)
        joint_mean, joint_covar = \
            joint_density_from_linear_conditional_relationship(
                mean1, cov1, cov2g1, Amat, bvec)

        true_mean2 = joint_mean[nvars1:]
        indices2 = np.arange(nvars1, nvars1+nvars2)
        true_cov2 = joint_covar[np.ix_(indices2, indices2)]

        assert np.allclose(true_mean2, mean2)
        assert np.allclose(true_cov2, cov2)

    def test_conditional_density_from_linear_conditional_relationship(self):
        """
        This tests only checks whether conditional denisty P(x2|x1) can be
        obtained after starting with conditional density P(x2|x1) and
        marginal density P(x1) forming joint density P(x1,x2) and
        then conditioning the data again to obtain P(x1|x2). Thus it relies on
        condition_gaussian_on_data and
        joint_density_from_Linear_conditional_relationship working.
        """
        nvars1, nvars2 = [3, 2]
        Amat = np.random.normal(0., 1., (nvars2, nvars1))
        bvec = np.random.normal(0, 1, (nvars2))
        mean1 = np.random.normal(0, 1, nvars1)
        temp = np.random.normal(0., 1., (nvars1, nvars1))
        cov1 = temp.T.dot(temp)
        temp = np.random.normal(0., 1., (nvars2, nvars2))
        cov2g1 = temp.T.dot(temp)

        # Compute joint probability P(x1,x2)
        joint_mean, joint_covar = \
            joint_density_from_linear_conditional_relationship(
                mean1, cov1, cov2g1, Amat, bvec)

        # Compute conditonal probability P(x1|x2) from P(x1) and P(x2|x1)
        values = np.array([0.25]*nvars2)
        mean_1g2, cov_1g2 =\
            conditional_density_from_linear_conditional_relationship(
                mean1, cov1, cov2g1, Amat, bvec, values)

        # Compute conditonal probability P(x1|x2) from P(x1,x2)
        fixed_indices = np.arange(nvars1, nvars1+nvars2)
        true_mean_1g2, true_cov_1g2 = condition_gaussian_on_data(
            joint_mean, joint_covar, fixed_indices, values)

        assert np.allclose(true_mean_1g2, mean_1g2)
        assert np.allclose(true_cov_1g2, cov_1g2)

    def test_convert_conditional_probability_density_to_canonical_form(self):
        nvars1, nvars2 = [3, 2]
        Amat = np.random.normal(0., 1., (nvars2, nvars1))
        bvec = np.random.normal(0, 1, (nvars2))
        mean1 = np.random.normal(0, 1, nvars1)
        true_joint_mean = np.concatenate([mean1, Amat.dot(mean1)+bvec])
        temp = np.random.normal(0., 1., (nvars1, nvars1))
        cov1 = temp.T.dot(temp)
        temp = np.random.normal(0., 1., (nvars2, nvars2))
        cov2g1 = temp.T.dot(temp)

        # Compute joint probability P(x1,x2)
        joint_mean, joint_covariance = \
            joint_density_from_linear_conditional_relationship(
                mean1, cov1, cov2g1, Amat, bvec)

        precision_matrix, shift, normalization = \
            convert_gaussian_to_canonical_form(mean1, cov1)
        factor1 = GaussianFactor(
            precision_matrix, shift, normalization, [0], [nvars1])

        factor2 = GaussianFactor(
            *convert_conditional_probability_density_to_canonical_form(
                Amat, bvec, cov2g1, [0], [nvars1], [1], [nvars2]))

        joint_factor = factor1*factor2
        joint_factor_mean, joint_factor_covariance = \
            convert_gaussian_from_canonical_form(
                joint_factor.precision_matrix, joint_factor.shift)

        assert np.allclose(joint_factor_mean, joint_mean)
        assert np.allclose(joint_factor_covariance, joint_covariance)

    def generate_gaussian_in_canonical_random(self, nvars):
        precision_matrix = np.random.normal(0, 1, (nvars, nvars))
        precision_matrix = precision_matrix.T.dot(precision_matrix)
        mean = np.random.normal(0., 1., nvars)
        shift = precision_matrix.dot(mean)
        normalization = compute_gaussian_pdf_canonical_form_normalization(
            mean, shift, precision_matrix)
        return precision_matrix, shift, normalization

    def test_multiply_gaussian_densities_in_canonical_form(self):
        var1_ids, var2_ids = [0, 1, 2], [0, 1, 2]
        nvars_per_var1, nvars_per_var2 = [2, 3, 1], [2, 3, 1]
        nvars1, nvars2 = sum(nvars_per_var1), sum(nvars_per_var2)
        precision_matrix1, shift1, normalization1 = \
            self.generate_gaussian_in_canonical_random(nvars1)
        precision_matrix2, shift2, normalization2 = \
            self.generate_gaussian_in_canonical_random(nvars2)

        (precision_matrix, shift, normalization, all_var_ids,
         nvars_per_all_vars) = \
             multiply_gaussian_densities_in_compact_canonical_form(
                 precision_matrix1, shift1, normalization1, var1_ids,
                 nvars_per_var1, precision_matrix2, shift2, normalization2,
                 var2_ids, nvars_per_var2)
        mean, covariance = convert_gaussian_from_canonical_form(
            precision_matrix, shift)

        mean1, covariance1 = convert_gaussian_from_canonical_form(
            precision_matrix1, shift1)
        mean2, covariance2 = convert_gaussian_from_canonical_form(
            precision_matrix2, shift2)
        true_mean, true_covariance = multiply_gaussian_densities(
            mean1, covariance1, mean2, covariance2)

        assert np.allclose(mean, true_mean)
        assert np.allclose(covariance, true_covariance)
        assert np.allclose(all_var_ids, np.concatenate([var1_ids]))
        assert np.allclose(nvars_per_all_vars, nvars_per_var1)

    def test_expand_scope_of_gaussian(self):
        nvars_per_new_var = np.array([4, 2, 3])
        old_var_ids = np.array([1, 2])
        new_var_ids = np.array([0, 1, 2])

        nold_vars = nvars_per_new_var[old_var_ids].sum()
        matrix = np.arange(nold_vars**2).reshape(nold_vars, nold_vars)
        vector = np.arange(nold_vars)

        new_matrix, new_vector = expand_scope_of_gaussian(
            old_var_ids, new_var_ids, nvars_per_new_var, matrix,
            vector)

        num_new_vars = nvars_per_new_var.sum()
        true_new_matrix = np.zeros((num_new_vars, num_new_vars))
        true_new_matrix[nvars_per_new_var[0]:, nvars_per_new_var[0]:] = matrix
        # print(true_new_matrix)
        assert np.allclose(new_matrix, true_new_matrix)

        nvars_per_new_var = np.array([4, 2, 3])
        old_var_ids = np.array([0, 2])
        new_var_ids = np.array([0, 1, 2])

        nold_vars = nvars_per_new_var[old_var_ids].sum()
        matrix = np.arange(nold_vars**2).reshape(nold_vars, nold_vars)
        vector = np.arange(nold_vars)

        new_matrix, new_vector = expand_scope_of_gaussian(
            old_var_ids, new_var_ids, nvars_per_new_var, matrix,
            vector)
        # print(new_matrix)

        num_new_vars = nvars_per_new_var.sum()
        true_new_matrix = np.zeros((num_new_vars, num_new_vars))
        true_new_matrix[:nvars_per_new_var[0], :nvars_per_new_var[0]] = \
            matrix[:nvars_per_new_var[0], :nvars_per_new_var[0]]
        true_new_matrix[nvars_per_new_var[:2].sum(
        ):, :nvars_per_new_var[0]] = matrix[
            nvars_per_new_var[0]:, :nvars_per_new_var[0]]
        true_new_matrix[:nvars_per_new_var[0], nvars_per_new_var[:2].sum(
        ):] = matrix[:nvars_per_new_var[0], nvars_per_new_var[0]:]
        true_new_matrix[nvars_per_new_var[:2].sum():,
                        nvars_per_new_var[:2].sum():] = matrix[
                            nvars_per_new_var[0]:, nvars_per_new_var[0]:]
        # print(true_new_matrix)
        assert np.allclose(new_matrix, true_new_matrix)


if __name__ == '__main__':
    gaussian_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGaussian)
    unittest.TextTestRunner(verbosity=2).run(gaussian_test_suite)
