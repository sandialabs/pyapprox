import unittest
import numpy as np

from pyapprox.util.misc import (
    split_indices,
    covariance_to_correlation,
    correlation_to_covariance,
    argsort_indices_leixographically,
    hash_array,
    unique_matrix_rows,
    unique_matrix_columns,
    get_all_sample_combinations,
    sublist,
    get_first_n_primes,
    all_primes_less_than_or_equal_to_n,
    lists_of_arrays_equal,
    unique_elements_from_2D_list,
    scipy_gauss_hermite_pts_wts_1D,
    scipy_gauss_legendre_pts_wts_1D,
    approx_jacobian_3D,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


class TestUtilities:
    def setUp(self):
        np.random.seed(1)

    def test_covariance_to_correlation(self):
        bkd = self.get_backend()
        nrows = 3
        A = bkd.asarray(np.random.normal(0, 1, (nrows, nrows)))
        cov = A.T @ A
        corr = covariance_to_correlation(cov, bkd)
        assert bkd.allclose(bkd.diag(corr), bkd.ones(nrows))
        recovered_cov = correlation_to_covariance(
            corr, bkd.sqrt(bkd.diag(cov))
        )
        assert bkd.allclose(cov, recovered_cov)

    def _check_split_indices(self, nelems, nsplits):
        bkd = self.get_backend()
        indices = split_indices(nelems, nsplits, bkd=bkd)
        split_array = np.array_split(np.arange(nelems), nsplits)
        true_indices = bkd.hstack(
            [bkd.zeros((1,), dtype=int)]
            + [bkd.array([a[-1] + 1], dtype=int) for a in split_array]
        )
        assert bkd.allclose(true_indices, indices)

    def test_split_indices(self):
        test_cases = [[10, 3], [6, 3], [3, 3]]
        for test_case in test_cases:
            self._check_split_indices(*test_case)

    def test_argsort_indices_leixographically(self):
        bkd = self.get_backend()
        indices = bkd.asarray([(1, 1), (2, 0), (1, 2), (0, 2)]).T
        true_sorted_indices = bkd.asarray([(0, 2), (1, 1), (2, 0), (1, 2)]).T
        assert bkd.allclose(
            indices[:, argsort_indices_leixographically(indices)],
            true_sorted_indices,
        )

    def test_hash_array(self):
        # test hash runs and produces the same hash when called twice
        # hash is not determistic is not guaranteed to be the same across
        # different Python versions, platforms or executions of the same
        # program so further testing is difficult
        bkd = self.get_backend()
        array = bkd.arange(3)
        assert hash_array(array, bkd) == hash_array(array, bkd)

        self.assertRaises(ValueError, hash_array, array[:, None], bkd)

    def test_unique_matrix_rows(self):
        bkd = self.get_backend()
        nrows, ncols = 4, 2
        mat = bkd.asarray(np.random.normal(0, 1, (nrows, ncols)))
        idx = [0, 1, 1, 2, 3]
        assert bkd.allclose(unique_matrix_rows(mat[idx], bkd), mat)

        self.assertRaises(ValueError, unique_matrix_rows, mat[0], bkd)

    def test_unique_matrix_cols(self):
        bkd = self.get_backend()
        nrows, ncols = 2, 4
        mat = bkd.asarray(np.random.normal(0, 1, (nrows, ncols)))
        idx = [0, 1, 1, 2, 3]
        assert bkd.allclose(unique_matrix_columns(mat[:, idx], bkd), mat)

    def test_get_all_sample_combinations(self):
        bkd = self.get_backend()
        samples1 = bkd.array([[1, 2], [2, 3]]).T
        samples2 = bkd.array([[0, 0, 0], [0, 1, 2]]).T
        true_samples = bkd.asarray(
            [
                [1, 2, 0, 0, 0],
                [1, 2, 0, 1, 2],
                [2, 3, 0, 0, 0],
                [2, 3, 0, 1, 2],
            ]
        ).T
        assert bkd.allclose(
            get_all_sample_combinations(samples1, samples2, bkd),
            true_samples,
        )

    def test_sublist(self):
        mylist = list(range(5))
        idx = [0, 1, 3]
        assert sublist(mylist, idx) == idx

    def test_lists_of_arrays_equal(self):
        bkd = self.get_backend()
        list1 = [bkd.arange(3), bkd.arange(1, 5)]
        assert lists_of_arrays_equal(list1, list1)
        assert not lists_of_arrays_equal(list1, list1 + list1)
        list2 = [bkd.arange(3), bkd.arange(1, 6)]
        assert not lists_of_arrays_equal(list1, list2)
        list3 = [bkd.arange(3), bkd.arange(2, 6)]
        assert not lists_of_arrays_equal(list1, list3)

    def test_unique_elements_from_2D_list(self):
        list1 = [list(range(3)), list(range(1, 4))]
        assert unique_elements_from_2D_list(list1) == [0, 1, 2, 3]

    def test_approx_jacobian_3D(self):
        bkd = self.get_backend()

        def fun(x):
            return bkd.arange(1, x.shape[0] + 1)[:, None] * x.T

        x0 = bkd.array([1, 0.5, 2])[:, None]
        fd_jac = approx_jacobian_3D(fun, x0, bkd=bkd)
        jac = []
        for ii in range(3):
            mat = bkd.zeros((3, 3))
            mat[:, ii] = ii + 1
            jac.append(mat)
        assert bkd.allclose(fd_jac, bkd.stack(jac))


class TestNumpyUtilities(TestUtilities, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin

    def test_get_first_n_primes(self):
        assert np.allclose(get_first_n_primes(5), [2, 3, 5, 7, 11])

    def test_all_primes_less_than_or_equal_to_n(self):
        assert np.allclose(all_primes_less_than_or_equal_to_n(7), [2, 3, 5, 7])

    def test_scipy_gauss_hermite_pts_wts_1D(self):
        xx, ww = scipy_gauss_hermite_pts_wts_1D(3)
        assert np.allclose(xx**2 @ ww, 1)

    def test_scipy_gauss_legendre_pts_wts_1D(self):
        xx, ww = scipy_gauss_legendre_pts_wts_1D(3)
        assert np.allclose(xx**2 @ ww, 1 / 3)


class TestTorchUtilities(TestUtilities, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
