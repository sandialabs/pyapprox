import unittest
import numpy as np
import scipy

from pyapprox.surrogates.kernels.numpykernels import (
    NumpyMaternKernel, NumpyConstantKernel, NumpySphericalCovariance)
from pyapprox.surrogates.kernels.torchkernels import (
    TorchMaternKernel, TorchSphericalCovariance)
from pyapprox.surrogates.autogp.numpytrends import NumpyMonomial
from pyapprox.surrogates.autogp.torchtrends import TorchMonomial
from pyapprox.surrogates.autogp.mokernels import (
    MultiLevelKernel, MultiPeerKernel, _get_recursive_scaling_matrix,
    ICMKernel, CollaborativeKernel)


class TestMultiOutputKernels(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _check_multilevel_kernel_scaling_matrix(self, noutputs, MaternKernel,
                                                Monomial):
        nvars, degree = 1, 0
        kernels = [
            MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
            for ii in range(noutputs)]
        scaling_vals = np.arange(2, noutputs+1)
        scalings = [
            Monomial(nvars, degree, scaling, [0, 2+noutputs],
                     name=f'scaling{ii}')
            for ii, scaling in enumerate(scaling_vals)]
        kernel = MultiLevelKernel(kernels, scalings)

        W = np.zeros((kernel.noutputs, kernel.nkernels))
        sample = np.zeros((nvars, 1))
        for ii in range(kernel.noutputs):
            for kk in range(kernel.noutputs):
                W[ii, kk] = kernel._get_kernel_combination_matrix_entry(
                    sample, ii, kk)
        W[np.isnan(W)] = 0.

        W_true = _get_recursive_scaling_matrix(scaling_vals, kernel.noutputs)
        assert np.allclose(W_true, W)

    def test_multilevel_kernel_scaling_matrix(self):
        for kk in range(2, 5):
            self._check_multilevel_kernel_scaling_matrix(
                kk, NumpyMaternKernel, NumpyMonomial)
        for kk in range(2, 5):
            self._check_multilevel_kernel_scaling_matrix(
                kk, TorchMaternKernel, TorchMonomial)

    def _check_spatially_scaled_multioutput_kernel_covariance(
            self, kernel, samples_per_output):
        nsamples_per_output = [s.shape[1] for s in samples_per_output]
        kmat = kernel(samples_per_output)
        assert np.allclose(kmat, kmat.T)
        assert np.allclose(kernel._la_get_diagonal(kmat),
                           kernel.diag(samples_per_output))

        # test evaluation when two sample sets are provided
        from copy import deepcopy
        noutputs = len(samples_per_output)
        samples_per_output_test = deepcopy(samples_per_output)
        samples_per_output_test[2:] = [np.array([[]])]*(noutputs-2)
        kmat_XY = kernel(samples_per_output_test, samples_per_output)
        cnt = sum([s.shape[1] for s in samples_per_output_test])
        assert np.allclose(kmat[:cnt, :], kmat_XY)
        kmat_diag = kernel.diag(samples_per_output_test)
        assert np.allclose(
            kmat_diag, kernel._la_get_diagonal(kmat[:cnt, :cnt]))

        samples_per_output_test = deepcopy(samples_per_output)
        samples_per_output_test[:1] = [np.array([[]])]
        kmat_XY = kernel(samples_per_output_test, samples_per_output)
        assert np.allclose(kmat[samples_per_output[0].shape[1]:, :], kmat_XY)

        kmat_diag = kernel.diag(samples_per_output_test)
        assert np.allclose(
            kmat_diag, kernel._la_get_diagonal(
                kmat[samples_per_output[0].shape[1]:,
                     samples_per_output[0].shape[1]:]))

        nsamples = int(5e6)
        DD_list_0 = [
            kernel._la_atleast2d(
                np.linalg.cholesky(
                    kernel.kernels[kk](samples_per_output[0])).dot(
                        np.random.normal(
                            0, 1, (nsamples_per_output[0], nsamples))))
            for kk in range(kernel.nkernels)]
        # samples must be nested for tests to work
        DD_lists = [[DD[:nsamples_per_output[ii], :] for DD in DD_list_0]
                    for ii in range(kernel.noutputs)]

        for ii in range(kernel.noutputs):
            vals = 0
            for kk in range(kernel.nkernels):
                wmat_iikk = kernel._get_kernel_combination_matrix_entry(
                    samples_per_output[ii], ii, kk)
                if wmat_iikk is not None:
                    vals += wmat_iikk*DD_lists[ii][kk]
            diag_block = np.cov(vals, ddof=1)
            assert np.allclose(
                diag_block,
                kernel._evaluate_block(
                    samples_per_output[ii], ii, samples_per_output[ii], ii,
                    False, True),
                rtol=1e-2)
            for jj in range(ii+1, kernel.noutputs):
                vals_ii = kernel._la_full(
                    (nsamples_per_output[ii], nsamples), 0.)
                vals_jj = kernel._la_full(
                    (nsamples_per_output[jj], nsamples), 0.)
                for kk in range(kernel.nkernels):
                    wmat_iikk = kernel._get_kernel_combination_matrix_entry(
                        samples_per_output[ii], ii, kk)
                    if wmat_iikk is not None:
                        vals_ii += wmat_iikk * DD_lists[ii][kk]
                for kk in range(kernel.nkernels):
                    wmat_jjkk = kernel._get_kernel_combination_matrix_entry(
                        samples_per_output[jj], jj, kk)
                    if wmat_jjkk is not None:
                        vals_jj += wmat_jjkk * DD_lists[jj][kk]
                kmat_iijj = kernel._evaluate_block(
                    samples_per_output[ii], ii, samples_per_output[jj], jj,
                    False, True)
                kmat_iijj_mc = np.cov(vals_ii, vals_jj, ddof=1)[
                    :nsamples_per_output[ii],
                    nsamples_per_output[ii]:]
                if np.abs(kmat_iijj).sum() > 0:
                    assert np.allclose(kmat_iijj, kmat_iijj_mc,  rtol=1e-2)
                else:
                    assert np.allclose(kmat_iijj, kmat_iijj_mc,  atol=2e-3)

    def _check_multioutput_kernel_3_outputs(
            self, nvars, degree, MOKernel, MaternKernel, Monomial):
        nsamples_per_output = [4, 3, 2]
        kernels = [MaternKernel(np.inf, 1.0, [1e-1, 1], nvars),
                   MaternKernel(np.inf, 2.0, [1e-2, 10], nvars),
                   MaternKernel(np.inf, .05, [1e-3, 0.1], nvars)]
        scalings = [
            Monomial(nvars, degree, 2, [-1, 2], name='scaling1'),
            Monomial(nvars, degree, -3, [-3, 3], name='scaling2')]
        kernel = MOKernel(kernels, scalings)
        base_training_samples = kernel._la_atleast2d(
            np.random.uniform(-1, 1, (nvars, nsamples_per_output[0])))
        # samples must be nested for tests to work
        samples_per_output = [
            base_training_samples[:, :nsamples]
            for nsamples in nsamples_per_output]
        self._check_spatially_scaled_multioutput_kernel_covariance(
           kernel, samples_per_output)

    def test_multioutput_kernels_3_outputs(self):
        test_cases = [
            [1, 0, MultiPeerKernel, NumpyMaternKernel, NumpyMonomial],
            [1, 1, MultiPeerKernel, NumpyMaternKernel, NumpyMonomial],
            [2, 1, MultiPeerKernel, NumpyMaternKernel, NumpyMonomial],
            [1, 0, MultiLevelKernel, NumpyMaternKernel, NumpyMonomial],
            [1, 0, MultiPeerKernel, TorchMaternKernel, TorchMonomial],
            [1, 1, MultiPeerKernel, TorchMaternKernel, TorchMonomial],
            [2, 1, MultiPeerKernel, TorchMaternKernel, TorchMonomial],
            [1, 0, MultiLevelKernel, TorchMaternKernel, TorchMonomial],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_multioutput_kernel_3_outputs(*test_case)

    def _check_coregionalization_kernel(
            self, noutputs, MaternKernel, SphericalCovariance):
        nvars = 1
        nsamples_per_output_0 = np.arange(2, 2+noutputs)[::-1]
        latent_kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
        radii = latent_kernel._la_arange(1, noutputs+1)
        radii_bounds = [0.1, 10]
        angles = np.pi/4
        output_kernel = SphericalCovariance(
            noutputs, radii, radii_bounds, angles=angles)
        kernel = ICMKernel(latent_kernel, output_kernel, noutputs)
        base_training_samples = np.random.uniform(
            -1, 1, (nvars, nsamples_per_output_0[0]))
        # samples must be nested for tests to work
        samples_per_output = [
            latent_kernel._la_atleast2d(base_training_samples[:, :nsamples])
            for nsamples in nsamples_per_output_0]
        kmat_diag = kernel.diag(samples_per_output)
        kmat = kernel(samples_per_output)
        assert np.allclose(latent_kernel._la_get_diagonal(kmat), kmat_diag)

        cnt = 0
        for nsamples, r in zip(nsamples_per_output_0, radii):
            assert np.allclose(kmat_diag[cnt:cnt+nsamples], r**2)
            cnt += nsamples
        cmat = kernel.output_kernels[0].get_covariance_matrix()
        assert np.allclose(
            kernel.get_output_kernel_correlations_from_psi(0),
            kernel._la_get_correlation_from_covariance(cmat)[0, 1:])

        # Test that when all samples are the same the kernel matrix is
        # equivalent to kronker-product of cov_matrix with kernels[0] matrix
        nsamples_per_output_0 = np.full((noutputs, ), 2)
        base_training_samples = kernel._la_atleast2d(
            np.random.uniform(-1, 1, (nvars, nsamples_per_output_0[0])))
        samples_per_output = [
            kernel._la_copy(base_training_samples)
            for nsamples in nsamples_per_output_0]
        kernel = ICMKernel(latent_kernel, output_kernel, noutputs)
        kmat = kernel(samples_per_output)
        cmat = kernel.output_kernels[0].get_covariance_matrix()
        assert np.allclose(
            kmat,
            kernel._la_kron(cmat, latent_kernel(base_training_samples)),
            atol=1e-12)

    def test_coregionalization_kernel(self):
        test_cases = [[kk, NumpyMaternKernel, NumpySphericalCovariance]
                      for kk in range(2, 6)]
        test_cases += [[kk, TorchMaternKernel, TorchSphericalCovariance]
                       for kk in range(2, 6)]
        for test_case in test_cases:
            self._check_coregionalization_kernel(*test_case)

    def _check_collaborative_kernel(self, noutputs, nlatent_kernels,
                                    MaternKernel, SphericalCovariance):
        nvars = 1
        nsamples_per_output_0 = np.arange(2, 2+noutputs)[::-1]
        latent_kernels = [
            MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
            for kk in range(nlatent_kernels)]
        radii, radii_bounds = np.arange(1, noutputs+1), [0.1, 10]
        angles = np.pi/4
        output_kernels = [
            SphericalCovariance(noutputs, radii, radii_bounds, angles=angles)
            for kk in range(nlatent_kernels)]
        discrepancy_kernels = [
            MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
            for ii in range(noutputs)]
        kernel = CollaborativeKernel(
            latent_kernels, output_kernels, discrepancy_kernels, noutputs)
        base_training_samples = np.random.uniform(
            -1, 1, (nvars, nsamples_per_output_0[0]))
        # samples must be nested for tests to work
        samples_per_output = [
            base_training_samples[:, :nsamples]
            for nsamples in nsamples_per_output_0]
        kmat_diag = kernel.diag(samples_per_output)
        kmat = kernel(samples_per_output)
        assert np.allclose(np.diag(kmat), kmat_diag)

    def test_collaborative_kernel(self):
        test_cases = [
            [2, 1, NumpyMaternKernel, NumpySphericalCovariance],
            [3, 2, NumpyMaternKernel, NumpySphericalCovariance],
            [4, 2, NumpyMaternKernel, NumpySphericalCovariance],
            [5, 1, NumpyMaternKernel, NumpySphericalCovariance],
            [2, 1, TorchMaternKernel, TorchSphericalCovariance],
            [3, 2, TorchMaternKernel, TorchSphericalCovariance],
            [4, 2, TorchMaternKernel, TorchSphericalCovariance],
            [5, 1, TorchMaternKernel, TorchSphericalCovariance]
        ]
        for test_case in test_cases:
            self._check_collaborative_kernel(*test_case)

        # check we can recover peer kernel when discrepancy kernel variances
        # are zero for two low-fidelity models and the low fidelity models
        # are only functions of a unique latent kernel
        noutputs, nvars = 3, 1
        peer_kernels = [
            NumpyMaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
            for kk in range(noutputs)]
        scalings = [
            NumpyMonomial(nvars, 0, 1, [-1, 2], name=f'scaling{ii}')
            for ii in range(noutputs-1)]
        peer_kernel = MultiPeerKernel(peer_kernels, scalings)
        nsamples_per_output_0 = np.arange(2, 2+noutputs)[::-1]
        base_training_samples = np.random.uniform(
            -1, 1, (nvars, nsamples_per_output_0[0]))
        # samples must be nested for tests to work
        samples_per_output = [
            base_training_samples[:, :nsamples]
            for nsamples in nsamples_per_output_0]
        peer_kmat = peer_kernel(samples_per_output)

        class HackKernel(NumpySphericalCovariance):
            def __init__(self, noutputs, cov_mat):
                super().__init__(noutputs)
                self.cov_mat = cov_mat
                # print(cov_mat)

            def __call__(self, ii, jj):
                return self.cov_mat[ii, jj]

        nlatent_kernels = 2
        latent_kernels = peer_kernels[:nlatent_kernels]
        # radii, radii_bounds = np.arange(1, noutputs+1), [0.1, 10]
        # angles = np.pi/4
        # output_kernels = [
        #     SphericalCovariance(noutputs, radii, radii_bounds, angles=angles)
        #     for kk in range(nlatent_kernels)]
        cov_mats = [np.array([[1., 0, 1], [0, 0, 0], [1, 0, 1]]),
                    np.array([[0., 0, 0], [0, 1, 1], [0, 1, 1]])]
        output_kernels = [
            HackKernel(noutputs, cov_mat) for cov_mat in cov_mats]
        discrepancy_kernels = [
            NumpyConstantKernel(0)*NumpyMaternKernel(
                np.inf, 1.0, [1e-1, 1], nvars)
            for ii in range(noutputs-1)] + [
                    NumpyMaternKernel(np.inf, 1.0, [1e-1, 1], nvars)]
        co_kernel = CollaborativeKernel(
            latent_kernels, output_kernels, discrepancy_kernels, noutputs)
        co_kmat = co_kernel(samples_per_output)
        assert np.allclose(peer_kmat, co_kmat)

    def _check_block_cholesky(self, MaternKernel, Monomial):
        noutputs, nvars, degree = 4, 1, 0
        nsamples_per_output = np.arange(2, 2+noutputs)[::-1]
        kernels = [MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
                   for ii in range(noutputs)]
        scalings = [
            Monomial(nvars, degree, 2, [-1, 2], name=f'scaling{ii}')
            for ii in range(noutputs-1)]
        kernel = MultiPeerKernel(kernels, scalings)
        base_training_samples = kernel._la_atleast2d(np.random.uniform(
            -1, 1, (nvars, nsamples_per_output[0])))
        # samples must be nested for tests to work
        samples_per_output = [
            base_training_samples[:, :nsamples]
            for nsamples in nsamples_per_output]

        kmat = kernel(samples_per_output, block_format=False)
        L_true = np.linalg.cholesky(kmat)

        blocks = kernel(samples_per_output, block_format=True)
        L = kernel._cholesky(noutputs, blocks, block_format=False, la=kernel)
        assert np.allclose(L, L_true)

        L_blocks = kernel._cholesky(
            noutputs, blocks, block_format=True, la=kernel)
        L = kernel._cholesky_blocks_to_dense(*L_blocks, la=kernel)
        assert np.allclose(L, L_true)
        assert np.allclose(
            kernel._logdet(*L_blocks, la=kernel), np.linalg.slogdet(kmat)[1])
        values = np.random.normal(0, 1, (L.shape[1], 1))
        assert np.allclose(
            kernel._lower_solve_triangular(*L_blocks, values, la=kernel),
            scipy.linalg.solve_triangular(L, values, lower=True))
        assert np.allclose(
            kernel._upper_solve_triangular(*L_blocks, values, la=kernel),
            scipy.linalg.solve_triangular(L.T, values, lower=False))
        assert np.allclose(
            kernel._cholesky_solve(*L_blocks, values, la=kernel),
            np.linalg.inv(kmat) @ values)

    def test_block_cholesky(self):
        test_cases = [
            [NumpyMaternKernel, NumpyMonomial]]
        for case in test_cases:
            self._check_block_cholesky(*case)


if __name__ == "__main__":
    multioutput_kernels_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestMultiOutputKernels))
    unittest.TextTestRunner(verbosity=2).run(multioutput_kernels_test_suite)
