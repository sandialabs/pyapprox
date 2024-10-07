import unittest
from pyapprox.sciml.util.torchutils import TorchUtilitiesSciML


class TestFCT(unittest.TestCase, TorchUtilitiesSciML):
    def setUp(self):
        self._la_random_seed(1389745)

    def test_fct_1d(self):
        n = 20
        pts = self._la_cos(self._la_pi() * self._la_arange(0, n+1)/n)
        values = self._la_cos(2*self._la_pi()*3.0*pts+0.5)
        w = self._la_empty(n+1)
        w[:] = 2
        w[0] = 1
        w[-1] = 1

        basis_mat = self._sciml_chebyshev_poly_basis(pts, n+1).T
        lstsq_coef = self._la_lstsq(basis_mat, values, rcond=None)[0]
        lstsq_coef = self._la_atleast1d(lstsq_coef)

        # Test forward Chebyshev transform
        coef = self._sciml_fct(values)
        assert self._la_allclose(coef, lstsq_coef), 'Error: Forward DCT-1D'

        # Test inverse Chebyshev transform
        recovered_values = self._sciml_ifct(coef)
        assert self._la_allclose(values, recovered_values), (
            'Error: Inverse DCT-1D')

        # Test batch Chebyshev transform
        batch_values = self._la_normal(0, 1, (n+1, 2))
        batch_coefs = self._sciml_fct(batch_values)
        assert self._la_allclose(batch_values,
                                 self._sciml_ifct(batch_coefs)), (
                'Error: Batch inverse DCT')
        assert self._la_allclose(self._sciml_fct(batch_values[:, 0]),
                                 batch_coefs[:, 0]), 'Error: Batch DCT'

        # Sanity check for circular convolution function
        u = self._la_normal(0, 1, (n+1, 1, 1))
        v = self._la_normal(0, 1, (n+1, 1, 1))
        u_conv_v = (
            self._sciml_circ_conv(u.flatten(), v.flatten())[:, None, None])
        assert self._la_allclose(self._la_fft(u_conv_v),
                                 self._la_fft(u)*self._la_fft(v)), (
                'Error: Violation of Fourier Convolution Theorem')
        assert self._la_allclose(self._la_ifft(u_conv_v),
                                 (n+1)*self._la_ifft(u)*self._la_ifft(v)), (
                'Error: Violation of Inverse Fourier Convolution Theorem')

        # Test forward Chebyshev convolution property
        u, v = u.flatten(), v.flatten()
        u_per = self._la_hstack([u, self._la_flip(u[1:-1], axis=0)])
        v_per = self._la_hstack([v, self._la_flip(v[1:-1], axis=0)])
        u_tconv_v = self._sciml_circ_conv(u_per, v_per)[:n+1]
        u_fct = self._sciml_fct(u)
        v_fct = self._sciml_fct(v)
        assert self._la_allclose(self._sciml_fct(u_tconv_v),
                                 u_fct*v_fct*2*n/w), (
            'Error: Forward Chebyshev convolution')

        # Test inverse Chebyshev convolution property
        u_tconv_v_w = w*u_tconv_v.flatten()
        u_w_ifct = self._sciml_ifct(w*u.flatten())
        v_w_ifct = self._sciml_ifct(w*v.flatten())
        assert self._la_allclose(self._sciml_ifct(u_tconv_v_w),
                                 u_w_ifct*v_w_ifct), (
            'Error: Inverse Chebyshev convolution')

    def test_fct_multidim(self):
        # interpolation in 2D
        n = 20
        pts = self._la_cos(self._la_pi() * self._la_arange(0, n+1)/n)
        (X, Y) = self._la_meshgrid(pts, pts, indexing='xy')
        Z = self._la_cos(2*self._la_pi()*3.0*X+0.5)*Y**2

        # Solve least-squares problem for coefficients
        basis_mat = self._sciml_chebyshev_poly_basis(pts, n+1).T
        Phi = self._la_kron(basis_mat, basis_mat)
        lstsq_coef = self._la_lstsq(Phi, Z.flatten(), rcond=None)[0]

        # Use FCT (extra dimensions for channels and realizations)
        coef = self._sciml_fct(Z[..., None, None]).flatten()
        assert self._la_allclose(coef, lstsq_coef), (
            'Error: 2D-DCT != Vandermonde')

        # tensor sizes
        n1, n2, n3, n4 = 17, 5, 9, 3
        ntrain = 10
        d_c = 1

        # 2D
        x = self._la_normal(0, 1, (n1, n2, d_c, ntrain))
        out = self._la_copy(x)
        for i in range(x.shape[0]):
            out[i, :, :] = self._sciml_fct(out[i, :, :, :])

        for j in range(x.shape[1]):
            out[:, j, :] = self._sciml_fct(out[:, j, :, :])

        assert self._la_allclose(out, self._sciml_fct(x)), (
            'Error: Forward DCT, 2D')
        assert self._la_allclose(self._sciml_ifct(self._sciml_fct(x)), x), (
            'Error: Inverse DCT, 2D')

        # 3D
        x = self._la_normal(0, 1, (n1, n2, n3, d_c, ntrain))
        out = self._la_copy(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i, j, :, :] = self._sciml_fct(out[i, j, :, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                out[i, :, j, :] = self._sciml_fct(out[i, :, j, :, :])

        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                out[:, i, j, :] = self._sciml_fct(out[:, i, j, :, :])

        assert self._la_allclose(out, self._sciml_fct(x)), (
            'Error: Forward DCT, 3D')
        assert self._la_allclose(self._sciml_ifct(self._sciml_fct(x)), x), (
            'Error: Inverse DCT, 3D')

        # 4D
        x = self._la_normal(0, 1, (n1, n2, n3, n4, d_c, ntrain))
        out = self._la_copy(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    out[i, j, k, :, :] = self._sciml_fct(out[i, j, k, :, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[3]):
                    out[i, j, :, k, :] = self._sciml_fct(out[i, j, :, k, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    out[i, :, j, k, :] = self._sciml_fct(out[i, :, j, k, :, :])

        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    out[:, i, j, k, :] = self._sciml_fct(out[:, i, j, k, :, :])

        assert self._la_allclose(out, self._sciml_fct(x)), (
            'Error: Forward DCT, 4D')
        assert self._la_allclose(self._sciml_ifct(self._sciml_fct(x)), x), (
            'Error: Inverse DCT, 4D')


if __name__ == '__main__':
    fct_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestFCT))
    unittest.TextTestRunner(verbosity=2).run(fct_test_suite)
