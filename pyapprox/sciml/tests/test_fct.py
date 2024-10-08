import unittest
import numpy as np
from pyapprox.sciml.util import fct
from pyapprox.sciml.util._torch_wrappers import asarray, hstack, flip


class TestFCT(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_fct_1d(self):
        n = 20
        pts = asarray(np.cos(np.pi*np.arange(0, n+1)/n))
        values = asarray(np.cos(2*np.pi*3.0*pts+0.5))
        w = 2*np.ones(n+1)
        w[0] = 1
        w[-1] = 1

        basis_mat = fct.chebyshev_poly_basis(pts, n+1).T
        lstsq_coef = np.linalg.lstsq(
            basis_mat.numpy(), values.numpy(), rcond=None)[0]

        # Test forward Chebyshev transform
        coef = fct.fct(values)
        assert np.allclose(coef.numpy(), lstsq_coef), 'Error: Forward DCT-1D'

        # Test inverse Chebyshev transform
        recovered_values = fct.ifct(coef)
        assert np.allclose(values.numpy(), recovered_values.numpy()), (
               'Error: Inverse DCT-1D')

        # Test batch Chebyshev transform
        batch_values = asarray(np.random.normal(0, 1, (n+1, 2)))
        batch_coefs = fct.fct(batch_values)
        assert np.allclose(batch_values, fct.ifct(batch_coefs)), ('Error: '
               'Batch inverse DCT')
        assert np.allclose(fct.fct(batch_values[:, 0]), batch_coefs[:, 0]), (
               'Error: Batch DCT')

        # Sanity check for circular convolution function
        u = asarray(np.random.normal(0, 1, (n+1,)))
        v = asarray(np.random.normal(0, 1, (n+1,)))
        assert np.allclose(
            np.fft.fft(fct.circ_conv(u, v)), np.fft.fft(u)*np.fft.fft(v)), (
                'Error: Violation of Fourier Convolution Theorem')
        assert np.allclose(np.fft.ifft(fct.circ_conv(u, v)),
                           (n+1)*np.fft.ifft(u)*np.fft.ifft(v)), ('Error: '
               'Violation of Inverse Fourier Convolution Theorem')

        # Test forward Chebyshev convolution property
        u_tconv_v = fct.circ_conv(hstack([u, flip(u[1:-1], dims=[0])]),
                                  hstack([v, flip(v[1:-1], dims=[0])]))[:n+1]
        assert np.allclose(fct.fct(u_tconv_v), fct.fct(u)*fct.fct(v)*2*n/w), (
               'Error: Forward Chebyshev convolution')

        # Test inverse Chebyshev convolution property
        assert np.allclose(fct.ifct(asarray(w)*u_tconv_v),
                           fct.ifct(asarray(w)*u)*fct.ifct(asarray(w)*v)), (
               'Error: Inverse Chebyshev convolution')

    def test_fct_multidim(self):
        # interpolation in 2D
        n = 20
        pts = np.cos(np.pi*np.arange(0, n+1)/n)
        (X, Y) = np.meshgrid(pts, pts)
        Z = np.cos(2*np.pi*3.0*X+0.5)*Y**2

        # Solve least-squares problem for coefficients
        basis_mat = fct.chebyshev_poly_basis(asarray(pts), n+1).T.numpy()
        Phi = np.kron(basis_mat, basis_mat)
        lstsq_coef = np.linalg.lstsq(Phi, Z.flatten(), rcond=None)[0]

        # Use FCT (extra dimensions for channels and realizations)
        coef = fct.fct(asarray(Z)[..., None, None])[..., 0, 0].flatten()
        assert np.allclose(coef, lstsq_coef), 'Error: 2D-DCT != Vandermonde'

        # tensor sizes
        n1, n2, n3, n4 = 17, 5, 9, 3
        ntrain = 10
        d_c = 1

        # 2D
        x = asarray(np.random.rand(n1, n2, d_c, ntrain))
        out = x.clone()
        for i in range(x.shape[0]):
            out[i, :, :] = fct.fct(out[i, :, :, :])

        for j in range(x.shape[1]):
            out[:, j, :] = fct.fct(out[:, j, :, :])

        assert np.allclose(out, fct.fct(x)), 'Error: Forward DCT, 2D'
        assert np.allclose(fct.ifct(fct.fct(x)), x), 'Error: Inverse DCT, 2D'

        # 3D
        x = asarray(np.random.rand(n1, n2, n3, d_c, ntrain))
        out = x.clone()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i, j, :, :] = fct.fct(out[i, j, :, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                out[i, :, j, :] = fct.fct(out[i, :, j, :, :])

        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                out[:, i, j, :] = fct.fct(out[:, i, j, :, :])

        assert np.allclose(out, fct.fct(x)), 'Error: Forward DCT, 3D'
        assert np.allclose(fct.ifct(fct.fct(x)), x), 'Error: Inverse DCT, 3D'

        # 4D
        x = asarray(np.random.rand(n1, n2, n3, n4, d_c, ntrain))
        out = x.clone()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    out[i, j, k, :, :] = fct.fct(out[i, j, k, :, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[3]):
                    out[i, j, :, k, :] = fct.fct(out[i, j, :, k, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    out[i, :, j, k, :] = fct.fct(out[i, :, j, k, :, :])

        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    out[:, i, j, k, :] = fct.fct(out[:, i, j, k, :, :])

        assert np.allclose(out, fct.fct(x)), 'Error: Forward DCT, 4D'
        assert np.allclose(fct.ifct(fct.fct(x)), x), 'Error: Inverse DCT, 4D'


if __name__ == '__main__':
    fct_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestFCT))
    unittest.TextTestRunner(verbosity=2).run(fct_test_suite)
