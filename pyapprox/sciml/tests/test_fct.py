import unittest
from pyapprox.sciml.util import FCT, NumpyLinAlgMixin


class TestFCT(unittest.TestCase):
    def setUp(self):
        self._bkd = NumpyLinAlgMixin
        self._bkd.random_seed(1)
        self.fct = FCT(backend=self._bkd)

    def test_fct_1d(self):
        n = 20
        fct = FCT(backend=self._bkd)
        pi = self._bkd.arccos(-1)
        pts = self._bkd.cos(pi * self._bkd.arange(0, n+1)/n)
        values = self._bkd.cos(2*pi*3.0*pts + 0.5)
        w = 2*self._bkd.ones(n+1)
        w[0] = 1
        w[-1] = 1

        basis_mat = fct.chebyshev_poly_basis(pts, n+1).T
        lstsq_coef = self._bkd.lstsq(basis_mat, values)

        # Test forward Chebyshev transform
        coef = fct.fct(values)
        assert self._bkd.allclose(coef, lstsq_coef), 'Error: Forward DCT-1D'

        # Test inverse Chebyshev transform
        recovered_values = fct.ifct(coef)
        assert self._bkd.allclose(values, recovered_values), (
               'Error: Inverse DCT-1D')

        # Test batch Chebyshev transform
        batch_values = self._bkd.normal(0, 1, (n+1, 2))
        batch_coefs = fct.fct(batch_values)
        assert self._bkd.allclose(batch_values, fct.ifct(batch_coefs)), (
               'Error: Batch inverse DCT')
        assert self._bkd.allclose(fct.fct(batch_values[:, 0]),
                                  batch_coefs[:, 0]), (
               'Error: Batch DCT')

        # Sanity check for circular convolution function
        u = self._bkd.normal(0, 1, (n+1, 1, 1))
        v = self._bkd.normal(0, 1, (n+1, 1, 1))
        assert self._bkd.allclose(
            self._bkd.fft(fct.circ_conv(u, v)),
            self._bkd.fft(u)*self._bkd.fft(v)), (
            'Error: Violation of Fourier Convolution Theorem')
        assert self._bkd.allclose(self._bkd.ifft(fct.circ_conv(u, v)),
                                  (n+1)*self._bkd.ifft(u)*self._bkd.ifft(v)), (
                   'Error: Violation of Inverse Fourier Convolution Theorem')

        # Test forward Chebyshev convolution property
        u = u.flatten()
        v = v.flatten()
        u_tconv_v = fct.circ_conv(
                        self._bkd.hstack([u, self._bkd.flip(u[1:-1], axis=0)]),
                        self._bkd.hstack([v, self._bkd.flip(v[1:-1], axis=0)])
                    )[:n+1]
        assert self._bkd.allclose(fct.fct(u_tconv_v),
                                  fct.fct(u)*fct.fct(v)*2*n/w), (
               'Error: Forward Chebyshev convolution')

        # Test inverse Chebyshev convolution property
        assert self._bkd.allclose(fct.ifct(w*u_tconv_v),
                                  fct.ifct(w*u)*fct.ifct(w*v)), (
               'Error: Inverse Chebyshev convolution')

    def test_fct_multidim(self):
        # interpolation in 2D
        n = 20
        pi = self._bkd.arccos(-1)
        pts = self._bkd.cos(pi*self._bkd.arange(0, n+1)/n)
        (X, Y) = self._bkd.meshgrid(pts, pts)
        Z = self._bkd.cos(2*pi*3.0*X + 0.5)*Y**2

        # Solve least-squares problem for coefficients
        basis_mat = self.fct.chebyshev_poly_basis(pts, n+1).T
        Phi = self._bkd.kron(basis_mat, basis_mat)
        lstsq_coef = self._bkd.lstsq(Phi, Z.flatten())

        # Use FCT (extra dimensions for channels and realizations)
        coef = self.fct.fct(Z[..., None, None])[..., 0, 0].flatten()
        assert self._bkd.allclose(coef, lstsq_coef), (
            'Error: 2D-DCT != Vandermonde')

        # tensor sizes
        n1, n2, n3, n4 = 17, 5, 9, 3
        ntrain = 10
        d_c = 1

        # 2D
        x = self._bkd.normal(0, 1, (n1, n2, d_c, ntrain))
        out = self._bkd.copy(x)
        for i in range(x.shape[0]):
            out[i, :, :] = self.fct.fct(out[i, :, :, :])

        for j in range(x.shape[1]):
            out[:, j, :] = self.fct.fct(out[:, j, :, :])

        assert self._bkd.allclose(out, self.fct.fct(x)), (
            'Error: Forward DCT, 2D')
        assert self._bkd.allclose(self.fct.ifct(self.fct.fct(x)), x), (
            'Error: Inverse DCT, 2D')

        # 3D
        x = self._bkd.normal(0, 1, (n1, n2, n3, d_c, ntrain))
        out = self._bkd.copy(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i, j, :, :] = self.fct.fct(out[i, j, :, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                out[i, :, j, :] = self.fct.fct(out[i, :, j, :, :])

        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                out[:, i, j, :] = self.fct.fct(out[:, i, j, :, :])

        assert self._bkd.allclose(out, self.fct.fct(x)), (
            'Error: Forward DCT, 3D')
        assert self._bkd.allclose(self.fct.ifct(self.fct.fct(x)), x), (
            'Error: Inverse DCT, 3D')

        # 4D
        x = self._bkd.normal(0, 1, (n1, n2, n3, n4, d_c, ntrain))
        out = self._bkd.copy(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    out[i, j, k, :, :] = self.fct.fct(out[i, j, k, :, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[3]):
                    out[i, j, :, k, :] = self.fct.fct(out[i, j, :, k, :, :])

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    out[i, :, j, k, :] = self.fct.fct(out[i, :, j, k, :, :])

        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    out[:, i, j, k, :] = self.fct.fct(out[:, i, j, k, :, :])

        assert self._bkd.allclose(out, self.fct.fct(x)), (
            'Error: Forward DCT, 4D')
        assert self._bkd.allclose(self.fct.ifct(self.fct.fct(x)), x), (
            'Error: Inverse DCT, 4D')


if __name__ == '__main__':
    fct_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestFCT))
    unittest.TextTestRunner(verbosity=2).run(fct_test_suite)
