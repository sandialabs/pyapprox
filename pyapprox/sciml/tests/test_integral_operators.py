import unittest
from functools import partial

import numpy as np
import torch

from pyapprox.sciml.util import fct
from pyapprox.sciml.util import _torch_wrappers as tw
from pyapprox.sciml.network import CERTANN
from pyapprox.sciml.integraloperators import (
    FourierConvolutionOperator, ChebyshevConvolutionOperator,
    DenseAffineIntegralOperator, DenseAffineIntegralOperatorFixedBias,
    ChebyshevIntegralOperator)
from pyapprox.sciml.layers import Layer
from pyapprox.sciml.activations import IdentityActivation
from pyapprox.sciml.optimizers import Adam


class TestIntegralOperators(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)

    def test_fourier_convolution_operator_1d(self):
        N = 101
        xx = np.linspace(-1, 1, N)
        u = tw.asarray(xx**2)
        v = tw.asarray(1 / (1 + (5*xx)**2))

        u_conv_v = fct.circ_conv(u, v)

        kmax = (N-1)//2
        ctn = CERTANN(N, [Layer(FourierConvolutionOperator(kmax))],
                      [IdentityActivation()])
        training_samples = u[:, None]
        training_values = u_conv_v[:, None]
        ctn.fit(training_samples, training_values, tol=1e-10)
        fcoef_target = tw.hstack([tw.fft(v).real[:kmax+1],
                                  tw.fft(v).imag[1:kmax+1]])

        assert (
            tw.norm(fcoef_target - ctn._hyp_list.get_values()) /
            tw.norm(fcoef_target) < 4e-4)

    def test_fourier_convolution_operator_multidim(self):
        N = 101
        xx = np.linspace(-1, 1, N)
        (X, Y) = np.meshgrid(xx, xx)
        u = tw.asarray((X+Y)**2)[..., None]
        v = tw.asarray(1 / (1 + (5*X*Y)**2))[..., None]

        u_conv_v = tw.ifft(tw.fft(u)*tw.fft(v)).real

        kmax = 10
        layers = [Layer(FourierConvolutionOperator(kmax, shape=X.shape))]
        ctn = CERTANN(X.size, layers, [IdentityActivation()])
        ctn.fit(u.flatten()[:, None], u_conv_v.flatten()[:, None], tol=1e-5)

        fftshift_v = tw.fftshift(tw.fft(v))
        nyquist = [n//2 for n in X.shape]
        slices = [slice(n-kmax, n+kmax+1) for n in nyquist]
        fftshift_v_proj = fftshift_v[slices].flatten()
        fftshift_v_proj_trim = fftshift_v_proj[fftshift_v_proj.shape[0]//2:]
        fcoef_target = tw.hstack([fftshift_v_proj_trim.real.flatten(),
                                  fftshift_v_proj_trim.imag.flatten()[1:]])

        assert (
            tw.norm(fcoef_target - ctn._hyp_list.get_values()) /
            tw.norm(fcoef_target) < 6e-5)

    def test_chebyshev_convolution_operator_1d(self):
        N = 101
        xx = np.linspace(-1, 1, N)
        u = tw.asarray(xx**2)
        v = tw.asarray(1 / (1 + (5*xx)**2))
        u_per = tw.hstack([u, tw.flip(u[1:-1], dims=[0])])
        v_per = tw.hstack([v, tw.flip(v[1:-1], dims=[0])])

        u_tconv_v = fct.circ_conv(u_per, v_per)[:N]

        kmax = N-1
        ctn = CERTANN(N, [Layer(ChebyshevConvolutionOperator(kmax))],
                      [IdentityActivation()])
        training_samples = u[:, None]
        training_values = u_tconv_v[:, None]
        ctn.fit(training_samples, training_values, tol=1e-10)

        assert (
            tw.norm(fct.fct(v)[:kmax+1] - ctn._hyp_list.get_values()) /
            tw.norm(fct.fct(v)[:kmax+1]) < 4e-4)

    def test_chebyshev_convolution_operator_multidim(self):
        N = 21
        xx = np.linspace(-1, 1, N)
        (X, Y) = np.meshgrid(xx, xx)
        u = tw.asarray((X+Y)**2)[..., None]
        v = tw.asarray(1 / (1 + (5*X*Y)**2))[..., None]
        u_per = fct.even_periodic_extension(u)
        v_per = fct.even_periodic_extension(v)
        u_tconv_v = tw.ifft(tw.fft(u_per) * tw.fft(v_per))[:N, :N].real

        kmax = N-1
        fct_v = fct.fct(v)[:kmax+1, :kmax+1, 0]
        v0 = (fct_v.flatten() *
              (1 + tw.asarray(np.random.normal(0, 0.1, ((kmax+1)**2,)))))

        # We do not have enough "quality" (def?) samples to recover fct(v).
        # Set initial iterate with 10% noise until we figure out sampling.
        layers = [Layer(ChebyshevConvolutionOperator(kmax, shape=X.shape,
                                                     v0=v0))]
        ctn = CERTANN(X.size, layers, [IdentityActivation()])
        ctn.fit(u.flatten()[..., None], u_tconv_v.flatten()[..., None],
                tol=1e-8)

        assert (
            tw.norm(fct_v.flatten() - ctn._hyp_list.get_values()) /
            tw.norm(fct_v.flatten()) < 2e-2)

    def test_chebyshev_tensor_product_operator(self):
        # Manufactured integral operator
        def cheb_measure(x):
            _x = x.flatten()
            w = 1.0 / (1e-14+np.sqrt(1-_x**2))
            w[0] = (w[1] + (_x[2] - _x[1]) / (_x[0] - _x[1]) * (w[2] - w[1]))
            w[-1] = w[0]
            return w

        def K(x, y, M):
            Phi_x = fct.chebyshev_poly_basis(tw.asarray(x), nterms).numpy()
            Phi_y = fct.chebyshev_poly_basis(tw.asarray(y), nterms).numpy()
            return np.diag(cheb_measure(x)) @ Phi_x.T @ M @ Phi_y

        def K_int(K, g, xx, M):
            quad_xx, quad_ww = np.polynomial.chebyshev.chebgauss(20)
            Kg = tw.asarray(K(xx, quad_xx, M))*g(quad_xx[None, :])[:, 0]
            return Kg @ quad_ww[:, None]

        # Define A
        nterms = 4
        A_tri = np.random.normal(0, 1, (nterms, nterms))
        A_mat = A_tri + A_tri.T

        # Generate training data
        nfterms = 4

        def parameterized_forc_fun(coef, xx):
            out = ((xx.T**np.arange(len(coef))[None, :]) @ coef)[:, None]
            return out

        level = 5
        nx = 2**level+1
        ntrain_samples = 10
        abscissa = np.cos(np.pi*np.arange(nx)/(nx-1))[None, :]
        kmax = nterms-1
        train_coef = np.random.normal(0, 1, (nfterms, ntrain_samples))
        train_forc_funs = [
            partial(parameterized_forc_fun, coef) for coef in train_coef.T]
        train_samples = np.hstack([f(abscissa) for f in train_forc_funs])
        train_values = np.hstack(
            [K_int(K, f, abscissa, A_mat) for f in train_forc_funs])

        # Fit the network
        ctn = CERTANN(nx, [Layer(ChebyshevIntegralOperator(kmax, chol=False))],
                      [IdentityActivation()])
        ctn.fit(train_samples, train_values, tol=1e-10)

        # Compare upper triangle of A to learned parameters
        A_upper = np.triu(A_mat).flatten()
        A_upper = A_upper[np.abs(A_upper) > 1e-10]
        assert np.allclose(A_upper, ctn._hyp_list.get_values())

    def test_dense_affine_integral_operator(self):
        N0, N1 = 5, 3
        W = tw.asarray(np.random.normal(0, 1, (N1, N0)))
        b = tw.asarray(np.random.normal(0, 1, (N1, 1)))
        XX = tw.asarray(np.random.normal(0, 1, (N0, 20)))
        YY = W @ XX + b
        ctn = CERTANN(N0, [Layer([DenseAffineIntegralOperator(N0, N1)])],
                      [IdentityActivation()])
        ctn.fit(XX, YY, tol=1e-14)
        assert np.allclose(tw.hstack([W, b]).flatten(),
                           ctn._hyp_list.get_values())

        ctn = CERTANN(
            N0, [Layer([DenseAffineIntegralOperator(N0, N1)])],
            [IdentityActivation()],
            optimizer=Adam(epochs=1000, lr=1e-2, batches=5))
        ctn.fit(XX, YY, tol=1e-12)
        assert (tw.norm(tw.hstack([W, b]).flatten() -
                ctn._hyp_list.get_values()) /
                tw.norm(ctn._hyp_list.get_values()) < 5.e-3)

    def test_dense_affine_integral_operator_fixed_bias(self):
        N0, N1 = 3, 5
        XX = tw.asarray(np.random.normal(0, 1, (N0, 20)))
        iop = DenseAffineIntegralOperatorFixedBias(N0, N1)
        b = tw.full((N1, 1), 0)
        W = iop._weights_biases.get_values().reshape(
            iop._noutputs, iop._ninputs+1)[:, :-1]
        YY = W @ XX + b
        assert np.allclose(iop._integrate(XX), YY)
        assert np.allclose(iop._hyp_list.nactive_vars(), N0*N1)


if __name__ == "__main__":
    integral_operators_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestIntegralOperators))
    unittest.TextTestRunner(verbosity=2).run(integral_operators_test_suite)
