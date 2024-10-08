import unittest
from functools import partial
import numpy as np
from pyapprox.sciml.util import fct
from pyapprox.sciml.util import _torch_wrappers as tw
import torch
from pyapprox.sciml.network import CERTANN
from pyapprox.sciml.integraloperators import (
    FourierConvolutionOperator, ChebyshevConvolutionOperator,
    DenseAffineIntegralOperator, DenseAffineIntegralOperatorFixedBias,
    ChebyshevIntegralOperator, KernelIntegralOperator, EmbeddingOperator,
    AffineProjectionOperator, DenseAffinePointwiseOperator,
    DenseAffinePointwiseOperatorFixedBias)
from pyapprox.sciml.layers import Layer
from pyapprox.sciml.activations import IdentityActivation
from pyapprox.sciml.optimizers import Adam
from pyapprox.sciml.kernels import MaternKernel
from pyapprox.sciml.quadrature import Fixed1DGaussLegendreIOQuadRule


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
        ctn.fit(training_samples, training_values, tol=1e-12)
        fcoef_target = tw.hstack([tw.fft(v).real[:kmax+1],
                                  tw.fft(v).imag[1:kmax+1]])

        assert (
            tw.norm(fcoef_target - ctn._hyp_list.get_values()) /
            tw.norm(fcoef_target) < 2e-4)

    def test_fourier_convolution_operator_multidim(self):
        N = 101
        xx = np.linspace(-1, 1, N)
        (X, Y) = np.meshgrid(xx, xx)
        u = tw.asarray((X+Y)**2)[..., None]
        v = tw.asarray(1 / (1 + (5*X*Y)**2))[..., None]

        u_conv_v = tw.ifft(tw.fft(u)*tw.fft(v)).real

        kmax = 10
        layers = [Layer(FourierConvolutionOperator(kmax, nx=X.shape))]
        ctn = CERTANN(X.size, layers, [IdentityActivation()])
        ctn.fit(u.flatten()[:, None, None], u_conv_v.flatten()[:, None, None],
                tol=1e-8)

        fftshift_v = tw.fftshift(tw.fft(v))
        nyquist = [n//2 for n in X.shape]
        slices = [slice(n-kmax, n+kmax+1) for n in nyquist]
        fftshift_v_proj = fftshift_v[slices].flatten()
        fftshift_v_proj_trim = fftshift_v_proj[fftshift_v_proj.shape[0]//2:]
        fcoef_target = tw.hstack([fftshift_v_proj_trim.real.flatten(),
                                  fftshift_v_proj_trim.imag.flatten()[1:]])

        tol = 4e-6
        relerr = (tw.norm(fcoef_target - ctn._hyp_list.get_values()) /
                  tw.norm(fcoef_target))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

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
        ctn.fit(training_samples, training_values, tol=1e-12)

        tol = 4e-4
        relerr = (tw.norm(fct.fct(v)[:kmax+1] - ctn._hyp_list.get_values()) /
                  tw.norm(fct.fct(v)[:kmax+1]))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_chebyshev_convolution_operator_multidim(self):
        N = 21
        xx = np.linspace(-1, 1, N)
        (X, Y) = np.meshgrid(xx, xx)
        u = tw.asarray((X+Y)**2)[..., None, None]
        v = tw.asarray(1 / (1 + (5*X*Y)**2))[..., None, None]
        u_per = fct.even_periodic_extension(u)
        v_per = fct.even_periodic_extension(v)
        u_tconv_v = tw.ifft(tw.fft(u_per) * tw.fft(v_per))[:N, :N, 0].real
        kmax = N-1
        fct_v = fct.fct(v)[:kmax+1, :kmax+1, 0]
        v0 = (fct_v.flatten() *
              (1 + tw.asarray(np.random.normal(0, 0.1, ((kmax+1)**2,)))))

        # We do not have enough "quality" (def?) samples to recover fct(v).
        # Set initial iterate with 10% noise until we figure out sampling.
        layers = [Layer(ChebyshevConvolutionOperator(kmax, nx=X.shape,
                                                     v0=v0))]
        ctn = CERTANN(X.size, layers, [IdentityActivation()])
        ctn.fit(u.flatten()[..., None], u_tconv_v.flatten()[..., None],
                tol=1e-10)

        tol = 2e-2
        relerr = (tw.norm(fct_v.flatten() - ctn._hyp_list.get_values()) /
                  tw.norm(fct_v.flatten()))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

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

        tol = 6e-7
        relerr = (np.linalg.norm(A_upper-ctn._hyp_list.get_values().numpy()) /
                  np.linalg.norm(A_upper))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_dense_affine_integral_operator(self):
        N0, N1 = 5, 3
        W = tw.asarray(np.random.normal(0, 1, (N1, N0)))
        b = tw.asarray(np.random.normal(0, 1, (N1, 1)))
        XX = tw.asarray(np.random.normal(0, 1, (N0, 20)))
        YY = W @ XX + b
        XX, YY = XX[:, None, :], YY[:, None, :]
        ctn = CERTANN(N0, [Layer([DenseAffineIntegralOperator(N0, N1)])],
                      [IdentityActivation()])
        ctn.fit(XX, YY, tol=1e-14)
        assert np.allclose(tw.hstack([W.flatten(), b.flatten()]),
                           ctn._hyp_list.get_values())

        ctn = CERTANN(
            N0, [Layer([DenseAffineIntegralOperator(N0, N1)])],
            [IdentityActivation()],
            optimizer=Adam(epochs=1000, lr=1e-2, batches=5))
        ctn.fit(XX, YY, tol=1e-12)

        tol = 1e-8
        relerr = (tw.norm(tw.hstack([W.flatten(), b.flatten()]) -
                          ctn._hyp_list.get_values()) /
                  tw.norm(tw.hstack([W.flatten(), b.flatten()])))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_dense_affine_integral_operator_fixed_bias(self):
        N0, N1 = 3, 5
        XX = tw.asarray(np.random.normal(0, 1, (N0, 20)))
        iop = DenseAffineIntegralOperatorFixedBias(N0, N1)
        b = tw.full((N1, 1), 0)
        W = iop._weights_biases.get_values()[:-N1].reshape(iop._noutputs,
                                                           iop._ninputs)
        YY = W @ XX + b
        XX, YY = XX[:, None, :], YY[:, None, :]
        assert np.allclose(iop._integrate(XX), YY), 'Quadrature error'
        assert np.allclose(iop._hyp_list.nactive_vars(), N0*N1), ('Dimension '
               'mismatch')

    def test_parameterized_kernels_parallel_channels(self):
        ninputs = 21

        matern_sqexp = MaternKernel(tw.inf, [0.2], [0.01, 0.5], 1)
        matern_exp = MaternKernel(0.5, [0.2], [0.01, 0.5], 1)

        # One block, two channels
        quad_rule_k = Fixed1DGaussLegendreIOQuadRule(ninputs)
        quad_rule_kp1 = Fixed1DGaussLegendreIOQuadRule(ninputs)
        iop = KernelIntegralOperator([matern_sqexp, matern_exp], quad_rule_k,
                                     quad_rule_kp1, channel_in=2,
                                     channel_out=2)
        xx = tw.asarray(np.linspace(0, 1, ninputs))[:, None]
        samples = tw.hstack([xx, xx])[..., None]
        values = iop(samples)

        # Two blocks, one channel
        iop_sqexp = KernelIntegralOperator([matern_sqexp], quad_rule_k,
                                           quad_rule_kp1, channel_in=1,
                                           channel_out=1)
        iop_exp = KernelIntegralOperator([matern_exp], quad_rule_k,
                                         quad_rule_kp1, channel_in=1,
                                         channel_out=1)

        # Results should be identical
        assert (np.allclose(iop_sqexp(xx), values[:, 0]) and
                np.allclose(iop_exp(xx), values[:, 1])), (
                'Kernel integral operators not acting on channels in '
                'parallel')

    def test_chebno_channels(self):
        n = 21
        w = fct.make_weights(n)[:, None]
        xx = np.cos(np.pi*np.arange(n)/(n-1))
        u = tw.asarray(np.cos(2*np.pi*3.0*xx + 0.5))[:, None]
        v1 = tw.asarray(np.random.normal(0, 1, (n,)))[:, None]
        v2 = tw.asarray(np.random.normal(0, 1, (n,)))[:, None]
        u_tconv_v1 = fct.ifct(fct.fct(u) * fct.fct(v1) * 2*(n-1)/w)
        u_tconv_v2 = fct.ifct(fct.fct(u) * fct.fct(v2) * 2*(n-1)/w)
        samples = u[..., None]
        values = tw.hstack([u_tconv_v1, u_tconv_v2])[..., None]

        kmax = n-1
        channel_in = 1
        channel_out = 2
        v0 = tw.zeros(channel_in * channel_out * n)
        v0[::2] = fct.fct(v1).flatten()
        v0[1::2] = fct.fct(v2).flatten()
        layers = [Layer(ChebyshevConvolutionOperator(kmax, nx=n,
                                                     channel_in=channel_in,
                                                     channel_out=channel_out))]
        ctn = CERTANN(n, layers, [IdentityActivation()])
        ctn.fit(samples, values, tol=1e-10, verbosity=0)

        tol = 4e-5
        relerr = (np.linalg.norm(v0 - ctn._hyp_list.get_values()) /
                  np.linalg.norm(v0))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_fno_channels(self):
        n = 21
        xx = np.cos(np.pi*np.arange(n)/(n-1))
        u = tw.asarray(np.cos(2*np.pi*3.0*xx + 0.5))
        v1 = tw.asarray(np.random.normal(0, 1, (n,)))
        v2 = tw.asarray(np.random.normal(0, 1, (n,)))
        u_conv_v1 = tw.ifft(tw.fft(u) * tw.fft(v1)).real
        u_conv_v2 = tw.ifft(tw.fft(u) * tw.fft(v2)).real
        samples = u[:, None, None]
        values = tw.hstack([u_conv_v1[:, None], u_conv_v2[:, None]])[..., None]

        kmax = n//2
        channel_in = 1
        channel_out = 2
        v0 = tw.zeros(channel_in * channel_out * (2*kmax+1))
        v0[:2*(kmax+1):2] = tw.fft(v1).real[:kmax+1]
        v0[1:2*(kmax+1):2] = tw.fft(v2).real[:kmax+1]
        v0[2*(kmax+1)::2] = tw.fft(v1).imag[1:kmax+1]
        v0[2*(kmax+1)+1::2] = tw.fft(v2).imag[1:kmax+1]

        layers = [Layer(FourierConvolutionOperator(kmax, nx=n,
                                                   channel_in=channel_in,
                                                   channel_out=channel_out))]
        ctn = CERTANN(n, layers, [IdentityActivation()])
        ctn.fit(samples, values, tol=1e-8, verbosity=0)

        tol = 6e-7
        relerr = (np.linalg.norm(v0 - ctn._hyp_list.get_values()) /
                  np.linalg.norm(v0))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_embedding_operator(self):
        nx = 17
        input_samples = tw.asarray(np.random.normal(0, 1, nx))[:, None, None]
        quad = Fixed1DGaussLegendreIOQuadRule(17)

        # Same kernel for all output channels
        lenscale = tw.asarray(np.asarray([0.5]))
        lenscale_bounds = tw.asarray(np.asarray([1e-5, 10]))
        kernel = MaternKernel(nu=0.5, lenscale=lenscale,
                              lenscale_bounds=lenscale_bounds, nvars=1)
        kio = KernelIntegralOperator(kernel, quad, quad)
        embedding = EmbeddingOperator(kio, channel_in=1, channel_out=10,
                                      nx=nx)
        out = embedding(input_samples)
        assert np.allclose(out, kio(input_samples))

        # Channels 1-2 have shared kernel; channels 3-10 have different kernel
        kernel2 = MaternKernel(nu=np.inf, lenscale=lenscale,
                               lenscale_bounds=lenscale_bounds, nvars=1)
        kio2 = KernelIntegralOperator(kernel2, quad, quad)
        embedding2 = EmbeddingOperator(2*[kio] + 8*[kio2], channel_in=1,
                                       channel_out=10, nx=nx)
        out2 = embedding2(input_samples)
        assert (np.allclose(out[:, :2, :], kio(input_samples)) and
                np.allclose(out2[:, 2:, :], kio2(input_samples))), (
                'Embedded values do not match corresponding kernels')

        assert not np.allclose(out2[:, 2:, :], kio(input_samples)), (
               'In unshared kernel case, channels 3-10 match kernel for '
               'channels 1-2')

    def test_affine_projection_operator(self):
        channel_in = 10
        nx = 17
        input_samples = np.tile(np.random.normal(0, 1, nx), (channel_in, 1)).T
        v0 = np.ones(channel_in + 1)
        v0[-1] = 1
        proj = AffineProjectionOperator(channel_in, v0=v0, nx=nx)
        out = proj(tw.asarray(input_samples)[..., None])
        assert np.allclose(out.squeeze(), input_samples.sum(axis=1)+1), (
               'Default affine projection does not match explicit sum')

    def test_dense_affine_pointwise_operator(self):
        channel_in = 2
        channel_out = 5
        nx = 5
        nsamples = 10
        v0 = np.random.normal(0, 1, (channel_out*(channel_in+1),))
        op = DenseAffinePointwiseOperator(channel_in=channel_in,
                                          channel_out=channel_out, v0=v0)
        samples = tw.asarray(np.random.normal(0, 1,
                                              (nx, channel_in, nsamples)))
        W = tw.asarray(np.reshape(v0[:-channel_out],
                                  (channel_out, channel_in)))
        b = tw.asarray(np.reshape(v0[-channel_out:], (channel_out,)))
        values = tw.einsum('ij,...jk->...ik', W, samples) + b[None, ..., None]
        assert np.allclose(op(samples), values), (
               'Pointwise affine operator does not match values')

    def test_dense_affine_pointwise_operator_fixed_bias(self):
        channel_in = 2
        channel_out = 5
        nx = 5
        nsamples = 10
        v0 = np.random.normal(0, 1, (channel_out*(channel_in+1),))
        op = DenseAffinePointwiseOperatorFixedBias(channel_in=channel_in,
                                                   channel_out=channel_out,
                                                   v0=v0)
        samples = tw.asarray(np.random.normal(0, 1,
                                              (nx, channel_in, nsamples)))
        W = tw.asarray(np.reshape(v0[:-channel_out],
                                  (channel_out, channel_in)))
        values = tw.einsum('ij,...jk->...ik', W, samples)
        assert np.allclose(op(samples), values), (
               'Pointwise affine operator with fixed bias does not match ' +
               'values')

    def test_fourier_hilbert_schmidt(self):
        # diagonal channel coupling
        kmax = 4
        d_c = 2
        num_entries = (2*(kmax+1)**2-1)*d_c
        v_float = tw.asarray(np.random.normal(0, 1, (num_entries,)))
        v = tw.zeros((2*kmax+1, 2*kmax+1, d_c), dtype=tw.cfloat)
        start = 0
        for i in range(kmax+1):
            stride = (2*kmax+1 - 2*i)*d_c
            cols = slice(i, 2*kmax+1-i)
            v[i, cols, ...].real.flatten()[:] = v_float[start:start+stride]
            if i < kmax:
                v[i, cols, ...].imag.flatten()[:] = v_float[start + stride:
                                                            start + 2*stride]
            start += 2*stride

        # Take Hermitian transpose in first two dimensions; torch operates on
        # last two dimensions by default
        v = tw.permute(v, list(range(v.ndim-1, -1, -1)))
        A = v + tw.tril(v, diagonal=-1).mH
        Atilde = tw.tril(tw.flip(A, dims=[-2]), diagonal=-1)
        Atilde = tw.conj(tw.flip(Atilde, dims=[-1]))
        R = A + Atilde
        R = tw.permute(R, list(range(R.ndim-1, -1, -1)))
        for k in range(d_c):
            R_H = R[..., k].mH.clone()
            for i in range(2*kmax+1):
                R_H[i, i] = R[i, i, k]
            assert np.allclose(R_H.resolve_conj(), R[..., k].resolve_conj()), (
                   'FourierHSOperator: Off-diagonal elements of kernel tensor '
                   + 'are not Hermitian-symmetric')

        y = tw.asarray(np.random.normal(0, 1, (2*kmax+1, d_c)))[..., None]
        fftshift_y = tw.fftshift(tw.fft(y))
        R_fft_y = tw.einsum('ijk,jkl->ikl', R, fftshift_y)
        out = tw.ifft(tw.ifftshift(R_fft_y))
        assert np.allclose(out.imag.squeeze(), np.zeros((2*kmax+1, d_c))), (
               'FourierHSOperator: Kernel tensor does not maintain conjugate-'
               + 'symmetry of outputs')


if __name__ == "__main__":
    integral_operators_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestIntegralOperators))
    unittest.TextTestRunner(verbosity=2).run(integral_operators_test_suite)
