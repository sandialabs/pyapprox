import unittest
from functools import partial
import torch
import numpy as np
from pyapprox.sciml.network import CERTANN
from pyapprox.sciml.torchintegraloperators import (
    TorchFourierConvolutionOperator, TorchChebyshevConvolutionOperator,
    TorchDenseAffineIntegralOperator,
    TorchDenseAffineIntegralOperatorFixedBias, TorchChebyshevIntegralOperator,
    TorchKernelIntegralOperator, TorchEmbeddingOperator,
    TorchAffineProjectionOperator, TorchDenseAffinePointwiseOperator,
    TorchDenseAffinePointwiseOperatorFixedBias)
from pyapprox.sciml.util.torchutils import TorchUtilitiesSciML
from pyapprox.sciml.layers import Layer
from pyapprox.sciml.activations import IdentityActivation
from pyapprox.sciml.optimizers import Adam
from pyapprox.surrogates.kernels.torchkernels import TorchMaternKernel
from pyapprox.sciml.quadrature import Fixed1DGaussLegendreIOQuadRule


class TestIntegralOperators(unittest.TestCase, TorchUtilitiesSciML):
    def setUp(self):
        self._la_random_seed(1389745)

    def test_fourier_convolution_operator_1d(self):
        N = 101
        xx = self._la_linspace(-1, 1, N)
        u = xx**2
        v = 1 / (1 + (5*xx)**2)
        v_fft = self._la_fft(v[:, None, None])[:, 0, 0]

        u_conv_v = self._sciml_circ_conv(u, v)

        kmax = (N-1)//2
        ctn = CERTANN(N, [Layer(TorchFourierConvolutionOperator(kmax))],
                      [IdentityActivation()])
        training_samples = u[:, None, None]
        training_values = u_conv_v[:, None, None]
        ctn.fit(training_samples, training_values, tol=1e-12)
        fcoef_target = self._la_hstack([v_fft.real[:kmax+1],
                                        v_fft.imag[1:kmax+1]])

        assert (
            self._la_norm(fcoef_target - ctn._hyp_list.get_values()) /
            self._la_norm(fcoef_target) < 2e-4)

    def test_fourier_convolution_operator_multidim(self):
        N = 101
        xx = self._la_linspace(-1, 1, N)
        (X, Y) = self._la_meshgrid(xx, xx, indexing='xy')
        u = ((X+Y)**2)[..., None, None]
        v = (1 / (1 + (5*X*Y)**2))[..., None, None]

        u_conv_v = self._la_ifft(self._la_fft(u)*self._la_fft(v)).real

        kmax = 10
        layers = [Layer(TorchFourierConvolutionOperator(kmax))]
        ctn = CERTANN(X.shape, layers, [IdentityActivation()])
        ctn.fit(u, u_conv_v, tol=1e-8)

        fftshift_v = self._la_fftshift(self._la_fft(v))
        nyquist = [n//2 for n in X.shape]
        slices = [slice(n-kmax, n+kmax+1) for n in nyquist]
        fftshift_v_proj = fftshift_v[slices].flatten()
        fftshift_v_proj_trim = fftshift_v_proj[fftshift_v_proj.shape[0]//2:]
        fcoef_target = self._la_hstack([fftshift_v_proj_trim.real.flatten(),
                                        fftshift_v_proj_trim.imag.flatten()[1:]
                                        ])

        tol = 4e-6
        relerr = (self._la_norm(fcoef_target - ctn._hyp_list.get_values()) /
                  self._la_norm(fcoef_target))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_chebyshev_convolution_operator_1d(self):
        N = 101
        xx = self._la_linspace(-1, 1, N)
        u = xx**2
        v = 1 / (1 + (5*xx)**2)
        u_per = self._la_hstack([u, self._la_flip(u[1:-1], axis=0)])
        v_per = self._la_hstack([v, self._la_flip(v[1:-1], axis=0)])

        u_tconv_v = self._sciml_circ_conv(u_per, v_per)[:N]

        kmax = N-1
        ctn = CERTANN(N, [Layer(TorchChebyshevConvolutionOperator(kmax))],
                      [IdentityActivation()])
        training_samples = u[:, None, None]
        training_values = u_tconv_v[:, None, None]
        ctn.fit(training_samples, training_values, tol=1e-12)

        tol = 4e-4
        true_coefs = self._sciml_fct(v)[:kmax+1]
        relerr = (self._la_norm(true_coefs - ctn._hyp_list.get_values()) /
                  self._la_norm(true_coefs))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_chebyshev_convolution_operator_multidim(self):
        N = 21
        xx = self._la_linspace(-1, 1, N)
        (X, Y) = self._la_meshgrid(xx, xx, indexing='xy')
        u = ((X+Y)**2)[..., None, None]
        v = 1 / (1 + (5*X*Y)**2)[..., None, None]
        u_per = self._sciml_even_extension(u)
        v_per = self._sciml_even_extension(v)
        u_tconv_v = self._la_ifft(self._la_fft(u_per) *
                                  self._la_fft(v_per))[:N, :N, ...].real
        kmax = N-1
        fct_v = self._sciml_fct(v)[:kmax+1, :kmax+1, ...]
        v0 = (fct_v.flatten() *
              (1 + self._la_normal(0, 0.05, size=((kmax+1)**2,))))

        # We do not have enough "quality" (def?) samples to recover fct(v).
        # Set initial iterate with 10% noise until we figure out sampling.
        layers = [Layer(TorchChebyshevConvolutionOperator(kmax, v0=v0))]
        ctn = CERTANN(X.shape, layers, [IdentityActivation()])
        ctn.fit(u, u_tconv_v, tol=1e-14)

        tol = 2e-2
        relerr = (self._la_norm(fct_v.flatten() - ctn._hyp_list.get_values()) /
                  self._la_norm(fct_v.flatten()))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_chebyshev_tensor_product_operator(self):
        # Manufactured integral operator
        def cheb_measure(x):
            _x = x.flatten()
            w = 1.0 / (1e-14+self._la_sqrt(1-_x**2))
            w[0] = (w[1] + (_x[2] - _x[1]) / (_x[0] - _x[1]) * (w[2] - w[1]))
            w[-1] = w[0]
            return w

        def K(x, y, M):
            Phi_x = self._sciml_chebyshev_poly_basis(x, nterms)
            Phi_y = self._sciml_chebyshev_poly_basis(y, nterms)
            return self._la_diag(cheb_measure(x)) @ Phi_x.T @ M @ Phi_y

        def K_int(K, g, xx, M):
            quad_xx, quad_ww = np.polynomial.chebyshev.chebgauss(20)
            quad_xx = self._la_atleast1d(quad_xx)
            quad_ww = self._la_atleast1d(quad_ww)
            Kg = K(xx, quad_xx, M)*g(quad_xx[None, :])[:, 0]
            return Kg @ quad_ww[:, None]

        # Define A
        nterms = 4
        A_tri = self._la_normal(0, 1, size=(nterms, nterms))
        A_mat = A_tri + A_tri.T

        # Generate training data
        nfterms = 4

        def parameterized_forc_fun(coef, xx):
            out = ((xx.T**self._la_arange(len(coef))[None, :]) @ coef)[:, None]
            return out

        level = 5
        nx = 2**level+1
        ntrain_samples = 10
        abscissa = self._la_cos(self._la_pi()*self._la_arange(nx)/(nx-1))[None, :]
        kmax = nterms-1
        train_coef = self._la_normal(0, 1, size=(nfterms, ntrain_samples))
        train_forc_funs = [
            partial(parameterized_forc_fun, coef) for coef in train_coef.T]
        train_samples = self._la_hstack([f(abscissa) for f in train_forc_funs])
        train_values = self._la_hstack(
            [K_int(K, f, abscissa, A_mat) for f in train_forc_funs])
        train_samples = train_samples[:, None, :]
        train_values = train_values[:, None, :]


        # Fit the network
        ctn = CERTANN(nx, [Layer(TorchChebyshevIntegralOperator(kmax,
                                                                chol=False))],
                      [IdentityActivation()])
        ctn.fit(train_samples, train_values, tol=1e-12)

        # Compare upper triangle of A to learned parameters
        A_upper = np.triu(A_mat).flatten()
        A_upper = A_upper[np.abs(A_upper) > 1e-10]

        tol = 6e-7
        relerr = (np.linalg.norm(A_upper-ctn._hyp_list.get_values().numpy()) /
                  np.linalg.norm(A_upper))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_dense_affine_integral_operator(self):
        N0, N1 = 5, 3
        W = self._la_normal(0, 1, size=(N1, N0))
        b = self._la_normal(0, 1, size=(N1, 1))
        XX = self._la_normal(0, 1, size=(N0, 20))
        YY = W @ XX + b
        XX, YY = XX[:, None, :], YY[:, None, :]
        ctn = CERTANN(N0, [Layer([TorchDenseAffineIntegralOperator(N0, N1)])],
                      [IdentityActivation()])
        ctn.fit(XX, YY, tol=1e-14)
        assert self._la_allclose(self._la_hstack([W.flatten(), b.flatten()]),
                                 ctn._hyp_list.get_values())

        ctn = CERTANN(
            N0, [Layer([TorchDenseAffineIntegralOperator(N0, N1)])],
            [IdentityActivation()],
            optimizer=Adam(epochs=1000, lr=1e-2, batches=5))
        ctn.fit(XX, YY, tol=1e-12)

        tol = 1e-8
        relerr = (self._la_norm(self._la_hstack([W.flatten(), b.flatten()]) -
                                ctn._hyp_list.get_values()) /
                  self._la_norm(self._la_hstack([W.flatten(), b.flatten()])))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_dense_affine_integral_operator_fixed_bias(self):
        N0, N1 = 3, 5
        XX = self._la_normal(0, 1, size=(N0, 20))
        iop = TorchDenseAffineIntegralOperatorFixedBias(N0, N1)
        b = self._la_empty((N1, 1))
        b[...] = 0.0
        W = iop._hyp_list.get_values()[:-N1].reshape(iop._noutputs,
                                                     iop._ninputs)
        YY = W @ XX + b
        XX, YY = XX[:, None, :], YY[:, None, :]
        assert self._la_allclose(iop._integrate(XX), YY), 'Quadrature error'
        assert iop._hyp_list.nactive_vars() == N0*N1, 'Dimension mismatch'

    def test_parameterized_kernels_parallel_channels(self):
        ninputs = 21

        matern_sqexp = TorchMaternKernel(self._la_inf(), [0.2], [0.01, 0.5], 1)
        matern_exp = TorchMaternKernel(0.5, [0.2], [0.01, 0.5], 1)

        # One block, two channels
        quad_rule_k = Fixed1DGaussLegendreIOQuadRule(ninputs)
        quad_rule_kp1 = Fixed1DGaussLegendreIOQuadRule(ninputs)
        iop = TorchKernelIntegralOperator([matern_sqexp, matern_exp],
                                          quad_rule_k, quad_rule_kp1,
                                          channel_in=2, channel_out=2)
        xx = self._la_linspace(0, 1, ninputs)[:, None]
        samples = self._la_hstack([xx, xx])[..., None]
        values = iop(samples)

        # Two blocks, one channel
        iop_sqexp = TorchKernelIntegralOperator([matern_sqexp], quad_rule_k,
                                                quad_rule_kp1, channel_in=1,
                                                channel_out=1)
        iop_exp = TorchKernelIntegralOperator([matern_exp], quad_rule_k,
                                              quad_rule_kp1, channel_in=1,
                                              channel_out=1)

        # Results should be identical
        assert (self._la_allclose(iop_sqexp(xx), values[:, 0]) and
                self._la_allclose(iop_exp(xx), values[:, 1])), (
                'Kernel integral operators not acting on channels in '
                'parallel')

    def test_chebno_channels(self):
        n = 21
        w = self._sciml_make_weights(n)[:, None, None]
        xx = self._la_cos(self._la_pi() *
                          self._la_arange(n)[:, None, None]/(n-1))
        u = self._la_cos(2*self._la_pi()*3.0*xx + 0.5)
        v1 = self._la_normal(0, 1, size=(n, 1, 1))
        v2 = self._la_normal(0, 1, size=(n, 1, 1))
        u_tconv_v1 = self._sciml_ifct(self._sciml_fct(u) * self._sciml_fct(v1)
                                      * 2*(n-1)/w)
        u_tconv_v2 = self._sciml_ifct(self._sciml_fct(u) * self._sciml_fct(v2)
                                      * 2*(n-1)/w)
        values = self._la_concatenate([u_tconv_v1, u_tconv_v2], axis=1)

        kmax = n-1
        channel_in = 1
        channel_out = 2
        layers = [Layer(TorchChebyshevConvolutionOperator(
                            kmax, channel_in=channel_in,
                            channel_out=channel_out))]
        ctn = CERTANN(n, layers, [IdentityActivation()])
        ctn.fit(u, values, tol=1e-10, verbosity=0)

        v0 = self._la_empty(channel_in * channel_out * n)
        v0[::2] = self._sciml_fct(v1).flatten()
        v0[1::2] = self._sciml_fct(v2).flatten()

        tol = 4e-5
        relerr = (self._la_norm(v0 - ctn._hyp_list.get_values()) /
                  self._la_norm(v0))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_fno_channels(self):
        n = 21
        xx = self._la_cos(self._la_pi() *
                          self._la_arange(n)[:, None, None]/(n-1))
        u = self._la_cos(2*self._la_pi()*3.0*xx + 0.5)
        v1 = self._la_normal(0, 1, size=(n, 1, 1))
        v2 = self._la_normal(0, 1, size=(n, 1, 1))
        u_conv_v1 = self._la_ifft(self._la_fft(u) * self._la_fft(v1)).real
        u_conv_v2 = self._la_ifft(self._la_fft(u) * self._la_fft(v2)).real
        values = self._la_concatenate([u_conv_v1, u_conv_v2], axis=1)

        kmax = n//2
        channel_in = 1
        channel_out = 2
        v0 = self._la_empty(channel_in * channel_out * (2*kmax+1))
        v0[:2*(kmax+1):2] = self._la_fft(v1).real[:kmax+1, 0, 0]
        v0[1:2*(kmax+1):2] = self._la_fft(v2).real[:kmax+1, 0, 0]
        v0[2*(kmax+1)::2] = self._la_fft(v1).imag[1:kmax+1, 0, 0]
        v0[2*(kmax+1)+1::2] = self._la_fft(v2).imag[1:kmax+1, 0, 0]

        layers = [Layer(TorchFourierConvolutionOperator(
                            kmax, channel_in=channel_in,
                            channel_out=channel_out))]
        ctn = CERTANN(n, layers, [IdentityActivation()])
        ctn.fit(u, values, tol=1e-9, verbosity=0)

        tol = 6e-7
        relerr = (self._la_norm(v0 - ctn._hyp_list.get_values()) /
                  self._la_norm(v0))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_embedding_operator(self):
        nx = 17
        input_samples = self._la_normal(0, 1, size=(nx, 1, 1))
        quad = Fixed1DGaussLegendreIOQuadRule(17)

        # Same kernel for all output channels
        lenscale = self._la_atleast1d(0.5)
        lenscale_bounds = self._la_atleast1d([1e-5, 10])
        kernel = TorchMaternKernel(nu=0.5, lenscale=lenscale,
                                   lenscale_bounds=lenscale_bounds, nvars=1)
        kio = TorchKernelIntegralOperator(kernel, quad, quad)
        embedding = TorchEmbeddingOperator(kio, channel_in=1, channel_out=10,
                                           nx=nx)
        out = embedding(input_samples)
        assert self._la_allclose(out, kio(input_samples))

        # Channels 1-2 have shared kernel; channels 3-10 have different kernel
        kernel2 = TorchMaternKernel(nu=self._la_inf(), lenscale=lenscale,
                                    lenscale_bounds=lenscale_bounds, nvars=1)
        kio2 = TorchKernelIntegralOperator(kernel2, quad, quad)
        embedding2 = TorchEmbeddingOperator(2*[kio] + 8*[kio2], channel_in=1,
                                            channel_out=10, nx=nx)
        out2 = embedding2(input_samples)
        assert (self._la_allclose(out[:, :2, :], kio(input_samples)) and
                self._la_allclose(out2[:, 2:, :], kio2(input_samples))), (
                'Embedded values do not match corresponding kernels')

        assert not self._la_allclose(out2[:, 2:, :], kio(input_samples)), (
               'In unshared kernel case, channels 3-10 match kernel for '
               'channels 1-2')

    def test_affine_projection_operator(self):
        channel_in = 10
        nx = 17
        input_samples = self._la_normal(0, 1, size=(nx, channel_in, 1))
        v0 = self._la_empty(channel_in+1)
        v0[:] = 1.0
        proj = TorchAffineProjectionOperator(channel_in, v0=v0)
        out = proj(input_samples)
        assert self._la_allclose(out.squeeze(),
                                 input_samples.sum(axis=1).squeeze()+1), (
                'Default affine projection does not match explicit sum')

    def test_dense_affine_pointwise_operator(self):
        channel_in = 2
        channel_out = 5
        nx = 5
        nsamples = 10
        v0 = self._la_normal(0, 1, size=(channel_out*(channel_in+1),))
        op = TorchDenseAffinePointwiseOperator(channel_in=channel_in,
                                               channel_out=channel_out, v0=v0)
        samples = self._la_normal(0, 1, size=(nx, channel_in, nsamples))
        W = v0[:-channel_out].reshape(channel_out, channel_in)
        b = v0[-channel_out:]
        values = (self._la_einsum('ij,...jk->...ik', W, samples) +
                  b[None, ..., None])
        assert self._la_allclose(op(samples), values), (
               'Pointwise affine operator does not match values')

    def test_dense_affine_pointwise_operator_fixed_bias(self):
        channel_in = 2
        channel_out = 5
        nx = 5
        nsamples = 10
        v0 = self._la_normal(0, 1, (channel_out*(channel_in+1),))
        op = TorchDenseAffinePointwiseOperatorFixedBias(
                channel_in=channel_in, channel_out=channel_out, v0=v0)
        samples = self._la_normal(0, 1, (nx, channel_in, nsamples))
        W = v0[:-channel_out].reshape(channel_out, channel_in)
        values = self._la_einsum('ij,...jk->...ik', W, samples)
        assert self._la_allclose(op(samples), values), (
               'Pointwise affine operator with fixed bias does not match ' +
               'values')

    def test_fourier_hilbert_schmidt(self):
        # diagonal channel coupling
        kmax = 4
        d_c = 2
        num_entries = (2*(kmax+1)**2-1)*d_c
        v_float = self._la_normal(0, 1, (num_entries,))
        v = self._la_empty((2*kmax+1, 2*kmax+1, d_c), dtype=self._la_cfloat())
        v[...] = 0.0
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
        v = self._la_transpose(v)
        A = v + self._la_tril(v, diagonal=-1).mH
        Atilde = self._la_tril(self._la_flip(A, axis=-2), diagonal=-1)
        Atilde = self._la_flip(Atilde, axis=-1).conj()
        R = A + Atilde
        R = self._la_transpose(R)
        for k in range(d_c):
            R_H = R[..., k].mH.clone()
            for i in range(2*kmax+1):
                R_H[i, i] = R[i, i, k]
            R_H_conj = self._la_atleast1d(R_H, dtype=self._la_cfloat())
            R_H_conj = R_H_conj.resolve_conj()
            R_k_conj = self._la_atleast1d(R[..., k], dtype=self._la_cfloat())
            R_k_conj = R_k_conj.resolve_conj()
            assert self._la_allclose(R_H_conj, R_k_conj), ('FourierHSOperator:'
                   ' Off-diagonal elements of kernel tensor '
                   + f'are not Hermitian-symmetric for channel {k}')

        y = self._la_normal(0, 1, (2*kmax+1, d_c))[..., None]
        fftshift_y = self._la_fftshift(self._la_fft(y))
        R_fft_y = self._la_einsum('ijk,jkl->ikl', R, fftshift_y)
        out = self._la_ifft(self._la_ifftshift(R_fft_y))
        zeros = self._la_empty((2*kmax+1, d_c))
        zeros[...] = 0.0
        assert self._la_allclose(out.imag.squeeze(), zeros), (
               'FourierHSOperator: Kernel tensor does not maintain conjugate-'
               + 'symmetry of outputs')


if __name__ == "__main__":
    integral_operators_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestIntegralOperators))
    unittest.TextTestRunner(verbosity=2).run(integral_operators_test_suite)
