import unittest
from functools import partial
from pyapprox.sciml.util import FCT, TorchLinAlgMixin
from pyapprox.sciml.network import CERTANN
from pyapprox.sciml.integraloperators import (
    FourierConvolutionOperator, ChebyshevConvolutionOperator,
    DenseAffineIntegralOperator, DenseAffineIntegralOperatorFixedBias,
    ChebyshevIntegralOperator, KernelIntegralOperator, EmbeddingOperator,
    AffineProjectionOperator, DenseAffinePointwiseOperator,
    DenseAffinePointwiseOperatorFixedBias)
from pyapprox.sciml.layers import Layer
from pyapprox.sciml.activations import IdentityActivation
from pyapprox.surrogates.kernels import MaternKernel
from pyapprox.surrogates.bases.orthopoly import (
    GaussLegendreQuadratureRule, Chebyshev1stKindGaussLobattoQuadratureRule)


class TestIntegralOperators(unittest.TestCase):
    def setUp(self):
        self._bkd = TorchLinAlgMixin
        self._bkd.random_seed(1)
        self._fct = FCT(backend=self._bkd)
        self.pi = 3.1415926535897932

    def test_fourier_convolution_operator_1d(self):
        N = 101
        xx = self._bkd.linspace(-1, 1, N)
        u = xx**2
        v = 1.0 / (1.0 + (5*xx)**2)

        u_conv_v = self._fct.circ_conv(u, v)[:, None, None]
        u, v = u[:, None, None], v[:, None, None]

        kmax = (N-1)//2
        ctn = CERTANN(N, [Layer(FourierConvolutionOperator(kmax))],
                      [IdentityActivation()])
        training_samples = u
        training_values = u_conv_v
        ctn.fit(training_samples, training_values, tol=1e-12)
        fcoef_target = self._bkd.hstack([self._bkd.fft(v).real[:kmax+1, 0, 0],
                                         self._bkd.fft(v).imag[1:kmax+1, 0, 0]]
                                        )

        tol = 2e-4
        relerr = (self._bkd.norm(fcoef_target - ctn._hyp_list.get_values()) /
                  self._bkd.norm(fcoef_target))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_fourier_convolution_operator_multidim(self):
        N = 101
        xx = self._bkd.linspace(-1, 1, N)
        (X, Y) = self._bkd.meshgrid(xx, xx)
        u = ((X+Y)**2)[..., None, None]
        v = (1 / (1 + (5*X*Y)**2))[..., None, None]

        u_conv_v = self._bkd.ifft(self._bkd.fft(u)*self._bkd.fft(v)).real

        kmax = 10
        layers = [Layer(FourierConvolutionOperator(kmax, nx=X.shape))]
        ctn = CERTANN(self._bkd.size(X), layers, [IdentityActivation()])
        ctn.fit(u.flatten()[:, None, None], u_conv_v.flatten()[:, None, None],
                tol=1e-8)

        fftshift_v = self._bkd.fftshift(self._bkd.fft(v))
        nyquist = [n//2 for n in X.shape]
        slices = [slice(n-kmax, n+kmax+1) for n in nyquist]
        fftshift_v_proj = self._bkd.get_slices(fftshift_v, slices).flatten()
        fftshift_v_proj_trim = fftshift_v_proj[fftshift_v_proj.shape[0]//2:]
        fcoef_target = self._bkd.hstack(
                            [fftshift_v_proj_trim.real.flatten(),
                             fftshift_v_proj_trim.imag.flatten()[1:]])

        tol = 4e-6
        relerr = (self._bkd.norm(fcoef_target - ctn._hyp_list.get_values()) /
                  self._bkd.norm(fcoef_target))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_chebyshev_convolution_operator_1d(self):
        N = 101
        xx = self._bkd.linspace(-1, 1, N)
        u = xx**2
        v = 1.0 / (1 + (5*xx)**2)
        u_per = self._bkd.hstack([u, self._bkd.flip(u[1:-1], axis=(0,))])
        v_per = self._bkd.hstack([v, self._bkd.flip(v[1:-1], axis=(0,))])
        u_tconv_v = self._fct.circ_conv(u_per, v_per)[:N]

        kmax = N-1
        ctn = CERTANN(N, [Layer(ChebyshevConvolutionOperator(kmax))],
                      [IdentityActivation()])
        training_samples = u[:, None, None]
        training_values = u_tconv_v[:, None, None]
        ctn.fit(training_samples, training_values, tol=1e-12)

        tol = 4e-4
        relerr = (self._bkd.norm(self._fct.fct(v)[:kmax+1] -
                                 ctn._hyp_list.get_values()) /
                  self._bkd.norm(self._fct.fct(v)[:kmax+1]))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_chebyshev_convolution_operator_multidim(self):
        N = 21
        xx = self._bkd.linspace(-1, 1, N)
        (X, Y) = self._bkd.meshgrid(xx, xx)
        u = ((X+Y)**2)[..., None, None]
        v = 1.0 / (1.0 + (5*X*Y)**2)[..., None, None]
        w = self._bkd.normal(0, 1, u.shape)
        u_per = self._fct.even_periodic_extension(u)
        v_per = self._fct.even_periodic_extension(v)
        w_per = self._fct.even_periodic_extension(w)
        u_tconv_v = self._bkd.ifft(self._bkd.fft(u_per)*self._bkd.fft(v_per))
        u_tconv_v = u_tconv_v[:N, :N, 0, 0].real
        w_tconv_v = self._bkd.ifft(self._bkd.fft(w_per)*self._bkd.fft(v_per))
        w_tconv_v = w_tconv_v[:N, :N, 0, 0].real
        kmax = N-1
        fct_v = self._fct.fct(v)[:kmax+1, :kmax+1, 0, 0]
        v0 = (fct_v.flatten() *
              (1 + self._bkd.normal(0, 1.0, ((kmax+1)**2,))))

        # We do not have enough "quality" (def?) samples to recover fct(v).
        # Set initial iterate with 4% noise until we figure out sampling.
        layers = [Layer(ChebyshevConvolutionOperator(kmax, nx=X.shape,
                                                     v0=v0))]
        ctn = CERTANN(self._bkd.size(X), layers, [IdentityActivation()])
        ctn.fit(self._bkd.vstack([u.flatten(), w.flatten()]).T[:, None, :],
                self._bkd.vstack([u_tconv_v.flatten(),
                                  w_tconv_v.flatten()]).T[:, None, :],
                tol=1e-6)

        tol = 7e-3
        relerr = (self._bkd.norm(fct_v.flatten() - ctn._hyp_list.get_values())
                  / self._bkd.norm(fct_v.flatten()))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_chebyshev_tensor_product_operator(self):
        # Manufactured integral operator
        def cheb_measure(x):
            _x = x.flatten()
            w = 1.0 / (1e-14+self._bkd.sqrt(1-_x**2))
            w[0] = (w[1] + (_x[2] - _x[1]) / (_x[0] - _x[1]) * (w[2] - w[1]))
            w[-1] = w[0]
            return w

        def K(x, y, M):
            Phi_x = self._fct.chebyshev_poly_basis(x, nterms)
            Phi_y = self._fct.chebyshev_poly_basis(y, nterms)
            return self._bkd.diag(cheb_measure(x)) @ Phi_x.T @ M @ Phi_y

        def K_int(K, g, xx, M):
            rule = Chebyshev1stKindGaussLobattoQuadratureRule(
                        bounds=[-1., 1.],
                        backend=self._bkd)
            rule.set_nnodes(20)
            quad_xx, quad_ww = rule()
            Kg = K(xx, quad_xx, M)*(g(quad_xx)[:, 0])
            return Kg @ quad_ww

        # Define A
        nterms = 4
        A_tri = self._bkd.normal(0, 1, (nterms, nterms))
        A_mat = A_tri + A_tri.T

        # Generate training data
        nfterms = 4

        def parameterized_forc_fun(coef, xx):
            out = (xx.T ** self._bkd.arange(len(coef))[None, :]) @ coef
            return out[:, None]

        level = 5
        nx = 2**level+1
        ntrain_samples = 10
        abscissa = self._bkd.cos(
            self.pi*self._bkd.arange(nx, dtype=float)/(nx-1))[None, :]
        kmax = nterms-1
        train_coef = self._bkd.normal(0, 1, (nfterms, ntrain_samples))
        train_forc_funs = [
            partial(parameterized_forc_fun, coef) for coef in train_coef.T]
        train_samples = self._bkd.hstack([f(abscissa)
                                          for f in train_forc_funs])
        train_values = self._bkd.hstack(
            [K_int(K, f, abscissa, A_mat) for f in train_forc_funs])

        # Fit the network
        ctn = CERTANN(nx, [Layer(ChebyshevIntegralOperator(kmax, chol=False))],
                      [IdentityActivation()])
        ctn.fit(train_samples[:, None, :], train_values[:, None, :], tol=1e-8)

        # Compare upper triangle of A to learned parameters
        A_upper = self._bkd.triu(A_mat).flatten()
        A_upper = A_upper[self._bkd.abs(A_upper) > 1e-10]

        tol = 1e-5
        relerr = (self._bkd.norm(A_upper-ctn._hyp_list.get_values()) /
                  self._bkd.norm(A_upper))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_dense_affine_integral_operator(self):
        N0, N1 = 5, 3
        W = self._bkd.normal(0, 1, (N1, N0))
        b = self._bkd.normal(0, 1, (N1, 1))
        XX = self._bkd.normal(0, 1, (N0, 20))
        YY = W @ XX + b
        XX, YY = XX[:, None, :], YY[:, None, :]
        ctn = CERTANN(N0, [Layer([DenseAffineIntegralOperator(N0, N1)])],
                      [IdentityActivation()])
        ctn.fit(XX, YY, tol=1e-14)
        true_weights = self._bkd.hstack([W.flatten(), b.flatten()])
        network_weights = self._bkd.atleast1d(ctn._hyp_list.get_values(),
                                              dtype=XX.dtype)
        assert self._bkd.allclose(true_weights, network_weights), (
                                  'Indexing mismatch in network weight '
                                  'initialization')

        ctn = CERTANN(
            N0, [Layer([DenseAffineIntegralOperator(N0, N1)])],
            [IdentityActivation()])
        ctn.fit(XX, YY, tol=1e-10)

        tol = 1e-6
        relerr = (self._bkd.norm(self._bkd.hstack([W.flatten(), b.flatten()]) -
                                 ctn._hyp_list.get_values()) /
                  self._bkd.norm(self._bkd.hstack([W.flatten(), b.flatten()])))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_dense_affine_integral_operator_fixed_bias(self):
        N0, N1 = 3, 5
        XX = self._bkd.normal(0, 1, (N0, 20))
        iop = DenseAffineIntegralOperatorFixedBias(N0, N1)
        b = self._bkd.zeros((N1, 1))
        W = iop._weights_biases.get_values()[:-N1].reshape(iop._noutputs,
                                                           iop._ninputs)
        YY = W @ XX + b
        XX, YY = XX[:, None, :], YY[:, None, :]
        assert self._bkd.allclose(iop._integrate(XX), YY), 'Quadrature error'
        assert (iop._hyp_list.nactive_vars() == N0*N1), 'Dimension mismatch'

    def test_parameterized_kernels_parallel_channels(self):
        ninputs = 21

        matern_sqexp = MaternKernel(self._bkd.inf(), [0.2], [0.01, 0.5], 1,
                                    backend=self._bkd)
        matern_exp = MaternKernel(0.5, [0.2], [0.01, 0.5], 1,
                                  backend=self._bkd)

        # One block, two channels
        quad_rule_k = GaussLegendreQuadratureRule([-1., 1.], backend=self._bkd)
        quad_rule_kp1 = GaussLegendreQuadratureRule([-1., 1.],
                                                    backend=self._bkd)
        quad_rule_k.set_nnodes(ninputs)
        quad_rule_kp1.set_nnodes(ninputs)
        iop = KernelIntegralOperator([matern_sqexp, matern_exp], quad_rule_k,
                                     quad_rule_kp1, channel_in=2,
                                     channel_out=2)
        xx = self._bkd.linspace(0, 1, ninputs)[:, None]
        samples = self._bkd.hstack([xx, xx])[..., None]
        values = iop(samples)

        # Two blocks, one channel
        iop_sqexp = KernelIntegralOperator([matern_sqexp], quad_rule_k,
                                           quad_rule_kp1, channel_in=1,
                                           channel_out=1)
        iop_exp = KernelIntegralOperator([matern_exp], quad_rule_k,
                                         quad_rule_kp1, channel_in=1,
                                         channel_out=1)

        # Results should be identical
        assert (self._bkd.allclose(iop_sqexp(xx), values[:, 0]) and
                self._bkd.allclose(iop_exp(xx), values[:, 1])), (
                'Kernel integral operators not acting on channels in '
                'parallel')

    def test_chebno_channels(self):
        n = 21
        w = self._fct.make_weights(n)[:, None]
        xx = self._bkd.cos(self.pi*self._bkd.arange(n)/(n-1))
        u = self._bkd.cos(2*self.pi*3.0*xx + 0.5)[:, None]
        v1 = self._bkd.normal(0, 1, (n,))[:, None]
        v2 = self._bkd.normal(0, 1, (n,))[:, None]
        u_tconv_v1 = self._fct.ifct(self._fct.fct(u) * self._fct.fct(v1) *
                                    2*(n-1)/w)
        u_tconv_v2 = self._fct.ifct(self._fct.fct(u) * self._fct.fct(v2) *
                                    2*(n-1)/w)
        samples = u[..., None]
        values = self._bkd.hstack([u_tconv_v1, u_tconv_v2])[..., None]

        kmax = n-1
        channel_in = 1
        channel_out = 2
        v0 = self._bkd.zeros((channel_in * channel_out * n,))
        v0[::2] = self._fct.fct(v1).flatten()
        v0[1::2] = self._fct.fct(v2).flatten()
        layers = [Layer(ChebyshevConvolutionOperator(kmax, nx=n,
                                                     channel_in=channel_in,
                                                     channel_out=channel_out))]
        ctn = CERTANN(n, layers, [IdentityActivation()])
        ctn.fit(samples, values, tol=1e-10, verbosity=0)

        tol = 2e-5
        relerr = (self._bkd.norm(v0 - ctn._hyp_list.get_values()) /
                  self._bkd.norm(v0))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_fno_channels(self):
        n = 21
        xx = self._bkd.cos(self.pi*self._bkd.arange(n)/(n-1))
        u = self._bkd.cos(2*self.pi*3.0*xx + 0.5)[:, None, None]
        v1 = self._bkd.normal(0, 1, (n,))[:, None, None]
        v2 = self._bkd.normal(0, 1, (n,))[:, None, None]
        u_conv_v1 = self._bkd.ifft(self._bkd.fft(u) * self._bkd.fft(v1)).real
        u_conv_v2 = self._bkd.ifft(self._bkd.fft(u) * self._bkd.fft(v2)).real
        samples = u
        values = self._bkd.concatenate([u_conv_v1, u_conv_v2], axis=1)

        kmax = n//2
        channel_in = 1
        channel_out = 2
        v0 = self._bkd.zeros((channel_in * channel_out * (2*kmax+1),))
        v0[:2*(kmax+1):2] = self._bkd.fft(v1).real[:kmax+1, 0, 0]
        v0[1:2*(kmax+1):2] = self._bkd.fft(v2).real[:kmax+1, 0, 0]
        v0[2*(kmax+1)::2] = self._bkd.fft(v1).imag[1:kmax+1, 0, 0]
        v0[2*(kmax+1)+1::2] = self._bkd.fft(v2).imag[1:kmax+1, 0, 0]

        layers = [Layer(FourierConvolutionOperator(kmax, nx=n,
                                                   channel_in=channel_in,
                                                   channel_out=channel_out))]
        ctn = CERTANN(n, layers, [IdentityActivation()])
        ctn.fit(samples, values, tol=1e-8, verbosity=0)

        tol = 3e-6
        relerr = (self._bkd.norm(v0 - ctn._hyp_list.get_values()) /
                  self._bkd.norm(v0))
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_embedding_operator(self):
        nx = 17
        input_samples = self._bkd.normal(0, 1, (nx, 1, 1))
        quad = GaussLegendreQuadratureRule([-1.0, 1.0], backend=self._bkd)
        quad.set_nnodes(17)

        # Same kernel for all output channels
        lenscale = self._bkd.atleast1d([0.5])
        lenscale_bounds = self._bkd.atleast1d([1e-5, 10])
        kernel = MaternKernel(nu=0.5, lenscale=lenscale,
                              lenscale_bounds=lenscale_bounds, nvars=1,
                              backend=self._bkd)
        kio = KernelIntegralOperator(kernel, quad, quad)
        embedding = EmbeddingOperator(kio, channel_in=1, channel_out=10,
                                      nx=nx)
        out = embedding(input_samples)
        assert self._bkd.allclose(out, kio(input_samples))

        # Channels 1-2 have shared kernel; channels 3-10 have different kernel
        kernel2 = MaternKernel(nu=self._bkd.inf(), lenscale=lenscale,
                               lenscale_bounds=lenscale_bounds, nvars=1,
                               backend=self._bkd)
        kio2 = KernelIntegralOperator(kernel2, quad, quad)
        embedding2 = EmbeddingOperator(2*[kio] + 8*[kio2], channel_in=1,
                                       channel_out=10, nx=nx)
        out2 = embedding2(input_samples)
        assert (self._bkd.allclose(out[:, :2, :], kio(input_samples)) and
                self._bkd.allclose(out2[:, 2:, :], kio2(input_samples))), (
                'Embedded values do not match corresponding kernels')

        assert not self._bkd.allclose(out2[:, 2:, :], kio(input_samples)), (
               'In unshared kernel case, channels 3-10 match kernel for '
               'channels 1-2')

    def test_affine_projection_operator(self):
        channel_in = 10
        nx = 17
        input_samples = self._bkd.tile(self._bkd.normal(0, 1, (nx,)),
                                       (channel_in, 1)).T
        v0 = self._bkd.ones((channel_in + 1, ))
        v0[-1] = 1
        proj = AffineProjectionOperator(channel_in, v0=v0, nx=nx)
        out = proj(input_samples[..., None])
        assert self._bkd.allclose(out.squeeze(),
                                  input_samples.sum(axis=1) + 1), (
               'Default affine projection does not match explicit sum')

    def test_dense_affine_pointwise_operator(self):
        channel_in = 2
        channel_out = 5
        nx = 5
        nsamples = 10
        v0 = self._bkd.normal(0, 1, (channel_out*(channel_in+1),))
        op = DenseAffinePointwiseOperator(channel_in=channel_in,
                                          channel_out=channel_out, v0=v0)
        samples = self._bkd.normal(0, 1, (nx, channel_in, nsamples))
        W = self._bkd.reshape(v0[:-channel_out], (channel_out, channel_in))
        b = self._bkd.reshape(v0[-channel_out:], (channel_out,))
        values = (self._bkd.einsum('ij,...jk->...ik', W, samples) +
                  b[None, ..., None])
        assert self._bkd.allclose(op(samples), values), (
               'Pointwise affine operator does not match values')

    def test_dense_affine_pointwise_operator_fixed_bias(self):
        channel_in = 2
        channel_out = 5
        nx = 5
        nsamples = 10
        v0 = self._bkd.normal(0, 1, (channel_out*(channel_in+1),))
        op = DenseAffinePointwiseOperatorFixedBias(channel_in=channel_in,
                                                   channel_out=channel_out,
                                                   v0=v0)
        samples = self._bkd.normal(0, 1, (nx, channel_in, nsamples))
        W = self._bkd.reshape(v0[:-channel_out], (channel_out, channel_in))
        values = self._bkd.einsum('ij,...jk->...ik', W, samples)
        assert self._bkd.allclose(op(samples), values), (
               'Pointwise affine operator with fixed bias does not match ' +
               'values')

    def test_fourier_hilbert_schmidt(self):
        # diagonal channel coupling
        kmax = 4
        d_c = 2
        num_entries = (2*(kmax+1)**2-1)*d_c
        v_float = self._bkd.normal(0, 1, (num_entries,))
        v = self._bkd.zeros((2*kmax+1, 2*kmax+1, d_c),
                            dtype=self._bkd.cfloat())
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
        v = self._bkd.transpose(v)
        A = v + self._bkd.tril(v, k=-1).mH
        Atilde = self._bkd.tril(self._bkd.flip(A, axis=(-2,)), k=-1)
        Atilde = self._bkd.flip(Atilde, axis=(-1,)).conj()
        R = A + Atilde
        R = self._bkd.transpose(R)
        for k in range(d_c):
            R_H = R[..., k].mH.clone()
            for i in range(2*kmax+1):
                R_H[i, i] = R[i, i, k]
            assert self._bkd.allclose(R_H, R[..., k]), (
                   'FourierHSOperator: Off-diagonal elements of kernel tensor '
                   + 'are not Hermitian-symmetric')

        y = self._bkd.normal(0, 1, (2*kmax+1, d_c))[..., None]
        fftshift_y = self._bkd.fftshift(self._bkd.fft(y))
        R_fft_y = self._bkd.einsum('ijk,jkl->ikl', R, fftshift_y)
        out = self._bkd.ifft(self._bkd.ifftshift(R_fft_y))
        assert self._bkd.allclose(out.imag.squeeze(),
                                  self._bkd.zeros((2*kmax+1, d_c))), (
               'FourierHSOperator: Kernel tensor does not maintain conjugate-'
               + 'symmetry of outputs')


if __name__ == "__main__":
    integral_operators_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestIntegralOperators))
    unittest.TextTestRunner(verbosity=2).run(integral_operators_test_suite)
