import unittest

import numpy as np
import torch

from pyapprox.sciml.kernels import MaternKernel, ConstantKernel
from pyapprox.sciml.integraloperators import (
    KernelIntegralOperator, DenseAffineIntegralOperator,
    FourierConvolutionOperator, ChebyshevConvolutionOperator)
from pyapprox.sciml.quadrature import Fixed1DGaussLegendreIOQuadRule
from pyapprox.sciml.activations import TanhActivation, IdentityActivation
from pyapprox.sciml.network import CERTANN
from pyapprox.sciml.util.hyperparameter import LogHyperParameterTransform
from pyapprox.sciml.layers import Layer


def smooth_fun(xx):
    assert xx.ndim == 2
    return -(xx*np.cos(4*np.pi*xx))


def nonsmooth_fun(xx):
    assert xx.ndim == 2
    return -(np.max(np.zeros(xx.shape), np.cos(4*np.pi*xx)))


def sqinv_elliptic_prior_samples(ninputs, nsamples=1):
    np.random.seed(1)
    dx = 2.0/(ninputs-1)
    M = 4.0*np.eye(ninputs)
    M[0, 0] = 2.0
    M[-1, -1] = 2.0
    for i in range(0, ninputs-1):
        M[i, i+1] = 1.0
        M[i+1, i] = 1.0
    M = (dx/6.0)*M

    S = 2.0*np.eye(ninputs)
    S[0, 0] = 1.0
    S[-1, -1] = 1.0
    for i in range(0, ninputs-1):
        S[i, i+1] = -1.0
        S[i+1, i] = -1.0
    S = (1.0/dx)*S
    E = (3.e-1) * S + M
    Z = np.random.normal(0, 1, (ninputs, nsamples))
    samples = np.linalg.solve(E, Z)
    return samples


class TestSingleLayerCERTANN(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)

    @unittest.skip("Broken")
    def test_matern(self):
        nvars = 1   # dimension of inputs
        noutputs = 1  # dimension of outputs
        nu = np.inf  # matern smoothness parameter
        nquad = 20  # number of quadrature points used for each layer
        nlayers = 3
        output_kernel = (
            ConstantKernel(1.0, [np.nan, np.nan], LogHyperParameterTransform())
            * MaternKernel(nu, [0.1], [1e-2, 10], nvars))

        kernels = [MaternKernel(nu, [0.1], [1e-2, 1], nvars)
                   for ii in range(nlayers-1)]+[output_kernel]
        mat = np.random.normal(0, 1, (nquad, nvars))
        quad_rules = (
            [Fixed1DGaussLegendreIOQuadRule(
                nvars if isinstance(lift, NoLift) else lift._noutputs)] +
            [Fixed1DGaussLegendreIOQuadRule(nquad)
             for kl in range(nlayers-1)] +
            [Fixed1DGaussLegendreIOQuadRule(noutputs)])
        integral_ops = (
            [KernelIntegralOperator(
                kernels[kk], quad_rules[kk], quad_rules[kk+1])
             for kk in range(len(kernels))])
        activations = (
            [TanhActivation() for ii in range(nlayers-1)] +
            [IdentityActivation()])
        ctn = CERTANN(nvars, integral_ops, activations, lift=lift)

        ntrain_samples = 11
        train_samples = np.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = smooth_fun(train_samples)

        ctn.fit(train_samples, train_values)

        print(ctn)

        # import matplotlib.pyplot as plt
        # plot_samples = np.linspace(-1, 1, 51)[None, :]
        # plt.plot(plot_samples[0], smooth_fun(plot_samples)[0])
        # plt.plot(ctn.train_samples, ctn.train_values, 'o')
        # plt.plot(plot_samples[0], ctn(plot_samples)[0])

        # from pyapprox.sciml.util._torch_wrappers import (asarray)
        # nparam_samples = 100
        # for ii in range(nparam_samples):
        #     bounds = ctn._hyp_list.get_active_opt_bounds().numpy()
        #     active_opt_params_np = (
        #         np.random.uniform(bounds[:, 0], bounds[:, 1]))
        #     active_opt_params = asarray(active_opt_params_np)
        #     ctn._hyp_list.set_active_opt_params(active_opt_params)
        #     print(ctn)
        #     plt.plot(plot_samples[0], ctn(plot_samples)[0], 'k-', lw=0.5)
        # plt.show()

    def test_single_layer_DenseAffine(self):
        ninputs = 21
        noutputs = ninputs

        # manufactured solution
        v0 = (1/ninputs) * np.ones((ninputs+1)*noutputs,)
        AffineBlock_manuf = DenseAffineIntegralOperator(ninputs, noutputs,
                                                        v0=v0)
        layers_manuf = Layer([AffineBlock_manuf])
        ctn_manuf = CERTANN(ninputs, layers_manuf, IdentityActivation())
        theta_manuf = ctn_manuf._hyp_list.get_values()

        # generate training samples from normal distribution with squared
        # inverse elliptic covariance
        ntrain = 1000
        training_samples = sqinv_elliptic_prior_samples(ninputs, ntrain)
        training_values = ctn_manuf(training_samples)

        # recover parameters
        v0 += np.random.normal(0, 1/ninputs, ((ninputs+1)*noutputs,))
        AffineBlock = DenseAffineIntegralOperator(ninputs, noutputs, v0=v0)
        layers = Layer([AffineBlock])

        ctn = CERTANN(ninputs, layers, IdentityActivation())
        ctn.fit(training_samples, training_values, tol=1e-14)
        theta_predicted = ctn._hyp_list.get_values()

        tol = 2e-5
        relerr = (theta_manuf-theta_predicted).norm() / theta_manuf.norm()
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_single_layer_FourierConv(self):
        # todo need test that checks when a layer has at least two
        # integral operators
        ninputs = 21
        kmax = 5

        # manufactured solution
        v0 = np.random.normal(0, 1, (2*kmax+1,))
        FourierConvBlock_manuf = FourierConvolutionOperator(kmax, v0=v0)
        layers_manuf = Layer([FourierConvBlock_manuf])
        ctn_manuf = CERTANN(ninputs, layers_manuf, IdentityActivation())
        theta_manuf = ctn_manuf._hyp_list.get_values()

        # generate training samples from normal distribution with squared
        # inverse elliptic covariance
        ntrain = 1000
        training_samples = sqinv_elliptic_prior_samples(ninputs, ntrain)
        training_values = ctn_manuf(training_samples)

        # recover parameters
        FourierConvBlock = FourierConvolutionOperator(kmax)
        layers = Layer([FourierConvBlock])

        ctn = CERTANN(ninputs, layers, IdentityActivation())
        ctn.fit(training_samples, training_values, tol=1e-8)
        theta_predicted = ctn._hyp_list.get_values()

        tol = 5e-6
        relerr = (theta_manuf-theta_predicted).norm() / theta_manuf.norm()
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_single_layer_ChebConv(self):
        ninputs = 21
        kmax = 5

        # manufactured solution
        v0 = np.random.normal(0, 1, (kmax+1,))
        ChebConvBlock_manuf = ChebyshevConvolutionOperator(kmax, v0=v0)
        layers_manuf = Layer([ChebConvBlock_manuf])
        ctn_manuf = CERTANN(ninputs, layers_manuf, IdentityActivation())
        theta_manuf = ctn_manuf._hyp_list.get_values()

        # generate training samples from normal distribution with squared
        # inverse elliptic covariance
        ntrain = 1000
        training_samples = sqinv_elliptic_prior_samples(ninputs, ntrain)
        training_values = ctn_manuf(training_samples)

        # recover parameters
        ChebConvBlock = ChebyshevConvolutionOperator(kmax)
        layers = Layer([ChebConvBlock])

        ctn = CERTANN(ninputs, layers, IdentityActivation())
        ctn.fit(training_samples, training_values, tol=1e-8)
        theta_predicted = ctn._hyp_list.get_values()

        relerr = (theta_manuf-theta_predicted).norm() / theta_manuf.norm()
        tol = 2e-6
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_single_layer_two_blocks(self):
        # When layer = [Affine, FourierConv], the parameter recovery problem is
        # under-determined, initial iterate must be close to true solution
        ninputs = 21
        noutputs = ninputs
        kmax = 5
        v0_affine = np.random.normal(0, 1, (ninputs+1)*noutputs)
        v0_conv = np.random.normal(0, 1, (2*kmax+1,))

        AffineBlock_manuf = DenseAffineIntegralOperator(ninputs, noutputs,
                                                        v0=v0_affine)
        FourierConvBlock_manuf = FourierConvolutionOperator(kmax, v0=v0_conv)
        layers_manuf = Layer([AffineBlock_manuf, FourierConvBlock_manuf])
        ctn_manuf = CERTANN(ninputs, layers_manuf, IdentityActivation())
        theta_manuf = ctn_manuf._hyp_list.get_values()

        # generate training samples from normal distribution with squared
        # inverse elliptic covariance
        ntrain = 1000
        training_samples = sqinv_elliptic_prior_samples(ninputs, ntrain)
        training_values = ctn_manuf(training_samples)
        noise_stdev = 1e-1  # standard deviation of additive noise
        v0_affine = ctn_manuf._hyp_list.hyper_params[0].get_values().numpy()
        v0_affine_rand = np.random.normal(0, noise_stdev, v0_affine.shape)
        v0_conv_rand = np.random.normal(0, noise_stdev, v0_conv.shape)

        AffineBlock = (
            DenseAffineIntegralOperator(ninputs, noutputs,
                                        v0=v0_affine+v0_affine_rand))
        FourierConvBlock = (
            FourierConvolutionOperator(kmax, v0=v0_conv+v0_conv_rand))
        layers = Layer([AffineBlock, FourierConvBlock])

        ctn = CERTANN(ninputs, layers, IdentityActivation())
        ctn.fit(training_samples, training_values, verbosity=0, tol=1e-5)
        theta_predicted = ctn._hyp_list.get_values()

        tol = 4e-2
        relerr = (theta_predicted-theta_manuf).norm() / theta_manuf.norm()
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'


if __name__ == "__main__":
    single_layer_certann_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestSingleLayerCERTANN))
    unittest.TextTestRunner(verbosity=2).run(single_layer_certann_test_suite)
