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
from pyapprox.sciml.util import _torch_wrappers as tw


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

    def test_single_layer_DenseAffine_single_channel(self):
        ninputs = 21
        noutputs = ninputs
        channel_in = 1
        channel_out = 1

        # manufactured solution
        v0 = (1/ninputs) * np.ones((ninputs+1)*noutputs*channel_out,)
        AffineBlock_manuf = DenseAffineIntegralOperator(ninputs, noutputs,
                                                        v0=v0,
                                                        channel_in=channel_in,
                                                        channel_out=channel_out
                                                        )
        layers_manuf = Layer([AffineBlock_manuf])
        ctn_manuf = CERTANN(ninputs, layers_manuf, IdentityActivation())
        theta_manuf = ctn_manuf._hyp_list.get_values()

        # generate training samples from normal distribution with squared
        # inverse elliptic covariance
        ntrain = 2000
        training_samples = sqinv_elliptic_prior_samples(ninputs, ntrain)
        training_values = ctn_manuf(training_samples)

        # recover parameters
        v0 += np.random.normal(0, 1/ninputs, v0.shape)
        AffineBlock = DenseAffineIntegralOperator(ninputs, noutputs,
                                                  channel_in=channel_in,
                                                  channel_out=channel_out,
                                                  v0=v0)
        layers = Layer([AffineBlock])

        ctn = CERTANN(ninputs, layers, IdentityActivation())
        ctn.fit(training_samples, training_values, tol=1e-14)
        theta_predicted = ctn._hyp_list.get_values()

        tol = 2e-5
        relerr = (theta_manuf-theta_predicted).norm() / theta_manuf.norm()
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_single_layer_DenseAffine_multichannel(self):
        ninputs = 21
        noutputs = ninputs
        channel_in = 1
        channel_out = 2

        # manufactured solution
        v0 = (1/ninputs) * np.ones((ninputs+1)*noutputs*channel_out,)
        AffineBlock_manuf = DenseAffineIntegralOperator(ninputs, noutputs,
                                                        v0=v0,
                                                        channel_in=channel_in,
                                                        channel_out=channel_out
                                                        )
        layers_manuf = Layer([AffineBlock_manuf])
        ctn_manuf = CERTANN(ninputs, layers_manuf, IdentityActivation())
        theta_manuf = ctn_manuf._hyp_list.get_values()

        # generate training samples from normal distribution with squared
        # inverse elliptic covariance
        ntrain = 2000
        training_samples = sqinv_elliptic_prior_samples(ninputs, ntrain)
        training_values = ctn_manuf(training_samples)

        # recover parameters
        v0 += np.random.normal(0, 1/ninputs, v0.shape)
        AffineBlock = DenseAffineIntegralOperator(ninputs, noutputs,
                                                  channel_in=channel_in,
                                                  channel_out=channel_out,
                                                  v0=v0)
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

    def test_single_layer_parameterized_kernel_single_channel(self):
        ninputs = 21
        matern_manuf = MaternKernel(np.inf, tw.asarray([0.2]), [0.01, 0.5], 1)

        quad_rule_k = Fixed1DGaussLegendreIOQuadRule(ninputs)
        quad_rule_kp1 = Fixed1DGaussLegendreIOQuadRule(ninputs)

        # Manufactured solution
        iop = KernelIntegralOperator([matern_manuf], quad_rule_k,
                                     quad_rule_kp1, channel_in=1,
                                     channel_out=1)
        ctn_manuf = CERTANN(ninputs, Layer([iop]), IdentityActivation())
        training_samples = tw.asarray(np.linspace(0, 1, ninputs)[:, None])
        training_values = ctn_manuf(training_samples)

        # Optimization problem
        matern_opt = MaternKernel(np.inf, tw.asarray([0.4]), [0.01, 0.5], 1)
        iop_opt = KernelIntegralOperator([matern_opt], quad_rule_k,
                                     quad_rule_kp1, channel_in=1,
                                     channel_out=1)
        layers = Layer([iop_opt])
        ctn = CERTANN(ninputs, layers, IdentityActivation())
        ctn.fit(training_samples, training_values, tol=1e-12, verbosity=0)
        relerr = tw.norm(ctn._hyp_list.get_values() - 0.2)/0.2
        tol = 4e-9
        assert relerr < tol, f'Relative error = {relerr:.2e} > {tol:.2e}'

    def test_single_layer_parameterized_kernel_multichannel(self):
        ninputs = 21

        matern_sqexp = MaternKernel(tw.inf, [0.25], [0.01, 0.5], 1)
        matern_exp = MaternKernel(0.5, [0.1], [0.01, 0.5], 1)
        quad_rule_k = Fixed1DGaussLegendreIOQuadRule(ninputs)
        quad_rule_kp1 = Fixed1DGaussLegendreIOQuadRule(ninputs)

        # Manufactured solution
        iop = KernelIntegralOperator([matern_sqexp, matern_exp], quad_rule_k,
                                     quad_rule_kp1, channel_in=2,
                                     channel_out=2)
        xx = tw.asarray(np.linspace(0, 1, ninputs))[:, None]
        samples = tw.hstack([xx, xx])[..., None]
        values = iop(samples)

        # Optimization problem
        matern_sqexp_opt = MaternKernel(np.inf, tw.asarray([0.4]), [0.01, 0.5],
                                        1)
        matern_exp_opt = MaternKernel(0.5, [0.1], [0.01, 0.5], 1)
        iop_opt = KernelIntegralOperator([matern_sqexp_opt, matern_exp_opt],
                                         quad_rule_k, quad_rule_kp1,
                                         channel_in=2, channel_out=2)
        layers = Layer([iop_opt])
        ctn = CERTANN(ninputs, layers, IdentityActivation())
        ctn.fit(samples, values, tol=1e-12, verbosity=0)
        relerr = (tw.norm(ctn._hyp_list.get_values() - tw.asarray([0.25, 0.1]))
                  / tw.norm(tw.asarray([0.25, 0.1])))
        tol = 4e-9
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
