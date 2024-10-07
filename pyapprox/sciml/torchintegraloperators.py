import torch
from pyapprox.sciml._integraloperators import (
    EmbeddingOperator, AffineProjectionOperator, KernelIntegralOperator,
    DenseAffineIntegralOperator, DenseAffineIntegralOperatorFixedBias,
    DenseAffinePointwiseOperator, DenseAffinePointwiseOperatorFixedBias,
    FourierConvolutionOperator, FourierHSOperator,
    ChebyshevConvolutionOperator, ChebyshevIntegralOperator,
    SparseAffineIntegralOperator)
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.hyperparameter.torchhyperparameter import (
    TorchIdentityHyperParameterTransform, TorchHyperParameter,
    TorchHyperParameterList)
from pyapprox.sciml.util.torchutils import TorchUtilitiesSciML

class TorchEmbeddingOperator(EmbeddingOperator, TorchLinAlgMixin):
    def __init__(self, integralops, channel_in: int, channel_out: int,
                 nx=None):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(integralops, channel_in, channel_out, nx=nx)


class TorchAffineProjectionOperator(AffineProjectionOperator,
                                    TorchLinAlgMixin):
    def __init__(self, channel_in: int, v0=None):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(channel_in, v0=v0)


class TorchKernelIntegralOperator(KernelIntegralOperator, TorchLinAlgMixin):
    def __init__(self, kernels, quad_rule_k, quad_rule_kp1, channel_in=1,
                 channel_out=1):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(kernels, quad_rule_k, quad_rule_kp1,
                         channel_in=channel_in, channel_out=channel_out)


class TorchDenseAffineIntegralOperator(DenseAffineIntegralOperator,
                                       TorchLinAlgMixin):
    def __init__(self, ninputs: int, noutputs: int, v0=None, channel_in=1,
                 channel_out=1):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(ninputs, noutputs, v0=v0, channel_in=channel_in,
                         channel_out=channel_out)


class TorchDenseAffineIntegralOperatorFixedBias(
        DenseAffineIntegralOperatorFixedBias, TorchLinAlgMixin):
    def __init__(self, ninputs: int, noutputs: int, v0=None, channel_in=1,
                 channel_out=1):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(ninputs, noutputs, v0=v0, channel_in=channel_in,
                         channel_out=channel_out)


class TorchSparseAffineIntegralOperator(SparseAffineIntegralOperator,
                                        TorchLinAlgMixin):
    def __init__(self, ninputs: int, noutputs: int, v0=None, channel_in=1,
                 channel_out=1, nonzero_inds=None):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(ninputs, noutputs, v0=v0, channel_in=channel_in,
                         channel_out=channel_out, nonzero_inds=nonzero_inds)


class TorchDenseAffinePointwiseOperator(DenseAffinePointwiseOperator,
                                        TorchLinAlgMixin):
    def __init__(self, v0=None, channel_in=1, channel_out=1):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(v0=v0, channel_in=channel_in, channel_out=channel_out)


class TorchDenseAffinePointwiseOperatorFixedBias(
        DenseAffinePointwiseOperatorFixedBias, TorchLinAlgMixin):
    def __init__(self, v0=None, channel_in=1, channel_out=1):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(v0=v0, channel_in=channel_in, channel_out=channel_out)


class TorchFourierHSOperator(FourierHSOperator, TorchLinAlgMixin):
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1,
                 channel_coupling='full'):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(kmax, nx=nx, v0=v0, channel_in=channel_in,
                         channel_out=channel_out,
                         channel_coupling=channel_coupling)


class TorchFourierConvolutionOperator(FourierConvolutionOperator,
                                      TorchLinAlgMixin):
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1,
                 channel_coupling='full'):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(kmax, nx=nx, v0=v0, channel_in=channel_in,
                         channel_out=channel_out,
                         channel_coupling=channel_coupling)


class TorchChebyshevConvolutionOperator(ChebyshevConvolutionOperator,
                                        TorchUtilitiesSciML):
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(kmax, nx=nx, v0=v0, channel_in=channel_in,
                         channel_out=channel_out)


class TorchChebyshevIntegralOperator(ChebyshevIntegralOperator,
                                     TorchUtilitiesSciML):
    def __init__(self, kmax, nx=None, v0=None, nonzero_inds=None,
                 chol=False):
        self._HyperParameter = TorchHyperParameter
        self._HyperParameterList = TorchHyperParameterList
        self._HyperParameterTransform = TorchIdentityHyperParameterTransform
        super().__init__(kmax, nx=nx, v0=v0, nonzero_inds=nonzero_inds,
                         chol=chol)
