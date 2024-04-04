from typing import Union

import numpy as np

from pyapprox.sciml.kernels import Kernel
from pyapprox.sciml.util._torch_wrappers import (
    array, asarray, where, sin, zeros, exp)
from pyapprox.sciml.util.hyperparameter import (
    HyperParameter, HyperParameterList, LogHyperParameterTransform)
# todo move HomogeneousLaplace1DGreensKernel here


class GreensFunctionSolver():
    def __init__(self, kernel, quad_rule):
        self._kernel = kernel
        self._quad_rule = quad_rule

    def __call__(self, forcing_fun, xx):
        quad_xx, quad_ww = self._quad_rule.get_samples_weights()
        assert quad_xx.shape[0] == xx.shape[0]
        forcing_vals = forcing_fun(quad_xx)
        assert forcing_vals.shape[1] == 1
        return (self._kernel(xx, quad_xx)*forcing_vals[:, 0]) @ quad_ww


class DrivenHarmonicOscillatorGreensKernel(Kernel):
    """
    \frac{d^2u}{dt^2}+w^2u=f
    u(0) = u'(0) = 0
    """
    def __init__(self,
                 omega: Union[float, array],
                 omega_bounds: array):
        self._nvars = 1
        self._omega = HyperParameter(
            "omega", 1, omega, omega_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._omega])

    def __call__(self, X1, X2=None):
        omega = self._omega.get_values()
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        K = sin(omega*(X1.T-X2))/omega
        K[X1.T-X2 < 0] = 0.
        return K


class Helmholtz1DGreensKernel(Kernel):
    def __init__(self,
                 wavenum: Union[float, array],
                 wavenum_bounds: array,
                 L: float = 1):
        self._nvars = 1
        self._L = L
        self._wavenum = HyperParameter(
            "wavenum", 1, wavenum, wavenum_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._wavenum])

    def _greens_function(self, k, L, X1, X2):
        return sin(k*(X1.T-L))*sin(k*X2)/(k*sin(k*L))

    def __call__(self, X1, X2=None):
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        wavenum = self._wavenum.get_values()
        K = zeros((X1.shape[1], X2.shape[1]))
        idx = where(X1.T >= X2)
        K_half = self._greens_function(wavenum, self._L, X1, X2)[idx]
        K[idx] = K_half
        idx = where(X1.T <= X2)
        K[idx] = self._greens_function(wavenum, self._L, X2, X1).T[idx]
        return K


class HeatEquation1DGreensKernel(Kernel):
    r"""
    du/dt-kdu^2/dx^2=Q(x,t)
    u(x, 0) = f(x) u(0, t) = 0 u(L, t) = 0

    u(x,t) = int_0^L f(\xi)G(x,t;\xi,\tau) dxi +
             int_0^L int_0^t Q(\xi, \tau)G(x,t;xi,tau)d\tau d\xi

    At tau = 0, G(x, t; \xi, \tau) expresses the influence of the
    initial temperature at x0 on the temperature at
    position x and time t. In addition, G(x, t; \xi, \tau) shows the
    influence of the source/sink term Q(\xi, \tau) at
    position x0 and time t0 on the temperature at position x and time t

    Non zero forcing Q requires 2D integration.
    """
    def __init__(self,
                 kappa: Union[float, array],
                 kappa_bounds: array,
                 L: float = 1,
                 nterms: int = 10):
        self._nvars = 1
        self._nterms = nterms
        self._L = L
        self._kappa = HyperParameter(
            "kappa", 1, kappa, kappa_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._kappa])

    def _series_term(self, ii, k, L, X1, X2):
        x, t = X1[:1], X1[1:2]
        xi, tau = X2[:1], X2[1:2]
        term = sin(ii*np.pi*x.T/L)*sin(ii*np.pi*xi/L)*exp(
            -k*(ii*np.pi/L)**2*(t.T-tau))
        term[t.T < tau] = 0
        return term

    def __call__(self, X1, X2=None):
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        vals = 0
        kappa = self._kappa.get_values()
        for ii in range(self._nterms):
            vals += self._series_term(ii, kappa, self._L, X1, X2)
        vals *= 2/self._L
        return vals
