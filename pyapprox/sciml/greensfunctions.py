from typing import Union

import numpy as np

from pyapprox.sciml.kernels import Kernel
from pyapprox.sciml.util._torch_wrappers import (
    array, asarray, where, sin, zeros, exp, cos, einsum, absolute)
from pyapprox.sciml.util.hyperparameter import (
    HyperParameter, HyperParameterList, LogHyperParameterTransform)
# todo move HomogeneousLaplace1DGreensKernel here


class GreensFunctionSolver():
    def __init__(self, kernel, quad_rule):
        self._kernel = kernel
        self._quad_rule = quad_rule

    def _eval(self, forcing_vals, xx):
        # quad_xx, quad_ww = self._quad_rule
        quad_xx, quad_ww = self._quad_rule.get_samples_weights()
        assert forcing_vals.ndim == 2
        # assert forcing_vals.shape[1] == 1
        # return (self._kernel(xx, quad_xx)*forcing_vals[:, 0]) @ quad_ww
        return einsum(
            "ijk,j->ik",
            asarray(self._kernel(xx, quad_xx)[..., None]*forcing_vals),
            asarray(quad_ww[:, 0]))

    def __call__(self, forcing_fun, xx):
        # quad_xx, quad_ww = self._quad_rule
        quad_xx, quad_ww = self._quad_rule.get_samples_weights()
        assert quad_xx.shape[0] == xx.shape[0]
        return self._eval(forcing_fun(quad_xx), xx)


class HomogeneousLaplace1DGreensKernel(Kernel):
    r"""
    The Laplace Equation with homogeneous boundary conditions in 1D is

    .. math:: -\kappa \nabla^2 u(x) &= f(x),\quad u(0)=u(1)=0

    """
    def __init__(self,
                 kappa: Union[float, array],
                 kappa_bounds: array):
        self._nvars = 1
        self._kappa = HyperParameter(
            "kappa", 1, kappa, kappa_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._kappa])

    def __call__(self, X1, X2=None):
        kappa = self._kappa.get_values()
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        K = (0.5*(X1.T+X2-absolute(X2-X1.T))-X1.T*X2)/kappa
        return K


class DrivenHarmonicOscillatorGreensKernel(Kernel):
    r"""
    The Driven Harmonic Oscillator satisfies
    
    .. math::   \frac{\partial^2 u}{\partial t^2}+\omega^2u(t)=f(t), \quad    u(0) = u'(0) = 0
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
    r"""
    The Helmholtz Equation in 1D is
    
    .. math::  \frac{\partial^2 u}{\partial x^2}+k^2\frac{\partial^2 u}{\partial t^2} = f(x), \quad u(0)=u(L)=0
    """
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
    Greens function for the heat equation

    .. math:: \dydx{u}{t}-k \frac{\partial^2 u}{\partial x^2}=Q(x,t)

    subject to

    .. math:: u(x, 0) = f(x), \quad u(0, t) = 0, \quad u(L, t) = 0

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


class WaveEquation1DGreensKernel(Kernel):
    r"""
    The wave equation in 1D is

    .. math:: \frac{\partial^2 u}{\partial t^2}+c^2\omega^2 u(t)=f(t), \quad    u(0, t) = u(L, t) = 0, \quad u(x, 0) = f(x), \dydx{u}{t}(x,0) = g(x)
    """
    def __init__(self,
                 coeff: Union[float, array],
                 coeff_bounds: array,
                 L: float = 1,
                 nterms: int = 10,
                 pos=True):
        self._nvars = 1
        self._L = L
        self._nterms = nterms
        self._pos = pos
        self._coeff = HyperParameter(
            "coeff", 1, coeff, coeff_bounds,
            LogHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._coeff])

    def _series_c_term(self, ii, c, L, X1, X2):
        x, t = X1[:1], X1[1:2]
        xi = X2[:1]
        term = sin(ii*np.pi*x.T/L)*sin(ii*np.pi*xi/L)*cos(ii*np.pi*c*t.T/L)
        return term

    def _series_s_term(self, ii, c, L, X1, X2):
        x, t = X1[:1], X1[1:2]
        xi = X2[:1]
        term = sin(ii*np.pi*x.T/L)*sin(ii*np.pi*xi/L)*sin(ii*np.pi*c*t.T/L)/(
            ii*np.pi*c/L)
        return term

    def _series_term(self, ii, c, L, X1, X2):
        if self._pos:
            return self._series_c_term(ii, c, L, X1, X2)
        return self._series_s_term(ii, c, L, X1, X2)

    def __call__(self, X1, X2=None):
        X1 = asarray(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = asarray(X2)
        vals = 0
        coeff = self._coeff.get_values()
        for ii in range(1, self._nterms+1):
            vals += self._series_term(ii, coeff, self._L, X1, X2)
        vals *= 2/self._L
        return vals


class ActiveGreensKernel():
    def __init__(self, kernel, inactive_X1, inactive_X2):
        self._kernel = kernel
        self._inactive_X1 = np.atleast_2d(inactive_X1)
        self._inactive_X2 = np.atleast_2d(inactive_X2)

    def __call__(self, X1, X2):
        X1 = np.vstack((X1, np.tile(self._inactive_X1, X1.shape[1])))
        if X2 is not None:
            X2 = np.vstack((X2, np.tile(self._inactive_X2, X2.shape[1])))
        return self._kernel(X1, X2)


# For good notes see
#https://math.libretexts.org/Bookshelves/Differential_Equations/Introduction_to_Partial_Differential_Equations_(Herman)/07%3A_Green%27s_Functions/7.02%3A_Boundary_Value_Greens_Functions
#https://math.libretexts.org/Bookshelves/Differential_Equations/Introduction_to_Partial_Differential_Equations_(Herman)/07%3A_Green%27s_Functions/7.04%3A_Greens_Functions_for_1D_Partial_Differential_Equations

#    To find solutions of stead state PDE with nonzero forcing note use superposition.
# e.g. u_xx = f(x) u(0)=a u(L)=b
#u = u1+u2
#where u1 solves u_xx = f(x) u(0)=0 u(L)=0
# which can be found with greens function for homgeneous boundary conditions
#u1 = int f(x)G(x, x') dx'
# and u2 solves u_xx=0 u(0)=a u(L)=b
#u2 = int f(x)G(x, x') dx'.
# for u_xx=0 everywhere u must be at most a linear polynomial u=cx+d
# then solve for unknowns
# u(0)=c*(0)+d=a => d=a
# u(L)=c*(L)+d=b => c=(b-d)/L
