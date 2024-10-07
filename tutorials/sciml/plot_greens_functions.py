r"""
Green's Functions
=================

Laplace Equation
----------------

Consider the constant-coefficient diffusion equation

.. math::

   -\kappa \nabla^2 u(x) &= f(x) && \qquad x\in \mathcal{D}\\
    u(x) &= 0 && \qquad x\in \partial \mathcal{D}

The Green's function :math:`G(x, y)`, for some :math:`y\in\mathcal{D}` is the
solution to

.. math::

   -\kappa \nabla^2 G(x, y) &= \delta(x-y) && \qquad x\in \mathcal{D}\\
    G(x, y) &= 0 && \qquad x\in \partial \mathcal{D}

Using the Green's function the solution of the PDE satisfies


.. math::
   u(x) = \int_\mathcal{D} G(x, y)f(y)\dx{y}


This can be verified by noting

.. math::

 -\kappa \nabla^2 u(x) &= -\kappa \int_\mathcal{D} \nabla^2 G(x, y)f(y)\dx{y}\\
 & = \int_\mathcal{D} \delta(x-y) f(y)\dx{y}\\
 &= f(x)


The Green's function for the constant coefficient diffusion equation with
:math:`\mathcal{D}=(0, 1)` and homogeneous boundary conditions is

.. math:: G(x, y) = \frac{1}{2\kappa}(x+y-|x-y|- 2x y)

The following code computes the solution to the Laplace equation by using the
trapezoid rule to compute the integral of the Green's function with the forcing
function and compares the result against the exact solution.
"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.sciml.greensfunctions import (
    HomogeneousLaplace1DGreensKernel, GreensFunctionSolver,
    HeatEquation1DGreensKernel, ActiveGreensKernel, Helmholtz1DGreensKernel,
    DrivenHarmonicOscillatorGreensKernel, WaveEquation1DGreensKernel)
from pyapprox.sciml.quadrature import (
    Fixed1DTrapezoidIOQuadRule, Transformed1DQuadRule)

np.random.seed(1)

kappa = 0.1
nquad = 100
greens_fun = HomogeneousLaplace1DGreensKernel(kappa, [1e-3, 1])
bounds = [0, 1]
quad_rule = Transformed1DQuadRule(
    Fixed1DTrapezoidIOQuadRule(nquad), bounds)
greens_solver = GreensFunctionSolver(greens_fun, quad_rule)


def forc_fun(xx):
    return (-19.2*xx**4*(1 - xx)**2 + 51.2*xx**3*(1 - xx)**3 -
            19.2*xx**2*(1 - xx)**4).T


def exact_solution(xx):
    return (16*xx**4*(1 - xx)**4).T


def greens_solution(kernel, forc, xx):
    quad_xx, quad_ww = quad_rule.get_samples_weights()
    return kernel(xx, quad_xx)*forc(quad_xx)[:, 0] @ quad_ww


plot_xx = np.linspace(*bounds, 101)[None, :]
green_sol = greens_solver(forc_fun, plot_xx)
ax = plt.figure().gca()
ax.plot(plot_xx[0], exact_solution(plot_xx), label=r"$u(x)$")
ax.plot(plot_xx[0], green_sol, '--', label=r"$u_G(x)$")
ax.plot(plot_xx[0], forc_fun(plot_xx), label=r"$f(x)=-\kappa\nabla^2 u(x)$")
ax.legend()


#%%
# Now plot the greens function
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
G = greens_fun(plot_xx, plot_xx)
ax = plt.figure().gca()
greens_plot = ax.imshow(G, origin="lower", extent=bounds+bounds, cmap="jet")


#%%
#Heat Equation
#-------------
#We can also compute the Green's function for the heat equation
#
#.. math:: \dydx{u}{t}-k \frac{\partial^2 u}{\partial x^2}=Q(x,t)
#
#subject to
#
#.. math:: u(x, 0) = f(x), \quad u(0, t) = 0, \quad u(L, t) = 0
#
#The solution to the heat equation using the greens function is
#
#.. math:: u(x,t) = \int_0^L f(\xi)G(x,t;\xi,0) d\xi + \int_0^L \int_0^t Q(\xi, \tau)G(x,t;\xi,\tau)d\tau d\xi
#
#where
#
#.. math:: G(x, \xi ; t, \tau)=\frac{2}{L} \sum_{n=1}^{\infty} \sin \frac{n \pi x}{L} \sin \frac{n \pi \xi}{L} e^{ -k(n\pi/L)^2 (t-\tau)}
#
#
#:math:`G(x, t; \xi, \tau)` quantifies the impact of the initial temperature at :math:`\xi` and time :math:`\tau = 0`  on the temperature at position :math:`x` and time :math:`t`. Similarly, :math:`G(x, t; \xi, \tau)` quantifies the impact of the forcing term :math:`Q(\xi, \tau)` at position :math:`\xi` and time :math:`\tau` on the temperature at position :math:`x` and time `t`
#
# Now plot the Green's function for :mat:`\tau=0`

L = 10
bounds = [0, L]
greens_fun_2d = HeatEquation1DGreensKernel(1, [1e-3, 100], 2*np.pi, nterms=100)
# Make greens function take 1D inputs by setting :math:`tau=0`
greens_fun = ActiveGreensKernel(greens_fun_2d, [3.], [0.])
plot_xx = np.linspace(*bounds, 101)[None, :]
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
G = greens_fun(plot_xx, plot_xx)
ax = plt.figure().gca()
greens_plot = ax.imshow(G, origin="lower", extent=bounds+bounds, cmap="jet")

#%%
#Helmholtz Equation
#------------------
#The Helmholtz Equation in 1D is
#
#.. math::  \frac{\partial^2 u}{\partial x^2}+k^2\frac{\partial^2 u}{\partial t^2} = f(x), \quad u(0)=u(L)=0
#
#where k is wave number
#
#The Green's function is
#
#.. math::  G(x, \xi) = \begin{cases}\frac{1}{\sin(kL)}\sin(k(x-L))\sin(k\xi) & x>\xi \\\frac{1}{\sin(kL)}\sin(k(\xi-L))\sin(kx) & x\leq \xi\end{cases}
#
bounds = [0, 1]
k = 10
greens_fun = Helmholtz1DGreensKernel(k, [1e-3, 100])
plot_xx = np.linspace(*bounds, 101)[None, :]
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
G = greens_fun(plot_xx, plot_xx)
ax = plt.figure().gca()
greens_plot = ax.imshow(
    G, origin="lower", extent=bounds+bounds, cmap="jet")

#%%
#Driven Harmonic Oscillator
#--------------------------
#The Driven Harmonic Oscillator satisfies
#
#.. math::   \frac{\partial^2 u}{\partial t^2}+\omega^2u(t)=f(t), \quad    u(0) = u'(0) = 0
#
#The Green's function is
#
#.. math:: G(t, \tau) = \begin{cases}\frac{1}{\omega}\sin(\omega(t-\tau)) & t\geq \tau \\0 & t <  \tau\end{cases}
final_time = 3
omega = 2
bounds = [0, final_time]
greens_fun = DrivenHarmonicOscillatorGreensKernel(omega, [1e-8, 10])
plot_xx = np.linspace(*bounds, 101)[None, :]
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
G = greens_fun(plot_xx, plot_xx)
ax = plt.figure().gca()
greens_plot = ax.imshow(
    G, origin="lower", extent=bounds+bounds, cmap="jet")


#%%
#Wave Equation
#-------------
#The wave equation in 1D is
#
#.. math::
#       \frac{\partial^2 u}{\partial t^2}+c^2\omega^2 u(t) &= f(t), \\
#       u(0, t) = u(L, t) &= 0, \\
#       u(x, 0) &= f(x), \\
#       \dydx{u}{t}(x,0) &= g(x)
#
#The Green's function is
#
#.. math:: G_\text{pos}(x, \xi, t, 0)=\frac{2}{L} \sum_{n=1}^{\infty} \sin \frac{n \pi x}{L} \sin \frac{n \pi \xi}{L} \cos \frac{n \pi c t}{L}
#
#.. math:: G_\text{vel}(x, \xi, t, 0)=\frac{2}{L} \sum_{n=1}^{\infty} \frac{L}{n \pi c}\sin \frac{n \pi x}{L} \sin \frac{n \pi \xi}{L} \sin \frac{n \pi c t}{L}
#
#The solution to the wave equation using the greens function is
#
#.. math:: u(x,t) = \int_0^L f(\xi)G_\text{pos}(x,t;\xi,0) d\xi + \int_0^L  g(\xi, \tau)G_\text{vel}(x,t;\xi,0)d\xi
#
#Here :math:`G_\text{pos}` quantifies the response to the initial position and :math:`G_\text{vel}` quantifies the response to the initial velocity
#
# Now plot the Green's function associated with the initial position. Note what it looks like while noting that
#
#.. math:: \delta(x-\xi)=\frac{2}{L} \sum_{n=1}^{\infty}\sin \frac{n \pi x}{L} \sin \frac{n \pi \xi}{L}
#
#is the Fourier series representation of the Dirac delta function :math:`\delta(x-\xi)`
omega, k = 2*np.pi/L, 5*np.pi/L
final_time = .1
coeff = omega/k
L = 10
bounds = [0, L]
greens_fun_2d = WaveEquation1DGreensKernel(
    coeff, [1e-3, 10], L=L, nterms=100, pos=False)
# Make greens function take 1D inputs by setting :math:`tau=0` and setting
# final time
greens_fun = ActiveGreensKernel(greens_fun_2d, [final_time], [0.])
plot_xx = np.linspace(*bounds, 101)[None, :]
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
G = greens_fun(plot_xx, plot_xx)
ax = plt.figure().gca()
greens_plot = ax.imshow(
    G, origin="lower", extent=bounds+bounds, cmap="jet")
plt.show()


