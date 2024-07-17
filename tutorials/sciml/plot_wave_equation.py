r"""
The Wave and Helmholtz Equations
================================

Wave Equation
-------------
The wave equation is

.. math:: -\partial_{tt} u(x,t) + c^2 \nabla^2 u(x,t) + f(x,t) = 0


Relationship to Helmholtz equation
----------------------------------
The Helmholtz equation can be derived from the wave equation using the Fourier transform.

Specifically noting

.. math:: \partial_{tt} e^{-\mathrm{i}\omega t} = -\omega^2 e^{-\mathrm{i}\omega t}

we have

.. math::
   0& =
   -\partial_{tt} u(x,t) + c^2 \nabla^2 u(x,t) + f(x,t)
   \\ & =
   -\partial_{tt} \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty U(x,\omega) e^{-\mathrm{i}\omega t} \mathrm d\omega
   + c^2 \nabla^2 \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty U(x,\omega) e^{-\mathrm{i}\omega t} \mathrm d\omega
   + \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty F(x,\omega) e^{-\mathrm{i}\omega t} \mathrm d\omega
   \\ & =
   \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty \left[
   -U(x,\omega) \partial_{tt} e^{-\mathrm{i}\omega t}
   + c^2 \nabla^2  U(x,\omega) e^{-\mathrm{i}\omega t}
   +  F(x,\omega) e^{-\mathrm{i}\omega t}
   \right]\mathrm d\omega
   \\ & =
   \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty \left[
   \omega^2U(x,\omega)
   + c^2 \nabla^2  U(x,\omega)
   +  F(x,\omega)
   \right] e^{-\mathrm{i}\omega t} \mathrm d\omega
   \\ & =
   \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty \left[
   \nabla^2 U(x,\omega)+k^2U(x,\omega)  + \frac{1}{c^2}F(x,\omega)\right] e^{-\mathrm{i}\omega t} \mathrm d\omega

where following convention we set :math:`k=\frac{\omega}{c}`.

The last line can only be zero for all values of of :math:`x` if

.. math::  \nabla^2 U(x,\omega)+k^2U(x,\omega)  + \frac{1}{c^2}F(x,\omega)=0

for all :math:`\omega`. The above equation is precisely the definition of Helmholtz equation. Note the Helmholtz equation is a type of reaction-diffusion equation.

Helmholtz Equation
------------------
The standard form of Helmholtz equation on :math:`[0,L]` with
:math:`u(0)=u(L)=0` is

.. math:: \nabla^2 U(x,\omega)+k^2U(x,\omega) - g(x,\omega)=0

The Greens Function for this standard form is (valid only when a is an integer muliple of :math:`\pi` because boundary conditios will not be satisfied otherwise))

.. math::

   K(x,y)=\begin{cases}
   \frac{\sin (k y) \sin (k (x-L))}{k\sin (k L)} & x > y\\
   \frac{\sin (k x) \sin (k (y-L))}{k\sin (k L)} & x < y
   \end{cases}

which can be used to solve the Helmholtz equation via

.. math:: u(x,\omega)=\int_0^L K(x,y)g(y,\omega)\dx{y}

Note when deriving the Helmholtz equation from a inhomogeneous wave equation

.. math:: g(x,\omega) = -\frac{1}{c^2}F(x,\omega)


The Fourier transform the frequency :math:`\omega` can be positive or negative. This results in either negative or positive values of the wavenumber. However, the Helmholtz equation depends on :math:`k^2` and is invariant with respect to a change of sign in :math:`k`.

Useful Indentities
------------------
Euler's formula

.. math:: \cos(x) = \frac{\exp(\mathrm{i}x)+\exp(-\mathrm{i}x)}{2} \qquad \cos(x) = \frac{\exp(\mathrm{i}x)-\exp(-\mathrm{i}x)}{2\mathrm{i}}

Example
-------
Consider the manufactured solution

.. math:: u(x,t)=\sin(a x)\cos(\omega_0 t)

Applying the differential operators we have

.. math:: \partial_{tt} u(x,t)=-\omega_0^2\sin(a x)\cos(\omega_0 t), \quad c^2\nabla^2 u(x,t)= a^2c^2 \sin(a x)\cos(\omega_0 t)

so

.. math:: f(x,t) = \partial_{tt} u(x,t)-c^2\nabla^2 u(x,t)=(a^2c^2-\omega_0^2) \sin(a x)\cos(\omega_0 t)

The Fourier transform on the forcing is

.. math::

   F(x,\omega) &= \sqrt{\frac{\pi}{2}}(a^2c^2-\omega_0^2) \sin(a x)\delta(\omega-\omega_0)+\sqrt{\frac{\pi}{2}}(a^2c^2-\omega_0^2) \sin(a x)\delta(\omega+\omega_0)\\
    &= F_1(x,\omega)+F_2(x,\omega)

Thus we must solve one Helmholtz equation

.. math::  \nabla^2 U(x,\omega_0)+\frac{\omega_0^2}{c^2}U(x,\omega_0)+\frac{1}{c^2}F_1(x,\omega_0)=0

Equivalently

.. math::

   \nabla^2 U(x,\omega_0)+k^2U(x,\omega_0)+\frac{k^2\sin(a x)}{\omega_0^2}\left(\sqrt{\frac{\pi}{2}}(\frac{a^2\omega_0^2}{k^2}-\omega_0^2)\right)&=0\\
   U(x,\omega_0)+k^2U(x,\omega_0)+\sqrt{\frac{\pi}{2}}(a^2-k^2)\sin(a x)

Using the Greens function above with :math:`L=1` yields

.. math::
   U(x, \omega_0) &= -\int_0^1 K(x,y)F_1(y,\omega_0)\dx{y} \\
   &=  -\int_0^x \frac{\sin (k y) \sin (k (x-L))}{k\sin (k L)}F_1(y,\omega_0)\dx{y} - \int_x^1 \frac{\sin (k x) \sin (k (y-L))}{k\sin (k L)}F_1(y,\omega_0)\dx{y}\\
   &= \frac{\sin(a x) - \frac{\sin(a)\sin(k x)}{\sin(k)}}{a^2 - k^2}\left(\sqrt{\frac{\pi}{2}}(a^2-k^2)\right)\\
   &= \left(\sin(a x) - \frac{\sin(a)\sin(k x)}{\sin(k)}\right)\sqrt{\frac{\pi}{2}}\\
   &= \sqrt{\frac{\pi}{2}}\sin(a x)

Notes: The second term on the last line is zero because to satisfy the boundary conditions sin(a)=0. The minus sign in the first line is because :math:`g(x,\omega)=-F_1(x,\omega)`.

To obtain the solution to the wave equation we must apply the inverse fourier transform.

.. math::
   u(x,t) &= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty U(x,\omega)e^{-\mathrm{i}\omega t}\dx{\omega}\\
   &=   \frac{1}{\sqrt{2\pi}}\left(U(x,\omega_0)e^{-\mathrm{i}\omega_0 t} + U(x,-\omega_0)e^{\mathrm{i}\omega_0 t}\right)\\
   &=U(x,\omega_0)\left(e^{-\mathrm{i}\omega_0 t}+e^{\mathrm{i}\omega_0 t})\right)\\
   &=U(x,\omega_0)\frac{1}{\sqrt{2\pi}}2\cos(\omega_0 t)\\
   &=U(x,\omega_0)\sqrt{\frac{2}{\pi}}\cos(\omega_0 t)\\
    &=\sqrt{\frac{\pi}{2}}\sin(a x)\sqrt{\frac{2}{\pi}}\cos(\omega_0 t)\\
   &=\sin(ax)\cos(\omega_0 t)


"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.sciml.quadrature import Fixed1DTrapezoidIOQuadRule


def _greens_function(k, L, X, Y):
    return np.sin(k*(X.T-L))*np.sin(k*Y)/(k*np.sin(k*L))


def greens_function(k, L, X, Y):
    K = np.zeros((X.shape[1], Y.shape[1]))
    idx = np.where(X.T >= Y)
    K_half = _greens_function(k, L, X, Y)[idx]
    K[idx] = K_half
    idx = np.where(X.T <= Y)
    K[idx] = _greens_function(k, L, Y, X).T[idx]
    return K


def greens_function_series(nterms, k, L, X, Y):
    series_sum = 0
    for nn in range(nterms):
        series_sum += (np.sin(nn*np.pi*X.T/L)*np.sin(nn*np.pi*Y/L) /
                       (k**2-(nn*np.pi/L)**2))
    return 2/L*series_sum


def greens_solution(quad_rule, kernel, forc, xx):
    quad_xx, quad_ww = quad_rule.get_samples_weights()
    return (kernel(xx, quad_xx.numpy())*forc(quad_xx.numpy())[:, 0] @
            quad_ww.numpy())


L = 1
wave_number = 10
# x_freq must be a integer multiple of np.pi otherwise BC will be violated
x_freq = 2*np.pi
t_freq = 3*np.pi
plot_xx = np.linspace(0, L, 101)[None, :]

axs = plt.subplots(1, 3, figsize=(3*8, 6))[1]
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
G = greens_function(wave_number, L, plot_xx, plot_xx)
greens_plot = axs[0].imshow(G, origin="lower", extent=[0, 1, 0, 1], cmap="jet")

# G1 = greens_function_series(100, wave_number, L, plot_xx, plot_xx)
# axs[1].imshow(G1, origin="lower", extent=[0, 1, 0, 1], cmap="jet")


# im = axs[2].imshow(abs(G-G1), origin="lower", extent=[0, 1, 0, 1], cmap="jet")
# plt.colorbar(im, ax=axs[2])
# plt.show()


# manufactured helmholtz_forcing_const
# def sol(a, xx):
#     return np.sin(a*xx.T)

# def forc(k, a, xx):
#     return (k**2-a**2)*np.sin(a*xx.T)
# plt.figure()
# gsol = greens_solution(
#         Fixed1DTrapezoidIOQuadRule(301),
#         lambda X, Y: greens_function(wave_number, L, X, Y),
#         lambda xx: forc(wave_number, x_freq, xx),
#         plot_xx)
# plt.plot(plot_xx[0], gsol)
# plt.plot(plot_xx[0], sol(x_freq, plot_xx))
# print(gsol-sol(x_freq, plot_xx))
# plt.show()
# assert False


def exact_wave_sol(k, a, w0, time, xx):
    return np.sin(a*xx.T)*np.cos(w0*time)


def wave_forcing_const(k, a, w0):
    return a**2*w0**2/k**2-w0**2


def wave_forcing_fun(k, a, w0, time, xx):
    const = wave_forcing_const(k, a, w0)
    return const*np.sin(a*xx.T)*np.cos(w0*time)


def helmholtz_forcing_const(a, k):
    return np.sqrt(np.pi/2)*(a**2-k**2)


def exact_helmholtz_sol(k, a, w0, xx):
    const = np.sqrt(np.pi/2)
    return -const*(-np.sin(a*xx.T) + 1/np.sin(k)*np.sin(a)*np.sin(k*xx.T))


def helmholtz_forcing_fun(k, a, w0, xx):
    const = helmholtz_forcing_const(k, a)
    return const*np.sin(a*xx.T)


axs[1].plot(
   plot_xx[0],
   exact_helmholtz_sol(wave_number, x_freq, t_freq, plot_xx),
   label="Exact Helmholtz Solution")
sol_plot = axs[1].plot(
    plot_xx[0],
    greens_solution(
        Fixed1DTrapezoidIOQuadRule(301),
        lambda X, Y: greens_function(wave_number, L, X, Y),
        lambda xx: helmholtz_forcing_fun(wave_number, x_freq, t_freq, xx),
        plot_xx), '--', label="Greens Helmholtz Solution")
# axs[1].plot(plot_xx[0], forcing_fun(wave_number, freq, plot_xx))
axs[1].legend()

time = 3/4
axs[2].plot(
    plot_xx[0],
    exact_wave_sol(wave_number, x_freq, t_freq, time, plot_xx),
    '-', label="Wave Exact Solution")
const = 2/np.sqrt(2*np.pi)*np.cos(t_freq*time)
axs[2].plot(
    plot_xx[0],
    exact_helmholtz_sol(wave_number, x_freq, t_freq, plot_xx)*const,
    '--', label="Fourier Transform Solution")
axs[2].legend()
plt.show()
