r"""
Fourier Transform
=================

The 1D fourier transform of a function :math:`f` is

.. math::
    \mathcal{F}[f] = F(\omega) =  \frac{1}{\sqrt{2\pi}}\int_\infty^\infty f(t)
    \exp\left(-\mathrm{i}\omega t\right)\dx{t}

The inverse fourier transform is

.. math::
    \mathcal{F}^{-1}[F] = f(t) =  \frac{1}{\sqrt{2\pi}}\int_\infty^\infty
    F(\omega) \exp\left(\mathrm{i}\omega t\right)\dx{\omega}



Convolution Theorem
-------------------
.. math::
  \mathcal{F}(f\star g) &=  \frac{1}{\sqrt{2\pi}}\int_\infty^\infty f\star g
        \exp\left(-\mathrm{i}\omega t\right) \dx{t} \\
  &=  \frac{1}{\sqrt{2\pi}}\int_\infty^\infty \int_\infty^\infty f(t-\tau)
        g(\tau)\dx{\tau} \exp\left(-\mathrm{i}\omega t\right) \dx{t} \\
  &=  \frac{1}{\sqrt{2\pi}}\int_\infty^\infty \int_\infty^\infty f(t)g(\tau)
        \dx{\tau} \exp\left(-\mathrm{i}\omega (\tau+t)\right) \dx{t} \\
  &=  \sqrt{2\pi}\frac{1}{\sqrt{2\pi}}\int_\infty^\infty f(t) \exp\left(-
        \mathrm{i}\omega t\right) \dx{t} \frac{1}{\sqrt{2\pi}}
        \int_\infty^\infty g(\tau)\exp\left(-\mathrm{i}\omega \tau\right)
        \dx{\tau}\\
  &= \sqrt{2\pi}F(\omega)G(\omega)

Where line 3 used the translation property of the Fourier transform

.. math::

  \mathcal{F}[f(t+a)](\omega) &=  \frac{1}{\sqrt{2\pi}}\int_\infty^\infty
        f(t+a) \exp\left(-\mathrm{i}\omega t\right) \dx{t}\\
  &=  \frac{1}{\sqrt{2\pi}}\int_\infty^\infty f(u) \exp\left(-\mathrm{i}\omega
        u-a\right) \dx{u}\\
  &=  \frac{1}{\sqrt{2\pi}}\exp\left(a\right)\int_\infty^\infty f(u) \exp\left(
        -\mathrm{i}\omega u\right) \dx{u}\\
  &=\exp\left(\mathrm{i}\omega a\right)\mathcal{F}[f(t)](\omega)

Discrete Fourier Transform
--------------------------
For frequencies :math:`k\in[0, N-1]` the discrete Fourier transform (DFT) is

.. math::
    F_k = \sum_{n=0}^{N-1} f_n \exp\left(-\frac{2\pi\mathrm{i}}{N}kn\right)


For :math:`n\in[0, N-1]`, the inverse transform is

.. math::
    f_n = \frac{1}{N}\sum_{n=0}^{N-1} F_n \exp\left(\frac{2\pi\mathrm{i}}{N}kn
        \right)

The following highlights the relationship between the continuous and discrete
Fourier transforms

.. math::

  F(\omega_k) &= \frac{1}{\sqrt{2\pi}}\sum_{n=0}^{N-1}\Delta t f(t_0+n\Delta t)
        \exp\left(-\mathrm{i}k\Delta \omega(t_0+\Delta t)\right) \\
  &\approx \frac{1}{\sqrt{2\pi}}\sum_{n=0}^{N-1} \Delta t f(t_0+n\Delta t)\exp
        \left(-\mathrm{i}k\Delta \omega(t_0+\Delta t)\right) \\


Now let us sample the fourier transform at equidistant frequences

.. math:: \omega_k = \frac{2\pi k}{N\Delta t}

where numpy assumes :math:`t_n=t_0+n\Delta t, n=0,\ldots,N-1`, with
:math:`\Delta t=T/N`. The point :math:`t_n=T` is left out because the function
is assumed periodic. We then have

.. math::

  F(\omega_k)&\approx \frac{1}{\sqrt{2\pi}}\sum_{n=0}^{N-1} \Delta t f(t_n)
        \exp\left(-\mathrm{i}\omega_k t_n\right) \\
  &= \frac{\Delta t}{\sqrt{2\pi}}\sum_{n=0}^{N-1}  f(t_0+n\Delta t)\exp\left(
        -\mathrm{i}\frac{2\pi k}{N\Delta t}(t_0+n\Delta t) \right) \\
  &= \frac{\Delta t}{\sqrt{2\pi}} \exp\left(-\mathrm{i}2\pi \frac{t_0 k}{
        N\Delta t}\right)\sum_{n=0}^{N-1}  f(t_0+n\Delta t)\exp\left(-
        \mathrm{i}\frac{2\pi nk}{N}\right)\\
  &= \frac{\Delta t}{\sqrt{2\pi}} \exp\left(-\mathrm{i}t_0w_k\right)
        \sum_{n=0}^{N-1}  f(t_0+n\Delta t)\exp\left(-\mathrm{i}\frac{2\pi nk}{
        N}\right)\\
  &= \underbrace{\phi(\omega_k)}_{\text{Phase Factor}}\underbrace{
        \sum_{n=0}^{N-1} f(t_0+n\Delta t)\exp\left(-\mathrm{i}\frac{2\pi nk}{N}
        \right)}_{\text{DFT}}

The phase factor is determined by the choice of origin (:math:`t_0`) for the
time coordinate :math:`t`.


The inverse DFT can be used to obtain the time signal from exact samples of the
continuous Fourier transform via

.. math::

 f(t_n) =\sum_{k=0}^{N-1}  \frac{F(\omega_k)}{\phi(\omega_k)}\exp\left(
        \mathrm{i}\frac{2\pi nk}{N}\right)

Example
-------
Consider the Fourier transform of the PDF :math:`f_{\sigma^2}(t)` of a
Gaussian with variance :math:`\sigma^2`:

.. math::

  F(\omega) &= \frac{1}{\sqrt{2\pi}}\int_\infty^\infty  f_\sigma^2(t)
        \exp\left(-\mathrm{i}\omega t\right)\dx{t}\\
  &=\frac{1}{\sqrt{2\pi}}\int_\infty^\infty  \frac{1}{\sqrt{2 \pi } \sigma}\exp
        \left(-\frac{t^2}{2 \sigma^2}\right) \exp\left(-\mathrm{i}\omega t
        \right)\dx{t} \\
  &= \frac{1}{\sqrt{2 \pi }}\exp\left(-\frac{\omega^2 \sigma^2}{2}\right)

Note there is no longer :math:`\sigma` in the fraction scaling the exponential
function and :math:`\sigma` now appears in the numerator inside the
exponential.

The convolution of the PDFs of two Gaussians with mean zero and variances
:math:`\sigma_1^2, \sigma_2^2` is

.. math::

    h(t) = \int_\infty^\infty f_{\sigma_1^2}(t-\tau)f_{\sigma_2^2}(\tau)
    \dx{\tau} = \frac{1}{\sqrt{2 \pi(\sigma_1^2+\sigma_2^2) }}\exp\left(-
    \frac{t^2}{2(\sigma_1^2+\sigma_2^2)}\right)=f_{\sigma_1^2+\sigma_2^2}(t)

This result can also be obtained using the convolution theorem, which states

.. math::

    (f\star g) (t) = \int_\infty^\infty f(t-\tau)(\tau)\dx{\tau} =
    \sqrt{2\pi}\mathcal{F}^{-1}[\mathcal{F}[f]\mathcal{F}[g]].

Using the Fourier transform of a Gaussian PDF yields

.. math::

  \sqrt{2\pi}\mathcal{F}^{-1}[\mathcal{F}[f]\mathcal{F}[g]] &= \sqrt{2\pi}
        \int_\infty^\infty \frac{1}{\sqrt{2 \pi }}\exp\left(-\frac{\omega^2
        \sigma_1^2}{2}\right)\frac{1}{\sqrt{2 \pi }} \exp\left(-\frac{\omega^2
        \sigma_2^2}{2}\right)\exp\left(-\mathrm{i}\omega t\right)\dx{\omega} \\
  &=f_{\sigma_1^2+\sigma_2^2}(t)

Now let's compute the compare the continuous and discrete Fourier transforms
numerically.

First define the Gaussian PDF and its Fourier transform
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


def gauss(x, var):
    return 1/(np.sqrt(var)*np.sqrt(2*np.pi))*np.exp(-x**2/(2*var))


def fourier_gauss(x, var):
    return 1/(np.sqrt(2*np.pi))*np.exp(-x**2*var/(2))


# Now generate discrete time series
t0, tfinal = -500, 500
s1, s2 = 1, 2
N = 40000
deltat = (tfinal-t0)/N
# final time is not included in tt because we assume signal is periodic
tt_centered = np.arange(N)*deltat+t0
tt = np.fft.ifftshift(tt_centered)
deltaw = 2*np.pi/(N*deltat)
ww = np.fft.fftfreq(N)*2*np.pi/deltat
ww_centered = np.fft.fftshift(ww)
assert np.allclose(deltaw, ww[1]-ww[0])

fx = gauss(tt_centered, s1**2)
gx = gauss(tt_centered, s2**2)

# %%
# Now compute the DFT of the two signals using the fast Fourier transform and
# plot
fx_fft = np.fft.fft(fx, axis=-1)
gx_fft = np.fft.fft(gx, axis=-1)

# compute the frequency samples
phase_factor = deltat/np.sqrt(2*np.pi)*np.exp(-complex(0, 1) * ww * t0)

ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ww_plot = np.linspace(-10, 10, 101)
ax.plot(ww, np.abs(fx_fft*phase_factor), 'or', label=r"DFT[f]", alpha=0.3)
ax.plot(ww, np.abs(gx_fft*phase_factor), 'sg', label=r"DFT[g]", alpha=0.3)
ax.plot(
    ww, np.abs(np.fft.fft(gauss(tt_centered, s2**2)*deltat/np.sqrt(2*np.pi),
                          axis=-1)), 'sg', label=r"DFT[g]", alpha=0.3)
ax.plot(ww_plot, fourier_gauss(ww_plot, s1**2), label=r"$\mathcal{F}[f]$",
        c='k', lw=3)
ax.plot(ww_plot, fourier_gauss(ww_plot, s2**2), label=r"$\mathcal{F}[g]$",
        c='b', lw=3)
ax.legend()
ax.set_xlim(-10, 10)

# %%
# Now compute the IDFT of the two signals and compare with their exact values
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
tt_plot = np.linspace(-10, 10, 101)
ax.plot(tt_plot, gauss(tt_plot, s1**2), label=r"$f$")
ax.plot(tt_plot, gauss(tt_plot, s2**2), label=r"$g$")

# the following two comments lines are equivalent to the third uncomment line
# ifft_fft_fx = np.fft.fftshift(
#    np.fft.ifft(fourier_gauss(ww, s1)/deltat*np.sqrt(2*np.pi)))
ifft_fft_fx = np.fft.ifft(fourier_gauss(ww, s1**2)/phase_factor)
ax.plot(tt_centered, ifft_fft_fx, '--k', label=r"DFT$^{-1}[DFT[f]]")
ifft_fft_gx = np.fft.fftshift(
    np.fft.ifft(fourier_gauss(ww, s2**2)))/deltat*np.sqrt(2*np.pi)
ax.plot(tt_centered, ifft_fft_gx, '--r', label=r"DFT$^{-1}[DFT[g]]")
ax.legend()
ax.set_xlim(-10, 10)

# %%
# Now compute the convolution of the time signals using the convolution theorem
# and compare with the analytical convolution
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
# the last sqrt is the factor from the convolution theorem
conv = np.fft.ifft(fourier_gauss(ww, s1**2)*fourier_gauss(ww, s2**2), axis=-1)
conv = np.fft.fftshift(conv)/deltat*np.sqrt(2*np.pi)**2

ax.plot(tt_plot, gauss(tt_plot, s1**2+s2**2), label=r"$f=g*h$", c='k')
ax.plot(tt_centered, np.abs(conv), '--r', label=r"DFT$^{-1}[DFT[f]DFT[g]]$")
ax.set_xlim(-10, 10)
ax.set_ylim(0, 0.2)
ax.legend()


# %%
# Now let's plot the kernel using its Fourier transformation.
# Previously we computed the Fourier transform of
#
# .. math::
#  K(t)=\frac{1}{\sqrt{2 \pi } \sigma}\exp\left(-\frac{t^2}{2 \sigma^2}\right)
#
# This is the Fourier transform of a scaled squared-exponential kernel with
# length-scale :math:`\sigma^2`
#
# .. math::
#    K(x,y), \qquad \text{where}~t=(x-y)
#
# The covariance will be scaled by :math:`\frac{1}{\sigma\sqrt{2 \pi}}`.

x0, x1 = -3, 3
sigma = 5
Nx = 101
deltax = (x1-x0)/Nx
xx = np.arange(Nx)*deltax+x0
yy = xx

flat_grid = (xx[None, :]-xx[:, None]).flatten()
tt_centered, indices, inv_indices = np.unique(
    flat_grid, return_index=True, return_inverse=True)
deltat = tt_centered[1]-tt_centered[0]

tt = np.fft.ifftshift(tt_centered)
Nt = tt.shape[0]
ww = np.fft.fftfreq(Nt)*2*np.pi/deltat
Kmat_flat = np.abs(np.fft.fftshift(
    np.fft.ifft(fourier_gauss(ww, sigma**2)))/deltat*np.sqrt(2*np.pi))
Kmat = Kmat_flat[inv_indices].reshape((Nx, Nx))
assert np.allclose(np.diag(np.sqrt(2*np.pi*sigma**2)*Kmat), 1.)


ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.imshow(Kmat)
