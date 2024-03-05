r"""
Chebyshev Transform Derivation
==============================

Concise Statement
-----------------

The Chebyshev transform computes the coefficients :math:`\hat{u}_n` of an
interpolating Chebyshev polynomial. Unlike the more famous Fourier
transform, the Chebyshev transform is designed for functions that are
*not* periodic. The forward and inverse transforms are given by

.. math::
    \hat{u}_n &=~~~~\mathcal{T}(\mat{u})_n &&= \frac{w_n}{2N} \Big[
        \sum_{j=0}^{N} w_j \, u_j \, \cos\left( \frac{\pi nj}{N} \right) \Big],
        &&\qquad n=0,\dots,N, \\
    u_n &= \mathcal{T}^{-1} (\mat{\hat{u}})_n &&= \sum_{j=0}^N \hat{u}_j \,
        \cos\left(\frac{\pi nj}{N} \right), &&\qquad n=0,\dots,N.

In the above equations,

* :math:`w_0 = w_N = 1`, with :math:`w_n = 2` otherwise, and
* :math:`u_n = u(x_n)`, with

.. math::
    x_n = \cos \Big( \frac{\pi n}{N} \Big), \qquad n=0,1,\dots,N.

**Note:** Some authors put a minus sign in front of :math:`\cos` so that
:math:`x_0 < \cdots < x_N`. Nothing is wrong with that, but it would
reverse the indexing in the frequency domain.

Derivation
----------

A key relationship allows us to recast Chebyshev approximation
in the frequency domain: the Chebyshev polynomials :math:`T_n` obey

.. math ::
    T_n(\cos(\theta)) = \cos(n \theta), \qquad n\geq 0 .

Consider the function :math:`u : [-1,1] \to \reals`. We make no
assumptions on :math:`u(x)` other than being continuous for
:math:`x \in (-1,1)`. Our goal is to determine the coefficients
:math:`\hat{u}_n` of a degree-:math:`N` interpolating polynomial

.. math::
    P_N(x) = \sum_{n=0}^N \hat{u}_n T_n(x)

such that :math:`P_N(x_j) = u(x_j) = u_j` at the nodes :math:`x_j` given above.

With the change of variables :math:`x = \cos(\theta)`, the interpolating
polynomial becomes the cosine series

.. math::
    R(\theta) = \sum_{n=0}^N \hat{u}_n \cos(n \theta)\, .

The target function is now :math:`f(\theta) = u(\cos(\theta))`, and the
interpolation conditions are

.. math::
    R(\pi j/N) = f(\pi j/N), \qquad 0 \leq j \leq N.

Importantly, :math:`f` is both even and periodic (example below).

.. _even-extension:

"""
import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(-1, 1, 21)
u = np.exp(xx)
xx_even = np.linspace(-3, 1, 41)
u_even = np.hstack([u[-1:0:-1], u])
theta = np.linspace(-2*np.pi, 2*np.pi, 81)
f = np.hstack([u[-1:0:-1], u[0:-1], u[-1:0:-1], u])

fig, ax = plt.subplots(1, 2)

ax[0].plot(xx, u, 'k')
ax[0].set_xlabel(r'$x$')
ax[0].set_title(r'$u(x) = \mathrm{e}^x$', fontsize=10)
ax[0].set_xlim([-2, 2])
ax[0].set_box_aspect(1)

ax[1].plot(theta, f, 'r')
ax[1].set_title(r'$f(\theta) = u(\cos(\theta))$', fontsize=10)
ax[1].set_xticks([-2*np.pi, 0, 2*np.pi], labels=[r'$-2\pi$', r'$0$', r'$2\pi$'])
ax[1].set_xlabel(r'$\theta$')
ax[1].set_xlim([-2*np.pi, 2*np.pi])
ax[1].set_box_aspect(1)

fig.set_figheight(fig.get_size_inches()[0]/2)
fig.tight_layout()
plt.show()

# %%
# The coefficients :math:`\hat{u}_n` satisfy the :math:`L^2` Fourier
# coefficient relations
#
# .. math::
#   \hat{u}_0 = \frac{1}{\pi} \int_0^{\pi} R(\theta) \dx{\theta}, \qquad
#   \hat{u}_n = \frac{2}{\pi}\int_0^{\pi} R(\theta) \cos(n \theta)
#   \dx{\theta}, \quad n = 1, \dots, N.
#
# Our next step is to compute these integrals using the data
# :math:`\{ u_j \}_{j=0}^N` that we already have. Applying the
# :ref:`lemma` below to :math:`v_n(\theta) = R(\theta) \cos(n\theta)`
# along with the interpolation conditions yields
#
# .. math::
#   \hat{u}_0 &= \frac{1}{2N} \Big[ v_0(0) + v_0(\pi) + 2 \sum_{j=1}^{N-1}
#       v_0(\pi j/N) \Big] \\
#   &= \frac{1}{2N} \Big[ u_0 + u_N  + 2 \sum_{j=1}^{N-1} u_j \Big], \\
#   \hat{u}_n &= \frac{1}{N} \Big[ v_n(0) + v_n(\pi) + 2 \sum_{j=1}^{N-1}
#       v_n(\pi j/N) \Big] \\
#   &= \frac{1}{N} \Big[ u_0 + (-1)^n u_N  + 2 \sum_{j=1}^{N-1} u_j \cos(\pi nj
#       / N) \Big], \qquad 1 \leq n < N.
#
# For :math:`\hat{u}_N`, the lemma does not apply since :math:`\cos^2(N\theta)`
# has degree :math:`2N`. We would, however, like for a similar
# discretization to hold. We have already shown that the interpolation is
# exact for every basis function except :math:`\cos(N\theta)`, so it is
# sufficient to consider :math:`R(\theta) = \cos(N\theta)`. In that case,
# we have
#
# .. math::
#   \hat{u}_N = \frac{2}{\pi}\int_0^\pi \cos^2(N\theta) \dx{\theta} = 1 .
#
# But :math:`\cos^2(j\pi) = 1` for integer :math:`j`, so
#
# .. math::
#   v_N(0) + v_N(\pi) + 2 \sum_{j=1}^{N-1} v_N(\pi j / N) = 2N,
#
# which means
#
# .. math::
#   \hat{u}_N = \frac{1}{2N} \Big[ v_N(0) + v_N(\pi) + 2 \sum_{j=1}^{N-1}
#       v_N(\pi j / N) \Big] .
#
# .. _lemma:
#
# Lemma
# ^^^^^
# If :math:`g(\theta)` is a cosine series of degree :math:`2N-1`, then
#
# .. math::
#   \frac{2}{\pi} \int_0^{\pi} g(\theta) \dx{\theta} = \frac{1}{N}\Big[g(0)
#   + g(\pi) + 2\sum_{j=1}^{N-1} g(\pi j/N) \Big] \, .
#
# **Proof:** The Euler--Maclaurin formula gives
#
# .. math::
#   \int_{-\pi}^{\pi} g(\theta) \dx{\theta} = \frac{\pi}{N}
#   \sum_{j=0}^{2N-1} g\Big( \pi - \frac{\pi j}{N} \Big) \, ,
#
# where we have used
#
# * the periodicity of :math:`g(\theta)` and all its derivatives over
#   :math:`[-\pi, \pi]`,
# * the change of variables :math:`\theta = \pi - \pi z/N`.
#
# No aliasing occurs in the :math:`2N`-point rule since there are exactly
# as many quadrature points as cosine modes. Because :math:`g` is even,
# then :math:`g(-\pi j/N) = g(\pi j/N)`, so we combine terms to obtain
#
# .. math::
#   \sum_{j=0}^{2N-1} g\Big( \pi - \frac{\pi j}{N} \Big) = g(0) + g(\pi)
#   + 2\sum_{j=1}^{N-1} g(\pi j/N) \, .
#
# Lastly, the evenness of :math:`g` gives
#
# .. math::
#    \int_{0}^{\pi} g(\theta) \dx{\theta} = \frac12 \int_{-\pi}^{\pi}
#    g(\theta) \dx{\theta},
#
# from which the result immediately follows. :math:`\blacksquare`
