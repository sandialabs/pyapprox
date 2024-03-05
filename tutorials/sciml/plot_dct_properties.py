r"""
Chebyshev Transform Properties
==============================

Recall the forward and inverse Chebyshev transforms:

.. math::
    \mathcal{T}(\mat{u})_n &= \frac{w_n}{2N} \Big[ \sum_{j=0}^{N} w_j \, u_j \,
        \cos\left( \frac{\pi nj}{N} \right) \Big], &&\qquad n=0,\dots,N, \\
    \mathcal{T}^{-1} ( \mat{\hat{u}})_n &= \sum_{j=0}^N \hat{u}_j \, \cos\left(
        \frac{\pi nj}{N} \right), &&\qquad n=0,\dots,N,

where

.. math::
    w_n = \begin{cases} 1, & n=0~\text{or}~n=N \\ 2, & 0<n<N \end{cases} &
        \qquad \mat{w} = [w_0, \dots, w_N] \in \reals^{N+1} \, , \\
    \mathbf{u}_n = u(x_n), & \qquad x_n = \cos(n \pi / N)\, .

For brevity, we introduce the following useful notation:

* **Even periodic extension:** The even periodic extension of a vector
  :math:`\mat{x} \in \reals^{N+1}` is

.. math::
  \mat{x}^\text{per} = [x_0, \dots, x_N, x_{N-1}, \dots, x_1]\in\reals^{2N}\, .

* **Componentwise product:** The componentwise product of two vectors
  :math:`\mat{x},\mat{y} \in \reals^{N+1}` is

.. math::
    \mat{x} \odot \mat{y} = [x_0 y_0, \dots, x_N y_N] \in \reals^{N+1} \, .

* **Componentwise division:** The componentwise quotient of a vector
  :math:`\mat{x} \in \reals^{N+1}` with a vector
  :math:`\mat{y} \in (\reals \backslash \{0\})^{N+1}` is

.. math::
    \frac{\mat{x}}{\mat{y}} = \Big[ \frac{x_0}{y_0}, \cdots,
    \frac{x_N}{y_N} \Big] \in \reals^{N+1}\, .

* **Circular convolution:** The circular convolution of
  :math:`\mat{x}, \mat{y} \in \reals^{N+1}`, denoted
  :math:`\mat{x} \circledast \mat{y} \in \reals^{N+1}`, is

.. math::
    (\mat{x} \circledast \mat{y})_n = \sum_{j=0}^N x_j \, y_{(n-j)\mod(N+1)},
    \qquad n=0, \dots, N.


Linearity
---------

The forward and inverse Chebyshev transforms are linear.


Relation to Fourier Transform
-----------------------------

We can use the fact that the Chebyshev transform is based on an even periodic
extension to connect the Chebyshev and Fourier transforms. Recall the forward
and inverse Fourier transforms of a length-:math:`M` signal:

.. math::
    \hat{f}_n = \mathcal{F}( \mat{f} )_n &= \sum_{j=0}^{M-1} f_j \, \exp(-2 \pi
        \text{i} n j/N), &&\qquad n=0,\dots,M-1, \\
    \mathcal{F}^{-1}( \mat{\hat{f}} )_n &= \frac{1}{N} \sum_{j=0}^{M-1}
        \hat{f}_j \, \exp(2\pi \text{i} n j/N), &&\qquad n=0,\dots,M-1,

where :math:`\text{i}^2 = 1`.

For :math:`n=0,\dots,N`, the following
properties hold:

.. math::
    \mathcal{T}( \mat{u} )_n &= w_n\,\mathcal{F}^{-1}(\mat{u}^\text{per})_n,\\
    \mathcal{T}^{-1} ( \mat{\hat{u}} )_n &= \mathcal{F}\Big( \frac{\mat{
        \hat{u}}^\text{per}}{\mat{w}^\text{per}} \Big)_n \, ,

These properties follow immediately from the cosine series for the even
periodic extension of :math:`u` (see :ref:`example <even-extension>`). As a
result, one can compute the Chebyshev transform with the fast Fourier transform
(FFT) in :math:`\mathcal{O}(N \log N)` time. We can verify these properties by
writing out the Fourier transform and using Euler's formula along with the
evenness of :math:`\cos`.

Convolution
-----------

The Chebyshev transform starts from an even periodic extension of the
data. Furthermore, for :math:`\mathbf{u}, \mathbf{v} \in \reals^{N+1}`, the
convolution of an even periodic extension is also even:

.. math::
    (\mat{u}^\text{per} \circledast \mat{v}^\text{per})_{N-k} =
    (\mat{u}^\text{per} \circledast \mat{v}^\text{per})_{N+k},
    \qquad k=1,\dots,N-1 \, .

Accordingly, we define the Chebyshev convolution
:math:`\overset{\small \text{T}}{\circledast}` as the (truncated) convolution
of even periodic extensions:

.. math::
    (\mat{u} \overset{\small \text{T}}{\circledast} \mat{v})_n =
    (\mat{u}^\text{per} \circledast\mat{v}^\text{per})_n,\qquad n=0,\dots,N\, .

By using even periodic extensions and keeping the books on :math:`\mathbf{w}`,
we can straightforwardly apply the Fourier convolution theorem to obtain

.. math::
    \mathcal{T}(\mat{u} \overset{\small \text{T}}{\circledast} \mat{v}) &=
        \frac{2N}{\mat{w}} \odot \mathcal{T}(\mat{u}) \odot
        \mathcal{T}(\mat{v})\, , \\
    \mathcal{T}^{-1}(\mat{w} \odot (\mat{\hat{u}} \overset{\small \text{T}}{
        \circledast} \mat{\hat{v}})) &= \Big( \mathcal{T}^{-1}(\mat{w} \odot
        \mat{\hat{u}}) \Big) \odot \Big( \mathcal{T}^{-1}(\mat{w} \odot
        \mat{\hat{v}}) \Big) \, .


.. _chebyshev-transform-inner-product:

:math:`L^2` Inner Product
-------------------------

Recall that Chebyshev transform of :math:`\mathbf{f} \in \reals^{N+1}` gives
the coefficients of the degree-:math:`N` Chebyshev interpolant. Therefore,

.. math::
    \mathcal{T}[\mathbf{f}]_n = \frac{\int_{-1}^1 T_n(x) f(x) \dx{\mu}}
        {\int_{-1}^1 (T_n(x))^2 \dx{\mu}} \,

where :math:`\mu` is the Chebyshev measure. Furthermore, since

.. math::
    \int_{-1}^1 (T_n(x))^2 \dx{\mu} = \begin{cases} \pi, & n=0 \\
    \pi/2, & n>0 \end{cases} \, ,

we can succinctly write

.. math::
    \int_{-1}^1 T_n(x) f(x) \dx{\mu} =
    \begin{cases} \pi \, \mathcal{T}[\mathbf{f}]_n, & n=0 \\
    (\pi/2) \, \mathcal{T}[\mathbf{f}]_n, & n>0 \end{cases} \ .
"""
