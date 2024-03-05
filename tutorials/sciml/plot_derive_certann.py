r"""
CERTANN Derivation
==================
CERTANNs are derived from the observation that in the limit of infinite width a
neural network can be expressed as sequence of intergral operators [RB2007]_.
Specifically, each layer of a CERTANN, with :math:`K` layers, that
approximates a function :math:`f(x):\reals^{D}\to\reals^{Q}` has the
continuous form

.. math::

   y_{k+1}(z_{k+1})&=\sigma_k\left(\int_{\mathcal{D}_{k}} \mathcal{K}_{k}(
        z_{k+1}, z_{k}; \theta_{k}) y_{k}(z_{k}) \dx{\mu_{k}(z_{k})})\right) \\
   &=\sigma_k\left(u_{k+1}(z_{k+1})\right),

where for :math:`k=0,\ldots, K-1`,

* :math:`\mathcal{D}_{k} \subset \reals^{D_k}`,
* :math:`\sigma_k:\reals\to\reals`,
* :math:`\mathcal{K}_k : \mathcal{D}_{k+1} \times \mathcal{D}_k \to \reals`,
  and
* :math:`y_k : \mathcal{D}_k \to \reals`.

To construct CERTANNs, we discretize the above integrals with quadrature so
that

.. math::

   u_{k+1}(z_{k+1})\approx \sum_{n=1}^{N_k} \mathcal{K}_{k}(z_{k+1}, z_k^{(n)};
   \theta_{k}) y_k(z_k^{(n)}) w_k^{(n)}.

We then discretize :math:`z_{k+1}` with another quadrature rule such that
:math:`\mat{K}_k\in\reals^{N_{k+1}\times N_k}` has entries
:math:`(\mat{K}_{k})_{m,n}=\mathcal{K}_{k}(z_{k+1}^{(m)}, z_k^{(n)})`,
:math:`\mat{W}_k=\mathrm{Diag}(w_k^{(1)},\ldots,w_{k}^{N_k})\in\reals^{N_k
\times N_k}` and :math:`\mat{y}_k=[y_k(z_k^{(1)}), \ldots, y_k(z_k^{(N_k)})
]^\top\in\reals^{N_k\times P}`, to obtain

.. math::

   \mat{u}_{k+1}&=\mat{K}_{k}\mat{W}_k\mat{y}_k&\in\reals^{N_{k+1}\times P}, \\
   \mat{y}_{k+1} &= \sigma(\mat{u}_{k+1}) &\in\reals^{N_{k+1}\times P},

where :math:`\sigma(\cdot)` acts elementwise. Special treatment must be given
to the input and output layers. When passing :math:`P` samples to the input
layer,

.. math::
    \mat{y}_0=\mat{x}\in\reals^{N_0\times P}, \qquad N_0=D \qquad
    \mathrm{and}\qquad W_0 = \mat{I}_{N_0}\in\reals^{N_0\times N_0},

where :math:`\mat{I}_{N_0}` is the identity matrix with :math:`N_0` diagonal
entries. For the final layer, the number of quadrature points must be equal to
the dimension :math:`Q` of the output :math:`f(x)`, that is :math:`N_K=Q`.

For a CERTANN with a single layer with no activation function applied to the
output layer, the discretized representation of each layer is

.. math::
   \mat{u}_{1} &= \mat{K}_{0}\mat{x}, &\qquad (\mat{K}_0)_{m,n}=\mathcal{K}_{0}
        (z_{1}^{(m)}, x^{(n)}) \qquad \mat{K}_0\in\reals^{N_1\times N_0}\qquad
        \mat{u}_1\in\reals^{N_1\times P} \\
   \mat{y}_{1} &= \sigma(\mat{u}_{0})& \\\
   \mat{u}_{2} &= \mat{K}_{1}\mat{W}_{1}\mat{y}_1&\qquad (\mat{K}_1)_{m,n}=
        \mathcal{K}_{1}(z_{2}^{(m)}, z_{1}^{(n)})\qquad \mat{K}_1\in\reals^{N_2
        \times N_1}\qquad \mat{u}_1\in\reals^{N_2\times P} \\
   \mat{y}_{2} &= \mat{u}_{2}&


Fourier Neural Operators
------------------------
Fourier Neural Operators (FNOs) [LKAKBSA2021]_ are a special case of CERTANNS
that set

.. math::
    y_{k+1}(z_{k+1}) = \sigma\left(\mathcal{W}_k \, y_k(z_{k}) +
    \int_{\mathcal{D}_k} \mathcal{K}_{k}(z_{k+1},z_k) y_{k}(z_{k})
    \dx{\mu_k(z_k)} \right)

where :math:`\mathcal{W}_k` is an affine transformation. In the original paper,
Li et al. introduce :math:`\mathcal{W}_k` to "track [... the] non-periodic
boundary.'' Also, the original paper maps :math:`y_k` into :math:`d_v` channels
**before** discretization, effectively using the continuous hidden layers

.. math:: \tilde{y}_k (z_k) = P(y_k(z_k)) \in \reals^{d_v},

where :math:`P: \reals \to \reals^{d_v}` is a lifting operator, typically a
shallow fully connected network. In contrast, we assume :math:`y_k: \reals \to
\reals`, and the quadrature discretization determines the shape of the network.

FNOs make the specific choice that :math:`\mathcal{K}_k` is a periodic
band-limited kernel with maximum frequency :math:`T_k`. Then efficient
integration can occur with the Fourier transform :math:`\mathcal{F}` and its
inverse :math:`\mathcal{F}^{-1}`. Specifically, FNOs compute

.. math::

    u_{k+1}(z_{k+1}) &= \mathcal{F}^{-1}\left(\mathcal{F}\mathcal{K}_{k}
        (z_{k+1},z_k) \odot \mathcal{F}y_k(z_{k+1})\right) \\
    &= \left( \mathcal{F}^{-1}\left(\mathcal{R}_{\theta_k} \odot \mathcal{F}y_k
        \right) \right)(z_{k+1}) \, .

The subscript :math:`\theta_k` denotes that the Fourier transform of the
kernel depends on hyper-parameters :math:`\theta_k`, which must be optimized,
and :math:`\odot` denotes elementwise multiplication.

In principle, FNOs permit an arbitrary discretization of the integral. In
practice, to use the Fast Fourier Transform (FFT), the domain of integration
:math:`\mathcal{D}_k` is discretized with :math:`N_k` points equidistantly
sampled in each dimension (:math:`s_{k,1} \times s_{k,2} \times \cdots \times
s_{k,D_k}= N_k`), and we denote the discretized transform as :math:`\mat{F}_k
\in \mathbb{C}^{N_k \times N_k}`. To perform projection into (and lifting from)
bandlimited space, we define

.. math::
    \mat{P}_{T_k, N_k} = [\mat{I}_{T_k} \ \ \mat{0}_{T_k \times (N_k-T_k)}] \in
    \reals^{T_k \times N_k} \, .

For :math:`\mat{y}_k \in \reals^{N_k \times P}` and :math:`\mat{R}_k =
\mathrm{Diag}(\theta_k^{(1)},\dots,\theta_{k}^{(T_k)})`,
we get

.. math::
    \mat{u}_{k+1} = \mat{F}_{k+1}^{-1} \mat{P}^{\top}_{T_k, N_{k+1}} \mat{R}_k
    \mat{P}_{T_k, N_k}\mat{F}_k \mat{y}_k\, \in\mathbb{C}^{N_{k+1}\times P}\, .

In contrast to [LKAKBSA2021]_, since we do not use a channel
embedding for :math:`y_k`, then :math:`\mat{R}_k` is not a three-way tensor. If
we take the original FNO formulation with :math:`d_v=1`, then we recover the
diagonal matrix above.


References
----------
.. [RB2007] `Le Roux and Bengio, Continuous Neural Networks. Proceedings of
   Machine Learning Research. 2007
   <https://proceedings.mlr.press/v2/leroux07a.html>`_

.. [LKAKBSA2021] `Li et al., Fourier Neural Operator for Parametric Partial
   Differential Equations. International Conference on Learning
   Representations. 2021. <https://arxiv.org/abs/2010.08895>`_
"""
