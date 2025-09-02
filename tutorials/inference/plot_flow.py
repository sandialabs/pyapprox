r"""
Normalizing Flows
=================
A normalizing flow is a differentiable invertible transformation :math:`T:\reals^D\to\reals^D` that maps a latent random variable :math:`u`, with pdf :math:`\pi(u)` to a target variable :math:`z` with pdf :math:`p(z)`.
Provided the transform is differentiable with Jacobian with respect to the variable u given by :math:`\nabla_u T(u)`, and the transform inverse is :math:`T^{-1}`, with Jacobian :math:`\nabla_z T^{-1}(z)`, then

.. math::

    p(\rvv) &= \pi(\vec{u})\left\lvert\text{Det}\left[\nabla_\vec{u} T(\vec{u})\right]\right\rvert^{-1}\\
         &= \pi(T^{-1}(\rvv))\left\lvert\text{Det}\left[\nabla_\rvv T^{-1}(T^{-1}(z))\right]\right\rvert

Here we used :math:`\left\lvert\text{Det}\left[\frac{dy}{dx}\right]\right\rvert=\left\lvert\frac{1}{\text{Det}\left[\frac{dx}{dy}\right]}\right\rvert.`
`

Normalzing flows typically consists of a composition of transformations :math:`T=T_K(z) \circ \cdots \circ T_1(\vec{u})` so that

.. math:: \log p(\rvv)= \log \pi(T^{-1}(\rvv))+\sum_{k=1}^K \log\left(\lvert\text{Det}\left[\nabla_\rvv T^{-1}_k(T^{-1}_k(\rvv_k))\right]\rvert\right)

where :math:`\rvv_k=T_k \circ \ldots \circ T_1(\vec{u})`

There are many invertible intermediate transforms :math:`T_k` we could use in the composition.
Here, we describe the intermediate transforms used Real Non Volume Preserving (RealNVP) flows. These intermediate transforms are often called coupling layers.

Let :math:`\rvv_A\subset \rvv`, :math:`\rvv_A\in\reals^s`, denote a subset of all variable :math:`\rvv` and let :math:`\rvv_B=\rvv\setminus \rvv_A` be the complemnt of :math:`\rvv_A`. Then RealNVP flows define the transform :math:`\rvv'=T_k(\rvv)` as

.. math::

    T_k(\rvv) = \begin{bmatrix}
    \rvv_A \\
    \rvv_B \odot \exp(\sigma_k(\rvv_A)) + \mu_k(\rvv_A)
    \end{bmatrix}

which has a simple inverse

.. math::

    T_k^{-1}(\rvv') = \begin{bmatrix}
    \rvv_A' \\
    (\rvv_B'-\mu_k(\rvv'_{A})) \odot \exp(-\sigma_k(\rvv_A'))
    \end{bmatrix}

This transformation gained popularity because inverting the transform does not require inverting the shift :math:`\mu_k:\reals^s\to\reals^{D-s}` and the scale :math:`\sigma_k:\reals^s\to\reals^{D-s}` which can be highly nonlinear functions of the untransformed variables :math:`\rvv_A`.

Because the first :math:`s` entries are unchanged by the transform the Jacobian
of the intermediate transforms are lower-triangular

.. math::

    \nabla_\rvv T_k(\rvv) = \begin{bmatrix}
    I_{s\times s} && 0_{s\times D-s}\\
    \frac{\partial \rvv'_{B}}{\partial \rvv_A} && \text{Diag}[\exp(\sigma_k(\rvv_A))]
    \end{bmatrix}

The determinant of a lower-triangular matrix is the product of its diagonals so

.. math::

    \text{Det}\left[\nabla_\rvv T_k(\rvv)\right]  = \prod_{d=1}^{D-s} \exp\left(\sigma_k(\rvv_A)\right)_d = \exp\left(\sum_{d=1}^{D-s} \sigma_k(\rvv_A)_d\right)

and for the inverse transform

.. math::

    \text{Det}\left[\nabla_{\rvv'} T_k^{-1}(\rvv')\right]  = \exp\left(-\sum_{d=1}^{D-s} \sigma_k(\rvv'_{A})_d\right)

The effectivness of such a flow depends on the ordering of the variables. So typically the ordering of the variables is changed for each intermediate transform.
Typically, we fix all variables that were transformed at the previous layer and vice-versa.

Example: Independent Gaussians
------------------------------
The following shows how to construct a flow that maps a 2D standard Normal
to an 2D independent Gaussian with scaled and shifted marginals.

Consider :math:`\rvv=[\rv_1,\rv_2], \vec{u}=[u_1,u_2]` and let :math:`s=1`.
Now let the shift and scale be:

.. math:: \sigma_k(\rvv) = \log(\tau_k) \qquad \mu_k(\rvv) = \nu_k

No substitute these expressions into:

.. math::

    T_k(\rvv) = \begin{bmatrix}
    \rvv_A \\
    \rvv_B\odot\exp(\sigma_k(\rvv_A)) + \mu_k(\rvv_B)
    \end{bmatrix}

with :math:`\rvv_A=[\rv_1]` and :math:`\rvv_B=[\rv_2]` so that:

.. math::

    \rvv_1=T_1(\vec{u})=
    \begin{bmatrix}
    u_1 \\
    u_2 \exp(\log(\tau_1)) + \nu_1
    \end{bmatrix}=
    \begin{bmatrix}
    u_1 \\
    u_2 \tau_1 + \nu_1
    \end{bmatrix}


Now let :math:`\rvv_A=[\rv_{12}]` and :math:`\rvv_B=[\rv_{11}]`

.. math::

    \rvv_2=T_2(\rvv_1)=
    \begin{bmatrix}
    u_1 \tau_2 + \nu_2\\
    u_2 \tau_1 + \nu_1
    \end{bmatrix}


Example: Correlated Gaussian
----------------------------
The following shows how to construct a flow that maps a 2D standard Normal
to a multivariate Gaussian with scaled and shifted correlated marginals.

Let the shift and scales of a two layer RealNVP with input :math:`\rvv=[\rv_1,\rv_2]` be:

.. math:: \sigma_1(\rvv) = \log(\tau_1) \qquad \mu_1(\rvv) = \nu_1 + \delta_1\rv_2

.. math:: \sigma_2(\rvv) = \log(\tau_2) \qquad \mu_2(\rvv) = \nu_2

so that:

.. math::

    \rvv_1=T_1(\vec{u})=
    \begin{bmatrix}
    u_1 \\
    u_2 \tau_1 + \nu_1 + \delta_1 u_2
    \end{bmatrix}

.. math::

    \rvv_2=T_2(\rvv_1)=
    \begin{bmatrix}
    u_1 \tau_2 + \nu_2\\
    u_2 \tau_1 + \nu_1 + \delta_1 u_1
    \end{bmatrix}


Let :math:`\rvv_2 =[Y_1, Y_2]`, then:

- :math:`Y_1` has mean :math:`\nu_2` and variance :math:`\tau_2^2`.
- :math:`Y_2` has mean :math:`\nu_1` and variance :math:`\tau_1^2 + \delta_1^2`.
- The covariance between :math:`Y_1` and :math:`Y_2` is :math:`\tau_2 \delta_1`.

Thus, the joint distribution of :math:`Y_1` and :math:`Y_2` is a bivariate normal distribution, given by:

.. math::

    \rvv_2=\begin{bmatrix}
    Y_1 \\
    Y_2
    \end{bmatrix} \sim N\left(
    \begin{bmatrix}
    \nu_2 \\
    \nu_1
    \end{bmatrix},
    \begin{bmatrix}
    \tau_2^2 & \tau_2 \delta_1 \\
    \tau_2 \delta_1 & \tau_1^2 + \delta_1^2
    \end{bmatrix}
    \right)


"""
