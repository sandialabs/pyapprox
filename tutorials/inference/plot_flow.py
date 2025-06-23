r"""
Normalizing Flows
=================
A normalizing flow is a differentiable invertible transformation :math:`T:\reals^D\to\reals^D` that maps a latent random variable :math:`u`, with pdf :math:`\pi(u)` to a target variable :math:`z` with pdf :math:`p(z)`.
Provided the transform is differentiable with Jacobian with respect to the variable u given by :math:`\nabla_u T(u)`, and the transform inverse is :math:`T^{-1}`, with Jacobian :math:`\nabla_z T^{-1}(z)`, then

.. math::

    p(z) &= \pi(u)\left\lvert\text{Det}\left[\nabla_u T(u)\right]\right\rvert^{-1}\\
         &= \pi(T^{-1}(z))\left\lvert\text{Det}\left[\nabla_z T^{-1}(T^{-1}(z))\right]\right\rvert

Here we used :math:`\left\lvert\text{Det}\left[\frac{dy}{dx}\right]\right\rvert=\left\lvert\frac{1}{\text{Det}\left[\frac{dx}{dy}\right]}\right\rvert.`
`

Normalzing flows typically consists of a composition of transformations :math:`T=T_K(z) \circ \cdots \circ T_1(u)` so that

.. math:: \log p(z)= \log \pi(T^{-1}(z))+\sum_{k=1}^K \left\lvert\text{Det}\left[\nabla_z T^{-1}_k(T^{-1}_k(z_k))\right]\right\rvert

where :math:`z_k=T_k \circ \ldots \circ T_1(u)`

There are many invertible intermediate transforms :math:`T_k` we could use in the composition.
Here, we describe the intermediate transforms used Real Non Volume Preserving (RealNVP) flows. These intermediate transforms are often called coupling layers.

Given an abitray index :math:`1<s<D` RealNVP flows define the transform :math:`z'=T_k(z)` as

.. math::

    T(z) = \begin{bmatrix}
    z_{1:s} \\
    z_{s+1:D} \odot \exp(\sigma(z_{1:s})) + \mu(z_{1:s})
    \end{bmatrix}

which has a simple inverse

.. math::

    T^{-1}(z') = \begin{bmatrix}
    z_{1:s}' \\
    (z_{s+1:D}'-\mu(z'_{1:s})) \odot \exp(-\sigma(z_{1:s}'))
    \end{bmatrix}

This transformation gained popularity because inverting the transform does not require inverting the shift :math:`\mu` and the scale :math:`\sigma` which can be highly nonlinear functions of the untransformed variables :math:`z_{1:s}`.

Because the first :math:`s` entries are unchanged by the transform the Jacobian
of the intermediate transforms are lower-triangular

.. math::

    \nabla_z T(z) = \begin{bmatrix}
    I_{s\times s} && 0_{s\times D-s}\\
    \frac{\partial z'_{s+1:D}}{\partial z_{1:s}} && \text{Diag}[\exp(\sigma(z_{1:s}))]
    \end{bmatrix}

The determinant of a lower-triangular matrix is the product of its diagonals so

.. math::

    \text{Det}\left[\nabla_z T(z)\right]  = \prod_{d=1}^{D-s} \exp\left(\sigma(z_{1:s})\right)_d = \exp\left(\sum_{d=1}^{D-s} \sigma(z_{1:s})_d\right)

and for the inverse transform

.. math::

    \text{Det}\left[\nabla_{z'} T^{-1}(z')\right]  = \exp\left(-\sum_{d=1}^{D-s} \sigma(z'_{1:s})_d\right)

The effectivness of such a flow depends on the ordering of the variables. So typically the ordering of the variables is changed for each intermediate transform.
Typically, we fix all variables that were transformed at the previous layer and vice-versa.

Example
-------
Consider \(z=[z_1,z_2], u=[u_1,u_2]\) and let \(s=1\).
Now let the shift and scale be linear functions of thier inputs. This is referred to as affine coupling.
Also it is also useful to express the coupling transform \(T_k(z)\) explicitly as a function \(h(z_A, \phi(z_B))\) of the fixed variables \(z_A\) and a nonlinear transform $\phi$ of the other variables \(z_B\). When using RealNVP layers, $h(z_A, \phi_(z_B))$
"""
