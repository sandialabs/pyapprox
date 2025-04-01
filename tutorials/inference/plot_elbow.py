r"""
Evidence Lower Bound
====================

Want to find a distribution in a family of distributions that is close to the posterior :math:`p(z\mid x)`

.. math:: q(z) = \text{arg} \min_{q(z)\in\mathcal{Q}} D_{\text{KL}}[q(z)\lVert p(z\mid x)]

.. math::

    D_{\text{KL}}[q(z) \lVert p(z | x)] &= \int q(z) \log \frac{q(z)}{p(z\mid x)} \;\mathrm{d}z\\
    &= \int q(z) \log q(z) - q(z) \log {p(z\mid x)} \;\mathrm{d}z\\
    & = \mathbb{E}_{q(z)}\left[ \log q(z)\right] - \mathbb{E}_{q(z)}\left[\log {p(z \mid x)} \right]\\
    & = \mathbb{E}_{q(z)}\left[ \log q(z)\right] - \mathbb{E}_{q(z)}\left[\log \frac{p(x, z)}{p(x)} \right]\\
    & = \mathbb{E}_{q(z)}\left[ \log q(z)\right] - \mathbb{E}_{q(z)}\left[\log p(x, z)-\log p(x) \right]\\
    &= \underbrace{\mathbb{E}_{q(z)}[ \log q(z)] - \mathbb{E}_{q(z)}[\log p(z, x)]}_{-\text{ELBO}(q)} + \log p(x).

Thus

.. math:: \mathcal{L}^A(q) = \text{ELBO}(q) =\mathbb{E}_{q(z)}[\log p(z, x)]-\mathbb{E}_{q(z)}[ \log q(z)]

and

.. math:: \log p(x) = \text{ELBO}(q) + D_{\text{KL}}[q(z) \lVert p(z | x)]

So ELBO stands for evidence lower bound because it bounds the evidence from below, i.e.

.. math::  \text{ELBO}(q) \ge \log p(x)


Jensen's inequality can also be used to derive the ELBO. Jensen's inequality states for any random variable X

.. math:: \log \mathbb{E}[X] \ge \mathbb{E}[\log X]

Thus

.. math::

   \log p(x) &= \log \int p(z, x) \;\mathrm{d}z\\
             & = \log \int p(z, x) \frac{q(z)}{q(z)} \;\mathrm{d}z\\
             & = \log \mathbb{E}_{q(z)}\left[ \frac{p(z, x)}{q(z)} \right]\\
             & \ge \underbrace{\mathbb{E}_{q(z)} \log \left[\frac{p(z, x)}{q(z)}\right]}_{\text{ELBO}(q)}


Kingma actually presents two estimators, which he denotes :math:`\mathcal{L}^A` and :math:`\mathcal{L}^B`. We have presnted the former, however often the ELBO is presented in a different form by performing the following manipulations

.. math::

    \mathcal{L}^B(q) = \text{ELBO}(q) &=  \mathbb{E}_{q(z)}[\log p(z, x)]-\mathbb{E}_{q(z)}[ \log q(z)]\\
            &=  \mathbb{E}_{q(z)}[\log p(x\mid z)p(z)]-\mathbb{E}_{q(z)}[ \log q(z)]\\
            &=  \mathbb{E}_{q(z)}[\log p(x\mid z)]+\mathbb{E}_{q(z)}[\log p(z) - \log q(z)]\\
            &=  \mathbb{E}_{q(z)}[\log p(x\mid z)]+\int {q(z)}\log \frac{p(z)}{q(z)}\;\mathrm{d}z\\
            &=  \mathbb{E}_{q(z)}[\log p(x\mid z)]-D_{\text{KL}}[q(z) \lVert p(z)]

Here we used

.. math:: D_{\text{KL}}[q(z) \lVert p(z)] = \int {q(z)}[\log \frac{q(z)}{p(z)}\;\mathrm{d}z = -\int {q(z)}\log \frac{p(z)}{q(z)}\;\mathrm{d}z\\


Reparameterization Trick
------------------------
The reparameterization trick is used to we can compute the gradient of the ELBO with respect to :math:`\theta`

Let

.. math::
    \epsilon &\sim p(\epsilon)\\
    z &= g_\theta(\epsilon, x)

Then letting :math:`x^{(i)}` denote the ith sample of the data and  :math:`z^{(i)}` denote the ith sample of the random variable 

.. math::
    \mathbb{E}_{p_{\theta}(z)}[f(z^{(i)})] = \mathbb{E}_{p(\epsilon)} [f(g_{\theta}(\epsilon, x^{(i)}))]

The gradient of a scalar function :math:`f`, is

.. math::

    \nabla_{\theta} \mathbb{E}_{p_{\theta}(z)}[f(z^{(i)})] &= \nabla_{\theta} \mathbb{E}_{p(\epsilon)} [f(g_{\theta}(\epsilon, x^{(i)}))]\\
    &= \mathbb{E}_{p(\epsilon)} [\nabla_{\theta} f(g_{\theta}(\epsilon, x^{(i)}))]\\
    &\approx \frac{1}{L} \sum_{l=1}^{L} \nabla_{\theta} f(g_{\theta}(\epsilon^{(l)}, x^{(i)}))

Thus

.. math::

    \nabla_{\theta, \phi} \mathcal{L}^B = \nabla_{\theta, \phi} \underbrace{\left[ \frac{1}{L} \sum_{l=1}^{L} \left( \log p_{\boldsymbol{\theta}}(x^{(i)} \mid z^{(l)}) \right)\right]}_{\text{Estimate with Monte Carlo}} - \nabla_{\theta, \phi} \underbrace{\left[\text{KL}[q_{\phi}(z) \lVert p_{\theta}(z)]\right]}_{\text{Analytically evaluate}}
"""
