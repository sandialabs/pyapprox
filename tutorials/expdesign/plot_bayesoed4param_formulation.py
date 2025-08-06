r"""
Bayesian Optimal Experimental Design
====================================

Goal
----
This tutorial demonstrates how to perform Bayesian Optimal Experimental Design (OED) that maximizes the expected information gain of the experiments when using a Gaussian likelihood function.

Formulation
-----------
We begin with a candidate pool of :math:`k` observations :math:`\obsv\in\reals^K` that noisy obsevations predicted by a observational model :math:`\vec{f}(\rvv):\reals^{D}\to\reals^K` parameterized by :math:`D`-random variables :math:`\rvv=[\rv_1,\ldots,\rv_d]\in\reals^D`. These observations satsify

.. math:: \obsv = \vec{f}(\rvv) + \vec{\epsilon}

where :math:`\vec{\epsilon}\sim\mathcal{N}(\vec{0}_K,\mat{\Gamma})` is mean zero Gaussian noise with covariance :math:`\mat{\Gamma}\in\reals^{K\times K}`.

Bayesian OED is then interested in finding the weights
:math:`\vec{w}=[w_1,\ldots,w_N]\in\reals^K` that determine the importance of each
observation in the log-likelihood

.. math:: \log p(\obsv\mid\rvv,\vec{w})=-\frac{1}{2}r(\obsv, \rvv)^\top \mat{W}^{1/2}\inv{\mat{\Gamma}} \mat{W}^{1/2}r(\obsv, \rvv)+C_w,
where

.. math::  C_w=-\frac{1}{2} \log |\mat{W}^{-1/2}\mat{\Gamma} \mat{W}^{-1/2}|+C
=\frac{1}{2} \log |\mat{W}^{1/2}\inv{\mat{\Gamma}} \mat{W}^{1/2}|+C

is a constant that depnds on :math:`\vec{w}`, 
:math:`\mat{W}=\text{Diag}[\vec{w}]\in\reals^{K \times K}`, :math:`r(\obsv, \rvv)=[r_1,\ldots,r_K]=\vec{f}(\rvv)-\obsv\in\reals^{K}` and the
constant :math:`C` that does not depend on :math:`\mat{W}`.

The weights are determined by maximixing the expected information gain :math:`U(\vec{w})`, that is

.. math:: \argmax_{\vec{w}\in[0, 1]^K} \;U(\vec{w}) \quad\text{such that}\quad \sum_{k=1}^K w_k=1.

The constraint ensures that the weights define a probability measure over the design space.

The expected information gain typically cannot be analytically and so it is often approximated using MC quadrature. Specifically, drawing  :math:`M` samples :math:`\rvv^{(m)}\sim p(\rvv)` from the prior distribution of the random model parameters  [Ryan_JCGS_2003]

.. math:: U(\vec{w})=\inv{M}\sum_{m=1}^M \left\{\log\left[p(\obsv^{(m)}\mid \rvv^{(m)}, \vec{w})\right] -\log\left[\inv{N}\sum_{n=1}^N p(\obsv^{(m)}\mid \rvv^{(n)}, \vec{w})\right]\right\},

where the :math:`N` samples :math:`\rvv^{(n)}\sim p(\rvv)` are drawn from the prior independently from the :math:`M` samples :math:`\rvv^{(m)}`.

First, let's focus on computing the the first term in the expected information gain :math:`U`

.. math:: \inv{M}\sum_{m=1}^M \log \left[p(\obsv^{(m)}\mid \rvv^{(m)}, \vec{w})\right],

which we can compute with the following steps:

1. Draw the samples :math:`\rVv=[\rvv^{(1)}, \ldots, \rvv^{(M)}]\in\reals^{D\times M}` from the prior, where :math:`\rvv^{(m)}\sim p(\rvv)`.
2. Draw samples :math:`\mat{Y}=[\obsv^{(1)},\ldots,\obsv^{(m)}]\in\reals^{K\times M}`, with :math:`\obsv^{(m)}\sim p(\obsv \mid \rvv, \vec{w})`, from the likelihood 
   A) Evaluate the model at these samples to obtain the model predictions :math:`\mat{F}=[\vec{f}^{(1)}, \ldots, \vec{f}^{(M)}]\in\reals^{K\times M}`, where :math:`\vec{f}^{(m)}=\vec{f}(\rvv^{(m)})`.
   B) Randomly draw noise :math:`\mat{E}=[\vec{\epsilon}^{(1)}, \ldots, \vec{\epsilon}^{(M)}]\in\reals^{K\times M}` independently from the noise distribution, where :math:`\vec{\epsilon}^{(m)}\sim \mathcal{M}(\vec{0}_K,\mat{\Gamma})` independently from the noise distribution.
   C) Generate the observations :math:`\mat{Y}` by adding noise to the model predictions such that :math:`\obsv^{(m)}=\vec{f}^{(m)}+\vec{\epsilon}^{(m)}`.
3. Compute the log of the Gaussian likelihood using the observations :math:`\mat{Y}` and the estimates of the mean of the likelihood :math:`\mat{F}`, i.e. the model predictions.
   A) Compute the residuals for each observation
   :math:`\mat{R}=\mat{Y}-\mat{F}\in\reals^{K\times M}`
   B) Compute the cholesky decomposition of the noise covariance
   :math:`\mat{\Gamma}=\mat{L}\mat{L}^\top`, where :math:`\mat{L}\in\reals^{K\times K}` is a lower-triangular matrix
   C) Multiply the residuals by the weigthed inverse of the Cholesky factor
   :math:`\mat{P}=\inv{\mat{L}}\mat{W}^{1/2}\mat{R}\in\reals^{K\times M}`
   D) Compute the log likelihood values
   :math:`\vec{l}=[l_1,\ldots,l_M]=-\text{Diag}\left[\mat{P}^\top\mat{P}\right]+C_w\in\reals^{M}`, where :math:`l_m=C_w+-\sum_{k=1}^K P_{mk}P_{km}` or using numpy's enisum `l-C_w=einsum("ij,ji->i", -P, P.T)`
3. Compute the average :math:`\inv{M}\sum_{m=1}^M l_m`

The second summation, which approximates the evidence,

.. math:: p(\obsv)=\inv{N}\sum_{n=1}^N p(\obsv^{(m)}\mid \rvv^{(n)}, \vec{w}),

is computed using a similar process. However, we do not compute new observations, but rather resue the observations :math:`\mat{Y}` from the previous steps. Specifically, the second summation is computed using the following steps:

1. Draw :math:`N` new samples :math:`\rVv^\star` from the prior.
2. Evaluate the model at these samples to obtain the model predictions :math:`\mat{F}^\star`.
3. Compute the likelihood using the observations :math:`\mat{Y}` (computed already using :math:`\mat{F}` - not :math:`\mat{F}^\star`) and the model predicions :math:`\mat{F}^\star`
   A) Compute the residuals :math:`\mat{R}^\star=\mat{Y}-\mat{F}^\star`
   B) Compute :math:`\mat{P}^\star=\inv{\mat{L}}\mat{W}^{1/2}\mat{R}^\star`
   C) Compute the log-likelihood values :math:`\vec{l}^\star=-\text{Diag}\left[(\mat{P}^\star)^\top\mat{P}^\star\right]+C_w`
   D) Compute the likelihood values :math:`\vec{v}=\exp(\vec{l})`
   E) Compute  :math:`\inv{N}\sum_{n=1}^N v_n^\star`

Note we can use quadrature instead of MC to estimate the intergals. Letting :math:`\vec{\eta}=[\eta_1,\ldots, \eta_M]\in\reals^M` be the quadrature weights associated witht the quadrature samples :math:`\mat{Q}=[\vec{q}^{(1)}, \ldots, \vec{q}^{(M)}]\in\reals^{(D+K)\times M}` and :math:`\vec{\eta}^\star` be the quadrature weights associated with the sampels :math:`\mat{Q}^\star` then we can approximate the first intergral and second integrals as

.. math:: \sum_{m=1}^M l_m\eta_m\quad\text{and}\quad \sum_{n=1}^N v^\star_m\eta_n^\star,

respectively. Note that we are using the quadrature samples to integrate with respect to the joint density of the samples :math:`\rvv` and the obsevations :math:`\obsv`. So, wgen using a Gaussian likelihoods the dimension of the integral is the dimension :math:`D` of the parameters plus the dimension :math:`K` of the observations. Thus, in the steps above we compute the observations using

.. math:: \obsv^{m} = \vec{f}(\vec{q}^{(m)}_\rv)+\vec{q}^{(m)}_\epsilon,

where we have split the quadrature samples :math:`\vec{q}^{(m)}=[\vec{q}^{(m)}_\rv, \vec{q}^{(m)}_\epsilon]` into parts integrating with respect to the prior density and the noise density, respectively.

Gradients of the the expected information gain with respect to the weights
--------------------------------------------------------------------------

Graidient of the Gaussian Likelihood assuming Independent Noise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When the noise is independent such that :math:`\mat{\Gamma}=\text{Diag}[\vec{\gamma}]\in\reals^{K\times K}`, where :math:`\vec{\gamma}=[\gamma_1,\ldots,\gamma_K]\in\reals^K` then

.. math:: l(\mat{W})=-\frac{1}{2}\mat{R}^\top \mat{W}\inv{\mat{\Gamma}}\mat{R}-\log |\mat{W}^{1/2}\inv{\mat{\Gamma}}_\epsilon \mat{W}^{1/2}|+C=l_1(\vec{w})+l_2(\vec{w})+C
so

.. math:: \nabla_{w_k} l_1(\vec{w})=-\frac{1}{2}\nabla_{w_k} \sum_{k=1}^K r_k  w_k\inv{\gamma}_k r_k=-\frac{1}{2}r_k^2\inv{\gamma}_k

.. math:: \nabla_w l_1(\vec{w}) = -(\frac{1}{2}\text{Diag}[\vec{\gamma}]^{-1}\circ \mat{R}^2)^\top,
where :math:`\circ` is the elementwise Hadamard product and :math:`\vec{R}^2` is applied elementwise.

Additionally, noting that for independent noise, :math:`\mat{W}^{1/2}\inv{\mat{\Gamma}}_\epsilon \mat{W}^{1/2}` is a diagonal matrix so its log determinant is the sum of the logs of its entries, that is

.. math:: l_2(\vec{w})=\frac{1}{2}\sum_{k=1}^K\log\left[\frac{w_k}{\gamma_k}\right] + C = \frac{1}{2}\sum_{k=1}^K\log\left[w_k\right]-\log\left[\gamma_k\right]


Thus, 

.. math:: \nabla_{w_k} l_2(\vec{w})=\frac{1}{2\vec{w}}


Lastly, the hessian of the log likelihood applied to a vector :math:`\vec{\xi}` is zero, i.e.

.. math:: (\nabla^2_w l)\vec{\xi} =0.

Gradient of the log of the evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To compute the gradient of the log evidence, first recall the chain rule for :math:`v = g\circ l`, where :math:`g:\reals\to\reals` and :math:`l:\reals^K\to\reals`, 

.. math:: \nabla_\vec{w} v(\vec{w}) = \nabla_w g(l(\vec{w}))=\dydx{g}{l}(l(\vec{w}))\dydx{l}{\vec{w}}(\vec{w})

and note

.. math:: g(l)=\log(l), \qquad \dydx{\log(l)}{l}=\frac{1}{l}

so that writing the liklihood :math:`v(\vec{w})` as a function of the design weights :math:`\vec{w}`

.. math::  v(\vec{w})=\sum_{n=1}^N \exp\left(l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})\right)\qquad \dydx{l}{\vec{w}}=\sum_{n=1}^N  \exp\left(l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})\right)\nabla_\vec{w} l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})

The gradient of the likelihood with resepect to the weights is

.. math:: 
    \begin{align}\nabla_\vec{w} \log&\left[\inv{N}\sum_{n=1}^N \exp\left(l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})\right)\right]=-\nabla_\vec{w} \log(N) + \nabla_\vec{w} \log\left[\sum_{n=1}^N \exp\left(l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})\right)\right]\\&=\inv{\left[\sum_{n=1}^N \exp\left(l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})\right)\right]}\left[\sum_{n=1}^N \exp\left(l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})\right)\nabla_\vec{w} l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})\right]
    \end{align}

Hessian vector product for the log of the evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To compute the Hessian vector product recall the chain rule for a Hessian with :math:`l:\reals^n\to\reals`, :math:`g:\reals\to\reals` and :math:`v(\vec{w})=g(l(\vec{w}))`

.. math:: \nabla^2_\vec{w} v(\vec{w})=g'(l(\vec{w}))\nabla^2_\vec{w} l(\vec{w})+ g''(l(\vec{w}))\nabla_\vec{w} l(\vec{w})^\top\nabla_\vec{w} l(\vec{w})

Using the expression we derived for the log of the evidence we observe that for a vector :math:`\vec{\xi}\in\reals^K`

.. math::  \nabla_\vec{w} v \cdot \vec{\xi} = \sum_{n=1}^N  \exp\left(l(\vec{w}; \obsv^{(m)}, \rvv^{(n)})\right)\nabla_\vec{w} l(\vec{w}; \obsv^{(m)}, \rvv^{(n)}) \cdot \vec{\xi},

which we can differentiate again to obtain the Hessian vector product

.. math:: 
    \begin{align}
    \nabla_\vec{w}^2 v(\vec{w})\cdot \vec{\xi}&=\inv{N}\sum_{n=1}^N \nabla_\vec{w}\left[\exp(l(\vec{w}; \obsv^{(m)},\rvv^{(n)}))\nabla_\vec{w} l(\vec{w}; \obsv^{(m)},\rvv^{(n)})\cdot \vec{\xi}\right]\\
    &=\inv{N}\sum_{n=1}^N  \exp(l(\vec{w}; \obsv^{(m)},\rvv^{(n)}))\nabla_\vec{w}l(\vec{w}; \obsv^{(m)},\rvv^{(n)})^\top \nabla_\vec{w} l(\vec{w}; \obsv^{(m)},\rvv^{(n)})\cdot \vec{\xi} \\& \qquad+\exp(l(\vec{w}; \obsv^{(m)},\rvv^{(n)}))\nabla_\vec{w}^2 l(\vec{w}; \obsv^{(m)},\rvv^{(n)})\cdot \vec{\xi}\\
    &=\inv{N}\sum_{n=1}^N  \exp(l(\vec{w}; \obsv^{(m)},\rvv^{(n)}))\nabla_\vec{w}l(\vec{w}; \obsv^{(m)},\rvv^{(n)})^\top \nabla_\vec{w} l(\vec{w}; \obsv^{(m)},\rvv^{(n)})\cdot \vec{\xi}
  \end{align}

where we used the Hessian chain rule to yield the second equlaity and the last equality holds because as we showed :math:`\nabla^2_\vec{w} l(\vec{w}) \xi = 0`.
"""
