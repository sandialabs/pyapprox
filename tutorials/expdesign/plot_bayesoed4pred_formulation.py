r"""
Goal-Oriented Bayesian Optimal Experimental Design
==================================================

Goal
----

This tutorial demonstrates how to perform goal-oriented Bayesian Optimal Experimental Design (OED) that maximizes risk-aware utility functions measuring the efficacy of different experimental designs when using a Gaussian likelihood function. We begin with a candidate pool of :math:`k` observations :math:`\obsv \in \reals^K` that are noisy observations predicted by an observational model :math:`\vec{f}(\rvv): \reals^{D} \to \reals^K` parameterized by :math:`D`-random variables :math:`\rvv = [\rv_1, \ldots, \rv_d] \in \reals^D`. These observations satisfy:

.. math::
   \obsv = \vec{f}(\rvv) + \vec{\epsilon},

where :math:`\vec{\epsilon} \sim \mathcal{N}(\vec{0}_K, \mat{\Gamma})` is mean-zero Gaussian noise with covariance :math:`\mat{\Gamma} \in \reals^{K \times K}`.



Goal-oriented Bayesian OED is then interested in finding the weights :math:`\vec{w} = [w_1, \ldots, w_N] \in \reals^K` that determine the importance of each observation in the log-likelihood:

.. math::
   \log p(\obsv \mid \rvv, \vec{w}) = -\frac{1}{2} r(\obsv, \rvv)^\top \mat{W}^{1/2} \inv{\mat{\Gamma}} \mat{W}^{1/2} r(\obsv, \rvv) + C_w,

where:

.. math::
   C_w = -\frac{1}{2} \log |\mat{W}^{-1/2} \mat{\Gamma} \mat{W}^{-1/2}| + C = \frac{1}{2} \log |\mat{W}^{1/2} \inv{\mat{\Gamma}} \mat{W}^{1/2}| + C,

is a constant that depends on :math:`\vec{w}`, :math:`\mat{W} = \text{Diag}[\vec{w}] \in \reals^{K \times K}`, :math:`r(\obsv, \rvv) = [r_1, \ldots, r_K] = \vec{f}(\rvv) - \obsv \in \reals^{K}`, and the constant :math:`C` that does not depend on :math:`\mat{W}`.

The weights are determined by maximizing a utility function :math:`U(\vec{w})`, that is:

.. math::
   \argmax_{\vec{w} \in [0, 1]^K} \; U(\vec{w}) \quad \text{such that} \quad \sum_{k=1}^K w_k = 1.

The constraint ensures that the weights define a probability measure over the design space.

Unlike traditonal Bayesian OED which targets that most influence the estimation of the model parameters \(\vec{z}\), goal-oriented OED targets changes in the push-forward of the posterior parameter distribution through a model that predictions quantities of interest:

.. math:: \vec{\gamma}=q(\rvv):\reals^{D}\to\reals^Q.

Utility Function
----------------

Utility functions should be tailored to the risk-preferences of the model stakeholders. Here, we present one such choice. This choice, like all other cannot be computed analytically for most problem setuyps and so it is often approximated using Monte Carlo (MC) quadrature. Here we use a double loop procedure to estimate the utiltity motivated by [@Ryan_JCGS_2003] for traditional BOED that targets information gain in parameters. Specifically, drawing :math:`M` samples :math:`\rvv^{(m)} \sim p(\rvv)` from the prior distribution of the random model parameters, tht utility :

.. math::    U(\vec{w}) = \mathbb{E}_{p(\obsv \mid \vec{w})} \left\{ \mathbb{V}_{p(\rvv \mid \obsv, \vec{w})} \left[ q(\rvv) \right] \right\},

can be approximated as:

.. math::
   U(\vec{w}) \approx \frac{1}{M} \sum_{m=1}^M \left\{
   \frac{1}{p(\obsv^{(m)} \mid \vec{w})} \mathbb{E}_{p(\rvv)} \left[ q^2(\rvv) p(\obsv^{(m)} \mid \rvv, \vec{w}) \right]
   - \frac{1}{p(\obsv^{(m)} \mid \vec{w})^2} \mathbb{E}_{p(\rvv)} \left[ q(\rvv) p(\obsv^{(m)} \mid \rvv, \vec{w}) \right]^2
   \right\}.

The following steps can be used to approximate the utility:

1. Draw the samples :math:`\rVv = [\rvv^{(1)}, \ldots, \rvv^{(M)}] \in \reals^{D \times M}` from the prior, where :math:`\rvv^{(m)} \sim p(\rvv)`.
2. Draw samples :math:`\mat{Y} = [\obsv^{(1)}, \ldots, \obsv^{(m)}] \in \reals^{K \times M}`, with :math:`\obsv^{(m)} \sim p(\obsv \mid \rvv, \vec{w})`, from the likelihood
     A) Evaluate the model at these samples to obtain the model predictions :math:`\mat{F} = [\vec{f}^{(1)}, \ldots, \vec{f}^{(M)}] \in \reals^{K \times M}`, where :math:`\vec{f}^{(m)} = \vec{f}(\rvv^{(m)})`.
     B) Randomly draw noise :math:`\mat{E} = [\vec{\epsilon}^{(1)}, \ldots, \vec{\epsilon}^{(M)}] \in \reals^{K \times M}` independently from the noise distribution, where :math:`\vec{\epsilon}^{(m)} \sim \mathcal{N}(\vec{0}_K, \mat{\Gamma})`.
     C) Generate the observations :math:`\mat{Y}` by adding noise to the model predictions such that :math:`\obsv^{(m)} = \vec{f}^{(m)} + \vec{\epsilon}^{(m)}`.
3. Draw samples :math:`\mat{Q} = [\vec{\gamma}^{(1)}, \ldots, \vec{\gamma}^{(m)}] \in \reals^{Q \times M}`, where :math:`\vec{\gamma}^{m}=q(\rvv{(m)})`.
4. Compute the evidence for each synthetic observation.
5. Compute the deviation of the posterior push-forward for each synthetic observation.

Evidence Approximation
----------------------

We approximate the evidence:

.. math:: p(\obsv^{(m)}\mid  \vec{w})\approx\inv{N}\sum_{n=1}^N p(\obsv^{(m)}\mid \rvv^{(n)}, \vec{w}),

with the following steps:

1. Draw :math:`N` new samples :math:`\rVv^\star` from the prior.
2. Evaluate the model at these samples to obtain the model predictions :math:`\mat{F}^\star`.
3. Compute the likelihood using the observations :math:`\mat{Y}` (computed already using :math:`\mat{F}` - not :math:`\mat{F}^\star`) and the model predicions :math:`\mat{F}^\star`
   A) Compute the residuals :math:`\mat{R}^\star=\mat{Y}-\mat{F}^\star`
   B) Compute :math:`\mat{P}^\star=\inv{\mat{L}}\mat{W}^{1/2}\mat{R}^\star`
   C) Compute the log-likelihood values :math:`\vec{l}^\star=-\text{Diag}\left[(\mat{P}^\star)^\top\mat{P}^\star\right]+C_w`
   D) Compute the likelihood values :math:`\vec{v}=\exp(\vec{l})`
   E) Compute  :math:`\inv{N}\sum_{n=1}^N v_n^\star`

A similar procedure is used to approximate the deviation of the posterior push-forward for each synthetic observation. For example, we compute:

.. math::  \mathbb{E}_{p(\rvv)} \left[ q(\rvv) p(\obsv^{(m)} \mid \rvv, \vec{w}) \right]\approx\inv{N}\sum_{n=1}^N \vec{\gamma}^{(n)}p(\obsv^{(m)}\mid \rvv^{(n)}, \vec{w}).


Quadrature Approximation
------------------------

Note we can use quadrature instead of MC to estimate the integrals. Letting :math:`\vec{\eta} = [\eta_1, \ldots, \eta_M] \in \reals^M` be the quadrature weights associated with the quadrature samples :math:`\mat{Q} = [\vec{q}^{(1)}, \ldots, \vec{q}^{(M)}] \in \reals^{(D+K) \times M}` and :math:`\vec{\eta}^\star` be the quadrature weights associated with the samples :math:`\mat{Q}^\star`, then we can approximate the first integral and second integrals as follows.

Vector-Valued Quantities of Interest
------------------------------------
If :math:`\vec{\gamma}^{(n)}` is a vector, i.e. :math:`Q>1`, then we must place a measure :math:`p(q)` on the elements of the vector and apply a risk measure over the output space of :math:`q` to make sure that the utility is a scalar-valued function. In this situtation, we compute:

.. math::    U(\vec{w}) = \mathbb{E}_{p(\obsv \mid \vec{w})} \left\{\mathcal{R}_{p(q)}\left[ \mathbb{V}_{p(\rvv \mid \obsv, \vec{w})} \left[ q(\rvv) \right]\right] \right\},

For example, we can place a discrete uniform measure with equal weights for each entry of the QoI and compute the average, that is we set :math:`\mathcal{R}_{p(q)}=\mathbb{E}_{p(q)}` and compute:

.. math::  \mathcal{R}_{p(q)}\left[ \mathbb{V}_{p(\rvv \mid \obsv, \vec{w})} \left[ q(\rvv) \right]\right]\approx Q^{-1}\sum_{i=1}^Q  \mathbb{V}_{p(\rvv \mid \obsv, \vec{w})} \left[ q_i(\rvv) \right]

"""
