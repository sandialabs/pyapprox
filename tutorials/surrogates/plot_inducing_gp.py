r"""Constructing A Gaussian Process with Inducing Points and Variational Inference
==================================================================================

Gaussian Processes (GPs) are powerful tools for modeling functions in a probabilistic framework. They are widely used in regression, classification, and optimization tasks. However, standard GPs scale poorly with the number of data points :math:`n`, as their computational complexity is :math:`\mathcal{O}(n^3)`. To address this, **inducing points** are introduced as a sparse approximation method to reduce computational costs.

This tutorial demonstrates how to construct a GP with inducing points and variational inference using the provided `InducingSamples` and `InducingGaussianProcess` classes. We will cover the mathematical foundations, demonstrate how to use the classes, and show how to optimize the GP model.

Gaussian Processes
------------------

A Gaussian Process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. It is fully specified by a mean function :math:`m(\mathbf{x})` and a covariance function :math:`k(\mathbf{x}, \mathbf{x}')`:

.. math::

    \vec{y} \sim \mathcal{GP}(m(\vec{x}), k(\vec{x}, \vec{x}')).

In regression tasks, the GP models the relationship between inputs :math:`\mat{X}` and outputs :math:`\vec{y}` taking the form


.. math::

    \vec{y} =  \vec{f}(\mat{X}) + \vec{\epsilon},

where :math:`\V{\epsilon}=\sigma^2\mat{I}` is independent Gausian noise.


Given training data, the posterior distribution of the GP can be used to make predictions at new input points :math:`\mat{X}'` and is a Gaussian distribution with mean and variance given by:

.. math:: \mu_{\vec{y}}(\mat{X}') =\mat{K}(\mat{X}',\mat{X})(\mat{K}(\mat{X},\mat{X})+\sigma^2\mat{I})^{-1}\vec{y},


.. math:: K_{\vec{y}}(\mat{X}', \mat{X}') =\mat{K}(\mat{X}',\mat{X}')-\mat{K}(\mat{X}',\mat{X})(\mat{K}(\mat{X},\mat{X})+\sigma^2\mat{I})^{-1}\mat{K}(\mat{X},\mat{X}')



Log Marginal Likelihood for an Exact Gaussian Process
-----------------------------------------------------

Training an exact GP, requires maximizing the log marginal likelihood, which quantifies how well the GP explains the observed data. It is derived from the joint Gaussian distribution of the observed outputs :math:`\vec{y}` and the latent function values :math:`\vec{f}`. The log marginal likelihood is given by:

.. math::

    \log p(\vec{y} \mid \mathbf{X}) = -\frac{1}{2} \vec{y}^\top \mat{K}^{-1} \vec{y} - \frac{1}{2} \log \mid\mat{K}\mid - \frac{n}{2} \log(2\pi),

where:

- :math:`\mat{K}` is the covariance matrix of the GP prior, computed using the kernel function :math:`k(\mathbf{x}_i, \mathbf{x}_j)`,
- :math:`\vec{y}` is the vector of observed outputs,
- :math:`n` is the number of data points.


Components of the Log Marginal Likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The log marginal likelihood consists of three terms:

1. **Data Fit Term**:

   .. math::

       -\frac{1}{2} \vec{y}^\top \mat{K}^{-1} \vec{y}

   This term measures how well the GP explains the observed data. It penalizes deviations of the observed outputs :math:`\vec{y}` from the GP's predictions.

2. **Complexity Penalty**:

   .. math::

       -\frac{1}{2} \log \mid\mat{K}\mid

   This term penalizes overly complex models by incorporating the determinant of the covariance matrix :math:`\mat{K}`. Larger determinants correspond to simpler models.

3. **Normalization Constant**:

   .. math::

       -\frac{n}{2} \log(2\pi)

   This term ensures that the log marginal likelihood is properly normalized.


Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~~

The computation of the log marginal likelihood involves:

1. **Matrix Inversion**:
   The term :math:`\mat{K}^{-1}` requires inverting the covariance matrix :math:`\mat{K}`, which scales as :math:`\mathcal{O}(n^3)`.

2. **Log Determinant**:
   The term :math:`\log \mid\mat{K}\mid` requires computing the determinant of :math:`\mat{K}`, which also scales as :math:`\mathcal{O}(n^3)`.

Due to this cubic complexity, exact GPs become computationally expensive for large datasets.



Sparse Approximation with Inducing Points
-----------------------------------------
Inducing points can be used to reduce the cost of training and predicting with Gaussian processes for large amounts of training data.
Inducing points are a small set of pseudo-inputs :math:`\vec{u}` that summarize the information in the full dataset.

**Definition of Inducing Variables and Input Locations**

1. Inducing Variables :math:`\vec{u}`:

   .. math:: \vec{u} = [u_1, u_2, \ldots, u_m]  \in \mathbb{R}^m,

   where:

   - :math:`u_i=f(\vec{z}_i)` represents the latent function value at the inducing input location :math:`z_i`,
   - :math:`m` is the number of inducing input locations.

2. Inducing Input Locations :math:`\mathbf{Z}`:

    .. math:: \mathbf{Z} = [\vec{z}_1, \vec{z}_2, \ldots, \vec{z}_m] \in \mathbb{R}^{m \times d},

   where:

   - :math:`z_i \in \mathbb{R}^d` is a :math:`d`-dimensional input location (e.g., in a :math:`d`-dimensional feature space),
   - :math:`m` is the number of inducing input locations.

The covariance matrix of the GP is approximated using these inducing points, reducing the computational complexity to :math:`\mathcal{O}(nm^2 + m^3)`, where :math:`m` is the number of inducing points (:math:`m \ll n`).


**Problem Setup

We aim to compute the log marginal likelihood of the observed data :math:`\vec{y}` given the inputs :math:`\mathbf{X}`. Using marginalization, we can make write the predictive Gaussian posterior distribution of a GP, for function values :math:`\vec{f}'` at unseen inputs, without explicitly conditioning on the locations  :math:`\mat{X}`, that is

.. math:: p(\vec{f}'\mid \vec{y})=\int p(\vec{f}'\mid \vec{f}) p(\vec{f}\mid \vec{y}) \text{d}\vec{f}

Similarly,  using inducing variables :math:`\vec{u}` evaluated at the inducing points :math:`\mat{Z]`, we can write:

.. math:: p(\vec{f}'\mid \vec{y})=\int p(\vec{f}'\mid \vec{f},\vec{u}) p(\vec{f}\mid\vec{u}, \vec{y})p(\vec{u}\mid\vec{y}) \text{d}\vec{f}\, \text{d}\vec{u},


where:

- :math:`\vec{f}` are the latent function values at the training points :math:`\mathbf{X}`,
- :math:`\vec{u}` are the inducing points.

Direct computation of this integral is intractable due to the high dimensionality of :math:`\vec{f}` and :math:`\vec{u}`. **Variational inference** provides an approximation by introducing a variational posterior :math:`q(\vec{u})` over the inducing points.


Variational Approximation
--------------------------
Variational inference can be used to approximate the posterior distribution of the inducing points.  Here we seek a variational distribution that factorizes as

.. math:: q_\gamma(\vec{f}',\vec{u})=q(\vec{f}'\mid\vec{u})q_\gamma(\vec{u})

where :math:`\gamma` are the hyperparamters of the distribution that must be optimized. Here we have dropped dependnce on the data :math:`\vec{y}` for convenience; we will reintroduce it later.

Inducing GPs assume that :math:`q_\gamma(\vec{u})` is Gaussian and the hyper parameters :math:`\gamma` are the mean :math:`\vec{\mu}` and covariance :math:`\mat{\Gamma}` of the Gaussian.

The variational distribution :math:`q_\gamma(\vec{f}')` can then be obtained by marginalizing over the inducing variables, that is:

.. math::

    q_{\gamma}(\vec{f}') = \int q_{\gamma}(\vec{f}',\vec{u})\,\text{d}\vec{u}

which is a Gaussian disribution :math:`\mathcal{N}(\vec{f}'\mid\vec{m}_\vec{y},\mat{S}_\vec{y})`

where:

- :math:`\vec{m}_\vec{y}(\mat{X}) = \mat{K}_{\mat{X}\mat{Z}}\mat{K}_{\mat{Z}\mat{Z}}^{-1}\vec{\mu}` is the mean vector of the inducing variables conditioned on the data,
- :math:`\mat{S}_\vec{y}(\mat{X},\mat{X}')=\mat{K}_{\mat{X}\mat{X}'} - \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}'} + \mat{K}_{\mat{X}\mat{Z}}  \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{\Gamma}\mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}'}`.

In the following, we will show that these :math:`\vec{\mu}` and :math:`\mat{\Gamma}` can be computed analytically if we instead optimize over the locations of the inducing points :math:`\mat{Z}`.


Marginal Prior Over Inducing Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To derive the posterior variational distribution, note that the joint distribution of the latent function values :math:`\vec{f}` and the inducing variables :math:`\vec{u}` according to the prior is:

.. math::

    p(\vec{f}, \vec{u}) = \mathcal{N} \left( \begin{bmatrix} \vec{f} \\ \vec{u} \end{bmatrix}, \begin{bmatrix} \mat{K}_{\mat{X}\mat{X}} & \mat{K}_{\mat{X}\mat{Z}} \\ \mat{K}_{\mat{Z}\mat{X}} & \mat{K}_{\mat{Z}\mat{Z}} \end{bmatrix} \right),

where:

- :math:`\mat{K}_{\mat{X}\mat{X}}` is the covariance matrix of the latent function values :math:`\vec{f}` at the input locations :math:`\mathbf{X}`,
- :math:`\mat{K}_{\mat{Z}\mat{Z}}` is the covariance matrix of the inducing variables :math:`\vec{u}` at the inducing input locations :math:`\mathbf{Z}`,
- :math:`\mat{K}_{\mat{X}\mat{Z}}` is the cross-covariance matrix between :math:`\vec{f}` and :math:`\vec{u}`, and :math:`\mat{K}_{\mat{Z}\mat{X}} = \mat{K}_{\mat{X}\mat{Z}}^\top`.

Factorizing the joint prior as:

.. math::

    p(\vec{f}, \vec{u}) = p(\vec{f} \mid \vec{u}) p(\vec{u}),


The marginal prior over inducing variables is simply given by:

.. math::

    p(\vec{u}) = \mathcal{N}(\vec{u} \mid \mathbf{0}, \mat{K}_{\mat{Z}\mat{Z}}),

where :math:`\mat{K}_{\mat{Z}\mat{Z}}` is the covariance matrix of the inducing variables.

Conditional Prior of :math:`\vec{f}` on :math:`\vec{u}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To defined the condiional prior, first define the vector-valued function :math:`\mathbf{k}_X(\vec{Z})` as:

.. math::

    \mathbf{k}_X(\mathbf{Z}) = \mat{K}_{\mat{X}\mat{Z}},

where :math:`\mathbf{k}_X(\mathbf{Z})` denotes the vector of covariances between :math:`\vec{f}` and the inducing inputs :math:`\mathbf{Z}`. Further, let :math:`\mat{K}_{\mat{X}\mat{Z}}` be the matrix containing values of the covariance function applied row-wise to the matrix of inputs :math:`\mathbf{X}` and inducing inputs :math:`\mathbf{Z}`.

Next, conditioning on the joint prior distribution on the inducing variables yields:

.. math::

    p(\vec{f} \mid \vec{u}) = \mathcal{N}(\vec{f} \mid \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \vec{u}, \mat{K}_{\mat{X}\mat{X}} - \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}}),

where the mean vector and covariance matrix are:

- **Mean vector**: :math:`\mathbf{m}=\mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \vec{u}`,
- **Covariance matrix**: :math:`\mat{S}=\mat{K}_{\mat{X}\mat{X}} - \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}}=\mat{K}_{\mat{X}\mat{X}}-\mat{K}_{\text{Nys}}`.

Here

.. math::

    \mat{K}_{\text{Nys}} = \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}},

is known as the Nyström approximation of the true covariance matrix :math:`\mat{K}_{\mat{X}\mat{X}}`.


ELBO Derivation
---------------
Using :math:`q(\vec{u})`, we can rewrite the log marginal likelihood as:

.. math::

    \log p(\vec{y} \mid \mathbf{X}) = \log \int p(\vec{y}, \vec{f}, \vec{u} \mid \mathbf{X}) \, d\vec{f} \, d\vec{u}.

Using **Jensen's inequality**, we derive a lower bound:

.. math::

    \log p(\vec{y} \mid \mathbf{X}) \geq \int q(\vec{u}) \log \frac{p(\vec{y}, \vec{f}, \vec{u} \mid \mathbf{X})}{q(\vec{u})} \, d\vec{f} \, d\vec{u}.

This lower bound is called the **Evidence Lower Bound (ELBO)**.


The ELBO is defined as:

.. math::

    \mathcal{L} = \int q(\vec{u}) \log \frac{p(\vec{y}, \vec{f}, \vec{u} \mid \mathbf{X})}{q(\vec{u})} \, d\vec{f} \, d\vec{u}.

We can decompose :math:`p(\vec{y}, \vec{f}, \vec{u} \mid \mathbf{X})` as:

.. math::

    p(\vec{y}, \vec{f}, \vec{u} \mid \mathbf{X}) = p(\vec{y} \mid \vec{f}) p(\vec{f} \mid \vec{u}) p(\vec{u}),

where:

- :math:`p(\vec{y} \mid \vec{f})` is the likelihood of the observed data given the latent function values,
- :math:`p(\vec{f} \mid \vec{u})` is the conditional distribution of the latent function values given the inducing points,
- :math:`p(\vec{u})` is the prior over the inducing points.

Substituting this decomposition into the ELBO:

.. math::

    \mathcal{L} = \int q(\vec{u}) \log \frac{p(\vec{y} \mid \vec{f}) p(\vec{f} \mid \vec{u}) p(\vec{u})}{q(\vec{u})} \, d\vec{f} \, d\vec{u}.

We can split the ELBO into two terms:

.. math::

    \mathcal{L} = \int q(\vec{u}) \int p(\vec{f} \mid \vec{u}) \log p(\vec{y} \mid \vec{f}) \, d\vec{f} \, d\vec{u} - \int q(\vec{u}) \log \frac{q(\vec{u})}{p(\vec{u})} \, d\vec{u}.


Likelihood Term
~~~~~~~~~~~~~~~

The first term is the **expected log likelihood**:

.. math::

    \mathbb{E}_{q(\vec{f})}[\log p(\vec{y} \mid \vec{f})],

where:

- :math:`q(\vec{f}) = \int p(\vec{f} \mid \vec{u}) q(\vec{u}) \, d\vec{u}` is the variational posterior over :math:`\vec{f}`.

This term measures how well the GP explains the observed data.

The inner integral involving the inducing variables :math:`\vec{u}` is computed as follows:

.. math::

    \log G(\vec{u}, \vec{y}) =
    \int p(\vec{f} \mid \vec{u}) \log p(\vec{y} \mid \vec{f}) \, d\vec{f}

Substituting the likelihood function and using :math:`(\vec{y}-\vec{f})^\top(\vec{y}-\vec{f}) = \text{Tr}\left(\vec{y}\vec{y}^\top - 2\vec{y}\vec{f}^\top + \vec{f}\vec{f}^\top \right)` :

.. math::

    \log G(\vec{u}, \vec{y}) =
    \int p(\vec{f} \mid \vec{u})
    \left[
        -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \text{Tr}\left( \vec{y}\vec{y}^\top - 2\vec{y}\vec{f}^\top + \vec{f}\vec{f}^\top \right)
    \right] \, d\vec{f}

Evaluating the integral:

.. math::

    \log G(\vec{u}, \vec{y}) =
    -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \text{Tr}\left( \vec{y}\vec{y}^\top - 2\vec{y}\boldsymbol{\alpha}^\top + \boldsymbol{\alpha}\boldsymbol{\alpha}^\top + \mathbf{K}_{\vec{X}\vec{X}} - \mat{Q}_\text{Nys} \right),

where:

- :math:`\boldsymbol{\alpha} = \mathbb{E}[\vec{f} \mid \vec{u}] = \mathbf{K}_{\vec{X}\vec{Z}} \mathbf{K}_{\vec{Z}\vec{Z}}^{-1} \vec{u}`,
- :math:`\mat{Q}_\text{Nys} = \mathbf{K}_{\vec{X}\vec{Z}} \mathbf{K}_{\vec{Z}\vec{Z}}^{-1} \mathbf{K}_{\vec{Z}\vec{X}}`.

Collecting the first three terms in the last Trace to form a Gaussian, yields

.. math::

    \log G(\vec{u}, \vec{y}) = \log \mathcal{N}(\vec{y} \mid \boldsymbol{\alpha}, \sigma^2 \mathbf{I}) - \frac{1}{2\sigma^2} \text{Tr}(\mathbf{K}_{\vec{X}\vec{X}} - \mat{Q}_\text{Nys}).



KL Divergence Term
~~~~~~~~~~~~~~~~~~

The second term is the **KL divergence** between the variational posterior :math:`q(\vec{u})` and the prior :math:`p(\vec{u})`:

.. math::

    \text{KL}(q(\vec{u}) \mid p(\vec{u})) = \int q(\vec{u}) \log \frac{q(\vec{u})}{p(\vec{u})} \, d\vec{u}.

This term penalizes deviations of :math:`q(\vec{u})` from :math:`p(\vec{u})`, ensuring that the variational posterior remains close to the prior.

We do not explicitly compute this but rather
substitute :math:`\log G(\vec{u}, \vec{y})` back into the reorganized ELBO:

.. math:: \mathcal{L} = \int p(\vec{u}) \left[\log G(\vec{u}, \vec{y}) + \log \frac{p(\vec{u})}{q(\vec{u})} \right]\, d\vec{u},

yielding

.. math::

    \mathcal{L} = \int q(\vec{u}) \frac{\log \mathcal{N}(\vec{y} | \vec{\alpha}, \sigma^2 \mathbf{I}) p(\vec{u})}{ q(\vec{u})} \, d\vec{u}
    - \frac{1}{2\sigma^2} \text{Tr}(\mat{K}_{\vec{X}\vec{X}} - \mat{K}_\text{Nys}),

Using Jensens inequality to move the log outside the intergral, so the two :math:`q(\vec{u})` cancel, yields

.. math::

    \mathcal{L} \ge \mathcal{L}' = \log \int \mathcal{N}(\vec{y} | \vec{\alpha}, \sigma^2 \mathbf{I}) p(\vec{u}) \, d\vec{u}
    - \frac{1}{2\sigma^2} \text{Tr}(\mat{K}_{\vec{X}\vec{X}} - \mat{K}_\text{Nys}),

Final ELBO Expression
~~~~~~~~~~~~~~~~~~~~~
Simplifying again yields the final form of the EBLO

.. math::

   \mathcal{L} = \log \mathcal{N}(\vec{y} | \vec{0}, \sigma^2 \mathbf{I} + \mat{K}_\text{Nys})
    - \frac{1}{2\sigma^2} \text{Tr}(\mat{K}_{\vec{X}\vec{X}} - \mat{Q}_\text{Nys}),

where we redefined the ELBO :math:`\mathcal{L}=\mathcal{L}` because :math:`\mathcal{L} \ge \mathcal{L}'` is still a lower bound, just a smaller one.



Computational Complexity of Estimating the ELBO
-----------------------------------------------

The Evidence Lower Bound (ELBO) is a key quantity in Variational Gaussian Processes (VGPs).
The first term is the likelihood term, computed using the Nyström approximation.

.. math::

    \mathbb{E}_{q(\vec{f})}[\log p(\vec{y} \mid \vec{f})] = -\frac{1}{2} \left[ \vec{y}^\top \mat{K}_{\text{noisy}}^{-1} \vec{y} + \log \mid\mat{K}_{\text{noisy}}\mid + n \log(2\pi) \right],

where:

- :math:`\mat{K}_{\text{noisy}} = \mat{K}_{\text{Nys}} + \sigma^2 \mathbf{I}_n`.

The second term is a regularization term that will tend to zero as more inducing points are used and will be zero when the inducing points are the training points.


Breakdown of Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

The computational complexity of estimating the ELBO can be broken down into the following components:

1. **Covariance Matrix Approximation**:
   Using the Nyström method, the covariance matrix :math:`\mat{K}_{\text{approx}}` is approximated as:

   .. math::

       \mat{K}_{\text{approx}} = \mat{K}_{\vec{X}\vec{Z}} \mat{K}_{\vec{Z}\vec{Z}}^{-1} \mat{K}_{\vec{X}\vec{Z}}^\top,

   where:

   - :math:`\mat{K}_{\vec{X}\vec{Z}}` is the covariance matrix between the data points :math:`\mathbf{X}` and the inducing points :math:`\vec{u}`,
   - :math:`\mat{K}_{\vec{Z}\vec{Z}}` is the covariance matrix between the inducing points.

   **Complexity**:

   - Computing :math:`\mat{K}_{\vec{X}\vec{Z}}` scales as :math:`\mathcal{O}(nm)`.
   - Computing :math:`\mat{K}_{\vec{Z}\vec{Z}}^{-1}` using Cholesky decomposition scales as :math:`\mathcal{O}(m^3)`.
   - Multiplying matrices to compute :math:`\mat{K}_{\text{approx}}` scales as :math:`\mathcal{O}(nm^2)`.

   **Total Complexity**:

   .. math::
       \mathcal{O}(nm^2 + m^3).

2. **Likelihood Term**:
   The likelihood term involves computing the log determinant and inverse of the noisy covariance matrix:

   .. math::

       \mat{K}_{\text{noisy}} = \mat{K}_{\text{approx}} + \sigma^2 \mathbf{I}_n.

   **Complexity**:

   - Computing the Cholesky decomposition of :math:`\mat{K}_{\text{noisy}}` scales as :math:`\mathcal{O}(m^3)`.
   - Solving triangular systems to compute :math:`\mat{K}_{\text{noisy}}^{-1}` scales as :math:`\mathcal{O}(nm)`.

   **Total Complexity**:

   .. math::
       \mathcal{O}(nm + m^3).

3. **Trace Regulariaztion Term**:
   The reglularization term is:

   .. math::

       - \frac{1}{2\sigma^2} \text{Tr}(\mat{K}_{\vec{X}\vec{X}} - \mat{Q}_\text{Nys}),

   **Complexity**:

   - Requires coputing the nystrom approximation and the covariance kernel at the training data`

   **Total Complexity**:

   .. math::
        \mathcal{O}(nm^2 + m^3).


The Optimal Variational Distribution
------------------------------------
Given inducing points, w can compute the mean and covariance of the optimal variational disribution :math:`q^*(\vec{u})`. This distribution can be found by taking the derivative of the ELBO, yielding:

.. math:: q^*(\vec{u}) = \frac{\mathcal{N}(\vec{y} | \vec{\alpha}, \sigma^2 \mat{I}) p(\vec{u})}
    {\int \mathcal{N}(\vec{y} | \vec{\alpha}, \sigma^2 \mat{I}) p(\vec{u}) \, d\vec{u}}.


Substituting the expressions, we have:

.. math::

    \begin{align*}
    q^*(\vec{u}) &\propto \mathcal{N}(\vec{y} | \vec{\alpha}, \sigma^2 \mat{I}) p(\vec{u}) \\
    &= c \exp \left(
        -\frac{1}{2} \vec{u}^\top \left( \mat{K}_{\vec{Z}\vec{Z}}^{-1} + \frac{1}{\sigma^2} \mat{K}_{\vec{Z}\vec{Z}}^{-1} \mat{K}_{\vec{Z}\vec{X}} \mat{K}_{\vec{X}\vec{Z}} \mat{K}_{\vec{Z}\vec{Z}}^{-1} \right) \vec{u}
        + \frac{1}{\sigma^2} \vec{y}^\top \mat{K}_{\vec{X}\vec{Z}} \mat{K}_{\vec{Z}\vec{Z}}^{-1} \vec{u}
    \right),
    \end{align*}

where :math:`c` is a constant.

Completing the square, produces the distribution:

.. math::

    q^*(\vec{u}) = \mathcal{N} \left( \vec{u} \mid \vec{\mu}, \mat{C} \right),

where:

- The mean of the optimal distribution over the latent variables :math:`\vec{u}` is

.. math:: \vec{\mu}=\sigma^{-2} \mat{K}_{\vec{Z}\vec{Z}} \mat{\Sigma} \mat{K}_{\vec{Z}\vec{X}} \vec{y}

- The variance of the optimal distribution is

.. math:: \mat{C} = \mat{K}_{\vec{Z}\vec{Z}} \mat{\Sigma} \mat{K}_{\vec{Z}\vec{Z}}

- and

.. math::

    \mat{\Sigma} = (\mat{K}_{\vec{Z}\vec{Z}} + \sigma^{-2} \mat{K}_{\vec{Z}\vec{X}} \mat{K}_{\vec{X}\vec{Z}})^{-1}.

Prediction
----------

Plugging :math:`\vec{\mu}` and :math:`\mat{\Sigma}` into the variational distribution,

.. math::

    q_{\gamma}(\vec{f}') = \mathcal{N}(\vec{f}' \mid \vec{m}_\vec{y}, \mat{S}_\vec{y}),

yields the mean vector :math:`\vec{m}_\vec{y}`:

.. math::

    \vec{m}_\vec{y} = \mat{K}_{\mat{X}\mat{Z}}\mat{K}_{\mat{Z}\mat{Z}}^{-1}\vec{\mu} =   \sigma^{-2}\mat{K}_{\mat{X}\mat{Z}}\mat{\Sigma} \mat{K}_{\mat{Z}\mat{X}} \vec{y},


and the covariance matrix :math:`\mat{S}_\vec{y}`:

.. math::

    \begin{align*}
    \mat{S}_\vec{y} &=  \mat{K}_{\mat{X}\mat{X}'} - \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}'}
    + \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1}\mat{C} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}'}\\
    &= \mat{K}_{\mat{X}\mat{X}'} - \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}'}
    + \mat{K}_{\mat{X}\mat{Z}} \mat{K}_{\mat{Z}\mat{Z}}^{-1} \left( \mat{K}_{\mat{Z}\mat{Z}} \mat{\Sigma} \mat{K}_{\mat{Z}\mat{Z}} \right) \mat{K}_{\mat{Z}\mat{Z}}^{-1} \mat{K}_{\mat{Z}\mat{X}'}.
    \end{align*}



Steps to Construct a GP with Inducing Points
--------------------------------------------

This tutorial now demonstrates how to construct and optimize a GP with inducing points and variational inference using the provided classes.
"""

# %%
# Step 1: Define the Kernel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The kernel function defines the covariance structure of the GP.


import numpy as np
from scipy import stats
from pyapprox.surrogates.kernels import MaternKernel
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.backends.torch import TorchMixin as bkd

nvars = 2  # Number of input variables
kernel = MaternKernel(np.inf, 0.1, [1e-1, 1], nvars, backend=bkd)


# %%
# Step 2: Initialize Inducing Samples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `InducingSamples` class manages the inducing points and noise hyperparameters. You can specify the number of inducing points and their bounds.

from pyapprox.surrogates.gaussianprocess.variationalgp import InducingSamples

ninducing_samples = 10  # Number of inducing points
inducing_samples = InducingSamples(
    nvars,
    ninducing_samples,
    bkd,
    inducing_sample_bounds=bkd.asarray([-1.0, 1.0]),
    noise_std=1e-2,  # Initial noise value
)


# %%
# Step 3: Setup the GP
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `InducingGaussianProcess` class constructs the GP model using the kernel and inducing samples.


from pyapprox.surrogates.gaussianprocess.variationalgp import (
    InducingGaussianProcess,
)

vi_gp = InducingGaussianProcess(
    nvars=nvars,
    kernel=kernel,
    inducing_samples=inducing_samples,
)

# The GP model is optimized by maximizing the ELBO using gradient-based methods.

from pyapprox.optimization.scipy import (
    DifferentialEvolutionScipyConstrainedGlobalLocalOptimizer,
)

optimizer = DifferentialEvolutionScipyConstrainedGlobalLocalOptimizer()
optimizer.set_verbosity(0)
optimizer.global_optimizer().set_options(maxiter=10)
vi_gp.set_optimizer(optimizer)


# %%
# Step : Train the GP
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The GP model is trained at training points


from pyapprox.interface.model import ModelFromVectorizedCallable

prior = IndependentMarginalsVariable(
    [stats.uniform(-1, 2) for ii in range(nvars)], backend=bkd
)


def fun(x):
    return (bkd.sin(np.pi * x[0, :]) + bkd.cos(np.pi * x[1, :]))[:, None]


model = ModelFromVectorizedCallable(1, nvars, fun, backend=bkd)

np.random.seed(0)
ntrain_samples = 50
X_train = prior.rvs(ntrain_samples)
y_train = fun(X_train)

vi_gp.fit(X_train, y_train)


# %%
# Compare the Variational GP to the exact GP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Evaluate the variational and exact  GP model at test points to compute
# and plot the posterior mean.


from pyapprox.surrogates.gaussianprocess.exactgp import ExactGaussianProcess
import matplotlib.pyplot as plt

exact_gp = ExactGaussianProcess(
    nvars,
    kernel,
    trend=None,
    kernel_reg=0,
)
exact_gp.set_optimizer(optimizer)
exact_gp.fit(X_train, y_train)

fig, axs = plt.subplots(1, 3, figsize=(3 * 8, 6))
model.plot(axs[0], prior.interval(1).flatten(), levels=31)
exact_gp.plot(axs[1], prior.interval(1).flatten(), levels=31)
axs[1].plot(*X_train, "ok")
vi_gp.plot(axs[2], prior.interval(1).flatten(), levels=31)
_ = axs[2].plot(*vi_gp.inducing_samples().get_samples(), "ok")


# %%
# Conclusion
# ----------
#
# This tutorial demonstrated how to construct a Gaussian Process with inducing points and variational inference using the provided classes. By leveraging sparse approximations and variational inference, the GP model scales efficiently to large datasets while maintaining accuracy.

# %%
# References
# ----------
#    .. [Titsias2009] `Michalis Titsias. *Variational Learning of Inducing Variables in Sparse Gaussian Processes*. Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics, 567–574, 2009.  <https://proceedings.mlr.press/v5/titsias09a.html>`_.
#
#    .. [Vanderwilk2020] `Mark van der Wilk, Vincent Dutordoir, ST John, Artem Artemev, Vincent Adam, and James Hensman. *A Framework for Interdomain and Multioutput Gaussian Processes*. 2020. <https://arxiv.org/abs/2003.01115>`_.
