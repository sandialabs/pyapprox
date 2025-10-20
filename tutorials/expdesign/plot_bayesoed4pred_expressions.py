r"""
Risk-Aware Bayesian Inference and Experimental Design: Analytical Expressions for Expected Utility with Gaussian Linear Models
=============================================================================================================================

In this tutorial, we derive and compute analytical expressions that quantify the dependence of the posterior mean and covariance parameters on observational data within the framework of Bayesian inference. Specifically, we focus on a linear observational model with Gaussian priors and likelihood distributions. While the posterior covariance is independent of the realizations of the data, the posterior mean is not—making the posterior mean a random variable that depends on the specific observation realizations used to compute it. We derive the distribution of the posterior mean as a function of the observations, as well as the distribution of posteriors when pushed forward through a linear quantity-of-interest model -- again as a function of the observations. Additionally, we derive expressions for various risk measures, such as entropic risk and Average Value at Risk (AVaR), which quantify uncertainty in the pushforward. These expressions are essential for the numerical verification of Bayesian Optimal Experimental Design (BOED) software that incorporates risk-awareness into utility functions used to construct experimental designs.

1. Linear Model Assumption
--------------------------
We assume a linear model of the form:

.. math::
   f(x, \rvv) = \sum_{i=0}^P \phi_i(x) \rvv, \quad \phi_i(x) = x^i

Here:

- :math:`x \in \mathbb{R}` is the input variable.
- :math:`\rvv = [\rv_0, \rv_1, \ldots, \rv_P] \in \mathbb{R}^P` are the model parameters.
- :math:`\phi_i(x) \in \mathbb{R}` are polynomial basis functions.

The design points :math:`x = [x_1, \ldots, x_{N_x}] \in \mathbb{R}^{N_x}` are selected from the interval :math:`[-1, 1]`, where :math:`N_x` is the total number of candidate design locations.


2. Vandermonde Matrix Representation
-------------------------------------
To represent the polynomial basis functions evaluated at the design points, we construct the Vandermonde matrix :math:`\mat{\Phi} \in \mathbb{R}^{N_x \times P}`, where :math:`N_x` is the number of design points and :math:`P` is the number of terms in the polynomial expansion. The entries of :math:`\mat{\Phi}` are defined as:

.. math::
   \mat{\Phi}_{ij} = \phi_j(x_i) = x_i^j

Here:

- :math:`x_i` is the :math:`i`-th design point.
- :math:`\phi_j(x_i)` is the :math:`j`-th polynomial basis function evaluated at :math:`x_i`.

This matrix provides a compact representation of the polynomial basis functions evaluated at all design points, enabling efficient computation in the subsequent steps.

3. Observational Model
----------------------
The observational model describes the relationship between the observations :math:`\vec{y}`, the Vandermonde matrix :math:`\mat{\Phi}`, the model parameters :math:`\rvv`, and the noise :math:`\vec{\epsilon}`. It is given by:

.. math::
   \vec{y} = \mat{\Phi} \rvv + \vec{\epsilon}, \quad \vec{\epsilon} \sim \mathcal{N}(0, \mat{\Gamma})

Here:

- :math:`\vec{y} \in \mathbb{R}^{N_x}` represents the observations at the design points.
- :math:`\mat{\Phi} \in \mathbb{R}^{N_x \times P}` is the Vandermonde matrix.
- :math:`\rvv \in \mathbb{R}^P` are the model parameters.
- :math:`\vec{\epsilon} \in \mathbb{R}^{N_x}` is Gaussian noise with covariance matrix :math:`\mat{\Gamma} \in \mathbb{R}^{N_x \times N_x}`.

This model assumes that the observations :math:`\vec{y}` are corrupted by Gaussian noise, which is independent of the model parameters :math:`\rvv`.

4. Prior Distribution
---------------------
The prior distribution on the model parameters :math:`\rvv` is assumed to be Gaussian:

.. math::
   \rvv \sim \mathcal{N}(\vec{\mu}, \mat{\Sigma})

where:

- :math:`\vec{\mu} \in \mathbb{R}^P` is the prior mean.
- :math:`\mat{\Sigma} \in \mathbb{R}^{P \times P}` is the prior covariance matrix.

This prior encodes our initial beliefs about the parameters :math:`\rvv` before observing any data.


5. Posterior Distribution
-------------------------
Given the observational model and the prior, the posterior distribution of :math:`\rvv` conditioned on the data :math:`\vec{y}` is also Gaussian:

.. math::
   \rvv \mid \vec{y} \sim \mathcal{N}(\vec{\mu}_\star, \mat{\Sigma}_\star)

where:

.. math::
   \mat{\Sigma}_\star = \left(\mat{\Sigma}^{-1} + \mat{\Phi}^\top \mat{\Gamma}^{-1} \mat{\Phi}\right)^{-1}

.. math::
   \vec{\mu}_\star = \mat{\Sigma}_\star \left(\mat{\Phi}^\top \mat{\Gamma}^{-1} \vec{y} + \mat{\Sigma}^{-1} \vec{\mu}\right)

Here:

- :math:`\mat{\Sigma}_\star \in \mathbb{R}^{P \times P}` is the posterior covariance matrix.
- :math:`\vec{\mu}_\star \in \mathbb{R}^P` is the posterior mean.

The posterior covariance matrix :math:`\mat{\Sigma}_\star` quantifies the uncertainty in the parameters after observing the data, while the posterior mean :math:`\vec{\mu}_\star` provides the updated estimate of the parameters.


6. Conditional Posterior Mean
-----------------------------
The posterior mean :math:`\vec{\mu}_\star` depends on the observational data :math:`\vec{y}`. Writing it explicitly as conditional on the parameters :math:`\rvv` and noise :math:`\vec{\epsilon}`, we have:

.. math::
   \vec{\mu}_\star \mid \rvv, \vec{\epsilon} = \mat{R} \mat{\Phi} \rvv + \mat{R} \vec{\epsilon} + \mat{\Sigma}_\star \mat{\Sigma}^{-1} \vec{\mu}

where:

.. math::
   \mat{R} = \mat{\Sigma}_\star \mat{\Phi}^\top \mat{\Gamma}^{-1}


7. Distribution of Posterior Mean
---------------------------------
Since the prior distribution on :math:`\rvv` and the noise :math:`\vec{\epsilon}` are independent, the posterior mean :math:`\vec{\mu}_\star` is itself a Gaussian random variable:

.. math::
   \vec{\mu}_\star \sim \mathcal{N}(\vec{\nu}, \mat{C})

where:

.. math::
   \vec{\nu} = \mat{R} \mat{\Phi} \vec{\mu} + \mat{\Sigma}_\star \mat{\Sigma}^{-1} \vec{\mu}

.. math::
   \mat{C} = \mathbb{V}[\mat{R} \mat{\Phi} \rvv] + \mathbb{V}[\mat{R} \vec{\epsilon}]

Expanding :math:`\mat{C}`, we get:

.. math::
   \mat{C} = \mat{R} \mat{\Phi} \mat{\Sigma} (\mat{R} \mat{\Phi})^\top + \mat{R} \mat{\Gamma} \mat{R}^\top

Here:

- :math:`\vec{\nu}` is the mean of the posterior mean :math:`\vec{\mu}_\star`.
- :math:`\mat{C}` is the covariance of the posterior mean :math:`\vec{\mu}_\star`.

8. Computing the Distribution of the Entropic Risk Deviation of Q | y
---------------------------------------------------------------------
Here we compute the distribution of the entropic risk when each posterior is pushed forward through a linear model using:

.. math::
   Q = \mat{\Psi} \rvv

where:

- :math:`\mat{\Psi} \in \mathbb{R}^{1 \times P}` is a row vector that predicts a quantity of interest.


Expression for the Entropic Risk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The entropic risk of :math:`Q \mid \vec{y}` is defined as:

.. math::
   \mathcal{R}(Q \mid \vec{y}) = \mat{\Psi} \vec{\mu}_\star + \frac{\lambda}{2} \mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T,

where:

- :math:`\mat{\Psi} \vec{\mu}_\star` is the mean of :math:`Q \mid \vec{y}`,
- :math:`\mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T` is the variance of :math:`Q \mid \vec{y}`,
- :math:`\lambda > 0` is a risk aversion parameter.


Splitting the Entropic Risk
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The entropic risk can be split into two components:

.. math::
   \mathcal{R}(Q \mid \vec{y}) = \mat{\Psi} \vec{\mu}_\star + C,

where:

- :math:`\mat{\Psi} \vec{\mu}_\star` captures the posterior mean,
- :math:`C = \frac{\lambda}{2} \mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T` is a deterministic scaling factor that depends on the posterior covariance.


Distribution of the Entropic Risk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The entropic risk depends on the posterior mean :math:`\vec{\mu}_\star`, which is Gaussian, and the posterior covariance :math:`\mat{\Sigma}_\star`, which is deterministic. Since :math:`\mat{\Psi} \vec{\mu}_\star` is Gaussian, the entropic risk follows a shifted Gaussian distribution:

.. math::
   \mathcal{R}(Q \mid \vec{y}) \sim \mathcal{N}(\tau + C, \sigma^2),

where:

- :math:`\tau = \mat{\Psi} \vec{\nu}` is the mean of :math:`\mat{\Psi} \vec{\mu}_\star`,
- :math:`\sigma^2 = \mat{\Psi} \mat{C} \mat{\Psi}^T` is the variance of :math:`\mat{\Psi} \vec{\mu}_\star`.


Computing the Distribution of the Entropic Risk-Based Deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The deviation associated with a risk measure of a variable :math:`X` is:

.. math::
   \mathcal{D}(X) = \mathcal{R}(X) - \mathbb{E}(X).

Thus, the entropic risk-based deviation for :math:`Q \mid \vec{y}` is:

.. math::
   \mathcal{D}(Q \mid \vec{y}) = \mathcal{R}(Q \mid \vec{y}) - \mathbb{E}[Q \mid \vec{y}].

Substituting the expressions for :math:`\mathcal{R}(Q \mid \vec{y})` and :math:`\mathbb{E}[Q \mid \vec{y}]`, we have:

.. math::
   \mathcal{D}(Q \mid \vec{y}) = \mat{\Psi} \vec{\mu}_\star + C - \mat{\Psi} \vec{\mu}_\star.

Simplifying:

.. math::
   \mathcal{D}(Q \mid \vec{y}) = C.

Since :math:`C = \frac{\lambda}{2} \mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T` is deterministic, the entropic risk-based deviation is constant and does not depend on the observations.


Expected Value of the Entropic Risk-Based Deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The mean of :math:`\mathcal{D}(Q \mid \vec{y})` is simply:

.. math::
   \mathbb{E}[\mathcal{D}(Q \mid \vec{y})] = C,

where:

.. math::
   C = \frac{\lambda}{2} \mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T.


9. Computing the AVaR Deviation of Q | y
----------------------------------------
Here we compute the AVaR deviation for the transformed variable :math:`Q \mid \vec{y}`, where :math:`Q = \mat{\Psi} \rvv`. The AVaR deviation is defined as the difference between the Average Value at Risk (AVaR) and the expected value of :math:`Q`. Using the properties of the Normal distribution and positive homogeneity of AVaR, we derive the expected AVaR deviation and its distribution.


AVaR of Q | y
^^^^^^^^^^^^^
The AVaR of :math:`Q \mid \vec{y}` at a quantile level :math:`p \in (0, 1)` is given by:

.. math::
   \mathrm{AVaR}[Q \mid \vec{y}] = \mat{\Psi} \vec{\mu}_\star + \sqrt{\mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T} \cdot \frac{\phi(\Phi^{-1}(p))}{1 - p},

where:

- :math:`\phi(\cdot)` is the PDF of the standard normal distribution,
- :math:`\Phi^{-1}(p)` is the inverse CDF (quantile function) of the standard normal distribution,
- :math:`\sqrt{\mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T}` is the standard deviation of :math:`Q \mid \vec{y}`.


Distribution of AVaR
^^^^^^^^^^^^^^^^^^^^
The AVaR depends on the posterior mean :math:`\vec{\mu}_\star`, which is Gaussian, and the posterior covariance matrix :math:`\mat{\Sigma}_\star`, which is deterministic. Since :math:`\mat{\Psi} \vec{\mu}_\star` is Gaussian, the AVaR of :math:`Q \mid \vec{y}` is a shifted Gaussian random variable:

.. math::
   \mathrm{AVaR}[Q \mid \vec{y}] \sim \mathcal{N}\left(\mat{\Psi} \vec{\nu} + \sqrt{\mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T} \cdot \frac{\phi(\Phi^{-1}(p))}{1 - p}, \mat{\Psi} \mat{C} \mat{\Psi}^T\right),

where:

- :math:`\mat{\Psi} \vec{\nu}` is the mean of :math:`\mat{\Psi} \vec{\mu}_\star`,
- :math:`\mat{\Psi} \mat{C} \mat{\Psi}^T` is the variance of :math:`\mat{\Psi} \vec{\mu}_\star`,
- :math:`\sqrt{\mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T} \cdot \frac{\phi(\Phi^{-1}(p))}{1 - p}` is the deterministic shift due to the AVaR formula.


Expected AVaR of Q | y
^^^^^^^^^^^^^^^^^^^^^^
The expected AVaR of :math:`Q \mid \vec{y}` is the mean of the shifted Gaussian distribution:

.. math::
   \mathbb{E}[\mathrm{AVaR}[Q \mid \vec{y}]] = \mat{\Psi} \vec{\nu} + \sqrt{\mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T} \cdot \frac{\phi(\Phi^{-1}(p))}{1 - p}.


Expected Value of Q | y
^^^^^^^^^^^^^^^^^^^^^^^
The expected value of :math:`Q \mid \vec{y}` is:

.. math::
   \mathbb{E}[Q \mid \vec{y}] = \mat{\Psi} \vec{\mu}_\star.

Since :math:`\mat{\Psi} \vec{\mu}_\star` is Gaussian, the expected value of :math:`Q \mid \vec{y}` is:

.. math::
   \mathbb{E}[\mathbb{E}[Q \mid \vec{y}]] = \mat{\Psi} \vec{\nu}.


Deviation of Q | y
^^^^^^^^^^^^^^^^^^
The AVaR deviation is defined as the difference between the AVaR and the expected value:

.. math::
   \mathbb{E}[\mathrm{AVaR}[Q \mid \vec{y}] - \mathbb{E}[Q \mid \vec{y}]] = \sqrt{\mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T} \cdot \frac{\phi(\Phi^{-1}(p))}{1 - p}.
"""
