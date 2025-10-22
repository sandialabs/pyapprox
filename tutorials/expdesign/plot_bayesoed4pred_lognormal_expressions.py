r"""
Risk-aware Utility Functions for Log-Normal Predictions
=======================================================

This tutorial focuses on deriving the distribution of the standard deviation of the transformed variable :math:`W = \exp(Q)`,
where :math:`Q \mid \vec{y}` follows a Gaussian distribution. We use the scaled log-normal distribution derived :ref:`here<sphx_glr_auto_tutorials_expdesign_plot_bayesoed4pred_gaussian_expressions.py>`
and conclude by computing the mean of the standard deviation distribution.

1. Transformation of :math:`Q` to :math:`W`
-------------------------------------------

Given :math:`Q \mid \vec{y} \sim \mathcal{N}(\mat{\Psi} \vec{\mu}_\star, \mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T)`,
the transformed variable :math:`W = \exp(Q)` follows a log-normal distribution:

.. math::
   W \mid \vec{y} \sim \mathcal{LN}(\tau, \sigma^2),

where:

- :math:`\tau = \mat{\Psi} \vec{\mu}_\star` is the mean of :math:`Q \mid \vec{y}`,
- :math:`\sigma^2 = \mat{\Psi} \mat{\Sigma}_\star \mat{\Psi}^T` is the variance of :math:`Q \mid \vec{y}`.

Additionally:

.. math::
   Q = \mat{\Psi} \rvv,

where:

- :math:`\mat{\Psi} \in \mathbb{R}^{1 \times P}` is a row vector that predicts a quantity of interest.

From earlier derivations, the standard deviation of :math:`W` was expressed as a scaled log-normal random variable:

.. math::
   \text{Std}(W \mid \vec{y}) = K \cdot \exp(\tau),

where:

- :math:`K = \exp\left(\frac{\sigma^2}{2}\right) \sqrt{\exp(\sigma^2) - 1}` is a deterministic constant.


2. Distribution of the Standard Deviation
-----------------------------------------

Since :math:`K` is deterministic, the randomness in :math:`\text{Std}(W \mid \vec{y})` comes entirely from
:math:`\tau = \mat{\Psi} \vec{\mu}_\star`, which is Gaussian. Let :math:`\mat{\Psi} \vec{\mu}_\star \sim \mathcal{N}(\nu, \sigma_\tau^2)`,
where:

- :math:`\nu = \mat{\Psi} \vec{\nu}` is the mean of :math:`\mat{\Psi} \vec{\mu}_\star`,
- :math:`\sigma_\tau^2 = \mat{\Psi} \mat{C} \mat{\Psi}^T` is the variance of :math:`\mat{\Psi} \vec{\mu}_\star`.

Then :math:`\tau = \mat{\Psi} \vec{\mu}_\star` is Gaussian:

.. math::
   \tau \sim \mathcal{N}(\nu, \sigma_\tau^2).

Since :math:`\text{Std}(W \mid \vec{y}) = K \cdot \exp(\tau)`, and :math:`\exp(\tau)` is the exponential of a Gaussian random variable,
:math:`\text{Std}(W \mid \vec{y})` follows a log-normal distribution.


3. Parameters of the Log-Normal Distribution
--------------------------------------------

The standard deviation of :math:`W \mid \vec{y}` follows the log-normal distribution:

.. math::
   \text{Std}(W \mid \vec{y}) \sim \mathcal{LN}(\nu, \sigma_\tau^2),

where:

- :math:`\nu = \mat{\Psi} \vec{\nu}` is the mean of :math:`\mat{\Psi} \vec{\mu}_\star`,
- :math:`\sigma_\tau^2 = \mat{\Psi} \mat{C} \mat{\Psi}^T` is the variance of :math:`\mat{\Psi} \vec{\mu}_\star`.

The deterministic constant :math:`K = \exp\left(\frac{\sigma^2}{2}\right) \sqrt{\exp(\sigma^2) - 1}` scales the log-normal distribution
but does not affect its shape.

4. Mean of the Standard Deviation Distribution
----------------------------------------------

The mean of a log-normal random variable :math:`X \sim \mathcal{LN}(\mu, \sigma^2)` is given by:

.. math::
   \mathbb{E}[X] = \exp\left(\mu + \frac{\sigma^2}{2}\right).

For :math:`\text{Std}(W \mid \vec{y}) \sim \mathcal{LN}(\nu, \sigma_\tau^2)`, the mean is:

.. math::
   \mathbb{E}[\text{Std}(W \mid \vec{y})] = K \cdot \exp\left(\nu + \frac{\sigma_\tau^2}{2}\right),

where:

- :math:`K = \exp\left(\frac{\sigma^2}{2}\right) \sqrt{\exp(\sigma^2) - 1}` is the deterministic scaling factor,
- :math:`\nu = \mat{\Psi} \vec{\nu}` is the mean of :math:`\mat{\Psi} \vec{\mu}_\star`,
- :math:`\sigma_\tau^2 = \mat{\Psi} \mat{C} \mat{\Psi}^T` is the variance of :math:`\mat{\Psi} \vec{\mu}_\star`.


5. Computing the Average Value at Risk (AVaR) of the Standard Deviation Distribution
------------------------------------------------------------------------------------

The Average Value at Risk (AVaR) is a risk measure that quantifies the expected value of a random variable beyond a given quantile
(Value at Risk, VaR). For the standard deviation distribution :math:`\text{Std}(W \mid \vec{y}) \sim \mathcal{LN}(\nu, \sigma_\tau^2)`,
we compute the AVaR using the positive homogeneity property of AVaR.

By the positive homogeneity property of AVaR, scaling a random variable by a positive constant scales its AVaR by the same constant.
Specifically, for a scaled random variable :math:`F \cdot X`, where :math:`F > 0` is deterministic and :math:`X` is a random variable:

.. math::
   \mathrm{AVaR}[F \cdot X] = F \cdot \mathrm{AVaR}[X].

For the standard deviation distribution :math:`\text{Std}(W \mid \vec{y}) = K \cdot \exp(\tau)`, where
:math:`K = \exp\left(\frac{\sigma^2}{2}\right) \sqrt{\exp(\sigma^2) - 1}` is deterministic, we can write:

.. math::
   \mathrm{AVaR}[\text{Std}(W \mid \vec{y})] = K \cdot \mathrm{AVaR}[\exp(\tau)].

We can then compute the AVaR of the log-normal variable :math:`\exp(\tau)` analytically.
"""
