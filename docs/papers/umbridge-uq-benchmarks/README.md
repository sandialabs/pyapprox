# Genz Integration Functions

## Overview
This benchmark consists of fix families of analytical functions $F_i:\mathbb{R}^D\to\mathbb{R}$, $i=1,\ldots,6$, with means $\mathbb{E}[F(\theta)]$ that can be computed analytically. The number of inputs $D$ and the anisotropy (relative importance of each variable and interactions) of the functions can be adjusted.

## Authors
John D. Jakeman

## Run
docker run -it -p 4243:4243 linusseelinger/benchmark-genz

## Properties

Model | Description
---|---
forward | Forward model

### forward
Mapping | Dimensions | Description
---|---|---
input | [D] | 2D coordinates $x \in \mathbb{R}^D$
output | [1] | Function $F(x)$ evaluated at $x$

Feature | Supported
---|---
Evaluate | True
Gradient | False
ApplyJacobian | False
ApplyHessian | False

Config | Type | Default | Description
---|---|---|---
nvars | int | D | Number of inputs
c_factor | double | 1.0 | Normalization parameter
w_factor | double | 0. | shift parameter
coef_type | string | "sqexp" | Coefficient decay type
name | string | "oscillatory" | Name of the test function

The supported values of the coef_type and name config variables are:

name=["oscillatory", "product_peak", "corner_peak", "gaussian", "c0continuous",  "discontinuous"]
coef_type=["none", "quadratic", "quartic", "exp", "sqexp"]


## Mount directories
Mount directory | Purpose
---|---
None |

## Source code

[Model sources here.](https://github.com/sandialabs/pyapprox/blob/master/pyapprox/benchmarks/genz.py)


## Description
The six Genz test function are:

Oscillatory ('oscillatory')

$$ f(z) = \cos\left(2\pi w_1 + \sum_{d=1}^D c_dz_d\right)$$

Product Peak ('product_peak')

$$ f(z) = \prod_{d=1}^D \left(c_d^{-2}+(z_d-w_d)^2\right)^{-1}$$

Corner Peak ('corner_peak')

$$ f(z)=\left( 1+\sum_{d=1}^D c_dz_d\right)^{-(D+1)}$$

Gaussian Peak ('gaussian')

$$ f(z) = \exp\left( -\sum_{d=1}^D c_d^2(z_d-w_d)^2\right)$$

C0 Continuous ('c0continuous')

$$ f(z) = \exp\left( -\sum_{d=1}^D c_d\lvert z_d-w_d\rvert\right)$$

Discontinuous ('discontinuous')

$$ f(z) = \begin{cases}
0 & z_1>w_1 \;\mathrm{or}\; z_2>w_2\\
\exp\left(\sum_{d=1}^D c_dz_d\right) & \mathrm{otherwise}
\end{cases}$$

Increasing $\lVert c \rVert$ will in general make
the integrands more difficult.

The $0\le w_d \le 1$ parameters do not affect the difficulty
of the integration problem. We set $w_1=w_2=\ldots=W_D$.

The coefficient types implement different decay rates for $c_d$.
This allows testing of methods that can identify and exploit anisotropy.
They are as follows:

No decay (none)

$$ \hat{c}_d=\frac{d+0.5}{D}$$

Quadratic decay (qudratic)

$$ \hat{c}_d = \frac{1}{(D + 1)^2}$$

Quartic decay (quartic)

$$ \hat{c}_d = \frac{1}{(D + 1)^4}$$

Exponential decay (exp)

$$ \hat{c}_d=\exp\left(\log(c_\mathrm{min})\frac{d+1}{D}\right)$$

Squared-exponential decay (sqexp)

$$ \hat{c}_d=10^{\left(\log_{10}(c_\mathrm{min})\frac{(d+1)^2}{D}\right)}$$

Here $c_\mathrm{min}$ is argument that sets the minimum value of $c_D$.

Once the formula are used the coefficients are normalized such that

$$ c_d = c_\text{factor}\frac{\hat{c}_d}{\sum_{d=1}^D \hat{c}_d}.$$

## References
This benchmark was first proposed [here.](https://doi.org/10.1007/978-94-009-3889-2_33)