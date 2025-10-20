r"""
Polynomial Quadrature
=====================

In this tutorial, we will explore quadrature methods for approximating integrals. Polynomial quadrature uses an interpolating polynomial to approximate the integral of a function over a given interval.

Quadrature Formula
------------------

Consider an :math:`N`-degree interpolating polynomial at the :math:`N+1` nodes :math:`x^{(0)}, \ldots, x^{(N)}`. Using this polynomial for quadrature yields:

.. math::
   I_N[f] = \int_{a}^b f(x) \, \mathrm{d}x \approx \int_{a}^b p_N(x) \, \mathrm{d}x = \int_{a}^b \sum_{n=0}^N f(x^{(n)}) \phi_n(x) \, \mathrm{d}x = \sum_{n=0}^N f(x^{(n)}) w_n

Here:

- :math:`p_N(x)` is the interpolating polynomial,
- :math:`\phi_n(x)` are the basis functions,
- :math:`w_n` are the quadrature weights.

Polynomial Interpolation Error
------------------------------
The polynomial interpolation error for :math:`x^{(n)} \in [a, b]` and :math:`\xi \in [a, b]` is given by:

.. math::
   f(x) - p_N(x) = \frac{f^{(N+1)}(\xi(x))}{(N+1)!} \prod_{n=0}^N (x - x^{(n)})

where:

- :math:`f^{(N+1)}` denotes the :math:`N+1` derivative of :math:`f`,
- :math:`\xi(x) \in [a, b]` is a point that depends on :math:`x`.

The error in the approximated integral is then given by:

.. math::
   E[f] = \int_{a}^b f(x) - p_N(x) \, \mathrm{d}x = \int_{a}^b \frac{f^{(N+1)}(\xi(x))}{(N+1)!} \pi_N(x) \, \mathrm{d}x

where:

.. math::
   \pi_N(x) = \prod_{n=0}^N (x - x^{(n)})

Below, we explore specific quadrature rules and their error bounds.

Left Rectangle Rule (:math:`N=0`)
---------------------------------

The left rectangle rule approximates the integral as:

.. math::
   \int_{a}^b f(x) \, \mathrm{d}x \approx (b - a) f(a)

Using the Taylor expansion at :math:`x = a`:

.. math::
   f(x) = f(x_0) + f^{(1)}(x_0)(x - x_0) + \cdots

Let :math:`\Delta = b - a`. Then:

.. math::
   \int_{a}^b f(x) \, \mathrm{d}x = \int_{a}^b f(x_0) + O(\Delta) \, \mathrm{d}x = \Delta f(x_0) + O(\Delta)

The integral of the linear term is zero, so the error is:

.. math::
   \int_{a}^b f(x) \, \mathrm{d}x - (b - a) f(x_0) = O(\Delta)

Right Rectangle Rule (:math:`N=0`)
----------------------------------

The right rectangle rule approximates the integral as:

.. math::
   \int_{a}^b f(x) \, \mathrm{d}x \approx (b - a) f(b)

Midpoint Rule (:math:`N=0`)
---------------------------

The midpoint rule approximates the integral as:

.. math::
   \int_{a}^b f(x) \, \mathrm{d}x \approx (b - a) f\left(\frac{a + b}{2}\right)

Using the Taylor expansion:

.. math::
   f(x) = f(x_0) + f^{(1)}(x_0)(x - x_0) + \frac{f^{(2)}(x_0)(x - x_0)^2}{2!} + \cdots

Let :math:`\Delta = b - a`. Then:

.. math::
   \int_{a}^b f(x) \, \mathrm{d}x = \int_{a}^b f(x_0) - \Delta f^{(1)}(x_0)(x - x_0) + O(\Delta^2) \, \mathrm{d}x = \Delta f(x_0) + O(\Delta^2)

The integral of the linear term is zero, so the error is:

.. math::
   \int_{a}^b f(x) \, \mathrm{d}x - (b - a) f(x_0) = O(\Delta^2)

Trapezoid Rule (:math:`N=1`)
----------------------------

The trapezoid rule uses a linear interpolation (:math:`N=1`) of :math:`f` at the points :math:`x_0 = a, x_1 = b`:

.. math::
   I_1[f] = \frac{b - a}{2} \left(f(a) + f(b)\right)

Applying the quadrature error formula:

.. math::
   E_1[f] = \int_{a}^b \frac{f^{(N+1)}(\xi(x))}{(N+1)!} \pi_N(x) \, \mathrm{d}x = \frac{f^{(2)}(c)}{2!} \frac{(b - a)^3}{6} = \frac{f^{(2)}(c)(b - a)^3}{12}

for some :math:`c \in (a, b)`.

Here, we applied the mean value theorem for definite integrals, which states:

If :math:`f : [a, b] \to \mathbb{R}` is continuous and :math:`g` is an integrable function that does not change sign on :math:`[a, b]`, then there exists :math:`c \in (a, b)` such that:

.. math::
   \int_a^b f(x) g(x) \, \mathrm{d}x = f(c) \int_a^b g(x) \, \mathrm{d}x

Simpson's Rule (:math:`N=2`)
----------------------------

Simpson's rule uses a quadratic interpolation (:math:`N=2`) of :math:`f` at the points :math:`x_0 = a, x_1 = \frac{a + b}{2}, x_2 = b`:

.. math::
   I_2[f] = \frac{x_2 - x_1}{6} \left(f(x_0) + 4f(x_1) + f(x_2)\right)

Substituting the Taylor expansion of :math:`f` around :math:`x_1`:

.. math::
   f(x) = f(x_1) + f^{(1)}(x_1)(x - x_1) + \frac{f^{(2)}(x_1)(x - x_1)^2}{2!} + \frac{f^{(3)}(x_1)(x - x_1)^3}{3!} + \cdots

Evaluating at the endpoints:

.. math::
   f(x_0) = f(x_1) - \Delta f^{(1)}(x_1) + \frac{\Delta^2 f^{(2)}(x_1)}{2!} - \frac{\Delta^3 f^{(3)}(x_1)}{3!} + \cdots

.. math::
   f(x_2) = f(x_1) + \Delta f^{(1)}(x_1) + \frac{\Delta^2 f^{(2)}(x_1)}{2!} + \frac{\Delta^3 f^{(3)}(x_1)}{3!} + \cdots

Substituting these values:

.. math::
   f(x_0) + f(x_2) = 2f(x_1) + f^{(2)}(x_1)\Delta^2 + \frac{2f^{(4)}(x_1)\Delta^4}{4!} + \cdots

The odd derivative terms cancel. Then:

.. math::
   \frac{\Delta}{3} \left(f(x_0) + 4f(x_1) + f(x_2)\right) = 2\Delta f(x_1) + \frac{\Delta^3}{3}f^{(2)}(x_1) + \frac{\Delta^5}{36}f^{(4)}(x_1) + \cdots

Comparing this to the exact integral:

.. math::
   \int_{x_0}^{x_2} f(x) \, \mathrm{d}x = 2\Delta f(x_1) + \frac{f^{(2)}(x_1)\Delta^3}{3} + \frac{f^{(4)}(x_1)\Delta^5}{60} + \cdots

The error is:

.. math::
   E_2[f] = -\frac{\Delta^4}{90}f^{(4)}(x_1) + O(\Delta^5)

Note that quadrature rules with even :math:`N` (e.g., Simpson's rule) can integrate degree :math:`N+1` polynomials exactly, while quadrature rules with odd :math:`N` (e.g., the trapezoid rule) can only integrate degree :math:`N` polynomials exactly.
"""
