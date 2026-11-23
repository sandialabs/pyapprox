r"""Second-order-in-time systems reduced to first order.

:math:`M \ddot u = f(u, t; p) + g(u, \dot u, t)` becomes the first-order
system of :math:`w = [u; v]`,

.. math::

    \begin{bmatrix} I & 0 \\ 0 & M \end{bmatrix} \dot w
    = \begin{bmatrix} v \\ f(u, t; p) + g(u, v, t) \end{bmatrix},

a :class:`~pyapprox.ode.mass_form_system.MassFormSystem` every mass-form
stepper and derivative sweep takes unchanged. Nothing here reads a
discretization: the restoring force :math:`f` comes as the first-order
system a stack already builds, and how the two blocks are stored is
injected.
"""
