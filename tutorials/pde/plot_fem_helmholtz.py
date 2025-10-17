r"""Solving the Helmholtz Equation on an Octagonal Domain with Finite Elements
==========================================================================

This tutorial demonstrates how to solve the Helmholtz equation using finite elements on an octagonal domain. The domain has speakers on each wall, corresponding to different Neumann boundary conditions. We will use the ``OctagonalHelmholtz`` model to compute the sound pressure and visualize the results.

Deriving the Weak Form for the Helmholtz Equation
-------------------------------------------------

The Helmholtz equation is a partial differential equation (PDE) that models wave propagation. In its strong form, the equation is given by:

.. math::

   -\Delta u - k^2 u = f \quad \text{in } \Omega,

where:

- :math:`u` is the unknown solution (e.g., sound pressure),
- :math:`\Delta u` is the Laplacian of :math:`u`,
- :math:`k` is the wave number (related to frequency and wave speed),
- :math:`f` is the source term,
- :math:`\Omega` is the domain.

Boundary conditions are typically specified as:
- Dirichlet boundary condition: :math:`u = g_D` on :math:`\Gamma_D`,
- Neumann boundary condition: :math:`\frac{\partial u}{\partial n} = g_N` on :math:`\Gamma_N`.

Weak Formulation
^^^^^^^^^^^^^^^^

To derive the weak form, we multiply the strong form of the Helmholtz equation by a test function :math:`v` (from a suitable function space) and integrate over the domain :math:`\Omega`. This gives:

.. math::

   \int_{\Omega} (-\Delta u - k^2 u) v \, dx = \int_{\Omega} f v \, dx.

Using integration by parts on the Laplacian term :math:`-\Delta u`, we obtain:

.. math::

   \int_{\Omega} \nabla u \cdot \nabla v \, dx - \int_{\Gamma} \frac{\partial u}{\partial n} v \, ds - \int_{\Omega} k^2 u v \, dx = \int_{\Omega} f v \, dx.

Here:

- :math:`\nabla u \cdot \nabla v` represents the gradient term,
- :math:`\frac{\partial u}{\partial n}` is the normal derivative of :math:`u` on the boundary :math:`\Gamma`,
- :math:`k^2 u v` is the mass term,
- :math:`f v` is the source term.

Applying Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The boundary integral term :math:`\int_{\Gamma} \frac{\partial u}{\partial n} v \, ds` depends on the boundary conditions:
- For Neumann boundary conditions, :math:`\frac{\partial u}{\partial n} = g_N`, so the term becomes :math:`\int_{\Gamma_N} g_N v \, ds`.
- For Dirichlet boundary conditions, :math:`u = g_D` is enforced directly, and the test function :math:`v` vanishes on :math:`\Gamma_D`.

Substituting the boundary conditions, the weak form becomes:

.. math::

   \int_{\Omega} \nabla u \cdot \nabla v \, dx - \int_{\Omega} k^2 u v \, dx = \int_{\Omega} f v \, dx + \int_{\Gamma_N} g_N v \, ds.

Final Weak Form
^^^^^^^^^^^^^^^

The final weak form of the Helmholtz equation is:

Find :math:`u \in V` such that:

.. math::

   a(u, v) = l(v) \quad \forall v \in V,

where:

- :math:`a(u, v) = \int_{\Omega} \nabla u \cdot \nabla v \, dx - \int_{\Omega} k^2 u v \, dx`,
- :math:`l(v) = \int_{\Omega} f v \, dx + \int_{\Gamma_N} g_N v \, ds`,
- :math:`V` is the appropriate function space (e.g., :math:`H^1(\Omega)`).

Problem Setup
-------------

The Helmholtz equation describes the propagation of sound waves in a domain.In this example, we use an octagonal domain with Neumann boundary conditions representing speakers on each wall. The equation is solved using finite element methods.

Boundary Conditions
-------------------

We set Neumann boundary conditions for the boundaries that are purely imaginary so that the solution is purely imaginary. Specifically:

- For the speaker segments, we set:

  .. math::

     \frac{\partial u}{\partial n} = 1.204 \cdot \omega \cdot A,

  where :math:`\omega` is the angular frequency and :math:`A` is the prescribed amplitude of the sound pressure.

- For the cabinet segments, we set:

  .. math::

     \frac{\partial u}{\partial n} = 0.

The speakers are centered on each side of the octagon and have a width equal to 60% of the length of each side. These boundary conditions model the behavior of speakers emitting sound waves and the cabinet walls reflecting sound without emitting any waves.

Define the Octagonal Helmholtz Model
------------------------------------

We initialize the ``OctagonalHelmholtz`` model with the following parameters:

- Frequency :math:`f`: 400 Hz
- Angular Frequency :math:`\omega`: :math:`2\pi f` radians/s
- Wave speed in outer domain: 343 m/s
- Wave speed in middle domain: 6320 m/s
- Wave speed in inner domain: 343 m/s
- Speaker amplitudes :math:`A`: 1 Pa
- Octagonal Radius: 0.5 m

"""

# Import the required modules
import numpy as np
from skfem.visuals.matplotlib import plot, plt
from pyapprox.pde.galerkin.parameterized import OctagonalHelmholtz


# Initialize the Model
model = OctagonalHelmholtz(2, 5, 400)


# %%
# Solve the Helmholtz Equation
# -----------------------------
#
# We solve the Helmholtz equation using the default parameters of the model.

pressure = model.solve()


# %%
# Solve with Perturbed Parameters
# -------------------------------
#
# We perturb the model parameters (the wave speeds) and solve the Helmholtz equation again.
model.set_params(np.array([400, 4000, 343]))
pressure_perturbed = model.solve()

# %%
# Visualize Results
# -----------------
#
# We visualize the results for both the default and perturbed solutions.


pressures = [pressure, pressure_perturbed]
axs = plt.subplots(1, 2, figsize=(2 * 8, 6), sharey=True)[1]
for ii in range(2):
    _ = plot(
        model.basis(),
        pressures[ii],
        ax=axs[ii],
        colorbar=True,
    )

# %%
# Conclusion
# ----------
#
# This tutorial demonstrated how to solve the Helmholtz equation on an octagonal domain using finite elements. We explored the effects of perturbing the model parameters and visualized the results. The ``OctagonalHelmholtz`` model provides a convenient framework for solving acoustic wave propagation problems in complex domains.
