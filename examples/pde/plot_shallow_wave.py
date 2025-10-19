r"""
Shallow Water Wave Equation
===========================

This tutorial demonstrates how to solve the shallow water wave equations in conservative form using the `pyapprox` library. The shallow water wave equations are a set of hyperbolic partial differential equations that describe the flow below a pressure surface in a fluid.

.. contents::
   :local:
   :depth: 2

The equations are solved using a **spectral collocation method**, which is a numerical technique that approximates solutions to partial differential equations using global basis functions, such as polynomials or Fourier series.

Strong Form of the Shallow Water Wave Equations
-----------------------------------------------

The shallow water wave equations in conservative form are given by:

.. math:: \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = \mathbf{S}(\mathbf{U}),

where:

- :math:`\mathbf{U} = \begin{bmatrix} h \\ hu \\ hv \end{bmatrix}` is the vector of conserved variables:
- :math:`h`: fluid height,
- :math:`hu`: momentum in the :math:`x`-direction,
- :math:`hv`: momentum in the :math:`y`-direction.
- :math:`\mathbf{F}(\mathbf{U})` is the flux tensor:
  
.. math::

  \mathbf{F}(\mathbf{U}) = \begin{bmatrix}
  hu & hv \\
  hu^2 + \frac{1}{2}gh^2 & huv \\
  huv & hv^2 + \frac{1}{2}gh^2
  \end{bmatrix},
- :math:`g` is the gravitational acceleration.
- :math:`\mathbf{S}(\mathbf{U})` is the source term, which accounts for external forces such as bed slope effects.

The goal is to compute the evolution of :math:`\mathbf{U}` over time, given initial and boundary conditions.

"""

# %% [markdown]
# Import Required Libraries
# -------------------------
import os
import sys
import numpy as np
from pyapprox.util.backends.template import Array
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.pde.collocation.adjoint import TransientAdjointFunctional
from pyapprox.pde.collocation.parameterized_pdes import ShallowWaterWaveModel
from pyapprox.pde.collocation.timeintegration import (
    BackwardEulerResidual,
    SymplecticMidpointResidual,
    CrankNicholsonResidual,
)
from pyapprox.pde.collocation.functions import (
    animate_transient_2d_vector_solution,
    get_water_cmap,
)
from pyapprox.util.newton import NewtonSolver

if sys.platform == "darwin":
    import matplotlib

    matplotlib.use("TKAgg")

# %% [markdown]
# Numerical Method: Spectral Collocation
# --------------------------------------
#
# The spectral collocation method is used to solve the shallow water wave equations. This method approximates the solution using a set of global basis functions, such as Chebyshev or Legendre polynomials, defined on a computational grid. The derivatives are computed at the collocation points, ensuring high accuracy.
#
# Advantages of the spectral collocation method:
#
# - High-order accuracy.
# - Efficient for smooth solutions.
#
# The method is implemented in the `pyapprox` library, which provides tools for solving parameterized partial differential equations.\
#
#
# Set Up the Domain and Model
# ----------------------------

# Setup domain
bkd = TorchMixin
np.random.seed(1)

# Define time period
init_time, final_time, deltat = 0, 20, 0.5

# Initialize Newton solver
newton_solver = NewtonSolver(
    verbosity=2,
    maxiters=5,
    atol=1e-6,
    rtol=1e-6,
)

# Initialize the shallow water wave model
model = ShallowWaterWaveModel(
    init_time,
    final_time,
    deltat,
    BackwardEulerResidual,
    newton_solver=newton_solver,
    backend=bkd,
)

# %% [markdown]
# Define Initial Conditions
# -------------------------
#
# The initial conditions for the shallow water wave equations include the fluid height :math:`h`, momentum in the :math:`x`-direction :math:`hu`, and momentum in the :math:`y`-direction :math:`hv`. These are specified as a sample vector.
#
#
# The initial conditions for the shallow water wave equations are computed using a combination of Beta distribution functions. The fluid surface is initialized as the sum of two Beta-shaped surfaces, each defined by specific shape parameters. The equations for computing the initial surface are as follows:
#
# Beta Surface Function
# ^^^^^^^^^^^^^^^^^^^^^
#
# The Beta surface function is defined as the tensor product of two univariate functions( which resemble the PDFs of Beta random variables but should not be interpreted as such here):
#
# .. math:: \beta(x) = \frac{x_1^{a_0 - 1} (1 - x_1)^{b_0 - 1} \cdot x_2^{a_1 - 1} (1 - x_2)^{b_1 - 1}}{\text{Beta}(a_0, b_0) \cdot \text{Beta}(a_1, b_1) \cdot 20},
#
#
# where:
#
# - :math:`x_1, x_2` are normalized spatial coordinates, computed as:
#
# .. math:: x_n = \frac{x}{\text{bounds}},
#
# - :math:`\text{bounds}` representing the domain bounds.
# - :math:`a_0, b_0, a_1, b_1` are shape parameters that control the shape of the Beta distribution.
# - :math:`\text{Beta}(a, b)` is the Beta function:
#
# .. math::  \text{Beta}(a, b) = \int_0^1 t^{a-1}(1-t)^{b-1} \, dt.
#
#
# Initial Surface Function
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The initial surface is computed as the sum of two Beta surface functions, each defined by a set of shape parameters:
#
# .. math:: h(x) = \beta_0(x) + \beta_1(x),
#
# where:
#
# - :math:`\beta_0(x)` is the Beta surface function computed using the shape parameters :math:`[a_0, b_0, a_1, b_1]`.
# - :math:`\beta_1(x)` is the Beta surface function computed using the shape parameters :math:`[a_2, b_2, a_3, b_3]`.
#
# The shape parameters for :math:`\beta_0(x)` and :math:`\beta_1(x)` are provided as input to the model, and the initial surface :math:`h(x)` is computed by evaluating these functions at the spatial coordinates :math:`x`.
#
# This approach allows for flexible initialization of the fluid surface, enabling the simulation of various scenarios by adjusting the shape parameters.

# Define sample parameters
sample = bkd.array([25, 20, 20, 20, 5, 20, 20, 20])[:, None]

# %%
# Define the Boundary Conditions
# ------------------------------
#
# Reflective boundary conditions are applied to the shallow water wave equations. These conditions ensure that the velocity component perpendicular to the boundary normal is set to zero, effectively modeling a no-flow condition across the boundary. This is essential for simulating a closed domain.
#
#
# Let :math:`\mathbf{n}` denote the unit normal vector to the boundary, and let :math:`\mathbf{v} = \begin{bmatrix} u \\ v \end{bmatrix}` represent the velocity vector, where:
#
# - :math:`u`: velocity component in the :math:`x`-direction,
# - :math:`v`: velocity component in the :math:`y`-direction.
#
# The reflective boundary conditions are defined as:
#
# - On **horizontal boundaries** (e.g., :math:`y = 0` or :math:`y = L_y`), the vertical velocity :math:`v` is set to zero:
#
# .. math::
#    \mathbf{v} \cdot \mathbf{n} = v = 0.
#
# - On **vertical boundaries** (e.g., :math:`x = 0` or :math:`x = L_x`), the horizontal velocity :math:`u` is set to zero:
#
# .. math::
#    \mathbf{v} \cdot \mathbf{n} = u = 0.
#
# These conditions ensure that the fluid does not flow across the boundaries, effectively reflecting the flow back into the domain. This is par


# %% [markdown]
# Solve the Shallow Water Wave Equations
# --------------------------------------
#
# The equations are solved using the backward Euler method for time integration. The solution is saved to a file for later use.

solfilename = "shallowwater.npz"
if not os.path.exists(solfilename):
    model.forward_solve(sample)
    np.savez(solfilename, sols=model._sols, times=model._times)
    sols = model._sols
    times = model._times
else:
    data = np.load(solfilename)
    sols = bkd.asarray(data["sols"])
    times = bkd.asarray(data["times"])

# %% [markdown]
# Visualize the Results
# ----------------------
#
# The solution is visualized using an animation that shows the evolution of the fluid height :math:`h` over time. The animation is saved as a GIF file.

surface_plot_kwargs = {
    "cmap": get_water_cmap(),
    "alpha": 1,
    "antialiased": True,
    "linewidth": 0,
    "rstride": 1,
    "cstride": 1,
}
contour_plot_kwargs = {"cmap": get_water_cmap()}

ani = animate_transient_2d_vector_solution(
    model.basis(),
    sols,
    times,
    model.physics().ncomponents(),
    [0],
    [0],
    51,
    contour_plot_kwargs,
    surface_plot_kwargs,
)

# Save the animation
ani.save("shallowwater.gif", dpi=100)
