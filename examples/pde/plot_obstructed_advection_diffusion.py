r"""
Coupled Navier-Stokes and Advection-Diffusion Equations
=======================================================

This tutorial demonstrates how to use the `ObstructedAdvectionDiffusion` model to solve the Navier-Stokes equations to compute the velocity field and subsequently solve the advection-diffusion equation to model the propagation of a source through the domain.

The tutorial includes the following steps:
1. Define the model parameters.
2. Specify the forcing function.
3. Compute the forcing values at quadrature points.
4. Solve for Karhunen-Loève Expansion (KLE) coefficients.
5. Visualize the forcing function and KLE eigenvectors.
6. Solve the advection-diffusion equation.
7. Visualize the concentration snapshots and animate the solution.

.. contents::
   :local:
   :depth: 2

Model Setup
-----------

First, we define the parameters for the model, including the number of terms in the Karhunen-Loève Expansion (KLE), the final simulation time, and the hyperparameters for the KLE.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyapprox.pde.galerkin.parameterized import (
    ObstructedAdvectionDiffusion,
    KLEHyperParameters,
)
from pyapprox.util.backends.numpy import NumpyMixin

# Define the number of KLE terms and final simulation time
nterms = 10
final_time = 1.5

# Define KLE hyperparameters
kle_hyperparams = KLEHyperParameters(0.5, 1.0, np.inf, nterms)

# Initialize the ObstructedAdvectionDiffusion model
navier_stokes_params = np.array([10.0, 2.0, 2.0])
model = ObstructedAdvectionDiffusion(
    3, 3, 0.1, final_time, kle_hyperparams, True, navier_stokes_params
)

# %%
# Defining the Forcing Function
# -----------------------------
#
# The forcing function represents the source term in the advection-diffusion equation. In this example, the forcing function is defined as a Gaussian distribution centered at a specific location in the domain.


def forcing(x):
    scale = 100.0
    loc = [0.15, 0.4]
    return np.exp(-scale * ((x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2))


# %%
# Computing Forcing Values at Quadrature Points
# ---------------------------------------------
#
# The forcing values are computed at quadrature points using the basis interpolation and projection methods provided by the model. The logarithm of the forcing values is then computed, ensuring that no values are zero.


# Compute forcing values at quadrature points
forcing_vals = model._basis.interpolate(
    model._basis.project(forcing)
).flatten()

# Ensure that the log of forcing values is not zero
forcing_vals = np.maximum(forcing_vals, 1e-8)

# %%
# Solving for KLE Coefficients
# ----------------------------
#
# The KLE coefficients are computed in log space using the weighted eigenvectors of the KLE.

# Solve for KLE coefficients in log space
params = NumpyMixin.lstsq(
    model.kle().weighted_eigenvectors(), np.log(forcing_vals)
)

# %%
# Visualizing the Forcing Function and KLE Eigenvectors
# -----------------------------------------------------
#
# The forcing function and KLE eigenvectors are visualized to understand the spatial distribution of the source term and the basis functions used in the Karhunen-Loève Expansion.

# Plot the forcing function
model.plot_forcing(params, colorbar=True)

# Plot KLE eigenvectors
axs = plt.subplots(1, 3, figsize=(3 * 8, 6), sharey=True)[1]
model.plot_kle_eigenvecs(np.array([0, nterms // 2, -1]), axs)
plt.savefig("advec_diff_forcing_eigenvecs.pdf", bbox_inches="tight")

# %%
# Solving the Governing Equations
# -------------------------------
#
# The advection-diffusion equation is solved using the model with the computed KLE coefficients. The Navier-Stokes equations are solved using the default parameters

# Set the KLE coefficients in the model
model.set_params(params)

# Solve the advection-diffusion equation
sols, times = model.solve()
print("Advection-Diffusion ndof:", sols.shape[0])

# %%
# Plot the Navier-Stokes Velocity Field
# -------------------------------------
#
# Plot the velocity field
ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)[1]
stokes_sol = model.stokes_model().solve()
print("Navier-Stokes ndof:", stokes_sol.shape[0])
model.stokes_model().plot_velocity_magnitude(stokes_sol, ax=ax)
model.stokes_model().plot_velocity_field(stokes_sol, ax=ax)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# plt.savefig("advec_diff_velocity.pdf", bbox_inches="tight")

# %%
# Visualizing the Concentration Snapshots
# ---------------------------------------
#
# Snapshots of the concentration field at different time steps are visualized to observe the propagation of the source through the domain.

# Plot concentration snapshots
snapshot_indices = np.array([-1])
nsnapshots = snapshot_indices.shape[0]
axs = plt.subplots(1, nsnapshots, figsize=(nsnapshots * 8, 6), sharey=True)[1]
model.plot_concentration_snapshots(sols, snapshot_indices, axs)
axs.set_xlim(0, 1)
axs.set_ylim(0, 1)
# plt.savefig("advec_diff_concentration_snapshots.pdf", bbox_inches="tight")

# %%
# Animating the Solution
# ----------------------
#
# An animation of the solution is created to visualize the time evolution of the concentration field. The animation is saved as a GIF file.

# Create an animation of the solution
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
ani = model.animate_concentration_snapshots(fig, axs, sols)

# Save the animation as a GIF file
ani.save("solution_animation.gif", fps=sols.shape[1] / final_time, dpi=150)

# %% [markdown]
# Conclusion
# ----------
#
# In this tutorial, we demonstrated how to use the `ObstructedAdvectionDiffusion` model to solve the Navier-Stokes equations for the velocity field and the advection-diffusion equation for the propagation of a source through the domain. We visualized the forcing function, KLE eigenvectors, and concentration snapshots, and created an animation of the solution.
