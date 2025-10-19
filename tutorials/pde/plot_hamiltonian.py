r"""
Hamiltonian Systems
===================

In this tutorial, we will explore how to solve a Hamiltonian Ordinary Differential Equation (ODE).

Definition
----------

Hamiltonian systems are widely used in physics and engineering to describe the evolution of dynamical systems. These systems are governed by Hamilton's equations:

.. math:: \frac{dp}{dt} = \frac{\partial H}{\partial q}, \quad \frac{dq}{dt} = -\frac{\partial H}{\partial p}

where:

- :math:`p` is the generalized momentum,
- :math:`q` is the generalized coordinate,
- :math:`H(p, q)` is the Hamiltonian function, representing the total energy of the system (typically the sum of kinetic and potential energy).

For any solution :math:`X(t) = (p(t), q(t))`, the Hamiltonian is constant for all :math:`t`, meaning the total energy of the system is conserved:

.. math:: H(p(t), q(t)) = \text{const}

For any smooth function :math:`F(p, q)` the constant level curves of the function are solutions of a Hamiltonian system:

.. math:: \frac{dq}{dt} = \frac{\partial F}{\partial p}, \qquad \frac{dp}{dt} = -\frac{\partial F}{\partial q}

This can be easily checked by noting:

.. math:: \frac{dH}{dt} = \nabla_p H \frac{dp}{dt} + \nabla_q H \frac{dq}{dt} = -\nabla_p H \nabla_q + \nabla_q H \nabla_p = 0

Example
-------

Consider the smooth function:

.. math:: F(p, q) = q^2 + p^2(p - 1)^2

The Hamiltonian system associated with this function is:

.. math::

   \begin{align}
   \frac{dp}{dt} &= -\frac{\partial F}{\partial q} = 2q \\
   \frac{dq}{dt} &= \frac{\partial F}{\partial p} = -2p(p - 1)(2p - 1)
   \end{align}
"""

# %%
import numpy as np
import matplotlib.pyplot as plt


# Define the function F(p, q)
def F(p, q):
    return q**2 + p**2 * (p - 1) ** 2


# Define the derivatives for the Hamiltonian system
def dX_dt(p, q):
    return [2 * q, -2 * p * (p - 1) * (2 * p - 1)]


# Generate a grid of points for p and q
p = np.linspace(-0.25, 1.25, 51) + 1e-8  # Avoid division by zero
q = np.linspace(-0.4, 0.4, 51) + 1e-8
P, Q = np.meshgrid(p, q)

# Compute the values of F(p, q) on the grid
F_values = F(P, Q)

# Compute the derivatives (dp/dt and dq/dt) on the grid
dp, dq = dX_dt(P, Q)

# Normalize the vector field for better visualization
magnitude = np.sqrt(dp**2 + dq**2)
dp /= magnitude
dq /= magnitude

# Plot the phase plot
plt.figure(figsize=(10, 8))
plt.quiver(
    P,
    Q,
    dp,
    dq,
    color="blue",
    alpha=0.6,
    label="Phase Flow",
)

# Plot example level curves of F(p, q)
plt.contour(
    P,
    Q,
    F_values,
    levels=np.array([0.0125, 0.0625, 0.1]),
    colors="red",
    alpha=0.8,
    linewidths=1.5,
)

# Add labels and title
plt.xlabel("p")
plt.ylabel("q")
plt.title("Phase Plot and Level Curves of F(p, q)")
plt.legend(["Phase Flow", "Level Curves"])
plt.grid()
plt.show()

# %% [markdown]
# Along the red lines the value of :math:F(p,q) is constant.
# Note :math:`X=(1/2,0)` is a saddle point.

# %% [markdown]
# Numerically solving the ODE
# ---------------------------
#
# Now let's solve the ODE numerically using symplectic and non-symplectic integrators.

# %%
from pyapprox.util.newton import ParameterizedNewtonResidualMixin
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    ImplicitTimeIntegrator,
    BackwardEulerResidual,
    CrankNicholsonResidual,
)
from pyapprox.util.backends.numpy import NumpyMixin as bkd


class HamiltonianResidual(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    def set_time(self, time: float):
        self._time = time

    def nstates(self) -> int:
        return 2  # Two states: p and q

    def nvars(self) -> int:
        return 0

    def set_param(self, param):
        self._param = param
        # the residual does not depend on the parameters

    def __call__(self, sol: np.ndarray) -> np.ndarray:
        p, q = sol
        return self._bkd.stack([2 * q, -2 * p * (p - 1) * (2 * p - 1)])

    def _jacobian(self, sol: np.ndarray) -> np.ndarray:
        p, q = sol
        jac = self._bkd.zeros((2, 2))
        jac[0, 1] = 2.0
        jac[1, 0] = -2.0 * (6 * p**2 - 6 * p + 1.0)
        return jac


# %% [markdown]
# Symplectic Time Integration
# ---------------------------
#
# Let's solve the Hamiltonian system with a symplectic integrator. A symplectic time integrator is a numerical method specifically designed to solve Hamiltonian systems while preserving their symplectic structure. Hamiltonian systems have a geometric property called symplecticity, which ensures the conservation of certain quantities, such as energy and phase space volume, over time. Here we will use the implicit mid-point rule, also called the Crank-Nicholson integrator.

# %%
init_time = 0.0
final_time = 10.0
deltat = 0.05

time_int = ImplicitTimeIntegrator(
    CrankNicholsonResidual(HamiltonianResidual(backend=bkd)),
    init_time,
    final_time,
    deltat,
)

init_cond = bkd.array([1.0, 0.5])
sols, times = time_int.solve(init_cond)


# Plot the solutions
def plot_solution(sols, times):
    p_values = sols[0]
    q_values = sols[1]

    axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
    axs[0].plot(times, p_values, label="p(t)")
    axs[0].plot(times, q_values, label="q(t)")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Values")
    axs[0].set_title("Hamiltonian System Solutions")
    axs[0].legend()
    axs[0].grid()

    # Plot phase space trajectory
    axs[1].plot(p_values, q_values, label="Phase Space Trajectory")
    axs[1].set_xlabel("p")
    axs[1].set_ylabel("q")
    axs[1].set_title("Phase Space Trajectory")
    axs[1].legend()
    axs[1].grid()


plot_solution(sols, times)

# %% [markdown]
# Note the last phase plot replicates the first one in the tutorial computed using the constant level functions of :math:`F`.
#
# Non-symplectic Time Integration
# -------------------------------
#
# Now use a time integration scheme that is not a symplectic integrator.

# %%
time_int = ImplicitTimeIntegrator(
    BackwardEulerResidual(HamiltonianResidual(backend=bkd)),
    init_time,
    final_time,
    deltat,
)

init_cond = bkd.array([1.0, 0.5])
sols, times = time_int.solve(init_cond)

plot_solution(sols, times)

# %% [markdown]
# The plot shows that if you do not use a symplectic integrator, then the Hamiltonian will not be conserved.
