r"""
Computing Gradients of ODEs Using Adjoints
==========================================

In this tutorial, we will explore the computation of gradients using adjoints for time-dependent problems. For linear time-dependent residuals, the adjoint method can be formulated using linear algebra, and solving adjoints becomes analogous to integrating an ODE backward in time.

This tutorial assumes familarity with the introductory tutorial on computing gradients using adjoints for arbitrary functions.

Intuition
---------

Solving adjoints for time-dependent problems is akin to integrating an ODE backward in time. For example, consider the ODE parameterized by optimization variable :math:`\theta`

.. math:: \dydx{y}{t} = f(y, \theta)

Here we focus on computing the gradient of an objective function that depends on :math:`y` and :math:`\theta` by solving adjoints of the discretized ODE. However, it must be noted that this is not equivalent to solving adjoint equation of an ODE. The gradients obtained by these two different approaches will only coincide, when the time step asymptotes to zero. Up to numerical precision issues the approach described here will return the gradients of the discretized ODE that would also be obtained using automatic differentiation. However, the approach used here will be faster and more memory efficient as it takes advantage of the structure of ODE time stepping schemes.

Using using backward Euler to evolve a linear version of this ODE for two time-steps from an initial condition looks like solving the set of simultaneous equations:

.. math::

   c(y, \theta) =
   \begin{bmatrix}
     A_{00}(t_0) & 0 & 0\\
     A_{10}(t_0) & A_{11}(t_1) & 0\\
     0 & A_{21}(t_1) & A_{22}(t_2)
   \end{bmatrix}
   \begin{bmatrix}
   y_0\\ y_1\\ y_2
   \end{bmatrix}
   -
   \begin{bmatrix}
   b_0\\ b_1\\ b_2
   \end{bmatrix}

We can apply the three gradient steps to formulate and solve the time-dependent discretized adjoint equation:

.. math::

   \begin{bmatrix}
     A_{00}(t_0)^\top & A_{10}(t_0)^\top & 0\\
     0 & A_{11}(t_1)^\top & A_{21}(t_1)^\top\\
     0 & 0 & A_{22}(t_2)^\top
   \end{bmatrix}
   \begin{bmatrix}
   \lambda_0\\ \lambda_1\\ \lambda_2
   \end{bmatrix}
   =
   \begin{bmatrix}
   \nabla_{y_0} J(y_0, \theta)\\ \nabla_{y_1} J(y_1, \theta)\\ \nabla_{y_2} J(y_2, \theta)
   \end{bmatrix}

The gradient can then be computed using the usual formula

.. math:: \nabla_\theta J(\theta) = \nabla_\theta J(y(\theta), \theta) + \lambda^\top \nabla_\theta c(y(\theta), \theta)

Example: Linear ODE
-------------------

Let :math:`\theta = [a, b]^\top` and consider:

.. math:: \frac{\mathrm{d}y}{\mathrm{d}t} = by \quad y(0) = a

With the exact solution:

.. math:: y(t, \theta) = a \exp(bt)

First, focus on computing the gradient of the functional:

.. math:: J(\theta) = \int_0^T y(t, \theta) \, \mathrm{d}t = \int_0^T a \exp(bt) \, \mathrm{d}t = \frac{a}{b}(\exp(bT) - 1)

i.e.,

.. math:: \frac{\mathrm{d}J}{\mathrm{d}\theta} = 
   \begin{bmatrix}
   \frac{1}{b}(\exp(bT) - 1) \\
   \frac{aT}{b} \exp(bT) - \frac{a}{b^2}(\exp(bT) - 1)
   \end{bmatrix}




Step 1: Compute Forward Solution with Backward Euler
----------------------------------------------------

.. math::

   y = \begin{bmatrix}
   y^{(0)} \\ y^{(1)} \\ y^{(2)}
   \end{bmatrix}
   =
   \begin{bmatrix}
   a \\
   \frac{y^{(0)}}{\Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1} \\
   \frac{y^{(1)}}{\Delta^{(2)} b^{2} \left(t_{2} + 2\right) + 1}
   \end{bmatrix}

.. math::

   y = \begin{bmatrix}
   a \\
   \frac{a}{\Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1} \\
   \frac{a}{\left(\Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1\right) \left(\Delta^{(2)} b^{2} \left(t_{2} + 2\right) + 1\right)}
   \end{bmatrix}

Step 2: Compute Adjoint Solution
--------------------------------

.. math::

   \lambda = -\left(\nabla_y C(y(\theta), \theta)^\top\right)^{-1} \nabla_y J(y(\theta), \theta)^\top

Define Objective
----------------
Backward Euler integration corresponds to right-sided piecewise constant quadrature in time so that

.. math:: J(y,\theta)=\int_0^T y(t,\theta) \dx{t} \approx\sum_{n=1}^N \Delta^{(n)} y^{(n)}

Here the sum starts from 1 and not 0 because the piecewise constant quadrature is right sided.


Thus, for our 2 step (and initial condition)

.. math::   J(y(\theta), \theta) = \Delta^{(1)} y^{(1)} + \Delta^{(2)} y^{(2)}

The following list summarizes common examples of the connections between quadrature rules and time integration:

- Explicit Euler: left-sided rectangular quadrature rule
- Explicit Mid-Point Rule (Heun's Method): piecewise-linear (trapezoid) quadrature.
- Implicit Euler: right-sided rectangular quadrature rule
- Implicit Mid-Point Rule (Crank-Nicholson): piecewise-linear (trapezoid) quadrature.


Compute Necessary Quantities
----------------------------

.. math::

   C(y(\theta), \theta) =
   \begin{bmatrix}
   - a + y^{(0)} \\
   \Delta^{(1)} b^{2} y^{(1)} \left(t_{1} + 2\right) - y^{(0)} + y^{(1)} \\
   \Delta^{(2)} b^{2} y^{(2)} \left(t_{2} + 2\right) - y^{(1)} + y^{(2)}
   \end{bmatrix}
   = 0

.. math::

   \nabla_y C(y(\theta), \theta) =
   \begin{bmatrix}
   1 & 0 & 0 \\
   -1 & \Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1 & 0 \\
   0 & -1 & \Delta^{(2)} b^{2} \left(t_{2} + 2\right) + 1
   \end{bmatrix}

.. math::

   \nabla_y J(y(\theta), \theta) =
   \begin{bmatrix}
   0 & \Delta^{(1)} & \Delta^{(2)}
   \end{bmatrix}

Compute via Backward Substitution
---------------------------------

Solve the adjoint equation using backward substitution, i.e., time step backward in time:

.. math::

   \begin{bmatrix}
   1 & -1 & 0 \\
   0 & \Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1 & -1 \\
   0 & 0 & \Delta^{(2)} b^{2} \left(t_{2} + 2\right) + 1
   \end{bmatrix}
   \lambda =
   -\begin{bmatrix}
   0 \\ \Delta^{(1)} \\ \Delta^{(2)}
   \end{bmatrix}

.. math::

   \lambda =
   \begin{bmatrix}
   \frac{- \Delta^{(1)} \Delta^{(2)} b^{2} t_{2} - 2 \Delta^{(1)} \Delta^{(2)} b^{2} - \Delta^{(1)} - \Delta^{(2)}}{\Delta^{(1)} \Delta^{(2)} b^{4} t_{1} t_{2} + 2 \Delta^{(1)} \Delta^{(2)} b^{4} t_{1} + 2 \Delta^{(1)} \Delta^{(2)} b^{4} t_{2} + 4 \Delta^{(1)} \Delta^{(2)} b^{4} + \Delta^{(1)} b^{2} t_{1} + 2 \Delta^{(1)} b^{2} + \Delta^{(2)} b^{2} t_{2} + 2 \Delta^{(2)} b^{2} + 1} \\
   \frac{- \Delta^{(1)} \Delta^{(2)} b^{2} t_{2} - 2 \Delta^{(1)} \Delta^{(2)} b^{2} - \Delta^{(1)} - \Delta^{(2)}}{\Delta^{(1)} \Delta^{(2)} b^{4} t_{1} t_{2} + 2 \Delta^{(1)} \Delta^{(2)} b^{4} t_{1} + 2 \Delta^{(1)} \Delta^{(2)} b^{4} t_{2} + 4 \Delta^{(1)} \Delta^{(2)} b^{4} + \Delta^{(1)} b^{2} t_{1} + 2 \Delta^{(1)} b^{2} + \Delta^{(2)} b^{2} t_{2} + 2 \Delta^{(2)} b^{2} + 1} \\
   - \frac{\Delta^{(2)}}{\Delta^{(2)} b^{2} t_{2} + 2 \Delta^{(2)} b^{2} + 1}
   \end{bmatrix}


Let :math:`y` be the concatenation of all states of the discretized ODE, that is:

.. math::

   y = [y^{(0)}, y^{(1)}, \ldots, y^{(N)}]^\top

Then the linearity of the ODE leads to the following matrix equation for :math:`N=3` (including the initial condition) computed using backward Euler time-stepping:

.. math::

   c(y, \theta) =
   \begin{bmatrix}
     1 & 0 & 0\\
     -1 & (1+\Delta^{(1)} b^2) & 0\\
     0 & -1 & (1+\Delta^{(2)} b^2)
   \end{bmatrix}
   \begin{bmatrix}
     y^{(0)}\\ y^{(1)}\\ y^{(2)}
   \end{bmatrix}
   -
   \begin{bmatrix}
     a\\ 0\\ 0
   \end{bmatrix}
   = 0

This is a lower-diagonal matrix, which can be solved by forward substitution, equivalent to the time-stepping scheme laid out previously.

Gradient Computation
--------------------

The gradient computation involves the following matrices:

.. math::

   \nabla_y c(y, \theta) =
   \begin{bmatrix}
     1 & 0 & 0\\
     -1 & (1+\Delta^{(1)} b^2) & 0\\
     0 & -1 & (1+\Delta^{(2)} b^2)
   \end{bmatrix}, \quad
   \nabla_y J(y, \theta) =
   \begin{bmatrix}
   0 & \Delta^{(1)} & \Delta^{(2)}
   \end{bmatrix}

.. math::

   \nabla_\theta c(y, \theta) =
   \begin{bmatrix}
   0 & -1\\
   2\Delta^{(1)} b y^{(1)} & 0\\
   2\Delta^{(2)} b y^{(2)} & 0
   \end{bmatrix}

.. math::

   \nabla_\theta J(y, \theta) =
   \begin{bmatrix}
   0 & 0
   \end{bmatrix}


Adjoint Equation
----------------

Solve the adjoint equation using backward substitution:

.. math::

   -\nabla_y c(y(\theta), \theta)^\top \lambda = \nabla_y J(y(\theta), \theta)^\top

This is equivalent to time-stepping backward in time from the final solution:

.. math::

   \begin{align*}
   \lambda^{(2)} &= -\Delta^{(2)}(1+\Delta^{(2)} b^2)^{-1} \\
   -\lambda^{(1)}(1+\Delta^{(1)} b^2) + \lambda^{(2)} &= -\Delta^{(1)} \\
   \lambda^{(1)} &= (-\Delta^{(2)}(1+\Delta^{(2)} b^2)^{-1} - \Delta^{(1)})(1+\Delta^{(1)} b^2)^{-1} \\
   &= (-\Delta^{(2)} - \Delta^{(1)}(1+\Delta^{(2)} b^2))(1+\Delta^{(1)} b^2)^{-1}(1+\Delta^{(2)} b^2)^{-1} \\
   \lambda^{(0)} &= \lambda^{(1)}
   \end{align*}


Final Gradient Computation
--------------------------

Finally, compute the gradient:

.. math::

   \nabla_\theta J(\theta) = \nabla_\theta J(y(\theta), \theta) + \lambda^\top \nabla_\theta c(y(\theta), \theta)

Substituting the computed values:

.. math::

   \nabla_\theta J(\theta) =
   \begin{bmatrix}
   0 \\ 0
   \end{bmatrix} +
   \begin{bmatrix}
   \frac{2 b a \left(\Delta^{(1)} \Delta^{(2)} \left(2 \Delta^{(2)} b^2+1\right)+\left(\Delta^{(1)} \Delta^{(2)} b^2+\Delta^{(1)}\right)^2+(\Delta^{(2)})^2\right)}{\left(1+{\Delta^{(1)}} b^2\right)^2 \left(1+{\Delta^{(2)}} b^2\right)^2} \\
   \frac{{\Delta^{(1)}} {\Delta^{(2)}} b^2+{\Delta^{(1)}}+{\Delta^{(2)}}}{\left(1+{\Delta^{(1)}} b^2\right) \left(1+{\Delta^{(2)}} b^2\right)}
   \end{bmatrix}

We can verify these gradients by taking the gradient of the objective directly:

.. math::

   J(y, \theta) = \int_0^T y(t, \theta) dt \approx \sum_{n=1}^N \Delta^{(n)} y^{(n)}

A Non-linear Example
--------------------

The remainder of this tutorial demonstrates how to compute gradients using adjoints for solving an ordinary differential equation (ODE). 
We will use the provided Python code to compute the gradient of the functional:

.. math:: J(y, \theta) = J = \int_{t_0}^{t_f} y_1(t) \, dt

where  :math:`y=[y_1,y_2]` is the solution to the ODE:

..  math:: \frac{dy}{dt} = g(y), \quad g(y) = -\alpha \cdot y^2

subject to the initial condition :math:`y(0)=[a, a]`

and:

..  math:: \alpha = \text{Diag}(b^2,  2b^2)

and:

.. math:: \theta=[b, a].


Aligning the Code with the Mathematical Formulation
-----------------------------------------------

Before diving into the code, let's summarize the mathematical expressions that must be implemented by the user:

1. **ODE Residual**:
   The residual :math:` c(y, \theta)` corresponds to the discretized form of the ODE implemented by ``NonLinearDecoupledODE``

2. **State Jacobian of the Residual**:
   The Jacobian of the residual with respect to the states is :math:`\nabla_y c(y, \theta)` implemented by  ``NonLinearDecoupledODE._jacobian``

3. **Param Jacobian of the Residual**:
   The Jacobian of the residual with respect to the states is :math:`\nabla_\theta c(y, \theta)` implemented by  ``NonLinearDecoupledODE._param_jacobian``

4. **Param Jacobian of the Initial condition**:
   The Jacobian of the residual with respect to the states is :math:`\nabla_\theta y(y, \theta)` implemented by  ``NonLinearDecoupledODE._init_param_jacobian``

5. **State Jacobian of the Functional**:
   The Jacobian of the functional with respect to the states is :math:`\nabla_y f(y, \theta)` implemented by  ``TransientSingleStateNonLinearFunctional._qoi_state_jacobian``

6. **State Jacobian of the Functional**:
   The Jacobian of the functional with respect to the parameters is :math:`\nabla_\theta f(y, \theta)` implemented by  ``TransientSingleStateNonLinearFunctional._qoi_state_jacobian``


Step 1: Define the RHS of the ODE
---------------------------------

The mathematical formulation of the ODE residual is implemented in the ``NonLinearDecoupledODE`` class. Specifically:

- The residual :math:` c(y, \theta)` corresponds to the ``__call__`` method.
- The Jacobian :math:` \nabla_y c(y, \theta)` corresponds to the ``jacobian`` method.
"""

# %%
# Step 1: Define the RHS of the ODE
# ---------------------------------
#
# Here we create a ``NonLinearDecoupledODEModel`` class that must inherit from ``TransientNewtonResidual`` and ``ParameterizedNewtonResidualMixin``.
# Below is the implementation of the ODE residual and its Jacobians.

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    BackwardEulerResidual,
    ImplicitTimeIntegrator,
    TransientAdjointFunctional,
    TimeIntegratorNewtonResidual,
)
from pyapprox.util.newton import ParameterizedNewtonResidualMixin
from pyapprox.pde.collocation.adjoint import TransientAdjointModel


class NonLinearDecoupledODE(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    def __init__(self, nstates: int, backend: BackendMixin):
        self._nstates = nstates
        super().__init__(backend)

    def nvars(self) -> int:
        return 2

    def set_time(self, time: float):
        # Set the time at which the RHS of the ODE needs to be evaluated
        self._time = time

    def set_param(self, param: Array):
        # Set the parameters of the ODE.
        if param.shape != (2,):
            raise ValueError("param.shape must = (2,)")
        self._param = param
        self._coef = param[0]
        self._init_cond = param[1]

    def __call__(self, sol: Array) -> Array:
        # Return the values of the RHS of the ODE
        alpha = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        return -alpha * sol**2

    def _jacobian(self, sol: Array) -> Array:
        # Return the jacobian of the solution with respect to the states
        # This is used by implicit time steping schemes
        # when solving the forward and adjoint equations. Explicit
        # time schmes also use it to solve the adjoint equations
        alpha = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        return self._bkd.diag(-2 * alpha * sol)

    def _initial_param_jacobian(self) -> Array:
        # Return the jacobian of the initial condition with respect
        # to the parameters. Only used whe solving the adjoint equations
        nstates = self._nstates
        return self._bkd.stack(
            [
                self._bkd.full((nstates,), 0.0),
                self._bkd.full((nstates,), -1.0),
            ],
            axis=1,
        )

    def _param_jacobian(self, sol: Array) -> Array:
        # Return the jacobian of the solution with respect
        # to the parameters. Only used whe solving the adjoint equations.
        nstates = sol.shape[0]
        coef = 1
        return self._bkd.stack(
            (
                -2
                * coef
                * self._coef
                * self._bkd.arange(
                    1, nstates + 1, dtype=self._bkd.double_type()
                )
                * sol**2,
                self._bkd.full((nstates,), 0.0),
            ),
            axis=1,
        )


# Define ODE parameters
init_time, final_time = 0, 0.25
deltat = 0.1
nstates = 2

# Setup the ODE RHS (residual)
ode_residual = NonLinearDecoupledODE(nstates, bkd)

# Specify the time stepping scheme
time_residual = BackwardEulerResidual(ode_residual)

# Initialize the time integrator
time_int = ImplicitTimeIntegrator(time_residual, init_time, final_time, deltat)

# %%
# Step 2: Define the Functional
# -----------------------------
#
# The functional represents the quantity of interest (QoI) for which we want to compute gradients.
# In this example, the functional is the temporal integral of the first state over all time.


class TransientSingleStateNonLinearFunctional(TransientAdjointFunctional):
    def __init__(self, nstates: int, nparams: int, backend: BackendMixin):
        self._nstates = nstates
        self._nparams = nparams
        super().__init__(backend)

    def nqoi(self) -> int:
        # The number of QoI. Must be 1 when computing gradients with
        # adjoint equations
        return 1

    def nstates(self) -> int:
        # The number of states in the PDE
        return self._nstates

    def nparams(self) -> int:
        # The number of uncertain parameters
        return self._nparams

    def _value(self, sol: Array) -> Array:
        # Return the value of the functional
        return self._bkd.atleast1d(self._bkd.sum(sol[0, :] ** 2 * self._quadw))

    def _qoi_state_jacobian(self, sol: Array) -> Array:
        # Return the quantity of interest with respect to the state
        e1 = self._bkd.zeros((self.nstates(),))
        e1[0] = 1.0
        dqdu = e1[:, None] * 2 * sol * self._quadw
        return dqdu

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        # Return the jacobian of the functional on the parameters
        # This functional does not depend on any variables expliclty
        # so return zero
        return self._bkd.zeros((self.nparams(),))

    def nunique_functional_params(self) -> int:
        # Some functionals depend on parameters that do not enter the
        # state residual. Return the correct number here
        return 0


functional = TransientSingleStateNonLinearFunctional(nstates, 2, bkd)
time_int.set_functional(functional)

# %%
# Step 3: Set Initial Conditions
# ------------------------------
#
# Set the initial conditions for the ODE. Note that the initial condition depends on the parameters.
param = bkd.array(
    [4.0, 3.0]
)  # param[0] is the coefficient b, param[1] is the initial condition
functional.set_param(param)
ode_residual.set_param(param)
init_cond = param[1] * bkd.ones((functional.nstates(),))

# %%
# Step 4: Solve the ODE
# ---------------------
#
# Solve the forward ODE using the model.

fwd_sols, times = time_int.solve(init_cond)

# Evaluate the functional
functional.set_quadrature_sample_weights(
    *time_residual.quadrature_samples_weights(times)
)
qoi = functional(fwd_sols)

# %%
# Step 5: Solve the Adjoint ODE
# -----------------------------
#
# Solve the adjoint ODE using the model.

adj_sols = time_int.solve_adjoint(fwd_sols, times)

# %%
# Step 6: Compute Gradients
# -------------------------
#
# Compute the gradient of the functional with respect to the parameters using adjoints.

jacobian = time_int.gradient_from_adjoint_sols(adj_sols, fwd_sols, times)
print("Gradient (Jacobian):", jacobian)

# %%
# Step 7: Visualization
# ----------------------
#
# Visualize the ODE solution and adjoints using Matplotlib.

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot ODE solution
for state_id in range(nstates):
    axs[0].plot(times, fwd_sols[state_id], label=f"State {state_id}")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Solution")
axs[0].set_title("ODE Solution")
axs[0].legend()
axs[0].grid()

# Plot adjoints
for state_id in range(nstates):
    axs[1].plot(times, adj_sols[state_id], label=f"Adjoint {state_id+1}")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Adjoint")
axs[1].set_title("Adjoint Solution")
axs[1].legend()
axs[1].grid()

plt.show()

# %%
# Step 8: Verify Gradients
# ------------------------
#
# To verify the computed gradient, we compare it against a finite difference approximation.
# This ensures that the adjoint-based gradient computation is correct. The plot below shows the error
# between the adjoint-based gradient and the finite difference gradient as a function of the finite difference step size.


# Wrap the integrator in a model for gradient verification
class NonLinearDecoupledODEModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        nstates: int = 2,
        backend: BackendMixin = bkd,
    ):
        time_residual = self._setup_residual(
            time_residual_cls, nstates, backend
        )
        functional = self._setup_functional(nstates, backend)
        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual,
            functional,
            None,
            backend,
        )

    def jacobian_implemented(self) -> bool:
        return True

    def nvars(self) -> int:
        return self._time_residual.native_residual.nvars()

    def _setup_residual(self, time_residual_cls, nstates, bkd):
        return time_residual_cls(NonLinearDecoupledODE(nstates, bkd))

    def _setup_functional(self, nstates, bkd):
        return TransientSingleStateNonLinearFunctional(nstates, 2, bkd)

    def get_initial_condition(self):
        # Do not use bkd.full as it will mess up torch autograd
        return self._functional._param[1] * self._bkd.ones(
            (self._functional.nstates(),)
        )


# Initialize the model
model = NonLinearDecoupledODEModel(
    init_time, final_time, deltat, BackwardEulerResidual, nstates, bkd
)

# Define the sample point at which to compute the gradient
sample = param[:, None]

# Compute the functional value and gradient using the model
qoi = model(sample)
jacobian = model.jacobian(sample)

# Verify the gradient using finite differences

ax = plt.figure().gca()
fd_eps = bkd.flip(bkd.logspace(-13, 0, 14))  # Finite difference step sizes
errors = model.check_apply_jacobian(sample, fd_eps)  # Compute errors
ax.loglog(fd_eps, errors)
ax.set_xlabel("Finite Difference Step Size")
ax.set_ylabel("Error")
ax.set_title("Gradient Verification Using Finite Differences")
plt.show()

# %%
# The gradient is correct because the plot exhibits a V-cycle.
# This indicates that the adjoint-based gradient computation matches the finite difference approximation.

# %%
# Conclusion
# ----------
#
# In this tutorial, we demonstrated how to solve an ODE and compute gradients using adjoints.
# The provided code serves as an example of implementing adjoint-based gradient computation for ODEs.
# The visualization highlights the solution of the ODE and the adjoint variables, which are used to compute the gradient of the functional.
