"""
ADAM Optimization Tutorial
==========================
In this tutorial, we will compare the ADAM optimization algorithm with simple stochastic gradient descent variant used for training deep learning models.


Step 1: Define the Objective Function
-------------------------------------
The objective function is the function that we want to minimize using the ADAM optimizer. Let's define a simple quadratic function:
"""

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.optimization.minimize import (
    StochasticGradientDescentOptimizer,
    ADAMOptimizer,
)


def objective(x):
    return np.sum(x**3, axis=0)[:, None]


def jacobian(x):
    return 3 * (x.T) ** 2


objective = ModelFromSingleSampleCallable(1, 3, objective, jacobian)

# %%
# Step 2: Create the ADAM Optimizer
# ---------------------------------
# Next, we need to create an instance of the ADAM optimizer.
# We will set the options below:

adam_optimizer = ADAMOptimizer()
adam_optimizer.set_objective_function(objective)

# %%
# Step 3: Creating the SGD Optimizer
# ----------------------------------
# We also need to create an instance of the SGD optimizer again we will set the options below:
sgd_optimizer = StochasticGradientDescentOptimizer()
sgd_optimizer.set_objective_function(objective)


# %%
# Step 4: Plot the Results
# ------------------------
# We can now plot the convergence of the ADAM and SGD optimizers. We must set
# store to true to record the objective values at each iteration
iterate = np.array([1.0, 1.0, 0.5])[:, None]
adam_optimizer.set_options(store=True, learning_rate=1e-2, maxiters=100)
adam_result = adam_optimizer.minimize(iterate)
sgd_optimizer.set_options(store=True, learning_rate=1e-2, maxiters=100)
sgd_result = sgd_optimizer.minimize(iterate)

plt.loglog(adam_optimizer.get_history(), label="ADAM")
plt.loglog(sgd_optimizer.get_history(), label="SGD")
plt.xlabel("Number of iterations")
plt.ylabel("Objective value")
plt.title("Convergence of ADAM and SGD Optimizers")
_ = plt.legend()

# %%
# It is clear that ADAM converges more rapidly than SGD.
