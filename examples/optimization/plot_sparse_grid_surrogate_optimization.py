"""
Tutorial: Sparse Grid Optimization with PyApprox
================================================

This tutorial demonstrates how to use PyApprox to perform constrained optimization using sparse grids. We will set up a sparse grid for both the objective and constraint functions, compute derivatives and Hessian-vector products, and solve the optimization problem using a constrained optimizer.

"""

# %%
# Setup
# -----
# First, import the necessary modules:

from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.benchmarks.algebraic import (
    RosenbrockConstrainedOptimizationBenchmark,
)
from pyapprox.surrogates.univariate.base import ClenshawCurtisQuadratureRule
from pyapprox.surrogates.affine.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.univariate.lagrange import UnivariateLagrangeBasis
from pyapprox.surrogates.affine.multiindex import DoublePlusOneIndexGrowthRule
from pyapprox.surrogates.sparsegrids.combination import (
    IsotropicCombinationSparseGrid,
)
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.optimization.minimize import ConstraintFromModel

benchmark = RosenbrockConstrainedOptimizationBenchmark(backend=bkd)

# %%
# Setting Up Sparse Grids
# ------------------------
# We define a helper function to set up a sparse grid for a given model. This function uses a Clenshaw-Curtis quadrature rule and a tensor product interpolating basis.


def setup_sparse_grid(model):
    nvars, level = 2, 3
    quad_rule = ClenshawCurtisQuadratureRule(
        store=True, backend=bkd, bounds=[-1, 1]
    )
    bases_1d = [
        UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(nvars)
    ]
    basis = TensorProductInterpolatingBasis(bases_1d)
    sg = IsotropicCombinationSparseGrid(
        model.nqoi(),
        nvars,
        level,
        DoublePlusOneIndexGrowthRule(),
        basis,
        backend=bkd,
    )
    train_samples = sg.train_samples()
    train_values = model(train_samples)
    sg.fit(train_samples, train_values)
    return sg


# %%
# Set up sparse grids for the objective and constraint functions:

obj_sg = setup_sparse_grid(benchmark.objective())
cons_sg = setup_sparse_grid(benchmark.constraints()[0])

# %%
# Computing Derivatives and Hessian-Vector Products
# -------------------------------------------------
# We can compute the Jacobian of the objective function:

print(obj_sg.jacobian(benchmark.init_iterate()))

# %%
# We can compute the Hessian-vector product of the objective function. The vector `vec` is typically passed in by the optimizer:

vec = bkd.array([1, 2])[:, None]
print(obj_sg.apply_hessian(benchmark.init_iterate(), vec))

# %%
# Similarly, we can compute the Jacobian of the constraint function:

print(cons_sg.jacobian(benchmark.init_iterate()))

# %%
# We can compute the weighted Hessian-vector product of the constraint function. The vector `vec` and weights are typically passed in by the optimizer:

vec = bkd.array([1, 2])[:, None]
weights = bkd.array([0.5, 0.5])[:, None]
print(cons_sg.apply_weighted_hessian(benchmark.init_iterate(), vec, weights))

# %%
# Setting Up Constraints and Optimizer
# -------------------------------------
# Define the constraint using the sparse grid model and the bounds of the original constraint:

constraint = ConstraintFromModel(
    cons_sg, benchmark.constraints()[0].get_bounds()
)

# %%
# Set up the optimizer using the sparse grid model for the objective function, the constraint, and the bounds of the design variables:

optimizer = ScipyConstrainedOptimizer(
    obj_sg,
    constraints=[constraint],
    bounds=benchmark.design_variable().bounds(),
    opts={"gtol": 1e-16},
)

# %%
# Solving the Optimization Problem
# --------------------------------
# Solve the optimization problem starting from the initial iterate:

result = optimizer.minimize(benchmark.init_iterate())
print(result)

# %%
# Summary
# -------
# This tutorial demonstrated how to use PyApprox to perform constrained optimization using sparse grids. We set up sparse grids for the objective and constraint functions, computed derivatives and Hessian-vector products, and solved the optimization problem using a constrained optimizer.
