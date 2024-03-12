r"""
Design Under Uncertainty
========================

We will ue the Cantilever Beam benchmark to illustrate how to design under
uncertainty.

We will minimize the objective function

.. math:: wt

Subject to a stress constraint

.. math:: 6L\left(\frac{X}{tw^2}+\frac{Y}{t^2w}\right) < R

and a displacement constraint

.. math:: \frac{4L^3}{Ewt}\sqrt{\left(\frac{Y}{t}\right)^2+\left(\frac{X}{w}\right)^2} < D

The conceptual model is depicted in the figure below

.. figure:: ../figures/cantilever-beam.png
   :align: center

   Conceptual model of the cantilever-beam

The marginal distribution of the independent random variables are

.. table:: Uncertainties
   :align: center

   =============== ========= =======================
   Uncertainty     Symbol    Prior
   =============== ========= =======================
   Yield stress    :math:`R` :math:`N(40000,2000)`
   Young's modulus :math:`E` :math:`N(2.9e7,1.45e6)`
   Horizontal load :math:`X` :math:`N(500,100)`
   Vertical Load   :math:`Y` :math:`N(1000,100)`
   =============== ========= =======================


.. math:: \mean{f(\rv, \xi)} \approx \mu(\xi)=N^{-1}\sum_{n=1}^N f(\rv^{(n)}, \xi)

.. math:: \nabla_\xi \mu(\xi) =  N^{-1}\sum_{n=1}^N \nabla_\xi f(\rv^{(n)}, \xi)

.. math:: \var{f(\rv, \xi)} \approx \Sigma(\xi) = (2N(N-1))^{-1}\sum_{m=1}^N\sum_{n=1}^N (f(\rv^{(m)}, \xi)-f(\rv^{(n)}, \xi))^2

or

.. math:: \var{f(\rv, \xi)} \approx \Sigma(\xi) = (N-1)^{-1}\sum_{n=1}^N (f(\rv^{(m)}, \xi)-\mu(\xi))^2

Chain rule for :math:`g(h(x))`

.. math:: \dydx{g}{x}(x) = \dydx{g}{h}(h(x))\dydx{h}{x}(x)

So setting :math:`g(y) = y^2` and :math:`h(\xi)=(f(\rv^{(m)}, \xi)-f(\rv^{(n)}, \xi))` then

.. math:: \dydx{g}{h}(h(\xi))\dydx{h}{\xi}(\xi) = 2(f(\rv^{(m)}, \xi)-f(\rv^{(n)}, \xi))(\nabla_\xi f(\rv^{(m)}, \xi)-\nabla_\xi f(\rv^{(n)}, \xi))

Thus

.. math:: \nabla_\xi  \Sigma(\xi) = (2N(N-1))^{-1}\sum_{m=1}^N\sum_{n=1}^N 2(f(\rv^{(m)}, \xi)-f(\rv^{(n)}, \xi))(\nabla_\xi f(\rv^{(m)}, \xi)-\nabla_\xi f(\rv^{(n)}, \xi))

or setting  :math:`g(y) = y^2` and :math:`h(\xi)=(f(\rv^{(m)}, \xi)-\mu(\xi))` then

.. math:: \dydx{g}{h}(h(\xi))\dydx{h}{\xi}(\xi) = 2(f(\rv^{(m)}, \xi)-\mu(\xi))(\nabla_\xi f(\rv^{(m)}, \xi)-\nabla_\xi\mu(\xi))


Chain rule for Hessian of a scalar function :math:`g(h(x))`

.. math:: \nabla_x^2 \dydx{g}{x} = \dydx{g}{h}(h(x))\nabla_x^2 f(x) + \frac{\partial^2 g}{\partial h^2}(h(x))(\nabla_x h(x))(\nabla_x h(x))^\top

where :math:`\nabla_x h(x)\in\reals^{1\times D}`

First lets perform a deterministic optimization at the nominal values
of the random variables
"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.interface.model import ActiveSetVariableModel
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer, Constraint)
np.random.seed(1)

benchmark = setup_benchmark('cantilever_beam')

ndesign_vars = benchmark.design_variable.num_vars()
#nominal_values = benchmark.variable.get_statistics('ppf', q=0.9)
nominal_values = benchmark.variable.get_statistics('mean')
objective_model = ActiveSetVariableModel(
    benchmark.funs[0],
    benchmark.variable.num_vars()+ndesign_vars,
    nominal_values, benchmark.design_var_indices)

constraint_model = ActiveSetVariableModel(
    benchmark.funs[1],
    benchmark.variable.num_vars()+ndesign_vars,
    nominal_values, benchmark.design_var_indices)
constraint_bounds = np.hstack(
    [np.zeros((2, 1)), np.full((2, 1), np.inf)])
constraint = Constraint(constraint_model, constraint_bounds)

optimizer = ScipyConstrainedOptimizer(
    objective_model, constraints=[constraint],
    bounds=benchmark.design_variable.bounds)
result = optimizer.minimize(np.array([3, 3])[:, None])
print("optimal design vars", result.x)
print("optimal", result.fun)

#%%
# Plot objective and constraints
from pyapprox.util.visualization import get_meshgrid_function_data
import matplotlib.cm as cm
X, Y, Z_o = get_meshgrid_function_data(
    objective_model,
    np.hstack([benchmark.design_variable.bounds.lb[:, None],
               benchmark.design_variable.bounds.ub[:, None]]).flatten(), 101)
im = plt.contourf(X, Y, Z_o, levels=40, cmap="coolwarm")
plt.colorbar(im)
for ii in range(1):
    X, Y, Z_c = get_meshgrid_function_data(
        constraint_model,
        np.hstack([benchmark.design_variable.bounds.lb[:, None],
                   benchmark.design_variable.bounds.ub[:, None]]).flatten(),
        301, qoi=ii)
    II = np.where(Z_c < 0)
    JJ = np.where(Z_c >= 0)
    Z_c[II] = 1
    # set region that satisfies constraints to np.nan so contourf
    # does not plot anything in that area
    Z_c[JJ] = np.nan
    im = plt.contourf(X, Y, Z_c, levels=40, cmap="gray")
plt.plot(*result.x, 'og')
plt.show()


#robust design
#min f subject to variance<tol

#reliability design
#min f subject to prob failure<tol
