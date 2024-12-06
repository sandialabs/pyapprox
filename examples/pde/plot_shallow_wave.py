r"""
Shallow Water Wave Equation
---------------------------
"""
import os
import sys
import numpy as np
from pyapprox.util.linearalgebra.linalgbase import Array

# from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.collocation.adjoint_models import TransientAdjointFunctional
from pyapprox.pde.collocation.parameterized_pdes import ShallowWaterWaveModel
from pyapprox.pde.collocation.timeintegration import (
    BackwardEulerResidual,
    SymplecticMidpointResidual,
    CrankNicholsonResidual,
)
from pyapprox.pde.collocation.functions import (
    # ScalarKLEFunction,
    animate_transient_2d_vector_solution,
    get_water_cmap,
)
from pyapprox.pde.collocation.newton import NewtonSolver

# from pyapprox.util.print_wrapper import *

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TKAgg")

# setup domain
# bkd = NumpyLinAlgMixin
bkd = TorchLinAlgMixin
np.random.seed(1)


# define time period
init_time, final_time, deltat = 0, 20, 0.5

# TODO: WARNING INITIAL CONDITION IS NOT CONSISTENT WITH BOUNDARY CONDITIONS

# setup random bed function
# mean_bed = ConstantScalarFunction(
#     basis, -1.0, ninput_funs=mesh.nphys_vars() + 1
# )
# bed = ScalarKLEFunction(
#     basis,
#     0.1,
#     3,
#     sigma=0.1,
#     mean_field=mean_bed,
#     ninput_funs=mesh.nphys_vars() + 1,
# )

surface_plot_kwargs = {
    "npts_1d": 101,
    "cmap": get_water_cmap(),
    "alpha": 1,
    "antialiased": True,  # False,
    "linewidth": 0,
    "rstride": 1,
    "cstride": 1,
}
# init_surface.plot(init_surface.get_plot_axis(surface=True)[1], **plot_kwargs)
# plt.show()
newton_solver = NewtonSolver(
    verbosity=2,
    maxiters=5,
    atol=1e-6,
    rtol=1e-6,
)
model = ShallowWaterWaveModel(
    init_time,
    final_time,
    deltat,
    BackwardEulerResidual,
    # CrankNicholsonResidual,
    # SymplecticMidpointResidual, # TODO seems to have an error in computation of jacobian
    newton_solver=newton_solver,
    backend=bkd,
)

sample = bkd.array([25, 20, 20, 20, 5, 20, 20, 20])[:, None]

# ax = model._bed.get_plot_axis()[1]
# im = model._bed.plot(ax, levels=50)
# plt.colorbar(im, ax=ax)
# plt.show()

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

surface_plot_kwargs = {
    "cmap": get_water_cmap(),
    "alpha": 1,
    "antialiased": True,  # False,
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
    [0, 1, 2],
    [0],
    51,
    contour_plot_kwargs,
    surface_plot_kwargs,
)
import os

print(os.getcwd())
ani.save("shallowwater.gif", dpi=100)
