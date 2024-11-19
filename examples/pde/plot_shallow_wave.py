r"""
Shallow Water Wave Equation
---------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from pyapprox.util.linearalgebra.linalgbase import Array

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.collocation.adjoint_models import TransientAdjointFunctional
from pyapprox.pde.collocation.parameterized_pdes import ShallowWaterWaveModel
from pyapprox.pde.collocation.timeintegration import (
    BackwardEulerResidual,
    SymplecticMidpointResidual,
    CrankNicholsonResidual,
)
from pyapprox.pde.collocation.functions import (
    ScalarKLEFunction,
    ZeroScalarFunction,
    ConstantScalarFunction,
    VectorFunction,
)
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform2D,
)
from pyapprox.pde.collocation.mesh import ChebyshevCollocationMesh2D
from pyapprox.pde.collocation.basis import ChebyshevCollocationBasis2D
from pyapprox.pde.collocation.newton import NewtonSolver
# from pyapprox.util.print_wrapper import *
import sys

if sys.platform == "darwin":
    import matplotlib

    matplotlib.use("TKAgg")

# setup domain
bkd = NumpyLinAlgMixin
# bkd = TorchLinAlgMixin
np.random.seed(1)
transform = ScaleAndTranslationTransform2D(
    [-1, 1, -1, 1], [0, 100, 0, 100], bkd
)
mesh = ChebyshevCollocationMesh2D([30, 30], transform)
basis = ChebyshevCollocationBasis2D(mesh)

# define time period
init_time, final_time, deltat = 0, 10, 0.5

# TODO: WARNING INITIAL CONDITION IS NOT CONSISTENT WITH BOUNDARY CONDITIONS

# setup random bed function
mean_bed = ConstantScalarFunction(
    basis, -1.0, ninput_funs=mesh.nphys_vars() + 1
)
bed = ScalarKLEFunction(
    basis,
    0.1,
    3,
    sigma=0.1,
    mean_field=mean_bed,
    ninput_funs=mesh.nphys_vars() + 1,
)


def bed_fun(xx):
    xn = xx / 100
    return -1 - bkd.prod(xn * (xn - 1) ** 2, axis=0)


from pyapprox.pde.collocation.functions import ScalarFunctionFromCallable

bed = ScalarFunctionFromCallable(
    basis, bed_fun, ninput_funs=mesh.nphys_vars() + 1
)


class TransientTODOFunctional(TransientAdjointFunctional):
    def __init__(self, nstates, backend):
        self._nstates = nstates
        self._bkd = backend

    def nstates(self):
        return self._nstates

    def nparams(self):
        return 2

    def _value(self, sol: Array) -> Array:
        # RETURNS SUM OF FIRST STATE over all time change to something more realistic
        return self._bkd.atleast1d(self._bkd.sum(sol[0, :] * self._quadw))

    def _qoi_sol_jacobian(self):
        raise NotImplementedError


# setup model
functional = TransientTODOFunctional(basis.mesh.nmesh_pts(), bkd)
init_surface = ZeroScalarFunction(basis, basis.nphys_vars() + 1)


from scipy.special import beta as beta_fn
def init_surface_fun(xx):
    a0, b0 = 5, 20
    #a1, b1 = 20, 20
    # The higher these values the higher basis orders need to be
    #a0, b0 = 5, 10
    a1, b1 = 10, 10
    xn = xx/100
    const0 = 1./beta_fn(a0, b0)
    const1 = 1./beta_fn(a1, b1)
    return (
        xn[0] ** (a0-1) * (1 - xn[0]) ** (b0-1) *
        xn[1] ** (a1-1) * (1 - xn[1]) ** (b1-1)
    ) * const0 * const1 / 20


plot_kwargs = {
    "npts_1d": 51,
    "edgecolor": "royalblue",
    "cmap":  "Blues_r",
    "alpha": 0.75,
    "antialiased": True,
    "linewidth": 0,
}
init_surface = ScalarFunctionFromCallable(
    basis, init_surface_fun, basis.nphys_vars() + 1
)
# init_surface.plot(init_surface.get_plot_axis(surface=True)[1], **plot_kwargs)
# plt.show()
newton_solver = NewtonSolver(
    verbosity=2,
    maxiters=5,
    atol=1e-8,
    rtol=1e-8,
)
model = ShallowWaterWaveModel(
    init_time,
    final_time,
    deltat,
    BackwardEulerResidual,
    # CrankNicholsonResidual,
    # SymplecticMidpointResidual, # TODO seems to have an error in computation of jacobian
    functional,
    bed,
    init_surface,
    newton_solver=newton_solver,
)

# run model
# param = bkd.asarray(np.random.normal(0, 1, (model.nvars(),)))
# model.set_param(param)

# ax = model._bed.get_plot_axis()[1]
# im = model._bed.plot(ax, levels=50)
# plt.colorbar(im, ax=ax)
# plt.show()
model._fwd_solve()
assert False
# sol = VectorFunction(basis, bed.ninput_funs(), bed.ninput_funs())
# print(model._sols.shape)
# sol.set_values(model._sols[..., -1])
# axs = plt.subplots(1, 3, figsize=(3*8, 6))[1]
# depth = sol.get_components()[0]
# print(model._bed.get_jacobian())
# print(depth.get_jacobian())
# surface = model._bed + depth
# im = model._bed.plot(axs[0], levels=50)
# plt.colorbar(im, ax=axs[0])
# im = depth.plot(axs[1], levels=50)
# plt.colorbar(im, ax=axs[1])
# im = surface.plot(axs[2], levels=50)
# plt.colorbar(im, ax=axs[2])
# plt.show()


# fig, axs = plt.subplots(1, 4, figsize=(3*8, 6))
from matplotlib.gridspec import GridSpec

gs = GridSpec(10, 3, hspace=1)  # 10 rows, 3 columns
fig = plt.figure(figsize=(3 * 8, 6))
ax0 = fig.add_subplot(gs[:9, 0], projection="3d")
ax1 = fig.add_subplot(gs[:9, 1])
ax2 = fig.add_subplot(gs[:9, 2])
ax3 = fig.add_subplot(gs[-1, :])
axs = [ax0, ax1, ax2, ax3]
surface_vals = model._sols[0] + model._bed.get_values()[:, None]
states = [surface_vals, model._sols[1], model._sols[2]]
state_bounds = bkd.stack(
    [bkd.asarray([s.min(), s.max()]) for s in states], axis=0
)
# state_bounds should be determined based on interpolated values
# which can be larger/smaller than mesh values which can cause white values in plot
levels = [bkd.linspace(*b, 51) for b in state_bounds]
print(levels)

zmin, zmax = surface_vals.min(), surface_vals.max()
zmin -= 0.02
zmax += 0.07
plot_kwargs["zbounds"] = [zmin, zmax]


def animate(ii):
    [ax.clear() for ax in axs]
    sol = VectorFunction(basis, bed.ninput_funs(), bed.ninput_funs())
    sol.set_values(model._sols[..., ii])
    h, uh, vh = sol.get_components()
    u = uh / h
    v = vh / h
    vorticity = v.deriv(0) - u.deriv(1)
    surface = model._bed + h
    #im = surface.plot(axs[0], levels=levels[0])
    im = surface.plot(axs[0], **plot_kwargs)
    axs[0].set_zlim((zmin, zmax))
    axs[0].set_box_aspect((1, 1, 0.25))
    im = uh.plot(axs[1], levels=levels[1])
    # im = vh.plot(axs[2], levels=levels[2])
    im = vorticity.plot(axs[2])
    #axs[2].quiver(*basis.mesh.mesh_pts(), u.get_values(), v.get_values())
    timebar = bkd.zeros(model._times.shape[0])
    timebar[:ii] = 1.0
    axs[3].imshow(
        timebar[None, :],
        extent=[model._times[0], model._times[-1], 0, 1],
        aspect="auto",
    )
    axs[3].set_xlabel("Time")
    axs[3].set_yticks([])


import matplotlib.animation as animation

ani = animation.FuncAnimation(
    fig, animate, interval=500, frames=model._sols.shape[-1], repeat_delay=1000
)
ani.save("shallowwater.gif", dpi=100)
