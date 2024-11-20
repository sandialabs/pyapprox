r"""
Shallow Water Wave Equation
---------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
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
# bkd = NumpyLinAlgMixin
bkd = TorchLinAlgMixin
np.random.seed(1)
Lx, Ly = 100, 200
bounds = bkd.array([0, Lx, 0, Ly])
transform = ScaleAndTranslationTransform2D(
    [-1, 1, -1, 1], bounds, bkd
)
mesh = ChebyshevCollocationMesh2D([30, 30], transform)
basis = ChebyshevCollocationBasis2D(mesh)

# define time period
init_time, final_time, deltat = 0, 20, 0.5

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
    # wave propagates faster in deeper water. Make bed increase in elevation
    # close to bottom and top boundaries so that it 0.1 at boundaries
    # and 1.1 at midpoint off y domain
    xn = 1 / bounds[1::2, None] * xx
    return -1+(-0.1 + xn[1] * (xn[1] - 1)) * (1-0.9*xn[0])


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


from scipy.special import beta as beta_fn
def beta_surface_fun(shapes, xx):
    # if a0<b0  not value then bump will be closer to left boundary
    a0, b0, a1, b1 = shapes
    # The higher the shape values the higher basis orders need to be
    xn = 1 / bounds[1::2, None] * xx
    const0 = 1./beta_fn(a0, b0)
    const1 = 1./beta_fn(a1, b1)
    return (
        xn[0] ** (a0-1) * (1 - xn[0]) ** (b0-1) *
        xn[1] ** (a1-1) * (1 - xn[1]) ** (b1-1)
    ) * const0 * const1 / 20

def gaussian_surface_fun(loc, xx):
    xn = 1 / bounds[1::2, None] * xx
    width = 0.1
    vals = bkd.exp(-bkd.sum((xn-loc) ** 2, axis=0) / width ** 2) / np.sqrt(2*np.pi)
    # normalize to have largest value of 1
    # Note, interpolated val max will not be one because of interpolation error
    vals = vals/bkd.max(vals)
    return vals

def beta_surface_mixture(xx):
    return (beta_surface_fun([25, 20, 20, 20], xx) + beta_surface_fun(
        [5, 20, 20, 20], xx)
    )

def gaussian_surface_mixture(xx):
    loc0 = bkd.array([0.5, 0.5])[:, None]
    loc1 = bkd.array([0.2, 0.5])[:, None]
    return gaussian_surface_fun(loc0, xx) + gaussian_surface_fun(loc1, xx)

# init_surface_fun = partial(beta_surface_fun, [25, 20, 20, 20])
# init_surface_fun = partial(gaussian_surface_fun,  bkd.array([0.5, 0.5])[:, None])
#init_surface_fun = gaussian_surface_mixture
init_surface_fun = beta_surface_mixture



import matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = truncate_colormap(plt.cm.Blues, minval=0.1)
plot_kwargs = {
    "npts_1d": 101,
    "cmap": cmap,
    "alpha": 1,
    "antialiased": True,#False,
    "linewidth": 0,
    "rstride": 1,
    "cstride": 1,
}
init_surface = ScalarFunctionFromCallable(
    basis, init_surface_fun, basis.nphys_vars() + 1
)

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
import os
solfilename = "shallowwater.npz"
if not os.path.exists(solfilename):
    model._fwd_solve()
    np.savez(solfilename, sols=model._sols, times=model._times)
    sols = model._sols
    times = model._times
else:
    data = np.load(solfilename)
    sols = bkd.asarray(data["sols"])
    times = bkd.asarray(data["times"])

# sol = VectorFunction(basis, bed.ninput_funs(), bed.ninput_funs())
# print(sols.shape)
# sol.set_values(sols[..., -1])
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

gs = GridSpec(10, 4, hspace=1)  # 10 rows, 3 columns
fig = plt.figure(figsize=(3 * 8, 6))
ax0 = fig.add_subplot(gs[:9, 0], projection="3d")
ax1 = fig.add_subplot(gs[:9, 1])
ax2 = fig.add_subplot(gs[:9, 2])
ax3 = fig.add_subplot(gs[:9, 3])
ax4 = fig.add_subplot(gs[-1, :])
axs = [ax0, ax1, ax2, ax3, ax4]
surface_vals = sols[0] + model._bed.get_values()[:, None]
states = [surface_vals, sols[1], sols[2]]
state_bounds = bkd.stack(
    [bkd.asarray([s.min(), s.max()]) for s in states], axis=0
)
# state_bounds should be determined based on interpolated values
# which can be larger/smaller than mesh values which can cause white values in plot

zmin, zmax = surface_vals.min(), surface_vals.max()
#zmin -= 0.02
# zmax += 0.14
zmin -= 1
zmax += 0.7
plot_kwargs["zbounds"] = [zmin, zmax]
state_bounds[0] = bkd.array([zmin, zmax])
levels = [bkd.linspace(*b, 51) for b in state_bounds]


def animate(ii):
    [ax.clear() for ax in axs]
    sol = VectorFunction(basis, bed.ninput_funs(), bed.ninput_funs())
    sol.set_values(sols[..., ii])
    h, uh, vh = sol.get_components()
    # u = uh / h
    # v = vh / h
    # vorticity = v.deriv(0) - u.deriv(1)
    surface = model._bed + h
    surface.plot(axs[0], **plot_kwargs)
    surface.plot(axs[1], levels=levels[0], cmap=cmap)
    axs[0].set_zlim((zmin, zmax))
    axs[0].set_box_aspect((Lx/max(Lx, Ly), Ly/max(Lx, Ly), 0.25))
    # axs[0].view_init(elev=10)
    # ax.set_axis_off()
    axs[0].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axs[0].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axs[0].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    axs[0].xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    axs[0].yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    axs[0].zaxis._axinfo["grid"]['color'] = (1,1,1,0)
    uh.plot(axs[2], levels=levels[1])
    vh.plot(axs[3], levels=levels[2])
    timebar = bkd.zeros(times.shape[0])
    timebar[:ii] = 1.0
    axs[4].imshow(
        timebar[None, :],
        extent=[times[0], times[-1], 0, 1],
        aspect="auto",
    )
    axs[4].set_xlabel("Time")
    axs[4].set_yticks([])


# animate(0)
# plt.show()


import matplotlib.animation as animation

ani = animation.FuncAnimation(
    fig, animate, interval=125, repeat_delay=1000, frames=sols.shape[-1],
)
ani.save("shallowwater.gif", dpi=100)
