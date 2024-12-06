r"""
Shallow Water Wave Equation
---------------------------
"""
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
    ScalarKLEFunction,
    ZeroScalarFunction,
    ConstantScalarFunction,
    VectorFunction,
    animate_transient_2d_vector_solution,
    get_water_cmap,
)
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform2D,
)
from pyapprox.pde.collocation.mesh import ChebyshevCollocationMesh2D
from pyapprox.pde.collocation.basis import ChebyshevCollocationBasis2D
from pyapprox.pde.collocation.newton import NewtonSolver

# from pyapprox.util.print_wrapper import *

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TKAgg")

# setup domain
# bkd = NumpyLinAlgMixin
bkd = TorchLinAlgMixin
np.random.seed(1)
Lx, Ly = 100, 200
bounds = bkd.array([0, Lx, 0, Ly])
transform = ScaleAndTranslationTransform2D([-1, 1, -1, 1], bounds, bkd)
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
    return -1 + (-0.1 + xn[1] * (xn[1] - 1)) * (1 - 0.9 * xn[0])


from pyapprox.pde.collocation.functions import ScalarFunctionFromCallable

# bed = ScalarFunctionFromCallable(
#     basis, bed_fun, ninput_funs=mesh.nphys_vars() + 1
# )


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

    def nunique_functional_params(self):
        return 0


# setup model
functional = TransientTODOFunctional(basis.mesh.nmesh_pts(), bkd)


from scipy.special import beta as beta_fn


def beta_surface_fun(shapes, xx):
    # if a0<b0  not value then bump will be closer to left boundary
    a0, b0, a1, b1 = shapes
    # The higher the shape values the higher basis orders need to be
    xn = 1 / bounds[1::2, None] * xx
    const0 = 1.0 / beta_fn(a0, b0)
    const1 = 1.0 / beta_fn(a1, b1)
    return (
        (
            xn[0] ** (a0 - 1)
            * (1 - xn[0]) ** (b0 - 1)
            * xn[1] ** (a1 - 1)
            * (1 - xn[1]) ** (b1 - 1)
        )
        * const0
        * const1
        / 20
    )


def gaussian_surface_fun(loc, xx):
    xn = 1 / bounds[1::2, None] * xx
    width = 0.1
    vals = bkd.exp(-bkd.sum((xn - loc) ** 2, axis=0) / width**2) / np.sqrt(
        2 * np.pi
    )
    # normalize to have largest value of 1
    # Note, interpolated val max will not be one because of interpolation error
    vals = vals / bkd.max(vals)
    return vals


def beta_surface_mixture(xx):
    return beta_surface_fun([25, 20, 20, 20], xx) + beta_surface_fun(
        [5, 20, 20, 20], xx
    )


def gaussian_surface_mixture(xx):
    loc0 = bkd.array([0.5, 0.5])[:, None]
    loc1 = bkd.array([0.2, 0.5])[:, None]
    return gaussian_surface_fun(loc0, xx) + gaussian_surface_fun(loc1, xx)


# init_surface_fun = partial(beta_surface_fun, [25, 20, 20, 20])
# init_surface_fun = partial(gaussian_surface_fun,  bkd.array([0.5, 0.5])[:, None])
# init_surface_fun = gaussian_surface_mixture
init_surface_fun = beta_surface_mixture


surface_plot_kwargs = {
    "npts_1d": 101,
    "cmap": get_water_cmap(),
    "alpha": 1,
    "antialiased": True,  # False,
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
    bed,
    init_surface,
    functional=functional,
    newton_solver=newton_solver,
)

sample = bkd.asarray(np.random.normal(0, 1, (model.nvars(), 1)))

# ax = model._bed.get_plot_axis()[1]
# im = model._bed.plot(ax, levels=50)
# plt.colorbar(im, ax=ax)
# plt.show()
import os

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
    basis,
    sols,
    times,
    model.physics().ncomponents(),
    [1, 2],
    [0, 1],
    51,
    contour_plot_kwargs,
    surface_plot_kwargs,
)
import os

print(os.getcwd())
ani.save("shallowwater.gif", dpi=100)
