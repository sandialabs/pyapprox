import unittest
import torch
import numpy as np
from functools import partial

from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution,
    setup_steady_stokes_manufactured_solution,
    setup_shallow_ice_manufactured_solution,
    setup_helmholtz_manufactured_solution,
    setup_shallow_water_wave_equations_manufactured_solution,
    setup_shallow_shelf_manufactured_solution,
    setup_first_order_stokes_ice_manufactured_solution,
    setup_two_species_advection_diffusion_reaction_manufactured_solution,
    setup_linear_elasticity_manufactured_solution
)
from pyapprox.pde.autopde.mesh import (
    CartesianProductCollocationMesh,
    TransformedCollocationMesh, InteriorCartesianProductCollocationMesh,
    TransformedInteriorCollocationMesh, VectorMesh,
    subdomain_integral_functional
    )
from pyapprox.pde.autopde.solvers import (
    Function, TransientFunction, SteadyStatePDE, TransientPDE,
    SteadyStateAdjointPDE, TransientAdjointPDE
)
from pyapprox.pde.autopde.physics import (
    AdvectionDiffusionReaction, IncompressibleNavierStokes,
    LinearIncompressibleStokes, ShallowIce, EulerBernoulliBeam,
    Helmholtz, ShallowWaterWave, ShallowShelfVelocities,
    ShallowShelf, FirstOrderStokesIce, MultiSpeciesAdvectionDiffusionReaction,
    LinearElasticity
)
from pyapprox.pde.autopde.mesh_transforms import (
    ScaleAndTranslationTransform, PolarTransform,
    EllipticalTransform, CompositionTransform, SympyTransform
)
from pyapprox.util.utilities import approx_jacobian, check_gradients


# Functions and testing only for wrapping Sympy generated manufactured
# solutions
def _normal_flux(flux_funs, normal_fun, xx):
    normal_vals = torch.as_tensor(normal_fun(xx), dtype=torch.double)
    flux_vals = torch.as_tensor(flux_funs(xx), dtype=torch.double)
    vals = torch.sum(normal_vals*flux_vals, dim=1)[:, None]
    return vals


def _robin_bndry_fun(sol_fun, flux_funs, normal_fun, alpha, xx, time=None):
    if time is not None:
        if hasattr(sol_fun, "set_time"):
            sol_fun.set_time(time)
        if hasattr(flux_funs, "set_time"):
            flux_funs.set_time(time)
    vals = alpha*sol_fun(xx) + _normal_flux(flux_funs, normal_fun, xx)
    return vals


def _normal_flux_old(flux_funs, active_var, sign, xx):
    vals = sign*flux_funs(xx)[:, active_var:active_var+1]
    return vals


def _robin_bndry_fun_old(sol_fun, flux_funs, active_var, sign, alpha, xx,
                         time=None):
    if time is not None:
        if hasattr(sol_fun, "set_time"):
            sol_fun.set_time(time)
        if hasattr(flux_funs, "set_time"):
            flux_funs.set_time(time)
    vals = alpha*sol_fun(xx) + _normal_flux_old(
        flux_funs, active_var, sign, xx)
    return vals


def _canonical_normal(bndry_index, samples):
    normal_vals = np.zeros((samples.shape[1], samples.shape[0]))
    active_var = int(bndry_index >= 2)
    normal_vals[:, active_var] = (-1)**((bndry_index+1) % 2)
    return normal_vals


def _get_boundary_funs(nphys_vars, bndry_types, sol_fun, flux_funs,
                       bndry_normals=None):
    bndry_conds = []
    for dd in range(2*nphys_vars):
        if bndry_types[dd] == "D":
            import copy
            bndry_conds.append([copy.deepcopy(sol_fun), "D"])
        elif bndry_types[dd] == "P":
            bndry_conds.append([None, "P"])
        else:
            if bndry_types[dd] == "R":
                # an arbitray non-zero value just chosen to test use of
                # Robin BCs
                alpha = 1
            else:
                # Zero to reduce Robin BC to Neumann
                alpha = 0
            if bndry_normals is None:
                normal_fun = partial(_canonical_normal, dd)
            else:
                normal_fun = bndry_normals[dd]
            bndry_fun = partial(
                 _robin_bndry_fun, sol_fun, flux_funs, normal_fun, alpha)
            if hasattr(sol_fun, "set_time") or hasattr(flux_funs, "set_time"):
                bndry_fun = TransientFunction(bndry_fun)
            bndry_conds.append([bndry_fun, "R", alpha])
        if bndry_conds[-1][0] is not None:
            bndry_conds[-1][0]._name = f"bndry_{dd}"
    return bndry_conds


def _vel_component_fun(vel_fun, ii, x, time=None):
    if time is not None:
        if hasattr(vel_fun, "set_time"):
            vel_fun.set_time(time)
    vals = vel_fun(x)
    return vals[:, ii:ii+1]


def _sww_momentum_component_fun(vel_fun, depth_fun, ii, x, time=None):
    if time is not None:
        if hasattr(vel_fun, "set_time"):
            vel_fun.set_time(time)
        if hasattr(depth_fun, "set_time"):
            depth_fun.set_time(time)
    vals = depth_fun(x)*vel_fun(x)[:, ii:ii+1]
    return vals


def _adf_bndry_fun(bndry_fun, diff_fun, xx):
    return bndry_fun(xx)/diff_fun(xx)


class TestAutoPDE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)

    def _check_mesh_integrate(self, domain_bounds, orders, fun,
                              exact_integral):
        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders)
        integral = mesh.integrate(fun(mesh.mesh_pts))
        print(integral, exact_integral)
        assert np.allclose(integral, exact_integral)

    def test_mesh_integrate(self):
        test_cases = [
            [[0, 1], [20], lambda xx: (xx[0, :]**2)[:, None],
             1/3],
            [[0, 1, 0, 1], [20, 20], lambda xx: (xx**2).sum(axis=0)[:, None],
             2/3]]
        for test_case in test_cases:
            self._check_mesh_integrate(*test_case)

    def _check_subdomain_integral_functional(
            self, domain_bounds, subdomain_bounds, orders, fun,
            exact_integral):
        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders)
        integral = subdomain_integral_functional(
            np.asarray(subdomain_bounds), mesh, fun(mesh.mesh_pts), None)
        # print(integral, exact_integral)
        assert np.allclose(integral, exact_integral)

    def test_subdomain_integral_functional(self):
        test_cases = [
            [[0, 1], [0.5, 1], [20], lambda xx: (xx[0, :]**2)[:, None],
             7/24],
            [[0, 1, 0, 1], [0.5, 1, 0, 0.25], [20, 20],
             lambda xx: (xx**2).sum(axis=0)[:, None], 29/384]]
        for test_case in test_cases:
            self._check_subdomain_integral_functional(*test_case)

    def _check_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string, diff_string, vel_strings,
            react_funs, bndry_types, basis_types, transform=None,
            nl_diff_funs=[None, None]):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], False,
                nl_diff_funs[0]))

        diff_fun = Function(diff_fun)
        vel_fun = Function(vel_fun)
        forc_fun = Function(forc_fun)
        sol_fun = Function(sol_fun)

        nphys_vars = len(orders)

        if transform is None:
            bndry_conds = _get_boundary_funs(
                nphys_vars, bndry_types, sol_fun, flux_funs)
            mesh = CartesianProductCollocationMesh(
                domain_bounds, orders, basis_types)
        else:
            bndry_conds = _get_boundary_funs(
                nphys_vars, bndry_types, sol_fun, flux_funs,
                [partial(transform.normal, ii) for ii in range(2*nphys_vars)])
            mesh = TransformedCollocationMesh(orders, transform)

        # import matplotlib.pyplot as plt
        # print(transform)
        # for ii in range(4):
        #     idx = mesh._bndry_indices[ii]
        #     eps = 0.05
        #     normals = mesh._bndrys[ii].normals(mesh.mesh_pts[:, idx])*eps
        #     for kk in range(normals.shape[0]):
        #         plt.arrow(*mesh.mesh_pts[:, idx][:, kk], *normals[kk])
        # plt.show()

        # bndry_cond returned by _get_boundary_funs is du/dx=fun
        # apply boundary condition kdu/dx.n=fun
        # for bndry_cond in bndry_conds:
        #     bndry_cond[0] = partial(
        #         _adf_bndry_fun, bndry_cond[0], diff_fun)

        if ("_u_" in diff_string):
            assert nl_diff_funs[1] is not None
        solver = SteadyStatePDE(AdvectionDiffusionReaction(
            mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
            react_funs[1], nl_diff_fun=nl_diff_funs[0],
            nl_diff_jac=nl_diff_funs[1]))

        # import matplotlib.pyplot as plt
        # plt.plot(mesh.mesh_pts[0], mesh.mesh_pts[1], 'o')
        # can_pts = mesh._map_samples_to_canonical_domain(mesh.mesh_pts)
        # # print(mesh._canonical_mesh_pts, can_pts)
        # # assert np.allclose(mesh._canonical_mesh_pts, can_pts)
        # plt.plot(can_pts[0], can_pts[1], 'X')
        # plt.show()

        exact_sol = sol_fun(mesh.mesh_pts)
        if nl_diff_funs[0] is not None:
            solver.physics._auto_jac = True
            np.set_printoptions(linewidth=1000)
            j_auto = torch.autograd.functional.jacobian(
                lambda s: solver.physics._raw_residual(s)[0],
                exact_sol[:, 0].clone().requires_grad_(True), strict=True).numpy()
            solver.physics._auto_jac = False
            jac = solver.physics._raw_residual(exact_sol[:, 0])[1]
            print(jac.numpy())
            # def fun(s):
            #     return solver.physics._raw_residual(torch.as_tensor(s))[0].numpy()
            # j_fd = approx_jacobian(fun, exact_sol[:, 0].numpy())
            # print(j_fd)
            print(j_auto)
            assert np.allclose(j_auto, jac.numpy())

        print(np.abs(solver.physics._raw_residual(
            sol_fun(mesh.mesh_pts)[:, 0])[0]).max())
        # print(np.abs(solver.physics._residual(sol_fun(mesh.mesh_pts)[:, 0])[0]))
        assert np.allclose(
            solver.physics._raw_residual(sol_fun(mesh.mesh_pts)[:, 0])[0], 0)
        assert np.allclose(
            solver.physics._residual(sol_fun(mesh.mesh_pts)[:, 0])[0], 0)
        solver.physics._clear_data()
        sol = solver.solve(
            init_guess=exact_sol[:, 0]+np.random.normal(
                0, 1e-2, exact_sol.shape[0]), tol=1e-8, rtol=1e-12,
            verbosity=2)[:, None]
        assert np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol) < 1e-9

        # import matplotlib.pyplot as plt
        # im = mesh.plot(sol, nplot_pts_1d=40)
        # plt.colorbar(im)
        # plt.show()

        # print(bndry_conds)

        def functional(sol, params):
            # return sol[1:-1].sum()
            return sol[sol.shape[0]//2]

        param_vals = diff_fun(mesh.mesh_pts)[0:1, 0]
        residual = AdvectionDiffusionReaction(
            mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
            react_funs[1])
        fwd_solver = SteadyStatePDE(residual)
        adj_solver = SteadyStateAdjointPDE(fwd_solver, functional)

        def set_param_values(residual, param_vals):
            # assert param_vals.ndim == 1
            mesh_vals = torch.tile(param_vals, (mesh.mesh_pts.shape[1], ))
            residual._diff_fun = partial(
                residual.mesh.interpolate, mesh_vals)
            # grad = adj_solver.compute_gradient(
            #     set_param_values, param_vals, tol=1e-8)

        def fun(params):
            set_param_values(
                fwd_solver.physics,
                torch.as_tensor(params[:, 0], dtype=torch.double))
            # newton tol must be smaller than finite difference step size
            fd_sol = fwd_solver.solve(tol=1e-8, verbosity=0, rtol=1e-12)
            qoi = np.asarray([functional(fd_sol, params[:, 0])])
            return qoi

        # pp = torch.clone(param_vals).requires_grad_(True)
        # set_param_values(fwd_solver.physics, pp)
        # sol = fwd_solver.solve()
        # qoi = functional(sol, pp)
        # qoi.backward()
        # grad_pure_ad = pp.grad
        # print(grad_pure_ad.numpy()[0], 'pg')
        # print((sol[sol.shape[0]//2]/param_vals[0]).item())

        # fd_grad = approx_fprime(param_vals.detach().numpy()[:, None], fun)
        # print(grad.numpy(), 'g')
        # print(fd_grad.T, 'fd')
        # print(grad.numpy()[0]/fd_grad[0, 0])
        # print(grad.numpy()[0]-fd_grad[0, 0])
        # def tmp_fun(x):
        #     # compute sum dudk at mesh points
        #     if bndry_conds[1][1] == "R" or  bndry_conds[1][1] == "N":
        #         tmp = x*(x/2-1)  # u(0)=0 u'(1)=mms(1)
        #     else:
        #         tmp = x*(x-1)/2  # u(0)=u(1)=0
        #     k = residual._diff_fun(np.zeros((1, 1)))
        #     f = residual._forc_fun(np.zeros((1, 1)))
        #     return tmp*(f/k**2).numpy()
        # # true grad when constant forc and diff and qoi = sum(u)
        # print(tmp_fun(mesh.mesh_pts[0, :]).sum(), "hack g")
        # print(tmp_fun(mesh.mesh_pts[0, mesh.mesh_pts.shape[1]//2]), "hack g")
        # assert np.allclose(grad.numpy().T, fd_grad, atol=1e-6)
        # assert False

        errors = check_gradients(
            fun, lambda p: adj_solver.compute_gradient(
                set_param_values,
                torch.as_tensor(p, dtype=torch.double)[:, 0]).numpy(),
            param_vals.numpy()[:, None], plot=False,
            fd_eps=3*np.logspace(-13, 0, 14)[::-1],
            direction=np.array([1])[:, None])
        print(errors.min()/errors.max())
        assert errors.min()/errors.max() < 3.7e-6 # 2.5e-6

    def _get_vertical_transform(self, s0, depth, L, alpha):
        # transformation
        surf_string, bed_string = (
            f"{s0}-{alpha}*_r_**2", f"{s0}-{alpha}*_r_**2-{depth}")
        # brackets are essential around bed string
        y_from_orth_string = f"({surf_string}-({bed_string}))*_t_+{bed_string}"
        y_to_orth_string = (
            f"(_y_-({bed_string}))/({surf_string}-({bed_string}))".replace(
                "_r_", "_x_"))
        vertical_transform = CompositionTransform(
            [ScaleAndTranslationTransform([-1, 1, -1, 1], [-L, L, 0., 1.]),
             SympyTransform(["_r_", y_from_orth_string],
                            ["_x_", y_to_orth_string])])
        return vertical_transform

    def test_advection_diffusion_reaction(self):
        polar_transform = CompositionTransform(
            [ScaleAndTranslationTransform(
                [-1, 1, -1, 1], [0.5, 1, np.pi/4, 3*np.pi/4]),
             PolarTransform()])
        ellipse_transform = CompositionTransform(
            [ScaleAndTranslationTransform(
                [-1, 1, -1, 1], [0.5, 1, np.pi/4, 3*np.pi/4]),
             EllipticalTransform(1)])
        s0, depth, L, alpha = 2, .1, 1, 1e-1
        # s0, depth, L, alpha = 2, 2, 10, 1e-1 # this produces ill conditioned
        vertical_transform = self._get_vertical_transform(s0, depth, L, alpha)

        def nl_diff_jac(linear_diff, sol):
            return linear_diff*2*sol

        test_cases = [
            # 0
            [[0, 1], [4], "-(x-1)*x/2", "4", ["0"],
             [None, None],
             ["D", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["1"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["1"],
             [lambda sol: sol**2, lambda sol: 2*sol[:, 0]],
             # [lambda sol: 1*sol, lambda sol: 1*torch.eye(sol.shape[0])],
             ["D", "D"], ["C"]],
            # 4
            [[0, 1], [20], "0.5*(x-3)*x", "2+x", ["0"],
                [lambda sol: 0*sol,
                lambda sol: torch.zeros((sol.shape[0],))],
             ["N", "D"], ["C"]],
            [[0, 1], [4], "x**2", "1", ["0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "N"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["R", "D"], ["C"]],
            # When using periodic bcs must have reaction term to have a
            # unique solution
            [[0, 2*torch.pi], [30], "sin(x)", "1", ["0"],
             [lambda sol: 1*sol, lambda sol: torch.ones(sol.shape[0])],
             ["P", "P"], ["C"]],
            [[0, 1, 0, 1], [4, 4], "y**2*x**2", "1", ["0", "0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "N", "N", "D"], ["C", "C"]],
            # 9
            [[0, .5, 0, 1], [16, 14], "y**2*sin(pi*x)", "1", ["0", "0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "N", "N", "D"], ["C", "C"]],
            [[0, .5, 0, 1], [16, 14], "x**2*y**2", "1", ["0", "0"],
             [lambda sol: sol**2,
              lambda sol: 2*sol[:, 0]],
             ["D", "N", "N", "D"], ["C", "C"]],
            [[0, .5, 0, 1], [16, 16], "y**2*sin(pi*x)", "1", ["0", "0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "R", "D", "D"], ["C", "C"]],
            [None, [6, 6], "y**2*x**2", "1", ["0", "0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "D", "D", "D"], ["C", "C"], vertical_transform],
            [None, [6, 6], "y**2*x**2", "1", ["1", "0"],
             [lambda sol: 1*sol**2,
              lambda sol: 2*sol[:, 0]],
             ["D", "D", "D", "N"], ["C", "C"], vertical_transform],
            # while solution is quadratic in the user domain
            # the solution is not quadratic in the canonical domain
            # due to nonlinearity of polar coordinate transform
            # 14
            [None, [20, 20], "(x+y)**2", "1", ["0", "0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "D", "D", "N"], ["C", "C"], polar_transform],
            [None, [20, 20], "(x+y)**2", "1", ["0", "0"],
             [lambda sol: 1*sol**2,
              lambda sol: 2*sol[:, 0]],
             ["D", "D", "D", "N"], ["C", "C"], ellipse_transform],
            [[0, 1], [6], "(1+x)**2", "1", ["0"],
                [lambda sol: 0*sol,
                 lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "D"], ["C"], None,
             [lambda linear_diff, sol: linear_diff*(sol**2),
              nl_diff_jac]]
        ]
        ii = 0
        for test_case in test_cases:
            np.random.seed(2)  # controls direction of finite difference
            print(ii)
            print(test_case)
            self._check_advection_diffusion_reaction(*test_case)
            ii += 1

    def _check_adjoint(self, adj_solver, param_vals, functional,
                       set_param_values, init_sol, final_time):

        def fun(params, return_grad=True):
            # newton tol must be smaller than finite difference step size
            if return_grad is False:
                qoi = adj_solver.compute_qoi(
                    init_sol, 0, final_time, set_param_values,
                    torch.as_tensor(params[:, 0], dtype=torch.double))[2]
                return np.atleast_1d(qoi)
            qoi, grad = adj_solver.compute_gradient(
                init_sol, 0, final_time, set_param_values,
                torch.as_tensor(params[:, 0], dtype=torch.double))
            return qoi, grad

        # qoi, grad = adj_solver.compute_gradient(
        #     init_sol, 0, final_time,
        #     set_param_values, param_vals, tol=1e-12)
        # fd_grad = approx_fprime(
        #     param_vals.detach().numpy()[:, None], partial(fun, return_grad=False),
        #     eps=1e-6)
        # print(fd_grad.T, 'fd')
        # print(grad, 'g')
        p0 = param_vals.numpy()[:, None]
        errors = check_gradients(
            fun, True, p0, fd_eps=0.5*np.logspace(-13, 0, 14)[::-1])
        print(errors.min()/errors.max())
        assert errors.min()/errors.max() < 2.5e-6

    def test_decoupled_ode_adjoint(self):
        orders = [2]  # only mid point will be correct applying bndry_conds
        domain_bounds = [0, 1]  # does not effect result

        from pyapprox.pde.autopde.physics import (
            AbstractSpectralCollocationPhysics)

        class DecoupledODE(AbstractSpectralCollocationPhysics):
            def __init__(self, mesh, bndry_conds, b):
                super().__init__(mesh, bndry_conds)
                self._b = b
                self._funs = []

            def _raw_residual(self, sol):
                return -self._b**2*sol+1, -torch.eye(
                    sol.shape[0], dtype=torch.double)*self._b**2

        bndry_conds = [[lambda x: torch.zeros((1, 1)), "D"] for ii in range(2)]
        aparam = 3
        bparam = -2
        deltat = 0.125
        final_time = deltat*2
        tableau_name = "im_beuler1"
        mesh = CartesianProductCollocationMesh(domain_bounds, orders)
        solver = TransientPDE(
            DecoupledODE(mesh, bndry_conds, bparam), deltat, tableau_name)
        init_sol = aparam**2*torch.ones(
            mesh.mesh_pts.shape[1], dtype=torch.double)
        sols, times = solver.solve(
            init_sol, 0, final_time, newton_kwargs={"tol": 1e-8})
        # print(sols)

        def functional(sols, params):
            # return sols[1, -1]
            return deltat*sols[1, 1:].sum()
        param_vals = torch.as_tensor([bparam], dtype=torch.double)
        adj_solver = TransientAdjointPDE(
            DecoupledODE(mesh, bndry_conds, bparam),
            deltat, tableau_name, functional)

        def set_param_values(residual, init_sol, param_vals):
            # assert param_vals.ndim == 1
            residual._b = param_vals[0]
            # update init sol if params effects it. In this case it does not
            return init_sol
        self._check_adjoint(adj_solver, param_vals, functional,
                            set_param_values, init_sol, final_time)

    def _check_transient_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string,
            diff_string, vel_strings, react_funs, bndry_types,
            tableau_name):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], True))

        diff_fun = Function(diff_fun)
        vel_fun = Function(vel_fun)
        forc_fun = TransientFunction(forc_fun, name='forcing')
        sol_fun = TransientFunction(sol_fun, name='sol')
        flux_funs = TransientFunction(flux_funs, name='flux')

        nphys_vars = len(orders)
        bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, sol_fun, flux_funs)

        deltat = 1  # 0.1
        final_time = deltat*2  # 5
        mesh = CartesianProductCollocationMesh(domain_bounds, orders)
        solver = TransientPDE(
            AdvectionDiffusionReaction(
                mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
                react_funs[1]), deltat, tableau_name)
        sol_fun.set_time(0)
        init_sol = sol_fun(mesh.mesh_pts)
        sols, times = solver.solve(
            init_sol, 0, final_time, newton_kwargs={"tol": 1e-8})

        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sol_t = sol_fun(solver.physics.mesh.mesh_pts).numpy()
            model_sol_t = sols[:, ii:ii+1].numpy()
            # print(exact_sol_t)
            # print(model_sol_t, 'm')
            L2_error = np.sqrt(
                solver.physics.mesh.integrate((exact_sol_t-model_sol_t)**2))
            factor = np.sqrt(
                solver.physics.mesh.integrate(exact_sol_t**2))
            # print(time, L2_error, 1e-8*factor)
            assert L2_error < 1e-8*factor

        # print(bndry_conds)
        if tableau_name != "im_beuler1":
            return

        def functional(sols, params):
            # return sols[:, -1].sum()
            idx = sols.shape[0] // 2
            return deltat*sols[idx, 1:].sum()
        param_vals = diff_fun(mesh.mesh_pts)[:1, 0]
        # param_vals = diff_fun(mesh.mesh_pts)[0:1, 0]*0+1
        adj_solver = TransientAdjointPDE(AdvectionDiffusionReaction(
            mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
            react_funs[1]), deltat, tableau_name, functional)

        def set_param_values(residual, init_sol, param_vals):
            # assert param_vals.ndim == 1
            mesh_vals = torch.tile(param_vals, (mesh.mesh_pts.shape[1], ))
            residual._diff_fun = partial(
                residual.mesh.interpolate, mesh_vals)
            return init_sol

        self._check_adjoint(adj_solver, param_vals, functional,
                            set_param_values, init_sol, final_time)

    def test_transient_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"],
             [None, None], ["D", "D"], "im_beuler1"],
            [[0, 1], [4], "x**2*(1+t)", "3", ["0"],
             [None, None], ["D", "D"], "im_beuler1"],
            [[0, 1], [4], "x**2*(1+t)", "3", ["0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "N"], "im_beuler1"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["0"],
             [lambda sol: 0*sol,
              lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "D"], "im_crank2"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["1"],
             [lambda sol: 1*sol**2,
              lambda sol: 2*sol[:, 0]],
             ["D", "D"], "im_crank2"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["1"],
             [lambda sol: 1*sol**2,
              lambda sol: 2*sol[:, 0]],
             ["N", "D"], "im_crank2"],
            [[0, 1, 0, 1], [3, 3], "(x-1)*x*(1+t)*y**2", "1", ["1", "1"],
             [None, None], ["D", "N", "R", "D"], "im_beuler1"],
            [[0, 1, 0, 1], [3, 3], "(x-1)*x*(1+t)**2*y**2", "1", ["1", "1"],
             [lambda sol: 1*sol**2,
              lambda sol: 2*sol[:, 0]],
             ["D", "N", "R", "D"], "im_crank2"]
        ]
        ii = 0
        for test_case in test_cases:
            print(ii)
            print(test_case)
            self._check_transient_advection_diffusion_reaction(*test_case)
            ii += 1

    def _check_stokes_solver_mms(
            self, domain_bounds, orders, vel_strings, pres_string, bndry_types,
            navier_stokes, transform=None):
        (vel_fun, pres_fun, vel_forc_fun, pres_forc_fun, vel_grad_funs,
         pres_grad_fun) = setup_steady_stokes_manufactured_solution(
             vel_strings, pres_string, navier_stokes)

        vel_fun = Function(vel_fun)
        pres_fun = Function(pres_fun)
        vel_forc_fun = Function(vel_forc_fun)
        pres_forc_fun = Function(pres_forc_fun)
        pres_grad_fun = Function(pres_grad_fun)

        nphys_vars = len(orders)
        if transform is None:
            boundary_normals = None
        else:
            boundary_normals = [
                partial(transform.normal, ii) for ii in range(2*nphys_vars)]
        vel_bndry_conds = [
            _get_boundary_funs(
                nphys_vars, bndry_types,
                partial(_vel_component_fun, vel_fun, ii),
                vel_grad_funs[ii], boundary_normals)
            for ii in range(nphys_vars)]
        bndry_conds = vel_bndry_conds + [[[None, None]]*(2*nphys_vars)]

        if transform is None:
            vel_meshes = [
                CartesianProductCollocationMesh(
                    domain_bounds, orders)]*nphys_vars
            pres_mesh = InteriorCartesianProductCollocationMesh(
                domain_bounds, orders)
        else:
            vel_meshes = [TransformedCollocationMesh(
                orders, transform)]*nphys_vars
            pres_mesh = TransformedInteriorCollocationMesh(
                orders, transform)
        mesh = VectorMesh(vel_meshes + [pres_mesh])
        pres_idx = 0
        pres_val = pres_fun(pres_mesh.mesh_pts[:, pres_idx:pres_idx+1])
        if not navier_stokes:
            Residual = LinearIncompressibleStokes
        else:
            Residual = IncompressibleNavierStokes
        solver = SteadyStatePDE(Residual(
            mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
            (pres_idx, pres_val)))

        exact_vel_vals = vel_fun(vel_meshes[0].mesh_pts).numpy()
        exact_pres_vals = pres_fun(pres_mesh.mesh_pts).numpy()
        exact_sol = torch.vstack(
            [v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts).T] +
            [pres_fun(pres_mesh.mesh_pts)])

        print(np.abs(solver.physics._raw_residual(exact_sol[:, 0])[0]).max())
        assert np.allclose(
            solver.physics._raw_residual(exact_sol[:, 0])[0], 0, atol=2.3e-8)
        assert np.allclose(
            solver.physics._residual(exact_sol[:, 0])[0], 0, atol=2.2e-8)

        def fun(s):
            return solver.physics._raw_residual(
                torch.as_tensor(s, dtype=torch.double))[0].numpy()
        j_fd = approx_jacobian(fun, exact_sol[:, 0].numpy())
        j_man = solver.physics._raw_residual(
            torch.as_tensor(exact_sol[:, 0], dtype=torch.double))[1].numpy()
        j_auto = torch.autograd.functional.jacobian(
            lambda s: solver.physics._raw_residual(s)[0],
            exact_sol[:, 0].clone().requires_grad_(True), strict=True).numpy()
        # np.set_printoptions(precision=2, suppress=True, threshold=100000,
        #                     linewidth=1000)
        # print(j_auto[:16, 32:])
        # # print(j_fd[:16, 32:])
        # print(j_man[:16, 32:])
        # # print((j_auto-j_fd)[:16, 32:])
        # # print((j_auto-j_man)[:16, 32:])
        # print(np.abs(j_auto-j_man).max())
        # print(np.abs(j_auto-j_fd).max())
        assert np.allclose(j_auto, j_man)

        sol = solver.solve(maxiters=10, rtol=1e-12)[:, None].detach().numpy()

        split_sols = mesh.split_quantities(sol)

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3, figsize=(8*3, 6))
        # plt_objs = mesh.plot(
        #     [v[:, None] for v in exact_vel_vals.T]+[exact_pres_vals])
        # plt_objs = mesh.plot(split_sols, axs=axs)
        exact_sols = [v[:, None] for v in exact_vel_vals.T]+[exact_pres_vals]
        # plt_objs = mesh.plot(
        #     [v-u for v, u in zip(exact_sols, split_sols)])
        # for ax, obj in zip(axs, plt_objs):
        #     plt.colorbar(obj, ax=ax)
        #     plt.show()

        # check value used to enforce unique pressure is found correctly
        assert np.allclose(
            split_sols[-1][pres_idx], pres_val)

        for exact_v, v in zip(exact_vel_vals.T, split_sols[:-1]):
            assert np.allclose(exact_v, v[:, 0])
            print(np.abs(exact_pres_vals-split_sols[-1]).max())
            assert np.allclose(exact_pres_vals, split_sols[-1], atol=9e-8)

    def test_stokes_solver_mms(self):
        s0, depth, L, alpha = 2, .1, 1, 1e-1
        vertical_transform = self._get_vertical_transform(s0, depth, L, alpha)
        test_cases = [
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["D", "D"], False],
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["N", "D"], False],
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["D", "D"], True],
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["D", "N"], True],
            [[0, 1, 0, 1], [20, 20],
             ["-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"], "x**3*y**3",
             ["D", "D", "D", "D"], False],
            [[0, 1, 0, 1], [6, 7],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "D"], False],
            [[0, 1, 0, 1], [4, 4], #[12, 12],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "D"], True],
            [[0, 1, 0, 1], [8, 8],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "D"], True, vertical_transform],
            [[0, 1, 0, 1], [8, 8],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "N"], True, vertical_transform]
        ]
        for test_case in test_cases:
            self._check_stokes_solver_mms(*test_case)

    def _check_shallow_ice_solver_mms(
            self, domain_bounds, orders, depth_string, bed_string, beta_string,
            bndry_types, A, rho, n, g, transient):
        nphys_vars = len(orders)
        depth_fun, bed_fun, beta_fun, forc_fun, flux_funs = (
            setup_shallow_ice_manufactured_solution(
                depth_string, bed_string, beta_string, A, rho, n,
                g, nphys_vars, transient))

        depth_fun = Function(depth_fun)
        bed_fun = Function(bed_fun)
        beta_fun = Function(beta_fun)
        forc_fun = Function(forc_fun)
        flux_funs = Function(flux_funs)

        bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, depth_fun, flux_funs)
        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders)

        solver = SteadyStatePDE(ShallowIce(
            mesh, bndry_conds, bed_fun, beta_fun, forc_fun, A, rho, n, g))

        exact_sol = depth_fun(mesh.mesh_pts)
        # print(np.abs(solver.physics._raw_residual(exact_sol[:, 0])[0]).max())
        # print(np.abs(solver.physics._raw_residual(exact_sol[:, 0])))

        def fun(s):
            return solver.physics._raw_residual(
                torch.as_tensor(s, dtype=torch.double))[0].numpy()
        j_fd = approx_jacobian(fun, exact_sol[:, 0].numpy())
        # j_man = solver.physics._raw_residual(torch.as_tensor(exact_sol[:, 0]))[1].numpy()
        j_auto = torch.autograd.functional.jacobian(
            lambda s: solver.physics._raw_residual(s)[0],
            exact_sol[:, 0].clone().requires_grad_(True), strict=True).numpy()
        j_man = solver.physics._raw_jacobian(exact_sol[:, 0].clone()).numpy()
        # np.set_printoptions(precision=2, suppress=True, threshold=100000, linewidth=1000)
        # print(j_fd)
        # print(j_auto)
        # print(j_man)
        assert np.allclose(j_auto, j_man)

        assert np.allclose(
            solver.physics._raw_residual(exact_sol[:, 0])[0], 0, atol=2e-8)
        assert np.allclose(
            solver.physics._residual(exact_sol[:, 0])[0], 0, atol=2e-8)


    def test_shallow_ice_solver_mms(self):
        s0, depth, alpha = 2, .1, 1e-1
        test_cases = [
            [[-1, 1], [4], "1", f"{s0}-{alpha}*x**2-{depth}", "1",
             ["D", "D"], 1, 1, 1, 1, False],
            [[-1, 1, -1, 1], [4, 4], "1", f"{s0}-{alpha}*x**2-{depth}-(1+y)", "1",
             ["D", "D", "D", "D"], 1, 1, 1, 1, False]
        ]
        for test_case in test_cases:
            self._check_shallow_ice_solver_mms(*test_case)

    def test_euler_bernoulli_beam(self):
        # bndry_conds are None because they are imposed in the solver
        # This 4th order equation requires 4 boundary conditions, two at each
        # end of the domain. This cannot be done with usual boundary condition
        # functions and msut be imposed on the residual exactly
        domain_bounds, orders, bndry_conds = [0, 1], [4], [[None, None]]*2
        emod_val, smom_val, forcing_val = 1., 1., -2.
        mesh = CartesianProductCollocationMesh(domain_bounds, orders)
        solver = SteadyStatePDE(EulerBernoulliBeam(
            mesh, bndry_conds, Function(lambda x: np.full((x.shape[1], 1), 1)),
            Function(lambda x: np.full((x.shape[1], 1), 1)),
            Function(lambda x: np.full((x.shape[1], 1), forcing_val))))

        def sol_fun(x):
            length = domain_bounds[1]-domain_bounds[0]
            return (forcing_val*x**2*(6*length**2-4*length*x+x**2)/(
                24*emod_val*smom_val)).T

        exact_sol_vals = sol_fun(mesh.mesh_pts)
        assert np.allclose(
            solver.physics._raw_residual(
                torch.tensor(exact_sol_vals[:, 0]))[0], 0)

        sol = solver.solve().detach()[:, None]
        assert np.allclose(sol, exact_sol_vals)

    def _check_helmholtz(self, domain_bounds, orders, sol_string, wnum_string,
                         bndry_types):
        sol_fun, wnum_fun, forc_fun, flux_funs = (
            setup_helmholtz_manufactured_solution(
                sol_string, wnum_string, len(domain_bounds)//2))

        wnum_fun = Function(wnum_fun)
        forc_fun = Function(forc_fun)
        sol_fun = Function(sol_fun)

        nphys_vars = len(orders)
        bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, sol_fun, flux_funs)

        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders)
        solver = SteadyStatePDE(
            Helmholtz(mesh, bndry_conds, wnum_fun, forc_fun))
        sol = solver.solve().detach()

        print(np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol[:, None]))
        assert np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol[:, None]) < 1e-9

    def test_helmholtz_solver_mms(self):
        test_cases = [
            [[0, 1], [16], "x**2", "1", ["N", "D"]],
            [[0, .5, 0, 1], [16, 16], "y**2*x**2", "1", ["N", "D", "D", "D"]]]
        for test_case in test_cases:
            self._check_helmholtz(*test_case)

    def test_shallow_water_wave_solver_mms_setup(self):
        # or Analytical solutions see
        # https://hal.archives-ouvertes.fr/hal-00628246v6/document
        def bernoulli_realtion(q, bed_fun, C, x, h):
            return q**2/(2*9.81*h**2)+h+bed_fun(x)-C, None
        x = torch.linspace(0, 1, 11,dtype=torch.double)
        from pyapprox.pde.autopde.util import newton_solve
        def bed_fun(x):
            return -x**2*0
        q, C = 1, 1
        init_guess = C-bed_fun(x).requires_grad_(True)
        fun = partial(bernoulli_realtion, q, bed_fun, C, x)
        sol = newton_solve(fun, init_guess, tol=1e-12, verbosity=2,
                           maxiters=20)
        sol = sol.detach().numpy()
        assert np.allclose(sol, sol[0], atol=1e-12)

        # import matplotlib.pyplot as plt
        # plt.plot(x.numpy(), bed_fun(x).numpy())
        # plt.plot(x.numpy(), bed_fun(x).numpy()+sol)
        # plt.show()

        vel_strings = ["%f"%q]
        bed_string = "0"
        depth_string = "%f"%sol[0]
        depth_fun, vel_fun, depth_forc_fun, vel_forc_fun, bed_fun = (
            setup_shallow_water_wave_equations_manufactured_solution(
                vel_strings, depth_string, bed_string))
        xx = torch.linspace(0, 1, 11)[None, :]
        assert np.allclose(depth_forc_fun(xx), 0, atol=1e-12)
        assert np.allclose(vel_forc_fun(xx), 0, atol=1e-12)

    def _check_shallow_water_wave_solver_mms(
            self, domain_bounds, orders, vel_strings, depth_string, bed_string,
            bndry_types):
        nphys_vars = len(vel_strings)
        depth_fun, vel_fun, depth_forc_fun, vel_forc_fun, bed_fun = (
            setup_shallow_water_wave_equations_manufactured_solution(
                vel_strings, depth_string, bed_string))

        bed_fun = Function(bed_fun)

        depth_forc_fun = Function(depth_forc_fun)
        vel_forc_fun = Function(vel_forc_fun)
        depth_fun = Function(depth_fun)
        vel_fun = Function(vel_fun)

        # TODO test neumann boundary conditions so need flux funs
        # returned by MMS
        flux_funs = None
        depth_bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, depth_fun, flux_funs)
        vel_bndry_conds = [_get_boundary_funs(
            nphys_vars, bndry_types,
            partial(_sww_momentum_component_fun, vel_fun, depth_fun, ii),
            flux_funs) for ii in range(nphys_vars)]
        bndry_conds = [depth_bndry_conds]+vel_bndry_conds

        depth_mesh = CartesianProductCollocationMesh(
            domain_bounds, orders)
        vel_meshes = [
            CartesianProductCollocationMesh(domain_bounds, orders)]*nphys_vars
        mesh = VectorMesh([depth_mesh]+vel_meshes)

        solver = SteadyStatePDE(
            ShallowWaterWave(
                mesh, bndry_conds, depth_forc_fun, vel_forc_fun, bed_fun))
        exact_depth_vals = depth_fun(depth_mesh.mesh_pts)
        exact_mom_vals = [exact_depth_vals*v[:, None]
                          for v in vel_fun(vel_meshes[0].mesh_pts).T]
        # split_sols = [q1, q2, q3] = [h, uh, vh]
        init_guess = torch.cat([exact_depth_vals] + exact_mom_vals)

        # solver.physics._auto_jac = True
        res_vals = solver.physics._raw_residual(init_guess.squeeze())[0]
        assert np.allclose(res_vals, 0)

        init_guess = init_guess+torch.randn(init_guess.shape)*1e-3
        # np.set_printoptions(precision=2, suppress=True, threshold=100000, linewidth=1000)
        j_man = solver.physics._raw_residual(init_guess.squeeze())[1]
        j_auto = torch.autograd.functional.jacobian(
            lambda s: solver.physics._raw_residual(s)[0],
            init_guess[:, 0].clone().requires_grad_(True), strict=True).numpy()
        # print((j_man.numpy()-j_auto)[32:, 32:])
        assert np.allclose(j_man, j_auto)

        init_guess = init_guess+torch.randn(init_guess.shape)*1e-3
        sol = solver.solve(init_guess, tol=1e-8, maxiters=10, verbosity=0)
        split_sols = mesh.split_quantities(sol[:, None])
        assert np.allclose(exact_depth_vals, split_sols[0])
        for exact_v, v in zip(exact_mom_vals, split_sols[1:]):
            # print(exact_v[:, 0]-v[:, 0])
            assert np.allclose(exact_v[:, 0], v[:, 0])

    def test_shallow_water_wave_solver_mms(self):
        # order must be odd or Jacobian will be almost uninvertable and
        # newton solve will diverge
        test_cases = [
            [[0, 1], [5], ["-x**2"], "1+x", "0", ["D", "D"]],
            [[0, 1, 0, 1], [5, 5], ["-x**2", "-y**2"], "1+x+y", "0",
             ["D", "D", "D", "D"]]
        ]
        for test_case in test_cases:
            self._check_shallow_water_wave_solver_mms(*test_case)

    def _check_shallow_water_wave_transient_solver_mms(
            self, domain_bounds, orders, vel_strings, depth_string, bed_string,
            bndry_types, tableau_name):
        nphys_vars = len(vel_strings)
        depth_fun, vel_fun, depth_forc_fun, vel_forc_fun, bed_fun = (
            setup_shallow_water_wave_equations_manufactured_solution(
                vel_strings, depth_string, bed_string, True))

        bed_fun = Function(bed_fun)

        depth_forc_fun = TransientFunction(depth_forc_fun)
        vel_forc_fun = TransientFunction(vel_forc_fun)
        depth_fun = TransientFunction(depth_fun)
        vel_fun = TransientFunction(vel_fun)

        depth_forc_fun._name = 'depth_f'
        vel_forc_fun._name = 'vel_f'
        bed_fun._name = 'bed'
        vel_fun._name = 'vel'
        depth_fun._name = 'depth'

        # TODO test neumann boundary conditions so need flux funs
        # returned by MMS
        flux_funs = None
        depth_bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, depth_fun, flux_funs)
        mom_bndry_conds = [_get_boundary_funs(
            nphys_vars, bndry_types,
            TransientFunction(partial(
                _sww_momentum_component_fun, vel_fun, depth_fun, ii)),
            flux_funs) for ii in range(nphys_vars)]
        bndry_conds = [depth_bndry_conds]+mom_bndry_conds

        depth_mesh = CartesianProductCollocationMesh(domain_bounds, orders)
        mom_meshes = [
            CartesianProductCollocationMesh(domain_bounds, orders)]*nphys_vars
        mesh = VectorMesh([depth_mesh]+mom_meshes)

        depth_fun.set_time(0)
        vel_fun.set_time(0)
        depth_forc_fun.set_time(0)
        vel_forc_fun.set_time(0)

        deltat = 0.1
        final_time = deltat
        solver = TransientPDE(
            ShallowWaterWave(mesh, bndry_conds, depth_forc_fun, vel_forc_fun,
                             bed_fun), deltat,
            tableau_name)
        init_depth_vals = depth_fun(depth_mesh.mesh_pts)
        init_sol = torch.cat(
            [init_depth_vals] +
            [init_depth_vals*v[:, None]
             for v in vel_fun(mom_meshes[0].mesh_pts).T])
        sols, times = solver.solve(
            init_sol, 0, final_time, newton_kwargs={"tol": 1e-8})
        sols = sols.numpy()

        import matplotlib.pyplot as plt
        for ii, time in enumerate(times):
            depth_fun.set_time(time)
            vel_fun.set_time(time)
            depth_vals = depth_fun(depth_mesh.mesh_pts)
            exact_sol_t = np.vstack(
                [depth_vals] +
                [depth_vals*v[:, None]
                 for v in vel_fun(mom_meshes[0].mesh_pts).T])
            model_sol_t = sols[:, ii:ii+1]
            # print(np.hstack((mesh.split_quantities(
            #     exact_sol_t)[2], mesh.split_quantities(model_sol_t)[2])))
            # print(np.hstack((exact_sol_t, model_sol_t)))
            # print(mesh.split_quantities((exact_sol_t-model_sol_t))[1])
            if False: #ii >= 0:
                fig, axs = plt.subplots(
                    1, mesh.nphys_vars+1, figsize=(8*(mesh.nphys_vars+1), 6))
                # mesh.plot(mesh.split_quantities(exact_sol_t), axs=axs)
                mesh.plot(mesh.split_quantities(model_sol_t), axs=axs, ls='--')
            # print(exact_sol_t.shape)
            # print(model_sol_t.shape)
            L2_error = np.sqrt(
                mesh.integrate(
                    mesh.split_quantities((exact_sol_t-model_sol_t)**2)))
            # print(time, L2_error, 'l')
            # plt.show()
            assert np.all(L2_error < 1e-8)

    def test_shallow_water_wave_transient_solver_mms(self):
        # order must be odd or Jacobian will be almost uninvertable and
        # newton solve will diverge

        test_cases = [
            [[0, 1], [5], ["-x**2*(t+1)"], "(1+x)*(t+1)", "0", ["D", "D"],
             "im_crank2"],
            [[0, 1, 0, 1], [5, 5], ["-x**2*(t+1)", "-y**2*(t+1)"],
             "(1+x+y)*(t+1)", "0",
             ["D", "D", "D", "D"], "im_crank2"]
        ]
        for test_case in test_cases:
            self._check_shallow_water_wave_transient_solver_mms(*test_case)

    def _check_shallow_shelf_solver_mms(
            self, domain_bounds, orders, vel_strings, depth_string, bed_string,
            beta_string, bndry_types, velocities_only):
        A, rho = 1, 1
        nphys_vars = len(vel_strings)
        depth_fun, vel_fun, vel_forc_fun, bed_fun, beta_fun, depth_forc_fun = (
            setup_shallow_shelf_manufactured_solution(
                depth_string, vel_strings, bed_string, beta_string, A, rho))

        bed_fun = Function(bed_fun, 'bed')
        beta_fun = Function(beta_fun, 'beta')
        depth_fun = Function(depth_fun, 'depth')
        depth_forc_fun = Function(depth_forc_fun, 'depth_forc')

        vel_forc_fun = Function(vel_forc_fun, 'vel_forc')
        vel_fun = Function(vel_fun, 'vel')

        # TODO test neumann boundary conditions so need flux funs
        # returned by MMS
        flux_funs = None
        vel_bndry_conds = [_get_boundary_funs(
            nphys_vars, bndry_types,
            partial(_vel_component_fun, vel_fun, ii),
            flux_funs) for ii in range(nphys_vars)]
        depth_bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, depth_fun, flux_funs)

        vel_meshes = [
            CartesianProductCollocationMesh(domain_bounds, orders)]*nphys_vars
        depth_mesh = CartesianProductCollocationMesh(domain_bounds, orders)
        if velocities_only:
            mesh = VectorMesh(vel_meshes)
            bndry_conds = vel_bndry_conds
        else:
            mesh = VectorMesh(vel_meshes+[depth_mesh])
            bndry_conds = vel_bndry_conds + [depth_bndry_conds]

        exact_vel_vals = [
            v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts).T]
        exact_depth_vals = depth_fun(vel_meshes[0].mesh_pts)
        if velocities_only:
            solver = SteadyStatePDE(
                ShallowShelfVelocities(
                    mesh, bndry_conds, vel_forc_fun, bed_fun, beta_fun,
                    depth_fun, A, rho, 1e-15))
            init_guess = torch.cat(exact_vel_vals)
        else:
            solver = SteadyStatePDE(
                ShallowShelf(mesh, bndry_conds, vel_forc_fun, bed_fun,
                             beta_fun, depth_forc_fun, A, rho, 1e-15))
            init_guess = torch.cat(exact_vel_vals+[exact_depth_vals])

        # np.set_printoptions(
        #     precision=2, suppress=True, threshold=100000, linewidth=1000)
        # print(init_guess, 'i')
        res_vals = solver.physics._raw_residual(init_guess.squeeze())[0]
        # print(np.abs(res_vals.detach().numpy()).max(), 'r')
        assert np.allclose(res_vals, 0, atol=5e-8)

        if velocities_only:
            init_guess = torch.randn(init_guess.shape, dtype=torch.double)*0.1
        else:
            init_guess = (init_guess+torch.randn(init_guess.shape)*5e-3)

        dudx_ij = solver.physics._derivs(
            mesh.split_quantities(init_guess[:, 0]))
        j_visc_man = torch.hstack(solver.physics._effective_strain_rate_jac(
            dudx_ij))
        j_visc_auto = torch.autograd.functional.jacobian(
            lambda s: solver.physics._effective_strain_rate(
                solver.physics._derivs(mesh.split_quantities(s))),
            init_guess[:, 0].clone().requires_grad_(True), strict=True).numpy()
        # print(j_visc_man.numpy()[:16, :16]-j_visc_auto[:16, :16])
        # print(j_visc_man.numpy()[:16, 16:]-j_visc_auto[:16, 16:32])
        # print(j_visc_man.shape, j_visc_auto.shape)
        assert np.allclose(
            j_visc_auto[:, :j_visc_man.shape[1]], j_visc_man.numpy())

        j_man = solver.physics._vector_components_jac(dudx_ij)
        for ii in range(nphys_vars):
            for jj in range(nphys_vars):
                j_auto = torch.autograd.functional.jacobian(
                    lambda s: solver.physics._vector_components(
                        solver.physics._derivs(
                            mesh.split_quantities(s)))[ii][:, jj],
                    init_guess[:, 0].clone().requires_grad_(True),
                    strict=True).numpy()
                for dd in range(nphys_vars):
                    cnt = j_man[ii][dd][jj].shape[1]
                    assert np.allclose(j_man[ii][dd][jj].numpy(),
                                       j_auto[:, dd*cnt:(dd+1)*cnt])

        j_man = solver.physics._raw_residual(init_guess.squeeze())[1]
        j_auto = torch.autograd.functional.jacobian(
            lambda s: solver.physics._raw_residual(s)[0],
            init_guess[:, 0].clone().requires_grad_(True), strict=True).numpy()
        assert np.allclose(j_man, j_auto)

        if velocities_only:
            init_guess = torch.randn(init_guess.shape, dtype=torch.double)*0+1
        else:
            init_guess = (init_guess+torch.randn(init_guess.shape)*5e-3)

        sol = solver.solve(
            init_guess, tol=1e-7, verbosity=2, maxiters=10)[:, None]
        split_sols = mesh.split_quantities(sol)
        for exact_v, v in zip(exact_vel_vals, split_sols):
            # print(exact_v[:, 0]-v[:, 0])
            assert np.allclose(exact_v[:, 0], v[:, 0])
        if not velocities_only:
            assert np.allclose(exact_depth_vals, split_sols[-1])

    def test_shallow_shelf_solver_mms(self):
        # Avoid velocity=0 in any part of the domain
        test_cases = [
            [[0, 1], [9], ["(x+2)**2"], "1+x**2", "-x**2", "1",
             ["D", "D"], True],
            [[0, 1], [9], ["(x+2)**2"], "1+x**2", "-x**2", "1",
             ["D", "D"], False],
            [[0, 1, 0, 1], [11, 11],
             ["(x+1)**2*(y+1)", "(y+1)*(x+1)"], "1+x+y",
             "1+x+y**2", "1", ["D", "D", "D", "D"], True],
            # odd order for at least one direction is needed for this example
            # for newton solver to converge. I think this has to do
            # with adding advection of depth, same happens for shallow water
            [[0, 1, 0, 1], [11, 11], ["(x+1)**2", "(y+1)**2"], "1+x+y",
             "0-x-y", "1", ["D", "D", "D", "D"], False]
        ]
        for test_case in test_cases:
            self._check_shallow_shelf_solver_mms(*test_case)

    def test_first_order_stokes_ice_mms(self):
        """
        Match manufactured solution from
        I .K. Tezaur et al.: A finite element,
        first-order Stokes approximation ice sheet solver

        There seems to be a mistake in that paper in the definition of the
        forcing (f1 below)
        """
        L, s0, H, alpha, beta, n, rho, g, A = (
            50, 2, 1, 4e-5, 1, 3, 910, 9.8, 1e-4)
        s = f"{s0}-{alpha}*x**2"
        dsdx = f"(-2*{alpha}*x)"
        vel_string = (
            f"2*{A}*({rho}*{g})**{n}/({n}+1)" +
            f"*((({s})-z)**({n}+1)-{H}**({n}+1))" +
            f"*{dsdx}**({n}-1)*{dsdx}-{rho}*{g}*{H}*{dsdx}/{beta}")
        test_case = [
            f"{H}", [vel_string], f"{s}-{H}",
            f"{beta}", A, rho, g, alpha, n, 50, True]
        (depth_fun, vel_fun, vel_forc_fun, bed_fun, beta_fun, bndry_funs,
         depth_expr, vel_expr, vel_forc_expr, bed_expr, beta_expr,
         bndry_exprs, ux, visc_expr, surface_normal) = (
             setup_first_order_stokes_ice_manufactured_solution(*test_case))

        import sympy as sp
        sp_x, sp_z = sp.symbols(['x', 'z'])
        symbs = (sp_x, sp_z)
        surface_expr = bed_expr+depth_expr
        phi1 = sp_z-surface_expr
        phi2 = 4*A*(alpha*rho*g)**3*sp_x
        phi3 = 4*sp_x**3*phi1**5*phi2**2
        phi4 = (8*alpha*sp_x**3*phi1**3*phi2 -
                (2*depth_expr*alpha*rho*g)/beta_expr +
                3*sp_x*phi2*(phi1**4-depth_expr**4))
        phi5 = (56*alpha*sp_x**2*phi1**3*phi2 +
                48*alpha**2*sp_x**4*phi1**2*phi2 +
                6*phi2*(phi1**4-depth_expr**4))
        mu = 1/2*(A*phi4**2+A*sp_x*phi1*phi3)**(-1/3)
        f1 = 16/3*A*mu**4*(
            -2*phi4**2*phi5+24*phi3*phi4*(phi1+2*alpha*sp_x**2) -
            6*sp_x**3*phi1**3*phi2*phi3-18*sp_x**2*phi1**2*phi2*phi4**2 -
            6*sp_x*phi1*phi3*phi5)

        # phi4 = -du/dx = -ux[0]
        # phi2 = 4*A*alpha**2*(rho*g)**3*(ds/dx)

        # below does not equal zero exactly unless A = 1 but is zero
        # to machine precision. This can only be checked though by lamdifying
        #  and evaluating expression for values of x
        xx = np.array([-1, -0.5, 0.5, 1])
        # assert (sp.simplify(ux[0]+phi4)) == 0
        # assert sp.simplify(ux[1]**2/4 - sp_x*phi1*phi3) == 0
        # assert (sp.simplify(visc_expr-mu) == 0)
        # print(sp.lambdify(symbs, visc_expr-mu, "numpy")(
        #         xx, bed_fun(xx[None, :])[:, 0]))
        assert np.allclose(
            sp.lambdify(symbs, visc_expr-mu, "numpy")(
                xx, bed_fun(xx[None, :])[:, 0]), 0)

        # print(vel_forc_expr[0])
        # assert np.allclose(
        #     sp.lambdify(symbs, vel_forc_expr[0]-f1, "numpy")(
        #         xx, bed_fun(xx[None, :])[:, 0]), 0)

        assert np.allclose(
            sp.lambdify(symbs, -(-4*phi4*mu)-bndry_exprs[0], "numpy")(
                xx, bed_fun(xx[None, :])[:, 0]), 0)
        assert np.allclose(
            sp.lambdify(symbs, -4*phi4*mu-bndry_exprs[1], "numpy")(
                xx, bed_fun(xx[None, :])[:, 0]), 0)
        assert (
            surface_normal[0]-(2*alpha*sp_x)/(4*alpha**2*sp_x**2+1)**(1/2) == 0)
        assert np.allclose(
            sp.lambdify(
                symbs,
                (-4*phi4*mu*surface_normal[0]-4*phi2*sp_x**2*phi1**3*mu*1) -
                bndry_exprs[3], "numpy")(
                    xx, (bed_fun(xx[None, :])+depth_fun(xx[None, :]))[:, 0]), 0)
        assert np.allclose(
            sp.lambdify(
                symbs,
                (-4*phi4*mu*(-surface_normal[0]) -
                 4*phi2*sp_x**2*phi1**3*mu*(-1) +
                 2*depth_expr*alpha*rho*g*sp_x -
                 beta_expr*sp_x**2*phi2*(phi1**4-depth_expr**4)) -
                bndry_exprs[2], "numpy")(xx, bed_fun(xx[None, :])[:, 0]), 0)

    def _check_first_order_stokes_ice_solver_mms(
            self, orders, vel_strings, depth_string, bed_string,
            beta_string, A, rho, g, alpha, n, L):
        nphys_vars = 2
        depth_fun, vel_fun, vel_forc_fun, bed_fun, beta_fun, bndry_funs = (
            setup_first_order_stokes_ice_manufactured_solution(
                depth_string, vel_strings, bed_string, beta_string, A, rho, g,
                alpha, n, L))

        bed_fun = Function(bed_fun, 'bed')
        beta_fun = Function(beta_fun, 'beta')
        depth_fun = Function(depth_fun, 'depth')

        vel_forc_fun = Function(vel_forc_fun, 'vel_forc')
        vel_fun = Function(vel_fun, 'vel')

        s0 = 2
        depth = 1
        transform = self._get_vertical_transform(s0, depth, L, alpha)

        vel_meshes = [TransformedCollocationMesh(orders, transform)]*(
            nphys_vars-1)
        mesh = VectorMesh(vel_meshes)

        # vel_meshes[0].plot(vel_fun(vel_meshes[0].mesh_pts)[:, :1], 10)
        # import matplotlib.pyplot as plt
        # plt.plot(vel_meshes[0].mesh_pts[0, :], vel_meshes[0].mesh_pts[1, :], 'o')

        # placeholder so that custom boundary conditions can be added
        # after residual is created
        bndry_conds = [_get_boundary_funs(
            nphys_vars, ["D", "D", "D", "D"], vel_fun, None)]

        exact_vel_vals = [
            v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts).T]
        solver = SteadyStatePDE(
            FirstOrderStokesIce(
                mesh, bndry_conds, vel_forc_fun, bed_fun, beta_fun,
                depth_fun, A, rho, 0))
        # define correct custom boundaries
        for ii in range(len(mesh._meshes[0]._bndrys)):
            solver.physics._bndry_conds[0][ii] = [
                Function(bndry_funs[ii]), "C",
                solver.physics._strain_boundary_conditions]
        solver.physics._n = n
        init_guess = torch.cat(exact_vel_vals)
        res_vals = solver.physics._raw_residual(init_guess.squeeze())[0]
        res_error = (np.linalg.norm(res_vals.detach().numpy()) /
                     np.linalg.norm(solver.physics._forc_vals[:, 0].numpy()))
        # print(np.linalg.norm(res_vals.detach().numpy()))
        print(res_error, 'r')
        assert res_error < 4e-5

        # solver.physics._n = 1
        # init_guess = torch.randn(init_guess.shape, dtype=torch.double)
        sol = solver.solve(
            init_guess, tol=1e-5, verbosity=2, maxiters=20).detach()[:, None]
        split_sols = mesh.split_quantities(sol)

        # print(exact_vel_vals[0][:, 0].numpy())
        # print(split_sols[0][:, 0])

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        # p0 = vel_meshes[0].plot(vel_fun(vel_meshes[0].mesh_pts).numpy()[:, :1], 10, ax=axs[0])
        # p1 = vel_meshes[0].plot(split_sols[0], 10, ax=axs[1])
        # plt.colorbar(p0, ax=axs[0])
        # plt.colorbar(p1, ax=axs[1])
        # # plt.plot(vel_meshes[0].mesh_pts[0, :], vel_meshes[0].mesh_pts[1, :], 'ko')
        # plt.show()

        for exact_v, v in zip(exact_vel_vals, split_sols):
            # print(np.linalg.norm(exact_v[:, 0].numpy()-v[:, 0]))
            # print(exact_v[:, 0].numpy())
            # print(v[:, 0])
            assert np.allclose(exact_v[:, 0], v[:, 0])

    def test_first_order_stokes_ice_solver_mms(self):
        # Avoid velocity=0 in any part of the domain
        L, s0, H, alpha, beta, n, rho, g, A, order = (
            50, 2, 1, 4e-5, 1, 3, 910, 9.8, 1e-4, 30)
            # 50, 2, 1, 4e-5, 1, 1, 910, 9.8, 1e-4, 10)
        # L, s0, H, alpha, beta, n, rho, g, A = 1, 1/25, 1/50, 1, 1, 3, 1, 1, 1
        s = f"{s0}-{alpha}*x**2"
        dsdx = f"(-2*{alpha}*x)"
        vel_string = (
            f"2*{A}*({rho}*{g})**{n}/({n}+1)" +
            f"*((({s})-z)**({n}+1)-{H}**({n}+1))" +
            f"*{dsdx}**({n}-1)*{dsdx}-{rho}*{g}*{H}*{dsdx}/{beta}")
        test_cases = [
            [[order, order], [vel_string], f"{H}", f"{s}-{H}",
             f"{beta}", A, rho, g, alpha, n, 50],
        ]
        # may need to setup backtracking for Newtons method
        for test_case in test_cases:
            self._check_first_order_stokes_ice_solver_mms(*test_case)

    def _check_transient_multi_species_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_strings,
            diff_strings, vel_strings, react_funs, react_jacs, bndry_types,
            tableau_name):
        (sol_fun_1, diff_fun_1, vel_fun_1, forc_fun_1, flux_funs_1,
         sol_fun_2, diff_fun_2, vel_fun_2, forc_fun_2, flux_funs_2) = (
            setup_two_species_advection_diffusion_reaction_manufactured_solution(
                sol_strings[0], diff_strings[0], vel_strings[0], react_funs[0],
                sol_strings[1], diff_strings[1], vel_strings[1], react_funs[1],
                True))

        diff_funs = [Function(diff_fun_1), Function(diff_fun_2)]
        vel_funs = [Function(vel_fun_1), Function(vel_fun_2)]
        forc_funs = [
            TransientFunction(forc_fun_1, name='forcing_1'),
            TransientFunction(forc_fun_2, name='forcing_2')]
        sol_funs = [TransientFunction(sol_fun_1, name='sol_1'),
                    TransientFunction(sol_fun_2, name='sol_2')]
        flux_funs = [TransientFunction(flux_funs_1, name='flux_1'),
                     TransientFunction(flux_funs_2, name='flux_2')]

        nphys_vars = len(orders)
        bndry_conds = [_get_boundary_funs(
            nphys_vars, bndry_types[ii], sol_funs[ii], flux_funs[ii])
                       for ii in range(len(sol_funs))]

        deltat = 0.1
        final_time = deltat*3  # 5
        mesh = VectorMesh(
            [CartesianProductCollocationMesh(domain_bounds, orders)]*2)
        solver = TransientPDE(
            MultiSpeciesAdvectionDiffusionReaction(
                mesh, bndry_conds, diff_funs, vel_funs, react_funs, forc_funs,
                react_jacs), deltat, tableau_name)

        for sol_fun in sol_funs:
            sol_fun.set_time(0)
        for forc_fun in forc_funs:
            forc_fun.set_time(0)
        exact_sol = torch.cat(
            [sol_fun(mesh.mesh_pts)
             for sol_fun, mesh in zip(sol_funs, mesh._meshes)])[:, 0]
        # print(exact_sol.shape)
        j_man = solver.physics._raw_residual(
            exact_sol)[1].numpy()
        j_auto = torch.autograd.functional.jacobian(
            lambda s: solver.physics._raw_residual(s)[0],
            exact_sol.clone().requires_grad_(True), strict=True).numpy()

        import numpy as np
        # np.set_printoptions(precision=2, suppress=True, linewidth=500)
        # print('j_man', j_man)
        # print('j_auto', j_auto)
        assert np.allclose(j_man, j_auto)

        for sol_fun in sol_funs:
            sol_fun.set_time(0)
        init_sol = torch.cat(
            [sol_fun(mesh.mesh_pts)
             for sol_fun, mesh in zip(sol_funs, mesh._meshes)])
        sols, times = solver.solve(
            init_sol, 0, final_time, newton_kwargs={"tol": 1e-8})

        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sols_t = [
                sol_fun(mesh.mesh_pts).numpy()
                for sol_fun, mesh in zip(sol_funs, mesh._meshes)]
            model_sols_t = [
                s.numpy() for s in mesh.split_quantities(sols[:, ii:ii+1])]
            # print(exact_sol_t)
            # print(model_sol_t, 'm')
            for jj in range(len(model_sols_t)):
                # print(exact_sols_t[jj].shape, model_sols_t[jj].shape)
                L2_error = np.sqrt(
                    mesh._meshes[jj].integrate(
                        (exact_sols_t[jj]-model_sols_t[jj])**2))
                factor = np.sqrt(
                    mesh._meshes[jj].integrate(exact_sols_t[jj]**2))
                print(time, L2_error.item(), 1e-8*factor.item())
                assert L2_error < 1e-8*factor

    def test_transient_multi_species_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [4], ["0.5*(x-3)*x", "x**3+1"], ["1", "2"],
             [["0"], ["0"]],
             [lambda sol: sol[0]**2*sol[1], lambda sol: -sol[0]**2*sol[1]],
             [lambda sol: [2*sol[0]*sol[1], sol[0]**2],
              lambda sol: [-2*sol[0]*sol[1], -sol[0]**2]],
             [["D", "D"], ["D", "D"]], "im_beuler1"],
        ]
        for test_case in test_cases:
            self._check_transient_multi_species_advection_diffusion_reaction(
                *test_case)

    def _check_linear_elasticity(
            self, domain_bounds, orders, disp_strings, lambda_string, mu_string,
            body_forc_strings, bndry_types, basis_types):
        disp_fun, lambda_fun, mu_fun, forc_fun, flux_funs = (
            setup_linear_elasticity_manufactured_solution(
                disp_strings, lambda_string, mu_string, body_forc_strings))

        disp_fun = Function(disp_fun)
        lambda_fun = Function(lambda_fun)
        mu_fun = Function(mu_fun)
        forc_fun = Function(forc_fun)

        nphys_vars = len(orders)
        bndry_conds = [_get_boundary_funs(
            nphys_vars, bndry_types,
            partial(_vel_component_fun, disp_fun, ii),
            flux_funs[ii]) for ii in range(nphys_vars)]

        disp_meshes = [
            CartesianProductCollocationMesh(domain_bounds, orders)]*nphys_vars
        mesh = VectorMesh(disp_meshes)

        solver = SteadyStatePDE(LinearElasticity(
            mesh, bndry_conds, forc_fun, lambda_fun, mu_fun))

        # change N (really R) BCs to custom
        for ii in range(nphys_vars):
            for jj in range(len(mesh._meshes[0]._bndrys)):
                res_bcs = solver.physics._bndry_conds[ii][jj]
                if res_bcs[1] == "R":
                    # res_bcs[0] is already the correct flux function
                    res_bcs[1] = "C"
                    res_bcs[2] = partial(solver.physics._traction_bcs, ii)

        exact_disp_vals = [
            v[:, None] for v in disp_fun(disp_meshes[0].mesh_pts).T]
        init_guess = torch.cat(exact_disp_vals)
        res_vals = solver.physics._raw_residual(init_guess.squeeze())[0]
        res_error = (np.linalg.norm(res_vals.detach().numpy()) /
                     np.linalg.norm(solver.physics._forc_vals[:, 0].numpy()))
        print(res_error, 'r')
        assert res_error < 1e-12

        sol = solver.solve().detach().numpy()

        split_sols = mesh.split_quantities(sol)
        exact_disp_vals = disp_fun(disp_meshes[0].mesh_pts).numpy()
        # import matplotlib.pyplot as plt
        # mesh.plot(split_sols)
        # mesh.plot([v for v in exact_disp_vals.T])
        # plt.show()
        for exact_disp, disp in zip(exact_disp_vals.T, split_sols):
            print(np.abs(exact_disp - disp).max())
            assert np.allclose(exact_disp, disp)

    def test_linear_elasticity(self):
        test_cases = [
            [[0, 1, 0, 1], [4, 4], ["y**2 * x**2", "x**3 * y**4"], "1", "1",
             ["0", "0"], ["D", "D", "D", "D"], ["C", "C"]],
            [[0, 1, 0, 1], [4, 4], ["y**2 * x**2", "x**3 * y**4"], "1", "1",
             ["0", "0"], ["D", "D", "N", "N"], ["C", "C"]]
        ]
        for test_case in test_cases:
            self._check_linear_elasticity(*test_case)


if __name__ == "__main__":
    auto_pde_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestAutoPDE)
    unittest.TextTestRunner(verbosity=2).run(auto_pde_test_suite)
