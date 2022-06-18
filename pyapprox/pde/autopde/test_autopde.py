import torch
import unittest
import numpy as np
from functools import partial

from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution,
    setup_helmholtz_manufactured_solution,
    setup_steady_stokes_manufactured_solution,
    setup_shallow_wave_equations_manufactured_solution,
    setup_shallow_shelf_manufactured_solution,
    setup_first_order_stokes_ice_manufactured_solution
)
from pyapprox.pde.autopde.autopde import (
    CartesianProductCollocationMesh, AdvectionDiffusionReaction,
    Function, EulerBernoulliBeam, Helmholtz,
    TransientFunction, TransientPDE, LinearStokes, NavierStokes,
    InteriorCartesianProductCollocationMesh, VectorMesh,
    SteadyStatePDE, ShallowWater, ShallowShelfVelocities, ShallowShelf,
    FirstOrderStokesIce
)


# Functions and testing only for wrapping Sympy generated manufactured
# solutions
def _normal_flux(flux_funs, active_var, sign, xx):
    vals = sign*flux_funs(xx)[:, active_var:active_var+1]
    return vals


def _robin_bndry_fun(sol_fun, flux_funs, active_var, sign, alpha, xx,
                     time=None):
    if time is not None:
        if hasattr(sol_fun, "set_time"):
            sol_fun.set_time(time)
        if hasattr(flux_funs, "set_time"):
            flux_funs.set_time(time)
    vals = alpha*sol_fun(xx) + _normal_flux(flux_funs, active_var, sign, xx)
    return vals


def _get_boundary_funs(nphys_vars, bndry_types, sol_fun, flux_funs):
    bndry_conds = []
    for dd in range(2*nphys_vars):
        if bndry_types[dd] == "D":
            bndry_conds.append([sol_fun, "D"])
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
            bndry_fun = partial(_robin_bndry_fun, sol_fun, flux_funs, dd//2,
                                (-1)**(dd+1), alpha)
            if hasattr(sol_fun, "set_time") or hasattr(flux_funs, "set_time"):
                bndry_fun = TransientFunction(bndry_fun)
            bndry_conds.append([bndry_fun, "R", alpha])
    return bndry_conds


def _vel_component_fun(vel_fun, ii, x):
    vals = vel_fun(x)
    return vals[:, ii:ii+1]


class TestAutoPDE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)

    def _check_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string, diff_string, vel_strings,
            react_fun, bndry_types, basis_types):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_fun))

        diff_fun = Function(diff_fun)
        vel_fun = Function(vel_fun)
        forc_fun = Function(forc_fun)
        sol_fun = Function(sol_fun)

        nphys_vars = len(orders)
        bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, sol_fun, flux_funs)

        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, bndry_conds, basis_types)

        solver = SteadyStatePDE(AdvectionDiffusionReaction(
            mesh, diff_fun, vel_fun, react_fun, forc_fun))

        print(solver.residual._raw_residual(sol_fun(mesh.mesh_pts)[:, 0]))
        assert np.allclose(
            solver.residual._raw_residual(sol_fun(mesh.mesh_pts)[:, 0]), 0)
        sol = solver.solve()

        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        # mesh.plot(sol, ax=ax, marker='o', ls='None', c='k', ms=20)
        # mesh.plot(
        #     sol_fun(mesh.mesh_pts).numpy(), ax=ax, marker='s',
        #     ls='None', c='b')
        # # mesh.plot(sol, ax=ax, nplot_pts_1d=100, label="Collocation", c="k')
        # mesh.plot(
        #     sol_fun(mesh.mesh_pts).numpy(), ax=ax, nplot_pts_1d=100,
        #     ls="--", label="Exact", c='b')
        # plt.legend()
        # plt.show()

        print(np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol))
        assert np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol) < 1e-9


        # normals = solver.mesh._get_bndry_normals(np.arange(nphys_vars*2))
        # if nphys_vars == 2:
        #     assert np.allclose(
        #         normals, np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]))
        # else:
        #     assert np.allclose(normals, np.array([[-1], [1]]))
        # _normal_fluxes = solver.compute_bndry_fluxes(
        #     sol, np.arange(nphys_vars*2))
        # for ii, indices in enumerate(solver.mesh.boundary_indices):
        #     assert np.allclose(
        #         np.array(_normal_fluxes[ii]),
        #         flux_funs(solver.mesh.mesh_pts[:, indices],).dot(
        #             normals[ii]))

    def test_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], lambda x: 0*x**2,
             ["D", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], lambda x: 0*x**2,
             ["N", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], lambda x: 0*x**2,
             ["R", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], lambda x: x**2,
             ["D", "D"], ["C"]],
            # When using periodic bcs must have reaction term to have a
            # unique solution
            [[0, 2*np.pi], [30], "sin(x)", "1", ["0"], lambda x: 1*x,
             ["P", "P"], ["C"]],
            # [[0, 2*np.pi], [5], "sin(x)", "1", ["0"], lambda x: 1*x,
            #  ["P", "P"], ["F"]],
            [[0, 1, 0, 1], [4, 4], "y**2*x**2", "1", ["0", "0"],
             lambda x: 0*x**2, ["D", "N", "N", "D"], ["C", "C"]],
            # [[0, .5, 0, 1], [14, 16], "y**2*sin(pi*x)", "1", ["0", "0"],
            #  lambda x: 0*x**2, ["D", "N", "N", "D"], ["C", "C"]],
            # [[0, .5, 0, 1], [16, 16], "y**2*sin(pi*x)", "1", ["0", "0"],
            #  lambda x: 0*x**2, ["D", "R", "D", "D"], ["C", "C"]]
        ] 
        for test_case in test_cases:
            self._check_advection_diffusion_reaction(*test_case)

    def test_euler_bernoulli_beam(self):
        # bndry_conds are None because they are imposed in the solver
        # This 4th order equation requires 4 boundary conditions, two at each
        # end of the domain. This cannot be done with usual boundary condition
        # functions and msut be imposed on the residual exactly
        domain_bounds, orders, bndry_conds = [0, 1], [4], [[None, None]]*2
        emod_val, smom_val, forcing_val = 1., 1., -2.
        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, bndry_conds)
        solver = SteadyStatePDE(EulerBernoulliBeam(
            mesh, Function(lambda x: np.full((x.shape[1], 1), 1)),
            Function(lambda x: np.full((x.shape[1], 1), 1)),
            Function(lambda x: np.full((x.shape[1], 1), forcing_val))))

        def sol_fun(x):
            length = domain_bounds[1]-domain_bounds[0]
            return (forcing_val*x**2*(6*length**2-4*length*x+x**2)/(
                24*emod_val*smom_val)).T

        exact_sol_vals = sol_fun(mesh.mesh_pts)
        assert np.allclose(
            solver.residual._raw_residual(torch.tensor(exact_sol_vals[:, 0])), 0)
        
        sol = solver.solve()

        assert np.allclose(sol, exact_sol_vals)
        # mesh.plot(sol, nplot_pts_1d=100)
        # import matplotlib.pyplot as plt
        # plt.show()

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
            domain_bounds, orders, bndry_conds)
        solver = SteadyStatePDE(Helmholtz(mesh, wnum_fun, forc_fun))
        sol = solver.solve()

        print(np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol))
        assert np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol) < 1e-9

    def test_helmholtz(self):
        test_cases = [
            [[0, 1], [16], "x**2", "1", ["N", "D"]],
            [[0, .5, 0, 1], [16, 16], "y**2*x**2", "1", ["N", "D", "D", "D"]]]
        for test_case in test_cases:
            self._check_helmholtz(*test_case)

    def _check_transient_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string,
            diff_string, vel_strings, react_fun, bndry_types,
            tableau_name):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_fun, True))

        diff_fun = Function(diff_fun)
        vel_fun = Function(vel_fun)
        forc_fun = TransientFunction(forc_fun, name='forcing')
        sol_fun = TransientFunction(sol_fun, name='sol')
        flux_funs = TransientFunction(flux_funs, name='flux')

        nphys_vars = len(orders)
        bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, sol_fun, flux_funs)

        deltat = 0.1
        final_time = deltat*5
        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, bndry_conds)
        solver = TransientPDE(
            AdvectionDiffusionReaction(
                mesh, diff_fun, vel_fun, react_fun, forc_fun),
            deltat, tableau_name)
        sol_fun.set_time(0)
        sols, times = solver.solve(
            sol_fun(mesh.mesh_pts), 0, final_time, newton_opts={"tol": 1e-8})

        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sol_t = sol_fun(solver.residual.mesh.mesh_pts).numpy()
            model_sol_t = sols[:, ii:ii+1]
            L2_error = np.sqrt(
                solver.residual.mesh.integrate((exact_sol_t-model_sol_t)**2))
            factor = np.sqrt(
                solver.residual.mesh.integrate(exact_sol_t**2))
            print(time, L2_error, 1e-8*factor)
            assert L2_error < 1e-8*factor


    def test_transient_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["0"],
             lambda x: 0*x**2, ["D", "D"], "im_crank2"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["1"],
            lambda x: 1*x**2, ["D", "D"], "im_crank2"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["1"],
             lambda x: 1*x**2, ["N", "D"], "im_crank2"],
            [[0, 1, 0, 1], [3, 3], "(x-1)*x*(1+t)**2*y**2", "1", ["1", "1"],
            lambda x: 1*x**2, ["D", "N", "R", "D"], "im_crank2"]
        ]
        for test_case in test_cases:
            self._check_transient_advection_diffusion_reaction(*test_case)

    def _check_stokes_solver_mms(
            self, domain_bounds, orders, vel_strings, pres_string, bndry_types,
            navier_stokes):
        vel_fun, pres_fun, vel_forc_fun, pres_forc_fun, pres_grad_fun = (
            setup_steady_stokes_manufactured_solution(
                vel_strings, pres_string, navier_stokes))

        vel_fun = Function(vel_fun)
        pres_fun = Function(pres_fun)
        vel_forc_fun = Function(vel_forc_fun)
        pres_forc_fun = Function(pres_forc_fun)
        pres_grad_fun = Function(pres_grad_fun)

        # TODO Curently not test stokes with Neumann Boundary conditions

        flux_funs = None
        nphys_vars = len(orders)
        vel_bndry_conds = [
            _get_boundary_funs(
                nphys_vars, bndry_types,
                partial(_vel_component_fun, vel_fun, ii),
                flux_funs) for ii in range(nphys_vars)]

        vel_meshes = [CartesianProductCollocationMesh(
            domain_bounds, orders, vel_bndry_conds[ii])
                      for ii in range(nphys_vars)]
        pres_mesh = InteriorCartesianProductCollocationMesh(
            domain_bounds, orders)
        mesh = VectorMesh(vel_meshes + [pres_mesh])
        pres_idx = 0
        pres_val = pres_fun(pres_mesh.mesh_pts[:, pres_idx:pres_idx+1])
        if not navier_stokes:
            Residual = LinearStokes
        else:
            Residual = NavierStokes
        solver = SteadyStatePDE(Residual(
            mesh, vel_forc_fun, pres_forc_fun, (pres_idx, pres_val)))
        sol = solver.solve()

        exact_vel_vals = vel_fun(vel_meshes[0].mesh_pts).numpy()
        exact_pres_vals = pres_fun(pres_mesh.mesh_pts).numpy()

        split_sols = mesh.split_quantities(sol)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(8*3, 6))
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
            assert np.allclose(exact_pres_vals, split_sols[-1], atol=6e-8)

    def test_stokes_solver_mms(self):
        test_cases = [
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["D", "D"], False],
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["D", "D"], True],
            [[0, 1, 0, 1], [20, 20],
             ["-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"], "x**3*y**3",
             ["D", "D", "D", "D"], False],
            [[0, 1, 0, 1], [6, 7],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "D"], False],
            [[0, 1, 0, 1], [12, 12],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "D"], True]
        ]
        for test_case in test_cases:
            self._check_stokes_solver_mms(*test_case)

    def _check_shallow_water_solver_mms(
            self, domain_bounds, orders, vel_strings, depth_string, bed_string,
            bndry_types):
        nphys_vars = len(vel_strings)
        depth_fun, vel_fun, depth_forc_fun, vel_forc_fun, bed_fun = (
            setup_shallow_wave_equations_manufactured_solution(
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
            partial(_vel_component_fun, vel_fun, ii),
            flux_funs) for ii in range(nphys_vars)]

        depth_mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, depth_bndry_conds)
        vel_meshes = [CartesianProductCollocationMesh(
            domain_bounds, orders, vel_bndry_conds[ii])
                      for ii in range(nphys_vars)]
        mesh = VectorMesh([depth_mesh]+vel_meshes)

        solver = SteadyStatePDE(
            ShallowWater(mesh, depth_forc_fun, vel_forc_fun, bed_fun))
        exact_depth_vals = depth_fun(depth_mesh.mesh_pts)
        exact_vel_vals = [v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts).T]
        # split_sols = [q1, q2] = [h, u, v]
        init_guess = torch.cat([exact_depth_vals] + exact_vel_vals)
        # split_sols = [q1, q2] = [h, uh, vh]
        # init_guess = torch.cat(
        #     [exact_depth_vals] +[v*depth_fun(vel_meshes[0].mesh_pts)
        #                    for v in exact_vel_vals])

        # print(init_guess, 'i')
        res_vals = solver.residual._raw_residual(init_guess.squeeze())
        assert np.allclose(res_vals, 0)

        init_guess = init_guess+torch.randn(init_guess.shape)*1e-3
        sol = solver.solve(init_guess, tol=1e-8)
        split_sols = mesh.split_quantities(sol)
        assert np.allclose(exact_depth_vals, split_sols[0])
        for exact_v, v in zip(exact_vel_vals, split_sols[1:]):
            print(exact_v[:, 0]-v[:, 0])
            assert np.allclose(exact_v[:, 0], v[:, 0])

    def test_shallow_water_solver_mms_setup(self):
        # or Analytical solutions see
        # https://hal.archives-ouvertes.fr/hal-00628246v6/document
        def bernoulli_realtion(q, bed_fun, C, x, h):
            return q**2/(2*9.81*h**2)+h+bed_fun(x)-C
        x = torch.linspace(0, 1, 11,dtype=torch.double)
        from util import newton_solve
        def bed_fun(x):
            return -x**2*0
        q, C = 1, 1
        init_guess = C-bed_fun(x).requires_grad_(True)
        fun = partial(bernoulli_realtion, q, bed_fun, C, x)
        sol = newton_solve(fun, init_guess, tol=1e-12, verbosity=2, maxiters=20)
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
            setup_shallow_wave_equations_manufactured_solution(
                vel_strings, depth_string, bed_string))
        xx = torch.linspace(0, 1, 11)[None, :]
        assert np.allclose(depth_forc_fun(xx), 0, atol=1e-12)
        assert np.allclose(vel_forc_fun(xx), 0, atol=1e-12)

    def test_shallow_water_solver_mms(self):
        # order must be odd or Jacobian will be almost uninvertable and
        # newton solve will diverge
        test_cases = [
            [[0, 1], [5], ["-x**2"], "1+x", "0", ["D", "D"]],
            [[0, 1, 0, 1], [5, 5], ["-x**2", "-y**2"], "1+x+y", "0",
             ["D", "D", "D", "D"]]
        ]
        for test_case in test_cases:
            self._check_shallow_water_solver_mms(*test_case)

    def _check_shallow_water_transient_solver_mms(
            self, domain_bounds, orders, vel_strings, depth_string, bed_string,
            bndry_types, tableau_name):
        nphys_vars = len(vel_strings)
        depth_fun, vel_fun, depth_forc_fun, vel_forc_fun, bed_fun = (
            setup_shallow_wave_equations_manufactured_solution(
                vel_strings, depth_string, bed_string, True))

        bed_fun = Function(bed_fun)

        depth_forc_fun = TransientFunction(depth_forc_fun)
        vel_forc_fun = TransientFunction(vel_forc_fun)
        depth_fun = TransientFunction(depth_fun)
        vel_fun = TransientFunction(vel_fun)

        # TODO test neumann boundary conditions so need flux funs
        # returned by MMS
        flux_funs = None
        depth_bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, depth_fun, flux_funs)
        vel_bndry_conds = [_get_boundary_funs(
            nphys_vars, bndry_types,
            partial(_vel_component_fun, vel_fun, ii),
            flux_funs) for ii in range(nphys_vars)]

        depth_mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, depth_bndry_conds)
        vel_meshes = [CartesianProductCollocationMesh(
            domain_bounds, orders, vel_bndry_conds[ii])
                      for ii in range(nphys_vars)]
        mesh = VectorMesh([depth_mesh]+vel_meshes)

        depth_fun.set_time(0)
        vel_fun.set_time(0)
        depth_forc_fun.set_time(0)
        vel_forc_fun.set_time(0)

        deltat = 0.05
        final_time = deltat
        solver = TransientPDE(
            ShallowWater(mesh, depth_forc_fun, vel_forc_fun, bed_fun), deltat,
            tableau_name)
        init_sol = torch.cat(
            [depth_fun(depth_mesh.mesh_pts)] +
            [v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts).T])
        sols, times = solver.solve(
            init_sol, 0, final_time, newton_opts={"tol": 1e-8})

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(
        #     1, mesh.nphys_vars+1, figsize=(8*(mesh.nphys_vars+1), 6))
        for ii, time in enumerate(times):
            depth_fun.set_time(time)
            vel_fun.set_time(time)
            exact_sol_t = np.vstack([
                depth_fun(depth_mesh.mesh_pts).numpy()]+
                                    [v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts)])
            model_sol_t = sols[:, ii:ii+1]
            print(mesh.split_quantities(
                exact_sol_t)[0][[0, -1]],
                  mesh.split_quantities(model_sol_t)[0][[0, -1]])
            # mesh.plot(mesh.split_quantities(exact_sol_t), axs=axs)
            # mesh.plot(mesh.split_quantities(model_sol_t), axs=axs, ls='--')
            L2_error = np.sqrt(
                mesh.integrate(
                    mesh.split_quantities((exact_sol_t-model_sol_t)**2)))
            print(time, L2_error)
            assert np.all(L2_error < 1e-8)
            # plt.show()

    def test_shallow_water_transient_solver_mms(self):
        # order must be odd or Jacobian will be almost uninvertable and
        # newton solve will diverge

        test_cases = [
            [[0, 1], [5], ["-x**2"], "1+x", "0", ["D", "D"], "im_crank2"],
            # [[0, 1, 0, 1], [5, 5], ["-x**2", "-y**2"], "1+x+y", "0",
            #  ["D", "D", "D", "D"], "im_beuler1"]
        ]
        for test_case in test_cases:
            self._check_shallow_water_transient_solver_mms(*test_case)

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

        vel_meshes = [CartesianProductCollocationMesh(
            domain_bounds, orders, vel_bndry_conds[ii])
                      for ii in range(nphys_vars)]
        depth_mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, depth_bndry_conds)
        if velocities_only:
            mesh = VectorMesh(vel_meshes)
        else:
            mesh = VectorMesh(vel_meshes+[depth_mesh])

        exact_vel_vals = [
            v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts).T]
        exact_depth_vals = depth_fun(vel_meshes[0].mesh_pts)
        if velocities_only:
            solver = SteadyStatePDE(
                ShallowShelfVelocities(mesh, vel_forc_fun, bed_fun, beta_fun,
                                       depth_fun, A, rho, 1e-15))
            init_guess = torch.cat(exact_vel_vals)
        else:
            solver = SteadyStatePDE(
                ShallowShelf(mesh, vel_forc_fun, bed_fun, beta_fun,
                             depth_forc_fun, A, rho, 1e-15))
            init_guess = torch.cat(exact_vel_vals+[exact_depth_vals])

        # print(init_guess, 'i')
        res_vals = solver.residual._raw_residual(init_guess.squeeze())
        print(np.abs(res_vals.detach().numpy()).max(), 'r')
        assert np.allclose(res_vals, 0)

        if velocities_only:
            init_guess = torch.randn(init_guess.shape, dtype=torch.double)*0
        else:
            init_guess = (init_guess+torch.randn(init_guess.shape)*5e-3)
        sol = solver.solve(init_guess, tol=1e-7, verbosity=2, maxiters=100)
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
            [[0, 1, 0, 1], [10, 10], ["(x+1)**2", "(y+1)**2"], "1+x+y",
             "1+x+y**2", "1", ["D", "D", "D", "D"], True],
            [[0, 2, 0, 2], [15, 15], ["(x+1)**2", "(y+1)**2"], "1+x+y",
             "0-x-y", "1", ["D", "D", "D", "D"], False]
        ]
        # may need to setup backtracking for Newtons method
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
        print(sp.lambdify(symbs, visc_expr-mu, "numpy")(
                xx, bed_fun(xx[None, :])[:, 0]))
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
        domain_bounds = [-L, L, 0, 1]
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

        # placeholder so that custom boundary conditions can be added
        # after residual is created
        vel_bndry_conds = [[[None, None] for ii in range(nphys_vars*2)]]

        vel_meshes = [CartesianProductCollocationMesh(
            domain_bounds, orders, vel_bndry_conds[ii])
                      for ii in range(nphys_vars-1)]
        mesh = VectorMesh(vel_meshes)

        exact_vel_vals = [
            v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts).T]
        solver = SteadyStatePDE(
            FirstOrderStokesIce(mesh, vel_forc_fun, bed_fun, beta_fun,
                                depth_fun, A, rho, 0))
        solver.residual._n = n
        # for ii in range(len(mesh._meshes[0]._bndry_conds)):
        #     mesh._meshes[0]._bndry_conds[ii] = [
        #         Function(bndry_funs[ii]), "C",
        #         solver.residual._strain_boundary_conditions]
        init_guess = torch.cat(exact_vel_vals)
        res_vals = solver.residual._raw_residual(init_guess.squeeze())
        res_error = (np.linalg.norm(res_vals.detach().numpy()) /
                     np.linalg.norm(solver.residual._forc_vals[:, 0].numpy()))
        print(np.linalg.norm(res_vals.detach().numpy()))
        print(res_error, 'r')
        assert res_error < 4e-5

        # solver.residual._n = 1
        # init_guess = torch.randn(init_guess.shape, dtype=torch.double)
        sol = solver.solve(init_guess, tol=1e-7, verbosity=2, maxiters=20)
        split_sols = mesh.split_quantities(sol)
        for exact_v, v in zip(exact_vel_vals, split_sols):
            # print(exact_v[:, 0]-v[:, 0])
            assert np.allclose(exact_v[:, 0], v[:, 0])

    def xtest_first_order_stokes_ice_solver_mms(self):
        # Avoid velocity=0 in any part of the domain
        L, s0, H, alpha, beta, n, rho, g, A = (
            50, 2, 1, 4e-5, 1, 3, 910, 9.8, 1e-4)
           # 1, 2, 1, 1e-1, 1, 2, 910, 9.81, 1e-4)
        # L, s0, H, alpha, beta, n, rho, g, A = 1, 1/25, 1/50, 1, 1, 3, 1, 1, 1
        s = f"{s0}-{alpha}*x**2"
        dsdx = f"(-2*{alpha}*x)"
        vel_string = (
            f"2*{A}*({rho}*{g})**{n}/({n}+1)" +
            f"*((({s})-z)**({n}+1)-{H}**({n}+1))" +
            f"*{dsdx}**({n}-1)*{dsdx}-{rho}*{g}*{H}*{dsdx}/{beta}")
        test_cases = [
            [[60, 10], [vel_string], f"{H}", f"{s}-{H}",
             f"{beta}", A, rho, g, alpha, n, 50],
        ]
        # may need to setup backtracking for Newtons method
        for test_case in test_cases:
            self._check_first_order_stokes_ice_solver_mms(*test_case)



if __name__ == "__main__":
    autopde_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestAutoPDE)
    unittest.TextTestRunner(verbosity=2).run(autopde_test_suite)
