import unittest
import string
import numpy as np
from functools import partial
import sympy as sp
import torch

from pyapprox.pde.hdg.pde_coupling import (
    OneDDomainDecomposition, RectangularDomainDecomposition,
    ElbowDomainDecomposition, TurbineDomainDecomposition,
    SteadyStateDomainDecompositionSolver, TransientDomainDecompositionSolver)
from pyapprox.pde.autopde.physics import AdvectionDiffusionReaction
from pyapprox.pde.autopde.solvers import SteadyStatePDE, TransientPDE
from pyapprox.pde.autopde.mesh import (
    CartesianProductCollocationMesh, TransformedCollocationMesh)
from pyapprox.pde.autopde.tests.test_autopde import _get_boundary_funs
from pyapprox.pde.autopde.solvers import (
    Function, TransientFunction)

from pyapprox.util.utilities import cartesian_product, approx_jacobian
from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution)


def init_steady_state_subdomain_model(
        diff_fun, forc_fun, vel_fun, orders, nphys_vars, exact_sol,
        mesh_transform, subdomain_id, vary_orders=False):
    if isinstance(orders, (float, int)):
        orders = np.full(nphys_vars, orders)
    assert len(orders) == nphys_vars
    if vary_orders:
        orders += subdomain_id
    mesh = TransformedCollocationMesh(orders, mesh_transform)
    react_funs = [lambda sol: 0*sol,
                  lambda sol: torch.zeros((sol.shape[0], ))]
    bndry_conds = [
        [lambda x: torch.as_tensor(exact_sol(x)), "D"]
        for dd in range(nphys_vars*2)]
    # bndry_conds = [
    #     [partial(full_fun_axis_1, 0, oned=False), "D"]
    #     for dd in range(nphys_vars*2)]
    solver = SteadyStatePDE(AdvectionDiffusionReaction(
            mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
            react_funs[1]))
    return solver


def init_transient_subdomain_model(
        diff_fun, forc_fun, vel_fun, react_funs, order, deltat, nphys_vars,
        get_bndry_funs, mesh_transform, subdomain_id):
    if isinstance(order, (float, int)):
        orders = np.full(nphys_vars, order)
    else:
        orders = order
    assert len(orders) == nphys_vars
    mesh = TransformedCollocationMesh(
            orders, mesh_transform)
    solver = TransientPDE(AdvectionDiffusionReaction(
            mesh, get_bndry_funs(), diff_fun, vel_fun, react_funs[0], forc_fun,
            react_funs[1]), deltat, "im_beuler1")
    return solver


def get_forcing_for_steady_state_constant_advection_diffusion_2d_sympy(
        sol_string, diffusivity, advection_1, advection_2):
    # from sympy.abc import t as sp_t
    sp_x, sp_y = sp.symbols(['x', 'y'])
    # u = sp.sin(sp.pi*sp_x)*sp.cos(sp_t)
    u = sp.sympify(sol_string)
    kdxu = [diffusivity*u.diff(sp_x, 1), diffusivity*u.diff(sp_y, 1)]
    dxu2 = kdxu[0].diff(sp_x, 1) + kdxu[1].diff(sp_y, 1)  # diffusion
    # dtu = u.diff(sp_t, 1)   # time derivative
    dxu = advection_1*u.diff(sp_x, 1)+advection_2*u.diff(sp_y, 1)  # advection
    # sp_forcing = dtu-(diffusivity*dxu2+advection*dxu)
    sp_forcing = -(dxu2-dxu)
    # print(sp_forcing)
    # forcing_fun = sp.lambdify((sp_x, sp_y, sp_t), sp_forcing, "numpy")
    forcing_fun = sp.lambdify((sp_x, sp_y), sp_forcing, "numpy")
    return forcing_fun


class TestPDECoupling(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_domain_decomp_1d(self):
        """
        solve ((1+delta*x)*u(x)')' + 1 = 0, u(0) = 0, u(1) = 0
        """
        # order = 32
        # delta = 10
        order = 3
        delta = 0
        nsubdomains = 5
        ninterfaces = nsubdomains-1
        mesh = np.linspace(0, 1, nsubdomains+1)[None, :]

        sp_x = sp.symbols('x')
        interface_syms = sp.symbols(list(string.ascii_lowercase))[:ninterfaces]
        sp_u = sp.Function('u')
        sp_diff = (1+delta*sp_x)
        eq = sp.Eq(sp.diff(sp_diff*(sp_u(sp_x).diff(sp_x, 1)))+1, 0)
        interface_mesh = mesh[:, 1:-1]
        bc_syms = [[0, interface_syms[0]]]
        for ii in range(1, nsubdomains-1):
            bc_syms += [[interface_syms[ii-1], interface_syms[ii]]]
        bc_syms += [[interface_syms[ninterfaces-1], 0]]

        bcs = {sp_u(0): 0, sp_u(1): 0}
        exact_sol_sp = sp.dsolve(eq, sp_u(sp_x), ics=bcs).rhs
        exact_sol_np = sp.lambdify((sp_x,), exact_sol_sp, "numpy")
        # print(exact_sol_sp)

        def exact_sol(x):
            return exact_sol_np(x[0, :])[:, None]
        exact_interface_vals = exact_sol(interface_mesh)
        exact_adj_mat = np.zeros((ninterfaces, ninterfaces))
        exact_adj_mat[0, :2] = 1
        for ii in range(1, ninterfaces):
            exact_adj_mat[ii, ii-1:ii+2] = 1

        sols, fluxes, residuals = [], [], []
        for ii in range(nsubdomains):
            bcs = {sp_u(mesh[0, ii]): bc_syms[ii][0],
                   sp_u(mesh[0, ii+1]): bc_syms[ii][1]}
            sols.append(sp.dsolve(eq, sp_u(sp_x), ics=bcs).rhs)
        for ii in range(ninterfaces):
            fluxes.append(
                [(sp_diff*sols[ii].diff(sp_x)).subs(
                    sp_x, interface_mesh[0, ii]),
                 -(sp_diff*sols[ii+1].diff(sp_x)).subs(
                     sp_x, interface_mesh[0, ii])])
            residuals.append(fluxes[-1][0]+fluxes[-1][1])
        sp_jac = np.zeros((ninterfaces, ninterfaces))
        for ii in range(ninterfaces):
            res = residuals[ii]
            JJ = np.where(exact_adj_mat[ii] != 0)[0]
            subs_tuples = [
                (interface_syms[jj], exact_interface_vals[jj, 0]) for jj in JJ]
            # print([fluxes[ii][jj].subs(subs_tuples) for jj in [0, 1]])
            res_norm = res.subs(subs_tuples)
            assert np.absolute(res_norm) < 2e-14
            for jj in range(ninterfaces):
                sp_jac[ii, jj] = res.diff(interface_syms[jj])

        # boundary conditions cannot be passed into partial
        # as only shallow copy will be made and so when domain decomp
        # sets interpolation on some boundaries it inadvertently effects
        # others
        init_subdomain_model = partial(
            init_steady_state_subdomain_model,
            lambda x: torch.as_tensor(x.T*delta+1),
            lambda x: torch.as_tensor(x.T*0+1),
            lambda x: torch.as_tensor(x.T*0), order,
            1, exact_sol)

        domain_decomp = OneDDomainDecomposition([0, 1], nsubdomains, 1)
        solver = SteadyStateDomainDecompositionSolver(domain_decomp)
        domain_decomp.init_subdomains(init_subdomain_model)
        # print(domain_decomp.get_interface_dof_adjacency_matrix())
        # print(exact_adj_mat)
        assert np.allclose(
            domain_decomp.get_interface_dof_adjacency_matrix(), exact_adj_mat)

        interface_mesh = domain_decomp.interface_mesh()
        exact_dirichlet_vals = exact_sol(interface_mesh)
        # residual, subdomain_fluxes = domain_decomp._evaluate_flux_residuals(
        #     exact_dirichlet_vals, return_subdomain_fluxes=True)
        residual = domain_decomp._assemble_dirichlet_neumann_map_jacobian(
            exact_dirichlet_vals)[0]
        # print(np.abs(residual).max())
        assert np.absolute(residual).max() < 2e-11

        # dirichlet_vals = np.ones((domain_decomp.get_ninterfaces_dof(), 1))
        # jac_fd = domain_decomp._approx_jacobian_finite_difference(
        #     exact_dirichlet_vals, epsilon=1)
        jac_fd = approx_jacobian(
            lambda d: domain_decomp._assemble_dirichlet_neumann_map_jacobian(d)[0],
            exact_dirichlet_vals)
        # print('sp_jac\n', sp_jac)
        # print('jac_fd\n', jac_fd.dtype)
        assert np.allclose(jac_fd, sp_jac)

        jac = domain_decomp._assemble_dirichlet_neumann_map_jacobian(
            exact_dirichlet_vals)[1]
        # print(jac, sp_jac)
        assert np.allclose(jac, sp_jac)

        init_dirichlet_vals = np.ones((domain_decomp._ninterfaces, 1))
        # using sp_jac newton conveges in one iteration which it should
        # but because of finite difference errors using finite difference
        # jacobian takes more iterations
        dirichlet_vals = domain_decomp._compute_interface_values(
            init_dirichlet_vals, {})
        assert np.allclose(dirichlet_vals, exact_dirichlet_vals)
        assert np.allclose(dirichlet_vals, exact_sol(interface_mesh))

        xx = np.linspace(0, 1, 21)[None, :]
        solver = SteadyStateDomainDecompositionSolver(
            OneDDomainDecomposition([0, 1], nsubdomains, 1))
        solver._decomp.init_subdomains(init_subdomain_model)
        subdomain_sols = solver.solve()
        interp_values = solver._decomp.interpolate(subdomain_sols, xx)
        assert np.allclose(interp_values, exact_sol(xx))

    def test_domain_decomp_2d(self):

        order = 4# 5
        sol_string = "x**2+y**2"  # "x**2*sin(pi*y)"
        diff_string = "1+x"
        res_tol = 1e-4
        diff_sp = sp.sympify(diff_string)
        sp_forcing_fun = \
            get_forcing_for_steady_state_constant_advection_diffusion_2d_sympy(
                sol_string, diff_sp, 1, 0)

        sp_x, sp_y = sp.symbols(['x', 'y'])
        diff_sp_fun = sp.lambdify(sp_x, diff_sp, "numpy")

        def diff_fun(x):
            vals = diff_sp_fun(x[:1, :])
            if type(vals) == np.ndarray:
                return torch.as_tensor(vals.T)
            return torch.full((x.shape[1], 1), vals)

        exact_sol_sp = sp.sympify(sol_string)
        exact_sol_np = sp.lambdify((sp_x, sp_y), exact_sol_sp, "numpy")

        def exact_sol(x):
            return torch.as_tensor(exact_sol_np(x[0], x[1])[:, None])

        def forcing_fun(x):
            return torch.as_tensor(sp_forcing_fun(x[0, :], x[1, :])[:, None])

        vary_orders = True
        # vary_orders = False
        init_subdomain_model = partial(
            init_steady_state_subdomain_model,
            diff_fun, forcing_fun,
            lambda x: torch.hstack(
                (torch.ones((x.shape[1], 1)), torch.zeros((x.shape[1], 1)))),
            order, 2, exact_sol,
            vary_orders=vary_orders)

        domain = [0, 1, 0, 1]
        # nsubdomains_1d = [2, 1]
        nsubdomains_1d = [2, 2]
        # nsubdomains_1d = [4]*2
        ninterface_dof = order-1
        domain_decomp = RectangularDomainDecomposition(
            domain, nsubdomains_1d, ninterface_dof)
        solver = SteadyStateDomainDecompositionSolver(domain_decomp)
        domain_decomp.init_subdomains(init_subdomain_model)

        interface_dof_adj_mat = \
            domain_decomp.get_interface_dof_adjacency_matrix()
        if nsubdomains_1d == [2, 2]:
            exact_subdomain_adj_mat = np.array(
                [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]])
            assert np.allclose(
                domain_decomp.get_subdomain_adjacency_matrix(),
                exact_subdomain_adj_mat)
            # print(domain_decomp._subdomain_to_interface_map)
            # print(domain_decomp.get_interface_dof_adjacency_matrix())
            tmp = np.ones((
                domain_decomp.get_ninterfaces_dof(),
                domain_decomp.get_ninterfaces_dof()))
            lb = 0
            for ii in range(domain_decomp._ninterfaces):
                ub = lb + domain_decomp._interfaces[ii]._ndof
                tmp[lb:ub, lb:ub] = 0.0
                lb = ub
            exact_interface_dof_adj_mat = np.fliplr(tmp)
            assert np.allclose(interface_dof_adj_mat,
                               exact_interface_dof_adj_mat)

        interface_mesh = domain_decomp.interface_mesh()

        # plt.plot(interface_mesh[0], interface_mesh[1], 'o')
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.show()

        np.set_printoptions(linewidth=1000, precision=4, suppress=True)

        exact_dirichlet_vals = exact_sol(interface_mesh)
        residual = domain_decomp._assemble_dirichlet_neumann_map_jacobian(
            exact_dirichlet_vals)[0]
        # print(residual)
        # print(np.absolute(residual.max()))
        assert np.absolute(residual).max() < res_tol

        def fun(jj, dirichlet_vals):
            domain_decomp._set_dirichlet_values(dirichlet_vals)
            bndry_fluxes = domain_decomp._compute_interface_fluxes(jj)[0]
            iface_fluxes = []
            for mm, iface1 in enumerate(
                    domain_decomp._subdomain_to_interface_map[jj]):
                interface1 = domain_decomp._interfaces[iface1]
                II = np.where(
                    domain_decomp._interface_to_bndry_map[
                        iface1][::2] == jj)[0][0]
                bndry_id1 = domain_decomp._interface_to_bndry_map[
                    iface1][2*II+1]
                iface_fluxes.append(interface1._interpolate_from_subdomain(
                    domain_decomp._subdomain_models[jj], bndry_id1,
                    bndry_fluxes[mm])[:, 0])
            # iface_fluxes = bndry_fluxes # hack
            vals = np.hstack(iface_fluxes)
            return vals

        # finite difference has large error for larger orders
        jac_parts = domain_decomp._assemble_dirichlet_neumann_map_jacobian(
            exact_dirichlet_vals, return_jac_parts=True)[2]
        for jj in range(domain_decomp._nsubdomains):
            subdomain_jac_fd = approx_jacobian(
                partial(fun, jj), exact_dirichlet_vals)
            subdomain_jac = np.zeros_like(subdomain_jac_fd)
            # assumes same # dof on each interface
            stride1 = subdomain_jac_fd.shape[0]//len(
                domain_decomp._subdomain_to_interface_map[jj])
            stride2 = domain_decomp._ninterface_dof
            for mm, iface1 in enumerate(
                    domain_decomp._subdomain_to_interface_map[jj]):
                for nn, iface2 in enumerate(
                        domain_decomp._subdomain_to_interface_map[jj]):
                    subdomain_jac[
                        mm*stride1:(mm+1)*stride1,
                        iface2*stride2:(iface2+1)*stride2] = (
                            jac_parts[jj][mm][nn])
            # print(subdomain_jac-subdomain_jac_fd)
            assert np.allclose(subdomain_jac, subdomain_jac_fd, atol=5e-3)

        jac = domain_decomp._assemble_dirichlet_neumann_map_jacobian(
            exact_dirichlet_vals)[1]
        jac_fd = approx_jacobian(
             lambda d: domain_decomp._assemble_dirichlet_neumann_map_jacobian(d)[0],
             exact_dirichlet_vals)
        # print(jac.numpy())
        # print(jac_fd)
        # print(np.abs(jac.numpy()-jac_fd).max())
        assert np.allclose(jac, jac_fd, atol=3.5e-3)

        init_dirichlet_vals = np.ones(
            (domain_decomp._ninterfaces*ninterface_dof, 1))
        dirichlet_vals = domain_decomp._compute_interface_values(
            init_dirichlet_vals, {})

        assert np.allclose(dirichlet_vals, exact_dirichlet_vals)

        domain_decomp._set_dirichlet_values(dirichlet_vals)
        subdomain_sols = [
            model.solve()
            for model in domain_decomp._subdomain_models]
        for (sol, model) in zip(
                subdomain_sols, domain_decomp._subdomain_models):
            # print(np.linalg.norm(
            #     sol-exact_sol(model.physics.mesh.mesh_pts)[:, 0]))
            assert np.allclose(
                sol, exact_sol(model.physics.mesh.mesh_pts)[:, 0])

        # from pyapprox.util.visualization import get_meshgrid_function_data, plt
        # fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
        # subp = domain_decomp.plot(subdomain_sols, ax=axs[0])
        # from pyapprox.util.visualization import get_meshgrid_function_data, plt
        # axs[0].plot(interface_mesh[0, :], interface_mesh[1, :], 'ko')
        # subp = plt.colorbar(subp, ax=axs[0])
        # X, Y, Z = get_meshgrid_function_data(
        #     exact_sol, domain_decomp._bounds, 100, qoi=0)
        # subp = axs[1].contourf(
        #     X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 20))
        # subp = plt.colorbar(subp, ax=axs[1])
        # plt.show()

        xx = cartesian_product([np.linspace(0, 1, 21)]*2)
        subdomain_sols = solver.solve()
        interp_values = domain_decomp.interpolate(subdomain_sols, xx)
        # print(np.linalg.norm(interp_values-exact_sol(xx).numpy()))
        assert np.allclose(interp_values, exact_sol(xx))

    def _check_transient_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string, diff_string, vel_strings,
            react_funs, bndry_types, intervals=None,
            domain_type="rectangular", atol=1e-8):
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
        final_time = deltat*5
        mesh = CartesianProductCollocationMesh(domain_bounds, orders)
        solver = TransientPDE(
            AdvectionDiffusionReaction(
                mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
                react_funs[1]), deltat, "im_beuler1")
        sol_fun.set_time(0)
        init_sol = sol_fun(mesh.mesh_pts)
        if react_funs[0] is None:
            maxiters = 1
        else:
            maxiters = 10
        single_dom_sols, times = solver.solve(
            init_sol, 0, final_time,
            newton_kwargs={"tol": 1e-8, "maxiters": maxiters})

        init_subdomain_model = partial(
            init_transient_subdomain_model,
            diff_fun, forc_fun, vel_fun, react_funs, orders, deltat,
            nphys_vars, partial(_get_boundary_funs, nphys_vars, bndry_types,
                                sol_fun, flux_funs))

        if nphys_vars == 1:
            if intervals is None:
                nsubdomains = 2
            else:
                nsubdomains = len(intervals)-1
            decomp_solver = TransientDomainDecompositionSolver(
                OneDDomainDecomposition(
                    domain_bounds, nsubdomains, 1, intervals))
            domain_decomp = decomp_solver._decomp
            domain_decomp.init_subdomains(init_subdomain_model)
        else:
            # assert np.allclose(orders[0], orders)
            ninterface_dof = np.min(orders)-1
            if domain_type == "rectangular":
                if intervals is None:
                    nsubdomains_1d = [2, 2]
                else:
                    nsubdomains_1d = [len(intv)-1 for intv in intervals]
                decomp_solver = TransientDomainDecompositionSolver(
                    RectangularDomainDecomposition(
                        domain_bounds, nsubdomains_1d, ninterface_dof, intervals))
            elif domain_type == "elbow":
                decomp_solver = TransientDomainDecompositionSolver(
                    ElbowDomainDecomposition(ninterface_dof, intervals))
            elif domain_type == "turbine":
                decomp_solver = TransientDomainDecompositionSolver(
                    TurbineDomainDecomposition(ninterface_dof))
            domain_decomp = decomp_solver._decomp
            domain_decomp.init_subdomains(init_subdomain_model)

        interface_mesh = domain_decomp.interface_mesh()

        # domain_decomp.plot_mesh_grid(plt.subplots(1, 1)[1], color='k')
        # plt.plot(interface_mesh[0], interface_mesh[1], 'o')
        # plt.show()

        sol_fun.set_time(0)
        init_sols = [sol_fun(model.physics.mesh.mesh_pts)
                     for model in domain_decomp._subdomain_models]
        subdomain_sols = init_sols
        for ii in range(1, len(times)):
            # must be before time is updated
            exact_prev_sols = [sol_fun(model.physics.mesh.mesh_pts)
                               for model in domain_decomp._subdomain_models]
            # time should be when boundary conditions are applied
            # we are using backward euler so BCs are applied at t+deltat
            time = times[ii]
            sol_fun.set_time(time)
            exact_dirichlet_vals = sol_fun(interface_mesh)
            domain_decomp._set_dirichlet_values(exact_dirichlet_vals)
            for jj, model in enumerate(domain_decomp._subdomain_models):
                model.physics._set_time(time)
                for bndry_id in range(2*nphys_vars):
                    if model.physics._bndry_conds[bndry_id][1] != "D":
                        continue
                    idx = model.physics.mesh._bndry_indices[bndry_id]
                    bndry_pts = model.physics.mesh.mesh_pts[:, idx]
                    bndry_vals = model.physics._bndry_conds[bndry_id][0](
                        bndry_pts)
                    assert np.allclose(bndry_vals, sol_fun(bndry_pts))

            subdomain_sols = [
                model.solve(prev_sol, times[ii-1], time)[0][:, -1:]
                for prev_sol, model in zip(subdomain_sols,
                                           domain_decomp._subdomain_models)]

            # ax = plt.subplots(1, 1)[1]
            # domain_decomp.plot(subdomain_sols, 51, ax, color='k')
            # plt.show()
            # exact_subdomain_sols = [
            #     sol_fun(model.physics.mesh.mesh_pts)
            #     for model in domain_decomp._subdomain_models]
            # domain_decomp.plot(exact_subdomain_sols, 51, ax)
            # if nphys_vars == 2:
            #     plt.plot(interface_mesh[0], interface_mesh[1], 'ko')
            # for model_jj in domain_decomp._subdomain_models:
            #     mesh = model_jj.physics.mesh
            #     for ii in range(4):
            #         idx = mesh._bndry_indices[ii]
            #         eps = 0.05
            #         normals = mesh._bndrys[ii].normals(mesh.mesh_pts[:, idx])*eps
            #         for kk in range(normals.shape[0]):
            #             plt.arrow(*mesh.mesh_pts[:, idx][:, kk], *normals[kk])
            # plt.show()

            for (sol, model) in zip(
                    subdomain_sols, domain_decomp._subdomain_models):
                # print(sol, sol_fun(model.physics.mesh.mesh_pts))
                # print(np.abs(
                #     sol-sol_fun(model.physics.mesh.mesh_pts)).max())
                assert np.allclose(
                    sol, sol_fun(model.physics.mesh.mesh_pts), atol=atol)

            decomp_solver._data = [
                exact_prev_sols, time-deltat, deltat, False, False]
            decomp_solver._decomp._solve_subdomain = (
                decomp_solver._solve_subdomain_expanded)

            residual, jac = (
                domain_decomp._assemble_dirichlet_neumann_map_jacobian(
                    exact_dirichlet_vals))
            # decomp_solver._decomp.plot_mesh_grid(plt)
            # im = decomp_solver._decomp.plot(subdomain_sols, 51, plt, eps=1e-1)
            # plt.colorbar(im[0])
            # plt.show()
            # print('res', residual, atol)
            assert np.allclose(residual, 0, atol=atol)

        npts_1d = 21
        if nphys_vars == 1:
            xx = cartesian_product([
                np.linspace(*domain_bounds[2*ii:2*(ii+1)], npts_1d)
                for ii in range(nphys_vars)])
        else:
            xx = np.hstack(
                [model.physics.mesh._create_plot_mesh_2d(npts_1d)[2]
                 for model in domain_decomp._subdomain_models])
        subdomain_sols, times = decomp_solver.solve(
            init_sols, 0, final_time, deltat, verbosity=0,
            subdomain_newton_kwargs={"verbosity": 0, "tol": 1e-8, "rtol": 1e-8,
                                     "maxiters": maxiters},
            macro_newton_kwargs={"verbosity": 0, "tol": 1e-8, "rtol": 1e-8,
                                 "maxiters": maxiters})
        for ii, time in enumerate(times):
            sol_fun.set_time(time)

            # diffs = [sol_fun(model.physics.mesh.mesh_pts) -
            #      subdomain_sols[ii][jj]
            #      for jj, model in enumerate(
            #              domain_decomp._subdomain_models)]
            # # print(diffs)
            # errors = [np.abs(d).max() for d in diffs]
            # print(errors)
            # if (np.max(np.array([np.abs(d).max() for d in diffs]) - np.array(
            #         [np.abs(d).min() for d in diffs])) > 0):
            #     # will not work when error is constant everwhere

            #     interface = domain_decomp._interfaces[0]
            #     # domain_id, bndry_id = 0, 2
            #     domain_id, bndry_id = 1, 2
            #     mesh = domain_decomp._subdomain_models[domain_id].physics.mesh
            #     idx = mesh._bndry_indices[bndry_id]
            #     bndry_pts = mesh.mesh_pts[:, idx]
            #     print(domain_decomp._interface_to_bndry_map)
            #     print(mesh._transform._transforms[0]._ranges)
            #     bndry_normals = mesh._transform.normal(bndry_id, bndry_pts)
            #     plt.plot(bndry_pts[0], bndry_pts[1], 'x')
            #     plt.plot(mesh.mesh_pts[0], mesh.mesh_pts[1], 'o')
            #     for kk in range(bndry_pts.shape[1]):
            #         plt.arrow(*bndry_pts[:, kk], *bndry_normals[kk])
            #     # plt.figure()
            #     #can_bndry_pts = interface._left_transform.map_to_orthogonal(
            #     #    bndry_pts)[interface._left_active_dim]
            #     # plt.plot(can_bndry_pts, subdomain_sols[ii][0][idx], '-o')
            #     # plt.plot(can_bndry_pts, sol_fun(bndry_pts), '--')
            #     # plt.plot(can_bndry_pts, subdomain_sols[ii][0][idx]-sol_fun(bndry_pts))


            #     axs = plt.subplots(1, 3, sharey=True, figsize=(3*8, 6))[1]
            #     full_exact_sol = sol_fun(solver.physics.mesh.mesh_pts)
            #     levels = np.linspace(
            #         full_exact_sol.min(), full_exact_sol.max(), 21)
            #     levels = 21
            #     domain_decomp.plot(
            #         [sol_fun(model.physics.mesh.mesh_pts)
            #          for jj, model in enumerate(domain_decomp._subdomain_models)],
            #         51, axs[0], levels=levels)
            #     domain_decomp.plot(
            #         subdomain_sols[ii], 51, axs[1], levels=levels)
            #     ims = domain_decomp.plot(diffs, 51, axs[2], levels=21)
            #     plt.plot(interface_mesh[0], interface_mesh[1], 'o')
            #     plt.colorbar(ims[-1], ax=axs[2])
            #     plt.show()
            # mesh_pts = domain_decomp.mesh_points()
            # plt.plot(mesh_pts[0], mesh_pts[1], 'o')
            # domain_decomp.plot_mesh_grid(plt, c="k")
            # plt.plot(interface_mesh[0], interface_mesh[1], 'X')

            for (sol, model) in zip(
                    subdomain_sols[ii], domain_decomp._subdomain_models):
                # print(np.linalg.norm(sol), np.linalg.norm(
                #     sol_fun(model.physics.mesh.mesh_pts)))
                # print(ii, "time", time, ":", np.abs(
                #     sol-sol_fun(model.physics.mesh.mesh_pts)).max())
                # print(sol.min(), sol.max())
                assert np.allclose(
                    sol, sol_fun(model.physics.mesh.mesh_pts), atol=atol)

            interp_values = domain_decomp.interpolate(subdomain_sols[ii], xx)
            # print(np.abs(interp_values-sol_fun(xx).numpy()).max())
            # if ii > 0:
            #     II = np.where(np.abs(interp_values-sol_fun(xx).numpy())>1e-6)[0]
            #     plt.plot(xx[0, II], xx[1, II], 'o')
            #     plt.show()
            assert np.allclose(interp_values, sol_fun(xx), atol=atol)

            # print(subdomain_sols[-1])
            # ax = plt.figure().gca()
            # im = decomp_solver._decomp.plot(subdomain_sols[ii], 51, ax, eps=1e-2)
            # plt.colorbar(im[0], ax=ax)
            # plt.show()

    def test_transient_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [4], "x**2*(1+t)", "4", ["0"],
             [lambda sol: 0*sol, lambda sol: torch.zeros((sol.shape[0],))],
             ["D", "D"]],
            [[0, 1], [4], "x**2*(1+t)", "4", ["0"],
             [None, None],
             ["D", "D"], np.array([0, 0.2, 0.6, 0.8, 1.])],
            [[0, 1, 0, 1], [4, 4], "(x-1)*x*(1+t)*y**2", "(2+x*y)", ["1", "1"],
             [None, None],
             ["R", "R", "N", "N"]],
            [[0, 1, 0, 1], [4, 4], "(x-1)*x*(1+t)*y**2", "1", ["1", "1"],
             [lambda sol: 1*sol**2, lambda sol: 2*sol[:, 0]],
             ["D", "N", "R", "D"]],
            [[0, 1, 0, 1], [5, 5], "(x-1)*x*(1+t)*y**2", "1", ["1", "1"],
             [lambda sol: sol**2, lambda sol: 2*sol[:, 0]],
             ["D", "N", "R", "D"], [np.array([0, 0.3, 1])]*2, "rectangular", 2e-8],
            [[0, 1, 0, 1], [4, 4], "(x-1)*x*(1+t)*y**2", "1", ["1", "1"],
             [lambda sol: sol**2, lambda sol: 2*sol[:, 0]],
             ["D", "D", "D", "D"], np.array([0, 0.3, 1, 0., 0.3, 1.]),
             "elbow"],
            [[0, 1, 0, 1], [12, 12],
             # "x**2*y**2*(t+1)", "1", ["1", "1"],
             "(x+y)**2*(t+1)", "1", ["0", "0"],
             [None, None],
             # [lambda sol: 0*sol**2, lambda sol: 0*2*sol[:, 0]],
             ["D", "D", "D", "D"], None, "turbine", 1e-5]
        ]
        ii = 0
        for test_case in test_cases:
            # print(ii)
            # print(test_case)
            np.random.seed(1)  # controls direction of finite difference
            self._check_transient_advection_diffusion_reaction(*test_case)
            ii += 1

    def _check_integrate(self, decomp, orders):
        nvars = len(orders)
        print(orders)
        def fun(xx):
            return ((xx**2).sum(axis=0))[:, None]
        
        init_subdomain_model = partial(
            init_steady_state_subdomain_model,
            lambda x: torch.as_tensor(x[:1].T*0+1),
            lambda x: torch.as_tensor(x[:1].T*0+1),
            lambda x: torch.hstack(
                [torch.zeros((x.shape[1], 1))]*nvars),
            orders, len(orders), lambda x: x[:1].T*0)
        decomp.init_subdomains(init_subdomain_model)

        subdomain_vals = [fun(m.physics.mesh.mesh_pts) for m in
                          decomp._subdomain_models]
        integral = decomp.integrate(subdomain_vals)
        bb = np.reshape(decomp._bounds, (nvars, 2))
        if nvars == 1:
            true_integral = (bb[0][1]**3/3-bb[0][0]**3/3)
        else:
            true_integral = (
                (bb[0][1]**3/3-bb[0][0]**3/3)*(bb[1][1]-bb[1][0]) +
                (bb[1][1]**3/3-bb[1][0]**3/3)*(bb[0][1]-bb[0][0]))
        assert np.allclose(true_integral, integral)

    def test_integrate(self):
        orders = [5, 5]
        test_cases = [
            [OneDDomainDecomposition([0, 2], 3, 1, None), [5]],
            [RectangularDomainDecomposition(
                [0, 2, 1, 2], [2, 2], np.min(orders)-1, None), orders]
        ]

        for test_case in test_cases:
            self._check_integrate(*test_case)


if __name__ == "__main__":
    pde_coupling_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestPDECoupling)
    unittest.TextTestRunner(verbosity=2).run(pde_coupling_test_suite)


# TODO: use interface_dof_adj_mat to speed up finite difference calculation of jacobian
# It tells what models will be effected when a dof on a specific subdomain is
# perturbed
