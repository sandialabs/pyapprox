import unittest
import numpy as np
from functools import partial
from skfem import (ElementVector, Basis, condense, solve, Functional)

from pyapprox.pde.galerkin.util import (
    _get_mesh, _get_element)
from pyapprox.pde.galerkin.physics import (
    _assemble_advection_diffusion_reaction, _assemble_stokes)
from pyapprox.pde.galerkin.solvers import (
    newton_solve, SteadyStatePDE, TransientPDE, TransientFunction)
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction, Helmholtz, Stokes)
from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution,
    setup_steady_stokes_manufactured_solution,
    setup_helmholtz_manufactured_solution)
from pyapprox.pde.autopde.tests.test_autopde import _vel_component_fun


def _normal_flux(flux_funs, normal_fun, xx):
    normal_vals = normal_fun(xx)
    flux_vals = flux_funs(xx)
    vals = np.sum(normal_vals*flux_vals, axis=1)
    return vals


def _robin_bndry_fun(sol_fun, flux_funs, normal_fun, alpha, xx, time=None):
    if time is not None:
        if hasattr(sol_fun, "set_time"):
            sol_fun.set_time(time)
        if hasattr(flux_funs, "set_time"):
            flux_funs.set_time(time)
    vals = alpha*sol_fun(xx) + _normal_flux(flux_funs, normal_fun, xx)
    return vals


def _canonical_normal(bndry_index, samples):
    # different to autopde because samples.ndim==3 here compared to ndim==2
    normal_vals = np.zeros(
        (samples.shape[1], samples.shape[0], samples.shape[2]))
    active_var = int(bndry_index >= 2)
    normal_vals[:, active_var, :] = (-1)**((bndry_index+1) % 2)
    return normal_vals


def _get_mms_boundary_funs(nphys_vars, bndry_types, sol_fun, flux_funs,
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
            bndry_conds.append([bndry_fun, "R", alpha])
        if bndry_conds[-1][0] is not None:
            bndry_conds[-1][0]._name = f"bndry_{dd}"
    return bndry_conds


def _list_to_vector_bndry_cond_fun(bndry_conds, idx, key, xx):
    nvec = len(bndry_conds)
    # bndry_conds[ii] = [D, N, R]
    # idx is index into D, N, or R boundaries
    # key is boundary key
    vals = np.stack([bndry_conds[ii][0][key][0](xx) for ii in range(nvec)])
    return vals


def _bndrys_keys_from_bndry_types(mesh, bndry_types, bndry_type):
    nphys_vars = len(bndry_types)//2
    # orders of keys must correspond to order specified in bndry_types
    # The following returns the keys mesh.boundaries.keys() but not
    # necessarily in the correct order
    keys = ["left", "right"]
    if nphys_vars == 2:
        keys += ["bottom", "top"]
    assert len(keys) == len(bndry_types)
    active_key_indices = [
        ii for ii in range(len(keys)) if bndry_types[ii] == bndry_type]
    return [keys[idx] for idx in active_key_indices], active_key_indices


def _get_bndry_keys_indices_from_types(mesh, bndry_types):
    # get Dirichlet boundary names
    D_bndry_keys, D_indices = _bndrys_keys_from_bndry_types(
        mesh, bndry_types, "D")
    N_bndry_keys, N_indices = _bndrys_keys_from_bndry_types(
        mesh, bndry_types, "N")
    R_bndry_keys, R_indices = _bndrys_keys_from_bndry_types(
        mesh, bndry_types, "R")
    return (D_bndry_keys, D_indices, N_bndry_keys, N_indices,
            R_bndry_keys, R_indices)


class MSBoundaryConditionFunction():
    # Boundary condtion wrapper for manufactured solutions (MS)
    def __init__(self, idx, bndry_conds):
        self._idx = idx
        self._fun = bndry_conds[self._idx][0]
        if isinstance(self._fun, TransientFunction):
            self.set_time = self._fun.set_time

    def __call__(self, xx):
        vals = self._fun(xx)
        if isinstance(self._fun, TransientFunction):
            return vals
        return vals[:, 0]


def _get_advection_diffusion_reaction_bndry_conds(
        mesh, bndry_types, domain_bounds, sol_fun, flux_funs):

    (D_bndry_keys, D_indices, N_bndry_keys, N_indices,
     R_bndry_keys, R_indices) = _get_bndry_keys_indices_from_types(
         mesh, bndry_types)
    nphys_vars = len(domain_bounds)//2
    bndry_conds = _get_mms_boundary_funs(
        nphys_vars, bndry_types, sol_fun, flux_funs)

    D_bndry_conds = dict()
    for key, idx in zip(D_bndry_keys, D_indices):
        D_bndry_conds[key] = [MSBoundaryConditionFunction(idx, bndry_conds)]
        # lambda does not work due to wierd way python does shallow copying
        # D_bndry_conds[key] = [lambda x: bndry_conds[idx][0](x)[:, 0]]

    N_bndry_conds = dict()
    for key, idx in zip(N_bndry_keys, N_indices):
        assert bndry_conds[idx][2] == 0
        N_bndry_conds[key] = [MSBoundaryConditionFunction(idx, bndry_conds)]

    R_bndry_conds = dict()
    for key, idx in zip(R_bndry_keys, R_indices):
        R_bndry_conds[key] = [
            MSBoundaryConditionFunction(idx, bndry_conds),
            bndry_conds[idx][2]]
    return D_bndry_conds, N_bndry_conds, R_bndry_conds


def _get_stokes_boundary_conditions(mesh, bndry_types, domain_bounds, vel_fun,
                                    vel_grad_funs):
    (D_bndry_keys, D_indices, N_bndry_keys, N_indices,
     R_bndry_keys, R_indices) = _get_bndry_keys_indices_from_types(
         mesh, bndry_types)

    nphys_vars = len(domain_bounds)//2
    vel_bndry_conds = [
        _get_advection_diffusion_reaction_bndry_conds(
            mesh, bndry_types, domain_bounds,
            partial(_vel_component_fun, vel_fun, ii),
            vel_grad_funs[ii])
        for ii in range(nphys_vars)]

    # currenlty only dirichlet supported by tests
    assert len(R_bndry_keys) == 0 and len(N_bndry_keys) == 0
    N_bndry_conds, R_bndry_conds = [], []

    idx = 0
    D_bndry_conds = dict()
    for key in vel_bndry_conds[idx][0].keys():
        D_bndry_conds[key] = [partial(
            _list_to_vector_bndry_cond_fun, vel_bndry_conds, idx, key)]

    return D_bndry_conds, N_bndry_conds, R_bndry_conds


class Function():
    def __init__(self, fun, name="fun"):
        self._fun = fun
        self._name = name

    def __call__(self, xx):
        vals = self._fun(xx)
        # if self._fun is not implemented correctly this will fail
        # E.g. when manufactured solution, diff etc. string does not have x
        # in it. If not dependent on x then must use 1e-16*x
        assert xx.ndim == vals.ndim
        return vals[:, 0]

    def __repr__(self):
        return "{0}()".format(self._name)


class TestFiniteElements(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_integrate(self):
        mesh = _get_mesh([0, 1], 1)
        element = _get_element(mesh, 2)
        basis = Basis(mesh, element)

        @Functional
        def integrate(w):
            return w.y

        vals = basis.project(lambda x: x[0]**2)
        integral = integrate.assemble(basis, y=basis.interpolate(vals))
        print(integral)
        assert np.allclose(integral, 1/3)

    def check_advection_diffusion_reaction(
            self, domain_bounds, order, nrefine, sol_string, diff_string,
            vel_strings, react_funs, bndry_types, mms_nl_diff_funs=[None, None]):

        sol_fun, diff_fun, mms_vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], False,
                mms_nl_diff_funs[0]))

        # put manufactured vels in format required by FEM
        def vel_fun(x):
            vals = mms_vel_fun(x)
            vals = np.swapaxes(vals, 0, 1)
            return vals

        # manufactured solutions assumes nl_diff_fun takes linear diffusion
        # and solution as arguments. Now convert to format required by
        # fem which takes x and solution as arguments
        nl_diff_funs = [None, None]
        if mms_nl_diff_funs[0] is not None:
            nl_diff_funs[0] = lambda x, sol: mms_nl_diff_funs[0](
                diff_fun(x), sol)
            nl_diff_funs[1] = lambda x, sol: mms_nl_diff_funs[1](
                diff_fun(x), sol)
        else:
            nl_diff_funs[0] = lambda x, sol: diff_fun(x)
            nl_diff_funs[1] = lambda x, sol: x[0]*0

        diff_fun = Function(diff_fun)
        forc_fun = Function(forc_fun)

        mesh = _get_mesh(domain_bounds, nrefine)
        element = _get_element(mesh, order)
        basis = Basis(mesh, element)

        bndry_conds = _get_advection_diffusion_reaction_bndry_conds(
            mesh, bndry_types, domain_bounds, sol_fun, flux_funs)

        # Solve linear diffusion problem to get initial guess
        # starting with just zeros can cause singular matrix if
        physics = AdvectionDiffusionReaction(
            mesh, element, basis, bndry_conds, diff_fun, forc_fun,
            vel_fun, nl_diff_funs, react_funs)
        init_sol = physics.init_guess()

        exact_sol = basis.project(lambda x: sol_fun(x)[:, 0])
        # print(np.abs(init_sol-exact_sol).max(), 'a')

        assemble = partial(
            _assemble_advection_diffusion_reaction, diff_fun,
            forc_fun, nl_diff_funs, react_funs, vel_fun, bndry_conds, mesh,
            element, basis)
        bilinear_mat, res, D_vals, D_dofs = assemble(u_prev=exact_sol)
        # minus sign because res = -a(u_prev, v) + L(v)
        jac = -bilinear_mat
        # print(jac.toarray()[:K.blocks[0], :K.blocks[0]], 'jac', jac.shape)
        II = np.setdiff1d(np.arange(jac.shape[0]), D_dofs)
        # print(res[II], 'res')
        # assert False
        assert np.all(np.abs(res[II]) < 5e-7)

        res = assemble(u_prev=init_sol)[1]

        # fem_sol = newton_solve(
        #     assemble, init_sol, atol=1e-8, rtol=1e-8, maxiters=20)
        solver = SteadyStatePDE(physics)
        fem_sol = solver.solve(init_sol, atol=1e-8, rtol=1e-8, maxiters=20)

        @Functional
        def integrate(w):
            return w.y
        error = np.sqrt(integrate.assemble(basis, y=(exact_sol-fem_sol)**2))
        print("error", error)

        mesh_pts = mesh.p
        fem_sol_on_mesh = basis.interpolator(fem_sol)(mesh_pts)
        # print(fem_sol)
        # print(fem_sol_on_mesh)
        # print(sol_fun(mesh_pts)[:, 0])
        # print(fem_sol_on_mesh-sol_fun(mesh_pts)[:, 0])
        assert np.allclose(fem_sol_on_mesh, sol_fun(mesh_pts)[:, 0], atol=1e-7)

        # II = np.argsort(mesh_pts[0])
        # mesh_pts = mesh_pts[:, II]
        # # plt.plot(mesh_pts[0], sol_fun(mesh_pts)[:, 0], '-ok')
        # # plt.plot(mesh_pts[0], fem_sol_on_mesh[II], '--')
        # plt.semilogy(
        #     mesh_pts[0], np.abs(sol_fun(mesh_pts)[:, 0]-fem_sol_on_mesh[II]), '--')
        # plt.show()

    def test_advection_diffusion_reaction(self):
        power = 1  # power of nonlinear diffusion
        test_cases = [
            [[0, 1], 2, 1, "x*(1-x)", "4+x", ["0+1e-16*x"], [None, None],
             ["D", "D"]],
            [[0, 1], 2, 1, "x*x", "4+x", ["0+1e-16*x"], [None, None],
             ["D", "D"]],
            [[0, 1], 2, 1, "x*x", "4+x", ["0+1e-16*x"], [None, None],
             ["D", "R"]],
            [[0, 1], 2, 1, "x*x", "4+x", ["1+1e-16*x"], [None, None],
             ["D", "R"]],
            # for nonlinear diffusion be careful to ensure that nl_diff > 0
            [[0, 1], 2, 5, "1+x", "4+1e-16*x", ["0+1e-16*x"], [None, None],
             ["D", "D"],
             [lambda linear_diff, sol: (sol**power+1)*linear_diff,
              lambda linear_diff, sol: (power*sol**(power-1))*linear_diff
              if power > 0 else 0*sol]],
            [[0, 1, 0, 1], 2, 1, "x*(1-x)+2*y*(1-y)", "4+1e-16*x",
             ["0+1e-16*x", "0+1e-16*x"],
             [None, None], ["D", "D", "D", "D"]],
            [[0, 1, 0, 1], 2, 1, "x*(1-x)+2*y*(1-y)", "4+1e-16*x",
             ["0+1e-16*x", "0+1e-16*x"],
             [None, None], ["D", "R", "N", "D"]],
            # [[0, 1, 0, 1], 2, 1, "x*(1-x)+2*y*(1-y)", "4+1e-16*x", ["0", "0"],
            #  [None, None], ["D", "D", "N", "D"],
            #  [lambda linear_diff, sol: (sol**power+1)*linear_diff,
            #   lambda linear_diff, sol: (power*sol**(power-1))*linear_diff
            #   if power > 0 else 0*sol]],
            [[0, 1], 2, 1, "x*(1-x)", "4+x", ["0+1e-16*x"],
             [lambda x, sol: 2*sol, lambda x, sol: 0*sol+2], ["D", "D"]],
            [[0, 1], 2, 1, "(1-x)", "4+x", ["0+1e-16*x"],
             [lambda x, sol: sol**2, lambda x, sol: 2*sol], ["D", "D"]],
            [[0, 1, 0, 1], 2, 1, "x*(1-x)+2*y*(1-y)", "4+1e-16*x",
             ["1+x", "2+1e-16*x"], [None, None], ["D", "R", "N", "D"]],
        ]
        # currently robin and neumann conditions do not work when
        # nonlinear diffusion present, so skip test
        cnt = 0
        for test_case in test_cases:
            print(cnt)
            self.check_advection_diffusion_reaction(*test_case)
            cnt += 1

    def _check_helmholtz(self, domain_bounds, order, nrefine, sol_string,
                         wnum_string, bndry_types):
        sol_fun, wnum_fun, forc_fun, flux_funs = (
            setup_helmholtz_manufactured_solution(
                sol_string, wnum_string, len(domain_bounds)//2))

        wnum_fun = Function(wnum_fun)
        forc_fun = Function(forc_fun)

        mesh = _get_mesh(domain_bounds, nrefine)
        element = _get_element(mesh, order)
        basis = Basis(mesh, element)

        bndry_conds = _get_advection_diffusion_reaction_bndry_conds(
            mesh, bndry_types, domain_bounds, sol_fun, flux_funs)

        physics = Helmholtz(
            mesh, element, basis, bndry_conds, wnum_fun, forc_fun)
        init_sol = physics.init_guess()

        exact_sol = basis.project(lambda x: sol_fun(x)[:, 0])
        #print(exact_sol)
        #print(init_sol)
        #print(np.abs(init_sol-exact_sol).max(), 'a')
        assert np.allclose(exact_sol, init_sol)

    def test_helmholtz(self):
        test_cases = [
            [[0, 1], 2, 1, "x**2", "1*x", ["D", "D"]],
            [[0, .5, 0, 1], 2, 1, "y**2*x**2", "1+1e-16*x",
             ["N", "D", "D", "D"]]
        ]
        for test_case in test_cases:
            self._check_helmholtz(*test_case)

    def check_stokes(self, domain_bounds, nrefine, vel_strings, pres_string,
                     bndry_types, navier_stokes):
        """
        bndry_types only refer to velocity boundaries.
        No boundary condition is placed on pressure.
        We place dirichlet=0 on all pressure boundaries to enforce uniquness
        this must means true pressure solution must take that value along
        entire boundary for test to pass.
        TODO: one value of pressure needs to be enforced. Forcing along
        entire boundary is to restrictive.
        """
        mesh = _get_mesh(domain_bounds, nrefine)
        element = {'u': ElementVector(_get_element(mesh, 2)),
                   'p': _get_element(mesh, 1)}
        basis = {variable: Basis(mesh, e, intorder=4)
                 for variable, e in element.items()}

        (vel_fun, pres_fun, vel_forc_fun, pres_forc_fun, vel_grad_funs,
         pres_grad_fun) = setup_steady_stokes_manufactured_solution(
             vel_strings, pres_string, navier_stokes)

        bndry_conds = _get_stokes_boundary_conditions(
            mesh, bndry_types, domain_bounds, vel_fun, vel_grad_funs)

        bilinear_mat, linear_vec, D_vals, D_dofs, K = _assemble_stokes(
            vel_forc_fun, pres_forc_fun, False,
            bndry_conds, mesh, element, basis, return_K=True)

        physics = Stokes(
            mesh, element, basis, bndry_conds, navier_stokes,
            vel_forc_fun, pres_forc_fun)
        init_sol = physics.init_guess()

        # A, b, x, I = condense(bilinear_mat, linear_vec, x=D_vals, D=D_dofs)
        # init_sol = solve(A, b, x, I)
        assemble = partial(_assemble_stokes, vel_forc_fun, pres_forc_fun,
                           navier_stokes, bndry_conds, mesh, element, basis)
        exact_pres_sol = basis['p'].project(lambda x: pres_fun(x)[:, 0])

        # projection using vector basis requires function to return np.ndarray
        # with shape (nvec, ...)
        exact_vel_sol = basis['u'].project(
            lambda x: np.stack(
                [vel_fun(x)[:, ii] for ii in range(len(domain_bounds)//2)]))
        exact_sol = np.concatenate([exact_vel_sol, exact_pres_sol])

        # check dirichlet boundary conditions are enforced correctly
        assert np.allclose(init_sol[D_dofs], exact_sol[D_dofs])

        if not navier_stokes:
            # print(init_sol-exact_sol)
            assert np.allclose(init_sol, exact_sol)

        bilinear_mat, res, D_vals, D_dofs = assemble(u_prev=exact_sol)
        # minus sign because res = -a(u_prev, v) + L(v)
        jac = -bilinear_mat
        II = np.setdiff1d(np.arange(jac.shape[0]), D_dofs)
        # print(res[II], 'res')
        assert np.all(np.abs(res[II]) < 5e-7)

        # fem_sol = newton_solve(
        #     assemble, init_sol, atol=1e-7, rtol=1e-6, maxiters=10)

        solver = SteadyStatePDE(physics)
        fem_sol = solver.solve(init_sol)

        # print(np.abs(fem_sol - exact_sol).max())
        assert np.allclose(fem_sol, exact_sol)

    def test_stokes(self):
        test_cases = [
            [[0, 1], 1, ["((2*x-1))**2"], "x*(1-1e-16*x)", ["D", "D"], False],
            [[0, 1], 0, ["((2*x-1))*(1+1e-16*x)"], "x*(1-1e-16*x)",
             ["D", "D"], True],
            [[0, 1, 0, 1], 1, ["1e-16*x+y", "(x+1)*(y+1)"], "x*(1+1e-16*x)+5e-16*y*(1+1e-16*y)",
             ["D", "D", "D", "D"], False],
            [[0, 1, 0, 1], 0, ["x**2*y**2", "(x+1)*(y+1)"], "x*y",
             ["D", "D", "D", "D"], True],
        ]

        for test_case in test_cases[-1:]:
            self.check_stokes(*test_case)

    def _check_transient_advection_diffusion_reaction(
            self, domain_bounds, nrefine, order, sol_string,
            diff_string, react_funs, bndry_types,
            tableau_name, nl_diff_funs=[None, None]):
        vel_strings = ["0"]*(len(domain_bounds)//2)
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], True))

        diff_fun = Function(diff_fun)
        forc_fun = TransientFunction(forc_fun, name='forcing')
        flux_funs = TransientFunction(flux_funs, name='flux')
        sol_fun = TransientFunction(sol_fun, name='sol')

        mesh = _get_mesh(domain_bounds, nrefine)
        element = _get_element(mesh, order)
        basis = Basis(mesh, element)

        bndry_conds = _get_advection_diffusion_reaction_bndry_conds(
            mesh, bndry_types, domain_bounds, sol_fun, flux_funs)

        physics = AdvectionDiffusionReaction(
            mesh, element, basis, bndry_conds, diff_fun, forc_fun,
            None, nl_diff_funs, react_funs)

        deltat = 1  # 0.1
        final_time = deltat*2  # 5
        sol_fun.set_time(0)
        init_sol = basis.project(lambda x: sol_fun(x))
        print(init_sol.shape)

        solver = TransientPDE(physics, deltat, tableau_name)
        sols, times = solver.solve(
            init_sol, 0, final_time,
            newton_kwargs={"atol": 1e-8, "rtol": 1e-8, "maxiters": 2})
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

    @unittest.skip(reason="Test and code incomplete")
    def test_transient_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], 2, 1, "(1-x)*x", "4+x", [None, None],
             ["D", "D"], "im_beuler1"],
        ]

        for test_case in test_cases:
            self._check_transient_advection_diffusion_reaction(*test_case)


if __name__ == "__main__":

    fem_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestFiniteElements)
    unittest.TextTestRunner(verbosity=2).run(fem_test_suite)
