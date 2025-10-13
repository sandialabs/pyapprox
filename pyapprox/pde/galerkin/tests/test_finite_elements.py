import unittest
import itertools
import numpy as np
from functools import partial
from skfem import ElementVector, Basis, Functional
from typing import List, Tuple

from pyapprox.pde.galerkin.util import _get_mesh, _get_element
from pyapprox.pde.galerkin.solvers import SteadyStatePDE, TransientPDE
from pyapprox.pde.galerkin.physics import (
    NonLinearAdvectionDiffusionReaction,
    Helmholtz,
    Stokes,
    Burgers,
    FEMScalarFunctionFromCallable,
    FEMVectorFunctionFromCallable,
    LinearAdvectionDiffusionReaction,
    FEMNonLinearOperatorFromCallable,
    FEMTransientScalarFunctionFromCallable,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
    ManufacturedHelmholtz,
    ManufacturedBurgers1D,
    ManufacturedStokes,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedNonLinearAdvectionDiffusionReaction,
)
from pyapprox.util.newton import NewtonSolver


def _normal_flux(flux_funs, normal_fun, xx):
    normal_vals = normal_fun(xx)
    flux_vals = flux_funs(xx)
    vals = np.sum(normal_vals * flux_vals, axis=1)
    return vals


def _robin_bndry_fun(sol_fun, flux_funs, normal_fun, alpha, xx, time=None):
    if time is not None:
        if hasattr(sol_fun, "set_time"):
            sol_fun.set_time(time)
        if hasattr(flux_funs, "set_time"):
            flux_funs.set_time(time)
    vals = alpha * sol_fun(xx) - _normal_flux(flux_funs, normal_fun, xx)
    return vals


def _canonical_normal(bndry_index, samples):
    # different to autopde because samples.ndim==3 here compared to ndim==2
    normal_vals = np.zeros(
        (samples.shape[1], samples.shape[0], samples.shape[2])
    )
    active_var = int(bndry_index >= 2)
    normal_vals[:, active_var, :] = (-1) ** ((bndry_index + 1) % 2)
    return normal_vals


def _get_mms_boundary_funs(
    nphys_vars, bndry_types, sol_fun, flux_funs, bndry_normals=None
):
    bndry_conds = []
    for dd in range(2 * nphys_vars):
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
                _robin_bndry_fun, sol_fun, flux_funs, normal_fun, alpha
            )
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
    nphys_vars = len(bndry_types) // 2
    # orders of keys must correspond to order specified in bndry_types
    # The following returns the keys mesh.boundaries.keys() but not
    # necessarily in the correct order
    keys = ["left", "right"]
    if nphys_vars == 2:
        keys += ["bottom", "top"]
    assert len(keys) == len(bndry_types)
    active_key_indices = [
        ii for ii in range(len(keys)) if bndry_types[ii] == bndry_type
    ]
    return [keys[idx] for idx in active_key_indices], active_key_indices


def _get_bndry_keys_indices_from_types(mesh, bndry_types):
    # get Dirichlet boundary names
    D_bndry_keys, D_indices = _bndrys_keys_from_bndry_types(
        mesh, bndry_types, "D"
    )
    N_bndry_keys, N_indices = _bndrys_keys_from_bndry_types(
        mesh, bndry_types, "N"
    )
    R_bndry_keys, R_indices = _bndrys_keys_from_bndry_types(
        mesh, bndry_types, "R"
    )
    return (
        D_bndry_keys,
        D_indices,
        N_bndry_keys,
        N_indices,
        R_bndry_keys,
        R_indices,
    )


class MSBoundaryConditionFunction:
    # Boundary condtion wrapper for manufactured solutions (MS)
    def __init__(self, idx, bndry_conds):
        self._idx = idx
        self._fun = bndry_conds[self._idx][0]
        if isinstance(self._fun, FEMTransientScalarFunctionFromCallable):
            self.set_time = self._fun.set_time

    def __call__(self, xx):
        vals = self._fun(xx)
        if isinstance(self._fun, FEMTransientScalarFunctionFromCallable):
            return vals
        return vals[:, 0]


def _get_advection_diffusion_reaction_bndry_conds(
    mesh, bndry_types, domain_bounds, sol_fun, flux_funs
):

    (
        D_bndry_keys,
        D_indices,
        N_bndry_keys,
        N_indices,
        R_bndry_keys,
        R_indices,
    ) = _get_bndry_keys_indices_from_types(mesh, bndry_types)
    nphys_vars = len(domain_bounds) // 2
    bndry_conds = _get_mms_boundary_funs(
        nphys_vars, bndry_types, sol_fun, flux_funs
    )

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
            bndry_conds[idx][2],
        ]
    return D_bndry_conds, N_bndry_conds, R_bndry_conds


def _vel_component_fun(vel_fun, ii, x, time=None):
    # currently takes raw functions not ones with axesswaped
    if time is not None:
        if hasattr(vel_fun, "set_time"):
            vel_fun.set_time(time)
    vals = vel_fun(x)
    return vals[:, ii : ii + 1]


def _get_stokes_boundary_conditions(
    mesh, bndry_types, domain_bounds, vel_fun, vel_grad_funs
):
    (
        D_bndry_keys,
        D_indices,
        N_bndry_keys,
        N_indices,
        R_bndry_keys,
        R_indices,
    ) = _get_bndry_keys_indices_from_types(mesh, bndry_types)

    nphys_vars = len(domain_bounds) // 2
    vel_bndry_conds = [
        _get_advection_diffusion_reaction_bndry_conds(
            mesh,
            bndry_types,
            domain_bounds,
            partial(_vel_component_fun, vel_fun, ii),
            vel_grad_funs[ii],
        )
        for ii in range(nphys_vars)
    ]

    # currenlty only dirichlet supported by tests
    assert len(R_bndry_keys) == 0 and len(N_bndry_keys) == 0
    N_bndry_conds, R_bndry_conds = [], []

    idx = 0
    D_bndry_conds = dict()
    for key in vel_bndry_conds[idx][0].keys():
        D_bndry_conds[key] = [
            partial(_list_to_vector_bndry_cond_fun, vel_bndry_conds, idx, key)
        ]

    return D_bndry_conds, N_bndry_conds, R_bndry_conds


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

        vals = basis.project(lambda x: x[0] ** 2)
        integral = integrate.assemble(basis, y=basis.interpolate(vals))
        print(integral)
        assert np.allclose(integral, 1 / 3)

    def _check_advection_diffusion_reaction(
        self,
        domain_bounds: List[float],
        order: int,
        nrefine: int,
        bndry_types: List[str],
        sol_string: str,
        diff_tup: Tuple[str, str],
        vel_strings: str,
        react_tup: Tuple[str, callable, callable],
    ):
        nvars = len(vel_strings)
        react_str, react_fun, react_prime = react_tup
        linear_diff_str, nonlinear_diff_op_str, diff_fun, diff_prime = diff_tup
        if "x" not in linear_diff_str:
            # if what a constant then must include 1e-16*x, 0*x will not work
            raise ValueError("linear_diff_str must have x")
        for v_str in vel_strings:
            if "x" not in v_str:
                raise ValueError("velocity_strs must have x")

        man_sol = ManufacturedNonLinearAdvectionDiffusionReaction(
            sol_string,
            nvars,
            linear_diff_str,
            nonlinear_diff_op_str,
            react_str,
            vel_strings,
            conservative=False,
        )
        print(man_sol)
        sol_fun = man_sol.functions["solution"]
        linear_diff_fun = man_sol.functions["linear_diffusion"]
        forc_fun = man_sol.functions["forcing"]
        mms_vel_fun = man_sol.functions["velocity"]
        flux_funs = man_sol.functions["flux"]

        vel_fun = FEMVectorFunctionFromCallable(mms_vel_fun, "velocity")

        linear_diff_fun = FEMScalarFunctionFromCallable(
            linear_diff_fun, "linear_diffusion"
        )
        forc_fun = FEMScalarFunctionFromCallable(forc_fun, "forcing")
        diff_op = FEMNonLinearOperatorFromCallable(diff_fun, diff_prime)
        react_op = FEMNonLinearOperatorFromCallable(react_fun, react_prime)

        print(domain_bounds, nrefine, type(domain_bounds))
        mesh = _get_mesh(domain_bounds, nrefine)
        element = _get_element(mesh, order)
        basis = Basis(mesh, element)

        bndry_conds = _get_advection_diffusion_reaction_bndry_conds(
            mesh, bndry_types, domain_bounds, sol_fun, flux_funs
        )

        # Solve linear diffusion problem to get initial guess
        # starting with just zeros can cause singular matrix if
        physics = NonLinearAdvectionDiffusionReaction(
            mesh,
            element,
            basis,
            bndry_conds,
            linear_diff_fun,
            forc_fun,
            vel_fun,
            diff_op,
            react_op,
        )
        # physics = LinearAdvectionDiffusionReaction(
        #     mesh,
        #     element,
        #     basis,
        #     bndry_conds,
        #     linear_diff_fun,
        #     forc_fun,
        #     vel_fun,
        #     FEMScalarFunctionFromCallable(man_sol.functions["reaction"]),
        # )
        newton_solver = NewtonSolver(verbosity=2, maxiters=30, rtol=1e-12)
        solver = SteadyStatePDE(physics, newton_solver)
        exact_sol = basis.project(lambda x: sol_fun(x)[:, 0])

        res = solver.newton_solver._residual(exact_sol)
        print(bndry_types)
        print(res, "Residual")
        assert np.all(np.abs(res) < 5e-7)

        linear_newton_solver = NewtonSolver(verbosity=2, maxiters=1)
        linear_physics = LinearAdvectionDiffusionReaction(
            mesh,
            element,
            basis,
            bndry_conds,
            linear_diff_fun,
            forc_fun,
            vel_fun,
        )
        linear_solver = SteadyStatePDE(linear_physics, linear_newton_solver)
        init_sol = linear_solver.solve(linear_physics.init_guess())
        fem_sol = solver.solve(init_sol)

        @Functional
        def integrate(w):
            return w.y

        error = np.sqrt(
            integrate.assemble(basis, y=(exact_sol - fem_sol) ** 2)
        )
        print("error", error)

        mesh_pts = mesh.p
        fem_sol_on_mesh = basis.interpolator(fem_sol)(mesh_pts)
        # print(fem_sol)
        # print(fem_sol_on_mesh)
        # print(sol_fun(mesh_pts)[:, 0])

        # II = np.argsort(mesh_pts[0])
        # ordered_mesh_pts = mesh_pts[:, II]
        # import matplotlib.pyplot as plt
        # plt.plot(ordered_mesh_pts[0], sol_fun(ordered_mesh_pts)[:, 0], '-ok')
        # plt.plot(ordered_mesh_pts[0], fem_sol_on_mesh[II], '--')
        # # plt.semilogy(
        # #     mesh_pts[0], np.abs(sol_fun(mesh_pts)[:, 0]-fem_sol_on_mesh[II]), '--')
        # plt.show()

        print(fem_sol_on_mesh - sol_fun(mesh_pts)[:, 0])
        assert np.allclose(fem_sol_on_mesh, sol_fun(mesh_pts)[:, 0], atol=1e-6)

    def test_advection_diffusion_reaction(self):
        # check 1D domain linear elements
        test_case_args = [
            [[0, 1], [0, 1.1]],  # bounds
            [
                2,
            ],  # element_order
            [
                1,
            ],  # nrefine
            [["D", "D"], ["R", "D"], ["R", "R"]],  # bndry_types
            [
                "x",
            ],  # sol_str
            # functions in tuple must be linear_diff * f(u)
            [
                (
                    "(4+1e-16*x)",
                    "u**0",
                    lambda x, u: 4 + u * 0,
                    lambda x, u: u * 0,
                )
            ],  # diff_tup
            [["0+1e-16*x"], ["(1+x)/10"]],  # vel_strs
            [
                ("2*u", lambda x, u: 2 * u, lambda x, u: 2 + 0 * u),
                ("u**2", lambda x, u: u**2, lambda x, u: 2 * u),
            ],  # react_tup
        ]
        for test_case in itertools.product(*test_case_args):
            self._check_advection_diffusion_reaction(*test_case)

        # check 1D domain quadratic elements
        test_case_args = [
            # [[0, 1], [0, 1.1]],  # bounds
            [[0, 1.1]],  # bounds
            [
                2,
            ],  # element_order
            [
                2,
            ],  # nrefine
            # [["D", "D"], ["D", "R"]],  # bndry_types
            [["D", "R"], ["D", "R"]],  # bndry_types
            [
                "x",
            ],  # sol_str
            # functions in tuple must be linear_diff * f(u)
            [
                (
                    "(4+1e-16*x)",
                    "u**2",
                    lambda x, u: 4 * u**2,
                    lambda x, u: 8 * u,
                )
            ],  # diff_tup
            [["0+1e-16*x"], ["(1+x)/10"]],  # vel_strs
            [
                ("u**2", lambda x, u: u**2, lambda x, u: 2 * u),
            ],  # react_tup
        ]
        for test_case in itertools.product(*test_case_args):
            self._check_advection_diffusion_reaction(*test_case)

        # check 2D domain nolinear reaction but linear diffusion
        # TODO ROBIN BOUNDARY and Neumann boundary do not work with non-linear diffusion
        test_case_args = [
            [
                [0, 1, 0, 1],
            ],  # bounds
            [
                2,
            ],  # element_order
            [
                1,
            ],  # nrefine
            [["D", "D", "D", "D"], ["D", "R", "D", "N"]],  # bndry_types
            [
                "x+2*y",
            ],  # sol_str
            # functions in tuple must be linear_diff * f(u)
            [
                (
                    "(4+1e-16*x+1e-16*y)",
                    "(u+1)**0",
                    lambda x, u: 4 + 0 * u,
                    lambda x, u: 0 * u,
                )
            ],  # diff_tup
            [["0+1e-16*x", "0+1e-16*x"], ["1+1e-16*x", "1+x"]],  # vel_strs
            [
                ("u**2", lambda x, u: u**2, lambda x, u: 2 * u),
            ],  # react_tup
        ]
        for test_case in itertools.product(*test_case_args):
            self._check_advection_diffusion_reaction(*test_case)

        # check 2D domain nonlinear PDE
        # TODO ROBIN BOUNDARY and Neumann boundary do not work with non-linear diffusion
        test_case_args = [
            [
                [0, 1, 0, 1],
            ],  # bounds
            [
                2,
            ],  # element_order
            [
                1,
            ],  # nrefine
            [
                ["D", "D", "D", "D"],
            ],  # bndry_types
            [
                "x+2*y",
            ],  # sol_str
            # functions in tuple must be linear_diff * f(u)
            [
                (
                    "(4+1e-16*x+1e-16*y)",
                    "(u+1)**2",
                    lambda x, u: 4 * (u + 1) ** 2,
                    lambda x, u: 8 * (u + 1) ** 1,
                )
            ],  # diff_tup
            [
                ["0+1e-16*x", "0+1e-16*x"],
            ],  # vel_strs
            [
                ("u**2", lambda x, u: u**2, lambda x, u: 2 * u),
            ],  # react_tup
        ]
        for test_case in itertools.product(*test_case_args):
            self._check_advection_diffusion_reaction(*test_case)

    def _check_helmholtz(
        self,
        domain_bounds,
        order,
        nrefine,
        sol_string,
        wnum_string,
        bndry_types,
    ):
        nvars = len(domain_bounds) // 2
        man_sol = ManufacturedHelmholtz(sol_string, nvars, wnum_string)
        sol_fun = man_sol.functions["solution"]
        flux_funs = man_sol.functions["flux"]
        wnum_npfun = man_sol.functions["sqwavenum"]
        forc_fun = man_sol.functions["forcing"]

        # wnum_fun = FEMScalarFunctionFromCallable(wnum_fun)
        wnum_fun = FEMNonLinearOperatorFromCallable(
            lambda x, u: wnum_npfun(x)[0] * u,
            lambda x, u: wnum_npfun(x)[0] + 0 * u,
        )
        forc_fun = FEMScalarFunctionFromCallable(forc_fun)

        mesh = _get_mesh(domain_bounds, nrefine)
        element = _get_element(mesh, order)
        basis = Basis(mesh, element)

        bndry_conds = _get_advection_diffusion_reaction_bndry_conds(
            mesh, bndry_types, domain_bounds, sol_fun, flux_funs
        )

        physics = Helmholtz(
            mesh, element, basis, bndry_conds, wnum_fun, forc_fun
        )
        newton_solver = NewtonSolver(verbosity=2, maxiters=20, rtol=1e-10)
        solver = SteadyStatePDE(physics, newton_solver)
        fem_sol = solver.solve(physics.init_guess())

        exact_sol = basis.project(lambda x: sol_fun(x)[:, 0])
        # print(fem_sol)
        # print(exact_sol)
        # print(np.abs(fem_sol - exact_sol).max(), "a")
        assert np.allclose(fem_sol, exact_sol)

    def test_helmholtz(self):
        # note react fun for advection diffusion reaction equation (wnum here)
        # must be size of u. so if reaction term depends on x it must be on
        # a single entry of x or a scalar function of all x, e.g sum(x[0])
        test_cases = [
            [[0, 1], 2, 1, "x", "4 + 1e-16*x", ["D", "D"]],
            [
                [0, 0.5, 0, 1],
                2,
                1,
                "y**1*x**1",
                "4+1e-16*x",
                ["D", "D", "D", "D"],
            ],
        ]
        for test_case in test_cases:
            self._check_helmholtz(*test_case)

    def check_stokes(
        self,
        domain_bounds,
        nrefine,
        vel_strings,
        pres_string,
        bndry_types,
        navier_stokes,
    ):
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
        element = {
            "u": ElementVector(_get_element(mesh, 2)),
            "p": _get_element(mesh, 1),
        }
        basis = {
            variable: Basis(mesh, e, intorder=4)
            for variable, e in element.items()
        }

        nvars = len(domain_bounds) // 2
        man_sol = ManufacturedStokes(
            vel_strings + [pres_string], nvars, navier_stokes
        )
        print(man_sol)
        sol_fun = man_sol.functions["solution"]
        # flux_funs = man_sol.functions["flux"]
        # forc_fun = man_sol.functions["forcing"]
        # vel_grad_funs = man_sol.functions["flux"]
        vel_grad_funs = [None] * nvars

        vel_fun = sol_fun
        # next function currently takes raw functions not ones with axesswaped
        bndry_conds = _get_stokes_boundary_conditions(
            mesh, bndry_types, domain_bounds, vel_fun, vel_grad_funs
        )

        vel_forc_fun = FEMVectorFunctionFromCallable(
            man_sol.functions["vel_forcing"]
        )
        pres_forc_fun = FEMScalarFunctionFromCallable(
            man_sol.functions["pres_forcing"]
        )
        sol_fun = FEMVectorFunctionFromCallable(sol_fun, "sol")

        physics = Stokes(
            mesh,
            element,
            basis,
            bndry_conds,
            navier_stokes,
            vel_forc_fun,
            pres_forc_fun,
        )
        init_sol = physics.init_guess()

        exact_pres_sol = basis["p"].project(lambda x: sol_fun(x)[-1])

        # projection using vector basis requires function to return np.ndarray
        # with shape (nvec, ...)
        exact_vel_sol = basis["u"].project(
            lambda x: np.stack(
                [sol_fun(x)[ii, :] for ii in range(len(domain_bounds) // 2)]
            )
        )
        exact_sol = np.concatenate([exact_vel_sol, exact_pres_sol])

        if not navier_stokes:
            print(exact_vel_sol, "vel")
            print(exact_pres_sol, "pres")
            print(init_sol)
            print(exact_sol)
            print(init_sol - exact_sol)
            assert np.allclose(init_sol, exact_sol)

        solver = SteadyStatePDE(physics)
        res = solver.newton_solver._residual(exact_sol)
        assert np.all(np.abs(res) < 5e-7)

        fem_sol = solver.solve(init_sol)
        # print(np.abs(fem_sol - exact_sol).max())
        assert np.allclose(fem_sol, exact_sol)

    def test_stokes(self):
        test_cases = [
            [[0, 1], 1, ["((2*x-1))**2"], "x*(1-1e-16*x)", ["D", "D"], False],
            [
                [0, 1],
                0,
                ["((2*x-1))*(1+1e-16*x)"],
                "x*(1-1e-16*x)",
                ["D", "D"],
                True,
            ],
            [
                [0, 1, 0, 1],
                1,
                ["1e-16*x+y", "(x+1)*(y+1)"],
                "x*(1+1e-16*x)+5e-16*y*(1+1e-16*y)",
                ["D", "D", "D", "D"],
                False,
            ],
            [
                [0, 1, 0, 1],
                0,
                ["x**2*y**2", "(x+1)*(y+1)"],
                "x*y",
                ["D", "D", "D", "D"],
                True,
            ],
        ]

        for test_case in test_cases:
            self.check_stokes(*test_case)

    def _check_transient_advection_diffusion_reaction(
        self,
        domain_bounds,
        nrefine,
        order,
        sol_string,
        diff_string,
        react_funs,
        bndry_types,
        tableau_name,
        mms_nl_diff_funs=[None, None],
    ):
        vel_strings = ["0"] * (len(domain_bounds) // 2)
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], True
            )
        )

        nl_diff_funs = [None, None]
        if mms_nl_diff_funs[0] is not None:
            nl_diff_funs[0] = lambda x, sol: mms_nl_diff_funs[0](
                diff_fun(x), sol
            )
            nl_diff_funs[1] = lambda x, sol: mms_nl_diff_funs[1](
                diff_fun(x), sol
            )
        else:
            nl_diff_funs[0] = lambda x, sol: diff_fun(x)
            nl_diff_funs[1] = lambda x, sol: x[0] * 0

        diff_fun = FEMScalarFunctionFromCallable(diff_fun)
        forc_fun = FEMTransientScalarFunctionFromCallable(
            forc_fun, name="forcing"
        )
        flux_funs = FEMTransientScalarFunctionFromCallable(
            flux_funs, name="flux"
        )
        sol_fun = FEMTransientScalarFunctionFromCallable(sol_fun, name="sol")

        mesh = _get_mesh(domain_bounds, nrefine)
        element = _get_element(mesh, order)
        basis = Basis(mesh, element)

        bndry_conds = _get_advection_diffusion_reaction_bndry_conds(
            mesh, bndry_types, domain_bounds, sol_fun, flux_funs
        )

        physics = AdvectionDiffusionReaction(
            mesh,
            element,
            basis,
            bndry_conds,
            diff_fun,
            forc_fun,
            None,
            nl_diff_funs,
            react_funs,
        )

        deltat = 1
        final_time = deltat * 5
        sol_fun.set_time(0)
        init_sol = basis.project(lambda x: sol_fun(x))

        @Functional
        def integrate(w):
            return w.y

        solver = TransientPDE(physics, deltat, tableau_name)
        sols, times = solver.solve(
            init_sol,
            0,
            final_time,
            newton_kwargs={"atol": 1e-8, "rtol": 1e-8, "maxiters": 2},
        )
        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sol_t = basis.project(lambda x: sol_fun(x))
            model_sol_t = sols[:, ii]
            L2_error = np.sqrt(
                integrate.assemble(basis, y=(exact_sol_t - model_sol_t) ** 2)
            )
            factor = np.sqrt(integrate.assemble(basis, y=exact_sol_t**2))
            print(time, L2_error, 1e-8 * factor)
            assert L2_error < 1e-8 * factor

    # @unittest.skip(reason="Test and code incomplete")
    def test_transient_advection_diffusion_reaction(self):
        test_cases = [
            [
                [0, 1],
                2,
                2,
                "(1-x)*x",
                "4+x",
                [None, None],
                ["D", "D"],
                "im_beuler1",
            ],
            [
                [0, 1],
                2,
                2,
                "(1-x)*x*(1+t)",
                "4+x",
                [None, None],
                ["D", "D"],
                "im_beuler1",
            ],
        ]

        for test_case in test_cases:
            self._check_transient_advection_diffusion_reaction(*test_case)

    def _check_transient_burgers(
        self,
        order,
        nrefine,
        domain_bounds,
        sol_string,
        viscosity_string,
        transient,
        bndry_types,
    ):
        tableau_name = "im_beuler1"
        sol_fun, viscosity_fun, forc_fun, flux_funs = (
            setup_burgers_manufactured_solution(
                sol_string, viscosity_string, transient
            )
        )

        viscosity_fun = FEMScalarFunctionFromCallable(viscosity_fun)
        if transient:
            forc_fun = FEMTransientScalarFunctionFromCallable(
                forc_fun, name="forcing"
            )
            flux_funs = FEMTransientScalarFunctionFromCallable(
                flux_funs, name="flux"
            )
            sol_fun = FEMTransientScalarFunctionFromCallable(
                sol_fun, name="sol"
            )
        else:
            forc_fun = FEMScalarFunctionFromCallable(forc_fun, name="forcing")
            flux_funs = FEMScalarFunctionFromCallable(flux_funs, name="flux")
            # for some reason tests fail when sol fun is wrapped
            # sol_fun = FEMScalarFunctionFromCallable(sol_fun, name='sol')
            # so do not wrap

        periodic = bndry_types == ["P", "P"]
        mesh = _get_mesh(domain_bounds, nrefine, periodic)
        element = _get_element(mesh, order)
        basis = Basis(mesh, element)

        if not periodic:
            bndry_conds = _get_advection_diffusion_reaction_bndry_conds(
                mesh, bndry_types, domain_bounds, sol_fun, flux_funs
            )
        else:
            bndry_conds = ({}, {}, {})

        physics = Burgers(
            mesh, element, basis, bndry_conds, viscosity_fun, forc_fun
        )

        @Functional
        def integrate(w):
            return w.y

        if not transient:
            init_sol = basis.project(lambda x: sol_fun(x)[:, 0] + 1)
            newton_solver = NewtonSolver(verbosity=2, maxiters=5, rtol=1e-12)
            solver = SteadyStatePDE(physics, newton_solver)
            fem_sol = solver.solve(init_sol)
            # when projecting must use sol_fun(x)[:, 0]
            exact_sol = basis.project(lambda x: sol_fun(x)[:, 0])
            error = np.sqrt(
                integrate.assemble(basis, y=(exact_sol - fem_sol) ** 2)
            )
            print("error", error)
            # II = np.argsort(mesh.p[0])
            # assert error < 1e-12
            # import matplotlib.pyplot as plt
            # ax = plt.subplots(1, 1)[1]
            # mesh_pts = mesh.p[:, II]
            # ax.plot(mesh_pts[0], sol_fun(mesh_pts)[:, 0], '--r')
            # # basis.interpolator does not work for periodic mesh
            # # fem_sol_on_mesh = basis.interpolator(fem_sol)(mesh_pts)
            # # plt.plot(mesh_pts[0], fem_sol_on_mesh, '--')
            # # but fem_sol is the wrong quantity to plot for periodic mesh
            # # ax.plot(mesh_pts[0], fem_sol[II], 'o')
            # # so use basis.refinterp
            # # increase nrefs to get solution on finer mesh
            # mpts, vals = basis.refinterp(fem_sol, nrefs=0)
            # ix = np.argsort(mpts.p[0])
            # ax.plot(mpts.p[0][ix], vals[ix], 'b')
            # plt.show()
            return

        deltat = 1
        final_time = deltat * 5
        sol_fun.set_time(0)
        init_sol = basis.project(lambda x: sol_fun(x))

        @Functional
        def integrate(w):
            return w.y

        solver = TransientPDE(physics, deltat, tableau_name)
        sols, times = solver.solve(
            init_sol,
            0,
            final_time,
            newton_kwargs={"atol": 1e-8, "rtol": 1e-8, "maxiters": 2},
        )
        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sol_t = basis.project(lambda x: sol_fun(x))
            model_sol_t = sols[:, ii]
            L2_error = np.sqrt(
                integrate.assemble(basis, y=(exact_sol_t - model_sol_t) ** 2)
            )
            factor = np.sqrt(integrate.assemble(basis, y=exact_sol_t**2))
            print(time, L2_error, 1e-8 * factor)
            assert L2_error < 1e-8 * factor

    def test_transient_burgers(self):
        test_cases = [
            [2, 3, [0, 1], "x*(1.0-x)", "10+1e-16*x", False, ["D", "D"]],
            [2, 3, [0, 1], "x*(1.0-x)*(1+t)", "10+1e-16*x", True, ["D", "D"]],
            # periodic mesh and cosine needs a high nrefine to even get close
            [2, 10, [0, 1], "cos(2*pi*x)", "10+1e-16*x", False, ["P", "P"]],
            [
                2,
                10,
                [0, 1],
                "cos(2*pi*x)*(1+t)",
                "10+1e-16*x",
                True,
                ["P", "P"],
            ],
        ]
        for case in test_cases:
            self._check_transient_burgers(*case)


if __name__ == "__main__":
    unittest.main(verbosity=2)
