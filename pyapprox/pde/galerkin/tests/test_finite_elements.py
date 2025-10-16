import copy
import unittest
import itertools
from functools import partial

import numpy as np
from skfem import ElementVector, Basis, Functional, Mesh, Element
from typing import List, Tuple

from pyapprox.pde.galerkin.util import get_mesh, get_element
from pyapprox.pde.galerkin.solvers import SteadyStatePDE, TransientPDE
from pyapprox.pde.galerkin.physics import (
    BoundaryConditions,
    NonLinearAdvectionDiffusionReaction,
    Helmholtz,
    Stokes,
    Burgers,
    FEMScalarFunctionFromCallable,
    FEMVectorFunctionFromCallable,
    LinearAdvectionDiffusionReaction,
    FEMNonLinearOperatorFromCallable,
    FEMTransientScalarFunctionFromCallable,
    FEMTransientVectorFunctionFromCallable,
    TransientMixin,
    FEMVectorFunction,
    FEMScalarFunction,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
    ManufacturedHelmholtz,
    ManufacturedBurgers1D,
    ManufacturedStokes,
    ManufacturedSolution,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    ManufacturedNonLinearAdvectionDiffusionReaction,
)
from pyapprox.util.newton import NewtonSolver

from abc import abstractmethod


class ManufacturedRobinBoundaryConditionFunction(FEMScalarFunction):
    def __init__(
        self,
        alpha: float,
        sol_fun: FEMScalarFunction,
        flux_funs: FEMVectorFunction,
        bndry_normal: callable,
    ):
        self._alpha = alpha
        self._sol_fun = sol_fun
        self._flux_funs = flux_funs
        self._bndry_normal = bndry_normal

    @abstractmethod
    def _check_funs(
        self, sol_fun: FEMScalarFunction, flux_funs: FEMVectorFunction
    ):
        raise NotImplementedError

    def _normal_flux(self, xx: np.ndarray) -> np.ndarray:
        normal_vals = self._bndry_normal(xx)
        flux_vals = self._flux_funs(xx)
        vals = np.sum(normal_vals * flux_vals, axis=1)
        return vals

    def _values(self, xx: np.ndarray) -> np.ndarray:
        sol_vals = self._sol_fun(xx)
        flux_vals = self._normal_flux(xx)
        assert sol_vals.shape == flux_vals.shape
        vals = self._alpha * sol_vals - flux_vals
        return vals


class SteadyManufacturedRobinBoundaryConditionFunction(
    ManufacturedRobinBoundaryConditionFunction
):
    def _check_funs(
        self,
        sol_fun: FEMScalarFunctionFromCallable,
        flux_funs: FEMVectorFunctionFromCallable,
    ):
        if not isinstance(sol_fun, FEMScalarFunctionFromCallable):
            raise TypeError(
                "sol_fun must be an instance of FEMScalarFunctionFromCallable"
            )
        if not isinstance(flux_funs, FEMVectorFunction):
            raise TypeError(
                "flux_funs must be an instance of "
                "FEMVectorFunctionFromCallable"
            )


class TransientManufacturedRobinBoundaryConditionFunction(
    TransientMixin, ManufacturedRobinBoundaryConditionFunction
):
    def _check_funs(
        self,
        sol_fun: FEMTransientScalarFunctionFromCallable,
        flux_funs: FEMTransientVectorFunctionFromCallable,
    ):
        if not isinstance(sol_fun, FEMTransientScalarFunctionFromCallable):
            raise TypeError(
                "sol_fun must be an instance of "
                "FEMTransientScalarFunctionFromCallable"
            )
        if not isinstance(flux_funs, FEMTransientVectorFunctionFromCallable):
            raise TypeError(
                "flux_funs must be an instance of "
                "FEMTransientVectorFunctionFromCallable"
            )

    def set_time(self, time: float):
        super().set_time(time)
        self._sol_fun.set_time(time)
        self._flux_funs.set_time(time)


class ManufacturedSolutionToBoundaryConditions:
    def __init__(
        self,
        man_sol: ManufacturedSolution,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        time: float = None,
    ):
        if not isinstance(man_sol, ManufacturedSolution):
            raise TypeError(
                "man sol must be an instance of ManufacturedSolution"
            )
        self._time = time
        if self._time is None:
            self._sol_fun = FEMScalarFunctionFromCallable(
                man_sol.functions["solution"]
            )
            self._flux_funs = FEMVectorFunctionFromCallable(
                man_sol.functions["flux"], swapaxes=False
            )
        else:
            self._sol_fun = FEMTransientScalarFunctionFromCallable(
                man_sol.functions["solution"]
            )
            self._sol_fun.set_time(time)

            self._flux_funs = FEMTransientVectorFunctionFromCallable(
                man_sol.functions["flux"], swapaxes=False
            )
            self._flux_funs.set_time(time)
        self._mesh = mesh
        self._element = element
        self._basis = basis

    def _canonical_normal(self, bndry_index, samples):
        # different to autopde because samples.ndim==3 here compared to ndim==2
        normal_vals = np.zeros(
            (samples.shape[1], samples.shape[0], samples.shape[2])
        )
        active_var = int(bndry_index >= 2)
        normal_vals[:, active_var, :] = (-1) ** ((bndry_index + 1) % 2)
        return normal_vals

    def _setup_robin_boundary_condition(self, alpha: float, bndry_idx: int):
        # normal_fun used below only works when domain is a rectangle
        normal_fun = partial(self._canonical_normal, bndry_idx)
        if self._time is None:
            bndry_fun = SteadyManufacturedRobinBoundaryConditionFunction(
                alpha, self._sol_fun, self._flux_funs, normal_fun
            )

            return bndry_fun
        bndry_fun = TransientManufacturedRobinBoundaryConditionFunction(
            alpha, self._sol_fun, self._flux_funs, normal_fun
        )
        bndry_fun.set_time(self._time)
        return bndry_fun

    def _setup_boundary_functions(
        self, nphys_vars: int, bndry_types: List[str]
    ):
        if len(bndry_types) != 2 * nphys_vars:
            raise ValueError("must specify a bndry type for each boundary")
        bndry_conds = []
        for bndry_idx in range(2 * nphys_vars):
            if bndry_types[bndry_idx] == "D":
                bndry_conds.append([copy.deepcopy(self._sol_fun), "D"])
            elif bndry_types[bndry_idx] == "P":
                bndry_conds.append([None, "P"])
            else:
                # Use an arbitray non-zero value (1.0) to test Robin BCs
                # use zero to reduce Robin BC to Neumann BC
                alpha = 1.0
                bndry_fun = self._setup_robin_boundary_condition(
                    alpha if bndry_types[bndry_idx] == "R" else 0.0, bndry_idx
                )
                if bndry_types[bndry_idx] == "N":
                    bndry_conds.append([bndry_fun, "N"])
                else:
                    bndry_conds.append([bndry_fun, "R", alpha])
            if bndry_conds[-1][0] is not None:
                bndry_conds[-1][0]._name = f"bndry_{bndry_idx}"
        return bndry_conds

    def boundary_conditions(self, bndry_types: List[str]):
        bndry_cond_tuples = self._setup_boundary_functions(
            self._mesh.p.shape[0], bndry_types
        )

        # bndry_cond tuples ordered [left, right, bottom, top]
        bndry_names = ["left", "right", "bottom", "top"]
        D_bndry_names, D_bndry_funs = [], []
        N_bndry_names, N_bndry_funs = [], []
        R_bndry_names, R_bndry_funs, R_bndry_consts = [], [], []
        for ii, tup in enumerate(bndry_cond_tuples):
            if tup[1] == "D":
                D_bndry_names.append(bndry_names[ii])
                D_bndry_funs.append(tup[0])
            elif tup[1] == "N":
                N_bndry_names.append(bndry_names[ii])
                N_bndry_funs.append(tup[0])
            else:
                R_bndry_names.append(bndry_names[ii])
                R_bndry_funs.append(tup[0])
                R_bndry_consts.append(tup[2])

        bndry_conds = BoundaryConditions(
            self._mesh,
            self._element,
            self._basis,
            D_bndry_names,
            D_bndry_funs,
            N_bndry_names,
            N_bndry_funs,
            R_bndry_names,
            R_bndry_funs,
            R_bndry_consts,
        )
        return bndry_conds


def _normal_flux(flux_funs, normal_fun, xx):
    normal_vals = normal_fun(xx)
    flux_vals = flux_funs(xx)
    vals = np.sum(normal_vals * flux_vals, axis=1)
    return vals


def _robin_bndry_fun(sol_fun, flux_funs, normal_fun, alpha, xx, time=None):
    if not isinstance(sol_fun, FEMScalarFunctionFromCallable):
        raise TypeError(
            "sol_fun must be an instance of FEMScalarFunctionFromCallable"
        )
    if time is not None:
        if hasattr(sol_fun, "set_time"):
            sol_fun.set_time(time)
        if hasattr(flux_funs, "set_time"):
            flux_funs.set_time(time)
    sol_vals = sol_fun(xx)
    flux_vals = _normal_flux(flux_funs, normal_fun, xx)
    assert sol_vals.shape == flux_vals.shape
    vals = alpha * sol_vals - flux_vals
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
            if bndry_types[dd] == "N":
                bndry_conds.append([bndry_fun, "N"])
            else:
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


# class MSBoundaryConditionFunction(FEMScalarFunctionFromCallable):
#     # Boundary condtion wrapper for manufactured solutions (MS)
#     def __init__(self, fun: callable, name: str = None):
#         super().__init__(fun, name)
#         self._fun = fun
#         if isinstance(self._fun, FEMTransientScalarFunctionFromCallable):
#             self.set_time = self._fun.set_time


def _get_scalar_bndry_conds(
    mesh, element, basis, bndry_types, nphys_vars, sol_fun, flux_funs
):
    bndry_cond_tuples = _get_mms_boundary_funs(
        nphys_vars, bndry_types, sol_fun, flux_funs
    )

    # bndry_cond tuples ordered [left, right, top, bottom]
    bndry_names = ["left", "right", "top", "bottom"]
    D_bndry_names, D_bndry_funs = [], []
    N_bndry_names, N_bndry_funs = [], []
    R_bndry_names, R_bndry_funs, R_bndry_consts = [], [], []
    for ii, tup in enumerate(bndry_cond_tuples):
        if tup[1] == "D":
            D_bndry_names.append(bndry_names[ii])
            D_bndry_funs.append(MSBoundaryConditionFunction(tup[0]))
        elif tup[1] == "N":
            N_bndry_names.append(bndry_names[ii])
            N_bndry_funs.append(MSBoundaryConditionFunction(tup[0]))
        else:
            R_bndry_names.append(bndry_names[ii])
            R_bndry_funs.append(MSBoundaryConditionFunction(tup[0]))
            R_bndry_consts.append(tup[2])

    bndry_conds = BoundaryConditions(
        mesh,
        element,
        basis,
        D_bndry_names,
        D_bndry_funs,
        N_bndry_names,
        N_bndry_funs,
        R_bndry_names,
        R_bndry_funs,
        R_bndry_consts,
    )
    return bndry_conds

    # D_bndry_conds = dict()
    # for key, idx in zip(D_bndry_keys, D_indices):
    #     D_bndry_cond_tuples[key] = [MSBoundaryConditionFunction(idx, bndry_cond_tuples)]
    #     # lambda does not work due to wierd way python does shallow copying
    #     # D_bndry_cond_tuples[key] = [lambda x: bndry_cond_tuples[idx][0](x)[:, 0]]

    # N_bndry_cond_tuples = dict()
    # for key, idx in zip(N_bndry_keys, N_indices):
    #     assert bndry_cond_tuples[idx][2] == 0
    #     N_bndry_cond_tuples[key] = [MSBoundaryConditionFunction(idx, bndry_cond_tuples)]

    # R_bndry_cond_tuples = dict()
    # for key, idx in zip(R_bndry_keys, R_indices):
    #     R_bndry_cond_tuples[key] = [
    #         MSBoundaryConditionFunction(idx, bndry_cond_tuples),
    #         bndry_cond_tuples[idx][2],
    #     ]
    # return D_bndry_cond_tuples, N_bndry_cond_tuples, R_bndry_cond_tuples


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
        _get_scalar_bndry_conds(
            mesh,
            bndry_types,
            len(domain_bounds) // 2,
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
        mesh = get_mesh([0, 1], 1)
        element = get_element(mesh, 2)
        basis = Basis(mesh, element)

        @Functional
        def integrate(w):
            return w.y

        vals = basis.project(lambda x: x[0] ** 2)
        integral = integrate.assemble(basis, y=basis.interpolate(vals))
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

        # man_sol creates diffusion as linear_diff_str * nonlinear_diff_op_str
        # however NonLinearAdvectionDiffusionReaction
        # does not do this, linear_diff_fun is only used for initial guess
        # and diff_fun must return equivalent to
        # linear_diff_str * nonlinear_diff_op_str
        man_sol = ManufacturedNonLinearAdvectionDiffusionReaction(
            sol_string,
            nvars,
            linear_diff_str,
            nonlinear_diff_op_str,
            react_str,
            vel_strings,
            conservative=False,
            oned=True,  # must be true for FEM code
        )

        mesh = get_mesh(domain_bounds, nrefine, nx=3)
        # mesh = get_mesh(domain_bounds, nrefine, nx=4)
        element = get_element(mesh, order)
        basis = Basis(mesh, element)

        bndry_converter = ManufacturedSolutionToBoundaryConditions(
            man_sol, mesh, element, basis
        )
        bndry_conds = bndry_converter.boundary_conditions(bndry_types)
        # print(man_sol)
        # print(bndry_conds)

        # wrap manufactured functions into classes required by FEM code
        # these classes are useful because the check sizes of output
        # which if not carefully checked can result in errors
        # for example by unintentional broadcasting
        sol_fun = FEMScalarFunctionFromCallable(man_sol.functions["solution"])
        vel_fun = FEMVectorFunctionFromCallable(
            man_sol.functions["velocity"], "velocity"
        )
        linear_diff_fun = FEMScalarFunctionFromCallable(
            man_sol.functions["linear_diffusion"], "linear_diffusion"
        )
        forc_fun = FEMScalarFunctionFromCallable(
            man_sol.functions["forcing"], "forcing"
        )
        diff_op = FEMNonLinearOperatorFromCallable(diff_fun, diff_prime)
        react_op = FEMNonLinearOperatorFromCallable(react_fun, react_prime)

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
        newton_solver = NewtonSolver(verbosity=0, maxiters=30, rtol=1e-12)
        solver = SteadyStatePDE(physics, newton_solver)
        exact_sol = basis.project(lambda x: sol_fun(x))

        res = solver.newton_solver._residual(exact_sol)
        # print(res, "Residual")
        assert np.all(np.abs(res) < 5e-7)

        # Solve linear diffusion problem to get initial guess
        # starting with just zeros can cause singular matrix
        linear_newton_solver = NewtonSolver(verbosity=0, maxiters=1)
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
        error = solver.L2_error(exact_sol, fem_sol)
        # print("error", error)
        assert error < 3e-8

        mesh_pts = mesh.p
        fem_sol_on_mesh = basis.interpolator(fem_sol)(mesh_pts)
        assert np.allclose(fem_sol_on_mesh, sol_fun(mesh_pts))

    def test_advection_diffusion_reaction(self):
        test_case = [
            [0, 1, 0, 1],
            2,
            0,
            ["D", "R", "D", "D"],
            "x**2*y**2",
            (
                "(1+1e-16*x+1e-16*y)",
                "(u+1)**0",
                lambda x, u: 1 + 0 * u,
                lambda x, u: 0 * u,
            ),
            ["0+1e-16*x", "0+1e-16*x"],
            ("u*0", lambda x, u: u * 0, lambda x, u: u * 0),
        ]
        self._check_advection_diffusion_reaction(*test_case)

        # check 1D domain linear elements
        test_case_args = [
            [[0, 1], [0, 1.1]],  # bounds
            [
                2,
            ],  # element_order
            [
                0,
            ],  # nrefine
            [["D", "N"], ["R", "D"], ["R", "R"]],  # bndry_types
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
        test_case_args = [
            [
                [0, 1, 0, 1],
            ],  # bounds
            [
                2,
            ],  # element_order
            [
                0,
            ],  # nrefine
            [["D", "D", "D", "D"], ["D", "R", "D", "N"]],  # bndry_types
            [
                "x+2*y",
                # "x**2*y**2",
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
                ["D", "D", "D", "R"],
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
        man_sol = ManufacturedHelmholtz(
            sol_string, nvars, wnum_string, oned=True
        )
        # print(man_sol)
        sol_fun = FEMScalarFunctionFromCallable(man_sol.functions["solution"])
        forc_fun = FEMScalarFunctionFromCallable(man_sol.functions["forcing"])

        wnum_npfun = man_sol.functions["sqwavenum"]
        wnum_fun = FEMNonLinearOperatorFromCallable(
            lambda x, u: wnum_npfun(x) * u,
            lambda x, u: wnum_npfun(x) + 0 * u,
        )

        mesh = get_mesh(domain_bounds, nrefine)
        element = get_element(mesh, order)
        basis = Basis(mesh, element)

        bndry_converter = ManufacturedSolutionToBoundaryConditions(
            man_sol, mesh, element, basis
        )
        bndry_conds = bndry_converter.boundary_conditions(bndry_types)

        physics = Helmholtz(
            mesh, element, basis, bndry_conds, wnum_fun, forc_fun
        )
        newton_solver = NewtonSolver(verbosity=0, maxiters=1, rtol=1e-10)
        solver = SteadyStatePDE(physics, newton_solver)

        fem_sol = solver.solve(physics.init_guess())
        exact_sol = basis.project(lambda x: sol_fun(x))
        error = solver.L2_error(exact_sol, fem_sol)
        assert error < 3e-8
        # print(fem_sol)
        # print(exact_sol)
        # print(fem_sol - exact_sol, "a")
        assert np.allclose(fem_sol, exact_sol)

    def test_helmholtz(self):
        # note react fun for advection diffusion reaction equation (wnum here)
        # must be size of u. so if reaction term depends on x it must be on
        # a single entry of x or a scalar function of all x, e.g sum(x[0])
        test_cases = [
            [[0, 1], 2, 1, "x", "4 + 1e-16*x", ["D", "D"]],
            [[0, 1], 2, 1, "x**2", "1*x", ["N", "R"]],
            [
                [0, 0.5, 0, 1],
                2,
                1,
                "y**2*x**2",
                "1+1e-16*x",
                ["D", "D", "R", "N"],
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
        mesh = get_mesh(domain_bounds, nrefine)
        element = {
            "u": ElementVector(get_element(mesh, 2)),
            "p": get_element(mesh, 1),
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
        domain_bounds: List[float],
        order: int,
        nrefine: int,
        bndry_types: List[str],
        sol_string: str,
        diff_tup: Tuple[str, str],
        vel_strings: str,
        react_tup: Tuple[str, callable, callable],
        tableau_name,
    ):
        nvars = len(domain_bounds) // 2
        react_str, react_fun, react_prime = react_tup
        linear_diff_str, nonlinear_diff_op_str, diff_fun, diff_prime = diff_tup
        man_sol = ManufacturedNonLinearAdvectionDiffusionReaction(
            sol_string,
            nvars,
            linear_diff_str,
            nonlinear_diff_op_str,
            react_str,
            vel_strings,
            conservative=False,
            oned=True,
        )
        sol_fun = FEMTransientScalarFunctionFromCallable(
            man_sol.functions["solution"]
        )
        vel_fun = FEMVectorFunctionFromCallable(
            man_sol.functions["velocity"], "velocity"
        )
        forc_fun = FEMTransientScalarFunctionFromCallable(
            man_sol.functions["forcing"], "forcing"
        )
        linear_diff_fun = FEMScalarFunctionFromCallable(
            man_sol.functions["linear_diffusion"], "linear_diffusion"
        )
        diff_op = FEMNonLinearOperatorFromCallable(diff_fun, diff_prime)
        react_op = FEMNonLinearOperatorFromCallable(react_fun, react_prime)

        mesh = get_mesh(domain_bounds, nrefine)
        element = get_element(mesh, order)
        basis = Basis(mesh, element)

        init_time = 0
        bndry_converter = ManufacturedSolutionToBoundaryConditions(
            man_sol, mesh, element, basis, time=init_time
        )
        bndry_conds = bndry_converter.boundary_conditions(bndry_types)
        print(man_sol)
        print(bndry_conds)

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
            init_time,
            final_time,
            newton_kwargs={"atol": 1e-8, "rtol": 1e-8, "maxiters": 2},
        )
        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sol_t = basis.project(lambda x: sol_fun(x))
            model_sol_t = sols[:, ii]
            L2error = solver.L2_error_at_a_single_time(
                exact_sol_t, model_sol_t
            )
            factor = np.sqrt(solver.integrate(exact_sol_t**2))
            print(time, L2error, 1e-8 * factor)
            assert L2error < 1e-8 * factor

    def test_transient_advection_diffusion_reaction(self):
        test_case_args = [
            [[0, 1]],
            [2],
            [2],
            # [["D", "D"]] , ["D", "N"]],
            [["D", "N"]],
            # ["(1-x)*x + 1e-16*T", "(1-x)*x*(1+T)"],
            ["(1-x)*x*(1+T)"],
            [
                (
                    "(4+1e-16*x)",
                    "u**0",
                    lambda x, u: 4 + u * 0,
                    lambda x, u: u * 0,
                )
            ],
            [["0+1e-16*x"], ["(1+x)/10"]],  # vel_strs
            [
                ("0*u", lambda x, u: 0 * u, lambda x, u: 0 * u),
                ("u**2", lambda x, u: u**2, lambda x, u: 2 * u),
            ],  # react_tup
            ["im_beuler1"],
        ]

        ii = 0
        for test_case in itertools.product(*test_case_args):
            print("####", ii)
            ii += 1
            self._check_transient_advection_diffusion_reaction(*test_case)

    def _setup_burgers_domain(
        self,
        man_sol: ManufacturedSolution,
        order: int,
        nrefine: int,
        domain_bounds: List[float],
        bndry_types: List[str],
        transient: bool,
    ):
        periodic = bndry_types == ["P", "P"]
        mesh = get_mesh(domain_bounds, nrefine, periodic)
        element = get_element(mesh, order)
        basis = Basis(mesh, element)

        if not periodic:
            bndry_converter = ManufacturedSolutionToBoundaryConditions(
                man_sol, mesh, element, basis, time=0.0 if transient else None
            )
            bndry_conds = bndry_converter.boundary_conditions(bndry_types)
        else:
            bndry_conds = BoundaryConditions(mesh, element, basis)
        return mesh, element, basis, bndry_conds

    def _check_steady_burgers(
        self,
        order: int,
        nrefine: int,
        domain_bounds: List[float],
        sol_string: str,
        viscosity_string: str,
        bndry_types: List[str],
    ):

        man_sol = ManufacturedBurgers1D(
            sol_string, viscosity_string, oned=True
        )
        viscosity_fun = FEMScalarFunctionFromCallable(
            man_sol.functions["viscosity"]
        )
        sol_fun = FEMScalarFunctionFromCallable(
            man_sol.functions["solution"], name="sol"
        )
        forc_fun = FEMScalarFunctionFromCallable(
            man_sol.functions["forcing"], name="forcing"
        )
        mesh, element, basis, bndry_conds = self._setup_burgers_domain(
            man_sol, order, nrefine, domain_bounds, bndry_types, False
        )
        physics = Burgers(
            mesh, element, basis, bndry_conds, viscosity_fun, forc_fun
        )
        init_sol = basis.project(lambda x: sol_fun(x) + 1)
        newton_solver = NewtonSolver(verbosity=2, maxiters=5, rtol=1e-12)
        solver = SteadyStatePDE(physics, newton_solver)
        fem_sol = solver.solve(init_sol)
        exact_sol = basis.project(lambda x: sol_fun(x))
        L2error = solver.L2_error(exact_sol, fem_sol)
        factor = np.sqrt(solver.integrate(exact_sol**2))
        print(L2error, 1e-8 * factor)
        assert L2error < 1e-8 * factor

    def test_steady_burgers(self):
        test_cases = [
            [2, 3, [0, 1], "x*(1.0-x)", "10+1e-16*x", ["D", "D"]],
            [2, 3, [0, 1], "x*(1.0-x)", "10+1e-16*x", ["N", "R"]],
            # periodic does not work with steady state. Not sure why.
            # but nor really needed, steady burgers was only
            # created as a stepping stone to transient burgers because it
            # enabled easier testing
            # periodic mesh and cosine needs a high nrefine to even get close
            # [2, 10, [0, 1], "cos(2*pi*x)", "10+1e-16*x", ["P", "P"]],
        ]
        for case in test_cases:
            self._check_steady_burgers(*case)

    def _check_transient_burgers(
        self,
        order: int,
        nrefine: int,
        domain_bounds: List[float],
        sol_string: str,
        viscosity_string: str,
        bndry_types: List[str],
    ):
        tableau_name = "im_beuler1"
        man_sol = ManufacturedBurgers1D(
            sol_string, viscosity_string, oned=True
        )
        viscosity_fun = FEMScalarFunctionFromCallable(
            man_sol.functions["viscosity"]
        )
        forc_fun = FEMTransientScalarFunctionFromCallable(
            man_sol.functions["forcing"], name="forcing"
        )
        sol_fun = FEMTransientScalarFunctionFromCallable(
            man_sol.functions["solution"], name="sol"
        )
        mesh, element, basis, bndry_conds = self._setup_burgers_domain(
            man_sol, order, nrefine, domain_bounds, bndry_types, True
        )
        physics = Burgers(
            mesh, element, basis, bndry_conds, viscosity_fun, forc_fun
        )

        deltat = 1.0
        final_time = deltat * 5
        sol_fun.set_time(0.0)
        init_sol = basis.project(lambda x: sol_fun(x))

        solver = TransientPDE(physics, deltat, tableau_name)
        sols, times = solver.solve(
            init_sol,
            0.0,
            final_time,
            newton_kwargs={"atol": 1e-8, "rtol": 1e-8, "maxiters": 2},
        )
        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sol_t = basis.project(lambda x: sol_fun(x))
            model_sol_t = sols[:, ii]
            L2error = solver.L2_error_at_a_single_time(
                exact_sol_t, model_sol_t
            )
            factor = np.sqrt(solver.integrate(exact_sol_t**2))
            print(time, L2error, 1e-8 * factor)
            assert L2error < 1e-8 * factor

    def test_transient_burgers(self):
        test_cases = [
            [2, 3, [0, 1], "x*(1.0-x)*(1+T)", "10+1e-16*x", ["D", "D"]],
            # periodic mesh and cosine needs a high nrefine to even get close
            [
                2,
                10,
                [0, 1],
                "cos(2*pi*x)*(1+T)",
                "10+1e-16*x",
                ["P", "P"],
            ],
        ]
        for case in test_cases:
            self._check_transient_burgers(*case)


if __name__ == "__main__":
    unittest.main(verbosity=2)
