import sys

from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Tuple, Union

import numpy as np

if "pyodide" in sys.modules:
    from scipy.sparse.base import spmatrix
else:
    from scipy.sparse import spmatrix

from skfem import asm, bmat, LinearForm, BilinearForm, Basis, solve, condense
from skfem.mesh import Mesh
from skfem.element import Element
from skfem.helpers import dot, grad, mul
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence
from pyapprox.pde.galerkin.util import (
    forcing_linearform,
    vector_forcing_linearform,
)
from pyapprox.pde.galerkin.functions import (
    FEMScalarFunction,
    FEMVectorFunction,
    FEMNonLinearOperator,
    FEMFunctionTransientMixin,
    FEMScalarFunctionFromCallable,
    FEMVectorFunctionFromCallable,
    FEMNonLinearOperatorFromCallable,
)
from pyapprox.util.backends.template import Array


class BoundaryConditions:
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        dirichlet_bndry_names: List[str] = None,
        dirichlet_bndry_funs: List[str] = None,
        neumann_bndry_names: List[str] = None,
        neumann_bndry_funs: List[str] = None,
        robin_bndry_names: List[str] = None,
        robin_bndry_funs: List[str] = None,
        robin_bndry_constants: List[str] = None,
    ):
        # Specifying neumann_bndry_funs as None but not providing boundary
        # conditions for all boundaries mean that the natural Neumann boundary
        # condition will be applied at bondaries which do not have a condition
        # specified. The natural Neumann condition is that the normal flux is zero

        # see https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html
        # for useful notes
        self._mesh = mesh
        self._element = element
        self._basis = basis
        self._check_bndry_arguments(
            dirichlet_bndry_names, dirichlet_bndry_funs, "Dirichlet"
        )
        self._dbndry_names = dirichlet_bndry_names
        self._dbndry_funs = dirichlet_bndry_funs
        self._check_bndry_arguments(
            neumann_bndry_names, neumann_bndry_funs, "Neumann"
        )
        self._nbndry_names = neumann_bndry_names
        self._nbndry_funs = neumann_bndry_funs
        self._check_bndry_arguments(
            robin_bndry_names, robin_bndry_funs, "Robin", robin_bndry_constants
        )
        self._rbndry_names = robin_bndry_names
        self._rbndry_funs = robin_bndry_funs
        self._rbndry_consts = robin_bndry_constants

    def _check_bndry_arguments(
        self,
        bndry_names: List[str],
        bndry_funs: List[str],
        bndry_type: str,
        bndry_consts: List[float] = None,
    ):
        if bndry_consts is None and bndry_names is None and bndry_funs is None:
            return

        if bndry_names is None or bndry_funs is None:
            raise ValueError(
                f"Boundary names must be provided for all {bndry_type} "
                "boundaries"
            )
        if bndry_type == "Robin" and bndry_consts is None:
            raise ValueError(
                f"Boundary constants must be provided for all {bndry_type} "
                "boundaries"
            )

        if len(bndry_funs) != len(bndry_names):
            raise ValueError(
                f"must provide a function for each {bndry_type} boundary"
            )
        if bndry_type == "Robin" and len(bndry_consts) != len(bndry_names):
            raise ValueError(
                f"must provide a constant for each {bndry_type} boundary"
            )

    def ndirichlet_boundaries(self) -> int:
        return 0 if self._dbndry_names is None else len(self._dbndry_names)

    def nneumann_boundaries(self) -> int:
        """
        Number of non-natural neumann BCs, i.e. those that require modification
        of weak form
        """
        return 0 if self._nbndry_names is None else len(self._nbndry_names)

    def nrobin_boundaries(self) -> int:
        return 0 if self._rbndry_names is None else len(self._rbndry_names)

    def dirichlet_dofs_and_vals(self, uprev: Array):
        D_vals = self._basis.zeros()
        if self.ndirichlet_boundaries() == 0:
            return uprev if uprev is None else D_vals, np.empty(0, dtype=int)
        D_dofs = self._basis.get_dofs(self._dbndry_names)
        for bndry_name, bndry_fun in zip(
            self._dbndry_names, self._dbndry_funs
        ):
            # bndry_basis = self._basis.boundary(bndry_name)
            bndry_dofs = self._basis.get_dofs(bndry_name)
            D_vals[bndry_dofs] = bndry_fun(self._basis.doflocs[:, bndry_dofs])
        if uprev is not None:
            D_vals = uprev - D_vals
        return D_vals, D_dofs

    def _neumann_linear_form(self, v: Array, w: dict) -> Array:
        return w["fun"] * v

    def impose_neumann_boundaries(self, vec: Array) -> Array:
        if self.nneumann_boundaries() == 0:
            return vec
        for bndry_name, bndry_fun in zip(
            self._nbndry_names, self._nbndry_funs
        ):
            bndry_basis = self._basis.boundary(bndry_name)
            vec += asm(
                LinearForm(self._neumann_linear_form),
                bndry_basis,
                fun=bndry_basis.project(bndry_fun),
            )
        return vec

    def _robin_bilinear_form(self, u: Array, v: Array, w: dict) -> Array:
        return w["alpha"] * u * v

    def _robin_linear_form(self, v: Array, w: Array) -> Array:
        return w["alpha"] * w["uprev"] * v

    def impose_robin_boundaries(
        self, mat: Array, vec: Array, uprev: Array
    ) -> Tuple[Array, Array]:
        if self.nrobin_boundaries() == 0:
            return mat, vec
        for bndry_name, bndry_fun, bndry_const in zip(
            self._rbndry_names, self._rbndry_funs, self._rbndry_consts
        ):
            bndry_basis = self._basis.boundary(bndry_name)
            # robin is assumed to take the form -flux = alpha*u + fun(x)
            mat += asm(
                BilinearForm(self._robin_bilinear_form),
                bndry_basis,
                alpha=bndry_const,
            )
            # add contribution of robin boundary to rhs
            vec += asm(
                LinearForm(self._neumann_linear_form),
                bndry_basis,
                fun=bndry_basis.project(bndry_fun),
            )
            if uprev is None:
                continue
            # when uprev is not None. then linear vec represents the residual
            # and so robin contribution to mass term must be reflected in res
            vec -= asm(
                LinearForm(self._robin_linear_form),
                bndry_basis,
                alpha=bndry_const,
                uprev=uprev,
            )
        return mat, vec

    def set_time(self, time: float):
        if self.ndirichlet_boundaries() > 0:
            for bndry_fun in self._dbndry_funs:
                if isinstance(bndry_fun, FEMFunctionTransientMixin):
                    bndry_fun.set_time(time)
        if self.nneumann_boundaries() > 0:
            for bndry_fun in self._nbndry_funs:
                if isinstance(bndry_fun, FEMFunctionTransientMixin):
                    bndry_fun.set_time(time)
        if self.nrobin_boundaries() > 0:
            for bndry_fun in self._rbndry_funs:
                if isinstance(bndry_fun, FEMFunctionTransientMixin):
                    bndry_fun.set_time(time)

    def __repr__(self) -> str:
        return "{0}(dirichlet={1}, neumann={2}, robin={3})".format(
            self.__class__.__name__,
            self._dbndry_names,
            self._nbndry_names,
            self._rbndry_names,
        )


class DiffusionResidual(ABC):
    def _zero_fun(self, x, *args):
        return x[0] * 0

    def _zero_vel_fun(self, x, *args):
        return x * 0

    def _advection_term(self, u, v, w):
        # this is for non-conservative form of advection
        print(w.x.shape)
        vel = self._vel_fun(w.x)
        # du = u.grad
        # return sum([v*vel[ii]*du[ii] for ii in range(w.x.shape[0])])
        return dot(v * vel, grad(u))

    @abstractmethod
    def linear_form(self, v, w):
        raise NotImplementedError

    @abstractmethod
    def bilinear_form(self, u, v, w):
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class LinearDiffusionResidual(DiffusionResidual):
    def __init__(self, forc_fun, diff_fun, react_fun, vel_fun):
        # for use with direct solvers, i.e. not in residual form,
        # typically used for computing initial guess for newton solve
        self._forc_fun = forc_fun
        self._diff_fun = diff_fun
        if react_fun is None:
            react_fun = self._zero_fun
        self._react_fun = react_fun
        if vel_fun is None:
            vel_fun = self._zero_vel_fun
        self._vel_fun = vel_fun
        self.__name__ = self.__class__.__name__

    def linear_form(self, v, w):
        # TODO instead of setting functions to zero if they are passed in as None
        # do not add them to form to avoid unnecessary computations
        forc = self._forc_fun(w.x)
        diff = self._diff_fun(w.x)
        react = self._react_fun(w.x)
        return (
            forc * v
            - dot(diff * grad(w.u_prev), grad(v))
            + react * v
            - self._advection_term(w.u_prev, v, w)
        )

    def bilinear_form(self, u, v, w):
        diff = self._diff_fun(w.x)
        react = self._react_fun(w.x)
        return (
            dot(diff * grad(u), grad(v))
            - react * u * v
            + self._advection_term(u, v, w)
        )


class NonLinearDiffusionResidual(DiffusionResidual):
    def __init__(
        self,
        forc_fun,
        react_fun,
        react_prime,
        nl_diff_fun,
        nl_diff_prime,
        vel_fun,
    ):
        self._forc_fun = forc_fun
        self._nl_diff_fun = nl_diff_fun
        if nl_diff_prime is None:
            nl_diff_prime = self._zero_fun
        self._nl_diff_prime = nl_diff_prime
        if react_fun is None:
            react_fun = self._zero_fun
            react_prime = None
        if vel_fun is None:
            vel_fun = self._zero_vel_fun
        self._vel_fun = vel_fun
        self._react_fun = react_fun
        if react_prime is None:
            react_prime = self._zero_fun
        self._react_prime = react_prime
        self.__name__ = self.__class__.__name__

    def linear_form(self, v, w):
        # this is actually the residual R of du/dt = R(u)
        forc = self._forc_fun(w.x)
        diff = self._nl_diff_fun(w.x, w.u_prev)
        react = self._react_fun(w.x, w.u_prev)
        return (
            forc * v
            - dot(diff * grad(w.u_prev), grad(v))
            + react * v
            - self._advection_term(w.u_prev, v, w)
        )

    def bilinear_form(self, u, v, w):
        diff = self._nl_diff_fun(w.x, w.u_prev)
        diff_prime = self._nl_diff_prime(w.x, w.u_prev)
        react_prime = self._react_prime(w.x, w.u_prev)
        mat = (
            dot(diff * grad(u), grad(v))
            + dot(diff_prime * u * grad(w.u_prev), grad(v))
            - react_prime * u * v
            + self._advection_term(u, v, w)
        )
        return mat


class Physics(ABC):
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        bndry_conds: BoundaryConditions,
    ):
        if not isinstance(mesh, Mesh):
            raise ValueError("mesh must be an instance of Mesh")
        if not isinstance(element, Element) and not isinstance(element, dict):
            raise ValueError("element must be an instance of Element")
        if not isinstance(basis, Basis) and not isinstance(element, dict):
            raise ValueError("basis must be an instance of Basis")

        if isinstance(bndry_conds, list):
            for bc in bndry_conds:
                if not isinstance(bc, BoundaryConditions):
                    raise ValueError(
                        "bndry_conds must be a list BoundaryConditions"
                    )
        else:
            if not isinstance(bndry_conds, BoundaryConditions):
                raise ValueError(
                    "bndry_conds must be an instance of BoundaryConditions"
                )

        self._mesh = mesh
        self._element = element
        self._basis = basis
        self._bndry_conds = bndry_conds
        self._funs = self._set_funs()

    def _set_funs(self) -> List:
        return []

    @abstractmethod
    def raw_assemble(
        self, sol: np.ndarray
    ) -> Tuple[spmatrix, Union[np.ndarray, spmatrix], np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def apply_boundary_conditions(
        self,
        sol: np.ndarray,
        bilinear_mat: spmatrix,
        linear_vec: Union[np.ndarray, spmatrix],
    ) -> Tuple[spmatrix, Union[np.ndarray, spmatrix], np.ndarray, np.ndarray]:
        linear_vec = self._bndry_conds.impose_neumann_boundaries(linear_vec)
        bilinear_mat, linear_vec = self._bndry_conds.impose_robin_boundaries(
            bilinear_mat, linear_vec, sol
        )
        dirichlet_vals, dirichlet_dofs = (
            self._bndry_conds.dirichlet_dofs_and_vals(sol)
        )
        return bilinear_mat, linear_vec, dirichlet_vals, dirichlet_dofs

    def assemble(
        self, sol: np.ndarray = None
    ) -> Tuple[spmatrix, Union[np.ndarray, spmatrix], np.ndarray, np.ndarray]:
        return self.apply_boundary_conditions(sol, *self.raw_assemble(sol))

    def _transient_residual(self, sol: np.ndarray, time: float):
        # correct equations for boundary conditions
        self._set_time(time)
        res, jac = self._raw_residual(sol)
        if jac is None:
            assert self._auto_jac
        return res, jac

    def mass_matrix(self) -> Tuple[spmatrix]:
        mass_mat = asm(mass, self._basis)
        return mass_mat

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class AdvectionDiffusionReaction(Physics):
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        bndry_conds: BoundaryConditions,
        forc_fun: FEMScalarFunction,
        vel_fun: FEMVectorFunction,
    ):
        if not isinstance(forc_fun, FEMScalarFunction):
            raise ValueError(
                "forc_fun must be an instance of FEMScalarFunction"
            )
        if not isinstance(vel_fun, FEMVectorFunction):
            raise ValueError(
                "forc_fun must be an instance of FEMVectorFunction"
            )
        self._forc_fun = forc_fun
        self._vel_fun = vel_fun
        super().__init__(mesh, element, basis, bndry_conds)

    def _set_funs(self) -> List:
        return [self._vel_fun, self._forc_fun]

    def raw_assemble(
        self,
        sol: np.ndarray,
    ) -> Tuple[spmatrix, Union[np.ndarray, spmatrix]]:
        residual = self._setup_residual()
        u_prev_interp = self._basis.interpolate(sol)
        bilinear_mat = asm(
            BilinearForm(residual.bilinear_form),
            self._basis,
            u_prev=u_prev_interp,
        )
        residual_vec = asm(
            LinearForm(residual.linear_form), self._basis, u_prev=u_prev_interp
        )
        return bilinear_mat, residual_vec


class LinearAdvectionDiffusionReaction(AdvectionDiffusionReaction):
    # Only to be used to get initial guess for nonlinear advection diffusion
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        bndry_conds: BoundaryConditions,
        diff_fun: FEMScalarFunction,
        forc_fun: FEMScalarFunction,
        vel_fun: FEMVectorFunction,
        react_fun: Optional[FEMNonLinearOperator] = None,
    ):
        if not isinstance(diff_fun, FEMScalarFunction):
            raise TypeError(
                "diff_fun must be an instance of FEMScalarFunction "
                f"but was {diff_fun.__class__.__name__}"
            )
        if react_fun is not None and not isinstance(
            react_fun, FEMScalarFunction
        ):
            raise TypeError(
                "react_fun must be an instance of FEMScalarFunction "
                f"but was {react_fun.__class__.__name__}"
            )
        self._diff_fun = diff_fun
        self._react_fun = react_fun
        super().__init__(mesh, element, basis, bndry_conds, forc_fun, vel_fun)

    def _setup_residual(self):
        return LinearDiffusionResidual(
            self._forc_fun, self._diff_fun, self._react_fun, self._vel_fun
        )

    def _set_funs(self) -> List:
        funs = super()._set_funs()
        return funs + [self._react_fun, self._forc_fun]

    def init_guess(self) -> np.ndarray:
        return np.ones((self._basis.N,))


# TODO allow anisotropic diffusion
# see https://github.com/kinnala/scikit-fem/discussions/923
# @BilinearForm
# def anipoisson(u, v, w):
#     C = np.array([[1.0, 0.1],
#                   [0.1, 1.0]])
#     return dot(mul(C, grad(u)), grad(v))


class NonLinearAdvectionDiffusionReaction(AdvectionDiffusionReaction):
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        bndry_conds: BoundaryConditions,
        linear_diff_fun: FEMScalarFunction,
        forc_fun: FEMScalarFunction,
        vel_fun: FEMVectorFunction,
        diff_op: FEMNonLinearOperator = None,
        react_op: FEMNonLinearOperator = None,
    ):
        # used to generate an initial guess
        if not isinstance(linear_diff_fun, FEMScalarFunction):
            raise ValueError(
                "linear_diff_fun must be an instance of FEMScalarFunction"
            )
        if not isinstance(diff_op, FEMNonLinearOperator):
            raise ValueError(
                "diff_op must be an instance of FEMNonLinearOperator"
            )
        if not isinstance(react_op, FEMNonLinearOperator):
            raise ValueError(
                "react_op must be an instance of FEMNonLinearOperator"
            )
        self._linear_diff_fun = linear_diff_fun
        self._diff_op = diff_op
        self._react_op = react_op
        super().__init__(mesh, element, basis, bndry_conds, forc_fun, vel_fun)

    def _setup_residual(self):
        return NonLinearDiffusionResidual(
            self._forc_fun,
            self._react_op.__call__,
            self._react_op.jacobian,
            self._diff_op.__call__,
            self._diff_op.jacobian,
            self._vel_fun,
        )

    def _set_funs(self) -> List:
        funs = super()._set_funs()
        return funs + [self._linear_diff_fun, self._diff_op, self._react_op]


class Helmholtz(NonLinearAdvectionDiffusionReaction):
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        bndry_conds: BoundaryConditions,
        wave_number_fun: Callable[[np.ndarray], np.ndarray],
        forc_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):

        # if len(bndry_conds[1]) > 0 or len(bndry_conds[1]) > 0:
        #     raise NotImplementedError(
        #         "Currently tests do not pass with Robin or Neumann BCs"
        #     )
        if forc_fun is None:
            forc_fun = FEMScalarFunctionFromCallable(
                self._zero_forcing, "forcing"
            )

        super().__init__(
            mesh,
            element,
            basis,
            bndry_conds,
            FEMScalarFunctionFromCallable(self._unit_diff, "diffusion"),
            forc_fun,
            FEMVectorFunctionFromCallable(self._zero_vel, "velocity", False),
            FEMNonLinearOperatorFromCallable(
                lambda x, u: 0 * u + 1.0, lambda x, u: 0 * u
            ),
            wave_number_fun,
        )

    def init_guess(self) -> Array:
        return np.ones((self._basis.N,))

    def _unit_diff(self, x: Array) -> Array:
        # only used if needing to generate a linear initial guess
        # which will not be needed here as PDE is linear
        # negative is taken because advection diffusion code solves
        # -k*\nabla^2 u + c*u = f
        # but Helmholtz solves
        # \nabla^2 u + c*u = 0 which is equivalent when k=-1
        return -1.0 + 0.0 * x

    def _zero_vel(self, x: Array) -> Array:
        # Helmholtz is a special case of the advection diffusion reaction
        # equation when there is no advection, i.e. velocity field is zero
        return 0.0 * x

    def _zero_forcing(self, x: Array) -> Array:
        return 0.0 * x[0]


class Stokes(Physics):
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        bndry_conds: List[BoundaryConditions],
        navier_stokes: bool,
        vel_forc_fun: FEMVectorFunction,
        pres_forc_fun: FEMScalarFunction,
        viscosity: float = 1.0,
    ):
        super().__init__(mesh, element, basis, bndry_conds)

        if not isinstance(vel_forc_fun, FEMVectorFunction):
            raise ValueError(
                "vel_forc_fun must be an instance of FEMVectorFunction"
            )
        if not isinstance(pres_forc_fun, FEMScalarFunction):
            raise ValueError(
                "pres_fun must be an instance of FEMScalarFunction"
            )
        self._vel_forc_fun = vel_forc_fun
        self._pres_forc_fun = pres_forc_fun
        self._navier_stokes = navier_stokes
        self._viscosity = viscosity

    def _navier_stokes_linearized_terms(self, u, v, w):
        z = w["u_prev"]
        dz = z.grad
        du = u.grad
        if u.shape[0] == 2:
            return (
                v[0] * (u[0] * dz[0][0] + u[1] * dz[0][1])
                + v[1] * (u[0] * dz[1][0] + u[1] * dz[1][1])
                + v[0] * (z[0] * du[0][0] + z[1] * du[0][1])
                + v[1] * (z[0] * du[1][0] + z[1] * du[1][1])
            )
        if u.shape[0] == 1:
            return v[0] * (u[0] * dz[0][0]) + v[0] * (z[0] * du[0][0])

    def _navier_stokes_nonlinear_term_residual(self, v, w):
        u = w["u_prev"]
        du = u.grad
        if u.shape[0] == 2:
            return v[0] * (u[0] * du[0][0] + u[1] * du[0][1]) + v[1] * (
                u[0] * du[1][0] + u[1] * du[1][1]
            )
        if u.shape[0] == 1:
            return v[0] * (u[0] * du[0][0])
        raise ValueError("Only 1D and 2D navier stokes supported")

    def _assemble_stokes_linear_vec(self, basis, vel_forc_fun, pres_forc_fun):
        pres_forc = basis["p"].interpolate(basis["p"].project(pres_forc_fun))
        vel_forces = basis["u"].interpolate(basis["u"].project(vel_forc_fun))
        vel_loads = asm(
            LinearForm(vector_forcing_linearform), basis["u"], forc=vel_forces
        )
        # - sign on pressure load because I think pressure equation
        # in manufactured_solutions.py has the wrong sign it is \nabla\cdot u = f_p
        # but convention is -\nabla\cdot u = f_p
        linear_vec = np.concatenate(
            [
                vel_loads,
                -asm(
                    LinearForm(forcing_linearform), basis["p"], forc=pres_forc
                ),
            ]
        )
        return linear_vec

    def _raw_assemble_stokes(
        self,
        vel_forc_fun,
        pres_forc_fun,
        navier_stokes,
        mesh,
        element,
        basis,
        u_prev=None,
        return_K=False,
        viscosity=1,
    ):
        A = viscosity * asm(vector_laplace, basis["u"])
        B = -asm(divergence, basis["u"], basis["p"])

        # C = 1e-6*asm(mass, basis['p'])
        C = None
        K = bmat([[A, B.T], [B, C]], "csr")

        if u_prev is not None:
            vel_prev, pres_prev = np.split(u_prev, K.blocks)

        if navier_stokes:
            A_nl = asm(
                BilinearForm(self._navier_stokes_linearized_terms),
                basis["u"],
                u_prev=vel_prev,
            )
            bilinear_mat = bmat([[(A + A_nl), B.T], [B, C]], "csr")
        else:
            bilinear_mat = K  # jacobian

        linear_vec = self._assemble_stokes_linear_vec(
            basis, vel_forc_fun, pres_forc_fun
        )

        if u_prev is not None:
            # minus sign because res = -a(u_prev, v) + L(v), b = res
            b_pres = -B.dot(vel_prev)
            b_vel = -(A.dot(vel_prev) + B.T.dot(pres_prev))
            if C is not None:
                b_pres -= C.dot(pres_prev)
            if navier_stokes:
                b_vel += -asm(
                    LinearForm(self._navier_stokes_nonlinear_term_residual),
                    basis["u"],
                    u_prev=vel_prev,
                )
            linear_vec += np.concatenate([b_vel, b_pres])
            # the following is true only for stokes (not navier stokes)
            # linear_vec -= K.dot(u_prev)

        if not return_K:
            return bilinear_mat, linear_vec
        return bilinear_mat, linear_vec, K

    def _assemble_linear_stokes(
        self,
        vel_forc_fun,
        pres_forc_fun,
        navier_stokes,
        bndry_conds,
        mesh,
        element,
        basis,
        u_prev=None,
        return_K=False,
        viscosity=1.0,
    ):
        result = self._raw_assemble_stokes(
            vel_forc_fun,
            pres_forc_fun,
            navier_stokes,
            mesh,
            element,
            basis,
            u_prev,
            return_K,
            viscosity,
        )
        bilinear_mat, linear_vec = result[:2]
        bilinear_mat, linear_vec, D_vals, D_dofs = (
            self._enforce_stokes_boundary_conditions(
                mesh,
                element,
                basis,
                bilinear_mat,
                linear_vec,
                bndry_conds,
                u_prev,
            )
        )
        if not return_K:
            return bilinear_mat, linear_vec, D_vals, D_dofs
        return bilinear_mat, linear_vec, D_vals, D_dofs, result[-1]

    def init_guess(self) -> np.ndarray:
        bilinear_mat, linear_vec, D_vals, D_dofs = (
            self._assemble_linear_stokes(
                self._vel_forc_fun,
                self._pres_forc_fun,
                False,
                self._bndry_conds,
                self._mesh,
                self._element,
                self._basis,
                return_K=False,
                viscosity=self._viscosity,
            )
        )
        return solve(*condense(bilinear_mat, linear_vec, x=D_vals, D=D_dofs))

    def raw_assemble(
        self, sol: Optional[np.ndarray] = None
    ) -> Tuple[spmatrix, Union[np.ndarray, spmatrix], np.ndarray, np.ndarray]:
        bilinear_mat, linear_vec = self._raw_assemble_stokes(
            self._vel_forc_fun,
            self._pres_forc_fun,
            self._navier_stokes,
            self._mesh,
            self._element,
            self._basis,
            sol,
            viscosity=self._viscosity,
        )
        return bilinear_mat, linear_vec

    def _enforce_stokes_boundary_conditions(
        self,
        mesh,
        element,
        basis,
        bilinear_mat,
        linear_vec,
        bndry_conds,
        u_prev=None,
    ):

        # Note When all boundaries are dirichlet must set a pressure value to
        # make presure unique. Otherwise it is only unique up to a constant
        # When not all boundaries are dirichlet  pressure is made unique because
        # it appears in the boundary integral arising from integration by parts
        # of pressure term in weak form of stokes equations

        nvars = mesh.p.shape[0]
        D_vals = np.hstack([basis["u"].zeros(), basis["p"].zeros()])
        vel_D_dofs = []
        # loop over velocity component boundary conditions
        for idx in range(nvars):
            for bndry_name, bndry_fun in zip(
                bndry_conds[idx]._dbndry_names, bndry_conds[idx]._dbndry_funs
            ):
                # bndry_basis = self._basis.boundary(bndry_name)
                # can use basis["u"].get_dofs(bndry_name) because u is first
                # in flattened list of dofs.

                # warning appending dofs like this may cause repeated entries at corners
                # and I am nor sure what effect this will have
                if nvars == 2:
                    dofnames = basis["u"].get_dofs().obj.element.dofnames
                    skip = dofnames[nvars - idx - 1]
                    bndry_dofs = basis["u"].get_dofs(bndry_name, skip=skip)
                elif nvars == 1:
                    bndry_dofs = basis["u"].get_dofs(bndry_name)
                else:
                    raise NotImplementedError("nvars must be <= 2")
                vel_D_dofs.append(bndry_dofs)
                D_vals[bndry_dofs] = bndry_fun(
                    basis["u"].doflocs[:, bndry_dofs]
                )

        # set pressure boundary conditions
        if bndry_conds[nvars].ndirichlet_boundaries() > 0:
            pres_D_dofs = []
            for bndry_name, bndry_fun in zip(
                bndry_conds[nvars]._dbndry_names,
                bndry_conds[nvars]._dbndry_funs,
            ):
                # Accesing p requires adding
                # basis["p"].get_dofs(bndry_name) and the total number of u basis dofs
                #  basis["u"].N is nrows of velocity block in stiffness matrix
                # need to get DOF for presure in global array
                p_dofs = basis["p"].get_dofs(bndry_name)
                shifted_p_dofs = p_dofs.flatten() + basis["u"].N
                pres_D_dofs.append(shifted_p_dofs)
                D_vals[shifted_p_dofs] = bndry_fun(
                    basis["p"].doflocs[:, p_dofs]
                )

            pres_D_dofs = np.hstack(pres_D_dofs)
            # create a copy DOFView that can
            # D_dofs = copy.deepcopy(vel_D_dofs[0])
            # D_dofs.flatten = lambda: np.union1d(
            #     np.unique(np.hstack(vel_D_dofs)), pres_D_dofs
            # )
            # typically D_dofs is a DOFView but if D_dofs is only used for
            # condense then we can just pass array. I can creat a view like above
            # but flatten will be inconsistent with the rest of its attributes an
            # d functions
            D_dofs = np.union1d(np.unique(np.hstack(vel_D_dofs)), pres_D_dofs)
        else:
            D_dofs = np.unique(np.hstack(vel_D_dofs))

        if u_prev is not None:
            D_vals = u_prev - D_vals
        return bilinear_mat, linear_vec, D_vals, D_dofs

    def apply_boundary_conditions(
        self,
        sol: np.ndarray,
        bilinear_mat: spmatrix,
        linear_vec: Union[np.ndarray, spmatrix],
    ) -> Tuple[spmatrix, Union[np.ndarray, spmatrix], np.ndarray, np.ndarray]:
        for bc in self._bndry_conds:
            if bc.nneumann_boundaries() + bc.nrobin_boundaries() > 0:
                raise NotImplementedError(
                    "Stokes only supports Dirichlet and natural Neuammn BCs"
                )
        bilinear_mat, linear_vec, D_vals, D_dofs = (
            self._enforce_stokes_boundary_conditions(
                self._mesh,
                self._element,
                self._basis,
                bilinear_mat,
                linear_vec,
                self._bndry_conds,
                sol,
            )
        )
        return bilinear_mat, linear_vec, D_vals, D_dofs


def _diffusion_reaction(u, v, w):
    return dot(mul(w["diff"], grad(u)), grad(v)) + w["react"] * u * v


class BiLaplacianPrior:
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        gamma: int,
        delta: int,
        anisotropic_tensor: Optional[np.ndarray] = None,
    ):
        r"""
        :math:`\delta\gamma` controls the variance of the prior
        :math:`\frac{\gamma}{\delta}` controls the correlation length
        The smaller the raio the smaller the correlation length
        Anisotropic tensor K controls correlation length in each direction
        E.g. in 2D K = [[K11, K12], [K21, K22]]
        K22<<K11 will cause correlation length to be small in vertical
        direction and long in horizontal direction
        """
        self._mesh = mesh
        self._element = element
        self._basis = basis
        self._gamma = gamma
        self._delta = delta
        self._beta = np.sqrt(self._gamma * self._delta) * 1.42
        if anisotropic_tensor is None:
            anisotropic_tensor = np.eye(mesh.p.shape[0]) * gamma
        else:
            anisotropic_tensor *= gamma
        if anisotropic_tensor.shape != (mesh.p.shape[0], mesh.p.shape[0]):
            raise ValueError("anisotropic_tensor has incorrect shape")
        self.anisotropic_tensor = anisotropic_tensor
        self._bndry_conds = BoundaryConditions(
            self._mesh,
            self._element,
            self._basis,
            robin_bndry_names=[key for key in self._mesh.boundaries.keys()],
            robin_bndry_funs=[
                FEMScalarFunctionFromCallable(self._bndry_fun, name=key)
                for key in self._mesh.boundaries.keys()
            ],
            robin_bndry_constants=[
                self._beta for key in self._mesh.boundaries.keys()
            ],
        )
        self._linear_system_data = None

    def _setup_linear_system(self):
        if self._linear_system_data is None:
            self._linear_system_data = list(
                self._assemble_stiffness(
                    self._diff_fun,
                    self._react_fun,
                    self._bndry_conds,
                    self._mesh,
                    self._element,
                    self._basis,
                )
            )
            mass_mat = asm(mass, self._basis)
            lumped_mass_mat = np.asarray(mass_mat.sum(axis=1))[:, 0]
            self._linear_system_data += [lumped_mass_mat]

        return self._linear_system_data

    def _bndry_fun(self, x):
        # x : np.ndarray (nvars, nelems, nbndry_dofs_per_elem)
        # dofs are Lagrange basis nodes not quad points
        return 0 * x[0]

    def _zero_forcing(self, v, w):
        return 0 * v

    def _react_fun(self, x):
        # x : np.ndarray (nvars, nelems, nquad_pts_per_elem)
        return self._delta + 0 * x[0]

    def _diff_fun(self, x):
        # x (nvars, nelems, nquad_pts_per_elem)
        return np.full((x.shape[1:]), self._gamma)

    def _assemble_stiffness(
        self, diff_fun, react_fun, bndry_conds, mesh, element, basis
    ):
        react_proj = basis.project(react_fun)
        react = basis.interpolate(react_proj)
        bilinear_mat = asm(
            BilinearForm(_diffusion_reaction),
            basis,
            diff=self.anisotropic_tensor,
            react=react,
        )
        linear_vec = asm(LinearForm(self._zero_forcing), basis)
        return self._bndry_conds.impose_robin_boundaries(
            bilinear_mat, linear_vec, None
        )

    def rvs(self, nsamples):
        bilinear_mat, linear_vec, lumped_mass_mat = self._setup_linear_system()
        white_noise = np.random.normal(
            0, 1, (lumped_mass_mat.shape[0], nsamples)
        )
        samples = np.empty((lumped_mass_mat.shape[0], nsamples))
        for ii in range(nsamples):
            rhs = np.sqrt(lumped_mass_mat) * white_noise[:, ii]
            samples[:, ii] = solve(
                *condense(bilinear_mat, rhs, D=np.empty(0, dtype=int))
            )
        return samples


class BurgersResidual:
    def __init__(self, forc_fun, viscosity_fun):
        # for use with direct solvers, i.e. not in residual form,
        # typically used for computing initial guess for newton solve
        self._forc_fun = forc_fun
        self._viscosity_fun = viscosity_fun
        self.__name__ = self.__class__.__name__

    def _advection_term(self, u, v):
        du = u.grad[0]
        return v * u * du

    def linear_form(self, v, w):
        forc = self._forc_fun(w.x)
        viscosity = self._viscosity_fun(w.x)
        return (
            forc * v
            - dot(viscosity * grad(w.u_prev), grad(v))
            - self._advection_term(w.u_prev, v)
        )

    def bilinear_form(self, u, v, w):
        # quasilinear burgers form derived from conservative form
        # u_t + (u(x)^2/2)_x
        # using the chain rule
        # g(u(x))=u(x)^2/2 : g(y) = y^2/2 dg/dy = y
        # dg/dx = dg/du(u(x))du(x)/dx = u(x)du(x)/dx
        viscosity = self._viscosity_fun(w.x)
        return (
            dot(viscosity * grad(u), grad(v))
            + v * w.u_prev * u.grad[0]
            + v * u * w.u_prev.grad[0]
        )


class Burgers(Physics):
    def __init__(
        self,
        mesh: Mesh,
        element: Element,
        basis: Basis,
        bndry_conds: BoundaryConditions,
        viscosity_fun: FEMScalarFunction,
        forc_fun: FEMScalarFunction,
    ):
        if not isinstance(viscosity_fun, FEMScalarFunction):
            raise ValueError(
                "viscosity_fun must be an instance of FEMScalarFunction"
            )
        if not isinstance(forc_fun, FEMScalarFunction):
            raise ValueError(
                "forc_fun must be an instance of FEMScalarFunction"
            )
        self._viscosity_fun = viscosity_fun
        self._forc_fun = forc_fun
        super().__init__(mesh, element, basis, bndry_conds)

    def _set_funs(self) -> List:
        return [self._viscosity_fun, self._forc_fun]

    def raw_assemble(
        self, sol: np.ndarray = None
    ) -> Tuple[spmatrix, Union[np.ndarray, spmatrix]]:
        residual = BurgersResidual(self._forc_fun, self._viscosity_fun)
        u_prev_interp = self._basis.interpolate(sol)
        bilinear_mat = asm(
            BilinearForm(residual.bilinear_form),
            self._basis,
            u_prev=u_prev_interp,
        )
        linear_vec = asm(
            LinearForm(residual.linear_form), self._basis, u_prev=u_prev_interp
        )
        return bilinear_mat, linear_vec

    def _unit_vel_fun(self, x, *args):
        return x * 0 + 1
