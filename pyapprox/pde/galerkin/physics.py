import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Tuple, Union
import sys
if "pyodide" in sys.modules:
    from scipy.sparse.base import spmatrix
else:
    from scipy.sparse import spmatrix

from skfem import (
    asm, bmat, LinearForm, BilinearForm, FacetBasis, Basis, solve, condense)
from skfem.mesh import Mesh
from skfem.element import Element
from skfem.helpers import dot, grad, mul
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence
from pyapprox.pde.galerkin.util import (
    _forcing, _vector_forcing, _vector_fun_to_skfem_vector_fun,
    _robin, _robin_prev_sol)


def _enforce_dirichlet_scalar_boundary_conditions(
        mesh, element, basis, bilinear_mat, linear_vec,
        D_bndry_conds, u_prev):
    """
    u_prev is none indicates that newtons method is calling enforce boundaries
    In this case linear_vec represents the residual and so contributions
    to residual must be added accordingly
    """
    # define Dirichlet boundary basis
    D_bases = [
        FacetBasis(mesh, element, facets=mesh.boundaries[key])
        for key, fun in D_bndry_conds.items()]
    # evaluate Dirichlet boundary values
    D_vals = basis.zeros()
    for b, key in zip(D_bases, D_bndry_conds.keys()):
        _dofs = basis.get_dofs(key)
        D_vals[_dofs] = b.project(D_bndry_conds[key][0])[_dofs]

    if u_prev is not None:
        D_vals = u_prev-D_vals
    # get dofs on Dirichlet boundary
    if len(D_bndry_conds) > 0:
        D_dofs = basis.get_dofs(list(D_bndry_conds.keys()))
    else:
        D_dofs = np.empty(0, dtype=int)

    return D_vals, D_dofs


def _enforce_scalar_robin_neumann_boundary_conditions(
        mesh, element, bilinear_mat, linear_vec, N_bndry_conds, R_bndry_conds,
        u_prev):
    N_bases = [
        FacetBasis(mesh, element, facets=mesh.boundaries[key])
        for key in N_bndry_conds.keys()]

    for b, key in zip(N_bases, N_bndry_conds.keys()):
        fun = N_bndry_conds[key][0]
        linear_vec += asm(LinearForm(_forcing), b, forc=b.interpolate(
            b.project(fun)))

    R_bases = [
        FacetBasis(mesh, element, facets=mesh.boundaries[key])
        for key in R_bndry_conds.keys()]
    for b, key in zip(R_bases, R_bndry_conds.keys()):
        fun, alpha = R_bndry_conds[key]
        bilinear_mat += asm(BilinearForm(_robin), b, alpha=alpha)
        if u_prev is not None:
            # when u_prev is not None. then linear vec represents the residual
            # and so robin contribution to mass term must be reflected in res
            linear_vec -= asm(
                LinearForm(_robin_prev_sol), b, alpha=alpha, u_prev=u_prev)
        linear_vec += asm(LinearForm(_forcing), b, forc=b.interpolate(
            b.project(fun)))
    # see https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html
    # for useful notes
    return bilinear_mat, linear_vec


def _enforce_scalar_boundary_conditions(
        mesh, element, basis, bilinear_mat, linear_vec,
        D_bndry_conds, N_bndry_conds, R_bndry_conds, u_prev=None):
    bilinear_mat, linear_vec = (
        _enforce_scalar_robin_neumann_boundary_conditions(
            mesh, element, bilinear_mat, linear_vec, N_bndry_conds,
            R_bndry_conds, u_prev))
    D_vals, D_dofs = _enforce_dirichlet_scalar_boundary_conditions(
        mesh, element, basis, bilinear_mat, linear_vec,
        D_bndry_conds, u_prev)
    return bilinear_mat, linear_vec, D_vals, D_dofs


def _diffusion(u, v, w):
    return dot(w["diff"] * grad(u), grad(v))


def _reaction(u, v, w):
    return w["react"] * v


def _linearized_nonlinear_diffusion(u, v, w):
    return (dot(w["diff"] * grad(u), grad(v)) +
            dot(w['diff_prime']*u*grad(w['u_prev']), grad(v)) +
            w["react_prime"] * u * v)


def _diffusion_residual(v, w):
    return (w["forc"] * v - dot(w["diff"]*grad(w['u_prev']), grad(v)) -
            w["react"] * v)


class LinearDiffusionResidual():
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

    def _zero_fun(self, x, *args):
        return x[0]*0

    def _zero_vel_fun(self, x, *args):
        return x*0

    def linear_form(self, v, w):
        return self._forc_fun(w.x) * v

    def _advection_term(self, u, v, w):
        du = u.grad
        vel = self._vel_fun(w.x)
        return sum([v*vel[ii]*du[ii] for ii in range(w.x.shape[0])])

    def bilinear_form(self, u, v, w):
        diff = self._diff_fun(w.x)
        react = self._react_fun(w.x)
        return (dot(diff * grad(u), grad(v)) + react*u*v +
                self._advection_term(u, v, w))


class NonLinearDiffusionResidual():
    def __init__(self, forc_fun, react_fun, react_prime, nl_diff_fun,
                 nl_diff_prime, vel_fun):
        self._forc_fun = forc_fun
        self._nl_diff_fun = nl_diff_fun
        self._nl_diff_prime = nl_diff_prime
        if react_fun is None:
            react_fun = self._zero_fun
            react_prime = self._zero_fun
        if vel_fun is None:
            vel_fun = self._zero_vel_fun
        self._vel_fun = vel_fun
        self._react_fun = react_fun
        self._react_prime = react_prime
        self.__name__ = self.__class__.__name__

    def _zero_fun(self, x, *args):
        return x[0]*0

    def _zero_vel_fun(self, x, *args):
        return x*0

    def linear_form(self, v, w):
        forc = self._forc_fun(w.x)
        diff = self._nl_diff_fun(w.x, w.u_prev)
        react = self._react_fun(w.x, w.u_prev)
        return (forc * v - dot(diff*grad(w.u_prev), grad(v)) -
                react * v - self._advection_term(w.u_prev, v, w))

    def _advection_term(self, u, v, w):
        du = u.grad
        vel = self._vel_fun(w.x)
        return sum([v*vel[ii]*du[ii] for ii in range(w.x.shape[0])])

    def bilinear_form(self, u, v, w):
        diff = self._nl_diff_fun(w.x, w.u_prev)
        diff_prime = self._nl_diff_prime(w.x, w.u_prev)
        react_prime = self._react_prime(w.x, w.u_prev)
        mat = (dot(diff * grad(u), grad(v)) +
               dot(diff_prime*u*grad(w.u_prev), grad(v)) +
               react_prime * u * v) + self._advection_term(u, v, w)
        return mat


# TODO allow anisotropic diffusion
# see https://github.com/kinnala/scikit-fem/discussions/923
# @BilinearForm
# def anipoisson(u, v, w):
#     C = np.array([[1.0, 0.1],
#                   [0.1, 1.0]])
#     return dot(mul(C, grad(u)), grad(v))


def _raw_assemble_advection_diffusion_reaction(
        diff_fun, forc_fun, nl_diff_funs, react_funs, vel_fun,
        bndry_conds, mesh, element, basis, u_prev=None):

    # Note, project only takes functions that return 1D arrays
    # project takes functions that can compute when x is a 3D array,
    # thus manufactured solutions must be created with x in most strings.
    # However, there are some exceptions

    nl_diff_fun, nl_diff_prime = nl_diff_funs

    if nl_diff_fun is not None:
        if nl_diff_prime is None:
            raise ValueError(
                "nl_diff_prime must be provided with nl_diff_fun")

    if u_prev is None:
        assert nl_diff_fun is None
        assert react_funs[0] is None
        residual = LinearDiffusionResidual(
            forc_fun, diff_fun, react_funs[0], vel_fun)
        bilinear_mat = asm(
            BilinearForm(residual.bilinear_form), basis)
        linear_vec = asm(
            LinearForm(residual.linear_form), basis)
    else:
        residual = NonLinearDiffusionResidual(
            forc_fun, *react_funs, *nl_diff_funs, vel_fun)
        # use below if using newton solve
        u_prev_interp = basis.interpolate(u_prev)
        bilinear_mat = asm(
            BilinearForm(residual.bilinear_form), basis, u_prev=u_prev_interp)
        linear_vec = asm(
            LinearForm(residual.linear_form), basis, u_prev=u_prev_interp)

    bilinear_mat, linear_vec = (
        _enforce_scalar_robin_neumann_boundary_conditions(
            mesh, element, bilinear_mat, linear_vec, *bndry_conds[1:], u_prev))
    return bilinear_mat, linear_vec


def _assemble_advection_diffusion_reaction(
        diff_fun, forc_fun, nl_diff_funs, react_funs, vel_fun,
        bndry_conds, mesh, element, basis, u_prev=None):
    bilinear_mat, linear_vec = _raw_assemble_advection_diffusion_reaction(
        diff_fun, forc_fun, nl_diff_funs, react_funs, vel_fun,
        bndry_conds, mesh, element, basis, u_prev)
    D_vals, D_dofs = _enforce_dirichlet_scalar_boundary_conditions(
        mesh, element, basis, bilinear_mat, linear_vec,
        bndry_conds[0], u_prev)
    return bilinear_mat, linear_vec, D_vals, D_dofs


def _assemble_stokes_linear_vec(basis, vel_forc_fun, pres_forc_fun):
    pres_forc = basis['p'].interpolate(
        basis['p'].project(lambda x: pres_forc_fun(x)[:, 0]))
    vel_forces = basis['u'].interpolate(
        basis['u'].project(
            partial(_vector_fun_to_skfem_vector_fun, vel_forc_fun)))
    vel_loads = asm(
        LinearForm(_vector_forcing), basis['u'], forc=vel_forces)
    # - sign on pressure load because I think pressure equation
    # in manufactured_solutions.py has the wrong sign it is \nabla\cdot u = f_p
    # but convention is -\nabla\cdot u = f_p
    linear_vec = np.concatenate(
        [vel_loads, -asm(LinearForm(_forcing), basis['p'], forc=pres_forc)])
    return linear_vec


def _navier_stokes_nonlinear_term_residual(v, w):
    u = w["u_prev"]
    du = u.grad
    if u.shape[0] == 2:
        return (v[0] * (u[0] * du[0][0] + u[1] * du[0][1])
                + v[1] * (u[0] * du[1][0] + u[1] * du[1][1]))
    if u.shape[0] == 1:
        return v[0] * (u[0] * du[0][0])
    raise ValueError("Only 1D and 2D navier stokes supported")


def _navier_stokes_linearized_terms(u, v, w):
    z = w["u_prev"]
    dz = z.grad
    du = u.grad
    if u.shape[0] == 2:
        return (v[0] * (u[0] * dz[0][0] + u[1] * dz[0][1])
                + v[1] * (u[0] * dz[1][0] + u[1] * dz[1][1])
                + v[0] * (z[0] * du[0][0] + z[1] * du[0][1])
                + v[1] * (z[0] * du[1][0] + z[1] * du[1][1]))
    if u.shape[0] == 1:
        return v[0] * (u[0] * dz[0][0]) + v[0] * (z[0] * du[0][0])


def _enforce_stokes_boundary_conditions(
        mesh, element, basis, bilinear_mat, linear_vec,
        D_bndry_conds, N_bndry_conds, R_bndry_conds, A_shape, u_prev=None):
    # currently only dirichlet supported and zero neumann condition
    # which is enforced by doing nothing
    assert len(R_bndry_conds) == 0 and len(N_bndry_conds) == 0

    D_bases = [
        FacetBasis(mesh, element['u'], facets=mesh.boundaries[key])
        for key, fun in D_bndry_conds.items()]

    # get DOF for Dirichlet boundaries for velocity
    D_dofs = basis['u'].get_dofs(list(D_bndry_conds.keys()))

    # condense requires D_vals to be number of dofs.
    # however only the entries associated with D_dofs are ever used
    D_vals = np.hstack([basis['u'].zeros(), basis['p'].zeros()])
    for b, key in zip(D_bases, D_bndry_conds.keys()):
        _dofs = basis['u'].get_dofs(key)
        D_vals[_dofs] = b.project(D_bndry_conds[key][0])[_dofs]

    if (len(D_bndry_conds) == len(mesh.boundaries)):
        # all boundaries are dirichlet so must set a pressure value to
        # make presure unique. Otherwise it is only unique up to a constant

        # degrees of freedom for vector basis stored
        # [u1, v1, u2, v2, u3, v3...]
        # set a unique value for pressure at first dof of first element
        # D_dofs = basis['u'].get_dofs(list(D_bndry_conds.keys()))
        # A_shape is shape of velocity block in stiffness matrix
        # need to get DOF for presure in global array
        pres_idx = A_shape[0]+basis['p'].get_dofs(0).flatten()[:1]
        D_dofs = np.hstack([D_dofs, pres_idx])
        # setting to zero is an arbitrary choice. Test will only pass
        # if exact pressure solution is zero at this degree of freedom
        D_vals[pres_idx] = 0.
    # else:
       # Do nothing.  pressure is made unique because it appears in
       # the boundary integral arising from integration by parts of pressure
       # term in weak form of stokes equations

    if u_prev is not None:
        D_vals = u_prev-D_vals
    return bilinear_mat, linear_vec, D_vals, D_dofs


def _raw_assemble_stokes(
        vel_forc_fun, pres_forc_fun, navier_stokes,
        bndry_conds, mesh, element, basis, u_prev=None, return_K=False,
        viscosity=1):
    A = viscosity*asm(vector_laplace, basis['u'])
    B = -asm(divergence, basis['u'], basis['p'])

    # C = 1e-6*asm(mass, basis['p'])
    C = None
    K = bmat([[A, B.T],
              [B, C]], 'csr')

    if u_prev is not None:
        vel_prev, pres_prev = np.split(u_prev, K.blocks)

    if navier_stokes:
        A_nl = asm(BilinearForm(_navier_stokes_linearized_terms),
                   basis['u'], u_prev=vel_prev)
        bilinear_mat = bmat([[(A+A_nl), B.T],
                             [B, C]], 'csr')
    else:
        bilinear_mat = K  # jacobian

    linear_vec = _assemble_stokes_linear_vec(
        basis, vel_forc_fun, pres_forc_fun)

    if u_prev is not None:
        # minus sign because res = -a(u_prev, v) + L(v), b = res
        b_pres = -B.dot(vel_prev)
        b_vel = -(A.dot(vel_prev) + B.T.dot(pres_prev))
        if C is not None:
            b_pres -= C.dot(pres_prev)
        if navier_stokes:
            b_vel += -asm(LinearForm(_navier_stokes_nonlinear_term_residual),
                          basis['u'], u_prev=vel_prev)
        linear_vec += np.concatenate([b_vel, b_pres])
        # the following is true only for stokes (not navier stokes)
        # linear_vec -= K.dot(u_prev)

    if not return_K:
        return bilinear_mat, linear_vec, A.shape
    return bilinear_mat, linear_vec, A.shape, K


def _assemble_stokes(
        vel_forc_fun, pres_forc_fun, navier_stokes,
        bndry_conds, mesh, element, basis, u_prev=None, return_K=False,
        viscosity=1):
    result = _raw_assemble_stokes(
        vel_forc_fun, pres_forc_fun, navier_stokes,
        bndry_conds, mesh, element, basis, u_prev, return_K,
        viscosity)
    bilinear_mat, linear_vec, A_shape = result[:3]
    bilinear_mat, linear_vec, D_vals, D_dofs = (
        _enforce_stokes_boundary_conditions(
            mesh, element, basis, bilinear_mat, linear_vec, *bndry_conds,
            A_shape, u_prev))
    if not return_K:
        return bilinear_mat, linear_vec, D_vals, D_dofs
    return bilinear_mat, linear_vec, D_vals, D_dofs, result[3]


class Physics(ABC):
    def __init__(self,
                 mesh: Mesh,
                 element: Element,
                 basis: Basis,
                 bndry_conds: List):
        self.mesh = mesh
        self.element = element
        self.basis = basis
        self.bndry_conds = bndry_conds
        self.funs = self._set_funs()

    def _set_funs(self) -> List:
        return []

    @abstractmethod
    def init_guess(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def raw_assemble(self, sol: np.ndarray) -> Tuple[
            spmatrix,
            Union[np.ndarray, spmatrix],
            np.ndarray,
            np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def apply_dirichlet_boundary_conditions(
            self,
            sol: np.ndarray,
            bilinear_mat: spmatrix,
            linear_vec: Union[np.ndarray, spmatrix]) -> Tuple[
                spmatrix,
                Union[np.ndarray, spmatrix],
                np.ndarray,
                np.ndarray]:
        raise NotImplementedError()

    def assemble(self, sol: np.ndarray = None) -> Tuple[
            spmatrix,
            Union[np.ndarray, spmatrix],
            np.ndarray,
            np.ndarray]:
        return self.apply_dirichlet_boundary_conditions(
            sol, *self.raw_assemble(sol))

    def _transient_residual(self,
                            sol: np.ndarray,
                            time: float):
        # correct equations for boundary conditions
        self._set_time(time)
        res, jac = self._raw_residual(sol)
        if jac is None:
            assert self._auto_jac
        return res, jac

    def mass_matrix(self) -> Tuple[spmatrix]:
        mass_mat = asm(mass, self.basis)
        return mass_mat


class LinearAdvectionDiffusionReaction(Physics):
    def __init__(
            self,
            mesh: Mesh,
            element: Element,
            basis: Basis,
            bndry_conds: List,
            diff_fun: Callable[[np.ndarray],  np.ndarray],
            forc_fun: Callable[[np.ndarray],  np.ndarray],
            vel_fun: Optional[Callable[[np.ndarray],  np.ndarray]] = None,
            react_fun: Optional[Callable[[np.ndarray],  np.ndarray]] = None):

        self.diff_fun = diff_fun
        self.vel_fun = vel_fun
        self.forc_fun = forc_fun
        self.react_fun = react_fun

        super().__init__(mesh, element, basis, bndry_conds)

    def _set_funs(self) -> List:
        return [self.diff_fun, self.vel_fun, self.forc_fun, self.react_fun]

    def apply_dirichlet_boundary_conditions(
            self,
            sol: np.ndarray,
            bilinear_mat: spmatrix,
            linear_vec: Union[np.ndarray, spmatrix]) -> Tuple[
                spmatrix,
                Union[np.ndarray, spmatrix],
                np.ndarray,
                np.ndarray]:
        D_vals, D_dofs = _enforce_dirichlet_scalar_boundary_conditions(
            self.mesh, self.element, self.basis, bilinear_mat, linear_vec,
            self.bndry_conds[0], sol)
        return bilinear_mat, linear_vec, D_vals, D_dofs

    def raw_assemble(self, sol: np.ndarray = None) -> Tuple[
            spmatrix,
            Union[np.ndarray, spmatrix]]:
        residual = LinearDiffusionResidual(
             self.forc_fun, self.diff_fun, self.react_fun, self.vel_fun)
        bilinear_mat = asm(
            BilinearForm(residual.bilinear_form), self.basis)
        linear_vec = asm(
            LinearForm(residual.linear_form), self.basis)
        bilinear_mat, linear_vec = (
            _enforce_scalar_robin_neumann_boundary_conditions(
                self.mesh, self.element, bilinear_mat, linear_vec,
                *self.bndry_conds[1:], sol))
        return bilinear_mat, linear_vec

    def init_guess(self) -> np.ndarray:
        bilinear_mat, linear_vec, D_vals, D_dofs = self.assemble()
        return solve(*condense(bilinear_mat, linear_vec, x=D_vals, D=D_dofs))


class AdvectionDiffusionReaction(Physics):
    def __init__(
            self,
            mesh: Mesh,
            element: Element,
            basis: Basis,
            bndry_conds: List,
            diff_fun: Callable[[np.ndarray],  np.ndarray],
            forc_fun: Callable[[np.ndarray],  np.ndarray],
            vel_fun: Optional[Callable[[np.ndarray],  np.ndarray]],
            nl_diff_funs: Optional[
                Tuple[Callable[[np.ndarray, np.ndarray],  np.ndarray],
                      Callable[[np.ndarray, np.ndarray],  np.ndarray]]] = (
                          [None, None]),
            react_funs: Optional[
                Tuple[Callable[[np.ndarray],  np.ndarray],
                      Callable[[np.ndarray],  np.ndarray]]] = [None, None]):

        self.diff_fun = diff_fun
        self.vel_fun = vel_fun
        self.forc_fun = forc_fun
        self.nl_diff_funs = nl_diff_funs
        self.react_funs = react_funs

        super().__init__(mesh, element, basis, bndry_conds)

    def _set_funs(self) -> List:
        return [self.diff_fun, self.vel_fun, self.forc_fun, self.nl_diff_funs,
                self.react_funs]

    def init_guess(self) -> np.ndarray:
        bilinear_mat, linear_vec, D_vals, D_dofs = (
            _assemble_advection_diffusion_reaction(
                self.diff_fun, self.forc_fun, [None, None], [None, None],
                self.vel_fun,
                self.bndry_conds, self.mesh, self.element, self.basis))
        return solve(*condense(bilinear_mat, linear_vec, x=D_vals, D=D_dofs))

    def raw_assemble(self, sol: np.ndarray = None) -> Tuple[
            spmatrix,
            Union[np.ndarray, spmatrix]]:
        residual = NonLinearDiffusionResidual(
            self.forc_fun, *self.react_funs, *self.nl_diff_funs, self.vel_fun)
        # use below if using newton solve
        u_prev_interp = self.basis.interpolate(sol)
        bilinear_mat = asm(
            BilinearForm(residual.bilinear_form), self.basis,
            u_prev=u_prev_interp)
        linear_vec = asm(
            LinearForm(residual.linear_form), self.basis, u_prev=u_prev_interp)
        bilinear_mat, linear_vec = (
            _enforce_scalar_robin_neumann_boundary_conditions(
                self.mesh, self.element, bilinear_mat, linear_vec,
                *self.bndry_conds[1:], sol))
        return bilinear_mat, linear_vec

    def apply_dirichlet_boundary_conditions(
            self,
            sol: np.ndarray,
            bilinear_mat: spmatrix,
            linear_vec: Union[np.ndarray, spmatrix]) -> Tuple[
                spmatrix,
                Union[np.ndarray, spmatrix],
                np.ndarray,
                np.ndarray]:
        D_vals, D_dofs = _enforce_dirichlet_scalar_boundary_conditions(
            self.mesh, self.element, self.basis, bilinear_mat, linear_vec,
            self.bndry_conds[0], sol)
        return bilinear_mat, linear_vec, D_vals, D_dofs


class Helmholtz(LinearAdvectionDiffusionReaction):
    def __init__(
            self,
            mesh: Mesh,
            element: Element,
            basis: Basis,
            bndry_conds: List,
            wave_number_fun: Callable[[np.ndarray],  np.ndarray],
            forc_fun: Optional[Callable[[np.ndarray],  np.ndarray]] = None):

        if forc_fun is None:
            forc_fun = self._zero_forcing

        super().__init__(
            mesh, element, basis, bndry_conds, self._unit_diff,
            forc_fun, self._zero_vel, wave_number_fun)

    def _unit_diff(self, x):
        # negative is taken because advection diffusion code solves
        # -k*\nabla^2 u + c*u = f
        # but Helmholtz solves
        # \nabla^2 u + c*u = 0 which is equivalent when k=-1
        return -1+0*x[0]

    def _zero_vel(self, x):
        # Helmholtz is a special case of the advection diffusion reaction
        # equation when there is no advection, i.e. velocity field is zero
        return 0*x

    def _zero_forcing(self, x):
        return 0*x[0]


class Stokes(Physics):
    def __init__(
            self,
            mesh: Mesh,
            element: Element,
            basis: Basis,
            bndry_conds: List,
            navier_stokes: bool,
            vel_forc_fun: Callable[[np.ndarray],  np.ndarray],
            pres_forc_fun: Callable[[np.ndarray],  np.ndarray],
            viscosity: Optional[float] = 1.0):
        super().__init__(mesh, element, basis, bndry_conds)

        self.vel_forc_fun = vel_forc_fun
        self.pres_forc_fun = pres_forc_fun
        self.navier_stokes = navier_stokes
        self.viscosity = viscosity

        self.A_shape = None

    def init_guess(self) -> np.ndarray:
        bilinear_mat, linear_vec, D_vals, D_dofs = _assemble_stokes(
            self.vel_forc_fun, self.pres_forc_fun, False,
            self.bndry_conds, self.mesh, self.element, self.basis,
            return_K=False, viscosity=self.viscosity)
        return solve(*condense(bilinear_mat, linear_vec, x=D_vals, D=D_dofs))

    def raw_assemble(self, sol: Optional[np.ndarray] = None) -> Tuple[
            spmatrix,
            Union[np.ndarray, spmatrix],
            np.ndarray,
            np.ndarray]:
        bilinear_mat, linear_vec, self.A_shape = _raw_assemble_stokes(
            self.vel_forc_fun, self.pres_forc_fun, self.navier_stokes,
            self.bndry_conds, self.mesh, self.element, self.basis, sol,
            viscosity=self.viscosity)
        return bilinear_mat, linear_vec

    def apply_dirichlet_boundary_conditions(
            self,
            sol: np.ndarray,
            bilinear_mat: spmatrix,
            linear_vec: Union[np.ndarray, spmatrix]) -> Tuple[
                spmatrix,
                Union[np.ndarray, spmatrix],
                np.ndarray,
                np.ndarray]:
        bilinear_mat, linear_vec, D_vals, D_dofs = (
            _enforce_stokes_boundary_conditions(
                self.mesh, self.element, self.basis, bilinear_mat, linear_vec,
                *self.bndry_conds, self.A_shape, sol))
        return bilinear_mat, linear_vec, D_vals, D_dofs


def _diffusion_reaction(u, v, w):
    return dot(mul(w["diff"], grad(u)), grad(v)) + w["react"] * u * v


class BiLaplacianPrior():
    def __init__(
            self,
            mesh: Mesh,
            element: Element,
            basis: Basis,
            gamma: int,
            delta: int,
            anisotropic_tensor: Optional[np.ndarray] = None):
        r"""
        :math:`\delta\gamma` controls the variance of the prior
        :math:`\frac{\gamma}{\delta}` controls the correlation length
        The smaller the raio the smaller the correlation length
        Anisotropic tensor K controls correlation length in each direction
        E.g. in 2D K = [[K11, K12], [K21, K22]]
        K22<<K11 will cause correlation length to be small in vertical
        direction and long in horizontal direction
        """
        self.mesh = mesh
        self.element = element
        self.basis = basis
        self.gamma = gamma
        self.delta = delta
        self.beta = np.sqrt(self.gamma*self.delta)*1.42
        if anisotropic_tensor is None:
            anisotropic_tensor = np.eye(mesh.p.shape[0])*gamma
        else:
            anisotropic_tensor *= gamma
        if anisotropic_tensor.shape != (mesh.p.shape[0], mesh.p.shape[0]):
            raise ValueError("anisotropic_tensor has incorrect shape")
        self.anisotropic_tensor = anisotropic_tensor
        self._bndry_conds = [
            {}, {},
            dict(zip(self.mesh.boundaries.keys(),
                     [[self._bndry_fun, self.beta]
                      for nn in range(len(self.mesh.boundaries))]))
        ]
        self._linear_system_data = None

    def _setup_linear_system(self):
        if self._linear_system_data is None:
            self._linear_system_data = list(
                self._assemble_stiffness(
                    self._diff_fun, self._react_fun,
                    self._bndry_conds, self.mesh, self.element, self.basis))
            mass_mat = asm(mass, self.basis)
            lumped_mass_mat = np.asarray(mass_mat.sum(axis=1))[:, 0]
            self._linear_system_data += [lumped_mass_mat]

        return self._linear_system_data

    def _bndry_fun(self, x):
        # x : np.ndarray (nvars, nelems, nbndry_dofs_per_elem)
        # dofs are Lagrange basis nodes not quad points
        return 0*x[0]

    def _zero_forcing(self, v, w):
        return 0 * v

    def _react_fun(self, x):
        # x : np.ndarray (nvars, nelems, nquad_pts_per_elem)
        return self.delta+0*x[0]

    def _diff_fun(self, x):
        # x (nvars, nelems, nquad_pts_per_elem)
        return np.full((x.shape[1:]), self.gamma)

    def _assemble_stiffness(
            self, diff_fun, react_fun, bndry_conds, mesh, element, basis):
        react_proj = basis.project(react_fun)
        react = basis.interpolate(react_proj)
        bilinear_mat = asm(
            BilinearForm(_diffusion_reaction), basis,
            diff=self.anisotropic_tensor, react=react)
        linear_vec = asm(
            LinearForm(self._zero_forcing), basis)
        return _enforce_scalar_boundary_conditions(
            mesh, element, basis, bilinear_mat, linear_vec, *bndry_conds,
            None)

    def rvs(self, nsamples):
        bilinear_mat, linear_vec, D_vals, D_dofs, lumped_mass_mat = (
            self._setup_linear_system())
        white_noise = np.random.normal(
            0, 1, (lumped_mass_mat.shape[0], nsamples))
        samples = np.empty((lumped_mass_mat.shape[0], nsamples))
        for ii in range(nsamples):
            rhs = np.sqrt(lumped_mass_mat)*white_noise[:, ii]
            samples[:, ii] = solve(
                *condense(bilinear_mat, rhs, x=D_vals, D=D_dofs))
        return samples
