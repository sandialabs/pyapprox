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
from skfem.helpers import dot, grad
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence
from pyapprox.pde.galerkin.util import (
    _forcing, _vector_forcing, _vector_fun_to_skfem_vector_fun,
    _robin, _robin_prev_sol)


def _enforce_scalar_boundary_conditions(
        mesh, element, basis, bilinear_mat, linear_vec,
        D_bndry_conds, N_bndry_conds, R_bndry_conds, u_prev=None):
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
    D_dofs = basis.get_dofs(list(D_bndry_conds.keys()))

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

    return bilinear_mat, linear_vec, D_vals, D_dofs


def _diffusion(u, v, w):
    return dot(w["diff"] * grad(u), grad(v))


def _linearized_nonlinear_diffusion(u, v, w):
    return (dot(w["diff"] * grad(u), grad(v)) +
            dot(w['diff_prime']*u*grad(w['u_prev']), grad(v)))


def _diffusion_residual(v, w):
    return w["forc"] * v - dot(w["diff"]*grad(w['u_prev']), grad(v))


def _assemble_advection_diffusion_reaction(
        diff_fun, forc_fun, nl_diff_funs,
        bndry_conds, mesh, element, basis, u_prev=None):

    # project only takes functions that return 1D arrays
    # project takes functions that can compute when x is a 3D array
    # thus manufactured solutions must be created with x in most strings
    # there are some exceptions
    forc = basis.interpolate(basis.project(lambda x: forc_fun(x)[:, 0]))
    # diff is only linear component
    diff_proj = basis.project(lambda x: diff_fun(x)[:, 0])
    diff = basis.interpolate(diff_proj)

    nl_diff_fun, nl_diff_prime = nl_diff_funs
    if u_prev is not None and nl_diff_fun is not None:
        if nl_diff_prime is None:
            raise ValueError("nl_diff_prime must be provided with nl_diff_fun")
        diff = basis.interpolate(nl_diff_fun(diff_proj, u_prev))

    # TODO add reaction term to bilinear form and to linear form
    # associated with the residual

    if u_prev is None:
        assert nl_diff_fun is None
        # use below if solving directly with linear solve
        bilinear_mat = asm(BilinearForm(_diffusion), basis, diff=diff)
        linear_vec = asm(
            LinearForm(_forcing), basis, forc=forc)
    else:
        # use below if using newton solve
        u_prev_interp = basis.interpolate(u_prev)
        if nl_diff_fun is None:
            bilinear_mat = asm(
                BilinearForm(_diffusion), basis, diff=diff)
        else:
            diff_prime = basis.interpolate(nl_diff_prime(diff_proj, u_prev))
            bilinear_mat = asm(
                BilinearForm(_linearized_nonlinear_diffusion), basis, diff=diff,
                diff_prime=diff_prime, u_prev=u_prev_interp)
        linear_vec = asm(
            LinearForm(_diffusion_residual),
            basis, forc=forc,  diff=diff, u_prev=u_prev_interp)

    bilinear_mat, linear_vec, D_vals, D_dofs = (
        _enforce_scalar_boundary_conditions(
            mesh, element, basis, bilinear_mat, linear_vec, *bndry_conds,
            u_prev))
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


def _assemble_stokes(
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

    bilinear_mat, linear_vec, D_vals, D_dofs = (
        _enforce_stokes_boundary_conditions(
            mesh, element, basis, bilinear_mat, linear_vec, *bndry_conds,
            A.shape, u_prev))
    if not return_K:
        return bilinear_mat, linear_vec, D_vals, D_dofs
    return bilinear_mat, linear_vec, D_vals, D_dofs, K


class Physics(ABC):
    def __init__(self,
                 mesh : Mesh,
                 element : Element,
                 basis : Basis,
                 bndry_conds : List):
        self.mesh = mesh
        self.element = element
        self.basis = basis
        self.bndry_conds = bndry_conds
    
    @abstractmethod
    def init_guess(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def assemble(self, sol : np.ndarray = None)-> Tuple[
            spmatrix,
            Union[np.ndarray, spmatrix],
            np.ndarray,
            np.ndarray]:
        raise NotImplementedError()


class AdvectionDiffusionReaction(Physics):
    def __init__(
            self,
            mesh : Mesh,
            element : Element,
            basis : Basis,
            bndry_conds : List,
            diff_fun : Callable [[np.ndarray],  np.ndarray],
            forc_fun : Callable [[np.ndarray],  np.ndarray],
            vel_fun : Optional[Callable [[np.ndarray],  np.ndarray]],
            nl_diff_funs : Optional[
                Tuple[Callable  [[np.ndarray, np.ndarray],  np.ndarray],
                      Callable  [[np.ndarray, np.ndarray],  np.ndarray]]] = (
                          [None, None]),
            react_funs : Optional[
                Tuple[Callable  [[np.ndarray],  np.ndarray],
                      Callable  [[np.ndarray],  np.ndarray]]] = [None, None]):
        
        super().__init__(mesh, element, basis, bndry_conds)
        self.diff_fun = diff_fun
        self.vel_fun = vel_fun
        self.forc_fun = forc_fun
        self.nl_diff_funs = nl_diff_funs
        self.react_funs = react_funs

        if self.vel_fun is not None or self.react_funs[0] is not None:
            # TODO add these terms
            raise NotImplementedError("Options currently not supported")
    
    def init_guess(self) -> np.ndarray:
        bilinear_mat, linear_vec, D_vals, D_dofs = (
            _assemble_advection_diffusion_reaction(
                self.diff_fun, self.forc_fun, [None, None],
                self.bndry_conds, self.mesh, self.element, self.basis))
        return solve(*condense(bilinear_mat, linear_vec, x=D_vals, D=D_dofs))
    
    def assemble(self, sol : np.ndarray = None)-> Tuple[
            spmatrix,
            Union[np.ndarray, spmatrix],
            np.ndarray,
            np.ndarray]:
        return _assemble_advection_diffusion_reaction(
            self.diff_fun, self.forc_fun, self.nl_diff_funs,
            self.bndry_conds, self.mesh, self.element, self.basis, sol)
    

class Stokes(Physics):
    def __init__(
            self,
            mesh : Mesh,
            element : Element,
            basis : Basis,
            bndry_conds : List,
            navier_stokes : bool,
            vel_forc_fun : Callable [[np.ndarray],  np.ndarray],
            pres_forc_fun : Callable  [[np.ndarray],  np.ndarray],
            viscosity : Optional[float] = 1.0):
        super().__init__(mesh, element, basis, bndry_conds)

        self.vel_forc_fun = vel_forc_fun
        self.pres_forc_fun = pres_forc_fun
        self.navier_stokes = navier_stokes
        self.viscosity = viscosity
    
    def init_guess(self) -> np.ndarray:
        bilinear_mat, linear_vec, D_vals, D_dofs = _assemble_stokes(
            self.vel_forc_fun, self.pres_forc_fun, False,
            self.bndry_conds, self.mesh, self.element, self.basis,
            return_K=False, viscosity=self.viscosity)
        return solve(*condense(bilinear_mat, linear_vec, x=D_vals, D=D_dofs))

    def assemble(self, sol : Optional[np.ndarray] = None) -> Tuple[
            spmatrix,
            Union[np.ndarray, spmatrix],
            np.ndarray,
            np.ndarray]:
        return _assemble_stokes(
            self.vel_forc_fun, self.pres_forc_fun, self.navier_stokes,
            self.bndry_conds, self.mesh, self.element, self.basis, sol,
            viscosity=self.viscosity)
