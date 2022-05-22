import unittest
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from functools import partial

from abc import ABC, abstractmethod
from pyapprox.pde.spectral_diffusion import (
    chebyshev_derivative_matrix,
    chebyshev_second_derivative_matrix,
    RectangularCollocationMesh, OneDCollocationMesh, ones_fun_axis_0
)

from pyapprox.surrogates.orthopoly.quadrature import (
    gauss_jacobi_pts_wts_1D)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d)
from pyapprox.util.utilities import cartesian_product, approx_jacobian


def lagrange_polynomial_derivative_matrix_1d(eval_samples, abscissa):
    nabscissa = abscissa.shape[0]
    samples_diff = eval_samples[:, None]-abscissa[None, :]
    abscissa_diff = abscissa[:, None]-abscissa[None, :]
    basis_vals = np.ones((eval_samples.shape[0], nabscissa))
    deriv_mat = np.empty_like(basis_vals)
    for ii in range(nabscissa):
        indices = np.delete(np.arange(nabscissa), ii)
        numer = samples_diff[:, indices].prod(axis=1)
        denom = abscissa_diff[ii, indices].prod(axis=0)
        basis_vals[:, ii] = numer/denom
        numer_deriv = 0
        for jj in range(nabscissa):
            if ii != jj:
                numer_deriv += np.delete(
                    samples_diff, (ii, jj), axis=1).prod(axis=1)
        deriv_mat[:, ii] = numer_deriv/denom
    return deriv_mat, basis_vals


def lagrange_polynomial_derivative_matrix_2d(eval_samples, abscissa_1d):
    nabscissa_1d = [a.shape[0] for a in abscissa_1d]
    basis_vals = np.ones((eval_samples.shape[1], np.prod(nabscissa_1d)))
    deriv_mat = np.empty((2, eval_samples.shape[1], np.prod(nabscissa_1d)))
    numer = [[], []]
    denom = [[], []]
    samples_diff = [None, None]
    for dd in range(2):
        samples_diff[dd] = eval_samples[dd][:, None]-abscissa_1d[dd][None, :]
        abscissa_diff = abscissa_1d[dd][:, None]-abscissa_1d[dd][None, :]
        for jj in range(nabscissa_1d[dd]):
            indices = np.delete(np.arange(nabscissa_1d[dd]), jj)
            numer[dd].append(samples_diff[dd][:, indices].prod(axis=1))
            denom[dd].append(abscissa_diff[jj, indices].prod(axis=0))

    cnt = 0
    for jj in range(nabscissa_1d[1]):
        for ii in range(nabscissa_1d[0]):
            basis_vals[:, cnt] = (
                numer[0][ii]/denom[0][ii]*numer[1][jj]/denom[1][jj])
            numer_deriv_0 = 0
            for kk in range(nabscissa_1d[0]):
                if ii != kk:
                    numer_deriv_0 += np.delete(
                        samples_diff[0], (ii, kk), axis=1).prod(axis=1)
            deriv_mat[0, :, cnt] = (
                numer_deriv_0/denom[0][ii]*numer[1][jj]/denom[1][jj])

            numer_deriv_1 = 0
            for kk in range(nabscissa_1d[1]):
                if jj != kk:
                    numer_deriv_1 += np.delete(
                        samples_diff[1], (jj, kk), axis=1).prod(axis=1)
            deriv_mat[1, :, cnt] = (
                numer[0][ii]/denom[0][ii]*numer_deriv_1/denom[1][jj])
            cnt += 1
    abscissa = cartesian_product(abscissa_1d)
    return deriv_mat, basis_vals, abscissa


class AbstractSpectralPDE(ABC):
    def __init__(self):
        self._qoi_functional = None
        self._qoi_functional_deriv = None
        self._mesh = None

    def _set_domain(self, domain, order):
        if len(domain) == 2:
            self._mesh = OneDCollocationMesh(domain, order)
        elif len(domain) == 4:
            self._mesh = RectangularCollocationMesh(domain, order)
        raise ValueError("Only 2D domains supported")

    @abstractmethod
    def _set_boundary_conditions(self, bndry_conds):
        raise NotImplementedError()

    def initialize(self, bndry_conds, order, domain):
        self._set_domain(domain, order)
        self._set_boundary_conditions(bndry_conds)

    @abstractmethod
    def _form_collocation_matrix(self, *args):
        raise NotImplementedError

    def set_qoi_functional(self, qoi_functional, qoi_functional_deriv=None):
        self._qoi_functional = qoi_functional
        self._qoi_functional_deriv = qoi_functional_deriv


class InteriorOneDCollocationMesh(OneDCollocationMesh):
    def form_derivative_matrices(self):
        # will work but divergence condition is only satisfied on interior
        # so if want to drive flow with only boundary conditions on velocity
        # it will now work
        # print(self.order)
        self.mesh_pts_1d = [
            -np.cos(np.linspace(0, np.pi, o+1))[1:-1] for o in self.order]
        # self.mesh_pts_1d = [chebyshev_derivative_matrix(o)[0]
        #                    for o in self.order-2]
        self._mesh_pts_1d_barycentric_weights = [
            compute_barycentric_weights_1d(xx) for xx in self.mesh_pts_1d]
        eval_samples = cartesian_product(
            [-np.cos(np.linspace(0, np.pi, o+1))  for o in self.order])
        deriv_mat = lagrange_polynomial_derivative_matrix_1d(
                eval_samples[0], self.mesh_pts_1d[0])

        self.mesh_pts = self.map_samples_from_canonical_domain(
            np.asarray(self.mesh_pts_1d))
        self.derivative_matrices = [deriv_mat[0]*2./(
            self.domain[1]-self.domain[0])]

        deriv_mat_alt = lagrange_polynomial_derivative_matrix_1d(
            self.mesh_pts_1d[0], eval_samples[0])
        self.derivative_matrices_alt = [deriv_mat_alt[0]*2./(
            self.domain[1]-self.domain[0])]
        
    def _determine_boundary_indices(self):
        self._boundary_indices = None

    def _set_bndry_segment_interpolation_data(self):
        self._bndry_segment_canonical_abscissa = []
        self._bndry_segment_canonical_bary_weights = []


class InteriorRectangularCollocationMesh(RectangularCollocationMesh):

    def form_derivative_matrices(self):
        # will work but divergence condition is only satisfied on interior
        # so if want to drive flow with only boundary conditions on velocity
        # it will now work
        self.mesh_pts_1d = [
            -np.cos(np.linspace(0, np.pi, o+1))[1:-1] for o in self.order]
        # self.mesh_pts_1d = [
        #     -np.cos(np.linspace(0, np.pi, self.order[0]+1))[1:-1],
        #     -np.cos(np.linspace(0, np.pi, self.order[0]-1))]
        self._mesh_pts_1d_barycentric_weights = [
            compute_barycentric_weights_1d(xx) for xx in self.mesh_pts_1d]
        eval_samples = cartesian_product(
            [-np.cos(np.linspace(0, np.pi, o+1)) for o in self.order])
        deriv_mat, basis_vals, canonical_mesh_pts = (
            lagrange_polynomial_derivative_matrix_2d(
                eval_samples, self.mesh_pts_1d))

        self.mesh_pts = self.map_samples_from_canonical_domain(
            canonical_mesh_pts.copy())

        self.derivative_matrices = [None, None]
        self.derivative_matrices[0] = deriv_mat[0]*2./(
            self.domain[1]-self.domain[0])
        self.derivative_matrices[1] = deriv_mat[1]*2./(
            self.domain[3]-self.domain[2])

        deriv_mat_alt = lagrange_polynomial_derivative_matrix_2d(
            canonical_mesh_pts,
            [-np.cos(np.linspace(0, np.pi, o+1)) for o in self.order])[0]

        self.derivative_matrices_alt = [None, None]
        self.derivative_matrices_alt[0] = deriv_mat_alt[0]*2./(
            self.domain[1]-self.domain[0])
        self.derivative_matrices_alt[1] = deriv_mat_alt[1]*2./(
            self.domain[3]-self.domain[2])

    def _determine_boundary_indices(self):
        self._boundary_indices = None

    def _set_bndry_segment_interpolation_data(self):
        self._bndry_segment_canonical_abscissa = []
        self._bndry_segment_canonical_bary_weights = []


class StokesFlowModel(AbstractSpectralPDE):
    def __init__(self):
        self.forcing_fun = None
        super().__init__()

    def initialize(self, bndry_conds, order, domain,
                   forcing_fun):
        super().initialize(bndry_conds, order, domain)
        self.forcing_fun = forcing_fun

    def _set_domain(self, domain, order):
        if len(domain) == 2:
            self._mesh = OneDCollocationMesh(domain, order)
            self._pres_mesh = InteriorOneDCollocationMesh(domain, order)
        elif len(domain) == 4:
            self._mesh = RectangularCollocationMesh(domain, order)
            self._pres_mesh = InteriorRectangularCollocationMesh(domain, order)

        self._interior_indices = np.delete(
            np.arange(self._mesh.mesh_pts.shape[1]),
            np.hstack(self._mesh.boundary_indices))

    def _set_boundary_conditions(self, bndry_conds):
        self._mesh.set_boundary_conditions(bndry_conds)

    def _form_collocation_matrix(self):
        # laplace vel  -diff grad pressure
        # TODO rearrange mesh derivative matrices so boundary conditions
        # are all the last rows
        sub_mats = []
        for dd in range(self._mesh.nphys_vars):
            sub_mats.append([None for ii in range(self._mesh.nphys_vars+1)])
            for ii in range(self._mesh.nphys_vars):
                if ii == dd:
                    Dmat = (
                        self._mesh.derivative_matrices[0])
                    vel_mat = Dmat.dot(Dmat)
                    for jj in range(1, self._mesh.nphys_vars):
                        Dmat = (
                            self._mesh.derivative_matrices[jj])
                        vel_mat += Dmat.dot(Dmat)
                    sub_mats[-1][ii] = -vel_mat
                else:
                    sub_mats[-1][ii] = np.zeros_like(vel_mat)

            # for ii in range(self._mesh.nphys_vars):
            #     if ii == dd:
            #         Dmat = self._mesh.derivative_matrices[dd]
            #         vel_mat = Dmat.dot(Dmat)
            #         sub_mats[-1][ii] = -vel_mat
            #     else:
            #         sub_mats[-1][ii] = np.zeros_like(vel_mat)
            pres_mat = self._pres_mesh.derivative_matrices[dd]
            sub_mats[-1][self._mesh.nphys_vars] = pres_mat

        # divergence constraint
        sub_mats.append([])
        for dd in range(self._mesh.nphys_vars):
            Dmat = self._pres_mesh.derivative_matrices_alt[dd].copy()
            sub_mats[-1].append(Dmat)
        sub_mats[-1].append(np.zeros((Dmat.shape[0],
                                      Dmat.shape[0])))

        # for s in sub_mats:
        #      print([t.shape for t in s])

        matrix = np.block(sub_mats)
        return matrix

    def get_num_degrees_of_freedom(self, config_sample):
        return np.prod(config_sample)

    def _apply_boundary_conditions_to_matrix(self, matrix):
        nvel_dof = self._mesh.mesh_pts.shape[1]
        cnt = 0
        for dd in range(self._mesh.nphys_vars):
            # TODO test this will work for neumann boundaries
            matrix[cnt:cnt+nvel_dof, :][np.hstack(
                self._mesh.boundary_indices), :] = 0
            vel_matrix = matrix[cnt:cnt+nvel_dof, cnt:cnt+nvel_dof].copy()
            updated_vel_matrix = (
                self._mesh._apply_boundary_conditions_to_matrix(vel_matrix))
            matrix[cnt:cnt+nvel_dof, cnt:cnt+nvel_dof] = updated_vel_matrix
            cnt += nvel_dof
        return matrix

    def _apply_boundary_conditions_to_rhs(self, rhs):
        nvel_dof = self._mesh.mesh_pts.shape[1]
        for ii, bndry_cond in enumerate(self._mesh.bndry_conds):
            idx = self._mesh.boundary_indices[ii]
            bndry_vals = bndry_cond[0](self._mesh.mesh_pts[:, idx])
            # only apply boundary conditions that define values of solution
            # consistent with degrees of freedom under consideration
            # i.e. only apply u_1 = 0 for rhs elements associated with u_1 mesh
            # (first row block) and u_2 = 0 for rhs elements associated with
            # u_2 mesh (second row block)
            for dd in range(self._mesh.nphys_vars):
                rhs[idx+dd*nvel_dof] = bndry_vals[dd]
        return rhs

    def solve(self, sample, pres=(0, 1)):
        forcing = self.forcing_fun(self._mesh.mesh_pts, sample[:, None])
        assert len(forcing) == self._mesh.nphys_vars+1
        for ii in range(len(forcing)):
            assert forcing[ii].ndim == 2 and forcing[ii].shape[1] == 1
        # forcing = forcing + [np.zeros((self._pres_mesh.mesh_pts.shape[1], 1))]
        # todo we call forcing fun twice which computes forcing on both meshes
        # which is wasted effort
        forcing[-1] = self.forcing_fun(
            self._pres_mesh.mesh_pts, sample[:, None])[-1]
        forcing = np.vstack(forcing)
        # we need another copy so that forcing can be used when solving adjoint
        self.forcing_vals = forcing.copy()
        # print([s[:, 0] for s in self._split_quantities(forcing)])
        self.collocation_matrix = self._form_collocation_matrix()
        print(np.linalg.matrix_rank(self.collocation_matrix),
              self.collocation_matrix.shape)
        matrix = self._apply_boundary_conditions_to_matrix(
            self.collocation_matrix.copy())
        print(np.linalg.matrix_rank(matrix), matrix.shape)
        rhs = self._apply_boundary_conditions_to_rhs(forcing.copy())

        # set pressure value at one location to make pressure unique
        matrix[self._mesh.nphys_vars*self._mesh.mesh_pts.shape[1]+pres[0], :] = 0
        matrix[self._mesh.nphys_vars*self._mesh.mesh_pts.shape[1]+pres[0],
               self._mesh.nphys_vars*self._mesh.mesh_pts.shape[1]+pres[0]] = 1
        rhs[self._mesh.nphys_vars*self._mesh.mesh_pts.shape[1]+pres[0], 0] = pres[1]

        print(np.linalg.cond(matrix))
        print(np.linalg.matrix_rank(matrix), matrix.shape)
        tmp = self._split_quantities(np.round(matrix, decimals=2))
        for t in tmp:
            print("#")
            tmp1 = self._split_quantities(t.T)
            for s in tmp1:
                print(s.T)

        for t in self._split_quantities(rhs):
            print(t)
        for t in self._split_quantities(forcing):
            print(t)
        
        solution = np.linalg.solve(matrix, rhs)
        split_solutions = self._split_quantities(solution)
        self._matrix = matrix
        self._rhs = rhs
        return split_solutions

    def _split_quantities(self, vector):
        nvel_dof = self._mesh.mesh_pts.shape[1]
        split = [vector[ii*nvel_dof:(ii+1)*nvel_dof]
                 for ii in range(self._mesh.nphys_vars)]
        split.append(vector[self._mesh.nphys_vars*nvel_dof:])
        return split

    def interpolate(self, sol_vals, xx):
        Z = [None, None, None]
        for ii in range(self._mesh.nphys_vars):
            Z[ii] = self._mesh.interpolate(sol_vals[ii], xx)
        Z[-1] = self._pres_mesh.interpolate(sol_vals[-1], xx)
        return Z

def evaluate_sp_lambda(sp_lambda, xx):
    # sp_lambda returns a singel function output
    vals = sp_lambda(*[x for x in xx])
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals)


def evaluate_sp_lambda_list(sp_lambda, xx):
    # sp_lambda returns list of values from multiple functions
    vals = sp_lambda(*[x for x in xx])  # vals is a list
    for ii in range(len(vals)):
        tmp = vals[ii]
        if type(tmp) == np.ndarray:
            vals[ii] = tmp[:, None]
        else:
            vals[ii] = np.full((xx.shape[1], 1), tmp)
    return vals


class TestStokes(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_chebyshev_derivative_matrix_interior_nodes(self):
        order = 5
        degree = order-2

        def fun(x): return x**degree
        def deriv(x): return degree*x**(degree-1)

        pts, deriv_mat_1d = chebyshev_derivative_matrix(order)

        deriv_mat_barycentric, basis_vals = (
            lagrange_polynomial_derivative_matrix_1d(pts, pts))
        assert np.allclose(basis_vals.dot(fun(pts)), fun(pts))
        assert np.allclose(deriv_mat_1d, deriv_mat_barycentric)
        assert np.allclose(deriv_mat_1d.dot(fun(pts)), deriv(pts))

        interior_pts = pts[1:-1]
        deriv_mat_1d_interior = np.zeros((pts.shape[0], pts.shape[0]))
        dmat_int = lagrange_polynomial_derivative_matrix_1d(
                pts, interior_pts)[0]
        assert np.allclose(
            dmat_int.dot(fun(interior_pts)), deriv(pts))


        interior_pts = pts[1:-1]
        deriv_mat_1d_interior = np.zeros((pts.shape[0], pts.shape[0]))
        dmat_int = lagrange_polynomial_derivative_matrix_1d(
            interior_pts, pts)[0]
        print(dmat_int)
        assert np.allclose(
            dmat_int.dot(fun(pts)), deriv(interior_pts))

    def test_lagrange_polynomial_derivative_matrix_2d(self):
        np.set_printoptions(linewidth=300, threshold=2000)
        order = [4, 4]
        abscissa_1d = [chebyshev_derivative_matrix(o)[0][1:-1] for o in order]
        eval_samples = cartesian_product(
            [chebyshev_derivative_matrix(o)[0] for o in order])
        deriv_mat, basis_vals, abscissa = (
            lagrange_polynomial_derivative_matrix_2d(
                eval_samples, abscissa_1d))

        def wrapper(xx):
            basis_vals = lagrange_polynomial_derivative_matrix_2d(
                xx, abscissa_1d)[1]
            vals = basis_vals[0, :]
            return vals

        jac1, jac2 = [], []
        for sample in eval_samples.T:
            tmp = approx_jacobian(wrapper, sample[:, None]).T
            jac1.append(tmp[0])
            jac2.append(tmp[1])
        jac = np.array([jac1, jac2])
        assert np.allclose(jac, deriv_mat, atol=1e-7)

        # print(np.round(deriv_mat[0], decimals=2))

        def fun(xx): return (xx[0, :]**2*xx[1, :])[:, None]

        def deriv(xx):
            return np.vstack(((2*xx[0, :]*xx[1, :])[None, :],
                              (xx[0, :]**2)[None, :]))
        assert np.allclose(
            basis_vals.dot(fun(abscissa)), fun(eval_samples))
        assert np.allclose(
            deriv_mat.dot(fun(abscissa)[:, 0]), deriv(eval_samples))


        order = [4, 4]
        abscissa_1d = [chebyshev_derivative_matrix(o)[0] for o in order]
        eval_samples = cartesian_product(
            [chebyshev_derivative_matrix(o-2)[0] for o in order])
        deriv_mat, basis_vals, abscissa = (
            lagrange_polynomial_derivative_matrix_2d(
                eval_samples, abscissa_1d))

        print(np.round(deriv_mat[0], decimals=2))

        def fun(xx): return (xx[0, :]**2*xx[1, :])[:, None]

        def deriv(xx):
            return np.vstack(((2*xx[0, :]*xx[1, :])[None, :],
                              (xx[0, :]**2)[None, :]))
        assert np.allclose(
            basis_vals.dot(fun(abscissa)), fun(eval_samples))
        assert np.allclose(
            deriv_mat.dot(fun(abscissa)[:, 0]), deriv(eval_samples))

        def wrapper(xx):
            basis_vals = lagrange_polynomial_derivative_matrix_2d(
                xx, abscissa_1d)[1]
            vals = basis_vals[0, :]
            return vals

        jac1, jac2 = [], []
        for sample in eval_samples.T:
            tmp = approx_jacobian(wrapper, sample[:, None]).T
            jac1.append(tmp[0])
            jac2.append(tmp[1])
        jac = np.array([jac1, jac2])
        print(np.round(jac[0], decimals=2))
        assert np.allclose(jac, deriv_mat, atol=1e-7)


    def test_stokes_mms(self):
        np.set_printoptions(linewidth=400, threshold=2000)
        nphys_vars = 2
        sp_x, sp_y = sp.symbols(['x', 'y'])
        if nphys_vars == 2:
            domain = [0, 1, 0, 1]
            symbs = (sp_x, sp_y)
            order = 20
            velocity_strings = ["-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"]
            pressure_string = "x**2+y**2"
            order = 6
            velocity_strings = ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"]
            # pressure_string = "1"
            pressure_string = "x**1*y**2"
            # order = 4
            # velocity_strings = ["-x**2", "-y**3"]
            # pressure_string = "x**2+y**2"
        else:
            domain = [0, 1]
            symbs = (sp_x,)
            order = 4
            velocity_strings = ["(1-x)**2"]
            pressure_string = "x**2"

        sp_pres = sp.sympify(pressure_string)
        sp_vel = [sp.sympify(s) for s in velocity_strings]
        sp_forc = []
        for vel, s1 in zip(sp_vel, symbs):
            sp_forc.append(
                sum([-vel.diff(s2, 2) for s2 in symbs])+
                sp_pres.diff(s1, 1))
        sp_div = sum([vel.diff(s, 1) for vel, s in zip(sp_vel, symbs)])
        print('v', sp_vel)
        print('p', sp_pres)
        print('f', sp_forc)
        print(([vel.diff(s, 1) for vel, s in zip(sp_vel, symbs)]))
        print('div', sp_div)

        exact_vel_lambda = [sp.lambdify(symbs, fun, "numpy")
                            for fun in sp_vel]
        exact_pres_lambda = sp.lambdify(symbs, sp_pres, "numpy")
        vel_forcing_lambda = sp.lambdify(symbs, sp_forc, "numpy")
        div_lambda = sp.lambdify(symbs, sp_div, "numpy")
        exact_pres = partial(evaluate_sp_lambda, exact_pres_lambda)
        vel_forcing_fun = partial(evaluate_sp_lambda_list, vel_forcing_lambda)
        exact_pres_grad = [sp_pres.diff(s, 1) for s in symbs]
        exact_pres_grad_lambda = [
            sp.lambdify(symbs, pg, "numpy")
            for s, pg in zip(symbs, exact_pres_grad)]
        print('pgrad', exact_pres_grad)

        def exact_pres_grad(xx):
            vals = [
                evaluate_sp_lambda(fun, xx) for fun in exact_pres_grad_lambda]
            return vals

        def exact_vel(xx):
            vals = [evaluate_sp_lambda(fun, xx) for fun in exact_vel_lambda]
            return vals

        def forcing_fun(xx, z):
            vel_forcing_vals = vel_forcing_fun(xx)
            div_vals = evaluate_sp_lambda(div_lambda, xx)
            return vel_forcing_vals+[div_vals]

        bndry_conds = [[lambda x: exact_vel(x), "D"],
                       [lambda x: exact_vel(x), "D"],
                       [lambda x: exact_vel(x), "D"],
                       [lambda x: exact_vel(x), "D"]]

        bndry_conds = bndry_conds[:len(domain)]

        model = StokesFlowModel()
        model.initialize(bndry_conds, order, domain, forcing_fun)

        assert np.allclose(model._mesh.mesh_pts[:, model._interior_indices],
                           model._pres_mesh.mesh_pts)

        exact_vel_vals = exact_vel(model._mesh.mesh_pts)
        exact_pres_vals = exact_pres(model._pres_mesh.mesh_pts)
        exact_sol_vals = np.vstack(exact_vel_vals+[exact_pres_vals])

        sample = np.zeros(0)  # dummy
        # pres_idx = 0  # index of fixed pressure in split solution
        pres_idx = model._pres_mesh.mesh_pts.shape[1]//2
        #pres_idx = model._pres_mesh.mesh_pts.shape[1]-1
        pres_val = exact_pres_vals[pres_idx]
        sol_vals = model.solve(
            sample, pres=(pres_idx, pres_val))

        for dd in range(model._mesh.nphys_vars):
            assert np.allclose(
                model._pres_mesh.derivative_matrices[dd].dot(exact_pres_vals),
                exact_pres_grad(model._mesh.mesh_pts)[dd])

        bndry_indices = np.hstack(model._mesh.boundary_indices)
        recovered_forcing = model._split_quantities(
            model._matrix.dot(exact_sol_vals))
        exact_forcing = forcing_fun(model._mesh.mesh_pts, sample)
        for dd in range(nphys_vars):
            assert np.allclose(sol_vals[dd][bndry_indices],
                               exact_vel_vals[dd][bndry_indices])
            assert np.allclose(recovered_forcing[dd][model._interior_indices],
                               exact_forcing[dd][model._interior_indices])
            assert np.allclose(exact_vel_vals[dd][bndry_indices],
                               recovered_forcing[dd][bndry_indices])

        # check value used to enforce unique pressure is found correctly
        print(sol_vals[nphys_vars][pres_idx], pres_val)
        assert np.allclose(
            sol_vals[nphys_vars][pres_idx], pres_val)
        # check pressure at all but point used for enforcing unique value
        # are set correctly
        assert np.allclose(
            np.delete(recovered_forcing[nphys_vars], pres_idx),
            np.delete(
                exact_forcing[nphys_vars][model._interior_indices], pres_idx))

        for exact_v, v in zip(exact_vel_vals, sol_vals[:-1]):
            assert np.allclose(exact_v, v)

        recovered_div = sum(
            [model._mesh.derivative_matrices[dd].dot(sol_vals[dd])
             for dd in range(model._mesh.nphys_vars)])
        # print(recovered_div[model._interior_indices])
        # print(exact_forcing[-1][model._interior_indices])
        assert np.allclose(recovered_div[model._interior_indices],
                           exact_forcing[-1][model._interior_indices])

        num_pts_1d = 50
        plot_limits = domain
        from pyapprox.util.visualization import get_meshgrid_samples
        X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d)
        Z = model.interpolate(sol_vals, pts)
        for ii in range(model._mesh.nphys_vars):
            assert np.allclose(Z[ii], exact_vel(pts)[ii])
            Z[ii] = np.reshape(Z[ii], (X.shape[0], X.shape[1]))
        assert np.allclose(Z[-1], exact_pres(pts))
        Z[-1] = np.reshape(Z[-1], (X.shape[0], X.shape[1]))
        fig, axs = plt.subplots(1, 4, figsize=(8*4, 6))
        axs[0].quiver(X, Y, Z[0], Z[1])
        for ii in range(3):
            if Z[ii].min() != Z[ii].max():
                pl = axs[ii+1].contourf(
                    X, Y, Z[ii],
                    levels=np.linspace(Z[ii].min(), Z[ii].max(), 20))
            plt.colorbar(pl, ax=axs[ii+1])
        plt.show()

    def test_lid_driven_cavity_flow(self):
        np.set_printoptions(linewidth=400, threshold=5000)
        # todo ensure boundary conditions are being enforced exactly
        # they are not yet

        # drive cavity using left boundary as top and bottom boundaries
        # do not hvae dof at the corners
        def bndry_condition(xx):
            cond = [(1*xx[0, :]**2*(1-xx[0, :]**2))[:, None],
                    np.zeros((xx.shape[1], 1))]
            # cond = [(np.exp(-(xx[0, :]-0.5)**2/.01))[:, None],
            #         np.zeros((xx.shape[1], 1))]
            return cond
        print(bndry_condition(
            np.hstack((np.linspace(0, 1, 11)[None, :], np.ones((1, 11))))))

        order = 20
        domain = [0, 1, 0, 1]
        bndry_conds = [
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"],
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"],
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"],
            [bndry_condition, "D"]]

        def forcing_fun(xx, zz):
            #return [np.zeros((xx.shape[1], 1)) for ii in range(3)]
            return [2*np.exp(2*xx[0, :]+2*xx[1, :])[:, None],
                    2*np.exp(2*xx[0, :]+2*xx[1, :])[:, None],
                    np.zeros((xx.shape[1], 1))]

        pres_idx = 0
        unique_pres_val = 0
        model = StokesFlowModel()
        model.initialize(bndry_conds, order, domain, forcing_fun)
        sample = np.zeros(0)  # dummy
        sol_vals = model.solve(sample, (pres_idx, unique_pres_val))

        rhs = model._split_quantities(model._rhs)
        assert np.allclose(
            sol_vals[0][model._mesh.boundary_indices[3], 0],
            rhs[0][model._mesh.boundary_indices[3], 0])
        # assert False

        # check value used to enforce unique pressure is found correctly
        # assert np.allclose(
        #     sol_vals[model._mesh.nphys_vars][pres_idx], unique_pres_val)

        num_pts_1d = 50
        plot_limits = domain
        from pyapprox.util.visualization import get_meshgrid_samples
        X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d)
        Z = model.interpolate(sol_vals, pts)
        print('v', Z[1].min(), Z[1].max())
        print('p', Z[2].min(), Z[2].max())

        Z = [np.reshape(zz, (X.shape[0], X.shape[1])) for zz in Z]
        fig, axs = plt.subplots(1, 4, figsize=(8*4, 6))
        axs[0].quiver(X, Y, Z[0], Z[1])
        from pyapprox.util.visualization import plot_2d_samples
        for ii in range(3):
            if Z[ii].min() != Z[ii].max():
                pl = axs[ii+1].contourf(
                    X, Y, Z[ii],
                    levels=np.linspace(Z[ii].min(), Z[ii].max(), 40))
                # plot_2d_samples(
                #     model._mesh.mesh_pts, ax=axs[ii+1], c='r', marker='o')
                # plot_2d_samples(
                #     model._pres_mesh.mesh_pts, ax=axs[ii+1], c='k', marker='o')
                plt.colorbar(pl, ax=axs[ii+1])
        print(model._pres_mesh.mesh_pts.min(axis=1),
              model._pres_mesh.mesh_pts.max(axis=1))
        print(model._mesh.mesh_pts.min(axis=1),
              model._mesh.mesh_pts.max(axis=1))
        plt.show()


if __name__ == "__main__":
    stokes_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestStokes)
    unittest.TextTestRunner(verbosity=2).run(stokes_test_suite)

#TODO clone conda environment when coding on branch
#diff spectral diffusion on master and on pde branch
#make sure that  _form_1d_derivative_matrices in this file is being called
#extract timestepper from advection diffusion and make its own object
#which just takes an steady state solver like diffusion or stokes and
#integrates it in time
