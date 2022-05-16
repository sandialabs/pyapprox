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

from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d)


def lagrange_polyniomial_derivative_matrix_1d(eval_samples, abscissa):
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
    def _form_1d_derivative_matrices(self, order):
        mpts, der_mat = chebyshev_derivative_matrix(order)
        interior_pts = mpts[1:-1]
        der_mat_interior = lagrange_polyniomial_derivative_matrix_1d(
            interior_pts, interior_pts)[0]
        # der_mat = np.vstack((
        #     np.zeros((1, order-1)), der_mat_interior, np.zeros((1, order-1))))
        der_mat[1:-1, 1:-1] = der_mat_interior
        der_mat[0, :] = 0.
        der_mat[-1, :] = 0.
        return interior_pts, der_mat

    def _determine_boundary_indices(self):
        self._boundary_indices = None

    def _set_bndry_segment_interpolation_data(self):
        self._bndry_segment_canonical_abscissa = []
        self._bndry_segment_canonical_bary_weights = []


class InteriorRectangularCollocationMesh(RectangularCollocationMesh):
    def _form_1d_derivative_matrices(self, order):
        mpts, der_mat = chebyshev_derivative_matrix(order)
        interior_pts = mpts[1:-1]
        der_mat_interior = lagrange_polyniomial_derivative_matrix_1d(
            interior_pts, interior_pts)[0]
        # der_mat = np.vstack((
        #     np.zeros((1, order-1)), der_mat_interior, np.zeros((1, order-1))))
        der_mat[1:-1, 1:-1] = der_mat_interior
        der_mat[0, :] = 0.
        der_mat[-1, :] = 0.
        return interior_pts, der_mat

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
            Dmat = self._mesh.derivative_matrices[dd]
            vel_mat = Dmat.dot(Dmat)
            for ii in range(self._mesh.nphys_vars):
                if ii == dd:
                    sub_mats[-1][ii] = vel_mat
                else:
                    sub_mats[-1][ii] = np.zeros_like(vel_mat)
            pres_mat = self._pres_mesh.derivative_matrices[dd]
            sub_mats[-1][self._mesh.nphys_vars] = (
                -pres_mat[:, self._interior_indices])

        # divergence constraint
        sub_mats.append([])
        for dd in range(self._mesh.nphys_vars):
            Dmat = self._mesh.derivative_matrices[dd]
            Dmat_interior = Dmat[self._interior_indices, :]
            sub_mats[-1].append(Dmat_interior)
        pres_mat = self._pres_mesh.derivative_matrices[dd]
        sub_mats[-1].append(
            np.zeros((self._interior_indices.shape[0],
                      self._interior_indices.shape[0])))

        # for s in sub_mats:
        #     print([t.shape for t in s])

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

    def solve(self, sample, pres=1):
        forcing = self.forcing_fun(self._mesh.mesh_pts, sample[:, None])
        assert len(forcing) == self._mesh.nphys_vars+1
        for ii in range(len(forcing)):
            assert forcing[ii].ndim == 2 and forcing[ii].shape[1] == 1
        # forcing = forcing + [np.zeros((self._pres_mesh.mesh_pts.shape[1], 1))]
        forcing[-1] = forcing[-1][self._interior_indices]
        forcing = np.vstack(forcing)
        # we need another copy so that forcing can be used when solving adjoint
        self.forcing_vals = forcing.copy()
        # print([s[:, 0] for s in self._split_quantities(forcing)])
        self.collocation_matrix = self._form_collocation_matrix()
        matrix = self._apply_boundary_conditions_to_matrix(
            self.collocation_matrix.copy())
        rhs = self._apply_boundary_conditions_to_rhs(forcing)

        # set pressure value at one location to make pressure unique
        matrix[-1, :] = 0
        matrix[-1, -1] = 1
        # assert False
        rhs[-1, 0] = pres
        
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
        order = 4
        degree = order-2

        def fun(x): return x**degree
        def deriv(x): return degree*x**(degree-1)

        pts, deriv_mat_1d = chebyshev_derivative_matrix(order)

        deriv_mat_barycentric, basis_vals = (
            lagrange_polyniomial_derivative_matrix_1d(pts, pts))
        assert np.allclose(basis_vals.dot(fun(pts)), fun(pts))
        assert np.allclose(deriv_mat_1d, deriv_mat_barycentric)

        assert np.allclose(deriv_mat_1d.dot(fun(pts)), deriv(pts))

        interior_pts = pts[1:-1]
        deriv_mat_1d_interior = lagrange_polyniomial_derivative_matrix_1d(
            interior_pts, interior_pts)[0]
        assert np.allclose(
            deriv_mat_1d_interior.dot(fun(interior_pts)), deriv(interior_pts))

    def test_stokes_mms(self):
        np.set_printoptions(linewidth=150, threshold=2000)
        nphys_vars = 2
        sp_x, sp_y = sp.symbols(['x', 'y'])
        if nphys_vars == 2:
            domain = [0, 1, 0, 1]
            symbs = (sp_x, sp_y)
            velocity_strings = ["-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"]
            pressure_string = "x**2+y**2"
            order = 20
            # velocity_strings = ["-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"]
            # order = 4
            # velocity_strings = ["-x**3", "-y**3"]
            # pressure_string = "x**2+y**2"
        else:
            domain = [0, 1]
            symbs = (sp_x,)
            order = 4
            velocity_strings = ["-x**3"]
            pressure_string = "x**2"

        sp_pres = sp.sympify(pressure_string)
        sp_vel = [sp.sympify(s) for s in velocity_strings]
        sp_forc = [(vel.diff(s, 2)-sp_pres.diff(s, 1))
                   for vel, s in zip(sp_vel, symbs)]
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
        sol_vals = model.solve(sample, pres=exact_sol_vals[-1])
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
        # check pressure at all but point used for enforcing unique value
        # are set correctly
        assert np.allclose(
            recovered_forcing[nphys_vars][:-1],
            exact_forcing[nphys_vars][model._interior_indices][:-1])
        # check value used to enforce unique pressure is found correctly
        assert np.allclose(
            recovered_forcing[nphys_vars][-1], exact_sol_vals[-1])

        for exact_v, v in zip(exact_vel_vals, sol_vals[:-1]):
            assert np.allclose(exact_v, v)

        num_pts_1d = 100
        plot_limits = domain
        from pyapprox.util.visualization import get_meshgrid_samples
        X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d)
        Z = model.interpolate(sol_vals, pts)
        for ii in range(model._mesh.nphys_vars):
            assert np.allclose(Z[ii], exact_vel(pts)[ii])
            Z[ii] = np.reshape(Z[ii], (X.shape[0], X.shape[1]))
        assert np.allclose(Z[-1], exact_pres(pts))
        Z[-1] = np.reshape(Z[-1], (X.shape[0], X.shape[1]))
        plt.quiver(X, Y, Z[0], Z[1])
        plt.show()

    def lid_driven_cavity_flow(self):
        # todo ensure boundary conditions are being enforced exactly
        # they are not yet

        # drive cavity using left boundary as top and bottom boundaries
        # do not hvae dof at the corners
        def bndry_condition(xx):
            cond = [np.zeros((xx.shape[1], 1)),
                    (16*xx[1, :]**2*(1-xx[1, :]**2))[:, None]]
            return cond

        order = 128
        domain = [0, 1, 0, 1]
        bndry_conds = [
            [bndry_condition, "D"],
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"],
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"],
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"]]

        def forcing_fun(xx, zz):
            return [np.zeros((xx.shape[1], 1)) for ii in range(3)]
        
        model = StokesFlowModel()
        model.initialize(bndry_conds, order, domain, forcing_fun)
        sample = np.zeros(0)  # dummy
        sol_vals = model.solve(sample)

        rhs = model._split_quantities(model._rhs)
        assert np.allclose(
            sol_vals[1][model._mesh.boundary_indices[0], 0],
            rhs[1][model._mesh.boundary_indices[0], 0])
        # assert False

        num_pts_1d = 100
        plot_limits = domain
        from pyapprox.util.visualization import get_meshgrid_samples
        X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d)
        Z = model.interpolate(sol_vals, pts)
        II = np.where(pts[0, :] == 0)[0]
        print(pts[:, II])
        print(Z[1][II, 0])
        Z = [np.reshape(zz, (X.shape[0], X.shape[1])) for zz in Z]
        print(sol_vals[0][model._mesh.boundary_indices[0], 0])
        print(sol_vals[1][model._mesh.boundary_indices[0], 0])
        print(sol_vals[-1].max())
        plt.quiver(X, Y, Z[0], Z[1])
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
