import unittest
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

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
        self._mesh = None

    def _set_domain(self, domain, order):
        if len(domain) == 2:
            self.mesh = OneDCollocationMesh(domain, order)
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
        mpts, der_mat = chebyshev_derivative_matrix(self.order[ii])
        interior_pts = mpts[1:-1]
        deriv_mat_interior = lagrange_polyniomial_derivative_matrix_1d(
            interior_pts, interior_pts)[0]
        return interior_pts, deriv_mat_interior

class InteriorRectangularCollocationMesh(RectangularCollocationMesh):
    def _form_1d_derivative_matrices(self, order):
        mpts, der_mat = chebyshev_derivative_matrix(self.order[ii])
        interior_pts = mpts[1:-1]
        deriv_mat_interior = lagrange_polyniomial_derivative_matrix_1d(
            interior_pts, interior_pts)[0]
        print(order, interior_pts.shape, deriv_mat_interior.shape)
        assert False
        return interior_pts, deriv_mat_interior


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
                    sub_mats[-1][ii] = vel_mat*0
            pres_mat = self._pres_mesh.derivative_matrices[dd]
            print(vel_mat.shape, pres_mat.shape)
            sub_mats[-1][self._mesh.nphys_vars] = pres_mat
        assert False
        # divergence constraint
        sub_mats.append([])
        for dd in range(self._mesh.nphys_vars):
            Dmat = self._mesh.derivative_matrices[dd]
            sub_mats[-1].append(Dmat)
        sub_mats[-1].append(0*Dmat)

        return np.block(sub_mats)

    def get_num_degrees_of_freedom(self, config_sample):
        return np.prod(config_sample)

    def _apply_boundary_conditions_to_matrix(self, matrix):
        nvel_dof = self._mesh.mesh_pts.shape[1]
        cnt = 0
        for dd in range(self._mesh.nphys_vars):
            vel_matrix = matrix[cnt:cnt+nvel_dof, :]
            # TODO this will not work for neumann boundaries because
            # _apply_neumann_boundary_conditions_to_matrix currently
            # only sets row to derivative matrix but this will not have
            # the correct number of columns
            vel_matrix = self._mesh._apply_boundary_conditions_to_matrix(
                vel_matrix)
            matrix[cnt:cnt+nvel_dof, :] = vel_matrix
            cnt += nvel_dof
        return matrix

    def _apply_boundary_conditions_to_rhs(self, rhs):
        nvel_dof = self._mesh.mesh_pts.shape[1]
        for ii, bndry_cond in enumerate(self._mesh.bndry_conds):
            idx = self._mesh.boundary_indices[ii]
            bndry_vals = bndry_cond[0](self._mesh.mesh_pts[:, idx])
            for dd in range(len(bndry_vals)):
                for kk in range(self._mesh.nphys_vars):
                    rhs[idx+kk*nvel_dof] = bndry_vals[dd]
        return rhs

    def solve(self, sample):
        forcing = self.forcing_fun(self._mesh.mesh_pts, sample[:, None])
        assert len(forcing) == self._mesh.nphys_vars
        for ii in range(len(forcing)):
            assert forcing[ii].ndim == 2 and forcing[ii].shape[1] == 1
        # forcing will be overwritten with bounary values so must take a
        # deep 
        forcing = forcing + [np.zeros((self._pres_mesh.mesh_pts.shape[1], 1))]
        forcing = np.vstack(forcing)
        # we need another copy so that forcing can be used when solving adjoint
        self.forcing_vals = forcing.copy()
        self.collocation_matrix = self._form_collocation_matrix()
        matrix = self._apply_boundary_conditions_to_matrix(
            self.collocation_matrix.copy())
        # print(self._split_quantities(matrix)[0])
        forcing = self._apply_boundary_conditions_to_rhs(forcing)
        print(self._split_quantities(forcing))
        solution = np.linalg.solve(matrix, forcing)
        split_solutions = self._split_quantities(solution)
        return split_solutions

    def _split_quantities(self, vector):
        nvel_dof = self._mesh.mesh_pts.shape[1]
        split = [vector[ii*nvel_dof:(ii+1)*nvel_dof]
                 for ii in range(self._mesh.nphys_vars)]
        split.append(vector[self._mesh.nphys_vars*nvel_dof:])
        return split


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

    def test_stokes_2d(self):
        sp_x, sp_y = sp.symbols(['x', 'y'])
        symbs = (sp_x, sp_y)
        velocity_strings = ["-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"]
        pressure_string = "x**2+y**2"
        domain = [0, 1, 0, 1]
        # symbs = (sp_x, )
        # velocity_strings = ["-cos(pi*x)"]
        # pressure_string = "x**2"
        # domain = [0, 1]
        
        sp_pres = sp.sympify(pressure_string)
        sp_vel = [sp.sympify(s) for s in velocity_strings]
        sp_forc = [(vel.diff(s, 2)-sp_pres.diff(s, 1))
                   for vel, s in zip(sp_vel, symbs)]
        print(sp_forc)
        div = sum([vel.diff(s, 1) for vel, s in zip(sp_vel, symbs)])
        # assert (div == 0)
        exact_vel_lambda = [sp.lambdify(symbs, fun, "numpy")
                            for fun in sp_vel]
        exact_pres_lambda = sp.lambdify(symbs, sp_pres, "numpy")
        def exact_vel(xx):
            vals = [fun(*[x for x in xx])[:, None] for fun in exact_vel_lambda]
            return vals
        def exact_pres(xx):
            vals = exact_pres_lambda(*[x for x in xx])[:, None]
            return vals

        order = 5
        # bndry_conds = [[lambda x: exact_vel(x), "D"],
        #                [lambda x: exact_vel(x), "D"],
        #                [lambda x: exact_vel(x), "D"],
        #                [lambda x: exact_vel(x), "D"]]

        # todo ensure boundary conditions are being enforced exactly
        # they are not yet
        bndry_conds = [[lambda x: [90*np.ones((x.shape[1], 1))]*2, "D"],
                       [lambda x: [90*np.ones((x.shape[1], 1))]*2, "D"],
                       [lambda x: [90*np.ones((x.shape[1], 1))]*2, "D"],
                       [lambda x: [90*np.ones((x.shape[1], 1))]*2, "D"]]

        bndry_conds = bndry_conds[:len(domain)]
        
        model = StokesFlowModel()
        forc_fun_np = sp.lambdify(symbs, sp_forc, "numpy")
        def forcing_fun(xx, z):
            vals = forc_fun_np(*[x for x in xx]) # vals is a list
            return [v[:, None] for v in vals]
        model.initialize(bndry_conds, order, domain, forcing_fun)

        sample = np.zeros(0) # dummy
        sol = model.solve(sample)

        print(sol[0])

        exact_vel_vals = exact_vel(model._mesh.mesh_pts)
        exact_pres_vals = exact_pres(model._pres_mesh.mesh_pts)

        num_pts_1d = 30
        plot_limits = domain
        from pyapprox.util.visualization import get_meshgrid_samples
        X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d)
        Z = exact_vel(pts)
        # print(sol[1])
        # print(exact_vel_vals[0])
        # plt.quiver(X, Y, Z[0], Z[1])
        # plt.show()


if __name__ == "__main__":
    stokes_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestStokes)
    unittest.TextTestRunner(verbosity=2).run(stokes_test_suite)

#TODO clone conda environment when coding on branch
#diff spectral diffusion on master and on pde branch
#make sure that  _form_1d_derivative_matrices in this file is being calle
