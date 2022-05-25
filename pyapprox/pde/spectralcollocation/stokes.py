import numpy as np

from pyapprox.util.utilities import cartesian_product
from pyapprox.pde.spectralcollocation.spectral_collocation import (
    OneDCollocationMesh, RectangularCollocationMesh, AbstractSpectralPDE,
    lagrange_polynomial_derivative_matrix_2d,
    lagrange_polynomial_derivative_matrix_1d
)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d
)


class InteriorOneDCollocationMesh(OneDCollocationMesh):
    def form_derivative_matrices(self):
        self.mesh_pts_1d = [
            -np.cos(np.linspace(0, np.pi, o+1))[1:-1] for o in self.order]
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
        # todo we call forcing fun twice which computes forcing on both meshes
        # which is wasted effort
        forcing[-1] = self.forcing_fun(
            self._pres_mesh.mesh_pts, sample[:, None])[-1]
        forcing = np.vstack(forcing)
        # we need another copy so that forcing can be used when solving adjoint
        self.forcing_vals = forcing.copy()
        # print([s[:, 0] for s in self._split_quantities(forcing)])
        self.collocation_matrix = self._form_collocation_matrix()
        matrix = self._apply_boundary_conditions_to_matrix(
            self.collocation_matrix.copy())
        rhs = self._apply_boundary_conditions_to_rhs(forcing.copy())

        # set pressure value at one location to make pressure unique
        matrix[self._mesh.nphys_vars*self._mesh.mesh_pts.shape[1]+pres[0], :] = 0
        matrix[self._mesh.nphys_vars*self._mesh.mesh_pts.shape[1]+pres[0],
               self._mesh.nphys_vars*self._mesh.mesh_pts.shape[1]+pres[0]] = 1
        rhs[self._mesh.nphys_vars*self._mesh.mesh_pts.shape[1]+pres[0], 0] = pres[1]
        
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
