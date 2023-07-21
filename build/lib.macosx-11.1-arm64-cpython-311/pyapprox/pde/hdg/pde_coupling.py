import numpy as np
from abc import ABC, abstractmethod
import matplotlib.tri as tri
import torch

from pyapprox.util.utilities import cartesian_product
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d, univariate_lagrange_polynomial)
from pyapprox.pde.autopde.mesh import CanonicalCollocationMesh
from pyapprox.pde.autopde.util import newton_solve
from pyapprox.util.utilities import cartesian_product, outer_product
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D


class SubdomainInterface(ABC):
    def __init__(self, ndof):
        self._set_ndof(ndof)

    @abstractmethod
    def _set_ndof(self, ndof):
        raise NotImplementedError()

    def _set_values(self, values):
        if values.shape[0] != self._ndof:
            raise ValueError("values.shape is inconsistent with self._ndof")
        self._values = values

    @abstractmethod
    def _interpolate(self, subdomain_pts):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_from_subdomain(self, subdomain, bndry_seg, flux):
        raise NotImplementedError()

    @abstractmethod
    def _from_interface_basis(self, subdomain, bndry_seg):
        raise NotImplementedError()

    @abstractmethod
    def _to_interface_basis(self, subdomain, bndry_seg):
        raise NotImplementedError()


class SubdomainInterface0D(SubdomainInterface):
    def _set_ndof(self, ndof):
        if ndof != 1:
            raise ValueError("OD interface can only have one DOF")
        self._ndof = 1

    def _interpolate(self, subdomain_pts):
        if subdomain_pts.shape[1] != self._values.shape[0]:
            msg = "subdomain_pts.shape is inconsistent with self._values"
            raise ValueError(msg)
        return self._values

    def _interpolate_from_subdomain(self, subdomain, bndry_seg, flux):
        return flux[:, None]

    def _from_interface_basis(self, subdomain, bndry_seg):
        return torch.tensor([[1]])

    def _to_interface_basis(self, subdomain, bndry_seg):
        return torch.tensor([[1]])


class SubdomainInterface1D(SubdomainInterface):
    def __init__(self, nmesh_pts, left_active_dim, left_transform,
                 left_bndry_id):
        # left and right transforms are for first in
        # self._interface_to_bndry_map[jj]
        # and second entry in the same map for interface[jj]
        self._ndof = None
        self._canonical_mesh_pts_1d = None
        self._canonical_mesh_pts = None
        self._left_transform = left_transform
        self._left_bndry_id = left_bndry_id
        self._left_active_dim = left_active_dim

        super().__init__(nmesh_pts)
        self._nphys_vars = 2
        self._canonical_domain = np.ones(2*self._nphys_vars)
        self._canonical_domain[::2] = -1.
        self._set_ndof(nmesh_pts)
        self._barycentric_weights_1d = [
            compute_barycentric_weights_1d(xx)
            for xx in self._canonical_mesh_pts_1d]

        self._subdomain_pts = []

        self._to_iface_bases = [None for ii in range(4)]

    def _set_ndof(self, ndof):
        # currently exact computation of dirichlet to neumann jacobian
        # assumes that boundary points are subsets of mesh points that lie on
        # boundary. This is purely for convienience so that additional
        # chain rule steps do not need to be applied.
        self._canonical_mesh_pts_1d = [
            -np.cos(np.linspace(0., np.pi, ndof+2))[1:-1]]
        # eps = 1e-2
        # self._canonical_mesh_pts_1d = [
        #     -np.cos(np.linspace(eps, np.pi-eps, ndof))]
        # from pyapprox.surrogates.orthopoly.quadrature import (
        #     gauss_jacobi_pts_wts_1D)
        # self._canonical_mesh_pts_1d = [
        #     gauss_jacobi_pts_wts_1D(ndof, 0, 0)[0]]
        self._canonical_mesh_pts = cartesian_product(
            self._canonical_mesh_pts_1d)
        self._ndof = self._canonical_mesh_pts.shape[1]

    def _canonical_interpolate(
            self, canonical_subdomain_pts, canonical_mesh_pts_1d):
        # print(canonical_subdomain_pts.shape, canonical_mesh_pts_1d.shape)
        if torch.is_tensor(self._values):
            values = self._values[:, 0].numpy()
        else:
            values = self._values[:, 0]
        # interp_vals = barycentric_interpolation_1d(
        #     canonical_mesh_pts_1d,
        #     self._barycentric_weights_1d[0], values,
        #     canonical_subdomain_pts[0, :])[:, None]
        basis = univariate_lagrange_polynomial(
            self._canonical_mesh_pts[0],
            canonical_subdomain_pts[0, :])
        interp_vals = basis.dot(values)[:, None]
        return interp_vals

    def _interpolate(self, subdomain_pts):
        # called when subdomain dirichlet boundary conditions interfaces
        # are called
        # basis = [basis for pts, basis in zip(self._subdomain_pts, self._bases)
        #          if pts.shape == subdomain_pts.shape and np.linalg.norm(
        #                  pts-subdomain_pts) < 1e-12]
        # # for pts, basis in zip(self._subdomain_pts, self._bases):
        # #     print(np.linalg.norm(pts-subdomain_pts) < 1e-12)
        # # print(basis)
        # # print(self._bases)
        # if len(basis) == 1:
        #     return basis[0].dot(self._values)
        # self._subdomain_pts.append(subdomain_pts)
        canonical_subdomain_pts = self._left_transform.map_to_orthogonal(
            subdomain_pts)[self._left_active_dim:self._left_active_dim+1]
        interp_vals = self._canonical_interpolate(
            canonical_subdomain_pts, self._canonical_mesh_pts_1d[0])
        return interp_vals

    def _interpolate_from_subdomain(self, subdomain, bndry_seg, flux):
        # if self._to_iface_basis is None:
        self._to_iface_bases[bndry_seg] = self._to_interface_basis(
            subdomain, bndry_seg)
        return self._to_iface_bases[bndry_seg].dot(flux.numpy())[:, None]

    def _from_interface_basis(self, subdomain, bndry_seg):
        # cannot use self._active_dim which only applies to subdomain
        # left of interface
        mesh = subdomain.physics.mesh
        idx = mesh._bndry_indices[bndry_seg]
        bndry_pts = mesh._bndry_slice(mesh.mesh_pts, idx, 1)
        basis = univariate_lagrange_polynomial(
            self._canonical_mesh_pts[0],
            self._left_transform.map_to_orthogonal(
                bndry_pts)[self._left_active_dim])
        return basis

    def _to_interface_basis(self, subdomain, bndry_seg):
        # cannot use self.active_dim which only applies to subdomain
        # left of interface
        # active_dim = int(bndry_seg < 2)
        mesh = subdomain.physics.mesh
        idx = mesh._bndry_indices[bndry_seg]
        if bndry_seg >= 2:
            # add missing corners (deleted so degrees of freedom on boundary
            # are unique) back in so interpolation is more accurate
            idx = np.hstack(([idx[0]-1], idx, idx[-1]+1))
        bndry_pts = mesh._bndry_slice(mesh.mesh_pts, idx, 1)
        basis = univariate_lagrange_polynomial(
            # mesh._transform.map_to_orthogonal(bndry_pts)[active_dim],
            self._left_transform.map_to_orthogonal(
                bndry_pts)[self._left_active_dim],
            self._canonical_mesh_pts[0])
        return basis

    def _mesh_pts(self):
        # self._canonical_mesh_pts is only 1D varing along active dimension
        # i.e. dimension in (canonical coordinates) that varies along the
        # boundary
        canonical_mesh_pts = np.empty((2, self._canonical_mesh_pts.shape[1]))
        canonical_mesh_pts[self._left_active_dim] = self._canonical_mesh_pts
        # if horizontal inteface (active_dim==0)
        # then inactive canonical coordinate
        # will be -1 because we assume transform is from bottom subdomain
        # if vertical inteface (active_dim==1) then
        # inactive canonical coordinate
        # will be 1 because we assume transform is from left subdomain
        canonical_mesh_pts[(self._left_active_dim+1) % 2] = (-1)**(
            self._left_bndry_id+1)
        return self._left_transform.map_from_orthogonal(canonical_mesh_pts)

    def __str__(self):
        string = self.__class__.__name__+":"
        string += f"\n\tactive_dim: {self._left_active_dim}"
        string += f"\n\tmesh_pts: {self._mesh_pts()}"
        return string

    def __repr__(self):
        return self.__str__()+"\n"


class AbstractDomainDecomposition(ABC):
    # Note that when switching to allow for arbitray points on interface that
    # dirichlet vals and thus
    # boundary conditions are not recovered exactly when
    # ninterface_dof > order-1. In these cases there is not enough
    # information to determine the coefficients of the interface
    # interpolating polynomial exactly. The true solution can
    # still be recovered exactly though as the interface polynomial
    # will interpolate the true solution at the boundary values
    # of the subdomain models
    def __init__(self):
        self._nsubdomains = None

        # self._interface_to_bndry_map contains tuples
        # [subdomain1, bdnry_seg1, subdomain2, bdnry_seg2]
        # which are the subdomain and bndry segment index of left subdomain
        # and right subdomain. E.g. in 1D [0, 1, 1, 0] is the entry for the
        # first interface between subdomains 1 and 2 which share a right
        # and left boundary respectively
        self._interface_to_bndry_map = None
        # self._subdomain_to_interface_map contains the global indices of the
        # intefaces that are shared by each subdomain. E.g. in 1D
        # subdomain 0 has [0] and subdomain 1 has [0, 1]
        self._subdomain_to_interface_map = None
        # self._subdomain_interface_bndry_indices contains
        # indices of boundaries that of each subdomain that lie
        # on an interface. In 2D there are four boundaries so indices
        # of interior subdomain are [0, 1, 2, 3], whereas indices
        # of bottom left subdomain are [1, 3]. Boundaries labeled
        # left, right, bottom top
        self._subdomain_interface_bndry_indices = None
        # self._interface_dof_starts contains the start index in the global
        # array of all interface dofs that contains the degeres of freedom
        # associated with each interface
        self._interface_dof_starts = None
        self._interfaces = None
        self._ninterfaces = None
        self._subdomain_models = None
        # _solve subdomain must be set by DomainDecompositionSolvers
        self._solve_subdomain = None

        self._iface_bases = []
        self._normals = None

        # updated with sols of each subdomain at each newton iteration
        # After newton sol the final solutions can be obtained from this
        # variable
        self._sols = None

    def get_subdomain_adjacency_matrix(self):
        # TODO make this a sparse matrix
        adjacency_mat = np.zeros((self._ninterfaces, self._nsubdomains))
        for ii in range(self._ninterfaces):
            adjacency_mat[ii, self._interface_to_bndry_map[ii, ::2]] = 1
        return adjacency_mat

    def get_ninterfaces_dof(self):
        return np.sum([iface._ndof for iface in self._interfaces])

    def get_interface_dof_adjacency_matrix(self):
        subdomain_adj_mat = self.get_subdomain_adjacency_matrix()
        ninterfaces_dof = self.get_ninterfaces_dof()
        dof_adj_mat = []
        for ii in range(self._ninterfaces):
            rows_ii = np.zeros(
                (self._interfaces[ii]._ndof, ninterfaces_dof))
            for neigh in range(self._nsubdomains):
                if subdomain_adj_mat[ii, neigh] == 1:
                    for iface in self._subdomain_to_interface_map[neigh]:
                        cnt = self._interface_dof_starts[iface]
                        rows_ii[:, cnt:cnt+self._interfaces[iface]._ndof] = 1.0
            dof_adj_mat.append(rows_ii)
        return np.vstack(dof_adj_mat)

    def _get_active_subdomains(self):
        active_subdomains = np.zeros(
            (self._nsubdomains, self._nsubdomains), dtype=int)
        for ii in range(self._nsubdomains):
            active = []
            for iface in self._subdomain_to_interface_map[ii]:
                neigh0, segm0, neigh1, segm1 = (
                    self._interface_to_bndry_map[iface])
                active += [neigh0, neigh1]
            active_subdomains[ii,  active] = 1
        return active_subdomains

    def _set_dirichlet_values(self, dirichlet_vals):
        if dirichlet_vals.ndim != 2 or dirichlet_vals.shape[1] != 1:
            raise ValueError("dirichlet_vals must be column vector")
        cnt = 0
        # update boundary conditions along interfaces
        for ii in range(self._ninterfaces):
            self._interfaces[ii]._set_values(
                dirichlet_vals[cnt:cnt+self._interfaces[ii]._ndof])
            cnt += self._interfaces[ii]._ndof

    def _invert_drdu(self, drdu, jj):
        return torch.linalg.inv(drdu)

    def _compute_interface_fluxes(self, jj, **subdomain_newton_kwargs):
        physics = self._subdomain_models[jj].physics
        sol, drdu = self._solve_subdomain(jj, **subdomain_newton_kwargs)
        drdu_inv = self._invert_drdu(drdu, jj)
        flux, drdp = [], []
        mesh = physics.mesh
        # tmp = [-drdu_inv[:, mesh._bndry_indices[ii]]
        #        for ii in self._subdomain_interface_bndry_indices[jj]]
        tmp = [-mesh._bndry_slice(drdu_inv, mesh._bndry_indices[ii], 1)
               for ii in self._subdomain_interface_bndry_indices[jj]]
        if self._normals is None:
            self._normals = [[None for ii in range(mesh.nphys_vars*2)]
                             for jj in range(self._nsubdomains)]
        for ii, bndry_id in enumerate(
                self._subdomain_interface_bndry_indices[jj]):
            idx = mesh._bndry_indices[bndry_id]
            if bndry_id >= 2:
                # add missing corners (deleted so degrees of freedom on
                # boundary are unique) back in so interpolation is more
                # accurate
                idx = np.hstack(([idx[0]-1], idx, idx[-1]+1))
            if self._normals[jj][ii] is None:
                normal_vals_ii = mesh._bndrys[bndry_id].normals(
                    mesh._bndry_slice(mesh.mesh_pts, idx, 1))
                self._normals[jj][ii] = normal_vals_ii
            else:
                normal_vals_ii = self._normals[jj][ii]
            physics._store_data = True
            dfdu_ii = physics._scalar_flux_jac(
                physics.mesh, idx)
            physics._store_data = False
            dfdu_ii_n = sum([
                normal_vals_ii[:, dd:dd+1]*dfdu_ii[dd]
                for dd in range(mesh.nphys_vars)])
            drdp.append(
                [torch.linalg.multi_dot((-dfdu_ii_n, tt)) for tt in tmp])
            flux.append(torch.linalg.multi_dot((dfdu_ii_n, sol)))
        return flux, drdp, sol

    def _compute_subdomain_flux_jacobians(self, dirichlet_vals,
                                          **subdomain_newton_kwargs):
        self._set_dirichlet_values(dirichlet_vals)
        flux_jacobians = []
        fluxes = []
        self._sols = []
        for jj in range(self._nsubdomains):
            flux, drdp, sol = self._compute_interface_fluxes(
                jj, **subdomain_newton_kwargs)
            flux_jacobians.append(drdp)
            fluxes.append(flux)
            self._sols.append(sol[:, None])
        return fluxes, flux_jacobians

    def _assemble_dirichlet_neumann_map_jacobian(
            self, dirichlet_vals, return_jac_parts=False,
            **subdomain_newton_kwargs):
        fluxes, flux_jacobians = self._compute_subdomain_flux_jacobians(
            dirichlet_vals, **subdomain_newton_kwargs)
        ndof = self.get_ninterfaces_dof()
        jac = torch.zeros((ndof, ndof), dtype=torch.double)
        residual = torch.zeros(ndof, dtype=torch.double)
        # jac_parts only useful for testing
        jac_parts = [[] for ii in range(self._nsubdomains)]
        cnt = 0
        store = False
        if len(self._iface_bases) == 0:
            store = True
            # self._to_interface_bases = []
        for ii in range(self._nsubdomains):
            for mm, iface1 in enumerate(self._subdomain_to_interface_map[ii]):
                # The values of fluxes, i.e. different rows of Jacobian
                cnt1 = self._interface_dof_starts[iface1]
                II = np.where(
                    self._interface_to_bndry_map[iface1][::2] == ii)[0][0]
                bndry_id1 = self._interface_to_bndry_map[iface1][2*II+1]
                interface1 = self._interfaces[iface1]
                residual[cnt1:cnt1+interface1._ndof] += (
                    interface1._interpolate_from_subdomain(
                        self._subdomain_models[ii], bndry_id1,
                        fluxes[ii][mm])[:, 0])
                if store:
                    basis_to_iface1 = torch.as_tensor(
                        interface1._to_interface_basis(
                            self._subdomain_models[ii], bndry_id1),
                        dtype=torch.double)
                    self._iface_bases.append(basis_to_iface1)
                basis_to_iface1 = self._iface_bases[cnt]
                cnt += 1

                jac_parts[ii].append([])
                for nn, iface2 in enumerate(
                        self._subdomain_to_interface_map[ii]):
                    # The dirichlet values that affect the fluxes in outer loop
                    # i.e. different variables (columns) of Jacobian
                    cnt2 = self._interface_dof_starts[iface2]
                    interface2 = self._interfaces[iface2]
                    JJ = np.where(
                        self._interface_to_bndry_map[iface2][::2] == ii)[0][0]
                    bndry_id2 = self._interface_to_bndry_map[iface2][2*JJ+1]
                    if store:
                        basis_from_iface2 = torch.as_tensor(
                            interface2._from_interface_basis(
                                self._subdomain_models[ii], bndry_id2),
                            dtype=torch.double)
                        self._iface_bases.append(basis_from_iface2)
                    basis_from_iface2 = self._iface_bases[cnt]
                    cnt += 1

                    jac_part = torch.linalg.multi_dot((
                        flux_jacobians[ii][mm][nn], basis_from_iface2))
                    jac_part = torch.linalg.multi_dot(
                        (basis_to_iface1, jac_part))
                    if return_jac_parts:
                        jac_parts[ii][mm].append(jac_part)
                    jac[cnt1:cnt1+interface1._ndof,
                        cnt2:cnt2+interface2._ndof] += (jac_part)
        if not return_jac_parts:
            return residual, jac
        return residual, jac, jac_parts

    def _compute_interface_values(self, dirichlet_vals,
                                  subdomain_newton_kwargs,
                                  **macro_newton_kwargs):
        dirichlet_vals = newton_solve(
            lambda d: self._assemble_dirichlet_neumann_map_jacobian(
                d[:, None], **subdomain_newton_kwargs),
            torch.as_tensor(dirichlet_vals[:, 0], dtype=torch.double),
            **macro_newton_kwargs)
        return dirichlet_vals[:, None]

    def _set_subdomain_interface_boundary_conditions(self):
        self._subdomain_interface_bndry_indices = [
            [] for jj in range(self._nsubdomains)]
        for ii in range(self._ninterfaces):
            for neighbor_idx, bndry_idx in zip(
                    self._interface_to_bndry_map[ii][::2],
                    self._interface_to_bndry_map[ii][1::2]):
                self._subdomain_interface_bndry_indices[neighbor_idx].append(
                    bndry_idx)
                neigh_model = self._subdomain_models[neighbor_idx]
                # using partial here will not work because set_values will be
                # ignored
                neigh_model.physics._bndry_conds[bndry_idx] = [
                    self._interfaces[ii]._interpolate, "D"]
        # convert inner list of indices to arrays for special indexing
        # latter
        self._subdomain_interface_bndry_indices = [
            np.array(inds) for inds in self._subdomain_interface_bndry_indices]

    def mesh_points(self):
        """
        Return all the mesh points of all the subdomains
        """
        return np.hstack(
            [model.physics.mesh.mesh_pts for model in self._subdomain_models])

    def init_subdomains(self, init_subdomain_model):
        self._subdomain_transforms = self._define_subdomain_transforms()
        self._set_interface_data()
        self._subdomain_models = [
            init_subdomain_model(self._subdomain_transforms[ii], ii)
            for ii in range(self._nsubdomains)]
        self._set_subdomain_interface_boundary_conditions()

    def _find_interfaces(self, subdomain_id):
        npts = 21
        s = np.linspace(-1, 1, npts)[None, :]
        # boundaries ordered left, right, bottom, top
        # this ordering is assumed by tests
        orth_lines = [
            np.vstack((np.full(s.shape, -1), s)),
            np.vstack((np.full(s.shape, 1), s)),
            np.vstack((s, np.full(s.shape, -1))),
            np.vstack((s, np.full(s.shape, 1)))]
        transform = self._subdomain_transforms[subdomain_id]
        # some transforms will not have bottom orthogonal boundary
        # on the bottom in the user domain so must loop over all boundaries
        # colors = ["k", "r", "b", "g"]
        # ls = ["-"]*4 #[["-", "--", "-.", ":"][subdomain_id]]*4
        for jj in range(4):
            orth_active_dim = int(jj < 2)
            orth_inactive_dim = int(jj >= 2)
            orth_line = orth_lines[jj]
            # line = transform.map_from_orthogonal(orth_line)
            # plt.plot(line[0], line[1], color=colors[jj], ls=ls[jj], lw=0.5,
            #          label=(subdomain_id, jj))
            found = False
            for neigh_id in range(subdomain_id+1, self._nsubdomains):
                neigh_transform = self._subdomain_transforms[neigh_id]
                # neigh_orth_line = neigh_transform.map_to_orthogonal(line)
                for kk in range(4):
                    #  neigh_orth_inactive_dim = int(kk >= 2)
                    # neigh_orth_active_dim = int(kk < 2)
                    # inactive orth dimensions must be the same and
                    # ends of boundary must be the same in active direction
                    # points in this may not be the same. If different transform types
                    # are used for the domains on either side of the interface, e.g.
                    # a polar transform and a sympy transform with y=np.sqrt(r**2-x**2)
                    if (np.allclose(
                            transform.map_to_orthogonal(
                                neigh_transform.map_from_orthogonal(orth_lines[kk]))[
                                    orth_inactive_dim],
                            orth_line[orth_inactive_dim]) and
                        np.allclose(
                            [-1, 1],
                            np.sort(transform.map_to_orthogonal(
                                neigh_transform.map_from_orthogonal(orth_lines[kk]))[
                                    orth_active_dim, [0, -1]]))):
                        # An interface exists
                        interface_cnt = len(self._interfaces)
                        self._interface_to_bndry_map.append(
                            [subdomain_id, jj, neigh_id, kk])
                        self._interfaces.append(SubdomainInterface1D(
                            self._ninterface_dof, orth_active_dim,
                            self._subdomain_transforms[subdomain_id], jj))
                        self._subdomain_to_interface_map[subdomain_id].append(
                            interface_cnt)
                        self._subdomain_to_interface_map[neigh_id].append(
                            interface_cnt)
                        found = True
                if found:
                    break

    def _set_interface_data(self):
        assert self._nsubdomains == len(self._subdomain_transforms)
        self._interface_to_bndry_map = []
        self._interfaces = []
        self._subdomain_to_interface_map = [
            [] for jj in range(self._nsubdomains)]
        for subdomain_id in range(self._nsubdomains):
            self._find_interfaces(subdomain_id)
            # the above procedure relies on subdomains being defined correctly
            # if not all subdomains line up exactly, then the assert below will
            # fail
        if len(self._interfaces) != self._ninterfaces:
            msg = f"number of interfaces found {len(self._interfaces)}"
            msg += f" does not match number specified {self._ninterfaces}"
            raise ValueError(msg)
        self._interface_to_bndry_map = np.array(self._interface_to_bndry_map)
        self._interface_dof_starts = np.hstack((0, np.cumsum(
            [iface._ndof for iface in self._interfaces])[:-1]))

    @abstractmethod
    def _define_subdomain_transforms(self):
        raise NotImplementedError()

    @abstractmethod
    def interpolate(self, subdomain_mesh_values, samples):
        raise NotImplementedError()

    @abstractmethod
    def interface_mesh(self):
        raise NotImplementedError()

    def subdomain_quadrature_data(self):
        subdomain_rules = []
        for ii, model in enumerate(self._subdomain_models):
            subdomain_rules.append(model.physics.mesh._get_quadrature_rule())
        return subdomain_rules

    def integrate(self, subdomain_vals):
        val = 0
        subdomain_quad_data = self.subdomain_quadrature_data()
        for ii, quad_data in enumerate(subdomain_quad_data):
            interp_vals = self._subdomain_models[ii].physics.mesh.interpolate(
                subdomain_vals[ii],  quad_data[0])
            val += np.asarray(interp_vals[:, 0]).dot(quad_data[1])
        return val


class OneDDomainDecomposition(AbstractDomainDecomposition):
    def __init__(self, bounds, nsubdomains, ninterface_dof, intervals=None):
        super().__init__()
        self._bounds = bounds
        self._nsubdomains = nsubdomains
        self._ninterfaces = self._nsubdomains-1
        self._ninterface_dof = ninterface_dof

        if intervals is None:
            intervals = np.linspace(
                self._bounds[0], self._bounds[1], nsubdomains+1)
        if (len(intervals) != nsubdomains+1):
            raise ValueError("intervals does not match nsubdomains")
        if (not np.allclose(intervals[[0, -1]], self._bounds[:2])):
            raise ValueError("intervals does not match bounds")
        self._intervals = intervals
        self._subdomain_bounds = self._set_subdomain_bounds()

    def _set_subdomain_bounds(self):
        subdomain_bounds = np.empty(self._nsubdomains*2)
        subdomain_bounds[::2] = self._intervals[:-1]
        subdomain_bounds[1::2] = self._intervals[1:]
        subdomain_bounds = subdomain_bounds.reshape(self._nsubdomains, 2)
        return subdomain_bounds

    def _define_subdomain_transforms(self):
        nphys_vars = 1
        canonical_domain_bounds = (
            CanonicalCollocationMesh._get_canonical_domain_bounds(
                nphys_vars,
                CanonicalCollocationMesh._get_basis_types(nphys_vars, None)))
        transforms = []
        for bounds in self._subdomain_bounds:
            transforms.append(ScaleAndTranslationTransform(
                canonical_domain_bounds, bounds))
        return transforms

    def _set_interface_data(self):
        self._interface_to_bndry_map = np.array([
            [ii, 1, ii+1, 0] for ii in range(self._ninterfaces)])
        self._interfaces = [
            SubdomainInterface0D(1) for ii in range(self._ninterfaces)]
        self._subdomain_to_interface_map = (
            [[0]]+[[ii-1, ii] for ii in range(1, self._nsubdomains-1)] +
            [[self._nsubdomains-2]])
        self._interface_dof_starts = np.hstack((0, np.cumsum(
            [iface._ndof for iface in self._interfaces])[:-1]))

    def _in_subdomains(self, samples):
        masks = []
        for (lb, ub) in self._subdomain_bounds[:-1]:
            masks.append(
                np.where((samples[0, :] >= lb) & (samples[0, :] < ub))[0])
        lb, ub = self._subdomain_bounds[-1]
        masks.append(
            np.where((samples[0, :] >= lb) & (samples[0, :] <= ub))[0])
        return masks

    def interpolate(self, subdomain_mesh_values, samples):
        sample_mask_per_subdomain = self._in_subdomains(samples)
        if subdomain_mesh_values[0].ndim == 2:
            ncols = subdomain_mesh_values[0].shape[1]
        else:
            ncols = 1
        values = np.ones((samples.shape[1], ncols))
        for jj in range(self._nsubdomains):
            values[sample_mask_per_subdomain[jj]] = (
                self._subdomain_models[jj].physics.mesh.interpolate(
                    subdomain_mesh_values[jj],
                    samples[:, sample_mask_per_subdomain[jj]]))
        return values

    def interface_mesh(self):
        return self._subdomain_bounds.flatten()[1:-2:2][None, :]

    def plot(self, subdomain_values, npts_1d, ax, **kwargs):
        ims = [model.physics.mesh.plot(
            subdomain_values[jj], npts_1d, ax, **kwargs)
               for jj, model in enumerate(self._subdomain_models)]
        return ims


class AbstractTwoDDomainDecomposition(AbstractDomainDecomposition):
    def interpolate(self, subdomain_mesh_values, samples, default_val=0.):
        sample_mask_per_subdomain = self._in_subdomains(samples)
        if subdomain_mesh_values[0].ndim == 2:
            ncols = subdomain_mesh_values[0].shape[1]
        else:
            ncols = 1
        values = np.full((samples.shape[1], ncols), default_val)
        for jj in range(self._nsubdomains):
            if sample_mask_per_subdomain[jj].shape[0] > 0:
                values[sample_mask_per_subdomain[jj]] = (
                    self._subdomain_models[jj].physics.mesh.interpolate(
                        subdomain_mesh_values[jj],
                        samples[:, sample_mask_per_subdomain[jj]]))
        return values

    def interface_mesh(self):
        pts = []
        for ii in range(self._ninterfaces):
            pts.append(self._interfaces[ii]._mesh_pts())
        return np.hstack(pts)

    def _in_subdomains(self, samples):
        masks = []
        for ii in range(self._nsubdomains):
            mesh = self._subdomain_models[ii].physics.mesh
            canonical_samples = mesh._map_samples_to_canonical_domain(
                samples)
            lb0, ub0, lb1, ub1 = mesh._canonical_domain_bounds

            eps, eps_l = 1e-12, 1e-12
            masks.append(np.where(
                    (canonical_samples[0, :] >= lb0-eps_l) &
                    (canonical_samples[0, :] <= ub0+eps) &
                    (canonical_samples[1, :] >= lb1-eps_l) &
                    (canonical_samples[1, :] <= ub1+eps))[0])
        return masks

    @staticmethod
    def _plot_subdomain_mesh(mesh, npts, ax, **kwargs):
        can_pts_1d = mesh._canonical_mesh_pts_1d
        xx = np.linspace(
            can_pts_1d[0][0], can_pts_1d[0][-1], npts)[None, :]
        yy = np.linspace(
            can_pts_1d[1][0], can_pts_1d[1][-1], npts)[None, :]
        for ii in range(can_pts_1d[0].shape[0]):
            can_line = np.vstack(
                (np.full(yy.shape, can_pts_1d[0][ii]), yy))
            line = mesh._map_samples_from_canonical_domain(can_line)
            ax.plot(line[0], line[1], **kwargs)
        for jj in range(can_pts_1d[1].shape[0]):
            can_line = np.vstack(
                (xx, np.full(xx.shape, can_pts_1d[1][jj])))
            line = mesh._map_samples_from_canonical_domain(can_line)
            ax.plot(line[0], line[1], **kwargs)

    def plot_subdomain_boundary(self, subdomain_id, bndry_id, ax, **kwargs):
        npts = 100
        mesh = self._subdomain_models[subdomain_id].physics.mesh
        can_pts_1d = mesh._canonical_mesh_pts_1d
        if bndry_id < 2:
            yy = np.linspace(
                can_pts_1d[1][0], can_pts_1d[1][-1], npts)[None, :]
            if bndry_id == 0:
                idx = 0
            else:
                idx = -1
            can_line = np.vstack(
                (np.full(yy.shape, can_pts_1d[0][idx]), yy))
        else:
            xx = np.linspace(
                can_pts_1d[0][0], can_pts_1d[0][-1], npts)[None, :]
            if bndry_id == 2:
                idx = 0
            else:
                idx = -1
            can_line = np.vstack(
                (xx, np.full(xx.shape, can_pts_1d[1][idx])))

        line = mesh._map_samples_from_canonical_domain(can_line)
        ax.plot(line[0], line[1], **kwargs)

    def plot_mesh_grid(self, ax, **kwargs):
        npts = 101
        for model in self._subdomain_models:
            mesh = model.physics.mesh
            self._plot_subdomain_mesh(mesh, npts, ax, **kwargs)

    def _plot_2d_from_subdomain_interpolate(
            self, subdomain_values, npts_1d, ax, **kwargs):
        levels = kwargs.get("levels", 21)
        subdomain_values = [
            s[:, None] if s.ndim == 1 else s for s in subdomain_values]
        subdomain_plot_data = [model.physics.mesh._plot_data(
            subdomain_values[jj], npts_1d)
               for jj, model in enumerate(self._subdomain_models)]
        if isinstance(levels, int):
            z_min = np.min([d[0].min() for d in subdomain_plot_data])
            z_max = np.max([d[0].max() for d in subdomain_plot_data])
            levels = np.linspace(z_min, z_max, levels)
        kwargs["levels"] = levels
        ims = [model.physics.mesh._plot_from_data(
            subdomain_plot_data[jj], ax, **kwargs)
               for jj, model in enumerate(self._subdomain_models)]
        # ims = [model.physics.mesh.plot(
        #     subdomain_values[jj], npts_1d, ax, **kwargs)
        #        for jj, model in enumerate(self._subdomain_models)]
        return ims

    def _2d_triangluation(self, npts_1d):
        data = []
        for model in self._subdomain_models:
            data.append(model.physics.mesh._create_plot_mesh_2d(npts_1d))
        pts = np.hstack([d[2] for d in data])
        triang = tri.Triangulation(pts[0], pts[1])
        x = pts[0, triang.triangles].mean(axis=1)
        y = pts[1, triang.triangles].mean(axis=1)
        masks_in = self._in_subdomains(np.vstack((x[None, :], y[None, :])))
        masks_out = []
        for m in masks_in:
            # mask required by triang is points not in domain
            # mask returned by _in_subdomains are points in domain
            # masks are only where true need a vector that has true and false
            tmp = np.full((triang.triangles.shape[0], 1), 1)
            tmp[m, 0] = 0
            masks_out.append(tmp)
        masks_out = np.hstack(masks_out)
        mask_out = np.all(masks_out, axis=1)
        triang.set_mask(mask_out)
        return triang, pts

    def _plot_2d_from_global_interpolate(
            self, subdomain_values, npts_1d, ax, **kwargs):
        triang, pts = self._2d_triangluation(npts_1d)
        Z = self.interpolate(subdomain_values, pts)
        levels = kwargs.get("levels", 21)
        if isinstance(levels, int):
            # the same contour levels must be used for all subdomains
            z_min = np.min([sv.min() for sv in subdomain_values])
            z_max = np.max([sv.max() for sv in subdomain_values])
            levels = np.linspace(z_min, z_max, levels)
        return ax.tricontourf(triang, Z[:, 0], levels=levels)

    def plot(self, subdomain_values, npts_1d, ax, **kwargs):
        # return self._plot_2d_from_global_interpolate(
        #    subdomain_values, npts_1d, ax, **kwargs)
        return self._plot_2d_from_subdomain_interpolate(
            subdomain_values, npts_1d, ax, **kwargs)


class RectangularDomainDecomposition(AbstractTwoDDomainDecomposition):
    def __init__(self, bounds, nsubdomains_1d, ninterface_dof, intervals=None):
        super(). __init__()
        self._bounds = bounds
        self._nsubdomains_1d = nsubdomains_1d
        self._nsubdomains = np.prod(self._nsubdomains_1d)
        self._ninterface_dof = ninterface_dof
        self._ninterfaces = (
            nsubdomains_1d[0]*(nsubdomains_1d[1]-1) +
            (nsubdomains_1d[0]-1)*nsubdomains_1d[1])

        if intervals is None:
            intervals = [
                np.linspace(self._bounds[0], self._bounds[1],
                            self._nsubdomains_1d[0]+1),
                np.linspace(self._bounds[2], self._bounds[3],
                            self._nsubdomains_1d[1]+1)]
        # print(len(intervals[0]),  self._nsubdomains_1d[0]+1)
        # print(len(intervals[1]),  self._nsubdomains_1d[1]+1)
        if (len(intervals[0]) != self._nsubdomains_1d[0]+1 or
                len(intervals[1]) != self._nsubdomains_1d[1]+1):
            raise ValueError("intervals does not match self._nsubdomains")
        if (not np.allclose(intervals[0][[0, -1]], self._bounds[:2]) or
                not np.allclose(intervals[1][[0, -1]], self._bounds[2:])):
            raise ValueError("intervals does not match bounds")
        self._intervals = intervals
        self._subdomain_bounds = self._set_subdomain_bounds()

    def _set_subdomain_bounds(self):
        subdomain_bounds = np.empty((self._nsubdomains, 4))
        for jj in range(self._nsubdomains_1d[1]):
            for ii in range(self._nsubdomains_1d[0]):
                idx = self._subdomain_index(ii, jj)
                subdomain_bounds[idx] = np.array([
                    self._intervals[0][ii], self._intervals[0][ii+1],
                    self._intervals[1][jj], self._intervals[1][jj+1]])
        return subdomain_bounds

    def _define_subdomain_transforms(self):
        nphys_vars = 2
        canonical_domain_bounds = (
            CanonicalCollocationMesh._get_canonical_domain_bounds(
                nphys_vars,
                CanonicalCollocationMesh._get_basis_types(nphys_vars, None)))
        transforms = []
        for bounds in self._subdomain_bounds:
            transforms.append(ScaleAndTranslationTransform(
                canonical_domain_bounds, bounds))
        return transforms

    def _subdomain_index(self, ii, jj):
        assert ii < self._nsubdomains_1d[0] and jj < self._nsubdomains_1d[1]
        return jj*self._nsubdomains_1d[0]+ii

    def _set_interface_data(self):
        """
        Domain


        y3   _B_ _B_ _B_
            |   |   |   |
            B   10  11  B
        y2  |_6_|_8_|_9_|
            |   |   |   |
            B   5   7   B
        y1  |_1_|_3_|_4_|
            |   |   |   |
            B   0   2   B
        y0  |_B_|_B_|_B_|
            x0 x1 x2 x3

        subdomains ordered bottom left, bottom right, top left, top right

        Boundaries of subdomains not on interfaces are labeled B. This is where
        PDE boundary conditions are enforced

        interfaces for each subdomain are labeld on the edges of each subdomain
        For this example the interfaces of each subdomain are
        [[0, 1], [1, 2, 3], [3, 4], [1, 5, 6], [3, 5, 7, 8], [4, 7, 9],
        [6, 10],[8, 10, 11], [9, 11]]

        intervals : [iterable, iterable]
            Intervals, e.g. [[x0, x1, x2, x3], [y0, y1, y2, y3]].
            Each interval can have different lengths
        """
        # the tuple (subdomain1, boundary_index1, subdomain2, boundary_index2)
        # of each interface
        self._interface_to_bndry_map = []
        self._interfaces = []
        self._subdomain_to_interface_map = [
            [] for jj in range(self._nsubdomains)]
        for jj in range(self._nsubdomains_1d[1]):
            for ii in range(self._nsubdomains_1d[0]):
                idx = self._subdomain_index(ii, jj)
                if ii < self._nsubdomains_1d[0]-1:
                    # right interface of subdomain [ii, jj]
                    right_neigh = self._subdomain_index(ii+1, jj)
                    self._interface_to_bndry_map.append(
                        [idx, 1, right_neigh, 0])
                    self._interfaces.append(
                        SubdomainInterface1D(
                            self._ninterface_dof, 1,
                            self._subdomain_transforms[idx], 1))
                    interface_cnt = len(self._interfaces)-1
                    self._subdomain_to_interface_map[idx].append(interface_cnt)
                    self._subdomain_to_interface_map[right_neigh].append(
                        interface_cnt)

                if jj < self._nsubdomains_1d[1]-1:
                    # top interface of subdomain [ii, jj]
                    top_neigh = self._subdomain_index(ii, jj+1)
                    self._interface_to_bndry_map.append(
                        [idx, 3, top_neigh, 2])
                    self._interfaces.append(
                        SubdomainInterface1D(
                            self._ninterface_dof, 0,
                            self._subdomain_transforms[idx], 3))
                    interface_cnt = len(self._interfaces)-1
                    self._subdomain_to_interface_map[idx].append(interface_cnt)
                    self._subdomain_to_interface_map[top_neigh].append(
                        interface_cnt)

        self._interface_to_bndry_map = np.array(self._interface_to_bndry_map)
        self._interface_dof_starts = np.hstack((0, np.cumsum(
            [iface._ndof for iface in self._interfaces])[:-1]))
        assert self._ninterfaces == len(self._interface_to_bndry_map)


class ElbowDomainDecomposition(AbstractTwoDDomainDecomposition):
    def __init__(self, ninterface_dof, intervals):
        super(). __init__()
        self._nsubdomains = 3
        self._ninterface_dof = ninterface_dof
        self._ninterfaces = 2
        self._subdomain_bounds = self._set_subdomain_bounds(intervals)

    @staticmethod
    def _set_subdomain_bounds(intervals):
        if len(intervals) != 6:
            raise ValueError("Must provide 6 intervals")
        subdomain_bounds = np.empty((3, 4))
        subdomain_bounds[0] = [
            intervals[0], intervals[1],
            intervals[4], intervals[5]]
        subdomain_bounds[1] = [
            intervals[0], intervals[1],
            intervals[3], intervals[4]]
        subdomain_bounds[2] = [
            intervals[1], intervals[2],
            intervals[3], intervals[4]]
        return subdomain_bounds

    def _define_subdomain_transforms(self):
        nphys_vars = 2
        canonical_domain_bounds = (
            CanonicalCollocationMesh._get_canonical_domain_bounds(
                nphys_vars,
                CanonicalCollocationMesh._get_basis_types(nphys_vars, None)))
        transforms = []
        for bounds in self._subdomain_bounds:
            transforms.append(ScaleAndTranslationTransform(
                canonical_domain_bounds, bounds))
        return transforms

    def _set_interface_data(self):
        """
        Elbow:
        y2   _B_
            |   |
            B   B
        y1  |_1_|_B_
            |   |   |
            B   0   B
        y0  |_B_|_B_|
            x0 x1 x2

        subdomains are indexed 0, 1, 2 for upper left, lower left, and lower
        right, respectively.

        intervals : iterable
            Define size of elbows [x0, x1, x2, y0, y1, y2]
        """
        interface_to_bndry_map = []
        interfaces = []
        subdomain_to_interface_map = [
            [] for jj in range(self._nsubdomains)]
        # subdomain 1
        idx = 1
        right_neigh, top_neigh = 2, 0
        interface_to_bndry_map.append(
            [idx, 1, right_neigh, 0])
        interfaces.append(
            SubdomainInterface1D(
                self._ninterface_dof, 1, self._subdomain_transforms[idx], 1))
        interface_cnt = len(interfaces)-1
        subdomain_to_interface_map[idx].append(interface_cnt)
        subdomain_to_interface_map[right_neigh].append(interface_cnt)

        interface_to_bndry_map.append(
            [idx, 3, top_neigh, 2])
        interfaces.append(
            SubdomainInterface1D(
                self._ninterface_dof, 0, self._subdomain_transforms[idx], 3))
        interface_cnt = len(interfaces)-1
        subdomain_to_interface_map[idx].append(interface_cnt)
        subdomain_to_interface_map[top_neigh].append(
            interface_cnt)

        interface_to_bndry_map = np.array(interface_to_bndry_map)
        ninterfaces = len(interface_to_bndry_map)
        interface_dof_starts = np.hstack((0, np.cumsum(
            [iface._ndof for iface in interfaces])[:-1]))
        assert ninterfaces == self._ninterfaces
        (self._interface_to_bndry_map, self._interfaces, self._ninterfaces,
         self._interface_dof_starts, self._subdomain_to_interface_map) = (
             interface_to_bndry_map, interfaces, ninterfaces,
             interface_dof_starts, subdomain_to_interface_map)


class SteadyStateDomainDecompositionSolver():
    def __init__(self, domain_decomp):
        self._decomp = domain_decomp
        self._decomp._solve_subdomain = self._solve_subdomain

    def solve(self, init_guess=None, macro_newton_kwargs={},
              subdomain_newton_kwargs={}):
        if init_guess is None:
            init_guess = np.ones((self._decomp.get_ninterfaces_dof(), 1))
            # Next line will set dirichlet values internally
        self._decomp._compute_interface_values(
            init_guess, subdomain_newton_kwargs, **macro_newton_kwargs)
        subdomain_sols = [
            model.solve(None, **subdomain_newton_kwargs)
            for model in self._decomp._subdomain_models]
        return subdomain_sols

    def _solve_subdomain(self, jj, **subdomain_newton_kwargs):
        sol = self._decomp._subdomain_models[jj].solve(
            None, **subdomain_newton_kwargs)
        drdu = self._decomp._subdomain_models[jj].physics._residual(sol)[1]
        return sol, drdu


class TransientDomainDecompositionSolver():
    def __init__(self, domain_decomp):
        self._decomp = domain_decomp
        self._dirichlet_vals = None
        self._prev_sols = None

        # # stores inverse of time stepping jacobian for linear physics
        self._drdu_inv = [None for ii in range(self._decomp._nsubdomains)]
        # even for linear physics drdu depends on timstep so this
        # is used to make sure timestep is the same before using stored drdu_inv
        self._tstep = None
        self._decomp._invert_drdu = self._invert_drdu

    def _invert_drdu(self, drdu, jj):
        if (self._drdu_inv[jj] is None or self._tstep != self._data[2]):
            # only works for linear physics
            # It does work even for time independent boundary conditions because
            # the time aspect does not effect jacobian. We assume boundary type is
            # the same for all time
            # todo add checks
            self._drdu_inv[jj] = torch.linalg.inv(drdu)
            self._tstep = self._data[2]
        return self._drdu_inv[jj]

    def _solve_subdomain_expanded(
            self, jj, **subdomain_newton_kwargs):
        prev_sols, time, deltat, store_data, clear_data = self._data
        sol = self._decomp._subdomain_models[jj].solve(
            prev_sols[jj], time, time+deltat, 0, store_data, clear_data,
            newton_kwargs=subdomain_newton_kwargs)[0][:, -1]
        physics = self._decomp._subdomain_models[jj].physics
        physics._store_data = store_data
        raw_drdu = physics._transient_residual(
            sol, time+deltat)[1]
        physics._store_data = False
        Id = torch.eye(sol.shape[0], dtype=torch.double)
        rhs = torch.empty(sol.shape[0], dtype=torch.double)  # dummy
        drdu = physics.mesh._apply_boundary_conditions(
            physics._bndry_conds, rhs, Id-deltat*raw_drdu, sol,
            physics.flux_jac)[1]
        return sol, drdu

    def _update_subdomain_sols(
            self, prev_sols, time, deltat, subdomain_newton_kwargs,
            macro_newton_kwargs, store_data, clear_data):
        if self._dirichlet_vals is None:
            self._dirichlet_vals = np.ones(
                (self._decomp.get_ninterfaces_dof(), 1))
        self._data = prev_sols, time, deltat, store_data, clear_data
        self._decomp._solve_subdomain = self._solve_subdomain_expanded
        self._dirichlet_vals = self._decomp._compute_interface_values(
            self._dirichlet_vals, subdomain_newton_kwargs,
            **macro_newton_kwargs)
        # Sols  has already been computed avoid recomputing and store
        # and reuse here
        # sols = []
        # for jj in range(self._decomp._nsubdomains):
        #     sol_jj = self._decomp._subdomain_models[jj].solve(
        #         prev_sols[jj], time, time+deltat, 0, store_data, clear_data,
        #         subdomain_newton_kwargs)[0][:, -1:]
        #     sols.append(sol_jj)
        #     print(self._decomp._sols[jj])
        #     print(sol_jj)
        #     assert np.allclose(self._decomp._sols[jj], sol_jj)
        # return sols
        return self._decomp._sols

    def solve(self, init_sols, init_time, final_time, deltat, verbosity=0,
              subdomain_newton_kwargs={}, macro_newton_kwargs={}):
        """
        subdomain_sols : list
           [ntimesteps+1, nsubdomains, ndof_per_subdomain]
        """
        if len(init_sols) != self._decomp._nsubdomains:
            raise ValueError(
                "must provide an initial condition for each subdomain")
        sols, times = [], []
        time = init_time
        sols.append([sol[:, None] if sol.ndim == 1 else sol
                     for sol in init_sols])
        times.append(time)
        while time < final_time-1e-12:
            if verbosity >= 1:
                print("Time", time)
            deltat = min(deltat, final_time-time)
            store_data = True
            clear_data = abs(time-final_time) <= 1e-12 or time == init_time
            subdomain_sols = self._update_subdomain_sols(
                sols[-1], time, deltat, subdomain_newton_kwargs,
                macro_newton_kwargs,
                store_data, clear_data)
            sols.append(subdomain_sols)
            time += deltat
            times.append(time)
        self._drdu_inv = [None for ii in range(self._decomp._nsubdomains)]
        return sols, times


from pyapprox.pde.autopde.mesh_transforms import (
    CompositionTransform, ScaleAndTranslationTransform, EllipticalTransform,
    SympyTransform, PolarTransform)
class TurbineDomainDecomposition(AbstractTwoDDomainDecomposition):
    def __init__(self, ninterface_dof, height_max=0.25, length=1):
        super(). __init__()
        self._nsubdomains = 13
        self._ninterfaces = 14
        self._ninterface_dof = ninterface_dof
        # max height from y=0 axis
        self._height_max = height_max
        self._length = length

    # @staticmethod
    # def _get_elliptical_nose_subdomain(width_max, height_max, thickness_ratio):
    #     foci = np.sqrt(width_max**2-height_max**2)
    #     rmax = np.arccosh(width_max/foci)
    #     height_max = np.sqrt(width_max**2-foci**2)
    #     rmin = rmax*thickness_ratio
    #     return CompositionTransform(
    #         [ScaleAndTranslationTransform(
    #             [-1, 1, -1, 1], [rmin, rmax, np.pi/2, 3*np.pi/2]),
    #          EllipticalTransform(foci)])

    @staticmethod
    def _get_polar_subdomain(rmin, rmax, scale, theta_min, theta_max):
        return CompositionTransform(
            [ScaleAndTranslationTransform(
                [-1, 1, -1, 1],
                [rmin, rmax, theta_min, theta_max]),
             PolarTransform(),
             ScaleAndTranslationTransform(
                [-rmax, rmax, rmin, rmax],
                 [-scale*rmax, scale*rmax, rmin, rmax])])

    @staticmethod
    def _get_subdomain(surf_string, bed_string, x0, x1):
        scale_transform = ScaleAndTranslationTransform(
            [-1, 1, -1, 1], [x0, x1, 0, 1])
        y_from_orth_string = f"({surf_string}-({bed_string}))*_t_+{bed_string}"
        y_to_orth_string = (
            f"(_y_-({bed_string}))/({surf_string}-({bed_string}))".replace(
                "_r_", "_x_"))
        return CompositionTransform(
            [scale_transform,
             SympyTransform(["_r_", y_from_orth_string],
                            ["_x_", y_to_orth_string])])

    def _define_subdomain_transforms(self):
        """
        The order of the upper and lower bounds of the polar transform matter.
        otherwise interfaces will not be defined correctly
        The order used here is correct.
        I am not sure how general this is and so am unsure how to add
        a check in the code (I can test for it though - the probelm manifests
        itself by making residual non zero when using exact dirichlet vals)
        """
        width_max = 1.0
        height_max = self._height_max
        rmax = height_max
        thickness_ratio = 0.7
        rmin = rmax*thickness_ratio
        # control length of front section left (of x=0) relative to
        # length of right section
        scale = width_max/rmax
        # increasing alpha moves left side of first column towards front nose
        # increasing beta moves moves right side of first column towards rear
        alpha, beta = 1.6, 0.5
        alpha, beta = 1.6, 0.3
        # alpha, beta = 1, 0.4
        theta0 = alpha*np.pi/2
        transform_0 = self._get_polar_subdomain(
            rmin, rmax, scale, theta0, 2*np.pi-theta0)
        theta1 = (alpha-beta)*np.pi/2
        transform_1 = self._get_polar_subdomain(
            rmin, rmax, scale, theta1, theta0)
        surf_string = f"sqrt({rmin**2}-_r_**2)"
        bed_string = f"-sqrt({rmin**2}-_r_**2)"
        x0 = rmin*np.cos(theta0)
        x1 = rmin*np.cos(theta1)
        transform_2 = CompositionTransform([
            self._get_subdomain(
                surf_string, bed_string, x0, x1),
            ScaleAndTranslationTransform(
                [x0, x1, 0, 1],
                [scale*x0, scale*x1, 0, 1])
            ])
        transform_3 = self._get_polar_subdomain(
            rmin, rmax, scale, 2*np.pi-theta0, 2*np.pi-theta1)
        transform_4 = self._get_polar_subdomain(
            rmin, rmax, scale, np.pi/2, theta1)
        transform_5 = self._get_polar_subdomain(
             rmin, rmax, scale, 2*np.pi-theta1, 2*np.pi-np.pi/2)
        # increasing x2 moves left side of second column towards rear
        x2 = width_max*0.05
        x_end = 1.0*width_max
        delta = height_max/2
        surf_string = f"-{delta}/{x_end**2}*_r_**2+{rmax}"
        bed_string = f"-{delta}/{x_end**2}*_r_**2+{rmin}"
        transform_6 = self._get_subdomain(
                surf_string, bed_string, 0, x2)
        surf_string_low = f"{delta}/{x_end**2}*_r_**2-{rmin}"
        bed_string_low = f"{delta}/{x_end**2}*_r_**2-{rmax}"
        transform_7 = self._get_subdomain(
                surf_string_low, bed_string_low, 0, x2)
        # increasing x3 moves right side of second column towards rear
        x3 = width_max*0.3
        x3 = width_max*0.6
        transform_8 = self._get_subdomain(
                surf_string, bed_string, x2, x3)
        transform_9 = self._get_subdomain(
                bed_string, surf_string_low, x2, x3)
        transform_10 = self._get_subdomain(
                surf_string_low, bed_string_low, x2, x3)
        transform_11 = self._get_subdomain(
                surf_string, bed_string, x3, x_end)
        transform_12 = self._get_subdomain(
                surf_string_low, bed_string_low, x3, x_end)
        transforms = [
            transform_0, transform_1, transform_2, transform_3,
            transform_4, transform_5, transform_6, transform_7,
            transform_8, transform_9, transform_10, transform_11, transform_12]

        final_transform = ScaleAndTranslationTransform(
            [-1, 1, -height_max, height_max],
            [0, self._length, -height_max, height_max])
        transforms = [CompositionTransform([transform, final_transform])
                      for transform in transforms]

        # The subdomain 2 causes issues when solving PDEs
        # I think because of different coordinate transformations.
        # So while cannot resolve just exclude
        selected_transform_idx = np.delete(np.arange(13), 2)
        self._ninterfaces = 12
        transforms = [transforms[idx] for idx in selected_transform_idx]
        self._nsubdomains = len(transforms)
        # transforms = [transforms[4], transforms[6]]
        # transforms = [transforms[5], transforms[7]]
        # transforms = [transforms[8], transforms[11]]
        # transforms = [transforms[2], transforms[3]]
        # self._nsubdomains = 2
        # self._ninterfaces = 1
        # transforms = [transforms[1], transforms[2], transforms[3],
        #               transforms[4], transforms[6]]
        # self._nsubdomains = 5
        # self._ninterfaces = 4
        return transforms

    def rvs(self, nsamples, batchsize=100, return_acceptance_rate=False):
        # randomly generate samples inside domain
        from pyapprox.variables.joint import (IndependentMarginalsVariable,
                                              stats)
        variable = IndependentMarginalsVariable(
            [stats.uniform(0., self._length),
             stats.uniform(-self._height_max, 2*self._height_max)])
        samples = np.empty((2, nsamples))
        idx1 = 0
        niters = 0
        accepted = 0
        while True:
            candidates = variable.rvs(batchsize)
            mask = np.hstack(self._in_subdomains(candidates))
            idx2 = min(idx1+mask.shape[0], nsamples)
            samples[:, idx1:idx2] = candidates[:, mask[:idx2-idx1]]
            idx1 = idx2
            accepted += mask.shape[0]
            niters += 1
            if idx2 == nsamples:
                break
        if not return_acceptance_rate:
            return samples
        acceptance_rate = accepted/(batchsize*niters)
        return samples, acceptance_rate


class GappyRectangularDomainDecomposition(RectangularDomainDecomposition):
    """
    Split rectanugar domain into rectangular subdomains with some subdomains
    missing. For fluid flow this amounts to placing obstructions to the flow
    """
    def __init__(self, bounds, nsubdomains_1d, ninterface_dof,
                 missing_subdomain_indices, ninterfaces, intervals=None):
        missing_subdomain_indices = np.asarray(missing_subdomain_indices)
        assert np.all(
            missing_subdomain_indices.max(axis=0) < np.asarray(nsubdomains_1d))
        self._missing_subdomain_indices = [
            ind[1]*nsubdomains_1d[0]+ind[0]
            for ind in missing_subdomain_indices]
        super().__init__(bounds, nsubdomains_1d, ninterface_dof, intervals)
        self._nsubdomains = np.prod(nsubdomains_1d) - len(
            self._missing_subdomain_indices)
        self._ninterfaces = ninterfaces

    def _define_subdomain_transforms(self):
        subdomain_transforms = super()._define_subdomain_transforms()
        subdomain_transforms = [
            trans for jj, trans in enumerate(subdomain_transforms)
            if jj not in self._missing_subdomain_indices]
        return subdomain_transforms

    def _set_interface_data(self):
        return super(RectangularDomainDecomposition,
                     self)._set_interface_data()

    def plot_subdomains(self, ax):
        # TODO make this general by mapping lines of points in canonical domain
        # via transform
        for ii in range(self._nsubdomains):
            ranges = self._subdomain_transforms[ii]._ranges
            square_pts = np.array(
                [[ranges[0], ranges[2]], [ranges[0], ranges[3]],
                 [ranges[1], ranges[3]], [ranges[1], ranges[2]],
                 [ranges[0], ranges[2]]]).T
            ax.plot(*square_pts, c='k')


def get_active_subdomain_indices(nsubdomains_1d, missing_indices):
    indices = cartesian_product([np.arange(nn) for nn in nsubdomains_1d])
    # indices_flat = np.arange(indices.shape[1])
    missing_indices_flat = [
        ind[1]*nsubdomains_1d[0]+ind[0] for ind in missing_indices]
    mask = np.ones(indices.shape[1], dtype=bool)
    mask[missing_indices_flat] = False
    indices = indices[:, mask]
    return indices



# Notes
# When using interface points that are on the boundary of the interface,
# e.g. the end of the interval, then need to make sure not to pass in duplicate
# values in dirichlet_vals when computing residuals. Otherwise a singluar
# jacobian will result in newtons method
#
# Have to use the same transform on an interface for each domain on the interface
# even thought the transforms of both subdomains may differ. This is because
# interface._values is stored using just one transform. If the transforms are different
# the orthogonal coordinates will differ and thus must use the same transform for
# both subdomains when mapping boundary coordinates to and from interface.
# The TurbineDomainDecomposition is a good example of the need for this.
# This does not come up for RectangularDomainDecomposition
