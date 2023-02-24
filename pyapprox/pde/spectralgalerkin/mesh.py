import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from pyapprox.util.utilities import cartesian_product, hash_array
from pyapprox.pde.autopde.mesh_transforms import (
    ScaleAndTranslationTransform, _map_hypercube_samples)


def canonical_linear_basis_1d(xx, ii):
    """
    one-dimensional canonical linear basis on [-1,1]
    """
    if (ii==0):
        vals = (1.-xx)/2.
        II = np.where((xx < -1.) | (xx >= 1.))[0]
    elif (ii==1):
        vals = (1.+xx)/2.
        II = np.where((xx < -1.) | (xx >= 1.))[0]
    else:
        raise ValueError('incorrect basis index given: %d' %ii)

    # II = np.where((xx < -1.) | (xx > 1.))[0]
    vals[II] = 0.
    return vals


def canonical_linear_basis_2d(xx, ii, jj):
    """
    two-dimensional linear basis on [-1,1,-1,1]
    """
    # assert np.max(xx) <= 1, (np.max(xx, axis=1), np.min(xx, axis=1))
    assert xx.ndim == 2
    assert xx.shape[0] == 2
    return canonical_linear_basis_1d(xx[0, :], ii)*canonical_linear_basis_1d(
        xx[1, :], jj)


def canonical_quadratic_basis_1d(xx, ii):
    """
    one-dimensional canonical quadratic basis on [-1,1]
    """
    if (ii==0):
        II = np.where((xx < -1.) | (xx >= 1.))[0]
        vals = -xx*(1.-xx)/2.
    elif (ii==1):
        vals = (1.-xx**2)
        II = np.where((xx < -1.) | (xx > 1.))[0]
    elif (ii==2):
        vals = xx*(1.+xx)/2.
        II = np.where((xx < -1.) | (xx >= 1.))[0]
    else:
        raise ValueError('incorrect basis index given: %d' %ii)

    vals[II] = 0.
    return vals


def canonical_quadratic_basis_2d(xx, ii, jj):
    """
    two-dimensional quadratic basis on [-1,1,-1,1]
    """
    # assert np.max(xx) <= 1, (np.max(xx, axis=1), np.min(xx, axis=1))
    assert xx.ndim == 2
    assert xx.shape[0] == 2
    return canonical_quadratic_basis_1d(
        xx[0, :], ii)*canonical_quadratic_basis_1d(xx[1, :], jj)


def near_1d_bndry(val, xx, tol=1e-12):
    return (np.absolute(xx[0]-val) < tol)


def near_horizontal_2d_bndry(val, lb, ub, xx, tol=1e-12):
    return ((np.absolute(xx[1]-val) < tol) &
            (xx[0] >= lb-tol) & (xx[0] <= ub+tol))


def near_vertical_2d_bndry(val, lb, ub, xx, tol=1e-12):
    return ((np.absolute(xx[0]-val) < tol) &
            (xx[1] >= lb-tol) & (xx[1] <= ub+tol))


class Basis(ABC):
    def basis_matrix(self, xx):
        assert xx.ndim == 2
        basis_matrix = np.empty((xx.shape[1], self._indices.shape[1]))
        for kk, index in enumerate(self._indices.T):
            basis_matrix[:, kk] = self(*index, xx)
        return basis_matrix

    @abstractmethod
    def __call__(self):
        raise NotImplementedError()


class CanonicalIntervalBasis(Basis):
    def __init__(self, order):
        self._order = order

        self._canonical_dof = cartesian_product(
            [np.linspace(-1, 1, (order+1))]*1)
        self._canonical_bounds = np.array([-1, 1])
        self._indices = cartesian_product([np.arange(0, (order+1))])

    def __call__(self, ii, xx):
        if self._order == 1:
            return canonical_linear_basis_1d(xx[0], ii)

        if self._order == 2:
            return canonical_quadratic_basis_1d(xx[0], ii)

        raise ValueError("Order not supported")


class CanonicalRectangularBasis(Basis):
    def __init__(self, order):
        self._order = order

        self._canonical_dof = cartesian_product(
            [np.linspace(-1, 1, (order+1))]*2)
        self._canonical_bounds = np.array([-1, 1, -1, 1])
        self._indices = cartesian_product(
            [np.arange(0, (order+1))]*2)

    def __call__(self, ii, jj, xx):
        if self._order == 1:
            return canonical_linear_basis_2d(xx, ii, jj)

        if self._order == 2:
            print(ii, jj)
            return canonical_quadratic_basis_2d(xx, ii, jj)

        raise ValueError("Order not supported")

    


class Mesh(ABC):
    def __init__(self, domain_bounds, nelems_1d, canonical_basis):
        self._domain_bounds = domain_bounds
        self._nphys_vars = len(domain_bounds)//2
        self._canonical_basis = canonical_basis

        self._canonical_mesh_pts = self._set_canonical_mesh(nelems_1d)
        self._elem_to_vertex_map = self._set_elem_to_vertex_map(nelems_1d)
           
        self._nelems = self._elem_to_vertex_map.shape[1]
        
        self._transform = ScaleAndTranslationTransform(
            np.hstack([-1, 1]*self._nphys_vars), self._domain_bounds)
        self.mesh_pts = self._transform.map_from_orthogonal(
            self._canonical_mesh_pts)

        # self._dofs : degrees of freedom are the points that
        # define finite element basis
        # self._elem_to_dof_map : for each element specifies the indices of
        # its dofs in the global array
        self._dofs, self._elem_to_dof_map = self._set_degrees_of_freedom()    
        self._ndofs = self._dofs.shape[1]

    def _set_canonical_mesh(self, nelems_1d):
        assert len(nelems_1d) == len(self._domain_bounds)//2
        mesh_pts = cartesian_product(
            [np.linspace(-1, 1, ne+1) for ne in nelems_1d])
        return mesh_pts

    def _set_degrees_of_freedom(self):
        cnt = 0
        dofs = []
        elem_to_dof_map = [[] for kk in range(self._nelems)]
        unique_dof_dict = dict()
        for kk in range(self._nelems):
            elem_dof = self._get_elem_dof(kk)
            for ed in elem_dof.T:
                key = hash_array(ed)
                if key not in unique_dof_dict:
                    unique_dof_dict[key] = cnt
                    dofs.append(ed)
                    elem_to_dof_map[kk].append(cnt)
                    cnt += 1
                else:
                    elem_to_dof_map[kk].append(unique_dof_dict[key])

        dofs = np.array(dofs).T
        return dofs, elem_to_dof_map

    def mark_boundaries(self, rules):
        dof_bndrys_indices = np.full((self._ndofs), np.inf, dtype=int)
        for ii, rule in enumerate(rules):
            print(rule(self._dofs), 'r')
            dof_bndrys_indices[rule(self._dofs)] = ii
        self._dof_bndry_indices = dof_bndrys_indices
        if not np.all(np.isfinite(dof_bndrys_indices)):
            raise ValueError("rules do not mark all boundaries")

    def plot_mesh(self):
        for ii in range(self._elem_to_vertex_map.shape[1]):
            self._plot_elem(ii)

    def interpolate(self, quantity, xx):
        # quantity (ndofs, nqoi) defined on dofs
        vals = np.zeros((xx.shape[1], quantity.shape[1]))
        for kk in range(self._nelems):
            canonical_xx = self._get_canonical_elem_xx(kk, xx)
            basis_matrix = self._canonical_basis.basis_matrix(canonical_xx)
            vals += quantity[self._elem_to_dof_map[kk]].T.dot(basis_matrix.T).T
        return vals

    def _get_elem_dof(self, kk):
        elem_vertices = self.mesh_pts[:, self._elem_to_vertex_map[:, kk]]
        elem_bounds = self._get_elem_bounds(elem_vertices)
        return _map_hypercube_samples(
            self._canonical_basis._canonical_dof,
            self._canonical_basis._canonical_bounds, elem_bounds)

    def _get_canonical_elem_xx(self, kk, xx):
        elem_vertices = self.mesh_pts[:, self._elem_to_vertex_map[:, kk]]
        elem_bounds = self._get_elem_bounds(elem_vertices)
        # map xx to canonical domain of basis. This is different
        # to canonical domain of mesh
        canonical_xx = _map_hypercube_samples(
            xx, elem_bounds, self._canonical_basis._canonical_bounds)
        return canonical_xx

    @abstractmethod
    def _set_elem_to_vertex_map(self, nelems_1d):
        raise NotImplementedError()
    
    @abstractmethod
    def _get_elem_bounds(self, elem_vertices):
        raise NotImplementedError()

    @abstractmethod
    def _plot_elem(self, kk):
        raise NotImplementedError()


class IntervalMesh1D(Mesh):
    def _set_elem_to_vertex_map(self, nelems_1d):
        elem_to_vertex_map = np.empty((2, nelems_1d[0]), dtype=int)
        for kk in range(nelems_1d[0]):
            elem_to_vertex_map[0, kk] = kk
            elem_to_vertex_map[1, kk] = kk+1
        return elem_to_vertex_map

    def _get_elem_bounds(self, elem_vertices):
        elem_bounds = np.asarray(
            [elem_vertices[0, 0], elem_vertices[0, -1]])
        return elem_bounds

    def _plot_elem(self, kk):
        # plot interval element
        elem_vertices = self.mesh_pts[:, self._elem_to_vertex_map[:, kk]]
        plt.plot(elem_vertices[0], elem_vertices[0]*0, '-k')
    

class RectangularMesh2D(Mesh):
    def _set_elem_to_vertex_map(self, nelems_1d):
        nelems = np.prod(nelems_1d)
        # Assign the indices that map element vertices with the global
        # vertex coordinate mesh
        kk = 0
        elem_to_vertex_map = np.empty((4, nelems), dtype=int)
        for jj in range(nelems_1d[1]):
            for ii in range(nelems_1d[0]):
                elem_to_vertex_map[0, kk] = jj*(nelems_1d[0]+1)+ii
                elem_to_vertex_map[1, kk] = jj*(nelems_1d[0]+1)+ii+1
                elem_to_vertex_map[2, kk] = (jj+1)*(nelems_1d[0]+1)+ii+1
                elem_to_vertex_map[3, kk] = (jj+1)*(nelems_1d[0]+1)+ii
                kk += 1
        return elem_to_vertex_map

    def _get_elem_bounds(self, elem_vertices):
        elem_bounds = np.asarray(
            [elem_vertices[0, 0], elem_vertices[0, 1],
             elem_vertices[1, 0], elem_vertices[1, -1]])
        return elem_bounds

    def _plot_elem(self, kk):
        # plot square element
        elem_vertices = self.mesh_pts[:, self._elem_to_vertex_map[:, kk]]
        plt.plot(*elem_vertices, '-k')
        plt.plot(*elem_vertices[:, [0, 3]], '-k')
