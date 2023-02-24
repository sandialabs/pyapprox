import unittest
import numpy as np


import matplotlib.pyplot as plt
from functools import partial
from pyapprox.util.visualization import (
    get_meshgrid_function_data, create_3d_axis)
from pyapprox.util.utilities import cartesian_product


from pyapprox.pde.spectralgalerkin.mesh import (
    RectangularMesh2D, CanonicalRectangularBasis, near_horizontal_2d_bndry,
    near_vertical_2d_bndry, near_1d_bndry, IntervalMesh1D,
    CanonicalIntervalBasis)


class TestFEMesh(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_1d_mesh(self):
        basis_order, nelems_1d, domain_bounds = 2, [3], [0, 1]
        basis = CanonicalIntervalBasis(basis_order)
        mesh = IntervalMesh1D(domain_bounds, nelems_1d, basis)
        bndry_rules = [
            # left bndry
            partial(near_1d_bndry, domain_bounds[0]),
            # right bndry
            partial(near_1d_bndry, domain_bounds[1])]
        mesh.mark_boundaries(bndry_rules)
               
        # mesh.plot_mesh()
        # plt.plot(*mesh._canonical_dofs, 'rs')
        # for ii in range(4):
        #     plt.plot(mesh._dofs[:, mesh._dof_bndry_indices==ii], 0, 'X')

        if basis_order == 1:
            assert mesh._dofs.shape[1] == mesh.mesh_pts.shape[1]
        else:
            assert (mesh._dofs.shape[1] ==
                (mesh.mesh_pts.shape[1] + mesh._nelems))

        plt.plot(mesh._dofs[:, mesh._elem_to_dof_map[2]], 0, 'o')
        def fun(xx):
            return np.sum(xx**basis_order, axis=0)[:, None]
        mesh_vals = fun(mesh._dofs)
        interp_vals = mesh.interpolate(mesh_vals, mesh._dofs)

        canonical_xx = np.linspace(
            *mesh._canonical_basis._canonical_bounds, 101)[None, :]
        plt.plot(canonical_xx[0],
                 mesh._canonical_basis.basis_matrix(canonical_xx))


        plt.plot(mesh._dofs[0], mesh_vals, 'ko')
        plt.plot(mesh._dofs[0], interp_vals, 'rX')

        xx = np.linspace(*domain_bounds, 101)[None, :]
        # Exclude endpoint because basis is zero on right boudnary of
        # canonical element which messes up error. TODO make so basis
        # is non zero at this point only for element on right boundary.
        # Or make vertex associated with mesh point only associated with
        # one basis. Currently it is associated with 2 (in 2D) which
        # means when all bases are summed. The value at mesh point is added
        # twice
        xx = xx[:, :-1]
        plt.plot(xx[0], fun(xx), 'k')
        plt.plot(xx[0], mesh.interpolate(mesh_vals, xx), 'r--')
        print(mesh_vals[:, 0])
        print(interp_vals[:, 0])
        print(mesh_vals[:, 0]-interp_vals[:, 0])
         # Exclude endpoint see not above
        assert np.allclose(mesh_vals[:-1], interp_vals[:-1])
            
        # plt.show()



    def test_2d_mesh(self):
        basis_order, nelems_1d, domain_bounds = 1, [2, 2], [0, 1, 0, 1]
        basis = CanonicalRectangularBasis(basis_order)
        mesh = RectangularMesh2D(domain_bounds, nelems_1d, basis)
        bndry_rules = [
            # left bndry
            partial(near_vertical_2d_bndry, domain_bounds[0],
                    *domain_bounds[2:]),
            # right bndry
            partial(near_vertical_2d_bndry, domain_bounds[1],
                    *domain_bounds[2:]),
            # bottom bndry
            partial(near_horizontal_2d_bndry, domain_bounds[2],
                    *domain_bounds[:2]),
            # top bndry
            partial(near_horizontal_2d_bndry, domain_bounds[3],
                    *domain_bounds[:2]),]
        mesh.mark_boundaries(bndry_rules)
        
        mesh.plot_mesh()
        # plt.plot(*mesh._canonical_dofs, 'rs')
        for ii in range(4):
            plt.plot(*mesh._dofs[:, mesh._dof_bndry_indices==ii], 'X')
        
        for ii in range(2):
            # -2 because first and last dof on left and right boundary will
            # be ovewritten when top and bottom boundaries are marked.
            # If top and bottom boundaries were marked first then this would
            # reverse
            assert (mesh._dofs[:, mesh._dof_bndry_indices==ii].shape[1] == (
                basis_order-1)*nelems_1d[1]+(nelems_1d[1]+1)-2)
            #plt.plot(*mesh._dofs[:, mesh._dof_bndry_indices==ii], 'X')
        for ii in range(2, 4):
            assert (mesh._dofs[:, mesh._dof_bndry_indices==ii].shape[1] == (
                basis_order-1)*nelems_1d[0]+(nelems_1d[0]+1))

        if basis_order == 1:
            assert mesh._dofs.shape[1] == mesh.mesh_pts.shape[1]
        else:
            assert (mesh._dofs.shape[1] ==
                (mesh.mesh_pts.shape[1] + mesh._nelems +
                 (nelems_1d[1]+1)*nelems_1d[0]+(nelems_1d[0]+1)*nelems_1d[1]))

        plt.plot(*mesh._dofs[:, mesh._elem_to_dof_map[2]], 'o')
        def fun(xx):
            return np.sum(xx**basis_order, axis=0)[:, None]
        mesh_vals = fun(mesh._dofs)
        axs = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))[1]
        # Exclude endpoint because basis is zero on right boudnary of
        # canonical element which messes up error. See not in test_1d_mesh
        bounds = [mesh._domain_bounds[0], mesh._domain_bounds[1]-1e-3,
                  mesh._domain_bounds[2], mesh._domain_bounds[3]-1e-3]
        X, Y, Z = get_meshgrid_function_data(fun, bounds, 51)
        axs[0].plot_surface(X, Y, Z)
        X, Y, Z = get_meshgrid_function_data(
            partial(mesh.interpolate, mesh_vals), mesh._domain_bounds, 51)
        axs[1].plot_surface(X, Y, Z)
        pts = cartesian_product(
            [np.linspace(*bounds[:2], 51), np.linspace(*bounds[2:], 51)])
        
        assert np.allclose(fun(pts), mesh.interpolate(mesh_vals, pts))

        # axs = plt.subplots(
        #     basis_order+1, basis_order+1, subplot_kw=dict(projection='3d'))[1]
        # for kk in range(mesh._canonical_basis._indices.shape[1]):
        #     X, Y, Z = get_meshgrid_function_data(
        #         basis.basis_matrix, mesh._canonical_basis._canonical_bounds,
        #         51, qoi=kk)
        #     # plt.contourf(X, Y, Z, levels=21)
        #     ii, jj = mesh._canonical_basis._indices[:, kk]
        #     axs[jj][ii].plot_surface(X, Y, Z)
        #     axs[jj][ii].set_title(f"{ii}, {jj}")
        # plt.show()


if __name__ == "__main__":
    femesh_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestFEMesh)
    unittest.TextTestRunner(verbosity=2).run(femesh_test_suite)
