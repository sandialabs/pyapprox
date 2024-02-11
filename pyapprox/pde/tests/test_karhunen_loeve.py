import unittest

import numpy as np

from pyapprox.pde.karhunen_loeve_expansion import (
    multivariate_chain_rule, MeshKLE, compute_kle_gradient_from_mesh_gradient,
    KLE1D)
from pyapprox.util.utilities import approx_jacobian


class TestKLE(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_multivariate_chain_rule(self):
        r"""
        Test computing gradient using multivariate chain rule.

        .. math::  y(u_1,u_2)=u_1^2+u_2*y, u_1(x_1,x_2)=x_1\sin(x_2), u_2(x_1,x_2)=sin^2(x_2)

        .. math::

           \frac{\partial u}{\partial r} = \frac{\partial u}{\partial x}\frac{\partial x}{\partial r} + \frac{\partial u}{\partial y}\frac{\partial y}{\partial r} = (2x)(\sin(t)+(2)(0)=2r\sin^2(t)

        .. math::

           \frac{\partial u}{\partial t} = \frac{\partial u}{\partial x}\frac{\partial x}{\partial t} + \frac{\partial u}{\partial y}\frac{\partial y}{\partial t} = (2x)(r\cos(t)+(2)(2\sin(t)\cos(t))=(r^2+2)\sin(2t)

        """
        def yfun(u):
            return u[0]**2+2.*u[1]

        def ufun(x): return np.array(
            [x[0]*np.sin(x[1]), np.sin(x[1])**2])

        sample = np.random.normal(0., 1., (2))

        exact_gradient = np.array(
            [2.*sample[0]*np.sin(sample[1])**2,
             (sample[0]**2+2.)*np.sin(2.*sample[1])])

        uvec = ufun(sample)
        jac_yu = np.array([2*uvec[0], 2.])
        jac_ux = np.array(
            [np.array([np.sin(sample[1]), 0.]),
             np.array([sample[0]*np.cos(sample[1]), np.sin(2.*sample[1])])]).T

        gradient = multivariate_chain_rule(jac_yu, jac_ux)
        assert np.allclose(exact_gradient, gradient, atol=1e-7)

    def test_compute_kle_gradient_from_mesh_gradient(self):
        nvars, sigma = 2, 3
        length_scale = 1
        mesh = np.linspace(0., 1., 11)[None, :]
        kle_mean = mesh[0, :]+2

        for use_log in [False, True]:
            kle = MeshKLE(mesh, kle_mean, use_log)
            kle.compute_basis(length_scale, sigma, nvars)

            def scalar_function_of_field(field):
                return np.dot(field[:, 0], field[:, 0])

            sample = np.random.normal(0., 1., (nvars, 1))
            kle_vals = kle(sample)

            mesh_gradient = kle_vals.T*2
            assert np.allclose(
                mesh_gradient,
                approx_jacobian(scalar_function_of_field, kle_vals), atol=1e-7)

            gradient = compute_kle_gradient_from_mesh_gradient(
                mesh_gradient, kle.eig_vecs, kle.mean_field,
                kle.use_log, sample[:, 0])

            def scalar_function_of_sample(sample):
                field = kle(sample)
                return scalar_function_of_field(field)

            fd_gradient = approx_jacobian(scalar_function_of_sample, sample)
            # print((fd_gradient, gradient))
            assert np.allclose(fd_gradient, gradient)

    def test_mesh_kle_1D(self):
        level = 10
        nterms = 3
        len_scale, sigma = 1, 1
        from pyapprox.surrogates.orthopoly.quadrature import (
            clenshaw_curtis_pts_wts_1D)
        mesh_coords, quad_weights = clenshaw_curtis_pts_wts_1D(level)
        quad_weights *= 2   # remove pdf of uniform variable
        # map to [lb, ub]
        lb, ub = 0, 2
        dom_len = ub-lb
        mesh_coords = (mesh_coords+1)/2*dom_len+lb
        quad_weights *= (ub-lb)/2
        mesh_coords = mesh_coords[None, :]
        kle = MeshKLE(mesh_coords, matern_nu=0.5, quad_weights=quad_weights)
        kle.compute_basis(len_scale, sigma, nterms)

        opts = {"mean_field": 0, "sigma2": sigma, "corr_len": len_scale,
                "num_vars": int(kle.nterms), "use_log": False,
                "dom_len": dom_len}
        kle_exact = KLE1D(opts)
        kle_exact.update_basis_vals(mesh_coords[0, :])

        # kle.eig_vecs are pre multiplied by sqrt_eigvals
        eig_vecs = kle.eig_vecs/kle.sqrt_eig_vals

        # check eigenvalues match
        # print((kle.sqrt_eig_vals**2-kle_exact.eig_vals)/kle_exact.eig_vals)
        assert np.allclose(kle.sqrt_eig_vals**2, kle_exact.eig_vals, rtol=3e-5)

        # Check basis is orthonormal
        assert np.allclose(
            np.sum(quad_weights[:, None]*kle_exact.basis_vals**2, axis=0), 1.0)
        assert np.allclose(kle_exact.basis_vals.T.dot(
            quad_weights[:, None]*kle_exact.basis_vals), np.eye(nterms),
                           atol=1e-6)
        # print(np.sum(quad_weights[:, None]*eig_vecs**2, axis=0))
        assert np.allclose(
            eig_vecs.T.dot(quad_weights[:, None]*eig_vecs), np.eye(nterms),
            atol=1e-6)

    def test_mesh_kle_1D_discretization(self):
        # this shows that the quadrature rule does not matter
        level1, level2 = 6, 8
        nterms = 3
        len_scale, sigma = 1, 1
        from pyapprox.surrogates.orthopoly.quadrature import (
            clenshaw_curtis_pts_wts_1D)

        def trapezoid_rule(level):
            npts = 2**level+1
            pts = np.linspace(-1, 1, npts)
            deltax = pts[1]-pts[0]
            weights = np.ones(npts)*deltax
            weights[[0, -1]] /= 2
            return pts, weights
        # quad_rule = clenshaw_curtis_pts_wts_1D
        quad_rule = trapezoid_rule
        mesh_coords, quad_weights = quad_rule(level2+1)
        quad_weights *= 2   # remove pdf of uniform variable
        # map to [lb, ub]
        lb, ub = 0, 2
        dom_len = ub-lb
        mesh_coords = (mesh_coords+1)/2*dom_len+lb
        quad_weights *= (ub-lb)/2
        mesh_coords = mesh_coords[None, :]
        kle = MeshKLE(mesh_coords, matern_nu=0.5, quad_weights=quad_weights)
        kle.compute_basis(len_scale, sigma, nterms)

        # quad_rule = clenshaw_curtis_pts_wts_1D
        mesh_coords1, quad_weights1 = quad_rule(level1)
        quad_weights1 *= 2   # remove pdf of uniform variable
        # map to [lb, ub]
        lb1, ub1 = 0, 2  # hack
        dom_len1 = ub1-lb1
        mesh_coords1 = (mesh_coords1+1)/2*dom_len1+lb1
        quad_weights1 *= (ub1-lb1)/2
        mesh_coords1 = mesh_coords1[None, :]

        kle1 = MeshKLE(
            mesh_coords1, matern_nu=0.5, quad_weights=quad_weights1)
        kle1.compute_basis(len_scale, sigma, nterms)

        # kle.eig_vecs are pre multiplied by sqrt_eigvals
        eig_vecs = kle.eig_vecs/kle.sqrt_eig_vals
        eig_vecs1 = kle1.eig_vecs/kle1.sqrt_eig_vals

        assert np.allclose(
            np.sum(quad_weights[:, None]*eig_vecs**2, axis=0), 1)
        assert np.allclose(
            np.sum(quad_weights1[:, None]*eig_vecs1**2, axis=0), 1)
        # print(kle.sqrt_eig_vals-kle1.sqrt_eig_vals)
        assert np.allclose(kle.sqrt_eig_vals, kle1.sqrt_eig_vals, atol=3e-4)

        # import matplotlib.pyplot as plt
        # plt.plot(mesh_coords[0, :], eig_vecs, '-ko')
        # plt.plot(mesh_coords1[0, :], eig_vecs1, 'r--s')
        # plt.show()

    def check_mesh_kle_1D_discretization(self):
        # this shows that the quadrature rule does not matter
        level1, level2 = 6, 8
        # level1, level2 = 3, 3
        level = max(level1, level2)+1
        nterms = 6
        len_scale, sigma = 1, 1

        def trapezoid_rule(level):
            npts = 2**level+1
            pts = np.linspace(-1, 1, npts)
            deltax = pts[1]-pts[0]
            weights = np.ones(npts)*deltax
            weights[[0, -1]] /= 2
            return pts, weights
        quad_rule = trapezoid_rule
        mesh_coords, quad_weights = quad_rule(level)
        quad_weights *= 2   # remove pdf of uniform variable
        # map to [lb, ub]
        lb, ub = 0, 2
        dom_len = ub-lb
        mesh_coords = (mesh_coords+1)/2*dom_len+lb
        quad_weights *= (ub-lb)/2
        mesh_coords = mesh_coords[None, :]
        quad_weights = None
        kle = MeshKLE(mesh_coords, matern_nu=0.5, quad_weights=quad_weights)
        kle.compute_basis(len_scale, sigma, nterms)

        # quad_rule = clenshaw_curtis_pts_wts_1D
        mesh_coords1, quad_weights1 = quad_rule(level1)
        quad_weights1 *= 2   # remove pdf of uniform variable
        # map to [lb, ub]
        lb1, ub1 = 0, 1.75
        dom_len1 = ub1-lb1
        mesh_coords1 = (mesh_coords1+1)/2*dom_len1+lb1
        quad_weights1 *= dom_len1/2

        mesh_coords2, quad_weights2 = quad_rule(level2)
        quad_weights2 *= 2   # remove pdf of uniform variable
        lb2, ub2 = 1.75, 2
        dom_len2 = ub2-lb2
        mesh_coords2 = (mesh_coords2+1)/2*dom_len2+lb2
        quad_weights2 *= dom_len2/2

        mesh_coords_mix = np.hstack(
            (mesh_coords1[None, :-1], mesh_coords2[None, :]))
        # remove duplicate point
        quad_weights2[0] += quad_weights1[-1]
        quad_weights_mix = np.hstack([quad_weights1[:-1], quad_weights2])
        quad_weights_mix = None
        # following only true if level1=level2=level+1
        # assert np.allclose(mesh_coords, mesh_coords_mix)
        # assert np.allclose(quad_weights, quad_weights_mix)

        kle_mix = MeshKLE(
            mesh_coords_mix, matern_nu=0.5, quad_weights=quad_weights_mix)
        kle_mix.compute_basis(len_scale, sigma, nterms)

        # kle.eig_vecs are pre multiplied by sqrt_eigvals
        eig_vecs = kle.eig_vecs/kle.sqrt_eig_vals
        eig_vecs_mix = kle_mix.eig_vecs/kle_mix.sqrt_eig_vals

        # print(np.sum(quad_weights[:, None]*eig_vecs**2, axis=0))
        # print(np.sum(quad_weights_mix[:, None]*eig_vecs_mix**2, axis=0))

        import matplotlib.pyplot as plt
        plt.plot(mesh_coords[0, :], eig_vecs, '-ko')
        plt.plot(mesh_coords_mix[0, :], eig_vecs_mix, 'r--s')
        plt.show()


if __name__ == "__main__":
    kle_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestKLE)
    unittest.TextTestRunner(verbosity=2).run(kle_test_suite)
