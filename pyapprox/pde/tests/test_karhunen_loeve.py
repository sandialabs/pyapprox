import unittest

import numpy as np

from pyapprox.pde.kle.kle import (
    multivariate_chain_rule, compute_kle_gradient_from_mesh_gradient, KLE1D,
    MeshKLE, DataDrivenKLE)

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.utilities import approx_jacobian
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_pts_wts_1D
)
from pyapprox.surrogates.bases.orthopoly import GaussLegendreQuadratureRule


class TestKLE():

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
        bkd = self.get_backend()
        nvars, sigma = 2, 3
        length_scale = 1
        mesh = bkd.linspace(0., 1., 11)[None, :]
        kle_mean = mesh[0, :]+2

        for use_log in [False, True]:
            kle = MeshKLE(
                mesh, length_scale, mean_field=kle_mean, use_log=use_log,
                sigma=sigma, nterms=nvars, backend=bkd)

            def scalar_function_of_field(field):
                return field[:, 0] @ field[:, 0]

            sample = bkd.array(np.random.normal(0., 1., (nvars, 1)))
            kle_vals = kle(sample)

            mesh_gradient = kle_vals.T*2
            assert np.allclose(
                mesh_gradient,
                approx_jacobian(scalar_function_of_field, kle_vals), atol=1e-7)

            gradient = compute_kle_gradient_from_mesh_gradient(
                mesh_gradient, kle._eig_vecs, kle._mean_field,
                kle._use_log, sample[:, 0])

            def scalar_function_of_sample(sample):
                field = kle(sample)
                return scalar_function_of_field(field)

            fd_gradient = approx_jacobian(scalar_function_of_sample, sample)
            # print((fd_gradient, gradient))
            assert np.allclose(fd_gradient, gradient)

    def test_mesh_kle_1D(self):
        bkd = self.get_backend()
        level = 10
        nterms = 3
        len_scale, sigma = 1, 1
        lb, ub = 0, 2
        mesh_coords, quad_weights = GaussLegendreQuadratureRule(
            [lb, ub], backend=bkd
        )(2**level+1)

        kle = MeshKLE(
            mesh_coords, len_scale, sigma=sigma, nterms=nterms,
            matern_nu=0.5, quad_weights=quad_weights[:, 0], backend=bkd
        )

        opts = {"mean_field": 0, "sigma2": sigma, "corr_len": len_scale,
                "num_vars": int(kle._nterms), "use_log": False,
                "dom_len": ub-lb}
        kle_exact = KLE1D(opts)
        kle_exact.update_basis_vals(mesh_coords[0, :])

        # kle._eig_vecs are pre multiplied by sqrt_eigvals
        eig_vecs = kle._eig_vecs/kle._sqrt_eig_vals

        # check eigenvalues match
        assert np.allclose(
            kle._sqrt_eig_vals**2, kle_exact.eig_vals, rtol=3e-5)

        # Check basis is orthonormal
        assert np.allclose(
            bkd.sum(
                quad_weights*kle_exact.basis_vals**2, axis=0), 1.0)
        exact_basis_vals = bkd.array(kle_exact.basis_vals)
        assert np.allclose(exact_basis_vals.T @ (
            quad_weights*exact_basis_vals), np.eye(nterms),
                           atol=1e-6)
        # print(np.sum(quad_weights[:, None]*eig_vecs**2, axis=0))
        assert np.allclose(
            eig_vecs.T @ (quad_weights*eig_vecs), np.eye(nterms),
            atol=1e-6)

    def test_mesh_kle_1D_discretization(self):
        # this shows that the quadrature rule does not matter
        level1, level2 = 6, 8
        nterms = 3
        len_scale, sigma = 1, 1

        def trapezoid_rule(level):
            npts = 2**level+1
            pts = np.linspace(-1, 1, npts)
            deltax = pts[1]-pts[0]
            weights = np.ones(npts)*deltax
            weights[[0, -1]] /= 2
            return pts, weights
        quad_rule = trapezoid_rule
        mesh_coords, quad_weights = quad_rule(level2+1)
        quad_weights *= 2   # remove pdf of uniform variable
        # map to [lb, ub]
        lb, ub = 0, 2
        dom_len = ub-lb
        mesh_coords = (mesh_coords+1)/2*dom_len+lb
        quad_weights *= (ub-lb)/2
        mesh_coords = mesh_coords[None, :]
        kle = MeshKLE(
            mesh_coords, len_scale, sigma=sigma, nterms=nterms,
            matern_nu=0.5, quad_weights=quad_weights)

        mesh_coords1, quad_weights1 = quad_rule(level1)
        quad_weights1 *= 2   # remove pdf of uniform variable
        # map to [lb, ub]
        lb1, ub1 = 0, 2  # hack
        dom_len1 = ub1-lb1
        mesh_coords1 = (mesh_coords1+1)/2*dom_len1+lb1
        quad_weights1 *= (ub1-lb1)/2
        mesh_coords1 = mesh_coords1[None, :]

        kle1 = MeshKLE(
            mesh_coords1, len_scale, sigma=sigma, nterms=nterms,
            matern_nu=0.5, quad_weights=quad_weights1)

        # kle._eig_vecs are pre multiplied by sqrt_eigvals
        eig_vecs = kle._eig_vecs/kle._sqrt_eig_vals
        eig_vecs1 = kle1._eig_vecs/kle1._sqrt_eig_vals

        assert np.allclose(
            np.sum(quad_weights[:, None]*eig_vecs**2, axis=0), 1)
        assert np.allclose(
            np.sum(quad_weights1[:, None]*eig_vecs1**2, axis=0), 1)
        assert np.allclose(kle._sqrt_eig_vals, kle1._sqrt_eig_vals, atol=3e-4)

        # import matplotlib.pyplot as plt
        # plt.plot(mesh_coords[0, :], eig_vecs, '-ko')
        # plt.plot(mesh_coords1[0, :], eig_vecs1, 'r--s')
        # plt.show()

    def test_data_driven_kle(self):
        bkd = self.get_backend()
        nterms = 3
        level = 6
        len_scale, sigma = 1, 1
        lb, ub = 0, 2
        mesh_coords, quad_weights = GaussLegendreQuadratureRule(
            [lb, ub], backend=bkd
        )(2**level+1)
        kle = MeshKLE(mesh_coords, len_scale, sigma=sigma, nterms=nterms,
                      matern_nu=0.5, quad_weights=None, backend=bkd)

        nsamples = 10000
        samples = bkd.atleast2d(
            np.random.normal(0., 1., (nterms, nsamples)))
        kle_realizations = kle(samples)

        kle_data = DataDrivenKLE(kle_realizations, nterms=nterms, backend=bkd)
        assert bkd.allclose(
           kle_data._sqrt_eig_vals, kle._sqrt_eig_vals, atol=1e-2, rtol=1e-2)

        kle_data_from_cov = DataDrivenKLE(
            kle_realizations, nterms=nterms, backend=bkd
        )
        super(type(kle_data_from_cov), kle_data_from_cov)._compute_basis()
        # there is precision loss when using from_cov
        assert bkd.allclose(
            kle_data._sqrt_eig_vals,
            kle_data_from_cov._sqrt_eig_vals,
            rtol=1e-3,
            atol=1e-3
        )
        assert bkd.allclose(
            kle_data._eig_vecs, kle_data_from_cov._eig_vecs, atol=1e-4)

        # Test use of quadrature weights
        kle = MeshKLE(
            mesh_coords, len_scale, sigma=sigma, nterms=nterms,
            matern_nu=0.5, quad_weights=quad_weights[:, 0], backend=bkd
        )
        nsamples = 10000
        samples = bkd.atleast2d(
            np.random.normal(0., 1., (nterms, nsamples)))
        kle_realizations = kle(samples)
        kle_data = DataDrivenKLE(
            kle_realizations, nterms=nterms, quad_weights=quad_weights[:, 0],
            backend=bkd
        )
        assert bkd.allclose(
           kle_data._sqrt_eig_vals, kle._sqrt_eig_vals, atol=1e-2, rtol=1e-2)


class TestNumpyKLE(TestKLE, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchKLE(TestKLE, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
