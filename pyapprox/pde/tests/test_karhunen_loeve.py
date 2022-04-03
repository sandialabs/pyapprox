import unittest
import numpy as np

from pyapprox.pde.karhunen_loeve_expansion import (
    multivariate_chain_rule, MeshKLE, compute_kle_gradient_from_mesh_gradient
)
from pyapprox.utilities.utilities import approx_jacobian


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

            sample = np.random.normal(0. ,1., (nvars, 1))
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
        

if __name__ == "__main__":
    kle_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestKLE)
    unittest.TextTestRunner(verbosity=2).run(kle_test_suite)
