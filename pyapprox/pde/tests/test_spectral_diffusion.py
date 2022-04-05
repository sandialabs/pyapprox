import numpy as np
import unittest
import sympy as sp
import matplotlib.pyplot as plt

from pyapprox.pde.spectral_diffusion import (
    SteadyStateAdvectionDiffusionEquation1D,
    TransientAdvectionDiffusionEquation1D,
    SteadyStateAdvectionDiffusionEquation2D,
    TransientAdvectionDiffusionEquation2D
)
from pyapprox.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.util.utilities import check_gradients


def get_forcing_for_steady_state_constant_advection_diffusion_2d_sympy(
        sol_string, diffusivity, advection_1, advection_2):
    # from sympy.abc import t as sp_t
    sp_x, sp_y = sp.symbols(['x', 'y'])
    # u = sp.sin(sp.pi*sp_x)*sp.cos(sp_t)
    u = sp.sympify(sol_string)
    dxu2 = u.diff(sp_x, 2) + u.diff(sp_y, 2)  # diffusion
    # dtu = u.diff(sp_t, 1)   # time derivative
    dxu = advection_1*u.diff(sp_x, 1)+advection_2*u.diff(sp_y, 1)  # advection
    # sp_forcing = dtu-(diffusivity*dxu2+advection*dxu)
    sp_forcing = -(diffusivity*dxu2-dxu)
    print(sp_forcing)
    # forcing_fun = sp.lambdify((sp_x, sp_y, sp_t), sp_forcing, "numpy")
    forcing_fun = sp.lambdify((sp_x, sp_y), sp_forcing, "numpy")
    return forcing_fun


class TestSpectralDiffusion2D(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.eps = 2 * np.finfo(float).eps

    def test_derivative_matrix(self):
        order = 4
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [-1, 1]
        model.initialize(bndry_conds, lambda x, z: x.T*0+1,
                         lambda x, z: x.T*0, lambda x, z: x.T*0, order, domain)
        derivative_matrix = model.get_derivative_matrices()[0]
        true_matrix = \
            [[5.5,        -6.82842712,  2.,         -1.17157288,  0.5],
             [1.70710678, -0.70710678, -1.41421356,  0.70710678, -0.29289322],
             [-0.5,         1.41421356, -0.,         -1.41421356,  0.5],
             [0.29289322, -0.70710678,  1.41421356,  0.70710678, -1.70710678],
             [-0.5,         1.17157288, -2.,          6.82842712, -5.5]]
        # I return points and calculate derivatives using reverse order of
        # points compared to what is used by Matlab cheb function thus the
        # derivative matrix I return will be the negative of the matlab version
        assert np.allclose(-derivative_matrix, true_matrix)

        def fun(x): return (np.exp(x)*np.sin(5*x)).T
        def grad(x): return (np.exp(x)*(np.sin(5*x)+5*np.cos(5*x))).T

        order = 20
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [-1, 1]
        model.initialize(bndry_conds, lambda x, z: x.T*0+1,
                         lambda x, z: x.T*0, lambda x, z: x.T*0, order, domain)
        cheb_grad = model.mesh.derivative_matrices[0].dot(
            fun(model.mesh.mesh_pts))
        # from pyapprox import plt
        # plt.plot(model.mesh.mesh_pts, grad(model.mesh.mesh_pts))
        # plt.plot(model.mesh.mesh_pts, cheb_grad)
        # plt.show()
        error = np.absolute(grad(model.mesh.mesh_pts) - cheb_grad)
        # print(error)
        assert np.max(error) < 1e-9

    def test_homogeneous_possion_equation(self):
        """
        solve -u(x)'' = 0, u(0) = 0, u(1) = 0.5
        """

        order = 4
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0+0.5, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: x.T*0+1,
                         lambda x, z: x.T*0, lambda x, z: x.T*0, order, domain)
        mesh_pts = model.get_collocation_points()
        sample = np.zeros((0))  # dummy for this example
        solution = model.solve(sample)
        def exact_sol(x): return 0.5*x.T
        # print(np.linalg.norm(exact_sol(mesh_pts)-solution))
        assert np.linalg.norm(
            exact_sol(mesh_pts)-solution) < 20*self.eps

    def test_neumann_boundary_conditions(self):
        """
        Solve -u(x)''=exp(4x) u(-1)'=0 and u(1)=0
        """
        order = 20
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "N"],
                       [lambda x: x.T*0, "D"]]
        domain = [-1, 1]
        model.initialize(bndry_conds, lambda x, z: x.T*0+1,
                         lambda x, z: -np.exp(4*x.T),
                         lambda x, z: x.T*0, order, domain)
        mesh_pts = model.get_collocation_points()
        sample = np.zeros((0))  # dummy for this example
        solution = model.solve(sample)
        def exact_sol(x): return (
            np.exp(4*x)-4*np.exp(-4)*(x-1)-np.exp(4)).T/16
        # print(np.linalg.norm(exact_sol(mesh_pts)-solution))
        assert np.linalg.norm(
            exact_sol(mesh_pts)-solution) < 1e-11

    def test_inhomogeneous_possion_equation(self):
        """
        solve -u(x)'' = -1, u(0) = 0, u(1) = 1
        solution u(x) =  0.5*(x-3.)*x
        """
        order = 4
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0-1, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: x.T*0+1,
                         lambda x, z: 0*x.T-1,
                         lambda x, z: x.T*0, order, domain)
        mesh_pts = model.get_collocation_points()
        sample = np.zeros((0))  # dummy for this example
        solution = model.solve(sample)
        def exact_sol(x): return (0.5*(x-3.)*x).T
        print(np.linalg.norm(exact_sol(mesh_pts)-solution))
        assert np.linalg.norm(
            exact_sol(mesh_pts)-solution) < 30*self.eps

    def test_inhomogeneous_advection_diffusion_equation(self):
        """
        solve -u(x)'' + a u(x)' = a cos(x)+sin(x), u(0) = 0, u(1) = sin(1)
        solution u(x) =  sin(x)
        """
        a = 10
        order = 20
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0+np.sin(1), "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: x.T*0+1,
                         lambda x, z: (a*np.cos(x)+np.sin(x)).T,
                         lambda x, z: x.T*0+a, order, domain)
        mesh_pts = model.get_collocation_points()
        sample = np.zeros((0))  # dummy for this example
        solution = model.solve(sample)
        def exact_sol(x): return np.sin(x.T)
        # print(np.linalg.norm(exact_sol(mesh_pts)-solution))
        assert np.linalg.norm(
            exact_sol(mesh_pts)-solution) < 6e-14

    def test_homogeneous_advection_diffusion_equation(self):
        """
        solve -a u(x)'' - u(x)' = 0, u(0) = 0, u(1) = 1
        solution u(x) =  sin(x)
        """
        a = 1
        order = 20
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0+1, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: x.T*0+a,
                         lambda x, z: x.T*0,
                         lambda x, z: x.T*0-1, order, domain)
        mesh_pts = model.get_collocation_points()
        sample = np.zeros((0))  # dummy for this example
        solution = model.solve(sample)
        def exact_sol(x): return ((np.exp(-x/a)-1)/(np.exp(-1/a)-1)).T
        # plt.plot(mesh_pts[0, :], exact_sol(mesh_pts)[:, 0])
        # plt.plot(mesh_pts[0, :], solution[:, 0], '--')
        # plt.show()
        print(np.linalg.norm(exact_sol(mesh_pts)-solution))
        assert np.linalg.norm(
            exact_sol(mesh_pts)-solution) < 2e-13

    def test_inhomogeneous_diffusion_equation_with_variable_coefficient(self):
        """
        solve -((1+x)*u(x)')' = 1, u(0) = 0, u(1) = 0
        solution u(x) = log(x+1)/log(2) - x
        """
        order = 20
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: x.T+1,
                         lambda x, z: x.T*0+1,
                         lambda x, z: x.T*0, order, domain)
        mesh_pts = model.get_collocation_points()
        sample = np.zeros((0))  # dummy for this example
        solution = model.solve(sample)
        def exact_sol(x): return (np.log(x+1.) / np.log(2.) - x).T
        # print(np.linalg.norm(exact_sol(mesh_pts)-solution))
        assert np.linalg.norm(
            exact_sol(mesh_pts)-solution) < 4e-14

    def test_integrate_1d(self):
        order = 4
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: x.T+1,
                         lambda x, z: x.T*0+1,
                         lambda x, z: x.T*0, order, domain)
        mesh_pts = model.get_collocation_points()
        print(model.mesh.integrate(mesh_pts.T**2))
        assert np.allclose(model.mesh.integrate(mesh_pts.T**2), 1./3.)
        assert np.allclose(model.mesh.integrate(mesh_pts.T**3), 1./4.)

        order = 4
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [-1, 1]
        model.initialize(bndry_conds, lambda x, z: x.T+1,
                         lambda x, z: x.T*0-1,
                         lambda x, z: x.T*0, order, domain)
        mesh_pts = model.get_collocation_points()
        print(model.mesh.integrate(mesh_pts.T**2))
        assert np.allclose(model.mesh.integrate(mesh_pts.T**2), 2./3.)
        assert np.allclose(model.mesh.integrate(mesh_pts.T**3), 0.)

    def test_evaluate(self):
        """
        for the PDE -((1+z*x)*u(x)')' = 1, u(0) = 0, u(1) = 0
        buse model.evaluate to extract QoI
        """
        order = 20
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: z*x.T + 1.,
                         lambda x, z:  0*x.T+1,
                         lambda x, z: x.T*0, order, domain)

        qoi_coords = np.array([0.05, 0.5, 0.95])
        model.qoi_functional = lambda x: model.mesh.interpolate(
            x, qoi_coords)[:, 0]

        sample = np.ones((1, 1), float)
        qoi = model(sample)
        assert np.allclose(np.log(qoi_coords+1.)/np.log(2.)-qoi_coords, qoi)

        sample = 0.5*np.ones((1, 1), float)
        qoi = model(sample)
        assert np.allclose(
            -(qoi_coords*np.log(9./4.)-2.*np.log(qoi_coords+2.) +
              np.log(4.))/np.log(3./2.), qoi)

    def test_evaluate_gradient_1d(self):
        """
        for the PDE -((1+sum(z^2)*x)*u(x)')' = 2, u(0) = 0, u(1) = 1
        use model.evaluate_gradient to evaluate the gradient of the QoI
        with respect to the random parameter vector z.
        The QoI is the intergral of the solution over the entire domain
        The adjoint rhs is then just 1.
        """
        order = 20
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        model.initialize(
            bndry_conds, lambda x, z: ((z[0]**2+z[1]**2)*x + 1.).T,
            lambda x, z: 0*x.T+2, lambda x, z: x.T*0, order, domain)

        sample = np.random.RandomState(2).uniform(-1, 1, (2, 1))
        # derivatives with respect to the mesh x
        model.diffusivity_derivs_fun = lambda x, z, i: 2.*x.T*z[i]
        model.forcing_derivs_fun = lambda x, z, i: 0.0*x.T
        model.advection_derivs_fun = lambda x, z, i: 0.0*x.T
        model(sample)
        # evaluate_gradient has to be called before any more calls to
        # model.solve with different parameters, because we need to
        # access self.fwd_solution, which will change with any subsuquent calls
        errors = check_gradients(
            model, lambda x: model.evaluate_gradient(x[:, 0]), sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 8e-7

    @unittest.skip("Gradient does not work when advection is turned on")
    def test_evaluate_advection_gradient_1d(self):
        """
        For the PDE
             -((1+sum(z^2)*x)*u(x)')'+2*sum(z)*u(x)' = 2, u(0) = 0, u(1) = 1
        use model.evaluate_gradient to evaluate the gradient of the QoI
        with respect to the random parameter vector z.
        The QoI is the intergral of the solution over the entire domain
        The adjoint rhs is then just 1.
        """
        aa = 10
        order = 20
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        model.initialize(
            bndry_conds, lambda x, z: ((z[0]**2+z[1]**2)*x + 1.).T,
            lambda x, z: 0*x.T+2,
            lambda x, z: aa*(z[0]+z[1])+0*x.T,
            order, domain)

        sample = np.random.RandomState(2).uniform(-1, 1, (2, 1))
        # sample = np.ones((2, 1))
        # derivatives with respect to z
        model.diffusivity_derivs_fun = lambda x, z, i: 2.*x.T*z[i]
        model.forcing_derivs_fun = lambda x, z, i: 0.0*x.T
        model.advection_derivs_fun = lambda x, z, i: x.T*0+aa
        model(sample)
        # evaluate_gradient has to be called before any more calls to
        # model.solve with different parameters, because we need to
        # access self.fwd_solution, which will change with any subsuquent calls
        errors = check_gradients(
            model, lambda x: model.evaluate_gradient(x[:, 0]), sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 6e-7

    @unittest.skip("Not fully implemented")
    def test_compute_error_estimate(self):
        """
        for the PDE -((1+z*x)*u(x)')' = 1, u(0) = 0, u(1) = 0
        use model.compute_error_estomate to compute an error estimate of
        the deterministic error in the foward solution.
        The QoI is the intergral of the solution over the entire domain
        The adjoint rhs is then just 1.
        """
        order = 5
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)

        model.diffusivity_fun = lambda x, z: z[0]*x + 1.
        model.forcing_func = lambda x, z: 0.*x+1.

        sample = np.ones((1, 1), float)
        qoi = model(sample)
        error_estimate = model.compute_error_estimate(sample)

        # solution = model.solve(sample[:, 0])
        def exact_solution(x): return np.log(x+1.)/np.log(2.)-x
        gl_pts, gl_wts = gauss_jacobi_pts_wts_1D(50, 0, 0)
        x_range = model.xlim[1]-model.xlim[0]
        gl_pts = x_range*(gl_pts+1.)/2.+model.xlim[0]
        gl_wts *= x_range
        gl_vals = exact_solution(gl_pts)
        exact_qoi = np.dot(gl_vals, gl_wts)

        exact_error = abs(exact_qoi-qoi)

        # print('err estimate', error_estimate)
        # print('exact err', exact_error)
        # print('effectivity ratio', error_estimate / exact_error)
        # should be very close to 1. As adjoint order is increased
        # it will converge to 1

        sample = 0.5*np.ones((1), float)
        qoi = model.evaluate(sample)
        exact_solution = -(model.mesh.mesh_pts*np.log(9./4.) -
                           2.*np.log(model.mesh.mesh_pts+2.) +
                           np.log(4.))/np.log(3./2.)
        exact_qoi = model.qoi_functional(exact_solution)
        error = abs(exact_qoi-qoi)
        error_estimate = model.compute_error_estimate(sample)

        # print(error_estimate, error)
        # print model.mesh.integrate( (exact_solution - solution )**2 )
        assert np.allclose(error_estimate, error)

    def test_timestepping_without_forcing(self):
        r"""
        solve u_t(x,t) = u_xx(x,t), u(-1,t) = 0, u(1,t) = 0,
        u(x,0) = \sin(\pi*x)

        Exact solution
        u(x,t) = \exp(-\pi^2t)*sin(\pi*x)
        """
        def exact_sol(x, t): return np.exp(-np.pi**2*t)*np.sin(np.pi*x.T)

        order = 16
        model = TransientAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        final_time = 1
        time_step_size = 1e-3
        domain = [-1, 1]
        model.initialize(
            bndry_conds, lambda x, z: 0*x.T + 1., lambda x, z, t: 0*x.T,
            lambda x, z: 0*x.T, order, domain, final_time, time_step_size,
            lambda x: exact_sol(x, 0))

        sample = np.ones((1), float)  # dummy argument for this example
        solution = model.solve(sample)

        for i, time in enumerate(model.times):
            exact_sol_t = exact_sol(model.mesh.mesh_pts, time)
            model_sol_t = solution[:, i:i+1]
            L2_error = np.sqrt(
                model.mesh.integrate((exact_sol_t-model_sol_t)**2))
            factor = np.sqrt(
                model.mesh.integrate(exact_sol_t**2))
            # print(time, L2_error, 1e-3*factor)
            assert L2_error < 1e-4*factor  # crank-nicholson

    def test_timestepping_with_time_independent_forcing_1d(self):
        r"""
        solve u_t(x,t) = u_xx(x,t)+sin(3\pi x), u(0,t) = 0, u(1,t) = 0,
        u(x,0) = 5\sin(2\pi x)+2\sin(3\pi x)

        Exact solution
        u(x,t) = 5\exp(-4\pi^2t)*sin(2\pi*x)+(2\exp(-9\pi^2t)+(1-\exp(-9\pi^2t))/(9\pi^2))*\sin(3\pi x)
        """
        def exact_sol(x, t): return (
                5.*np.exp(-4.*np.pi**2*t)*np.sin(2.*np.pi*x) +
                (2.*np.exp(-9.*np.pi**2*t)+(1.-np.exp(-9.*np.pi**2*t)) /
                 (9.*np.pi**2))*np.sin(3.*np.pi*x)).T

        order = 32
        model = TransientAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        final_time = 1
        time_step_size = 1e-3
        domain = [0, 1]
        model.initialize(
            bndry_conds, lambda x, z: 0*x.T + 1.,
            lambda x, z, t: np.sin(3*np.pi*x).T,
            lambda x, z: 0*x.T, order, domain, final_time, time_step_size,
            lambda x: exact_sol(x, 0))

        sample = np.ones((1), float)  # dummy argument for this example

        solution = model.solve(sample)
        for i, time in enumerate(model.times):
            exact_sol_t = exact_sol(model.mesh.mesh_pts, time)
            model_sol_t = solution[:, i:i+1]
            L2_error = np.sqrt(
                model.mesh.integrate((exact_sol_t-model_sol_t)**2))
            factor = np.sqrt(model.mesh.integrate(exact_sol_t**2))
            # print(time, L2_error, 1e-3*factor)
            assert L2_error < 1e-3*factor  # crank-nicholson

    def test_timestepping_with_time_dependent_forcing_1d(self):
        r"""
        solve
            u_t(x,t) = u_xx(x,t)-sin(t)sin(pi*x)+pi**2*cos(t)sin(pi*x),
            u(0,t) = 0, u(1,t) = 0,
        Exact_solution
          u(x,t) = sin(pi*x)cos(t)
        """
        def exact_sol(x, t): return np.sin(np.pi*x.T)*np.cos(t)

        order = 32
        model = TransientAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        final_time = 1.0
        time_step_size = 1e-2
        domain = [0, 1]
        model.initialize(
            bndry_conds, lambda x, z: 0*x.T + 1.,
            lambda x, z, t: (-np.sin(t)*np.sin(np.pi*x) +
                             np.pi**2*np.cos(t)*np.sin(np.pi*x)).T,
            lambda x, z: 0*x.T, order, domain, final_time, time_step_size,
            lambda x: exact_sol(x, 0))

        sample = np.ones((1), float)  # dummy argument for this example

        # model.set_time_step_method('backward-euler')
        solution = model.solve(sample)
        test_mesh_pts = np.linspace(domain[0], domain[1], 100)[None, :]
        for i, time in enumerate(model.times):
            exact_sol_t = exact_sol(model.mesh.mesh_pts, time)
            model_sol_t = solution[:, i:i+1]
            L2_error = np.sqrt(
                model.mesh.integrate((exact_sol_t-model_sol_t)**2))
            factor = np.sqrt(
                model.mesh.integrate(exact_sol_t**2))
            plot = False
            if plot and i == len(model.times)-1:
                test_exact_sol_t = exact_sol(test_mesh_pts, time)
                test_model_sol_t = model.mesh.interpolate(
                    model_sol_t, test_mesh_pts)
                plt.plot(test_mesh_pts[0, :], test_model_sol_t, 'k',
                             label='collocation', linewidth=2)
                plt.plot(test_mesh_pts[0, :], test_exact_sol_t,
                             'r--', label='exact', linewidth=2)
                plt.plot(
                    model.mesh.mesh_pts[0, :], model_sol_t[:, 0], 'ko')
                plt.plot(
                    model.mesh.mesh_pts[0, :], exact_sol_t[:, 0], 'rs')
                plt.legend(loc=0)
                plt.title('$t=%1.4f$' % time)
                plt.ylim(-1, 1)
                # plt.show()
            assert L2_error < 1e-4*factor  # crank-nicholson

    def test_timestepping_with_time_dependent_forcing_2d(self):
        r"""
        solve
            u_t(x,t) = u_xx(x,t)-f(x,t)
            u(0,t) = 0, u(1,t) = 0,
        Exact_solution
          u(x,t) = sin(pi*x)cos(pi*y)cos(t)
        """
        def exact_sol(x, t):
            return np.sin(np.pi*x[:1].T)*np.cos(np.pi*x[1:2].T)*np.cos(t)

        order = [40, 32]
        model = TransientAdvectionDiffusionEquation2D()
        bndry_conds = [[lambda x: exact_sol(x, 0), "D"],
                       [lambda x: exact_sol(x, 0), "D"],
                       [lambda x: exact_sol(x, 0), "D"],
                       [lambda x: exact_sol(x, 0), "D"]]

        final_time = 1.0
        time_step_size = 1e-2
        domain = [0, 1, 0.5, 1.5]
        model.initialize(
            bndry_conds, lambda x, z: 0*x[:1].T + 1.,
            lambda x, z, t: (
                -np.sin(t)*np.sin(np.pi*x[:1])*np.cos(np.pi*x[1:2]) +
                2*np.pi**2*np.cos(t)*np.sin(np.pi*x[:1])*np.cos(np.pi*x[1:2])
            ).T,
            lambda x, z: 0*x.T, order, domain, final_time, time_step_size,
            lambda x: exact_sol(x, 0))

        sample = np.ones((1), float)  # dummy argument for this example

        # model.set_time_step_method('backward-euler')
        solution = model.solve(sample)
        for i, time in enumerate(model.times):
            exact_sol_t = exact_sol(model.mesh.mesh_pts, time)
            model_sol_t = solution[:, i:i+1]
            L2_error = np.sqrt(
                model.mesh.integrate((exact_sol_t-model_sol_t)**2))
            factor = np.sqrt(
                model.mesh.integrate(exact_sol_t**2))
            plot = False
            if plot and (i == 0 or i == len(model.times)-1):
                fig, axs = plt.subplots(1, 2, figsize=(8, 6))
                p1 = model.mesh.plot(model_sol_t, 100, 20, axs[0])
                p2 = model.mesh.plot(exact_sol_t, 100, 20, axs[1])
                plt.colorbar(p1, ax=axs[0])
                plt.colorbar(p2, ax=axs[1])
                plt.show()
            # print(time, L2_error, 1e-4*factor)
            assert L2_error < 1e-4*factor  # crank-nicholson

    def test_convergence(self):

        def exact_sol(x, t):
            return (5.*np.exp(-4.*np.pi**2*t)*np.sin(2.*np.pi*x) +
                    (2.*np.exp(-9.*np.pi**2*t) +
                     (9.*np.pi**2*np.sin(t)-np.cos(t)+np.exp(-9.*np.pi**2*t)) /
                     (1+81.*np.pi**4))*np.sin(3.*np.pi*x)).T

        # order = 8  # 1e-5
        # order = 16 #1e-11
        order = 20  # 2e-15
        model = TransientAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        final_time = 1.
        time_step_size = 1e-2
        sample = np.ones((1), float)  # dummy argument for this example

        num_convergence_steps = 4
        errors = np.empty((num_convergence_steps), float)
        time_step_sizes = np.empty((num_convergence_steps), float)
        num_time_steps = np.empty((num_convergence_steps), float)
        for i in range(num_convergence_steps):
            model.initialize(
                bndry_conds, lambda x, z: 0*x.T + 1.,
                lambda x, z, t: np.sin(3*np.pi*x.T)*np.sin(t),
                lambda x, z: 0*x.T, order, domain, final_time, time_step_size,
                lambda x: exact_sol(x, 0))

            solution = model.solve(sample)
            assert np.allclose(model.times[-1], final_time, atol=1e-15)
            exact_sol_t = exact_sol(model.mesh.mesh_pts, final_time)
            model_sol_t = solution[:, -1:]
            L2_error = np.sqrt(model.mesh.integrate((exact_sol_t-model_sol_t)**2))
            errors[i] = L2_error
            # print(L2_error, solution.shape)
            time_step_sizes[i] = time_step_size
            num_time_steps[i] = model.num_time_steps
            time_step_size /= 2

        conv_rate = -np.log10(errors[-1]/errors[0])/np.log10(
            num_time_steps[-1]/num_time_steps[0])
        # print(conv_rate)
        assert np.allclose(conv_rate, 2, atol=1e-4)
        # plt.loglog(
        #     num_time_steps, errors, 'o-r',
        #     label=r'$\lVert u(x,T)-\hat{u}(x,T)\\rVert_{\ell_2(D)}$',
        #     linewidth=2)
        # # print errors[0]*num_time_steps[0]/num_time_steps
        # order = 1
        # plt.loglog(
        #     num_time_steps,
        #     errors[0]*num_time_steps[0]**order/num_time_steps**order,
        #     'o--', label=r'$(\Delta t)^{-%d}$' % order, linewidth=2)
        # order = 2
        # plt.loglog(
        #     num_time_steps,
        #     errors[0]*num_time_steps[0]**order/num_time_steps**order,
        #     'o--', label=r'$(\Delta t)^{-%d}$' % order, linewidth=2)
        # plt.legend(loc=0)
        # plt.show()

    def test_inhomogeneous_diffusion_equation_2d_variable_coefficient(self):
        """
        wolfram alpha z random variable x and w are spatial dimension
        d/dx 16*exp(-z^2)*(x^2-1/4)*(w^2-1/4)
        d/dx (1+t/pi^2*z*cos(pi/2*(x^2+w^2)))*32*(w^2-1/4)*x*exp(-z^2)
        Peter zaspels thesis is wrong it is 1 = sigma * not 1 + sigma +
        """

        sigma = 1

        def forcing_fun(x, z):
            vals = -(32.*(1.+sigma*z[0]*sigma*np.cos(
                np.pi/2.*(x[0, :]**2+x[1, :]**2))/np.pi**2) *
                np.exp(-z[0]**2)*(x[0, :]**2+x[1, :]**2-0.5) -
                32./np.pi*z[0]*sigma*np.sin(np.pi/2.*(x[0, :]**2+x[1, :]**2)) *
                (x[0, :]**2 * np.exp(-z[0]**2)*(x[1, :]**2-0.25)+x[1, :]**2 *
                 np.exp(-z[0]**2)*(x[0, :]**2-0.25)))[:, None]
            return vals

        def diffusivity_fun(x, z):
            return (1.+sigma/np.pi**2*z[0]*np.cos(
                np.pi/2.*(x[0, :]**2+x[1, :]**2)))[:, None]

        # only well posed if |y| < pi^2/sigma
        def exact_sol(x, y): return (
                16.*np.exp(-y**2) *
                (x[0, :]**2-0.25)*(x[1, :]**2-0.25))[:, None]

        order = 16
        model = SteadyStateAdvectionDiffusionEquation2D()
        domain = [-0.5, 0.5, -0.5, 0.5]
        bndry_conds = [[lambda x: x[1:2, :].T*0, "D"],
                       [lambda x: x[1:2, :].T*0, "D"],
                       [lambda x: x[0:1, :].T*0, "D"],
                       [lambda x: x[0:1, :].T*0, "D"]]
        model.initialize(
                bndry_conds, diffusivity_fun, forcing_fun,
                lambda x, z: np.zeros((x.shape[1], 2)), order, domain)

        num_dims = 1
        rng = np.random.RandomState(1)
        sample = rng.uniform(-np.sqrt(3), np.sqrt(3), (num_dims))
        mesh_pts = model.get_collocation_points()
        solution = model.solve(sample)
        # print (np.linalg.norm(exact_sol(mesh_pts, sample)-solution))
        assert np.linalg.norm(exact_sol(mesh_pts, sample) - solution) < 2.e-12

    def test_2d_advection_diffusion_neumann_x_dim_bcs(self):
        sol_string = "x**2*sin(pi*y)"
        sp_forcing_fun = \
            get_forcing_for_steady_state_constant_advection_diffusion_2d_sympy(
                sol_string, 1, 1, 0)

        def exact_sol(x): return (x[0, :]**2*np.sin(np.pi*x[1, :]))[:, None]

        def forcing_fun(x, z):
            return sp_forcing_fun(x[0, :], x[1, :])[:, None]

        order = 16
        model = SteadyStateAdvectionDiffusionEquation2D()
        domain = [0, 1, 0, 1]
        bndry_conds = [
            [lambda x: np.zeros((x.shape[1], 1)), "N"],
            [lambda x: exact_sol(x), "D"],
            [lambda x: np.zeros((x.shape[1], 1)), "D"],
            [lambda x: np.zeros((x.shape[1], 1)), "D"]]
        model.initialize(
            bndry_conds, lambda x, z: np.ones((x.shape[1], 1)), forcing_fun,
            lambda x, z: np.hstack(
                (np.ones((x.shape[1], 1)), np.zeros((x.shape[1], 1)))),
            order, domain)

        sample = np.zeros((0))  # dummy for this example
        solution = model.solve(sample)

        # print(np.linalg.norm(exact_sol(model.mesh.mesh_pts)-solution))
        # fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
        # X, Y, Z = get_meshgrid_function_data(exact_sol, model.domain, 30)
        # p = axs[0].contourf(
        #     X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 10))
        # plt.colorbar(p, ax=axs[0])
        # X, Y, Z = get_meshgrid_function_data(
        #     partial(model.mesh.interpolate, solution), model.domain, 30)
        # p = axs[1].contourf(
        #     X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 10))
        # plt.colorbar(p, ax=axs[1])
        # plt.show()

        assert np.linalg.norm(
            exact_sol(model.mesh.mesh_pts)-solution) < 1e-9

    def test_2d_advection_diffusion_neumann_y_dim_bcs(self):
        sol_string = "y**2*sin(pi*x)"
        sp_forcing_fun = \
            get_forcing_for_steady_state_constant_advection_diffusion_2d_sympy(
                sol_string, 1, 1, 0)

        def exact_sol(x): return (x[1, :]**2*np.sin(np.pi*x[0, :]))[:, None]

        def forcing_fun(x, z):
            return sp_forcing_fun(x[0, :], x[1, :])[:, None]

        order = 16
        model = SteadyStateAdvectionDiffusionEquation2D()
        domain = [0, 1, 0, 1]
        bndry_conds = [
            [lambda x: np.zeros((x.shape[1], 1)), "D"],
            [lambda x: np.zeros((x.shape[1], 1)), "D"],
            [lambda x: np.zeros((x.shape[1], 1)), "N"],
            [lambda x: exact_sol(x), "D"]]
        model.initialize(
            bndry_conds, lambda x, z: np.ones((x.shape[1], 1)), forcing_fun,
            lambda x, z: np.hstack(
                (np.ones((x.shape[1], 1)), np.zeros((x.shape[1], 1)))),
            order, domain)

        sample = np.zeros((0))  # dummy for this example
        solution = model.solve(sample)

        assert np.linalg.norm(
            exact_sol(model.mesh.mesh_pts)-solution) < 1e-9

    def test_integrate_2d(self):
        order = 4
        model = SteadyStateAdvectionDiffusionEquation2D()
        domain = [0, 1, 0, 1]
        bndry_conds = [[lambda x: x[1:2, :].T*0, "D"],
                       [lambda x: x[1:2, :].T*0, "D"],
                       [lambda x: x[0:1, :].T*0, "D"],
                       [lambda x: x[0:1, :].T*0, "D"]]
        model.initialize(
            bndry_conds, lambda x, z: np.ones((x.shape[1], 1)),
            lambda x, z: np.zeros((x.shape[1], 1)),
            lambda x, z: np.zeros((x.shape[1], 2)), order, domain)
        mesh_pts = model.get_collocation_points()
        assert np.allclose(
            model.mesh.integrate(np.sum(mesh_pts**2, axis=0)[:, None]), 2./3.)

        order = 4
        model = SteadyStateAdvectionDiffusionEquation2D()
        domain = [-1, 1, -1, 1]
        bndry_conds = [[lambda x: x[1:2, :].T*0, "D"],
                       [lambda x: x[1:2, :].T*0, "D"],
                       [lambda x: x[0:1, :].T*0, "D"],
                       [lambda x: x[0:1, :].T*0, "D"]]
        model.initialize(
            bndry_conds, lambda x, z: np.ones((x.shape[1], 1)),
            lambda x, z: np.zeros((x.shape[1], 1)),
            lambda x, z: np.zeros((x.shape[1], 1)), order, domain)
        mesh_pts = model.get_collocation_points()
        assert np.allclose(
            model.mesh.integrate(np.sum(mesh_pts**2, axis=0)[:, None]), 8./3.)

    def test_evaluate_gradient_2d(self):
        """
        for the PDE -((1+sum(z^2)*x)*u(x)')' = 2, u(0) = 0, u(1) = 1
        use model.evaluate_gradient to evaluate the gradient of the QoI
        with respect to the random parameter vector z.
        The QoI is the intergral of the solution over the entire domain
        The adjoint rhs is then just 1.
        """
        order = 20
        model = SteadyStateAdvectionDiffusionEquation2D()
        domain = [0, 1, 0, 1]
        bndry_conds = [[lambda x: x[1:2, :].T*0, "D"],
                       [lambda x: x[1:2, :].T*0, "D"],
                       [lambda x: x[0:1, :].T*0, "D"],
                       [lambda x: x[0:1, :].T*0, "D"]]
        model.initialize(
            bndry_conds,
            lambda x, z: ((z[0]**2+z[1]**2)*(x[0]+x[1]) + 1.)[:, None],
            lambda x, z: 0*x[:1].T+2,
            lambda x, z: np.zeros((x.shape[1], 2)), order, domain)

        sample = np.random.RandomState(2).uniform(-1, 1, (2, 1))
        model.diffusivity_derivs_fun = \
            lambda x, z, i: (2.*(x[0]+x[1])*z[i])[:, None]
        model.forcing_derivs_fun = \
            lambda x, z, i: np.zeros((x.shape[1], 1))
        model.advection_derivs_fun = \
            lambda x, z, i: np.zeros((x.shape[1], 2))
        model(sample)
        # evaluate_gradient has to be called before any more calls to
        # model.solve with different parameters, because we need to
        # access self.fwd_solution, which will change with any subsuquent calls
        errors = check_gradients(
            model, lambda x: model.evaluate_gradient(x[:, 0]), sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 4e-6


if __name__ == "__main__":
    spectral_diffusion_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSpectralDiffusion2D)
    unittest.TextTestRunner(verbosity=2).run(spectral_diffusion_test_suite)
