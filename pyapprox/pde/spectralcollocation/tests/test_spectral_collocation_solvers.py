import unittest
import numpy as np
import sympy as sp
from functools import partial

from pyapprox.pde.spectralcollocation.stokes import StokesFlowModel
from pyapprox.pde.spectralcollocation.diffusion import (
    SteadyStateAdvectionDiffusionEquation1D,
    SteadyStateAdvectionDiffusionEquation2D,
    TransientAdvectionDiffusionEquation1D,
    TransientAdvectionDiffusionEquation2D,
    SteadyStateAdvectionDiffusionReaction)
from pyapprox.pde.tests.manufactured_solutions import (
    setup_steady_advection_diffusion_manufactured_solution,
    setup_steady_advection_diffusion_reaction_manufactured_solution,
    setup_steady_stokes_manufactured_solution
)
from pyapprox.util.utilities import check_gradients, approx_jacobian
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D


class TestSolvers(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_diffusion_integrate_1d(self):
        order = 4
        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: x.T+1,
                         lambda x, z: x.T*0+1,
                         lambda x, z: x.T*0, order, domain)
        mesh_pts = model.get_collocation_points()
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
        assert np.allclose(model.mesh.integrate(mesh_pts.T**2), 2./3.)
        assert np.allclose(model.mesh.integrate(mesh_pts.T**3), 0.)

    def test_diffusion_evaluate(self):
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

    def test_advection_diffusion_reaction_residual(self):
        # check finite difference of residual gives collocation matrix
        order = 4
        model = SteadyStateAdvectionDiffusionReaction()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: z*x.T + 1.,
                         lambda x, z:  0*x.T+1,
                         lambda x, z: x.T*0,
                         lambda x, z: x.T*0,
                         order, domain)
        model1 = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds1 = [[lambda x: x.T*0, "D"],
                        [lambda x: x.T*0, "D"]]
        model1.initialize(bndry_conds1, lambda x, z: z*x.T + 1.,
                          lambda x, z:  0*x.T+1,
                          lambda x, z: x.T*0,
                          order, domain)
        sample = np.ones((1), float)
        sol1 = model1.solve(sample)
        # sol = model.solve(sample, initial_guess=sol1)
        residual = model._residual(sol1, sample)
        assert np.allclose(residual, np.zeros_like(residual))

        guess = np.ones((model.mesh.mesh_pts.shape[1], 1))
        residual = model._residual(guess, sample)
        residual_fun = partial(model._residual, sample=sample)
        jac_fd = approx_jacobian(residual_fun, guess)
        jac = model1.form_collocation_matrix(
            model.diffusivity_fun(model.mesh.mesh_pts, sample[:, None]),
            model.velocity_fun(model.mesh.mesh_pts, sample[:, None]))
        jac = model1.mesh._apply_boundary_conditions_to_matrix(jac)
        assert np.allclose(jac, jac_fd)
        jac_ad = model._compute_jacobian(residual_fun, guess)
        assert np.allclose(jac, jac_ad, rtol=1e-15)

        order = 4
        model = SteadyStateAdvectionDiffusionReaction()
        bndry_conds = [[lambda x: x.T*0, "D"],
                       [lambda x: x.T*0, "D"]]
        domain = [0, 1]
        model.initialize(bndry_conds, lambda x, z: z*x.T + 1.,
                         lambda x, z:  0*x.T+1,
                         lambda x, z: x.T*0,
                         lambda x, z: x.T*0,
                         order, domain)
        model1 = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds1 = [[lambda x: x.T*0, "D"],
                        [lambda x: x.T*0, "D"]]
        model1.initialize(bndry_conds1, lambda x, z: z*x.T + 1.,
                          lambda x, z:  0*x.T+1,
                          lambda x, z: x.T*0,
                          order, domain)
        sample = np.ones((1), float)
        sol1 = model1.solve(sample)
        residual = model._residual(sol1, sample)
        assert np.allclose(residual, np.zeros_like(residual))

        guess = np.ones((model.mesh.mesh_pts.shape[1], 1))
        residual = model._residual(guess, sample)
        jac_fd = approx_jacobian(
            partial(model._residual, sample=sample), guess)
        jac = model.form_collocation_matrix(
            model.diffusivity_fun(model.mesh.mesh_pts, sample[:, None]),
            model.velocity_fun(model.mesh.mesh_pts, sample[:, None]))
        jac = model.mesh._apply_boundary_conditions_to_matrix(jac)
        assert np.allclose(jac, jac_fd)
        jac_ad = model._compute_jacobian(residual_fun, guess)
        # print(jac)
        # print(jac_ad)
        assert np.allclose(jac, jac_ad)

    def check_advection_diffusion(self, domain_bounds, orders, sol_string,
                                  diff_string, vel_strings, bndry_types):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_steady_advection_diffusion_manufactured_solution(
                sol_string, diff_string, vel_strings))

        def normal_flux(flux_funs, active_var, sign, xx, sample):
            vals = sign*flux_funs(xx, sample)[:, active_var:active_var+1]
            return vals

        def robin_bndry_fun(sol_fun, flux_funs, active_var, sign, alpha,
                            xx, sample):
            vals = alpha*sol_fun(xx, sample) + normal_flux(
                flux_funs, active_var, sign, xx, sample)
            return vals

        sample = np.zeros((0))  # dummy

        nphys_vars = len(orders)
        bndry_conds = []
        for dd in range(2*nphys_vars):
            if bndry_types[dd] == "N":
                bndry_conds.append(
                    [partial(normal_flux, flux_funs, dd//2, (-1)**(dd+1),
                             sample=sample), "N"])
            elif bndry_types[dd] == "D":
                bndry_conds.append([lambda xx: sol_fun(xx, sample), "D"])
            elif bndry_types[dd] == "R":
                alpha = 1
                bndry_conds.append(
                    [partial(robin_bndry_fun, sol_fun, flux_funs, dd//2,
                             (-1)**(dd+1), alpha, sample=sample), "R", alpha])
                # warning use of lists and lambda like commented code below
                # causes error because of shallow pointer copies
                # bndry_conds.append(
                #     [lambda xx: alpha*sol_fun(xx, sample)+normal_flux(
                #         flux_funs, dd, -1, xx, sample), "R", alpha])

        model = SteadyStateAdvectionDiffusionEquation2D()
        model.initialize(
            bndry_conds, diff_fun, forc_fun, vel_fun,
            orders, domain_bounds)

        import time
        t0 = time.time()
        sol = model.solve(sample)
        print(t0-time.time(), 'sec')

        print(np.linalg.norm(
            sol_fun(model.mesh.mesh_pts, sample)-sol))
        assert np.linalg.norm(
            sol_fun(model.mesh.mesh_pts, sample)-sol) < 1e-9

        normals = model.mesh._get_bndry_normals(np.arange(nphys_vars*2))
        if nphys_vars == 2:
            assert np.allclose(
                normals, np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]))
        else:
            assert np.allclose(normals, np.array([[-1], [1]]))
        normal_fluxes = model.compute_bndry_fluxes(
            sol, np.arange(nphys_vars*2), sample)
        for ii, indices in enumerate(model.mesh.boundary_indices):
            assert np.allclose(
                np.array(normal_fluxes[ii]),
                flux_funs(model.mesh.mesh_pts[:, indices], sample).dot(
                    normals[ii]))

    def test_advection_diffusion(self):
        test_cases = [
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], ["D", "D"]],
            [[0, 1], [20], "0.5*(x-3)*x", "1", ["0"], ["D", "N"]],
            [[0, 1], [20], "0.5*(x-3)*x", "1", ["0"], ["N", "D"]],
            [[0, 1], [20], "0.5*(x-3)*x", "1", ["0"], ["R", "D"]],
            [[0, 1], [20], "log(x+1)/log(2)-x", "1+x", ["0"], ["D", "D"]],
            [[-1, 1], [20], "(exp(4*x)-4*exp(-4)*(x-1)-exp(4))/16", "1", ["0"],
             ["D", "N"]],
            [[0, .5, 0, 1], [14, 16], "y**2*sin(pi*x)", "1", ["0", "0"],
             ["D", "N", "N", "D"]],
            [[0, .5, 0, 1], [25, 25], "y**2*sin(pi*x)", "1", ["0", "0"],
             ["D", "R", "D", "D"]]
        ]
        for test_case in test_cases[-1:]:
            self.check_advection_diffusion(*test_case)

    def test_diffusion_evaluate_gradient_1d(self):
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
        model.velocity_derivs_fun = lambda x, z, i: 0.0*x.T
        model(sample)
        # evaluate_gradient has to be called before any more calls to
        # model.solve with different parameters, because we need to
        # access self.fwd_solution, which will change with any subsuquent calls
        errors = check_gradients(
            model, lambda x: model.evaluate_gradient(x[:, 0]), sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 8e-7

    def test_diffusion_evaluate_advection_gradient_1d(self):
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
        model.velocity_derivs_fun = lambda x, z, i: x.T*0+aa
        model(sample)
        # evaluate_gradient has to be called before any more calls to
        # model.solve with different parameters, because we need to
        # access self.fwd_solution, which will change with any subsuquent calls
        errors = check_gradients(
            model, lambda x: model.evaluate_gradient(x[:, 0]), sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 6e-7

    def test_diffusion_compute_error_estimate(self):
        """
        for the PDE -((1+z*x)*u(x)')' = 1, u(0) = 0, u(1) = 0
        use model.compute_error_estomate to compute an error estimate of
        the deterministic error in the foward solution.
        The QoI is the intergral of the solution over the entire domain
        The adjoint rhs is then just 1.
        """
        # ensure that degree used to solve forward problem results in inexact
        # solution but that degree (order+2) used to solve adjoint problem solves
        # the adjoint exactly. Then error estimate will be exact
        (domain_bounds, orders, sol_string, diff_string, vel_strings) = (
             [0, 1], [3], "0.5*(x-3)*x**3", "1", ["0"])
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_steady_advection_diffusion_manufactured_solution(
                sol_string, diff_string, vel_strings))

        sample = np.ones((0, 1), float) # dummy

        model = SteadyStateAdvectionDiffusionEquation1D()
        bndry_conds = [[partial(sol_fun, sample=sample), "D"],
                       [partial(sol_fun, sample=sample), "D"]]
        model.initialize(
            bndry_conds, diff_fun, forc_fun, vel_fun, orders, domain_bounds)
        from pyapprox.pde.spectralcollocation.spectral_collocation import (
            ones_fun_axis_0)
        # qoi must always be linear functional involving integration
        model.set_qoi_functional(model.mesh.integrate, ones_fun_axis_0)

        qoi = model(sample)
        error_estimate = model.compute_error_estimate(sample)

        # solution = model.solve(sample[:, 0])
        gl_pts, gl_wts = gauss_jacobi_pts_wts_1D(50, 0, 0)
        x_range = model.mesh.domain[1]-model.mesh.domain[0]
        gl_pts = x_range*(gl_pts+1.)/2.+model.mesh.domain[0]
        gl_wts *= x_range
        gl_vals = sol_fun(gl_pts[None, :], sample)
        exact_qoi = np.dot(gl_vals[:, 0], gl_wts)

        exact_error = abs(exact_qoi-qoi)

        print('err estimate', error_estimate)
        print('exact err', exact_error)
        print('effectivity ratio', error_estimate / exact_error)
        # should be very close to 1. As adjoint order is increased
        # it will converge to 1

        assert np.allclose(error_estimate, error_estimate)

    def test_diffusion_timestepping_without_forcing(self):
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

    def test_diffusion_timestepping_with_time_independent_forcing_1d(self):
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

    def test_diffusion_timestepping_with_time_dependent_forcing_1d(self):
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

    def test_diffusion_timestepping_with_time_dependent_forcing_2d(self):
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

    def test_diffusion_convergence(self):

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

    def test_diffusion_integrate_2d(self):
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

    def test_diffusion_evaluate_gradient_2d(self):
        """
        for the PDE -((1+sum(z^2)*x)*u(x)')' = 2, u(0) = 0, u(1) = 1
        use model.evaluate_gradient to evaluate the gradient of the QoI
        with respect to the random parameter vector z.
        The QoI is the intergral of the solution over the entire domain
        The adjoint rhs is then just 1.
        """
        order = 20
        model = SteadyStateAdvectionDiffusionEquation2D()
        domain = [0, 1, 0, .5]
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
        model.velocity_derivs_fun = \
            lambda x, z, i: np.zeros((x.shape[1], 2))
        model(sample)
        # evaluate_gradient has to be called before any more calls to
        # model.solve with different parameters, because we need to
        # access self.fwd_solution, which will change with any subsuquent calls
        errors = check_gradients(
            model, lambda x: model.evaluate_gradient(x[:, 0]), sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 4e-6

    def check_stokes_mms(self, domain_bounds, orders, vel_strings, pres_string):
        vel_fun, pres_fun, forc_fun, pres_grad_fun = (
            setup_steady_stokes_manufactured_solution(
                vel_strings, pres_string))
        bndry_conds = [[lambda x: vel_fun(x), "D"],
                       [lambda x: vel_fun(x), "D"],
                       [lambda x: vel_fun(x), "D"],
                       [lambda x: vel_fun(x), "D"]]

        sample = np.zeros(0)  # dummy
        vel_fun = partial(vel_fun, sample=sample)
        pres_fun = partial(pres_fun, sample=sample)
        pres_grad_fun = partial(pres_grad_fun, sample=sample)

        nphys_vars = len(domain_bounds)//2
        bndry_conds = bndry_conds[:2*nphys_vars]

        model = StokesFlowModel()
        model.initialize(bndry_conds, orders, domain_bounds, forc_fun)

        assert np.allclose(model._mesh.mesh_pts[:, model._interior_indices],
                           model._pres_mesh.mesh_pts)

        exact_vel_vals = vel_fun(model._mesh.mesh_pts)
        exact_pres_vals = pres_fun(model._pres_mesh.mesh_pts)
        exact_sol_vals = np.vstack(exact_vel_vals+[exact_pres_vals])

        pres_idx = 0# model._pres_mesh.mesh_pts.shape[1]//2
        pres_val = exact_pres_vals[pres_idx]
        sol_vals = model.solve(sample, pres=(pres_idx, pres_val))

        print(sol_vals)

        # for dd in range(model._mesh.nphys_vars):
        #     assert np.allclose(
        #         model._pres_mesh.derivative_matrices[dd].dot(exact_pres_vals),
        #         pres_grad_fun(model._mesh.mesh_pts)[dd])

        # bndry_indices = np.hstack(model._mesh.boundary_indices)
        # recovered_forcing = model._split_quantities(
        #     model._matrix.dot(exact_sol_vals))
        # exact_forcing = forc_fun(model._mesh.mesh_pts, sample)
        # for dd in range(nphys_vars):
        #     assert np.allclose(sol_vals[dd][bndry_indices],
        #                        exact_vel_vals[dd][bndry_indices])
        #     assert np.allclose(recovered_forcing[dd][model._interior_indices],
        #                        exact_forcing[dd][model._interior_indices])
        #     assert np.allclose(exact_vel_vals[dd][bndry_indices],
        #                        recovered_forcing[dd][bndry_indices])

        # # check value used to enforce unique pressure is found correctly
        # assert np.allclose(
        #     sol_vals[nphys_vars][pres_idx], pres_val)
        # # check pressure at all but point used for enforcing unique value
        # # are set correctly
        # assert np.allclose(
        #     np.delete(recovered_forcing[nphys_vars], pres_idx),
        #     np.delete(
        #         exact_forcing[nphys_vars][model._interior_indices], pres_idx))

        # num_pts_1d = 50
        # plot_limits = domain_bounds
        # from pyapprox.util.visualization import get_meshgrid_samples, plt
        # X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d)
        # # Z = model.interpolate(sol_vals, pts)
        # Z = model.interpolate(exact_vel_vals+[exact_pres_vals], pts)

        # Z = [np.reshape(zz, (X.shape[0], X.shape[1])) for zz in Z]
        # fig, axs = plt.subplots(1, 4, figsize=(8*4, 6))
        # axs[0].quiver(X, Y, Z[0], Z[1])
        # from pyapprox.util.visualization import plot_2d_samples
        # for ii in range(3):
        #     if Z[ii].min() != Z[ii].max():
        #         pl = axs[ii+1].contourf(
        #             X, Y, Z[ii],
        #             levels=np.linspace(Z[ii].min(), Z[ii].max(), 40))
        #         # plot_2d_samples(
        #         #     model._mesh.mesh_pts, ax=axs[ii+1], c='r', marker='o')
        #         # plot_2d_samples(
        #         #     model._pres_mesh.mesh_pts, ax=axs[ii+1], c='k', marker='o')
        #         plt.colorbar(pl, ax=axs[ii+1])
        # plt.show()

        for exact_v, v in zip(exact_vel_vals, sol_vals[:-1]):
            print(exact_v[:, 0])
            print(v[:, 0])
            assert np.allclose(exact_v, v)

        print(exact_pres_vals-sol_vals[-1])
        assert np.allclose(exact_pres_vals, sol_vals[-1])

    def test_stokes_mms(self):
        test_cases = [
            # [[0, 1], [4], ["(1-x)**2"], "x**2"],
            [[0, 1, 0, 1], [20, 20],
              ["-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"], "x**2+y**2"],
            # [[0, 1, 0, 1], [6, 7],
            #  ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2"]
        ]
        for test_case in test_cases:
            self.check_stokes_mms(*test_case)

    def test_lid_driven_cavity_flow(self):
        # drive cavity using left boundary as top and bottom boundaries
        # do not hvae dof at the corners
        def bndry_condition(xx):
            cond = [(1*xx[0, :]**2*(1-xx[0, :]**2))[:, None],
                    np.zeros((xx.shape[1], 1))]
            return cond

        order = 20
        domain = [0, 1, 0, 1]
        bndry_conds = [
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"],
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"],
            [lambda x: [np.zeros((x.shape[1], 1)) for ii in range(2)], "D"],
            [bndry_condition, "D"]]

        def forcing_fun(xx, zz):
            return [np.zeros((xx.shape[1], 1)) for ii in range(3)]
            # split pressure into p = p_solve + np.exp(2*x+2*y)
            # to lessen singularity at top right corner
            # return [2*np.exp(2*xx[0, :]+2*xx[1, :])[:, None],
            #         2*np.exp(2*xx[0, :]+2*xx[1, :])[:, None],
            #         np.zeros((xx.shape[1], 1))]

        pres_idx = 0
        unique_pres_val = 0
        model = StokesFlowModel()
        model.initialize(bndry_conds, order, domain, forcing_fun)
        sample = np.zeros(0)  # dummy
        sol_vals = model.solve(sample, (pres_idx, unique_pres_val))

        rhs = model._split_quantities(model._rhs)
        assert np.allclose(
            sol_vals[0][model._mesh.boundary_indices[3], 0],
            rhs[0][model._mesh.boundary_indices[3], 0])

        # check value used to enforce unique pressure is found correctly
        assert np.allclose(
            sol_vals[model._mesh.nphys_vars][pres_idx], unique_pres_val)

        # num_pts_1d = 50
        # plot_limits = domain
        # from pyapprox.util.visualization import get_meshgrid_samples
        # X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d)
        # Z = model.interpolate(sol_vals, pts)

        # Z = [np.reshape(zz, (X.shape[0], X.shape[1])) for zz in Z]
        # fig, axs = plt.subplots(1, 4, figsize=(8*4, 6))
        # axs[0].quiver(X, Y, Z[0], Z[1])
        # from pyapprox.util.visualization import plot_2d_samples
        # for ii in range(3):
        #     if Z[ii].min() != Z[ii].max():
        #         pl = axs[ii+1].contourf(
        #             X, Y, Z[ii],
        #             levels=np.linspace(Z[ii].min(), Z[ii].max(), 40))
        #         # plot_2d_samples(
        #         #     model._mesh.mesh_pts, ax=axs[ii+1], c='r', marker='o')
        #         # plot_2d_samples(
        #         #     model._pres_mesh.mesh_pts, ax=axs[ii+1], c='k', marker='o')
        #         plt.colorbar(pl, ax=axs[ii+1])
        # # plt.show()

    def check_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string,
            diff_string, vel_strings, rate_string, bndry_types):
        sol_fun, diff_fun, vel_fun, forc_fun, rate_fun, flux_funs = (
            setup_steady_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, rate_string))

        def normal_flux(flux_funs, active_var, sign, xx, sample):
            vals = sign*flux_funs(xx, sample)[:, active_var:active_var+1]
            return vals

        def robin_bndry_fun(sol_fun, flux_funs, active_var, sign, alpha,
                            xx, sample):
            return alpha*sol_fun(xx, sample) + normal_flux(
                flux_funs, active_var, sign, xx, sample)

        sample = np.zeros((0))  # dummy

        nphys_vars = len(orders)
        bndry_conds = []
        for dd in range(nphys_vars):
            if bndry_types[2*dd] == "N":
                bndry_conds.append(
                    [partial(normal_flux, flux_funs, dd, -1, sample=sample),
                     "N"])
            elif bndry_types[2*dd] == "D":
                bndry_conds.append([lambda xx: sol_fun(xx, sample), "D"])
            elif bndry_types[2*dd] == "R":
                alpha = 1
                bndry_conds.append(
                    [partial(robin_bndry_fun, sol_fun, flux_funs, dd, -1, alpha,
                             sample=sample), "R", alpha])
                # warning use of lists and lambda like commented code below
                # causes error because of shallow pointer copies
                # bndry_conds.append(
                #     [lambda xx: alpha*sol_fun(xx, sample)+normal_flux(
                #         flux_funs, dd, -1, xx, sample), "R", alpha])
            if bndry_types[2*dd+1] == "N":
                bndry_conds.append(
                    [partial(normal_flux, flux_funs, dd, 1, sample=sample),
                     "N"])
            elif bndry_types[2*dd+1] == "D":
                bndry_conds.append([lambda xx: sol_fun(xx, sample), "D"])
            elif bndry_types[2*dd+1] == "R":
                alpha = 2
                bndry_conds.append(
                    [partial(robin_bndry_fun, sol_fun, flux_funs, dd, 1, alpha,
                             sample=sample), "R", alpha])

        model = SteadyStateAdvectionDiffusionReaction()
        model.initialize(
            bndry_conds, diff_fun, forc_fun, vel_fun, rate_fun,
            orders, domain_bounds)

        sol = model.solve(sample)

        print(np.linalg.norm(
            sol_fun(model.mesh.mesh_pts, sample)-sol))
        # print(sol[[0, -1]], sol_fun(model.mesh.mesh_pts, sample)[[0, -1]])
        # model.mesh.plot(sol)
        # model.mesh.plot(sol_fun(model.mesh.mesh_pts, sample))
        # import matplotlib.pyplot as plt
        # plt.show()
        assert np.linalg.norm(
            sol_fun(model.mesh.mesh_pts, sample)-sol) < 1e-7

        normals = model.mesh._get_bndry_normals(np.arange(nphys_vars*2))
        if nphys_vars == 2:
            assert np.allclose(
                normals, np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]))
        else:
            assert np.allclose(normals, np.array([[-1], [1]]))
        normal_fluxes = model.compute_bndry_fluxes(
            sol, np.arange(nphys_vars*2), sample)
        for ii, indices in enumerate(model.mesh.boundary_indices):
            assert np.allclose(
                np.array(normal_fluxes[ii]),
                flux_funs(model.mesh.mesh_pts[:, indices], sample).dot(
                    normals[ii]))

    def test_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [6], "0.5*(x-3)*x", "1", ["0"], "1", ["D", "D"]],
            [[0, 1], [6], "0.5*(x-3)*x", "1", ["0"], "1", ["D", "N"]],
            [[0, 1], [6], "0.5*(x-3)*x", "1", ["0"], "1", ["D", "R"]],
            [[0, .5, 0, 1], [14, 16], "y**2*sin(pi*x)", "1", ["0", "0"], "1",
             ["D", "N", "N", "D"]],
            [[0, .5, 0, 1], [16, 16], "y**2*sin(pi*x)", "1", ["0", "0"], "1",
             ["D", "R", "D", "D"]]
        ]
        for test_case in test_cases:
            self.check_advection_diffusion_reaction(*test_case)


if __name__ == "__main__":
    solvers_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSolvers)
    unittest.TextTestRunner(verbosity=2).run(solvers_test_suite)

#TODO clone conda environment when coding on branch
#diff spectral diffusion on master and on pde branch
#make sure that  _form_1d_derivative_matrices in this file is being called
#extract timestepper from advection diffusion and make its own object
#which just takes an steady state solver like diffusion or stokes and
#integrates it in time
