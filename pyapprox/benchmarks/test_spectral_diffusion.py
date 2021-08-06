import numpy as np
import unittest

from pyapprox.benchmarks.spectral_diffusion import (
    kronecker_product_2d, chebyshev_derivative_matrix,
    SteadyStateDiffusionEquation2D, SteadyStateDiffusionEquation1D
)
from pyapprox.univariate_polynomials.quadrature import gauss_jacobi_pts_wts_1D
import pyapprox as pya


class TestSpectralDiffusion2D(unittest.TestCase):
    def setUp(self):
        self.eps = 2 * np.finfo(np.float).eps

    def test_derivative_matrix(self):
        order = 4
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0., 0.0]
        xlim = [-1, 1]
        model.initialize(order, bndry_cond, xlim)
        derivative_matrix = model.get_derivative_matrix()
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

    def test_homogeneous_possion_equation(self):
        """
        solve u(x)'' = 0, u(0) = 0, u(1) = 0.5
        """

        order = 4
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.5]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)
        mesh_pts = model.get_collocation_points()

        diff_vals = 0*mesh_pts.squeeze()+1
        forcing_vals = 0*mesh_pts.squeeze()
        solution = model.solve(diff_vals, forcing_vals)
        def exact_sol(x): return 0.5*x
        assert np.linalg.norm(exact_sol(mesh_pts.squeeze())-solution) < 20*self.eps

    def test_inhomogeneous_possion_equation(self):
        """
        solve u(x)'' = -1, u(0) = 0, u(1) = 1
        solution u(x) =  -0.5*(x-3.)*x
        """
        order = 4
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 1.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)
        mesh_pts = model.get_collocation_points()
        diff_vals = 0*mesh_pts.squeeze()+1
        forcing_vals = 0*mesh_pts.squeeze()-1
        solution = model.solve(diff_vals, forcing_vals)
        def exact_sol(x): return -0.5*(x-3.)*x
        assert np.linalg.norm(
            exact_sol(mesh_pts.squeeze())-solution) < 30*self.eps

    def test_inhomogeneous_diffusion_equation_with_variable_coefficient(self):
        """
        solve ((1+x)*u(x)')' = -1, u(0) = 0, u(1) = 0
        solution u(x) = log(x+1)/log(2) - x
        """
        order = 20
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)
        mesh_pts = model.get_collocation_points()
        def diffusivity_function(x): return x + 1
        diff_vals = diffusivity_function(mesh_pts.squeeze())
        forcing_vals = 0*mesh_pts.squeeze()-1
        solution = model.solve(diff_vals, forcing_vals)
        def exact_sol(x): return np.log(x+1.) / np.log(2.) - x
        assert np.linalg.norm(exact_sol(mesh_pts.squeeze())-solution) < 3e-13

    def test_integrate(self):
        order = 4
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)
        mesh_pts = model.get_collocation_points()
        assert np.allclose(model.integrate(mesh_pts.T**2), 1./3.)
        assert np.allclose(model.integrate(mesh_pts.T**3), 1./4.)

        order = 4
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [-1, 1]
        model.initialize(order, bndry_cond, xlim)
        mesh_pts = model.get_collocation_points()
        assert np.allclose(model.integrate(mesh_pts.T**2), 2./3.)
        assert np.allclose(model.integrate(mesh_pts.T**3), 0.)

    def test_evaluate(self):
        """
        for the PDE ((1+z*x)*u(x)')' = -1, u(0) = 0, u(1) = 0
        use model.evaluate to extract QoI
        """
        order = 20
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)

        model.diffusivity_function = lambda x, z: z*x + 1.
        model.forcing_function = lambda x, z: 0*x-1
        qoi_coords = np.array([0.05, 0.5, 0.95])
        model.qoi_functional = lambda x: model.interpolate(x, qoi_coords)[:, 0]

        sample = np.ones((1, 1), float)
        qoi = model(sample)
        assert np.allclose(np.log(qoi_coords+1.)/np.log(2.)-qoi_coords, qoi)

        sample = 0.5*np.ones((1, 1), float)
        qoi = model(sample)
        assert np.allclose(
            -(qoi_coords*np.log(9./4.)-2.*np.log(qoi_coords+2.) +
              np.log(4.))/np.log(3./2.), qoi)

    def test_evaluate_gradient(self):
        """
        for the PDE ((1+sum(z^2)*x)*u(x)')' = -2, u(0) = 0, u(1) = 1
        use model.evaluate_gradient to evaluate the gradient of the QoI
        with respect to the random parameter vector z.
        The QoI is the intergral of the solution over the entire domain
        The adjoint rhs is then just 1.
        """
        order = 20
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)

        model.diffusivity_function = lambda x, z: (z[0]**2+z[1]**2)*x + 1.
        model.forcing_function = lambda x, z: 0*x-2
        model.qoi_coords = np.array([0.05, 0.5, 0.95])

        sample = np.random.RandomState(2).uniform(-1, 1, (2, 1))
        model.diffusivity_derivs_function = \
            lambda x, z, i: np.array([2.*x*z[i]]).T
        model.forcing_derivs_function = \
            lambda x, z, i: np.array([0.*x]).T
        model(sample)
        # evaluate_gradient has to be called before any more calls to
        # model.solve with different parameters, because we need to
        # access self.fwd_solution, which will change with any subsuquent calls
        errors = pya.check_gradients(
            model, lambda x: model.evaluate_gradient(x[:, 0]), sample)
        errors = errors[np.isfinite(errors)]
        assert errors.max() > 0.1 and errors.min() <= 6e-7

    @unittest.skip("Not fully implemented")
    def test_compute_error_estimate(self):
        """
        for the PDE ((1+z*x)*u(x)')' = -1, u(0) = 0, u(1) = 0
        use model.compute_error_estomate to compute an error estimate of
        the deterministic error in the foward solution.
        The QoI is the intergral of the solution over the entire domain
        The adjoint rhs is then just 1.
        """
        order = 5
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)

        model.diffusivity_function = lambda x, z: z[0]*x + 1.
        model.forcing_function = lambda x, z: 0.*x-1.

        sample = np.ones((1, 1), float)
        qoi = model(sample)
        error_estimate = model.compute_error_estimate(sample)

        solution = model.run(sample[:, 0])
        def exact_solution(x): return np.log(x+1.)/np.log(2.)-x
        gl_pts, gl_wts = gauss_jacobi_pts_wts_1D(50, 0, 0)
        x_range = model.xlim[1]-model.xlim[0]
        gl_pts = x_range*(gl_pts+1.)/2.+model.xlim[0]
        gl_wts *= x_range
        gl_vals = exact_solution(gl_pts)
        exact_qoi = np.dot(gl_vals, gl_wts)

        exact_error = abs(exact_qoi-qoi)

        print('err estimate', error_estimate)
        print('exact err', exact_error)
        print('effectivity ratio', error_estimate / exact_error)
        # should be very close to 1. As adjoint order is increased
        # it will converge to 1

        sample = 0.5*np.ones((1), float)
        qoi = model.evaluate(sample)
        exact_solution = -(model.mesh_pts*np.log(9./4.) -
                           2.*np.log(model.mesh_pts+2.) +
                           np.log(4.))/np.log(3./2.)
        exact_qoi = model.qoi_functional(exact_solution)
        error = abs(exact_qoi-qoi)
        error_estimate = model.compute_error_estimate(sample)

        print(error_estimate, error)
        # print model.integrate( (exact_solution - solution )**2 )
        assert np.allclose(error_estimate, error)

    def test_timestepping_without_forcing(self):
        r"""
        solve u_t(x,t) = u_xx(x,t), u(-1,t) = 0, u(1,t) = 0,
        u(x,0) = \sin(\pi*x)

        Exact solution
        u(x,t) = \exp(-\pi^2t)*sin(\pi*x)
        """
        order = 16
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [-1, 1]
        model.initialize(order, bndry_cond, xlim)
        model.diffusivity_function = lambda x, z: 0*x + 1.
        model.forcing_function = lambda x, t, z: 0*x
        sample = np.ones((1), float)  # dummy argument for this example
        model.num_time_steps = 1000
        model.initial_sol = np.sin(np.pi*model.mesh_pts)
        model.time_step_size = 1e-4
        model.time_step_method = 'adams-moulton-3'
        # model.time_step_method = 'crank-nicholson'
        model.time_step_method = 'backward-euler'
        model.num_stored_timesteps = 100
        solution = model.transient_solve(sample)
        def exact_sol(x, t): return np.exp(-np.pi**2*t)*np.sin(np.pi*x)
        test_mesh_pts = np.linspace(xlim[0], xlim[1], 100)
        plot = False  # True
        for i, t in enumerate(model.times):
            if plot:
                exact_sol_t = exact_sol(test_mesh_pts, t)
                model_sol_t = model.interpolate(solution[:, i], test_mesh_pts)
                pya.plt.plot(test_mesh_pts, model_sol_t, 'k',
                             label='collocation', linewidth=2)
                pya.plt.plot(test_mesh_pts, exact_sol_t,
                             'r--', label='exact', linewidth=2)
                pya.plt.legend(loc=0)
                pya.plt.title('$t=%1.2f$' % t)
                pya.plt.show()
            L2_error = np.sqrt(model.integrate(
                (exact_sol(model.mesh_pts, t)-solution[:, i])**2))
            factor = np.sqrt(
                model.integrate(exact_sol(model.mesh_pts, t)**2))
            # print L2_error, 1e-3*factor
            assert L2_error < 1e-3*factor

    def test_timestepping_with_time_independent_forcing(self):
        r"""
        solve u_t(x,t) = u_xx(x,t)+sin(3\pi x), u(0,t) = 0, u(1,t) = 0,
        u(x,0) = 5\sin(2\pi x)+2\sin(3\pi x)

        Exact solution
        u(x,t) = 5\exp(-4\pi^2t)*sin(2\pi*x)+(2\exp(-9\pi^2t)+(1-\exp(-9\pi^2t))/(9\pi^2))*\sin(3\pi x)
        """
        order = 32
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)
        model.diffusivity_function = lambda x, z: 0*x + 1.
        model.forcing_function = lambda x, t, z: np.sin(3*np.pi*x)
        sample = np.ones((1), float)  # dummy argument for this example
        model.num_time_steps = 10000
        model.initial_sol = 5*np.sin(2*np.pi*model.mesh_pts) + \
            2*np.sin(3*np.pi*model.mesh_pts)
        model.time_step_size = 1e-4
        # model.time_step_method = 'adams-moulton-3'
        model.time_step_method = 'crank-nicholson'
        # model.time_step_method = 'backward-euler'
        model.num_stored_timesteps = 100
        solution = model.transient_solve(sample)
        def exact_sol(x, t): return 5.*np.exp(-4.*np.pi**2*t)*np.sin(2.*np.pi*x) + \
            (2.*np.exp(-9.*np.pi**2*t)+(1.-np.exp(-9.*np.pi**2*t))/(9.*np.pi**2))*np.sin(3.*np.pi*x)
        # test_mesh_pts = np.linspace(xlim[0], xlim[1], 100)
        for i, t in enumerate(model.times):
            # exact_sol_t = exact_sol(test_mesh_pts,t)
            # model_sol_t = model.interpolate(solution[:,i],test_mesh_pts)
            # pya.plt.plot(test_mesh_pts,model_sol_t,'k',label='collocation',linewidth=2)
            # pya.plt.plot(test_mesh_pts,exact_sol_t,'r--',label='exact',linewidth=2)
            # pya.plt.legend(loc=0)
            # pya.plt.title('$t=%1.2f$'%t)
            # pya.plt.show()
            L2_error = np.sqrt(model.integrate(
                (exact_sol(model.mesh_pts, t)-solution[:, i])**2))
            factor = np.sqrt(
                model.integrate(exact_sol(model.mesh_pts, t)**2))
            # print(L2_error, 1e-4*factor)
            assert L2_error < 1e-4*factor

    def test_timestepping_with_time_dependent_forcing(self):
        r"""
        solve u_t(x,t) = u_xx(x,t)+np.sin(3\pi x)*np.sin(t), u(0,t) = 0, u(1,t) = 0,
        u(x,0) = 5sin(2\pi x)+2sin(3\pi x)

        Exact solution
        u(x,t) = 5\exp(-4\pi^2t)*np.sin(2\pi*x)+(2\exp(-9\pi^2t)+\exp(-9\pi^2t)(9\pi^2sin(t)-cos(t)+\exp(-9\pi^2t))/(1+81\pi^4))*sin(3\pi x)
        """
        order = 32
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)
        model.diffusivity_function = lambda x, z: 0*x + 1.
        model.forcing_function = lambda x, t, z: np.sin(3*np.pi*x)*np.sin(t)
        sample = np.ones((1), float)  # dummy argument for this example
        model.num_time_steps = int(1e4)
        model.initial_sol = 5*np.sin(2*np.pi*model.mesh_pts) + \
            2*np.sin(3*np.pi*model.mesh_pts)
        model.time_step_size = 1e-4
        model.num_stored_timesteps = 100
        # model.time_step_method = 'adams-moulton-3'
        model.time_step_method = 'crank-nicholson'
        # model.time_step_method = 'backward-euler'
        # model.time_step_method = 'RK4'
        solution = model.transient_solve(sample)

        def exact_sol(x, t): return 5.*np.exp(
                -4.*np.pi**2*t)*np.sin(2.*np.pi*x)+(
                    2.*np.exp(-9.*np.pi**2*t)+(
                        9.*np.pi**2*np.sin(t)-np.cos(t) +
                        np.exp(-9.*np.pi**2*t))/(1+81.*np.pi**4))*np.sin(
                            3.*np.pi*x)
        test_mesh_pts = np.linspace(xlim[0], xlim[1], 100)
        plot = False
        for i, t in enumerate(model.times):
            if plot:
                exact_sol_t = exact_sol(test_mesh_pts, t)
                model_sol_t = model.interpolate(solution[:, i], test_mesh_pts)
                pya.plt.plot(test_mesh_pts, model_sol_t, 'k',
                             label='collocation', linewidth=2)
                pya.plt.plot(test_mesh_pts, exact_sol_t, 'r--', label='exact',
                             linewidth=2)
                pya.plt.legend(loc=0)
                pya.plt.title('$t=%1.3f$' % t)
                pya.plt.show()
            L2_error = np.sqrt(model.integrate(
                (exact_sol(model.mesh_pts, t)-solution[:, i])**2))
            factor = np.sqrt(
                model.integrate(exact_sol(model.mesh_pts, t)**2))
            # print(L2_error, 1e-4*factor)
            assert L2_error < 1e-4*factor
            # print('time %1.2e: L2 error %1.2e' % (t, L2_error))

    def test_convergence(self):
        order = 8  # 1e-5
        # order = 16 #1e-11
        order = 20  # 2e-15
        model = SteadyStateDiffusionEquation1D()
        bndry_cond = [0.0, 0.0]
        xlim = [0, 1]
        model.initialize(order, bndry_cond, xlim)
        model.diffusivity_function = lambda x, z: 0*x + 1.
        model.forcing_function = lambda x, t, z: np.sin(3*np.pi*x)*np.sin(t)
        sample = np.ones((1), float)  # dummy argument for this example
        model.initial_sol = 5*np.sin(2*np.pi*model.mesh_pts) + \
            2*np.sin(3*np.pi*model.mesh_pts)
        final_time = 1.
        model.time_step_size = 1e-2
        model.num_stored_timesteps = 1
        # model.time_step_method = 'crank-nicholson'
        # model.time_step_method = 'backward-euler'
        # model.time_step_method = 'RK4' needs bug fixes and testing

        def exact_sol(x, t): return 5.*np.exp(
                -4.*np.pi**2*t)*np.sin(2.*np.pi*x)+(2.*np.exp(-9.*np.pi**2*t) + (
                    9.*np.pi**2*np.sin(t)-np.cos(t)+np.exp(-9.*np.pi**2*t))/(1+81.*np.pi**4))*np.sin(3.*np.pi*x)
        # test_mesh_pts = np.linspace(xlim[0], xlim[1], 1000)
        num_convergence_steps = 4
        errors = np.empty((num_convergence_steps), float)
        time_step_sizes = np.empty((num_convergence_steps), float)
        num_time_steps = np.empty((num_convergence_steps), float)
        for i in range(num_convergence_steps):
            model.num_time_steps = int(
                np.ceil(final_time/model.time_step_size))
            solution = model.transient_solve(sample)
            assert np.allclose(model.times[0], final_time, atol=1e-15)
            L2_error = np.sqrt(model.integrate(
                (exact_sol(model.mesh_pts, final_time)-solution[:, 0])**2))
            # interpolated_sol = model.interpolate(exact_sol(model.mesh_pts,final_time),test_mesh_pts)
            # print(np.linalg.norm(exact_sol(test_mesh_pts,final_time)-interpolated_sol)/np.sqrt(interpolated_sol.shape[0]))
            # print(model.num_time_steps, L2_error)
            errors[i] = L2_error
            time_step_sizes[i] = model.time_step_size
            num_time_steps[i] = model.num_time_steps
            model.time_step_size /= 2
        # print(errors)
        conv_rate = -np.log10(errors[-1]/errors[0])/np.log10(
            num_time_steps[-1]/num_time_steps[0])
        assert np.allclose(conv_rate, 2, atol=1e-4)
        # pya.plt.loglog(
        #     num_time_steps, errors, 'o-r',
        #     label=r'$\lVert u(x,T)-\hat{u}(x,T)\\rVert_{\ell_2(D)}$',
        #     linewidth=2)
        # # print errors[0]*num_time_steps[0]/num_time_steps
        # order = 1
        # pya.plt.loglog(
        #     num_time_steps,
        #     errors[0]*num_time_steps[0]**order/num_time_steps**order,
        #     'o--', label=r'$(\Delta t)^{-%d}$' % order, linewidth=2)
        # order = 2
        # pya.plt.loglog(
        #     num_time_steps,
        #     errors[0]*num_time_steps[0]**order/num_time_steps**order,
        #     'o--', label=r'$(\Delta t)^{-%d}$' % order, linewidth=2)
        # pya.plt.legend(loc=0)
        # pya.plt.show()

    def test_inhomogeneous_diffusion_equation_2d_variable_coefficient(self):
        """
        wolfram alpha z random variable x and w are spatial dimension
        d/dx 16*exp(-z^2)*(x^2-1/4)*(w^2-1/4)
        d/dx (1+t/pi^2*z*cos(pi/2*(x^2+w^2)))*32*(w^2-1/4)*x*exp(-z^2)
        Peter zaspels thesis is wrong it is 1 = sigma * not 1 + sigma +
        """

        sigma = 1
        num_dims = 1

        order = 16
        model = SteadyStateDiffusionEquation2D()
        lims = [-0.5, 0.5, -0.5, 0.5]
        bndry_cond = [0., 0.]
        model.initialize(order, bndry_cond, lims)

        def forcing_function(x, y): return \
            32.*(1.+sigma*y[0]*sigma*np.cos(np.pi/2.*(x[0, :]**2+x[1, :]**2))/np.pi**2) * \
            np.exp(-y[0]**2)*(x[0, :]**2+x[1, :]**2-0.5) -\
            32./np.pi*y[0]*sigma*np.sin(np.pi/2.*(x[0, :]**2+x[1, :]**2)) *\
            (x[0, :]**2 * np.exp(-y[0]**2)*(x[1, :]**2-0.25)+x[1, :]**2 *
             np.exp(-y[0]**2)*(x[0, :]**2-0.25))

        def diffusivity_function(x, y):
            return 1.+sigma/np.pi**2*y[0]*np.cos(
                np.pi/2.*(x[0, :]**2+x[1, :]**2))

        # only well posed if |y| < pi^2/sigma
        def exact_sol(x, y): return 16.*np.exp(-y**2) * \
            (x[0, :]**2-0.25)*(x[1, :]**2-0.25)

        rng = np.random.RandomState(1)
        sample = rng.uniform(-np.sqrt(3), np.sqrt(3), (num_dims))
        mesh_pts = model.get_collocation_points()
        diff_vals = diffusivity_function(mesh_pts, sample)
        forcing_vals = forcing_function(mesh_pts, sample)
        solution = model.solve(diff_vals, forcing_vals)
        # print np.linalg.norm(exact_sol( mesh_pts, sample )- solution )
        assert np.linalg.norm(exact_sol(mesh_pts, sample) - solution) < 2.e-12

    def test_2d_matlab_example(self):
        """
        Example from Spectral methods in Matlab. Specifically program 16 on page
        70 (90 PDF page number)

        Solve Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary
        and forcing 10*np.sin(8*xx.*(yy-1))

        true_solution at (xx,yy)=(1/np.sqrt(2),1/np.sqrt(2))= 0.32071594511
        """
        num_dims = 10
        order = 24
        model = SteadyStateDiffusionEquation2D()
        lims = [-1, 1, -1, 1]
        bndry_cond = [0., 0.]
        model.initialize(order, bndry_cond, lims)
        def diffusivity(x, y): return np.ones(x.shape[1])
        def forcing(x, y): return 10.*np.sin(8.*(x[0, :])*(x[1, :]-1))
        rng = np.random.RandomState(1)
        sample = rng.uniform(-1, 1., (num_dims))
        mesh_pts = model.get_collocation_points()
        diff_vals = diffusivity(mesh_pts, sample)
        forcing_vals = forcing(mesh_pts, sample)
        solution = model.solve(diff_vals, forcing_vals)

        # because I used reverse order of chebyshev points
        # and thus negative sign
        # of derivative matrix the solution returned here will have different
        # order to matlab which can be obtained by applying flipud(fliplr(x)),
        # e.g. we can obtain the correct coordinates used in the example with
        # index = np.arange((order+1)**2).reshape(
        #     (order+1, order+1))[3*order//4, 3*order//4]
        # print(mesh_pts[:, index])
        eval_samples = np.array([[1./np.sqrt(2), 1./np.sqrt(2)]]).T
        qoi = model.interpolate(solution, eval_samples)
        assert np.allclose(qoi, 0.32071594511)


if __name__ == "__main__":
    spectral_diffusion_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSpectralDiffusion2D)
    unittest.TextTestRunner(verbosity=2).run(spectral_diffusion_test_suite)
