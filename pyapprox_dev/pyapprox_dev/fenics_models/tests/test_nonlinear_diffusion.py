import sys

from functools import partial
import unittest, pytest

import sympy as sp
import matplotlib.pyplot as plt


if sys.platform == 'win32':
    pytestmark = pytest.mark.skip("Skipping test on Windows")
    dl = None
    skiptest = unittest.skipIf(
        True, reason="fenics_adjoint package not available on Windows")
else:
    import dolfin as dl
    from pyapprox_dev.fenics_models.fenics_utilities import *
    from pyapprox_dev.fenics_models.nonlinear_diffusion import *
    from pyapprox_dev.fenics_models.tests.test_advection_diffusion import \
        get_exact_solution as get_advec_exact_solution, \
        get_forcing as get_advec_forcing

    dl.set_log_level(40)
    skiptest = unittest.skipIf(
        not has_dla, reason="fenics_adjoint package missing")


def quadratic_diffusion(u):
    """Nonlinear coefficient in the PDE."""
    return 1 + u**2


def get_quadratic_diffusion_exact_solution_sympy():
    # Use SymPy to compute f given manufactured solution u
    from sympy.abc import t
    x, y = sp.symbols('x[0] x[1]')
    u = sp.cos(np.pi*t)*sp.sin(np.pi*x)*sp.sin(np.pi*y)  # 1 + x + 2*y
    return u, x, y, t


def get_quadratic_diffusion_exact_solution(mesh, degree):
    u, x, y, t = get_quadratic_diffusion_exact_solution_sympy()
    exact_sol = dla.Expression(
        sp.printing.ccode(u), cell=mesh.ufl_cell(),
        domain=mesh, t=0, degree=degree)
    return exact_sol


def get_diffusion_forcing(q, sol_func, mesh, degree):
    u, x, y, t = sol_func()
    f = u.diff(t, 1)-sum(sp.diff(q(u)*sp.diff(u, xi), xi) for xi in (x, y))
    f = sp.simplify(f)
    forcing = dla.Expression(
        sp.printing.ccode(f), cell=mesh.ufl_cell(),
        domain=mesh, t=0, degree=degree)
    return forcing


class TestNonlinearDiffusion(unittest.TestCase):
    def setUp(self):
        pass

    def test_quadratic_diffusion_dirichlet_boundary_conditions(self):
        """
        du/dt = div((1+u**2)*grad(u))+f   in the unit square.
            u = u_D on the boundary.
        """
        nx, ny, degree = 21, 21, 2
        mesh = dla.RectangleMesh(dl.Point(0, 0), dl.Point(1, 1), nx, ny)
        function_space = dl.FunctionSpace(mesh, "Lagrange", degree)

        bndry_obj = get_2d_unit_square_mesh_boundaries()
        boundary_conditions = get_dirichlet_boundary_conditions_from_expression(
            get_quadratic_diffusion_exact_solution(mesh, degree), 0, 1, 0, 1)
        forcing = get_diffusion_forcing(
            quadratic_diffusion, get_quadratic_diffusion_exact_solution_sympy,
            mesh, degree)

        options = {'time_step': 0.05, 'final_time': 1,
                   'forcing': forcing,
                   'boundary_conditions': boundary_conditions,
                   'second_order_timestepping': True,
                   'init_condition': get_quadratic_diffusion_exact_solution(
                       mesh, degree), 'nonlinear_diffusion':
                   quadratic_diffusion}
        sol = run_model(function_space, **options)

        exact_sol = get_quadratic_diffusion_exact_solution(mesh, degree)
        exact_sol.t = options['final_time']
        error = dl.errornorm(exact_sol, sol, mesh=mesh)
        print('Abs. Error', error)
        assert error <= 8e-5

    def test_constant_diffusion_dirichlet_boundary_conditions(self):
        kappa = 3
        nx, ny, degree = 31, 31, 2
        mesh = dla.RectangleMesh(dl.Point(0, 0), dl.Point(1, 1), nx, ny)
        function_space = dl.FunctionSpace(mesh, "Lagrange", degree)

        def constant_diffusion(u):
            return dla.Constant(kappa)

        boundary_conditions = get_dirichlet_boundary_conditions_from_expression(
            get_advec_exact_solution(mesh, degree), 0, 1, 0, 1)

        nlsparam = dict()

        options = {'time_step': 0.05, 'final_time': 1,
                   'forcing': get_advec_forcing(kappa, mesh, degree),
                   'boundary_conditions': boundary_conditions,
                   'second_order_timestepping': True,
                   'init_condition': get_advec_exact_solution(mesh, degree),
                   'nlsparam': nlsparam,
                   'nonlinear_diffusion': constant_diffusion}
        sol = run_model(function_space, **options)

        exact_sol = get_advec_exact_solution(mesh, degree)
        exact_sol.t = options['final_time']
        error = dl.errornorm(exact_sol, sol, mesh=mesh)
        print('Abs. Error', error)
        assert error <= 1e-4

    @skiptest
    def test_adjoint(self):

        np_kappa = 2
        nx, ny, degree = 11, 11, 2
        mesh = dla.RectangleMesh(dl.Point(0, 0), dl.Point(1, 1), nx, ny)
        function_space = dl.FunctionSpace(mesh, "Lagrange", degree)

        def constant_diffusion(kappa, u):
            return kappa

        boundary_conditions = \
            get_dirichlet_boundary_conditions_from_expression(
                get_advec_exact_solution(mesh, degree), 0, 1, 0, 1)

        class NonlinearDiffusivity(object):
            def __init__(self, kappa):
                self.kappa = kappa

            def __call__(self, u):
                return (self.kappa+u**2)

        dl_kappa = dla.Constant(np_kappa)
        options = {'time_step': 0.05, 'final_time': 1.,
                   'forcing': dla.Constant(1),
                   'boundary_conditions': boundary_conditions,
                   'init_condition': get_advec_exact_solution(mesh, degree),
                   'nonlinear_diffusion': NonlinearDiffusivity(dl_kappa),
                   'second_order_timestepping': True,
                   'nlsparam': dict()}

        def dl_qoi_functional(sol):
            return dla.assemble(sol*dl.dx)

        def dl_fun(np_kappa):
            kappa = dla.Constant(np_kappa)
            options_copy = options.copy()
            options_copy['forcing'] = dla.Constant(1.0)
            # using class avoids pickling
            options_copy['nonlinear_diffusion'] = NonlinearDiffusivity(kappa)
            sol = run_model(function_space, **options_copy)
            return sol, kappa

        def fun(np_kappa):
            np_kappa = np_kappa[0, 0]
            sol, kappa = dl_fun(np_kappa)
            J = dl_qoi_functional(sol)
            control = dla.Control(kappa)
            dJd_kappa = dla.compute_gradient(J, [control])[0]
            return np.atleast_1d(float(J)), np.atleast_2d(float(dJd_kappa))

        sol, kappa = dl_fun(np_kappa)

        J = dl_qoi_functional(sol)
        control = dla.Control(kappa)
        Jhat = dla.ReducedFunctional(J, control)
        # h = dla.Constant(np.random.normal(0, 1, 1))
        # conv_rate = dla.taylor_test(Jhat, kappa, h)
        # assert np.allclose(conv_rate, 2.0, atol=1e-2)

        from pyapprox.optimization import check_gradients
        x0 = np.atleast_2d(np_kappa)
        errors = check_gradients(fun, True, x0)
        assert errors.min() < 1e-7 and errors.max() > 1e-1


class TestShallowIceEquation(unittest.TestCase):
    def setUp(self):
        pass

    def run_shallow_ice_halfar(self, nphys_dim):
        """
        See 'Exact time-dependent similarity solutions for isothermal shallow 
        ice sheets'
        https://pdfs.semanticscholar.org/5e57/ffc51586717cb4db33c1c20ebed54c3bfbfb.pdf

        Compute the similarity solution to the isothermal flat-bed SIA from
        Halfar (1983).  Constants H0 = 3600 m and R0 = 750 km are as in Test B in
        Bueler et al (2005).
        """

        nx, degree = 41, 1
        glen_exponent = 3
        Gamma = 2.8457136065980445e-05
        positivity_tol = 0  # 1e-6

        Lx = 1200e3

        if nphys_dim == 1:
            mesh = dla.IntervalMesh(nx, -Lx, Lx)

        elif nphys_dim == 2:
            ny, Ly = nx, Lx
            mesh = dla.RectangleMesh(
                dl.Point(-Lx, -Ly), dl.Point(Lx, Ly), nx, ny)
        function_space = dl.FunctionSpace(mesh, "Lagrange", degree)

        bed = None
        forcing = dla.Constant(0.0)
        exact_solution = get_halfar_shallow_ice_exact_solution(
            Gamma, mesh, degree, nphys_dim)
        if nphys_dim == 1:
            boundary_conditions = \
                get_1d_dirichlet_boundary_conditions_from_expression(
                    exact_solution, -Lx, Lx)
        elif nphys_dim == 2:
            boundary_conditions =\
                get_dirichlet_boundary_conditions_from_expression(
                    exact_solution, -Lx, Lx, -Ly, Ly)

        # exact solution is undefined at t=0 so set initial condition to
        # some later time
        secpera = 31556926  # seconds per anum
        exact_solution.t = 200*secpera

        nlsparams = get_default_snes_nlsparams()

        beta = None
        diffusion = partial(
            shallow_ice_diffusion, glen_exponent, Gamma, bed, positivity_tol,
            beta)
        options = {'time_step': 10*secpera, 'final_time': 600*secpera,
                   'forcing': forcing,
                   'boundary_conditions': boundary_conditions,
                   'second_order_timestepping': True,
                   'init_condition': exact_solution,
                   'nonlinear_diffusion': diffusion,
                   'nlsparam': nlsparams,
                   'positivity_tol': positivity_tol}
        sol = run_model(function_space, **options)

        # exact_solution.t=forcing.t+options['time_step']
        exact_solution.t = options['final_time']
        error = dl.errornorm(exact_solution, sol, mesh=mesh)
        print('Abs. Error', error)
        rel_error = error/dl.sqrt(
            dla.assemble(exact_solution**2*dl.dx(degree=5)))
        print('Rel. Error', rel_error)

        plot = False
        if plot and nphys_dim == 1:
            function_space = dl.FunctionSpace(mesh, "CG", degree)
            x = function_space.tabulate_dof_coordinates()
            indices = np.argsort(x)
            x = x[indices]
            values = sol.vector().get_local()
            values = values[indices]
            plt.plot(x, values)
            exact_values = dla.interpolate(
                exact_solution, function_space).vector().get_local()
            plt.plot(x, exact_values[indices])
            plt.show()
        elif plot and nphys_dim == 2:
            fig = plt.figure(figsize=(3*8, 6))
            ax = plt.subplot(1, 3, 1)
            pl = plot(sol, mesh=mesh)
            plt.colorbar(pl, ax=ax)
            ax = plt.subplot(1, 3, 2)
            pl = plot(exact_solution, mesh=mesh)
            plt.colorbar(pl, ax=ax)
            ax = plt.subplot(1, 3, 3)
            pl = plot(exact_solution-sol, mesh=mesh)
            plt.colorbar(pl, ax=ax)
            plt.show()

        assert rel_error <= 3e-4

    def test_shallow_ice_halfar_1d(self):
        self.run_shallow_ice_halfar(1)

    def test_shallow_ice_halfar_2d(self):
        self.run_shallow_ice_halfar(2)

    def test_halfar_model(self):
        nlsparams = get_default_snes_nlsparams()
        # nlsparams = get_default_newton_nlsparams()

        def dl_qoi_functional(sol):
            return dla.assemble(sol*dl.dx)

        def qoi_functional(sol):
            return np.atleast_1d(float(dl_qoi_functional(sol)))

        def qoi_functional_grad(sol, model):
            J = dl_qoi_functional(sol)
            control = dla.Control(model.shallow_ice_diffusivity.Gamma)
            dJd_gamma = dla.compute_gradient(J, [control])[0]
            # apply chain rule. we want gradient of qoi as a function of x
            # but fenics compute gradient with respect to g(x)=(1+x)*Gamma
            # dq/dx = dq/dg*dg/dx
            dJd_gamma *= model.shallow_ice_diffusivity.Gamma
            # h = dla.Constant(1e-5) # h must be similar magnitude to Gamma
            #Jhat = dla.ReducedFunctional(J, control)
            # conv_rate = dla.taylor_test(
            #    Jhat, model.shallow_ice_diffusivity.Gamma, h)
            return np.atleast_2d(float(dJd_gamma))

        if not has_dla:
            qoi_functional_grad = None

        secpera = 31556926  # seconds per anum
        init_time = 200*secpera
        final_time, degree, nphys_dim = 300*secpera, 1, 1  # 600*secpera, 1, 1
        model = HalfarShallowIceModel(
            nphys_dim, init_time, final_time, degree, qoi_functional,
            second_order_timestepping=True, nlsparams=nlsparams,
            qoi_functional_grad=qoi_functional_grad)

        # for nphys_dim=1 [8, 8] will produce error of 2.7 e-5
        # but stagnates for a while at around 1e-4 for values
        # 5, 6, 7
        random_sample = np.array([[0]]).T
        config_sample = np.array([[4]*nphys_dim + [4]]).T
        sample = np.vstack((random_sample, config_sample))
        sol = model.solve(sample)

        exact_solution = get_halfar_shallow_ice_exact_solution(
            model.Gamma, model.mesh, model.degree, model.nphys_dim)
        exact_solution.t = final_time
        error = dl.errornorm(exact_solution, sol, mesh=model.mesh)
        print('Abs. Error', error)
        rel_error = error/dl.sqrt(
            dla.assemble(exact_solution**2*dl.dx(degree=5)))
        print('Rel. Error', rel_error)

        assert rel_error < 1e-3

        if not has_dla:
            return
        # TODO: complete test qoi grad but first add taylor_test
        val, grad = model(sample, True)
        print(val, grad)

        from pyapprox.optimization import check_gradients
        from pyapprox.models.wrappers import SingleFidelityWrapper
        fun = SingleFidelityWrapper(
            partial(model, jac=True), config_sample[:, 0])
        x0 = np.atleast_2d(model.Gamma)
        errors = check_gradients(fun, True, x0, direction=np.atleast_2d(1))
        assert errors.min() < 3e-5 and errors.max() > 1e-1


if __name__ == "__main__":
    nonlinear_diffusion_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestNonlinearDiffusion)
    unittest.TextTestRunner(verbosity=2).run(nonlinear_diffusion_test_suite)

    shallow_ice_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestShallowIceEquation)
    unittest.TextTestRunner(verbosity=2).run(shallow_ice_test_suite)
