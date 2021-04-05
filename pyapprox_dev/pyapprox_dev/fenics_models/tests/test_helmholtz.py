import sys
import unittest, pytest
import pyapprox as pya


if pya.PYA_DEV_AVAILABLE:
    import dolfin as dl
    from pyapprox_dev.fenics_models.helmholtz import *
else:
    pytestmark = pytest.mark.skip("Skipping test on Windows")

    # Create stub class
    class dl(object):
        UserExpression = object

try:
    import mshr
    mshr_package_missing = False
except:
    mshr_package_missing = True

mshr_skiptest = unittest.skipIf(
    mshr_package_missing, reason="mshr package missing")


class ExactSolution(dl.UserExpression):
    def __init__(self, kappa, **kwargs):
        self.kappa = kappa
        if '2017' not in dl.__version__:
            # does not work for fenics 2017 only >=2018
            super().__init__(**kwargs)
        # in 2017 base class __init__ does not need to be called.

    def eval(self, values, x):
        vals = exact_solution(x)
        values[0] = vals[0, 0]
        values[1] = vals[0, 1]

    def value_shape(self):
        return (2,)


class Forcing(dl.UserExpression):
    def __init__(self, kappa, component, **kwargs):
        self.kappa = kappa
        self.component = component
        if '2017' not in dl.__version__:
            # does not work for fenics 2017 only >=2018
            super().__init__(**kwargs)
        # in 2017 base class __init__ does not need to be called.

    def eval(self, values, x):
        if self.component == 'real':
            vals = forcing_function_real(x, self.kappa)
        else:
            vals = forcing_function_imag(x, self.kappa)
        values[0] = vals

    def value_shape(self):
        return []


class ForcingVariableKappa(dl.UserExpression):
    def __init__(self, kappa1, kappa2, component, **kwargs):
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.component = component
        if '2017' not in dl.__version__:
            # does not work for fenics 2017 only >=2018
            super().__init__(**kwargs)
        # in 2017 base class __init__ does not need to be called.

    def eval(self, values, x):
        tol = 1e-14
        if x[1] <= 0.5+tol:
            kappa = self.kappa1
        else:
            kappa = self.kappa2

        if self.component == 'real':
            vals = forcing_function_real(x, kappa)
        else:
            vals = forcing_function_imag(x, kappa)
        values[0] = vals

    def value_shape(self):
        return []


class RobinBoundaryRHS(dl.UserExpression):
    def __init__(self, kappa, n, alpha, component, phys_var, **kwargs):
        """
        n = -1 or 1 specifies left or right boundary
        """
        self.kappa = kappa
        self.n = n
        self.alpha = alpha
        self.component = component
        self.phys_var = phys_var

        if '2017' not in dl.__version__:
            # does not work for fenics 2017 only >=2018
            super().__init__(**kwargs)
        # in 2017 base class __init__ does not need to be called.

    def eval(self, values, x):
        if self.component == 'real':
            if self.phys_var == 0:
                vals = robin_x_real(x, self.kappa, self.n, self.alpha)
            else:
                vals = robin_y_real(x, self.kappa, self.n, self.alpha)
        else:
            if self.phys_var == 0:
                vals = robin_x_imag(x, self.kappa, self.n, self.alpha)
            else:
                vals = robin_y_imag(x, self.kappa, self.n, self.alpha)
        values[0] = vals

    def value_shape(self):
        return []


def exact_solution_real(x):
    assert x.ndim == 2
    assert x.shape[0] == 2
    return np.sin(2*np.pi*x[0, :])*np.cos(2*np.pi*x[1, :]+np.pi/2)


def exact_solution_imag(x):
    assert x.ndim == 2
    assert x.shape[0] == 2
    return np.sin(np.pi*x[0, :])*np.cos(np.pi*x[1, :]+np.pi/2)


def exact_solution(x):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    assert x.ndim == 2
    assert x.shape[0] == 2
    # real and imaginary parts
    return np.array([exact_solution_real(x), exact_solution_imag(x)]).T


def exact_solution_magnitude(x):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    assert x.ndim == 2
    assert x.shape[0] == 2
    sol = exact_solution(x)
    magnitude = np.sqrt(sol[:, 0]**2+sol[:, 1]**2)
    return magnitude


def forcing_function_real(x, kappa):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    assert x.shape[0] == 2
    return (-8*np.pi**2*np.sin(2*np.pi*x[0, :])*np.cos(2*np.pi*x[1, :]+np.pi/2) +
            kappa**2*exact_solution_real(x))


def forcing_function_imag(x, kappa):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    assert x.shape[0] == 2
    return (-2*np.pi**2*np.sin(np.pi*x[0, :])*np.cos(np.pi*x[1, :]+np.pi/2) +
            kappa**2*exact_solution_imag(x))


def robin_x_real(x, kappa, n, alpha):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    assert x.shape[0] == 2
    assert n == 1 or n == -1
    return n*2*np.pi*np.cos(2*np.pi*x[0, :])*np.cos(2*np.pi*x[1, :]+np.pi/2) +\
        alpha*exact_solution_real(x)


def robin_x_imag(x, kappa, n, alpha):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    assert x.shape[0] == 2
    assert n == 1 or n == -1
    return n*np.pi*np.cos(np.pi*x[0, :])*np.cos(np.pi*x[1, :]+np.pi/2) +\
        alpha*exact_solution_imag(x)


def robin_y_real(x, kappa, n, alpha):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    assert x.shape[0] == 2
    assert n == 1 or n == -1
    return -n*2*np.pi*np.sin(2*np.pi*x[0, :])*np.sin(2*np.pi*x[1, :]+np.pi/2) +\
        alpha*exact_solution_real(x)


def robin_y_imag(x, kappa, n, alpha):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    assert x.shape[0] == 2
    assert n == 1 or n == -1
    return -n*np.pi*np.sin(np.pi*x[0, :])*np.sin(np.pi*x[1, :]+np.pi/2) +\
        alpha*exact_solution_imag(x)


def get_robin_bndry_conditions(kappa, alpha, Vh):
    """
    Do not pass element=function_space.ufl_element()
    as want forcing to be a scalar pass degree instead
    """
    bndry_obj = get_2d_unit_square_mesh_boundaries()
    boundary_conditions = []
    ii = 0
    for phys_var in [0, 1]:
        for normal in [1, -1]:
            boundary_conditions.append(
                ['robin',
                 bndry_obj[ii],
                 [RobinBoundaryRHS(kappa, normal, alpha, 'real', phys_var,
                                   degree=Vh.ufl_element().degree()),
                  RobinBoundaryRHS(kappa, normal, alpha, 'imag', phys_var,
                                   degree=Vh.ufl_element().degree())],
                 [dl.Constant(0), alpha]])
            ii += 1
    return boundary_conditions


def generate_mesh_with_cicular_subdomain(resolution, radius, plot_mesh=False):
    cx1, cy1 = 0.5, 0.5
    lx, ly = 1.0, 1.0

    # Define 2D geometry
    domain = mshr.Rectangle(dl.Point(0.0, 0.0), dl.Point(lx, ly))
    domain.set_subdomain(1, mshr.Circle(dl.Point(cx1, cy1), radius))
    cx2, cy2 = cx1-radius/np.sqrt(8), cy1-radius/np.sqrt(8)
    domain.set_subdomain(2, mshr.Circle(dl.Point(cx2, cy2), radius/2))

    # Generate and plot mesh
    mesh2d = mshr.generate_mesh(domain, resolution)
    if plot_mesh:
        dl.plot(mesh2d, "2D mesh")

        class Circle1(dl.SubDomain):
            def inside(self, x, on_boundary):
                return pow(x[0] - cx1, 2) + pow(x[1] - cy1, 2) <= pow(radius, 2)

        class Circle2(dl.SubDomain):
            def inside(self, x, on_boundary):
                return pow(x[0] - cx2, 2) + pow(x[1] - cy2, 2) <= pow(radius/2, 2)

        # Convert subdomains to mesh function for plotting
        mf = dl.MeshFunction("size_t", mesh2d, 2)
        mf.set_all(0)
        circle1 = Circle1()
        circle2 = Circle2()

        for c in dl.cells(mesh2d):
            if circle1.inside(c.midpoint(), True):
                mf[c.index()] = 1
            if circle2.inside(c.midpoint(), True):
                mf[c.index()] = 2

        dl.plot(mf, "Subdomains")
        # show must be called here or plot gets messed up
        # plt.show()
    return mesh2d


class TestHelmholtz(unittest.TestCase):
    # @dolfin_skiptest
    def test_dirichlet_boundary(self):
        frequency = 50
        omega = 2.*np.pi*frequency
        c1, c2 = [343.4, 6320]
        gamma = 8.4e-4

        kappa = omega/c1

        Nx, Ny = [21, 21]
        Lx, Ly = [1, 1]

        # create function space
        mesh = dl.RectangleMesh(dl.Point(0., 0.), dl.Point(Lx, Ly), Nx, Ny)
        degree = 1
        P1 = dl.FiniteElement('Lagrange', mesh.ufl_cell(), degree)
        element = dl.MixedElement([P1, P1])
        function_space = dl.FunctionSpace(mesh, element)

        boundary_conditions = None
        kappa = dl.Constant(kappa)
        # do not pass element=function_space.ufl_element()
        # as want forcing to be a scalar pass degree instead
        forcing = [
            Forcing(
                kappa, 'real', degree=function_space.ufl_element().degree()),
            Forcing(
                kappa, 'imag', degree=function_space.ufl_element().degree())]

        p = run_model(kappa, forcing, function_space, boundary_conditions)
        error = dl.errornorm(
            ExactSolution(kappa, element=function_space.ufl_element()), p)
        print('Error', error)
        assert error <= 3e-2

    # @dolfin_skiptest
    def test_robin_boundary(self):
        frequency = 50
        omega = 2.*np.pi*frequency
        c1, c2 = [343.4, 6320]
        gamma = 8.4e-4

        kappa = omega/c1
        alpha = kappa*gamma

        Nx, Ny = 21, 21
        Lx, Ly = 1, 1

        mesh = dl.RectangleMesh(dl.Point(0., 0.), dl.Point(Lx, Ly), Nx, Ny)
        degree = 1
        P1 = dl.FiniteElement('Lagrange', mesh.ufl_cell(), degree)
        element = dl.MixedElement([P1, P1])
        function_space = dl.FunctionSpace(mesh, element)

        boundary_conditions = get_robin_bndry_conditions(
            kappa, alpha, function_space)
        kappa = dl.Constant(kappa)
        forcing = [
            Forcing(
                kappa, 'real', degree=function_space.ufl_element().degree()),
            Forcing(
                kappa, 'imag', degree=function_space.ufl_element().degree())]

        p = run_model(kappa, forcing, function_space, boundary_conditions)
        error = dl.errornorm(
            ExactSolution(kappa, element=function_space.ufl_element()), p,)
        print('Error', error)
        assert error <= 3e-2

    @mshr_skiptest
    def test_variable_kappa(self):
        frequency = 50
        omega = 2.*np.pi*frequency
        c1, c2 = [343.4, 6320*100]
        gamma = 8.4e-4

        kappa1 = omega/c1
        kappa2 = omega/c2

        Nx, Ny = 21, 21
        Lx, Ly = 1, 1

        #mesh = RectangleMesh(Point(0., 0.), Point(Lx, Ly), Nx, Ny)
        mesh = generate_mesh_with_cicular_subdomain(15, 0.25, False)
        degree = 1
        P1 = dl.FiniteElement('Lagrange', mesh.ufl_cell(), degree)
        element = dl.MixedElement([P1, P1])
        function_space = dl.FunctionSpace(mesh, element)

        boundary_conditions = None
        forcing = [
            ForcingVariableKappa(
                kappa1, kappa2, 'real',
                degree=function_space.ufl_element().degree()),
            ForcingVariableKappa(
                kappa1, kappa2, 'imag',
                degree=function_space.ufl_element().degree())]
        kappa = dl.Expression('x[1] <= 0.5 + tol ? k_0 : k_1', degree=0,
                              tol=1e-14, k_0=kappa1, k_1=kappa2)
        # use mesh with circular subdomain as this should not effect result
        # significantly but will test if subdomain mesh works correctly
        p = run_model(kappa, forcing, function_space, boundary_conditions)
        error = dl.errornorm(ExactSolution(
            kappa, element=function_space.ufl_element()), p)
        print('Error', error)
        assert error <= 3e-2

    @mshr_skiptest
    def test_superposition(self):
        radius = 0.125
        mesh_resolution = 101
        mesh = generate_mesh_with_cicular_subdomain(
            mesh_resolution, radius, False)

        degree = 1
        P1 = dl.FiniteElement('Lagrange', mesh.ufl_cell(), degree)
        element = dl.MixedElement([P1, P1])
        function_space = dl.FunctionSpace(mesh, element)

        frequency = 400*3
        omega = 2.*np.pi*frequency
        sound_speed = np.array([343.4, 6320, 60])
        gamma = 8.4e-4
        speaker_amplitude = .1

        kappas = omega/sound_speed
        cr1, cr2 = 0.5, 0.5-radius/np.sqrt(8)
        kappa = dl.Expression(
            '((x[0]-c1)*(x[0]-c1)+(x[1]-c1)*(x[1]-c1) >= r*r + tol) ? k_0 : ((x[0]-c2)*(x[0]-c2)+(x[1]-c2)*(x[1]-c2)>=r*r/4+tol ? k_1 : k_2)',
            degree=0, tol=1e-14, k_0=kappas[0], k_1=kappas[1], k_2=kappas[2],
            r=radius, c1=cr1, c2=cr2)

        forcing = [dl.Constant(0), dl.Constant(0)]

        alpha = kappa*dl.Constant(gamma)
        beta = dl.Constant(1.204*omega*speaker_amplitude)
        bndry_obj = get_2d_square_mesh_boundary_segments()

        def get_boundary_conditions():
            boundary_conditions = [
                ['neumann', bndry_obj[ii], [dl.Constant(0), beta]]
                for ii in [1, 4, 7, 10]]
            boundary_conditions += [
                ['robin', bndry_obj[ii],
                 [dl.Constant(0), dl.Constant(0)], [dl.Constant(0), alpha]]
                for ii in [0, 2, 3, 5, 6, 8, 9, 11]]
            tmp = [None for ii in range(len(boundary_conditions))]
            for ii, jj in enumerate([1, 4, 7, 10]):
                tmp[jj] = boundary_conditions[ii]
            for ii, jj in enumerate([0, 2, 3, 5, 6, 8, 9, 11]):
                tmp[jj] = boundary_conditions[ii+4]
            boundary_conditions = tmp
            return boundary_conditions

        boundary_conditions = get_boundary_conditions()
        sol = run_model(kappa, forcing, function_space, boundary_conditions)

        sols = []
        for jj in [1, 4, 7, 10]:
            boundary_conditions = get_boundary_conditions()
            for kk in range(12):
                if jj != kk:
                    boundary_conditions[kk][2][1] = dl.Constant(0)
            pii = run_model(kappa, forcing, function_space,
                            boundary_conditions)
            sols.append(pii)

        # for jj in [0,2,3,5,6,8,9,11]:
        #     boundary_conditions = get_boundary_conditions()
        #     for kk in range(12):
        #         if jj!=kk:
        #             boundary_conditions[kk][2][1]=dl.Constant(0)
        #     pii=run_model(kappa,forcing,function_space,boundary_conditions)
        #     sols.append(pii)

        boundary_conditions = get_boundary_conditions()
        for kk in range(12):
            if kk not in [0, 2, 3, 5, 6, 8, 9, 11]:
                boundary_conditions[kk][2][1] = dl.Constant(0)
        pii = run_model(kappa, forcing, function_space, boundary_conditions)
        sols.append(pii)

        superposition_sol = sols[0]
        for ii in range(1, len(sols)):
            superposition_sol += sols[ii]
        superposition_sol = dl.project(superposition_sol, function_space)

        pr, pi = sol.split()
        pr_super, pi_super = superposition_sol.split()
        # print('error',dl.errornorm(pr_super,pr))
        # print('error',dl.errornorm(pi_super,pi))
        assert dl.errornorm(pr_super, pr) < 1e-10
        assert dl.errornorm(pi_super, pi) < 1e-10

        # plt.subplot(1,4,1)
        # dl.plot(pr)
        # plt.subplot(1,4,3)
        # pp=dl.plot(pi)
        # plt.colorbar(pp)
        # plt.subplot(1,4,2)

        # dl.plot(pr_super)
        # plt.subplot(1,4,4)
        # pp=dl.plot(pi_super)
        # plt.colorbar(pp)
        # plt.figure()
        # pp=dl.plot(pi_super-pi)
        # print(pp)
        # plt.colorbar(pp)

        # plt.show()


if __name__ == "__main__":
    # dl.set_log_level(40)
    helmholtz_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestHelmholtz)
    unittest.TextTestRunner(verbosity=2).run(helmholtz_test_suite)
