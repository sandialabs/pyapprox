import numpy as np
from pyapprox.fenics_models.fenics_utilities import *


def run_model(function_space, time_step, final_time, forcing, boundary_conditions, init_condition, 
              nonlinear_diffusion, second_order_timestepping=False, nlsparam=dict(), positivity_tol=0):

    dt = time_step
    mesh = function_space.mesh()
    num_bndrys = len(boundary_conditions)

    if (len(boundary_conditions) == 1 and
            isinstance(boundary_conditions[0][2], DirichletBC)):
        ds = dl.Measure('ds', domain=mesh)
        dirichlet_bcs = [boundary_conditions[0][2]]
    else:
        boundaries = mark_boundaries(mesh, boundary_conditions)
        dirichlet_bcs = collect_dirichlet_boundaries(
            function_space, boundary_conditions, boundaries)
        ds = dl.Measure('ds', domain=mesh, subdomain_data=boundaries)

    dx = dl.Measure('dx', domain=mesh)

    u = dl.TrialFunction(function_space)
    v = dl.TestFunction(function_space)

    # Previous solution
    #assert init_condition.t==0
    u_1 = dla.interpolate(init_condition, function_space)

    u_2 = dla.Function(function_space)
    u_2.assign(u_1)

    if not second_order_timestepping:
        theta = 1
    else:
        theta = 0.5

    if second_order_timestepping and hasattr(forcing, 't'):
        forcing_1 = copy_expression(forcing)
    else:
        forcing_1 = forcing

    kappa = nonlinear_diffusion(u)
    a = u*v*dx + theta*dt*kappa*dl.inner(dl.grad(u), dl.grad(v))*dx
    L = (u_1 + theta*dt*forcing)*v*dx

    # subtract of positivity preserving part added to diffusion
    if positivity_tol > 0:
        a -= positivity_tol*dl.inner(dl.grad(u), dl.grad(v))*dx

    if second_order_timestepping:
        kappa_1 = nonlinear_diffusion(u_1)
        L -= (1-theta)*dt*kappa_1*dl.inner(dl.grad(u_1), dl.grad(v))*dx
        L += (1-theta)*dt*forcing_1*v*dx

    beta_1_list = []
    alpha_1_list = []
    for ii in range(num_bndrys):
        if (boundary_conditions[ii][0] == 'robin'):
            alpha = boundary_conditions[ii][3]
            a += theta*dt*alpha*u*v*ds(ii)
            if second_order_timestepping:
                if hasattr(alpha, 't'):
                    alpha_1 = copy_expression(alpha)
                    alpha_1_list.append(alpha_1)
                else:
                    alpha_1 = alpha
                L -= (1-theta)*dt*alpha_1*u_1*v*ds(ii)

        if ((boundary_conditions[ii][0] == 'robin') or
                (boundary_conditions[ii][0] == 'neumann')):
            beta = boundary_conditions[ii][2]
            L -= theta*dt*beta*v*ds(ii)
            if second_order_timestepping:
                if hasattr(beta, 't'):
                    beta_1 = copy_expression(beta)
                    beta_1_list.append(beta_1)
                else:
                    # boundary condition is constant in time
                    beta_1 = beta
                L -= (1-theta)*dt*beta_1*v*ds(ii)

    if hasattr(init_condition, 't'):
        t = init_condition.t
    else:
        t = 0.0
    # print(init_condition.t)
    while t < final_time:
        # print('TIME',t)
        # Update current time
        prev_t = t
        forcing_1.t = prev_t
        t += dt
        t = min(t, final_time)
        forcing.t = t

        # set current time for time varying boundary conditions
        for ii in range(num_bndrys):
            if hasattr(boundary_conditions[ii][2], 't'):
                boundary_conditions[ii][2].t = t

        # set previous time for time varying boundary conditions when
        # using second order timestepping. lists will be empty if using
        # first order timestepping
        for jj in range(len(beta_1_list)):
            beta_1_list[jj].t = prev_t
        for jj in range(len(alpha_1_list)):
            alpha_1_list[jj].t = prev_t

        # solver must be redefined at every timestep
        F = a-L
        F = dl.action(F, u_2)
        dla.solve(F == 0, u_2, dirichlet_bcs, solver_parameters=nlsparam)
        
        # import matplotlib.pyplot as plt
        # pl = dl.plot(sol); plt.colorbar(pl); plt.show()
        # import matplotlib.pyplot as plt
        # pl = dl.plot(sol); plt.show()

        # Update previous solution
        u_1.assign(u_2)

    return u_1


def compute_gamma(A, n=3):
    """n = 3 # GlenExponent"""
    g = 9.81 #GravityAcceleration
    rho = 910 #IceDensity
    gamma = 2.0*A*(rho*g)**n/(n+2)
    return gamma


def var_form_max(a, b): return (a+b+abs(a-b))/dl.Constant(2)


def shallow_ice_diffusion(n, gamma, bed,positivity_tol, beta, thickness):
    r"""
    Nonlinear diffusivity function of the shallow ice equation

    dH/dt -div(Gamma * H**(n+2) * |\nabla h|**(n-1) \nabla h) = f

    n: integer
        Glen exponent

    gamma : float
        Diffusivity strength

    bed : Fenics Function
        Bed elevation

    thickness : Fenics Function
        Thickness H of the ice-sheet
    """
    height = thickness
    if bed is not None:
        height += bed

    var_form = gamma*thickness**(n+2)*dl.inner(
        dl.grad(height),dl.grad(height))**((n-1)/2)
    #var_form = gamma*thickness**(n+2)*(dl.inner(    
    #    dl.grad(height),dl.grad(height))+positivity_tol)**((n-1)/2)

    if beta is not None:
        g = 9.81 #GravityAcceleration
        rho = 910 #IceDensity
        var_form += rho*g/2*height**2/dla.exp(beta)

    # var_form = var_form_max(var_form,positivity_tol) # doesnt work well
    var_form += positivity_tol
    return var_form


from pyapprox.fenics_models.advection_diffusion_wrappers import AdvectionDiffusionModel
class HalfarShallowIceModel(AdvectionDiffusionModel):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # Change the following functions to modify governing equations
    #     initialize_random_expressions()
    #     get_initial_condition()
    #     get_boundary_conditions_and_function_space()
    #     get_velocity()
    #     get_forcing()
    #     get_diffusivity()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def set_constants(self):
        self.secpera = 31556926  # seconds per anum
        self.Gamma = 2.8457136065980445e-05
        self.glen_exponent = 3
        self.Lx = 1200e3
        self.bed = None
        self.beta = None

        self.positivity_tol = 0

    def initialize_random_expressions(self, random_sample):
        r"""
        Overide this class to split random_samples into the parts that effect
        the 5 random quantities
        """
        self.set_constants()
        init_condition = self.get_initial_condition(None)
        boundary_conditions, function_space = \
            self.get_boundary_conditions_and_function_space(None)
        beta = self.get_velocity(None)
        forcing = self.get_forcing(None)
        kappa = self.get_diffusivity(random_sample)
        return init_condition, boundary_conditions, function_space, beta, \
            forcing, kappa

    def get_initial_condition(self, random_sample):
        r"""By Default the initial condition is deterministic and set to zero"""
        assert random_sample is None
        init_solution = get_shallow_ice_exact_solution(
            self.Gamma, self.mesh, self.degree, self.nphys_dim)
        init_solution.t = self.init_time
        return initial_condition

    def get_boundary_conditions_and_function_space(self, random_sample):
        r"""By Default the boundary conditions are deterministic, Dirichlet and 
           and set to zero"""
        assert random_sample is None
        exact_solution = get_shallow_ice_exact_solution(
            self.Gamma, self.mesh, self.degree, self.nphys_dim)
        exact_solution.t = self.init_time
        if self.nphys_dim == 1:
            boundary_conditions = \
                get_1d_dirichlet_boundary_conditions_from_expression(
                    exact_solution, -self.Lx, self.Lx)
        elif self.nphys_dim == 2:
            boundary_conditions =\
                get_dirichlet_boundary_conditions_from_expression(
                    exact_solution, -self.Lx, self.Lx, -self.Lx, self.Lx)

        
        function_space = dl.FunctionSpace(self.mesh, "CG", self.degree)
        return boundary_conditions, function_space

    def get_velocity(self, random_sample):
        r"""By Default the advection is deterministic and set to zero"""
        assert random_sample is None
        beta = dla.Expression((str(0), str(0)), degree=self.degree)
        return beta

    def get_forcing(self, random_sample):
        r"""By Default the forcing is deterministic and set to 

        .. math:: (1.5+\cos(2\pi t))*cos(x_1)

        where :math:`t` is time and :math:`x_1` is the first spatial dimension.
        """
        forcing = dla.Constant(0.0)
        return forcing

    def get_diffusivity(self, random_sample):
        r"""
        Use the random diffusivity specified in [JEGGIJNME2020].
        """
        assert random_sample is None
        kappa = partial(
            shallow_ice_diffusion, self.glen_exponent, self.Gamma, self.bed, self.positivity_tol, self.beta)
        return kappa

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # Change the following functions to modify mapping of discretization
    # parameters to mesh and timestep resolution
    #     get_timestep()
    #     get_mesh_resolution()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def get_timestep(self, dt_level):
        dt = self.final_time/2**(dt_level+2)
        return dt

    def get_mesh_resolution(self, mesh_levels):
        nx_level, ny_level = mesh_levels
        nx = 2**(nx_level+2)
        ny = 2**(ny_level+2)
        return nx, ny

    def get_mesh(self, resolution_levels):
        r"""The arguments to this function are the outputs of 
        get_degrees_of_freedom_and_timestep()"""
        nx, ny = np.asarray(resolution_levels, dtype=int)
        mesh = dla.RectangleMesh(dl.Point(0, 0), dl.Point(1, 1), nx, ny)
        return mesh

    def set_num_config_vars(self):
        r"""
        Should be equal to the number of physical dimensions + 1 
        (for the temporal resolution)
        """
        self.num_config_vars = 3

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # Do not change the following functions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def __init__(self, init_time, final_time, degree, qoi_functional,
                 second_order_timestepping=True, options={},
                 qoi_functional_grad=None):
        self.init_time = init_time
        self.final_time = final_time
        self.qoi_functional = qoi_functional
        self.degree = degree
        self.second_order_timestepping = second_order_timestepping
        self.set_num_config_vars()
        self.options = options
        self.qoi_functional_grad = qoi_functional_grad

    def solve(self, samples):
        r"""
        Run the simulation

        Notes
        -----
        Dolfin objects must be initialized inside this function otherwise 
        this object cannot be pickled and used with multiprocessing.Pool
        """
        assert samples.ndim == 2
        assert samples.shape[1] == 1

        resolution_levels = samples[-self.num_config_vars:, 0]
        dt = self.get_timestep(resolution_levels[-1])
        self.mesh = self.get_mesh(
            self.get_mesh_resolution(resolution_levels[:-1]))

        random_sample = samples[:-self.num_config_vars, 0]

        init_condition, boundary_conditions, function_space, beta, \
            forcing, kappa = self.initialize_random_expressions(
                random_sample)
        # when dla is dolfin_adjoint
        # Must project dla.CompiledExpression to avoid error
        # site-packages/pyadjoint/overloaded_type.py", line 136,
        # in _ad_convert_type raise NotImplementedError
        self.kappa = dla.interpolate(kappa, function_space)
        # this is not necessary when just using dolfin

        sol = run_model(
            function_space, self.kappa, forcing,
            init_condition, dt, self.final_time,
            boundary_conditions, velocity=beta,
            second_order_timestepping=self.second_order_timestepping,
            intermediate_times=self.options.get('intermediate_times', None))
        return sol

    def __call__(self, samples, jac=False):
        sol = self.solve(samples)
        vals = np.atleast_1d(self.qoi_functional(sol))
        if vals.ndim == 1:
            vals = vals[:, np.newaxis]

        if jac is True:
            assert self.qoi_functional_grad is not None
            grad = self.qoi_functional_grad(sol, self)
            return vals, grad

        return vals


