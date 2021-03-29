import os, sys
import time
import numpy as np


if sys.platform == 'win32':
    raise ImportError("Not available on Windows")

try:
    import dolfin as dl
    from pyapprox_dev.fenics_models.fenics_utilities import *
    import fenics_adjoint as dla
except (ImportError, ModuleNotFoundError):
    has_dla = False

    # Create stub class
    class dla(object):
        UserExpression = object

else:
    # If no exceptions raised
    has_dla = True


def run_model(function_space, kappa, forcing, init_condition, dt, final_time,
              boundary_conditions=None, second_order_timestepping=False,
              exact_sol=None, velocity=None, point_sources=None,
              intermediate_times=None):
    """
    Use implicit euler to solve transient advection diffusion equation

    du/dt = grad (k* grad u) - vel*grad u + f

    WARNING: when point sources solution changes significantly when mesh is 
    varied
    """
    if not has_dla:
        raise RuntimeError("DLA not available")

    mesh = function_space.mesh()

    time_independent_boundaries = False
    if boundary_conditions is None:
        bndry_obj = dl.CompiledSubDomain("on_boundary")
        boundary_conditions = [['dirichlet', bndry_obj, dla.Constant(0)]]
        time_independent_boundaries = True

    num_bndrys = len(boundary_conditions)
    boundaries = mark_boundaries(mesh, boundary_conditions)
    dirichlet_bcs = collect_dirichlet_boundaries(
        function_space, boundary_conditions, boundaries)

    # To express integrals over the boundary parts using ds(i), we must first
    # redefine the measure ds in terms of our boundary markers:
    ds = dl.Measure('ds', domain=mesh, subdomain_data=boundaries)
    dx = dl.Measure('dx', domain=mesh)

    # Variational problem at each time
    u = dl.TrialFunction(function_space)
    v = dl.TestFunction(function_space)

    # Previous solution
    if hasattr(init_condition, 't'):
        assert init_condition.t == 0
    u_1 = dla.interpolate(init_condition, function_space)

    if not second_order_timestepping:
        theta = 1
    else:
        theta = 0.5

    if hasattr(forcing, 't'):
        forcing_1 = copy_expression(forcing)
    else:
        forcing_1 = forcing

    def steady_state_form(u, v, f):
        F = kappa*dl.inner(dl.grad(u), dl.grad(v))*dx
        F -= f*v*dx
        if velocity is not None:
            F += dl.dot(velocity, dl.grad(u))*v*dx
        return F

    F = u*v*dx-u_1*v*dx + dt*theta*steady_state_form(u, v, forcing) + \
        dt*(1.-theta)*steady_state_form(u_1, v, forcing_1)
    a, L = dl.lhs(F), dl.rhs(F)

    # a = u*v*dx + theta*dt*kappa*dl.inner(dl.grad(u), dl.grad(v))*dx
    # L = (u_1 + dt*theta*forcing)*v*dx

    # if velocity is not None:
    #     a += theta*dt*v*dl.dot(velocity,dl.grad(u))*dx

    # if second_order_timestepping:
    #     L -= (1-theta)*dt*dl.inner(kappa*dl.grad(u_1), dl.grad(v))*dx
    #     L += (1-theta)*dt*forcing_1*v*dx

    #     if velocity is not None:
    #         L -= (1-theta)*dt*(v*dl.dot(velocity,dl.grad(u_1)))*dx

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

    if time_independent_boundaries:
        # TODO this can be used if dirichlet and robin conditions are not
        # time dependent.
        A = dla.assemble(a)
        for bc in dirichlet_bcs:
            bc.apply(A)
        solver = dla.LUSolver(A)
        #solver.parameters["reuse_factorization"] = True
    else:
        solver = None

    u_2 = dla.Function(function_space)
    u_2.assign(u_1)
    t = 0.0

    dt_tol = 1e-12
    n_time_steps = 0
    if intermediate_times is not None:
        intermediate_u = []
        intermediate_cnt = 0
        # assert in chronological order
        assert np.allclose(intermediate_times, np.array(intermediate_times))
        assert np.all(intermediate_times < final_time)

    while t < final_time-dt_tol:
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

        #A, b = dl.assemble_system(a, L, dirichlet_bcs)
        # for bc in dirichlet_bcs:
        #    bc.apply(A,b)
        if boundary_conditions is not None:
            A = dla.assemble(a)
            for bc in dirichlet_bcs:
                bc.apply(A)

        b = dla.assemble(L)
        for bc in dirichlet_bcs:
            bc.apply(b)

        if point_sources is not None:
            ps_list = []
            for ii in range(len(point_sources)):
                point, expr = point_sources[ii]
                ps_list.append((dl.Point(point[0], point[1]), expr(t)))
            ps = dla.PointSource(function_space, ps_list)
            ps.apply(b)

        if solver is None:
            dla.solve(A, u_2.vector(), b)
        else:
            solver.solve(u_2.vector(), b)

        # tape = dla.get_working_tape()
        # tape.visualise()

        #print ("t =", t, "end t=", final_time)

        # Update previous solution
        u_1.assign(u_2)
        # import matplotlib.pyplot as plt
        # plt.subplot(131)
        # pp=dl.plot(u_1)
        # plt.subplot(132)
        # dl.plot(forcing,mesh=mesh)
        # plt.subplot(133)
        # dl.plot(forcing_1,mesh=mesh)
        # plt.colorbar(pp)
        # plt.show()

        # compute error
        if exact_sol is not None:
            exact_sol.t = t
            error = dl.errornorm(exact_sol, u_2)
            print('t = %.2f: error = %.3g' % (t, error))
            # dl.plot(exact_sol,mesh=mesh)
            # plt.show()

        if (intermediate_times is not None and
            intermediate_cnt < intermediate_times.shape[0] and
                t >= intermediate_times[intermediate_cnt]):
            # save solution closest to intermediate time
            u_t = dla.Function(function_space)
            u_t.assign(u_2)
            intermediate_u.append(u_t)
            intermediate_cnt += 1
        n_time_steps += 1
    # print ("t =", t, "end t=", final_time,"# time steps", n_time_steps)

    if intermediate_times is None:
        return u_2
    else:
        return intermediate_u+[u_2]


def run_steady_state_model(function_space, kappa, forcing,
                           boundary_conditions=None, velocity=None):
    """
    Solve steady-state diffusion equation

    -grad (k* grad u) = f
    """
    mesh = function_space.mesh()

    if boundary_conditions == None:
        bndry_obj = dl.CompiledSubDomain("on_boundary")
        boundary_conditions = [['dirichlet', bndry_obj, dla.Constant(0)]]

    num_bndrys = len(boundary_conditions)
    boundaries = mark_boundaries(mesh, boundary_conditions)
    dirichlet_bcs = collect_dirichlet_boundaries(
        function_space, boundary_conditions, boundaries)

    # To express integrals over the boundary parts using ds(i), we must first
    # redefine the measure ds in terms of our boundary markers:
    ds = dl.Measure('ds', domain=mesh, subdomain_data=boundaries)
    dx = dl.Measure('dx', domain=mesh)

    # Variational problem at each time
    u = dl.TrialFunction(function_space)
    v = dl.TestFunction(function_space)

    a = kappa*dl.inner(dl.grad(u), dl.grad(v))*dx
    L = forcing*v*dx

    if velocity is not None:
        a += v*dl.dot(velocity, dl.grad(u))*dx

    beta_1_list = []
    alpha_1_list = []
    for ii in range(num_bndrys):
        if (boundary_conditions[ii][0] == 'robin'):
            alpha = boundary_conditions[ii][3]
            a += alpha*u*v*ds(ii)

        elif ((boundary_conditions[ii][0] == 'robin') or
                (boundary_conditions[ii][0] == 'neumann')):
            beta = boundary_conditions[ii][2]
            L -= beta*v*ds(ii)

    u = dla.Function(function_space)
    # dl.assemble, apply and solve does not work with
    # fenics adjoint
    # A, b = dla.assemble_system(a, L, dirichlet_bcs)
    # # apply boundary conditions
    # for bc in dirichlet_bcs:
    #     bc.apply(A, b)
    # dla.solve(A, u.vector(), b)
    dla.solve(a == L, u, dirichlet_bcs)
    return u


class Diffusivity(dla.UserExpression):
    # def __init__(self,**kwargs):
    #    if '2019' in dl.__version__:
    #        # does not work for fenics 2017 only 2019
    #        super().__init__(**kwargs)

    def initialize_kle(self, num_vars, corr_len):
        domain_len = 1
        self.num_vars = num_vars
        self.sqrtpi = np.sqrt(np.pi)
        self.Lp = max(domain_len, 2*corr_len)
        self.L = corr_len/self.Lp
        self.nn = np.arange(2, num_vars+1)
        self.eigenvalues = np.sqrt(self.sqrtpi*self.L)*np.exp(
            -((np.floor(self.nn/2)*np.pi*self.L))**2/4)
        # -((np.floor(self.nn/2)*np.pi*self.L))**2/8)#nobile eigvals
        self.eigenvectors = np.empty((self.num_vars-1), dtype=float)

    def set_random_sample(self, sample):
        assert sample.shape[0] == self.num_vars
        self.sample = sample

    def eval(self, values, x):
        self.eigenvectors[::2] = np.sin(
            ((np.floor(self.nn[::2]/2)*np.pi*x[0]))/self.Lp)
        self.eigenvectors[1::2] = np.cos(
            ((np.floor(self.nn[1::2]/2)*np.pi*x[0]))/self.Lp)
        self.eigenvectors *= self.eigenvalues

        values[0] = self.eigenvectors.dot(self.sample[1:])
        values[0] += 1+self.sample[0]*np.sqrt(self.sqrtpi*self.L/2)
        values[0] = np.exp(values[0])+0.5

#-------------#
# useful code #
#-------------#

# # Report flux across left boundary
# # this will be zero for cosine analytical examples because integral
# # of one period sine function is zero
# n = FacetNormal(mesh)
# flux = assemble(kappa*dot(grad(u), n)*ds(0))
# info('t = %g, flux = %e' % (t, flux))
