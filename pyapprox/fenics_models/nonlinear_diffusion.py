import numpy as np
from pyapprox.fenics_models.fenics_utilities import *

def extract_options(options):
    forcing = options.get("forcing", dla.Constant(0))
    nonlinear_diffusion = options.get('nonlinear_diffusion', None)
    second_order_timestepping = options.get('second_order_timestepping', True)
    nlsparam = options.get('nlsparam', dict())
    positivity_tol = options.get('positivity_tol', 0.0)
    return options['time_step'], options['final_time'],\
        forcing, options['init_condition'],\
        options['boundary_conditions'], nonlinear_diffusion,\
        second_order_timestepping, nlsparam, positivity_tol


def run_model(function_space, options):
    dt, final_time, forcing, init_condition, boundary_conditions, \
        nonlinear_diffusion, second_order_timestepping, nlsparam, \
        positivity_tol = extract_options(options)

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
