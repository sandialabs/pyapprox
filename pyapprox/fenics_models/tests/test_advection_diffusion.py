from pyapprox.fenics_models.advection_diffusion import *
from pyapprox.fenics_models.advection_diffusion_wrappers import *
from pyapprox.fenics_models.fenics_utilities import *
import unittest
import matplotlib.pyplot as plt

def get_repeated_random_samples_with_varying_config_values(
        num_vars,config_vars,generate_random_sample,num_samples):
    num_config_vars=config_vars.shape[0]
    samples=np.empty((num_vars+num_config_vars,0))
    random_samples=generate_random_sample(num_vars,num_samples)
    random_samples=np.vstack(
        (random_samples,np.empty((config_vars.shape[0],num_samples))))
    for ii in range(num_samples):
        samples_ii=random_samples[:,ii:ii+1]
        samples_ii=np.tile(samples_ii,(1,config_vars.shape[1]))
        samples_ii[-num_config_vars:,:]=config_vars
        samples=np.hstack((samples,samples_ii))
    return samples

import sympy as sp
def get_exact_solution_sympy(steady_state):
    from sympy.abc import t
    x,y = sp.symbols('x[0] x[1]')
    a,b,c = 2*dl.pi,2*dl.pi,2*dl.pi
    # using sp.pi instead of pi can cause JIT compiler to fail. Not sure why
    u = sp.sin(a*x)*sp.sin(b*y)
    if not steady_state:
        u *= sp.cos(c*t)
    return u, x, y, t

def get_exact_solution(mesh,degree,steady_state=False):
    exact_sol_sympy = get_exact_solution_sympy(steady_state)[0]
    exact_sol = dl.Expression(
        sp.printing.ccode(exact_sol_sympy), cell=mesh.ufl_cell(),
        domain=mesh,t=0,degree=degree)
    #print (sp.printing.ccode(exact_sol_sympy))
    return exact_sol

def get_forcing(kappa,mesh,degree,steady_state=False,advection=False):
    """

    u(x,y,t)=sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*t)

    is a solution to du/dt=k d^2u/dx^2+f on [0,1]^2

    f = -2*pi*sin(2*pi*t)*sin(2*pi*x[0])*sin(2*pi*x[1])
      + kappa*8*pi**2*sin(2*pi*x[0])*sin(2*pi*x[1])*cos(2*pi*t)
    """
    u, x, y, t = get_exact_solution_sympy(steady_state)
    
    dxu2 = sum(u.diff(xi, 2) for xi in (x, y))
    dtu = u.diff(t, 1)
    if advection:
        bdxu = sum(u.diff(xi, 1) for xi in (x, y))# for beta = Constant([1,1])
        #bdxu = sum(u.diff(xi, 1) for xi in (x,)) # for beta = Constant([1,0])
        forcing_sympy=dtu-kappa*dxu2+bdxu
        assert dtu==kappa*dxu2-bdxu+forcing_sympy
    else:
        forcing_sympy=(dtu-kappa*dxu2)
        assert dtu==kappa*dxu2+forcing_sympy
    #print (sp.printing.ccode(forcing_sympy))
    forcing = dl.Expression(
        sp.printing.ccode(forcing_sympy), cell=mesh.ufl_cell(), domain=mesh,
        degree=degree,t=0)

    return forcing

class ExactSolutionPy(dl.UserExpression):
    def __init__(self,**kwargs):
        self.kappa=kappa
        self.t=0
        if '2019' in dl.__version__:
            # does not work for fenics 2017 only 2019
            super().__init__(**kwargs)
        # in 2017 base class __init__ does not need to be called.
    def eval(self, values, x):
        values[0] = np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[1])*cos(
            2*np.pi*self.t)
        if abs(x[0])<1e-12:
            #print(x,values,self.t)
            assert values[0]==0

class ForcingPy(dl.UserExpression):
    def __init__(self,kappa, **kwargs):
        self.kappa=kappa
        self.t=0
        if '2019' in dl.__version__:
            # does not work for fenics 2017 only 2019
            super().__init__(**kwargs)
        # in 2017 base class __init__ does not need to be called.

    def eval(self, values, x):
        values[0] = \
        -2*np.pi*sin(
            2*np.pi*self.t)*sin(2*np.pi*x[0])*sin(2*np.pi*x[1])\
        +self.kappa*8*np.pi**2*sin(
            2*np.pi*x[0])*sin(2*np.pi*x[1])*cos(2*np.pi*self.t)

def get_exact_solution_py(mesh,degree):
    return ExactSolution(cell=mesh.ufl_cell(),
                         domain=mesh,degree=degree)

def get_forcing_py(kappa,mesh,degree):
    return Forcing(kappa,cell=mesh.ufl_cell(),
                   domain=mesh,degree=degree)

def get_gradu_dot_n(kappa,alpha,mesh,degree,phys_var,n):
    u, x, y, t = get_exact_solution_sympy(False)
    xi = [x,y][phys_var]
    gradu = u.diff(xi, 1)
    expr_sympy = kappa*gradu*n+alpha*u
    expr = dl.Expression(
        sp.printing.ccode(expr_sympy), cell=mesh.ufl_cell(), domain=mesh,
        degree=degree,t=0)
    #print(expr_sympy)
    return expr
    

def get_quadratic_exact_solution(alpha,beta,mesh,degree):
    exact_sol = dl.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    alpha=alpha, beta=beta,cell=mesh.ufl_cell(),
                    domain=mesh,t=0,degree=degree)
    return exact_sol

def get_quadratic_solution_forcing(alpha,beta,mesh,degree):
    f = dl.Expression('beta - 2 - 2*alpha', beta=beta, alpha=alpha,degree=degree)
    return f

class TestTransientDiffusion(unittest.TestCase):
    def test_quadratic_solution(self):
        dt = 0.01; t = 0; final_time = 2
        nx,ny=2,2
        degree=2
        alpha,beta=3,1.2
        kappa=dl.Constant(1)
        mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(1, 1),nx,ny)
        function_space=dl.FunctionSpace(mesh, "Lagrange", degree)

        boundary_conditions = get_dirichlet_boundary_conditions_from_expression(
            get_quadratic_exact_solution(alpha,beta,mesh,degree),0,1,0,1)
        sol = run_model(
            function_space,kappa,
            get_quadratic_solution_forcing(alpha,beta,mesh,degree),
            get_quadratic_exact_solution(alpha,beta,mesh,degree),
            dt,final_time,boundary_conditions=boundary_conditions)#,
            #exact_sol=get_quadratic_exact_solution(alpha,beta,mesh,degree))

        exact_sol = get_quadratic_exact_solution(alpha,beta,mesh,degree)
        exact_sol.t=final_time
        error = dl.errornorm(exact_sol,sol,mesh=mesh)
        print('Error',error)
        assert error <= 3e-14

    def test_cosine_solution_dirichlet_boundary_conditions(self):
        dt = 0.05; t = 0; final_time = 1
        nx,ny=31,31
        degree=2
        kappa=dl.Constant(3)

        mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(1, 1),nx,ny)
        function_space=dl.FunctionSpace(mesh, "Lagrange", degree)
        boundary_conditions = get_dirichlet_boundary_conditions_from_expression(
            get_exact_solution(mesh,degree),0,1,0,1)
        sol = run_model(
            function_space,kappa,get_forcing(kappa,mesh,degree),
            get_exact_solution(mesh,degree),
            dt,final_time,boundary_conditions=boundary_conditions,
            second_order_timestepping=True,exact_sol=None)

        exact_sol = get_exact_solution(mesh,degree)
        exact_sol.t=final_time
        error = dl.errornorm(exact_sol,sol,mesh=mesh)
        print('Error',error)
        assert error <= 1e-4

    def test_cosine_solution_robin_boundary_conditions(self):
        dt = 0.05; t = 0; final_time = 1
        nx,ny=31,31
        degree=2
        kappa=dl.Constant(3)

        mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(1, 1),nx,ny)
        function_space=dl.FunctionSpace(mesh, "Lagrange", degree)
        # bc : kappa * grad u.dot(n)+alpha*u=beta
        alpha=1
        from functools import partial
        expression = partial(get_gradu_dot_n,kappa,alpha,mesh,degree)
        boundary_conditions = get_robin_boundary_conditions_from_expression(
            expression,dl.Constant(alpha))
            
        sol = run_model(
            function_space,kappa,get_forcing(kappa,mesh,degree),
            get_exact_solution(mesh,degree),
            dt,final_time,boundary_conditions=boundary_conditions,
            second_order_timestepping=True,exact_sol=None)

        exact_sol = get_exact_solution(mesh,degree)
        exact_sol.t=final_time
        error = dl.errornorm(exact_sol,sol,mesh=mesh)
        print('Error',error)
        assert error <= 1e-4

    def test_superposition_dirichlet_boundary_conditions(self):
        dt = 0.05; t = 0; final_time = 1
        nx,ny=31,31
        degree=2
        kappa=dl.Constant(3)
        xl,xr,yb,yt=0.25,1.25,0.25,1.25

        sols = []
        mesh = dl.RectangleMesh(dl.Point(xl, yb),dl.Point(xr, yt),nx,ny)
        function_space=dl.FunctionSpace(mesh, "Lagrange", degree)
        for ii in range(4):
            boundary_conditions=\
                get_dirichlet_boundary_conditions_from_expression(
                    get_exact_solution(mesh,degree),xl,xr,yb,yt)    
            for jj in range(4):
                if jj!=ii:
                    boundary_conditions[jj][2]=dl.Constant(0.)

            sol = run_model(
                function_space,kappa,dl.Constant(0.0),
                dl.Constant(0.0),
                dt,final_time,boundary_conditions=boundary_conditions,
                second_order_timestepping=True,exact_sol=None)
            sols.append(sol)
            
        sol = run_model(
            function_space,kappa,get_forcing(kappa,mesh,degree),
            get_exact_solution(mesh,degree),
            dt,final_time,boundary_conditions=None,
            second_order_timestepping=True,exact_sol=None)
        sols.append(sol)

        superposition_sol = sols[0]
        for ii in range(1,len(sols)):
            superposition_sol += sols[ii]
        superposition_sol = dl.project(superposition_sol,function_space)

        boundary_conditions = get_dirichlet_boundary_conditions_from_expression(
            get_exact_solution(mesh,degree),xl,xr,yb,yt)
        sol = run_model(
            function_space,kappa,get_forcing(kappa,mesh,degree),
            get_exact_solution(mesh,degree),
            dt,final_time,boundary_conditions=boundary_conditions,
            second_order_timestepping=True,exact_sol=None)

        exact_sol = get_exact_solution(mesh,degree,True)
        error = dl.errornorm(exact_sol,sol,mesh=mesh)
        print('Error',error)
        assert error <= 1e-4

        error = dl.errornorm(superposition_sol,sol,mesh=mesh)
        print('error',error)
        assert error<1e-14

class TestSteadyStateDiffusion(unittest.TestCase):
    def test_superposition_dirichlet_boundary_conditions(self):
        nx,ny=31,31
        degree=2
        kappa=dl.Constant(3)
        xl,xr,yb,yt=0.25,1.25,0.25,1.25

        sols = []
        mesh = dl.RectangleMesh(dl.Point(xl, yb),dl.Point(xr, yt),nx,ny)
        function_space=dl.FunctionSpace(mesh, "Lagrange", degree)
        for ii in range(4):
            boundary_conditions=\
                get_dirichlet_boundary_conditions_from_expression(
                    get_exact_solution(mesh,degree,True),xl,xr,yb,yt)    
            for jj in range(4):
                if jj!=ii:
                    boundary_conditions[jj]=[
                        'dirichlet',boundary_conditions[jj][1],0]
                                
            sol = run_steady_state_model(
                function_space,kappa,dl.Constant(0.0),
                boundary_conditions=boundary_conditions)
            sols.append(sol)
            
        sol = run_steady_state_model(
            function_space,kappa,get_forcing(kappa,mesh,degree,True),
            boundary_conditions=None)
        sols.append(sol)

        superposition_sol = sols[0]
        for ii in range(1,len(sols)):
            superposition_sol += sols[ii]
        #pp=dl.plot(superposition_sol)
        #plt.colorbar(pp); plt.show()
        superposition_sol = dl.project(superposition_sol,function_space)

        boundary_conditions = get_dirichlet_boundary_conditions_from_expression(
            get_exact_solution(mesh,degree,True),xl,xr,yb,yt)
        sol = run_steady_state_model(
            function_space,kappa,get_forcing(kappa,mesh,degree,True),
            boundary_conditions=boundary_conditions)
        #plt.figure()
        #pp=dl.plot(sol-superposition_sol)
        #plt.colorbar(pp); plt.show()


        exact_sol = get_exact_solution(mesh,degree,True)
        error = dl.errornorm(exact_sol,sol,mesh=mesh)
        print('Error',error)
        assert error <= 1e-4

        error = dl.errornorm(superposition_sol,sol,mesh=mesh)
        print('error',error)
        assert error<1e-14

    def test_superposition_mixed_boundary_conditions(self):
        nx,ny=31,31
        degree=2
        kappa=dl.Constant(3)
        xl,xr,yb,yt=0.0,1.0,0.0,1.0

        """
        based upon the test in Fenics example ft_poisson_extended
        """
        x, y = sp.symbols('x[0], x[1]')             # needed by UFL
        u = 1 + x**2 + 2*y**2                       # exact solution
        u_e = u                                     # exact solution
        u_00 = u.subs(x, 0)                         # restrict to x = 0
        u_01 = u.subs(x, 1)                         # restrict to x = 1
        u_10 = u.subs(y, 0)                         # restrict to y = 0
        u_11 = u.subs(y, 1)                         # restrict to y = 1
        f = -sp.diff(u, x, 2) - sp.diff(u, y, 2)    # -Laplace(u)
        f = sp.simplify(f)                          # simplify f
        g_10 = sp.diff(u, y).subs(y, 0)            # compute g = -du/dn
        g_11 = -sp.diff(u, y).subs(y, 1)            # compute g = -du/dn

        # Collect variables
        variables = [u_e, u_00, u_01, u_10, u_11, f, g_10, g_11]
        
        # Turn into C/C++ code strings
        variables = [sp.printing.ccode(var) for var in variables]
        
        # Turn into FEniCS Expressions
        variables = [dl.Expression(var, degree=2) for var in variables]
        
        # Extract variables
        u_e, u_00, u_01, u_10, u_11, f, g_10, g_11 = variables

        # Extract variables
        u_e, u_00, u_01, u_10, u_11, f, g_10, g_11 = variables

        # Define boundary conditions
        bndry_objs = get_2d_rectangular_mesh_boundaries(xl,xr,yb,yt)
        boundary_conditions = [
            ['dirichlet',bndry_objs[0],u_00],     # x = 0
            ['dirichlet',bndry_objs[1],u_01],     # x = 1
            #['dirichlet',bndry_objs[2],u_10],    # y = 0
            #['dirichlet',bndry_objs[3],u_11]]    # y = 1
            ['robin',    bndry_objs[2],g_10,0.],  # y = 0
            ['neumann',  bndry_objs[3],g_11]]     # y = 1
        
        # Compute solution
        kappa = dl.Constant(1)

        mesh = dl.RectangleMesh(dl.Point(xl, yb),dl.Point(xr, yt),nx,ny)
        function_space=dl.FunctionSpace(mesh, "Lagrange", degree)
        sol = run_steady_state_model(
            function_space,kappa,f,
            boundary_conditions=boundary_conditions)
        #pp=dl.plot(sol)
        #plt.colorbar(pp); plt.show()

        error = dl.errornorm(u_e,sol,mesh=mesh)
        print('error',error)
        assert error<2e-12

        sols=[]
        for ii in range(4):
            bndry_objs = get_2d_rectangular_mesh_boundaries(xl,xr,yb,yt)
            boundary_conditions_ii = [
                ['dirichlet',bndry_objs[0],u_00],     # x = 0
                ['dirichlet',bndry_objs[1],u_01],     # x = 1
                #['dirichlet',bndry_objs[2],u_10],    # y = 0
                #['dirichlet',bndry_objs[3],u_11]]    # y = 1
                ['robin',    bndry_objs[2],0.,g_10 ], # y = 0
                ['neumann',  bndry_objs[3],g_11]]  # y = 1
        
            for jj in range(4):
                if jj!=ii:
                    #boundary_conditions_ii[jj]=['dirichlet',bndry_objs[jj],0]
                    boundary_conditions_ii[jj][2]=0
                    
            sol_ii = run_steady_state_model(
                function_space,kappa,dl.Constant(0.0),
                boundary_conditions=boundary_conditions_ii)
            sols.append(sol_ii)
            #plt.figure(ii+1)
            #pp=dl.plot(sol_ii)
            #plt.colorbar(pp);

            
        bndry_objs = get_2d_rectangular_mesh_boundaries(xl,xr,yb,yt)
        boundary_conditions_ii = [
            ['dirichlet',bndry_objs[0],0],     # x = 0
            ['dirichlet',bndry_objs[1],0],     # x = 1
            #['dirichlet',bndry_objs[2],0],    # y = 0
            #['dirichlet',bndry_objs[3],0]]    # y = 1
            ['robin',    bndry_objs[2],0.,0], # y = 0
            ['neumann',  bndry_objs[3],0]]  # y = 1

        sol_ii = run_steady_state_model(
            function_space,kappa,f,
            boundary_conditions=boundary_conditions_ii)
        sols.append(sol_ii)
        #plt.figure(ii+1)
        #pp=dl.plot(sol_ii)
        #plt.show()

        superposition_sol = sols[0]
        for ii in range(1,len(sols)):
            superposition_sol += sols[ii]
        #pp=dl.plot(superposition_sol)
        #plt.colorbar(pp); plt.show()
        superposition_sol = dl.project(superposition_sol,function_space)

        error = dl.errornorm(superposition_sol,sol,mesh=mesh)
        print('error',error)
        assert error<5e-12

class TestTransientAdvectionDiffusionEquation(unittest.TestCase):
    def test_maunfactured_solution_dirichlet_boundaries(self):

        # Define time stepping
        final_time = 0.7
        exact_sol_sympy = get_exact_solution_sympy(steady_state=False)[0]
        exact_sol = dl.Expression(
            sp.printing.ccode(exact_sol_sympy),t=0,degree=6)
        exact_sol.t=final_time

        # Create mesh
        def run(nx,degree):
            dt = final_time/nx
            mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(1, 1), nx, nx)
            # Define function spaces
            function_space = dl.FunctionSpace(mesh, "CG", degree)
            # Define initial condition
            initial_condition = dl.Constant(0.0)

            # Define velocity field
            beta = dl.Expression(('1.0','1.0'),cell=mesh.ufl_cell(),domain=mesh,
                                 degree=degree)
            # Define diffusivity field
            kappa = dl.Constant(1.0)

            # Define forcing
            forcing = get_forcing(
                kappa,mesh,degree,steady_state=False,advection=True)

            #boundary_conditions = \
            #    get_dirichlet_boundary_conditions_from_expression(
            #        get_exact_solution(mesh,degree),0,1,0,1)
            boundary_conditions=None
            sol = run_model(
                function_space,kappa,forcing,initial_condition,dt,final_time,
                boundary_conditions,velocity=beta,
                second_order_timestepping=True)
            return sol

        # use refined reference solution instead of exact solution
        #exact_sol=run(128,2)
        
        etypes, degrees, rates, errors = compute_convergence_rates(
            run,exact_sol,max_degree=1, num_levels=4)
        degree=1
        print(rates[degree]['L2 norm'])
        assert np.allclose(
            rates[degree]['L2 norm'][-1:],(degree+1)*np.ones(1),atol=1e-2)
        #for error_type in etypes:
        #    print('\n' + error_type)
        #    for degree in degrees:
        #        print('P%d: %s' %(degree, str(rates[degree][error_type])[1:-1]))
        
    def test_maunfactured_solution_dirichlet_boundaries_using_object(self):
        # Define time stepping
        final_time = 0.7
        exact_sol_sympy = get_exact_solution_sympy(steady_state=False)[0]
        exact_sol = dl.Expression(
            sp.printing.ccode(exact_sol_sympy),t=0,degree=6)
        exact_sol.t=final_time
        
        class NewModel(AdvectionDiffusionModel):
            def initialize_random_expressions(self,random_sample):
                init_condition = self.get_initial_condition(None)
                boundary_conditions, function_space = \
                    self.get_boundary_conditions_and_function_space(None)
                beta = self.get_velocity(None)
                forcing = self.get_forcing(None)
                kappa = self.get_diffusivity(None)
                return init_condition, boundary_conditions, function_space, \
                    beta, forcing, kappa
            
            def get_velocity(self,random_sample):
                assert random_sample is None
                beta = dl.Expression(
                    ('1.0','1.0'),cell=self.mesh.ufl_cell(),domain=self.mesh,
                    degree=self.degree)
                return beta

            def get_diffusivity(self,random_sample):
                assert random_sample is None
                return dl.Constant(1.0)
            
            def get_forcing(self,random_sample):
                assert random_sample is None
                kappa = self.get_diffusivity(None)
                return get_forcing(
                    kappa,self.mesh,self.degree,steady_state=False,
                    advection=True)
                
        
        def run(n,degree):
            nx=np.log2(n)-2
            model = NewModel(final_time,degree,qoi_functional_misc)
            samples = np.array([[nx,nx,nx]]).T
            sol = model.solve(samples)
            return sol
        etypes, degrees, rates, errors = compute_convergence_rates(
            run,exact_sol,max_degree=1, num_levels=4)
        degree=1
        print(rates[degree]['L2 norm'])
        assert np.allclose(
            rates[degree]['L2 norm'][-1:],(degree+1)*np.ones(1),atol=1e-2)

    def test_advection_diffusion_base_class(self):
        """
        Just check the benchmark runs
        """
        nvars,corr_len=2,0.1
        benchmark = setup_advection_diffusion_benchmark(
            nvars=nvars,corr_len=corr_len,max_eval_concurrency=1)
        model = benchmark.fun
        #random_samples = np.zeros((nvars,1))
        random_samples = -np.sqrt(3)*np.ones((nvars,1))
        config_samples = 3*np.ones((3,1))
        samples = np.vstack([random_samples,config_samples])
        sol = model.base_model.solve(samples)
        qoi = model.base_model(samples)
        print(qoi)
        assert np.isfinite(qoi)
        
        #dl.plot(sol)
        #plt.show()

    def test_advection_diffusion_source_inversion_model(self):
        """
        Just check the benchmark runs
        """
        benchmark = setup_advection_diffusion_source_inversion_benchmark(
            measurement_times=np.array([0.15]),source_strength=0.5,source_width=0.1)
        model = benchmark.fun
        #random_samples = np.zeros((nvars,1))
        random_samples = np.array([[0.25,0.75]]).T
        config_samples = 3*np.ones((3,1))
        samples = np.vstack([random_samples,config_samples])
        sol = model.base_model.solve(samples)
        print(sol.vector().get_local().max())
        
        p=dl.plot(sol)
        plt.colorbar(p)
        plt.show()

if __name__== "__main__":
    dl.set_log_level(40)
    transient_diffusion_test_suite=\
        unittest.TestLoader().loadTestsFromTestCase(TestTransientDiffusion)
    unittest.TextTestRunner(verbosity=2).run(transient_diffusion_test_suite)
    steady_state_diffusion_test_suite=\
        unittest.TestLoader().loadTestsFromTestCase(TestSteadyStateDiffusion)
    unittest.TextTestRunner(verbosity=2).run(steady_state_diffusion_test_suite)
    transient_advection_diffusion_equation_test_suite=\
        unittest.TestLoader().loadTestsFromTestCase(
            TestTransientAdvectionDiffusionEquation)
    unittest.TextTestRunner(verbosity=2).run(
        transient_advection_diffusion_equation_test_suite)

#TODO implement a test that has time varying dirichlet conditions and another with time varing alpha in robin conditions. Then write code to preassemble a when these two things are not time varying.
