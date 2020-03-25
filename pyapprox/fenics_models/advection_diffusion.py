import dolfin as dl
import numpy as np
from pyapprox.fenics_models.fenics_utilities import *
import os, time

def run_model(function_space,kappa,forcing,init_condition,dt,final_time,
              boundary_conditions=None,second_order_timestepping=False,
              exact_sol=None,velocity=None,point_sources=None):
    """
    Use implicit euler to solve transient advection diffusion equation

    du/dt = grad (k* grad u) - vel*grad u + f

    WARNINGarningW: when point sources solution changes significantly when mesh is varied
    """
    mesh = function_space.mesh()

    time_independent_boundaries=False
    if boundary_conditions==None:
        bndry_obj  = dl.CompiledSubDomain("on_boundary")
        boundary_conditions = [['dirichlet',bndry_obj,dl.Constant(0)]]
        time_independent_boundaries=True

    num_bndrys = len(boundary_conditions)
    boundaries = mark_boundaries(mesh,boundary_conditions)
    dirichlet_bcs = collect_dirichlet_boundaries(
        function_space,boundary_conditions, boundaries)

    # To express integrals over the boundary parts using ds(i), we must first
    # redefine the measure ds in terms of our boundary markers:
    ds = dl.Measure('ds', domain=mesh, subdomain_data=boundaries)
    dx = dl.Measure('dx', domain=mesh)

    # Variational problem at each time
    u = dl.TrialFunction(function_space)
    v = dl.TestFunction(function_space)

    # Previous solution
    if hasattr(init_condition,'t'):
        assert init_condition.t==0
    u_1=dl.interpolate(init_condition,function_space)

    if not second_order_timestepping:
        theta=1
    else:
        theta=0.5

    if hasattr(forcing,'t'):
        forcing_1 = copy_expression(forcing)
    else:
        forcing_1 = forcing

    def steady_state_form(u,v,f):
        F =  kappa*dl.inner(dl.grad(u), dl.grad(v))*dx
        F -= f*v*dx
        if velocity is not None:
            F += dl.dot(velocity,dl.grad(u))*v*dx
        return F

    F =  u*v*dx-u_1*v*dx + dt*theta*steady_state_form(u,v,forcing) + \
         dt*(1.-theta)*steady_state_form(u_1,v,forcing_1)
    a,L = dl.lhs(F),dl.rhs(F)

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
        if (boundary_conditions[ii][0]=='robin'):
            alpha = boundary_conditions[ii][3]
            a += theta*dt*alpha*u*v*ds(ii)
            if second_order_timestepping:
                if hasattr(alpha,'t'):
                    alpha_1 = copy_expression(alpha)
                    alpha_1_list.append(alpha_1)
                else:
                    alpha_1=alpha
                L -= (1-theta)*dt*alpha_1*u_1*v*ds(ii)

        if ((boundary_conditions[ii][0]=='robin') or
            (boundary_conditions[ii][0]=='neumann')):
            beta = boundary_conditions[ii][2]
            L -= theta*dt*beta*v*ds(ii)
            if second_order_timestepping:
                if hasattr(beta,'t'):
                    beta_1 = copy_expression(beta)
                    beta_1_list.append(beta_1)
                else:
                    # boundary condition is constant in time
                    beta_1=beta
                L -= (1-theta)*dt*beta_1*v*ds(ii)

    if time_independent_boundaries:
        # TODO this can be used if dirichlet and robin conditions are not
        # time dependent.
        A = dl.assemble(a);
        for bc in dirichlet_bcs:
            bc.apply(A)
        solver = dl.LUSolver(A)
        #solver.parameters["reuse_factorization"] = True
    else:
        solver=None


    u_2=dl.Function(function_space)
    u_2.assign(u_1)
    t=0.0

    dt_tol=1e-12
    n_time_steps=0
    while t < final_time-dt_tol:
        # Update current time
        t += dt
        forcing.t=t
        forcing_1.t=t-dt

        # set current time for time varying boundary conditions
        for ii in range(num_bndrys):
            if hasattr(boundary_conditions[ii][2],'t'):
                boundary_conditions[ii][2].t=t

        # set previous time for time varying boundary conditions when
        # using second order timestepping. lists will be empty if using
        # first order timestepping
        for jj in range(len(beta_1_list)):
            beta_1_list[jj].t=t-dt
        for jj in range(len(alpha_1_list)):
            alpha_1_list[jj].t=t-dt

        #A, b = dl.assemble_system(a, L, dirichlet_bcs)
        #for bc in dirichlet_bcs:
        #    bc.apply(A,b)
        if boundary_conditions is not None:
            A = dl.assemble(a);
            for bc in dirichlet_bcs:
                bc.apply(A)
                
        b = dl.assemble(L)
        for bc in dirichlet_bcs:
            bc.apply(b)

        if point_sources is not None:
            ps_list = []
            for ii in range(len(point_sources)):
                point,expr = point_sources[ii]
                ps_list.append((dl.Point(point[0],point[1]), expr(t)))
            ps = dl.PointSource(function_space,ps_list)
            ps.apply(b)
                
        if solver is None:
            dl.solve(A, u_2.vector(), b)
        else:
            solver.solve(u_2.vector(), b)

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
            exact_sol.t=t
            error = dl.errornorm(exact_sol,u_2)
            print('t = %.2f: error = %.3g' % (t, error))
            #dl.plot(exact_sol,mesh=mesh)
            #plt.show()

        t = min(t,final_time)
        n_time_steps+=1
    #print ("t =", t, "end t=", final_time,"# time steps", n_time_steps)
    

    return u_2

def run_steady_state_model(function_space,kappa,forcing,
                           boundary_conditions=None,velocity=None):
    """
    Solve steady-state diffusion equation

    -grad (k* grad u) = f
    """
    mesh = function_space.mesh()
    
    if boundary_conditions==None:
        bndry_obj  = dl.CompiledSubDomain("on_boundary")
        boundary_conditions = [['dirichlet',bndry_obj,dl.Constant(0)]]

    num_bndrys = len(boundary_conditions)
    boundaries = mark_boundaries(mesh,boundary_conditions)
    dirichlet_bcs = collect_dirichlet_boundaries(
        function_space,boundary_conditions, boundaries)

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
        a += v*dl.dot(velocity,dl.grad(u))*dx

    beta_1_list = []
    alpha_1_list = []
    for ii in range(num_bndrys):
        if (boundary_conditions[ii][0]=='robin'):
            alpha = boundary_conditions[ii][3]
            a += alpha*u*v*ds(ii)

        if ((boundary_conditions[ii][0]=='robin') or
            (boundary_conditions[ii][0]=='neumann')):
            beta = boundary_conditions[ii][2]
            L -= beta*v*ds(ii)

    u=dl.Function(function_space)
    A, b = dl.assemble_system(a, L, dirichlet_bcs)
    # apply boundary conditions
    for bc in dirichlet_bcs:
        bc.apply(A, b)
    dl.solve(A, u.vector(), b)

    return u

class Diffusivity(dl.UserExpression):
    #def __init__(self,**kwargs):
    #    if '2019' in dl.__version__:
    #        # does not work for fenics 2017 only 2019
    #        super().__init__(**kwargs)

    def initialize_kle(self,num_vars,corr_len):
        domain_len = 1
        self.num_vars = num_vars
        self.sqrtpi = np.sqrt(np.pi)
        self.Lp = max(domain_len,2*corr_len)
        self.L = corr_len/self.Lp
        self.nn = np.arange(2,num_vars+1)
        self.eigenvalues = np.sqrt(self.sqrtpi*self.L)*np.exp(
            -((np.floor(self.nn/2)*np.pi*self.L))**2/4)
        #-((np.floor(self.nn/2)*np.pi*self.L))**2/8)#nobile eigvals
        self.eigenvectors = np.empty((self.num_vars-1),dtype=float)

    def set_random_sample(self,sample):
        assert sample.shape[0]==self.num_vars
        self.sample=sample

    def eval(self, values, x):
        self.eigenvectors[::2] = np.sin(
            ((np.floor(self.nn[::2]/2)*np.pi*x[0]))/self.Lp)
        self.eigenvectors[1::2] = np.cos(
            ((np.floor(self.nn[1::2]/2)*np.pi*x[0]))/self.Lp)
        self.eigenvectors *= self.eigenvalues
        
        values[0] =  self.eigenvectors.dot(self.sample[1:])
        values[0] += 1+self.sample[0]*np.sqrt(self.sqrtpi*self.L/2)
        values[0] =  np.exp(values[0])+0.5

class AdvectionDiffusionModel(object):
    def __init__(self,num_vars,corr_len,final_time,degree,qoi_functional,
                 add_work_to_qoi=False, periodic_boundary=False,
                 num_phys_dims=2,second_order_timestepping=True,
                 parameterized_forcing=False,velocity=None):

        self.num_phys_dims=num_phys_dims
        self.num_config_vars=3
        self.num_vars=num_vars
        self.corr_len = corr_len
        self.final_time=final_time
        self.qoi_functional=qoi_functional
        self.degree=degree
        self.add_work_to_qoi=add_work_to_qoi
        self.periodic_boundary=periodic_boundary
        self.second_order_timestepping=second_order_timestepping
        # forcing is gaussian bump with random location. The forcing
        # is steady state so after 1 timestep with implicit timestepping
        # the answer should not change
        self.parameterized_forcing = parameterized_forcing
        self.velocity=velocity

        path = os.path.abspath(os.path.dirname(__file__))
        if '2017' in dl.__version__:
            filename = os.path.join(
                path,"src,""nobile_diffusivity_fenics_class_2017.cpp")
        else:
            filename = os.path.join(
                path,"src","nobile_diffusivity_fenics_class.cpp")
        with open(filename,'r') as kappa_file:
            self.kappa_code=kappa_file.read()

    def get_timestep(self,dt_level):
        dt_level = 2**(-(dt_level+2))
        return dt_level

    def get_mesh_resolution(self,nx_level,ny_level):
        nx = 2**(nx_level+2)
        ny = 2**(ny_level+2)
        return nx,ny

    def get_degrees_of_freedom_and_timestep(self,config_sample):
        assert config_sample.ndim==1
        assert config_sample.shape[0]==self.num_config_vars
        nx_level,ny_level,dt_level = config_sample
        #assert nx_level.is_integer()
        #assert ny_level.is_integer()
        #assert dt_level.is_integer()
        nx_level = int(nx_level)
        ny_level = int(ny_level)
        dt_level = int(dt_level)
        dt = self.get_timestep(dt_level)
        nx,ny = self.get_mesh_resolution(nx_level,ny_level)
        return nx,ny,dt
        
    def __call__(self,samples):
        # have to have dl objects in call otherwise model cannot be pickled
        # and used with multiprocessing.Pool
        if self.velocity is None and self.num_phys_dims==2:
            beta = dl.Expression(('1.0','1.0'),degree=self.degree)
        elif self.velocity is None:
            beta = dl.Constant(1.0)
        elif self.num_phys_dims==2:
            beta = dl.Expression(
                (str(self.velocity[0]),str(self.velocity[1])),degree=self.degree)
        else:
            beta = dl.Constant(self.velocity)
            
        if '2017' in dl.__version__:
            kappa = dl.UserExpression(self.kappa_code,degree=self.degree)
        else:
            kappa = dl.CompiledExpression(
                dl.compile_cpp_code(
                    self.kappa_code).NobileDiffusivityExpression(),
                degree=self.degree)
        if not self.parameterized_forcing:
            kappa.initialize_kle(self.num_vars,self.corr_len)
        else:
            kappa.initialize_kle(max(1,self.num_vars-2),self.corr_len)
            

        if not self.parameterized_forcing:
            forcing = dl.Expression(
                '(1.5+cos(2*pi*t))*cos(x[0])',degree=self.degree,t=0)
            
        initial_condition=dl.Constant(0.0)
        
        assert samples.ndim==2
        vals = []
        for ii in range(samples.shape[1]):
            nx,ny,dt = self.get_degrees_of_freedom_and_timestep(
                samples[-3:,ii])
            if self.num_phys_dims==2:
                mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(1, 1), nx, ny)
            else:
                mesh = dl.UnitIntervalMesh(nx)
            if self.periodic_boundary:
                pbc =  RectangularMeshPeriodicBoundary(1)
                function_space = dl.FunctionSpace(
                    mesh, "CG", self.degree, constrained_domain=pbc)
                bndry_obj = dl.CompiledSubDomain(
                    "on_boundary&&(near(x[0],0)||near(x[0],1))")
                boundary_conditions = [['dirichlet',bndry_obj,dl.Constant(0)]]
            else:
                function_space = dl.FunctionSpace(mesh, "CG", self.degree)
                boundary_conditions = None

            random_sample = samples[:-3,ii]

            if self.parameterized_forcing:
                #assert dt==self.final_time
                forcing = dl.Expression(
                    '1./(sigma*sigma*2*pi)*std::exp(-(std::pow(x[0]-xk,2)+std::pow(x[1]-yk,2))/sigma*sigma)',xk=random_sample[0],yk=random_sample[1],sigma=0.16,degree=self.degree)
                random_sample = random_sample[2:]
                if random_sample.shape[0]==0:
                    random_sample = np.array([0.])
            
            if '2017' in dl.__version__:
                for ii in range(random_sample.shape[0]):
                    kappa.set_random_sample(random_sample[ii],ii)
            else:
                kappa.set_random_sample(random_sample)

            t0=time.time()
            sol = run_model(
                function_space,kappa,forcing,
                initial_condition,dt,self.final_time,
                boundary_conditions,velocity=beta,
                second_order_timestepping=self.second_order_timestepping)

            work=time.time()-t0
            
            vals.append(self.qoi_functional(sol))
            ndofs = function_space.dim()
            ntimesteps = self.final_time/dt

            if self.add_work_to_qoi:
                if type(vals[-1])==np.ndarray:
                    vals[-1]=np.concatenate((vals[-1],[work,ndofs,ntimesteps]))
                elif type(vals[-1])==list:
                    vals[-1]+=[work,ndofs,ntimesteps]
                else:
                    raise Exception()
                
        if type(vals[0])==np.ndarray:
            vals = np.asarray(vals)
        return vals

def qoi_functional_1(u):
    return np.asarray([u([0.5,0.5])])

def qoi_functional_2(u):
    return [u]

def qoi_functional_misc(u):
    expr = dl.Expression(
        '1./(sigma*sigma*2*pi)*std::exp(-(std::pow(x[0]-xk,2)+std::pow(x[1]-yk,2))/sigma*sigma)',
        xk=0.3,yk=0.5,sigma=0.16,degree=2)
    qoi = dl.assemble(u*expr*dl.dx(u.function_space().mesh()))
    return np.asarray([qoi])

#-------------#
# useful code #
#-------------#

# # Report flux across left boundary
# # this will be zero for cosine analytical examples because integral
# # of one period sine function is zero
# n = FacetNormal(mesh)
# flux = assemble(kappa*dot(grad(u), n)*ds(0))
# info('t = %g, flux = %e' % (t, flux))
            
