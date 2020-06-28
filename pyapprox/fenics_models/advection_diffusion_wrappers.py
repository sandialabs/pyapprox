from pyapprox.fenics_models.advection_diffusion import *

def qoi_functional_misc(u):
    """
    Use the QoI from [JEGGIJNME2020]

    To reproduce adaptive multi index results use following 
    expr = dl.Expression(
        '1./(sigma*sigma*2*pi)*std::exp(-(std::pow(x[0]-xk,2)+std::pow(x[1]-yk,2))/sigma*sigma)',
        xk=0.3,yk=0.5,sigma=0.16,degree=2)

    The /sigma*sigma is an error it should be 1/(sigma*sigma)
    """
    expr = dl.Expression(
        '1./(sigma*sigma*2*pi)*std::exp(-(std::pow(x[0]-xk,2)+std::pow(x[1]-yk,2))/(sigma*sigma))',
        xk=0.3,yk=0.5,sigma=0.16,degree=2)
    qoi = dl.assemble(u*expr*dl.dx(u.function_space().mesh()))
    return np.asarray([qoi])

def get_misc_forcing(degree):
    """
    Use the forcing from [JEGGIJNME2020]
    """
    forcing = dl.Expression(
        '(1.5+cos(2*pi*t))*cos(x[0])',degree=degree,t=0)
    return forcing

def get_gaussian_source_forcing(degree,random_sample):
    forcing = dl.Expression(
        'A/(sig2*2*pi)*std::exp(-(std::pow(x[0]-xk,2)+std::pow(x[1]-yk,2))/(2*sig2))',xk=random_sample[0],yk=random_sample[1],sig2=0.05**2,A=2.,degree=degree)
    return forcing

def get_nobile_diffusivity(corr_len,degree,random_sample):
    nvars = random_sample.shape[0]
    path = os.path.abspath(os.path.dirname(__file__))
    if '2017' in dl.__version__:
        filename = os.path.join(
            path,"src,""nobile_diffusivity_fenics_class_2017.cpp")
    else:
        filename = os.path.join(
            path,"src","nobile_diffusivity_fenics_class.cpp")
    with open(filename,'r') as kappa_file:
        kappa_code=kappa_file.read()
    if '2017' in dl.__version__:
        kappa = dl.UserExpression(kappa_code,degree=degree)
    else:
        kappa = dl.CompiledExpression(
            dl.compile_cpp_code(kappa_code).NobileDiffusivityExpression(),
            degree=degree)
    kappa.initialize_kle(nvars,corr_len)

    if '2017' in dl.__version__:
        for ii in range(random_sample.shape[0]):
            kappa.set_random_sample(random_sample[ii],ii)
    else:
        kappa.set_random_sample(random_sample)
    return kappa

def get_default_velocity(degree,vel_vec):
    if vel_vec.shape[0]==2:
        beta = dl.Expression((str(vel_vec[0]),str(vel_vec[1])),degree=degree)
    else:
        beta = dl.Constant(velocity[0])
    return beta

def setup_dirichlet_and_periodic_boundary_conditions_and_function_space(
        degree,random_sample):
    assert random_sample is None
    pbc =  RectangularMeshPeriodicBoundary(1)
    function_space = dl.FunctionSpace(
        mesh, "CG", degree, constrained_domain=pbc)
    bndry_obj = dl.CompiledSubDomain(
        "on_boundary&&(near(x[0],0)||near(x[0],1))")
    boundary_conditions = [['dirichlet',bndry_obj,dl.Constant(0)]]
    return boundary_conditions

def setup_zero_flux_neumann_boundary_conditions(degree,random_sample):
    assert random_sample is None
    function_space = dl.FunctionSpace(mesh, "CG", degree)
    bndry_objs = get_2d_rectangular_mesh_boundaries(0,1,0,1)
    boundary_conditions = [
        ['neumann',  bndry_objs[0],dl.Constant(0)],
        ['neumann',  bndry_objs[1],dl.Constant(0)],
        ['neumann',  bndry_objs[2],dl.Constant(0)],
        ['neumann',  bndry_objs[3],dl.Constant(0)]]
    return boundary_conditions

class AdvectionDiffusionModel(object):
        

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # Change the following functions to modify governing equations
    #     initialize_random_expressions()
    #     get_initial_condition()
    #     get_boundary_conditions_and_function_space()
    #     get_velocity()
    #     get_forcing()
    #     get_diffusivity()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def initialize_random_expressions(self,random_sample):
        """
        Overide this class to split random_samples into the parts that effect
        the 5 random quantities
        """
        init_condition = self.get_initial_condition(None)
        boundary_conditions, function_space = \
            self.get_boundary_conditions_and_function_space(None)
        beta = self.get_velocity(None)
        forcing = self.get_forcing(None)
        kappa = self.get_diffusivity(random_sample)
        return init_condition, boundary_conditions, function_space, beta, \
            forcing, kappa
            

    def get_initial_condition(self,random_sample):
        """By Default the initial condition is deterministic and set to zero"""
        assert random_sample is None
        initial_condition=dl.Constant(0.0)
        return initial_condition

    def get_boundary_conditions_and_function_space(
            self,random_sample):
        """By Default the boundary conditions are deterministic, Dirichlet and 
           and set to zero"""
        assert random_sample is None
        function_space = dl.FunctionSpace(self.mesh, "CG", self.degree)
        boundary_conditions = None
        return boundary_conditions,function_space

    def get_velocity(self,random_sample):
        """By Default the advection is deterministic and set to zero"""
        assert random_sample is None
        beta = dl.Expression((str(0),str(0)),degree=self.degree)
        return beta

    def get_forcing(self,random_sample):
        """By Default the forcing is deterministic and set to 

        .. math:: (1.5+\cos(2\pi t))*cos(x_1)
        
        where :math:`t` is time and :math:`x_1` is the first spatial dimension.
        """
        forcing = get_misc_forcing(self.degree)
        return forcing

    def get_diffusivity(self,random_sample):
        """
        Use the random diffusivity specified in [JEGGIJNME2020].
        """
        kappa = get_nobile_diffusivity(
            self.options['corr_len'],self.degree,random_sample)
        return kappa

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # Change the following functions to modify mapping of discretization
    # parameters to mesh and timestep resolution
    #     get_timestep()
    #     get_mesh_resolution()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def get_timestep(self,dt_level):
        dt = self.final_time/2**(dt_level+2)
        return dt

    def get_mesh_resolution(self,mesh_levels):
        nx_level,ny_level = mesh_levels
        nx = 2**(nx_level+2)
        ny = 2**(ny_level+2)
        return nx,ny

    def get_mesh(self,resolution_levels):
        """The arguments to this function are the outputs of 
        get_degrees_of_freedom_and_timestep()"""
        nx,ny=np.asarray(resolution_levels,dtype=int)
        mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(1, 1), nx, ny)
        return mesh

    def set_num_config_vars(self):
        """
        Should be equal to the number of physical dimensions + 1 
        (for the temporal resolution)
        """
        self.num_config_vars=3

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # Do not change the following functions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def __init__(self,final_time,degree,qoi_functional,
                 second_order_timestepping=True,options={}):
        self.final_time=final_time
        self.qoi_functional=qoi_functional
        self.degree=degree
        self.second_order_timestepping=second_order_timestepping
        self.set_num_config_vars()
        self.options=options

    def solve(self,samples):
        """
        Run the simulation

        Notes
        -----
        Dolfin objects must be initialized inside this function otherwise 
        this object cannot be pickled and used with multiprocessing.Pool
        """
        assert samples.ndim==2
        assert samples.shape[1]==1

        resolution_levels = samples[-self.num_config_vars:,0]
        dt = self.get_timestep(resolution_levels[-1])
        self.mesh = self.get_mesh(
            self.get_mesh_resolution(resolution_levels[:-1]))
        
        random_sample = samples[:-self.num_config_vars,0]

        init_condition, boundary_conditions, function_space, beta, \
            forcing, kappa = self.initialize_random_expressions(random_sample)

        sol = run_model(
            function_space,kappa,forcing,
            init_condition,dt,self.final_time,
            boundary_conditions,velocity=beta,
            second_order_timestepping=self.second_order_timestepping)
        return sol
        
    def __call__(self,samples):
        sol = self.solve(samples)
        vals = np.atleast_1d(self.qoi_functional(sol))
        if vals.ndim==1:
            vals = vals[:,np.newaxis]
        return vals

def setup_advection_diffusion_benchmark(nvars,corr_len,max_eval_concurrency=1):
    r"""
    Compute functionals of the following model of transient advection-diffusion

    .. math::

       \frac{\partial u}{\partial t}(x,t,\rv) + \nabla u(x,t,\rv)-\nabla\cdot\left[k(x,\rv) \nabla u(x,t,\rv)\right] &=g(x,t) \qquad (x,t,\rv)\in D\times [0,1]\times\rvdom\\
       \mathcal{B}(x,t,\rv)&=0 \qquad\qquad (x,t,\rv)\in \partial D\times[0,1]\times\rvdom\\
       u(x,t,\rv)&=u_0(x,\rv) \qquad (x,t,\rv)\in D\times\{t=0\}\times\rvdom

    Following [NTWSIAMNA2008]_, [JEGGIJNME2020]_ we set 

    .. math:: g(x,t)=(1.5+\cos(2\pi t))\cos(x_1),

    the initial condition as :math:`u(x,z)=0`, :math:`B(x,t,z)` to be zero dirichlet boundary conditions.

    and we model the diffusivity :math:`k` as a random field represented by the
    Karhunen-Loeve (like) expansion (KLE)

    .. math::

       \log(k(x,\rv)-0.5)=1+\rv_1\left(\frac{\sqrt{\pi L}}{2}\right)^{1/2}+\sum_{k=2}^d \lambda_k\phi(x)\rv_k,

    with

    .. math::

       \lambda_k=\left(\sqrt{\pi L}\right)^{1/2}\exp\left(-\frac{(\lfloor\frac{k}{2}\rfloor\pi L)^2}{4}\right) k>1,  \qquad\qquad  \phi(x)=
       \begin{cases}
       \sin\left(\frac{(\lfloor\frac{k}{2}\rfloor\pi x_1)}{L_p}\right) & k \text{ even}\,,\\
       \cos\left(\frac{(\lfloor\frac{k}{2}\rfloor\pi x_1)}{L_p}\right) & k \text{ odd}\,.
       \end{cases}

    where :math:`L_p=\max(1,2L_c)`, :math:`L=\frac{L_c}{L_p}`.

    Parameters
    ----------
    nvars : integer
        The number of variables of the KLE

    corr_len : float
        The correlation length :math:`L_c` of the covariance kernel

    max_eval_concurrency : integer
        The maximum number of simulations that can be run in parallel. Should be         no more than the maximum number of cores on the computer being used

    Returns
    --------
    benchmark : pya.Benchmark
       Object containing the benchmark attributes
    
    """
    
    from scipy import stats
    from pyapprox.models.wrappers import TimerModelWrapper, PoolModel, \
        WorkTrackingModel
    from pyapprox.models.wrappers import PoolModel
    from pyapprox.variables import IndependentMultivariateRandomVariable
    from pyapprox.benchmarks.benchmarks import Benchmark
    univariate_variables = [stats.uniform(-np.sqrt(3),2*np.sqrt(3))]*nvars
    variable=IndependentMultivariateRandomVariable(univariate_variables)
    final_time, degree = 1.0,1
    options={'corr_len':corr_len}
    base_model = AdvectionDiffusionModel(
        final_time,degree,qoi_functional_misc,second_order_timestepping=False,
        options=options)
    # add wrapper to allow execution times to be captured
    timer_model = TimerModelWrapper(base_model,base_model)
    pool_model=PoolModel(timer_model,max_eval_concurrency,base_model=base_model)
    # add wrapper that tracks execution times.
    model = WorkTrackingModel(pool_model,base_model)
    attributes = {'fun':model,'variable':variable}
    return Benchmark(attributes)
     
