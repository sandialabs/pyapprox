import numpy as np
from scipy.optimize import minimize, Bounds
from functools import partial
from scipy.stats import gaussian_kde as KDE
from pyapprox.configure_plots import *
import scipy.stats as ss
from pyapprox.utilities import get_all_sample_combinations

def approx_jacobian(func,x,*args,epsilon=np.sqrt(np.finfo(float).eps)):
    x0 = np.asfarray(x)
    assert x0.ndim==1
    f0 = np.atleast_1d(func(*((x0,)+args)))
    if f0.ndim==2:
        assert f0.shape[1]==1
        f0 = f0[:,0]
    jac = np.zeros([len(x0),len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        f1 = func(*((x0+dx,)+args))
        if f1.ndim==2:
            assert f1.shape[1]==1
            f1 = f1[:,0]
        jac[i] = (f1 - f0)/epsilon
        dx[i] = 0.0

    return jac.transpose()
    

def eval_function_at_multiple_design_and_random_samples(function,uq_samples,design_samples):
    """
    for functions which only take 1d arrays for uq_samples and design_samples
    loop over all combinations and evaluate function at each combination

    design_samples vary slowest and uq_samples vary fastest

    Let design samples = [[1,2],[2,3]]
    uq_samples = [[0, 0, 0],[0, 1, 2]]
    Then samples will be

    ([1, 2], [0, 0, 0])
    ([1, 2], [0, 1, 2])
    ([3, 4], [0, 0, 0])
    ([3, 4], [0, 1, 2])

    function(uq_samples,design_samples)
    """
    vals = []
    # put design samples first so that samples iterates over uq_samples fastest
    samples = get_all_sample_combinations(design_samples,uq_samples)
    for xx,zz in zip(
            samples[:design_samples.shape[0]].T,
            samples[design_samples.shape[0]:].T):
        # flip xx,zz because functions assumed to take uq_samples then
        # design_samples
        vals.append(function(zz,xx))
    return np.asarray(vals)

def eval_mc_based_jacobian_at_multiple_design_samples(grad,stat_func,
                                                      uq_samples,design_samples):
    """
    Alternatively I could use
    jacobian = [np.mean([constraint_grad_single(z,x) for z in zz.T],axis=0) for x in xx.T]
    But I think this implementation will allow better use of concurent evaluations in the 
    future. For example eval_function_at_multiple_design_and_random_samples could
    utilize an asynchronous call over all the sample combinations

    TODO combine uq_samples and design samples into one matrix and assume functions
    always take a single matrix and not two matrices
    """
    grads = eval_function_at_multiple_design_and_random_samples(
        grad,uq_samples,design_samples)
    
    ndesign_samples = design_samples.shape[1]
    nuq_samples = uq_samples.shape[1]
    jacobian = np.array(
        [stat_func(grads[ii*nuq_samples:(ii+1)*nuq_samples])
         for ii in range(ndesign_samples)])
    return jacobian

def check_inputs(uq_samples,design_samples):
    if design_samples.ndim==1:
        design_samples = design_samples[:,np.newaxis]
    if uq_samples is not None and uq_samples.ndim==1:
        uq_samples = design_samples[:,np.newaxis]
    if (uq_samples is not None and
        (design_samples.shape[1]>1 and uq_samples.shape[1]>1)):
        assert design_samples.shape[1]==uq_samples.shape[1]
    return uq_samples,design_samples

def deterministic_lower_bound_constraint(constraint_function,lower_bound,
                                         uq_samples,design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    assert design_samples.shape[1]==1
    val = lower_bound-constraint_function(uq_samples,design_samples)
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce upper bound
    return -val

def variance_lower_bound_constraint(constraint_function,lower_bound,uq_samples,
                                    design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    assert design_samples.shape[1]==1
    # scipy minimize enforces constraints are non-negative
    vals = constraint_function(uq_samples,design_samples)
    val = lower_bound-np.std(vals)**2
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce upper bound
    return -val

def mean_lower_bound_constraint(constraint_function,lower_bound,uq_samples,
                                design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    assert design_samples.shape[1]==1
    # scipy minimize enforces constraints are non-negative
    vals = constraint_function(uq_samples,design_samples)
    val = lower_bound-np.mean(vals)**2
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce upper bound
    return -val

def mean_lower_bound_constraint_jacobian(constraint_function_jacobian,uq_samples,
                                         design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    assert design_samples.shape[1]==1
    # scipy minimize enforces constraints are non-negative
    vals = constraint_function_jacobian(uq_samples,design_samples)
    val = -np.mean(vals)**2
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce upper bound
    return -val

def quantile_lower_bound_constraint(constraint_function,quantile,lower_bound,
                                    uq_samples,design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    assert design_samples.shape[1]==1
    vals = constraint_function(uq_samples,design_samples)
    val = (lower_bound-ss.mstats.mquantiles(vals,prob=[quantile]))
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce lower bound
    return -val

from pyapprox.cvar_regression import smoothed_conditional_value_at_risk, \
    conditional_value_at_risk
def cvar_lower_bound_constraint(constraint_function,quantile,lower_bound,eps,
                                uq_samples,design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    assert design_samples.shape[1]==1
    vals = constraint_function(uq_samples,design_samples)
    # -vals because we want to minimize lower tail
    val = (lower_bound-smoothed_conditional_value_at_risk(-vals,quantile,eps))
    #val = (lower_bound-conditional_value_at_risk(-vals,quantile))
    return val

class MultipleConstraints(object):
    def __init__(self,constraints):
        self.constraints=constraints

    def __call__(self,design_sample,constraint_idx=None):
        if constraint_idx is None:
            constraint_idx = np.arange(len(self.constraints))
        nconstraints = len(constraint_idx)
        vals = np.empty(nconstraints)
        for ii,jj in enumerate(constraint_idx):
            vals[ii]=self.constraints[jj](design_sample)
        return vals

class MCStatisticConstraint(object):
    def __init__(self,constraint_function,generate_samples,info):
        self.constraint_function = constraint_function
        self.generate_samples=generate_samples
        self.info=info

    def __call__(self,design_samples):
        uq_samples = self.generate_samples()
        constraint_type=self.info['type']
        if constraint_type=='quantile':
            quantile = self.info['quantile']
            lower_bound = self.info['lower_bound']
            return quantile_lower_bound_constraint(
                self.constraint_function,quantile,lower_bound,
                uq_samples,design_samples)
        elif constraint_type=='cvar':
            quantile = self.info['quantile']
            lower_bound = self.info['lower_bound']
            eps = self.info['smoothing_eps']
            return cvar_lower_bound_constraint(
                constraint_functions[ii], quantile, lower_bound, eps,
                uq_samples, design_samples)
        elif constraint_type=='var':
            var_lower_bound = self.info['lower_bound']
            return variance_lower_bound_constraint(
                constraint_functions[ii],lower_bound,uq_samples,design_samples)
        else:
            raise Exception(
                'constraint type (%s) not implemented'%constraint_type[ii])

class DeterministicConstraint(object):
    def __init__(self,constraint_function,info):
        self.constraint_function = constraint_function
        self.info=info

    def __call__(self,design_samples):
        lower_bound = self.info['lower_bound']
        uq_nominal_sample = self.info['uq_nominal_sample']
        return deterministic_lower_bound_constraint(
            self.constraint_function,lower_bound,uq_nominal_sample,
            design_samples)
    
def setup_inequality_constraints(constraint_functions,constraints_info,
                                 uq_samples):
    constraints = []
    for ii in range(len(constraint_functions)):
        info = constraints_info[ii]
        constraint_type = info['type']
        if constraint_type=='quantile':
            quantile = info['quantile']
            quantile_lower_bound = info['quantile_lower_bound']
            ineq_cons_fun = partial(
                quantile_lower_bound_constraint, constraint_functions[ii],
                quantile, quantile_lower_bound, uq_samples)
        elif constraint_type=='cvar':
            quantile = info['quantile']
            quantile_lower_bound = info['cvar_lower_bound']
            eps = info['smoothing_eps']
            ineq_cons_fun = partial(
                cvar_lower_bound_constraint, constraint_functions[ii],
                quantile, quantile_lower_bound, eps, uq_samples)
        elif constraint_type=='var':
            var_lower_bound = info['var_lower_bound']
            ineq_cons_fun = partial(
               variance_lower_bound_constraint, constraint_functions[ii],
               var_lower_bound, uq_samples)
        elif constraint_type=='deterministic':
            lower_bound = info['lower_bound']
            ineq_cons_fun = partial(
                deterministic_lower_bound_constraint, constraint_functions[ii],
                lower_bound, uq_samples)
        else:
            raise Exception(
                'constraint type (%s) not implemented'%constraint_type[ii])
        ineq_cons = {'type': 'ineq', 'fun' : ineq_cons_fun}
        constraints.append(ineq_cons)
    return constraints

def run_design(objective, init_design_sample,
               constraints, bounds, optim_options):

    opt_history = [init_design_sample[:,0]]
    def callback(xk):
        opt_history.append(xk)
        #print(objective(xk))
        #print([constraints[ii]['fun'](xk) for ii in [0,1]])

    # opt_method = 'SLSQP'
    # res = minimize(
    #     objective, init_design_sample[:,0], method=opt_method, jac=None,
    #     constraints=constraints,
    #     options=optim_options,bounds=bounds,callback=callback)

    from scipy.optimize import fmin_slsqp
    res = fmin_slsqp(objective, init_design_sample[:,0], f_ieqcons=constraints,
                     bounds=bounds, callback=callback, full_output=True)#, **optim_options)
    class result():
        def __init__(self,x,fun):
            self.x=np.atleast_1d(x)
            self.fun=fun
    res = result(res[0],res[1])

    opt_history = (np.array(opt_history)).T
    return res, opt_history

def plot_optimization_history(obj_function,constraints,uq_samples,opt_history,
                              plot_limits):

    # fig,axs=plot_optimization_objective_and_constraints_2D(
    #     [constraints[ii]['fun'] for ii in range(len(constraints))],
    #     partial(obj_function,uq_samples[:,0]),plot_limits)

    fig,axs=plot_optimization_objective_and_constraints_2D(
        constraints,partial(obj_function,uq_samples[:,0]),plot_limits)
    # objective can only be evaluated at one uq_sample thus use of
    # uq_samples[:,0]
    
    for ii in range(len(axs)):
        axs[ii].plot(opt_history[0,:],opt_history[1,:],'ko')
        for jj, txt in enumerate(range(opt_history.shape[1])):
            axs[ii].annotate(
                '%d'%txt,(opt_history[0,jj],opt_history[1,jj]))
    return fig,axs

#def plot_optimization_objective_and_constraints_2D(
#        constraint_functions,objective,plot_limits):

def plot_optimization_objective_and_constraints_2D(
        constraints,objective,plot_limits):
    from pyapprox.visualization import get_meshgrid_function_data
    num_pts_1d = 100; num_contour_levels=30
    fig,axs=plt.subplots(1,3,figsize=(3*8,6))
    #for ii in range(len(constraint_functions)+1):
    for ii in range(len(constraints.constraints)+1):

        #if ii==len(constraint_functions):
        if ii==len(constraints.constraints):
            function=objective
        else:
            # def function(design_samples):
            #     vals = np.empty((design_samples.shape[1]))
            #     for jj in range(design_samples.shape[1]):
            #         vals[jj]=constraint_functions[ii](design_samples[:,jj])
            #     return vals
            def function(design_samples):
                vals = np.empty((design_samples.shape[1]))
                for jj in range(design_samples.shape[1]):
                    vals[jj]=constraints(design_samples[:,jj],[ii])
                return vals
        
        X,Y,Z = get_meshgrid_function_data(
            function, plot_limits, num_pts_1d)
        norm = None
        cset = axs[ii].contourf(
            X, Y, Z, levels=np.linspace(Z.min(),Z.max(),num_contour_levels),
            cmap=mpl.cm.coolwarm,
            norm=norm)
        #for kk in range(len(constraint_functions)):
        for kk in range(len(constraints.constraints)):
            if ii==kk:
                ls = '-'
            else:
                ls = '--'
            axs[kk].contour(X,Y,Z,levels=[0],colors='k',linestyles=ls)
        plt.colorbar(cset,ax=axs[ii])

    return fig,axs

def plot_constraint_pdfs(constraint_functions,uq_samples,design_sample,
                         fig_pdf=None,axs_pdf=None,label=None,color=None):
    colors = ['b','gray']
    nconstraints = len(constraint_functions)
    if axs_pdf is None:
        fig_pdf,axs_pdf = plt.subplots(1,nconstraints,figsize=(nconstraints*8,6))
    for ii in range(nconstraints):
        # evaluate constraint function at each of the uq samples
        constraint_function_vals = constraint_functions[ii](
            uq_samples,design_sample)

        constraint_kde = KDE(constraint_function_vals)
        yy = np.linspace(constraint_function_vals.min(),
                         constraint_function_vals.max(),101)

        axs_pdf[ii].fill_between(yy,0,constraint_kde(yy),alpha=0.5,label=label,
                                 color=color)
        axs_pdf[ii].axvline(0,color='k')
        #axs_pdf[ii].axvline(constraints[ii]['fun'](design_sample),color='r')
    return fig_pdf,axs_pdf
    
def plot_constraint_cdfs(constraints,constraint_functions,uq_samples,
                         design_sample,quantile,fig_cdf,axs_cdf=None,label=None,
                         color=None):
    nconstraints = len(constraint_functions)
    if axs_cdf is None:
        fig_cdf,axs_cdf = plt.subplots(
            1,nconstraints,figsize=(nconstraints*8,6))

    for ii in range(nconstraints):
        constraint_function_vals = constraint_functions[ii](
            uq_samples,design_sample)

        cvar = (conditional_value_at_risk(-constraint_function_vals,0.9))
        cvars = (smoothed_conditional_value_at_risk(-constraint_function_vals,0.9,1e-3))
        print ('cvar',cvar)
        print ('cvars',cvars)
        #constraint_val = constraints[ii]['fun'](design_sample)
        constraint_val = constraints(design_sample,[ii])
        constraint_function_vals.sort()
        cdf_vals = np.linspace(0,1,constraint_function_vals.shape[0]+1)[1:]
        axs_cdf[ii].plot(constraint_function_vals,cdf_vals,label=label,
                         color=color)
        #I = np.where(constraint_function_vals<=constraint_val)[0]
        I = np.where(constraint_function_vals<=0)[0]
        axs_cdf[ii].fill_between(
            constraint_function_vals[I],0,cdf_vals[I],alpha=0.5,color=color)
        axs_cdf[ii].axvline(0,color='k')
        J = np.where(constraint_function_vals<=0)[0]
        #print (J.shape[0]/float(constraint_function_vals.shape[0]),'p failure',constraint_val,J.shape[0])
        # Compute the constraint value. This combines constraint_function_vals
        # into a scalar value
        #axs_cdf[ii].axvline(constraint_val,color='r')
        #axs_cdf[ii].plot(
        #    np.linspace(constraint_function_vals[0],constraint_val,101),
        #    quantile*np.ones(101),'-r')
        #axs_cdf[ii].set_yticks(list(axs_cdf[ii].get_yticks()) + [quantile])
        axs_cdf[ii].set_ylim(0,1.05)
        axs_cdf[ii].set_xlim(
            constraint_function_vals[0],constraint_function_vals[-1])
    return fig_cdf,axs_cdf

def check_gradients(function,grad_function,xx,plot=False,disp=True):
    assert xx.ndim==1
    if callable(grad_function):
        function_val = function(xx)
        grad_val = grad_function(xx)
    elif grad_function==True:
        function_val,grad_val = function(xx)
    direction = np.random.normal(0,1,xx.shape[0])
    direction /= np.linalg.norm(direction)
    directional_derivative = grad_val.dot(direction)
    fd_eps = np.logspace(-13,0,14)[::-1]
    errors = []
    row_format = "{:<25} {:<25} {:<25}"
    if disp:
        print(row_format.format("Eps","Errors (max)","Errors (min)"))
    for ii in range(fd_eps.shape[0]):
        xx_perturbed = xx.copy()+fd_eps[ii]*direction
        perturbed_function_val = function(xx_perturbed)
        if grad_function==True:
            perturbed_function_val = perturbed_function_val[0]
        fd_directional_derivative = (
            perturbed_function_val-function_val)/fd_eps[ii]
        errors.append(np.absolute(
            fd_directional_derivative-directional_derivative))
        if disp:
            print(row_format.format(fd_eps[ii],errors[ii].max(),
                                    errors[ii].min()))
            #print(fd_directional_derivative,directional_derivative)

    if plot:
        plt.loglog(fd_eps,errors,'o-')
        plt.ylabel(r'$\lvert\nabla_\epsilon f-\nabla f\rvert$')
        plt.xlabel(r'$\epsilon$')
        plt.show()

    return np.asarray(errors)

import scipy.sparse as sp
from scipy.optimize import LinearConstraint, NonlinearConstraint, BFGS, linprog, OptimizeResult
def basis_pursuit(Amat,bvec,options):
    nunknowns = Amat.shape[1]
    nslack_variables = nunknowns

    c = np.zeros(nunknowns+nslack_variables)
    c[nunknowns:] = 1.0

    I = sp.identity(nunknowns)
    tmp = np.array([[1,-1],[-1,-1]])
    A_ub = sp.kron(tmp,I)
    b_ub = np.zeros(nunknowns+nslack_variables)

    A_eq = sp.lil_matrix((Amat.shape[0],c.shape[0]),dtype=float)
    A_eq[:,:Amat.shape[1]] = Amat
    b_eq = bvec

    bounds = [(-np.inf,np.inf)]*nunknowns + [(0,np.inf)]*nslack_variables
    
    res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options=options)
    
    return res.x[:nunknowns]
    

def nonlinear_basis_pursuit(func,func_jac,func_hess,init_guess,options):
    method = 'trust-constr'
    nunknowns = init_guess.shape[0]
    nslack_variables = nunknowns
    def obj(x):
        val = np.sum(x[nunknowns:])
        grad = np.zeros(x.shape[0])
        grad[nunknowns:]=1.0
        return val, grad
    def hessp(x,p):
        matvec = np.zeros(x.shape[0])
        return matvec
    
    I = sp.identity(nunknowns)
    tmp = np.array([[1,-1],[-1,-1]])
    A_con = sp.kron(tmp,I)
    #A_con = A_con.A#dense
    lb_con = -np.inf*np.ones(nunknowns+nslack_variables)
    ub_con = np.zeros(nunknowns+nslack_variables)
    #print(A_con.A)
    linear_constraint = LinearConstraint(
        A_con, lb_con, ub_con, keep_feasible=False)
    constraints = [linear_constraint]
    
    def constraint_obj(x):
        val = func(x[:nunknowns])
        if func_jac == True:
            return val[0]
        return val
    
    def constraint_jac(x):
        if func_jac == True:
            jac = func(x[:nunknowns])[1]
        else:
            jac = func_jac(x[:nunknowns])

        jac = sp.hstack([jac,sp.csr_matrix((jac.shape[0],jac.shape[1]),dtype=float)])
        #np.concatenate([jac,np.zeros(nslack_variables)])[np.newaxis,:]
        jac = sp.csr_matrix(jac)
        return jac
    
    if func_hess is not None:
        def constraint_hessian(x,v):
            # see https://prog.world/scipy-conditions-optimization/
            # for example how to define NonlinearConstraint hess
            H = func_hess(x[:nunknowns])
            hess = sp.lil_matrix((x.shape[0],x.shape[0]),dtype=float)
            hess[:nunknowns,:nunknowns]=H*v[0]
            return hess
    else:
        constraint_hessian = BFGS()   

    # experimental parameter. does not enforce interpolation but allows some
    # deviation
    nonlinear_constraint = NonlinearConstraint(
        constraint_obj,0,0,jac=constraint_jac,hess=constraint_hessian,
        keep_feasible=False)
    constraints.append(nonlinear_constraint)
    
    lbs = np.zeros(nunknowns+nslack_variables);
    lbs[:nunknowns]=-np.inf
    ubs = np.inf*np.ones(nunknowns+nslack_variables)
    bounds = Bounds(lbs,ubs)
    x0 = np.concatenate([init_guess,np.absolute(init_guess)])
    #print('obj',obj(x0)[0])
    #print('jac',obj(x0)[1])
    #print('nl_constr_obj',nonlinear_constraint.fun(x0))
    #print('nl_constr_jac',nonlinear_constraint.jac(x0).A)
    #print('l_constr_obj',linear_constraint.A.dot(x0),linear_constraint.ub)
    res = minimize(
        obj, x0, method=method, jac=True, hessp=hessp, options=options,
        bounds=bounds, constraints=constraints)
    
    return res.x[:nunknowns]

def kouri_smooth_absolute_value(t,r,x):
    vals = np.zeros(x.shape[0])
    I = np.where(r*x+t<-1)[0]
    vals[I] = -1/r*(r * x[I] + t[I] + 1/2 + 1/2 * t[I]**2)
    J = np.where((-1<=r*x+t)&(r*x+t<=1))[0]
    vals[J] = t[J] * x[J] + r/2 * x[J]**2
    K = np.where(1<r*x+t)[0]
    vals[K] = 1/r * (r * x[K] + t[K] - 1/2 - 1/2 * t[K]**2)
    return vals

def kouri_smooth_absolute_value_gradient(t,r,x):
    grad = np.zeros(x.shape[0])
    I = np.where(r*x+t<-1)[0]
    grad[I] = -1
    J = np.where((-1<=r*x+t)&(r*x+t<=1))[0]
    grad[J] = t[J] + r * x[J]
    K = np.where(1<r*x+t)[0]
    grad[K] = 1
    return grad

def kouri_smooth_absolute_value_hessian(t,r,x):
    hess = np.zeros(x.shape[0])
    J = np.where((-1<=r*x+t)&(r*x+t<=1))[0]
    hess[J] = r
    return hess

def kouri_smooth_l1_norm(t,r,x):
    vals = kouri_smooth_absolute_value(t,r,x)
    norm = vals.sum()
    return norm

def kouri_smooth_l1_norm_gradient(t,r,x):
    grad = kouri_smooth_absolute_value_gradient(t,r,x)
    return grad

def kouri_smooth_l1_norm_hessian(t,r,x):
    hess = kouri_smooth_absolute_value_hessian(t,r,x)
    hess = np.diag(hess)
    return hess

def kouri_smooth_l1_norm_hessp(t,r,x,v):
    hess = kouri_smooth_absolute_value_hessian(t,r,x)
    return hess*v[0]
        
def basis_pursuit_denoising(func,func_jac,func_hess,init_guess,eps,options,homotopy_options):

    t = np.ones_like(init_guess)
    r = 1

    method = 'trust-constr'
    nunknowns = init_guess.shape[0]

    def constraint_obj(x):
        val = func(x)
        if func_jac == True:
            return val[0]
        return val
    
    def constraint_jac(x):
        if func_jac == True:
            jac = func(x)[1]
        else:
            jac = func_jac(x)
        return jac

    # if func_hess is None:
    #    constraint_hessian = BFGS()
    # else:
    #    def constraint_hessian(x,v):
    #        H = func_hess(x)
    #        return H*v.sum()
    constraint_hessian = BFGS()

    nonlinear_constraint = NonlinearConstraint(
        constraint_obj,-eps,eps,jac=constraint_jac,hess=constraint_hessian,
        keep_feasible=False)
    constraints = [nonlinear_constraint]

    niter=0
    prev_obj = np.inf
    x0=init_guess
    maxiter,xtol = homotopy_options.get('maxiter',1000),homotopy_options.get('xtol',1e-8)
    verbose=homotopy_options.get('verbose',1)
    nfev,njev,nhev=0,0,0
    constr_nfev,constr_njev,constr_nhev=0,0,0
    while True:
        obj  = partial(kouri_smooth_l1_norm,t,r)
        jac  = partial(kouri_smooth_l1_norm_gradient,t,r)
        #hessp = partial(kouri_smooth_l1_norm_hessp,t,r)
        hessp=None

        res = minimize(
            obj, x0, method=method, jac=jac, hessp=hessp, options=options,
            constraints=constraints)
        assert res.status==1 or res.status==2

        if abs(prev_obj-res.fun)<xtol:
            msg = f'homotopy xtol {xtol} reached after {niter} iters.'
            status=2
            break
        if niter >= maxiter:
            msg = f'maxiter {maxiter} reached. f(x_prev)-f(x)={prev_obj-res.fun}'
            status=0
            break
        
        if verbose>1:
            print (f'Current objective value {res.fun}')
        niter+=1
        x = res.x
        prev_obj=res.fun
        t = np.maximum(-1, np.minimum(1, t + r*x))
        r *= 2
        x0=x.copy()
        nfev+=res.nfev;njev+=res.njev;nhev+=res.nhev
        constr_nfev+=res.constr_nfev[0];constr_njev+=res.constr_njev[0];constr_nhev+=res.constr_nhev[0]

    if verbose>0:
        print(msg)

    res = OptimizeResult(fun=res.fun,x=res.x,nit=niter,msg=msg,nfev=nfev,njev=njev,nhev=nhev,constr_nfev=constr_nfev,constr_njev=constr_njev,constr_nhev=constr_nhev,status=status)
    return res
