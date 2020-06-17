import numpy as np
import scipy.sparse as sp
from functools import partial
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, BFGS, \
    linprog, OptimizeResult, Bounds
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
    

def nonlinear_basis_pursuit(func,func_jac,func_hess,init_guess,options,eps=0):
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

        if jac.ndim==1:
            jac = jac[np.newaxis,:]
        jac = sp.hstack([jac,sp.csr_matrix((jac.shape[0],jac.shape[1]),dtype=float)])
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
        constraint_obj,0,eps,jac=constraint_jac,hess=constraint_hessian,
        keep_feasible=False)
    constraints.append(nonlinear_constraint)
    
    lbs = np.zeros(nunknowns+nslack_variables);
    lbs[:nunknowns]=-np.inf
    ubs = np.inf*np.ones(nunknowns+nslack_variables)
    bounds = Bounds(lbs,ubs)
    x0 = np.concatenate([init_guess,np.absolute(init_guess)])
    method = options.get('method','slsqp')
    if 'method' in options:
        del options['method']
    if method!='ipopt':
        res = minimize(
            obj, x0, method=method, jac=True, hessp=hessp, options=options,
            bounds=bounds, constraints=constraints)
    else:
        from ipopt import minimize_ipopt
        from scipy.optimize._constraints import new_constraint_to_old
        con = new_constraint_to_old(constraints[0],x0)
        res = minimize_ipopt(
            obj,x0,method=method,jac=True,options=options,
            constraints=con)

    
    return res.x[:nunknowns]

def kouri_smooth_absolute_value(t,r,x):
    vals = np.zeros(x.shape[0])
    z = r*x+t
    I = np.where(z<-1)[0]
    vals[I] = -1/r*(z[I] + 0.5 + 0.5 * t[I]**2)
    J = np.where((-1<=z)&(z<=1))[0]
    vals[J] = t[J] * x[J] + 0.5*r * x[J]**2
    K = np.where(1<z)[0]
    vals[K] = 1/r * (z[K] - 0.5 - 0.5 * t[K]**2)
    return vals

def kouri_smooth_absolute_value_gradient(t,r,x):
    z = r*x+t
    grad = np.maximum(-1,np.minimum(1,z))
    return grad

def kouri_smooth_absolute_value_hessian(t,r,x):
    hess = np.zeros(x.shape[0])
    z = r*x+t
    J = np.where(np.absolute(z)<=1)[0]
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
        
def basis_pursuit_denoising(func,func_jac,func_hess,init_guess,eps,options):

    t = np.zeros_like(init_guess)

    method = options.get('method','slsqp')
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
        constraint_obj,0,eps**2,jac=constraint_jac,hess=constraint_hessian,
        keep_feasible=False)
    constraints = [nonlinear_constraint]

    # Maximum Number Outer Iterations
    maxiter = options.get('maxiter',100)
    # Maximum Number Outer Iterations
    maxiter_inner = options.get('maxiter_inner',1000)
    # Desired Dual Tolerance 
    ttol = options.get('dualtol',1e-6)
    # Verbosity Level
    verbose = options.get('verbose',1)
    # Initial Penalty Parameter
    r = options.get('r0',1)
    # Max Penalty Parameter
    rmax = options.get('rmax',1e6)
    # Optimization Tolerance Update Factor
    tfac = options.get('tfac',1e-1)
    # Penalty Parameter Update Factor
    rfac = options.get('rfac',2)
    # Desired Feasibility Tolerance
    ctol = options.get('ctol',1e-8)
    # Desired Optimality Tolerance
    gtol = options.get('gtol',1e-8)
    # Initial Dual Tolerance
    ttol0  = options.get('ttol0',1)
    # Initial Feasiblity Tolerance
    ctol0  = options.get('ctol0',1e-2)
    # Initial Optimality Tolerance
    gtol0  = options.get('gtol0',1e-2)
    # Tolerance for termination for change in objective
    ftol = options.get('ftol',1e-8)

    niter=0
    x0=init_guess
    f0 = np.inf
    nfev,njev,nhev=0,0,0
    constr_nfev,constr_njev,constr_nhev=0,0,0
    while True:
        obj  = partial(kouri_smooth_l1_norm,t,r)
        jac  = partial(kouri_smooth_l1_norm_gradient,t,r);
        #hessp = partial(kouri_smooth_l1_norm_hessp,t,r)
        hessp=None

        if method=='slsqp':
            options0 = {'ftol':gtol0,'verbose':max(0,verbose-2),
                        'maxiter':maxiter_inner,'disp':(verbose>2)}
        elif method=='trust-constr':
            options0 = {'gtol':gtol0,'tol':gtol0,'verbose':max(0,verbose-2),
                        'barrier_tol':ctol,'maxiter':maxiter_inner,
                        'disp':(verbose>2)}
        elif method=='cobyla':
            options0 = {'tol':gtol0,'verbose':max(0,verbose-2),
                        'maxiter':maxiter_inner,'rhoend':gtol0,'rhobeg':1,
                        'disp':(verbose>2),'catol':ctol0}
        if method!='ipopt':
            #init_guess=x0
            res = minimize(
                obj, init_guess,method=method,jac=jac,hessp=hessp,options=options0,
                constraints=constraints)
        else:
            from ipopt import minimize_ipopt
            options0 = {'tol':gtol0,'print_level':max(0,verbose-1),
                        'maxiter':int(maxiter_inner),'acceptable_constr_viol_tol':ctol0,
                        'derivative_test':'first-order','nlp_scaling_constr_target_gradient':1.}
            from scipy.optimize._constraints import new_constraint_to_old
            con = new_constraint_to_old(constraints[0],init_guess)
            res = minimize_ipopt(
                obj, init_guess,method=method,jac=jac,hessp=hessp,options=options0,
                constraints=con)
            #assert res.success, res
        
        if method=='trust-constr':
            assert res.status==1 or res.status==2
        elif method=='slsqp':
            assert res.status==0
        assert res.success==True

        fdiff = np.linalg.norm(f0-res.fun)
        xdiff = np.linalg.norm(x0-res.x)
        t0=t.copy()
        t = np.maximum(-1, np.minimum(1, t0 + r*res.x))

        tdiff = np.linalg.norm(t0-t)
        niter+=1
        x0 = res.x.copy()
        f0 = res.fun

        nfev+=res.nfev;
        if hasattr(res,'njev'):
            njev+=res.njev;
        if hasattr(res,'nhev'):
            nhev+=res.nhev
        
        if verbose>1:
            #print('  i = %d  tdiff = %11.10e  r = %11.10e  ttol = %3.2e  ctol = %3.2e  gtol = %3.2e  iter = %d'%(niter,tdiff,r,ttol0,ctol0,gtol0,0))
            print('  i = %d  tdiff = %11.10e  fdiff = %11.10e  xdiff = %11.10e  r = %11.10e  ttol = %3.2e  gtol = %3.2e  nfev = %d'%(niter,tdiff,fdiff,xdiff,r,ttol0,gtol0,nfev))

        if tdiff < ttol:
            msg = f'ttol {ttol} reached'
            status=0
            #break

        if fdiff < ftol:
            msg = f'ftol {ftol} reached'
            status=0
            break

        if niter >= maxiter:
            msg = f'maxiter {maxiter} reached'
            status=1
            break

            
        if tdiff > ttol0:
            r = min(r*2,rmax)
        ttol0,gtol0=max(tfac*ttol0,ttol),max(tfac*gtol0,gtol)
        ctol0=max(tfac*ctol0,ctol)
            
        #constr_nfev only for trust-constr
        #constr_nfev+=res.constr_nfev[0];constr_njev+=res.constr_njev[0];constr_nhev+=res.constr_nhev[0]

    if verbose>0:
        print(msg)

        
    res = OptimizeResult(fun=res.fun,x=res.x,nit=niter,msg=msg,nfev=nfev,njev=njev,status=status)#constr_nfev=constr_nfev,constr_njev=constr_njev)
    return res

def lasso(func,func_jac,func_hess,init_guess,lamda,options):
    nunknowns = init_guess.shape[0]
    nslack_variables = nunknowns
    def obj(lamda,x):
        vals = func(x[:nunknowns])
        if func_jac==True:
            grad=vals[1]
            vals=vals[0]
        else:
            grad = func_jac(x[:nunknowns])
        vals += lamda*np.sum(x[nunknowns:])
        grad = np.concatenate([grad,lamda*np.ones(nslack_variables)])
        return vals, grad

    def hess(x):
        H = sp.lil_matrix((x.shape[0],x.shape[0]),dtype=float)
        H[:nunknowns,:nunknowns] = func_hess(x[:nunknowns])
        return H
    if func_hess is None:
        hess=None
    
    I = sp.identity(nunknowns)
    tmp = np.array([[1,-1],[-1,-1]])
    A_con = sp.kron(tmp,I)
    lb_con = -np.inf*np.ones(nunknowns+nslack_variables)
    ub_con = np.zeros(nunknowns+nslack_variables)
    linear_constraint = LinearConstraint(
        A_con, lb_con, ub_con, keep_feasible=False)
    constraints = [linear_constraint]
    #print(A_con.A)
    
    lbs = np.zeros(nunknowns+nslack_variables);
    lbs[:nunknowns]=-np.inf
    ubs = np.inf*np.ones(nunknowns+nslack_variables)
    bounds = Bounds(lbs,ubs)
    x0 = np.concatenate([init_guess,np.absolute(init_guess)])
    method = options.get('method','slsqp')
    if 'method' in options:
        del options['method']
    if method!='ipopt':
        res = minimize(
            partial(obj,lamda), x0, method=method, jac=True, hess=hess, options=options,
            bounds=bounds, constraints=constraints)
    else:
        #jac_structure_old = lambda : np.nonzero(np.tile(np.eye(nunknowns), (2, 2)))
        def jac_structure():
            rows = np.repeat(np.arange(2*nunknowns),2)
            cols = np.empty_like(rows)
            cols[::2] =np.hstack([ np.arange(nunknowns)]*2)
            cols[1::2] =np.hstack([np.arange(nunknowns,2*nunknowns)]*2)
            return rows,cols
        #assert np.allclose(jac_structure()[0],jac_structure_old()[0])
        #assert np.allclose(jac_structure()[1],jac_structure_old()[1])
        
        #jac_structure=None
        def hess_structure():
            h = np.zeros((2*nunknowns, 2*nunknowns))
            h[:nunknowns, :nunknowns] = np.tril(np.ones((nunknowns,nunknowns)))
            return np.nonzero(h)
        if hess is None:
            hess_structure=None
        
        from ipopt import minimize_ipopt
        from scipy.optimize._constraints import new_constraint_to_old
        con = new_constraint_to_old(constraints[0],x0)
        res = minimize_ipopt(
            partial(obj,lamda),x0,method=method,jac=True,options=options,
            constraints=con,jac_structure=jac_structure, hess_structure=hess_structure, hess=hess)
    #print(res)
    
    return res.x[:nunknowns], res
    
    
