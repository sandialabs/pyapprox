import numpy as np
from scipy import sparse
from scipy.sparse import eye as speye
from scipy.sparse import lil_matrix,csc_matrix
from cvxopt import matrix as cvxopt_matrix, solvers, spmatrix as cvxopt_spmatrix
from functools import partial
from scipy.optimize import minimize

def build_inequality_contraints(Y,basis_matrix,p,eta_indices):
    """
    Construct the matrix form of the constraints of quadratic program.
       Ax<=b

    Utility formulation
    Z dominates Y

    s_{ik} + z_k >= y_i, i=1,...,N, k=1,...,N
    sum_{k=1}^N p_k s_{ik} <= v_i = E[(y_i-Y)^{+}], k=1,...,N
    s_ik>=0, i=1,...,N, k=1,...,N
    

    Note that A contains contraints that enforce s_ik >=0. This can be removed
    if solver being used allows bounds to be enforced separately to constraints.

    Parameters
    ----------
    basis_matrix : np.ndarray (N,M)
       The Vandermonde type matrix obtained by evaluting the basis
       at each of the training points

    Y : np.ndarray (N,1)
       The values y_i, i=1,...,M

    Yvec : np.ndarray (N,1)
        The values v_i = E[(y_i-Y)^{+}], v_i=1,...,M

    p : np.ndarray (N,1)
        The probabilities p_k, k=1,...,M

    eta_indices : np.ndarray (P)
        The elements of Y at which to enforce the dominance constraints.

    Returns
    -------
    A : np.ndarray (2N*P+P,N*P+M)
       The constraints matrix. Contains contraints that enforce s_ik >=0
       for i=1,..P and k=1,..N the constraints rows are ordered
        
       ...
       z_{k}  + s_{i,k}   >= eta_i
       z_{k+1}+ s_{i,k+1} >= eta_i
       ...
       z_{k}  + s_{i+1,k}   >= eta_i+1
       z_{k+1}+ s_{i+1,k+1} >= eta_i+1
       ...
       \sum_{k=1}^N p_k s_{i,k} <= v_i   = E[(eta_{i}  -Y)^{+}]
       \sum_{k=1}^N p_k s_{i+1,k} <= v_i = E[(eta_{i+1}-Y)^{+}]
       ...
       s_{i,k}   >= 0
       s_{i,k+1} >= 0
       ...
       s_{i+1,k}   >= 0
       s_{i+1,k+1} >= 0
       ...
    

    b : np.ndarray (2N*P+P,1)
       The constraints RHS. Contains contraints that enforce s_ik >=0
       The rows of b are aranged as follows
       b = [ c_0 ... c_M  s_{0,0} ... s_{0,k} s_{0,k+1} ... s_{P,k} s_{P,k+1} ... s_PN].T

    """
    assert (Y.ndim==2 and Y.shape[1]==1)
    nsamples = Y.shape[0]
    num_eta = eta_indices.shape[0]
    assert num_eta<=nsamples
    nbasis = basis_matrix.shape[1]
    assert basis_matrix.shape[0] == nsamples
    assert p.shape[0]==nsamples
    assert p.sum()==1

    eta=Y[eta_indices,0]
    reduced_cond_exps=compute_conditional_expectations(
        eta,Y[:,0],False)
    
    num_opt_vars  = nbasis           # number of polynomial coefficients
    num_opt_vars += num_eta*nsamples # number of decision vars s_{ik}

    # z_k+s_{ik} >= y_i
    num_constraints  = num_eta*nsamples
    # \sum_{k=1}^N p_k s_{ik} <= v_i = E[(y_i-Y)^{+}]
    num_constraints += num_eta
    # s_ik>=0
    num_constraints += num_eta*nsamples
    I = speye(nsamples,nsamples)
    
    A = lil_matrix((num_constraints,num_opt_vars))
    b = np.empty((num_constraints,1))
    for ii in range(num_eta):
        row = ii*nsamples
        col = nbasis + ii*nsamples
        # s_{ik}+z_k >= eta_i
        # The following was old (incorrect constraint
        #A[row:row+nsamples,:nbasis] = -basis_matrix[ii]
        A[row:row+nsamples,:nbasis] = -basis_matrix
        A[row:row+nsamples,col:col+nsamples]= -I
        b[row:row+nsamples,0]               = -eta[ii]
        
    row=num_eta*nsamples
    col = nbasis
    # \sum_{k=1}^N p_k s_{ik} <= v_i = E[(eta_i-Y)^{+}]
    for ii in range(num_eta):
        A[row+ii,col:col+nsamples] = p.T
        col+=nsamples
    b[row:row+num_eta,0] = reduced_cond_exps
            
    # s_ik>=0
    idx=num_eta*nsamples
    A[-idx:,-idx:] = -speye(num_eta*nsamples)
    b[-idx:]     = 0
    
    #np.set_printoptions(linewidth=500)
    #print('pyapprox')
    #print(A.todense())
    #print(b)
    return A,b


def compute_conditional_expectations(eta,samples,disutility_formulation=True):
    """
    Compute the conditional expectation of :math:`Y`
    .. math::
      \mathbb{E}\left[\max(0,\eta-Y)\right]

    or of :math:`-Y` (disutility form)
    .. math::
      \mathbb{E}\left[\max(0,Y-\eta)\right]

    where \math:`\eta\in Y' in the domain of :math:`Y'

    The conditional expectation is convex non-negative and non-decreasing.

    Parameters
    ==========
    eta : np.ndarray (num_eta)
        values of :math:`\eta`

    samples : np.ndarray (nsamples)
        The samples of :math:`Y`

    disutility_formulation : boolean
        True  - evaluate \mathbb{E}\left[\max(0,\eta-Y)\right]
        False - evaluate \mathbb{E}\left[\max(0,Y-\eta)\right]

    Returns
    =======
    values : np.ndarray (num_eta)
        The conditional expectations
    """
    assert samples.ndim==1
    assert eta.ndim==1
    if disutility_formulation:
        values = np.maximum(
            0,samples[:,np.newaxis]+eta[np.newaxis,:]).mean(axis=0)
    else:
        values = np.maximum(
            0,eta[np.newaxis,:]-samples[:,np.newaxis]).mean(axis=0)
    return values

def gradient(coeff0,N,M,P):
    """
    gradient g of x'Hx+g'x

    M : num basis terms
    N : num samples
    P : num eta
    """
    g = np.zeros((M+P*N,1))
    g[:M] = -coeff0
    #g = lil_matrix((M+P*N,1))
    #g[:M] = -coeff0
    return g

def hessian(basis_matrix,P,use_sample_average=True):
    """
    Hessian H of x'Hx+g'x

    P : num eta
    """
    N, M = basis_matrix.shape
    H = lil_matrix((M+P*N,M+P*N))
    if use_sample_average:
        H[:M,:M] = basis_matrix.T.dot(basis_matrix)/N
    else:
        H[:M,:M] = speye(M,M)
    return H

def solve_stochastic_dominance_constrained_least_squares(
        samples,values,eval_basis_matrix,lstsq_coef=None,
        eta_indices=None):
    # Compute coefficients with second order stochastic dominance constraints
    num_samples = samples.shape[1]
    basis_matrix = eval_basis_matrix(samples)
    probabilities = np.ones((num_samples,1))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)
    [A_lil,b]=build_inequality_contraints(
        values,basis_matrix,probabilities,eta_indices)

    #if lstsq_coef is None:
    #    lstsq_coef = np.linalg.lstsq(
    #        eval_basis_matrix(samples),values,rcond=None)[0]
    lstsq_coef = eval_basis_matrix(samples).T.dot(values)/num_samples

    num_basis_terms = basis_matrix.shape[1]
    # minimize distance between lstsq solution and ssd solution
    g = gradient(lstsq_coef,num_samples,num_basis_terms,eta_indices.shape[0])
    H = hessian(basis_matrix,eta_indices.shape[0])

    # Convert scipy lil format to sparse cvxopt format
    # TODO use CVXOPT sparse matrix format
    I,J,data = sparse.find(A_lil)
    A_sparse = cvxopt_spmatrix(data,I,J,size=A_lil.shape)
    I,J,data = sparse.find(H)
    H_sparse = cvxopt_spmatrix(data,I,J,size=H.shape);

    A = A_sparse
    H = H_sparse
    # g and b must be of type matrix and not type cvxopt_spmatrix
    g = cvxopt_matrix(g)
    b = cvxopt_matrix(b)

    # solve least squares problem
    # H = hessian(basis_matrix,eta_indices.shape[0])
    # H = H.todense()[:num_basis_terms,:num_basis_terms]
    # I,J,data = sparse.find(H)
    # H = cvxopt_spmatrix(data,I,J,size=H.shape);
    # g = g[:num_basis_terms]
    # A=None
    # b=None

    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8
    solvers.options['maxiters'] = 1000
    # Minimize x'Hx+g'x subject to Ax <= b
    #print ("optimizing")
    result = solvers.qp(H,g,A,b,Aeq=None,beq=None)
    #print ("done")
    ssd_solution = np.array(result['x'])
    coef = ssd_solution[:num_basis_terms]
    return coef

def build_disutility_constraints_scipy(ssd_obj):
    constraints = []
    for ii in range(ssd_obj.nnl_constraints):
        ineq_cons_fun = partial(
            ssd_obj.nonlinear_constraints,constraint_indices=ii)
        ineq_cons = {'type': 'ineq', 'fun' : ineq_cons_fun}
        ineq_cons['jac'] = partial(ssd_obj.nonlinear_constraint_gradients,
            constraint_indices=ii)
        constraints.append(ineq_cons)
    return constraints

def solve_disutility_stochastic_dominance_constrained_least_squares_slsqp(
        samples,values,eval_basis_matrix,eta_indices=None):
    """
    Disutility formuation
    -Y dominates -Z
    """
    num_samples = samples.shape[1]
    basis_matrix = eval_basis_matrix(samples)
    probabilities = np.ones((num_samples))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)

    # define objective
    ssd_functor = SLSQPDisutilitySSDFunctor(
        basis_matrix,values[:,0],values[eta_indices,0],probabilities)
    objective = ssd_functor.objective
    jac = ssd_functor.objective_gradient

    # define nonlinear inequality constraints
    constraints = build_disutility_constraints_scipy(ssd_functor)
    ineq_cons_fun = lambda x: (ssd_functor.linear_constraint_matrix.dot(x)-ssd_functor.linear_constraint_vector)
    ineq_cons = {'type': 'ineq', 'fun' : ineq_cons_fun}
    ineq_cons['jac'] = lambda x: ssd_functor.linear_constraint_matrix.todense()
    
    constraints.append(ineq_cons)

    # solve ssd problem
    optim_options={'ftol': 1e-4, 'disp': True, 'maxiter':1000}
    opt_method = 'SLSQP'
    res = minimize(
        objective, ssd_functor.init_guess, method=opt_method, jac=jac,
        constraints=constraints,options=optim_options,
        bounds=ssd_functor.bounds)
    coef = res.x[:basis_matrix.shape[1],np.newaxis]
    slack_variables = res.x[basis_matrix.shape[1]:]
    
    #print('t',slack_variables)
    if not res.success:
        raise Exception(res.message)
    
    return coef


class SLSQPDisutilitySSDFunctor(object):
    def __init__(self,basis_matrix,values,eta,probabilities):
        self.basis_matrix=basis_matrix
        self.values=values
        self.eta=eta
        self.probabilities=probabilities

        assert values.ndim==1
        assert eta.ndim==1
        assert probabilities.ndim==1

        #number of nonlinear constraints
        self.nnl_constraints = eta.shape[0]
        #number of linear constraints
        self.nl_constaints = self.nnl_constraints
        self.nsamples,self.ncoef=basis_matrix.shape
        #self.nslack_variables = self.nnl_constraints*self.nsamples
        self.nslack_variables = 2*self.nnl_constraints*self.nsamples
        self.nunknowns=self.ncoef+self.nslack_variables

        self.cond_exps = np.maximum(
            0,self.values[:,np.newaxis]-self.eta[np.newaxis,:]).mean(axis=0)

        # define bounds
        from scipy.optimize import Bounds
        # lb = np.zeros(self.nunknowns)
        # lb[:self.ncoef] = -np.inf
        # ub = np.ones(self.nunknowns)
        # ub[:self.ncoef] = np.inf
        # self.bounds = Bounds(lb,ub)

        lb = np.zeros(self.nunknowns)
        lb[:self.ncoef] = -np.inf
        ub = np.ones(self.nunknowns)
        ub[:self.ncoef] = np.inf
        ub[:self.ncoef+self.nslack_variables//2:] = np.inf
        self.bounds = Bounds(lb,ub)
        
        # define initial guess
        self.init_guess = np.zeros((self.nunknowns))
        #self.init_guess[self.ncoef:]=1
        self.init_guess[self.ncoef:self.ncoef+self.nslack_variables//2]=1
        self.init_guess[self.ncoef+self.nslack_variables//2:]=1



        lstsq_coef = np.linalg.lstsq(
            self.basis_matrix,self.values,rcond=None)[0]
        #lstsq_coef[0]-=min(0,self.nonlinear_constraints(self.init_guess).min())
        lstsq_coef[0]=1
        self.init_guess[:self.ncoef] = lstsq_coef
        self.init_guess[:]=0

        self.define_linear_constraints()


    def objective(self,x):
        coef = x[:self.ncoef]
        return 0.5*np.sum((self.values-self.basis_matrix.dot(coef))**2)

    def nonlinear_constraints(self,x,constraint_indices=None):
        if constraint_indices is None:
            constraint_indices=np.arange(self.nnl_constraints)
        constraint_indices = np.atleast_1d(constraint_indices)
            
        coef = x[:self.ncoef]
        approx_values = self.basis_matrix.dot(coef)
        constraint_values = np.zeros(constraint_indices.shape)
        for ii,index in enumerate(constraint_indices):
            lb = self.ncoef+index*self.nsamples
            ub = lb+self.nsamples
            t = x[lb:ub]
            lb = self.ncoef+self.nslack_variables//2+index*self.nsamples
            ub = lb+self.nsamples
            s = x[lb:ub]
            #constraint_values[ii] = np.dot(
            #    self.probabilities*t,approx_values-self.eta[index])
            #constraint_values[ii] -= self.cond_exps[index]
            constraint_values[ii] = np.dot(
                self.probabilities*t,approx_values-approx_values[index])
            constraint_values[ii] -= self.probabilities.dot(s)
        return constraint_values

    def objective_gradient(self,x):
        coef = x[:self.ncoef]
        grad = np.zeros(x.shape)
        grad[:self.ncoef] = -self.basis_matrix.T.dot(
            self.values-self.basis_matrix.dot(coef))
        return grad

    def nonlinear_constraint_gradients(self,x,constraint_indices=None):
        if constraint_indices is None:
            constraint_indices=np.arange(self.nnl_constraints)
        constraint_indices = np.atleast_1d(constraint_indices)

        grad = np.zeros((constraint_indices.shape[0],self.nunknowns))
        coef = x[:self.ncoef]
        approx_values = self.basis_matrix.dot(coef)
        for ii,index in enumerate(constraint_indices):
            lb = self.ncoef+index*self.nsamples
            ub = lb+self.nsamples
            t = x[lb:ub]
            #grad[ii,:self.ncoef] = self.basis_matrix.T.dot(
            #    self.probabilities*t)
            #grad[ii,lb:ub]  = self.probabilities*(
            #    approx_values-self.eta[index])
            grad[ii,1:self.ncoef] = (
                self.basis_matrix[:,1:]-self.basis_matrix[index,1:]).T.dot(
                    self.probabilities*t)
            grad[ii,lb:ub]  = self.probabilities*(
                approx_values-approx_values[index])
            lb = self.ncoef+self.nslack_variables//2+index*self.nsamples
            ub = lb+self.nsamples
            s = x[lb:ub]
            grad[ii,lb:ub]  = -self.probabilities
            
        if grad.ndim==2 and grad.shape[0]==1:
            grad = grad[0,:]

        return grad

    def define_linear_constraints(self):
        self.linear_constraint_vector = np.tile(
            self.values,self.nnl_constraints)

        nnonzero_entries = (self.ncoef+1)*self.nslack_variables//2
        data = np.empty((nnonzero_entries),dtype=float)
        I = np.empty((nnonzero_entries),dtype=int)
        J = np.empty_like(I)
        kk=0
        for ii in range(self.nnl_constraints):
            for jj in range(self.nsamples):
                data[kk:kk+self.ncoef]=self.basis_matrix[jj,:]
                data[kk+self.ncoef]=1
                I[kk:kk+self.ncoef+1]=ii*self.nsamples+jj
                J[kk:kk+self.ncoef]=np.arange(self.ncoef)
                J[kk+self.ncoef]=self.ncoef+self.nslack_variables//2+\
                    ii*self.nsamples+jj
                kk+=self.ncoef+1
        assert kk==nnonzero_entries

        self.linear_constraint_matrix = csc_matrix(
            (data,(I,J)),shape=(self.nslack_variables//2,self.nunknowns))


class TrustRegionDisutilitySSDFunctor(SLSQPDisutilitySSDFunctor):
    def __init__(self,basis_matrix,values,eta,probabilities):
        super().__init__(basis_matrix,values,eta,probabilities)
        self.constraint_hessians=[None for ii in range(self.nnl_constraints)]

    def objective_jacobian(self,x):
        coef = x[:self.ncoef]
        data = -self.basis_matrix.T.dot(self.values-self.basis_matrix.dot(
            coef))
        I = np.zeros(self.ncoef,dtype=int)
        J = np.arange(self.ncoef,dtype=int)

        grad = csc_matrix(
            (data,(I,J)),shape=(1,self.nunknowns))
        return grad

    def nonlinear_constraints_jacobian(self,x):
        coef = x[:self.ncoef]
        approx_values = self.basis_matrix.dot(coef)

        #nnonzero_entries=(self.ncoef+self.nsamples)*self.nnl_constraints
        nnonzero_entries=(self.ncoef-1+2*self.nsamples)*self.nnl_constraints
        data = np.empty(nnonzero_entries,dtype=float)
        I = np.empty(nnonzero_entries,dtype=int)
        J = np.empty_like(I)
        kk=0
        for ii in range(self.nnl_constraints):
            lb = self.ncoef+ii*self.nsamples
            ub = lb+self.nsamples
            t = x[lb:ub]
            #I[kk:kk+self.ncoef] = ii;
            #J[kk:kk+self.ncoef] = np.arange(self.ncoef)
            #data[kk:kk+self.ncoef]=self.basis_matrix.T.dot(
            #    self.probabilities*t)
            #kk+=self.ncoef
            I[kk:kk+self.ncoef-1] = ii;
            J[kk:kk+self.ncoef-1] = np.arange(1,self.ncoef)
            data[kk:kk+self.ncoef-1]=(
                self.basis_matrix[:,1:]-self.basis_matrix[ii,1:]).T.dot(
                    self.probabilities*t)
            kk+=self.ncoef-1

            I[kk:kk+(ub-lb)] = ii;
            J[kk:kk+(ub-lb)] = np.arange(lb,ub)
            #data[kk:kk+(ub-lb)]=self.probabilities*(
            #    approx_values-self.eta[ii])
            # approx_values[ii] should be approx_values[eta_indices[ii]]
            # this change needs to occur elsewhere too
            data[kk:kk+(ub-lb)]=self.probabilities*(
                approx_values-approx_values[ii])
            kk+=(ub-lb)

            lb = self.ncoef+self.nslack_variables//2+ii*self.nsamples
            ub = lb+self.nsamples
            I[kk:kk+(ub-lb)] = ii;
            J[kk:kk+(ub-lb)] = np.arange(lb,ub)
            s = x[lb:ub]
            data[kk:kk+(ub-lb)] = -self.probabilities
            kk+=(ub-lb)

        assert kk==nnonzero_entries

        grad = csc_matrix(
            (data,(I,J)),shape=(self.nnl_constraints,self.nunknowns))
        return grad

    def objective_hessian(self,x):
        from pyapprox.optimization import approx_jacobian
        xx = np.array(x)
        data = self.basis_matrix.T.dot(
            self.basis_matrix).flatten()
        I = np.repeat(np.arange(self.ncoef),self.ncoef)
        J = np.tile(np.arange(self.ncoef),self.ncoef)
        hessian = csc_matrix(
            (data,(I,J)),shape=(self.nunknowns,self.nunknowns))
        return hessian
    
    def define_nonlinear_constraint_hessian(self,x,ii):        
        # temp = self.probabilities[:,np.newaxis]*self.basis_matrix
        # data1 = (temp.T).flatten()
        # I1 = np.repeat(np.arange(self.ncoef),self.nsamples)
        # J1 = np.tile(np.arange(self.nsamples),self.ncoef)
        # J1 += self.ncoef+self.nsamples*ii
        # data2 = temp.flatten()
        # I2 = np.repeat(np.arange(self.nsamples),self.ncoef)
        # J2 = np.tile(np.arange(self.ncoef),self.nsamples)
        # I2 += self.ncoef+self.nsamples*ii

        temp = self.probabilities[:,np.newaxis]*(
            self.basis_matrix[:,1:]-self.basis_matrix[ii,1:])
        data1 = (temp.T).flatten()
        I1 = np.repeat(np.arange(1,self.ncoef),self.nsamples)
        J1 = np.tile(np.arange(self.nsamples),self.ncoef-1)
        J1 += self.ncoef+self.nsamples*ii
        data2 = temp.flatten()
        I2 = np.repeat(np.arange(self.nsamples),self.ncoef-1)
        J2 = np.tile(np.arange(1,self.ncoef),self.nsamples)
        I2 += self.ncoef+self.nsamples*ii
        
        I = np.concatenate([I1,I2])
        J = np.concatenate([J1,J2])
        data = np.concatenate([data1,data2])

        hessian = csc_matrix(
            (data,(I,J)),shape=(self.nunknowns,self.nunknowns))
        return hessian

    def nonlinear_constraints_hessian(self,x,v):
        assert v.shape[0]==self.nnl_constraints
        result = 0
        for ii in range(v.shape[0]):
            if self.constraint_hessians[ii] is None:
                self.constraint_hessians[ii]=\
                    self.define_nonlinear_constraint_hessian(x,ii)
            result += v[ii]*self.constraint_hessians[ii]
        return result

def solve_disutility_stochastic_dominance_constrained_least_squares_trust_region(samples,values,eval_basis_matrix,eta_indices=None):
    """
    Disutility formuation
    -Y dominates -Z
    """
    num_samples = samples.shape[1]
    basis_matrix = eval_basis_matrix(samples)
    probabilities = np.ones((num_samples))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)

    # define objective
    ssd_functor = TrustRegionDisutilitySSDFunctor(
        basis_matrix,values[:,0],values[eta_indices,0],probabilities)

    from scipy.optimize import NonlinearConstraint, LinearConstraint
    keep_feasible=False
    nonlinear_constraint = NonlinearConstraint(
        ssd_functor.nonlinear_constraints, 0, np.inf,
        jac=ssd_functor.nonlinear_constraints_jacobian,
        hess=ssd_functor.nonlinear_constraints_hessian,
        keep_feasible=keep_feasible)

    linear_constraint = LinearConstraint(
        ssd_functor.linear_constraint_matrix,
        ssd_functor.linear_constraint_vector,np.inf,keep_feasible=keep_feasible)
    
    # for optimization options see
    # docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
    tol=1e-3
    optim_options={'verbose': 3, 'maxiter':1000,
                   'gtol':tol, 'xtol':tol, 'barrier_tol':tol}

    if keep_feasible and (np.any(ssd_functor.nonlinear_constraints(ssd_functor.init_guess)<=0) or np.any(linear_constraint_matrix.dot(ssd_functor.init_guess)<linear_constraint_vector)):
        msg = "initial_guess is infeasiable"
        print('c',ssd_functor.nonlinear_constraints(ssd_functor.init_guess))
        print(linear_constraint_matrix.dot(ssd_functor.init_guess)-linear_constraint_vector)
        raise Exception(msg)

    #constraints = [nonlinear_constraint]
    constraints = [linear_constraint,nonlinear_constraint]
    res = minimize(
        ssd_functor.objective, ssd_functor.init_guess, method='trust-constr',
        jac=ssd_functor.objective_gradient,
        hess=ssd_functor.objective_hessian,
        constraints=constraints,options=optim_options,
        bounds=ssd_functor.bounds)

    coef = res.x[:basis_matrix.shape[1],np.newaxis]
    slack_variables = res.x[basis_matrix.shape[1]:]

    # constraints at last iterate
    # print('constraint vals',res['constr'][0])
    
    # print('slack vars',slack_variables[-ssd_functor.nsamples:])
    # print('eta',ssd_functor.eta)
    # print('cond exps',ssd_functor.cond_exps)
    if not res.success:
        raise Exception(res.message)
    
    return coef
