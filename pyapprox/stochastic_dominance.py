import numpy as np
from scipy import sparse
from scipy.sparse import eye as speye
from scipy.sparse import lil_matrix
from cvxopt import matrix as cvxopt_matrix, solvers, spmatrix as cvxopt_spmatrix
from functools import partial

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
    for ii in range(ssd_obj.nconstraints):
        ineq_cons_fun = partial(ssd_obj.constraints,constraint_indices=ii)
        ineq_cons = {'type': 'ineq', 'fun' : ineq_cons_fun}
        ineq_cons['jac'] = partial(ssd_obj.constraint_gradients,
            constraint_indices=ii)
        constraints.append(ineq_cons)
    return constraints

def solve_disutility_stochastic_dominance_constrained_least_squares_scipy(
        samples,values,eval_basis_matrix,eta_indices=None):
    """
    Disutility formuation
    -Y dominates -Z

    sum_{k=1}^N p_k t_{ik} (z_k-y_i) >= v_i = E[(y_i+Y)^{+}], k=1,...,N
    t_ik>=0, i=1,...,N, k=1,...,N
    """
    num_samples = samples.shape[1]
    basis_matrix = eval_basis_matrix(samples)
    probabilities = np.ones((num_samples))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)

    # define objective
    ssd_functor = DisutilitySSDFunctor(
        basis_matrix,values[:,0],values[eta_indices,0],probabilities)
    objective = ssd_functor.objective
    jac = ssd_functor.objective_gradient

    # define initial guess
    init_guess = np.zeros((ssd_functor.nunknowns))
    init_guess[ssd_functor.ncoef:]=1
    init_guess[:ssd_functor.ncoef]=np.linalg.lstsq(
        basis_matrix,values,rcond=None)[0][:,0]

    # define nonlinear inequality constraints
    constraints = build_disutility_constraints_scipy(ssd_functor)

    # define bounds
    from scipy.optimize import minimize, Bounds
    lb = np.zeros(ssd_functor.nunknowns)
    lb[:basis_matrix.shape[1]] = -np.inf
    ub = np.ones(ssd_functor.nunknowns)
    ub[:basis_matrix.shape[1]] = np.inf
    bounds = Bounds(lb,ub)

    # solve ssd problem
    optim_options={'ftol': 1e-4, 'disp': True, 'maxiter':1000}
    opt_method = 'SLSQP'
    res = minimize(
        objective, init_guess, method=opt_method, jac=jac,
        constraints=constraints,options=optim_options,bounds=bounds)
    coef = res.x[:basis_matrix.shape[1],np.newaxis]
    slack_variables = res.x[basis_matrix.shape[1]:]
    
    print('t',slack_variables)
    if not res.success:
        raise Exception(res.message)
    
    return coef


class DisutilitySSDFunctor(object):
    def __init__(self,basis_matrix,values,eta,probabilities):
        self.basis_matrix=basis_matrix
        self.values=values
        self.eta=eta
        self.probabilities=probabilities
        self.nonlinear_inequalities_lowerbound=True

        assert values.ndim==1
        assert eta.ndim==1
        assert probabilities.ndim==1

        self.nconstraints = eta.shape[0]
        self.nsamples,self.ncoef=basis_matrix.shape
        self.nunknowns = basis_matrix.shape[1]+self.nconstraints*self.nsamples

        self.cond_exps = np.maximum(
            0,self.values[:,np.newaxis]-self.eta[np.newaxis,:]).mean(axis=0)

    def objective(self,x):
        coef = x[:self.ncoef]
        return 0.5*np.sum((self.values-self.basis_matrix.dot(coef))**2)

    def constraints(self,x,constraint_indices=None):
        if constraint_indices is None:
            constraint_indices=np.arange(self.nconstraints)
        constraint_indices = np.atleast_1d(constraint_indices)
            
        coef = x[:self.ncoef]
        approx_values = self.basis_matrix.dot(coef)
        constraint_values = np.zeros(constraint_indices.shape)
        for ii,index in enumerate(constraint_indices):
            lb = self.ncoef+index*self.nsamples
            ub = lb+self.nsamples
            t = x[lb:ub]
            constraint_values[ii] = np.dot(
                self.probabilities*t,approx_values-self.eta[index])
            constraint_values[ii] -= self.cond_exps[index]
        if not self.nonlinear_inequalities_lowerbound:
            constraint_values *= -1 
        return constraint_values

    def objective_gradient(self,x):
        coef = x[:self.ncoef]
        grad = np.zeros(x.shape)
        grad[:self.ncoef] = -self.basis_matrix.T.dot(
            self.values-self.basis_matrix.dot(coef))
        return grad

    def constraint_gradients(self,x,constraint_indices=None):
        if constraint_indices is None:
            constraint_indices=np.arange(self.nconstraints)
        constraint_indices = np.atleast_1d(constraint_indices)

        grad = np.zeros((constraint_indices.shape[0],self.nunknowns))
        coef = x[:self.ncoef]
        approx_values = self.basis_matrix.dot(coef)
        for ii,index in enumerate(constraint_indices):
            lb = self.ncoef+index*self.nsamples
            ub = lb+self.nsamples
            t = x[lb:ub]
            grad[ii,:self.ncoef] = self.basis_matrix.T.dot(self.probabilities*t)
            grad[ii,lb:ub]  = self.probabilities*(approx_values-self.eta[index])
        if grad.ndim==2 and grad.shape[0]==1:
            grad = grad[0,:]

        if not self.nonlinear_inequalities_lowerbound:
            grad *= -1 
        return grad

class CVXOptDisutilitySSDFunctor(DisutilitySSDFunctor):
    def __init__(self,basis_matrix,values,eta,probabilities):
        super().__init__(basis_matrix,values,eta,probabilities)
        self.nonlinear_inequalities_lowerbound=False
        self.H=[None for ii in range(self.nconstraints+1)]

        nslack_variables = eta.shape[0]*self.nsamples
        nlinear_constraints = 2*nslack_variables
        self.G = cvxopt_matrix(0.0, (nlinear_constraints,self.nunknowns))
        self.h = cvxopt_matrix(0.0, (nlinear_constraints,1))

        self.G[:nslack_variables,self.ncoef:]=-np.eye(nslack_variables)
        self.G[nslack_variables:,self.ncoef:]=np.eye(nslack_variables)
        self.h[:nslack_variables]=0.
        self.h[nslack_variables:]=1.

        print(np.array(self.G))
    
    def __call__(self,x=None,z=None):
        if x is None:
            return self.nconstraints,  cvxopt_matrix(1.0, (self.nunknowns,1))

        # convert from cvxopt to numpy
        x = np.array(x)[:,0]

        f = cvxopt_matrix(0.0, (self.nconstraints+1,1))
        f[0] = self.objective(x)
        f[1:] = self.constraints(x)
        #print('f',f[0])
        print(self.constraints(x))

        Df = cvxopt_matrix(0.0, (self.nconstraints+1,self.nunknowns))
        Df[0,:] = self.objective_gradient(x)
        Df[1:,:] = self.constraint_gradients(x)


        if z is None:  return f, Df

        H = cvxopt_matrix(0.0, (self.nunknowns,self.nunknowns))
        for kk in range(self.nconstraints+1):
            hess_kk = self.hessian(x,kk)
            H += z[kk]*hess_kk
            # # TODO perhaps only need to compute components of hessian using
            # # self.hessian once. Use this check once optimizer is working
            # if self.H[kk] is not None:
            #     assert np.allclose(self.H[kk],hess_kk)
            #     self.H[kk]=hess_kk

        #print('rank H ',np.linalg.matrix_rank(np.array(H)))
        #print('rank Df',np.linalg.matrix_rank(np.array(Df)))
        #print('n',x.shape[0])
        #print(np.array(Df).shape)

        return f, Df, H

    def hessian(self,x,ii):
        from pyapprox.optimization import approx_jacobian
        hessian = cvxopt_matrix(0.0, (self.nunknowns,self.nunknowns))
        xx = np.array(x)
        if ii==0:
            fd_hess = approx_jacobian(self.objective_gradient,xx)
            fed_hess = 0.5*(fd_hess.T+fd_hess)
            hessian[:self.ncoef,:self.ncoef] = self.basis_matrix.T.dot(
                self.basis_matrix)
            #print('Ho',fd_hess,np.array(hessian))
        else:
            fd_hess = approx_jacobian(
                partial(self.constraint_gradients,constraint_indices=ii-1),xx)
            fed_hess = 0.5*(fd_hess.T+fd_hess)
            #print('Hc',fd_hess)
            hessian[:,:]=fd_hess

        #print('H',np.array(hessian))
        return hessian
        

def solve_disutility_stochastic_dominance_constrained_least_squares_cvxopt(
        samples,values,eval_basis_matrix,eta_indices=None):
    """
    Disutility formuation
    -Y dominates -Z

    sum_{k=1}^N p_k t_{ik} (z_k-y_i) >= v_i = E[(y_i+Y)^{+}], k=1,...,N
    t_ik>=0, i=1,...,N, k=1,...,N
    """
    num_samples = samples.shape[1]
    basis_matrix = eval_basis_matrix(samples)
    probabilities = np.ones((num_samples))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)

    # define objective
    ssd_functor = CVXOptDisutilitySSDFunctor(
        basis_matrix,values[:,0],values[eta_indices,0],probabilities)

    solvers.options['show_progress'] = True#False
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8
    solvers.options['maxiters'] = 10
    result = solvers.cp(ssd_functor,G=ssd_functor.G,h=ssd_functor.h)
    #print ("done")
    ssd_solution = np.array(result['x'])
    print(result)
    coef = ssd_solution[:ssd_functor.ncoef]
    slack_variables = ssd_solution[ssd_functor.ncoef:]
    print(slack_variables)
    if result['status']!='optimal':
        msg = 'Failed with status: %s'%result['status']
        raise Exception(msg)
    
    return coef
