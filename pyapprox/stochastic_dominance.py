import numpy as np
from scipy import sparse
from scipy.sparse import eye as speye
from scipy.sparse import lil_matrix
from cvxopt import matrix, solvers, spmatrix
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
    A_sparse = spmatrix(data,I,J,size=A_lil.shape)
    I,J,data = sparse.find(H)
    H_sparse = spmatrix(data,I,J,size=H.shape);

    A = A_sparse
    H = H_sparse
    # g and b must be of type matrix and not type spmatrix
    g = matrix(g)
    b = matrix(b)

    # solve least squares problem
    # H = hessian(basis_matrix,eta_indices.shape[0])
    # H = H.todense()[:num_basis_terms,:num_basis_terms]
    # I,J,data = sparse.find(H)
    # H = spmatrix(data,I,J,size=H.shape);
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

def disutility_stochastic_dominance_objective(basis_matrix,values,xx):
    nsamples,ncoef = basis_matrix.shape
    coef = xx[:ncoef]
    return 0.5*np.sum((values-basis_matrix.dot(coef))**2)

def disutility_stochastic_dominance_objective_gradient(basis_matrix,values,xx):
    nsamples,ncoef = basis_matrix.shape
    coef = xx[:ncoef]
    grad = np.zeros(xx.shape)
    grad[:ncoef] =  -basis_matrix.T.dot(values-basis_matrix.dot(coef))
    return grad

def solve_disutility_stochastic_dominance_constrained_least_squares(
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
    constraints = build_disutility_constraints(
        basis_matrix,probabilities,values[:,0],eta_indices)

    from scipy.optimize import minimize, Bounds
    optim_options={'ftol': 1e-4, 'disp': True, 'maxiter':1000}
    opt_method = 'SLSQP'
    objective = partial(
        disutility_stochastic_dominance_objective,basis_matrix,values[:,0])
    nunknowns = basis_matrix.shape[1]+eta_indices.shape[0]*num_samples
    init_guess = np.zeros((nunknowns))
    init_guess[basis_matrix.shape[1]:]=1
    init_guess[:basis_matrix.shape[1]]=np.linalg.lstsq(
        basis_matrix,values,rcond=None)[0][:,0]
    lb = np.zeros(nunknowns)
    lb[:basis_matrix.shape[1]] = -np.inf
    ub = np.ones(nunknowns)
    ub[:basis_matrix.shape[1]] = np.inf
    bounds = Bounds(lb,ub)
    jac = partial(disutility_stochastic_dominance_objective_gradient,
                  basis_matrix,values[:,0])
    from scipy.optimize import check_grad, approx_fprime
    xx = np.ones(nunknowns)
    #assert check_grad(objective, jac, xx)<5e-7
    #for con in constraints:
    #    if 'jac' in con:
    #        #print(approx_fprime(xx,con['fun'],1e-7))
    #        #print(con['jac'](xx))
    #        assert check_grad(con['fun'], con['jac'], xx)<5e-7
    #jac=None
    #constraints = [] # return lstsq solution
    res = minimize(
        objective, init_guess, method=opt_method, jac=jac,
        constraints=constraints,options=optim_options,bounds=bounds)
    coef = res.x[:basis_matrix.shape[1],np.newaxis]
    slack_variables = res.x[basis_matrix.shape[1]:]
    #print('t',slack_variables)
    #if not res.success:
    #    raise Exception(res.message)

    # def F(x=None, z=None):
    #     if x is None: return eta_indices.shape[0], matrix(1.0, (nunknowns,1))
    #     if min(x) <= 0.0: return None
    #     f = matrix(0.0,(eta_indices.shape[0]+1,1))
    #     Df = 
    #     f[0] = objective(x)
    #     for con in constraints:
    #         f[ii] = con['fun'](x)
    #     Df = -(x**-1).T
    #     if z is None: return f, Df
    #     H = spdiag(z[0] * x**-2)
    #     return f, Df, H
    # coef = solvers.cp(F)['x']
    
    return coef

def evaluate_disutility_inequality_constraint(basis_matrix,probabilities,values,
                                              eta,cond_exps,constraint_num,xx):
    # turn on for debugging
    #assert values.ndim==1
    #assert probabilities.ndim==1
    #assert eta.ndim==1
    #assert cond_exps.ndim==1
    nsamples,ncoef = basis_matrix.shape
    coef = xx[:ncoef]
    approx_values = basis_matrix.dot(coef)
    lb = ncoef+constraint_num*nsamples
    ub = lb+nsamples
    tt = xx[lb:ub]
    value = np.dot(probabilities*tt,approx_values-eta[constraint_num])
    value -= cond_exps[constraint_num]
    return value

def evaluate_disutility_inequality_constraint_gradient(
        basis_matrix,probabilities,values,
        eta,cond_exps,constraint_num,xx):
    # turn on for debugging
    #assert values.ndim==1
    #assert probabilities.ndim==1
    #assert eta.ndim==1
    #assert cond_exps.ndim==1
    nsamples,ncoef = basis_matrix.shape
    coef = xx[:ncoef]
    approx_values = basis_matrix.dot(coef)
    lb = ncoef+constraint_num*nsamples
    ub = lb+nsamples
    tt = xx[lb:ub]
    
    grad = np.zeros(xx.shape)
    grad[:ncoef] = basis_matrix.T.dot(probabilities*tt)
    grad[lb:ub]  = probabilities*(approx_values-eta[constraint_num])
    return grad

def build_disutility_constraints(basis_matrix,probabilities,values,
                                 eta_indices=None):
    assert values.ndim==1
    assert probabilities.ndim==1
    constraints = []
    nsamples = probabilities.shape[0]
    eta=values[eta_indices]
    cond_exps = np.maximum(
            0,values[:,np.newaxis]-eta[np.newaxis,:]).mean(axis=0)
    for ii in range(eta.shape[0]):
        ineq_cons_fun = partial(
            evaluate_disutility_inequality_constraint,basis_matrix,probabilities,
            values,eta,cond_exps,ii)
        ineq_cons = {'type': 'ineq', 'fun' : ineq_cons_fun}
        ineq_cons['jac']=partial(
            evaluate_disutility_inequality_constraint_gradient,
            basis_matrix,probabilities,values,eta,cond_exps,ii)
        constraints.append(ineq_cons)
    return constraints
