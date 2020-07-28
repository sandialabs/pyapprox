import numpy as np
from scipy import sparse
from scipy.sparse import eye as speye
from scipy.sparse import lil_matrix,csc_matrix
from functools import partial
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, BFGS,\
    Bounds

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

def solve_SSD_constrained_least_squares(
        samples,values,eval_basis_matrix,lstsq_coef=None,
        eta_indices=None,return_full=False):
    from cvxopt import matrix as cvxopt_matrix, solvers, spmatrix as cvxopt_spmatrix
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
    coef = ssd_solution[:num_basis_terms,0]
    if return_full:
        return coef, None
    return coef

class SLSQPDisutilitySSDOptProblem(object):
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
        # lb = np.zeros(self.nunknowns)
        # lb[:self.ncoef] = -np.inf
        # ub = np.ones(self.nunknowns)
        # ub[:self.ncoef] = np.inf
        # self.bounds = Bounds(lb,ub)

        lb = np.zeros(self.nunknowns)
        lb[:self.ncoef] = -np.inf
        ub = np.ones(self.nunknowns)
        ub[:self.ncoef] = np.inf
        ub[self.ncoef+self.nslack_variables//2:] = np.inf
        self.bounds = Bounds(lb,ub)

        lstsq_coef = np.linalg.lstsq(
            self.basis_matrix,self.values,rcond=None)[0]
        # Ensure approximation has value >= largest value
        I = np.argmax(self.values)
        residual = self.values[I]-self.basis_matrix[I].dot(lstsq_coef)
        lstsq_coef[0]+=max(0,residual)

        residual = self.values[I]-self.basis_matrix[I].dot(lstsq_coef)
        assert residual<=10*np.finfo(float).eps, residual

        approx_values = self.basis_matrix.dot(lstsq_coef)
        # define initial guess
        self.init_guess = np.zeros((self.nunknowns))
        self.init_guess[:self.ncoef] = lstsq_coef
        for ii in range(self.nnl_constraints):
            lb = self.ncoef+(ii)*self.nsamples
            ub = lb+self.nsamples
            self.init_guess[lb:ub]=np.maximum(
                0,np.sign(approx_values-approx_values[ii]))
            lb = self.ncoef+(ii)*self.nsamples
            lb+=self.nslack_variables//2
            ub+=self.nslack_variables//2
            self.init_guess[lb:ub]=np.maximum(0,self.values-approx_values[ii])

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

    def objective_jacobian(self,x):
        coef = x[:self.ncoef]
        grad = np.zeros(x.shape)
        grad[:self.ncoef] = -self.basis_matrix.T.dot(
            self.values-self.basis_matrix.dot(coef))
        return grad

    def nonlinear_constraints_jacobian(self,x,constraint_indices=None):
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
                data[kk:kk+self.ncoef]=self.basis_matrix[ii,:]
                data[kk+self.ncoef]=1
                I[kk:kk+self.ncoef+1]=ii*self.nsamples+jj
                J[kk:kk+self.ncoef]=np.arange(self.ncoef)
                J[kk+self.ncoef]=self.ncoef+self.nslack_variables//2+\
                    ii*self.nsamples+jj
                kk+=self.ncoef+1
        assert kk==nnonzero_entries

        self.linear_constraint_matrix = csc_matrix(
            (data,(I,J)),shape=(self.nslack_variables//2,self.nunknowns))

    def solve(self,optim_options=None):
        if optim_options is None:
            optim_options={'ftol': 1e-8, 'disp': True, 'maxiter':1000,'iprint':3}

            
        objective = self.objective
        jac = self.objective_jacobian

        # define nonlinear inequality constraints
        ineq_cons_fun = lambda x: self.nonlinear_constraints(x)
        ineq_cons_jac = lambda x: self.nonlinear_constraints_jacobian(x)
        ineq_cons1 = {'type': 'ineq', 'fun' : ineq_cons_fun, 'jac': ineq_cons_jac}
        
        ineq_cons_fun = lambda x: (
            self.linear_constraint_matrix.dot(x)-self.linear_constraint_vector)
        ineq_cons_jac = lambda x: self.linear_constraint_matrix.todense()
        ineq_cons2 = {'type': 'ineq', 'fun' : ineq_cons_fun, 'jac': ineq_cons_jac}
        constraints = [ineq_cons1,ineq_cons2]

        opt_method = 'SLSQP'
        res = minimize(
            objective, self.init_guess, method=opt_method, jac=jac,
            constraints=constraints,options=optim_options,
            bounds=self.bounds)
        coef = res.x[:self.ncoef]
        slack_variables = res.x[self.ncoef:]

        if not res.success:
            raise Exception(res.message)

        return coef

def solve_disutility_SSD_constrained_least_squares_slsqp(samples,values,eval_basis_matrix,eta_indices=None,probabilities=None):
    """
    Disutility formuation
    -Y dominates -Z
    """
    num_samples = samples.shape[1]
    if probabilities is None:
        probabilities = np.ones((num_samples))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)

    basis_matrix = eval_basis_matrix(samples)

    ssd_opt_problem = SLSQPDisutilitySSDOptProblem(
        basis_matrix,values[:,0],values[eta_indices,0],probabilities)

    coef = ssd_opt_problem.solve()

    return coef, ssd_opt_problem



class TrustRegionDisutilitySSDOptProblem(SLSQPDisutilitySSDOptProblem):
    def __init__(self,basis_matrix,values,eta,probabilities):
        super().__init__(basis_matrix,values,eta,probabilities)
        self.constraint_hessians = None

    def objective_jacobian(self,x):
        coef = x[:self.ncoef]
        data = -self.basis_matrix.T.dot(self.values-self.basis_matrix.dot(
            coef))
        # I = np.zeros(self.ncoef,dtype=int)
        # J = np.arange(self.ncoef,dtype=int)

        # grad = csc_matrix(
        #     (data,(I,J)),shape=(1,self.nunknowns))
        grad = np.zeros(x.shape[0])
        grad[:self.ncoef]=data
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
        initialize=False
        if self.constraint_hessians is None:
            self.constraint_hessians=[
                None for ii in range(self.nnl_constraints)]
            initialize=True

        assert v.shape[0]==self.nnl_constraints
        result = 0
        for ii in range(v.shape[0]):
            if initialize:
                self.constraint_hessians[ii]=\
                    self.define_nonlinear_constraint_hessian(x,ii)
            result += v[ii]*self.constraint_hessians[ii]
        return result

    def design_feasiable(self,x,feastol=1e-15):
        return (np.all(self.nonlinear_constraints(x)<-feastol) and
                np.all(self.linear_constraint_matrix.dot(x)<
                       self.linear_constraint_vector-feastol))
        

    def solve(self,optim_options=None):
        """
        for optimization options see
        docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
        """
        if optim_options is None:
            tol=1e-7
            optim_options={'verbose': 3, 'maxiter':1000,
                           'gtol':tol, 'xtol':tol, 'barrier_tol':tol}
        
        keep_feasible=False
        nonlinear_constraint = NonlinearConstraint(
            self.nonlinear_constraints, 0, np.inf,
            jac=self.nonlinear_constraints_jacobian,
            hess=self.nonlinear_constraints_hessian,
            keep_feasible=keep_feasible)

        linear_constraint = LinearConstraint(
            self.linear_constraint_matrix,
            self.linear_constraint_vector,np.inf,
            keep_feasible=keep_feasible)

        #if not self.design_feasiable(self.init_guess):
        #    msg = "initial_guess is infeasiable"
        #    raise Exception(msg)

        #constraints = [nonlinear_constraint]
        constraints = [linear_constraint,nonlinear_constraint]
        res = minimize(
            self.objective, self.init_guess,
            method='trust-constr',
            jac=self.objective_jacobian,
            hess=self.objective_hessian,
            constraints=constraints,options=optim_options,
            bounds=self.bounds)

        coef = res.x[:self.ncoef]
        slack_variables = res.x[self.ncoef:]

        if not res.success:
            raise Exception(res.message)

        return coef

def solve_disutility_SSD_constrained_least_squares_trust_region(samples,values,eval_basis_matrix,eta_indices=None,probabilities=None):
    """
    Disutility formuation
    -Y dominates -Z
    """
    num_samples = samples.shape[1]
    if probabilities is None:
        probabilities = np.ones((num_samples))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)

    basis_matrix = eval_basis_matrix(samples)

    ssd_opt_problem = TrustRegionDisutilitySSDOptProblem(
        basis_matrix,values[:,0],values[eta_indices,0],probabilities)

    coef = ssd_opt_problem.solve()

    return coef, ssd_opt_problem


class SmoothDisutilitySSDOptProblem(TrustRegionDisutilitySSDOptProblem):
    def __init__(self,basis_matrix,values,eta,probabilities,smoother_type=0,
                 eps=1e-3):
        self.eps=eps
        self.smoother_type=smoother_type
        if self.eps is None:
            if self.smoother_type==0:
                self.eps=1e-2
            else:
                self.eps=1e-3

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
        self.nunknowns=self.ncoef

        # define bounds
        lb = np.zeros(self.nunknowns)
        lb[:self.ncoef] = -np.inf
        ub = np.ones(self.nunknowns)
        ub[:self.ncoef] = np.inf
        self.bounds = Bounds(lb,ub)
        
        # define initial guess
        self.init_guess = np.zeros((self.nunknowns))
        lstsq_coef = np.linalg.lstsq(
            self.basis_matrix,self.values,rcond=None)[0]
        self.init_guess[:self.ncoef] = lstsq_coef
        shift=np.max(self.values-self.basis_matrix.dot(
            self.init_guess))
        # for some reason just shifting exactly causes algorithm to not converge
        self.init_guess[0]+=shift*(1+self.eps)

        self.constraint_hessians=None

        self.smoother1=self.smooth_max_function
        self.smoother1_first_derivative=self.smooth_max_function_first_derivative
        self.smoother1_second_derivative=\
            self.smooth_max_function_second_derivative

        self.smoother2=self.smoother1
        self.smoother2_first_derivative=self.smoother1_first_derivative
        self.smoother2_second_derivative=self.smoother1_second_derivative

    def design_feasiable(self,x,feastol=1e-15):
        return np.all(self.nonlinear_constraints(x)<-feastol)

    def smooth_max_function(self,x):
        if self.smoother_type==0:
            I = np.where(np.isfinite(np.exp(-x/self.eps)))
            vals = np.zeros_like(x)
            vals[I] = (x[I] + self.eps*np.log(1+np.exp(-x[I]/self.eps)))
            assert np.all(np.isfinite(vals))
            return vals
        elif self.smoother_type==1:
            vals = np.zeros(x.shape)
            I = np.where((x>0)&(x<self.eps))#[0]
            vals[I]=x[I]**3/self.eps**2*(1-x[I]/(2*self.eps))
            J = np.where(x>=self.eps)#[0]
            vals[J]=x[J]-self.eps/2
            return vals
        else:
            msg="incorrect smoother_type"
            raise Exception(msg)

    def smooth_max_function_first_derivative(self,x):
        if self.smoother_type==0:
            #vals = 1.-1./(1+np.exp(x/self.eps))
            vals = 1./(1+np.exp(-x/self.eps))
            assert np.all(np.isfinite(vals))
            return vals
        elif self.smoother_type==1:
            vals = np.zeros(x.shape)
            I = np.where((x>0)&(x<self.eps))#[0]
            vals[I]=(x[I]**2*(3*self.eps-2*x[I]))/self.eps**3
            J = np.where(x>=self.eps)#[0]
            vals[J]=1
            return vals
        else:
            msg="incorrect smoother_type"
            raise Exception(msg)

    def smooth_max_function_second_derivative(self,x):
        if self.smoother_type==0:
            vals = 1/(self.eps*(np.exp(-x/self.eps)+2+np.exp(x/self.eps)))
            assert np.all(np.isfinite(vals))
            return vals
        elif self.smoother_type==1:
            vals = np.zeros(x.shape)
            I = np.where((x>0)&(x<self.eps))#[0]
            vals[I]=6*x[I]*(self.eps-x[I])/self.eps**3
            return vals
        else:
            msg="incorrect smoother_type"
            raise Exception(msg)
        
    def nonlinear_constraints(self,x,constraint_indices=None):
        if constraint_indices is None:
            constraint_indices=np.arange(self.nnl_constraints)
        constraint_indices = np.atleast_1d(constraint_indices)
            
        coef = x[:self.ncoef]
        approx_values = self.basis_matrix.dot(coef)
        constraint_values = np.zeros(constraint_indices.shape)
        for ii,index in enumerate(constraint_indices):
            constraint_values[ii] = self.probabilities.dot(
                self.smoother1(approx_values-approx_values[index]))
            constraint_values[ii] -= self.probabilities.dot(
                self.smoother2(self.values-approx_values[index]))
        assert np.all(np.isfinite(constraint_values))
        return constraint_values

    def nonlinear_constraints_jacobian(self,x):
        coef = x[:self.ncoef]
        approx_values = self.basis_matrix.dot(coef)
        grad = np.empty((self.nnl_constraints,self.ncoef),dtype=float)
        for ii in range(self.nnl_constraints):
            tmp1=self.smoother1_first_derivative(approx_values-approx_values[ii])
            tmp2=self.smoother2_first_derivative(self.values-approx_values[ii])
            grad[ii,:] = self.probabilities.dot(
                tmp1[:,np.newaxis]*(self.basis_matrix-self.basis_matrix[ii,:]))
            grad[ii,:] -= self.probabilities.dot(
                tmp2[:,np.newaxis]*(-self.basis_matrix[ii,:]))
        return grad

    def define_nonlinear_constraint_hessian(self,x,ii):
        """
        d^2/dx^2 f(g(x))=g'(x)^2 f''(g(x))+g''(x)f'(g(x))

        g''(x)=0 for all x
        """
        coef = x[:self.ncoef]
        approx_values = self.basis_matrix.dot(coef)
        hessian = np.zeros((self.nunknowns,self.nunknowns))
        tmp1 = self.smoother1_second_derivative(approx_values-approx_values[ii])
        tmp2 = self.smoother2_second_derivative(self.values-approx_values[ii])
        if np.all(tmp1==0) and np.all(tmp2==0):
            # Hessian will be zero
            return None

        tmp3 = self.basis_matrix-self.basis_matrix[ii,:]
        tmp4 = -self.basis_matrix[ii,:]
        
        for jj in range(self.nunknowns):
            for kk in range(jj,self.nunknowns):
                hessian[jj,kk] = self.probabilities.dot(
                    tmp1*(tmp3[:,jj]*tmp3[:,kk]))
                hessian[jj,kk] -= self.probabilities.dot(
                    tmp2*(tmp4[jj]*tmp4[kk]))
                hessian[kk,jj]=hessian[jj,kk]
        
        return hessian

    def nonlinear_constraints_hessian(self,x,v):
        initialize=False
        if self.constraint_hessians is None:
            self.constraint_hessians=[
                None for ii in range(self.nnl_constraints)]
            initialize=True

        assert v.shape[0]==self.nnl_constraints
        result = np.zeros((self.nunknowns,self.nunknowns))
        for ii in range(v.shape[0]):
            if initialize:
                self.constraint_hessians[ii]=\
                    self.define_nonlinear_constraint_hessian(x,ii)
            if self.constraint_hessians[ii] is not None:
                result += v[ii]*self.constraint_hessians[ii]
            #else hessian is zero
        return result

    def objective(self,x):
        from pyapprox.optimization import check_gradients
        #print(self.nonlinear_constraints(x))
        #print(self.nonlinear_constraints_jacobian(x))
        #check_gradients(
        #    self.nonlinear_constraints,self.nonlinear_constraints_jacobian,x,plot=True)
        return super().objective(x)

    def solve(self,optim_options=None):
        if optim_options is None:
            tol=1e-4
            optim_options={'verbose': 3, 'maxiter':1000,
                           'gtol':tol, 'xtol':tol, 'barrier_tol':tol}

        
        keep_feasible=False
        nonlinear_constraint = NonlinearConstraint(
            self.nonlinear_constraints, 0, np.inf,
            jac=self.nonlinear_constraints_jacobian,#jac='2-point',
            hess=self.nonlinear_constraints_hessian,
            #hess=BFGS(),
            keep_feasible=keep_feasible)

        constraints = [nonlinear_constraint]
        res = minimize(
            self.objective, self.init_guess,
            method='trust-constr',
            jac=self.objective_jacobian,
            hess=self.objective_hessian,
            constraints=constraints,options=optim_options,
            bounds=self.bounds)

        coef = res.x[:self.ncoef]

        if not res.success:
            raise Exception(res.message)

        return coef


def solve_disutility_SSD_constrained_least_squares_smooth(
        samples,values,eval_basis_matrix,eta_indices=None,probabilities=None,
        eps=None,smoother_type=0,return_full=False):
    """
    Disutility formuation
    -Y dominates -Z
    """
    num_samples = samples.shape[1]
    if probabilities is None:
        probabilities = np.ones((num_samples))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)
        
    basis_matrix = eval_basis_matrix(samples)

    ssd_opt_problem = SmoothDisutilitySSDOptProblem(
        basis_matrix,values[:,0],values[eta_indices,0],probabilities,eps=eps,
        smoother_type=smoother_type)

    coef = ssd_opt_problem.solve()

    if return_full:
        return coef, ssd_opt_problem
    else:
        return coef


class FSDOptProblem(SmoothDisutilitySSDOptProblem):
    def __init__(self,basis_matrix,values,eta,probabilities,smoother_type=0,
                 eps=None):
        super().__init__(basis_matrix,values,eta,probabilities,smoother_type,eps)

        smoother1,smoother2='shifted','shifted'
        #smoother1,smoother2='unshifted','unshifted'
        #smoother1,smoother2='shifted','unshifted'

        self.set_smoothers(smoother1,smoother2)

    def set_smoothers(self,smoother1,smoother2):
        if smoother1=='shifted':
            self.smoother1=self.shifted_smooth_heaviside_function
            self.smoother1_first_derivative=\
                self.shifted_smooth_heaviside_function_first_derivative
            self.smoother1_second_derivative=\
                self.shifted_smooth_heaviside_function_second_derivative

        if smoother1!='shifted':
            self.smoother1=self.smooth_heaviside_function
            self.smoother1_first_derivative=\
                self.smooth_heaviside_function_first_derivative
            self.smoother1_second_derivative=\
                self.smooth_heaviside_function_second_derivative

        if smoother2!='shifted':
            self.smoother2=self.smooth_heaviside_function
            self.smoother2_first_derivative=\
                self.smooth_heaviside_function_first_derivative
            self.smoother2_second_derivative=\
                self.smooth_heaviside_function_second_derivative

        if smoother2=='shifted':
            self.smoother2=self.shifted_smooth_heaviside_function
            self.smoother2_first_derivative=\
                self.shifted_smooth_heaviside_function_first_derivative
            self.smoother2_second_derivative=\
                self.shifted_smooth_heaviside_function_second_derivative

    def smooth_heaviside_function(self,x):
        """
            Heaviside function is approximated by the first derivative of
            the approximate postive part function
            x + self.eps*np.log(1+np.exp(-x/self.eps)
        """
        # one minus sign because using right heaviside function but want
        # left heaviside function
        if self.smoother_type==2:
            x=-x
            vals = np.zeros(x.shape)
            I = np.where((x>0)&(x<self.eps))
            vals[I] = 6*(x[I]/self.eps)**2-8*(x[I]/self.eps)**3+\
                      3*(x[I]/self.eps)**4
            J = np.where(x>=self.eps)
            vals[J]=1
            return vals

        vals = super().smooth_max_function_first_derivative(-x)
        return vals

    def smooth_heaviside_function_first_derivative(self,x):
        # two minus signs because using right heaviside function but want
        # left heaviside function
        if self.smoother_type==2:
            x=-x
            vals = np.zeros(x.shape)
            I = np.where((x>0)&(x<self.eps))
            vals[I] = 12*x[I]*(self.eps-x[I])**2/self.eps**4
            return -vals
        vals = -super().smooth_max_function_second_derivative(-x)
            
        return vals

    def smooth_max_function_third_derivative(self,x):
        # third derivative of max function
        if self.smoother_type==0:
            vals = np.zeros(x.shape)
            I = np.where(np.isfinite(np.exp(-x/self.eps)**3))
            vals[I]=np.exp(-x[I]/self.eps)*(np.exp(-x[I]/self.eps)-1)/(
                self.eps**2*(1+np.exp(-x[I]/self.eps))**3)
            assert np.all(np.isfinite(vals))
            return vals
        elif self.smoother_type==1:
            vals = np.zeros(x.shape)
            I = np.where((x>0)&(x<self.eps))#[0]
            vals[I]=6*(self.eps-2*x[I])/self.eps**3
            return vals
        elif self.smoother_type==2:
            vals = np.zeros(x.shape[0])
            I = np.where((x>0)&(x<self.eps))
            vals[I] = 12*(self.eps**2-4*self.eps*x[I]+3*x[I]**2)/self.eps**4
            return vals
        else:
            msg="incorrect smoother_type"
            raise Exception(msg)

    def smooth_heaviside_function_second_derivative(self,x):
        return self.smooth_max_function_third_derivative(-x)

    def shifted_smooth_heaviside_function(self,x):
        return self.smooth_heaviside_function(x-self.eps)
    
    def shifted_smooth_heaviside_function_first_derivative(self,x):
        return self.smooth_heaviside_function_first_derivative(x-self.eps)

    def shifted_smooth_heaviside_function_second_derivative(self,x):
        return self.smooth_heaviside_function_second_derivative(x-self.eps)

    def left_heaviside_function(self,x):
        vals = np.zeros_like(x)
        vals[x<=0]=1
        return vals
        
    # def nonlinear_constraints(self,x,constraint_indices=None):
    #     if constraint_indices is None:
    #         constraint_indices=np.arange(self.nnl_constraints)
    #     constraint_indices = np.atleast_1d(constraint_indices)
            
    #     coef = x[:self.ncoef]
    #     approx_values = self.basis_matrix.dot(coef)
    #     constraint_values = np.zeros(constraint_indices.shape)
    #     for ii,index in enumerate(constraint_indices):
    #         constraint_values[ii] = self.probabilities.dot(
    #             self.shifted_smooth_heaviside_function(
    #                 approx_values-approx_values[index]))
    #         #constraint_values[ii] -= self.probabilities.dot(
    #         #    self.smooth_heaviside_function(self.values-approx_values[index]))
    #         constraint_values[ii] -= self.probabilities.dot(
    #             self.shifted_smooth_heaviside_function(self.values-approx_values[index]))
    #     return constraint_values

    # def nonlinear_constraints_jacobian(self,x):
    #     coef = x[:self.ncoef]
    #     approx_values = self.basis_matrix.dot(coef)
    #     grad = np.empty((self.nnl_constraints,self.ncoef),dtype=float)
    #     for ii in range(self.nnl_constraints):
    #         tmp1 = self.shifted_smooth_heaviside_function_first_derivative(
    #             approx_values-approx_values[ii])
    #         grad[ii,:] = self.probabilities.dot(
    #             tmp1[:,np.newaxis]*(self.basis_matrix-self.basis_matrix[ii,:]))
    #         #tmp2 = self.smooth_heaviside_function_first_derivative(
    #         #    (self.values-approx_values[ii]))
    #         tmp2 = self.shifted_smooth_heaviside_function_first_derivative(
    #             (self.values-approx_values[ii]))
    #         grad[ii,:] -= self.probabilities.dot(
    #             tmp2[:,np.newaxis]*(-self.basis_matrix[ii,:]))

    #     return grad

    # def define_nonlinear_constraint_hessian(self,x,ii):
    #     """
    #     d^2/dx^2 f(g(x))=g'(x)^2 f''(g(x))+g''(x)f'(g(x))

    #     g''(x)=0 for all x
    #     """
    #     coef = x[:self.ncoef]
    #     approx_values = self.basis_matrix.dot(coef)
    #     hessian = np.zeros((self.nunknowns,self.nunknowns))
    #     # two minus signs because using right heaviside function but want
    #     # left heaviside function
    #     tmp1 = self.shifted_smooth_heaviside_function_second_derivative(
    #         approx_values-approx_values[ii])
    #     #tmp2 = self.smooth_heaviside_function_second_derivative(
    #     #    self.values-approx_values[ii])
    #     tmp2 = self.shifted_smooth_heaviside_function_second_derivative(
    #         self.values-approx_values[ii])
    #     if np.all(tmp1==0) and np.all(tmp2==0):
    #         # Hessian will be zero
    #         return None

    #     tmp3 = self.basis_matrix-self.basis_matrix[ii,:]
    #     tmp4 = -self.basis_matrix[ii,:]
        
    #     for jj in range(self.nunknowns):
    #         for kk in range(jj,self.nunknowns):
    #             hessian[jj,kk] = self.probabilities.dot(
    #                 tmp1*(tmp3[:,jj]*tmp3[:,kk]))
    #             hessian[jj,kk] -= self.probabilities.dot(
    #                 tmp2*(tmp4[jj]*tmp4[kk]))
    #             hessian[kk,jj]=hessian[jj,kk]
        
    #     return hessian

    # def nonlinear_constraints_hessian(self,x,v):
    #     initialize=False
    #     if self.constraint_hessians is None:
    #         self.constraint_hessians=[
    #             None for ii in range(self.nnl_constraints)]
    #         initialize=True

    #     assert v.shape[0]==self.nnl_constraints
    #     result = np.zeros((self.nunknowns,self.nunknowns))
    #     for ii in range(v.shape[0]):
    #         if initialize:
    #             self.constraint_hessians[ii]=\
    #                 self.define_nonlinear_constraint_hessian(x,ii)
    #         if self.constraint_hessians[ii] is not None:
    #             result += v[ii]*self.constraint_hessians[ii]
    #         #else hessian is zero
    #     return result

    def solve_slsqp(self,optim_options):
        if optim_options is None:
            tol=1e-6
            optim_options={'disp': True, 'maxiter':1000,
                           'ftol':tol,'iprint':3}

        # define nonlinear inequality constraints        
        ineq_cons_fun = lambda x: -self.nonlinear_constraints(x)
        ineq_cons_jac = lambda x: -self.nonlinear_constraints_jacobian(x)
        ineq_cons = {'type': 'ineq', 'fun' : ineq_cons_fun, 'jac': ineq_cons_jac}

        constraints = [ineq_cons]
        res = minimize(
            self.objective, self.init_guess, method='SLSQP',
            jac=self.objective_jacobian,
            constraints=constraints,options=optim_options,
            bounds=self.bounds)

        coef = res.x[:self.ncoef]
        
        if not res.success:
            #print(res.message)
            raise Exception(res.message)

        return coef

    def solve(self,optim_options=None):
        #np.seterr(all='raise')
        #import warnings
        #warnings.filterwarnings('error')
        assert self.smoother_type==0
        
        # sstore user defined options
        eps=self.eps
        smoother_type = self.smoother_type

        #if local smoother requested first use global smoother to get
        if smoother_type!=0:
            # a good initial guess. The global smoother can only use larger
            # eps tolerances
            self.eps=max(1e-3,eps)
        self.smoother_type=0
        #coef = self.solve_slsqp(optim_options)
        coef = self.solve_trust_region(optim_options)

        if smoother_type!=0:
            self.eps=eps
            self.init_guess=coef
            self.smoother_type=smoother_type
            #coef = self.solve_slsqp(optim_options)
            coef = self.solve_trust_region(optim_options)

        return coef

    def solve_trust_region(self,optim_options):

        if optim_options is None:
            tol=1e-7
            optim_options={'verbose': 3, 'maxiter':1000,
                           'gtol':tol, 'xtol':1e-15, 'barrier_tol':tol}

        # define constraints
        #keep_feasible=False
        keep_feasible=True
        nonlinear_constraint = NonlinearConstraint(
            self.nonlinear_constraints, -np.inf, 0,
            jac=self.nonlinear_constraints_jacobian,
            hess=self.nonlinear_constraints_hessian,
            #jac='2-point',
            #hess=BFGS(),
            keep_feasible=keep_feasible)
    
        constraints = [nonlinear_constraint]
        
        #return self.init_guess[:,np.newaxis], self

        res = minimize(
            self.objective, self.init_guess,
            method='trust-constr',
            jac=self.objective_jacobian,
            hess=self.objective_hessian,
            #jac='2-point',
            #hess=BFGS(),
            constraints=constraints,options=optim_options,
            bounds=self.bounds)

        coef = res.x[:self.ncoef]

        # print(self.eps)
        # print('lag grad  ',res.lagrangian_grad)
        # print('obj grad  ',res.grad)
        # print('constr jac',res.jac)
        # print('constr    ',res.constr)

        if not res.success:
            #print(res.message)
            raise Exception(res.message)
    
        return coef

        
    
def solve_FSD_constrained_least_squares_smooth(
        samples,values,eval_basis_matrix,eta_indices=None,probabilities=None,
        eps=None):
    """
    First order stochastic dominance FSD
    """
    num_samples = samples.shape[1]
    if probabilities is None:
        probabilities = np.ones((num_samples))/num_samples
    if eta_indices is None:
        eta_indices = np.arange(0,num_samples)

    basis_matrix = eval_basis_matrix(samples)

    smoother_type=0
    fsd_opt_problem = FSDOptProblem(
        basis_matrix,values[:,0],values[eta_indices,0],probabilities,eps=eps,
        smoother_type=smoother_type)

    coef = fsd_opt_problem.solve()

    return coef, fsd_opt_problem
