import numpy as np
from cvxopt import matrix, solvers, spmatrix
from matplotlib import pyplot as plt

from scipy.stats.mstats import mquantiles as quantile
from scipy.stats import norm as normal_rv
from scipy.special import erfinv
from scipy import sparse

def value_at_risk(samples,alpha,weights=None,samples_sorted=False):
    """
    Compute the value at risk of a variable Y using a set of samples.

    Parameters
    ----------
    samples : np.ndarray (num_samples)
        Samples of the random variable Y

    alpha : integer
        The superquantile parameter

    weights : np.ndarray (num_samples)
        Importance weights associated with each sample. If samples Y are drawn
        from biasing distribution g(y) but we wish to compute VaR with respect
        to measure f(y) then weights are the ratio f(y_i)/g(y_i) for 
        i=1,...,num_samples.

    Returns
    -------
    cvar : float
        The conditional value at risk of the random variable Y
    """
    num_samples = samples.shape[0]
    if weights is None:
        weights = np.ones(num_samples)
    if not samples_sorted:
        I = np.argsort(samples)
        xx,ww = samples[I],weights[I]
    else:
        xx,ww = samples,weights
    ecdf = ww.cumsum()
    ecdf/=ecdf[-1]
    index = np.arange(num_samples)[ecdf>=alpha][0]
    return xx[index],index

def conditional_value_at_risk(samples,alpha,weights=None):
    """
    Compute conditional value at risk of a variable Y using a set of samples.

    Note accuracy of Monte Carlo Estimate of CVaR is dependent on alpha.
    As alpha increases more samples will be needed to achieve a fixed
    level of accruracy.

    Parameters
    ----------
    samples : np.ndarray (num_samples)
        Samples of the random variable Y

    alpha : integer
        The superquantile parameter

    weights : np.ndarray (num_samples)
        Importance weights associated with each sample. If samples Y are drawn
        from biasing distribution g(y) but we wish to compute VaR with respect
        to measure f(y) then weights are the ratio f(y_i)/g(y_i) for 
        i=1,...,num_samples.

    Returns
    -------
    cvar : float
        The conditional value at risk of the random variable Y
    """
    assert samples.ndim==1
    num_samples = samples.shape[0]
    if weights is None:
        weights = np.ones(num_samples)
    I = np.argsort(samples)
    xx,ww = samples[I],weights[I]
    VaR,index = value_at_risk(xx,alpha,ww,samples_sorted=True)
    CVaR=VaR+1/((1-alpha)*num_samples)*np.sum((xx[index+1:]-VaR)*ww[index+1:])
    return CVaR

def cvar_smoothing_function_I(samples,eps):
    return (samples + eps*np.log(1+np.exp(-samples/eps)))

def smoothed_conditional_value_at_risk(samples,alpha,eps):
    assert samples.ndim==1
    num_samples = samples.shape[0]
    q = quantile(samples,alpha)
    cvar_eps = cvar_smoothing_function_I(samples-q,eps).sum()
    cvar_eps /= ((1-alpha)*num_samples)
    return cvar_eps + np.asscalar(q)

def cvar_regression_quadrature(basis_matrix,values,alpha,nquad_intervals,
                               verbosity=1,trapezoid_rule=False,
                               solver_name='cvxopt'):
    """
    solver_name = 'cvxopt'
    solver_name='glpk'

    trapezoid works but default option is better.
    """
    
    assert alpha<1 and alpha>0
    basis_matrix=basis_matrix[:,1:]
    assert basis_matrix.ndim==2
    assert values.ndim==1
    nsamples,nbasis = basis_matrix.shape
    assert values.shape[0]==nsamples

    if not trapezoid_rule:
        # left-hand piecewise constant quadrature rule
        beta = np.linspace(alpha,1,nquad_intervals+2)[:-1]# quadrature points
        dx = beta[1]-beta[0]
        weights = dx*np.ones(beta.shape[0])
        nuvars = weights.shape[0]
        nvconstraints = nsamples*nuvars
        num_opt_vars = nbasis + nuvars + nvconstraints
        num_constraints = 2*nsamples*nuvars
    else:
        beta = np.linspace(alpha,1,nquad_intervals+1)# quadrature points
        dx = beta[1]-beta[0]
        weights = dx*np.ones(beta.shape[0])
        weights[0]/=2; weights[-1]/=2
        weights = weights[:-1] # ignore left hand side
        beta = beta[:-1]
        nuvars = weights.shape[0]
        nvconstraints = nsamples*nuvars
        num_opt_vars = nbasis + nuvars + nvconstraints + 1
        num_constraints = 2*nsamples*nuvars+nsamples
    
    v_coef = weights/(1-beta)*1./nsamples*1/(1-alpha)
    #print (beta
    #print (weights
    #print (v_coef

    Iquad = np.identity(nuvars)
    Iv  = np.identity(nvconstraints)

    # num_quad_point = mu
    # nsamples = nu
    # nbasis = m

    # design vars [c_1,...,c_m,u_1,...,u_{mu+1},v_1,...,v_{mu+1}nu]
    # v_ij variables ordering: loop through j fastest, e.g. v_11,v_{12} etc

    if not trapezoid_rule:
        c_arr = np.hstack((
            basis_matrix.sum(axis=0)/nsamples,
            1/(1-alpha)*weights,
            np.repeat(v_coef,nsamples)))

        # # v_ij+h'c+u_i <=y_j
        # constraints_1 = np.hstack((
        #     -np.tile(basis_matrix,(nuvars,1)),
        #     -np.repeat(Iquad,nsamples,axis=0),-Iv))
        # # v_ij >=0
        # constraints_3 = np.hstack((
        #    np.zeros((nvconstraints,nbasis+nuvars)),-Iv))

        # G_arr = np.vstack((constraints_1,constraints_3))
        # assert G_arr.shape[0]==num_constraints
        # assert G_arr.shape[1]==num_opt_vars
        # assert c_arr.shape[0]==num_opt_vars
        # I,J,data = sparse.find(G_arr)
        # G = spmatrix(data,I,J,size=G_arr.shape)

        #v_ij+h'c+u_i <=y_j
        constraints_1_shape = (nvconstraints,num_opt_vars)
        constraints_1a_I = np.repeat(np.arange(nvconstraints),nbasis)
        constraints_1a_J = np.tile(np.arange(nbasis),nvconstraints)
        constraints_1a_data = -np.tile(basis_matrix,(nquad_intervals+1,1))

        constraints_1b_I = np.arange(nvconstraints)
        constraints_1b_J = np.repeat(
            np.arange(nquad_intervals+1),nsamples)+nbasis
        constraints_1b_data = -np.repeat(np.ones(nquad_intervals+1),nsamples)

        ii = nbasis+nquad_intervals+1; jj = ii+nvconstraints
        constraints_1c_I = np.arange(nvconstraints)
        constraints_1c_J = np.arange(ii,jj)
        constraints_1c_data = -np.ones((nquad_intervals+1)*nsamples)

        constraints_1_data = np.hstack((
            constraints_1a_data.flatten(),constraints_1b_data,
            constraints_1c_data))
        constraints_1_I = np.hstack(
            (constraints_1a_I,constraints_1b_I,constraints_1c_I))
        constraints_1_J = np.hstack(
            (constraints_1a_J,constraints_1b_J,constraints_1c_J))

        # v_ij >=0
        constraints_3_I = np.arange(
            constraints_1_shape[0],constraints_1_shape[0]+nvconstraints)
        constraints_3_J = np.arange(
            nbasis+nquad_intervals+1,nbasis+nquad_intervals+1+nvconstraints)
        constraints_3_data = -np.ones(nvconstraints)

        constraints_shape=(num_constraints,num_opt_vars)
        constraints_I = np.hstack((constraints_1_I,constraints_3_I))
        constraints_J = np.hstack((constraints_1_J,constraints_3_J))
        constraints_data = np.hstack((constraints_1_data,constraints_3_data))
        G = spmatrix(
            constraints_data,constraints_I,constraints_J,size=constraints_shape)
        # assert np.allclose(np.asarray(matrix(G)),G_arr)

        # print (constraints_shape
        # print (np.asarray(matrix(G))

        h_arr = np.hstack((
            -np.tile(values,nuvars),
            np.zeros(nvconstraints)))
        #print (G_arr
        #print (c_arr
        #print (h_arr

    else:
        c_arr = np.hstack((
            basis_matrix.sum(axis=0)/nsamples,
            1/(1-alpha)*weights,
            np.repeat(v_coef,nsamples),
            1/(nsamples*(1-alpha))*np.ones(1)))

        # v_ij+h'c+u_i <=y_j
        constraints_1 = np.hstack((
            -np.tile(basis_matrix,(nquad_intervals,1)),
            -np.repeat(Iquad,nsamples,axis=0),
            -Iv,np.zeros((nvconstraints,1))))

        #W+h'c<=y_j
        constraints_2 = np.hstack((
            -basis_matrix,
            np.zeros((nsamples,nuvars)),
            np.zeros((nsamples,nvconstraints)),
            -np.ones((nsamples,1))))
        
        # v_ij >=0
        constraints_3 = np.hstack((
            np.zeros((nvconstraints,nbasis+nquad_intervals)),-Iv,
            np.zeros((nvconstraints,1))))
        
        G_arr = np.vstack((constraints_1,constraints_2,constraints_3))

        h_arr = np.hstack((
            -np.tile(values,nuvars),
            -values,
            np.zeros(nvconstraints)))

        assert G_arr.shape[0]==num_constraints
        assert G_arr.shape[1]==num_opt_vars
        assert c_arr.shape[0]==num_opt_vars
        I,J,data = sparse.find(G_arr)
        G = spmatrix(data,I,J,size=G_arr.shape)

    c = matrix(c_arr)
    h = matrix(h_arr)
    if verbosity<1:
        solvers.options['show_progress'] = False
    else:
        solvers.options['show_progress'] = True

    # solvers.options['abstol'] = 1e-10
    # solvers.options['reltol'] = 1e-10
    # solvers.options['feastol'] = 1e-10
        
    sol = np.asarray(
        solvers.lp(c=c, G=G, h=h, solver=solver_name)['x'])[:nbasis]
    residuals = values-basis_matrix.dot(sol)[:,0]
    coef = np.append(conditional_value_at_risk(residuals,alpha),sol)
    return coef

def cvar_regression(basis_matrix, values, alpha,verbosity=1):
    # do not include constant basis in optimization
    assert alpha<1 and alpha>0
    basis_matrix=basis_matrix[:,1:]
    assert basis_matrix.ndim==2
    assert values.ndim==1
    nsamples,nbasis = basis_matrix.shape
    assert values.shape[0]==nsamples

    active_index = int(np.ceil(alpha*nsamples))-1# 0 based index 0,...,nsamples-1
    nactive_samples = nsamples-(active_index+1)
    assert nactive_samples>0, ('no samples in alpha quantile')
    #print (nactive_samples,active_index,nsamples
    beta = np.arange(1,nsamples+1,dtype=float)/nsamples
    #print (beta
    beta[active_index-1]=alpha
    
    #print (beta
    #print (beta[active_index-1],nactive_samples
    
    beta_diff = np.diff(beta[active_index-1:-1])
    #print (beta_diff
    assert beta_diff.shape[0]==nactive_samples
    v_coef = np.log(1 - beta[active_index-1:-2]) - np.log(
        1 - beta[active_index:-1])
    v_coef /= nsamples*(1-alpha)
    #print (beta[active_index-1:-2],beta[active_index:-1]
    #print (v_coef

    nvconstraints = nsamples*nactive_samples
    Iv  = np.identity(nvconstraints)
    Isamp = np.identity(nsamples)
    Iactsamp = np.identity(nactive_samples)

    # nactive_samples = p
    # nsamples = m
    # nbasis = n

    # design vars [c_1,...,c_n,u1,...,u_{m-p},v_1,...,v_{m-p}m,w]

    # print ('a'
    # print ((        basis_matrix.sum(axis=0).shape,
    #     beta_diff.shape,
    #     #np.tile(v_coef,nsamples).shape,  # tile([1,2],2)   = [1,2,1,2] 
    #     np.repeat(v_coef,nsamples).shape, # repeat([1,2],2) = [1,1,2,2]
    #     np.ones(1).shape,v_coef.shape,nactive_samples,nsamples)
    
    c_arr = np.hstack((
        basis_matrix.sum(axis=0)/nsamples,
        1/(1-alpha)*beta_diff,
        #np.tile(v_coef,nsamples),  # tile([1,2],2)   = [1,2,1,2] 
        np.repeat(v_coef,nsamples), # repeat([1,2],2) = [1,1,2,2]
        1./(nsamples*(1-alpha))*np.ones(1)))

    #print (basis_matrix.sum(axis=0).shape,nsamples,nactive_samples
    # print ((
    #     np.tile(basis_matrix,(nactive_samples,1)).shape,
    #     np.repeat(Iactsamp,nsamples,axis=0).shape,
    #     Iv.shape,
    #     np.zeros((nvconstraints,1)).shape)

    num_opt_vars = nbasis + nactive_samples + nvconstraints + 1
    # v_ij variables ordering: loop through j fastest, e.g. v_11,v_{12} etc
    
    #v_ij+h'c+u_i <=y_j
    constraints_1 = np.hstack((
        -np.tile(basis_matrix,(nactive_samples,1)),
        -np.repeat(Iactsamp,nsamples,axis=0),
        -Iv,
        np.zeros((nvconstraints,1))))
    
    #W+h'c<=y_j
    constraints_2 = np.hstack((
        -basis_matrix,
        np.zeros((nsamples,nactive_samples)),
        np.zeros((nsamples,nvconstraints)),
        -np.ones((nsamples,1))))

    # v_ij >=0
    constraints_3 = np.hstack((
            np.zeros((nvconstraints,nbasis+nactive_samples)),
            -Iv,np.zeros((nvconstraints,1))))
    
    #print ((constraints_1.shape, constraints_2.shape, constraints_3.shape)
    G_arr = np.vstack((constraints_1,constraints_2,constraints_3))
    
    h_arr = np.hstack((
        -np.tile(values,nactive_samples),
        -values,np.zeros(nvconstraints)))

    # print (G_arr
    # print (c_arr
    # print (h_arr

    assert G_arr.shape[1]==num_opt_vars
    assert G_arr.shape[0]==h_arr.shape[0]
    assert c_arr.shape[0]==num_opt_vars


    #print (c_arr
    # print (G_arr
    # print (h_arr
    # print ('rank',np.linalg.matrix_rank(G_arr), G_arr.shape
    # print (h_arr.shape
    
    c = matrix(c_arr)
    #G = matrix(G_arr)
    h = matrix(h_arr)

    from scipy import sparse
    I,J,data = sparse.find(G_arr)
    G = spmatrix(data,I,J,size=G_arr.shape)    
    if verbosity<1:
        solvers.options['show_progress'] = False
    else:
        solvers.options['show_progress'] = True

    # solvers.options['abstol'] = 1e-10
    # solvers.options['reltol'] = 1e-10
    # solvers.options['feastol'] = 1e-10
        
    sol = np.asarray(solvers.lp(c=c, G=G, h=h)['x'])[:nbasis]
    residuals = values-basis_matrix.dot(sol)[:,0]
    coef = np.append(conditional_value_at_risk(residuals,alpha),sol)
    return coef

#sudo yum install glpk-devel glpk
# pip install cvxopt
