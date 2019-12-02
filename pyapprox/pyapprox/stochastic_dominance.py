import numpy as np
from scipy import sparse
from scipy.sparse import eye as speye
from scipy.sparse import lil_matrix
from cvxopt import matrix, solvers, spmatrix

def build_inequality_contraints(Y,Yvec,basis_matrix,p,
                                disutility_formulation=True):
    """
    Construct the matrix form of the constraints of quadratic program.
       Ax<=b

    z_k+s_{ik} >= y_i, i=1,...,N, k=1,...,N
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

    Returns
    -------
    A : np.ndarray (2N**2+N,N**2+M)
       The constraints matrix. Contains contraints that enforce s_ik >=0

    A : np.ndarray (2N**2+N,1)
       The constraints RHS. Contains contraints that enforce s_ik >=0
    """
    N,M = basis_matrix.shape
    I = speye(N,N)

    num_opt_vars  = M  # number of polynomial coefficients
    num_opt_vars += N*N # number of decision vars s_{ik}
    
    num_constraints  = N*N # z_k+s_{ik} >= y_i
    num_constraints += N   # \sum_{k=1}^N p_k s_{ik} <= v_i = E[(y_i-Y)^{+}]
    num_constraints += N*N # s_ik>=0
    A = lil_matrix((num_constraints,num_opt_vars))
    b = lil_matrix((num_constraints,1))
    for i in range(N):
        row = i*N + i
        col = M + i*N
        if not disutility_formulation:
            # z_k+s_{ik} >= y_i
            A[row:row+N,:M]        = -basis_matrix
            A[row:row+N,col:col+N] = -I
            b[row:row+N]           = -Y
            ## \sum_{k=1}^N p_k s_{ik} <= v_i = E[(y_i-Y)^{+}]
            A[row+N,col:col+N] = p.T
            b[row+N]           = Yvec[i]
        else:
            # z_k-s_{ik} <= y_i
            A[row:row+N,:M]        = basis_matrix
            A[row:row+N,col:col+N] = -I
            b[row:row+N]           = Y
            # \sum_{k=1}^N p_k s_{ik} >= v_i = E[(Y-y_i)^{+}]
            A[row+N,col:col+N] = -p.T
            b[row+N]           = -Yvec[i]
            
        # s_ik>=0
        row = i*N+N*N+N
        A[row:row+N,col:col+N] = -I
        b[row:row+N]     = 0
    return A,b


def compute_conditional_expectations(eta,samples,disutility_formulation=True):
    assert samples.ndim==1
    assert eta.ndim==1
    if disutility_formulation:
        values = np.asarray(
            [np.maximum(samples-eta[ii],0).mean() for ii in range(eta.shape[0])])
    else:
        values = np.asarray(
            [np.maximum(eta[ii]-samples,0).mean() for ii in range(eta.shape[0])])
    return values

def gradient(coeff0,N,M):
    """
    gradient g of x'Hx+g'x
    """
    #g = np.zeros((M+N*N,1))
    #g[:M] = -coeff0
    g = lil_matrix((M+N*N,1))
    g[:M] = -coeff0
    return g

def hessian(basis_matrix,use_sample_average=True):
    """
    Hessian H of x'Hx+g'x
    """
    N, M = basis_matrix.shape
    H = lil_matrix((M+N*N,M+N*N))
    if use_sample_average:
        H[:M,:M] = basis_matrix.T.dot(basis_matrix)/N
    else:
        H[:M,:M] = speye(M,M)
    return H

def solve_stochastic_dominance_contrained_least_squares(
        samples,values,eval_basis_matrix,lstsq_coef=None,
        disutility_formulation=False):
    # Compute coefficients with second order stochastic dominance constraints
    num_samples = samples.shape[1]
    basis_matrix = eval_basis_matrix(samples)
    Yvec    = compute_conditional_expectations(
        values[:,0],values[:,0],disutility_formulation)
    probabilities = np.ones((num_samples,1))
    [A,b]=build_inequality_contraints(
        values,Yvec,basis_matrix,probabilities,disutility_formulation)

    #if lstsq_coef is None:
    #    lstsq_coef = np.linalg.lstsq(
    #        eval_basis_matrix(samples),values,rcond=None)[0]
    lstsq_coef = eval_basis_matrix(samples).T.dot(values)/num_samples

    num_basis_terms = basis_matrix.shape[1]
    # minimize distance between lstsq solution and ssd solution
    g = gradient(lstsq_coef,num_samples,num_basis_terms)
    H = hessian(basis_matrix)

    # TODO use CVXOPT sparse matrix format
    I,J,data = sparse.find(A)
    A_sparse = spmatrix(data,I,J,size=A.shape)
    I,J,data = sparse.find(H)
    H_sparse = spmatrix(data,I,J,size=H.shape);

    A = A_sparse
    H = H_sparse
    # g and b must be of type matrix and not type spmatrix
    g = matrix(g.todense())
    b = matrix(b.todense())

    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8
    solvers.options['maxiters'] = 1000
    # Minimize x'Hx+g'x subject to Ax <= b
    print ("optimizing")
    result = solvers.qp(H,g,A,b,Aeq=None,beq=None)
    print ("done")
    ssd_solution = np.array(result['x'])
    coef = ssd_solution[:num_basis_terms]
    return coef

