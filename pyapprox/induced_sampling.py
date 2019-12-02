from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np, os
from pyapprox.orthonormal_polynomials_1d import jacobi_recurrence,\
     gauss_quadrature
from scipy.special import betaln
from scipy.special import beta as betafn
#from scipy.optimize import fsolve
from scipy.optimize import brenth
from functools import partial
from pyapprox.utilities import discrete_sampling
from pyapprox.variables import is_bounded_continuous_variable
def C_eval(a, b, x, N):
    """
    C_eval -- Evaluates Christoffel-normalized orthogonal polynomials

    C = C_eval(a, b, x, N)
    Uses the recurrence coefficients a and b to evaluate C_n(x),
    defined as

    C_n(x) = p_n(x) / sqrt(sum_{j=0}^{n-1} p_j^2(x)),

    where p_n(x) is the degree-n orthonormal polynomial associated with the
    recurrence coefficients a, b (with positive leading coefficient). We define
    C_{-1} = 0, and C_0 = p_0.

    The p_n satisfy the recurrences
    
    sqrt(b_{n+1}) p_{n+1} = (x - a_n) p_n - sqrt(b_n) p_{n-1}

    With the input arrays a and b, we have a_n = a(n+1), b_n = b(n+1). Note
    that b_0 = b(1) is only used to initialize p_0.

    The output matrix C has size numel(x) x N, and hence the first N (up to
    degree N-1) polynomials are evaluated.

    Inputs:
        x : array of doubles
        N : positive integer, N - 1 < length(a) == length(b)
        a : array of recurrence coefficients
        b : array of reucrrence coefficients
    """
    if np.isscalar(x):
        x = np.asarray([x])
    nx = x.shape[0]

    assert(N >= 0)

    assert(N <= a.shape[0])
    assert(N <= b.shape[0])

    C = np.zeros((nx,N+1))

    # Flatten x
    xf = x.flatten()

    # To initialize C, we need p_0 and p_1
    C[:,0] = 1./np.sqrt(b[0]) # C0 = p0
    if N > 0:
        C[:,1] = 1/np.sqrt(b[1]) * (xf - a[0])
        # C1 = p1/p0

    if N > 1:
        C[:,2] = 1./np.sqrt(1+C[:,1]**2)*((xf-a[1])*C[:,1]-np.sqrt(b[1]))
        C[:,2] = C[:,2]/np.sqrt(b[2])

    for n in range(2,N):
        C[:,n+1] = 1./np.sqrt(1+C[:,n]**2)*\
          ((xf-a[n])*C[:,n]-np.sqrt(b[n])*C[:,n-1]/np.sqrt(1+C[:,n-1]**2))
        C[:,n+1] = C[:,n+1]/np.sqrt(b[n+1]);
        
    return C

     
def ratio_eval(a, b, x, N):
    """
    ratio_eval -- Evaluates ratios between successive orthogonal polynomials

    r = ratio_eval(a, b, x, N)
    Uses the recurrence coefficients a and b to evaluate the ratio r at
    locations x, of orders n = 1, ..., N. A ratio r_n(x) of order n is defined
    by

    r_n(x) = p_n(x) / p_{n-1}(x),

    where p_n(x) is the degree-n orthonormal polynomial associated with the
    recurrence coefficients a, b (with positive leading coefficient).

    The p_n and r_n satisfy the recurrences

    sqrt(b_{n+1}) p_{n+1} = (x - a_n) p_n - sqrt(b_n) p_{n-1}
    sqrt(b_{n+1}) r_{n+1} = (x - a_n)  - sqrt(b_n) / r_n

    With the input arrays a and b, we have a_n = a(n+1), b_n = b(n+1). Note
    that b_0 = b(1) is only used to initialize p_0.

    For the Nevai class, we expect a_n ---> 0, b_n ---> 1/4

    The output matrix r has size numel(x) x N

    Inputs:
        x : array of doubles
        N : positive integer, N - 1 <= length(a) == length(b)
        a : array of recurrence coefficients
        b : array of reucrrence coefficients
    """

    if np.isscalar(x):
        x = np.asarray([x])
    
    nx = x.shape[0]
    assert(N > 0)
    assert(N < a.shape[0])
    assert(N < b.shape[0])

    r = np.zeros((nx, N))

    # Flatten x
    xf = x.flatten()

    # To initialize r, we need p_0 and p_1
    p0 = 1/np.sqrt(b[0]) * np.ones((nx, 1))
    p1 = 1/np.sqrt(b[1]) * (xf - a[0])*p0

    r1 = p1/p0
    r[:,0] = r1

    for q in range(1,N):
        # Derived from three-term recurrence
        r2 = (xf - a[q]) - np.sqrt(b[q]) / r1
        r1 = 1/np.sqrt(b[q+1]) * r2

        r[:,q] = r1

    if nx==1:
        r = r.reshape(N,1)
    return r

def linear_modification(alph, bet, x0):
    """
    linear_modification -- Modifies recurrence coefficients

    [a,b] = linear_modification(alph, bet, x0)
    
    Performs a linear modification of orthogonal polynomial recurrence
    coefficients. The inputs alph and bet are three-term recurrence
    coefficients for a d(mu)-orthonormal polynomial family p_n:

    sqrt(bet_{n+1}) p_{n+1} = (x - alph_n) p_n - sqrt(bet_n) p_{n-1}

    This function transforms the alph, bet into new coefficients a, b such that
    the new coefficients determine a polynomial family that is orthonormal
    under the weight |x-x0|*d(mu), when x0 \not\in \supp \mu.

    The appropriate sign of the modification (+/- (x-x0)) is inferred from the
    sign of (alph(1) - x0). Since alph(1) is the zero of p_1, then it is in
    \supp \mu.

    Outputs:
        a: column vector (size (N-1)) of recurrence coefficients
        b: column vector (size (N-1)) of recurrence coefficients

    Inputs:
        x0   : real scalar, assumed \not \in \supp \mu
        alph : column vector (length-N) of recurrence coefficients
        bet  : column vector (length-N) of reucrrence coefficients
        x0   : real scalar, +1 or -1, so that sgn*(x-x0) > 0 on \supp \mu
    """

    N = alph.shape[0]
    assert(bet.shape[0] == N)
    assert(N > 1)

    sgn = np.sign(alph[0] - x0)

    r = np.abs(ratio_eval(alph, bet, x0, N-1))
    # r is length N-1

    acorrect = np.zeros((N-1, 1))
    bcorrect = np.zeros((N-1, 1))

    acorrect = np.sqrt(bet[1:N])/r
    acorrect[1:] = np.diff(acorrect,axis=0)

    bcorrect = np.sqrt(bet[1:N])*r
    bcorrect[1:] = bcorrect[1:]/bcorrect[:-1]

    b = bet[:N-1]  * bcorrect[:N-1]
    a = alph[:N-1] + sgn*acorrect[:N-1]
    return a,b

def quadratic_modification_C(alph, bet, x0):
    """
    quadratic_modification_C -- Modifies recurrence coefficients
    
    [a,b] = quadratic_modification_C(alph, bet, x0)
    
    Performs a quadratic modification of orthogonal polynomial recurrence
    coefficients. The inputs alph and bet are three-term recurrence
    coefficients for a d(mu)-orthonormal polynomial family p_n:
    
      sqrt(bet_{n+1}) p_{n+1} = (x - alph_n) p_n - sqrt(bet_n) p_{n-1}
    
    This function transforms the alph, bet into new coefficients a, b such that
    the new coefficients determine a polynomial family that is orthonormal
    under the weight (x-x0)^2*d(mu).
    
    This function uses the q functions to perform the recurrence updates,
    instead of the standard orthogonal polynomial basis.
    
    Outputs:
        a: column vector (size (N-2)) of recurrence coefficients
        b: column vector (size (N-2)) of recurrence coefficients
    
    Inputs:
        x0   : real scalar
        alph : column vector (length-N) of recurrence coefficients
        bet  : column vector (length-N) of reucrrence coefficients
    """

    N = alph.shape[0]
    assert((bet.shape[0] == N ) and (bet.ndim==2))
    assert(alph.ndim==2)
    assert(N > 2)

    # Output recurrence coefficients
    a = np.zeros((N-2, 1))
    b = np.zeros((N-2, 1))

    C = np.reshape(C_eval(alph, bet, x0, N-1), (N, 1))

    # q is length N --- new coefficients have length N-2
    acorrect = np.zeros((N-2, 1))
    bcorrect = np.zeros((N-2, 1))

    temp1 = np.sqrt(bet[1:N])*C[1:N]*C[:(N-1)]/np.sqrt(1 + C[:(N-1)]**2)
    temp1[0] = np.sqrt(bet[1])*C[1] # special case

    acorrect = np.diff(temp1,axis=0)

    temp1 = 1 + C[:(N-1)]**2
    bcorrect = temp1[1:]/temp1[:-1]

    bcorrect[0] = (1 + C[1]**2)/C[0]**2

    b = bet[1:N-1]  * bcorrect
    a = alph[1:N-1] + acorrect
    return a,b

def medapprox_jacobi(alph, bet, n):
    """
    x0 = medapprox_jacobi(alph, bet, n)

    Returns a guess for the median of the order-n Jacobi induced distribution
    with parameters alph, bet.
    """

    if n > 0:
        x0 = (bet**2 - alph**2)/(2*n + alph + bet)**2
    else:
        x0 = 2/(1 + (alph+1)/(bet+1)) - 1
    return x0

def idist_jacobi(x, n, alph, bet, M=10):
    """
    idist_jacobi -- Evaluation of induced distribution

    F = idist_jacobi(x, n, alph, bet, {M = 10})
    
    Evaluates the integral
    
        F = \int_{-1}**x p_n**2(x) \dx{\mu(x)},

    where mu is the (a,b) Jacobi polynomial measure, scaled to be a 
    probability distribution on [-1,1], and p_n is the corresponding 
    degree-n orthonormal polynomial.

    This function evaluates this via a transformation, measure modification,
    and Gauss quadrature, the ending Gauss quadrature has M points.
    """

    assert ((alph > -1) and (bet > -1))
    assert ( np.all(np.abs(x) <= 1) )
    assert np.all(n >= 0)
    if x.ndim==2:
        assert x.shape[1]==1
        x = x[:,0]

    A = int(np.floor(abs(alph))) # is an integer
    Aa = alph - A

    F = np.zeros((x.shape[0],1))

    mrs_centroid = medapprox_jacobi(alph, bet, n)
    xreflect = x > mrs_centroid

    if x[xreflect].shape[0]>0:
        F[xreflect] = 1 - idist_jacobi(-x[xreflect], n, bet, alph, M)

    recursion_coeffs = jacobi_recurrence(n+1, alph, bet, True)
    # All functions that accept b assume they are receiving b
    # but recusion_coeffs:,1=np.sqrt(b)
    a = recursion_coeffs[:,0]; b = recursion_coeffs[:,1]**2
    assert b[0] == 1 # To make it a probability measure

    if n > 0:
        # Zeros of p_n
        xn = gauss_quadrature(recursion_coeffs, n)[0]

        # This is the (inverse) n'th root of the leading coefficient square
        # of p_n. We'll use it for scaling later
    
        kn_factor = np.exp(-1./n*np.sum(np.log(b)))
        

    for xq in range(x.shape[0]):

        if x[xq] == -1:
            F[xq] = 0
            continue

        if xreflect[xq]:
            continue

        # Recurrence coefficients for quadrature rule
        recursion_coeffs = jacobi_recurrence(2*n+A+M+1, 0, bet, True)
        # All functions that accept b assume they are receiving b
        # but recusion_coeffs:,1=np.sqrt(b)
        a = recursion_coeffs[:,0:1]; b = recursion_coeffs[:,1:]**2
        assert b[0] == 1 # To make it a probability measure

        if n > 0:
            # Transformed
            un = (2./(x[xq]+1.)) * (xn + 1) - 1

        # Keep this so that bet(1) always equals what it did before
        logfactor = 0 

        # Successive quadratic measure modifications
        for j in range(n):
            a,b = quadratic_modification_C(a, b, un[j])

            logfactor += np.log(b[0]*((x[xq]+1)/2)**2 * kn_factor)
            b[0] = 1


        # Linear modification by factors (2 - 1/2*(u+1)*(x+1)),
        # having root u = (3-x)/(1+x)
        root = (3.-x[xq])/(1.+x[xq])
        for aq in range(A):
            [a, b] = linear_modification(a, b, root)

            logfactor += np.log(b[0] * 1/2 * (x[xq]+1))
            b[0] = 1

        # M-point Gauss quadrature for evaluation of auxilliary integral I
        # gauss quadrature requires np.sqrt(b)
        u,w = gauss_quadrature(np.hstack((a, np.sqrt(b))), M)
        I = np.dot(w.T,(2. - 1./2. * (u+1.) * (x[xq]+1) )**Aa)
        F[xq] = np.exp(logfactor - alph*np.log(2) - betaln(bet+1, alph+1) -
                       np.log(bet+1) + (bet+1)*np.log((x[xq]+1)/2)) * I
    return F

def idist_mixture_sampling(indices,univ_inv,weights=None,num_samples=None,
                           seed=None):
    """
    x = idist_mixture_sampling(indices, univ_inv, M)
    
    Performs tensorial inverse transform sampling from an additive mixture of
    tensorial induced distributions, generating M samples. The measure this
    samples from is the order-indices induced measure, which is an additive
    mixture of tensorial measures. Each tensorial measure is defined a row
    of indices. 
    
    The function univ_inv is a function handle that inverts a univariate 
    order-n induced distribution. It must support the syntax
    
       univ_inv( u, n )
    
    where u can be any size array of elements between 0 and 1, and n is a
    non-negative integer.
    
    The second calling syntax sets M = size(indices, 1), so that indices 
    already repesents randomly-generated multi-indices.

    seed : integer
       Useful to ensure different processors generate different samples 
       when generating samples in parallel
    """
    if num_samples is None:
        d,M = indices.shape
    else:
        M = num_samples
        d,K = indices.shape

    assert(M > 0)

    if seed is not None:
        np.random.seed(seed)

    if num_samples is not None:
        if weights is None:
            # This selects M random samples on 0, 1, 2, ..., K-1
            ks = np.asarray(
                np.ceil(K*np.random.uniform(0.,1.,(M))),dtype=int)-1
            ks[ks > K]= K
        else:
            ks = discrete_sampling(M,weights,states=np.arange(K))
            #n,bins,__ = plt.hist(ks,np.arange(K+1))
            #plt.show()
        indices = indices[:,ks]

    x = univ_inv(np.random.uniform(0.,1.,(d,M)), indices)
    return x

def histcounts(n,edges):
    indices = np.digitize(n, bins=edges)
    nn = np.zeros(edges.shape[0])
    for ii in indices:
        nn[indices-1]+=1
    return nn, indices

def idist_inverse(u, n, primitive, a, b, supp):
    """
    [x] = idist_inverse(u, n, primitive, a, b, supp)

    Uses bisection to compute the (approximate) inverse of the order-n induced
    primitive function F_n. 

    The ouptut x = F_n^{-1}(u). The input function primitive should be a 
    function handle accepting a single input and outputs the primitive F_n 
    evaluated at the input. 

    The last inputs a, b, and supp are three-term recurrence coefficients 
    (a and b) for the original measure, and its support (supp). 
    These are required to formulate a good initial guess via the 
    Markov-Stiltjies inequalities.
    """
    if np.isscalar(n):
        n = np.asarray([n])

    if n.shape[0] == 1:
        intervals = markov_stiltjies_initial_guess(u, n, a, b, supp)
    else:
        intervals = np.zeros((n.shape[0],2))
        nmax = np.max(n)
        edges = np.arange(-0.5,nmax+0.5)
        nn,bins = histcounts(n, edges)
        for qq in range(0,n.max()+1):
            flags = bins==(qq+1);
            intervals[flags,:] = markov_stiltjies_initial_guess(
                u[flags], qq, a, b, supp)

    x = np.zeros(u.shape)

    for q in range(u.shape[0]):

        #fun = lambda xx: (primitive(xx) - u[q]).squeeze();
        #x[q] = fsolve(
        #    fun, intervals[q,:].sum()/2., xtol=10*np.finfo(float).eps)
        #print (fun(np.asarray([0])))
        fun = lambda xx: (primitive(np.asarray([xx])) - u[q]).squeeze();
        x[q] = brenth(
            fun, intervals[q,0], intervals[q,1], xtol=10*np.finfo(float).eps)
    return x

def markov_stiltjies_initial_guess(u, n, a, b, supp):
    """
    intervals = markov_stiltjies_initial_guess(u, n, a, b, supp)
    
    Uses the Markov-Stiltjies inequalities to provide a bounding interval for x
    the solution to 
    
      F_n(x) = u,
    
    where n is the the order-n induced distribution function associated to the
    measure with three-term recurrrence coefficients a, b, having support on 
    the real-line interval defined by the length-2 vector supp.
    
    If u is a length-M vector, the output intervals is an (M x 2) matrix, with
    row m the bounding interval for u = u(m).
    """
    n = np.asscalar(n)
    assert(a.shape[0] == b.shape[0])
    assert(a.shape[0] > 2*n)

    # Compute quadratic modifications modifications.
    if n>0:
        [x,w] = gauss_quadrature(np.hstack((a, np.sqrt(b))), n)
    b[0] = 1;
    for k in range(n):
        [a,b] = quadratic_modification_C(a, b, x[k])
        b[0] = 1;

    
    ## Markov-Stiltjies inequalities
    # Use all the remaining coefficients for the Markov-Stiltjies inequalities
    N = a.shape[0];
    [y,w] = gauss_quadrature(np.hstack((a, np.sqrt(b))), N);
    if supp[1] > y[-1]:
        X = np.hstack((supp[0], y, supp[1]))
        W = np.hstack((0,np.cumsum(w)))
    else:
        X = np.hstack((supp[0], y, y[-1]))
        W = np.hstack((0,np.cumsum(w)))

    W = W/W[-1]

    W[W > 1] = 1# Just in case for machine eps issues
    W[-1] = 1

    #[~,j] = histc(u, W)
    j = np.digitize(u,W)
    #j = j(:)
    jleft = j
    jright = jleft + 2

    # Fix endpoints
    flags = (jleft == (N+1))
    jleft[flags] = N+2
    jright[flags] = N+2

    intervals = np.hstack(
        (X[jleft-1][:,np.newaxis], X[jright-1][:,np.newaxis]))
    return intervals
    
def idistinv_jacobi(u, n, alph, bet):
    """
    [x] = idistinv_jacobi(u, n, alph, bet)

    Computes the inverse of the order-n induced primitive for the Jacobi
    distribution with parameters alph and bet. Uses a bisection method in
    conjunction with forward evaluation given by idist_jacobi.
    """
    if np.isscalar(n):
        n = np.array([n])

    assert((np.all(u >= 0)) and (np.all(u <=1)))
    assert((alph > -1) and (bet > -1) )
    assert(np.all(n >= 0))

    x = np.zeros(u.shape);

    supp = [-1.,1.]

    if n.shape[0] == 1:
        primitive = partial(idist_jacobi, n=n, alph=alph, bet=bet)

        # Need 2*n + K coefficients, where K is the size of the
        # Markov-Stiltjies binning procedure
        recursion_coeffs = jacobi_recurrence(2*n + 400, alph, bet)
        # All functions that accept b assume they are receiving b
        # but recusion_coeffs:,1=np.sqrt(b)
        a = recursion_coeffs[:,0:1]; b = recursion_coeffs[:,1:]**2
        
        x = idist_inverse(u, n, primitive, a, b, supp)
      
    else:

        nmax = n.max()
        #[nn, ~, bin] = histcounts(n, -0.5:(nmax+0.5))
        edges = np.arange(-0.5,nmax+0.5)
        nn,bins = histcounts(n,edges)
                              

        recursion_coeffs = jacobi_recurrence(2*nmax + 400, alph, bet)
        # All functions that accept b assume they are receiving b
        # but recusion_coeffs:,1=np.sqrt(b)
        a = recursion_coeffs[:,0:1]; b = recursion_coeffs[:,1:]**2

        # need to use u_flat code to be be consistent with akils code
        # inside idist_inverse. but if that code is correct final result
        # will be the same
        #u_flat = u.flatten(order='F')
        for qq in range(0,nmax+1):
            flags = bins==(qq+1)
            #flat_flags = flags.flatten(order='F')
            primitive = partial(idist_jacobi,n=qq,alph=alph,bet=bet)
            #xtemp = idist_inverse(
            #    u_flat[flat_flags], qq, primitive, a, b, supp)
            xtemp = idist_inverse(
                u[flags], qq, primitive, a, b, supp)
            x[flags] = xtemp
    return x

from pyapprox.orthonormal_polynomials_1d import \
     evaluate_orthonormal_polynomial_1d
def demo_idist_jacobi():
    alph = -0.8
    bet = np.sqrt(101)
    #n = 8;    M = 1001
    n = 4;    M = 5

    x = np.cos(np.linspace(0, np.pi, M+2))[:,np.newaxis]
    x = x[1:-1]

    F = idist_jacobi(x, n, alph, bet)

    recursion_coeffs = jacobi_recurrence(n+1, alph, bet, True);
    #recursion_coeffs[:,1]=np.sqrt(recursion_coeffs[:,1])
    polys = evaluate_orthonormal_polynomial_1d(x[:,0],n,recursion_coeffs)
    wt_function = (1-x)**alph*(1+x)**bet/(2**(alph+bet+1)*betafn(bet+1,alph+1))
    f = polys[:,-1:]**2*wt_function
    
    fig,ax = plt.subplots(1,1)
    plt.plot(x,f)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p_n^2(x) \mathrm{d}\mu(x)$')
    ax.set_xlim(-1,1)
    ax.set_ylim(0,4)

    fig,ax=plt.subplots(1,1)
    plt.plot(x,F)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$F_n(x)$')
    ax.set_xlim(-1,1)
    plt.show()

from pyapprox.indexing import sort_indices_lexiographically

def idist_mixture_sampling_pool_helper(indices,univ_inv,weights,args):
    return idist_mixture_sampling(
        indices,univ_inv,num_samples=args[0],weights=weights,seed=args[1])

def idist_mixture_sampling_parallel(indices,univ_inv,num_samples=None,
                                    weights=None,max_eval_concurrency=1,
                                    seed=None,assert_omp=True):
    """
    seed : integer
       seed random number generator. Mutiple calls to idist_mixture_sampling
       must be made with different seeds. The ith seed passed to the
       ith procesor is seed+i

    assert_omp : boolean
        True - the script calling this function must be run with 
           OMP_NUM_THREADS=1 python induced_sampling.py, or 
           OMP_NUM_THREADS must be known to shell, e.g export OMP_NUM_THREADS=1 
           in .bashrc or .bashrc_profile
        False - No check for OMP_NUM_THREADS is made. If OMP_NUM_THREADS != 1
           then speed of this function will degrade signficantly because
           numpy uses threads under the hood and these threads will be 
           overloaded when muliprocessing.pool is called
    """
    from multiprocessing import Pool

    if assert_omp:
        assert int(os.environ['OMP_NUM_THREADS'])==1

    if num_samples is None:
        num_samples = indices.shape[1]

    if seed is None:
        seed = np.random.randint(int(1e6))

    batch_sizes = np.array_split(np.arange(
        num_samples),max_eval_concurrency)
    args = []
    for ii in range(len(batch_sizes)):
        if batch_sizes[ii].shape[0]>0:
            args.append((batch_sizes[ii].shape[0],seed+ii))

    partial_func = partial(
        idist_mixture_sampling_pool_helper,indices,univ_inv,weights)
    pool = Pool(max_eval_concurrency)
    result = pool.map(partial_func,args)

    cnt = 0
    num_vars = indices.shape[0]
    samples = np.empty((num_vars,num_samples))
    for ii in range(len(result)):
        samples[:,cnt:cnt+result[ii].shape[1]]=result[ii]
        cnt += result[ii].shape[1]
    return samples
    
def demo_multivariate_sampling_jacobi():
    from work.adaptive_surrogates_for_inference.apc_study import \
     plot_random_equlibrium_density_corrrelated_beta,\
     plot_random_equlibrium_density_iid_beta
    from pyapprox.arbitrary_polynomial_chaos import \
     compute_coefficients_of_unrotated_basis
    from pyapprox.utilities import get_tensor_product_quadrature_rule
    from pyapprox.polynomial_sampling import christoffel_function
    from pyapprox.visualization import get_meshgrid_function_data

    num_vars = 2
    num_samples = 10000
    alpha_stat=5; beta_stat=2
    degree=1
    
    alph=beta_stat-1; bet=alpha_stat-1

    #case = 'iid'
    case = 'corr'
    if case != 'iid':
        pce, random_variable_density = \
            plot_random_equlibrium_density_corrrelated_beta()
    else:
        pce, random_variable_density = plot_random_equlibrium_density_iid_beta()

    indices = compute_hyperbolic_indices(num_vars,degree,1.0)
    pce.set_indices(indices)

    christoffel_func = partial(
        christoffel_function,basis_matrix_generator=pce.basis_matrix)
    induced_density = lambda x : \
      christoffel_func(x)*random_variable_density(x)/(pce.indices.shape[1])

    num_contour_levels=30
    num_samples_1d = 101
    eps = 1e-2
    plot_limits = [eps,1-eps,eps,1-eps]

    def plot(fun,ax):
        X,Y,Z = get_meshgrid_function_data(
            fun, plot_limits, num_samples_1d)
        dx = (plot_limits[1]-plot_limits[0])/num_samples_1d
        dy = (plot_limits[3]-plot_limits[2])/num_samples_1d
        print(('integral of func', Z.sum()*(dx*dy)))# should converge to 1 as num_samples_1d->\infty

        z_max = np.percentile(Z.flatten(),99.9)
        if Z.max()/z_max<10:
            z_max = Z.max()
        levels = np.linspace(0,z_max,num_contour_levels)
        cset = ax.contourf(
            X, Y, Z, levels=levels,cmap=mpl.cm.coolwarm)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')

    fig,axs=plt.subplots(1,5,figsize=(5*8,6),sharey=True)
    plot(induced_density,axs[0])
    plot(random_variable_density,axs[1])
    plot(christoffel_func,axs[2])
    #plt.show()
    if case!='iid':
        coefficients = np.ones((indices.shape[1],1))
        unrotated_basis_coefficients = compute_coefficients_of_unrotated_basis(
            coefficients,pce.R_inv)
        weights = np.absolute(unrotated_basis_coefficients)
    else:
        weights = np.ones((indices.shape[1],1))
        
    univ_inv = partial(idistinv_jacobi,alph=alph, bet=bet)
    #unrotated_basis_coefficients = unrotated_basis_coefficients*0+1
    import time
    t0 = time.time()
    samples = idist_mixture_sampling_parallel(
        pce.indices,univ_inv,num_samples=num_samples,
        weights=weights,max_eval_concurrency=20)
    # samples = idist_mixture_sampling(
    #     pce.indices,univ_inv,num_samples=num_samples,
    #     weights=unrotated_basis_coefficients)
    print(('generating samples took', (time.time()-t0)))
    # samples are in canonical domain
    samples = pce.var_trans.map_from_canonical_space(samples)
        
    
    from scipy.stats import gaussian_kde as GKDE
    kde = GKDE(samples,'silverman')
    plot(kde,axs[3])
    # max is just for plotting purposes to prevent division by zero
    approx_christoffel = lambda x: kde(x).squeeze()/np.maximum(
        1e-2,random_variable_density(x)).squeeze()
    plot(approx_christoffel,axs[4])
    plt.show()

def discrete_inverse_transform_sampling_1d(probability_mesh,probability_masses,
                                           num_samples):
    """
    probability_mesh : np.ndarray (num_discrete_masses)
        The locations of non-zero probability mass. 
        Must be ascending order.

    probability_masses : np.ndarray (num_discrete_masses)
        The non-zero probability masses at the locations in probability_mesh
    """
    assert probability_mesh.shape[0] == probability_masses.shape[0]
    u_samples = np.random.uniform(0.,1.,(num_samples))
    sample_indices = np.searchsorted(probability_masses,u_samples)
    #print(sample_indices,probability_mesh.shape)
    #print(probability_masses,u_samples)
    samples = probability_mesh[sample_indices]
    return samples


def basis_matrix_generator_1d(pce,nmax,dd,samples):
    vals = evaluate_orthonormal_polynomial_1d(
        samples,nmax,pce.recursion_coeffs[pce.basis_type_index_map[dd]])
    return vals

def discrete_induced_sampling(basis_matrix_generator_1d,basis_indices,
                              probability_mesh_list,
                              probability_masses_list,num_samples):

    num_vars = len(probability_masses_list)
    assert len(probability_mesh_list)==num_vars
    
    basis_cdfs = []
    max_degree_1d = basis_indices.max(axis=1)
    num_basis_indices = basis_indices.shape[1]
    for dd in range(num_vars):    
        basis_probability_masses=basis_matrix_generator_1d(
            dd,probability_mesh_list[dd])**2
        num_indices_1d = basis_probability_masses.shape[1]
        assert num_indices_1d>=max_degree_1d[dd]
        #print (basis_probability_masses,probability_masses_list[dd],dd,max_degree_1d)
        basis_probability_masses=(
            basis_probability_masses.T*probability_masses_list[dd]).T
        #print ('basis l2 norm', basis_probability_masses.sum(axis=0))
        basis_cdfs.append(np.cumsum(basis_probability_masses,axis=0))
        basis_cdfs[-1]/=basis_cdfs[-1][-1]

    
    # Selects random samples on 0, 1, 2, ..., num_indices-1
    mixture_indices = np.random.randint(0,num_basis_indices,(num_samples))
    unique_mixture_indices, mixture_indices_counts = np.unique(
        mixture_indices,return_counts=True)

    idx1 = 0
    samples = np.empty((num_vars,num_samples))
    for ii in range(unique_mixture_indices.shape[0]):
        idx2 = idx1 + mixture_indices_counts[ii]
        for dd in range(num_vars):
            samples[dd,idx1:idx2] = discrete_inverse_transform_sampling_1d(
                probability_mesh_list[dd],
                basis_cdfs[dd][:,basis_indices[dd,unique_mixture_indices[ii]]],
                mixture_indices_counts[ii])
        idx1 = idx2
    # shuffle so that not all samples from one mixture are next to eachother
    return samples[:,np.random.permutation(np.arange(num_samples))]

def basis_matrix_generator_1d_active_vars_wrapper(
        basis_matrix_generator,active_vars,dd,samples):
    return basis_matrix_generator(active_vars[dd],samples)
    

def mixed_continuous_discrete_induced_sampling(
        basis_matrix_generator_1d,basis_indices,probability_mesh_list,
        probability_masses_list,num_vars,discrete_var_indices,num_samples):
    
    num_discrete_vars = len(discrete_var_indices)
    assert num_discrete_vars<=num_vars
    assert len(probability_masses_list)==num_discrete_vars
    assert len(probability_mesh_list)==num_discrete_vars

    mask = np.ones((num_vars),dtype=bool)
    mask[discrete_vars]=False
    continuous_vars = np.arange(num_vars)[mask]
    assert discrete_vars.shape[0]+continous_vars.shape[0]==num_vars
    
    discrete_samples = discrete_induced_sampling(
        partial(basis_matrix_generator_1d_vars_wrapper,
                basis_matrix_generator_1d,discrete_vars),
        basis_indices,
        probability_mesh_list,
        probability_masses_list,num_samples)

    #TODO remove hard coding which only supports jacobi polynomials
    # not even varying jacobi polynomials are supported
    univ_inv = partial(idistinv_jacobi,alph=alph, bet=bet)
    continuous_samples = idist_mixture_sampling_parallel(
        basis_indices,univ_inv,num_samples=num_samples,
        weights=None,max_eval_concurrency=10)

    samples = np.empty((num_vars,num_samples))
    samples[discrete_vars,:] = discrete_samples
    samples[continuous_vars,:] = continuous_samples
    return samples

def float_rv_discrete_inverse_transform_sampling_1d(xk,pk,ab,ii,u_samples):
    poly_vals = evaluate_orthonormal_polynomial_1d(
        np.asarray(xk,dtype=float),ii,ab)[:,-1]
    probability_masses = pk*poly_vals**2
    cdf_vals = np.cumsum(probability_masses)
    assert np.allclose(cdf_vals[-1],1),cdf_vals[-1]
    #cdf_vals/=cdf_vals[-1]
    sample_indices = np.searchsorted(cdf_vals,u_samples)
    samples = xk[sample_indices]
    return samples

from scipy import integrate
def continuous_induced_measure_cdf(pdf,ab,ii,lb,ub,tol,x):
    x = np.atleast_1d(x)
    assert x.ndim==1
    assert x.min()>=lb and x.max()<=ub
    integrand = lambda xx: evaluate_orthonormal_polynomial_1d(
        np.atleast_1d(xx),ii,ab)[:,-1]**2*pdf(xx)
    #from pyapprox.cython.orthonormal_polynomials_1d import\
    #    induced_measure_pyx
    #integrand = lambda x: induced_measure_pyx(x,ii,ab,pdf)
    vals = np.empty_like(x,dtype=float)
    for jj in range(x.shape[0]):
        integral,err = integrate.quad(
            integrand,lb,x[jj],epsrel=tol,epsabs=tol,limit=100)
        vals[jj]=integral
        # avoid numerical issues at boundary of domain
        if vals[jj]>1 and vals[jj]-1<tol:
            vals[jj]=1.
    return vals

from pyapprox.random_variable_algebra import invert_monotone_function
from pyapprox.variables import get_distribution_info
def continuous_induced_measure_ppf(var,ab,ii,u_samples,
                                   quad_tol=1e-8,opt_tol=1e-6):
    if is_bounded_continuous_variable(var):
        # canonical domain of bounded variables does not coincide with scipy
        # variable canonical pdf domain with loc,scale=0,1
        lb,ub = -1,1
        loc,scale = -1,2
    else:
        lb,ub = var.support()
        loc,scale=0,1
    shapes=get_distribution_info(var)[2]

    # need to map x from canonical polynomial domain to canonical domain of pdf
    pdf = lambda x: var.dist._pdf((x-loc)/scale,**shapes)/scale
    def pdf(x):
        vals = var.dist._pdf((x-loc)/scale,**shapes)/scale
        #print('x',x,(x-loc)/scale,vals)
        return  vals
    #pdf = var.pdf
    #func = partial(continuous_induced_measure_cdf,pdf,ab,ii,lb,ub,quad_tol)
    from pyapprox.cython.orthonormal_polynomials_1d import\
        continuous_induced_measure_cdf_pyx
    func = partial(continuous_induced_measure_cdf_pyx,pdf,ab,ii,lb,quad_tol)
    method='bisect'
    samples = invert_monotone_function(func,[lb,ub],u_samples,method,opt_tol)
    assert np.all(np.isfinite(samples))
    return samples

from scipy.stats import _continuous_distns, _discrete_distns
def inverse_transform_sampling_1d(var,ab,ii,u_samples):
    name = var.dist.name
    if name=='float_rv_discrete':
        return float_rv_discrete_inverse_transform_sampling_1d(
            var.dist.xk,var.dist.pk,ab,ii,u_samples)
    elif name in _continuous_distns._distn_names:
        return continuous_induced_measure_ppf(var,ab,ii,u_samples)
    else:
        msg = 'induced sampling not yet implemented for var type %s'%name
        raise Exception(msg)
    return samples

def generate_induced_samples(pce,num_samples):
    num_vars,num_basis_indices = pce.indices.shape
    
    # Selects random samples on 0, 1, 2, ..., num_indices-1
    mixture_indices = np.random.randint(0,num_basis_indices,(num_samples))
    unique_mixture_indices, mixture_indices_counts = np.unique(
        mixture_indices,return_counts=True)
    
    idx1 = 0
    samples = np.empty((num_vars,num_samples))
    for ii in range(unique_mixture_indices.shape[0]):
        idx2 = idx1 + mixture_indices_counts[ii]
        for jj in range(pce.var_trans.variable.nunique_vars):
            var = pce.var_trans.variable.unique_variables[jj]
            for dd in pce.var_trans.variable.unique_variable_indices[jj]:
                kk = pce.indices[dd,unique_mixture_indices[ii]]
                u_samples = np.random.uniform(
                    0.,1.,(mixture_indices_counts[ii]))
                samples[dd,idx1:idx2] = inverse_transform_sampling_1d(
                    var,pce.recursion_coeffs[pce.basis_type_index_map[dd]],
                    kk,u_samples)
        idx1 = idx2
    # shuffle so that not all samples from one mixture are next to eachother
    return samples[:,np.random.permutation(np.arange(num_samples))]

def generate_induced_samples_migliorati(pce,num_samples_per_index):
    num_vars,num_indices = pce.indices.shape

    idx1 = 0
    samples = np.empty((num_vars,num_indices*num_samples_per_index))
    for ii in range(num_indices):
        idx2 = idx1 + num_samples_per_index
        for jj in range(pce.var_trans.variable.nunique_vars):
            var = pce.var_trans.variable.unique_variables[jj]
            for dd in pce.var_trans.variable.unique_variable_indices[jj]:
                kk = pce.indices[dd,ii]
                u_samples = np.random.uniform(0.,1.,(num_samples_per_index))
                if kk>0:
                    samples[dd,idx1:idx2] = inverse_transform_sampling_1d(
                        var,pce.recursion_coeffs[pce.basis_type_index_map[dd]],
                        kk,u_samples)
                else:
                    samples[dd,idx1:idx2]=var.rvs(size=(1,num_samples_per_index))
                    
        idx1=idx2
    # shuffle so that not all samples from one mixture are next to eachother
    return samples[:,np.random.permutation(np.arange(samples.shape[1]))]

def compute_preconditioned_canonical_basis_matrix_condition_number(pce,samples):
    return compute_preconditioned_basis_matrix_condition_number(
        pce.canonical_basis_matrix,samples)

def compute_preconditioned_basis_matrix_condition_number(
        basis_matrix_func,samples):
    basis_matrix = basis_matrix_func(samples)
    basis_matrix = basis_matrix*np.sqrt(
        christoffel_weights(basis_matrix))[:,np.newaxis]
    cond = np.linalg.cond(basis_matrix)
    return cond

from pyapprox.polynomial_sampling import christoffel_weights
def generate_induced_samples_migliorati_tolerance(pce,cond_tol,samples=None,
                                                  verbosity=0):
    """
    Parameters
    ----------
    samples : np.ndarray
        samples in the canonical domain of the polynomial
    """
    if samples is None:
        new_samples = generate_induced_samples_migliorati(pce,1)
    else:
        new_samples = samples.copy()
    cond = compute_preconditioned_canonical_basis_matrix_condition_number(
        pce,new_samples)
    
    if verbosity>0:
        print('\tCond No.', cond, 'No. samples', new_samples.shape[1])
    while cond>cond_tol:
        new_samples=np.hstack(
            (new_samples,generate_induced_samples_migliorati(pce,1)))
        cond = compute_preconditioned_canonical_basis_matrix_condition_number(
            pce,new_samples)
        if verbosity>0:
            print('\tCond No.', cond, 'No. samples', new_samples.shape[1])
    return new_samples

def increment_induced_samples_migliorati(pce,cond_tol,samples,indices,
                                         new_indices,verbosity=0):
    """
    Parameters
    ----------
    samples : np.ndarray
        samples in the canonical domain of the polynomial
    """
    pce_indices = pce.indices.copy()
    num_samples = samples.shape[1]
    num_samples_per_new_index = num_samples//indices.shape[1]
    pce.set_indices(new_indices)
    new_samples = np.hstack((
        samples,generate_induced_samples_migliorati(
            pce,num_samples_per_new_index)))
    pce.set_indices(np.hstack((indices,new_indices)))
    new_samples = generate_induced_samples_migliorati_tolerance(
        pce,cond_tol,new_samples)
    pce.set_indices(pce_indices)

    if verbosity>0:
        print('N',num_samples,indices.shape[1],num_samples//indices.shape[1])
        print('No. initial samples', samples.shape[1])
        print('No. samples from new mixture components',
              num_samples_per_new_index*new_indices.shape[1])
        print('No. additional new samples from all mixture components',
              new_samples.shape[1]-num_samples_per_new_index*new_indices.shape[1]-
              samples.shape[1])
    
    return new_samples
    
    
if __name__=='__main__':
    from pyapprox.indexing import compute_hyperbolic_indices
    from pyapprox.configure_plots import *
    demo_idist_jacobi()
    demo_multivariate_sampling_jacobi()
