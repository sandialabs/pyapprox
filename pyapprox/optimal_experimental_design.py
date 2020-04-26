import numpy as np
from scipy.linalg import solve as scipy_solve

def compute_heteroscedastic_outer_products(factors,noise_multiplier):
    """
    Compute 

    .. math:: \eta(x_i)^2\sigma**2f(x_i)f(x_i)^T\quad \forall i=0,\ldots,M

    at a set of design pts :math:`x_i`.
    
    for the linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon

    WARNING: CHECK this documentation. It may be only relevant to least squares
    estimation

    Parameters
    ---------
    factors : np.ndarray (M,N)
        The N factors F of the linear model evaluated at the M design pts

    noise_multiplier : np.ndarray (M)
        The design dependent noise :math:`\eta(x)\sigma**2`

    Returns
    -------
    heteroscedastic_outer_products : np.ndarray (N,N,M)
       The outer products of each row of F with itself weighted by the 
       noise_multiplier**2 of that row,  i.e. 
       :math:`\eta(x_i)^2\sigma**2f(x_i)f(x_i)^T`
    """
    num_design_pts,num_factors = factors.shape
    temp = (factors.T*noise_multiplier).T
    heteroscedastic_outer_products = np.empty(
        (num_factors,num_factors,num_design_pts),dtype=float)
    for ii in range(num_design_pts):
        heteroscedastic_outer_products[:,:,ii] = np.outer(temp[ii,:],temp[ii,:])
    return heteroscedastic_outer_products

def compute_homoscedastic_outer_products(factors):
    """
    Compute 

    .. math:: f(x_i)f(x_i)^T\quad \forall i=0,\ldots,M

    at a set of design pts :math:`x_i`.
    
    for the linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon

    Parameters
    ---------
    factors : np.ndarray (M,N)
        The N factors F of the linear model evaluated at the M design pts

    Returns
    -------
    homoscedastic_outer_products : np.ndarray (N,N,M)
       The outer products of each row of F with itself, i.e. 
       :math:`f(x_i)f(x_i)^T`
    """
    num_design_pts,num_factors = factors.shape
    homoscedastic_outer_products = np.empty(
        (num_factors,num_factors,num_design_pts),dtype=float)
    for ii in range(num_design_pts):
        homoscedastic_outer_products[:,:,ii]=np.outer(
            factors[ii,:],factors[ii,:])
    return homoscedastic_outer_products

def ioptimality_criterion(homog_outer_prods,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          hetero_outer_prods=None,
                          noise_multiplier=None):
    r"""
    Evaluate the I-optimality criterion for a given design probability measure.

    The criteria is
    \int_\Xi g(\xi) C(\mu) g(\xi) d\nu(\xi)

    where
    C(\mu) = M_1^{-1} M_0 M^{-1}

    For Homoscedastic noise M_0=M_1 and
    C(\mu) = M_1^{-1}

    Parameters
    ----------
    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure \mu on the design points

    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The hessian M_1 of the error for each design point

    hetero_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                          num_design_pts)
       The asymptotic covariance M_0 of the subgradient of the model for 
       each design point. If None homoscedastic noise is assumed.

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    pred_factors : np.ndarray (num_pred_pts,num_pred_factors)
       The prediction factors g evaluated at each of the prediction points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """

    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors   = pred_factors.shape
    # [:,:,0] just changes shape from (N,N,1) to (N,N)
    M1 = homog_outer_prods.dot(design_prob_measure)[:,:,0]
    if hetero_outer_prods is not None:
        # [:,:,0] just changes shape from (N,N,1) to (N,N)
        M0 = hetero_outer_prods.dot(design_prob_measure)[:,:,0]
        Q,R = np.linalg.qr(M1)
        u = scipy_solve(R,Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        value = np.sum(u*M0u) / num_pred_factors;
        if (return_grad):
            assert noise_multiplier is not None
            gamma   = -scipy_solve(R,(Q.T.dot(M0u)))
            Fu  = design_factors.dot(u)
            t   = noise_multiplier[:,np.newaxis] * Fu
            Fgamma  = design_factors.dot(gamma)
            gradient = 2*np.sum(Fu*Fgamma,axis=1) + np.sum(t**2,axis=1)
            gradient /= num_pred_factors
            return value, gradient
        else:
            return value
    else:
        u = np.linalg.solve(M1,pred_factors.T)
        # Value
        # We want to sum the variances, i.e. the enties of the diagonal of
        # pred_factors.dot(M1.dot(pred_factors.T))
        # We know that diag(A.T.dot(B)) = (A*B).axis=0)
        # The following sums over all entries of A*B we get the mean of the
        # variance
        value    = np.sum(pred_factors*u.T) / num_pred_factors;
        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_P = design_factors.dot(u);
        gradient   = -np.sum(F_M1_inv_P**2,axis=1) / num_pred_factors;
        return value, gradient


def coptimality_criterion(homog_outer_prods,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          hetero_outer_prods=None,
                          noise_multiplier=None):
    r"""
    Evaluate the C-optimality criterion for a given design probability measure.

    The criteria is
    c^T C(\mu) c 
    for some vector c. Here we assume c=(1,1,...,1)

    where
    C(\mu) = M_1^{-1} M_0 M^{-1}

    For Homoscedastic noise M_0=M_1 and
    C(\mu) = M_1^{-1}

    Parameters
    ----------
    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure \mu on the design points

    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The hessian M_1 of the error for each design point

    hetero_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                          num_design_pts)
       The asymptotic covariance M_0 of the subgradient of the model for 
       each design point. If None homoscedastic noise is assumed.

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    pred_factors : np.ndarray (num_pred_pts,num_pred_factors)
       The prediction factors g evaluated at each of the prediction points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
        
    


        
def doptimality_criterion(homog_outer_prods,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          hetero_outer_prods=None,
                          noise_multiplier=None):
    r"""
    Evaluate the D-optimality criterion for a given design probability measure.

    The criteria is
    determinant [ C(\mu) ]

    where
    C(\mu) = M_1^{-1} M_0 M^{-1}

    For Homoscedastic noise M_0=M_1 and
    C(\mu) = M_1^{-1}

    Parameters
    ----------
    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure \mu on the design points

    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The hessian M_1 of the error for each design point

    hetero_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                          num_design_pts)
       The asymptotic covariance M_0 of the subgradient of the model for 
       each design point. If None homoscedastic noise is assumed.

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    pred_factors : np.ndarray (num_pred_pts,num_pred_factors)
       The prediction factors g evaluated at each of the prediction points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if re
    """

    num_design_pts, num_design_factors = design_factors.shape
    c  = np.ones((num_design_factors),dtype=float)
    # [:,:,0] just changes shape from (N,N,1) to (N,N)
    M1 = homog_outer_prods.dot(design_prob_measure)[:,:,0]
    M1_inv = np.linalg.inv(M1)
    if hetero_outer_prods is not None:
        #print(homog_outer_prods.shape)
        print(hetero_outer_prods.shape)
        # [:,:,0] just changes shape from (N,N,1) to (N,N)
        M0 =  hetero_outer_prods.dot(design_prob_measure)[:,:,0]
        u = M1_inv
        gamma = M0.dot(u)
        value = np.linalg.det(gamma)
        if (return_grad):
            gamma   = -scipy_solve(R,(Q.T.dot(M0u)))
            assert noise_multiplier is not None
            Fu    = design_factors.dot(u)
            t   = noise_multiplier * Fu
            Fgamma  = design_factors.dot(gamma)
            gradient = 2*Fu*Fgamma + t**2;
            return value, gradient
        else:
            return value
    else:
        value  = np.linalg.det(M1_inv)
        assert False # gradient not yet derived
        # Gradient
        if (return_grad):
            FM1_inv = design_factors.dot(M1_inv)
            gradient = 2*value M1 
            return value, gradient
        else:
            return value
