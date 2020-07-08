import numpy as np
from scipy.linalg import solve_triangular
import copy

def compute_homoscedastic_outer_products(factors):
    r"""
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

def get_M0_and_M1_matrices(
        homog_outer_prods,design_prob_measure,noise_multiplier,regression_type):
    """
    Compute the matrices :math:`M_0` and :math:`M_1` used to compute the
    asymptotic covariance matrix :math:`C(\mu) = M_1^{-1} M_0 M^{-1}` of the
    linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon.

    For least squares

    .. math:: M_0 = \sum_{i=1}^M\eta(x_i)^2f(x_i)f(x_i)^T\r_i

    .. math:: M_1 = \sum_{i=1}^Mf(x_i)f(x_i)^T\r_i
    
    and for quantile regression

    .. math:: M_0 = \sum_{i=1}^M\frac{1}{\eta(x_i)}f(x_i)f(x_i)^Tr_i

    .. math:: M_1 = \sum_{i=1}^Mf(x_i)f(x_i)^T\r_i

    Parameters
    ----------
    homog_outer_prods : np.ndarray(num_design_pts,num_design_pts,num_design_pts)
        The outer products :math:`f(x_i)f(x_i)^T` for each design point 
        :math:`x_i`

    design_prob_measure : np.ndarray (ndesign_pts)
        The weights :math:`r_i` for each design point

    noise_multiplier : np.ndarray (num_design_pts)
        The design dependent noise function :math:`\eta(x)`

    regression_type : string
        The method used to compute the coefficients of the linear model. 
        Currently supported options are ``lstsq`` and ``quantile``.

    Returns
    -------
    M0 : np.ndarray (num_design_pts,num_design_pts)
        The matrix :math:`M_0`

    M1 : np.ndarray (num_design_pts,num_design_pts)
        The matrix :math:`M_1`
    
    """
    if noise_multiplier is None:
        return None, homog_outer_prods.dot(design_prob_measure)
    
    if regression_type=='lstsq':
        M0 = homog_outer_prods.dot(design_prob_measure*noise_multiplier**2)
        M1 = homog_outer_prods.dot(design_prob_measure)
    elif regression_type=='quantile':
        M0 = homog_outer_prods.dot(design_prob_measure)
        M1 = homog_outer_prods.dot(design_prob_measure/noise_multiplier)
    else:
        msg = f'regression type {regression_type} not supported'
        raise Exception(msg)
    return M0,M1

def ioptimality_criterion(homog_outer_prods,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          noise_multiplier=None,
                          regression_type='lstsq'):
    r"""
    Evaluate the I-optimality criterion for a given design probability measure
    for the linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon.

    The criteria is

    .. math::  \int_\Xi g(\xi) C(\mu) g(\xi) d\nu(\xi)

    where

    .. math:: C(\mu) = M_1^{-1} M_0 M^{-1}

    Parameters
    ----------
    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The outer_products :math:`f(x_i)f(x_i)^T` for each design point 
       :math:`x_i`

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    pred_factors : np.ndarray (num_pred_pts,num_pred_factors)
       The prediction factors :math:`g` evaluated at each of the prediction 
       points

    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure :math:`\mu` on the design points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    noise_multiplier : np.ndarray (num_design_pts)
        The design dependent noise function :math:`\eta(x)`

    regression_type : string
        The method used to compute the coefficients of the linear model. 
        Currently supported options are ``lstsq`` and ``quantile``.

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors   = pred_factors.shape
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M0,M1=get_M0_and_M1_matrices(
        homog_outer_prods,design_prob_measure,noise_multiplier,regression_type)
    if noise_multiplier is not None:
        Q,R = np.linalg.qr(M1)
        u = solve_triangular(R,Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        value = np.sum(u*M0u) / num_pred_pts;
        if (return_grad):
            gamma   = -solve_triangular(R,(Q.T.dot(M0u)))
            Fu  = design_factors.dot(u)
            t   = noise_multiplier[:,np.newaxis] * Fu
            Fgamma  = design_factors.dot(gamma)
            if regression_type=='lstsq':
                gradient = 2*np.sum(Fu*Fgamma,axis=1) + np.sum(t**2,axis=1)
            elif regression_type=='quantile':
                gradient = 2*np.sum(Fu*Fgamma/noise_multiplier[:,np.newaxis],axis=1) + \
                    np.sum(Fu**2,axis=1)
            gradient /= num_pred_pts
            return value, gradient.T
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
        value    = np.sum(pred_factors*u.T) / num_pred_pts;
        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_P = design_factors.dot(u);
        gradient   = -np.sum(F_M1_inv_P**2,axis=1) / num_pred_pts;
        return value, gradient.T


def coptimality_criterion(homog_outer_prods,design_factors,
                          design_prob_measure,return_grad=True,
                          noise_multiplier=None,regression_type='lstsq'):
    r"""
    Evaluate the C-optimality criterion for a given design probability measure 
    for the linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon.

    The criteria is

    .. math:: c^T C(\mu) c 

    where

    .. math:: C(\mu) = M_1^{-1} M_0 M^{-1}

    for some vector :math:`c`. Here we assume without loss of genearlity 
    :math:`c=(1,1,...,1)^T`

    Parameters
    ----------
    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The hessian M_1 of the error for each design point

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure :math:`\mu` on the design points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    noise_multiplier : np.ndarray (num_design_pts)
        The design dependent noise function :math:`\eta(x)`

    regression_type : string
        The method used to compute the coefficients of the linear model. 
        Currently supported options are ``lstsq`` and ``quantile``.

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
    num_design_pts, num_design_factors = design_factors.shape
    c = np.ones((num_design_factors,1))
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M0,M1=get_M0_and_M1_matrices(
        homog_outer_prods,design_prob_measure,noise_multiplier,regression_type)
    if noise_multiplier is not None:
        Q,R = np.linalg.qr(M1)
        u = solve_triangular(R,Q.T.dot(c))
        M0u = M0.dot(u)
        value = u.T.dot(M0u)
        if (return_grad):
            gamma   = -solve_triangular(R,(Q.T.dot(M0u)))
            Fu  = design_factors.dot(u)
            t   = noise_multiplier[:,np.newaxis] * Fu
            Fgamma  = design_factors.dot(gamma)
            if regression_type=='lstsq':
                gradient = 2*Fu*Fgamma + t**2
            elif regression_type=='quantile':
                gradient = 2*Fu*Fgamma/noise_multiplier[:,np.newaxis] + Fu**2
            return value, gradient.T
        else:
            return value
    else:
        u = np.linalg.solve(M1,c)
        value    = c.T.dot(u);
        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_c = design_factors.dot(u);
        gradient   = -F_M1_inv_c**2
        return value, gradient.T
        
def doptimality_criterion(homog_outer_prods,design_factors,
                          design_prob_measure,return_grad=True,
                          noise_multiplier=None,
                          regression_type='lstsq'):
    r"""
    Evaluate the D-optimality criterion for a given design probability measure
    for the linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon.

    The criteria is
    
    .. math:: \log \mathrm{determinant} [ C(\mu) ]

    where

    .. math:: C(\mu) = M_1^{-1} M_0 M^{-1}

    Parameters
    ----------
    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The outer_products :math:`f(x_i)f(x_i)^T` for each design point 
       :math:`x_i`

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure :math:`\mu` on the design points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    noise_multiplier : np.ndarray (num_design_pts)
        The design dependent noise function :math:`\eta(x)`

    regression_type : string
        The method used to compute the coefficients of the linear model. 
        Currently supported options are ``lstsq`` and ``quantile``.

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
    num_design_pts, num_design_factors = design_factors.shape
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    #M1 = homog_outer_prods.dot(design_prob_measure)
    M0,M1=get_M0_and_M1_matrices(
        homog_outer_prods,design_prob_measure,noise_multiplier,regression_type)
    M1_inv = np.linalg.inv(M1)
    if noise_multiplier is not None:
        gamma = M0.dot(M1_inv)
        value = np.log(np.linalg.det(M1_inv.dot(gamma)))
        if (return_grad):
            ident = np.eye(gamma.shape[0])
            M0_inv = np.linalg.inv(M0)
            kappa  = M1.dot(M0_inv)
            gradient = np.zeros(num_design_pts)
            for ii in range(num_design_pts):
                #gradient[ii]=np.trace(kappa.dot(homog_outer_prods[:,:,ii]).dot(
                #    -2*gamma.T+noise_multiplier[ii]**2*ident).dot(M1_inv))
                #TODO order multiplications to be most efficient. Probably
                # need to work on f_i rather than stored outer product
                # f_i.dot(f_i.T)
                if regression_type=='lstsq':
                    gradient[ii] = np.sum(kappa.dot(homog_outer_prods[:,:,ii])*(
                        -2*gamma.T+noise_multiplier[ii]**2*ident).dot(M1_inv))
                elif regression_type=='quantile':
                    gradient[ii] = np.sum(kappa.dot(homog_outer_prods[:,:,ii])*(
                        -2/noise_multiplier[:,np.newaxis][ii]*gamma.T+ident).dot(M1_inv))
            return value, gradient.T
        else:
            return value
    else:
        value  = np.log(np.linalg.det(M1_inv))
        # Gradient
        if (return_grad):
            #gradient = -np.array([np.trace(M1_inv.dot(homog_outer_prods[:,:,ii])) for ii in range(homog_outer_prods.shape[2])])
            #TODO order multiplications to be most efficient. Probably need to
            # work on f_i rather than stored outer product f_i.dot(f_i.T)
            gradient = -np.array([(homog_outer_prods[:,:,ii]*M1_inv).sum() for ii in range(homog_outer_prods.shape[2])])
            return value, gradient.T
        else:
            return value


def aoptimality_criterion(homog_outer_prods,design_factors,
                          design_prob_measure,return_grad=True,
                          noise_multiplier=None,
                          regression_type='lstsq'):
    r"""
    Evaluate the A-optimality criterion for a given design probability measure 
    for the linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon.

    The criteria is

    .. math:: \mathrm{trace}[ C(\mu) ]

    where

    .. math:: C(\mu) = M_1^{-1} M_0 M^{-1}

    Parameters
    ----------
    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                    num_design_pts)
       The hessian M_1 of the error for each design point

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure :math:`\mu` on the design points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    noise_multiplier : np.ndarray (num_design_pts)
        The design dependent noise function :math:`\eta(x)`

    regression_type : string
        The method used to compute the coefficients of the linear model. 
        Currently supported options are ``lstsq`` and ``quantile``.

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """

    num_design_pts, num_design_factors = design_factors.shape
    # [:,:,0] just changes shape from (N,N,1) to (N,N)
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M0,M1=get_M0_and_M1_matrices(
        homog_outer_prods,design_prob_measure,noise_multiplier,regression_type)
    M1_inv = np.linalg.inv(M1)
    if noise_multiplier is not None:
        gamma = M0.dot(M1_inv)
        value = np.trace(M1_inv.dot(gamma))
        if (return_grad):
            ident = np.eye(gamma.shape[0])
            M0_inv = np.linalg.inv(M0)
            kappa  = M1.dot(M0_inv)
            gradient = np.zeros(num_design_pts)
            for ii in range(num_design_pts):
                if regression_type=='lstsq':
                    gradient[ii]=np.trace(
                        M1_inv.dot(homog_outer_prods[:,:,ii]).dot(
                        -2*gamma.T+noise_multiplier[ii]**2*ident).dot(M1_inv))
                elif regression_type=='quantile':
                    gradient[ii]=np.trace(
                        M1_inv.dot(homog_outer_prods[:,:,ii]).dot(
                        -2*gamma.T/noise_multiplier[:,np.newaxis][ii]+ident).dot(M1_inv))
            return value, gradient.T
        else:
            return value
    else:
        value  = np.trace(M1_inv)
        # Gradient
        if (return_grad):
            gradient = -np.array([(M1_inv*homog_outer_prods[:,:,ii].dot(M1_inv)).sum() for ii in range(homog_outer_prods.shape[2])])
            return value, gradient.T
        else:
            return value

def roptimality_criterion(beta,homog_outer_prods,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          noise_multiplier=None,regression_type='lstsq'):
    r"""
    Evaluate the R-optimality criterion for a given design probability measure 
    for the linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon.

    The criteria is

    .. math:: \mathrm{CVaR}\left[\sigma^2f(x)^T\left(F(\mathcal{X})^TF(\mathcal{X})\right)^{-1}f(x)\right]

    where

    .. math:: C(\mu) = M_1^{-1} M_0 M^{-1}

    Parameters
    ----------
    beta : float
       The confidence level of CVAR 

    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The hessian M_1 of the error for each design point

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    pred_factors : np.ndarray (num_pred_pts,num_pred_factors)
       The prediction factors :math:`g` evaluated at each of the prediction 
       points

    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure :math:`\mu` on the design points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    noise_multiplier : np.ndarray (num_design_pts)
        The design dependent noise function :math:`\eta(x)`

    regression_type : string
        The method used to compute the coefficients of the linear model. 
        Currently supported options are ``lstsq`` and ``quantile``.

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
    assert beta>=0 and beta<=1
    from pyapprox.cvar_regression import conditional_value_at_risk, \
        conditional_value_at_risk_gradient
    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors   = pred_factors.shape
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M0,M1=get_M0_and_M1_matrices(
        homog_outer_prods,design_prob_measure,noise_multiplier,regression_type)
    if noise_multiplier is not None:
        Q,R = np.linalg.qr(M1)
        u = solve_triangular(R,Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        variances = np.sum(u*M0u,axis=0)
        value = conditional_value_at_risk(variances,beta)
        if (return_grad):
            gamma   = -solve_triangular(R,(Q.T.dot(M0u)))
            Fu  = design_factors.dot(u)
            t   = noise_multiplier[:,np.newaxis] * Fu
            Fgamma  = design_factors.dot(gamma)
            cvar_grad = conditional_value_at_risk_gradient(variances,beta)
            if regression_type=='lstsq':
                gradient = np.sum((2*Fu*Fgamma+t**2).T*cvar_grad[:,np.newaxis],axis=0)
            elif regression_type=='quantile':
                gradient = np.sum((2*Fu*Fgamma/noise_multiplier[:,np.newaxis]+Fu**2).T*cvar_grad[:,np.newaxis],axis=0)
                
            return value, gradient.T
        else:
            return value
    else:
        u = np.linalg.solve(M1,pred_factors.T)
        # Value
        # We want to sum the variances, i.e. the enties of the diagonal of
        # pred_factors.dot(M1.dot(pred_factors.T))
        # We know that diag(A.T.dot(B)) = (A*B).axis=0)
        # The following sums over all entries of A*B we get the diagonal
        # variances
        variances = np.sum(pred_factors*u.T,axis=1)
        value = conditional_value_at_risk(variances,beta)
        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_P = design_factors.dot(u);
        cvar_grad = conditional_value_at_risk_gradient(variances,beta)
        gradient   = -np.sum(F_M1_inv_P.T**2*cvar_grad[:,np.newaxis],axis=0)
        return value, gradient.T

def goptimality_criterion(homog_outer_prods,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          noise_multiplier=None,regression_type='lstsq'):
    r"""
    valuate the G-optimality criterion for a given design probability measure 
    for the linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon.

    The criteria is

    .. math:: \max{sup}_{xi\in\Xi_\text{pred}} \sigma^2f(x)^T\left(F(\mathcal{X})^TF(\mathcal{X})\right)^{-1}f(x)

    where

    .. math:: C(\mu) = M_1^{-1} M_0 M^{-1}

    Parameters
    ----------
    homog_outer_prods : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The hessian M_1 of the error for each design point

    design_factors : np.ndarray (num_design_pts,num_design_factors)
       The design factors evaluated at each of the design points

    pred_factors : np.ndarray (num_pred_pts,num_pred_factors)
       The prediction factors g evaluated at each of the prediction points

    design_prob_measure : np.ndarray (num_design_pts)
       The prob measure :math:`\mu` on the design points

    return_grad : boolean
       True  - return the value and gradient of the criterion
       False - return only the value of the criterion

    noise_multiplier : np.ndarray (num_design_pts)
        The design dependent noise function :math:`\eta(x)`

    regression_type : string
        The method used to compute the coefficients of the linear model. 
        Currently supported options are ``lstsq`` and ``quantile``.

    Returns
    -------
    value : np.ndarray (num_pred_pts)
        The value of the objective function

    grad : np.ndarray (num_pred_pts,num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors   = pred_factors.shape
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M0,M1=get_M0_and_M1_matrices(
        homog_outer_prods,design_prob_measure,noise_multiplier,regression_type)
    if noise_multiplier is not None:
        Q,R = np.linalg.qr(M1)
        u = solve_triangular(R,Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        variances = np.sum(u*M0u,axis=0)
        value = variances
        if (return_grad):
            gamma   = -solve_triangular(R,(Q.T.dot(M0u)))
            Fu  = design_factors.dot(u)
            t   = noise_multiplier[:,np.newaxis] * Fu
            Fgamma  = design_factors.dot(gamma)
            if regression_type=='lstsq':
                gradient = 2*Fu*Fgamma + t**2
            elif regression_type=='quantile':
                gradient = 2*Fu*Fgamma/noise_multiplier[:,np.newaxis] + Fu**2
            return value, gradient.T
        else:
            return value
    else:
        u = np.linalg.solve(M1,pred_factors.T)
        # Value
        # We want to sum the variances, i.e. the enties of the diagonal of
        # pred_factors.dot(M1.dot(pred_factors.T))
        # We know that diag(A.T.dot(B)) = (A*B).axis=0)00
        # The following sums over all entries of A*B we get the diagonal
        # variances
        variances = np.sum(pred_factors*u.T,axis=1)
        value = variances
        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_P = design_factors.dot(u);
        gradient   = -F_M1_inv_P**2
        return value, gradient.T

from scipy.optimize import Bounds, minimize, LinearConstraint, NonlinearConstraint
from functools import partial

def minimax_oed_constraint_objective(local_oed_obj,x):
    return x[0]-local_oed_obj(x[1:])

def minimax_oed_constraint_jacobian(local_oed_jac,x):
    jac = -local_oed_jac(x[1:])
    jac = np.atleast_2d(jac)
    return np.hstack([np.ones((jac.shape[0],1)),jac]) 

class AlphabetOptimalDesign(object):
    """
    Notes
    -----
        # Even though scipy.optimize.minimize may print the warning
        # UserWarning: delta_grad == 0.0. Check if the approximated function is
        # linear. If the function is linear better results can be obtained by
        # defining the Hessian as zero instead of using quasi-Newton
        # approximations.
        # The Hessian is not zero.
    """
    def __init__(self,criteria,design_factors,noise_multiplier=None,
                 opts=None,regression_type='lstsq'):
        self.criteria=criteria
        self.noise_multiplier = noise_multiplier
        self.design_factors = design_factors
        self.opts=opts
        self.regression_type=regression_type
        
    def get_objective_and_jacobian(self,design_factors,homog_outer_prods,
                                   noise_multiplier,opts):

        #criteria requiring pred_factors
        pred_criteria_funcs = {
            'G':goptimality_criterion,'I':ioptimality_criterion}
        #criteria not requiring pred_factors
        other_criteria_funcs = {
            'A':aoptimality_criterion,'C':coptimality_criterion,
            'D':doptimality_criterion}

        if self.criteria in other_criteria_funcs:
            criteria_fun =  other_criteria_funcs[self.criteria]
            objective = partial(
                criteria_fun,homog_outer_prods,design_factors,
                noise_multiplier=noise_multiplier,return_grad=False,
                regression_type=self.regression_type)
            jac = lambda r: criteria_fun(
                homog_outer_prods,design_factors,r,return_grad=True,
                noise_multiplier=noise_multiplier,
                regression_type=self.regression_type)[1]
        elif self.criteria in pred_criteria_funcs:
            criteria_fun =  pred_criteria_funcs[self.criteria]
            pred_factors = opts['pred_factors']
            objective = partial(
                criteria_fun,homog_outer_prods,design_factors,
                pred_factors,noise_multiplier=noise_multiplier,
                return_grad=False,regression_type=self.regression_type)
            jac = lambda r: criteria_fun(
                homog_outer_prods,design_factors,pred_factors,r,
                return_grad=True,noise_multiplier=noise_multiplier,
                regression_type=self.regression_type)[1]
        elif self.criteria=='R':
            beta = opts['beta']
            pred_factors = opts['pred_factors']
            objective = partial(
                roptimality_criterion,beta,homog_outer_prods,
                design_factors,pred_factors,
                noise_multiplier=noise_multiplier,return_grad=False,
                regression_type=self.regression_type)
            jac = lambda r: roptimality_criterion(
                beta,homog_outer_prods,design_factors,pred_factors,r,
                return_grad=True,noise_multiplier=noise_multiplier,
                regression_type=self.regression_type)[1]
        else:
            msg = f'Optimality criteria: {self.criteria} is not supported. '
            msg += 'Supported criteria are:\n'
            for key in other_criteria_funcs.keys():
                msg += f"\t{key}\n"
            for key in pred_criteria_funcs.keys():
                msg += f"\t{key}\n"
            msg += '\tR\n'
            raise Exception(msg)
        return objective,jac
        

    def solve(self,options=None,init_design=None,return_full=False):
        num_design_pts = self.design_factors.shape[0]
        homog_outer_prods = compute_homoscedastic_outer_products(
            self.design_factors)
        objective,jac = self.get_objective_and_jacobian(
            self.design_factors,homog_outer_prods,
            self.noise_multiplier,self.opts)
            
        if self.criteria=='G': 
            constraint_obj = partial(minimax_oed_constraint_objective,objective)
            constraint_jac = partial(minimax_oed_constraint_jacobian,jac)
            nonlinear_constraints = [NonlinearConstraint(
                constraint_obj,0,np.inf,jac=constraint_jac)]
            return self._solve_minimax(
                nonlinear_constraints,num_design_pts,options,return_full,
                init_design)
            
        self.bounds = Bounds([0]*num_design_pts,[1]*num_design_pts)
        lb_con = ub_con = np.atleast_1d(1)
        A_con = np.ones((1,num_design_pts))
        self.linear_constraint = LinearConstraint(A_con, lb_con, ub_con)
        
        if init_design is None:
            x0 = np.ones(num_design_pts)/num_design_pts
        else:
            x0 = init_design

        #method='trust-constr'
        method='slsqp'
        res = minimize(
            objective, x0, method=method, jac=jac, hess=None,
            constraints=[self.linear_constraint],options=options,
            bounds=self.bounds)
        weights = res.x

        if not return_full:
            return weights
        
        return weights,res.x

    def minimax_nonlinear_constraints(self,parameter_samples,design_samples):
        constraints = []
        for ii in range(parameter_samples.shape[1]):
            design_factors = self.design_factors(
                parameter_samples[:,ii],design_samples)
            homog_outer_prods = compute_homoscedastic_outer_products(
                design_factors)
            opts = copy.deepcopy(self.opts)
            if opts is not None and 'pred_factors' in opts:
                opts['pred_factors']=opts['pred_factors'](
                    parameter_samples[:,ii],opts['pred_samples'])
            if self.noise_multiplier is None:
                noise_multiplier=None
            else:
                noise_multiplier=self.noise_multiplier(
                    parameter_samples[:,ii],design_samples).squeeze()
                assert noise_multiplier.ndim==1
                assert noise_multiplier.shape[0]==design_samples.shape[1]
            obj,jac = self.get_objective_and_jacobian(
                design_factors.copy(),homog_outer_prods.copy(),
                noise_multiplier,copy.deepcopy(opts))
            constraint_obj = partial(minimax_oed_constraint_objective,obj)
            constraint_jac = partial(minimax_oed_constraint_jacobian,jac)
            constraint = NonlinearConstraint(
                constraint_obj,0,np.inf,jac=constraint_jac)
            constraints.append(constraint)
            
        return constraints

    def _solve_minimax(self,nonlinear_constraints,num_design_pts,options,
                       return_full,x0):
        lb_con = ub_con = np.atleast_1d(1)
        A_con = np.ones((1,num_design_pts+1))
        A_con[0,0] = 0
        linear_constraint = LinearConstraint(A_con, lb_con, ub_con)
        constraints = [linear_constraint]
        constraints += nonlinear_constraints
        
        minimax_objective = lambda x: x[0]
        def jac(x):
            vec = np.zeros_like(x)
            vec[0] = 1
            return vec
        bounds = Bounds(
            [0]+[0]*num_design_pts,[np.inf]+[1]*num_design_pts)

        if x0 is None:
            x0 = np.ones(num_design_pts+1)/num_design_pts
            x0[0]=1
        #method = 'trust-constr'
        method = 'slsqp'
        res = minimize(
            minimax_objective, x0, method=method,jac=jac, hess=None,
            constraints=constraints,options=options,
            bounds=bounds)
        
        weights = res.x[1:]
        if not return_full:
            return weights
        else:
            return weights, res

    def solve_nonlinear_minimax(self,parameter_samples,design_samples,
                                options=None,return_full=False,x0=None):
        assert callable(self.design_factors)
        if self.noise_multiplier is not None:
            assert callable(self.noise_multiplier)
        nonlinear_constraints = self.minimax_nonlinear_constraints(
            parameter_samples,design_samples)
        num_design_pts = design_samples.shape[1]
        return self._solve_minimax(nonlinear_constraints,num_design_pts,options,
                                   return_full,x0)

    def bayesian_objective_jacobian_components(
            self,parameter_samples,design_samples):
        objs,jacs=[],[]
        for ii in range(parameter_samples.shape[1]):
            design_factors = self.design_factors(
                parameter_samples[:,ii],design_samples)
            homog_outer_prods = compute_homoscedastic_outer_products(
                design_factors)
            if self.noise_multiplier is None:
                noise_multiplier=None
            else:
                noise_multiplier=self.noise_multiplier(
                    parameter_samples[:,ii],design_samples).squeeze()
                assert noise_multiplier.ndim==1
                assert noise_multiplier.shape[0]==design_samples.shape[1]
            opts = copy.deepcopy(self.opts)
            if opts is not None and 'pred_factors' in opts:
                opts['pred_factors']=opts['pred_factors'](
                    parameter_samples[:,ii],opts['pred_samples'])
            obj,jac = self.get_objective_and_jacobian(
                design_factors.copy(),homog_outer_prods.copy(),
                noise_multiplier,copy.deepcopy(opts))
            objs.append(obj)
            jacs.append(jac)
            
        num_design_pts = homog_outer_prods.shape[2]
        return objs,jacs,num_design_pts

    def solve_nonlinear_bayesian(self,samples,design_samples,
                                 sample_weights=None,options=None,
                                 return_full=False,x0=None):
        assert callable(self.design_factors)
        objs,jacs,num_design_pts = self.bayesian_objective_jacobian_components(
            samples,design_samples)
        lb_con = ub_con = np.atleast_1d(1)
        A_con = np.ones((1,num_design_pts))
        linear_constraint = LinearConstraint(A_con, lb_con, ub_con)
        constraints = [linear_constraint]

        if sample_weights is None:
            sample_weights = np.ones(
                samples.shape[1])/samples.shape[1]
        assert sample_weights.shape[0]==samples.shape[1]
        
        def objective(x):
            objective = 0
            for obj,weight in zip(objs,sample_weights):
                objective += obj(x)*weight
            return objective
            
        def jacobian(x):
            vec = 0
            for jac,weight in zip(jacs,sample_weights):
                vec += jac(x)*weight
            return vec
                
        bounds = Bounds(
            [0]*num_design_pts,[1]*num_design_pts)

        if x0 is None:
            x0 = np.ones(num_design_pts)/num_design_pts
        #method = 'trust-constr'
        method = 'slsqp'
        res = minimize(
            objective, x0, method=method,jac=jacobian, hess=None,
            constraints=constraints,options=options,
            bounds=bounds)

        res['obj_fun']=objective
        
        weights = res.x
        if not return_full:
            return weights
        else:
            return weights, res
