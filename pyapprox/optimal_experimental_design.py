import numpy as np
from scipy.linalg import solve as scipy_solve

def compute_subgradient_covariances(factors,noise_multiplier):
    num_design_pts,num_factors = factors.shape
    temp = (factors.T*noise_multiplier).T
    covars = np.empty((num_factors,num_factors,num_design_pts),dtype=float)
    for ii in range(num_design_pts):
        covars[:,:,ii] = np.outer(temp[ii,:],temp[ii,:])
    return covars

def compute_error_hessians(factors):
    num_design_pts,num_factors = factors.shape
    covars = np.empty((num_factors,num_factors,num_design_pts),dtype=float)
    for ii in range(num_design_pts):
        covars[:,:,ii] = np.outer(factors[ii,:],factors[ii,:])
    return covars

def ioptimality_criterion(error_hessians,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          subgradient_covariances=None,
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

    error_hessians : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The hessian M_1 of the error for each design point

    subgradient_covariances : np.ndarray (num_design_factors,num_design_factors,
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
    M1      = np.reshape(
        error_hessians.dot(design_prob_measure),
        (num_design_factors,num_design_factors))
    if subgradient_covariances is not None:
        M0 =  np.reshape(
            subgradient_covariances.dot(design_prob_measure),
            (num_design_factors,num_design_factors))
        Q,R = np.linalg.qr(M1)
        u = scipy_solve(R,Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        value = np.sum(u*M0u) / num_pred_factors;
        if (return_grad):
            assert noise_multiplier is not None
            lam   = - 2 * scipy_solve(R,(Q.T.dot(M0u)))
            Fu    = design_factors.dot(u).sum(axis=1)
            Fnu   = noise_multiplier * Fu
            Flam  = design_factors.dot(lam).sum(axis=1)
            gradient = Fu*Flam + Fnu**2;
            gradient /= num_pred_factors
            return value, gradient
        else:
            return value
    else:
        u = np.linalg.solve(M1,pred_factors.T)
        # Value
        value    = np.sum(pred_factors*u.T) / num_pred_factors;
        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_P = design_factors.dot(u);
        gradient   = -np.sum(F_M1_inv_P**2,axis=1) / num_pred_factors;
        return value, gradient


def coptimality_criterion(error_hessians,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          subgradient_covariances=None,
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

    error_hessians : np.ndarray (num_design_factors,num_design_factors,
                                 num_design_pts)
       The hessian M_1 of the error for each design point

    subgradient_covariances : np.ndarray (num_design_factors,num_design_factors,
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
    c  = np.ones((num_design_factors),dtype=float)
    M1 = np.reshape(
        error_hessians.dot(design_prob_measure),
        (num_design_factors,num_design_factors))
    # scipy solve can take advantage of matrix being having structure, e.g
    # Upper triangular or symmetic positive definite (latter not used here)
    if subgradient_covariances is not None:
        M0 =  np.reshape(
            subgradient_covariances.dot(design_prob_measure),
            (num_design_factors,num_design_factors))
        Q,R = np.linalg.qr(M1)
        u = scipy_solve(R,Q.T.dot(c))
        M0u = M0.dot(u)
        value = np.asscalar(u.T.dot(M0u))
        if (return_grad):
            assert noise_multiplier is not None
            lam   = -2*scipy_solve(R,(Q.T.dot(M0u)))
            Fu    = design_factors.dot(u)
            Fnu   = noise_multiplier * Fu
            Flam  = design_factors.dot(lam)
            gradient = Fu*Flam + Fnu**2;
            return value, gradient
        else:
            return value
    else:
        M1c_inv = np.linalg.solve(M1,c)
        value   = np.asscalar(c.T.dot(M1c_inv))
        # Gradient
        if (return_grad):
            FM1c_inv = design_factors.dot(M1c_inv)
            gradient = -FM1c_inv**2
            return value, gradient
        else:
            return value


        
