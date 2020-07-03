import numpy as np
from scipy.linalg import solve_triangular
import copy

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
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M1 = homog_outer_prods.dot(design_prob_measure)
    if hetero_outer_prods is not None:
        M0 = hetero_outer_prods.dot(design_prob_measure)
        Q,R = np.linalg.qr(M1)
        u = solve_triangular(R,Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        value = np.sum(u*M0u) / num_pred_pts;
        if (return_grad):
            assert noise_multiplier is not None
            gamma   = -solve_triangular(R,(Q.T.dot(M0u)))
            Fu  = design_factors.dot(u)
            t   = noise_multiplier[:,np.newaxis] * Fu
            Fgamma  = design_factors.dot(gamma)
            gradient = 2*np.sum(Fu*Fgamma,axis=1) + np.sum(t**2,axis=1)
            gradient /= num_pred_pts
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
        value    = np.sum(pred_factors*u.T) / num_pred_pts;
        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_P = design_factors.dot(u);
        gradient   = -np.sum(F_M1_inv_P**2,axis=1) / num_pred_pts;
        return value, gradient


def coptimality_criterion(homog_outer_prods,design_factors,
                          design_prob_measure,return_grad=True,
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
    c = np.ones((num_design_factors,1))
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M1 = homog_outer_prods.dot(design_prob_measure)
    if hetero_outer_prods is not None:
        M0 = hetero_outer_prods.dot(design_prob_measure)
        Q,R = np.linalg.qr(M1)
        u = solve_triangular(R,Q.T.dot(c))
        M0u = M0.dot(u)
        value = u.T.dot(M0u)
        if (return_grad):
            assert noise_multiplier is not None
            gamma   = -solve_triangular(R,(Q.T.dot(M0u)))
            Fu  = design_factors.dot(u)
            t   = noise_multiplier[:,np.newaxis] * Fu
            Fgamma  = design_factors.dot(gamma)
            gradient = 2*Fu*Fgamma + t**2
            return value, gradient
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
        return value, gradient


        
def doptimality_criterion(homog_outer_prods,design_factors,
                          design_prob_measure,return_grad=True,
                          hetero_outer_prods=None,
                          noise_multiplier=None):
    r"""
    Evaluate the D-optimality criterion for a given design probability measure.

    The criteria is
    log determinant [ C(\mu) ]

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
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M1 = homog_outer_prods.dot(design_prob_measure)
    M1_inv = np.linalg.inv(M1)
    if hetero_outer_prods is not None:
        M0 =  hetero_outer_prods.dot(design_prob_measure)
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
                gradient[ii] = np.sum(kappa.dot(homog_outer_prods[:,:,ii])*(
                    -2*gamma.T+noise_multiplier[ii]**2*ident).dot(M1_inv))
            return value, gradient
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
            return value, gradient
        else:
            return value


def aoptimality_criterion(homog_outer_prods,design_factors,
                          design_prob_measure,return_grad=True,
                          hetero_outer_prods=None,
                          noise_multiplier=None):
    r"""
    Evaluate the A-optimality criterion for a given design probability measure.

    The criteria is
    Trace [ C(\mu) ]

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
    # [:,:,0] just changes shape from (N,N,1) to (N,N)
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M1 = homog_outer_prods.dot(design_prob_measure)
    M1_inv = np.linalg.inv(M1)
    if hetero_outer_prods is not None:
        # [:,:,0] just changes shape from (N,N,1) to (N,N)
        M0 =  hetero_outer_prods.dot(design_prob_measure)
        gamma = M0.dot(M1_inv)
        value = np.trace(M1_inv.dot(gamma))
        if (return_grad):
            ident = np.eye(gamma.shape[0])
            M0_inv = np.linalg.inv(M0)
            kappa  = M1.dot(M0_inv)
            gradient = np.zeros(num_design_pts)
            for ii in range(num_design_pts):
                gradient[ii]=np.trace(M1_inv.dot(homog_outer_prods[:,:,ii]).dot(
                    -2*gamma.T+noise_multiplier[ii]**2*ident).dot(M1_inv))
                #TODO order multiplications to be most efficient. Probably
                # need to work on f_i rather than stored outer product
                # f_i.dot(f_i.T)
                #gradient[ii] = np.sum(kappa.dot(homog_outer_prods[:,:,ii])*(
                #    -2*gamma.T+noise_multiplier[ii]**2*ident).dot(M1_inv))
            return value, gradient
        else:
            return value
    else:
        value  = np.trace(M1_inv)
        # Gradient
        if (return_grad):
            #gradient = -np.array([np.trace(M1_inv.dot(homog_outer_prods[:,:,ii]).dot(M1_inv)) for ii in range(homog_outer_prods.shape[2])])
            #TODO order multiplications to be most efficient. Probably need to
            # work on f_i rather than stored outer product f_i.dot(f_i.T)
            gradient = -np.array([(M1_inv*homog_outer_prods[:,:,ii].dot(M1_inv)).sum() for ii in range(homog_outer_prods.shape[2])])
            return value, gradient
        else:
            return value

def roptimality_criterion(beta,homog_outer_prods,design_factors,
                          pred_factors,design_prob_measure,return_grad=True,
                          hetero_outer_prods=None,
                          noise_multiplier=None):
    r"""
    Evaluate the R-optimality criterion for a given design probability measure.

    The criteria is


    where
    C(\mu) = M_1^{-1} M_0 M^{-1}

    For Homoscedastic noise M_0=M_1 and
    C(\mu) = M_1^{-1}

    Parameters
    ----------
    beta : float
       The confidence level of CVAR 

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
    assert beta>=0 and beta<=1
    from pyapprox.cvar_regression import conditional_value_at_risk, \
        conditional_value_at_risk_gradient
    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors   = pred_factors.shape
    if design_prob_measure.ndim==2:
        assert design_prob_measure.shape[1]==1
        design_prob_measure = design_prob_measure[:,0]
    M1 = homog_outer_prods.dot(design_prob_measure)
    if hetero_outer_prods is not None:
        M0 = hetero_outer_prods.dot(design_prob_measure)
        Q,R = np.linalg.qr(M1)
        u = solve_triangular(R,Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        variances = np.sum(u*M0u,axis=0)
        value = conditional_value_at_risk(variances,beta)
        if (return_grad):
            assert noise_multiplier is not None
            gamma   = -solve_triangular(R,(Q.T.dot(M0u)))
            Fu  = design_factors.dot(u)
            t   = noise_multiplier[:,np.newaxis] * Fu
            Fgamma  = design_factors.dot(gamma)
            cvar_grad = conditional_value_at_risk_gradient(variances,beta)
            #gradient = 2*np.sum(Fu*Fgamma*cvar_grad,axis=1) + np.sum(t**2*cvar_grad,axis=1)
            gradient = np.sum((2*Fu*Fgamma+t**2).T*cvar_grad[:,np.newaxis],axis=0)
            return value, gradient
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
        return value, gradient

from scipy.optimize import Bounds, minimize, LinearConstraint, NonlinearConstraint
from functools import partial

def minmax_oed_constraint_objective(local_oed_obj,x):
    return x[0]-local_oed_obj(x[1:])

def minmax_oed_constraint_jacobian(local_oed_jac,x):
    return np.concatenate([np.ones(1),-local_oed_jac(x[1:])])    

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
                 opts=None):
        self.criteria=criteria
        self.noise_multiplier = noise_multiplier
        self.design_factors = design_factors
        self.opts=opts
        
    def get_objective_and_jacobian(self,design_factors,homog_outer_prods,
                                   hetero_outer_prods,noise_multiplier,opts):
        if self.criteria=='A':
            objective = partial(
                aoptimality_criterion,homog_outer_prods,design_factors,
                hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier,return_grad=False)
            jac = lambda r: aoptimality_criterion(
                homog_outer_prods,design_factors,r,return_grad=True,
                hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier)[1]
        elif self.criteria=='C':
            objective = partial(
                coptimality_criterion,homog_outer_prods,design_factors,
                hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier,return_grad=False)
            jac = lambda r: coptimality_criterion(
                homog_outer_prods,design_factors,r,return_grad=True,
                hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier)[1]
        elif self.criteria=='D':
            objective = partial(
                doptimality_criterion,homog_outer_prods,design_factors,
                hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier,return_grad=False)
            jac = lambda r: doptimality_criterion(
                homog_outer_prods,design_factors,r,return_grad=True,
                hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier)[1]
        elif self.criteria=='I':
            pred_factors = opts['pred_factors']
            objective = partial(
                ioptimality_criterion,homog_outer_prods,design_factors,
                pred_factors,hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier,return_grad=False)
            jac = lambda r: ioptimality_criterion(
                homog_outer_prods,design_factors,pred_factors,r,
                return_grad=True,hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier)[1]
        elif self.criteria=='R':
            beta = opts['beta']
            pred_factors = opts['pred_factors']
            objective = partial(
                roptimality_criterion,beta,homog_outer_prods,
                design_factors,
                pred_factors,hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier,return_grad=False)
            jac = lambda r: roptimality_criterion(
                beta,homog_outer_prods,design_factors,pred_factors,r,
                return_grad=True,hetero_outer_prods=hetero_outer_prods,
                noise_multiplier=noise_multiplier)[1]
        else:
            msg = f'Optimality criteria: {self.criteria} is not supported'
            raise Exception(msg)
        return objective,jac
        

    def solve(self,options=None,init_design=None):
        num_design_pts = self.design_factors.shape[0]
        self.bounds = Bounds([0]*num_design_pts,[1]*num_design_pts)
        lb_con = ub_con = np.atleast_1d(1)
        A_con = np.ones((1,num_design_pts))
        self.linear_constraint = LinearConstraint(A_con, lb_con, ub_con)

        homog_outer_prods = compute_homoscedastic_outer_products(
            self.design_factors)
        
        if self.noise_multiplier is not None:
            hetero_outer_prods = compute_heteroscedastic_outer_products(
                self.design_factors,self.noise_multiplier)
        else:
            hetero_outer_prods=None

        if init_design is None:
            x0 = np.ones(num_design_pts)/num_design_pts
        else:
            x0 = init_design
        objective,jac = self.get_objective_and_jacobian(
            self.design_factors,homog_outer_prods,hetero_outer_prods,
            self.noise_multiplier,self.opts)
        res = minimize(
            objective, x0, method='trust-constr', jac=jac, hess=None,
            constraints=[self.linear_constraint],options=options,
            bounds=self.bounds)
        weights = res.x
        return weights

    def minmax_nonlinear_constraints(self,parameter_samples,design_samples):
        constraints = []
        for ii in range(parameter_samples.shape[1]):
            design_factors = self.design_factors(
                parameter_samples[:,ii],design_samples)
            homog_outer_prods = compute_homoscedastic_outer_products(
                design_factors)
            if self.noise_multiplier is not None:
                hetero_outer_prods = compute_heteroscedastic_outer_products(
                    design_factors,self.noise_multiplier)
            else:
                hetero_outer_prods=None
            opts = copy.deepcopy(self.opts)
            if opts is not None and 'pred_factors' in opts:
                opts['pred_factors']=opts['pred_factors'](
                    parameter_samples[:,ii],opts['pred_samples'])
            obj,jac = self.get_objective_and_jacobian(
                design_factors.copy(),homog_outer_prods.copy(),
                hetero_outer_prods,self.noise_multiplier,copy.deepcopy(opts))
            constraint_obj = partial(minmax_oed_constraint_objective,obj)
            constraint_jac = partial(minmax_oed_constraint_jacobian,jac)
            num_design_pts=design_factors.shape[0]
            constraint = NonlinearConstraint(
                constraint_obj,0,np.inf,jac=constraint_jac)
            constraints.append(constraint)
            
        num_design_pts = homog_outer_prods.shape[2]
        return constraints,num_design_pts

    def solve_minmax(self,parameter_samples,design_samples,options=None,
                     return_full=False,x0=None):
        assert callable(self.design_factors)
        nonlinear_constraints,num_design_pts = self.minmax_nonlinear_constraints(
            parameter_samples,design_samples)
        lb_con = ub_con = np.atleast_1d(1)
        A_con = np.ones((1,num_design_pts+1))
        A_con[0,0] = 0
        linear_constraint = LinearConstraint(A_con, lb_con, ub_con)
        constraints = [linear_constraint]
        constraints += nonlinear_constraints
        
        minmax_objective = lambda x: x[0]
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
            minmax_objective, x0, method=method,jac=jac, hess=None,
            constraints=constraints,options=options,
            bounds=bounds)
        
        weights = res.x[1:]
        if not return_full:
            return weights
        else:
            return weights, res

    def bayesian_objective_jacobian_components(
            self,parameter_samples,design_samples):
        objs,jacs=[],[]
        for ii in range(parameter_samples.shape[1]):
            design_factors = self.design_factors(
                parameter_samples[:,ii],design_samples)
            homog_outer_prods = compute_homoscedastic_outer_products(
                design_factors)
            if self.noise_multiplier is not None:
                hetero_outer_prods = compute_heteroscedastic_outer_products(
                    design_factors,self.noise_multiplier)
            else:
                hetero_outer_prods=None
            opts = copy.deepcopy(self.opts)
            if opts is not None and 'pred_factors' in opts:
                opts['pred_factors']=opts['pred_factors'](
                    parameter_samples[:,ii],opts['pred_samples'])
            obj,jac = self.get_objective_and_jacobian(
                design_factors.copy(),homog_outer_prods.copy(),
                hetero_outer_prods,self.noise_multiplier,copy.deepcopy(opts))
            objs.append(obj)
            jacs.append(jac)
            
        num_design_pts = homog_outer_prods.shape[2]
        return objs,jacs,num_design_pts

    def solve_bayesian(self,samples,design_samples,
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
        
        weights = res.x
        if not return_full:
            return weights
        else:
            return weights, res
