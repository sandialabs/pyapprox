import copy
from functools import partial
import numpy as np
from scipy.linalg import solve_triangular

from scipy.optimize import (
    Bounds, minimize, LinearConstraint, NonlinearConstraint
)
from pyapprox.variables.risk import conditional_value_at_risk

from pyapprox.optimization.cvar_regression import (
    conditional_value_at_risk_subgradient,
    smooth_conditional_value_at_risk_split,
    smooth_conditional_value_at_risk_gradient_split
)


def compute_prediction_variance(design_prob_measure, pred_factors,
                                homog_outer_prods, noise_multiplier=None,
                                regression_type='lstsq'):
    M0, M1 = get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier,
        regression_type)
    u = np.linalg.solve(M1, pred_factors.T)
    if M0 is not None:
        M0u = M0.dot(u)
        variances = np.sum(u*M0u, axis=0)
    else:
        variances = np.sum(pred_factors.T*u, axis=0)
    return variances


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
    num_design_pts, num_factors = factors.shape
    homoscedastic_outer_products = np.empty(
        (num_factors, num_factors, num_design_pts), dtype=float)
    for ii in range(num_design_pts):
        homoscedastic_outer_products[:, :, ii] = np.outer(
            factors[ii, :], factors[ii, :])
    return homoscedastic_outer_products


def get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier,
        regression_type):
    r"""
    Compute the matrices :math:`M_0` and :math:`M_1` used to compute the
    asymptotic covariance matrix :math:`C(\mu) = M_1^{-1} M_0 M^{-1}` of the
    linear model

    .. math::  y(x) = F(x)\theta+\eta(x)\epsilon.

    For least squares

    .. math:: M_0 = \sum_{i=1}^M\eta(x_i)^2f(x_i)f(x_i)^Tr_i

    .. math:: M_1 = \sum_{i=1}^Mf(x_i)f(x_i)^Tr_i

    and for quantile regression

    .. math:: M_0 = \sum_{i=1}^M\frac{1}{\eta(x_i)}f(x_i)f(x_i)^Tr_i

    .. math:: M_1 = \sum_{i=1}^Mf(x_i)f(x_i)^Tr_i

    Parameters
    ----------
    homog_outer_prods : np.ndarray(num_factors,num_factors,num_design_pts)
        The outer products :math:`f(x_i)f(x_i)^T` for each design point
        :math:`x_i`

    design_prob_measure : np.ndarray (num_design_pts)
        The weights :math:`r_i` for each design point

    noise_multiplier : np.ndarray (num_design_pts)
        The design dependent noise function :math:`\eta(x)`

    regression_type : string
        The method used to compute the coefficients of the linear model.
        Currently supported options are ``lstsq`` and ``quantile``.

    Returns
    -------
    M0 : np.ndarray (num_factors,num_factors)
        The matrix :math:`M_0`

    M1 : np.ndarray (num_factors,num_factors)
        The matrix :math:`M_1`

    """
    design_prob_measure = design_prob_measure.squeeze()
    if noise_multiplier is None:
        M1 = homog_outer_prods.dot(design_prob_measure)
        assert M1.ndim == 2
        return None, homog_outer_prods.dot(design_prob_measure)

    if regression_type == 'lstsq':
        M0 = homog_outer_prods.dot(design_prob_measure*noise_multiplier**2)
        M1 = homog_outer_prods.dot(design_prob_measure)
    elif regression_type == 'quantile':
        M0 = homog_outer_prods.dot(design_prob_measure)
        M1 = homog_outer_prods.dot(design_prob_measure/noise_multiplier)
    else:
        msg = f'regression type {regression_type} not supported'
        raise Exception(msg)
    return M0, M1


def ioptimality_criterion(homog_outer_prods, design_factors,
                          pred_factors, design_prob_measure, return_grad=True,
                          noise_multiplier=None, regression_type='lstsq',
                          pred_prob_measure=None):
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

    pred_prob_measure : np.ndarray (num_pred_pts)
        The prob measure :math:`\nu` on the prediction points. If none then
        assumes samples are random and
        pred_prob_measure[ii] =  1/num_pred_factors for all uu

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
    # import time
    # t0=time.time()
    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors = pred_factors.shape

    if pred_prob_measure is None:
        pred_prob_measure = np.ones(num_pred_pts)/num_pred_pts

    if design_prob_measure.ndim == 2:
        assert design_prob_measure.shape[1] == 1
        design_prob_measure = design_prob_measure[:, 0]
    M0, M1 = get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier,
        regression_type)
    if noise_multiplier is not None:
        Q, R = np.linalg.qr(M1)
        u = solve_triangular(R, Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        # value = np.sum(u*M0u) / num_pred_pts
        value = np.sum(u*pred_prob_measure*M0u)
        if return_grad:
            gamma = -solve_triangular(R, (Q.T.dot(M0u)))
            Fu = design_factors.dot(u)
            t = noise_multiplier[:, np.newaxis] * Fu
            Fgamma = design_factors.dot(gamma)
            if regression_type == 'lstsq':
                # gradient = 2*np.sum(Fu*Fgamma, axis=1) + np.sum(t**2, axis=1)
                # gradient /= num_pred_pts
                gradient = 2*np.sum(Fu*Fgamma*pred_prob_measure, axis=1) +\
                    np.sum(t**2*pred_prob_measure, axis=1)
            elif regression_type == 'quantile':
                # gradient = 2*np.sum(
                #    Fu*Fgamma/noise_multiplier[:, np.newaxis], axis=1) + \
                #    np.sum(Fu**2, axis=1)
                # gradient /= num_pred_pts
                gradient = 2*np.sum(
                    Fu*Fgamma/noise_multiplier[:, np.newaxis] *
                    pred_prob_measure, axis=1) + np.sum(
                        Fu**2*pred_prob_measure, axis=1)
            # print('that took',time.time()-t0)
            return value, gradient.T
        else:
            return value
    else:
        # import time
        # t0=time.time()
        u = np.linalg.solve(M1, pred_factors.T)
        # Value
        # We want to sum the variances, i.e. the enties of the diagonal of
        # pred_factors.dot(M1.dot(pred_factors.T))
        # We know that diag(A.T.dot(B)) = (A*B).axis=0)
        # The following sums over all entries of A*B we get the mean of the
        # variance
        # value = np.sum(pred_factors*u.T) / num_pred_pts
        # print(pred_prob_measure)
        value = np.sum(np.sum(pred_factors*u.T, axis=1)*pred_prob_measure)
        if not return_grad:
            return value
        # Gradient
        # F_M1_inv_P = design_factors.dot(u)
        # gradient = -np.sum(F_M1_inv_P**2, axis=1) / num_pred_pts
        F_M1_inv_P = design_factors.dot(u)
        gradient = -np.sum(F_M1_inv_P**2*pred_prob_measure, axis=1)
        # print('That took', time.time()-t0)
        return value, gradient.T


def ioptimality_criterion_more_design_pts_than_params(
        homog_outer_prods, design_factors,
        pred_factors, design_prob_measure, return_grad=True,
        noise_multiplier=None, regression_type='lstsq',
        pred_prob_measure=None):

    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors = pred_factors.shape
    if pred_prob_measure is None:
        pred_prob_measure = np.ones(num_pred_pts)/num_pred_pts

    pred_outer_prods = compute_homoscedastic_outer_products(pred_factors)
    B = pred_outer_prods.dot(pred_prob_measure)
    B[B < 0] = 0
    M0, M1 = get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier,
        regression_type)

    # TODO: do not use u_vecs, v_vecs in computations when identity
    # TODO: do not compute wv_vecs when noise_multiplier is None
    num_unknowns = M1.shape[0]
    u_vecs = np.eye(num_unknowns)
    v_vecs = B

    # print(pred_prob_measure)

    # solve state equations
    # wu_vecs = np.linalg.solve(M1, u_vecs)
    # wv_vecs = np.linalg.solve(M1, v_vecs)
    Q, R = np.linalg.qr(M1)
    wu_vecs = solve_triangular(R, Q.T.dot(u_vecs))
    if noise_multiplier is not None:
        wv_vecs = solve_triangular(R, Q.T.dot(v_vecs))
        # value = np.trace(wu_vecs.T.dot(M0).dot(wv_vecs))
        value = np.sum(wu_vecs*M0.dot(wv_vecs))
    else:
        # dont need inv(M).dot(v_vecs) for value but do need it for gradient
        # see below
        wv_vecs = v_vecs
        # value = np.trace(wu_vecs.T.dot(wv_vecs))
        value = np.sum(wu_vecs*wv_vecs)

    if not return_grad:
        return value

    # solve adjoint equations
    if noise_multiplier is not None:
        lamu_vecs = np.linalg.solve(M1, -M0.dot(wv_vecs))
        lamv_vecs = np.linalg.solve(M1, -M0.dot(wu_vecs))
    else:
        lamu_vecs = -solve_triangular(R, Q.T.dot(v_vecs))
        lamv_vecs = -wu_vecs

    # compute gradient
    gradient = np.zeros(num_design_pts)
    for ii in range(num_design_pts):
        M0_grad_ii = homog_outer_prods[:, :, ii]
        if noise_multiplier is not None:
            M0_grad_ii *= noise_multiplier[ii]**2
        M1_grad_ii = homog_outer_prods[:, :, ii]
        for jj in range(num_unknowns):
            gradient[ii] += (
                wu_vecs[:, jj].T.dot(M0_grad_ii.dot(wv_vecs[:, jj])) +
                lamu_vecs[:, jj].T.dot(M1_grad_ii).dot(wu_vecs[:, jj]) +
                lamv_vecs[:, jj].T.dot(M1_grad_ii).dot(wv_vecs[:, jj]))

    return value, gradient


def coptimality_criterion(homog_outer_prods, design_factors,
                          design_prob_measure, return_grad=True,
                          noise_multiplier=None, regression_type='lstsq'):
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
    c = np.ones((num_design_factors, 1))
    if design_prob_measure.ndim == 2:
        assert design_prob_measure.shape[1] == 1
        design_prob_measure = design_prob_measure[:, 0]
    M0, M1 = get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier, regression_type)
    if noise_multiplier is not None:
        Q, R = np.linalg.qr(M1)
        u = solve_triangular(R, Q.T.dot(c))
        M0u = M0.dot(u)
        value = u.T.dot(M0u)
        if return_grad:
            gamma = -solve_triangular(R, (Q.T.dot(M0u)))
            Fu = design_factors.dot(u)
            t = noise_multiplier[:, np.newaxis] * Fu
            Fgamma = design_factors.dot(gamma)
            if regression_type == 'lstsq':
                gradient = 2*Fu*Fgamma + t**2
            elif regression_type == 'quantile':
                gradient = 2*Fu*Fgamma/noise_multiplier[:, np.newaxis] + Fu**2
            return value, gradient.T
        else:
            return value
    else:
        u = np.linalg.solve(M1, c)
        value = c.T.dot(u)
        if not return_grad:
            return value
        # Gradient
        F_M1_inv_c = design_factors.dot(u)
        gradient = -F_M1_inv_c**2
        return value, gradient.T


def doptimality_criterion(homog_outer_prods, design_factors,
                          design_prob_measure, return_grad=True,
                          noise_multiplier=None,
                          regression_type='lstsq',
                          use_cholesky=False,
                          return_hessian=False):
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
    if return_hessian:
        assert return_grad
    num_design_pts, num_design_factors = design_factors.shape
    if design_prob_measure.ndim == 2:
        assert design_prob_measure.shape[1] == 1
        design_prob_measure = design_prob_measure[:, 0]
    # M1 = homog_outer_prods.dot(design_prob_measure)
    M0, M1 = get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier,
        regression_type)
    M1_inv = np.linalg.inv(M1)
    if noise_multiplier is not None:
        gamma = M0.dot(M1_inv)
        value = np.log(np.linalg.det(M1_inv.dot(gamma)))
        if return_grad:
            M0_inv = np.linalg.inv(M0)
            # ident = np.eye(gamma.shape[0])
            # kappa  = M1.dot(M0_inv)
            # gradient = np.zeros(num_design_pts)
            # for ii in range(num_design_pts):
            #     if regression_type=='lstsq':
            #         gradient[ii] = np.sum(
            #              kappa.dot(homog_outer_prods[:,:,ii])*(
            #             -2*gamma.T+noise_multiplier[ii]**2*ident).dot(M1_inv))
            #     elif regression_type=='quantile':
            #         gradient[ii] = np.sum(
            #            kappa.dot(homog_outer_prods[:,:,ii])*(
            #                -2/noise_multiplier[:,np.newaxis][ii]*gamma.T+
            #                ident).dot(M1_inv))
            # return value, gradient

            if regression_type == 'lstsq':
                # computing diagonal elements with trace is more efficient than
                # extracting diagonal (below) and looping through each element
                # (above)
                # gradient = -2*np.diag(design_factors.dot(M1_inv.dot(design_factors.T)))+np.diag(noise_multiplier[:,np.newaxis]*design_factors.dot(M0_inv.dot((noise_multiplier[:,np.newaxis]*design_factors).T)))
                gradient = -2*np.sum(design_factors.T*(M1_inv.dot(design_factors.T)), axis=0)+np.sum(
                    (noise_multiplier[:, np.newaxis]*design_factors).T*(M0_inv.dot((noise_multiplier[:, np.newaxis]*design_factors).T)), axis=0)
            elif regression_type == 'quantile':
                gradient = -2*np.sum(design_factors.T*(M1_inv.dot((design_factors/noise_multiplier[:, np.newaxis]).T)), axis=0)+np.sum(
                    design_factors.T*(M0_inv.dot(design_factors.T)), axis=0)
            return value, gradient
        else:
            return value
    else:
        if use_cholesky:
            chol_factor = np.linalg.cholesky(M1)
            value = -2 * np.log(np.diag(chol_factor)).sum()
        else:
            value = np.log(np.linalg.det(M1_inv))
        # Gradient
        if (return_grad):
            if use_cholesky:
                from scipy.linalg import solve_triangular
                temp = solve_triangular(
                    chol_factor, design_factors.T, lower=True)
                gradient = -(temp**2).sum(axis=0)
                temp = temp.T.dot(temp)  # precompute for hessian
            else:
                temp = design_factors.T*M1_inv.dot(design_factors.T)
                gradient = -(temp).sum(axis=0)
            if not return_hessian:
                return value, gradient.T
            else:
                hessian = temp.T.dot(temp)
                return value, gradient.T, hessian

        else:
            return value


def aoptimality_criterion(homog_outer_prods, design_factors,
                          design_prob_measure, return_grad=True,
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
    if design_prob_measure.ndim == 2:
        assert design_prob_measure.shape[1] == 1
        design_prob_measure = design_prob_measure[:, 0]
    M0, M1 = get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier,
        regression_type)
    M1_inv = np.linalg.inv(M1)
    if noise_multiplier is not None:
        gamma = M0.dot(M1_inv)
        value = np.trace(M1_inv.dot(gamma))
        if (return_grad):
            ident = np.eye(gamma.shape[0])
            gradient = np.zeros(num_design_pts)
            for ii in range(num_design_pts):
                if regression_type == 'lstsq':
                    gradient[ii] = np.trace(
                        M1_inv.dot(homog_outer_prods[:, :, ii]).dot(
                            -2*gamma.T+noise_multiplier[ii]**2*ident).dot(
                                M1_inv))
                elif regression_type == 'quantile':
                    gradient[ii] = np.trace(
                        M1_inv.dot(homog_outer_prods[:, :, ii]).dot(
                            -2*gamma.T/noise_multiplier[:, np.newaxis][ii] +
                            ident).dot(M1_inv))
            return value, gradient.T
        else:
            return value
    else:
        value = np.trace(M1_inv)
        # Gradient
        if (return_grad):
            # gradient = -np.array([(M1_inv*homog_outer_prods[:,:,ii].dot(M1_inv)).sum() for ii in range(homog_outer_prods.shape[2])])
            # below is faster than above
            temp = M1_inv.dot(M1_inv)
            gradient = -np.sum(design_factors.T *
                               (temp).dot(design_factors.T), axis=0)
            return value, gradient.T
        else:
            return value


def roptimality_criterion(beta, homog_outer_prods, design_factors,
                          pred_factors, design_prob_measure, return_grad=True,
                          noise_multiplier=None, regression_type='lstsq',
                          eps=0, tt=None, pred_prob_measure=None):
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

    eps : float
        Hyper-parameter controlling the smoothness of smooth cvar
        if eps==0 then no smoothing is used.

    tt : float
        The current estimate of t used to compute cvar

    pred_prob_measure : np.ndarray (num_pred_pts)
        The prob measure :math:`\nu` on the prediction points. If none then
        assumes samples are random and
        pred_prob_measure[ii] =  1/num_pred_factors for all uu

    Returns
    -------
    value : float
        The value of the objective function

    grad : np.ndarray (num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
    if (eps > 0 and tt is None) or (eps == 0 and tt is not None):
        raise ValueError("If eps>0 tt must both be specified")
    if beta < 0 or beta >= 1:
        raise ValueError("Beta must be in [0,1)")

    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors = pred_factors.shape

    if pred_prob_measure is None:
        pred_prob_measure = np.ones(num_pred_pts)/num_pred_pts
    if design_prob_measure.ndim == 2:
        assert design_prob_measure.shape[1] == 1
        design_prob_measure = design_prob_measure[:, 0]

    M0, M1 = get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier,
        regression_type)
    if noise_multiplier is not None:
        Q, R = np.linalg.qr(M1)
        u = solve_triangular(R, Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        variances = np.sum(u*M0u, axis=0)
        if eps == 0:
            value = conditional_value_at_risk(
                variances, beta, pred_prob_measure)
        else:
            value = smooth_conditional_value_at_risk_split(
                0, eps, beta, variances, tt, pred_prob_measure)
        if not return_grad:
            return value

        gamma = -solve_triangular(R, (Q.T.dot(M0u)))
        Fu = design_factors.dot(u)
        t = noise_multiplier[:, np.newaxis] * Fu
        Fgamma = design_factors.dot(gamma)
        if eps == 0:
            cvar_grad = conditional_value_at_risk_subgradient(
                variances, beta, pred_prob_measure)
        else:
            grad = smooth_conditional_value_at_risk_gradient_split(
                0, eps, beta, variances, tt, pred_prob_measure)
            cvar_grad, t_grad = grad[:-1], grad[-1]
        if regression_type == 'lstsq':
            gradient = (np.sum((2*Fu*Fgamma+t**2).T *
                               cvar_grad[:, np.newaxis], axis=0)).T
        elif regression_type == 'quantile':
            gradient = (np.sum(
                (2*Fu*Fgamma/noise_multiplier[:, np.newaxis]+Fu**2).T *
                cvar_grad[:, np.newaxis], axis=0)).T
        if eps == 0:
            return value, gradient

        return value, np.hstack((gradient, t_grad))
    else:
        u = np.linalg.solve(M1, pred_factors.T)
        # Value
        # We want to sum the variances, i.e. the enties of the diagonal of
        # pred_factors.dot(M1.dot(pred_factors.T))
        # We know that diag(A.T.dot(B)) = (A*B).axis=0)
        # The following sums over all entries of A*B we get the diagonal
        # variances
        variances = np.sum(pred_factors*u.T, axis=1)
        if eps == 0:
            value = conditional_value_at_risk(
                variances, beta, pred_prob_measure)
        else:
            value = smooth_conditional_value_at_risk_split(
                0, eps, beta, variances, tt, pred_prob_measure)

        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_P = design_factors.dot(u)
        if eps == 0:
            cvar_grad = conditional_value_at_risk_subgradient(
                variances, beta, pred_prob_measure)
        else:
            grad = smooth_conditional_value_at_risk_gradient_split(
                0, eps, beta, variances, tt, pred_prob_measure)
            cvar_grad = grad[:-1]
            t_grad = grad[-1]
        gradient = -np.sum(F_M1_inv_P.T**2*cvar_grad[:, np.newaxis], axis=0).T
        if eps == 0:
            return value, gradient

        return value, np.hstack((gradient, t_grad))


def goptimality_criterion(homog_outer_prods, design_factors,
                          pred_factors, design_prob_measure, return_grad=True,
                          noise_multiplier=None, regression_type='lstsq',
                          pred_prob_measure=None):
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

    pred_prob_measure : np.ndarray (num_pred_pts)
         The prob measure :math:`\nu` on the prediction points. G optimality
         ignores this

    Returns
    -------
    value : np.ndarray (num_pred_pts)
        The value of the objective function

    grad : np.ndarray (num_pred_pts,num_design_pts)
        The gradient of the objective function. Only if return_grad is True.
    """
    num_design_pts, num_design_factors = design_factors.shape
    num_pred_pts,   num_pred_factors = pred_factors.shape
    if design_prob_measure.ndim == 2:
        assert design_prob_measure.shape[1] == 1
        design_prob_measure = design_prob_measure[:, 0]
    M0, M1 = get_M0_and_M1_matrices(
        homog_outer_prods, design_prob_measure, noise_multiplier,
        regression_type)
    if noise_multiplier is not None:
        Q, R = np.linalg.qr(M1)
        u = solve_triangular(R, Q.T.dot(pred_factors.T))
        M0u = M0.dot(u)
        variances = np.sum(u*M0u, axis=0)
        value = variances
        if (return_grad):
            gamma = -solve_triangular(R, (Q.T.dot(M0u)))
            Fu = design_factors.dot(u)
            t = noise_multiplier[:, np.newaxis] * Fu
            Fgamma = design_factors.dot(gamma)
            if regression_type == 'lstsq':
                gradient = 2*Fu*Fgamma + t**2
            elif regression_type == 'quantile':
                gradient = 2*Fu*Fgamma/noise_multiplier[:, np.newaxis] + Fu**2
            return value, gradient.T
        else:
            return value
    else:
        u = np.linalg.solve(M1, pred_factors.T)
        # Value
        # We want to sum the variances, i.e. the enties of the diagonal of
        # pred_factors.dot(M1.dot(pred_factors.T))
        # We know that diag(A.T.dot(B)) = (A*B).axis=0)00
        # The following sums over all entries of A*B we get the diagonal
        # variances
        variances = np.sum(pred_factors*u.T, axis=1)
        value = variances
        if (not return_grad):
            return value
        # Gradient
        F_M1_inv_P = design_factors.dot(u)
        gradient = -F_M1_inv_P**2
        return value, gradient.T


def minimax_oed_objective(x):
    return x[0]


def minimax_oed_objective_jacobian(x):
    vec = np.zeros_like(x)
    vec[0] = 1
    return vec


def minimax_oed_constraint_objective(local_oed_obj, x):
    return x[0]-local_oed_obj(x[1:])


def minimax_oed_constraint_jacobian(local_oed_jac, x):
    jac = -local_oed_jac(x[1:])
    jac = np.atleast_2d(jac)
    return np.hstack([np.ones((jac.shape[0], 1)), jac])


def get_minimax_bounds(num_design_pts):
    return Bounds(
        [0]+[0]*num_design_pts, [np.inf]+[1]*num_design_pts)


def get_minimax_default_initial_guess(num_design_pts):
    x0 = np.ones(num_design_pts+1)/num_design_pts
    x0[0] = 1
    return x0


def get_minimax_linear_constraints(num_design_pts):
    lb_con = ub_con = np.atleast_1d(1)
    A_con = np.ones((1, num_design_pts+1))
    A_con[0, 0] = 0
    return LinearConstraint(A_con, lb_con, ub_con)


def extract_minimax_design_from_optimize_result(res):
    return res.x[1:]


def r_oed_objective(beta, pred_weights, x):
    num_pred_pts = pred_weights.shape[0]
    u = x[1:num_pred_pts+1].squeeze()
    return x[0]+1/(1-beta)*u.dot(pred_weights)


def r_oed_objective_jacobian(beta, pred_weights, x):
    num_pred_pts = pred_weights.shape[0]
    vec = np.zeros(x.shape[0])
    vec[0] = 1
    vec[1:num_pred_pts+1] = pred_weights/(1-beta)
    return vec


def r_oed_constraint_objective(num_design_pts, local_oed_obj, x):
    num_pred_pts = x.shape[0]-(1+num_design_pts)
    t = x[0]
    u = x[1:num_pred_pts+1].squeeze()
    return t+u-local_oed_obj(x[num_pred_pts+1:]).T


def r_oed_constraint_jacobian(num_design_pts, local_oed_jac, x):
    num_pred_pts = x.shape[0]-(1+num_design_pts)
    jac = -local_oed_jac(x[num_pred_pts+1:])
    assert jac.ndim == 2 and jac.shape == (num_pred_pts, num_design_pts)
    jac = np.hstack(
        [np.ones((jac.shape[0], 1)), np.eye(jac.shape[0], num_pred_pts), jac])
    return jac


def r_oed_sparse_constraint_jacobian(num_design_pts, local_oed_jac, x):
    num_pred_pts = x.shape[0]-(1+num_design_pts)
    local_jac = -local_oed_jac(x[num_pred_pts+1:])
    assert local_jac.ndim == 2 and local_jac.shape == (
        num_pred_pts, num_design_pts)
    jac = np.empty((num_pred_pts, 2+num_design_pts))
    jac[:, :2] = 1
    jac[:, 2:] = local_jac
    jac = jac.ravel()
    return jac


def get_r_oed_bounds(num_pred_pts, num_design_pts):
    return Bounds(
        [-np.inf]+[0]*(num_pred_pts+num_design_pts),
        [np.inf]*(num_pred_pts+1)+[1]*(num_design_pts))


def get_r_oed_default_initial_guess(num_pred_pts, num_design_pts):
    x0 = np.ones(1+num_pred_pts+num_design_pts)/num_design_pts
    x0[:num_pred_pts+1] = 1
    return x0


def get_r_oed_linear_constraints(num_pred_pts, num_design_pts):
    lb_con = ub_con = np.atleast_1d(1)
    A_con = np.ones((1, 1+num_pred_pts+num_design_pts))
    A_con[0, :num_pred_pts+1] = 0
    return LinearConstraint(A_con, lb_con, ub_con)


def extract_r_oed_design_from_optimize_result(num_design_pts, res):
    num_pred_pts = res.x.shape[0]-(1+num_design_pts)
    return res.x[num_pred_pts+1:]


def get_r_oed_jacobian_structure(num_pred_pts, num_design_pts):
    nonlinear_constraint_structure = np.hstack(
        [np.ones((num_pred_pts, 1)), np.eye(num_pred_pts, num_pred_pts),
         np.ones((num_pred_pts, num_design_pts))])
    linear_constraint_structure = np.zeros((1, num_pred_pts+num_design_pts+1))
    linear_constraint_structure[0, 1+num_pred_pts:] = 1
    structure = np.vstack(
        [linear_constraint_structure, nonlinear_constraint_structure])
    return np.nonzero(structure)


class AlphabetOptimalDesign(object):
    r"""
    Construct optimal experimental designs using functions of the fisher
    information matrix

    Notes
    -----
    Even though scipy.optimize.minimize may print the warning
    "UserWarning: delta_grad == 0.0. Check if the approximated function is
    linear. If the function is linear better results can be obtained by
    defining the Hessian as zero instead of using quasi-Newton
    approximations"
    the Hessian is not zero so do not make this change
    """

    def __init__(self, criteria, design_factors, noise_multiplier=None,
                 opts=None, regression_type='lstsq'):
        r"""
        Parameters
        ----------
        criteria : string
            The name of the optimality criteria

        design_factors : np.ndarray (num_design_pts, num_design_factors)
            The design factors evaluated at each of the design points

        noise_multiplier : np.ndarray (num_design_pts)
            The design dependent noise function :math:`\eta(x)`

        opts : dict
            Options passed to the non-linear optimizer which solves
            the  OED problem

        regression_type : string
            The method used to compute the coefficients of the linear model.
            Currently supported options are ``lstsq`` and ``quantile``.
        """

        self.criteria = criteria
        self.noise_multiplier = noise_multiplier
        self.design_factors = design_factors
        self.opts = opts
        self.regression_type = regression_type

        homog_outer_prods = compute_homoscedastic_outer_products(
            self.design_factors)
        self.__objective = self.setup_objective(
            self.criteria, homog_outer_prods, self.design_factors,
            self.noise_multiplier, self.opts)
        self.__constraints = self.__setup_constraints(
            self.criteria, homog_outer_prods, self.design_factors,
            self.noise_multiplier, self.opts)

    def objective(self, xx, **kwargs):
        # kwargs needed for smooth version of r-optimality
        return self.__objective(xx, **kwargs)

    def constraint(self, xx):
        assert len(self.__constraints) == 1
        return self.__constraints[0](xx)

    def setup_objective(self, criteria, homog_outer_prods, design_factors,
                        noise_multiplier, opts):
        pred_criteria = ["I", "R", "G"]
        other_criteria = ["A", "C", "D"]
        if self.criteria in pred_criteria:
            return self.__setup_pred_objective(
                criteria, homog_outer_prods, design_factors,
                noise_multiplier, opts)

        elif self.criteria in other_criteria:
            return self.__setup_other_objective(
                criteria, homog_outer_prods, design_factors,
                noise_multiplier, opts)

        else:
            msg = f'Optimality criteria: {self.criteria} is not supported. '
            msg += 'Supported criteria are:\n'
            for key in pred_criteria+other_criteria:
                msg += f"\t{key}\n"
                msg += '\tR\n'
            raise Exception(msg)

    def __setup_pred_objective(self, criteria, homog_outer_prods,
                               design_factors, noise_multiplier, opts):
        pred_factors = opts['pred_factors']
        pred_prob_measure = opts.get("pred_prob_measure", None)

        if criteria == "I":
            return partial(
                ioptimality_criterion, homog_outer_prods,
                design_factors,
                pred_factors, noise_multiplier=noise_multiplier,
                return_grad=True, regression_type=self.regression_type,
                pred_prob_measure=pred_prob_measure)
        elif criteria == "R":
            eps = opts.get('eps', 0)
            beta = opts['beta']
            if eps > 0:
                return lambda xx: roptimality_criterion(
                    beta, homog_outer_prods,
                    design_factors, pred_factors, xx[:-1],
                    noise_multiplier=noise_multiplier, return_grad=True,
                    regression_type=self.regression_type, eps=eps,
                    pred_prob_measure=pred_prob_measure, tt=xx[-1])
            else:
                if pred_prob_measure is None:
                    num_pred_pts = opts['pred_factors'].shape[0]
                    pred_prob_measure = np.ones(num_pred_pts)/num_pred_pts
                r_obj = partial(r_oed_objective, beta, pred_prob_measure)
                r_jac = partial(
                    r_oed_objective_jacobian, beta, pred_prob_measure)
                return lambda xx: (r_obj(xx), r_jac(xx))
        elif criteria == "G":
            # pred_prob_measure is not used here because we are looking
            # for worst case
            return partial(
                goptimality_criterion, homog_outer_prods,
                design_factors,
                pred_factors, noise_multiplier=noise_multiplier,
                return_grad=True, regression_type=self.regression_type)

    def __setup_other_objective(self, criteria, homog_outer_prods,
                                design_factors, noise_multiplier, opts):
        other_criteria_funcs = {
            'A': aoptimality_criterion, 'C': coptimality_criterion,
            'D': doptimality_criterion}
        criteria_fun = other_criteria_funcs[criteria]
        return partial(
            criteria_fun, homog_outer_prods, design_factors,
            noise_multiplier=noise_multiplier, return_grad=True,
            regression_type=self.regression_type)

    def __setup_constraints(self, criteria, homog_outer_prods,
                            design_factors, noise_multiplier, opts):
        num_design_pts = design_factors.shape[0]
        if criteria == "R" and opts.get('eps', 0) == 0:
            objective = self.setup_objective(
                "G", homog_outer_prods,
                design_factors, noise_multiplier, opts)

            def fun(xx): return objective(xx)[0]
            def jac(xx): return objective(xx)[1]
            constraint_obj = partial(
                r_oed_constraint_objective, num_design_pts, fun)
            constraint_jac = partial(
                r_oed_constraint_jacobian, num_design_pts, jac)
            return [NonlinearConstraint(
                constraint_obj, 0, np.inf, jac=constraint_jac)]
        elif criteria == "G":
            objective = self.setup_objective(
                "G", homog_outer_prods,
                design_factors, noise_multiplier, opts)
            constraint_obj = partial(
                minimax_oed_constraint_objective, partial(
                    obj_from_obj_plus_jac_fun, objective))
            constraint_jac = partial(
                minimax_oed_constraint_jacobian, partial(
                    jac_from_obj_plus_jac_fun, objective))
            return [NonlinearConstraint(
                constraint_obj, 0, np.inf, jac=constraint_jac)]
        return None

    def __minimize(self, x0, objective, jac, options):
        if 'solver' in options:
            options = options.copy()
            method = options['solver']
            del options['solver']
        else:
            # method='trust-constr'
            method = 'slsqp'

        if method == 'ipopt':
            # when printing results of derivative_test The first floating point
            # number is the value given by the user code, and the second number
            # (after "~") is the finite differences estimation. Finally, the
            # number in square brackets is the relative difference between
            # these two numbers.
            bounds = [
                [lb, ub] for lb, ub in zip(self.bounds.lb, self.bounds.ub)]
            from scipy.optimize._constraints import new_constraint_to_old
            con = new_constraint_to_old(self.linear_constraint, x0)
            from ipopt import minimize_ipopt
            res = minimize_ipopt(
                objective, x0, jac=jac, bounds=bounds, constraints=con,
                options=options)
        else:
            res = minimize(
                objective, x0, method=method, jac=jac, hess=None,
                constraints=[self.linear_constraint], options=options,
                bounds=self.bounds)

        return res

    def __setup_bounds(self):
        """
        Define the bounds of the optimization variables
        """
        num_design_pts = self.design_factors.shape[0]
        if self.criteria == "R" and self.opts.get('eps', 0) > 0:
            # smoothed roptimaility must solve for design and variable tt which
            # estimates the quantile used in cvar so make bounds correct size
            return Bounds(
                [0]*num_design_pts+[-np.inf], [1]*num_design_pts+[np.inf])
        else:
            return Bounds([0]*num_design_pts, [1]*num_design_pts)

    def __setup_linear_constraint(self):
        """
        Enforce the constraint that the sum of the design probability masses
        sum to one
        """
        num_design_pts = self.design_factors.shape[0]
        lb_con = ub_con = np.atleast_1d(1)
        if self.criteria == "R" and self.opts.get('eps', 0) > 0:
            # smoothed roptimaility must solve for design and variable tt which
            # estimates the quantile used in cvar so make constraint correct
            # size
            A_con = np.hstack((np.ones((1, num_design_pts)), np.zeros((1, 1))))
        else:
            A_con = np.ones((1, num_design_pts))
        return LinearConstraint(A_con, lb_con, ub_con)

    def __setup_initial_guess(self, init_design):
        num_design_pts = self.design_factors.shape[0]
        if self.criteria == "R" and self.opts.get('eps', 0) > 0:
            # smoothed roptimaility must solve for design and variable tt which
            # estimates the quantile used in cvar so add initial guess of 1
            if init_design is None:
                return np.hstack((np.ones(num_design_pts)/num_design_pts, 1))
            else:
                return np.hstack((init_design, 1))

        if init_design is None:
            return np.ones(num_design_pts)/num_design_pts
        else:
            return init_design

    def __generic_solve(self, options, init_design, return_full):
        """
        Some criteria can be solved in exactly the same way, that is using
        the workflow in this function.
        """
        self.bounds = self.__setup_bounds()
        self.linear_constraint = self.__setup_linear_constraint()
        x0 = self.__setup_initial_guess(init_design)
        res = self.__minimize(x0, self.objective, True, options)
        weights = res.x

        if not return_full:
            return weights

        return weights, res

    def __roptimality_solve(self, options, init_design, return_full):
        if self.opts.get("eps", 0) == 0:
            return self.__roptimality_solve_slack(
                options, init_design, return_full)

        return self.__roptimality_solve_smooth(
            options, init_design, return_full)

    def __roptimality_solve_slack(self, options, init_design, return_full):
        """
        Solve exact reformulation of roptimality by adding slack variables.
        This adds a lot of constraints and so can be computationally demanding
        """
        num_design_pts = self.design_factors.shape[0]
        num_pred_pts = self.opts['pred_factors'].shape[0]
        return self._solve_minimax(
            self.__constraints, num_design_pts, options, return_full,
            init_design, objective=self.objective, jac=True,
            get_bounds=partial(get_r_oed_bounds, num_pred_pts),
            get_init_guess=partial(
                get_r_oed_default_initial_guess, num_pred_pts),
            get_linear_constraint=partial(
                get_r_oed_linear_constraints, num_pred_pts),
            extract_design_from_optimize_result=partial(
                extract_r_oed_design_from_optimize_result, num_design_pts))

    def __roptimality_solve_smooth(self, options, init_design, return_full):
        """
        Solve the smoothed version of roptimality. The smoothed version
        introduces a small error but is much more computationaly efficient
        than the slack variable baed reformulation.
        """
        weights, res = self.__generic_solve(options, init_design, return_full)

        # smoothed roptimaility must solve for design and variable tt which
        # estimates the quantile used in cvar so subtract off tt from design
        # returned
        if not return_full:
            return weights[:-1]
        return weights[:-1], res

    def __goptimality_solve(self, options, init_design, return_full):
        num_design_pts = self.design_factors.shape[0]
        return self._solve_minimax(
            self.__constraints, num_design_pts, options, return_full,
            init_design)

    def solve(self, options=None, init_design=None, return_full=False):
        if self.criteria in ["A", "D", "C", "I"]:
            return self.__generic_solve(options, init_design, return_full)

        if self.criteria == "R":
            return self.__roptimality_solve(options, init_design, return_full)

        if self.criteria == 'G':
            return self.__goptimality_solve(options, init_design, return_full)

    def _solve_minimax(self, nonlinear_constraints, num_design_pts, options,
                       return_full, x0, objective=minimax_oed_objective,
                       jac=minimax_oed_objective_jacobian,
                       get_bounds=get_minimax_bounds,
                       get_init_guess=get_minimax_default_initial_guess,
                       get_linear_constraint=get_minimax_linear_constraints,
                       extract_design_from_optimize_result=extract_minimax_design_from_optimize_result):

        self.linear_constraint = get_linear_constraint(num_design_pts)
        constraints = [self.linear_constraint]
        constraints += nonlinear_constraints

        self.bounds = get_bounds(num_design_pts)

        if x0 is None:
            x0 = get_init_guess(num_design_pts)

        if 'solver' in options:
            method = options['solver']
            options = options.copy()
            del options['solver']
        else:
            # method='trust-constr'
            method = 'slsqp'

        if method == 'ipopt':
            bounds = [[lb, ub]
                      for lb, ub in zip(self.bounds.lb, self.bounds.ub)]
            from scipy.optimize._constraints import new_constraint_to_old
            constraints = [
                new_constraint_to_old(con, x0)[0] for con in constraints]
            from ipopt import minimize_ipopt
            try:
                # if version of ipopt supports it pass in jacobian structure
                options = options.copy()
                constraint_jacobianstructure = options.get(
                    'constraint_jacobianstructure', None)
                if 'constraint_jacobianstructure' in options:
                    del options['constraint_jacobianstructure']
                res = minimize_ipopt(
                    objective, x0, jac=jac, bounds=bounds,
                    constraints=constraints,
                    options=options,
                    constraint_jacobianstructure=constraint_jacobianstructure)
            except:
                res = minimize_ipopt(
                    objective, x0, jac=jac, bounds=bounds, constraints=constraints,
                    options=options)
        else:
            res = minimize(
                objective, x0, method=method, jac=jac, hess=None,
                constraints=constraints, options=options,
                bounds=self.bounds)

        weights = extract_design_from_optimize_result(res)
        if not return_full:
            return weights
        else:
            return weights, res


def obj_from_obj_plus_jac_fun(fun, xx):
    return fun(xx)[0]


def jac_from_obj_plus_jac_fun(fun, xx):
    return fun(xx)[1]


class NonLinearAlphabetOptimalDesign(AlphabetOptimalDesign):
    r"""
    Construct minimax optimal experimental designs of non-linear models
    by sampling fisher information matrix at multiple uncertain parameter
    realizations.
    """

    def __init__(self, criteria, design_factors, noise_multiplier=None,
                 opts=None, regression_type='lstsq'):
        r"""
        Parameters
        ----------
        criteria : string
            The name of the optimality criteria

        design_factors : np.ndarray (num_design_pts, num_design_factors)
            The design factors evaluated at each of the design points

        noise_multiplier : np.ndarray (num_design_pts)
            The design dependent noise function :math:`\eta(x)`

        opts : dict
            Options passed to the non-linear optimizer which solves
            the  OED problem

        regression_type : string
            The method used to compute the coefficients of the linear model.
            Currently supported options are ``lstsq`` and ``quantile``.
        """
        if not callable(design_factors):
            msg = "design_factors must be a function that returns local"
            msg += " design factors for a given estimate of the unknown"
            msg += "model parameters"
            raise TypeError(msg)

        self.criteria = criteria
        self.noise_multiplier = noise_multiplier
        self.design_factors = design_factors
        self.opts = opts
        self.regression_type = regression_type

    def setup_minimax_nonlinear_constraints(
            self, parameter_samples, design_samples):
        constraints = []
        for ii in range(parameter_samples.shape[1]):
            design_factors = self.design_factors(
                parameter_samples[:, ii], design_samples)
            homog_outer_prods = compute_homoscedastic_outer_products(
                design_factors)
            opts = copy.deepcopy(self.opts)
            if opts is not None and 'pred_factors' in opts:
                opts['pred_factors'] = opts['pred_factors'](
                    parameter_samples[:, ii], opts['pred_samples'])
            if self.noise_multiplier is None:
                noise_multiplier = None
            else:
                noise_multiplier = self.noise_multiplier(
                    parameter_samples[:, ii], design_samples).squeeze()
                assert noise_multiplier.ndim == 1
                assert noise_multiplier.shape[0] == design_samples.shape[1]
            #obj, jac = self.get_objective_and_jacobian(
            #    design_factors.copy(), homog_outer_prods.copy(),
            #    noise_multiplier, copy.deepcopy(opts))
            objective = self.setup_objective(
                self.criteria, homog_outer_prods.copy(), design_factors.copy(),
                noise_multiplier, copy.deepcopy(opts))
            constraint_obj = partial(
                minimax_oed_constraint_objective, partial(
                    obj_from_obj_plus_jac_fun, objective))
            constraint_jac = partial(
                minimax_oed_constraint_jacobian, partial(
                    jac_from_obj_plus_jac_fun, objective))
            constraint = NonlinearConstraint(
                constraint_obj, 0, np.inf, jac=constraint_jac)
            constraints.append(constraint)

        return constraints

    def setup_objective(self, criteria, homog_outer_prods, design_factors,
                        noise_multiplier, opts):
        if criteria == "R":
            pred_factors = opts['pred_factors']
            pred_prob_measure = opts.get("pred_prob_measure", None)
            eps = 0  # just assume cvar is differentiable and
            # use subgradients to compute design. This may not always work
            # but has worked for all examples I have testsed
            # we can introduce slack variables to make sure problem is
            # differentiable everywhere but this will be computationally
            # expensive
            beta = opts['beta']
            return lambda xx: roptimality_criterion(
                beta, homog_outer_prods,
                design_factors, pred_factors, xx,
                noise_multiplier=noise_multiplier, return_grad=True,
                regression_type=self.regression_type, eps=eps,
                pred_prob_measure=pred_prob_measure)

        return super().setup_objective(
            criteria, homog_outer_prods, design_factors,
            noise_multiplier, opts)

    def solve_nonlinear_minimax(self, parameter_samples, design_samples,
                                options=None, return_full=False, x0=None):
        assert callable(self.design_factors)
        if self.noise_multiplier is not None:
            assert callable(self.noise_multiplier)
        self.nonlinear_constraints = self.setup_minimax_nonlinear_constraints(
            parameter_samples, design_samples)
        num_design_pts = design_samples.shape[1]
        return self._solve_minimax(
            self.nonlinear_constraints, num_design_pts, options, return_full, x0)

    def bayesian_objective_jacobian_components(
            self, parameter_samples, design_samples):
        objs = []
        for ii in range(parameter_samples.shape[1]):
            design_factors = self.design_factors(
                parameter_samples[:, ii], design_samples)
            homog_outer_prods = compute_homoscedastic_outer_products(
                design_factors)
            if self.noise_multiplier is None:
                noise_multiplier = None
            else:
                noise_multiplier = self.noise_multiplier(
                    parameter_samples[:, ii], design_samples).squeeze()
                assert noise_multiplier.ndim == 1
                assert noise_multiplier.shape[0] == design_samples.shape[1]
            opts = copy.deepcopy(self.opts)
            if opts is not None and 'pred_factors' in opts:
                opts['pred_factors'] = opts['pred_factors'](
                    parameter_samples[:, ii], opts['pred_samples'])
            objective = self.setup_objective(
                self.criteria, homog_outer_prods.copy(), design_factors.copy(),
                noise_multiplier, copy.deepcopy(opts))
            objs.append(objective)

        num_design_pts = homog_outer_prods.shape[2]
        return objs, num_design_pts

    def solve_nonlinear_bayesian(self, samples, design_samples,
                                 sample_weights=None, options=None,
                                 return_full=False, x0=None):
        assert callable(self.design_factors)
        objs, num_design_pts = \
            self.bayesian_objective_jacobian_components(
                samples, design_samples)
        lb_con = ub_con = np.atleast_1d(1)
        A_con = np.ones((1, num_design_pts))
        linear_constraint = LinearConstraint(A_con, lb_con, ub_con)
        constraints = [linear_constraint]

        if sample_weights is None:
            sample_weights = np.ones(
                samples.shape[1])/samples.shape[1]
        assert sample_weights.shape[0] == samples.shape[1]

        def objective(xx):
            objective = 0
            vec = 0
            for obj, weight in zip(objs, sample_weights):
                val, grad = obj(xx)
                objective += val*weight
                vec += grad*weight
            return objective, vec

        # def jacobian(x):
        #     vec = 0
        #     for jac, weight in zip(jacs, sample_weights):
        #         vec += jac(x)*weight
        #     return vec

        bounds = Bounds(
            [0]*num_design_pts, [1]*num_design_pts)

        if x0 is None:
            x0 = np.ones(num_design_pts)/num_design_pts

        if 'solver' in options:
            options = options.copy()
            method = options['solver']
            del options['solver']
        else:
            # method='trust-constr'
            method = 'slsqp'

        if method == 'ipopt':
            bounds = [[lb, ub]
                      for lb, ub in zip(self.bounds.lb, self.bounds.ub)]
            from ipopt import minimize_ipopt
            from scipy.optimize._constraints import new_constraint_to_old
            con = new_constraint_to_old(self.linear_constraint, x0)
            res = minimize_ipopt(
                objective, x0, jac=True, bounds=bounds, constraints=con,
                options=options)
        else:
            res = minimize(
                objective, x0, method=method, jac=True, hess=None,
                constraints=constraints, options=options,
                bounds=bounds)

        res['obj_fun'] = objective

        weights = res.x
        if not return_full:
            return weights
        
        return weights, res


def optimal_experimental_design(
        design_pts, fun, criteria, regression_type='lstsq',
        noise_multiplier=None, solver_opts=None, pred_factors=None,
        cvar_tol=None):
    r"""
    Compute optimal experimental designs for models of the form

    .. math:: y(\rv)=m(\rv;\theta)+\eta(\rv)\epsilon

    to be used with estimators, such as least-squares and quantile regression,
    to find approximate parameters
    :math:`\hat{\theta}` that are the solutions of

    .. math:: \mathrm{argmin}_\theta \frac{1}{M}\sum_{i=1}^M e(y_i-m(\rv_i;\theta))

    for some loss function :math:`e`

    Parameters
    ----------
    design_pts : np.ndarray (nvars,nsamples)
        All possible experimental conditions

    design_factors : callable or np.ndarray
       The function :math:`m(\rv;\theta)` with the signature

       `design_factors(z,p)->np.ndarray`

       where `z` are the design points and `p` are the unknown
       parameters of the function which will be estimated
       from data collected using the optimal design

       A np.ndarray with shape (nsamples,nfactors) where
       each column is the jacobian of :math:`m(\rv,\theta)` for
       some :math:`\theta`

    criteria : string
       The optimality criteria. Supported criteria are

       - ``'A'``
       - ``'D'``
       - ``'C'``
       - ``'I'``
       - ``'R'``
       - ``'G'``

       The criteria I,G and R require pred_factors to be provided. A, C and D
       optimality do not. R optimality requires cvar_tol to be provided.

       See [KJHSIAMUQ2020]_ for a definition of these criteria

    regression_type : string
        The method used to compute the coefficients of the linear model.
        This defines the loss function :math:`e`.
        Currently supported options are

        - ``'lstsq'``
        - ``'quantile'``

        Both these options will produce the same design if noise_multiplier
        is None

    noise_multiplier : np.ndarray (nsamples)
        An array specifying the noise multiplier :math:`\eta` at each
        design point

    solver_opts : dict
        Options passed to the non-linear optimizer which solves
        the OED problem

    pred_factors : callable or np.ndarray
        The function :math:`g(\rv;\theta)` with the signature

        `design_factors(z,p)->np.ndarray`

        where `z` are the prediction points and `p` are the unknown
        parameters

        A np.ndarray with shape (nsamples,nfactors) where
        each column is the jacobian of :math:`g(\rv,\theta)` for
        some :math:`\theta`

    cvar_tol : float
        The :math:`0\le\beta<1` quantile defining the R-optimality criteria.
        When :math:`\beta=0`, I and R optimal designs will be the same.

    Returns
    -------
    final_design_pts : np.ndarray (nvars,nfinal_design_pts)
        The design points used in the experimental design

    nrepetitions : np.ndarray (nfinal_design_pts)
        The number of times to evaluate the model at each
        design point

    References
    ----------
    .. [KJHSIAMUQ2020] `D.P. Kouri, J.D. Jakeman, G. Huerta, Risk-Adapted Optimal Experimental Design.`

    """
    if not callable(fun):
        design_factors = fun

    opts = None
    if pred_factors is not None:
        opts = {'pred_factors': pred_factors}
        if cvar_tol is not None:
            opts['beta'] = cvar_tol

    ncandidate_design_pts = design_pts.shape[1]
    opt_problem = AlphabetOptimalDesign(
        criteria, design_factors, regression_type=regression_type,
        noise_multiplier=noise_multiplier, opts=opts)
    if solver_opts is None:
        solver_opts = {'iprint': 1, 'ftol': 1e-8}
    mu = opt_problem.solve(solver_opts)
    mu = np.round(mu*ncandidate_design_pts)
    II = np.where(mu > 0)[0]
    return design_pts[:, II], mu[II]
