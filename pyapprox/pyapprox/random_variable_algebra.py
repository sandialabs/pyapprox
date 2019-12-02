#!/usr/bin/env python
import numpy as np
from scipy.optimize import bisect, newton
from pyapprox.utilities import cartesian_product, outer_product
import matplotlib.pyplot as plt

def scalar_multiple_of_random_variable(pdf_func,coefficient,xx):
    assert np.isscalar(coefficient)
    if abs(coefficient)>0:
        return pdf_func(xx/coefficient)*abs(1./coefficient)
    else:
        return np.zeros(xx.shape[0])

def power_of_random_variable_pdf(pdf_func,power,xx):
    if power==1:
        return pdf_func(xx)
    root = 1./power
    I = np.where(xx>=0)[0]
    J = np.where(xx<0)[0]
    vals = np.zeros_like(xx)
    vals[I]=pdf_func(xx[I]**root)*np.absolute(root*(xx[I]**(root-1)))
    vals[J]=pdf_func(-((-xx[J])**root))*np.absolute(-root*((-xx[J])**(root-1)))
    if power%2==0:
        vals[I]+=pdf_func(-(xx[I]**root))*np.absolute(-root*(xx[I]**(root-1)))
        vals[J]+=pdf_func(-((-xx[J])**root))*np.absolute(
            -root*(-xx[J])**(root-1))
    return vals

def sum_of_independent_random_variables_pdf(pdf,gauss_quadrature_rules,zz):
    """
    Compute PDF of Z = X_1+X_2+...+X_d
    Parameters
    ---------
    pdf : callable
        PDF of X_1
    
    gauss_quadrature_rules : list [x,w]
        List of gaussian quadrature rules which integrate with respect to PDFs
        of X_2,...X_d

    zz : np.ndarray (num_samples)
       The locations to evaluate the pdf of Z
    """
    num_vars = len(gauss_quadrature_rules)+1
    xx = cartesian_product(
        [gauss_quadrature_rules[ii][0] for ii in range(num_vars-1)])
    ww = outer_product(
        [gauss_quadrature_rules[ii][1] for ii in range(num_vars-1)])
    
    vals = np.zeros_like(zz)
    for ii in range(vals.shape[0]):
        vals[ii] = pdf(zz[ii]-xx.sum(axis=0)).dot(ww)
    return vals

def product_of_independent_random_variables_pdf(pdf,gauss_quadrature_rules,zz):
    """
    Compute PDF of Z = X_1*X_2*...*X_d
    Parameters
    ---------
    pdf : callable
        PDF of X_1
    
    gauss_quadrature_rules : list [x,w]
        List of gaussian quadrature rules which integrate with respect to PDFs
        of X_2,...X_d

    zz : np.ndarray (num_samples)
       The locations to evaluate the pdf of Z
    """
    num_vars = len(gauss_quadrature_rules)+1
    xx = cartesian_product(
        [gauss_quadrature_rules[ii][0] for ii in range(num_vars-1)])
    ww = outer_product(
        [gauss_quadrature_rules[ii][1] for ii in range(num_vars-1)])
    
    vals = np.zeros_like(zz)
    for ii in range(vals.shape[0]):
        vals[ii] = np.dot(
            ww,pdf(zz[ii]/xx.prod(axis=0))/np.absolute(xx.prod(axis=0)))
    return vals
    

def invert_monotone_function(poly,bounds,zz,method='newton',tol=1e-12):
    """
    because poly is monotone can start each newton iteration from the same
    point at left of domain
    """
    lb,ub=bounds
    zz = np.atleast_1d(zz)
    assert zz.ndim==1
    roots = np.zeros((zz.shape[0]))
    if type(poly)==np.poly1d:
        deriv_poly = poly.deriv()
    else:
        def deriv_poly(x):
            if x<ub-fd_eps:
                return (poly(x+eps)-poly(x))/eps
            elif x>lb+fd_eps:
                return (poly(x)-poly(x-eps))/eps
            
    #fprime2 = poly.deriv(m=2)
    #fprime=deriv_poly
    fprime=None
    fprime2=None
    for ii in range(zz.shape[0]):
        func = lambda x: poly(x)-zz[ii]
        flag1 = (np.isfinite(lb) and np.isfinite(ub) and
                 np.sign(func(lb))==np.sign(func(ub)))
        flag2 = (np.isfinite(lb) and not np.isfinite(ub) and
                 deriv_poly(lb)<0 and func(lb)<0)
        flag3 = (np.isfinite(lb) and not np.isfinite(ub) and
                 deriv_poly(lb)>0 and func(lb)>0)
        flag2 = (np.isfinite(ub) and not np.isfinite(lb) and
                 deriv_poly(ub)<0 and func(ub)<0)
        flag4 = (np.isfinite(ub) and not np.isfinite(lb) and
                 deriv_poly(ub)>0 and func(ub)>0)
        # Warning: If lb and ub are not finite cannot check if polynomial is
        # all above or below root. In this case newton solve will fail
        if (flag1 or flag2 or flag3 or flag4):
            # there is no root in the region defined by bounds
            root=np.nan
        else:
            if not np.isfinite(lb) and not np.isfinite(ub):
                initial_guess=0
            elif not np.isfinite(lb):
                # numerical precision errors can cause newton to move in wrong
                # direction if initial guess is near local extrema so push away
                initial_guess = ub-2*np.finfo(float).eps
            elif not np.isfinite(ub):
                # numerical precision errors can cause newton to move in wrong
                # direction if initial guess is near local extrema so push away
                initial_guess = lb+2*np.finfo(float).eps
            else:
                initial_guess = (lb+ub)/2
            if method=='newton':
                root,result = newton(
                    func,initial_guess,fprime=fprime,fprime2=fprime2,
                    disp=True,maxiter=1000,tol=tol,full_output=True)
            else:
                root,result = bisect(
                    func,lb,ub,disp=True,full_output=True,xtol=tol,rtol=tol)
        roots[ii] = root
    return roots

def get_all_local_extrema_of_monomial_expansion_1d(poly,lb,ub):
    critical_points=poly.deriv().r
    critical_points=critical_points[critical_points.imag==0].real
    critical_points.sort()
    critical_points=critical_points[(critical_points>=lb)&(critical_points<=ub)]
    # due to numerical precision errors sometimes critical points are not unique
    # so remove non-unique values
    critical_points = np.unique(critical_points)
    return critical_points

def get_global_maxima_and_minima_of_monomial_expansion(poly,lb,ub):
    critical_points = get_all_local_extrema_of_monomial_expansion_1d(poly,lb,ub)
    poly_vals_at_critical_points = poly(critical_points)
    I = np.argsort(poly_vals_at_critical_points)
    if not np.all(np.isfinite([lb,ub])):
        msg = 'Cannot find extrema over unbounded interval'
        raise Exception(msg)
    poly_vals_at_bounds = poly([lb,ub])
    poly_min,poly_max = np.sort(poly_vals_at_bounds)
    if I.shape[0]>0:
        poly_min = min(poly_vals_at_critical_points[I[0]],poly_min)
        poly_max = max(poly_vals_at_critical_points[I[-1]],poly_max)
    return poly_min,poly_max

def get_inverse_derivatives(poly,bounds,zz,fd_eps=np.sqrt(np.finfo(float).eps)):
    inverse_vals=invert_monotone_function(poly,bounds,zz)
    indices = np.where(np.isfinite(inverse_vals))[0]
    inverse_derivs = np.array([np.nan]*zz.shape[0])
    if indices.shape[0]==0:
        return inverse_vals, inverse_derivs, indices
    defined_inverse_vals=inverse_vals[indices]
    assert np.allclose(poly(defined_inverse_vals),zz[indices])
    assert np.all(
        (defined_inverse_vals>=bounds[0]-np.finfo(float).eps)&
        (defined_inverse_vals<=bounds[1]+np.finfo(float).eps))
    # for calculating forward finite difference
    zz_perturbed = zz[indices]+fd_eps
    # use backwards difference for last point
    zz_perturbed[-1] = zz_perturbed[-1]-2*fd_eps
    defined_inverse_vals_perturbed = invert_monotone_function(
        poly,bounds,zz_perturbed)
    defined_inverse_derivs=(
        defined_inverse_vals_perturbed-defined_inverse_vals)/fd_eps
    # on upper bound use backwards difference
    defined_inverse_derivs[-1] *= -1
    inverse_derivs[indices] = defined_inverse_derivs
    return inverse_vals,inverse_derivs, indices

def get_pdf_from_monomial_expansion(coef,lb,ub,x_pdf,zz):
    """
    my code assumes monomial coefficients are ordered smallest degree to largest
    scipy assumes reverse ordering
    """
    poly = np.poly1d(coef[::-1])
    critical_points = get_all_local_extrema_of_monomial_expansion_1d(poly,lb,ub)
    # intervals containing monotone regions of polynomial
    intervals = critical_points
    if (len(intervals)==0 or not np.isfinite(intervals[0]) or
        abs(intervals[0]-lb)>1e-15):
        intervals=np.concatenate(([lb],intervals))
    if not np.isfinite(intervals[-1]) or abs(intervals[-1]-ub)>1e-15:
        intervals=np.concatenate((intervals,[ub]))
        
    zz_pdf_vals = np.zeros((zz.shape[0]))
    for jj in range(intervals.shape[0]-1):
        inverse_vals,inverse_derivs,defined_indices = get_inverse_derivatives(
            poly,intervals[jj:jj+2],zz)
        x_pdf_vals = x_pdf(inverse_vals[defined_indices])
        zz_pdf_vals[defined_indices] += x_pdf_vals*np.absolute(
            inverse_derivs[defined_indices])
    return zz_pdf_vals
