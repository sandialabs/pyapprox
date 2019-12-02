from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from pyapprox.models.genz import evaluate_1darray_function_on_2d_array
from pyapprox.utilities import get_low_rank_matrix
class RosenbrockLogLikelihoodModel(object):
    def __init__( self ):
        pass

    def loglikelihood_function( self, x ):
        return (1. - x[0])**2 + 100*(x[1]-x[0]**2)**2

    def gradient( self, x ):
        return np.array( [2.*(200.*x[0]**3-200*x[0]*x[1]+x[0]-1.),
                             200.*(x[1]-x[0]**2)] )

    def hessian( self, x ):
        return np.array( [[1200.*x[0]**2-400*x[1]+2.,-400.*x[0]],
                             [-400.*x[0],200.]])

    def value( self, x ):
        assert x.ndim == 1
        vals = self.loglikelihood_function( x )
        return np.array( [vals] )

    def __call__(self, x):
        return evaluate_1darray_function_on_2d_array(self.value,x,{})

class ExponentialQuarticLogLikelihoodModel(object):
    def __init__(self,modify=True):
        self.modify = modify
        self.a = 3.0
        
    def loglikelihood_function(self, x):
        #x[0] = np.cos((x[0]+3)/2); x[1] = np.sin(x[1]/6)
        value = (0.1*x[0]**4 + 0.5*(2.*x[1]-x[0]**2)**2)
        if self.modify:
            value += np.cos((x[0]+x[1]*3)/6)**2
            value *= 3.
            value = value**(1./self.a)+1
        return value

    def gradient(self, x):
        assert x.ndim==1
        grad = np.array([12./5.*x[0]**3-4.*x[0]*x[1],
                            4.*x[1]-2.*x[0]**2])
        if self.modify:
            grad[0] += -3*np.sin(x[0]/3.+x[1])/6.
            grad[1] += -3*np.sin(x[0]/3.+x[1])/2.
            # the following is not tested
            #grad = self.loglikelihood_function(x)**(1./self.a-1.)*grad/self.a
        return grad

    def gradient_set(self,samples):
        assert samples.ndim==2
        num_vars, num_samples = samples.shape
        gradients = np.empty((num_vars,num_samples),dtype=float)
        for i in range(num_samples):
            gradients[:,i] = self.gradient(samples[:,i])
        return gradients

    def hessian(self, x):
        hess = np.array([[36./5.*x[0]**2-4.*x[1],-4.*x[0]],
                           [-4.*x[0],4.]])
        if self.modify:
            hess[0,0]+=-3.*np.cos(x[0]/3.+x[1])/18.
            hess[0,1]+=-3.*np.cos(x[0]/3.+x[1])/6.
            hess[1,0]+=-3.*np.cos(x[0]/3.+x[1])/6.
            hess[1,1]+=-3.*np.cos(x[0]/3.+x[1])/2.
            # the following is not tested
            #val = self.loglikelihood_function(x)
            #grad = self.gradient(x)
            #hess = val**(1./self.a-2.)*(self.a*val*hess-(self.a-1.0)*grad**2)/self.a**2
        return hess

    def value(self, x):
        assert x.ndim == 1
        vals = self.loglikelihood_function(x)
        return np.array([vals])
    
    def __call__(self, x):
        return evaluate_1darray_function_on_2d_array(self.value,x,{})

class QuadraticMisfitModel(object):
    def __init__(self,num_vars,rank,num_qoi,
                 obs=None,noise_covariance=None,Amatrix=None):
        self.num_vars = num_vars
        self.rank=rank
        self.num_qoi=num_qoi
        if Amatrix is None:
            self.Amatrix = get_low_rank_matrix(num_qoi,num_vars,rank)
        else:
            self.Amatrix=Amatrix
            
        if obs is None:
            self.obs = np.zeros(num_qoi)
        else:
            self.obs=obs
        if noise_covariance is None:
            self.noise_covariance = np.eye(num_qoi)
        else:
            self.noise_covariance=noise_covariance
            
        self.noise_covariance_inv = np.linalg.inv(self.noise_covariance)

    def value(self,sample):
        assert sample.ndim==1
        residual = np.dot(self.Amatrix,sample)-self.obs
        return np.asarray(
            [0.5*np.dot(residual.T,np.dot(self.noise_covariance_inv,residual))])

    def gradient(self,sample):
        assert sample.ndim==1
        grad = np.dot(self.Amatrix.T,np.dot(self.noise_covariance_inv,
                      np.dot(self.Amatrix,sample)-self.obs))
        return grad

    def gradient_set(self,samples):
        assert samples.ndim==2
        num_vars, num_samples = samples.shape
        gradients = np.empty((num_vars,num_samples),dtype=float)
        for i in range(num_samples):
            gradients[:,i] = self.gradient(samples[:,i])
        return gradients

    def hessian(self,sample):
        assert sample.ndim==1 or sample.shape[1]==1
        return np.dot(
            np.dot(self.Amatrix.T,self.noise_covariance_inv),self.Amatrix)

    def __call__(self,samples,opts=dict()):
        eval_type=opts.get('eval_type','value')
        if eval_type=='value':
            return evaluate_1darray_function_on_2d_array(
                self.value,samples,opts)
        elif eval_type=='value_grad':
            vals = evaluate_1darray_function_on_2d_array(
                self.value,samples,opts)
            return np.hstack((vals,self.gradient_set(samples).T))
        elif eval_type=='grad':
            return self.gradient_set(samples).T
        else:
            raise Exception('%s is not a valid eval_type'%eval_type)
        
class LogUnormalizedPosterior(object):

    def __init__(self, misfit, misfit_gradient, prior_pdf, prior_log_pdf,
                 prior_log_pdf_gradient):
        """
        Initialize the object.

        Parameters
        ----------
        """
        self.misfit          = misfit
        self.misfit_gradient = misfit_gradient
        self.prior_pdf       = prior_pdf
        self.prior_log_pdf   = prior_log_pdf
        self.prior_log_pdf_gradient   = prior_log_pdf_gradient

    def gradient(self,samples):
        """
        Evaluate the gradient of the logarithm of the unnormalized posterior
           likelihood(x)*posterior(x)
        at a sample x

        Parameters
        ----------
        samples : (num_vars,num_samples) vector
            The location at which to evalute the unnormalized posterior

        Returns
        -------
        val : (1x1) vector
            The logarithm of the unnormalized posterior
        """
        if samples.ndim==1:
            samples=samples[:,np.newaxis]
        grad = -self.misfit_gradient(samples) + \
               self.prior_log_pdf_gradient(samples)
        return grad

    def __call__(self,samples,opts=dict()):
        """
        Evaluate the logarithm of the unnormalized posterior
           likelihood(x)*posterior(x)
        at samples x

        Parameters
        ----------
        sampels : np.ndarray (num_vars, num_samples)
            The samples at which to evalute the unnormalized posterior

        Returns
        -------
        values : np.ndarray (num_samples,1)
            The logarithm of the unnormalized posterior
        """
        if samples.ndim==1:
            samples=samples[:,np.newaxis]

        eval_type = opts.get('eval_type','value')
        if eval_type=='value':
            values = -self.misfit(samples)+self.prior_log_pdf(samples)
            assert values.ndim==2
            
        elif eval_type=='grad':
            values = self.gradient(samples).T
            
        elif eval_type=='value_grad':
            values = -self.misfit(samples)+self.prior.log_pdf(samples)
            grad = self.gradient(samples)
            values = np.hstack((values,grad))
        else:
            raise Exception()
            
        return values
