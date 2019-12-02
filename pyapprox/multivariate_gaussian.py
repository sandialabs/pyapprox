from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np

def get_operator_diagonal(operator,num_vars,eval_concurrency,transpose=False,
                          active_indices=None):
    """
    Dont want to solve for all vectors at once because this will 
    likely be to large to fit in memory.

    active indices: np.ndarray 
       only do some entries of diagonal
    """
    if active_indices is None:
        active_indices=np.arange(num_vars)
    else:
        assert active_indices.shape[0]<=num_vars
        
    num_active_indices = active_indices.shape[0]
    
    #diagonal = np.empty((num_vars),float)
    diagonal = np.empty((num_active_indices),float)
    cnt = 0
    #while cnt<num_vars:
    while cnt<num_active_indices:
        #num_vectors = min(eval_concurrency, num_vars-cnt)
        num_vectors = min(eval_concurrency, num_active_indices-cnt)
        vectors = np.zeros((num_vars,num_vectors),dtype=float)
        for j in range(num_vectors):
            #vectors[cnt+j,j]=1.0
            vectors[active_indices[cnt+j],j]=1.0
        tmp = operator(vectors, transpose=transpose)
        for j in range(num_vectors):
            diagonal[cnt+j]=tmp[active_indices[cnt+j],j]
        cnt += num_vectors
    return diagonal


class GaussianSqrtCovarianceOperator(object):
    def apply(self, vectors):
        """
        Apply the sqrt of the covariance to a set of vectors x.
        If the vectors are standard normal samples then the result
        will be a samples from the Gaussian.

        Parameters
        ----------
        vector : np.ndarray (num_vars x num_vectors)
            The vectors x
        """
        raise Exception('Derived classes must implement this function')

    def num_vars(self):
        raise Exception('Derived classes must implement this function')

    def __call__(self, vectors, transpose):
        return self.apply(vectors, transpose)

class CholeskySqrtCovarianceOperator(GaussianSqrtCovarianceOperator):
    def __init__(self, covariance, eval_concurrency=1, fault_percentage=0):
        super(CholeskySqrtCovarianceOperator,self).__init__()

        assert fault_percentage < 100
        self.fault_percentage = fault_percentage

        self.covariance = covariance
        self.chol_factor = np.linalg.cholesky(self.covariance)
        self.eval_concurrency=eval_concurrency

    def apply(self, vectors, transpose):
        """

        Apply the sqrt of the covariance to a st of vectors x.
        If a vector is a standard normal sample then the result
        will be a sample from the prior.

        C_sqrt*x

        Parameters
        ----------
        vectors : np.ndarray (num_vars x num_vectors)
           The vectors x

        transpose : boolean
            Whether to apply the action of the sqrt of the covariance
            or its tranpose.

        Returns
        -------
        result : np.ndarray (num_vars x num_vectors)
            The vectors C_sqrt*x
        """
        assert vectors.ndim==2
        num_vectors = vectors.shape[1]
        if transpose:
            result = np.dot(self.chol_factor.T,vectors)
        else:
            result = np.dot(self.chol_factor,vectors)
        u = np.random.uniform(0.,1.,(num_vectors))
        I = np.where(u<self.fault_percentage/100.)[0]
        result[:,I] = np.nan
        return result

    def num_vars(self):
        """
        Return the number of variables of the multivariate Gaussian
        """
        return self.covariance.shape[0]

class CovarianceOperator(object):
    """
    Class to compute the action of the a covariance opearator on a 
    vector. 

    We can compute the matrix rerpesentation of the operator C = C_op(I)
    where I is the identity matrix. 
    """
    def __init__(self, sqrt_covariance_operator):
        self.sqrt_covariance_operator=sqrt_covariance_operator

    def apply(self, vectors, transpose):
        """
        Compute action of covariance C by applying action of sqrt_covariance L
        twice, where C=L*L.T. E.g.
        C(x) = L(L.T(x))
        Tranpose is ignored because covariance operators are symmetric
        """
        tmp = self.sqrt_covariance_operator.apply(vectors, True)
        result = self.sqrt_covariance_operator.apply(tmp, False)
        return result

    def __call__(self, vectors, transpose):
        """
        Tranpose is ignored because covariance operators are symmetric
        """
        return self.apply(vectors, None)

    def num_rows(self):
         return self.sqrt_covariance_operator.num_vars()

    def num_cols(self):
         return self.num_rows()


class MultivariateGaussian(object):
    def __init__(self,sqrt_covariance_operator,mean=0.):
        self.sqrt_covariance_operator=sqrt_covariance_operator
        if np.isscalar(mean):
            mean = mean*np.ones((self.num_vars()))
        assert mean.ndim==1 and mean.shape[0]==self.num_vars()
        self.mean=mean

    def num_vars(self):
        """
        Return the number of variables of the multivariate Gaussian
        """
        return self.sqrt_covariance_operator.num_vars()

    def apply_covariance_sqrt(self, vectors, transpose):
        return self.sqrt_covariance_operator(vectors, transpose)

    def generate_samples(self,nsamples):
        std_normal_samples = np.random.normal(0.,1.,(self.num_vars(),nsamples))
        samples = self.apply_covariance_sqrt(std_normal_samples,False)
        samples += self.mean[:,np.newaxis]
        return samples

    def pointwise_variance(self,active_indices=None):
        """
        Get the diagonal of the Gaussian covariance matrix.
        Default implementation is two apply the sqrt operator twice.
        """
        covariance_operator=CovarianceOperator(self.sqrt_covariance_operator)
        return get_operator_diagonal(
            covariance_operator,self.num_vars(),
            self.sqrt_covariance_operator.eval_concurrency,
            active_indices=active_indices)
