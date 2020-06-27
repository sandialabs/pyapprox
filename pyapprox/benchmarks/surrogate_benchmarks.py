import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess_prod
def rosenbrock_function(samples):
    return rosen(samples)[:,np.newaxis]

def rosenbrock_function_jacobian(samples):
    assert samples.shape[1]==1
    return rosen_der(samples)

def rosenbrock_function_hessian_prod(samples,vec):
    assert samples.shape[1]==1
    return rosen_hess_prod(samples[:,0],vec[:,0])[:,np.newaxis]
