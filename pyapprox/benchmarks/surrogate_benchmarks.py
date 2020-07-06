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

def rosenbrock_function_mean(nvars):
    """
    Mean of rosenbrock function with uniform variables in [-2,2]^d
    """
    assert nvars%2==0
    lb,ub=-2,2
    import sympy as sp
    x,y = sp.Symbol('x'),sp.Symbol('y')
    exact_mean = nvars/2*float(
        sp.integrate(100*(y-x**2)**2+(1-x)**2,(x,lb,ub),(y,lb,ub)))/(4**nvars)
    return exact_mean
