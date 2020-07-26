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

from scipy import stats
from pyapprox.variables import IndependentMultivariateRandomVariable, DesignVariable
def define_beam_random_variables():
    # traditional parameterization
    X = stats.norm(loc=500,scale=np.sqrt(100)**2)
    Y = stats.norm(loc=1000,scale=np.sqrt(100)**2)
    E = stats.norm(loc=2.9e7,scale=np.sqrt(1.45e6)**2)
    R = stats.norm(loc=40000,scale=np.sqrt(2000)**2)

    # increased total variance contribution from E
    X = stats.norm(loc=500,scale=np.sqrt(100)**2/10)
    Y = stats.norm(loc=1000,scale=np.sqrt(100)**2/10)
    E = stats.norm(loc=2.9e7,scale=np.sqrt(1.45e6)**2)
    R = stats.norm(loc=40000,scale=np.sqrt(2000)**2/10)

    from scipy.optimize import Bounds
    design_bounds = Bounds([0,0],[4,4])
    design_variable = DesignVariable(design_bounds)
    
    variable = IndependentMultivariateRandomVariable([X,Y,E,R])
    return variable, design_variable

def cantilever_beam_objective(samples):
    X,Y,E,R,w,t = samples
    values = np.empty((samples.shape[1],1))
    values[:,0] = w*t
    return values

def cantilever_beam_objective_grad(samples):
    X,Y,E,R,w,t = samples
    grad = np.empty((samples.shape[1],2))
    grad[:,0]=t
    grad[:,1]=w
    return grad

def cantilever_beam_constraints(samples):
    values = np.hstack([beam_constraint_I(samples),beam_constraint_II(samples)])
    return values

def cantilever_beam_constraints_jacobian(samples):
    jac = np.vstack(
        [beam_constraint_I_design_jac(samples),
         beam_constraint_II_design_jac(samples)])
    return jac

def beam_constraint_I(samples):
    X,Y,E,R,w,t = samples
    L = 100                  # length of beam
    vals = 1-6*L/(w*t)*(X/w+Y/t)/R # scaled version
    return vals[:,np.newaxis]

def beam_constraint_I_design_jac(samples):
    """
    Jacobian with respect to the design variables
    """
    X,Y,E,R,w,t = samples
    L = 100 
    grad = np.empty((samples.shape[1],2))
    grad[:,0] = (L*(12*t*X + 6*w*Y))/(R*t**2*w**3)
    grad[:,1] = (L*(6*t*X + 12*w*Y))/(R*t**3*w**2)
    return grad

def beam_constraint_II(samples):
    X,Y,E,R,w,t = samples
    L,D0 = 100,2.2535
    # scaled version
    vals = 1-4*L**3/(E*w*t)*np.sqrt(X**2/w**4+Y**2/t**4)/D0
    return vals[:,np.newaxis]

def beam_constraint_II_design_jac(samples):
    """
    Jacobian with respect to the design variables
    """
    X,Y,E,R,w,t = samples
    L,D0 = 100,2.2535
    grad = np.empty((samples.shape[1],2))
    grad[:,0] = (4*L**3*(3*t**4*X**2 + w**4*Y**2))/(D0*t**3*w**4*E*np.sqrt(t**4*X**2 + w**4*Y**2))
    grad[:,1] = (4*L**3*(t**4*X**2 + 3*w**4*Y**2))/(D0*t**4*w**3*E*np.sqrt(t**4*X**2 + w**4*Y**2))
    return grad
