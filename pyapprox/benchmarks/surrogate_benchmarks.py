from scipy import stats
import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess_prod
from scipy import integrate
from numba import njit

from pyapprox.variables import (
    IndependentMultivariateRandomVariable,
    DesignVariable
)

from pyapprox.models.wrappers import (
    evaluate_1darray_function_on_2d_array
)


def rosenbrock_function(samples):
    return rosen(samples)[:, np.newaxis]


def rosenbrock_function_jacobian(samples):
    assert samples.shape[1] == 1
    return rosen_der(samples).T


def rosenbrock_function_hessian_prod(samples, vec):
    assert samples.shape[1] == 1
    return rosen_hess_prod(samples[:, 0], vec[:, 0])[:, np.newaxis]


def rosenbrock_function_mean(nvars):
    """
    Mean of rosenbrock function with uniform variables in [-2,2]^d
    """
    assert nvars % 2 == 0
    lb, ub = -2, 2
    import sympy as sp
    x, y = sp.Symbol('x'), sp.Symbol('y')
    exact_mean = nvars/2*float(
        sp.integrate(
            100*(y-x**2)**2+(1-x)**2, (x, lb, ub), (y, lb, ub)))/(4**nvars)
    return exact_mean


def define_beam_random_variables():
    # traditional parameterization
    X = stats.norm(loc=500, scale=np.sqrt(100)**2)
    Y = stats.norm(loc=1000, scale=np.sqrt(100)**2)
    E = stats.norm(loc=2.9e7, scale=np.sqrt(1.45e6)**2)
    R = stats.norm(loc=40000, scale=np.sqrt(2000)**2)

    # increased total variance contribution from E
    # X = stats.norm(loc=500,scale=np.sqrt(100)**2/10)
    # Y = stats.norm(loc=1000,scale=np.sqrt(100)**2/10)
    # E = stats.norm(loc=2.9e7,scale=np.sqrt(1.45e6)**2)
    # R = stats.norm(loc=40000,scale=np.sqrt(2000)**2/10)

    from scipy.optimize import Bounds
    design_bounds = Bounds([1, 1], [4, 4])
    design_variable = DesignVariable(design_bounds)

    variable = IndependentMultivariateRandomVariable([X, Y, E, R])
    return variable, design_variable


def cantilever_beam_objective(samples):
    X, Y, E, R, w, t = samples
    values = np.empty((samples.shape[1], 1))
    values[:, 0] = w*t
    return values


def cantilever_beam_objective_grad(samples):
    X, Y, E, R, w, t = samples
    grad = np.empty((samples.shape[1], 2))
    grad[:, 0] = t
    grad[:, 1] = w
    return grad


def cantilever_beam_constraints(samples):
    values = np.hstack([beam_constraint_I(samples),
                        beam_constraint_II(samples)])
    return values


def cantilever_beam_constraints_jacobian(samples):
    jac = np.vstack(
        [beam_constraint_I_design_jac(samples),
         beam_constraint_II_design_jac(samples)])
    return jac


def beam_constraint_I(samples):
    """
    Desired behavior is when constraint is less than 0
    """
    X, Y, E, R, w, t = samples
    L = 100                  # length of beam
    vals = 1-6*L/(w*t)*(X/w+Y/t)/R  # scaled version
    return -vals[:, np.newaxis]


def beam_constraint_I_design_jac(samples):
    """
    Jacobian with respect to the design variables
    Desired behavior is when constraint is less than 0
    """
    X, Y, E, R, w, t = samples
    L = 100
    grad = np.empty((samples.shape[1], 2))
    grad[:, 0] = (L*(12*t*X + 6*w*Y))/(R*t**2*w**3)
    grad[:, 1] = (L*(6*t*X + 12*w*Y))/(R*t**3*w**2)
    return -grad


def beam_constraint_II(samples):
    """
    Desired behavior is when constraint is less than 0
    """
    X, Y, E, R, w, t = samples
    L, D0 = 100, 2.2535
    # scaled version
    vals = 1-4*L**3/(E*w*t)*np.sqrt(X**2/w**4+Y**2/t**4)/D0
    return -vals[:, np.newaxis]


def beam_constraint_II_design_jac(samples):
    """
    Jacobian with respect to the design variables
    Desired behavior is when constraint is less than 0
    """
    X, Y, E, R, w, t = samples
    L, D0 = 100, 2.2535
    grad = np.empty((samples.shape[1], 2))
    grad[:, 0] = (4*L**3*(3*t**4*X**2 + w**4*Y**2)) / \
        (D0*t**3*w**4*E*np.sqrt(t**4*X**2 + w**4*Y**2))
    grad[:, 1] = (4*L**3*(t**4*X**2 + 3*w**4*Y**2)) / \
        (D0*t**4*w**3*E*np.sqrt(t**4*X**2 + w**4*Y**2))
    return -grad


def define_piston_random_variables():
    M = stats.uniform(loc=30., scale=30.)
    S = stats.uniform(loc=0.005, scale=0.015)
    V_0 = stats.uniform(loc=0.002, scale=0.008)
    k = stats.uniform(loc=1000., scale=4000.)
    P_0 = stats.uniform(loc=90000., scale=20000.)
    T_a = stats.uniform(loc=290., scale=6.)
    T_0 = stats.uniform(loc=340., scale=20.)

    variable = IndependentMultivariateRandomVariable([M, S, V_0, k,
                                                      P_0, T_a, T_0])
    return variable


def piston_function(samples):
    M, S, V_0, k, P_0, T_a, T_0 = samples

    Z = P_0 * V_0 / T_0 * T_a
    A = P_0 * S + 19.62 * M - k * V_0 / S
    V = S / (2. * k) * (np.sqrt(A**2 + 4. * k * Z) - A)
    C = 2. * np.pi * np.sqrt(M / (k + S**2 * Z / V**2))

    return C[:, None]


def wing_weight_function(samples):
    """
    Weight of a light aircraft wing
    """
    S_w, W_fw, A, Lamda, q, lamda, tc, N_z, W_dg, W_p = samples.copy()
    Lamda *= np.pi/180.
    vals = ((.036*(S_w**.758)*(W_fw**.0035)*(A**.6) *
             (np.cos(Lamda)**-.9)*(q**.006)*(lamda**.04)*(100**-.3) *
             (tc**-.3)*(N_z**.49)*(W_dg**.49))+S_w*W_p)
    return vals[:, None]


def wing_weight_gradient(samples):
    """
    Gradient of weight of a light aircraft wing

    Parameters
    ----------
    samples : np.ndarray (nsamples, nvars)
        The samples at which the gradient is requested

    Returns
    -------
    grads : np.ndarray (nsamples, nvars)
    """
    S_w, W_fw, A, Lamda, q, lamda, tc, N_z, W_dg, W_p = samples.copy()
    Lamda *= np.pi/180.

    vals = wing_weight_function(samples)[:, 0] - S_w*W_p

    nvars, nsamples = samples.shape
    grad = np.empty((nvars, nsamples))
    grad[0] = (.758*vals/S_w + W_p)
    grad[1] = (.0035*vals/W_fw)
    grad[2] = (.6*vals/A)
    grad[3] = (.9*vals*np.sin(Lamda)/np.cos(Lamda))*np.pi/180
    grad[4] = (.006*vals/q)
    grad[5] = (.04*vals/lamda)
    grad[6] = (-.3*vals/tc)
    grad[7] = (.49*vals/N_z)
    grad[8] = (.49*vals/W_dg)
    grad[9] = (S_w)
    return grad.T


def define_wing_weight_random_variables():
    univariate_variables = [
        stats.uniform(150, 50),
        stats.uniform(220, 80),
        stats.uniform(6, 4),
        stats.uniform(-10, 20),
        stats.uniform(16, 29),
        stats.uniform(0.5, 0.5),
        stats.uniform(0.08, 0.1),
        stats.uniform(2.5, 3.5),
        stats.uniform(1700, 800),
        stats.uniform(0.025, 0.055)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    return variable


def get_chemical_reaction_variable_ranges():
    nominal_vars = np.array(
        [1.6, 20.75, 0.04, 1.0, 0.36, 0.016], np.double)
    ranges = np.empty(2*nominal_vars.shape[0])
    ranges[:4] = [0., 4, 5., 35.]
    ranges[4::2] = nominal_vars[2:]*0.9
    ranges[5::2] = nominal_vars[2:]*1.1
    return nominal_vars, ranges


def define_chemical_reaction_random_variables():
    nominal_vars, ranges = get_chemical_reaction_variable_ranges()
    univariate_variables = [
        stats.uniform(ranges[2*ii], ranges[2*ii+1]-ranges[2*ii])
        for ii in range(len(ranges)//2)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    return variable


@njit(cache=True)
def chemical_reaction_rhs(sol, time, params):
    a, b, c, d, e, f = params
    z = 1. - sol[0] - sol[1] - sol[2]
    return (a*z - c*sol[0] - 4*d*sol[0]*sol[1],
            2*b*z**2 - 4*d*sol[0]*sol[1],
            e*z - f*sol[2])


def solve_chemical_reaction_model(z, time):
    y0 = [0., 0., 0.]
    y = integrate.odeint(chemical_reaction_rhs, y0, time,
                         args=(z,))
    return y


class ChemicalReactionModel(object):
    """
    Model of species absorbing onto a surface out of gas phase
    # u = y[0] = monomer species
    # v = y[1] = dimer species
    # w = y[2] = inert species

    Vigil et al., Phys. Rev. E., 1996; Makeev et al., J. Chem. Phys., 2002
    Bert dubescere used this example 2014 talk
    """
    def __init__(self, qoi_functional=None, final_time=100):
        self.qoi_functional = qoi_functional
        self.time = np.arange(0, final_time+0.2, 0.2)
        self.nominal_vars, self.var_ranges = \
            get_chemical_reaction_variable_ranges()
        if self.qoi_functional is None:
            self.qoi_functional = lambda sol: np.array([sol[-1, 0]])

    def value(self, sample):
        assert sample.ndim == 1
        y = solve_chemical_reaction_model(sample, self.time)
        return self.qoi_functional(y)

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.value, samples, None)
