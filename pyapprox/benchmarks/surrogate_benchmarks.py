from scipy import stats
import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess_prod
from scipy import integrate

from pyapprox.util.pya_numba import njit
from pyapprox.variables.joint import (
    IndependentMarginalsVariable, DesignVariable
)
from pyapprox.interface.wrappers import (
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

    variable = IndependentMarginalsVariable([X, Y, E, R])
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
    grad[:, 0] = (L*(12*t*X+6*w*Y))/(R*t**2*w**3)
    grad[:, 1] = (L*(6*t*X+12*w*Y))/(R*t**3*w**2)
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
    grad[:, 0] = (4*L**3*(3*t**4*X**2+w**4*Y**2)) /\
        (D0*t**3*w**4*E*np.sqrt(t**4*X**2+w**4*Y**2))
    grad[:, 1] = (4*L**3*(t**4*X**2+3*w**4*Y**2)) /\
        (D0*t**4*w**3*E*np.sqrt(t**4*X**2+w**4*Y**2))
    return -grad


def define_piston_random_variables():
    """
    M piston mass (kg)
    S surface area (m^2)
    V_0 initial gas volume (m^3)
    k spring coefficient (N/m)
    P_0 atmospheric pressure (N/m^2)
    T_a ambient temperature (K)
    T_0 filling gas temperature (K)
    """
    M = stats.uniform(loc=30., scale=30.)
    S = stats.uniform(loc=0.005, scale=0.015)
    V_0 = stats.uniform(loc=0.002, scale=0.008)
    k = stats.uniform(loc=1000., scale=4000.)
    P_0 = stats.uniform(loc=90000., scale=20000.)
    T_a = stats.uniform(loc=290., scale=6.)
    T_0 = stats.uniform(loc=340., scale=20.)

    variable = IndependentMarginalsVariable([M, S, V_0, k,
                                                      P_0, T_a, T_0])
    return variable


def piston_function(samples):
    """Predict cycle time of a piston in a cylinder
    """
    M, S, V_0, k, P_0, T_a, T_0 = samples

    Z = P_0*V_0/T_0*T_a
    A = P_0*S+19.62*M - k*V_0/S
    V = S/(2.*k)*(np.sqrt(A**2+4.*k*Z) - A)
    C = 2.*np.pi*np.sqrt(M/(k+S**2*Z/V**2))

    return C[:, None]


def piston_function_gradient(samples):
    """Gradient of cycle time of a piston in a cylinder
    """
    M, S, V_0, k, P_0, T_a, T_0 = samples

    Z = P_0*V_0/T_0*T_a
    A = P_0*S+19.62*M - k*V_0/S
    V = S/(2.*k)*(np.sqrt(A**2+4.*k*Z) - A)

    tmp0 = S**2*P_0*V_0
    tmp1 = k+tmp0*T_a/(T_0*V**2)
    tmp2 = (A**2+4*k*P_0*V_0*T_a/T_0)**(-.5)
    tmp3 = np.pi*M**.5*tmp1**(-1.5)

    grad_M = (np.pi*(M*tmp1)**(-.5)+2*tmp3*S**3*P_0*V_0*T_a /
              (2*k*T_0*V**3)*(tmp2*A*19.62-19.62))
    grad_S = (-tmp3*(2*S*P_0*V_0*T_a/(T_0*V**2)-2*tmp0*T_a/(T_0*V**3) *
                     (V/S+S/(2*k)*(tmp2*A*(P_0+k*V_0/S**2)-P_0-k*V_0/S**2))))
    grad_V_0 = (-tmp3*(S**2*P_0*T_a/(T_0*V**2)-2*S**3*P_0*V_0*T_a /
                       (2*k*T_0*V**3)*(tmp2/2*(4*k*P_0*T_a/T_0-2*A*k/S)+k/S)))
    grad_k = (-tmp3*(1-2*tmp0*T_a/(T_0*V**3) *
                     (-V/k+S/(2*k)*(tmp2/2*(4*P_0*V_0*T_a/T_0-2*A*V_0/S) +
                                    V_0/S))))
    grad_P_0 = (-tmp3*(S**2*V_0*T_a/(T_0*V**2)-2*S**3*P_0*V_0*T_a /
                       (2*k*T_0*V**3)*(tmp2/2*(4*k*V_0*T_a/T_0+2*A*S)-S)))
    grad_T_a = (-tmp3*(tmp0/(T_0*V**2)-2*S**3*P_0*V_0*T_a/(2*k*T_0*V**3) *
                       (tmp2/2*4*k*P_0*V_0/T_0)))
    grad_T_0 = (tmp3*(tmp0*T_a/(T_0**2*V**2)+2*S**3*P_0*V_0*T_a/(2*k*T_0*V**3)
                      * (-tmp2/2*P_0*4*k*V_0*T_a/T_0**2)))
    return np.vstack((grad_M, grad_S, grad_V_0, grad_k, grad_P_0, grad_T_a,
                      grad_T_0)).T


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

    vals = wing_weight_function(samples)[:, 0]-S_w*W_p

    nvars, nsamples = samples.shape
    grad = np.empty((nvars, nsamples))
    grad[0] = (.758*vals/S_w+W_p)
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
    variable = IndependentMarginalsVariable(univariate_variables)
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
    variable = IndependentMarginalsVariable(univariate_variables)
    return variable


@njit(cache=True)
def chemical_reaction_rhs(sol, time, params):
    a, b, c, d, e, f = params
    z = 1.-sol[0]-sol[1]-sol[2]
    return (a*z-c*sol[0]-4*d*sol[0]*sol[1],
            2*b*z**2-4*d*sol[0]*sol[1],
            e*z-f*sol[2])


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


def get_random_oscillator_nominal_values():
    return np.array([0.1, 0.035, 0.1, 1.0, 0.5, 0.],
                    np.double)


def define_random_oscillator_random_variables():
    ranges = np.array([0.08, 0.12, 0.03, 0.04, 0.08, 0.12, 0.8, 1.2,
                       0.45, 0.55, -0.05, 0.05], np.double)
    univariate_variables = [
        stats.uniform(ranges[2*ii], ranges[2*ii+1]-ranges[2*ii])
        for ii in range(len(ranges)//2)]
    variable = IndependentMarginalsVariable(univariate_variables)
    return variable


@njit(cache=True)
def random_oscillator_rhs(y, t, z):
    return (y[1], -z[0]*y[1]-z[1]*y[0]+z[2]*np.sin(z[3]*t))


class RandomOscillator(object):
    def __init__(self, qoi_functional=None):
        self.num_vars = 6
        self.t = np.arange(0, 20, 0.1)
        self.name = 'random-oscillator-6'
        if qoi_functional is None:
            self.qoi_functional = lambda sol: np.array([sol[-1]])
        else:
            self.qoi_functional = qoi_functional

    def run(self, z):
        # return self.analytical_solution(z, self.t)
        return self.numerical_solution(z)[:, 0]

    def analytical_solution(self, z, t):
        # only computes position and not velocity

        # x''+b*x'+k*x = F*sin(w*t)
        b, k, F, w, x0, v0 = z

        g = b/2.

        zeta = np.sqrt((b*w)**2+(k-w**2)**2)
        phi = np.arctan(-(b*w)/(k-w**2))
        if (k-w**2)/zeta**2 < 0:
            phi += np.pi

        # Steady state solution (ys) for rhs = 0
        ys = F*np.sin(w*t+phi)/zeta

        # Compute transient (yt) component of solution
        B1 = -F*(b*w)/zeta**2
        B2 = F*(k-w**2)/zeta**2

        if np.sqrt(k) > g:
            # Under damped
            wd = np.sqrt(k)*np.sqrt(1.-g**2/k)
            # No forcing
            # A1 = x0
            # A2 = (g*x0+v0)/wd
            # Forcing
            A1 = x0-B1
            A2 = (v0+g*A1-w*B2)/wd
            yt = np.exp(-g*t)*(A1*np.cos(wd*t)+A2*np.sin(wd*t))

        elif np.sqrt(k) < g:
            # Over damped
            l1 = np.sqrt(k)*(-g/np.sqrt(k)+np.sqrt(g**2/k-1.))
            l2 = np.sqrt(k)*(-g/np.sqrt(k)-np.sqrt(g**2/k-1.))
            if np.isnan(l1):
                raise RuntimeError(f"{z}")
            # No forcing
            # A1 = x0+(l1*x0-v0)/(l2-l1)
            #  A2 = -(l1*x0-v0)/(l2-l1)
            # Forcing
            A2 = (v0-l1*(x0+B1)-w*B2)/(l2-l1)
            A1 = x0-A2-B1
            yt = A1*np.exp(l1*t)+A2*np.exp(l2*t)

        else:
            # np.sqrt(k) = g
            # Critically damped
            # No forcing
            # A1 = x0
            # A2 = b/2.*x0+v0
            # With forcing
            A1 = x0-B1
            A2 = b/2.*A1-w*B2+v0
            yt = np.exp(-g*t)*(A1+A2*t)
        return yt+ys

    def numerical_solution(self, z):
        # Computes position and velocity
        assert z.ndim == 1
        y0 = [z[4], z[5]]
        return integrate.odeint(random_oscillator_rhs, y0, self.t, args=(z,))

    def value(self, z):
        assert z.ndim == 1
        return self.qoi_functional(
            self.analytical_solution(z, self.t))

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.value, samples, None)


def get_coupled_springs_nominal_values():
    return np.array([1., 1.5, 8., 40., 0.5, 1.0, 0.8, 0.5,
                     0.5, 0.0, 2.25, 0.0], np.double)


def define_coupled_springs_random_variables():
    ranges = np.array([0.9, 1.1,  1.4, 1.6,  7., 9.,  39., 41.,
                       0.4, 0.6,  0.9, 1.1,  0.7, 0.9,  0.4, 0.6,
                       0.4, 0.6,  -0.1, 0.1,  2.2, 2.3,  -0.1, 0.1],
                      np.double)
    univariate_variables = [
        stats.uniform(ranges[2*ii], ranges[2*ii+1]-ranges[2*ii])
        for ii in range(len(ranges)//2)]
    variable = IndependentMarginalsVariable(univariate_variables)
    return variable


@njit(cache=True)
def coupled_springs_rhs(y, t, z):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
    w :  vector of the state variables:
    w = [x1, y1, x2, y2]
    t :  time
    p :  vector of the parameters:
    p = [m1, m2, k1, k2, L1, L2, b1, b2

    m1 and m2 are the masses
    k1 and k2 are the spring constants
    L1 and L2 are the natural lengths
    b1 abd b2 are the friction coefficients
    x1 and x2 are the initial displacements;
    y1 and y2 are the initial velocities
    """
    x1, y1, x2, y2 = y
    m1, m2, k1, k2, L1, L2, b1, b2 = z[:8]
    return [y1, (-b1*y1-k1*(x1-L1)+k2*(x2-x1-L2))/m1,
            y2, (-b2*y2-k2*(x2-x1-L2))/m2]


class CoupledSprings(object):
    """
    Two objects with masses m1 and m2 are coupled through springs with
    spring constants k1 and k2. The left end of the left spring is fixed.
    We assume that the lengths of the springs, when subjected to no external
    forces, are L1 and L2.

    The masses are sliding on a surface that creates friction,
    so there are two friction coefficients, b1 and b2.
    http://www.scipy.org/Cookbook/CoupledSpringMassSystem
    """

    def __init__(self, qoi_functional=None):
        self.num_vars = 12
        self.t = np.arange(0., 10., .1)
        self.name = 'coupled-springs-12'
        if qoi_functional is None:
            self.qoi_functional = lambda sol: np.array([sol[-1, 0]])
        else:
            self.qoi_functional = qoi_functional

    def run(self, z):
        assert z.ndim == 1
        abserr = 1.0e-8
        relerr = 1.0e-6
        y0 = [z[8], z[9], z[10], z[11]]
        return integrate.odeint(
            coupled_springs_rhs, y0, self.t, args=(z,),
            atol=abserr, rtol=relerr)[:, ::2]

    def value(self, z):
        assert z.ndim == 1
        sol = self.run(z)
        return self.qoi_functional(sol)

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.value, samples, None)


def get_nondim_hastings_ecology_nominal_values():
    # return np.array([5.0, 4.1, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0],
    #                 np.double)
    return np.array([5.0, 3, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0],
                    np.double)


def define_nondim_hastings_ecology_random_variables():
    nominal_sample = get_nondim_hastings_ecology_nominal_values()
    ranges = np.zeros((2*len(nominal_sample)), np.double)
    ranges[::2] = nominal_sample*0.95
    ranges[1::2] = nominal_sample*1.05
    # ranges[:2] = 4.9, 5.1
    # ranges[12:14] = 0, 1
    # ranges[14:16] = 0, 1
    # ranges[16:18] = 5, 12
    univariate_variables = [
        stats.uniform(ranges[2*ii], ranges[2*ii+1]-ranges[2*ii])
        for ii in range(len(ranges)//2)]
    variable = IndependentMarginalsVariable(univariate_variables)
    return variable


@njit(cache=True)
def nondim_hastings_ecology_rhs(y, t, z):
    """
    """
    y1, y2, y3 = y
    a1, b1, a2, b2, d1, d2 = z[:6]
    return [y1*(1-y1)-a1*y1*y2/(1+b1*y1),
            a1*y1*y2/(1.0 + b1*y1) - a2*y2*y3/(1.0 + b2*y2) - d1*y2,
            a2*y2*y3/(1.0 + b2*y2) - d2*y3]


def non_dimensionalize_hastings_ecology_variables(z):
    R0, K0, C1, C2, D1, D2, A1, A2, B1, B2 = z
    a1 = K0*A1/(R0*B1)
    b1 = K0/B1
    a2 = C2*A2*K0/(C1*R0*B2)
    b2 = K0/(C1*B2)
    d1 = D1/R0
    d2 = D2/R0
    return np.array([a1, b1, a2, b2, d1, d2])


@njit(cache=True)
def dim_hastings_ecology_rhs(y, t, z):
    """
    """
    y1, y2, y3 = y
    nondim_z = non_dimensionalize_hastings_ecology_variables(z[:10])
    nondim_z = np.vstack((nondim_z, z[10:]))
    return nondim_hastings_ecology_rhs(y, t, z)


class HastingsEcology(object):
    """
    http://www.jstor.org/stable/1940591

    Original model

    dY1/dT = R_0 Y_1(1-Y_1/K_0)-C_1 F_1(Y_1) Y_2
    dY1/dT = F_1(Y_1) Y_2 - F_2(Y_2) Y_3 - D_1 Y_2
    dY1/dT = C_2 F_2(Y_2) Y_3 - D_2 Y_3

    F_i(U) = A_iU/(B_i+U) i=1,2

    T is time
    R0 intrinsic growth rate
    K0 carry capacity
    C1 conversion rate to prey for species y2
    C2 conversion rate to prey for species y3
    D1 constant death rate for species y2
    D2 constant death rate for species y3
    A1 saturating functional response for f1
    A2 saturating functional response for f2
    B1 prey population level where the predator rate per unit prey
       is half its maximum value for f1
    B1 prey population level where the predator rate per unit prey
       is half its maximum value for f2

    Model is non-dimensionalized
    a_1 = K_0 A_1/(R_0 B_1)
    b_1 = K_0/B_1
    a_2 = C_2 A_2 K_0/(C_1 R_0 B_2)
    b_2 = K_0/(C_1 B_2)
    d_1 = D_1/R_0
    d_2 = D_2/R_0
    """

    def __init__(self, qoi_functional=None, nondim=True, time=None):
        self.num_vars = 9
        self.opts = {"rtol": 1e-10, "atol": 1e-10}
        if time is None:
            self.t = np.linspace(0., 100, 101)
        else:
            self.t = time
        self.nondim = nondim
        if self.nondim:
            self.run = self.nondim_run
        else:
            self.run = self.dim_run
        if qoi_functional is None:
            self.qoi_functional = lambda sol: np.array(
                [sol[-1, 2]])
        else:
            self.qoi_functional = qoi_functional
        self.name = 'hastings-ecology-9'

    def nondim_run(self, z):
        assert z.ndim == 1
        y0 = z[6:]
        return integrate.odeint(
            nondim_hastings_ecology_rhs, y0, self.t, args=(z,), **self.opts)

    def dim_run(self, z):
        assert z.ndim == 1
        y0 = z[10:]
        return integrate.odeint(
            nondim_hastings_ecology_rhs, y0, self.t, args=(z,), **self.opts)

    def value(self, z):
        assert z.ndim == 1
        sol = self.run(z)
        return self.qoi_functional(sol)

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.value, samples, None)


class ParameterizedNonlinearModel(object):
    def __init__(self):
        self.qoi = 1
        self.ranges = np.array(
            [0.79, 0.99, 1-4.5*np.sqrt(0.1), 1+4.5*np.sqrt(0.1)],
            np.double)

    def num_qoi(self):
        if np.isscalar(self.qoi):
            return 1
        else:
            return len(self.qoi)

    def evaluate(self, samples):
        assert samples.ndim == 1
        assert samples.ndim == 1

        sol = np.ones((2), float)

        x1 = samples[0]
        x2 = samples[1]
        u1 = sol[0]
        u2 = sol[1]

        res1 = 1.-(x1*u1*u1+u2*u2)
        res2 = 1.-(u1*u1-x2*u2*u2)

        norm_res = np.sqrt(res1*res1 + res2*res2)

        it = 0
        max_iters = 20
        while (norm_res > 1e-10) and (it < max_iters):
            det = -4*u1*u2*(x1*x2+1.0)
            j11i = -2.0*x2*u2 / det
            j12i = -2.0*u2 / det
            j21i = -2.0*u1 / det
            j22i = 2*x1*u1 / det

            du1 = j11i*res1 + j12i*res2
            du2 = j21i*res1 + j22i*res2

            u1 += du1
            u2 += du2

            res1 = 1.-(x1*u1*u1+u2*u2)
            res2 = 1.-(u1*u1-x2*u2*u2)

            norm_res = np.sqrt(res1*res1 + res2*res2)
            it += 1

        sol[0] = u1
        sol[1] = u2

        if np.isscalar(self.qoi):
            values = np.array([sol[self.qoi]])
        else:
            values = sol[self.qoi]
        return values

    def __call__(self, samples):
        num_samples = samples.shape[1]
        values = np.empty((num_samples, self.num_qoi()), float)
        for i in range(samples.shape[1]):
            values[i, :] = self.evaluate(samples[:, i])
        return values
