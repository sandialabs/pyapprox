#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
try:
    import ROL
    from ROL.numpy_vector import NumpyVector
    has_ROL = True
except:
    has_ROL = False
from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds, \
    OptimizeResult, BFGS
from scipy.optimize import minimize as scipy_minimize
from functools import partial


class ROLObj(ROL.Objective):
    def __init__(self, fun, jac, hess=None):
        ROL.Objective.__init__(self)
        self.fun = fun
        self.jac = jac
        self.hess = hess

        assert callable(self.jac)
        if hess is not None:
            assert callable(self.hess)

    def value(self, x, tol):
        return self.fun(x.data)

    def gradient(self, g, x, tol):
        grad = self.jac(x.data)
        for ii in range(grad.shape[0]):
            g[ii] = grad[ii]

    def hessVec(self, hv, v, x, tol):
        res = self.hess(x.data).dot(v.data)
        for ii in range(res.shape[0]):
            hv[ii] = res[ii]


class ROLConstraint(ROL.Constraint):
    def __init__(self, fun, jac, hessp=None):
        super().__init__()
        self.fun = fun
        self.jac = jac
        self.hessp = hessp
        
        assert callable(self.jac)
        if self.hessp is not None:
            assert callable(self.hessp)
            self.applyAdjointHessian = self._applyAdjointHessian
            #self.hessp = None

    def value(self, cvec, x, tol):
        vals = self.fun(x.data)
        for ii in range(vals.shape[0]):
            cvec[ii] = vals[ii]

    def applyJacobian(self, jv, v, x, tol):
        res = self.jac(x.data).dot(v.data)
        for ii in range(res.shape[0]):
            jv[ii] = res[ii]
  
    def applyAdjointJacobian(self, jv, v, x, tol):
        res = self.jac(x.data).T.dot(v.data)
        for ii in range(res.shape[0]):
            jv[ii] = res[ii]

    def _applyAdjointHessian(self, ahuv, u, v, x, tol):
        # x : optimization variables
        # u : Lagrange multiplier (size of number of constraints)
        # v : vector (size of x)
        res = self.hessp(x.data, u.data).dot(v.data)
        for ii in range(res.shape[0]):
            ahuv[ii] = res[ii]


def linear_constraint_fun(A, x):
    return A.dot(x)


def linear_constraint_jac(A, x):
    return A


def get_rol_parameters(method, use_bfgs, options):
    paramsDict = {}
    paramsDict["Step"] = {}
    # paramsDict["Step"]["Type"] = 'Fletcher'
    # paramsDict["Step"]["Fletcher"] = {'Regularization Parameter':1e-2}
    
    paramsDict["Step"]["Type"] = method
    paramsDict["Step"]["Trust Region"] = {}
    paramsDict["Step"]["Trust Region"]["Subproblem Solver"] = "Truncated CG"
    paramsDict["Step"]["Augmented Lagrangian"]  = {
        'Initial Optimality Tolerance':1e-1,
        'Initial Feasibility Tolerance':1e-1,
        'Use Default Problem Scaling':True,
        'Print Intermediate Optimization History':(options.get('verbose',0)>2),
        #'Use Default Initial Penalty Parameter':False,
        #'Initial Penalty Parameter':100
    }
    paramsDict["General"] = {'Print Verbosity':0}
    paramsDict["General"]["Secant"] = {}
    if not use_bfgs:
        paramsDict["General"]["Secant"]["Use as Hessian"] = False
    else:
        paramsDict["General"]["Secant"]["Use as Hessian"] = True
        paramsDict["Step"]["Line Search"] = {}
        paramsDict["Step"]["Line Search"]["Descent Method"] = {}
        paramsDict["Step"]["Line Search"]["Descent Method"]["Type"]= \
            "Quasi-Newton Method"
        paramsDict["General"]["Secant"]["Use as Hessian"] = True
        
    paramsDict["Status Test"] = {
            "Gradient Tolerance" : options.get('gtol', 1e-8),
            "Step Tolerance" : options.get('xtol', 1e-14),
            "Constraint Tolerance" : options.get('ctol', 1e-8),
            "Iteration Limit" : options.get("maxiter", 100)}
    return paramsDict


def get_rol_bounds(py_lb, py_ub):
    nvars = len(py_lb)
    if np.all(py_lb == -np.inf) and np.all(py_ub == np.inf):
        raise Exception('Bounds not needed lb and ub are -inf, +inf')
    if np.any(py_lb == np.inf):
        raise Exception('A lower bound was set to +inf')
    if np.any(py_ub == -np.inf):
        raise Exception('An upper bound was set to -inf')
        
    
    lb, ub = NumpyVector(nvars), NumpyVector(nvars)
    for ii in range(nvars):
        lb[ii], ub[ii] = py_lb[ii], py_ub[ii]
        
    if np.all(py_lb == -np.inf):
        return ROL.Bounds(ub, False, 1.0)
    elif np.all(py_ub == np.inf):
        return ROL.Bounds(lb, True, 1.0)

    # avoid overflow warnings created by numpy_vector.py
    I = np.where(py_lb==-np.inf)[0]
    J = np.where(py_ub==np.inf)[0]
    for ii in I:
        lb[ii] = -np.finfo(float).max/100
    for jj in J:
        ub[jj] = np.finfo(float).max/100
    return ROL.Bounds(lb, ub, 1.0)


def get_rol_numpy_vector(np_vec):
    vec = NumpyVector(np_vec.shape[0])
    for ii in range(np_vec.shape[0]):
        vec[ii] = np_vec[ii]
    return vec

def get_constraints(scipy_constraints, scipy_bounds, x0=None):
    bnd, econs, emuls, icons, imuls, ibnds = None, [], [], [], [], []
    if scipy_bounds is not None:
        bnd = get_rol_bounds(scipy_bounds.lb, scipy_bounds.ub)
        
    if len(scipy_constraints) > 0:
        for constr in scipy_constraints:
            if type(constr) == LinearConstraint:
                cfun = partial(linear_constraint_fun, np.asarray(constr.A))
                cjac = partial(linear_constraint_jac, np.asarray(constr.A))
                rol_constr = ROLConstraint(cfun, cjac)
                if type(constr.lb) != np.ndarray:
                    constr.lb = constr.lb*np.ones(len(constr.A))
                    constr.ub = constr.ub*np.ones(len(constr.A))
            elif type(constr) == NonlinearConstraint:
                rol_constr = ROLConstraint(constr.fun, constr.jac, constr.hess)
            else:
                raise Exception('incorrect argument passed as constraint')
            if x0 is not None:
                check_constraint_gradient(constr, rol_constr, x0)
            if np.all(constr.lb == constr.ub):
                econs.append(rol_constr)
                # value of emul does not matter just size
                emuls.append(NumpyVector(len(constr.lb)))
            else:
                assert type(constr.lb) == np.ndarray
                icons.append(rol_constr)
                imuls.append(NumpyVector(len(constr.lb)))
                ibnds.append(get_rol_bounds(constr.lb, constr.ub))
            
    return bnd, econs, emuls, icons, imuls, ibnds


def check_constraint_gradient(constr, rol_constr, x0):
    #x0 = np.random.normal(0,1,(x0.shape))
    print("Testing constraint Jacobian")
    x = get_rol_numpy_vector(x0)
    v = get_rol_numpy_vector(np.random.normal(0, 1, x0.shape[0]))
    jv = get_rol_numpy_vector(np.zeros(len(constr.lb)))
    rol_constr.checkApplyJacobian(x, v, jv, 12, 1)
    w = get_rol_numpy_vector(np.random.normal(0, 1, len(constr.lb)))
    rol_constr.checkAdjointConsistencyJacobian(w, v, x)
    if rol_constr.hessp is not None:
        u = get_rol_numpy_vector(np.random.normal(0, 1, len(constr.lb)))
        hv = get_rol_numpy_vector(np.zeros(len(constr.lb)))
        print("Testing constraint Hessian")
        rol_constr.checkApplyAdjointHessian(x, u, v, hv, 12, 1)
    

def rol_minimize(fun, x0, method=None, jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None,
                 options={}, x_grad=None):
    if method is None:
        method = "Trust Region"
    obj = ROLObj(fun, jac, hess)
    if x_grad is not None:
        print("Testing objective")
        xg = get_rol_numpy_vector(x_grad)
        d = get_rol_numpy_vector(np.random.normal(0, 1, (x_grad.shape[0])))
        obj.checkGradient(xg, d, 12, 1)
        obj.checkHessVec(xg, d, 12, 1)

    use_bfgs = False
    if hess is None:
        use_bfgs = True
    if type(hess) == BFGS:
        use_bfgs = True
    for constr in constraints:
        if type(constr) != LinearConstraint and type(constr.hess) == BFGS:
            use_bfgs = True
            constr.hess = None

    paramsDict = get_rol_parameters(method, use_bfgs, options)
    params = ROL.ParameterList(paramsDict, "Parameters")
    x = NumpyVector(x0.shape[0])
    bnd, econ, emul, icon, imul, ibnd = get_constraints(
        constraints, bounds, x_grad)
    optimProblem = ROL.OptimizationProblem(
        obj, x, bnd=bnd, econs=econ, emuls=emul, icons=icon, imuls=imul,
        ibnds=ibnd)
    solver = ROL.OptimizationSolver(optimProblem, params)
    solver.solve()
    state = solver.getAlgorithmState()
    res = OptimizeResult(
        x=x.data, fun=state.value, cnorm=state.cnorm, gnorm=state.gnorm,
        snorm=state.snorm, success=True)
    print(res)
    return res


def pyapprox_minimize(fun, x0, args=(), method=None, jac=None, hess=None,
                      hessp=None, bounds=None, constraints=(), tol=None,
                      callback=None, options={}, x_grad=None):

    if x_grad is not None and 'rol' not in method:
        # Fix this limitation
        msg = f"Method {method} does not currently support gradient checking"
        raise Exception(msg)

    if 'rol' in method and has_ROL:
        if callback is not None:
            raise Exception(f'Method {method} cannot use callbacks')
        if args != ():
            raise Exception(f'Method {method} cannot use args')
        rol_methods = {'rol_trust_constr':'Trust Region'}
        if method in rol_methods:
            rol_method = rol_methods[method]
        else:
            raise Exception(f"Method {method} not found")
        return rol_minimize(
            fun, x0, rol_method, jac, hess, hessp, bounds, constraints, tol,
            options, x_grad)
    
    if method == 'trust_constr':
        return scipy_minimize(
            x0, args, method, jac, hess, hessp, bounds, constraints, tol,
            callback, options)
    
    if method == 'slsqp':
        return scipy_minimize(
            x0, args, method, jac, hess, hessp, bounds, constraints, tol,
            callback, options)

    raise Exception(f"Method {method} was not found")



def test_TR():
    def fun(x):
        return (x[0] - 1)**2 + x[1]**2

    def jac(x):
        g = np.zeros(2)
        g[0] = 2 * (x[0] - 1)
        g[1] = 2 * x[1]
        return g

    x0 = np.zeros(2)
    x = rol_minimize(fun, x0, jac=jac).x
    assert round(x[0] - 1.0, 6) == 0.0
    assert round(x[1], 6) == 0.0

from scipy.optimize import rosen, rosen_der, rosen_hess
def test_rosenbrock_TR():
    x0 = np.zeros(2)
    x = rol_minimize(rosen, x0, jac=rosen_der, hess=rosen_hess).x
    assert np.allclose(x, np.ones(2))

    
def test_rosenbrock_TR_constrained():
    bounds = Bounds([0, -0.5], [1.0, 2.0])
    
    linear_constraint = LinearConstraint(
        np.array([[1, 2], [2, 1]]), [-np.inf, 1], [1, 1])
    
    def cons_f(x):
        return np.array([x[0]**2 + x[1], x[0]**2 - x[1]])

    def cons_J(x):
        return np.array([[2*x[0], 1], [2*x[0], -1]])

    def cons_H(x, v):
        return v[0]*np.array([[2, 0], [0, 0]]) + \
            v[1]*np.array([[2, 0], [0, 0]])

    nonlinear_constraint = NonlinearConstraint(
        cons_f, -np.inf*np.ones(2), 1*np.ones(2), jac=cons_J, hess=cons_H)

    constraints = [linear_constraint, nonlinear_constraint]
    
    x0 = np.array([0.5, 0])
    x = rol_minimize(
        rosen, x0, jac=rosen_der, hess=rosen_hess, constraints=constraints,
        bounds=bounds, options={'maxiter':10, 'gtol':1e-8, 'ctol':1e-8},
        x_grad = x0).x
    assert np.allclose(x, [0.41494531, 0.17010937], atol=1e-6)
    
if __name__ == '__main__':
    np.seterr(all='raise')
    #test_TR()
    #test_rosenbrock_TR()
    test_rosenbrock_TR_constrained()
