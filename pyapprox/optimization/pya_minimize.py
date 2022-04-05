import numpy as np
from scipy.optimize import minimize as scipy_minimize

from pyapprox.util.sys_utilities import package_available
if package_available("ROL"):
    has_ROL = True
    from pyapprox.optimization._rol_minimize import rol_minimize
else:
    has_ROL = False


def pyapprox_minimize(fun, x0, args=(), method='rol-trust-constr', jac=None,
                      hess=None, hessp=None, bounds=None, constraints=(),
                      tol=None, callback=None, options={}, x_grad=None):
    options = options.copy()
    if x_grad is not None and 'rol' not in method:
        # Fix this limitation
        msg = f"Method {method} does not currently support gradient checking"
        # raise Exception(msg)
        print(msg)

    if 'rol' in method and has_ROL:
        if callback is not None:
            raise ValueError(f'Method {method} cannot use callbacks')
        if args != ():
            raise ValueError(f'Method {method} cannot use args')
        rol_methods = {'rol-trust-constr': None}
        if method in rol_methods:
            rol_method = rol_methods[method]
        else:
            raise ValueError(f"Method {method} not found")
        return rol_minimize(
            fun, x0, rol_method, jac, hess, hessp, bounds, constraints, tol,
            options, x_grad)

    x0 = x0.squeeze()  # scipy only takes 1D np.ndarrays
    x0 = np.atleast_1d(x0)  # change scalars to np.ndarrays
    assert x0.ndim <= 1
    if method == 'rol-trust-constr' and not has_ROL:
        print('ROL requested by not available switching to scipy.minimize')
        method = 'trust-constr'
    
    if method == 'trust-constr':
        if 'ctol' in options:
            del options['ctol']
        return scipy_minimize(
            fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol,
            callback, options)
    elif method == 'slsqp':
        hess, hessp = None, None
        if 'ctol' in options:
            del options['ctol']
        if 'gtol' in options:
            ftol = options['gtol']
            del options['gtol']
        options['ftol'] = ftol
        if 'verbose' in options:
            verbose = options['verbose']
            options['disp'] = verbose
            del options['verbose']
        return scipy_minimize(
            fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol,
            callback, options)

    raise ValueError(f"Method {method} was not found")
