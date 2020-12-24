try:
    from _rol_minimize import *
    has_ROL = True
except:
    has_ROL = False

from scipy.optimize import minimize as scipy_minimize


def pyapprox_minimize(fun, x0, args=(), method='rol-trust-constr', jac=None,
                      hess=None, hessp=None, bounds=None, constraints=(),
                      tol=None, callback=None, options={}, x_grad=None):

    options = options.copy()
    if x_grad is not None and 'rol' not in method:
        # Fix this limitation
        msg = f"Method {method} does not currently support gradient checking"
        #raise Exception(msg)
        print(msg)

    if 'rol' in method and has_ROL:
        if callback is not None:
            raise Exception(f'Method {method} cannot use callbacks')
        if args != ():
            raise Exception(f'Method {method} cannot use args')
        rol_methods = {'rol-trust-constr':None}
        if method in rol_methods:
            rol_method = rol_methods[method]
        else:
            raise Exception(f"Method {method} not found")
        return rol_minimize(
            fun, x0, rol_method, jac, hess, hessp, bounds, constraints, tol,
            options, x_grad)
    
    if method == 'trust-constr':
        if 'ctol' in options:
            del options['ctol']
        return scipy_minimize(
            fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol,
            callback, options)
    
    if method == 'slsqp':
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

    raise Exception(f"Method {method} was not found")



if __name__ == '__main__':
    np.seterr(all='raise')
    #test_TR()
    #test_rosenbrock_TR()
    test_rosenbrock_TR_constrained()
