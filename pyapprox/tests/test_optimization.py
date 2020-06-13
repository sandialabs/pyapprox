import unittest
from pyapprox.optimization import *

class TestOptimization(unittest.TestCase):
    def test_approx_jacobian(self):
        constraint_function = lambda x: np.array([1-x[0]**2-x[1]])
        constraint_grad = lambda x: [-2*x[0],-1]

        x0 = np.random.uniform(0,1,(2,1))
        true_jacobian = constraint_grad(x0[:,0])
        assert np.allclose(true_jacobian,approx_jacobian(constraint_function,x0[:,0]))

        constraint_function = lambda x: np.array(
            [1 - x[0] - 2*x[1],1 - x[0]**2 - x[1],1 - x[0]**2 + x[1]])
        constraint_grad = lambda x: np.array([[-1.0, -2.0],
                                              [-2*x[0], -1.0],
                                              [-2*x[0], 1.0]])

        
        x0 = np.random.uniform(0,1,(2,1))
        true_jacobian = constraint_grad(x0[:,0])
        assert np.allclose(true_jacobian,approx_jacobian(constraint_function,x0[:,0]))

    def test_eval_mc_based_jacobian_at_multiple_design_samples(self):
        constraint_function_single = lambda z,x: np.array([z[0]*(1-x[0]**2-x[1])])
        constraint_grad_single = lambda z,x: [-2*z[0]*x[0],-z[0]]

        x0 = np.random.uniform(0,1,(2,2))
        zz = np.arange(0,6,2)[np.newaxis,:]

        vals = eval_function_at_multiple_design_and_random_samples(
            constraint_function_single,zz,x0)

        stat_func = lambda vals: np.mean(vals,axis=0)
        jacobian = eval_mc_based_jacobian_at_multiple_design_samples(
            constraint_grad_single,stat_func,zz,x0)

        true_jacobian=[np.mean([constraint_grad_single(z,x) for z in zz.T],axis=0)
                       for x in x0.T]
        assert np.allclose(true_jacobian,jacobian)

        constraint_function_single = lambda z,x: z[0]*np.array(
            [1 - x[0] - 2*x[1],1 - x[0]**2 - x[1],1 - x[0]**2 + x[1]])
        constraint_grad_single = lambda z,x: z[0]*np.array([[-1.0, -2.0],
                                              [-2*x[0], -1.0],
                                              [-2*x[0], 1.0]])

        x0 = np.random.uniform(0,1,(2,2))
        zz = np.arange(0,6,2)[np.newaxis,:]

        vals = eval_function_at_multiple_design_and_random_samples(
            constraint_function_single,zz,x0)

        stat_func = lambda vals: np.mean(vals,axis=0)
        jacobian = eval_mc_based_jacobian_at_multiple_design_samples(
            constraint_grad_single,stat_func,zz,x0)
        
        true_jacobian=[np.mean([constraint_grad_single(z,x) for z in zz.T],axis=0)
                       for x in x0.T]
        assert np.allclose(true_jacobian,jacobian)


        
        # lower_bound=0
        # nsamples = 100
        # uq_samples = np.random.uniform(0,1,(2,nsamples))
        # func = partial(mean_lower_bound_constraint,constraint_function,lower_bound,uq_samples)
        # grad = partial(mean_lower_bound_constraint_jacobian,constraint_grad,uq_samples)

    def test_basis_pursuit(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 6, 7, 2 
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        options=None
        coef = basis_pursuit(basis_matrix,vals,options)
        print(coef)
        assert np.allclose(coef,true_coef)

    def test_least_squares(self):
        """for tutorial purposes. Perhaps move to a tutorial"""
        np.random.seed(1)
        tol=1e-14
        nsamples, degree, sparsity = 100, 7, 2 
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        def objective(x,return_grad=True):
            residual = basis_matrix.dot(x)-vals
            obj = 0.5*residual.dot(residual)
            grad = basis_matrix.T.dot(residual)
            if return_grad:
                return obj, grad
            return obj

        def hessian(x):
            return basis_matrix.T.dot(basis_matrix)
            
        lstsq_coef = np.linalg.lstsq(basis_matrix,vals,rcond=0)[0]

        init_guess = np.random.normal(0,0.1,(true_coef.shape[0]))
        #init_guess = lstsq_coef

        errors = check_gradients(objective,True,init_guess,disp=True)
        assert errors.min()<2e-7

        method = 'trust-constr'
        func = partial(objective,return_grad=True); jac=True; hess=hessian; 
        options = {'gtol':tol,'verbose':0,'disp':True,'xtol':tol,'maxiter':1000}
        res = minimize(
            func, init_guess, method=method, jac=jac, hess=hess,options=options)

        #print(lstsq_coef)
        print(res.x,true_coef)
        assert np.allclose(res.x,true_coef,atol=1e-4)
        

    def test_nonlinear_basis_pursuit_with_linear_model(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 100, 7, 2 
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        def func(x):
            return basis_matrix.dot(x)-vals, basis_matrix
        jac=True
        def hess(x):
            return sp.lil_matrix((x.shape[0],x.shape[0]),dtype=float)

        tol=1e-12
        options = {'gtol':tol,'verbose':2,'disp':True,'xtol':tol,'maxiter':10000}
        init_guess = true_coef+np.random.normal(0,1e-5,true_coef.shape[0])
        #fd_jac = approx_jacobian(lambda x: func(x)[0],init_guess,epsilon=1e-7)
        #exact_jac = func(init_guess)[1]
        l1_coef = nonlinear_basis_pursuit(func,jac,hess,init_guess,options)
        #print(l1_coef)
        assert np.allclose(l1_coef,true_coef)

    def test_nonlinear_basis_pursuit(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 7, 7, 2 
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        def model(x):
            val = basis_matrix.dot(x[:-1])*np.exp(samples[0,:]*x[-1])
            grad = np.hstack(
                [basis_matrix*np.exp(samples[0,:]*x[-1])[:,np.newaxis],
                 (samples[0,:]*val)[:,np.newaxis]])
            return val, grad

        true_coef = np.zeros(basis_matrix.shape[1]+1)
        true_coef[np.random.permutation(true_coef.shape[0]-1)[:sparsity-1]]=1.
        true_coef[-1]=1
        vals = model(true_coef)[0]

        def func(x):
            model_vals,grad = model(x)
            return model_vals-vals, grad
        jac=True
        hess = None


        init_guess = true_coef+np.random.normal(0,1,true_coef.shape[0])
        fd_jac = approx_jacobian(lambda x: model(x)[0],init_guess,epsilon=1e-7)
        analytical_jac = model(init_guess)[1]
        #print(analytical_jac-fd_jac)
        assert np.allclose(analytical_jac,fd_jac,atol=1e-8)

        tol=1e-12
        options = {'gtol':tol,'verbose':2,'disp':True,'xtol':tol,'maxiter':1000}
        init_guess = true_coef+np.random.normal(0,1,true_coef.shape[0])
        l1_coef = nonlinear_basis_pursuit(func,jac,hess,init_guess,options)
        #print(true_coef,l1_coef)
        assert np.allclose(l1_coef,true_coef,atol=2e-4)

    def test_basis_pursuit_smooth_l1_norm(self):
        np.random.seed(1)
        x = np.linspace(-1,1,101)
        t = np.zeros_like(x)
        t = np.ones_like(x)
        r = 10
        #plt.plot(x,kouri_smooth_absolute_value(t,r,x))
        #plt.show()

        t = np.ones(5);r=1
        init_guess = np.random.normal(0,1,(t.shape[0]))
        func = partial(kouri_smooth_l1_norm,t,r)
        jac  = partial(kouri_smooth_l1_norm_gradient,t,r)
        errors = check_gradients(func,jac,init_guess,disp=True)
        assert errors.min()<3e-7

        fd_hess = approx_jacobian(jac,init_guess)
        assert np.allclose(fd_hess,kouri_smooth_l1_norm_hessian(t,r,init_guess))

        nsamples, degree, sparsity = 6, 7, 2
        #nsamples, degree, sparsity = 15, 20, 3
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        def func(x):
            return basis_matrix.dot(x)-vals, basis_matrix
        jac=True
        def hess(x):
            return sp.lil_matrix((x.shape[0],x.shape[0]),dtype=float)

        tol=1e-10
        eps=1e-6
        #init_guess = np.random.normal(0,1,(true_coef.shape[0]))
        init_guess = true_coef
        options = {'gtol':tol,'verbose':2,'disp':True,'xtol':tol,'maxiter':1000}
        homotopy_options = {'gtol':tol,'verbose':2,'disp':True,'xtol':1e-10,
                            'maxiter':100}
        res = basis_pursuit_denoising(
            func,jac,hess,init_guess,eps,options,homotopy_options)
        coef =  res.x

        print(true_coef,coef)
        assert np.allclose(true_coef,coef,atol=1e-7)

    def test_basis_pursuit_denoising_smooth_l1_norm(self):
        np.random.seed(1)
        x = np.linspace(-1,1,101)
        t = np.zeros_like(x)
        t = np.ones_like(x)
        r = 10
        #plt.plot(x,kouri_smooth_absolute_value(t,r,x))
        #plt.show()

        t = np.ones(5);r=1
        init_guess = np.random.normal(0,1,(t.shape[0]))
        func = partial(kouri_smooth_l1_norm,t,r)
        jac  = partial(kouri_smooth_l1_norm_gradient,t,r)
        errors = check_gradients(func,jac,init_guess,disp=True)
        assert errors.min()<3e-7

        fd_hess = approx_jacobian(jac,init_guess)
        assert np.allclose(fd_hess,kouri_smooth_l1_norm_hessian(t,r,init_guess))

        nsamples, degree, sparsity = 6, 7, 2
        #nsamples, degree, sparsity = 15, 20, 3
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        def func(x,return_grad=True):
            residual = basis_matrix.dot(x)-vals
            obj = 0.5*residual.dot(residual)
            grad = basis_matrix.T.dot(residual)
            if return_grad:
                return obj, grad
            return obj

        def hess(x):
            return basis_matrix.T.dot(basis_matrix)
        jac=True

        init_guess = np.random.normal(0,1,(true_coef.shape[0]))
        assert np.allclose(func(init_guess)[0],0.5*np.linalg.norm(basis_matrix.dot(init_guess)-vals)**2)

        tol=1e-8
        eps=1e-14
        #init_guess = np.random.normal(0,1,(true_coef.shape[0]))
        init_guess = true_coef
        options = {'gtol':tol,'verbose':2,'disp':True,'xtol':tol,'maxiter':1000}
        homotopy_options = {'gtol':tol,'verbose':2,'disp':True,'xtol':1e-4,'maxiter':10}
        res = basis_pursuit_denoising(func,jac,hess,init_guess,eps,options,homotopy_options)
        print (res)
        coef =  res.x

        print(true_coef,coef)
        assert np.allclose(true_coef,coef)
        

if __name__ == '__main__':
    optimization_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestOptimization)
    unittest.TextTestRunner(verbosity=2).run(optimization_test_suite)


    
