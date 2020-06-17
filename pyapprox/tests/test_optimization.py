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
        nsamples, degree, sparsity = 20, 7, 2 
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        np.savetxt('basis_matrix.txt',basis_matrix)
        np.savetxt('rhs.txt',vals)
        np.savetxt('true_coef.txt',true_coef)

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
        #init_guess = lstsq_coef+np.random.normal(0,1e-3,(true_coef.shape[0]))

        errors = check_gradients(objective,True,init_guess,disp=True)
        assert errors.min()<2e-7

        method = 'trust-constr'
        func = partial(objective,return_grad=True); jac=True; hess=hessian; 
        options = {'gtol':tol,'verbose':2,'disp':True,'xtol':tol,'maxiter':10000}
        constraints = []

        # def constraint_obj(x):
        #     val = objective(x)[0]
        #     return val
    
        # def constraint_jac(x):
        #     jac = objective(x)[1]
        #     return jac

        # def constraint_hessian(x,v):
        #     return hessian(x)*v[0]
                
        # #constraint_hessian = BFGS()

        # eps=1e-4
        # nonlinear_constraint = NonlinearConstraint(
        #     constraint_obj,0,eps,jac=constraint_jac,hess=constraint_hessian,
        #     keep_feasible=False)
        # constraints = [nonlinear_constraint]
        
        res = minimize(
            func, init_guess, method=method, jac=jac, hess=hess,options=options,
            constraints=constraints)

        #print(lstsq_coef)
        print(res.x,true_coef)
        assert np.allclose(res.x,true_coef,atol=1e-4)
        

    def test_nonlinear_basis_pursuit_with_linear_model(self):
        np.random.seed(1)
        # nsamples, degree, sparsity = 100, 7, 2 
        # samples = np.random.uniform(0,1,(1,nsamples))
        # basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        # true_coef = np.zeros(basis_matrix.shape[1])
        # true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        # vals = basis_matrix.dot(true_coef)

        basis_matrix,true_coef,vals = self.SPARCO_problem_7(1024//4,256//4,32//4)

        def func(x):
            return basis_matrix.dot(x)-vals, basis_matrix
        jac=True
        def hess(x):
            return sp.lil_matrix((x.shape[0],x.shape[0]),dtype=float)

        tol=1e-12
        options = {'gtol':tol,'verbose':2,'disp':True,'xtol':tol,'maxiter':10000}
        #options = {'tol':tol,'maxiter':1000,'print_level':3,
        #           'method':'ipopt'}
        init_guess = true_coef+np.random.normal(0,1,true_coef.shape[0])
        #fd_jac = approx_jacobian(lambda x: func(x)[0],init_guess,epsilon=1e-7)
        #exact_jac = func(init_guess)[1]
        l1_coef = nonlinear_basis_pursuit(func,jac,hess,init_guess,options)
        print(np.linalg.norm(l1_coef-true_coef))
        assert np.allclose(l1_coef,true_coef,atol=1e-5)

    def test_nonlinear_basis_pursuit_denoising_with_linear_model(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 20, 7, 2 
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        #basis_matrix,true_coef,vals = self.SPARCO_problem_7(1024//4,256//4,32//4)

        eps=1e-3
        def func(x,return_grad=True):
            residual = basis_matrix.dot(x)-vals
            obj = residual.dot(residual)
            grad = 2*basis_matrix.T.dot(residual)
            if return_grad:
                return obj, grad
            return obj
        jac=True; hess=None

        tol=1e-6
        #options = {'gtol':tol,'verbose':2,'disp':True,'xtol':tol,'maxiter':1000,
        #           'method':'trust-constr'}
        options = {'ftol':tol,'disp':True,'maxiter':1000,'iprint':3,
                   'method':'slsqp'}
        #options = {'tol':tol,'maxiter':1000,'print_level':3,
        #           'method':'ipopt'}
        init_guess = np.random.normal(0,1,true_coef.shape[0])
        #fd_jac = approx_jacobian(lambda x: func(x)[0],init_guess,epsilon=1e-7)
        #exact_jac = func(init_guess)[1]
        l1_coef = nonlinear_basis_pursuit(func,jac,hess,init_guess,options,eps**2)
        print(np.linalg.norm(l1_coef-true_coef))
        assert np.allclose(l1_coef,true_coef,atol=1e-5)

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

    def test_smooth_l1_norm_gradients(self):
        #x = np.linspace(-1,1,101)
        #t = np.ones_like(x)
        #r = 1e1
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

    def SPARCO_problem_7(self, basis_length, num_samples, sparsity):
        sparsity_matrix = np.eye(basis_length)
        measurement_matrix = np.random.normal(0,1,(num_samples,basis_length))
        signal = np.zeros( ( basis_length ), float )
        non_zero_indices = np.random.permutation( basis_length )[:sparsity]
        signal[non_zero_indices] = np.sign(np.random.normal(0,1,(sparsity)))
        measurements = np.dot( measurement_matrix, signal ) 
        A_matrix = np.dot( measurement_matrix, sparsity_matrix )
        true_solution = signal
        return A_matrix,true_solution,measurements

    def test_basis_pursuit_smooth_l1_norm(self):
        np.random.seed(1)
        #nsamples, degree, sparsity = 6, 7, 2
        nsamples, degree, sparsity = 15, 20, 3
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        basis_matrix,true_coef,vals = self.SPARCO_problem_7(1024//4,256//4,32//4)

        np.savetxt('basis_matrix.txt',basis_matrix)
        np.savetxt('rhs.txt',vals)
        np.savetxt('true_coef.txt',true_coef)


        def func(x):
            return basis_matrix.dot(x)-vals, basis_matrix
        jac=True
        def hess(x):
            return sp.lil_matrix((x.shape[0],x.shape[0]),dtype=float)

        tol=1e-6
        eps=0
        init_guess = np.random.normal(0,1,(true_coef.shape[0]))*0
        method='ipopt'
        #init_guess = true_coef
        options = {'gtol':tol,'verbose':2,'disp':True,'xtol':1e-10,
                   'maxiter':20,'method':method,'ftol':1e-10}
        res = basis_pursuit_denoising(
            func,jac,hess,init_guess,eps,options)
        coef =  res.x

        #print(true_coef,coef)
        print(np.linalg.norm(coef-true_coef))
        assert np.allclose(true_coef,coef,atol=1e-7)

    def test_basis_pursuit_denoising_smooth_l1_norm(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 20, 7, 2
        #nsamples, degree, sparsity = 15, 20, 3
        samples = np.random.uniform(0,1,(1,nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        vals = basis_matrix.dot(true_coef)

        #basis_matrix,true_coef,vals = self.SPARCO_problem_7(1024//4,256//4,32//4)

        def func(x,return_grad=True):
            residual = basis_matrix.dot(x)-vals
            obj = residual.dot(residual)
            grad = 2*basis_matrix.T.dot(residual)
            if return_grad:
                return obj, grad
            return obj

        def hess(x):
            return 2*basis_matrix.T.dot(basis_matrix)
        jac=True

        init_guess = np.random.normal(0,1,(true_coef.shape[0]))
        assert np.allclose(func(init_guess)[0],np.linalg.norm(basis_matrix.dot(init_guess)-vals)**2)

        print(true_coef)
        eps=1e-3
        method = 'slsqp'
        #method = 'ipopt'
        #init_guess = np.random.normal(0,0.1,(true_coef.shape[0]))
        init_guess = np.random.normal(0,1,(true_coef.shape[0]))*0
        #init_guess = true_coef
        options = {'gtol':1e-8,'verbose':6,'disp':True,'dualtol':1e-6,'maxiter_inner':1e3,'r0':1e4,'maxiter':1e2,'ftol':1e-10,'method':method}
        res = basis_pursuit_denoising(func,jac,hess,init_guess,eps,options)
        coef =  res.x

        print(np.linalg.norm(true_coef-coef))
        assert np.allclose(true_coef,coef)
        

if __name__ == '__main__':
    optimization_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestOptimization)
    unittest.TextTestRunner(verbosity=2).run(optimization_test_suite)


    
