import unittest
from pyapprox.optimization import *
from scipy.optimize import check_grad

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



        

if __name__ == '__main__':
    optimization_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestOptimization)
    unittest.TextTestRunner(verbosity=2).run(optimization_test_suite)


    
