import unittest
from pyapprox.variables import *
from scipy.linalg import lu_factor, lu as scipy_lu
from scipy.stats import norm, beta, gamma, binom, uniform
from pyapprox.utilities import lists_of_arrays_equal

class TestVariables(unittest.TestCase):
    def test_get_distribution_params(self):
        name,scales,shapes = get_distribution_info(beta(a=1,b=2,loc=0,scale=1))
        assert name=='beta'
        assert shapes=={'a':1,'b':2}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(beta(1,b=2,loc=0,scale=1))
        assert name=='beta'
        assert shapes=={'a':1,'b':2}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(beta(1,2,loc=0,scale=1))
        assert name=='beta'
        assert shapes=={'a':1,'b':2}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(beta(1,2,0,scale=1))
        assert name=='beta'
        assert shapes=={'a':1,'b':2}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(beta(1,2,0,1))
        assert name=='beta'
        assert shapes=={'a':1,'b':2}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(norm(0,1))
        assert name=='norm'
        assert shapes==dict()
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(norm(0,scale=1))
        assert name=='norm'
        assert shapes==dict()
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(norm(loc=0,scale=1))
        assert name=='norm'
        assert shapes==dict()
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(gamma(a=1,loc=0,scale=1))
        assert name=='gamma'
        assert shapes=={'a':1}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(gamma(1,loc=0,scale=1))
        assert name=='gamma'
        assert shapes=={'a':1}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(gamma(1,0,scale=1))
        assert name=='gamma'
        assert shapes=={'a':1}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(gamma(1,0,1))
        assert name=='gamma'
        assert shapes=={'a':1}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(gamma(1))
        assert name=='gamma'
        assert shapes=={'a':1}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(gamma(1,loc=0))
        assert name=='gamma'
        assert shapes=={'a':1}
        assert scales=={'loc':0,'scale':1}


        name,scales,shapes = get_distribution_info(gamma(1,scale=1))
        assert name=='gamma'
        assert shapes=={'a':1}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(binom(n=1,p=1,loc=0))
        assert name=='binom'
        assert shapes=={'n':1,'p':1}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(binom(1,p=1,loc=0))
        assert name=='binom'
        assert shapes=={'n':1,'p':1}
        assert scales=={'loc':0,'scale':1}


        name,scales,shapes = get_distribution_info(binom(1,1,loc=0))
        assert name=='binom'
        assert shapes=={'n':1,'p':1}
        assert scales=={'loc':0,'scale':1}

        name,scales,shapes = get_distribution_info(binom(1,1,0))
        assert name=='binom'
        assert shapes=={'n':1,'p':1}
        assert scales=={'loc':0,'scale':1}

    
    def test_define_iid_random_variables(self):
        """
        Construct a independent and identiically distributed (iid) 
        multivariate random variable from the tensor-product of
        the same one-dimensional variable.
        """
        var=norm(loc=2,scale=3)
        num_vars=2
        iid_variable = define_iid_random_variables(var,num_vars)

        assert len(iid_variable.unique_variables)==1
        assert np.allclose(
            iid_variable.unique_variable_indices,np.arange(num_vars))

    def test_define_mixed_tensor_product_random_variable_I(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable type 
        the distribution parameters ARE the same
        """
        univariate_variables = [
            uniform(-1,2),beta(1,1,-1,2),norm(0,1),uniform(-1,2),
            uniform(-1,2),beta(1,1,-1,2)]
        variable = IndependentMultivariateRandomVariable(univariate_variables)
        
        assert len(variable.unique_variables)==3
        assert lists_of_arrays_equal(variable.unique_variable_indices,
                                     [[0,3,4],[1,5],[2]])

    def test_define_mixed_tensor_product_random_variable_II(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable 
        type the distribution parameters ARE NOT the same
        """
        univariate_variables = [
            uniform(-1,2),beta(1,1,-1,2),norm(-1,2),uniform(),uniform(-1,2),
            beta(2,1,-2,3)]
        variable = IndependentMultivariateRandomVariable(univariate_variables)

        assert len(variable.unique_variables)==5
        assert lists_of_arrays_equal(variable.unique_variable_indices,
                                     [[0,4],[1],[2],[3],[5]])

    def test_float_discrete_variable(self):
        nmasses1=10
        mass_locations1 = np.geomspace(1.0, 32.0, num=nmasses1)
        masses1 = np.ones(nmasses1,dtype=float)/nmasses1
        var1 = float_rv_discrete(
            name='var1',values=(mass_locations1,masses1))()

        for power in [1,2,3]:
            assert np.allclose(
                var1.moment(power),(mass_locations1**power).dot(masses1))

        np.random.seed(1)
        num_samples = int(1e6)
        samples = var1.rvs(size=(1,num_samples))
        assert np.allclose(samples.mean(),var1.moment(1),atol=1e-2)

        #import matplotlib.pyplot as plt
        #xx = np.linspace(0,33,301)
        #plt.plot(mass_locations1,np.cumsum(masses1),'rss')
        #plt.plot(xx,var1.cdf(xx),'-'); plt.show()
        assert np.allclose(np.cumsum(masses1),var1.cdf(mass_locations1))

        #import matplotlib.pyplot as plt
        #yy = np.linspace(0,1,51)
        #plt.plot(mass_locations1,np.cumsum(masses1),'rs')
        #plt.plot(var1.ppf(yy),yy,'-o',ms=2); plt.show()
        xx = mass_locations1
        assert np.allclose(xx,var1.ppf(var1.cdf(xx)))

        xx = mass_locations1
        assert np.allclose(xx,var1.ppf(var1.cdf(xx+1e-1)))

    def test_get_statistics(self):
        univariate_variables = [
            uniform(2,4),beta(1,1,-1,2),norm(0,1)]
        variable = IndependentMultivariateRandomVariable(univariate_variables)
        mean = variable.get_statistics('mean')
        assert np.allclose(mean.squeeze(),[4,0,0])

        intervals = variable.get_statistics('interval',alpha=1)
        assert np.allclose(intervals,np.array([[2,6],[-1,1],[-np.inf,np.inf]]))


if __name__== "__main__":    
    variables_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestVariables)
    unittest.TextTestRunner(verbosity=2).run(variables_test_suite)

