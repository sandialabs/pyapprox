import unittest
from pyapprox.probability_measure_sampling import *
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.variables import IndependentMultivariateRandomVariable, \
    float_rv_discrete

from scipy.stats import beta as beta, uniform, norm, rv_discrete

class TestProbabilitySampling(unittest.TestCase):
    def setUp(self):
        uniform_var1={'var_type':'uniform','range':[-1,1]}
        uniform_var2={'var_type':'uniform','range':[0,1]}
        beta_var1   ={'var_type':'beta','range':[-1,1],
                      'alpha_stat':1,'beta_stat':1}
        beta_var2   ={'var_type':'beta','range':[-2,1],
                      'alpha_stat':2,'beta_stat':1}
        gaussian_var={'var_type':'gaussian','mean':-1.,'variance':4.}

        #self.continuous_variables = [
        #    uniform_var1,beta_var1,gaussian_var,uniform_var2,uniform_var1,
        #    beta_var2]
        self.continuous_variables = [
            uniform(-1,2),beta(1,1,-1,2),norm(-1,2),uniform(),uniform(-1,2),
            beta(2,1,-2,3)]
        
        self.continuous_mean = np.array(
            [0.,0.,-1,0.5,0.,beta.mean(a=2,b=1,loc=-2,scale=3)])

        nmasses1=10
        mass_locations1 = np.geomspace(1.0, 32.0, num=nmasses1)
        masses1 = np.ones(nmasses1,dtype=float)/nmasses1

        nmasses2=10
        mass_locations2 = np.arange(0,nmasses2)
        masses2 = np.geomspace(1.0, 32.0, num=nmasses2)
        masses2 /= masses2.sum()
        # second () is to freeze variable which creates var.dist member
        # variable
        var1 = float_rv_discrete(name='var1',values=(mass_locations1,masses1))()
        var2 = float_rv_discrete(name='var2',values=(mass_locations2,masses2))()
        self.discrete_variables = [var1,var2]
        self.discrete_mean = np.empty(len(self.discrete_variables))
        for ii,var in enumerate(self.discrete_variables):
            self.discrete_mean[ii]=var.moment(1)
    
    
    def test_independent_continuous_samples(self):
        
        variable = IndependentMultivariateRandomVariable(
            self.continuous_variables)

        var_trans = AffineRandomVariableTransformation(variable)

        num_samples = int(1e6)
        samples = generate_independent_random_samples(
            var_trans.variable,num_samples)

        mean = samples.mean(axis=1)
        assert np.allclose(mean,self.continuous_mean,atol=1e-2)

    def test_independent_discrete_samples(self):

        variable  = IndependentMultivariateRandomVariable(
            self.discrete_variables)
        var_trans = AffineRandomVariableTransformation(variable)

        num_samples = int(1e6)
        samples = generate_independent_random_samples(
            var_trans.variable,num_samples)
        mean = samples.mean(axis=1)
        assert np.allclose(mean,self.discrete_mean,rtol=1e-2)

    def test_independent_mixed_continuous_discrete_samples(self):

        univariate_variables=self.continuous_variables+self.discrete_variables
        I = np.random.permutation(len(univariate_variables))
        univariate_variables = [univariate_variables[ii] for ii in I]
        
        variable  = IndependentMultivariateRandomVariable(univariate_variables)
        var_trans = AffineRandomVariableTransformation(variable)

        num_samples = int(1e6)
        samples = generate_independent_random_samples(
            var_trans.variable,num_samples)
        mean = samples.mean(axis=1)

        true_mean = np.concatenate([self.continuous_mean,self.discrete_mean])[I]
        assert np.allclose(mean,true_mean,atol=1e-2)

if __name__=='__main__':
    probability_sampling_test_suite=unittest.TestLoader().loadTestsFromTestCase(
          TestProbabilitySampling)
    unittest.TextTestRunner(verbosity=2).run(probability_sampling_test_suite)
