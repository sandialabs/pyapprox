import unittest
from pyapprox.control_variate_monte_carlo import *

def mlmc_variance_reduction(pilot_samples, sample_ratios, nhf_samples):
    """
    Parameters
    ----------
    pilot_samples : np.ndarray (npilot_samples,nlevel)
        Pilot samples of the quantitiy of interest at all levels in 
        decending order. The highest fidelity model is the first model.

    sample_ratios : np.ndarray (nlevel-1)
        The list of ratios (LF/HF) of the number of samples that will be used
        in final study. This will be different to npilot_samples

    nhf_samples : integer
        The number of high-fidelity samples to be used in final study 
        (not in the pilot sample)

    Returns
    -------
    variance_ratio : float 
        The variance ratio (MLMC / MC)
    """
    nmodels = pilot_samples.shape[1]
    assert len(sample_ratios)==nmodels-1
    var = 0.0
    nruns = nhf_samples
    for ii in range(nmodels-1):
        var += np.var(pilot_samples[:,ii]-pilot_samples[:,ii+1])/nruns
        nruns = sample_ratios[ii] * nhf_samples - nruns

    var += np.var(pilot_samples[:,-1]) / nruns

    varhf = np.var(pilot_samples[:,0]) / nhf_samples

    return var / (varhf)

class TunableExample(object):
    def __init__(self,theta0,theta1,theta2):
        self.A0 = np.sqrt(11)
        self.A1 = np.sqrt(7)
        self.A2 = np.sqrt(3)
        self.nmodels=3
        self.theta0=theta0
        self.theta1=theta1
        self.theta2=theta2
        
    def m1(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return A0 * (np.cos(self.theta0) * x**5 + np.sin(self.theta0) * y**5)
    
    def m2(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return A1 * (np.cos(self.theta1) * x**3 + np.sin(self.theta1) * y**3)
    
    def m3(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return A2 * (np.cos(self.theta2) * x + np.sin(self.theta2) * y)

    def get_covariance_matrix(self,npilot=None):
        if npilot is None:
            cov = np.eye(self.nmodels)
            cov[0, 1] = self.A0*self.A1/9*(np.sin(self.theta0)*np.sin(
                self.theta1)+np.cos(self.theta0)*np.cos(self.theta1))
            cov[1, 0] = cov[0,1]
            cov[0, 2] = self.A0*self.A2/7*(np.sin(self.theta0)*np.sin(
                self.theta2)+np.cos(self.theta0)*np.cos(self.theta2))
            cov[2, 0] = cov[0, 2]
            cov[1, 2] = self.A1*self.A2/5*(
                np.sin(self.theta1)*np.sin(self.theta2)+np.cos(
                self.theta1)*np.cos(self.theta2))
            cov[2, 1] = cov[1,2]
        else:
            samples = self.generate_samples(npilot)
            values  = np.zeros((npilot, 3))
            values[:,0] = self.m1(samples)
            values[:,1] = self.m2(samples)
            values[:,2] = self.m3(samples)
            cov = np.cov(samples,rowvar=False)

        return cov

    def generate_samples(nsamples):
        return np.random.uniform(-1,1,(2,nsamples))

    def __call__(self,sample_sets):
        assert len(samples_sets)==3
        models = [self.m1,self.m2,self.m3]
        value_sets = []
        for ii in range(len(sample_sets)):
            if samples_sets[ii] is not None:
                value_sets.append(models[ii](samples))
        return value_sets
                    

class TestCVMC(unittest.TestCase):

    def test_standardize_sample_ratios(self):
        nhf_samples, nsample_ratios = 9.8, [2.1]
        nhf_samples_std, nsample_ratios_std = standardize_sample_ratios(
            nhf_samples,nsample_ratios)
        assert np.allclose(nhf_samples_std,10)
        assert np.allclose(nsample_ratios_std,[2])

    def test_MLMC_tunable_example(self):
        example = TunableExample(np.pi/2,0.5497787143782138,np.pi/6)
        costs = np.array([1.0, 1.0/100, 1.0/100/100])
        cov = example.get_covariance_matrix()

        
    
    def test_MLMC_variance_reduction(self):
        np.random.seed(1)
        matr = np.random.randn(2,2)
        exact_covariance = np.dot(matr, matr.T)
        chol_factor = np.linalg.cholesky(exact_covariance)
        
        samples = np.random.normal(0,1,(100000,2))
        values = np.dot(samples,chol_factor.T)
        covariance = np.cov(values, rowvar=False)
        costs = [1, 1e-1]

        target_cost=10
        nhf_samples, nsample_ratios, log10_variance = allocate_samples_mlmc(
            covariance, costs, target_cost, nhf_samples_fixed=None)
        
        actual_cost=costs[0]*nhf_samples+\
            costs[1]*(nsample_ratios[0]*nhf_samples)

        assert actual_cost==10.4
        assert np.allclose(nsample_ratios,[3])
        assert nhf_samples==8

        mlmc_variance = mlmc_variance_reduction(
            values,nsample_ratios,nhf_samples)*covariance[0,0]/nhf_samples
        assert np.allclose(mlmc_variance,10**log10_variance)

    def test_CVMC(self):
        pass

    def test_ACVMC(self):
        pass
    
if __name__== "__main__":    
    cvmc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestCVMC)
    unittest.TextTestRunner(verbosity=2).run(cvmc_test_suite)

