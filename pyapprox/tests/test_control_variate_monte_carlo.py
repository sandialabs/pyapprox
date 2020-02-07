import unittest
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.configure_plots import *

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
    def __init__(self,theta1):
        """
        Parameters
        ----------
        theta0 : float
            Angle controling 
        Notes
        -----
        The choice of A0, A1, A2 here results in unit variance for each model
        """
        self.A0 = np.sqrt(11)
        self.A1 = np.sqrt(7)
        self.A2 = np.sqrt(3)
        self.nmodels=3
        self.theta0=np.pi/2
        self.theta1=theta1
        self.theta2=np.pi/6
        assert self.theta0>self.theta1 and self.theta1>self.theta2
        
    def m1(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return self.A0*(np.cos(self.theta0) * x**5 + np.sin(self.theta0) * y**5)
    
    def m2(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return self.A1*(np.cos(self.theta1) * x**3 + np.sin(self.theta1) * y**3)
    
    def m3(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return self.A2*(np.cos(self.theta2) * x + np.sin(self.theta2) * y)

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
            return cov
        else:
            samples = self.generate_samples(npilot)
            values  = np.zeros((npilot, 3))
            values[:,0] = self.m1(samples)
            values[:,1] = self.m2(samples)
            values[:,2] = self.m3(samples)
            cov = np.cov(samples,rowvar=False)

            return cov, samples, values

    def generate_samples(self,nsamples):
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
        nhf_samples_std, nsample_ratios_std = pya.standardize_sample_ratios(
            nhf_samples,nsample_ratios)
        assert np.allclose(nhf_samples_std,10)
        assert np.allclose(nsample_ratios_std,[2])

    def test_MLMC_tunable_example(self):
        example = TunableExample(np.pi/4)
        #costs = np.array([1.0, 1.0/100, 1.0/100/100])
        cov, samples , values= example.get_covariance_matrix(int(1e3))
        import seaborn as sns
        from pandas import DataFrame
        df = DataFrame(
            index=np.arange(values.shape[0]),
            data=dict([(r'$z_%d$'%ii,values[:,ii])
                       for ii in range(values.shape[1])]))
        # heatmap does not currently work with matplotlib 3.1.1 downgrade to
        # 3.1.0 using pip install matplotlib==3.1.0
        #sns.heatmap(df.corr(),annot=True,fmt='.2f',linewidth=0.5)
        exact_cov = example.get_covariance_matrix()
        exact_cor = pya.get_correlation_from_covariance(exact_cov)
        print(exact_cor)
        print(df.corr())
        #plt.tight_layout()
        #plt.show()

        theta1 = np.linspace(example.theta2*1.05,example.theta0*0.95,5)
        covs = []
        var_reds = []
        for th1 in theta1:
            example.theta1=th1
            covs.append(example.get_covariance_matrix())
            OCV_var_red = pya.get_variance_reduction(
                pya.get_control_variate_rsquared,covs[-1],None)
            # use model with largest covariance with high fidelity model
            idx = [0,np.argmax(covs[-1][0,1:])+1]
            assert idx == [0,1] #it will always be the first model
            OCV1_var_red = pya.get_variance_reduction(
                pya.get_control_variate_rsquared,covs[-1][np.ix_(idx,idx)],None)
            var_reds.append([OCV_var_red,OCV1_var_red])
        covs = np.array(covs)
        var_reds = np.array(var_reds)

        fig,axs = plt.subplots(1,2,figsize=(2*8,6))
        for ii,jj, in [[0,1],[0,2],[1,2]]:
            axs[0].plot(theta1,covs[:,ii,jj],'o-',
                        label=r'$\rho_{%d%d}$'%(ii,jj))
        axs[1].plot(theta1,var_reds[:,0],'o-',label=r'$\textrm{OCV}$')
        axs[1].plot(theta1,var_reds[:,1],'o-',label=r'$\textrm{OCV1}$')
        axs[1].plot(theta1,var_reds[:,0]/var_reds[:,1],'o-',
                    label=r'$\textrm{OCV/OCV1}$')
        axs[0].set_xlabel(r'$\theta_1$')
        axs[0].set_ylabel(r'$\textrm{Correlation}$')
        axs[1].set_xlabel(r'$\theta_1$')
        axs[1].set_ylabel(r'$\textrm{Variance reduction ratio} \ \gamma$')
        axs[0].legend()
        axs[1].legend()
        #plt.show()

        print('####')
        target_cost = 100
        cost_ratio = 10
        costs = np.array([1,1/cost_ratio,1/cost_ratio**2])
        example.theta0=1.4/0.95
        example.theta2=0.6/1.05
        theta1 = np.linspace(example.theta2*1.05,example.theta0*0.95,5)
        #allocate = pya.allocate_samples_mlmc
        #get_rsquared = pya.get_rsquared_mlmc
        #allocate = pya.allocate_samples_mfmc
        #get_rsquared = pya.get_rsquared_mfmc
        allocate = pya.allocate_samples_acv
        get_rsquared = pya.get_rsquared_acv1
        for th1 in theta1:
            example.theta1=th1
            cov = example.get_covariance_matrix()
            nhf_samples, nsample_ratios, log10_var = allocate(
                cov,costs,target_cost)
            var_red = pya.get_variance_reduction(
                get_rsquared,cov,nsample_ratios)
            assert np.allclose(var_red,(10**log10_var)/cov[0,0]*nhf_samples)
            print(var_red)
            assert False
    
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
        nhf_samples, nsample_ratios, log10_variance = pya.allocate_samples_mlmc(
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

