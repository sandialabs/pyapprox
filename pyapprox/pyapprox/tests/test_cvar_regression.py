import unittest
from pyapprox.cvar_regression import *

class TestCVaR(unittest.TestCase):

    def test_value_at_risk_normal(self):
        weights=None
        alpha=0.8
        num_samples = int(1e2)
        samples = np.random.normal(0,1,num_samples)
        xx = np.sort(samples)
        VaR,VaR_index = value_at_risk(xx,alpha,weights,samples_sorted=True)
        index = int(np.ceil(alpha*num_samples)-1)
        assert np.allclose(VaR_index,index)
        assert np.allclose(VaR,xx[index])

    def test_value_at_risk_lognormal(self):
        mu,sigma=0,1

        f = lambda x: np.exp(x).T
        VaR  = lambda p: np.exp(mu+sigma*np.sqrt(2)*erfinv(2*p-1))
        mean = np.exp(mu+sigma**2/2)
        CVaR = lambda p: mean*normal_rv.cdf(
            (mu+sigma**2-np.log(VaR(p)))/sigma)/(1-p)

        weights=None
        alpha=0.8
        num_samples = int(1e6)
        samples = f(np.random.normal(0,1,num_samples))
        xx = np.sort(samples)
        empirical_VaR,__ = value_at_risk(xx,alpha,weights,samples_sorted=True)
        #print(VaR(alpha),empirical_VaR)
        assert np.allclose(VaR(alpha),empirical_VaR,1e-2)

    def test_weighted_value_at_risk_normal(self):
        mu,sigma=1,1
        #bias_mu,bias_sigma=1.0,1
        bias_mu,bias_sigma=mu,sigma

        from scipy.special import erf
        VaR = lambda alpha: normal_rv.ppf(alpha,loc=mu,scale=sigma)
        def CVaR(alpha):
            vals =  0.5*mu
            vals -= 0.5*mu*erf((VaR(alpha)-mu)/(np.sqrt(2)*sigma))-\
                sigma*np.exp(-(mu-VaR(alpha))**2/(2*sigma**2))/np.sqrt(2*np.pi)
            vals /= (1-alpha)
            return vals

        alpha=0.8
        num_samples = int(1e5)
        samples = np.random.normal(bias_mu,bias_sigma,num_samples)
        target_pdf_vals = normal_rv.pdf(samples,loc=mu,scale=sigma)
        bias_pdf_vals   = normal_rv.pdf(samples,loc=bias_mu,scale=bias_sigma)
        I = np.where(bias_pdf_vals<np.finfo(float).eps)[0]
        assert np.all(target_pdf_vals[I]<np.finfo(float).eps)
        J = np.where(bias_pdf_vals>=np.finfo(float).eps)[0]
        weights = np.zeros_like(target_pdf_vals)
        weights[J] = target_pdf_vals[J]/bias_pdf_vals[J]
        
        empirical_VaR,__ = value_at_risk(
            samples,alpha,weights,samples_sorted=False)
        #print('VaR',VaR(alpha),empirical_VaR)
        assert np.allclose(VaR(alpha),empirical_VaR,rtol=1e-2)

        empirical_CVaR = conditional_value_at_risk(
            samples,alpha,weights)
        #print('CVaR',CVaR(alpha),empirical_CVaR)
        assert np.allclose(CVaR(alpha),empirical_CVaR,rtol=1e-2)
        


if __name__== "__main__":    
    cvar_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestCVaR)
    unittest.TextTestRunner(verbosity=2).run(cvar_test_suite)
