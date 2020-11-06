import unittest
from pyapprox.quantile_regression import *
from pyapprox.cvar_regression import *
from pyapprox.stochastic_dominance import *
from scipy.special import erf, erfinv, factorial
from scipy.stats import truncnorm as truncnorm_rv, triang as triangle_rv, \
    lognorm as lognormal_rv, norm as normal_rv
from pyapprox.configure_plots import *
from pyapprox.optimization import check_gradients
from pyapprox.rol_minimize import has_ROL

try:
    import cvxopt
    has_cvxopt=True
except:
    has_cvxopt=False
skiptest = unittest.skipIf(
    not has_cvxopt, reason="cvxopt package not found")

skiptest_rol = unittest.skipIf(
    not has_ROL, reason="rol package not found")

def plot_1d_functions_and_statistics(
            functions,labels,samples,values,stat_function,eta):

    xlb,xub = samples.min()-abs(samples.max())*.1,\
        samples.max()+abs(samples.max())*.1
    xx = np.linspace(xlb,xub,101)
    fig,axs = plt.subplots(1,2,figsize=(2*8,6))
    colors = ['k','r','b','g'][:len(functions)]
    linestyles = ['-',':','--','-.'][:len(functions)]
    for function,label,color,ls in zip(functions,labels,colors,linestyles):
        axs[0].plot(xx,function(xx[np.newaxis,:]),ls=ls,c=color,label=label)
    axs[0].plot(samples[0,:],values[:,0],'ok',label='Train data')
    axs[0].set_xlim(xlb,xub)

    for function,label,color,ls in zip(functions,labels,colors,linestyles):
        stats = stat_function(function(samples)[:,0])
        I = stats.argmax()
        axs[1].plot(eta,stats,ls=ls,c=color,label=label)

    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$f(x)$')
    axs[1].set_xlabel(r'$\eta$')
    axs[0].legend()
    axs[1].legend()

    return fig,axs


def triangle_quantile(u,c,loc,scale):
    """
    Also known as inverse CDF
    """
    if np.isscalar(u):
        u = np.asarray([u])
    assert u.ndim==1
    lb = loc
    mid = loc+c*scale
    ub = loc+scale
    I = np.where((u>=0)&(u < (mid-lb)/(ub-lb)))[0]
    J = np.where((u<=1)&(u >= (mid-lb)/(ub-lb)))[0]
    if I.shape[0]+J.shape[0]!=u.shape[0]:
        raise Exception('Ensure u in [0,1] and loc<=loc+c*scale <=loc+scale')
    
    quantiles = np.empty_like(u)
    quantiles[I]=lb + np.sqrt((ub-lb)*(mid-lb)*u[I])
    quantiles[J]=ub - np.sqrt((ub-lb)*(ub-mid)*(1-u[J]))
    return quantiles

def triangle_superquantile(u,c,loc,scale):
    lb = loc
    mid = loc+c*scale
    ub = loc+scale

    if np.isscalar(u):
        u = np.asarray([u])
    assert u.ndim==1

    left_integral  = lambda u: 2./3*u*np.sqrt(u*(lb - ub)*(lb - mid)) + lb*u
    right_integral = lambda u: ub*u-2./3*(u-1)*np.sqrt((u-1)*(lb-ub)*(ub-mid)) 
    
    I = np.where((u>=0)&(u < (mid-lb)/(ub-lb)))[0]
    J = np.where((u<1)&(u >= (mid-lb)/(ub-lb)))[0]
    K = np.where((u==1))[0]
    
    if I.shape[0]+J.shape[0]+K.shape[0]!=u.shape[0]:
        raise Exception('Ensure u in [0,1] and loc<=loc+c*scale <=loc+scale')

    superquantiles = np.empty_like(u)
    superquantiles[I]=(left_integral((mid-lb)/(ub-lb))-left_integral(u[I])+right_integral(1)-right_integral((mid-lb)/(ub-lb)))/(1-u[I])
    superquantiles[J]=(right_integral(1)-right_integral(u[J]))/(1-u[J])
    superquantiles[K]=triangle_rv.interval(1,c,loc,scale)[1]
    return superquantiles

def get_lognormal_example_exact_quantities(mu, sigma):
    #print('mu,sigma',mu,sigma)
    f= lambda x: np.exp(x).T

    mean = np.exp(mu+sigma**2/2)
    
    def f_cdf(y):
        y = np.atleast_1d(y)
        vals = np.zeros_like(y)
        II = np.where(y>0)[0]
        vals[II]=normal_rv.cdf((np.log(y[II])-mu)/sigma)
        return vals
    
    # PDF of output variable (lognormal PDF)
    def f_pdf(y):
        vals = np.zeros_like(y)
        II = np.where(y>0)[0]
        vals[II] = np.exp(-(np.log(y[II])-mu)**2/(2*sigma**2))/(
            sigma*np.sqrt(2*np.pi)*y[II])
        return vals
    
    # Analytic VaR of model output
    VaR  = lambda p: np.exp(mu+sigma*np.sqrt(2)*erfinv(2*p-1))
    
    # Analytic VaR of model output
    CVaR = lambda p: mean*normal_rv.cdf(
        (mu+sigma**2-np.log(VaR(p)))/sigma)/(1-p)
    
    def cond_exp_le_eta(y):
        vals = np.zeros_like(y)
        II = np.where(y>0)[0]
        vals[II]=mean*normal_rv.cdf((np.log(y[II])-mu-sigma**2)/sigma)/f_cdf(
            y[II])
        return vals
        
    ssd  =  lambda y: f_cdf(y)*(y-cond_exp_le_eta(y))
    
    def cond_exp_y_ge_eta(y):
        vals = np.ones_like(y)*mean
        II = np.where(y>0)[0]
        vals[II] = mean*normal_rv.cdf(
            (mu+sigma**2-np.log(y[II]))/sigma)/(1-f_cdf(y[II]))
        return vals

    ssd_disutil  =  lambda eta: (1-f_cdf(-eta))*(eta+cond_exp_y_ge_eta(-eta))
    return f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil

def get_truncated_lognormal_example_exact_quantities(lb,ub,mu,sigma):
    f = lambda x: np.exp(x).T

    #lb,ub passed to truncnorm_rv are defined for standard normal.
    #Adjust for mu and sigma using
    alpha, beta = (lb-mu)/sigma, (ub-mu)/sigma
    
    denom = normal_rv.cdf(beta)-normal_rv.cdf(alpha)
    #truncated_normal_cdf = lambda x: (
    #    normal_rv.cdf((x-mu)/sigma)-normal_rv.cdf(alpha))/denom
    truncated_normal_cdf=lambda x: truncnorm_rv.cdf(
        x,alpha,beta,loc=mu,scale=sigma)
    truncated_normal_pdf=lambda x: truncnorm_rv.pdf(
        x,alpha,beta,loc=mu,scale=sigma)
    truncated_normal_ppf=lambda p: truncnorm_rv.ppf(
        p,alpha,beta,loc=mu,scale=sigma)

    # CDF of output variable (log truncated normal PDF)
    def f_cdf(y):
        vals = np.zeros_like(y)
        II = np.where((y>np.exp(lb))&(y<np.exp(ub)))[0]
        vals[II]=truncated_normal_cdf(np.log(y[II]))
        JJ = np.where((y>=np.exp(ub)))[0]
        vals[JJ]=1.
        return vals
    
    # PDF of output variable (log truncated normal PDF)
    def f_pdf(y):
        vals = np.zeros_like(y)
        II = np.where((y>np.exp(lb))&(y<np.exp(ub)))[0]
        vals[II]=truncated_normal_pdf(np.log(y[II]))/y[II]
        return vals
    
    # Analytic VaR of model output
    VaR  = lambda p: np.exp(truncated_normal_ppf(p))
    
    const = np.exp(mu+sigma**2/2)
   
    # Analytic VaR of model output
    CVaR = lambda p: -0.5/denom*const/(1-p)*(
        erf((mu+sigma**2-ub)/(np.sqrt(2)*sigma))
        -erf((mu+sigma**2-np.log(VaR(p)))/(np.sqrt(2)*sigma)))
    
    def cond_exp_le_eta(y):
        vals = np.zeros_like(y)
        II = np.where((y>np.exp(lb))&(y<np.exp(ub)))[0]
        vals[II]=-0.5/denom*const*(
            erf((mu+sigma**2-np.log(y[II]))/(np.sqrt(2)*sigma))
            -erf((mu+sigma**2-lb)/(np.sqrt(2)*sigma)))/f_cdf(y[II])
        JJ = np.where((y>=np.exp(ub)))[0]
        vals[JJ]=mean
        return vals
    ssd  =  lambda y: f_cdf(y)*(y-cond_exp_le_eta(y))

    mean = CVaR(np.zeros(1))
    def cond_exp_y_ge_eta(y):
        vals = np.ones_like(y)*mean
        II = np.where((y>np.exp(lb))&(y<np.exp(ub)))[0]
        vals[II] = -0.5/denom*const*(
            erf((mu+sigma**2-ub)/(np.sqrt(2)*sigma))
            -erf((mu+sigma**2-np.log(y[II]))/(np.sqrt(2)*sigma)))/(
                1-f_cdf(y[II]))
        JJ = np.where((y>np.exp(ub)))[0]
        vals[JJ]=0
        return vals
    ssd_disutil = lambda eta: (1-f_cdf(-eta))*(eta+cond_exp_y_ge_eta(-eta))

    return f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil

def plot_truncated_lognormal_example_exact_quantities(
        num_samples=int(1e5),plot=False,mu=0,sigma=1):
    if plot:
        assert num_samples<=1e5
    num_vars=1.
    lb,ub = -1,3
    
    f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
        get_truncated_lognormal_example_exact_quantities(lb,ub,mu,sigma)
    #lb,ub passed to truncnorm_rv are defined for standard normal.
    #Adjust for mu and sigma using
    alpha, beta = (lb-mu)/sigma, (ub-mu)/sigma
    samples = truncnorm_rv.rvs(
        alpha,beta,mu,sigma,size=num_samples)[np.newaxis,:]
    values = f(samples)[:,0]
    
    fig,axs=plt.subplots(1,6,sharey=False,figsize=(16,6))
    
    from pyapprox.density import EmpiricalCDF
    ygrid = np.linspace(np.exp(lb)-1,np.exp(ub)*1.1,100)
    ecdf = EmpiricalCDF(values)
    if plot:
        axs[0].plot(ygrid,ecdf(ygrid),'-')
        axs[0].plot(ygrid,f_cdf(ygrid),'--')
        axs[0].set_xlim(ygrid.min(),ygrid.max())
        axs[0].set_title('CDF')

    if plot:
        ygrid = np.linspace(np.exp(lb)-1,np.exp(ub)*1.1,100)
        axs[1].hist(values,bins='auto',density=True) 
        axs[1].plot(ygrid,f_pdf(ygrid),'--')
        axs[1].set_xlim(ygrid.min(),ygrid.max())
        axs[1].set_title('PDF')
    
    pgrid = np.linspace(0.01,1-1e-2,10)
    evar = np.array([value_at_risk(values,p)[0] for p in pgrid]).squeeze()
    if plot:
        axs[2].plot(pgrid,evar,'-')
        axs[2].plot(pgrid,VaR(pgrid),'--')
        axs[2].set_title('VaR')
    else:
        assert np.allclose(evar,VaR(pgrid),rtol=2e-2)

    pgrid = np.linspace(0,1-1e-2,100)
    ecvar = np.array([conditional_value_at_risk(values,p) for p in pgrid])
    # CVaR for alpha=0 should be the mean
    assert np.allclose(ecvar[0],values.mean())
    if plot:
        axs[3].plot(pgrid,ecvar,'-')
        axs[3].plot(pgrid,CVaR(pgrid),'--')
        axs[3].set_xlim(pgrid.min(),pgrid.max())
        axs[3].set_title('CVaR')
    else:
        assert np.allclose(ecvar.squeeze(),CVaR(pgrid).squeeze(),rtol=1e-2)
    
    ygrid = np.linspace(np.exp(lb)-10,np.exp(ub)+1,100)
    essd = compute_conditional_expectations(ygrid,values,False)
    if plot:
        axs[4].plot(ygrid,essd,'-')
        axs[4].plot(ygrid,ssd(ygrid),'--')
        axs[4].set_xlim(ygrid.min(),ygrid.max())
        axs[4].set_title(r'$E[(\eta-Y)^+]$')
        axs[5].set_xlabel(r'$\eta$')
    else:
        assert np.allclose(essd.squeeze(),ssd(ygrid),rtol=2e-2)

    # zoom into ygrid over high probability region of -Y
    ygrid = -ygrid[::-1]
    disutil_essd = compute_conditional_expectations(ygrid,values,True)
    assert np.allclose(disutil_essd,compute_conditional_expectations(
        ygrid,-values,False))
    #print(np.linalg.norm(disutil_essd.squeeze()-ssd_disutil(ygrid),ord=np.inf))
    if plot:
        axs[5].plot(ygrid,disutil_essd,'-',label='Empirical')
        axs[5].plot(ygrid,ssd_disutil(ygrid),'--',label='Exact')
        axs[5].set_xlim(ygrid.min(),ygrid.max())
        axs[5].set_title(r'$E[(\eta-(-Y))^+]$')
        axs[5].set_xlabel(r'$\eta$')
        axs[5].legend()
    
        plt.show()    

def plot_lognormal_example_exact_quantities(num_samples=int(2e5),plot=False,
                                            mu=0,sigma=1):
    num_vars = 1
    if plot:
        assert num_samples<=1e5
    else:
        assert num_samples>=1e4
    
    f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
        get_lognormal_example_exact_quantities(mu,sigma)
    from pyapprox.utilities import transformed_halton_sequence
    #samples = np.random.normal(mu,sigma,(num_vars,num_samples))
    #values = f(samples)[:,0]
    samples = transformed_halton_sequence(
        [partial(normal_rv.ppf,loc=mu,scale=sigma)],num_vars,num_samples)
    values = f(samples)[:,0]

    
    fig,axs=plt.subplots(1,6,sharey=False,figsize=(16,6))
    
    from pyapprox.density import EmpiricalCDF
    if plot:
        ygrid = np.linspace(-1,5,100)
        #ecdf = EmpiricalCDF(values)
        #axs[0].plot(ygrid,ecdf(ygrid),'-')
        axs[0].plot(ygrid,f_cdf(ygrid),'--')
        #axs[0].set_xlim(ygrid.min(),ygrid.max())
        axs[0].set_title('CDF')
        #ecdf = EmpiricalCDF(-values)
        #axs[0].plot(-ygrid,ecdf(-ygrid),'-')

        ygrid = np.linspace(-1,20,100)
        #axs[1].hist(values,bins='auto',density=True) 
        axs[1].plot(ygrid,f_pdf(ygrid),'--')
        axs[1].set_xlim(ygrid.min(),ygrid.max())
        axs[1].set_title('PDF')
    
    pgrid = np.linspace(1e-2,1-1e-2,100)
    evar = np.array([value_at_risk(values,p)[0] for p in pgrid])
    #print(np.linalg.norm(evar.squeeze()-VaR(pgrid),ord=np.inf))
    if plot:
        axs[2].plot(pgrid,evar,'-')
        axs[2].plot(pgrid,VaR(pgrid),'--')
        axs[2].set_title('VaR')
    else:
        assert np.allclose(evar.squeeze(),VaR(pgrid),atol=2e-1)

    pgrid = np.linspace(1e-2,1-1e-2,100)
    ecvar = np.array([conditional_value_at_risk(values,y) for y in pgrid])
    #print(np.linalg.norm(ecvar.squeeze()-CVaR(pgrid).squeeze(),ord=np.inf))
    print(CVaR(0.8))
    if plot:
        axs[3].plot(pgrid,ecvar,'-')
        axs[3].plot(pgrid,CVaR(pgrid),'--')
        axs[3].set_xlim(pgrid.min(),pgrid.max())
        axs[3].set_title('CVaR')
    else:
        assert np.allclose(ecvar.squeeze(),CVaR(pgrid).squeeze(),rtol=4e-2)
        
    #ygrid = np.linspace(-1,10,100)
    ygrid = np.linspace(
        lognormal_rv.ppf(0.0,np.exp(mu),sigma),
        lognormal_rv.ppf(0.9,np.exp(mu),sigma),101)
    essd = compute_conditional_expectations(ygrid,values,False)
    #print(np.linalg.norm(essd.squeeze()-ssd(ygrid),ord=np.inf))
    if plot:
        axs[4].plot(ygrid,essd,'-')
        axs[4].plot(ygrid,ssd(ygrid),'--')
        axs[4].set_xlim(ygrid.min(),ygrid.max())
        axs[4].set_title(r'$E[(\eta-Y)^+]$')
        axs[4].set_xlabel(r'$\eta$')
    else:
        assert np.allclose(essd.squeeze(),ssd(ygrid),atol=1e-3)

        
    # zoom into ygrid over high probability region of -Y
    ygrid = -ygrid[::-1]
    disutil_essd = compute_conditional_expectations(ygrid,values,True)
    assert np.allclose(disutil_essd,compute_conditional_expectations(
        ygrid,-values,False))
    #print(np.linalg.norm(disutil_essd.squeeze()-ssd_disutil(ygrid),ord=np.inf))
    if plot:
        axs[5].plot(ygrid,disutil_essd,'-',label='Empirical')
        axs[5].plot(ygrid,ssd_disutil(ygrid),'--',label='Exact')
        axs[5].set_xlim((ygrid).min(),(ygrid).max())
        axs[5].set_title(r'$E[(\eta-(-Y))^+]$')
        axs[5].set_xlabel(r'$\eta$')
        axs[5].plot([0],[np.exp(mu+sigma**2/2)],'o')
        axs[5].legend()
        plt.show()
    else:
        assert np.allclose(disutil_essd.squeeze(),ssd_disutil(ygrid),atol=1e-3) 

def help_check_stochastic_dominance(solver, nsamples, degree,
                                    disutility=None, plot=False):
    """
    disutilty is none plot emprical CDF
    disutility is True plot disutility SSD
    disutility is False plot standard SSD
    """
    from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
    from pyapprox.variable_transformations import \
        define_iid_random_variable_transformation
    from pyapprox.indexing import compute_hyperbolic_indices
    num_vars = 1
    mu,sigma = 0, 1
    f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
        get_lognormal_example_exact_quantities(mu, sigma)

    samples = np.random.normal(0, 1, (1, nsamples))
    samples = np.sort(samples)
    values = f(samples[0, :])[:, np.newaxis]

    pce = PolynomialChaosExpansion()
    var_trans = define_iid_random_variable_transformation(
        normal_rv(mu, sigma), num_vars) 
    pce.configure({'poly_type':'hermite', 'var_trans':var_trans})
    indices = compute_hyperbolic_indices(1, degree, 1.)
    pce.set_indices(indices)

    eta_indices=None
    #eta_indices=np.argsort(values[:,0])[nsamples//2:]
    coef, sd_opt_problem = solver(
        samples, values, pce.basis_matrix, eta_indices=eta_indices)

    pce.set_coefficients(coef[:, np.newaxis])
    pce_values = pce(samples)[:, 0]

    ygrid = pce_values.copy()
    if disutility is not None:
        if disutility:
            ygrid = -ygrid[::-1]
        stat_function = partial(compute_conditional_expectations,
                                ygrid, disutility_formulation=disutility)
        if disutility:
            # Disutility SSD
            eps=1e-12
            assert np.all(
                stat_function(values[:, 0])<=stat_function(pce_values)+eps)
        else:
            # SSD
            assert np.all(
                stat_function(pce_values)<=stat_function(values[:, 0]))
    else:
        # FSD
        from pyapprox.density import EmpiricalCDF
        stat_function = lambda x: EmpiricalCDF(x)(ygrid)
        print(stat_function(pce_values), stat_function(values[:, 0]))
        assert np.all(stat_function(values[:, 0])<=stat_function(values[:, 0]))

    if plot:
        lstsq_pce = PolynomialChaosExpansion()
        lstsq_pce.configure({'poly_type':'hermite', 'var_trans':var_trans})
        lstsq_pce.set_indices(indices)

        lstsq_coef = solve_least_squares_regression(
            samples, values, lstsq_pce.basis_matrix)
        lstsq_pce.set_coefficients(lstsq_coef)

        #axs[1].plot(ygrid,stat_function(values[:,0]),'ko',ms=12)
        #axs[1].plot(ygrid,stat_function(pce_values),'rs')
        #axs[1].plot(ygrid,stat_function(lstsq_pce(samples)[:,0]),'b*')

        ylb,yub = values.min()-abs(values.max())*.1,\
                  values.max()+abs(values.max())*.1

        ygrid = np.linspace(ylb, yub, 101)
        ygrid = np.sort(np.concatenate([ygrid, pce_values]))
        if disutility is not None:
            if disutility:
                ygrid = -ygrid[::-1]
            stat_function = partial(compute_conditional_expectations,
                                    ygrid, disutility_formulation=disutility)
        else:
            def stat_function(x):
                assert x.ndim == 1
                #vals = sd_opt_problem.smoother1(
                #x[np.newaxis,:]-ygrid[:,np.newaxis]).mean(axis=1)
                vals = EmpiricalCDF(x)(ygrid)
                return vals

        fig, axs=plot_1d_functions_and_statistics(
            [f, pce, lstsq_pce], ['Exact', 'SSD', 'Lstsq'], samples, values,
            stat_function, ygrid)

        plt.show()

def help_check_stochastic_dominance_gradients(sd_opt_problem):

    np.random.seed(1)
    xx = sd_opt_problem.init_guess
    if hasattr(sd_opt_problem, "eps"):
        # smoothers often only have nonzero derivative is a region or
        # diameter epsilon
        xx[0]-=sd_opt_problem.eps/10

    from pyapprox.optimization import approx_jacobian
    fd_jacobian = approx_jacobian(sd_opt_problem.objective, xx, epsilon=1e-8)
    jacobian = sd_opt_problem.objective_jacobian(xx)
    #check_gradients(
    #    sd_opt_problem.objective,sd_opt_problem.objective_jacobian,xx,False)
    #print('jac ex',fd_jacobian)
    #print('jac fd',jacobian)
    assert np.allclose(fd_jacobian, jacobian, atol=1e-7)

    fd_jacobian = approx_jacobian(
        sd_opt_problem.nonlinear_constraints, xx)
    jacobian = sd_opt_problem.nonlinear_constraints_jacobian(xx)
    if hasattr(jacobian,'todense'):
        jacobian = jacobian.todense()
    print('jac ex',fd_jacobian)
    print('jac fd',jacobian)
    msg = 'change x, current value is not an effective test'
    #check_gradients(
    #    sd_opt_problem.nonlinear_constraints,
    #    sd_opt_problem.nonlinear_constraints_jacobian,xx,False)
    assert not np.all(np.absolute(jacobian)<1e-15), msg 
    assert np.allclose(fd_jacobian,jacobian, atol=1e-7)

    if hasattr(sd_opt_problem,'objective_hessian'):
        hessian = sd_opt_problem.objective_hessian(xx)
        fd_hessian = approx_jacobian(sd_opt_problem.objective_jacobian,xx)
        if hasattr(hessian,'todense'):
            hessian = hessian.todense()
        assert np.allclose(hessian, fd_hessian)

    if hasattr(sd_opt_problem,'define_nonlinear_constraint_hessian'):
        at_least_one_hessian_nonzero = False
        for ii in range(sd_opt_problem.nnl_constraints):
            def grad(xx):
                row=sd_opt_problem.nonlinear_constraints_jacobian(xx)[ii, :]
                if hasattr(row, 'todense'):
                    row = np.asarray(row.todense())[0, :]
                return row

            fd_hessian = approx_jacobian(grad,xx)
            hessian=sd_opt_problem.define_nonlinear_constraint_hessian(
                xx, ii)
            #np.set_printoptions(linewidth=1000)
            #print('h',hessian)
            #print('h_fd',fd_hessian)
            if hessian is None:
                assert np.allclose(
                    fd_hessian,np.zeros_like(fd_hessian), atol=1e-7)
            else:
                if hasattr(hessian,'todense'):
                    hessian = hessian.todense()
                print(hessian, '\n', fd_hessian)
                assert np.allclose(hessian, fd_hessian, atol=1e-4, rtol=1e-4)
                if not at_least_one_hessian_nonzero:
                    at_least_one_hessian_nonzero = np.any(
                        np.absolute(hessian)<1e-15)
                at_least_one_hessian_nonzero=True

        if not at_least_one_hessian_nonzero:
            msg = 'change x, current value is not an effective test'
            assert False, msg 

    return sd_opt_problem


class TestRiskMeasures(unittest.TestCase):

    def test_smooth_max_function_gradients(self):
        smoother_type,eps=0,1e-1
        #x = np.linspace(-1,1,101)
        #plt.plot(x,smooth_max_function(smoother_type,eps,x));plt.show()
        #plt.plot(x,smooth_max_function_first_derivative(smoother_type,eps,x-0.5));plt.show()
        #plt.plot(x,smooth_max_function_second_derivative(smoother_type,eps,x-0.5));plt.show()
        x = np.array([0.01])
        errors = check_gradients(
            partial(smooth_max_function,smoother_type,eps),
            partial(smooth_max_function_first_derivative,smoother_type,eps),
            x[:,np.newaxis])

        errors = check_gradients(
            partial(smooth_max_function_first_derivative,smoother_type,eps),
            partial(smooth_max_function_second_derivative,smoother_type,eps),
            x[:,np.newaxis])
        assert errors.min()<1e-6

    def test_smooth_conditioal_value_at_risk_gradient(self):
        smoother_type,eps,alpha=0,1e-1,0.7
        samples = np.linspace(-1,1,11)
        t=0.1
        x0 = np.concatenate([samples,[t]])[:,np.newaxis]
        errors = check_gradients(
            partial(smooth_conditional_value_at_risk,smoother_type,eps,alpha),
            partial(smooth_conditional_value_at_risk_gradient,smoother_type,
                    eps,alpha),x0)
        assert errors.min()<1e-6

        weights = np.random.uniform(1,2,samples.shape[0])
        weights /= weights.sum()
        errors = check_gradients(
            partial(smooth_conditional_value_at_risk,smoother_type,eps,alpha,
                    weights=weights),
            partial(smooth_conditional_value_at_risk_gradient,smoother_type,
                    eps,alpha,weights=weights),x0)
        print(errors.min())
        assert errors.min()<1e-6

    def test_smooth_conditional_value_at_risk_composition_gradient(self):
        smoother_type,eps,alpha=0,1e-1,0.7
        nsamples,nvars=4,2
        samples = np.arange(nsamples*nvars).reshape(nvars,nsamples)
        t=0.1
        x0 = np.array([2,3,t])[:,np.newaxis]
        fun = lambda x: (np.sum((x*samples)**2,axis=0).T)[:,np.newaxis]
        jac = lambda x: 2*(x*samples**2).T

        errors = check_gradients(fun,jac,x0[:2],disp=False)
        assert (errors.min()<1e-6)

        #import pyapprox as pya
        #f = lambda x: smooth_conditional_value_at_risk_composition(smoother_type,eps,alpha,fun,jac,x)[0]
        #print(pya.approx_jacobian(f,x0))
        #print(smooth_conditional_value_at_risk_composition(smoother_type,eps,alpha,fun,jac,x0)[1])
     
        errors = check_gradients(
            partial(smooth_conditional_value_at_risk_composition,smoother_type,eps,alpha,fun,jac),
            True,x0)
        assert errors.min()<1e-7

    def test_triangle_quantile(self):
        rv_1 = triangle_rv(0.5,loc=-0.5, scale=2)

        alpha=np.array([0.3,0.75])
        assert np.allclose(rv_1.ppf(alpha),triangle_quantile(alpha,0.5,-0.5,2))
    
    def test_triangle_superquantile(self):
        c = 0.5; loc = -0.5; scale=2
        u = np.asarray([0.3,0.75])
        cvar = triangle_superquantile(u,c,loc,scale)
        samples = triangle_quantile(np.random.uniform(0,1,(100000)),c,loc,scale)
        for ii in range(len(u)):
            mc_cvar = conditional_value_at_risk(samples,u[ii])
        assert abs(cvar[ii]-mc_cvar)<1e-2

    def test_lognormal_example_exact_quantities(self):
        plot_lognormal_example_exact_quantities(int(2e5))

    def test_truncated_lognormal_example_exact_quantities(self):
        plot_truncated_lognormal_example_exact_quantities(int(2e5))

    def test_value_at_risk_normal(self):
        weights=None
        alpha=0.8
        num_samples = int(1e2)
        samples = np.random.normal(0,1,num_samples)
        samples = np.arange(1,num_samples+1)#[np.random.permutation(num_samples)]
        #xx = np.sort(samples)
        #VaR,VaR_index = value_at_risk(xx,alpha,weights,samples_sorted=True)
        xx = samples
        VaR,VaR_index = value_at_risk(xx,alpha,weights,samples_sorted=False)
        sorted_index = int(np.ceil(alpha*num_samples)-1)
        I = np.argsort(samples)
        index = I[sorted_index]
        print(index,VaR_index)
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
        weights/=weights.sum()
        
        empirical_VaR,__ = value_at_risk(
            samples,alpha,weights,samples_sorted=False)
        #print('VaR',VaR(alpha),empirical_VaR)
        assert np.allclose(VaR(alpha),empirical_VaR,rtol=1e-2)

        empirical_CVaR = conditional_value_at_risk(
            samples,alpha,weights)
        #print('CVaR',CVaR(alpha),empirical_CVaR)
        assert np.allclose(CVaR(alpha),empirical_CVaR,rtol=1e-2)

    def test_equivalent_formulations_of_cvar(self):
        mu,sigma=0,1
        f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
            get_lognormal_example_exact_quantities(mu,sigma)

        eps=1e-15
        lbx,ubx=normal_rv.ppf([eps,1-eps],loc=mu,scale=sigma)
        lbx,ubx = -np.inf,np.inf
        tol= 4*np.finfo(float).eps
        alpha=.8
        t = VaR(alpha)
        #fun1 = lambda x: np.maximum(f(x)-t,0)*normal_rv.pdf(x,loc=mu,scale=sigma)
        def fun1(x):
            x = np.atleast_1d(x)
            pdf_vals = normal_rv.pdf(x,loc=mu,scale=sigma)
            # Next line avoids problems caused by f(x) being to large or nan        
            I = np.where(pdf_vals>0)[0]
            vals = np.zeros_like(x)
            vals[I] = np.maximum(f(x[I])-t,0)*pdf_vals[I]
            return vals
        cvar1=t+1./(1.-alpha)*integrate.quad(
            fun1,lbx,ubx,epsabs=tol,epsrel=tol,full_output=True)[0]

        #fun2 = lambda x: f(x)*normal_rv.pdf(x,loc=mu,scale=sigma)
        def fun2(x):
            x = np.atleast_1d(x)
            pdf_vals = normal_rv.pdf(x,loc=mu,scale=sigma)
            # Next line avoids problems caused by f(x) being to large or nan
            I = np.where(pdf_vals>0)[0]
            vals = np.zeros_like(x)
            vals[I] = f(x[I])*pdf_vals[I]
            return vals
        x_cdf = partial(normal_rv.cdf,loc=mu,scale=sigma)
        integral = integrate.quad(
            fun2,np.log(t),ubx,epsabs=tol,epsrel=tol,full_output=True)[0]
        cvar2 = t+1./(1.-alpha)*(integral-t*(1-x_cdf(np.log(t))))
        #fun3=lambda x: np.absolute(f(x)-t)*normal_rv.pdf(x,loc=mu,scale=sigma)
        def fun3(x):
            x = np.atleast_1d(x)
            pdf_vals = normal_rv.pdf(x,loc=mu,scale=sigma)
            # Next line avoids problems caused by f(x) being to large or nan
            I = np.where(pdf_vals>0)[0]
            vals = np.zeros_like(x)
            vals[I] = np.absolute(f(x[I])-t)*pdf_vals[I]
            return vals

        mean = CVaR(0)
        integral = integrate.quad(
            fun3,lbx,ubx,epsabs=tol,epsrel=tol,full_output=True)[0]
        cvar3 = t+0.5/(1.-alpha)*(integral+mean-t)

        lbx, ubx =- np.inf, np.inf
        value_at_risk4,cvar4 = compute_cvar_from_univariate_function(
            f, partial(normal_rv.pdf, loc=mu, scale=sigma), lbx, ubx, alpha, 5,
            tol=1e-7)

        #print(abs(cvar1-CVaR(alpha)))
        #print(abs(cvar2-CVaR(alpha)))
        #print(abs(cvar3-CVaR(alpha)))
        #print(abs(cvar4-CVaR(alpha)))
        assert np.allclose(cvar1, CVaR(alpha))
        assert np.allclose(cvar2, CVaR(alpha))
        assert np.allclose(cvar3, CVaR(alpha))
        assert np.allclose(cvar4, CVaR(alpha))

    @skiptest
    def test_second_order_stochastic_dominance(self):
        np.random.seed(4)
        solver = partial(solve_SSD_constrained_least_squares, return_full=True)
        help_check_stochastic_dominance(solver, 10, 2, False)

    def test_disutility_second_order_stochastic_dominance(self):
        np.random.seed(2)
        # slsqp needs more testing. Dont think it is working, e.g. try
        solver = partial(solve_disutility_SSD_constrained_least_squares_slsqp,
                         return_full=True)
        help_check_stochastic_dominance(solver, 10, 2, True)

        solver = partial(
            solve_disutility_SSD_constrained_least_squares_trust_region,
            return_full=True)
        help_check_stochastic_dominance(solver, 10, 2, True)

        solver = partial(
            solve_disutility_SSD_constrained_least_squares_smooth,
            smoother_type=0, return_full=True)
        help_check_stochastic_dominance(solver, 10, 2, True)

        solver = partial(
            solve_disutility_SSD_constrained_least_squares_smooth,
            smoother_type=1, return_full=True)  
        help_check_stochastic_dominance(solver, 10, 2, True)

    @skiptest_rol
    def test_first_order_stochastic_dominance_rol(self):
        np.random.seed(4)
        solver=partial(
            solve_FSD_constrained_least_squares_smooth, eps=1e-6,
            return_full=True, smoother_type=0,
            method='rol-trust-constr')
        #help_check_stochastic_dominance(solver, 20, 3)

        optim_options = {'maxiter':100, 'verbose':3, 'ctol':1e-6, 'xtol':0,
                         'gtol':1e-6}
        solver=partial(
            solve_FSD_constrained_least_squares_smooth, eps=1e-3,
            return_full=True, smoother_type=2, optim_options=optim_options,
            method='rol-trust-constr')
        help_check_stochastic_dominance(solver, 10, 1, plot=False)


    def test_conditional_value_at_risk(self):
        N = 6
        p = np.ones(N)/N
        X = np.random.normal(0, 1, N)
        #X = np.arange(1,N+1)
        X = np.sort(X)
        beta = 7/12
        i_beta_exact = 3
        VaR_exact = X[i_beta_exact]
        cvar_exact = 1/5*VaR_exact+2/5*(np.sort(X)[i_beta_exact+1:]).sum()
        ecvar,evar = conditional_value_at_risk(
            X,beta,return_var=True)
        #print(cvar_exact,ecvar)
        assert np.allclose(cvar_exact, ecvar)

    def test_conditional_value_at_risk_subgradient(self):
        N = 6
        p = np.ones(N)/N
        X = np.random.normal(0, 1, N)
        #X = np.arange(1,N+1)
        X = np.sort(X)
        beta = 7/12
        beta = 2/3
        i_beta_exact=  3
        VaR_exact = X[i_beta_exact]
        cvar_exact = 1/5*VaR_exact+2/5*(np.sort(X)[i_beta_exact+1:]).sum()
        cvar_grad = conditional_value_at_risk_subgradient(X,beta)
        from pyapprox.optimization import approx_jacobian
        func = partial(conditional_value_at_risk,alpha=beta)
        cvar_grad_fd = approx_jacobian(func,X)
        assert np.allclose(cvar_grad, cvar_grad_fd,atol=1e-7)
        
    def test_conditional_value_at_risk_using_opitmization_formula(self):
        """
        Compare value obtained via optimization and analytical formula
        """
        plot=False
        num_samples = 5
        alpha = np.array([1./3.,0.5,0.85])
        samples = np.random.normal(0,1,(num_samples))
        for ii in range(alpha.shape[0]):
            ecvar,evar = conditional_value_at_risk(
                samples,alpha[ii],return_var=True)

            objective = lambda tt: np.asarray([t+1/(1-alpha[ii])*np.maximum(
                0,samples-t).mean() for t in tt])

            tol=1e-8
            method='L-BFGS-B'; options={'disp':False,'gtol':tol,'ftol':tol}
            init_guess = 0
            result=minimize(objective,init_guess,method=method,
                            options=options)
            value_at_risk = result['x']
            cvar = result['fun']
            print(alpha[ii],value_at_risk,cvar,ecvar)
            assert np.allclose(ecvar,cvar)
            assert np.allclose(evar,value_at_risk)

            if plot:
                rv = normal_rv
                lb,ub = rv.interval(.99)
                xx = np.linspace(lb,ub,101)
                obj_vals = objective(xx)
                plt.plot(xx,obj_vals)
                plt.plot(value_at_risk,cvar,'ro')
                plt.show()

    def test_compute_conditional_expectations(self):
        num_samples = 5
        samples = np.random.normal(0,1,(num_samples))
        eta = samples

        values = compute_conditional_expectations(eta,samples,True)
        values1 = [np.maximum(0,eta[ii]+samples).mean()
                   for ii in range(eta.shape[0])]
        assert np.allclose(values,values1)

        values = compute_conditional_expectations(eta,samples,False)
        values1 = [np.maximum(0,eta[ii]-samples).mean()
                   for ii in range(eta.shape[0])]
        assert np.allclose(values,values1)

    def setup_sd_opt_problem(self,SDOptProblem):
        from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
        from pyapprox.variable_transformations import \
            define_iid_random_variable_transformation
        from pyapprox.indexing import compute_hyperbolic_indices
        
        num_vars=1
        mu,sigma=0,1
        f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
            get_lognormal_example_exact_quantities(mu,sigma)
        
        nsamples=4
        degree=2
        samples = np.random.normal(0,1,(1,nsamples))
        values = f(samples[0,:])[:,np.newaxis]

        pce = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            normal_rv(mu,sigma),num_vars) 
        pce.configure({'poly_type':'hermite','var_trans':var_trans})
        indices = compute_hyperbolic_indices(1,degree,1.)
        pce.set_indices(indices)
        
        basis_matrix = pce.basis_matrix(samples)
        probabilities = np.ones((nsamples))/nsamples
        
        sd_opt_problem = SDOptProblem(
            basis_matrix,values[:,0],values[:,0],probabilities)
        return sd_opt_problem
    
    def test_stochastic_second_order_dominance_gradients(self):
        sd_opt_problem = self.setup_sd_opt_problem(
            TrustRegionDisutilitySSDOptProblem)
        help_check_stochastic_dominance_gradients(sd_opt_problem)

        sd_opt_problem =self.setup_sd_opt_problem(SLSQPDisutilitySSDOptProblem)
        help_check_stochastic_dominance_gradients(sd_opt_problem)

        sd_opt_problem = self.setup_sd_opt_problem(
            SmoothDisutilitySSDOptProblem)
        help_check_stochastic_dominance_gradients(sd_opt_problem)
        
    def test_fsd_gradients(self):
        np.random.seed(5)
        np.random.seed(3)
        fsd_opt_problem = self.setup_sd_opt_problem(FSDOptProblem)
        fsd_opt_problem.smoother_type = 2

        # import matplotlib.pyplot as plt
        # fsd_opt_problem.eps=1e-1
        # fsd_opt_problem.smoother_type=1
        # xx = np.linspace(-5*fsd_opt_problem.eps,5*fsd_opt_problem.eps,101)
        # plt.plot(xx,fsd_opt_problem.smooth_heaviside_function(xx),'k')
        # plt.plot(xx,fsd_opt_problem.shifted_smooth_heaviside_function(xx),'b')
        # plt.plot(xx,fsd_opt_problem.left_heaviside_function(xx),'r')
        # plt.show()
        # assert False
                 
        # test gradients of smoothing function
        fsd_opt_problem.eps = 1e-1
        from scipy.optimize import approx_fprime
        xx = -np.ones(1)*0.09
        # make sure xx will produce non zero grad
        assert -xx < fsd_opt_problem.eps and -xx != fsd_opt_problem.eps/2
        fd_grad = approx_fprime(
            xx, fsd_opt_problem.smooth_heaviside_function, 1e-7)
        grad = fsd_opt_problem.smooth_heaviside_function_first_derivative(xx)
        #print(grad, fd_grad)
        assert np.allclose(fd_grad, grad)

        xx = np.ones(1)*0.09
        # make sure xx will produce non zero grad
        assert xx < fsd_opt_problem.eps and xx != fsd_opt_problem.eps/2
        fd_grad = approx_fprime(
            xx,fsd_opt_problem.shifted_smooth_heaviside_function, 1e-7)
        grad = fsd_opt_problem.shifted_smooth_heaviside_function_first_derivative(xx)
        #print(fd_grad, grad)
        assert np.allclose(fd_grad, grad)

        fd_hess = approx_fprime(
            xx,fsd_opt_problem.smooth_heaviside_function_first_derivative, 1e-7)
        hess = fsd_opt_problem.smooth_heaviside_function_second_derivative(xx)
        #print(fd_hess, hess)
        assert np.allclose(fd_hess, hess)

        fd_hess = approx_fprime(
            xx,fsd_opt_problem.shifted_smooth_heaviside_function_first_derivative, 1e-7)
        hess = fsd_opt_problem.shifted_smooth_heaviside_function_second_derivative(xx)
        assert np.allclose(fd_hess, hess)

        # fsd_opt_problem.eps=1e-5
        # import matplotlib.pyplot as plt
        # xx = np.ones(fsd_opt_problem.nunknowns)
        # fsd_opt_problem.nonlinear_constraints(xx)
        # #plt.plot(fsd_opt_problem.values, fsd_opt_problem.constraint_rhs,'rs')
        # from pyapprox.density import EmpiricalCDF
        # ecdf = EmpiricalCDF(fsd_opt_problem.values)
        # plt.plot(fsd_opt_problem.values,ecdf(fsd_opt_problem.values), 'ko')
        # smooth_heaviside_ecdf_vals = np.asarray([fsd_opt_problem.smooth_heaviside_function(fsd_opt_problem.values-fsd_opt_problem.values[ii]) for ii in range(fsd_opt_problem.values.shape[0])]).mean(axis=1)
        # plt.plot(fsd_opt_problem.values, smooth_heaviside_ecdf_vals,'b*')
        # heaviside_ecdf_vals = np.asarray([fsd_opt_problem.left_heaviside_function(fsd_opt_problem.values-fsd_opt_problem.values[ii]) for ii in range(fsd_opt_problem.values.shape[0])]).mean(axis=1)
        # #print(heaviside_ecdf_vals-fsd_opt_problem.constraint_rhs)
        # plt.plot(fsd_opt_problem.values, heaviside_ecdf_vals,'g+')
        # plt.show()

        # fsd_opt_problem.smoother_type=1
        # xx = np.linspace(-fsd_opt_problem.eps,fsd_opt_problem.eps,101)
        # plt.plot(xx,fsd_opt_problem.smooth_heaviside_function(xx))
        # fsd_opt_problem.smoother_type=2
        # plt.plot(xx,fsd_opt_problem.smooth_heaviside_function(xx))
        # plt.show()
        
        help_check_stochastic_dominance_gradients(fsd_opt_problem)

    def test_quantile_regression(self):
        np.random.seed(1)
        nbasis = 20
        def func(x):
            return (1+x-x**2+x**3).T
        samples = np.random.uniform(-1, 1, (1, 201))
        values = func(samples)
        def eval_basis_matrix(x):
            return (x**np.arange(nbasis)[:, None]).T
        tau = 0.75
        quantile_coef = solve_quantile_regression(
            tau, samples, values, eval_basis_matrix)
        true_coef = np.zeros((nbasis))
        true_coef[:4] = [1, 1, -1, 1]
        assert np.allclose(quantile_coef[:, 0], true_coef)

    def test_second_order_stochastic_dominance_constraints(self):
        np.random.seed(2)
        nbasis = 5
        def func(x):
            return (1+x-x**2+x**3).T
        samples = np.random.uniform(-1, 1, (1, 20))
        values = func(samples)
        def eval_basis_matrix(x):
            return (x**np.arange(nbasis)[:, None]).T
        tau = 0.75
        tol = 1e-14
        eps = 1e-3
        optim_options = {'verbose': 3, 'maxiter':2000,
                         'gtol':tol, 'xtol':tol, 'barrier_tol':tol}
        ssd_coef = solve_disutility_SSD_constrained_least_squares_smooth(
            samples, values, eval_basis_matrix, optim_options=optim_options,
            eps=eps)
        true_coef = np.zeros((nbasis))
        true_coef[:4] = [1, 1, -1, 1]
        #print(ssd_coef)
        approx_vals = eval_basis_matrix(samples).dot(ssd_coef)
        #print(approx_vals.max(), values.max())
        assert approx_vals.max() >= values.max()
        assert approx_vals.mean() >= values.mean()
        assert np.allclose(ssd_coef, true_coef, atol=1e-5)

    def test_first_order_stochastic_dominance_constraints(self):
        np.random.seed(1)
        nbasis = 5
        def func(x):
            return (1+x-x**2+x**3).T
        samples = np.random.uniform(-1, 1, (1, 10))
        values = func(samples)
        def eval_basis_matrix(x):
            return (x**np.arange(nbasis)[:, None]).T
        tol = 1e-6
        eps = 1e-3
        method = 'trust-constr'
        optim_options = {'verbose': 1, 'maxiter':100,
                         'gtol':tol, 'xtol':tol, 'barrier_tol':tol}
        fsd_coef = solve_FSD_constrained_least_squares_smooth(
            samples, values, eval_basis_matrix, eps=eps,
            optim_options=optim_options, method=method, smoother_type=2)
        true_coef = np.zeros((nbasis))
        true_coef[:4] = [1, 1, -1, 1]
        #print(fsd_coef)
        assert np.allclose(fsd_coef, true_coef, atol=2e-3)


def compute_quartic_spline_of_right_heaviside_function():
    """
    Get spline approximation of step function enforcing all derivatives 
    at 0 and 1 are zero
    """
    from scipy.interpolate import BPoly
    poly = BPoly.from_derivatives([0, 1], [[0,0,0,0], [1,0,0,0]],orders=[4])
    poly = BPoly.from_derivatives([0, 1], [[0,0,0], [1,0,0]],orders=[3])
    basis = lambda x,p: x[:,np.newaxis]**np.arange(p+1)[np.newaxis,:]

    interp_nodes=(-np.cos(np.linspace(0,np.pi,5))+1)/2
    basis_mat = basis(interp_nodes,4)
    coef = np.linalg.inv(basis_mat).dot(poly(interp_nodes))
    print(coef)
    xx=np.linspace(0,1,101)
    print(np.absolute(basis(xx,4).dot(coef)-poly(xx)).max())
    #plt.plot(xx,basis(xx,4).dot(coef))
    #plt.plot(xx,poly(xx))

    eps=0.1
    a,b=0,eps
    xx=np.linspace(a,b,101)
    plt.plot(xx,basis((xx-a)/(b-a),4).dot(coef))
    f = lambda x: 6*((xx)/eps)**2-8*((xx)/eps)**3+3*((xx)/eps)**4
    plt.plot(xx,f(xx))
    plt.show()

             


if __name__== "__main__":    
    risk_measures_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestRiskMeasures)
    unittest.TextTestRunner(verbosity=2).run(risk_measures_test_suite)
