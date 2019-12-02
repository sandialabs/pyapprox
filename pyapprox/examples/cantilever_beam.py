import scipy.stats as ss
from pyapprox.optimization import *

def define_beam_random_variables():
    # traditional parameterization
    X = ss.norm(loc=500,scale=np.sqrt(100)**2)
    Y = ss.norm(loc=1000,scale=np.sqrt(100)**2)
    E = ss.norm(loc=2.9e7,scale=np.sqrt(1.45e6)**2)
    R = ss.norm(loc=40000,scale=np.sqrt(2000)**2)

    # increased total variance contribution from E
    X = ss.norm(loc=500,scale=np.sqrt(100)**2/10)
    Y = ss.norm(loc=1000,scale=np.sqrt(100)**2/10)
    E = ss.norm(loc=2.9e7,scale=np.sqrt(1.45e6)**2)
    R = ss.norm(loc=40000,scale=np.sqrt(2000)**2/10)

    variable = IndependentMultivariateRandomVariable([X,Y,E,R])
    return variable

def beam_obj(uq_samples,design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    w,t = design_samples
    return w*t

def beam_constraint_I(uq_samples,design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    w,t     = design_samples # width, height
    X,Y,E,R = uq_samples     # external forces, modulus, yield strength
    L = 100                  # length of beam
    vals = 1-6*L/(w*t)*(X/w+Y/t)/R # scaled version
    return vals

def beam_constraint_II(uq_samples,design_samples):
    uq_samples,design_samples = check_inputs(uq_samples,design_samples)
    w,t     = design_samples # width, height
    X,Y,E,R = uq_samples     # external forces, modulus, yield strength
    L = 100                  # length of beam
    D0 = 2.2535
    vals = 1-4*L**3/(E*w*t)*np.sqrt(X**2/w**4+Y**2/t**4)/D0 # scaled version
    return vals

def setup_beam_design():
    uq_vars = define_beam_random_variables()
    init_design_sample = np.array([[2,2]]).T
    #bounds = Bounds([1,1], [4,4])
    bounds = [[1,4], [1,4]]
    quantile_lower_bound=0
    constraint_functions = [beam_constraint_I,beam_constraint_II]
    objective = partial(beam_obj,None)
    return uq_vars, objective, constraint_functions, bounds, init_design_sample

def plot_beam_design(objective,constraints,constraint_functions,
                     uq_samples,design_sample,res,opt_history,
                     label,
                     fig_pdf=None,axs_pdf=None,
                     fig_cdf=None,axs_cdf=None):
    quantile=0.1
    fig_opt,axs_opt = plot_optimization_history(
        objective,constraints,uq_samples,opt_history,[1,4,1,4])
    plot_constraint_pdfs(constraint_functions,uq_samples,design_sample,
                        fig_pdf=fig_pdf,axs_pdf=axs_pdf,
                        label='deterministic',color='b')
    plot_constraint_cdfs(constraints,constraint_functions,uq_samples,
                        design_sample,quantile,fig_cdf=fig_cdf,
                        axs_cdf=axs_cdf,
                        label='deterministic',color='b')
    return res.fun


from pyapprox.probability_measure_sampling import \
    generate_independent_random_samples
from pyapprox.variables import IndependentMultivariateRandomVariable
def find_deterministic_beam_design():
    lower_bound = 0

    uq_vars, objective, constraint_functions, bounds, init_design_sample = \
        setup_beam_design()

    uq_nominal_sample = uq_vars.get_statistics('mean')
    #constraints_info = [{'type':'deterministic',
    #                     'lower_bound':lower_bound}]*len(constraint_functions)
    #constraints = setup_inequality_constraints(
    #    constraint_functions,constraints_info,uq_nominal_sample)

    constraint_info = {'lower_bound':lower_bound,
                       'uq_nominal_sample':uq_nominal_sample}
    individual_constraints = [
        DeterministicConstraint(f,constraint_info)
        for f in constraint_functions]
    constraints = MultipleConstraints(individual_constraints)
    
    optim_options={'ftol': 1e-9, 'disp': 3, 'maxiter':1000}
    res, opt_history = run_design(
        objective,init_design_sample,constraints,bounds,optim_options)

    nsamples = 10000
    uq_samples = generate_independent_random_samples(uq_vars,nsamples)
    return objective,constraints,constraint_functions,uq_samples,res,\
        opt_history

def find_uncertainty_aware_beam_design(constraint_type='quantile'):

    quantile=0.1
    print(quantile)
    quantile_lower_bound = 0

    uq_vars, objective, constraint_functions, bounds, init_design_sample = \
        setup_beam_design()
    if constraint_type=='quantile':
        constraint_info = [{'type':'quantile',
                             'lower_bound':quantile_lower_bound,
                             'quantile':quantile}]*len(constraint_functions)
    elif constraint_type=='cvar':
        constraint_info = [{'type':'cvar',
                             'lower_bound':0.05,
                             'quantile':quantile,
                             'smoothing_eps':1e-1}]*len(
                                 constraint_functions)
    else:
        raise Exception()

    nsamples=10000
    uq_samples = generate_independent_random_samples(uq_vars,nsamples)
    #constraints = setup_inequality_constraints(
    #    constraint_functions,constraint_info,uq_samples)

    def generate_samples():
        # always use the same samples avoid noise in constraint vals
        np.random.seed(1)
        return uq_samples
    individual_constraints = [
        MCStatisticConstraint(f,generate_samples,constraint_info[0])
        for f in constraint_functions]
    constraints = MultipleConstraints(individual_constraints)


    optim_options={'ftol': 1e-9, 'disp': 3, 'maxiter':1000}
    res, opt_history = run_design(
        objective,init_design_sample,constraints,bounds,optim_options)

    return objective,constraints,constraint_functions,uq_samples,res,\
        opt_history



if __name__=='__main__':
    objective,constraints,constraint_functions,uq_samples,res,opt_history = \
        find_deterministic_beam_design()
    plot_beam_design(
        beam_obj,constraints,constraint_functions,uq_samples,
        res.x,res,opt_history,'deterministic')

    objective,constraints,constraint_functions,uq_samples,res,opt_history = \
        find_uncertainty_aware_beam_design(constraint_type='quantile')
    plot_beam_design(
        beam_obj,constraints,constraint_functions,uq_samples,
        res.x,res,opt_history,'DUU')
    plt.show()
