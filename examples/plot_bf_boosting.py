#%%
# define imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
# use reload for debugging
# %load_ext autoreload 
# %autoreload 2
from pyapprox.surrogates.approximate import adaptive_approximate, approximate
from pyapprox.interface.wrappers import ModelEnsemble
from pyapprox.benchmarks import setup_benchmark
from pyapprox.surrogates.polychaos.gpc import PolynomialChaosExpansion, define_poly_options_from_variable_transformation
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.variables.transforms import AffineTransform
from pyapprox.interface.wrappers import WorkTrackingModel, TimerModel
#%%
## Define example
no_samp = 1000 
max_degree = 4
sketch_sz = 18
no_trials = 100 
no_runs = 1000 
np.random.seed(1)
## Set up high and low fidelity models 
time_scenario = {
    "final_time": 1.0,
    "butcher_tableau": "im_crank2",
    "deltat": 0.1,  # default will be overwritten
    "init_sol_fun": None,
    "sink": None
}
config_values = [
    [20,10],
    [10],
    [0.125]
]
benchmark = setup_benchmark(
    "multi_index_advection_diffusion",
    kle_nvars=3, kle_length_scale=0.5,
    time_scenario=time_scenario, config_values=config_values)
variables = benchmark.variable 
# Define samples to use for each problem
samples = variables.rvs(no_samp)
# get the same set of indices for everyone
var_trans = AffineTransform(benchmark.variable)
poly_opts = define_poly_options_from_variable_transformation(var_trans)
indices = compute_hyperbolic_indices(var_trans.num_vars(), max_degree)
# In practice we would do this 
funs = [WorkTrackingModel(
    TimerModel(fun), base_model=fun) for fun in reversed(benchmark.funs)]
model_ensemble = ModelEnsemble(funs, ["hi", "lo"])
#%% for speed of this experiment, cheat
# eval rhs for lofi and hifi models
print("Eval lofi")
start = time.time()
b_lofi = model_ensemble.functions[1](samples)
elapsed = time.time() - start
print(f"  elapsed {elapsed} s")
print("Eval hifi")
start = time.time()
b_hifi = model_ensemble.functions[0](samples)
elapsed = time.time() - start
print(f"  elapsed {elapsed} s")
#%%
# Define cheat stuff
class CheatModelEnsemble(ModelEnsemble):
    def __init__(self, arrays, names):
        self.functions = arrays 
        self.names = names
model_ensemble_cheat = CheatModelEnsemble([b_hifi,b_lofi],["hi","lo"])
# helper function to evauate relative error of difference pce
def getError(sketcher_or_booster):
    Ax = sketcher_or_booster['approx'].value(samples)
    return np.linalg.norm(Ax - b_hifi, 2) / np.linalg.norm(b_hifi, 2)
#%%
# Define deterministic models
full_pce_res = approximate(
    samples, 
    b_hifi, 
    'polynomial_chaos', 
    options={
        'basis_type':'fixed',
        'variable':benchmark.variable, 
        'options':{
            'indices':indices,
            'solver_type':'lstsq'
        }, 
        'poly_opts':poly_opts
    }
)
# sketchers
# deterministic algorithm (for reference)
qr_sketched_res = adaptive_approximate(
    b_hifi, 
    benchmark.variable, 
    'polynomial_chaos',
    options={
        'method':'sketched', 
        'options':{
            'degree':max_degree, 
            'samples':samples, 
            'sketch_sz':sketch_sz, 
            'sketch_type': "qr"
        }
    }
)
## Get error from deterministic models
# Full linear least squares problem error
error_unsketched = getError(full_pce_res)
# QR sketching is deterministic, so we need only do it once 
error_qr = getError(qr_sketched_res)
#%%
# Stochastic Models
## The other sketching techniques require sampling from different catagorical distributions, so we do so no_runs times 
error_unif = np.zeros(no_runs)
error_lev = np.zeros(no_runs)
error_unif_boosted = np.zeros(no_runs)
error_lev_boosted = np.zeros(no_runs)
for run_ix in tqdm(range(no_runs)):
    # stochastic algorithms
    unif_sketched_res = adaptive_approximate(
        b_hifi, 
        benchmark.variable, 
        'polynomial_chaos',
        options={
            'method':'sketched', 
            'options':{
                'degree':max_degree, 
                'samples':samples, 
                'sketch_sz':sketch_sz, 
                'sketch_type': "uniform"
            }
        }
    )
    lev_sketched_res = adaptive_approximate(
        b_hifi, 
        benchmark.variable, 
        'polynomial_chaos',
        options={
            'method':'sketched', 
            'options':{
                'degree':max_degree, 
                'samples':samples, 
                'sketch_sz':sketch_sz, 
                'sketch_type': "leverage_score"
            }
        }
    )
    # bifi boosters (both are stochastic)
    bifi_unif_sketched_res = adaptive_approximate(
        model_ensemble_cheat, 
        benchmark.variable, 
        'polynomial_chaos',
        options={
            'method':'bf_boosted', 
            'options':{
                'degree':max_degree, 
                'samples':samples, 
                'no_trials':no_trials,
                'sketch_sz':sketch_sz, 
                'sketch_type': "uniform"
            }
        }
    )
    bifi_lev_sketched_res = adaptive_approximate(
        model_ensemble_cheat, 
        benchmark.variable, 
        'polynomial_chaos',
        options={
            'method':'bf_boosted', 
            'options':{
                'degree':max_degree, 
                'samples':samples, 
                'no_trials':no_trials,
                'sketch_sz':sketch_sz, 
                'sketch_type': "leverage_score"
            }
        }
    )
    error_unif[run_ix] = getError(unif_sketched_res)
    error_lev[run_ix] = getError(lev_sketched_res)
    error_unif_boosted[run_ix] = getError(bifi_unif_sketched_res)
    error_lev_boosted[run_ix] = getError(bifi_lev_sketched_res)

# %%
# plot
err_name = np.concatenate(
    [
        np.repeat('Uniform', no_runs), 
        np.repeat('Leverage Score', no_runs),
        np.repeat('Uniform', no_runs), 
        np.repeat('Leverage Score', no_runs)
    ]
)
err_boosted = np.concatenate(
    [
        np.repeat('Regular', no_runs), 
        np.repeat('Regular', no_runs),
        np.repeat('Boosted', no_runs), 
        np.repeat('Boosted', no_runs)
    ]
)
err_val = np.concatenate([error_unif, error_lev, error_unif_boosted, error_lev_boosted])
dic = {'Error':err_val,'Sampling Method':err_name,'boosted':err_boosted}
df = pd.DataFrame(dic,index = np.arange(4*no_runs))

fig, axes = plt.subplots(1,1,figsize=(16,8),constrained_layout=True)
fig.suptitle('Isigami Comparison',fontsize=20)
fg = sns.boxplot(y='Error',x='Sampling Method',hue='boosted',data=df,ax=axes)
fg.axhline(error_qr,c='b',alpha=0.5,label="QR")
fg.axhline(error_unsketched,c='y',alpha=0.5,label="Full Problem")
fg.set_title(f'Max Degree = {max_degree}; Sketch Size=18',fontsize=14)
fg.set_yscale('log')
fg.legend(title=None,loc=1)
# %%
