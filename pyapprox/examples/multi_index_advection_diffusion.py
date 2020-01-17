import numpy as np
import matplotlib.pyplot as plt
from pyapprox.utilities import halton_sequence, \
    get_all_sample_combinations
from pyapprox.models.wrappers import PoolModel, WorkTracker
import multiprocessing
from pyapprox.fenics_models.advection_diffusion import AdvectionDiffusionModel,\
    qoi_functional_misc
from pyapprox.models.wrappers import TimerModelWrapper, WorkTrackingModel
def setup_model():
    num_vars,corr_len,num_levels,periodic_boundary=10,1/2,6,False
    second_order_timestepping=False

    qoi_functional=qoi_functional_misc
    degree=1
    base_model = AdvectionDiffusionModel(
        num_vars,corr_len,1.0e-0,degree,qoi_functional,add_work_to_qoi=False,
        periodic_boundary=periodic_boundary,
        second_order_timestepping=second_order_timestepping)
    timer_model = TimerModelWrapper(base_model,base_model)
    model = PoolModel(timer_model,max_eval_concurrency,base_model=base_model)
    return model

from pyapprox.utilities import cartesian_product
from functools import partial
def error_vs_cost(model,generate_random_samples,validation_levels):
    #import sys
    #sys.setrecursionlimit(10) 
    model=WorkTrackingModel(model,model.base_model)
    num_samples=10
    validation_levels = np.asarray(validation_levels)
    assert len(validation_levels)==model.base_model.num_config_vars
    config_vars = cartesian_product(
        [np.arange(ll) for ll in validation_levels])

    random_samples=generate_random_samples(num_samples)
    samples = get_all_sample_combinations(random_samples,config_vars)

    reference_samples = samples[:,::config_vars.shape[1]].copy()
    reference_samples[-model.base_model.num_config_vars:,:]=\
            validation_levels[:,np.newaxis]

    reference_values = model(reference_samples)
    reference_mean = reference_values[:,0].mean()

    values = model(samples)

    # put keys in order returned by cartesian product
    keys = sorted(model.work_tracker.costs.keys(),key=lambda x: x[::-1])
    keys = keys[:-1]# remove validation key associated with validation samples
    costs,ndofs,means,errors = [],[],[],[]
    for ii in range(len(keys)):
        key=keys[ii]
        costs.append(np.median(model.work_tracker.costs[key]))
        nx,ny,dt = model.base_model.get_degrees_of_freedom_and_timestep(
            np.asarray(key))
        ndofs.append(nx*ny*model.base_model.final_time/dt)
        print(key,ndofs[-1],nx,ny,model.base_model.final_time/dt)
        means.append(np.mean(values[ii::config_vars.shape[1],0]))
        errors.append(abs(means[-1]-reference_mean)/abs(reference_mean))

    times = costs.copy()
    # make costs relative
    costs /= costs[-1]

    n1,n2,n3 = validation_levels
    indices=np.reshape(np.arange(len(keys),dtype=int),(n1,n2,n3),order='F')
    costs = np.reshape(np.array(costs),(n1,n2,n3),order='F')
    ndofs = np.reshape(np.array(ndofs),(n1,n2,n3),order='F')
    errors = np.reshape(np.array(errors),(n1,n2,n3),order='F')
    times = np.reshape(np.array(times),(n1,n2,n3),order='F')
    
    validation_index = reference_samples[-model.base_model.num_config_vars:,0]
    validation_time = np.median(model.work_tracker.costs[tuple(validation_levels)])
    validation_cost = validation_time/costs[-1]
    validation_ndof = np.prod(reference_values[:,-2:],axis=1)

    data = {"costs":costs,"errors":errors,"indices":indices,
            "times":times,"validation_index":validation_index,
            "validation_cost":validation_cost,"validation_ndof":validation_ndof,
            "validation_time":validation_time,"ndofs":ndofs}

    return data

def plot_error_vs_cost(data,cost_type='ndof'):
    
    print(data['times'][-1,-1,-1])
    errors,costs,indices=data['errors'],data['costs'],data['indices']

    if cost_type=='ndof':
        costs = data['ndofs']/data['ndofs'].max()
    Z = costs[:,:,-1]
    print(costs.shape)
    print('validation_time',data['validation_time'],data['validation_index'])
    from pyapprox.convert_to_latex_table import convert_to_latex_table
    #column_indices = [np.arange(3),np.arange(validation_level)[-3:]]
    #time_index = [0,-1]
    # for ii in range(len(column_indices)):
    #    Z = costs[:,column_indices[ii],time_index[ii]]
    #    column_labels = [str(jj+1) for jj in column_indices[ii]]
    #    row_labels = np.arange(1,Z.shape[0]+1)
    #    convert_to_latex_table(
    #        Z,
    #        'advection-diffusion-cost-table-%d-l-%d.tex'%(ii,validation_level),
    #        column_labels,
    #        row_labels,None,output_dir='texdir',
    #        bold_entries=None,
    #        corner_labels=[r'$\alpha_1$',r'$\alpha_2$'])

    import matplotlib as mpl
    mpl.rcParams['axes.labelsize'] = 30
    mpl.rcParams['axes.titlesize'] = 30
    mpl.rcParams['xtick.labelsize'] = 30
    mpl.rcParams['ytick.labelsize'] = 30

    validation_levels = costs.shape
    fig,axs = plt.subplots(1,len(validation_levels),
                           figsize=(len(validation_levels)*8,6),
                           sharey=True)
    if len(validation_levels)==1:
        label=r'$(\cdot)$'
        axs.loglog(costs,errors,'o-',label=label)
    if len(validation_levels)==2:
        for ii in range(validation_levels[0]):
            label=r'$(\cdot,%d)$'%(ii)
            axs[0].loglog(costs[:,ii],errors[:,ii],'o-',label=label)
        for ii in range(validation_levels[0]):
            label=r'$(%d,\cdot)$'%(ii)
            axs[1].loglog(costs[ii,:],errors[ii,:],'o-',label=label)
    if len(validation_levels)==3:
        for ii in range(validation_levels[1]):
            jj = costs.shape[2]-1
            label=r'$(\cdot,%d,%d)$'%(ii,jj)
            axs[0].loglog(costs[:,ii,jj],errors[:,ii,jj],'o-',label=label)
        for ii in range(validation_levels[0]):
            jj = costs.shape[2]-1
            label=r'$(%d,\cdot,%d)$'%(ii,jj)
            axs[1].loglog(costs[ii,:,jj],errors[ii,:,jj],'o-',label=label)
            jj = costs.shape[1]-1
            label=r'$(%d,%d,\cdot)$'%(ii,jj)
            axs[2].loglog(costs[ii,jj,:],errors[ii,jj,:],'o-',label=label)

        # plot expected congergence rates
        ii = validation_levels[1]-1
        jj = validation_levels[2]-1
        axs[0].loglog(costs[:,ii,jj],costs[:,ii,jj]**(-2)/costs[0,ii,jj]**(-2),':',
                      color='gray')
        ii = validation_levels[0]-1
        axs[1].loglog(costs[ii,:,jj],costs[ii,:,jj]**(-2)/costs[ii,0,jj]**(-2),':',
                      color='gray')
        jj = validation_levels[1]-1
        axs[2].loglog(costs[ii,jj,:],
                      costs[ii,jj,:]**(-1)/costs[ii,jj,0]**(-1)*1e-1,
                      ':',color='gray')
        
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    for ii in range(len(validation_levels)):
        axs[ii].set_xlabel(r'$\mathrm{Work}$ $W_{\boldsymbol{\alpha}}$')
    axs[0].set_ylabel(r'$\left\lvert \mathbb{E}[f]-\mathbb{E}[f_{\boldsymbol{\alpha}}]\right\rvert / \left\lvert \mathbb{E}[f]\right\rvert$')
    return fig,axs
    
if __name__ == '__main__':
    from pyapprox.configure_plots import *
    #max_eval_concurrency = multiprocessing.cpu_count()-2
    max_eval_concurrency = 10
    def generate_random_samples(m,n):
        samples = halton_sequence(m,0,n)
        samples = samples*2*np.sqrt(3)-np.sqrt(3)
        return samples
    model = setup_model()
    #model.cost_function = WorkTracker()
    #from pyapprox.models.wrappers import MultiLevelWrapper
    #multilevel_model=MultiLevelWrapper(
    #    model,base_model.num_config_vars,model.cost_function)
    validation_levels = [5,5,5]
    #validation_levels = [3]*3
    data = error_vs_cost(
        model,partial(generate_random_samples,model.base_model.num_vars),
        validation_levels)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdirname:
        #costs_filename = os.path.join(tmpdirname,"costs.npz")
        #np.savez(costs_filename,**data)
        #data = np.load(costs_filename)
        plot_error_vs_cost(data,'time')
        #plt.savefig('advection-diffusion-error-vs-cost.pdf')
        plt.show()

