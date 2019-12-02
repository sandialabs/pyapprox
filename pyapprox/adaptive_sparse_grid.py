from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from pyapprox.sparse_grid import *
import copy
from pyapprox.utilities import lists_of_lists_of_arrays_equal, \
    lists_of_arrays_equal, partial_functions_equal
import pickle
from pyapprox.indexing import get_forward_neighbor, get_backward_neighbor
from functools import partial
# try:
#     # Python version < 3
#     import Queue as queue
# except:
#     import queue
    
import heapq
class mypriorityqueue():
    def __init__(self):
        self.list= []

    def empty(self):
        return len(self.list)==0
    
    def put(self,item):
        heapq.heappush(self.list,item)

    def get(self):
        item = heapq.heappop(self.list)
        return item

    def __eq__(self,other):
        return other.list==self.list

    def __neq__(self,other):
        return not self.__eq__(other)

    def __repr__(self):
        return str(self.list)
    
def extract_items_from_priority_queue(pqueue):
    """
    Return the items in a priority queue. The items will only be shallow copies
    of items in queue
    
    Priority queue is thread safe so does not support shallow or deep copy
    One can copy this queue by pushing and popping by original queue will 
    be destroyed. Return a copy of the original queue that can be used to 
    replace the destroyed queue
    """
    
    #pqueue1 = queue.PriorityQueue()
    pqueue1 = mypriorityqueue()
    items = []
    while not pqueue.empty():
        item=pqueue.get()
        items.append(item)
        pqueue1.put(item)
    return items, pqueue1
  
def update_smolyak_coefficients(new_index,subspace_indices,smolyak_coeffs):
    assert new_index.ndim==1
    assert subspace_indices.ndim==2

    new_smolyak_coeffs = smolyak_coeffs.copy()

    try:
        from pyapprox.cython.adaptive_sparse_grid import \
            update_smolyak_coefficients_pyx
        return update_smolyak_coefficients_pyx(new_index,subspace_indices,
                new_smolyak_coeffs)
        # from pyapprox.weave.adaptive_sparse_grid import \
        #     c_update_smolyak_coefficients failed
        # # new_index.copy is needed
        # return c_update_smolyak_coefficients(
        #    new_index.copy(),subspace_indices,smolyak_coeffs)
    except:
        print('update_smolyak_coefficients extension failed')

    num_vars, num_subspace_indices = subspace_indices.shape
    for ii in range(num_subspace_indices):
        diff = new_index-subspace_indices[:,ii]
        if np.all(diff>=0) and diff.max()<=1:
            new_smolyak_coeffs[ii]+=(-1.)**diff.sum()
    return new_smolyak_coeffs

def add_unique_poly_indices(poly_indices_dict,new_poly_indices):
    unique_poly_indices = []
    num_unique_poly_indices = len(poly_indices_dict)
    array_indices = np.empty((new_poly_indices.shape[1]),dtype=int)
    for jj in range(new_poly_indices.shape[1]):
        poly_index = new_poly_indices[:,jj]
        key = hash_array(poly_index)
        if key not in poly_indices_dict:
            unique_poly_indices.append(poly_index)
            poly_indices_dict[key]=num_unique_poly_indices
            array_indices[jj]=num_unique_poly_indices
            num_unique_poly_indices+=1
        else:
            array_indices[jj]=poly_indices_dict[key]
    return poly_indices_dict, np.asarray(unique_poly_indices).T, array_indices

def subspace_index_is_admissible(subspace_index, subspace_indices_dict):
    if hash_array(subspace_index) in subspace_indices_dict:
        return False
    
    if subspace_index.sum()<=1:
        return True

    num_vars = subspace_index.shape[0]
    for ii in range(num_vars):
        if subspace_index[ii]>0:
            neighbor_index = get_backward_neighbor(subspace_index,ii)
            if hash_array(neighbor_index) not in subspace_indices_dict:
                return False
    return True

def max_level_admissibility_function(max_level,max_level_1d,
                                     max_num_sparse_grid_samples, error_tol,
                                     sparse_grid, subspace_index):
    if subspace_index.sum()>max_level:
        return False
    
    if error_tol is not None:
        if sparse_grid.error.sum()<error_tol*np.absolute(
                sparse_grid.values[0,0]):
            print('Desired accuracy obtained',sparse_grid.error.sum(),
                  error_tol*np.absolute(sparse_grid.values[0,0]))
            return False
        
    if max_level_1d is not None:
        for dd in range(subspace_index.shape[0]):
            if subspace_index[dd]>max_level_1d[dd]:
                #print ('Max level reached in variable dd')
                return False
    if (max_num_sparse_grid_samples is not None and
        (sparse_grid.num_equivalent_function_evaluations>
         max_num_sparse_grid_samples)):
        #print ('Max num evaluations reached')
        return False
    return True
        
def default_combination_sparse_grid_cost_function(x):
    return np.ones(x.shape[1])
        
def get_active_subspace_indices(active_subspace_indices_dict,
                                sparse_grid_subspace_indices):
    I = []
    for key in active_subspace_indices_dict:
        I.append(active_subspace_indices_dict[key])
    return sparse_grid_subspace_indices[:,I], I

def partition_sparse_grid_samples(sparse_grid):
    num_vars = sparse_grid.samples.shape[0]
   
    active_subspace_indices,active_subspace_idx = get_active_subspace_indices(
        sparse_grid.active_subspace_indices_dict,
        sparse_grid.subspace_indices)

    sparse_grid_subspace_idx = np.ones(
        (sparse_grid.subspace_indices.shape[1]),dtype=bool)
    sparse_grid_subspace_idx[active_subspace_idx]=False
    
    samples = np.empty((num_vars,0),dtype=float)
    samples_dict = dict()
    kk=0;
    for ii in np.arange(
            sparse_grid_subspace_idx.shape[0])[sparse_grid_subspace_idx]:
        subspace_index = sparse_grid.subspace_indices[:,ii]
        subspace_poly_indices = \
            sparse_grid.subspace_poly_indices_list[ii]
        subspace_max_level=subspace_index.max()
        subspace_samples = get_sparse_grid_samples(
            subspace_poly_indices,
            sparse_grid.samples_1d,
            sparse_grid.config_variables_idx)
        for jj in range(subspace_samples.shape[1]):
            key = hash_array(subspace_samples[:,jj])
            if key not in samples_dict:
                samples_dict[key]=kk
                samples = np.hstack((samples,subspace_samples[:,jj:jj+1]))
                kk+=1
                    
    num_active_samples = sparse_grid.samples.shape[1]-\
            samples.shape[1]
    active_samples_idx = np.empty((num_active_samples),dtype=int)
    kk=0
    for ii in range(sparse_grid.samples.shape[1]):
        sample = sparse_grid.samples[:,ii]
        key = hash_array(sample)
        if key not in samples_dict:
            active_samples_idx[kk]=ii
            kk+=1
    assert kk==num_active_samples

    active_samples = sparse_grid.samples[:,active_samples_idx]
    return samples,active_samples


def plot_adaptive_sparse_grid_2d(sparse_grid,plot_grid=True):
    from pyapprox.visualization import plot_2d_indices
    active_subspace_indices,active_subspace_idx = get_active_subspace_indices(
        sparse_grid.active_subspace_indices_dict,
        sparse_grid.subspace_indices)

    # get subspace indices that have been added to the sparse grid,
    # i.e are not active
    sparse_grid_subspace_idx = np.ones(
        (sparse_grid.subspace_indices.shape[1]),dtype=bool)
    sparse_grid_subspace_idx[active_subspace_idx]=False

    if plot_grid:
        f,axs=plt.subplots(1,2,sharey=False,figsize=(16, 6))
    else:
        f,axs=plt.subplots(1,1,sharey=False,figsize=(8, 6))
        axs=[axs]
        
    plot_2d_indices(
        sparse_grid.subspace_indices[:,sparse_grid_subspace_idx],
        coeffs=sparse_grid.smolyak_coefficients[sparse_grid_subspace_idx],
        other_indices=active_subspace_indices,ax=axs[0])

    if sparse_grid.config_variables_idx is not None:
        axs[0].set_xlabel(r'$\beta_1$',rotation=0)
        axs[0].set_ylabel(r'$\alpha_1$')#,rotation=0)

    if plot_grid:
        samples,active_samples = partition_sparse_grid_samples(sparse_grid)
        if sparse_grid.config_variables_idx is None:
            axs[1].plot(samples[0,:],samples[1,:],'ko')
            axs[1].plot(active_samples[0,:],active_samples[1,:],'ro')
        else:
            for ii in range(samples.shape[1]):
                axs[1].plot(samples[0,ii],samples[1,ii],'ko')
            for ii in range(active_samples.shape[1]):
                axs[1].plot(active_samples[0,ii],active_samples[1,ii],'ro')
            from matplotlib.pyplot import MaxNLocator
            ya = axs[1].get_yaxis()
            ya.set_major_locator(MaxNLocator(integer=True))
            #axs[1].set_ylabel(r'$\alpha_1$',rotation=0)
            axs[1].set_xlabel('$z_1$',rotation=0,)

def isotropic_refinement_indicator(subspace_index,
                                   num_new_subspace_samples,
                                   sparse_grid):
    return float(subspace_index.sum()), np.inf


def tensor_product_refinement_indicator(subspace_index,num_new_subspace_samples,
                                        sparse_grid):
    return float(subspace_index.max()), np.inf
    

def variance_refinement_indicator_old(subspace_index,num_new_subspace_samples,
                                      sparse_grid,normalize=True,mean_only=False):
    """
    when config index is increased but the other indices are 0 the
    subspace will only have one random sample. Thus the variance
    contribution will be zero regardless of value of the function value
    at that sample
    """
    
    #return subspace_index.sum()
    moments = sparse_grid.moments()
    smolyak_coeffs = update_smolyak_coefficients(
        subspace_index,sparse_grid.subspace_indices,
        sparse_grid.smolyak_coefficients.copy())
    new_moments = integrate_sparse_grid(
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        smolyak_coeffs,sparse_grid.weights_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    if mean_only:
        error = np.absolute(new_moments[0]-moments[0])
    else:
        error = np.absolute(new_moments[0]-moments[0])**2+\
                np.absolute(new_moments[1]-moments[1])

    indicator=error.copy()
                
    # relative error will not work if value at first grid point is close to zero
    if normalize:
        assert np.all(np.absolute(sparse_grid.values[0,:])>1e-6)
        indicator/=np.absolute(sparse_grid.values[0,:])**2

    qoi_chosen = np.argmax(indicator)
    #print (qoi_chosen)

    indicator=indicator.max()

    cost_per_sample = sparse_grid.eval_cost_function(
        subspace_index[:,np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    # compute marginal benefit 
    indicator/=cost
    
    return -indicator, error[qoi_chosen]


def variance_refinement_indicator(subspace_index,num_new_subspace_samples,
                                  sparse_grid,normalize=True,mean_only=False):
    """
    when config index is increased but the other indices are 0 the
    subspace will only have one random sample. Thus the variance
    contribution will be zero regardless of value of the function value
    at that sample
    """
    
    #return subspace_index.sum()
    moments = sparse_grid.moments()
    smolyak_coeffs = update_smolyak_coefficients(
        subspace_index,sparse_grid.subspace_indices,
        sparse_grid.smolyak_coefficients.copy())

    new_moments = sparse_grid.moments_(smolyak_coeffs)

    if mean_only:
        error = np.absolute(new_moments[0]-moments[0])
    else:
        error = np.absolute(new_moments[0]-moments[0])**2+\
                np.absolute(new_moments[1]-moments[1])

    indicator=error.copy()
                
    # relative error will not work if value at first grid point is close to zero
    if normalize:
        assert np.all(np.absolute(sparse_grid.values[0,:])>1e-6)
        indicator/=np.absolute(sparse_grid.values[0,:])**2

    qoi_chosen = np.argmax(indicator)
    #print (qoi_chosen)

    indicator=indicator.max()

    cost_per_sample = sparse_grid.eval_cost_function(
        subspace_index[:,np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    # compute marginal benefit 
    indicator/=cost
    
    return -indicator, error[qoi_chosen]


def cv_refinement_indicator(validation_samples,validation_values,
                            subspace_index,num_new_subspace_samples,
                            sparse_grid):
    smolyak_coefficients = update_smolyak_coefficients(
        subspace_index,sparse_grid.subspace_indices,
        sparse_grid.smolyak_coefficients.copy())
    approx_values = evaluate_sparse_grid(
        validation_samples[:sparse_grid.config_variables_idx,:],
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        smolyak_coefficients,sparse_grid.samples_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    cost_per_sample = sparse_grid.eval_cost_function(
        subspace_index[:,np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    error = np.linalg.norm(
        approx_values-validation_values)/np.std(validation_values)
    current_approx_values = sparse_grid(validation_samples)
    current_error = np.linalg.norm(
        current_approx_values-validation_values)/np.std(validation_values)
    error = abs(error-current_error)
    indicator = error.max()/cost
    return -indicator,error

def compute_surpluses(subspace_index,sparse_grid,hierarchical=False):
    key = hash_array(subspace_index)
    ii = sparse_grid.active_subspace_indices_dict[key]

    subspace_samples = get_subspace_samples(
        subspace_index,
        sparse_grid.subspace_poly_indices_list[ii],
        sparse_grid.samples_1d,sparse_grid.config_variables_idx,
        unique_samples_only=False)

    if hierarchical:
        hier_indices = get_hierarchical_sample_indices(
            subspace_index,
            sparse_grid.subspace_poly_indices_list[ii],
            sparse_grid.samples_1d,sparse_grid.config_variables_idx)
        subspace_samples = subspace_samples[:,hier_indices]
    else:
        hier_indices = None

    current_approx_values = evaluate_sparse_grid(
        subspace_samples,
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        sparse_grid.smolyak_coefficients,
        sparse_grid.samples_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    smolyak_coefficients = update_smolyak_coefficients(
        subspace_index,sparse_grid.subspace_indices,
        sparse_grid.smolyak_coefficients.copy())

    new_approx_values = evaluate_sparse_grid(
        subspace_samples,
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        smolyak_coefficients,
        sparse_grid.samples_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    return new_approx_values-current_approx_values, hier_indices

def compute_hierarchical_surpluses_direct(subspace_index,sparse_grid):

    # only works if not used in multilevel setting
    assert sparse_grid.config_variables_idx is None
    key = hash_array(subspace_index)
    ii = sparse_grid.active_subspace_indices_dict[key]

    subspace_samples = get_subspace_samples(
        subspace_index,
        sparse_grid.subspace_poly_indices_list[ii],
        sparse_grid.samples_1d,sparse_grid.config_variables_idx,
        unique_samples_only=False)

    hier_indices = get_hierarchical_sample_indices(
        subspace_index,sparse_grid.subspace_poly_indices_list[ii],
        sparse_grid.samples_1d,sparse_grid.config_variables_idx)
    #hier_indices = np.arange(subspace_samples.shape[1])
    
    subspace_samples = subspace_samples[:,hier_indices]

    current_approx_values = evaluate_sparse_grid(
        subspace_samples,
        sparse_grid.values,
        sparse_grid.poly_indices_dict,
        sparse_grid.subspace_indices,
        sparse_grid.subspace_poly_indices_list,
        sparse_grid.smolyak_coefficients,
        sparse_grid.samples_1d,
        sparse_grid.subspace_values_indices_list,
        sparse_grid.config_variables_idx)

    subspace_values = get_subspace_values(
        sparse_grid.values,
        sparse_grid.subspace_values_indices_list[ii])
    subspace_values = subspace_values[hier_indices,:]

    surpluses = subspace_values-current_approx_values
    return surpluses

def surplus_refinement_indicator(subspace_index,num_new_subspace_samples,
                                 sparse_grid,output=False,hierarchical=False,
                                 norm_order=np.inf):

    surpluses, hier_indices = compute_surpluses(
        subspace_index,sparse_grid,hierarchical=hierarchical)

    subspace_weights = get_subspace_weights(
        subspace_index,sparse_grid.weights_1d,sparse_grid.config_variables_idx)
    if hier_indices is not None:
        subspace_weights = subspace_weights[hier_indices]

    if norm_order==np.inf:
        error = np.max(np.abs(surpluses),axis=0)
    elif norm_order==1:
        error = np.abs(np.dot(subspace_weights,surpluses))
    else:
        raise Exception("ensure norm_order in [np.inf,1]")

    assert error.shape[0]==surpluses.shape[1]

    # relative error will not work if value at first grid point is close to zero
    assert np.all(np.absolute(sparse_grid.values[0,:])>1e-6)
    error/=np.absolute(sparse_grid.values[0,:])
    
    error = np.max(error)   

    cost_per_sample = sparse_grid.eval_cost_function(
        subspace_index[:,np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    if output:
        print (hier_indices)
        #print (sparse_grid.subspace_poly_indices_list[ii][:,hier_indices])
        print(('s',subspace_values))
        print (approx_values)
        print (subspace_weights)
        print((subspace_values-approx_values))
        print((np.dot(subspace_weights,(subspace_values-approx_values))))
        print (cost)
        print((error/cost))

    indicator=error/cost

    return -indicator,error

def convert_sparse_grid_to_polynomial_chaos_expansion(sparse_grid, pce_opts,
                                                      debug=False):
    from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
    from pyapprox.manipulate_polynomials import add_polynomials
    pce = PolynomialChaosExpansion()
    pce.configure(pce_opts)
    if sparse_grid.config_variables_idx is not None:
        assert pce.num_vars() == sparse_grid.config_variables_idx
    else:
        assert pce.num_vars() == sparse_grid.num_vars

    def get_recursion_coefficients(N,dd):
        pce.update_recursion_coefficients([N]*pce.num_vars(),pce.config_opts)
        return pce.recursion_coeffs[pce.basis_type_index_map[dd]].copy()

    coeffs_1d=[
        convert_univariate_lagrange_basis_to_orthonormal_polynomials(
            sparse_grid.samples_1d[dd],
            partial(get_recursion_coefficients,dd=dd))
        for dd in range(pce.num_vars())]
    
    indices_list = []
    coeffs_list = []
    for ii in range(sparse_grid.subspace_indices.shape[1]):
        if (abs(sparse_grid.smolyak_coefficients[ii])>np.finfo(float).eps):
            subspace_index = sparse_grid.subspace_indices[:,ii]
            poly_indices=sparse_grid.subspace_poly_indices_list[ii]
            values_indices=\
                sparse_grid.subspace_values_indices_list[ii]
            subspace_values = get_subspace_values(
                sparse_grid.values,values_indices)
            subspace_coeffs = \
                convert_multivariate_lagrange_polys_to_orthonormal_polys(
                    subspace_index,subspace_values,coeffs_1d,poly_indices,
                    sparse_grid.config_variables_idx)

            if debug:
                pce.set_indices(
                    poly_indices[:sparse_grid.config_variables_idx,:])
                pce.set_coefficients(subspace_coeffs)
                subspace_samples = get_subspace_samples(
                    subspace_index,
                    sparse_grid.subspace_poly_indices_list[ii],
                    sparse_grid.samples_1d,sparse_grid.config_variables_idx,
                    unique_samples_only=False)
                poly_values = pce(
                    subspace_samples[:sparse_grid.config_variables_idx,:])
                assert np.allclose(poly_values,subspace_values)

            coeffs_list.append(
                subspace_coeffs*sparse_grid.smolyak_coefficients[ii])
            indices_list.append(poly_indices[:sparse_grid.config_variables_idx,:])

    indices, coeffs = add_polynomials(indices_list,coeffs_list)
    pce.set_indices(indices)
    pce.set_coefficients(coeffs)
    
    return pce

def extract_sparse_grid_quadrature_rule(asg):
    num_sparse_grid_points = (
        asg.poly_indices.shape[1])
    #must initialize to zero
    weights = np.zeros((num_sparse_grid_points),dtype=float)
    samples = get_sparse_grid_samples(
        asg.poly_indices,asg.samples_1d)
    for ii in range(asg.subspace_indices.shape[1]):
        if (abs(asg.smolyak_coefficients[ii])>np.finfo(float).eps):
            subspace_index = asg.subspace_indices[:,ii]
            subspace_poly_indices=asg.subspace_poly_indices_list[ii]
            subspace_weights = get_subspace_weights(
                subspace_index,asg.weights_1d)*asg.smolyak_coefficients[ii]
            for jj in range(subspace_poly_indices.shape[1]):
                poly_index = subspace_poly_indices[:,jj]
                key = hash_array(poly_index)
                if key in asg.poly_indices_dict:
                    weights[asg.poly_indices_dict[key]]+=subspace_weights[jj]
                else:
                    raise Exception('index not found')
    return samples,weights

from pyapprox.adaptive_sparse_grid import mypriorityqueue
class SubSpaceRefinementManager(object):
    def __init__(self,num_vars):
        self.num_vars = num_vars
        self.num_config_vars=0
        self.subspace_indices_dict = dict()
        self.subspace_indices = np.zeros((self.num_vars,0),dtype=int)
        self.active_subspace_indices_dict = dict()
        self.active_subspace_queue = mypriorityqueue()
        self.admissibility_function = None
        self.refinement_indicator = None
        self.univariate_growth_rule = None
        self.subspace_poly_indices_list = []
        self.poly_indices = np.zeros((self.num_vars,0),dtype=int)
        self.subspace_values_indices_list = []
        self.config_variables_idx=None
        self.samples = None
        self.values = None
        self.num_equivalent_function_evaluations = 0
        self.function=None
        self.variable_transformation = None
        self.work_qoi_index = None
        self.config_var_trans = None
        self.unique_quadrule_indices = None
        self.compact_univariate_growth_rule = None
        self.unique_poly_indices_idx=np.zeros((0),dtype=int)

    def initialize(self):
        self.poly_indices_dict = dict()
        self.samples = np.zeros((self.num_vars,0))
        self.add_new_subspaces(np.zeros((self.num_vars,1),dtype=int))
        self.error=np.zeros((0))
        self.prioritize_active_subspaces(
            self.subspace_indices, np.asarray([self.samples.shape[1]]))
        self.active_subspace_queue.list[0] = (np.inf,self.error[0],0)
        
    def refine(self):
        if self.subspace_indices.shape[1]==0:
            self.initialize()

        priority,error,best_subspace_idx = self.active_subspace_queue.get()
        best_active_subspace_index = self.subspace_indices[:,best_subspace_idx]

        new_active_subspace_indices, num_new_subspace_samples = \
            self.refine_and_add_new_subspaces(best_active_subspace_index)

        self.prioritize_active_subspaces(
            new_active_subspace_indices, num_new_subspace_samples)

        self.error[best_subspace_idx]=0.0 

    def refine_subspace(self,subspace_index):
        new_active_subspace_indices = np.zeros((self.num_vars,0),dtype=int)
        for ii in range(self.num_vars):
            neighbor_index = get_forward_neighbor(subspace_index,ii)
            if (subspace_index_is_admissible(
                neighbor_index,self.subspace_indices_dict) and
                    hash_array(neighbor_index) not in \
                    self.active_subspace_indices_dict and
                self.admissibility_function(self,neighbor_index)):
                new_active_subspace_indices = np.hstack(
                    (new_active_subspace_indices,neighbor_index[:,np.newaxis]))
        return new_active_subspace_indices

    def build(self):
        """
        """
        while (not self.active_subspace_queue.empty() or
               self.subspace_indices.shape[1]==0):
            self.refine()
        

    def refine_and_add_new_subspaces(self,best_active_subspace_index):
        key = hash_array(best_active_subspace_index)
        self.subspace_indices_dict[key]=\
          self.active_subspace_indices_dict[key]
    
        # get all new active subspace indices
        new_active_subspace_indices = self.refine_subspace(
            best_active_subspace_index)
        del self.active_subspace_indices_dict[key]

        if new_active_subspace_indices.shape[1]>0:
            num_new_subspace_samples=self.add_new_subspaces(
                new_active_subspace_indices)
        else:
            num_new_subspace_samples=0
        return new_active_subspace_indices, num_new_subspace_samples

    def get_subspace_samples(self,subspace_index,unique_poly_indices):
        """
        Must be implemented by derived class
        This function should only be called when updating grid not interogating
        grid
        """
        msg = "get_subspace_samples must be implemented by derived class"
        raise Exception(msg)


    def initialize_subspace(self,subspace_index):
        subspace_poly_indices = get_subspace_polynomial_indices(
                subspace_index, self.univariate_growth_rule,
                self.config_variables_idx)
            
        self.subspace_poly_indices_list.append(subspace_poly_indices)
        self.poly_indices_dict, unique_poly_indices, \
          subspace_values_indices = add_unique_poly_indices(
              self.poly_indices_dict, subspace_poly_indices)
        self.unique_poly_indices_idx=np.concatenate(
            [self.unique_poly_indices_idx,[self.poly_indices.shape[1]]])
        self.subspace_values_indices_list.append(subspace_values_indices)
        self.poly_indices = np.hstack((self.poly_indices,unique_poly_indices))
        return unique_poly_indices

    def initialize_subspaces(self,new_subspace_indices):
        num_vars, num_new_subspaces = new_subspace_indices.shape
        num_current_subspaces = self.subspace_indices.shape[1]
        cnt = num_current_subspaces
        for ii in range(num_new_subspaces):
            subspace_index = new_subspace_indices[:,ii]
            unique_poly_indices = self.initialize_subspace(subspace_index)
            self.active_subspace_indices_dict[hash_array(subspace_index)]=cnt
            cnt += 1

    def create_new_subspaces_data(self,new_subspace_indices):
        num_current_subspaces = self.subspace_indices.shape[1]
        self.initialize_subspaces(new_subspace_indices)
        num_vars, num_new_subspaces = new_subspace_indices.shape
        new_samples = np.empty((num_vars,0),dtype=float)
        #num_current_subspaces = self.subspace_indices.shape[1]
        #cnt = num_current_subspaces
        num_new_subspace_samples=np.empty((num_new_subspaces),dtype=int)
        for ii in range(num_new_subspaces):
            subspace_index = new_subspace_indices[:,ii]
            #unique_poly_indices = self.initialize_subspace(subspace_index)
            idx1=self.unique_poly_indices_idx[num_current_subspaces+ii]
            if ii < num_new_subspaces-1:
                idx2=self.unique_poly_indices_idx[num_current_subspaces+ii+1]
            else:
                idx2=self.poly_indices.shape[1]
            unique_poly_indices = self.poly_indices[:,idx1:idx2]
            unique_subspace_samples=self.get_subspace_samples(
                subspace_index,unique_poly_indices)
            new_samples = np.hstack(
                (new_samples,unique_subspace_samples))
            num_new_subspace_samples[ii]=unique_subspace_samples.shape[1]
            #self.active_subspace_indices_dict[hash_array(subspace_index)]=cnt
            #cnt += 1
        return new_samples,num_new_subspace_samples

    def add_new_subspaces(self,new_subspace_indices):
        new_samples, num_new_subspace_samples = self.create_new_subspaces_data(
            new_subspace_indices)

        new_values = self.eval_function(new_samples)
        self.subspace_indices = np.hstack(
            (self.subspace_indices,new_subspace_indices))
        self.samples = np.hstack((self.samples,new_samples))

        if self.values is None:
            self.values = new_values
        else:
            self.values = np.vstack((self.values, new_values))
            
        self.num_equivalent_function_evaluations += self.get_cost(
            new_subspace_indices,num_new_subspace_samples)

        return num_new_subspace_samples

    def prioritize_active_subspaces(self,new_active_subspace_indices,
                                    num_new_subspace_samples):
        cnt = self.subspace_indices.shape[1]-\
              new_active_subspace_indices.shape[1]
        for ii in range(new_active_subspace_indices.shape[1]):
            subspace_index = new_active_subspace_indices[:,ii]

            priority, error = self.refinement_indicator(
                subspace_index,num_new_subspace_samples[ii],self)
            self.active_subspace_queue.put((priority,error,cnt))
            self.error = np.concatenate([self.error,[error]])

            #print(subspace_index[-self.num_config_vars:],
            #      self.error.sum(),error,priority)
            cnt += 1

    def set_function(self,function,variable_transformation=None):
        self.function=function
        self.variable_transformation=variable_transformation

    def map_config_samples_from_canonical_space(self,samples):
        if self.config_variables_idx is None:
            config_variables_idx=self.num_vars
        else:
            config_variables_idx = self.config_variables_idx
        config_samples = samples[config_variables_idx:,:]
        if self.config_var_trans is not None:
            config_samples = self.config_var_trans.map_from_canonical_space(
                config_samples)
        return config_samples

    def map_random_samples_from_canonical_space(self,canonical_samples):
        if self.config_variables_idx is None:
            config_variables_idx=self.num_vars
        else:
            config_variables_idx = self.config_variables_idx
        random_samples = canonical_samples[:config_variables_idx,:]
        if self.variable_transformation is not None:                
            random_samples=self.variable_transformation.map_from_canonical_space(
                random_samples)
        return random_samples

    def eval_function(self,canonical_samples):
        random_samples = self.map_random_samples_from_canonical_space(
            canonical_samples)
        config_samples = self.map_config_samples_from_canonical_space(
            canonical_samples)
        samples = np .vstack((random_samples,config_samples))
        values = self.function(samples)
        
        if self.work_qoi_index is not None:
            costs = values[:,self.work_qoi_index]
            values = np.delete(values,self.work_qoi_index,axis=1)
            if self.config_variables_idx is None:
                # single fidelity so make up dummy unique key for work tracker
                config_samples=np.zeros((1,canonical_samples.shape[1]),dtype=int)
            self.cost_function.update(config_samples,costs)
        return values

    def set_univariate_growth_rules(self,univariate_growth_rule,
                                    unique_quadrule_indices):
        """
        self.config_variable_idx must be set if univariate_growth_rule is 
        a callable function and not a lisf of callable functions. Otherwise 
        errors such as assert len(growth_rule_1d)==config_variables_idx will
        be thrown

        TODO: eventually retire self.univariate_growth rule and just pass
        around compact_growth_rule. When doing this change from storing 
        samples_1d for each dimension to only storing for unique quadrature 
        rules
        """
        
        self.unique_quadrule_indices=unique_quadrule_indices
            
        if self.config_variables_idx is None:
            dd = self.num_vars
        else:
            dd = self.config_variables_idx
        if callable(univariate_growth_rule):
            self.compact_univariate_growth_rule=[univariate_growth_rule]
            self.unique_quadrule_indices = [np.arange(dd)]
        else:
            self.compact_univariate_growth_rule=univariate_growth_rule
            
        if self.unique_quadrule_indices is not None:
            cnt = 0
            for ii in self.unique_quadrule_indices:
                cnt+= len(ii)
            if cnt!=dd:
                msg = 'unique_quad_rule_indices is inconsistent with num_vars'
                raise Exception(msg)
            assert len(self.compact_univariate_growth_rule)==len(
                self.unique_quadrule_indices)
            self.univariate_growth_rule = [[] for dd in range(dd)]
            for ii in range(len(self.unique_quadrule_indices)):
                jj = self.unique_quadrule_indices[ii]
                for kk in jj:
                    self.univariate_growth_rule[kk]=\
                        self.compact_univariate_growth_rule[ii]
        else:
            if len(self.compact_univariate_growth_rule)!=dd:
                msg = 'length of growth_rules is inconsitent with num_vars.'
                msg += 'Maybe you need to set unique_quadrule_indices'
                raise Exception(msg)
            self.univariate_growth_rule = self.compact_univariate_growth_rule

        assert len(self.univariate_growth_rule)==dd
    
    def set_refinement_functions(self,refinement_indicator,
                                 admissibility_function,univariate_growth_rule,
                                 cost_function=None,work_qoi_index=None,
                                 unique_quadrule_indices=None):
        """
        cost_function : callable (or object is work_qoi_index is not None)
            Return the cost of evaluating a function with a unique indentifier.
            Identifiers can be strings, integers, etc. The identifier
            is found by mapping the sparse grid canonical_config_samples which
            are consecutive integers 0,1,... using self.config_var_trans

        work_qoi_index : integer (default None)
            If provided self.function is assumed to return the work (typically 
            measured in wall time) taken to evaluate each sample. The work
            for each sample return as a QoI in the column indexed by 
            work_qoi_index. The work QoI is ignored by the sparse grid
            eval_function() member function. If work_qoi_index is provided
            cost_function() must be a class with a member function 
            update(config_samples,costs). config_samples is a 2d array whose 
            columns are unique identifiers of the model being evaluated and 
            costs is the work needed to evaluate that model. If building single 
            fidelity sparse grid then config vars is set to be (0,...,0) for 
            each sample
        """
        self.refinement_indicator=refinement_indicator
        self.admissibility_function=admissibility_function
        self.set_univariate_growth_rules(
            univariate_growth_rule,unique_quadrule_indices)
        if cost_function is None:
            cost_function = default_combination_sparse_grid_cost_function
        self.cost_function=cost_function
        self.work_qoi_index=work_qoi_index
        if self.work_qoi_index is not None:
            if not hasattr(self.cost_function,'update'):
                msg = 'cost_function must have update() member function'
                raise Exception(msg)

    def set_config_variable_index(self,idx,config_var_trans=None):
        if  self.function is None:
            msg ='Must call set_function before entry'
            raise Exception(msg)
        self.config_variables_idx=idx
        self.config_var_trans=config_var_trans
        self.num_config_vars = self.num_vars-self.config_variables_idx
        if self.variable_transformation is not None:
            assert self.variable_transformation.nvars==self.config_variables_idx
        if self.config_var_trans is not None:
            assert self.num_config_vars==self.config_var_trans.num_vars()

    def eval_cost_function(self,samples):
        config_samples = self.map_config_samples_from_canonical_space(
            samples)
        if self.config_variables_idx is None:
            # single fidelity so make up dummy unique key for work tracker
            config_samples=np.zeros((1,samples.shape[1]),dtype=int)
        costs = self.cost_function(config_samples)
        return costs

    def get_cost(self, subspace_indices, num_new_subspace_samples):
        assert subspace_indices.shape[1]==num_new_subspace_samples.shape[0]
        # the cost of a single evaluate of each function
        function_costs = self.eval_cost_function(subspace_indices)
        assert function_costs.ndim==1
        # the cost of evaluating the unique points of each subspace
        subspace_costs = function_costs*num_new_subspace_samples
        # the cost of evaluating the unique points of all subspaces in
        # subspace_indices
        cost = np.sum(subspace_costs)
        return cost

    def __neq__(self,other):
        return not self.__eq__(other)

    def __eq__(self, other):
        """
        This function will compare all attributes of the derived class and this
        base class.
        """
        member_names = [
            m[0] for m in vars(self).items() if not m[0].startswith("__")]
        for m in member_names:
            attr = getattr(other,m)
            #print(m)
            #print(type(attr))
            #print(attr)
            if type(attr)==partial:
                if not partial_functions_equal(attr,getattr(self,m)):
                    return False
            elif type(attr)==list and type(attr[0])==np.ndarray:
                if not lists_of_arrays_equal(attr,getattr(self,m)):
                    return False
            elif type(attr)==list and type(attr[0])==list and type(
                    attr[0][0])==np.ndarray:
                if not lists_of_lists_of_arrays_equal(attr,getattr(self,m)):
                    return False
            elif np.any(getattr(other,m)!=getattr(self,m)):
                return False
        return True
    
from pyapprox.univariate_quadrature import leja_growth_rule
from pyapprox.univariate_quadrature import gaussian_leja_quadrature_rule,\
    beta_leja_quadrature_rule, candidate_based_leja_rule
def get_sparse_grid_univariate_leja_quadrature_rules(
        var_trans,growth_rules=None):
    assert var_trans is not None
    
    if growth_rules is None:
        growth_rules = leja_growth_rule
    if callable(growth_rules):
        growth_rules = [growth_rules]*var_trans.num_vars()
        
    assert var_trans is not None
    quad_rules = [[] for ii in range(var_trans.num_vars())]
    for var_type in var_trans.variables.unique_var_types:
        ii = var_trans.variables.unique_var_types[var_type]
        unique_quadrule_parameters = []
        for jj in range(len(var_trans.variables.unique_var_indices[ii])):
            var_index = var_trans.variables.unique_var_indices[ii][jj]
            var_parameters=var_trans.variables.unique_var_parameters[ii][jj]
            quad_rule = get_univariate_leja_quadrature_rule(
                var_type,var_parameters,growth_rules[var_index])
            quad_rules[var_index]=quad_rule
    
    return quad_rules, growth_rules#, None

from pyapprox.variables import variable_shapes_equivalent
def get_unique_quadrule_variables(var_trans):
    """
    This function will create a quad rule for each variable type with different 
    scaling. This can cause redundant computation of quad rules which
    may be significant when using leja sequences
    """
    unique_quadrule_variables = [var_trans.variable.unique_variables[0]]
    unique_quadrule_indices = [
        var_trans.variable.unique_variable_indices[0].copy()]
    for ii in range(1,var_trans.variable.nunique_vars):
        var = var_trans.variable.unique_variables[ii]
        var_indices = var_trans.variable.unique_variable_indices[ii].copy()
        found = False
        for jj in range(len(unique_quadrule_variables)):
            if variable_shapes_equivalent(var,unique_quadrule_variables[jj]):
                unique_quadrule_indices[jj]=np.concatenate(
                    [unique_quadrule_indices[jj],var_indices])
                found = True
                break
        if not found:
            unique_quadrule_variables.append(var)
            unique_quadrule_indices.append(var_indices)

    return unique_quadrule_variables, unique_quadrule_indices

def get_unique_quadrule_parameters_deprecated(var_trans):
    unique_quadrule_indices = []
    unique_quadrule_parameters = []
    var_types = []
    for var_type in var_trans.variables.unique_var_types:
        ii = var_trans.variables.unique_var_types[var_type]
        var_type_unique_quadrule_parameters = []
        for jj in range(len(var_trans.variables.unique_var_indices[ii])):
            var_parameters=var_trans.variables.unique_var_parameters[ii][jj]
            # make copy so not alter incoming dictionary
            var_parameters = copy.deepcopy(var_parameters)
            var_index = var_trans.variables.unique_var_indices[ii][jj]
            if 'range' in var_parameters:
                del var_parameters['range']
            elif var_type=='gaussian':
                del var_parameters['mean']
                del var_parameters['variance']
            index = -1
            for kk in range(len(var_type_unique_quadrule_parameters)):
                if var_parameters==var_type_unique_quadrule_parameters[kk]:
                    index = kk
                    break
            if index<0:
                var_type_unique_quadrule_parameters.append(var_parameters)
                unique_quadrule_parameters.append(var_parameters)
                unique_quadrule_indices.append([var_index])
                var_types.append(var_type)
            else:
                unique_quadrule_indices[ii+kk].append(var_index)
    return var_types, unique_quadrule_parameters, unique_quadrule_indices

def get_sparse_grid_univariate_leja_quadrature_rules_economical(
        var_trans,growth_rules=None):       
    assert var_trans is not None
    
    if growth_rules is None:
        growth_rules = leja_growth_rule

    unique_quadrule_variables,unique_quadrule_indices = \
        get_unique_quadrule_variables(var_trans)
        
    if callable(growth_rules):
        growth_rules = [growth_rules]*len(unique_quadrule_indices)

    if len(growth_rules)!=len(unique_quadrule_indices):
        msg ='growth rules and unique_quadrule_indices (derived from var_trans)'
        msg += ' are inconsistent'
        raise Exception(msg)

    quad_rules = []
    for ii in range(len(unique_quadrule_indices)):
        quad_rule = get_univariate_leja_quadrature_rule(
            unique_quadrule_variables[ii],growth_rules[ii])
        quad_rules.append(quad_rule)

    return quad_rules, growth_rules, unique_quadrule_indices

from pyapprox.variables import get_distribution_info
def get_univariate_leja_quadrature_rule(variable,growth_rule):
    var_type, __, shapes = get_distribution_info(variable)
    if var_type=='uniform':
        quad_rule = partial(
            beta_leja_quadrature_rule,1,1,growth_rule=growth_rule,
            samples_filename=None)
    elif var_type=='beta':
        quad_rule = partial(
            beta_leja_quadrature_rule,shapes['a'],shapes['b'],
            growth_rule=growth_rule)
    elif var_type=='norm':
        quad_rule = partial(
            gaussian_leja_quadrature_rule,growth_rule=growth_rule)
    elif var_type=='binom':
        num_trials = variable_parameters['num_trials']
        prob_success = variable_parameters['prob_success']
        def generate_candidate_samples(num_samples):
            assert num_samples==num_trials+1
            return np.arange(0,num_trials+1)[np.newaxis,:]
        recursion_coeffs = krawtchouk_recurrence(
            num_trials,num_trials,probability=True)
        quad_rule = partial(
            candidate_based_leja_rule,recursion_coeffs,
            generate_candidate_samples,
            num_trials+1,
            initial_samples=np.atleast_2d(
                [binomial_rv.ppf(0.5,num_trials,prob_success)]))
    else:
        raise Exception('var_type %s not implemented'%var_type)
    return quad_rule

class CombinationSparseGrid(SubSpaceRefinementManager):
    def __init__(self,num_vars):
        super(CombinationSparseGrid,self).__init__(num_vars)

        self.univariate_quad_rule = None
        self.samples_1d,self.weights_1d = [None,None]
        self.smolyak_coefficients = np.empty((0),np.float)
        self.variable_transformation = None
        self.compact_univariate_quad_rule = None
        self.subspace_moments = None

    def setup(self,function,config_variables_idx,refinement_indicator,
              admissibility_function,univariate_growth_rule,
              univariate_quad_rule,
              variable_transformation=None,config_var_trans=None,
              cost_function=None,work_qoi_index=None,
              unique_quadrule_indices=None):
        self.set_function(function,variable_transformation)
        if config_variables_idx is not None:
            self.set_config_variable_index(config_variables_idx,config_var_trans)
        self.set_refinement_functions(
            refinement_indicator,admissibility_function,univariate_growth_rule,
            cost_function,work_qoi_index,unique_quadrule_indices)
        self.set_univariate_rules(univariate_quad_rule)

    def set_univariate_rules(self,univariate_quad_rule):
        if self.univariate_growth_rule is None:
            msg="Must call set_refinement_functions before set_univariate rules"
            raise Exception(msg)
        max_level=2
        self.univariate_quad_rule=univariate_quad_rule

        if self.config_variables_idx is None:
            dd = self.num_vars
        else:
            dd = self.config_variables_idx

        num_random_vars = 0
        for ii in range(len(self.unique_quadrule_indices)):
            num_random_vars += len(self.unique_quadrule_indices[ii])
        if num_random_vars!=dd:
            msg =  'unique_quadrule_indices is inconsistent with '
            msg += 'self.config_variables_idx. If using config_variables try'
            msg += 'calling the following functions in this order'
            msg += """
                   set_function()
                   set_config_variable_index()
                   set_refinement_functions()
                   set_univariate_rules()
                   """
            raise Exception(msg)
        
        if callable(univariate_quad_rule):
            self.compact_univariate_quad_rule=[self.univariate_quad_rule]
        else:
            self.compact_univariate_quad_rule=univariate_quad_rule

        if self.unique_quadrule_indices is None:
            self.univariate_quad_rule = self.compact_univariate_quad_rule
        else:
            assert len(self.compact_univariate_quad_rule)==len(
                self.unique_quadrule_indices)
            self.univariate_quad_rule = [[] for dd in range(dd)]
            for ii in range(len(self.unique_quadrule_indices)):
                jj = self.unique_quadrule_indices[ii]
                for kk in jj:
                    self.univariate_quad_rule[kk]=\
                        self.compact_univariate_quad_rule[ii]

        assert len(self.univariate_quad_rule)==dd
            
        self.samples_1d, self.weights_1d = get_1d_samples_weights(
            self.compact_univariate_quad_rule,
            self.compact_univariate_growth_rule,
            [max_level]*dd,self.config_variables_idx,
            self.unique_quadrule_indices)

    def refine_and_add_new_subspaces(self,best_active_subspace_index):
        new_active_subspace_indices, num_new_subspace_samples = super(
            CombinationSparseGrid,self).refine_and_add_new_subspaces(
            best_active_subspace_index)
        self.smolyak_coefficients = update_smolyak_coefficients(
            best_active_subspace_index,self.subspace_indices,
            self.smolyak_coefficients)
        return new_active_subspace_indices, num_new_subspace_samples

    def get_subspace_samples(self,subspace_index,unique_poly_indices):
        samples_1d,weights_1d = update_1d_samples_weights(
            self.compact_univariate_quad_rule,
            self.compact_univariate_growth_rule,
            subspace_index,self.samples_1d,self.weights_1d,
            self.config_variables_idx,self.unique_quadrule_indices)

        self.smolyak_coefficients = np.hstack(
            (self.smolyak_coefficients,np.zeros(1)))

        return  get_sparse_grid_samples(
            unique_poly_indices,self.samples_1d,self.config_variables_idx)


    def __call__(self,samples):
        """
        config values are ignored. The sparse grid just returns its best 
        approximation of the highest fidelity model. TODO: consider enforcing
        that samples do not have configure variables
        """
        if self.variable_transformation is not None:
            canonical_samples = \
                self.variable_transformation.map_to_canonical_space(
                    samples[:self.config_variables_idx,:])
        else:
            canonical_samples = samples[:self.config_variables_idx,:]
        
        return evaluate_sparse_grid(
            canonical_samples[:self.config_variables_idx,:],
            self.values,
            self.poly_indices_dict,self.subspace_indices,
            self.subspace_poly_indices_list,
            self.smolyak_coefficients,self.samples_1d,
            self.subspace_values_indices_list,
            self.config_variables_idx)

    def moments_(self,smolyak_coefficients):
        return integrate_sparse_grid_from_subspace_moments(
            self.subspace_indices,smolyak_coefficients,
            self.subspace_moments)

        # return integrate_sparse_grid(
        #     self.values,
        #     self.poly_indices_dict,self.subspace_indices,
        #     self.subspace_poly_indices_list,
        #     smolyak_coefficients,self.weights_1d,
        #     self.subspace_values_indices_list,
        #     self.config_variables_idx)



    def moments(self):
        return self.moments_(self.smolyak_coefficients)

    def evaluate_using_all_data(self,samples):
        """
        Evaluate sparse grid using all subspace indices including
        active subspaces. __call__ only uses subspaces which are not active
        """
        # extract active subspaces from queue without destroying queue
        pairs, self.active_subspace_queue = \
            extract_items_from_priority_queue(self.active_subspace_queue)
        # copy smolyak coefficients so as not affect future refinement
        smolyak_coefficients = self.smolyak_coefficients.copy()
        # add all active subspaces to sparse grid by updating smolyak
        # coefficients
        for ii in range(len(pairs)):
            subspace_index = self.subspace_indices[:,pairs[ii][-1]]
            smolyak_coefficients = update_smolyak_coefficients(
                subspace_index,self.subspace_indices,
                smolyak_coefficients)
        
        if self.variable_transformation is not None:
            canonical_samples = \
                self.variable_transformation.map_to_canonical_space(
                    samples[:self.config_variables_idx,:])
        else:
            canonical_samples = samples[:self.config_variables_idx,:]
            
        # evaluate sparse grid includding active subspaces
        approx_values = evaluate_sparse_grid(
            canonical_samples,
            self.values,self.poly_indices_dict,
            self.subspace_indices,
            self.subspace_poly_indices_list,
            smolyak_coefficients,self.samples_1d,
            self.subspace_values_indices_list,
            self.config_variables_idx)
        return approx_values

    def add_new_subspaces(self,new_subspace_indices):
        num_new_subspaces = new_subspace_indices.shape[1]
        num_current_subspaces = self.subspace_indices.shape[1]
        num_new_subspace_samples = super(
            CombinationSparseGrid,self).add_new_subspaces(new_subspace_indices)
        
        cnt = num_current_subspaces
        new_subspace_moments = np.empty(
            (num_new_subspaces,self.values.shape[1],2),dtype=float)
        for ii in range(num_new_subspaces):
            subspace_index = new_subspace_indices[:,ii]
            subspace_values = get_subspace_values(
                self.values, self.subspace_values_indices_list[cnt])
            subspace_moments = integrate_sparse_grid_subspace(
                subspace_index,subspace_values,self.weights_1d,
                self.config_variables_idx)
            new_subspace_moments[ii,:,:] = subspace_moments.T
            cnt += 1

        if self.subspace_moments is None:
            self.subspace_moments = new_subspace_moments
        else:
            self.subspace_moments = np.vstack(
                (self.subspace_moments,new_subspace_moments))

        return num_new_subspace_samples

    def save(self,filename):
        try:
            with open(filename, 'wb') as file_object:
                pickle.dump(self,file_object)
        except:
            msg =  'Initial attempt to save failed. Likely self.function '
            msg += 'cannot be pickled. Trying to save setting function to None'
            print (msg)
            function = self.function
            self.function = None
            with open(filename, 'wb') as file_object:
                pickle.dump(self,file_object)
            self.function=function
            msg = 'Second save was successful'
            print(msg)
  
            
class WorkTracker(object):
    def __init__(self):
        self.costs = dict()

    def __call__(self,config_samples):
        num_config_vars, nqueries = config_samples.shape
        costs = np.empty((nqueries))
        for ii in range(nqueries):
            key = tuple([int(ll) for ll in config_samples[:,ii]])
            assert key in self.costs, key
            costs[ii] = np.median(self.costs[key])
        return costs

    def update(self,config_samples,costs):
        num_config_vars, nqueries = config_samples.shape
        assert costs.shape[0]==nqueries
        assert costs.ndim==1
        for ii in range(nqueries):
            key = tuple([int(ll) for ll in config_samples[:,ii]])
            if key in self.costs:
                self.costs[key].append(costs[ii])
            else:
                self.costs[key] = [costs[ii]]
            
    
def plot_adaptive_sparse_grid_3d(sparse_grid,plot_grid=True):
    from pyapprox.visualization import plot_3d_indices
    fig = plt.figure(figsize=plt.figaspect(0.5))
    active_subspace_indices,active_subspace_idx = get_active_subspace_indices(
        sparse_grid.active_subspace_indices_dict,
        sparse_grid.subspace_indices)

    # get subspace indices that have been added to the sparse grid,
    # i.e are not active
    sparse_grid_subspace_idx = np.ones(
        (sparse_grid.subspace_indices.shape[1]),dtype=bool)
    sparse_grid_subspace_idx[active_subspace_idx]=False

    nn=1
    if plot_grid:
        nn = 2
    ax=fig.add_subplot(1,nn,1,projection='3d')
    if active_subspace_indices.shape[1]==0:
        active_subspace_indices=None
    plot_3d_indices(sparse_grid.subspace_indices,ax,active_subspace_indices)

    if plot_grid:
        samples,active_samples = partition_sparse_grid_samples(sparse_grid)
        ax=fig.add_subplot(1,nn,2,projection='3d')
        ax.plot(samples[0,:],samples[1,:],samples[2,:],'ko')
        ax.plot(active_samples[0,:],active_samples[1,:],
                active_samples[2,:],'ro')

        angle=45
        ax.view_init(30, angle)
        #ax.set_axis_off()
        ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    

    
"""
Notes if use combination technique to manage only adaptive refinement in configure variables and another strategy (e.g. another independent combination technique to refine in stochastic space) then this will remove downward index constraint
# between subspaces that vary both models and parameters.

An Adaptive PCE can only use this aforementioned case. I do not see a way to
let each subspace still be a tensor product index and build an approximation only over tha subspace and then combine.
"""
