import numpy as np
from pyapprox.indexing import compute_hyperbolic_indices,hash_array
from pyapprox.utilities import nchoosek

def get_main_and_total_effect_indices_from_pce(coefficients,indices):
    """
    Assume basis is orthonormal
    Assume first coefficient is the coefficient of the constant basis. Remove
    this assumption by extracting array index of constant term from indices

    Returns
    -------
    main_effects : np.ndarray(num_vars)
        Contribution to variance of each variable acting alone

    total_effects : np.ndarray(num_vars)
        Contribution to variance of each variable acting alone or with 
        other variables

    """
    num_vars = indices.shape[0]
    num_terms,num_qoi = coefficients.shape
    assert num_terms==indices.shape[1]
    
    main_effects = np.zeros((num_vars,num_qoi), np.double )
    total_effects = np.zeros((num_vars,num_qoi), np.double )
    variance = np.zeros(num_qoi)

    for ii in range(num_terms):
        index = indices[:,ii]
        
        # calculate contribution to variance of the index
        var_contribution = coefficients[ii,:]**2
        
        # get number of dimensions involved in interaction, also known
        # as order
        non_constant_vars = np.where(index>0)[0]
        order = non_constant_vars.shape[0]

        if order>0:
            variance += var_contribution
        
        # update main effects
        if ( order == 1 ):
            var = non_constant_vars[0]
            main_effects[var,:] += var_contribution
            
        # update total effects
        for ii in range( order ):
            var = non_constant_vars[ii]
            total_effects[var,:] += var_contribution

    main_effects /= variance
    total_effects /= variance
    return main_effects, total_effects

def get_sobol_indices(coefficients,indices,max_order=2):
    num_terms,num_qoi = coefficients.shape
    variance = np.zeros(num_qoi)
    assert num_terms==indices.shape[1]
    interactions = dict()
    interaction_values = []
    interaction_terms = []
    kk=0 
    for ii in range(num_terms):
        index = indices[:,ii]
        var_contribution = coefficients[ii,:]**2
        non_constant_vars = np.where(index>0)[0]
        key = hash_array(non_constant_vars)

        if len(non_constant_vars) > 0:
            variance += var_contribution
        
        if len(non_constant_vars) > 0 and len(non_constant_vars)<=max_order:
            if key in interactions:
                interaction_values[interactions[key]] += var_contribution
            else:
                interactions[key]=kk
                interaction_values.append(var_contribution)
                interaction_terms.append(non_constant_vars)
                kk+=1

    interaction_terms = np.asarray(interaction_terms).T
    interaction_values = np.asarray(interaction_values)
    
    return interaction_terms, interaction_values/variance

def plot_main_effects(main_effects, ax, truncation_pct=0.95,
                      max_slices=5, rv='z', qoi=0):
    """
    Plot the main effects in a pie chart showing relative size.

    Parameters
    ----------
    main_effects : np.ndarray (nvars,nqoi)
        The variance based main effect sensitivity indices

    ax : :class:`matplotlib.pyplot.axes.Axes`
        Axes that will be used for plotting

    truncation_pct : float
        The proportion :math:`0<p\le 1` of the sensitivity indices 
        effects to plot

    max_slices : integer
        The maximum number of slices in the pie-chart. Will only
        be active if the turncation_pct gives more than max_slices

    rv : string
        The name of the random variables when creating labels
        
    qoi : integer
        The index 0<qoi<nqoi of the quantitiy of interest to plot
    """
    main_effects=main_effects[:,qoi]
    assert main_effects.sum()<=1.

    # sort main_effects in descending order
    I=np.argsort( main_effects )[::-1]
    main_effects=main_effects[I]

    labels=[]
    partial_sum=0.
    for i in range( I.shape[0] ):
        if partial_sum < truncation_pct and i < max_slices:
            labels.append( '$%s_{%d}$' %(rv,I[i]+1) )
            partial_sum += main_effects[i]
        else:
            break

    main_effects.resize( i + 1 )
    if abs( partial_sum - main_effects.sum()) > 100*np.finfo(np.double).eps:
        labels.append( 'other' )
        main_effects[-1]= main_effects.sum() - partial_sum

    ax.pie(main_effects, labels=labels, autopct='%1.1f%%',
           shadow=True )

def plot_total_effects( total_effects, ax, truncation_pct=0.95, max_slices=5,
                        rv='z', qoi=0):
    """
    Plot the total effects in a pie chart showing relative size.

    Parameters
    ----------
    total_effects : np.ndarray (nvars,nqoi)
        The variance based total effect sensitivity indices

    ax : :class:`matplotlib.pyplot.axes.Axes`
        Axes that will be used for plotting

    truncation_pct : float
        The proportion :math:`0<p\le 1` of the sensitivity indices 
        effects to plot

    max_slices : integer
        The maximum number of slices in the pie-chart. Will only
        be active if the turncation_pct gives more than max_slices

    rv : string
        The name of the random variables when creating labels
        
    qoi : integer
        The index 0<qoi<nqoi of the quantitiy of interest to plot
    """

    total_effects=total_effects[:,qoi]

    # normalise total effects
    total_effects /= total_effects.sum()

    # sort total_effects in descending order
    I=np.argsort( total_effects )[::-1]
    total_effects=total_effects[I]

    labels=[]
    partial_sum=0.
    for i in range( I.shape[0] ):
        if partial_sum < truncation_pct*total_effects.sum() and i < max_slices:
            labels.append( '$%s_{%d}$' %(rv,I[i]+1) )
            partial_sum += total_effects[i]
        else:
            break

    total_effects.resize( i + 1 )
    if abs( partial_sum - 1. ) > 10 * np.finfo( np.double ).eps:
        labels.append( 'other' )
        total_effects[-1]=1. - partial_sum

    ax.pie(total_effects, labels=labels, autopct='%1.1f%%',
           shadow=True )


def plot_interaction_values( interaction_values, interaction_terms, ax, 
                             truncation_pct=0.95, max_slices=5, rv='z', qoi=0):
    """
    Plot sobol indices in a pie chart showing relative size.

    Parameters
    ----------
    interaction_values : np.ndarray (nvars,nqoi)
        The variance based Sobol indices

    interaction_terms : nlist (nchoosek(nvars+max_order,nvars))
        Indices np.ndarrays of varying size specifying the variables in each 
        interaction in ``interaction_indices``

    ax : :class:`matplotlib.pyplot.axes.Axes`
        Axes that will be used for plotting

    truncation_pct : float
        The proportion :math:`0<p\le 1` of the sensitivity indices 
        effects to plot

    max_slices : integer
        The maximum number of slices in the pie-chart. Will only
        be active if the turncation_pct gives more than max_slices

    rv : string
        The name of the random variables when creating labels
        
    qoi : integer
        The index 0<qoi<nqoi of the quantitiy of interest to plot
    """

    assert interaction_values.shape[0]==len(interaction_terms)
    interaction_values=interaction_values[:,qoi]
    interaction_values /= interaction_values.sum()

    I = np.argsort(interaction_values)[::-1]
    interaction_values = interaction_values[I]
    interaction_terms = [interaction_terms[ii] for ii in I]

    labels=[]
    partial_sum=0.
    for i in range(interaction_values.shape[0]):
        if partial_sum < truncation_pct and i < max_slices:
            l='($'
            for j in range( len( interaction_terms[i] )-1 ):
                l += '%s_{%d},' %(rv,interaction_terms[i][j]+1)
            l+= '%s_{%d}$)' %(rv,interaction_terms[i][-1]+1)
            labels.append( l )
            partial_sum += interaction_values[i]
        else:
            break

    interaction_values=interaction_values[:i]
    if abs( partial_sum - 1. ) > 10 * np.finfo( np.double ).eps:
        labels.append( 'other' )
        interaction_values=np.concatenate([interaction_values,[1.-partial_sum]])


    assert interaction_values.shape[0] == len ( labels )
    ax.pie(interaction_values, labels=labels, autopct='%1.1f%%',
           shadow=True )

def get_morris_trajectory(nvars,nlevels,eps=0):
    """
    Compute a morris trajectory used to compute elementary effects

    Parameters
    ----------
    nvars : integer
        The number of variables

    nlevels : integer
        The number of levels used for to define the morris grid.

    eps : float 
        Set grid used defining the morris trajectory to [eps,1-eps].
        This is needed when mapping the morris trajectories using inverse
        CDFs of unbounded variables

    Returns
    -------
    trajectory : np.ndarray (nvars,nvars+1)
        The Morris trajectory which consists of nvars+1 samples
    """
    assert nlevels%2==0
    delta = nlevels/((nlevels-1)*2)
    samples_1d = np.linspace(eps, 1-eps, nlevels)
    
    initial_point=np.random.choice(samples_1d, nvars)
    shifts = np.diag(np.random.choice([-delta,delta],nvars))
    trajectory = np.empty((nvars,nvars+1))
    trajectory[:,0] = initial_point
    for ii in range(nvars):
        trajectory[:,ii+1]=trajectory[:,ii].copy()
        if (trajectory[ii,ii]-delta)>=0 and (trajectory[ii,ii]+delta)<=1:
            trajectory[ii,ii+1]+=shift[ii]
        elif (trajectory[ii,ii]-delta)>=0:
            trajectory[ii,ii+1]-=delta
        elif (trajectory[ii,ii]+delta)<=1:
            trajectory[ii,ii+1]+=delta
        else:
            raise Exception('This should not happen')
    return trajectory

def get_morris_samples(nvars,nlevels,ntrajectories,eps=0,icdfs=None):
    """
    Compute a set of Morris trajectories used to compute elementary effects

    Notes
    -----
    The choice of nlevels must be linked to the choice of ntrajectories.
    For example, if a large number of possible levels is used ntrajectories
    must also be high, otherwise if ntrajectories is small effort will be  
    wasted because many levels will not be explored. nlevels=4 and 
    ntrajectories=10 is often considered reasonable.

    Parameters
    ----------
    nvars : integer
        The number of variables

    nlevels : integer
        The number of levels used for to define the morris grid.

    ntrajectories : integer
        The number of Morris trajectories requested

    eps : float 
        Set grid used defining the Morris trajectory to [eps,1-eps].
        This is needed when mapping the morris trajectories using inverse
        CDFs of unbounded variables

    icdfs : list (nvars)
        List of inverse CDFs functions for each variable

    Returns
    -------
    trajectories : np.ndarray (nvars,ntrajectories*(nvars+1))
        The Morris trajectories
    """
    if icdfs is None:
        icdfs = [lambda x: x]*nvars
    assert len(icdfs)==nvars

    trajectories = np.hstack([get_morris_trajectory(nvars,nlevels,eps)
                              for n in range(ntrajectories)])
    for ii in range(nvars):
        trajectories[ii,:] = icdfs[ii](trajectories[ii,:])
    return trajectories

def get_morris_elementary_effects(samples,values):
    """
    Get the Morris elementary effects from a set of trajectories.

    Parameters
    ----------
    samples : np.ndarray (nvars,ntrajectories*(nvars+1))
        The morris trajectories

    values : np.ndarray (ntrajectories*(nvars+1),nqoi)
        The values of the vecto-valued target function with nqoi quantities 
        of interest (QoI)

    Returns
    -------
    elem_effects : np.ndarray(nvars,ntrajectories,nqoi)
        The elementary effects of each variable for each trajectory and QoI
    """
    nvars = samples.shape[0]
    nqoi=values.shape[1]
    assert samples.shape[1]%(nvars+1)==0
    assert samples.shape[1]==values.shape[0]
    ntrajectories = samples.shape[1]//(nvars+1)
    elem_effects = np.empty((nvars,ntrajectories,nqoi))
    ix1=0
    for ii in range(ntrajectories):
        ix2=ix1+nvars
        delta = np.diff(samples[:,ix1+1:ix2+1]-samples[:,ix1:ix2]).max()
        assert delta>0
        elem_effects[:,ii] = (values[ix1+1:ix2+1]-values[ix1:ix2])/delta
        ix1=ix2+1
    return elem_effects

def get_morris_sensitivity_indices(elem_effects):
    """
    Compute the Morris sensitivity indices mu and sigma from the elementary 
    effects computed for a set of trajectories.

    Mu is the mu^\star from Campolongo et al.

    Parameters
    ----------
    elem_effects : np.ndarray(nvars,ntrajectories,nqoi)
        The elementary effects of each variable for each trajectory and quantity
        of interest (QoI)

    Returns
    -------
    mu : np.ndarray(nvars,nqoi) 
        The sensitivity of each output to each input. Larger mu corresponds to 
        higher sensitivity
    
    sigma: np.ndarray(nvars,nqoi) 
        A measure of the non-linearity and/or interaction effects of each input
        for each output. Low values suggest a linear realationship between
        the input and output. Larger values suggest a that the output is 
        nonlinearly dependent on the input and/or the input interacts with 
        other inputs
    """
    mu = np.absolute(elem_effects).mean(axis=1)
    assert mu.shape==(elem_effects.shape[0],elem_effects.shape[2])
    sigma = np.std(elem_effects,axis=1)
    return mu,sigma

def print_morris_sensitivity_indices(mu,sigma,qoi=0):
    string = "Morris sensitivity indices\n"
    from pandas import DataFrame
    df=DataFrame({"mu*":mu[:,qoi],"sigma":sigma[:,qoi]})
    df.index = [f'Z_{ii+1}' for ii in range(mu.shape[0])]
    print(df)
    
from scipy.spatial.distance import cdist
from itertools import combinations
def downselect_morris_trajectories(samples,ntrajectories):    
    nvars = samples.shape[0]
    assert samples.shape[1]%(nvars+1)==0
    ncandidate_trajectories = samples.shape[1]//(nvars+1)
    #assert 10*ntrajectories<=ncandidate_trajectories

    trajectories=np.reshape(
        samples,(nvars,nvars+1,ncandidate_trajectories),order='F')
    
    distances = np.zeros((ncandidate_trajectories,ncandidate_trajectories))
    for ii in range(ncandidate_trajectories):
        for jj in range(ii+1):
            distances[ii,jj]=cdist(
                trajectories[:,:,ii].T,trajectories[:,:,jj].T).sum()
            distances[jj,ii]=distances[ii,jj]

    get_combinations=combinations(
        np.arange(ncandidate_trajectories),ntrajectories)
    ncombinations = nchoosek(ncandidate_trajectories,ntrajectories)
    print('ncombinations',ncombinations)
    values = np.empty(ncombinations)
    best_index = None
    best_value = -np.inf
    for ii,index in enumerate(get_combinations):
        value = np.sqrt(np.sum(
            [distances[ix[0],ix[1]]**2 for ix in combinations(index,2)]))
        if value>best_value:
            best_value=value
            best_index=index

    samples = trajectories[:,:,best_index].reshape(nvars,ntrajectories*(nvars+1),order='F')
    return samples

from scipy.optimize import OptimizeResult
class SensivitityResult(OptimizeResult):
    pass

def analyze_sensitivity_morris(fun,univariate_variables,ntrajectories,nlevels=4):
    """
    Compute sensitivity indices by constructing an adaptive polynomial chaos
    expansion.
    
    Parameters
    ----------
    fun : callable
        The function being analyzed

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

    ntrajectories : integer
        The number of Morris trajectories requested


    nlevels : integer
        The number of levels used for to define the morris grid.
        
    Returns
    -------
    result : :class:`pyapprox.sensitivity_analysis.SensivitityResult`
         Result object with the following attributes

    mu : np.ndarray (nvars,nqoi) 
        The sensitivity of each output to each input. Larger mu corresponds to 
        higher sensitivity
    
    sigma: np.ndarray (nvars,nqoi) 
        A measure of the non-linearity and/or interaction effects of each input
        for each output. Low values suggest a linear realationship between
        the input and output. Larger values suggest a that the output is 
        nonlinearly dependent on the input and/or the input interacts with 
        other inputs

    samples : np.ndarray(nvars,ntrajectories*(nvars+1))
        The coordinates of each morris trajectory

    values : np.ndarray(nvars,nqoi)
        The values of ``fun`` at each sample in ``samples``
    """
    
    nvars = len(univariate_variables)
    samples = get_morris_samples(nvars,nlevels,ntrajectories)
    values = function(samples)
    elem_effects = get_morris_elementary_effects(samples,values)
    mu,sigma = get_morris_sensitivity_indices(elem_effects)
    
    return SensivitityResult(
        {'morris_mu':pce_main_effects,
         'morris_sigma':pce_total_effects,
         'samples':samples,'values':values})

def analyze_sensitivity_sparse_grid(sparse_grid,max_order=2):
    """
    Compute sensitivity indices from a sparse grid
    by converting it to a polynomial chaos expansion

    Parameters
    ----------
    sparse_grid :class:`pyapprox.adaptive_sparse_grid:CombinationSparseGrid`
       The sparse grid

    max_order : integer
        The maximum interaction order of Sonol indices to compute. A value
        of 2 will compute all pairwise interactions, a value of 3 will 
        compute indices for all interactions involving 3 variables. The number
        of indices returned will be nchoosek(nvars+max_order,nvars). Warning 
        when nvars is high the number of indices will increase rapidly with 
        max_order.


    Returns
    -------
    result : :class:`pyapprox.sensitivity_analysis.SensivitityResult`
         Result object with the following attributes

    main_effects : np.ndarray (nvars)
        The variance based main effect sensitivity indices

    total_effects : np.ndarray (nvars)
        The variance based total effect sensitivity indices

    sobol_indices : np.ndarray (nchoosek(nvars+max_order,nvars),nqoi)
        The variance based Sobol sensitivity indices

    sobol_interaction_indices : np.ndarray(nvars,nchoosek(nvars+max_order,nvars))
        Indices specifying the variables in each interaction in 
        ``sobol_indices``

    pce : :class:`multivariate_polynomials.PolynomialChaosExpansion`
       The pce respresentation of the sparse grid ``approx``
    """
    from pyapprox.multivariate_polynomials import \
        define_poly_options_from_variable_transformation
    from pyapprox.adaptive_sparse_grid import \
        convert_sparse_grid_to_polynomial_chaos_expansion
    pce_opts=define_poly_options_from_variable_transformation(
        sparse_grid.variable_transformation)
    pce = convert_sparse_grid_to_polynomial_chaos_expansion(
        sparse_grid,pce_opts)
    pce_main_effects,pce_total_effects=\
        get_main_and_total_effect_indices_from_pce(
            pce.get_coefficients(),pce.get_indices())

    interaction_terms, pce_sobol_indices = get_sobol_indices(
            pce.get_coefficients(),pce.get_indices(),max_order=max_order)
    
    return SensivitityResult(
        {'main_effects':pce_main_effects,
         'total_effects':pce_total_effects,
         'sobol_indices':pce_sobol_indices,
         'sobol_interaction_indices':interaction_terms,
         'pce':pce})

def analyze_sensitivity_polynomial_chaos(pce,max_order=2):
    """
    Compute variance based sensitivity metrics from a polynomial chaos expansion

    Parameters
    ----------
    pce :class:`pyapprox.multivariate_polynomials.PolynomialChaosExpansion`
       The polynomial chaos expansion

    max_order : integer
        The maximum interaction order of Sonol indices to compute. A value
        of 2 will compute all pairwise interactions, a value of 3 will 
        compute indices for all interactions involving 3 variables. The number
        of indices returned will be nchoosek(nvars+max_order,nvars). Warning 
        when nvars is high the number of indices will increase rapidly with 
        max_order.

    Returns
    -------
    result : :class:`pyapprox.sensitivity_analysis.SensivitityResult`
         Result object with the following attributes

    main_effects : np.ndarray (nvars)
        The variance based main effect sensitivity indices

    total_effects : np.ndarray (nvars)
        The variance based total effect sensitivity indices

    sobol_indices : np.ndarray (nchoosek(nvars+max_order,nvars),nqoi)
        The variance based Sobol sensitivity indices

    sobol_interaction_indices : np.ndarray(nvars,nchoosek(nvars+max_order,nvars))
        Indices specifying the variables in each interaction in 
        ``sobol_indices``
    """
    pce_main_effects,pce_total_effects=\
        get_main_and_total_effect_indices_from_pce(
            pce.get_coefficients(),pce.get_indices())

    interaction_terms, pce_sobol_indices = get_sobol_indices(
        pce.get_coefficients(),pce.get_indices(),
        max_order=max_order)
    
    return SensivitityResult(
        {'main_effects':pce_main_effects,
         'total_effects':pce_total_effects,
         'sobol_indices':pce_sobol_indices,
         'sobol_interaction_indices':interaction_terms})
