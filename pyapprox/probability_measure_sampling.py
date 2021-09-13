import numpy as np

from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation


def print_statistics(samples, values, sample_labels=None, value_labels=None):
    """
    Print statistics about a set of samples and associated values

    Parameters
    ----------
    samples : np.ndarray (num_vars,num_samples)
        Random samples 

    values : np.ndarray (num_samples,num_qoi)
       Function values at samples

    Examples
    --------
    >>> num_vars=2
    >>> np.random.seed(1)
    >>> samples = np.random.normal(0,1,(num_vars,100))
    >>> values = np.array([np.sum(samples**2,axis=0),2*np.sum(samples**2,axis=0)]).T
    >>> print_statistics(samples,values)
                   z0          z1          y0          y1
    count  100.000000  100.000000  100.000000  100.000000
    mean     0.060583    0.152795    1.679132    3.358265
    std      0.889615    0.936690    1.714470    3.428941
    min     -2.301539   -2.434838    0.031229    0.062458
    25%     -0.613818   -0.300010    0.437509    0.875019
    50%      0.064074    0.236616    1.073139    2.146279
    75%      0.637410    0.743020    2.397208    4.794417
    max      2.185575    2.528326    9.575905   19.151810
    """
    if values.ndim == 1:
        values = values[:, np.newaxis]
    from pandas import DataFrame
    num_vars, nsamples = samples.shape
    num_qoi = values.shape[1]
    assert nsamples == values.shape[0]
    if sample_labels is None:
        sample_labels = ['z%d' % ii for ii in range(num_vars)]
    if value_labels is None:
        value_labels = ['y%d' % ii for ii in range(num_qoi)]

    data = [(label, s) for s, label in zip(samples, sample_labels)]
    data += [(label, s) for s, label in zip(values.T, value_labels)]
    data = dict(data)
    df = DataFrame(index=np.arange(nsamples), data=data)
    print(df.describe())


def generate_canonical_univariate_random_samples(
        var_type, variable_parameters, num_samples, num_vars):
    """
    Generate samples from a one-dimensional probability measure.

    This function only supports common variables types.
    uniform, beta, gaussian, exponential

    Note to developers: The canonical domain here must be consistent with
    the probability domain of AffineRandomVariableTransformation
    which is used to map canonical samples to the user space.

    Parameters
    ----------

    var_type : string
        The variable type 

    variable_parameters : dict
        The parameters that define the distribution

    num_samples : integer
        The number of samples to generate

    Returns
    -------
    samples : np.ndarray (num_samples)
        Independent samples from the target distribution
    """
    if var_type == 'uniform':
        samples = np.random.uniform(-1., 1., (num_vars, num_samples))
    elif var_type == 'beta':
        alpha_stat = variable_parameters['alpha_stat']
        beta_stat = variable_parameters['beta_stat']
        samples = 2.*np.random.beta(alpha_stat,
                                    beta_stat, (num_vars, num_samples))-1.
    elif var_type == 'gaussian':
        samples = np.random.normal(0., 1., (num_vars, num_samples))
    elif var_type == 'exponential':
        samples = np.random.exponential(1., (num_vars, num_samples))
    elif var_type == 'uniform_discrete':
        # samples = np.random.randint(0,variable_parameters['num_trials']+1,
        #                             (num_vars,num_samples))
        samples = np.random.randint(
            variable_parameters['range'][0], variable_parameters['range'][1]+1,
            (num_vars, num_samples))
    elif var_type == 'binomial_discrete':
        samples = np.random.binomial(
            variable_parameters['num_trials'],
            variable_parameters['prob_success'], (num_vars, num_samples))
    elif var_type == 'hypergeometric_discrete':
        samples = np.random.hypergeometric(
            variable_parameters['num_type1'], variable_parameters['num_type2'],
            variable_parameters['num_trials'], (num_vars, num_samples))
    elif var_type == 'arbitrary_discrete':
        masses = variable_parameters['prob_masses']
        mass_locations = variable_parameters['prob_mass_locations']
        samples = np.random.choice(mass_locations, size=(num_vars, num_samples),
                                   p=masses)
    else:
        raise TypeError('var_type %s not supported' % var_type)
    return samples


def generate_independent_random_samples_deprecated(var_trans, num_samples):
    """
    Generate samples from a tensor-product probability measure.

    Parameters
    ----------
    var_trans : AffineRandomVariableTransformation
        Object that maps samples from a canonical domain into
        the space required by the user

    num_samples : integer
        The number of samples to generate

    Returns
    -------
    samples : np.ndarray (num_vars, num_samples)
        Independent samples from the target distribution
    """
    assert type(var_trans) == AffineRandomVariableTransformation, \
        "`var_trans` must be of AffineRandomVariableTransformation type"
    num_vars = var_trans.nvars

    canonical_samples = np.empty((num_vars, num_samples), dtype=float)
    variables = var_trans.variables
    num_unique_var_types = len(variables.unique_var_types)
    for var_type in list(variables.unique_var_types.keys()):
        type_index = variables.unique_var_types[var_type]
        num_vars_of_type = len(variables.unique_var_indices[type_index])
        for jj in range(num_vars_of_type):
            var_index = variables.unique_var_indices[type_index][jj]
            canonical_samples[var_index, :] = \
                generate_canonical_univariate_random_samples(
                var_type, variables.unique_var_parameters[type_index][jj],
                num_samples, 1)[0, :]

    return var_trans.map_from_canonical_space(canonical_samples)


def generate_independent_random_samples(variable, num_samples,
                                        random_state=None):
    """
    Generate samples from a tensor-product probability measure.

    Parameters
    ----------
    num_samples : integer
        The number of samples to generate

    Returns
    -------
    samples : np.ndarray (num_vars, num_samples)
        Independent samples from the target distribution
    """
    assert type(variable) == IndependentMultivariateRandomVariable, \
        "`variable` must be of IndependentMultivariateRandomVariable type"
    num_vars = variable.nvars
    num_samples = int(num_samples)
    samples = np.empty((num_vars, num_samples), dtype=float)
    for ii in range(variable.nunique_vars):
        var = variable.unique_variables[ii]
        indices = variable.unique_variable_indices[ii]
        samples[indices, :] = var.rvs(
            size=(indices.shape[0], num_samples), random_state=random_state)

    return samples


def rejection_sampling(target_density, proposal_density,
                       generate_proposal_samples, envelope_factor,
                       num_vars, num_samples, verbose=False,
                       batch_size=None):
    """
    Obtain samples from a density f(x) using samples from a proposal 
    distribution g(x).

    Parameters
    ----------
    target_density : callable vals = target_density(samples)
        The target density f(x)

    proposal_density : callable vals = proposal_density(samples)
        The proposal density g(x)

    generate_proposal_samples : callable samples = generate_samples(num_samples)
        Generate samples from the proposal density

    envelope_factor : double
        Factor M that satifies f(x)<=Mg(x). Set M such that inequality is as 
        close to equality as possible

    num_vars : integer
        The number of variables

    num_samples : integer
        The number of samples required

    verbose : boolean
        Flag specifying whether to print diagnostic information

    batch_size : integer
        The number of evaluations of each density to be performed in a batch.
        Almost always we should set batch_size=num_samples

    Returns
    -------
    samples : np.ndarray (num_vars, num_samples)
        Independent samples from the target distribution
    """
    if batch_size is None:
        batch_size = num_samples

    cntr = 0
    num_proposal_samples = 0
    samples = np.empty((num_vars, num_samples), dtype=float)
    while cntr < num_samples:
        proposal_samples = generate_proposal_samples(batch_size)
        target_density_vals = target_density(proposal_samples)
        proposal_density_vals = proposal_density(proposal_samples)
        assert target_density_vals.shape[0] == batch_size
        assert proposal_density_vals.shape[0] == batch_size
        urand = np.random.uniform(0., 1., (batch_size))

        # ensure envelop_factor is large enough
        if np.any(target_density_vals > (envelope_factor*proposal_density_vals)):
            I = np.argmax(
                target_density_vals/(envelope_factor*proposal_density_vals))
            msg = 'proposal_density*envelop factor does not bound target '
            msg += 'density: %f,%f' % (
                target_density_vals[I],
                (envelope_factor*proposal_density_vals)[I])
            raise ValueError(msg)

        I = np.where(
            urand < target_density_vals/(envelope_factor*proposal_density_vals))[0]

        num_batch_samples_accepted = min(I.shape[0], num_samples-cntr)
        I = I[:num_batch_samples_accepted]
        samples[:, cntr:cntr+num_batch_samples_accepted] = proposal_samples[:, I]
        cntr += num_batch_samples_accepted
        num_proposal_samples += batch_size

    if verbose:
        print(('num accepted', num_samples))
        print(('num rejected', num_proposal_samples-num_samples))
        print(('inverse envelope factor', 1/envelope_factor))
        print(('acceptance probability', float(
            num_samples)/float(num_proposal_samples)))
    return samples
