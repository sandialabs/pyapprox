from scipy.stats._distn_infrastructure import rv_sample
import numpy as np


def is_continuous_variable(rv):
    from scipy.stats import _continuous_distns
    return bool((rv.dist.name in _continuous_distns._distn_names) or
                rv.dist.name == 'continuous_rv_sample')


def is_bounded_continuous_variable(rv):
    interval = rv.interval(1)
    return bool(is_continuous_variable(rv) and
                rv.dist.name != 'continuous_rv_sample'
                and np.isfinite(interval[0]) and np.isfinite(interval[1]))

def is_bounded_discrete_variable(rv):
    from scipy.stats import _discrete_distns
    interval = rv.interval(1)
    return bool(((rv.dist.name in _discrete_distns._distn_names) or
                (rv.dist.name == 'float_rv_discrete') or
                 (rv.dist.name == 'discrete_chebyshev')) and
                np.isfinite(interval[0]) and np.isfinite(interval[1]))

def get_probability_masses(rv):
    assert is_bounded_discrete_variable(rv)
    name, scales, shapes = get_distribution_info(rv)
    if name == 'float_rv_discrete' or name == 'discrete_chebyshev':
        return rv.dist.xk.copy(), rv.dist.pk.copy()
    elif name == 'hypergeom':
        M, n, N = [shapes[key] for key in ['M', 'n', 'N']]
        xk = np.arange(max(0, N-M+n), min(n, N)+1, dtype=float)
        pk = rv.pmf(xk)
        return xk, pk
    elif name == 'binom':
        n, p = shapes['n'], shapes['p']
        xk = np.arange(0, n+1, dtype=float)
        pk = rv.pmf(xk)
        return xk, pk
    else:
        raise Exception(f'{rv.dist.name} not supported')


def get_distribution_info(rv):
    """
    Shapes and scales can appear in either args of kwargs depending on how
    user initializes frozen object.
    """
    name = rv.dist.name
    args = rv.args
    shape_names = rv.dist.shapes
    if shape_names is not None:
        shape_names = [name.strip() for name in shape_names.split(",")]
        shape_values = [
            rv.args[ii] for ii in range(min(len(rv.args), len(shape_names)))]
        shape_values += [
            rv.kwds[shape_names[ii]]
            for ii in range(len(rv.args), len(shape_names))]
        shapes = dict(zip(shape_names, shape_values))
    else:
        shapes = dict()

    scale_values = [rv.args[ii] for ii in range(len(shapes), len(rv.args))]
    scale_values += [rv.kwds[key] for key in rv.kwds if key not in shapes]
    if len(scale_values) == 0:
        if (type(rv.dist) == float_rv_discrete and
            rv.dist.name != 'discrete_chebyshev' and
                rv.dist.name != 'continuous_rv_sample'):
            lb, ub = rv.dist.xk.min(), rv.dist.xk.max()
            scale_values = [lb, ub-lb]
        else:
            scale_values = [0, 1]
    elif len(scale_values) == 1 and len(rv.args) > len(shapes):
        scale_values += [1.]
    elif len(scale_values) == 1 and 'scale' not in rv.kwds:
        scale_values += [1.]
    elif len(scale_values) == 1 and 'loc' not in rv.kwds:
        scale_values = [0]+scale_values
    scale_names = ['loc', 'scale']
    scales = dict(zip(scale_names, np.atleast_1d(scale_values)))

    if type(rv.dist) == float_rv_discrete:
        # shapes={'xk':rv.dist.xk,'pk':rv.dist.pk}
        xk = rv.dist.xk.copy()
        if (rv.dist.name != 'discrete_chebyshev' and
                rv.dist.name != 'continuous_rv_sample'):
            lb, ub = xk.min(), xk.max()
            xk = (xk-lb)/(ub-lb)
        shapes = {'xk': xk, 'pk': rv.dist.pk}

    return name, scales, shapes


def define_iid_random_variables(rv, num_vars):
    unique_variables = [rv]
    unique_var_indices = [np.arange(num_vars)]
    return IndependentMultivariateRandomVariable(
        unique_variables, unique_var_indices)


def variables_equivalent(rv1, rv2):
    """
    Determine if 2 scipy variables are equivalent

    Let 
    a = beta(1,1,-1,2)
    b = beta(a=1,b=1,loc=-1,scale=2)

    then a==b will return False because .args and .kwds are different
    """
    name1, scales1, shapes1 = get_distribution_info(rv1)
    name2, scales2, shapes2 = get_distribution_info(rv2)
    # print(scales1, shapes1, scales2, shapes2)
    if name1 != name2:
        return False
    if scales1 != scales2:
        return False
    return variable_shapes_equivalent(rv1, rv2)


def get_unique_variables(variables):
    nvars = len(variables)
    unique_variables = [variables[0]]
    unique_var_indices = [[0]]
    for ii in range(1, nvars):
        found = False
        for jj in range(len(unique_variables)):
            if variables_equivalent(variables[ii], unique_variables[jj]):
                unique_var_indices[jj].append(ii)
                found = True
                break
        if not found:
            unique_variables.append(variables[ii])
            unique_var_indices.append([ii])
    return unique_variables, unique_var_indices


def variable_shapes_equivalent(rv1, rv2):
    name1, __, shapes1 = get_distribution_info(rv1)
    name2, __, shapes2 = get_distribution_info(rv2)
    if name1 != name2:
        return False
    if name1 == 'float_rv_discrete' or name1 == 'discrete_chebyshev':
        # xk and pk shapes are list so != comparison will not work
        not_equiv = np.any(shapes1['xk'] != shapes2['xk']) or np.any(
            shapes1['pk'] != shapes2['pk'])
        return not not_equiv
    else:
        return shapes1 == shapes2
    return True


class IndependentMultivariateRandomVariable(object):
    """
    Class representing independent random variables

    Examples
    --------
    >>> from pyapprox.variables import IndependentMultivariateRandomVariable
    >>> from scipy.stats import norm, beta
    >>> marginals = [norm(0,1),beta(0,1),norm()]
    >>> variable = IndependentMultivariateRandomVariable(marginals)
    >>> print(variable)
    I.I.D. Variable
    Number of variables: 3
    Unique variables and global id:
        norm(loc=0,scale=1): z0, z2
        beta(a=0,b=1,loc=0,scale=1): z1
    """

    def __init__(self, unique_variables, unique_variable_indices=None,
                 variable_labels=None):
        if unique_variable_indices is None:
            self.unique_variables, self.unique_variable_indices =\
                get_unique_variables(unique_variables)
        else:
            self.unique_variables = unique_variables.copy()
            self.unique_variable_indices = unique_variable_indices.copy()
        self.nunique_vars = len(self.unique_variables)
        assert self.nunique_vars == len(self.unique_variable_indices)
        self.nvars = 0
        for ii in range(self.nunique_vars):
            self.unique_variable_indices[ii] = np.asarray(
                self.unique_variable_indices[ii])
            self.nvars += self.unique_variable_indices[ii].shape[0]
        if unique_variable_indices is None:
            assert self.nvars == len(unique_variables)
        self.variable_labels = variable_labels

    def num_vars(self):
        return self.nvars

    def all_variables(self):
        all_variables = [None for ii in range(self.nvars)]
        for ii in range(self.nunique_vars):
            for jj in self.unique_variable_indices[ii]:
                all_variables[jj] = self.unique_variables[ii]
        return all_variables

    def get_statistics(self, function_name, **kwargs):
        """
        Get a statistic from each univariate random variable.

        Parameters
        ----------
        function_name : string
            The function name of the scipy random variable statistic of interest

        kwargs : kwargs
            The arguments to the scipy statistic function

        Returns
        -------
        stat : 
            The output of the stat function

        Examples
        --------
        >>> import pyapprox as pya
        >>> from scipy.stats import uniform
        >>> num_vars = 2
        >>> variable = pya.IndependentMultivariateRandomVariable([uniform(-2,3)],[np.arange(num_vars)])
        >>> variable.get_statistics('interval',alpha=1)
        array([[-2.,  1.],
               [-2.,  1.]])
        >>> variable.get_statistics('pdf',x=np.linspace(-2,1,3))
        array([[0.33333333, 0.33333333, 0.33333333],
               [0.33333333, 0.33333333, 0.33333333]])

        """
        for ii in range(self.nunique_vars):
            var = self.unique_variables[ii]
            indices = self.unique_variable_indices[ii]
            stats_ii = np.atleast_1d(getattr(var, function_name)(**kwargs))
            assert stats_ii.ndim == 1
            if ii == 0:
                stats = np.empty((self.num_vars(), stats_ii.shape[0]))
            stats[indices] = stats_ii
        return stats

    def evaluate(self, function_name, x):
        stats = None
        for ii in range(self.nunique_vars):
            var = self.unique_variables[ii]
            indices = self.unique_variable_indices[ii]
            for jj in indices:
                stats_jj = np.atleast_1d(getattr(var, function_name)(x[jj, :]))
                assert stats_jj.ndim == 1
                if stats is None:
                    stats = np.empty((self.num_vars(), stats_jj.shape[0]))
                stats[jj] = stats_jj
        return stats

    def pdf(self, x, log=False):
        assert x.shape[0] == self.num_vars()
        if log is False:
            marginal_vals = self.evaluate('pdf', x)
        else:
            marginal_vals = self.evaluate('logpdf', x)
        return np.prod(marginal_vals, axis=0)[:, None]

    def __str__(self):
        variable_labels = self.variable_labels
        if variable_labels is None:
            variable_labels = ['z%d' % ii for ii in range(self.num_vars())]
        string = 'I.I.D. Variable\n'
        string += f'Number of variables: {self.num_vars()}\n'
        string += 'Unique variables and global id:\n'
        for ii in range(self.nunique_vars):
            var = self.unique_variables[ii]
            indices = self.unique_variable_indices[ii]
            name, scales, shapes = get_distribution_info(var)
            shape_string = ','.join(
                [f'{name}={val}' for name, val in shapes.items()])
            scales_string = ','.join(
                [f'{name}={val}' for name, val in scales.items()])
            string += '    '+var.dist.name + '('
            if len(shapes) > 0:
                string += ','.join([shape_string, scales_string])
            else:
                string += scales_string
            string += '): '
            string += ', '.join(
                [variable_labels[idx] for idx in indices])
            if ii < self.nunique_vars-1:
                string += '\n'
        return string


class float_rv_discrete(rv_sample):
    """Discrete distribution defined on locations represented as floats.

    rv_discrete in scipy only allows for integer locations.

    Currently we only guarantee that overloaded functions and cdf, ppf and moment
    work and are tested
    """

    def __init__(self, a=0, b=np.inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, extradoc=None, seed=None):
        super(float_rv_discrete, self).__init__(
            a, b, name, badvalue, moment_tol, values, inc, longname, shapes, extradoc,
            seed)
        self.xk = self.xk.astype(dtype=float)

    def __new__(cls, *args, **kwds):
        return super(float_rv_discrete, cls).__new__(cls)

    def _rvs(self):
        samples = np.random.choice(self.xk, size=self._size, p=self.pk)
        return samples

    def rvs(self, *args, **kwds):
        """
        Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : None or int or ``np.random.RandomState`` instance, 
            optional
            If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = np.logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            raise ValueError("Domain error in arguments.")

        if np.all(scale == 0):
            return loc*ones(size, 'd')

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            self._random_state = check_random_state(rndm)

        # `size` should just be an argument to _rvs(), but for, um,
        # historical reasons, it is made an attribute that is read
        # by _rvs().
        self._size = size
        vals = self._rvs(*args)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # JDJAKEM: commenting this scipy code out allows for non integer
        # locations
        # # Cast to int if discrete
        # if discrete:
        #     if size == ():
        #         vals = int(vals)
        #     else:
        #         vals = vals.astype(int)

        return vals

    def pdf(self, x):
        x = np.atleast_1d(x)
        vals = np.zeros(x.shape[0])
        for jj in range(x.shape[0]):
            for ii in range(self.xk.shape[0]):
                if self.xk[ii] == x[jj]:
                    vals[jj] = self.pk[ii]
                    break
        return vals


class DesignVariable(object):
    def __init__(self, bounds):
        self.bounds = bounds

    def num_vars(self):
        return len(self.bounds.lb)
