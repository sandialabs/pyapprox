import numpy as np

from pyapprox.variables.marginals import (
    is_bounded_continuous_variable, transform_scale_parameters
)
from pyapprox.variables.rosenblatt import (
    rosenblatt_transformation, inverse_rosenblatt_transformation
)
from pyapprox.variables.nataf import (
    covariance_to_correlation, trans_x_to_u, trans_u_to_x,
    transform_correlations, scipy_gauss_hermite_pts_wts_1D
)
from pyapprox.variables.joint import (
    define_iid_random_variable, IndependentMarginalsVariable
)


def _map_hypercube_samples(current_samples, current_ranges, new_ranges):
    # no error checking or notion of active_vars
    clbs, cubs = current_ranges[0::2], current_ranges[1::2]
    nlbs, nubs = new_ranges[0::2], new_ranges[1::2]
    return ((current_samples.T-clbs)/(cubs-clbs)*(nubs-nlbs)+nlbs).T


def map_hypercube_samples(current_samples, current_ranges, new_ranges,
                          active_vars=None, tol=2*np.finfo(float).eps):
    """
    Transform samples from one hypercube to another hypercube with different
    bounds.

    Parameters
    ----------
    current_samples : np.ndarray (num_vars, num_samples)
        The samples to be transformed

    current_ranges : np.ndarray (2*num_vars)
        The lower and upper bound of each variable of the current samples
        [lb_1,ub_1,...,lb_d,ub_d]

    new_ranges : np.ndarray (2*num_vars)
        The desired lower and upper bound of each variable
        [lb_1,ub_1,...,lb_d,ub_d]

    active_vars : np.ndarray (num_active_vars)
        The active vars to which the variable transformation should be applied
        The inactive vars have the identity applied, i.e. they remain
        unchanged.

    tol : float
        Some functions such as optimizers will create points very close
         but outside current bounds. In this case allow small error
         and move these points to boundary

    Returns
    -------
    new_samples : np.ndarray (num_vars, num_samples)
        The transformed samples
    """
    new_ranges = np.asarray(new_ranges)
    current_ranges = np.asarray(current_ranges)
    if np.allclose(new_ranges, current_ranges):
        return current_samples
    new_samples = current_samples.copy()
    num_vars = current_samples.shape[0]
    assert num_vars == new_ranges.shape[0]//2
    assert num_vars == current_ranges.shape[0]//2
    clbs, cubs = current_ranges[0::2], current_ranges[1::2]
    nlbs, nubs = new_ranges[0::2], new_ranges[1::2]
    if np.any((current_samples.min(axis=1)) <= clbs-tol):
        raise ValueError("samples outside lower bounds")
    if np.any((current_samples.max(axis=1)) >= cubs+tol):
        raise ValueError("samples outside upper bounds")
    II = np.where(current_samples.T < clbs)
    JJ = np.where(current_samples.T > cubs)
    if II[0].shape[0] > 0:
        current_samples[II[1], II[0]] = clbs[II[1]]
    if JJ[0].shape[0] > 0:
        current_samples[JJ[1], JJ[0]] = cubs[JJ[1]]

    if active_vars is None:
        return _map_hypercube_samples(
            current_samples, current_ranges, new_ranges)

    new_samples[active_vars] = (
        (current_samples[active_vars].T-clbs[active_vars])/(
            cubs[active_vars]-clbs[active_vars])*(
                nubs[active_vars]-nlbs[active_vars])+nlbs[active_vars]).T
    return new_samples


class IdentityTransformation(object):
    def __init__(self, num_vars):
        self.nvars = num_vars

    def map_from_canonical(self, samples):
        return samples

    def map_to_canonical(self, samples):
        return samples

    def num_vars(self):
        return self.nvars

    def map_derivatives_from_canonical_space(self, derivatives):
        return derivatives


class AffineBoundedVariableTransformation(object):
    def __init__(self, canonical_ranges, user_ranges):
        assert len(user_ranges) == len(canonical_ranges)
        self.canonical_ranges = np.asarray(canonical_ranges)
        self.user_ranges = np.asarray(user_ranges)
        self.nvars = int(len(self.user_ranges)/2)

    def map_from_canonical(self, canonical_samples):
        return map_hypercube_samples(
            canonical_samples, self.canonical_ranges, self.user_ranges)

    def map_to_canonical(self, user_samples):
        return map_hypercube_samples(
            user_samples, self.user_ranges, self.canonical_ranges)

    def num_vars(self):
        return self.nvars


class AffineTransform(object):
    r"""
    Apply an affine transformation to a
    :py:class:`pyapprox.variables.IndependentMarginalsVariable`
    """

    def __init__(self, variable, enforce_bounds=False):
        """
        Variable uniquness dependes on both the type of random variable
        e.g. beta, gaussian, etc. and the parameters of that distribution
        e.g. loc and scale parameters as well as any additional parameters
        """
        if (type(variable) != IndependentMarginalsVariable):
            variable = IndependentMarginalsVariable(variable)
        self.variable = variable
        self.enforce_bounds = enforce_bounds
        self.identity_map_indices = None

        self.scale_parameters = np.empty((self.variable.nunique_vars, 2))
        for ii in range(self.variable.nunique_vars):
            var = self.variable.unique_variables[ii]
            # name, scale_dict, __ = get_distribution_info(var)
            # copy is essential here because code below modifies scale
            # loc, scale = scale_dict['loc'].copy(), scale_dict['scale'].copy()
            # if (is_bounded_continuous_variable(var) or
            #     (type(var.dist) == float_rv_discrete and
            #      var.dist.name != 'discrete_chebyshev')):
            #     lb, ub = -1, 1
            #     scale /= (ub-lb)
            #     loc = loc-scale*lb
            self.scale_parameters[ii, :] = np.hstack(
                transform_scale_parameters(var))

    def set_identity_maps(self, identity_map_indices):
        """
        Set the dimensions we do not want to map to and from
        canonical space

        Parameters
        ----------
        identity_map_indices : iterable
            The dimensions we do not want to map to and from
            canonical space
        """
        self.identity_map_indices = identity_map_indices

    def map_to_canonical(self, user_samples):
        canonical_samples = user_samples.copy().astype(float)
        for ii in range(self.variable.nunique_vars):
            indices = self.variable.unique_variable_indices[ii]
            loc, scale = self.scale_parameters[ii, :]

            bounds = [loc-scale, loc+scale]
            var = self.variable.unique_variables[ii]
            if self.identity_map_indices is not None:
                active_indices = np.setdiff1d(
                    indices, self.identity_map_indices, assume_unique=True)
            else:
                active_indices = indices
            if ((self.enforce_bounds is True) and
                (is_bounded_continuous_variable(var) is True) and
                ((np.any(user_samples[active_indices, :] < bounds[0])) or
                 (np.any(user_samples[active_indices, :] > bounds[1])))):
                II = np.where((user_samples[active_indices, :] < bounds[0]) |
                              (user_samples[active_indices, :] > bounds[1]))[1]
                print(user_samples[np.ix_(active_indices, II)], bounds)
                raise ValueError(f'Sample outside the bounds {bounds}')

            canonical_samples[active_indices, :] = (
                user_samples[active_indices, :]-loc)/scale

        return canonical_samples

    def map_from_canonical(self, canonical_samples):
        user_samples = canonical_samples.copy().astype(float)
        for ii in range(self.variable.nunique_vars):
            indices = self.variable.unique_variable_indices[ii]
            loc, scale = self.scale_parameters[ii, :]
            if self.identity_map_indices is not None:
                active_indices = np.setdiff1d(
                    indices, self.identity_map_indices, assume_unique=True)
            else:
                active_indices = indices
            user_samples[active_indices, :] = \
                canonical_samples[active_indices, :]*scale+loc

        return user_samples

    def map_to_canonical_1d(self, samples, ii):
        for jj in range(self.variable.nunique_vars):
            if ii in self.variable.unique_variable_indices[jj]:
                loc, scale = self.scale_parameters[jj, :]
                if self.identity_map_indices is None:
                    return (samples-loc)/scale
                else:
                    return samples
        raise Exception()

    def map_from_canonical_1d(self, canonical_samples, ii):
        for jj in range(self.variable.nunique_vars):
            if ii in self.variable.unique_variable_indices[jj]:
                loc, scale = self.scale_parameters[jj, :]
                if self.identity_map_indices is None:
                    return canonical_samples*scale+loc
                else:
                    return canonical_samples
        raise Exception()

    def map_derivatives_from_canonical_space(self, derivatives):
        """
        Parameters
        ----------
        derivatives : np.ndarray (nvars*nsamples, nqoi)
            Derivatives of each qoi. The ith column consists of the derivatives
            [d/dx_1 f(x^{(1)}), ..., f(x^{(M)}),
            d/dx_2 f(x^{(1)}), ..., f(x^{(M)})
            ...,
            d/dx_D f(x^{(1)}), ..., f(x^{(M)})]
            where M is the number of samples and D=nvars

            Derivatives can also be (nvars, nsamples) - transpose of Jacobian -
            Here each sample is considered a different QoI
        """
        assert derivatives.shape[0] % self.num_vars() == 0
        num_samples = int(derivatives.shape[0]/self.num_vars())
        mapped_derivatives = derivatives.copy()
        for ii in range(self.variable.nunique_vars):
            var_indices = self.variable.unique_variable_indices[ii]
            idx = np.tile(var_indices*num_samples, num_samples)+np.tile(
                np.arange(num_samples), var_indices.shape[0])
            loc, scale = self.scale_parameters[ii, :]
            mapped_derivatives[idx, :] /= scale
        return mapped_derivatives

    def map_derivatives_to_canonical_space(self, canonical_derivatives):
        """
        derivatives : np.ndarray (nvars*nsamples, nqoi)
            Derivatives of each qoi. The ith column consists of the derivatives
            [d/dx_1 f(x^{(1)}), ..., f(x^{(M)}),
             d/dx_2 f(x^{(1)}), ..., f(x^{(M)})
             ...,
             d/dx_D f(x^{(1)}), ..., f(x^{(M)})]
            where M is the number of samples and D=nvars

            Derivatives can also be (nvars, nsamples) - transpose of Jacobian -
            Here each sample is considered a different QoI
        """
        assert canonical_derivatives.shape[0] % self.num_vars() == 0
        num_samples = int(canonical_derivatives.shape[0]/self.num_vars())
        derivatives = canonical_derivatives.copy()
        for ii in range(self.variable.nunique_vars):
            var_indices = self.variable.unique_variable_indices[ii]
            idx = np.tile(var_indices*num_samples, num_samples)+np.tile(
                np.arange(num_samples), var_indices.shape[0])
            loc, scale = self.scale_parameters[ii, :]
            derivatives[idx, :] *= scale
        return derivatives

    def num_vars(self):
        return self.variable.num_vars()

    def samples_of_bounded_variables_inside_domain(self, samples):
        for ii in range(self.variable.nunique_vars):
            var = self.variable.unique_variables[ii]
            lb, ub = var.interval(1)
            indices = self.variable.unique_variable_indices[ii]
            if (samples[indices, :].max() > ub):
                print(samples[indices, :].max(), ub, 'ub violated')
                return False
            if samples[indices, :].min() < lb:
                print(samples[indices, :].min(), lb, 'lb violated')
                return False
        return True

    def get_ranges(self):
        ranges = np.empty((2*self.num_vars()), dtype=float)
        for ii in range(self.variable.nunique_vars):
            var = self.variable.unique_variables[ii]
            lb, ub = var.interval(1)
            indices = self.variable.unique_variable_indices[ii]
            ranges[2*indices], ranges[2*indices+1] = lb, ub
        return ranges


def define_iid_random_variable_transformation(variable_1d, num_vars):
    variable = define_iid_random_variable(variable_1d, num_vars)
    var_trans = AffineTransform(variable)
    return var_trans


class RosenblattTransform(object):
    r"""
    Apply the Rosenblatt transformation to an arbitraty multivariate
    random variable.
    """

    def __init__(self, joint_density, num_vars, opts):
        self.joint_density = joint_density
        self.limits = opts['limits']
        self.num_quad_samples_1d = opts['num_quad_samples_1d']
        self.tol = opts.get('tol', 1e-12)
        self.num_bins = opts.get('num_bins', 101)
        self.nvars = num_vars
        self.canonical_variable_types = ['uniform']*self.num_vars()

    def map_from_canonical(self, canonical_samples):
        user_samples = inverse_rosenblatt_transformation(
            canonical_samples, self.joint_density, self.limits,
            self.num_quad_samples_1d, self.tol, self.num_bins)
        return user_samples

    def map_to_canonical(self, user_samples):
        canonical_samples = rosenblatt_transformation(
            user_samples, self.joint_density, self.limits,
            self.num_quad_samples_1d)
        return canonical_samples

    def num_vars(self):
        return self.nvars


class UniformMarginalTransformation(object):
    r"""
    Transform variables to have uniform marginals on [0,1]
    """

    def __init__(self, x_marginal_cdfs, x_marginal_inv_cdfs,
                 enforce_open_bounds=True):
        """
        enforce_open_bounds: boolean
            If True  - enforce that canonical samples are in (0,1)
            If False - enforce that canonical samples are in [0,1]
        """
        self.nvars = len(x_marginal_cdfs)
        self.x_marginal_cdfs = x_marginal_cdfs
        self.x_marginal_inv_cdfs = x_marginal_inv_cdfs
        self.enforce_open_bounds = enforce_open_bounds

    def map_from_canonical(self, canonical_samples):
        # there is a singularity at the boundary of the unit hypercube when
        # mapping to the (semi) unbounded distributions
        if self.enforce_open_bounds:
            assert canonical_samples.min() > 0 and canonical_samples.max() < 1
        user_samples = np.empty_like(canonical_samples)
        for ii in range(self.nvars):
            user_samples[ii, :] = self.x_marginal_inv_cdfs[ii](
                canonical_samples[ii, :])
        return user_samples

    def map_to_canonical(self, user_samples):
        canonical_samples = np.empty_like(user_samples)
        for ii in range(self.nvars):
            canonical_samples[ii, :] = self.x_marginal_cdfs[ii](
                user_samples[ii, :])
        return canonical_samples

    def num_vars(self):
        return self.nvars


class NatafTransform(object):
    r"""
    Apply the Nataf transformation to an arbitraty multivariate
    random variable.
    """

    def __init__(self, x_marginal_cdfs, x_marginal_inv_cdfs,
                 x_marginal_pdfs, x_covariance, x_marginal_means,
                 bisection_opts=dict()):

        self.nvars = len(x_marginal_cdfs)
        self.x_marginal_cdfs = x_marginal_cdfs
        self.x_marginal_inv_cdfs = x_marginal_inv_cdfs
        self.x_marginal_pdfs = x_marginal_pdfs
        self.x_marginal_means = x_marginal_means

        self.x_correlation = covariance_to_correlation(x_covariance)
        self.x_marginal_stdevs = np.sqrt(np.diag(x_covariance))

        quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
        self.z_correlation = transform_correlations(
            self.x_correlation, self.x_marginal_inv_cdfs,
            self.x_marginal_means, self.x_marginal_stdevs, quad_rule,
            bisection_opts)

        self.z_correlation_cholesky_factor = np.linalg.cholesky(
            self.z_correlation)

    def map_from_canonical(self, canonical_samples):
        return trans_u_to_x(
            canonical_samples, self.x_marginal_inv_cdfs,
            self.z_correlation_cholesky_factor)

    def map_to_canonical(self, user_samples):
        return trans_x_to_u(
            user_samples, self.x_marginal_cdfs,
            self.z_correlation_cholesky_factor)

    def num_vars(self):
        return self.nvars


class ComposeTransforms(object):
    r"""
    Apply a composition of transformation to an multivariate
    random variable.
    """

    def __init__(self, transformations):
        """
        Parameters
        ----------
        transformations : list of transformation objects
            The transformations are applied first to last for
            map_to_canonical and in reverse order for
            map_from_canonical
        """
        self.transformations = transformations

    def map_from_canonical(self, canonical_samples):
        user_samples = canonical_samples
        for ii in range(len(self.transformations)-1, -1, -1):
            user_samples = \
                self.transformations[ii].map_from_canonical(user_samples)
        return user_samples

    def map_to_canonical(self, user_samples):
        canonical_samples = user_samples
        for ii in range(len(self.transformations)):
            canonical_samples = \
                self.transformations[ii].map_to_canonical(
                    canonical_samples)
        return canonical_samples

    def num_vars(self):
        return self.transformations[0].num_vars()


class ConfigureVariableTransformation(object):
    """
    Class which maps one-to-one configure indices in [0, 1, 2, 3,...]
    to a set of configure values accepted by a function
    """

    def __init__(self, config_values, labels=None):
        """
        Parameters
        ----------
        nvars : integer
            The number of configure variables

        config_values : list
            The list of configure values for each configure variable.
            Each entry in the list is a 1D np.ndarray with potentiallly
            different sizes
        """

        self.nvars = len(config_values)
        assert (type(config_values[0]) == list or
                type(config_values[0]) == np.ndarray)
        self.config_values = config_values
        self.variable_labels = labels

    def map_from_canonical(self, canonical_samples):
        """
        Map a configure multi-dimensional index to the corresponding
        configure values
        """
        assert canonical_samples.shape[0] == self.nvars
        samples = np.empty_like(canonical_samples, dtype=float)
        for ii in range(samples.shape[1]):
            for jj in range(self.nvars):
                kk = canonical_samples[jj, ii]
                samples[jj, ii] = self.config_values[jj][int(kk)]
        return samples

    def map_to_canonical(self, samples):
        """
        This is the naive slow implementation that searches through all
        canonical samples to find one that matches each sample provided
        """
        assert samples.shape[0] == self.nvars
        canonical_samples = np.empty_like(samples, dtype=float)
        for ii in range(samples.shape[1]):
            for jj in range(self.nvars):
                found = False
                for kk in range(len(self.config_values[jj])):
                    if samples[jj, ii] == self.config_values[jj][int(kk)]:
                        found = True
                        break
                if not found:
                    raise Exception("Configure value not found")
                canonical_samples[jj, ii] = kk
        return canonical_samples

    def num_vars(self):
        """Return the number of configure variables.

        Returns
        -------
        The number of configure variables
        """
        return self.nvars

    def __repr__(self):
        return "{0}(nvars={1}, {2})".format(
            self.__class__.__name__, self.num_vars(), self.config_values)
