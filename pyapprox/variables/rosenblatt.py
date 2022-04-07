import numpy as np

from pyapprox.util.utilities import get_tensor_product_quadrature_rule


def invert_cdf(F, cdffun, x_limits, tol=1e-12, num_bins=101, plot=False):
    """
    Evaluate the inverse cdf of the function handle cdffun at the points F.
    Just does bisection.

    Parameters
    ----------
    F : np.ndarray (num_samples)
        The locations at which to evaluate the inverse cdf

    cdffun : callable vals = cdffun(samples)
        Funciton that returns the value of the cdf at a set of samples

    limits : np.ndarray (2)
        Lower and upper bounds [lb,ub] of the random variable associated with
        cdffun

    tol : float
        Terminate bisection once we get samples that come this close

    num_bins : integer
        The number of bins use to get good initial point for inversion.

    Returns
    -------
    values : np.ndarray(num_samples)
        The values of the inverse cdf at the samples F.
    """
    # Do a cheap first-pass binning to speed things up
    bin_xs = np.linspace(x_limits[0], x_limits[1], (num_bins))
    bin_Fs = cdffun(bin_xs)
    if plot:
        import pylab
        pylab.plot(bin_xs, bin_Fs)
        pylab.show()
    num_samples = F.shape[0]
    # Return the indices of the bins to which each value in input array
    # belongs
    bins = np.digitize(F, bin_Fs)
    # if get error when using this function with Rosenblatt transformation
    #   bins = np.digitize( F, bin_Fs )
    #   bins must be monotonically increasing or decreasing
    # Then this is likely caused by inaccurate quadrature

    values = np.zeros(num_samples)
    for q in range(num_samples):
        if bins[q] == 0:
            # This happens when F(q) is not strictly in [0,1].
            # It's probably 1 + mach_eps
            # Force F(q) to be 1, augment bin to end
            bins[q] = bin_xs.shape[0]
            values[q] = 1

        elif bins[q] == bin_xs.shape[0]:
            # F has value 1 (or greater?!) and so we just return 1
            values[q] = 1

        else:
            left = bin_xs[bins[q]-1]
            right = bin_xs[bins[q]]

            while (right-left) > tol:
                mid = 1./2.*(left + right)
                mid_F = cdffun(mid)

                if mid_F <= F[q]:
                    left = mid
                else:
                    right = mid

            values[q] = 1./2.*(left + right)

    return values


def combine_samples_with_fixed_data(
        fixed_data, fixed_data_indices, sub_samples):
    assert sub_samples.ndim == 2
    if fixed_data.shape[0] == 0:
        return sub_samples
    # assert fixed_data.shape[1]==sub_samples.shape[1]
    num_vars = fixed_data.shape[0] + sub_samples.shape[0]
    num_samples = sub_samples.shape[1]
    samples = np.empty((num_vars, num_samples), dtype=float)
    samples[fixed_data_indices, :] = fixed_data[:, np.newaxis]
    mask = np.ones((num_vars), dtype=bool)
    mask[fixed_data_indices] = False
    samples[mask, :] = sub_samples
    return samples


def marginal_pdf(joint_density, active_vars, limits, samples,
                 num_quad_samples_1d=100, quad_rule=None):
    """
    Parameters
    ----------

    num_quad_samples_1d : integer
        The number of quadrature samples in the univariate quadrature rule
        used to construct tensor product quadrature rule
    """
    assert samples.ndim == 2
    assert active_vars.shape[0] == samples.shape[0]
    diff = np.diff(limits)[::2]
    assert np.all(diff > 0)

    num_vars = limits.shape[0]//2
    mask = np.ones((num_vars), dtype=bool)
    mask[active_vars] = False
    marginalized_vars = np.arange(num_vars)[mask]

    num_marginalized_vars = num_vars - samples.shape[0]
    if num_marginalized_vars > 0:
        if quad_rule is None:
            quad_x, quad_w = get_tensor_product_quadrature_rule(
                num_quad_samples_1d, num_marginalized_vars,
                np.polynomial.legendre.leggauss)
        else:
            quad_x, quad_w = quad_rule[0].copy(), quad_rule[1].copy()
            assert quad_x.min() >= -1. and quad_x.max() <= 1.
            assert quad_x.shape[0] == num_marginalized_vars

        for ii in range(num_marginalized_vars):
            lb = limits[2*marginalized_vars[ii]]
            ub = limits[2*marginalized_vars[ii]+1]
            quad_x[ii, :] = (quad_x[ii, :]+1.)/2*(ub-lb)+lb
            quad_w *= (ub-lb)/2.0

    num_samples = samples.shape[1]
    values = np.empty((num_samples), dtype=float)
    for jj in range(num_samples):
        fixed_data = samples[:, jj]
        xx = combine_samples_with_fixed_data(
            fixed_data, active_vars, quad_x)
        density_vals = joint_density(xx)
        values[jj] = np.dot(density_vals, quad_w)
    return values


def marginalized_cumulative_distribution_function(
        joint_density, limits, active_vars, active_var_samples,
        inactive_vars, fixed_var_samples, num_quad_samples_1d=100, quad_rule=None):
    """
    Given a set of fixed values for variables, indexed by I, and a set of
    variables to be marginalized, indexed by J, compute the CDF
    at samples of the values indexed by K, where 

    I\\intersect J\\intersect K=\\emptyset and
    I\\union J\\union = {1,...,num_vars}.

    Parameters
    ----------
    joint_density : callable vals = joint_density(samples)
        The joint density f(x) of the random variables x

    limits : np.ndarray (2*num_vars)
        The bounds of the random variables

    active_vars : np.ndarray (num_active_vars)
        The indices (K) of the variables at which we will be evaluating
        the CDF.

    active_var_samples : np.ndarray (num_vars, num_samples)
        The point at which to evaluate the CDF

    inactive_vars : np.ndarray (num_active_vars)
        The indices (J) of the variables that will be marginalized out.

    fixed_var_samples : np.ndarray (num_vars, num_samples)
        The samples of at which the joint density will be fixed before CDF
        is computed and marginalized.

    num_quad_samples_1d : integer
        The number of quadrature samples in the univariate quadrature rule
        used to construct tensor product quadrature rule

    Returns
    -------
    values : np.ndarray (num_samples)
       The values of the CDF at the samples
    """
    num_samples = active_var_samples.shape[1]
    if (fixed_var_samples.shape[1] == 1 and
            fixed_var_samples.shape[1] != num_samples):
        fixed_var_samples = np.tile(fixed_var_samples, (1, num_samples))

    num_samples = active_var_samples.shape[1]
    num_vars = limits.shape[0]//2
    num_active_vars = active_vars.shape[0]
    num_inactive_vars = inactive_vars.shape[0]
    num_fixed_vars = num_vars-(num_active_vars+num_inactive_vars)

    diff = np.diff(limits)[::2]
    assert np.all(diff > 0)
    assert active_var_samples.ndim == 2
    assert active_var_samples.shape[0] == num_active_vars
    assert num_fixed_vars == fixed_var_samples.shape[0]
    assert fixed_var_samples.shape[1] == num_samples

    mask = np.ones((num_vars), dtype=bool)
    mask[active_vars] = False
    mask[inactive_vars] = False
    fixed_vars = np.arange(num_vars)[mask]

    if quad_rule is None:
        quad_x, quad_w = get_tensor_product_quadrature_rule(
            num_quad_samples_1d, num_active_vars+num_inactive_vars,
            np.polynomial.legendre.leggauss)
    else:
        quad_x, quad_w = quad_rule
        assert quad_x.shape[0] == num_active_vars+num_inactive_vars

    values = np.empty((num_samples), dtype=float)
    integration_vars = np.hstack((active_vars, inactive_vars))
    inactive_ubs = limits[2*inactive_vars+1]
    for jj in range(num_samples):
        # limits of integration
        w = quad_w.copy()
        active_x = np.empty_like(quad_x)
        ubs = np.hstack((active_var_samples[:, jj], inactive_ubs))
        for ii in range(integration_vars.shape[0]):
            lb = limits[2*integration_vars[ii]]
            ub = ubs[ii]
            #assert (ub-lb)>0
            active_x[ii, :] = (quad_x[ii, :]+1.)/2.*(ub-lb)+lb
            w *= (ub-lb)/2.0

        fixed_data = fixed_var_samples[:, jj]
        xx = combine_samples_with_fixed_data(
            fixed_data, fixed_vars, active_x)
        density_vals = joint_density(xx)
        values[jj] = np.dot(density_vals.T, w)
    return values


def rosenblatt_transformation(samples, joint_density, limits, num_quad_samples_1d=100):
    assert samples.ndim == 2
    trans_samples = np.empty_like(samples, dtype=float)
    num_vars, num_samples = samples.shape

    trans_samples[0, :] = marginalized_cumulative_distribution_function(
        joint_density, limits, np.arange(
            1), samples[0:1, :], np.arange(1, num_vars),
        np.empty((0, num_samples)), num_quad_samples_1d)
    for ii in range(1, num_vars):
        trans_samples[ii, :] = \
            marginalized_cumulative_distribution_function(
            joint_density, limits, np.asarray([ii]), samples[ii:ii+1, :],
            np.arange(ii+1, num_vars), samples[:ii, :], num_quad_samples_1d)
        active_vars = np.arange(ii)
        trans_samples[ii, :] /= marginal_pdf(
            joint_density, active_vars, limits, samples[:ii, :])
    return trans_samples


def inverse_rosenblatt_transformation(samples, joint_density, limits,
                                      num_quad_samples_1d=100,
                                      tol=1e-12, num_bins=101):
    assert samples.ndim == 2
    num_vars, num_samples = samples.shape
    quad_x, quad_w = get_tensor_product_quadrature_rule(
        num_quad_samples_1d, num_vars, np.polynomial.legendre.leggauss)

    def cdffun(x):
        if np.isscalar(x):
            x = np.asarray([x])
        assert x.ndim == 1
        return marginalized_cumulative_distribution_function(
            joint_density, limits, np.arange(1), x[np.newaxis, :],
            np.arange(1, num_vars), np.empty(
                (0, x.shape[0])), num_quad_samples_1d,
            (quad_x, quad_w))

    trans_samples = np.empty((num_vars, num_samples), dtype=float)
    for jj in range(num_samples):
        trans_samples[0, jj] = invert_cdf(samples[0, jj:jj+1], cdffun, limits[:2],
                                          tol, num_bins)

    for ii in range(1, num_vars):
        active_vars = np.arange(ii)

        quad_x, quad_w = get_tensor_product_quadrature_rule(
            num_quad_samples_1d, 1+(num_vars-ii-1),
            np.polynomial.legendre.leggauss)

        # Even though invert_cdf can be used for multiple samples
        # The following cdf impicitly uses the inactive samples which
        # are size of num_samples but invert_cdf solves problem
        # one point at a time so active_samples and inactive samples
        # will be inconsistent
        for jj in range(num_samples):
            def cdffun(x):
                if np.isscalar(x):
                    x = np.asarray([x])
                assert x.ndim == 1
                cdf_val = marginalized_cumulative_distribution_function(
                    joint_density, limits, np.asarray([ii]), x[np.newaxis, :],
                    np.arange(ii+1, num_vars), trans_samples[:ii, jj:jj+1],
                    num_quad_samples_1d, (quad_x, quad_w))
                pdf_val = marginal_pdf(
                    joint_density, active_vars, limits,
                    trans_samples[:ii, jj:jj+1], num_quad_samples_1d,
                    quad_rule=(quad_x, quad_w))
                return cdf_val / pdf_val

            icdf_val = invert_cdf(
                samples[ii, jj:jj+1], cdffun, limits[2*ii:2*ii+2], tol, num_bins)
            trans_samples[ii, jj] = icdf_val
    return trans_samples


def inverse_rosenblatt_transformation_from_polynomial_chaos_expansion(
        samples, pce, limits, num_quad_samples_1d=100,
        tol=1e-12, num_bins=101):
    # Cannot precompute marginalizations like when using tensor-train

    # Can precompute unique indices and repeated_idx for each
    # marginalization. This will not change with the sample. Create class
    # compute these once then store as member variable

    # marginalized_cumulative_distribution_function will only require 1d
    # integral. Call conditional_moments_of_polynomial_chaos_expansion for
    # each point in quadrature rule

    # compute pce approximation of sqrt of likelihood then squaring pce, i.e.
    # increasing index set and collecting like terms. Alternatively I can
    # just use mean,var = conditional_moments_of_polynomial_chaos_expansion
    # and marginal of square of pce is var+mean**2
    pass
