import warnings
import numpy as np

def hermite_univariate_equilibrium_sampling(num_pts,degree=1):
    """
    Univariate scaled sampling from the weighted equilibrium measure on
    (-Inf, Inf).

    Samples num_pts samples from the univariate weighted equilibrium
    measure, whose unnormalized density is

        rho(x) = sqrt( 2 - x^2 ),        x \in ( -sqrt(2), sqrt(2) ).

    The samples from this density are subsequently expanded by
    sqrt(degree).
    """

    warnings.warn("Use hermite_equilibrium_sampling instead", DeprecationWarning)

    assert degree > 0

    # Sample from beta(1.5, 1.5) on [0, 1], and transform to [-1,1].
    # Then scale by sqrt(2*degree)
    return np.sqrt(2*degree) * (2 * np.random.beta(3/2., 3/2., size=num_pts) - 1.)

def hermite_radial_equilibrium_pdf(r, num_vars):
    """
    Computes the *marginal* density along the radial direction of the
    multivariate hermite equilibrium measure. Since the multivariate
    density has support on a l2 ball of radius sqrt(2), this
    marginalized density has an extra factor of r^(num_vars-1), which is
    a Jacobian factor on the (num_vars)-dimensional unit ball.

    The form of this density is

        rho(r) = sqrt( 2 - r^2 )^(num_vars/2) * r^(num_vars-1)

    for 0 <= r <= sqrt(2).
    """

    from scipy.special import beta

    normalization = 2**(num_vars-1.)*beta(num_vars/2., num_vars/2. + 1)

    return 1/normalization * (2.-r**2)**(num_vars/2.) * r**(num_vars-1)

def hermite_univariate_equilibrium_cdf(x):
    """
    Evalutes the cumulative distribution function of the one-dimensional
    Hermite equilibrium measure.
    """

    from scipy.stats import beta
    import scipy
    if scipy.__version__ == '0.11.0':
        return beta.cdf(x/(2*np.sqrt(2))+0.5, 1.5, 1.5)
    else:
        return beta.cdf(x/(2*np.sqrt(2))+0.5, a=1.5, b=1.5)

def hermite_radial_sampling(N, num_vars):
    """
    Returns N iid samples from a random variable corresponding to the
    radius (l2 norm) of a num_vars-dimensional hermite equilibrium
    measure sample.

    The density function for this random variable is

        rho(r) = sqrt( 2 - r^2 )^(num_vars/2) * r^(num_vars-1)

    for 0 <= r <= sqrt(2).
    """

    assert num_vars > 0

    r = np.random.beta(num_vars/2., num_vars/2.+1, size=N)
    return np.sqrt(2)*np.sqrt(r)

def hermite_equilibrium_sampling(num_pts, num_vars=1, degree=1, a=1.0):
    """
    Multivariate scaled sampling from the (conjectured) weighted
    equilibrium measure on (-Inf, Inf)^num_vars.

    Samples num_pts samples from the multivariate weighted equilibrium
    measure, whose unnormalized density is

        rho(x) = sqrt( 2 - ||x||^2 )^(num_vars/2),  ||x|| < sqrt(2)

    where ||x|| is the l2 norm of the vector x. The samples from this
    density are subsequently expanded by sqrt(degree).

    This assumes sampling for Hermite polynomials that are orthonormal
    under the weight function exp(-a*||x||^2).
    """

    # Generate uniformly on surface of sphere
    x = np.random.normal( 0., 1., ( num_vars, num_pts) );
    norms = np.sqrt( np.sum(x**2, axis=0) );
    assert norms.shape[0] == num_pts

    # Sample according to the sqrt(degree)-scaled equilibrium measure
    # Also build in 1/norms factor that puts x on surface of unit ball
    r = hermite_radial_sampling(num_pts, num_vars)
    r *= np.sqrt(degree)/norms
    r /= np.sqrt(a)

    for d in range( num_vars ):
        x[d,:] *= r

    return x
