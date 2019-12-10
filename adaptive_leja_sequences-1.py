import numpy as np
from pyapprox.configure_plots import *
from pyapprox.adaptive_polynomial_chaos import *
from pyapprox.variable_transformations import \
AffineBoundedVariableTransformation, AffineRandomVariableTransformation
from pyapprox.variables import IndependentMultivariateRandomVariable
from scipy.stats import beta
from pyapprox.probability_measure_sampling import \
    generate_independent_random_samples
from pyapprox.adaptive_sparse_grid import max_level_admissibility_function, \
    isotropic_refinement_indicator
from pyapprox.univariate_quadrature import clenshaw_curtis_rule_growth, \
    constant_increment_growth_rule
from functools import partial
from scipy.stats import uniform,beta
from pyapprox.models.genz import GenzFunction

def compute_l2_error(validation_samples,validation_values,pce,relative=True):
    pce_values = pce(validation_samples)
    error = np.linalg.norm(pce_values-validation_values,axis=0)
    if not relative:
        error /=np.sqrt(validation_samples.shape[1])
    else:
        error /= np.linalg.norm(validation_values,axis=0)

    return error

np.random.seed(1)