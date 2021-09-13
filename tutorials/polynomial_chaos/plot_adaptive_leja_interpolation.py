r"""
Adaptive Leja Sequences
=======================
This tutorial describes how to construct a polynomial chaos expansion (PCE) of a function with uncertain parameters using Leja sequences. This tutorial assumes that the reader is familiar with the tutorial in :ref:`Polynomial Chaos Regression`.
"""
#%%
#First lets import necessary modules and define a function useful for estimating the error in the PCE. We will also set the random seed for reproductibility

from pyapprox.sparse_grid import plot_sparse_grid_2d
import numpy as np
import pyapprox as pya
from pyapprox.configure_plots import *
from functools import partial
from pyapprox.benchmarks.benchmarks import setup_benchmark


def compute_l2_error(validation_samples, validation_values, pce, relative=True):
    pce_values = pce(validation_samples)
    error = np.linalg.norm(pce_values-validation_values, axis=0)
    if not relative:
        error /= np.sqrt(validation_samples.shape[1])
    else:
        error /= np.linalg.norm(validation_values, axis=0)

    return error


np.random.seed(1)

#%%
#Our goal is to demonstrate how to use a polynomial chaos expansion (PCE) to approximate a function :math:`f(z): \reals^d \rightarrow \reals` parameterized by the random variables :math:`z=(z_1,\ldots,z_d)`. with the joint probability density function :math:`\pdf(\V{\rv})`. In the following we will use a function commonly used in the literature, the oscillatory Genz function. This function is well suited for testing as the number of variables and the non-linearity can be adjusted. We define the random variables and the function with the following code

c = np.array([10, 0.01])
w = np.zeros(2)
benchmark = setup_benchmark('genz', nvars=2, test_name='oscillatory', coefficients=(c,w))
model = benchmark.fun
variable = benchmark.variable

#%% We can also use other benchmarks, for example by uncommenting the following code

# benchmark = setup_benchmark('ishigami', a=7, b=0.1)
# variable = benchmark.variable
# model = benchmark.fun

#%%
#Here we have intentionally set the coefficients :math:`c`: of the Genz function to be highly anisotropic, to emphasize the properties of the adaptive algorithm.
#
#PCE represent the model output :math:`f(\V{\rv})` as an expansion in orthonormal polynomials,
#
#.. math::
#
#  \begin{align*}
#  f(\V{\rv}) &\approx f_N(\V{\rv}) = \sum_{\lambda\in\Lambda}\alpha_{\lambda}\phi_{\lambda}(\V{\rv}), & |\Lambda| &= N.
#  \end{align*}
#
#where :math:`\lambda=(\lambda_1\ldots,\lambda_d)\in\mathbb{N}_0^d` is a multi-index and :math:`\Lambda` specifies the terms included in the expansion. In :ref:`Polynomial Chaos Regression` we set :math:`\Lambda` to be a total degree expansion. This choice was somewhat arbitray. The exact indices in :math:`\Lambda` should be chosen with more care. The number of terms in a PCE dictates how many samples are need to accurately compute the coefficients of the expansion. Consequently we should choose the index set :math:`\Lambda` in a way that minimizes error for a fixed computational budget. In this tutorial we use an adaptive algorithm to construct an index set that greedily minimizes the error in the PCE.
#
#Before starting the adaptive algorithm  we will generate some test data to estimate the error in the PCE as the adaptive algorithm evolves. We will compute the error at each step using a callback function.

var_trans = pya.AffineRandomVariableTransformation(variable)
validation_samples = pya.generate_independent_random_samples(
    var_trans.variable, int(1e3))
validation_values = model(validation_samples)

errors = []
num_samples = []


def callback(pce):
    error = compute_l2_error(validation_samples, validation_values, pce)
    errors.append(error)
    num_samples.append(pce.samples.shape[1])

#%%
#Now we setup the adaptive algorithm.

max_num_samples = 200
error_tol = 1e-10
candidate_samples = -np.cos(
    np.random.uniform(0, np.pi, (var_trans.nvars, int(1e4))))
pce = pya.AdaptiveLejaPCE(
    var_trans.nvars, candidate_samples, factorization_type='fast')

max_level = np.inf
max_level_1d = [max_level]*(pce.num_vars)

admissibility_function = partial(
    pya.max_level_admissibility_function, max_level, max_level_1d,
    max_num_samples, error_tol)

growth_rule = partial(pya.constant_increment_growth_rule, 2)
#growth_rule = pya.clenshaw_curtis_rule_growth
pce.set_function(model, var_trans)
pce.set_refinement_functions(
    pya.variance_pce_refinement_indicator, admissibility_function,
    growth_rule)

#%%
#The AdaptiveLejaPCE object is used to build an adaptive Leja sequence. Before building the sequence, let us first introduce the basic concepts of Leja sequences.
#
#A Leja sequence (LS) is essentially a doubly-greedy computation of a determinant maximization procedure. Given an existing set of nodes :math:`\mathcal{Z}_M`, a Leja sequence update chooses a new node :math:`\V{\rv}^{(M+1)}` by maximizing the determinant of a new Vandermonde-like matrix with an additional row and column: the additional column is formed by adding a single predetermined new basis element, :math:`\phi_{M+1}`, and the additional row is defined by the newly added point. Hence a LS is both greedy in the chosen interpolation points, and also assumes some *a priori* ordering of the basis elements.
#
#In one dimension, a weighted LS can be understood without linear algebra: Let :math:`\mathcal{Z}_N` be a set of nodes on :math:`\rvdom` with cardinality :math:`N \geq 1`. We will add a new point :math:`z^{(N+1)}` to :math:`\mathcal{Z}` determined by the following:
#
#.. math::
#
#  \argmax_{\rv \in \rvdom} v(\rv)\prod_{n=1}^N |\rv - \rv^{(n)}|
#
#We omit notation indicating the dependence of :math:`z^{(N+1)}` on :math:`\mathcal{Z}_N`.
# By iterating the above equation, one can progressively build up the Leja sequence :math:`\mathcal{Z}` by recomputing and maximizing the objective function for increasing :math:`N`.
#
#Traditionally Leja sequences were developed with :math:`v(\rv)=1`. In the following we use
#
#.. math:: v(\V{\rv})=\left(\sum_{n=1}^N \phi_n^2(\V{\rv}^{(i)})\right)^{-\frac{1}{2}}
#
#which is the square-root of the Christoffel function.
#
#Note univaraite weighted Leja sequence were intially developed setting :math:`v(\V{\rv})=\sqrt{\rho(\V{\rv}}` to be the square-root of the joint probability density of the random variables [NJ2014]_. However using the Christoffel function typically produces more well-conditioned Leja sequences and requires no explicit knowldege of the joint PDF.
#
#In multiple dimensions, formulating a generalization of the univariate procedure is challenging. The following linear algebra formulation greedily maximizes the weighted Vandermonde-like determinant
#
#.. math:: \V{\rv}^{(N+1)} = \argmax_{\rv \in \rvdom} |\det v(\V{\rv}) \Phi(\mathcal{Z}, \V{\rv}^{(N+1)})|.
#
#The above procedure is an optimization with no known explicit solution, so constructing a Leja sequence is challenging. In [NJ2014]_, gradient based optimization was used to construct weighted Leja sequences. However a simpler procedure based upon LU factorization can also be used [JFNMP2019]_. The simpler approach comes at a cost of slight degradation in the achieved determinant of the LS. We adopt the LU-based approach here due to its ease of implementation.
#
#The algorithm for generating weighted Leja sequences using LU factorization is outlined in Algorithm :ref:`Algorithm 1`. The algorithm consists of 5 steps. First a polynomial basis must be specified. The number of polynomial basis elements must be greater than or equal to the number of desired samples in the Leja sequence, i.e. :math:`N \geq M`. The input basis must also be ordered, and the Leja sequence is dependent on this ordering. In this paper we only consider total-degree polynomial spaces, that is we have
#
#.. math::
#   \begin{align*}
#   \mathrm{span}\{\phi_n\}_{n=1}^N &= \pi_\Lambda, & \Lambda = \Lambda_{k,1}^d,
#   \end{align*}
#
#for some polynomial degree :math:`k`. We use lexigraphical ordering on :math:`\Lambda` to define the basis. The second step consists of generating a set of :math:`S` candidate samples :math:`\mathcal{Z}_S`; ideally, :math:`S \gg M`. Our candidate samples will be generated as independent and identically-distributed realizations of a random variable. The precise choice of the random draw will be discussed in the next section. For now we only require that the measure of the draw have support identical with the measure of :math:`Z`. Once candidates have been generated we then form the :math:`S \times N` Vandermonde-like matrix :math:`\Phi`, precondition this matrix with :math:`V`, and compute a truncated LU factorization. (Computing the full LU factorization is expensive and unnecessary.) We terminate the LU factorization algorithm after computing the first :math:`M` pivots. These ordered pivots correspond to indices in the candidate samples that will make up the Leja sequence. If we assume that there is \textit{any} size-:math:`M` subset of :math:`\mathcal{Z}_S` that is unisolvent for interpolation, then by the pivoting procedure, a Leja sequence is always chosen so that the interpolation problem is unisolvent.
#
#Algorithm 1:
#
#   **Require** number of desired samples :math:`M`, preconditioning function :math:`v(\V{\rv})`, basis :math:`\{\phi\}_{n=1}^N`
#
#   #. Choose the index set :math:`\Lambda` such that :math:`N\ge M`
#   #. Specifying an ordering of the basis :math:`\phi`
#   #. Generate set of :math:`S\gg M` candidate samples :math:`\mathcal{Z}_S`
#   #. Build :math:`\Phi`, :math:`\Phi_{m,n} =\phi_n(\V{\rv}^{(m)})`, :math:`m\in[S]`, :math:`n\in[N]`
#   #. Compute preconditioning matrix :math:`V`, :math:`V_{mm}=v(\V{\rv}^{(m)})`
#   #. Compute first M pivots of LU factorization, :math:`PLU=LU(V \Phi`,M)
#
#Once a Leja sequence :math:`\mathcal{Z}_M` has been generated one can easily generate a polynomial interpolant with two simple steps. The first step evaluates the function at the samples in the sequence, i.e. :math:`y=f(\mathcal{Z})`. The coefficients of the PCE interpolant can then be computed via
#
# .. math:: \alpha=(LU)^{-1}P^{-1} V y
#
#where the matrices :math:`P`, :math:`L`, and :math:`U` are identified in :ref:`Algorithm 1`.
#
#These two steps are carried out at each iteration of the adaptive algorithm. The PCE coefficients are used to guide refinement of the polynomial index set :math:`\Lambda`.
#
#In the following we use an adaptive algorithm first developed for generalized sparse grid approximation (this is discussed in another tutorial). At each iteration the algorithm identifies a number of different sets :math:`\mathcal{S}\subset\Lambda` of candidate indices :math:`\V{\lambda}` which may significantly reduce the PCE error. The algorithm then chooses the set :math:`\mathcal{S}` which does produce the biggest change and uses this set to generate new candidate sets :math:`\mathcal{S}` for refinement. Here we use the change in variance induced by a set as a proxy for the change in PCE error. This change in variance is simply the sum of the coefficients squared associated with the set, i.e.
#
#.. math:: \sum_{\lambda\in \mathcal{S}} \alpha_\V{\lambda}^2
#
#We end this section by noting that (approximate) Fekete points are an alternative determinant-maximizing choice for interpolation points. We opt to use Leja sequences here because they are indeed a *sequence*, whereas a Fekete point construction is not nested.
#
#
#Now we are in a position to start the adaptive process

while (not pce.active_subspace_queue.empty() or
       pce.subspace_indices.shape[1] == 0):
    pce.refine()
    pce.recompute_active_subspace_priorities()
    if callback is not None:
        callback(pce)
#%%
#And finally we plot the final polynomial index set :math:`\Lambda` the subspace index set, the Leja sequence, and the decay in error as the number of samples increases.

plot_sparse_grid_2d(
    pce.samples, np.ones(pce.samples.shape[1]),
    pce.pce.indices, pce.subspace_indices)

plt.figure()
plt.loglog(num_samples, errors, 'o-')
plt.show()


#%%
#References
#^^^^^^^^^^
#.. [NJ2014] `Narayan A., Jakeman J.D. Adaptive Leja sparse grid constructions for stochastic collocation and high-dimensional approximation SIAM J. Sci. Comput., 36 (6) (2014), pp. A2952-A2983 <https://doi.org/10.1137/140966368>`_
#
#.. [JFNMP2019] `John D. Jakeman, Fabian Franzelin, Akil Narayan, Michael Eldred, and Dirk Plfuger.  Polynomial chaosexpansions for dependent random variables. Computer Methods in Applied Mechanics and Engineering, 351:643-666, 2019 <https://doi.org/10.1016/j.cma.2019.03.049>`_
