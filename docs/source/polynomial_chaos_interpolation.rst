.. _Polynomial Chaos Regression:

Polynomial Chaos Regression
===========================

This tutorial discusses how to construct a polynomial chaos expansion of a function with uncertain parameters using least squares regression.

Lets first import the necessary modules and set the random seed to ensure the exact plots of this tutorial are reproducbile,

.. plot::
   :include-source:
   :context:
      close-figs

   import numpy as np
   from pyapprox.variables.variables import IndependentMultivariateRandomVariable
   from pyapprox.variables.variable_transformations import \
   AffineRandomVariableTransformation
   from pyapprox.polychaos.gpc import PolynomialChaosExpansion,\
   define_poly_options_from_variable_transformation
   from pyapprox.variables.probability_measure_sampling import \
   generate_independent_random_samples
   from scipy.stats import uniform, beta
   from pyapprox.polychaos.indexing import compute_hyperbolic_indices, tensor_product_indices
   from pyapprox.interface.genz import GenzFunction
   from functools import partial
   from pyapprox.orthopoly.quadrature import gauss_jacobi_pts_wts_1D, \
   clenshaw_curtis_in_polynomial_order
   from pyapprox.utilities.utilities import get_tensor_product_quadrature_rule
   from pyapprox.polychaos.polynomial_sampling import christoffel_weights
   
   np.random.seed(1)

Our goal is to demonstrate how to use a polynomial chaos expansion (PCE) to approximate a function :math:`f(z): \reals^d \rightarrow \reals` parameterized by the random variables :math:`z=(z_1,\ldots,z_d)`. with the joint probability density function :math:`\pdf(\V{\rv})`. In the following we will use a function commonly used in the literature, the oscillatory Genz function. This function is well suited for testing as the number of variables and the non-linearity can be adjusted.

.. plot::
   :include-source:
   :context:
      close-figs

   univariate_variables = [uniform(),beta(3,3)]
   variable = IndependentMultivariateRandomVariable(univariate_variables)

   c = np.random.uniform(0.,1.,variable.num_vars())
   c*=4/c.sum()
   w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
   model = GenzFunction( "oscillatory",variable.num_vars(),c=c,w=w )

PCE represent the model output :math:`f(\V{\rv})` as an expansion in orthonormal polynomials, 


.. math::
   :label: eq:pce-integer-index
	   
   f(\V{\rv})\approx p(\V{\rv})=\sum_{n=1}^\infty \alpha_n\phi_n(\V{\rv}),

Here :math:`\alpha_n` are the PCE coefficients that must be computed, and the basis functions :math:`\phi_n` are polynomial basis functions that are pairwise orthonormal under an inner product weighted by the probability density of :math:`\V{\rv}`. Above we assume that :math:`f` is scalar-valued, but the procedures we describe carry over to vector- or function-valued outputs.
In this tutorial the components of :math:`\V{\rv}` are independent and so we can generate the multivariate polynomials :math:`\phi_n` from univariate orthogonal polynomials, but such a construction can also be accomplished when :math:`\V{\rv}` has dependent components. This will be demonstrated in another tutorial.

Polynomial chaos expansions are most easily constructed when the components of :math:`\rv` are independent.  Under the assumption of independence, we have

.. math::
   \begin{align*}
  \rvdom &= \times_{i=1}^d \rvdom_i, & \rvdom_i &\subseteq \reals, & \pdf(\V{\rv}) &= \prod_{i=1}^d \pdf_i(\rv_i),
  \end{align*}

where :math:`\pdf_i` are the marginal densities of the variables :math:`\V{\rv}_i`, which completely characterizes the distribution of :math:`\rv`. This allows us to express the basis functions :math:`\phi` as tensor products of univariate orthonormal polynomials. That is

.. math::
   \phi_\lambda(\V{\rv})=\prod_{i=1}^d \phi^i_{\lambda_i}(\rv_i),

where :math:`\lambda=(\lambda_1\ldots,\lambda_d)\in\mathbb{N}_0^d` is a multi-index, and the univariate basis functions :math:`\phi^i_j` are defined uniquely (up to a sign) for each :math:`i = 1, \ldots, d`, as

.. math::
   \begin{align*}
  \int_{\rvdom_i} \phi^i_{j}(z_i) \phi^i_{k}(z_i) \pdf_i(z_i) \;dz_i &= \delta_{j,k}, & j, k &\geq 0, & \deg \phi^i_j &= j.
  \end{align*}
  
With this notation, the degree of the multivariate polynomial :math:`\phi_\lambda` is :math:`|\lambda| \colon= \sum_{j=1}^d \lambda_j`.

The following code initializes a PCE with univariate polynomials orthonormal to the random variables.

.. plot::
   :include-source:
   :context:
      close-figs

   var_trans = AffineRandomVariableTransformation(variable)
   poly = PolynomialChaosExpansion()
   poly_opts = define_poly_options_from_variable_transformation(var_trans)
   poly.configure(poly_opts)

In practice the PCE  must be truncated to some finite number of terms, say :math:`N`, defined by a multi-index set :math:`\Lambda \subset \mathbb{N}_0^d`:

.. math::
   \begin{align*}
   \label{eq:pce-multi-index}
   f(\V{\rv}) &\approx f_N(\V{\rv}) = \sum_{\lambda\in\Lambda}\alpha_{\lambda}\phi_{\lambda}(\V{\rv}), & |\Lambda| &= N.
   \end{align*}
   :label: eq:pce-multi-index

Frequently the PCE is truncated to retain only the multivariate polynomials whose associated multi-indices have norm at most :math:`p`, i.e.,

.. math::
   \label{eq:hyperbolic-index-set}
   \begin{align*}
   \Lambda &= \Lambda^d_{p,q} = \{\lambda \mid \norm{\lambda}{q} \le p\}., & \left\| \lambda \right\|_q &\coloneqq \left(\sum_{i=1}^d \lambda^q_i\right)^{1/q}.
   \end{align*}

Taking :math:`q=1` results in a total-degree space having dimension :math:`\text{card}\; \Lambda^d_{p,1} \equiv N = { d+p \choose d }`. The choice of :math:`\Lambda` identifies a subspace in which :math:`f_N` has membership:

.. math::
   \begin{align*}
  \pi_\Lambda &\coloneqq \mathrm{span} \left\{ \phi_\lambda \;\; \big| \;\; \lambda \in \Lambda \right\}, & f_N &\in \pi_\Lambda.
  \end{align*}

Under an appropriate ordering of multi-indices, the expression :eq:`eq:pce-integer-index` , and the expression :eq:`eq:pce-multi-index` truncated to the first :math:`N` terms, are identical. Defining :math:`[N]:=\{1,\ldots,N\}`, for :math:`N\in\mathbb{N}`, we will in the following frequently make use of a linear ordering of the PCE basis, :math:`\phi_k` for :math:`k \in [N]` from :eq:`eq:pce-integer-index`, instead of the multi-index ordering of the PCE basis :math:`\phi_{\lambda}` for :math:`\lambda \in \Lambda` from :eq:`eq:pce-multi-index`.  Therefore,

.. math::
  \sum_{\lambda \in \Lambda} \alpha_\lambda \phi_\lambda(\V{\rv}) = \sum_{n=1}^N \alpha_n \phi_n(\V{\rv}).

Any bijective map between :math:`\Lambda` and :math:`[N]` will serve to define this linear ordering, and the particular choice of this map is not relevant in our discussion.

To set the PCE truncation to a third degree total-degree index set use

.. plot::
   :include-source:
   :context:
      close-figs

   degree=3
   indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
   poly.set_indices(indices)

Now we have defined the PCE, we are now must compute its coefficients. Pyapprox supports a number of methods to compute the polynomial coefficients. Here we will use interpolation. Specifically we evaluate the function at a set of samples :math:`\mathcal{Z}=[\V{\rv}^{(1)},\ldots,\V{\rv}^{(M)}]` to obtain a set of function values :math:`\V{f}=[\V{f}^{(1)},\ldots,\V{f}^{(M)}]^T`. The function may be vectored valued and thus each :math:`\V{f}^{(i)}\in\mathbb{R}^Q` is a vector and :math:`\V{F}\in\mathbb{R}^{M\times Q}` is a matrix

In the following we will generate training samples by randomly drawing samples from the tensor-product Chebyshev measure.

.. math::
   \begin{align*}
   v(\V{\rv})=\prod_{i=1}^d v(\rv_i) & & v(\rv_i)=\frac{1}{\pi\sqrt{1-\rv_i^2}}
   \end{align*}

Sampling from this measure is asymptorically optimal (as degree increases) for any bounded random variable [NJZ2017]_. The following code samples from the Chebyshev measure and evaluates the model at these samples.

.. plot::
   :include-source:
   :context:
      close-figs

   ntrain_samples = int(poly.indices.shape[1]*1.1)
   train_samples = -np.cos(np.random.uniform(0,2*np.pi,(poly.num_vars(),ntrain_samples)))
   train_samples = var_trans.map_from_canonical_space(train_samples)
   train_values  = model(train_samples)

Here we have used the variable transformation to map the samples from
:math:`[-1,1]^d\rightarrow[0,1]^d`. More details on how to use variable transformations will be covered
in another tutorial.

The function values we generated can now be used to approximate the polynomial coefficients by solving the least squares system

.. math:: \V{\Phi} \V{\alpha}=\V{F}
	  
where entries of the basis matrix :math:`\V{\Phi}\in\mathbb{R}^{M\times N}` are given by :math:`\Phi_{ij}=\phi_j(\V{\rv}^{(i)})`. Solving this system will be ill conditioned so we must precondition the system using an appropriate preconditioner. The optimal preconditioner when sampling from the Chebyshev measure is a diagonal matrix :math:`\V{w}` with entries 

.. math:: W_{ii}=\left(\sum_{n=1}^N \phi_n^2(\V{\rv}^{(i)})\right)^{-\frac{1}{2}}

We will use numpy's in built least squares function to solve the preconditioned system of equations

.. math:: \V{W}\V{\Phi} \V{\alpha}=\V{W}\V{F}

.. plot::
   :include-source:
   :context:
      close-figs

   basis_matrix = poly.basis_matrix(train_samples)
   precond_weights = christoffel_weights(basis_matrix)
   precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
   precond_train_values = precond_weights[:,np.newaxis]*train_values
   coef = np.linalg.lstsq(precond_basis_matrix,precond_train_values,rcond=None)[0]
   poly.set_coefficients(coef)

Now lets plot the Genz function and the error in the PCE approximation

.. plot::
   :include-source:
   :context:

   plot_limits = [0,1,0,1]
   num_pts_1d = 30
   from pyapprox.utilities.configure_plots import *
   from pyapprox.utilities.visualization import plot_surface, get_meshgrid_function_data

   fig = plt.figure(figsize=(2*8,6))
   ax=fig.add_subplot(1,2,1,projection='3d')
   X,Y,Z = get_meshgrid_function_data(model, plot_limits, num_pts_1d)
   plot_surface(X,Y,Z,ax)

   ax=fig.add_subplot(1,2,2,projection='3d')
   error = lambda x: np.absolute(model(x)-poly(x))
   X,Y,Z = get_meshgrid_function_data(error, plot_limits, num_pts_1d)
   plot_surface(X,Y,Z,ax)
   offset = -(Z.max()-Z.min())/2
   ax.plot(train_samples[0,:],train_samples[1,:],
   #offset*np.ones(train_samples.shape[1]),'o',zorder=100,color='b')
   error(train_samples)[:,0],'o',zorder=100,color='k')
   ax.view_init(80, 45)
   plt.show()

As you can see the error in the interpolant is small near the training points and larger further away from those points.

Notes
^^^^^
In this tutorial we sampled from the Chebyshev measure and applied a preconditioner (known as the Christoffel function) to generate a well-conditioned linear system. Other strategies exists for generating well conditioned systems. We will cover other choices and provide more information on the preconditioning techinque used here in another tutorial. However we want to emphasize that random sampling from the probability measure does not produce a well-conditioned system and should be avoided.
   
References
^^^^^^^^^^
.. [NJZ2017] `Narayan A., Jakeman J., Zhou T. A christoffel function weighted least squares algorithm for collocation approximations Math. Comp., 86 (306) (2017), pp. 1913-1947 <https://doi.org/10.1090/mcom/3192>`_

.. [JNZ2017] `Jakeman J.D., Narayan A., Zhou T. A generalized sampling and preconditioning scheme for sparse approximation of polynomial chaos expansions. SIAM J. Sci. Comput., 39 (3) (2017), pp. A1114-A1144. <https://epubs.siam.org/doi/10.1137/16M1063885>`_
