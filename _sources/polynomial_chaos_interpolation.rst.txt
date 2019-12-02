Polynomial Chaos Regression
===========================

Often an approximation is not defined over the user defined variable space :math:`\mathcal{X}` but over another space :math:`\mathcal{U}`. This is often done to enhance the numerical stability of the approximation or to facilitate the use of simpler approximations. For instance it usually best practice to convert a set of correlated Normal variables into a set of independent Normal variables using the tranformation

.. math:: T(x)=Lx
   
where :math:`L` is the Cholesky factor of the covariance of the correlated variables

In this case the monomial approximation is always defined on :math:`\mathcal{U}=[-1,1]^d` so lets define a transformation

.. math:: T:[0,1]^d\rightarrow[-1,1]^d

Here we focus on how to build a maximum-degree multivariate polynomial approximation

.. math:: f(x)\approx p(T(x))=p(u)=\sum_{\|\lambda\|_\infty\le p} c_\lambda\phi_\lambda(u),

for some degree :math:`p`.

Pyapprox supports a number of methods to compute the polynomial coefficients. Here we will use interpolation. Specifically we evaluate the function at a set of samples :math:`X=[x^{(1)},\ldots,x^{(M)}]` to obtain a set of function values :math:`Y=[f^{(1)},\ldots,f^{(M)}]^T`. The function may be vectored valued and thus each :math:`f^{(i)}\in\mathbb{R}^Q` is a vector and :math:`Y\in\mathbb{R}^{M\times Q}` is a matrix

In the following we will use the tensor product of the nodes of the univariate Clenshaw-Curtis quadrature rule. However any well conditioned sampling set can be used, such as Leja sequences. Random sampling from the probability measure is not well-conditioned.

These function values can the be used to approximate the polynomial coefficients using least squares system

.. math:: \Phi c=Y
	  
where entries of the basis matrix :math:`\Phi\in\mathbb{R}^{M\times N}` are given by :math:`\Phi_{ij}=\phi_j(x^{(i)})`

Here we will use numpy's in built least squares function to solve the interpolation problem. The following plots show the function being approximated and the magnitude of the error in the polynomial interpolant

.. plot::
   :include-source:

   import numpy as np
   from pyapprox.variables import IndependentMultivariateRandomVariable
   from pyapprox.variable_transformations import \
   AffineRandomVariableTransformation
   from pyapprox.multivariate_polynomials import PolynomialChaosExpansion,\
   define_poly_options_from_variable_transformation
   from pyapprox.probability_measure_sampling import \
   generate_independent_random_samples
   from scipy.stats import uniform, beta
   from pyapprox.indexing import compute_hyperbolic_indices, tensor_product_indices
   from pyapprox.models.genz import GenzFunction
   from functools import partial
   from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D, \
   clenshaw_curtis_in_polynomial_order
   from pyapprox.utilities import get_tensor_product_quadrature_rule

   np.random.seed(1)
   univariate_variables = [
   uniform(),beta(3,3)]
   variable = IndependentMultivariateRandomVariable(univariate_variables)
   var_trans = AffineRandomVariableTransformation(variable)

   c = np.random.uniform(0.,1.,var_trans.num_vars())
   c*=4/c.sum()
   w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
   model = GenzFunction( "oscillatory",var_trans.num_vars(),c=c,w=w )


   poly = PolynomialChaosExpansion()
   poly_opts = define_poly_options_from_variable_transformation(var_trans)
   poly.configure(poly_opts)

   #num_train_samples = indices.shape[1]#*2
   #train_samples = generate_independent_random_samples(
   #    var_trans.variable,num_train_samples)

   #univariate_quadrature_rules = [
   #    partial(gauss_jacobi_pts_wts_1D,alpha_poly=0,beta_poly=0),
   #    partial(gauss_jacobi_pts_wts_1D,alpha_poly=2,beta_poly=2)]
   level=1
   univariate_quadrature_rules = [
   partial(clenshaw_curtis_in_polynomial_order,
   return_weights_for_all_levels=False)]*poly.num_vars()
   train_samples, train_weights = get_tensor_product_quadrature_rule(
   level,var_trans.num_vars(),univariate_quadrature_rules,
   var_trans.map_from_canonical_space)

   degrees = [int(train_samples.shape[1]**(1/poly.num_vars()))]*poly.num_vars()
   #indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
   indices = tensor_product_indices(degrees)
   poly.set_indices(indices)

   train_values = model(train_samples)

   basis_matrix = poly.basis_matrix(train_samples)
   coef = np.linalg.lstsq(basis_matrix,train_values,rcond=None)[0]
   poly.set_coefficients(coef)

   plot_limits = [0,1,0,1]
   num_pts_1d = 30
   from pyapprox.configure_plots import *
   from pyapprox.visualization import plot_surface, get_meshgrid_function_data

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

As you can see the error in the interpolant is zero at the training points.
   
   
If all the variables of the model are independent and indentically distributed then we can construct the variable transformation with the helper function

.. code-block::
   
   var_trans = define_iid_random_variable_transformation(
   uniform(-1,2),num_vars) 
