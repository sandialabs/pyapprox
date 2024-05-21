r"""
Univariate Interpolation
========================

This tutorial will present methods to approximate univariate functions :math:`\hat{f}_{\alpha}:\reals\to\reals` using interpolation. An interpolant of a function :math:`f` is a weighted linear combination of basis functions

.. math:: f_{\alpha,\beta}(\rv_i)=\sum_{j=1}^M f_{\alpha}(\rv_i^{(j)})\phi_{i,j}(\rv_i),

where the weights are the evaluation on the function :math:`f` on a set of points :math:`\rv_i^{(j)}, j=1\ldots,M_\beta`. Here :math:`\beta` is an index that controls the number of points in the interpolant and the indices :math:`\alpha` and :math:`i` can be ingored in this tutorial.

I included :math:`\alpha` and :math:`i` because they will be useful in later tutotorials that build multi-fidelity multi-variate interpolants of functions :math:`f_\alpha:\reals^D\to\reals`. In such cases i is used to denote the dimension :math:`i=1\ldots,D` and :math:`\alpha` is an index functions of different accuracy (fidelity).


The properties of interpolants depends on two factors. The way the interpolation points :math:`\rv_i^{(j)}` are constructed and the form of the basis functions :math:`\phi`.

In this tutorial I will introduce two types on interpolation strategies based on local and global basis functions.


Lagrange Interpolation
----------------------
Lagrange interpolation uses global polynomials as basis functions

.. math:: \phi_{i,j}(\rv_i) = \prod_{k=1,k\neq j}^{m_{\beta_i}}\frac{\rv_i-\rv_i^{(k)}}{\rv_i^{(j)}-\rv_i^{(k)}}.

The error of a Lagrange interpolant on the interval :math:`[a,b]` is given by

.. math:: e(\rv_i) = f_\alpha(\rv_i)- f_{\alpha,\beta}(\rv_i)\leq\frac{f_\alpha^{(m_\beta+1)}(c)}{(m_\beta+1)!}\prod_{j=1}^{m_{\beta_i}}(\rv_i-\rv_i^{(j)})

where :math:`f_\alpha^{(n+1)}(c)` is the (n+1)-th derivative of the function at some point in :math:`c\in[a,b]`.

The error in the polynomial is bounded by

.. math:: \max_{\rv_i\in[a,b]}\lvert e(\rv_i)\rvert = \max_{\rv_i\in[a,b]}\left\lvert f_\alpha(\rv_i)- f_{\alpha,\beta}(\rv_i)\right\rvert  \leq \frac{1}{(n+1)!} \max_{\rv_i\in[a,b]}\prod_{j=1}^{m_{\beta_i}}\left\lvert (\rv_i-\rv_i^{(j)})\right\rvert\max_{c\in[a,b]} \left\lvert f_\alpha^{(n+1)}(c)\right\rvert

This result shows that the choice of inteprolation points matter. Specificallly, we can not change the derivatives of a function but we can attempt to minimize

.. math:: \prod_{j=1}^{m_{\beta_i}}\left\lvert(\rv_i-\rv_i^{(j)})\right\rvert

Chebyshev points are a very good choice of interpolation points that produce a small value of the quantity above. They are given by

.. math::

  z_{i}^{(j)}=\cos\left(\frac{(j-1)\pi}{m_{\beta_i}-1}\right),\qquad j=1,\ldots,m_{\beta_i}


Piecewise polynomial interpolation
----------------------------------


.. math:: \max_{\rv\in[\rv_i^{(j)},\rv_i^{(j+1)}]}\lvert f_\alpha(\rv_i)-f_{\alpha,\beta}(\rv_i)\lvert \leq \frac{h^3}{72\sqrt{3}} \max_{\rv_i\in[\rv_i^{(j)},\rv_i^{(j+1)}]} f_\alpha^{(3)}(\rv_i)

A proof of this lemma can be found `here <https://eng.libretexts.org/Workbench/Math%2C_Numerics%2C_and_Programming_(Ethan's)/01%3A_Unit_I_-_(Numerical)_Calculus_and_Elementary_Programming_Concepts/1.02%3A_Interpolation/1.2.01%3A_Interpolation_of_Univariate_Functions>`_
"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.surrogates.interp.tensorprod import (
    UnivariatePiecewiseQuadraticBasis, UnivariateLagrangeBasis,
    TensorProductInterpolant)


#The following code compares polynomial and piecewise polynomial univariate basis functions.
nnodes = 5
samples = np.linspace(-1, 1, 201)[None, :]
ax = plt.subplots(1, 2, figsize=(2*8, 6), sharey=True)[1]
cheby_nodes = np.cos(np.arange(nnodes)*np.pi/(nnodes-1))[None, :]
lagrange_basis = UnivariateLagrangeBasis()
lagrange_basis.set_nodes(cheby_nodes)
lagrange_basis_vals = lagrange_basis(samples)
ax[0].plot(samples[0], lagrange_basis_vals)
ax[0].plot(cheby_nodes[0], cheby_nodes[0]*0, 'ko')
equidistant_nodes = np.linspace(-1, 1, nnodes)[None, :]
quadratic_basis = UnivariatePiecewiseQuadraticBasis()
quadratic_basis.set_nodes(equidistant_nodes)
piecewise_basis_vals = quadratic_basis(samples)
_ = ax[1].plot(samples[0], piecewise_basis_vals)
_ = ax[1].plot(equidistant_nodes[0], equidistant_nodes[0]*0, 'ko')

#%%
#Notice that the unlike the lagrange basis the picewise polynomial basis is non-zero only on a local region of the input space.
#
#The compares the accuracy of lagrange basis the picewise polynomial approximations for a piecewise continuous function


def fun(samples):
    yy = samples[0].copy()
    yy[yy > 1/3] = 1.0
    yy[yy <= 1/3] = 0.
    return yy[:, None]


lagrange_interpolant = TensorProductInterpolant([lagrange_basis])
quadratic_interpolant = TensorProductInterpolant([quadratic_basis])
axs = plt.subplots(1, 2, figsize=(2*8, 6))[1]
[ax.plot(samples[0], fun(samples)) for ax in axs]
cheby_nodes = -np.cos(np.arange(nnodes)*np.pi/(nnodes-1))[None, :]
values = fun(cheby_nodes)
lagrange_interpolant.fit([cheby_nodes], values)
equidistant_nodes = np.linspace(-1, 1, nnodes)[None, :]
values = fun(equidistant_nodes)
quadratic_interpolant.fit([equidistant_nodes], values)
axs[0].plot(samples[0], lagrange_interpolant(samples), ':')
axs[0].plot(cheby_nodes[0], values, 'o')
axs[1].plot(samples[0], quadratic_interpolant(samples), '--')
_ = axs[1].plot(equidistant_nodes[0], values, 's')

#%%
#The Lagrange polynomials induce oscillations around the discontinuity, which significantly decreases the convergence rate of the approximation. The picewise quadratic also over and undershoots around the discontinuity, but the phenomena is localized.

#%%
#Now lets see how the error changes as we increase the number of nodes

axs = plt.subplots(1, 2, figsize=(2*8, 6))[1]
[ax.plot(samples[0], fun(samples)) for ax in axs]
for nnodes in [3, 5, 9, 17]:
    nodes = np.cos(np.arange(nnodes)*np.pi/(nnodes-1))[None, :]
    values = fun(nodes)
    lagrange_interpolant.fit([nodes], values)
    nodes = np.linspace(-1, 1, nnodes)[None, :]
    values = fun(nodes)
    quadratic_interpolant.fit([nodes], values)
    axs[0].plot(samples[0], lagrange_interpolant(samples), ':')
    _ = axs[1].plot(samples[0], quadratic_interpolant(samples), '--')


#%%
#Probability aware interpolation for UQ
#--------------------------------------
#When interpolants are used for UQ we do not need the approximation to be accurate everywhere but rather only in regions of high-probability. First lets see what happens when we approximate a function using an interpolant that targets accuracy with respect to a dominating measure :math:`\nu` when really needed an approximation that targets accuracy with respect to a different measure :math:`w`.

from scipy import stats
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.benchmarks import setup_benchmark
from pyapprox.surrogates.orthopoly.quadrature import (
    gauss_jacobi_pts_wts_1D)
from pyapprox.util.utilities import cartesian_product

nvars = 1
c = np.array([20])
w = np.array([0])
benchmark = setup_benchmark(
    "genz", test_name="oscillatory", nvars=nvars, coeff=[c, w])

alpha_stat, beta_stat = 11, 11
true_rv = IndependentMarginalsVariable(
    [stats.beta(a=alpha_stat, b=beta_stat)]*nvars)

interp = TensorProductInterpolant([lagrange_basis]*nvars)
opt_interp = TensorProductInterpolant([lagrange_basis]*nvars)

alpha_poly, beta_poly = 0, 0
ntrain_samples = 7
xx = gauss_jacobi_pts_wts_1D(ntrain_samples, alpha_poly, beta_poly)[0]
train_samples = cartesian_product([(xx+1)/2]*nvars)
train_values = benchmark.fun(train_samples)
interp.fit([(xx[None, :]+1)/2]*nvars, train_values)

opt_xx = gauss_jacobi_pts_wts_1D(ntrain_samples, beta_stat-1, alpha_stat-1)[0]
opt_train_samples = cartesian_product([(opt_xx+1)/2]*nvars)
opt_train_values = benchmark.fun(opt_train_samples)
opt_interp.fit([(opt_xx[None, :]+1)/2]*nvars, opt_train_values)


ax = plt.subplots(1, 1, figsize=(8, 6))[1]
plot_xx = np.linspace(0, 1, 101)
true_vals = benchmark.fun(plot_xx[None, :])
pbwt = r"\pi"
ax.plot(plot_xx, true_vals, '-r', label=r'$f(z)$')
ax.plot(plot_xx, interp(plot_xx[None, :]), ':k', label=r'$f_M^\nu$')
ax.plot(train_samples[0], train_values[:, 0], 'ko', ms=10,
        label=r'$\mathcal{Z}_{M}^{\nu}$')
ax.plot(plot_xx, opt_interp(plot_xx[None, :]), '--b', label=r'$f_M^%s$' % pbwt)
ax.plot(opt_train_samples[0], opt_train_values[:, 0], 'bs',
        ms=10, label=r'$\mathcal{Z}_M^%s$' % pbwt)

pdf_vals = true_rv.pdf(plot_xx[None, :])[:, 0]
ax.set_ylim(true_vals.min()-0.1*abs(true_vals.min()),
            max(true_vals.max(), pdf_vals.max())*1.1)
ax.fill_between(
    plot_xx, ax.get_ylim()[0], pdf_vals+ax.get_ylim()[0],
    alpha=0.3, visible=True,
    label=r'$%s(z)$' % pbwt)
ax.set_xlabel(r'$M$', fontsize=24)
_ = ax.legend(fontsize=18, loc="upper right")

#%%
#As you can see the approximation that targets the uniform norm is "more accurate" on average over the domain, but the interpolant that directly targets accuracy with respect to the desired Beta distribution is more accurate in the regions of non-negligible probability.

#%%
#Now lets looks at how the accuracy changes with the "distance" between the dominating and target measures. This demonstrates the numerical impact of the main theorem in [XJD2013]_.


def compute_density_ratio_beta(num, true_rv, alpha_stat_2, beta_stat_2):
    beta_rv2 = IndependentMarginalsVariable(
        [stats.beta(a=alpha_stat_2, b=beta_stat_2)]*nvars)
    print(beta_rv2, true_rv)
    xx = np.random.uniform(0, 1, (nvars, 100000))
    density_ratio = true_rv.pdf(xx)/beta_rv2.pdf(xx)
    II = np.where(np.isnan(density_ratio))[0]
    assert II.shape[0] == 0
    return density_ratio.max()


def compute_L2_error(interp, validation_samples, validation_values):
    nvalidation_samples = validation_values.shape[0]
    approx_vals = interp(validation_samples)
    l2_error = np.linalg.norm(validation_values-approx_vals)/np.sqrt(
        nvalidation_samples)
    return l2_error


nvars = 3
c = np.array([20, 20, 20])
w = np.array([0, 0, 0])
benchmark = setup_benchmark(
    "genz", test_name="oscillatory", nvars=nvars, coeff=[c, w])
true_rv = IndependentMarginalsVariable(
    [stats.beta(a=alpha_stat, b=beta_stat)]*nvars)

nvalidation_samples = 1000
validation_samples = true_rv.rvs(nvalidation_samples)
validation_values = benchmark.fun(validation_samples)
interp = TensorProductInterpolant([lagrange_basis]*nvars)

alpha_polys = np.arange(0., 11., 2.)
ntrain_samples_list = np.arange(2, 20, 2)
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
for alpha_poly in alpha_polys:
    beta_poly = alpha_poly
    density_ratio = compute_density_ratio_beta(
        nvars, true_rv, beta_poly+1, alpha_poly+1)
    results = []
    for ntrain_samples in ntrain_samples_list:
        xx = gauss_jacobi_pts_wts_1D(ntrain_samples, alpha_poly, beta_poly)[0]
        nodes_1d = [(xx+1)/2]*nvars
        train_samples = cartesian_product(nodes_1d)
        train_values = benchmark.fun(train_samples)
        interp.fit([(xx[None, :]+1)/2]*nvars, train_values)
        l2_error = compute_L2_error(
            interp, validation_samples, validation_values)
        results.append(l2_error)
    ax.semilogy(ntrain_samples_list, results,
                label="{0:1.2f}".format(density_ratio))

ax.set_xlabel(r'$M$', fontsize=24)
ax.set_ylabel(r'$\| f-f_M^\nu\|_{L^2_%s}$' % pbwt, fontsize=24)
_ = ax.legend(ncol=2)

#%%
#References
#----------
#.. [XJD2013] `Chen Xiaoxiao, Park Eun-Jae, Xiu Dongbin. A flexible numerical approach for quantification of epistemic uncertainty. J. Comput. Phys., 240 (2013), pp. 211-224 <https://doi.org/10.1016/j.jcp.2013.01.018>`_
