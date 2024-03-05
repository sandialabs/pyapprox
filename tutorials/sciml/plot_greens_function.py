r"""
Green's Function Example
========================

Consider the constant-coefficient diffusion equation

.. math::

   -\kappa \nabla^2 u(x) &= f(x) && \qquad x\in \mathcal{D}\\
    u(x) &= 0 && \qquad x\in \partial \mathcal{D}

The Green's function :math:`G(x, y)`, for some :math:`y\in\mathcal{D}` is the
solution to

.. math::

   -\kappa \nabla^2 G(x, y) &= \delta(x-y) && \qquad x\in \mathcal{D}\\
    G(x, y) &= 0 && \qquad x\in \partial \mathcal{D}

Using the Green's function the solution of the PDE satisfies


.. math::
   u(x) = \int_\mathcal{D} G(x, y)f(y)\dx{y}


This can be verified by noting

.. math::

 -\kappa \nabla^2 u(x) &= -\kappa \int_\mathcal{D} \nabla^2 G(x, y)f(y)\dx{y}\\
 & = \int_\mathcal{D} \delta(x-y) f(y)\dx{y}\\
 &= f(x)


The Green's function for the constant coefficient diffusion equation with
:math:`\mathcal{D}=(0, 1)` and homogeneous boundary conditions is

.. math:: G(x, y) = \frac{1}{2\kappa}(x+y-|x-y|- 2x y)

The following code computes the solution to the Laplace equation by using the
trapezoid rule to compute the integral of the Green's function with the forcing
function and compares the result against the exact solution.
"""
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.sciml.quadrature import Fixed1DGaussLegendreIOQuadRule
from pyapprox.sciml.network import CERTANN
from pyapprox.sciml.activations import TanhActivation, IdentityActivation
from pyapprox.sciml.util.hyperparameter import LogHyperParameterTransform
from pyapprox.sciml.integraloperators import (
    KernelIntegralOperator, ChebyshevIntegralOperator,
    DenseAffineIntegralOperator)
from pyapprox.sciml.kernels import (
    ConstantKernel, MaternKernel, Legendre1DHilbertSchmidtKernel)
from pyapprox.sciml.kernels import HomogeneousLaplace1DGreensKernel
from pyapprox.sciml.quadrature import Fixed1DTrapezoidIOQuadRule
from pyapprox.sciml.util import fct
from pyapprox.sciml.util._torch_wrappers import asarray

np.random.seed(1)

kappa = 0.1
nquad = 100
greens_fun = HomogeneousLaplace1DGreensKernel(kappa, [1e-3, 1])
# TODO currently quadrature rules defined on [0, 1] need to pass
# a transform that defines them on a user specified domain
quad_rule = Fixed1DTrapezoidIOQuadRule(nquad)


def forc_fun(xx):
    return (-19.2*xx**4*(1 - xx)**2 + 51.2*xx**3*(1 - xx)**3 -
            19.2*xx**2*(1 - xx)**4).T


def exact_solution(xx):
    return (16*xx**4*(1 - xx)**4).T


def greens_solution(kernel, forc, xx):
    quad_xx, quad_ww = quad_rule.get_samples_weights()
    return kernel(xx, quad_xx)*forc(quad_xx)[:, 0] @ quad_ww


plot_xx = np.linspace(0, 1, 101)[None, :]
green_sol = greens_solution(greens_fun, forc_fun, plot_xx)
ax = plt.figure().gca()
ax.plot(plot_xx[0], exact_solution(plot_xx), label=r"$u(x)$")
ax.plot(plot_xx[0], green_sol, '--', label=r"$u_G(x)$")
ax.plot(plot_xx[0], forc_fun(plot_xx), label=r"$f(x)=-\kappa\nabla^2 u(x)$")
ax.legend()
plt.show()


# %%
# Now plot the greens function
ax = plt.figure().gca()
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
G = greens_fun(plot_xx, plot_xx)
greens_plot = ax.imshow(G, origin="lower", extent=[0, 1, 0, 1], cmap="jet")
plt.show()


# %%
# CERTANN
# -------
# Now let's learn the Green's function using a PYAPPROX.SCIML. First load necessary
# modules


# %%
# Now plot the linear integral operator (not CERTANN) with fixed kernel
# hyper-parameters (the weights of the terms in the Hilbert-Schmidt sum)
nterms = 30
hs_kernel = Legendre1DHilbertSchmidtKernel(
    nterms, 1/np.arange(1, nterms+1)**1, [1e-2, 1])
# Replace above hs_kernel with Matern kernel to see how approximation changes
# hs_kernel = MaternKernel(0.5, 0.1, [1e-2, 1], 1)
const_kernel = ConstantKernel(
    10, [1e-2, 1e4], transform=LogHyperParameterTransform())
final_kernel = const_kernel*hs_kernel
green_sol_hs = greens_solution(final_kernel, forc_fun, plot_xx)
ax = plt.figure().gca()
ax.plot(plot_xx[0], exact_solution(plot_xx), label=r"$u(x)$")
ax.plot(plot_xx[0], green_sol_hs, '--', label=r"$u_{HS}(x)$")
ax.legend()
plt.show()


# %%
# Plot the Hilbert-Schmidt kernel used
ax = plt.figure().gca()
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
Z = final_kernel(plot_xx, plot_xx)
im = ax.imshow(
    Z, origin="lower", extent=[0, 1, 0, 1], cmap="jet")
plt.colorbar(im, ax=ax)
plt.show()


# %%
# Now let's build a CERTANN using random samples of a parameterized polynomial
# forcing function. The following defines the forcing function and generates
# training data.

nfterms = 4  # the number of unknown coefficients parameterizing the forcing


def parameterized_forc_fun(coef, xx):
    return ((xx.T**np.arange(len(coef))[None, :]) @ coef)[:, None]
    # coef = coef.reshape(coef.shape[0]//2, 2)
    # return np.hstack([np.cos(2*c[0]*np.pi*xx.T+c[1])
    #                  for c in coef]).sum(axis=1)[:, None]


nphys_vars = 1
# Set the number of evaluations of the forcing function per random sample
ninputs = 40
# Set the number of random training samples.
ntrain_samples = 10
abscissa = np.linspace(0, 1, ninputs)[None, :]
noutputs = abscissa.shape[1]
train_coef = np.random.normal(0, 1, (nfterms, ntrain_samples))
train_forc_funs = [
    partial(parameterized_forc_fun, coef) for coef in train_coef.T]
# The training samples shape is (ninputs, nntrain_samples)
train_samples = np.hstack([f(abscissa) for f in train_forc_funs])
# The training samples shape is (nntrain_samples, noutputs)
train_values = np.hstack(
    [greens_solution(greens_fun, f, abscissa) for f in train_forc_funs])


# Set the number of CERTANN layers
nlayers = 2
# Set the matern smoothness parameter of the first kernel
nu = np.inf
# Set the kernels for each layer
kernels = [MaternKernel(nu, [0.1], [1e-5, 1], nphys_vars)
           for ii in range(nlayers-1)]+[final_kernel]

# Use Gauss-Legendre Quadrature
QuadRule = Fixed1DGaussLegendreIOQuadRule

# Set the quadrature rules for each layer. Note Last quad rule is only
# used to set the locations X of the kernel(X,Y) in the final integral operator
quad_rules = (
    [QuadRule(ninputs)] +
    [QuadRule(nquad) for kl in range(nlayers-1)] +
    [QuadRule(noutputs)])

# Set the integral operators for each layer. They each need to know
# two quadrature rules
integral_ops = (
    [KernelIntegralOperator(
        kernels[kk], quad_rules[kk], quad_rules[kk+1])
     for kk in range(len(kernels))])

# Set the activations for each layer. The last layer has no activation function
activations = (
    [TanhActivation() for ii in range(nlayers-1)] +
    [IdentityActivation()])

# Initialize the CERTANN
ctn = CERTANN(ninputs, integral_ops, activations)


# Fit the CERTANN
ctn.fit(train_samples, train_values)

# Print the CERTANN
print(ctn, ctn._hyp_list.get_values().shape)

# %%
# Plot the CERTANN evaluations at the training samples to see if
# they resemble training values. Many Kernels will not even pass this
# weak test
ctn_sol = ctn(train_samples)
exact_sol = train_values
ax = plt.figure().gca()
ax.plot(abscissa[0], exact_sol, '-k')
ax.plot(abscissa[0], ctn_sol.numpy(), 'r--')
plt.show()


val_coef = np.random.normal(0, 1, (nfterms, ntrain_samples))
val_forc_funs = [
    partial(parameterized_forc_fun, coef) for coef in val_coef.T]
val_samples = np.hstack([f(abscissa) for f in val_forc_funs])
val_values = np.hstack(
    [greens_solution(greens_fun, f, abscissa) for f in val_forc_funs])
ctn_sol = ctn(val_samples)
exact_sol = val_values
print(np.linalg.norm(ctn_sol.numpy().flatten()-exact_sol.flatten()) /
      np.linalg.norm(exact_sol.flatten()))

# %%
# Plot the learnt kernel
plot_xx = np.linspace(0, 1, 101)[None, :]
ax = plt.figure().gca()
X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
Z = final_kernel(plot_xx, plot_xx)
im = ax.imshow(
    Z, origin="lower", extent=[0, 1, 0, 1], cmap="jet")
plt.colorbar(im, ax=ax)
plt.show()

# Print the final kernel variance
print(const_kernel)

# Print the Hilbert-Schmidt Kernel weights
print(hs_kernel)
# The __repr__ function called by print(hs_kernel)
# will not print all the weights because there are so many so call get_values
if isinstance(hs_kernel, Legendre1DHilbertSchmidtKernel):
    print(hs_kernel._weights.get_values())


# %%
# Now we'll examine how the Green's function performs when approximated with a
# truncated Fourier/Chebyshev expansion. For fixed :math:`x \in \mathcal{D}`,
#
# .. math::
#   u(x) &= \int_{-1}^1 G(x,y) \, f(y) \dx{y} \\
#   &\approx\int_{-1}^1 \left(\sum_{n=0}^N c_n \phi_n(y; x)\right)f(y)\dx{y} \\
#   &= \tilde{u}(x)

# %%
# First, we do a Fourier transform and retain 7 symmetric coefficients.


def greens_solution_fourier(kernel, forc, xx, N):
    quad_xx, quad_ww = quad_rule.get_samples_weights()
    coefs = np.fft.fft(kernel(quad_xx, xx).numpy(), axis=-1)
    if N == 0:
        coefs[:, 1:] = 0
    else:
        coefs[:, N:-N+1] = 0
    kvals = np.fft.ifft(coefs, axis=-1).T
    return kvals*forc(quad_xx)[:, 0].numpy() @ quad_ww.numpy()


plot_xx = np.arange(101)[None, :]/101
green_sol = greens_solution_fourier(greens_fun, forc_fun, plot_xx, N=4)
ax = plt.figure().gca()
ax.plot(plot_xx[0], exact_solution(plot_xx), label=r"$u(x)$")
ax.plot(plot_xx[0], green_sol, '--', label=r"$\tilde{u}_F(x)$")
ax.plot(plot_xx[0], forc_fun(plot_xx), label=r"$f(x)=-\kappa\nabla^2 u(x)$")
ax.set_title(r'Truncated Fourier expansion, 7 terms')
ax.legend()
plt.show()

# %%
# Now we'll do a Chebyshev transform and retain 7 coefficients.


def greens_solution_chebyshev(kernel, forc, xx, N):
    pts = (np.cos(np.arange(101)*np.pi/100)+1)/2
    coefs = fct.fct(kernel(xx, pts[None, :]).T)[:N, :]
    quad_xx, quad_ww = quad_rule.get_samples_weights()
    basis = fct.chebyshev_poly_basis(2*quad_xx-1, N)
    return (basis.T @ coefs).T*(forc(quad_xx)[:, 0]) @ quad_ww


plot_xx = np.linspace(0, 1, 101)[None, :]
green_sol = greens_solution_chebyshev(greens_fun, forc_fun, plot_xx, N=7)
ax = plt.figure().gca()
ax.plot(plot_xx[0], exact_solution(plot_xx), label=r"$u(x)$")
ax.plot(plot_xx[0], green_sol, '--', label=r"$\tilde{u}_C(x)$")
ax.plot(plot_xx[0], forc_fun(plot_xx), label=r"$f(x)=-\kappa\nabla^2 u(x)$")
ax.set_title(r'Truncated Chebyshev expansion, 7 terms')
ax.legend()
plt.show()

# %%
# We see that the Fourier and Chebyshev coefficients decay rapidly enough that
# only a handful of terms are necessary for an accurate Green's function.
#
# Chebyshev Tensor-Product Kernel
# -------------------------------
# We will now learn the action of integrating against a Green's function using
# a :ref:`Chebyshev tensor-product kernel <chebyshev-tensor-product-kernel>`.
# The two changes from before are the abscissas (Chebyshev extrema) and the
# parameter :math:`k_\text{max}`, the maximum degree.

# Set the number of random training samples.
ntrain_samples = 10
level = 5
nx = 2**level + 1
abscissa = 0.5*(1+np.cos(np.pi*np.arange(nx)/(nx-1))[None, :])
kmax = 6
noutputs = abscissa.shape[1]
train_coef = np.random.normal(0, 1, (nfterms, ntrain_samples))
train_forc_funs = [
    partial(parameterized_forc_fun, coef) for coef in train_coef.T]
train_samples = np.hstack([f(abscissa) for f in train_forc_funs])
train_values = np.hstack(
    [greens_solution(greens_fun, f, abscissa) for f in train_forc_funs])

ctn = CERTANN(nx, [ChebyshevIntegralOperator(kmax, chol=False)],
              [IdentityActivation()])
ctn.fit(train_samples, train_values, verbosity=1, tol=1e-14)

print(ctn)

# %%
# Now let's see how the CERTANN does on a test set.

ntest_samples = 5
test_coef = np.random.normal(0, 1, (nfterms, ntest_samples))
test_forc_funs = [
    partial(parameterized_forc_fun, coef) for coef in test_coef.T]
test_samples = np.hstack([f(abscissa) for f in test_forc_funs])
test_values = np.hstack(
    [greens_solution(greens_fun, f, abscissa) for f in test_forc_funs])
ctn_sol = ctn(test_samples)
exact_sol = test_values

ax = plt.figure().gca()
ax.plot(abscissa[0], exact_sol, '-k')
ax.plot(abscissa[0], ctn_sol.numpy(), 'r--')
plt.xlabel(r'$x$')
plt.title(r'Exact $u$ (black), predicted $u$ (red), $k_\mathrm{max} = %d$' %
          kmax)
plt.show()

print('Relative error:', np.linalg.norm(
    ctn_sol.numpy().flatten() - exact_sol.flatten()) / np.linalg.norm(
    exact_sol.flatten()))


# %%
# With similar training data and network sizes, a Chebyshev tensor-product
# kernel obtains significantly lower error than a general Hilbert--Schmidt
# kernel.
#
# Let's see how well we learn the Green's function with a Chebyshev kernel. An
# extra factor of 2 appears in :math:`K(x,y)` due to the change of variables
#
# .. math::
#   \tilde{x} = (x+1)/2,
#
# which maps the canonical Chebyshev domain :math:`[-1,1]` to
# :math:`\mathcal{D} = [0,1]`.

# Convert parameters to matrix form
cheb_U = ctn._hyp_list.get_values()
U = np.zeros((kmax+1, kmax+1))
c = 0
diag_idx = range(kmax+1)
for k in diag_idx:
    U[k, k:] = cheb_U[c:c+kmax+1-k]
    c += kmax+1-k
A = U.T + U
A[diag_idx, diag_idx] = U[diag_idx, diag_idx]

w = 1.0 / (1e-14+np.sqrt(1-(2*plot_xx[0]-1)**2))
w[0] = (w[1] + (plot_xx[0, 2] - plot_xx[0, 1]) / (
    plot_xx[0, 0] - plot_xx[0, 1]) * (w[2] - w[1]))
w[-1] = w[0]
Phi = fct.chebyshev_poly_basis(2*asarray(plot_xx)-1.0, kmax+1).numpy()
fig, ax = plt.subplots(1, 2)
K = 2 * np.diag(w) @ (Phi.T @ (A @ Phi)) @ np.diag(w)
ax[0].imshow(
    K, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=0, vmax=2.5)
ax[1].imshow(
    G, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=0, vmax=2.5)
ax[0].set_title(r'Learned $K(x,y)$, with $k_\mathrm{max} = %d$' % kmax)
ax[1].set_title(r'True $G(x,y)$')
ax[0].set_xlabel(r'$x$')
ax[1].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[1].set_ylabel(r'$y$')
fig.set_size_inches(10, 5)
plt.show()

# %%
# A Green's function corresponds to a space of input functions, so the sampling
# procedure of training functions will affect the learned operator. This is why
# :math:`K(x,y)` looks markedly different from :math:`G(x,y)`.
#
# How will the Chebyshev tensor kernel compare to a dense multilayer perceptron
# (MLP) with a single hidden layer? Let's start by generating training and
# testing data with a coarser discretization than we used for plotting.

# Use 9 nodes and 40 training samples of the forcing function
level = 3
nx = 2**level+1
ntrain_samples = 40
abscissa = 0.5*(1+np.cos(np.pi*np.arange(nx)/(nx-1))[None, :])
kmax = 6
noutputs = abscissa.shape[1]
train_coef = np.random.normal(0, 1, (nfterms, ntrain_samples))
train_forc_funs = [
    partial(parameterized_forc_fun, coef) for coef in train_coef.T]
train_samples = np.hstack([f(abscissa) for f in train_forc_funs])
train_values = np.hstack(
    [greens_solution(greens_fun, f, abscissa) for f in train_forc_funs])

# Use 10 test samples with the same nodes as before
ntest_samples = 10
test_coef = np.random.normal(0, 1, (nfterms, ntest_samples))
test_forc_funs = [
    partial(parameterized_forc_fun, coef) for coef in test_coef.T]
test_samples = np.hstack([f(abscissa) for f in test_forc_funs])
test_values = np.hstack(
    [greens_solution(greens_fun, f, abscissa) for f in test_forc_funs])

# %%
# With data in hand, let's run the experiments. First up: Chebyshev.
print('CHEBYSHEV TENSOR-PRODUCT KERNEL\n')
print('Network size | Rel test err')
print('---------------------------')
cheb_size, cheb_err = [], []
for kmax in range(0, 9, 2):
    ctn = CERTANN(
        nx, [ChebyshevIntegralOperator(kmax)], [IdentityActivation()])
    ctn.fit(train_samples, train_values, tol=1e-10)
    approx_values = ctn(test_samples)
    cheb_size.append(ctn._hyp_list.get_values().shape[0])
    cheb_err.append(
        np.linalg.norm(approx_values-test_values, 'fro') /
        np.linalg.norm(test_values, 'fro'))
    print('%8d     | %10.3e' % (cheb_size[-1], cheb_err[-1]))


# %%
# Now, let's do the MLP.

print('SINGLE-LAYER MLP\n')
print('Network size | Rel test err')
print('---------------------------')
mlp_size, mlp_err = [], []
for width in range(4):
    integralops = [DenseAffineIntegralOperator(nx, width),
                   DenseAffineIntegralOperator(width, nx)]
    activations = 2*[IdentityActivation()]
    ctn = CERTANN(nx, integralops, activations)
    ctn.fit(train_samples, train_values, tol=1e-14)
    approx_values = ctn(test_samples)
    mlp_size.append(ctn._hyp_list.get_values().shape[0])
    mlp_err.append(
        np.linalg.norm(approx_values-test_values, 'fro') /
        np.linalg.norm(test_values, 'fro'))
    print('%8d     | %10.3e' % (mlp_size[-1], mlp_err[-1]))

# %%
# A side-by-side plot shows a that the prediction error is an order of
# magnitude lower with Chebyshev kernels than with a dense MLP. Axes are chosen
# for consistency with later convergence plots.

plt.semilogy(cheb_size, cheb_err, 'ko-', label='Chebyshev kernel', linewidth=2)
plt.semilogy(mlp_size, mlp_err, 'bs--', label='Single-layer MLP', linewidth=2)
plt.grid()
plt.title(r'Approximation of $f \mapsto u$: %d training polynomials, %d nodes'
          % (ntrain_samples, nx))
plt.xlabel('Learnable parameters')
plt.ylabel('Relative validation error in $u$')
plt.tight_layout()
plt.xlim([0, 250])
plt.ylim([1e-4, 1.2])
plt.legend()
plt.show()


# %%
#
# Sampling Dirac Deltas
# ---------------------
#
# In this section, we will repeat the previous experiments using
# (approximations of) Dirac delta functions as input functions:

x = [0]
nfterms = 40
c = fct.chebyshev_poly_basis(asarray(x), nfterms).numpy()
xx = np.linspace(-1, 1, 201)
A = fct.chebyshev_poly_basis(asarray(xx), nfterms).numpy().T
plt.plot(xx, A @ c)
plt.ylim([-5, 25])
plt.grid()
plt.title(r'Chebyshev series for $\delta(x)$ with %d terms' % nfterms)
plt.show()

# %%
# Now we re-harvest training data with approximate Dirac deltas.


def dirac_delta_approx(mass_points, eval_points):
    nterms = 50  # num Chebyshev polynomials to approximate Dirac delta
    mass_points_transformed = 2.0*mass_points-1.0
    c = fct.chebyshev_poly_basis(asarray(mass_points_transformed),
                                 nterms).numpy()
    eval_points_transformed = 2.0*eval_points-1.0
    Phi = fct.chebyshev_poly_basis(asarray(eval_points_transformed),
                                   nterms).numpy().T
    return (Phi @ c)


nphys_vars = 1
# Set the number of evaluations of the forcing function per random sample
level = 5
nx = 2**level+1
# Set the number of random training samples.
ntrain_samples = 50
abscissa = 0.5*(1+np.cos(np.pi*np.arange(nx)/(nx-1))[None, :])
kmax = 20
noutputs = abscissa.shape[1]
train_mass_pts = np.random.uniform(0, 1, (ntrain_samples,))
train_forc_funs = [
    partial(dirac_delta_approx, mass_pt) for mass_pt in train_mass_pts]
train_samples = np.hstack([f(abscissa) for f in train_forc_funs])
train_values = np.hstack(
    [greens_solution(greens_fun, f, abscissa) for f in train_forc_funs])

# %%
# Now, train the CERTANN

ctn = CERTANN(nx, [ChebyshevIntegralOperator(kmax)], [IdentityActivation()])
ctn.fit(train_samples, train_values, tol=1e-12)

# %%
# Now let's see how the CERTANN does on a test set.

test_mass_pts = np.random.uniform(0, 1, (5,))
test_forc_funs = [
    partial(dirac_delta_approx, mass_pt) for mass_pt in test_mass_pts]
test_samples = np.hstack([f(abscissa) for f in test_forc_funs])
test_values = np.hstack(
    [greens_solution(greens_fun, f, abscissa) for f in test_forc_funs])
ctn_sol = ctn(test_samples)
exact_sol = test_values

ax = plt.figure().gca()
ax.plot(abscissa[0], exact_sol, '-k')
ax.plot(abscissa[0], ctn_sol.numpy(), 'r--')
plt.xlabel(r'$x$')
plt.title(r'Exact $u$ (black), predicted $u$ (red), $k_\mathrm{max} = %d$' %
          kmax)
plt.show()

print('Relative error:', np.linalg.norm(
    ctn_sol.numpy().flatten() - exact_sol.flatten()) / np.linalg.norm(
    exact_sol.flatten()))

# %%
# We do very well on out-of-training predictions. Now plot the learned
# :math:`K(x,y)`.

# Convert parameters to matrix form
cheb_U = ctn._hyp_list.get_values()
U = np.zeros((kmax+1, kmax+1))
c = 0
diag_idx = range(kmax+1)
for k in diag_idx:
    U[k, k:] = cheb_U[c:c+kmax+1-k]
    c += kmax+1-k
A = U.T + U
A[diag_idx, diag_idx] = U[diag_idx, diag_idx]

w = 1.0 / (1e-14+np.sqrt(1-(2*plot_xx[0]-1)**2))
w[0] = (w[1] + (plot_xx[0, 2] - plot_xx[0, 1]) / (
    plot_xx[0, 0] - plot_xx[0, 1]) * (w[2] - w[1]))
w[-1] = w[0]
Phi = fct.chebyshev_poly_basis(2*asarray(plot_xx)-1.0, kmax+1).numpy()
fig, ax = plt.subplots(1, 2)
K = 2 * np.diag(w) @ (Phi.T @ (A @ Phi)) @ np.diag(w)
ax[0].imshow(
    K, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=0, vmax=2.5)
ax[1].imshow(
    G, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=0, vmax=2.5)
ax[0].set_title(r'Learned $K(x,y)$, with $k_\mathrm{max} = %d$' % kmax)
ax[1].set_title(r'True $G(x,y)$')
ax[0].set_xlabel(r'$x$')
ax[1].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[1].set_ylabel(r'$y$')
fig.set_size_inches(10, 5)
plt.show()

# %%
# With Dirac deltas as inputs, the learned :math:`K(x,y)` is a more accurate
# representation of :math:`G(x,y)`.
#
# We now perform a convergence study for Chebyshev kernels vs. MLP.

print('CHEBYSHEV TENSOR-PRODUCT KERNEL\n')
print('Network size | Rel test err')
print('---------------------------')
cheb_size, cheb_err = [], []
for kmax in range(0, 21, 2):
    ctn = CERTANN(
        nx, [ChebyshevIntegralOperator(kmax)], [IdentityActivation()])
    ctn.fit(train_samples, train_values, tol=1e-10)
    approx_values = ctn(test_samples)
    cheb_size.append(ctn._hyp_list.get_values().shape[0])
    cheb_err.append(
        np.linalg.norm(approx_values-test_values, 'fro') /
        np.linalg.norm(test_values, 'fro'))
    cheb_U = ctn._hyp_list.get_values()
    print('%8d     | %10.3e' % (cheb_size[-1], cheb_err[-1]))

print('\n\nSINGLE-LAYER MLP\n')
print('Network size | Rel test err')
print('---------------------------')
mlp_size, mlp_err = [], []
for width in range(4):
    integralops = [DenseAffineIntegralOperator(nx, width),
                   DenseAffineIntegralOperator(width, nx)]
    activations = 2*[IdentityActivation()]
    ctn = CERTANN(nx, integralops, activations)
    ctn.fit(train_samples, train_values, tol=1e-10)
    approx_values = ctn(test_samples)
    mlp_size.append(ctn._hyp_list.get_values().shape[0])
    mlp_err.append(
        np.linalg.norm(approx_values-test_values, 'fro') /
        np.linalg.norm(test_values, 'fro'))
    print('%8d     | %10.3e' % (mlp_size[-1], mlp_err[-1]))

plt.semilogy(cheb_size, cheb_err, 'ko-', label='Chebyshev kernel', linewidth=2)
plt.semilogy(mlp_size, mlp_err, 'bs--', label='Single-layer MLP', linewidth=2)
plt.grid()
plt.title(r'Approximation of $f \mapsto u$: %d Dirac deltas, %d nodes' %
          (ntrain_samples, nx))
plt.xlabel('Learnable parameters')
plt.ylabel('Relative validation error in $u$')
plt.legend()
plt.xlim([0, 250])
plt.ylim([1e-4, 1.2])
plt.tight_layout()
plt.show()

# %%
# As expected, the convergence rates are significantly slower with Dirac
# delta-like functions than with polynomials, but Chebyshev kernels still
# outperform MLPs by an order of magnitude.
