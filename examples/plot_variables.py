r"""
Random Variables
----------------
This tutorial describes how to create multivariate random variable objects.

PyApprox primarily supports multivariate random variables comprised of
independent univariate variables. Such variables can be defined from a list of
scipy.stats variable objects. To create a Beta random variable defined on :math:`[0, 1]\times[1, 2]`
"""
from pyapprox import IndependentRandomVariable
from scipy import stats
scipy_vars = [stats.beta(2, 3, loc=0, scale=1),
              stats.beta(5, 2, loc=1, scale=1)]
variable = IndependentRandomVariable(scipy_vars)

#%%
#A summary of the random variable can be printed using
print(variable)

#%%
# To generate random samples from the multivariate variable use
nsamples = 10
samples = variable.rvs(nsamples)

#%%
#For such 2D variables comprised of continuous univariate random variables we
#can evaluate the joint probability density function (PDF) at a set of samples
#using
pdf_vals = variable.pdf(samples)
print(pdf_vals)

#%%
#Any statistics, supported by the univariate scipy.stats variables,
#can be accessed for all 1D variabels using
#:func:`pyapprox.variables.IndependentRandomVariable.get_statistics`
#For example,
mean = variable.get_statistics("mean")
variance = variable.get_statistics("var")
print("Mean", mean)
print("Variance", variance)

#%%
#Note, by convention PyApprox tries to always return 2D numpy arrays, e.g.
#here we are returning a column vector. Sometimes this is not possible
#because some functions in other packages, such as SciPy, require input as 1D
#arrays.
#
#We can also plot the joint PDF and overlay the random samples.
#Given a number of 1D samples specified by the user, the following plots
#evaluates the PDF (or any 2D function) on a cartesian grid of these 1D samples
#defined on the bounded ranges of the random variables. If some univariate
#variables are unbounded then the range corresponding to a fraction of
#the total probability will be used. See the documentation at
#:func:`pyapprox.util.visualization.get_meshgrid_function_data_from_variable`
from pyapprox import get_meshgrid_function_data_from_variable
nplot_pts_1d = 50
X, Y, Z = get_meshgrid_function_data_from_variable(
    variable.pdf, variable, nplot_pts_1d)

#%%
#Here we will create 2D subplots, a contour plot and a surface plot
from pyapprox.util.configure_plots import plt
import numpy as np
ncontours = 20
fig = plt.figure(figsize=(2*8, 6))
ax0 = fig.add_subplot(1, 2, 1)
ax0.plot(samples[0, :], samples[1, :], 'ro')
ax0.contourf(
    X, Y, Z, zdir='z', levels=np.linspace(Z.min(), Z.max(), ncontours))
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.plot_surface(X, Y, Z)
plt.show()
