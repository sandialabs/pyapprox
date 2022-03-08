r"""
Low-discrepancy quadrature
--------------------------
Monte Carlo
===========
It is often important to quantify statistics of numerical models. Monte Carlo
quadrature is the simplest and most robust method for doing so. For any model
we can compute statistics, such as mean and variance as follows
"""
import pyapprox as pya
from scipy import stats
from pyapprox.benchmarks.benchmarks import setup_benchmark
benchmark = setup_benchmark("genz", test_name="oscillatory", nvars=2)
nsamples = 100
mc_samples = benchmark.variable.rvs(nsamples)
values = benchmark.fun(mc_samples)
mean = values.mean(axis=0)
variance = values.var(axis=0)
print("mean", mean)
print("variance", variance)

#%%
#Although not necessary for this scalar valued function, we in geenral must
#use the args axis=0 to mean and variance so that we compute the statistics
#for each quantity of interest (column of values)
#
#Sobol Sequences
#===============
#Large number of samples are needed to compute statistics using Monte Carlo
#quadrature. Low-discrepancy sequences can be used to improve accuracy for a
#fixed number of samples. We can compute statistics using Sobol sequences as
#follows

sobol_samples = pya.sobol_sequence(
    benchmark.variable.num_vars(), nsamples, variable=benchmark.variable)
values = benchmark.fun(sobol_samples)
pya.print_statistics(sobol_samples, values)

#%%
#Here we have used print statistics to compute the sample stats. Note,
#the inverse cumulative distribution functions of the univariate variables in
#variable are used to ensure that the samples integrate with respect to the
#joint distribution of variable. If the variable argument is not provided the
#Sobol sequence will be generated on the unit hybercube :math:`[0,1]^D`.
#
#Low-discrepancy sequences are typically more evenly space over the parameter
#space. This can be seen by comparing the Monte Carlo and Sobol sequence samples
from pyapprox.configure_plots import plt
plt.plot(mc_samples[0, :], mc_samples[1, :], 'ko', label="MC")
plt.plot(sobol_samples[0, :], sobol_samples[1, :], 'rs', label="Sobol")
plt.legend()
plt.show()

#
#Halton Sequences
#================
#Pyapprox also supports Halton Sequences
halton_samples = pya.halton_sequence(
    benchmark.variable.num_vars(), nsamples, variable=benchmark.variable)
values = benchmark.fun(halton_samples)
pya.print_statistics(halton_samples, values)

#%%
#Latin Hypercube Designs
#=======================
#Latin Hypercube designs are another popular means to compute low-discrepancy
#samples for quadrature. Pyapprox does not support such designs as unlike
#Sobol and Halton Sequences they are not nested. For example, when using Latin
#Hypercube designs, it is not possible to requrest 100 samples
#then request 200 samples while resuing the first 100.
