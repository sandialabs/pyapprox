r"""
Experimental design for Gaussian processes
==========================================
The nature of the training samples significantly impacts the accuracy of a Gaussian process. Noting that the variance of a GP reflects the accuracy of a Gaussian process, [SWMW1989]_ developed an experimental design procedure which minimizes the average variance with respect to a specified measure. This measure is typically the probability measure :math:`\pdf(\rv)` of the random variables :math:`\rv`. Integrated variance designs, as they are often called, find a set of samples :math:`\mathcal{Z}\subset\Omega\subset\rvdom` from a set of candidate samples :math:`\Omega` by solving the minimization problem

.. math:: \mathcal{Z}^\dagger=\argmin_{\mathcal{Z}\subset\Omega\subset\rvdom, \lvert\mathcal{Z}\rvert=M} \int_{\rvdom} C^\star(\rv, \rv\mid \mathcal{Z})\pdf(\rv)d\rv

where we have made explicit the posterior variance dependence on :math:`\mathcal{Z}`.

The variance of a GP is not dependent on the values of the training data, only the sample locations, and thus the procedure can be used to generate batches of samples. The IVAR criterion - also called active learning Cohn (ALC) - can be minimized over discrete [HJZ2021]_ or continuous [GM2016]_ design spaces :math:`\Omega`. When employing a discrete design space, greedy methods [C2006]_ are used to sample one at a time from a finite set of candidate samples to minimize the learning objective.  This approach requires a representative candidate set which, we have found, can be generated with low-discrepancy sequences, e.g. Sobol sequences. The continuous optimization optimization is non-convex and thus requires a good initial guess to start the gradient based optimization. Greedy methods can be used to produce the initial guess, however in certain situation optimizing from the greedy design resulted in minimal improvement.

The following code plots the samples chosen by greedily minimizing the IVAR criterion

.. math:: \int_{\rvdom} C^\star(\rv, \rv\mid \mathcal{Z})\pdf(\rv)d\rv = 1-\mathrm{Trace}\left[A_\mathcal{Z}P_\mathcal{Z}\right]\qquad P_\mathcal{Z}=\int_{\rvdom} A_{\mathcal{Z}\cup\{\rv\}}A_{\mathcal{Z}\cup\{\rv\}}^\top\pdf(\rv)d\rv

from a set of candidate samples :math:`\mathcal{Z}_\mathrm{cand}`. Because the additive constant does not effect the design IVAR designs are found by greedily adding points such that the :math:`N+1` point satisfies

.. math:: \rv_{N+1}=\argmin_{\rv^\prime\in\mathcal{Z}_\mathrm{cand}} \mathrm{Trace}\left[A_{\mathcal{Z}_N\cup\{\rv^\prime\}}P_{\mathcal{Z}_N\cup\{\rv^\prime\}}\right].
"""

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.surrogates.kernels.kernels import MaternKernel
from pyapprox.surrogates.gaussianprocess.exactgp import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.activelearning import (
    GreedyIntegratedVarianceSampler,
    BruteForceGreedyIntegratedVarianceSampler,
    CholeskySampler,
)
from pyapprox.variables.joint import IndependentMarginalsVariable, stats

nvars = 1
variable = IndependentMarginalsVariable([stats.uniform(-1, 2)])

ncandidate_samples = 101
sampler = BruteForceGreedyIntegratedVarianceSampler(
    variable, nquad_nodes_1d=[100], nugget=1e-8
)
kernel = MaternKernel(np.inf, 0.5, (0.1, 1), fixed=True, nvars=nvars)
gp = ExactGaussianProcess(nvars, kernel)
sampler.set_gaussian_process(gp)
ncandidates = 101
candidate_samples = np.linspace(-1, 1, ncandidates)[None, :]
sampler.set_candidate_samples(candidate_samples)
init_pivots = np.array([ncandidates // 2], dtype=int)
sampler.set_initial_pivots(init_pivots)


def plot_gp_samples(sampler, ntrain_samples, kernel, variable):
    axs = plt.subplots(1, ntrain_samples, figsize=(ntrain_samples * 8, 6))[1]
    gp = ExactGaussianProcess(nvars, kernel)
    for ii in range(1, ntrain_samples + 1):
        gp.plot_1d(
            axs[ii - 1],
            variable.get_statistics("interval", 1)[0, :],
            npts_1d=101,
        )

    train_samples = sampler(ntrain_samples)
    # set train values to zero to make difference in GP variance clearer
    train_values = np.zeros((ntrain_samples, 1))
    for ii in range(1, ntrain_samples + 1):
        gp.fit(train_samples[:, :ii], train_values[:ii])
        gp.plot_1d(
            axs[ii - 1],
            variable.get_statistics("interval", 1)[0, :],
            npts_1d=101,
        )
        axs[ii - 1].plot(
            train_samples[0, : ii - 1], train_values[: ii - 1, 0], "ko", ms=15
        )
        axs[ii - 1].plot(
            train_samples[0, ii - 1 : ii],
            train_values[ii - 1 : ii, 0],
            "rs",
            ms=15,
        )


ntrain_samples = 5
plot_gp_samples(sampler, ntrain_samples, kernel, variable)

# %%
# As an alternative to integrated variance sampling, active learning Mckay (ALM) greedily chooses samples that minimizes the maximum variance of the Gaussian process. That is, given M training samples the next sample is chosen via
#
# .. math:: \rv^{(n+1)}=\argmax_{\mathcal{Z}\subset\Omega\subset\rvdom} C^\star(\rv, \rv\mid \mathcal{Z}_M)
#
# Although more computationally efficient than ALC, empirical studies suggest that ALM tends to produce GPs with worse predictive performance [GL2009]_.
#
# Accurately evaluating the ALC and ALM criterion is often challenging because inverting the covariance matrix :math:`C(\mathcal{Z}_M\cup \rv)` is poorly conditioned when :math:`\rv` is 'close' to a point in :math:`\mathcal{Z}_M`. Consequently a small constant (nugget) is often added to the diagonal of :math:`C(\mathcal{Z}_M\cup \rv)` to improve numerical stability [PW2014]_.
#
# Experimental design strategies similar to ALM and ALC have been developed for radial basis functions (RBFs). The strong connections between radial basis function and Gaussian process approximation mean that the RBF algorithms can often be used for constructing GPs. A popular RBF design strategy minimizes the worst case error function (power function) of kernel based approximations [SW2006]_. The minimization of the power function is equivalent to minimizing the ALM criteria [HJZ2021]_. As with ALM and ALC, evaluation of the power function is unstable [SW2006]_. However the authors of [PS2011]_ established that stability can be improved by greedily minimizing the power function using pivoted Cholesky factorization [PS2011]_. Specifically, the first :math:`M` pivots of the pivoted Cholesky factorization of a kernel (covariance matrix), evaluated a large set of candidate sample, define the :math:`M` samples which greedily minimize the power function (ALM criteria). Minimizing the power function does not take into account any available distribution information about the inputs :math:`\rv`. In [HJZ2021]_ this information was incorporated by weighting the power function by the density :math:`\pdf(\rv)` of the input variables. This procedure attempts to greedily minimizes the :math:`\pdf`-weighted :math:`L^2` error and produces GPs with predictive performance comparable to those based upon ALC designs while being much more computationally efficient because of its use of pivoted Cholesky factorization.
#
# Finally we remark that while ALM and ALC are the most popular experimental design strategies for GPs, alternative methods have been proposed. Of note are those methods which approximately minimize the mutual information between the Gaussian process evaluated at the training data and the Gaussian process evaluated at the remaining candidate samples [KSG2008]_, [BG2016]_. We do not consider these methods in our numerical comparisons.
#
# The following code shows how to use pivoted Cholesky factorization to greedily choose trainig samples for a GP.
kernel = MaternKernel(np.inf, 0.5, (0.1, 1), fixed=True, nvars=nvars)
gp = ExactGaussianProcess(nvars, kernel)
sampler = CholeskySampler(variable)
sampler.set_gaussian_process(gp)
ncandidate_samples = 101
candidate_samples = np.linspace(-1, 1, ncandidates)[None, :]
sampler.set_candidate_samples(candidate_samples)
init_pivots = np.array([ncandidates // 2], dtype=int)
sampler.set_initial_pivots(init_pivots)
ntrain_samples = 5
plot_gp_samples(sampler, ntrain_samples, kernel, variable)

# %%
# Active Learning
# ---------------
# The samples selected by the aforementioned methods depends on the kernel specified. Change the length_scale of the kernel above to see how the selected samples changes. Active learning chooses a small initial sample set then trains the GP to learn the best kernel hyper-parameters. These parameters are then used to increment the training set and then used to train the GP hyper-parameters again and so until a sufficient accuracy or computational budget is reached. PyApprox's AdaptiveGaussianProcess implements this procedure [HJZ2021]_.

# %%
# References
# ^^^^^^^^^^
# .. [RW2006] `C.E. Rasmussen and C. WIlliams. Gaussian Processes for Machine Learning. MIT Press (2006) <http://www.gaussianprocess.org/gpml/>`_
#
# .. [SWMW1989] `J. Sacks, W.J. Welch, T.J.Mitchell, H.P. Wynn Designs and analysis of computer experiments (with discussion). Statistical Science, 4:409-435 (1989) <http://www.jstor.org/stable/2245858>`_
#
# .. [HJZ2021] `H. Harbrecht, J.D. Jakeman, P. Zaspel. Cholesky-based experimental design for Gaussian process and kernel-based emulation and calibration . Communications in Computational Physics (2021) In press <https://edoc.unibas.ch/79042/>`_
#
# .. [GM2016] `A. Gorodetsky, Y. Marzouk. Mercer kernels and integrated variance experimental design. Connec- tions between Gaussian process regression and polynomial approximation. SIAM/ASA J. Uncertain. Quantif., 4(1):796–828 (2016) <https://doi.org/10.1137/15M1017119>`_
#
# .. [C2006] `D. Cohn Neural network exploration using optimal experiment design, Neural Netw., 9 (1996), pp. 1071–1083. <https://proceedings.neurips.cc/paper/1993/file/d840cc5d906c3e9c84374c8919d2074e-Paper.pdf>`_
#
# .. [KSG2008] `A. Krause, A. Singh, C. Guestrin, Near-optimal sensor placements in Gaussian processes: Theory, efficient algorithms and empirical studies, J. Mach. Learn. Res., 9 (2008), pp. 235–284. <https://doi.org/10.1002/env.769>`_
#
# .. [PW2014] `C.Y. Peng, J. Wu, On the choice of nugget in kriging modeling for deterministic computer experiments, J. Comput. Graph. Statist., 23 (2014), pp. 151–168. <https://doi.org/10.1080/10618600.2012.738961>`_
#
# .. [BG2016] `J. Beck, S. Guillas, Sequential Design with Mutual Information for Computer Experiments (MICE): Emulation of a Tsunami Model, SIAM/ASA J. UNCERTAINTY QUANTIFICATION Vol. 4, pp. 739–766 (2016) <https://doi.org/10.1137/140989613>`_
#
# .. [GL2009] `R.B. Gramacy, H.K.H. Lee, Adaptive design and analysis of supercomputer experiments, Technometrics, 51 (2009), pp. 130–145. <https://doi.org/10.1198/TECH.2009.0015>`_
#
# .. [SW2006] `R. Schaback and H. Wendland. Kernel techniques: From machine learning to meshless methods. Acta Numer., 15:543–639 (2006). <https://doi.org/10.1017/S0962492906270016>`_
#
# .. [PS2011] `M. Pazouki and R. Schaback. Bases for kernel-based spaces. J. Comput. Appl. Math., 236:575–588 (2011). <https://doi.org/10.1016/j.cam.2011.05.021>`_
