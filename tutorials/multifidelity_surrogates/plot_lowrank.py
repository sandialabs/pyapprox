r"""
Bi-fidelity Low Rank Model
==========================
"""

import numpy as np
import matplotlib.pyplot as plt
from pyapprox.surrogates.affine.tests.test_low_rank_multi_fidelilty import (
    OscillatoryPolyModel,
)
from pyapprox.surrogates.affine.low_rank_multifidelity import (
    BiFidelityModel,
    expected_l2_error,
)
from pyapprox.util.backends.numpy import NumpyMixin as bkd

eps = 1.0e-3
mesh_dof = 100
K = 35
hf_model = OscillatoryPolyModel(mesh_dof, 100, eps, backend=bkd)
lf_model = OscillatoryPolyModel(mesh_dof, 100, 1e-2, backend=bkd)


# %% plot spatial dependence for a fixed random sample
fig, ax = plt.subplots(1, 1, figsize=(1 * 8, 6))
samples = bkd.array([[5]])
hf_model.set_eps(1e-2)
ax.plot(hf_model.mesh(), hf_model(samples)[0, :], label="$u_0$")
hf_model.set_eps(1e-3)
ax.plot(hf_model.mesh(), lf_model(samples)[0, :], label="$u_1$")
_ = ax.legend()

# %% plot parameteric dependence of model value at 50th mesh point
fig, ax = plt.subplots(1, 1, figsize=(1 * 8, 6))
samples = bkd.linspace(0.01, np.pi * 10 - 0.1, 101)[None, :]
hf_model.set_eps(1e-2)
ax.plot(samples[0, :], hf_model(samples)[:, 50], label="$u_0$")
hf_model.set_eps(1e-3)
_ = ax.plot(samples[0, :], lf_model(samples)[:, 50], label="$u_1$")


# %% plot error in bifidelity model as the number of hf samples increases
nlf_candidates = int(1e4)
ntest_samples = int(1e3)
test_samples = hf_model.variable().rvs(ntest_samples)
hf_test_values = hf_model(test_samples)
lf_samples = hf_model.variable().rvs(nlf_candidates)
hf_runs = bkd.arange(1, 21, 2)
print(hf_runs)
error_mf = bkd.empty((len(hf_runs)))
fig, ax = plt.subplots(1, 1, figsize=(1 * 8, 6))
colors = ["k", "b", "r"]
for nterms, color in zip([15, 20, 35], colors):
    lf_model = OscillatoryPolyModel(mesh_dof, nterms, 1e-3, backend=bkd)
    lf_test_values = lf_model(test_samples)
    error_lf = expected_l2_error(hf_test_values, lf_test_values)[1]
    for jj in range(len(hf_runs)):
        nhf_runs = hf_runs[jj]

        mf_model = BiFidelityModel(backend=bkd)
        mf_model.build(lf_model, hf_model, lf_samples, nhf_runs)
        mf_test_values = mf_model(test_samples)
        error_mf[jj] = expected_l2_error(hf_test_values, mf_test_values)[1]
        print("|hf-mf|", error_mf[jj])

    ax.semilogy(hf_runs, error_mf, c=color, label=f"MF error, $N={nhf_runs}$")
    ax.axhline(y=error_lf, ls="--", c=color, label=f"LF error, $K={nterms}$")
_ = ax.legend()
ax.set_xlabel("Number of high-fidelity simulations")
ax.set_ylabel("Relative Expected L2 error")
plt.show()

# %%
# References
# ^^^^^^^^^^
# .. [NGXSISC2014] `Narayan, A. and Gittelson, C. and Xiu, D. A Stochastic Collocation Algorithm with Multifidelity Models. SIAM Journal on Scientific Computing 36(2), A495-A521, 2014. <https://doi.org/10.1137/130929461>`_
