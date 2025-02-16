r"""
Parameter Sweeps
----------------
When analyzing complex models or functions it is often useful to gain insight
into its smoothness and non-linearity before undertaking more computationally
intensive analysis such as uncertainty quantification or sensitivity analysis.
Knowledge about smoothness and non-linearity can be used to inform what
algorithms are used for these later tasks.

Lets first generate parameter sweeps for the oakley benchmark. Each sweep will 
be a random direction through the parameter domain. The domain is assumed to 
be the cartesian product of the centered truncated intervals of each
1D variable marginal. For unbounded variables each interval captures 99% of the
PDF. For bounded variables the true bounds are used (i.e. are not truncated)
"""

from pyapprox import analysis
from pyapprox.util.visualization import plt, mathrm_label
from pyapprox.benchmarks import OakleyBenchmark, SobolGBenchmark
from pyapprox.variables.transforms import AffineTransform
import numpy as np

np.random.seed(1)

benchmark = OakleyBenchmark()
mean = benchmark.variable().get_statistics("mean")
stdev = benchmark.variable().get_statistics("std")
cov = np.diag(stdev[:, 0])
gaussian_sweeper = analysis.GaussianParameterSweeper(
    mean,
    lambda x: np.linalg.cholesky(cov) @ x,
    sweep_radius=2,
    nsamples_per_sweep=20,
)
nsweeps = 3
sweep_samples = gaussian_sweeper.rvs(nsweeps)
sweep_values = benchmark.model()(sweep_samples)
ax = plt.figure().gca()
for sweep_id in range(nsweeps):
    gaussian_sweeper.plot_single_qoi_sweep(
        sweep_values[:, 0], sweep_id, ax, marker="o"
    )
ax.set_title(mathrm_label("Oakely model parameter sweeps"))

# %%
# Now lets plot parameter sweeps for the Sobol G function
benchmark = SobolGBenchmark(nvars=4)
bounded_sweeper = analysis.BoundedParameterSweeper(
    benchmark.variable().nvars(),
    AffineTransform(benchmark.variable()),
    nsamples_per_sweep=20,
)
sweep_samples = bounded_sweeper.rvs(nsweeps)
sweep_values = benchmark.model()(sweep_samples)
ax = plt.figure().gca()
for sweep_id in range(nsweeps):
    gaussian_sweeper.plot_single_qoi_sweep(
        sweep_values[:, 0], sweep_id, ax, marker="o"
    )
ax.set_title(mathrm_label("Sobol G model parameter sweeps"))
plt.show()

# %%
# The Sobol G function is not as smooth as the Oakely function. The former
# has discontinuous first derivatives which can be seen by inspecting their
# aprameter sweeps
