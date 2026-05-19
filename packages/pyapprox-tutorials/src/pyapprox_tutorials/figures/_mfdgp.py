"""Plotting functions for the multi-fidelity AR1-vs-DGP tutorial.

Covers: dgp_multifidelity_concept.qmd.

The figures here fit a single-fidelity GP baseline, a linear (AR1)
multi-output GP, and a multi-fidelity DGP on the Perdikaris/Cutajar
nonlinear-A benchmark, and reuse the same fitted models across all
figures so the comparison is consistent.

Caching keeps the fit cost to one run per Quarto render.
"""

from __future__ import annotations

import numpy as np
from pyapprox.surrogates.kernels.base import Kernel

from pyapprox.util.hyperparameter import HyperParameterList

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "_mfdgp.py requires PyTorch (the DGP code path uses TorchBkd). "
        "Install torch to render the multi-fidelity DGP tutorial."
    ) from exc


# ---------------------------------------------------------------------------
# Nonlinear-A benchmark (Perdikaris 2017, used in Cutajar 2019 Figure 4)
# ---------------------------------------------------------------------------
#
#   y_low(x)  = sin(8 pi x)
#   y_high(x) = (x - sqrt(2)) * y_low(x)^2
#
# Low-fidelity points are evenly spaced on [0, 1]; high-fidelity points
# are a nested subset. Both fidelities are noise-free for the demo.

_N_LOW = 50
_N_HIGH = 13     # every 4th low-fidelity point: 50 // 4 + 1 = 13


def _nonlinear_A_low(x):
    return np.sin(8.0 * np.pi * x)


def _nonlinear_A_high(x):
    return (x - np.sqrt(2.0)) * _nonlinear_A_low(x) ** 2


def _nonlinear_A_dataset():
    """Return X_low, y_low, X_high, y_high, x_grid, y_low_truth, y_high_truth.

    Shapes: X arrays are (1, N); y arrays are (1, N); grid arrays are
    (1, n_grid) for X and (n_grid,) for the truth curves.
    """
    x_low_flat = np.linspace(0.0, 1.0, _N_LOW)
    X_low = x_low_flat.reshape(1, -1)
    y_low = _nonlinear_A_low(x_low_flat).reshape(1, -1)

    # Nested: every 4th low-fidelity point becomes a high-fidelity point
    x_high_flat = x_low_flat[::4]
    X_high = x_high_flat.reshape(1, -1)
    y_high = _nonlinear_A_high(x_high_flat).reshape(1, -1)

    x_grid_flat = np.linspace(0.0, 1.0, 200)
    x_grid = x_grid_flat.reshape(1, -1)
    y_low_truth = _nonlinear_A_low(x_grid_flat)
    y_high_truth = _nonlinear_A_high(x_grid_flat)

    return X_low, y_low, X_high, y_high, x_grid, y_low_truth, y_high_truth


# ---------------------------------------------------------------------------
# Cached fits
# ---------------------------------------------------------------------------

_FIT_CACHE = {}


def _se_kernel_factory(nvars, bkd):
    """Kernel factory used by all models.

    Squared exponential (RBF) kernel with one lengthscale per input dim.
    The SE kernel is the correct choice for these infinitely-differentiable
    benchmark functions, and matches Cutajar 2019 which uses RBF throughout.
    """
    from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
    return SquaredExponentialKernel(
        lenscale=[0.1] * nvars,
        lenscale_bounds=(0.01, 5.0),
        nvars=nvars,
        bkd=bkd,
    )


def _fit_baseline():
    """Fit a single-fidelity GP on high-fidelity data only.

    This is the naive approach: ignore all low-fidelity data and
    fit a standard GP on the 13 high-fidelity points. Provides the
    reference against which AR1 and MF-DGP are compared.
    """
    if "baseline" in _FIT_CACHE:
        return _FIT_CACHE["baseline"]

    from pyapprox.surrogates.gaussianprocess.fitters.maximum_likelihood_fitter import (
        GPMaximumLikelihoodFitter,
    )
    from pyapprox.surrogates.gaussianprocess.torch_exact import (
        TorchExactGaussianProcess,
    )
    from pyapprox.util.backends.torch import TorchBkd

    bkd = TorchBkd()

    _, _, X_high, y_high, _, _, _ = _nonlinear_A_dataset()
    X_high_t = bkd.array(X_high)
    y_high_t = bkd.array(y_high)

    kernel = _se_kernel_factory(1, bkd)
    gp = TorchExactGaussianProcess(kernel, nvars=1, nugget=1e-2)

    fitter = GPMaximumLikelihoodFitter(bkd)
    result = fitter.fit(gp, X_high_t, y_high_t)

    _FIT_CACHE["baseline"] = (result.surrogate(), bkd)
    return _FIT_CACHE["baseline"]


def _fit_ar1():
    """Fit a 2-level AR1 model: f_high(x) = rho * f_low(x) + delta(x).

    Uses pyapprox's DAGMultiOutputKernel with a constant (degree-0)
    polynomial scaling, which is mathematically equivalent to the
    Kennedy-O'Hagan AR1 specification.
    """
    if "ar1" in _FIT_CACHE:
        return _FIT_CACHE["ar1"]

    import networkx as nx
    from pyapprox.surrogates.gaussianprocess.fitters.multioutput_fitter import (
        MultiOutputGPMaximumLikelihoodFitter,
    )
    from pyapprox.surrogates.gaussianprocess.torch_multioutput import (
        TorchMultiOutputGP,
    )
    from pyapprox.surrogates.kernels.scalings import PolynomialScaling
    from pyapprox.util.backends.torch import TorchBkd

    from pyapprox.surrogates.kernels.multioutput import (
        DAGMultiOutputKernel,
    )

    bkd = TorchBkd()

    X_low, y_low, X_high, y_high, _, _, _ = _nonlinear_A_dataset()
    X_low_t = bkd.array(X_low)
    y_low_t = bkd.array(y_low)
    X_high_t = bkd.array(X_high)
    y_high_t = bkd.array(y_high)

    # DAG: 0 (low) -> 1 (high)
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1])
    dag.add_edge(0, 1)

    # Discrepancy kernels at each level
    k_low = _se_kernel_factory(1, bkd)
    k_high = _se_kernel_factory(1, bkd)

    # Constant scaling rho_{0->1}(x) = c0 (Kennedy-O'Hagan AR1)
    rho = PolynomialScaling(
        coefficients=[1.0],
        bounds=(-3.0, 3.0),
        bkd=bkd,
        nvars=1,
    )
    edge_scalings = {(0, 1): rho}

    kernel = DAGMultiOutputKernel(
        dag, [k_low, k_high], edge_scalings,
    )

    gp = TorchMultiOutputGP(kernel, nugget=1e-2)

    fitter = MultiOutputGPMaximumLikelihoodFitter(bkd)
    result = fitter.fit(
        gp,
        X_train_list=[X_low_t, X_high_t],
        y_train=[y_low_t, y_high_t],
    )
    _FIT_CACHE["ar1"] = (result.surrogate(), bkd)
    return _FIT_CACHE["ar1"]


def _cutajar_kernel_factory(nvars, bkd):
    """Kernel factory that uses Cutajar composite for augmented layers.

    Root layers (nvars == 1) get a plain SE kernel. Non-root layers
    (nvars == 2, i.e. [x, f]) get the Cutajar composite kernel so
    that NARGP and MF-DGP use the same kernel structure for a fair
    comparison.
    """
    from pyapprox.surrogates.kernels.linear import LinearKernel
    from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel

    nvars_x = nvars - 1
    if nvars_x < 1:
        return _se_kernel_factory(nvars, bkd)

    k_corr = SquaredExponentialKernel(
        lenscale=[0.1] * nvars_x, lenscale_bounds=(0.01, 5.0),
        nvars=nvars_x, bkd=bkd,
    )
    k_prev = SquaredExponentialKernel(
        lenscale=[0.5], lenscale_bounds=(0.01, 5.0), nvars=1, bkd=bkd,
    )
    k_linear = LinearKernel(
        signal_variance=1.0, signal_variance_bounds=(0.01, 10.0),
        nvars=1, bkd=bkd,
    )
    k_delta = SquaredExponentialKernel(
        lenscale=[0.1] * nvars_x, lenscale_bounds=(0.01, 5.0),
        nvars=nvars_x, bkd=bkd,
    )
    return _CutajarMFKernel(nvars_x, k_corr, k_prev, k_linear, k_delta, bkd)


def _fit_nargp():
    """Fit a 2-level exact NARGP (Perdikaris 2017).

    Chain of exact GPs fitted sequentially: layer 0 is a standard GP
    on the low-fidelity data, layer 1 takes augmented input [x, f_0(x)]
    with the Cutajar composite kernel and is fitted on the high-fidelity
    data using the parent's posterior mean as a deterministic input.
    No inducing points or variational inference.
    """
    if "nargp" in _FIT_CACHE:
        return _FIT_CACHE["nargp"]

    import networkx as nx
    from pyapprox.surrogates.gaussianprocess.exact_nargp import (
        ExactNARGPFitter,
    )
    from pyapprox.util.backends.torch import TorchBkd

    bkd = TorchBkd()

    X_low, y_low, X_high, y_high, _, _, _ = _nonlinear_A_dataset()
    X_low_t = bkd.array(X_low)
    y_low_t = bkd.array(y_low)
    X_high_t = bkd.array(X_high)
    y_high_t = bkd.array(y_high)

    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1])
    dag.add_edge(0, 1)

    data = {0: (X_low_t, y_low_t), 1: (X_high_t, y_high_t)}

    fitter = ExactNARGPFitter(
        bkd, _cutajar_kernel_factory, nvars=1, nugget=1e-2,
    )
    result = fitter.fit(dag, data)

    _FIT_CACHE["nargp"] = (result.surrogate(), bkd)
    return _FIT_CACHE["nargp"]


def _predict_nargp_high(model, bkd, x_grid_np):
    """NARGP prediction at the high-fidelity (leaf) node."""
    x_t = bkd.array(x_grid_np)
    mean = bkd.to_numpy(model.predict(x_t)).ravel()
    std = bkd.to_numpy(model.predict_std(x_t)).ravel()
    return mean, std


def _training_inducing_points(bkd):
    """Inducing points placed at training data (Cutajar 2019 pattern).

    Layer 0: every-other low-fidelity input (25 of 50 points).
    Layer 1: all high-fidelity inputs augmented with observed low-fidelity
             values [X_high, y_low(X_high)] (13 points).
    """
    X_low, _, X_high, _, _, _, _ = _nonlinear_A_dataset()
    Z0 = bkd.array(X_low[:, ::2])
    y_low_at_high = _nonlinear_A_low(X_high[0]).reshape(1, -1)
    Z1 = bkd.array(np.concatenate([X_high, y_low_at_high], axis=0))
    return Z0, Z1


def _fit_mfdgp_random_inducing():
    """MF-DGP with Cutajar kernel but random inducing points.

    Same composite kernel structure as _fit_mfdgp_cutajar_kernel() but
    with random inducing points (not fixed at training data). This
    isolates the effect of inducing point placement.
    """
    if "mfdgp_random" in _FIT_CACHE:
        return _FIT_CACHE["mfdgp_random"]

    import networkx as nx
    from pyapprox.optimization.minimize.adam.adam_optimizer import (
        AdamOptimizer,
    )
    from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
        DeepGaussianProcess,
    )
    from pyapprox.surrogates.gaussianprocess.deep.initializers import (
        RandomUniformInitializer,
    )
    from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
    from pyapprox.surrogates.gaussianprocess.deep.propagator import (
        LayerPropagator,
    )
    from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
        TensorProductGHRule,
    )
    from pyapprox.surrogates.gaussianprocess.fitters.composition import (
        DGPFitterChain,
    )
    from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
        DGPMaximumLikelihoodFitter,
        MFDGPSequentialFitter,
    )
    from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
        InducingPoints,
    )
    from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
        GaussianVariationalDistribution,
    )
    from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
        GaussianLikelihood,
    )
    from pyapprox.surrogates.gaussianprocess.mean_functions import (
        ParentPassthroughMean,
        ZeroMean,
    )
    from pyapprox.surrogates.kernels.linear import LinearKernel
    from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
    from pyapprox.util.backends.torch import TorchBkd

    bkd = TorchBkd()

    X_low, y_low, X_high, y_high, _, _, _ = _nonlinear_A_dataset()
    X_low_t = bkd.array(X_low)
    y_low_t = bkd.array(y_low)
    X_high_t = bkd.array(X_high)
    y_high_t = bkd.array(y_high)

    n_ind = 15
    rng = np.random.RandomState(0)

    # Layer 0: SE kernel on x, random inducing
    init0 = RandomUniformInitializer(bounds=(0.0, 1.0))
    Z0_rand = init0.initialize(n_ind, 1, bkd, rng)
    kernel0 = _se_kernel_factory(1, bkd)
    ip0 = InducingPoints(1, n_ind, bkd, Z0_rand, (0.0, 1.0))
    vd0 = GaussianVariationalDistribution(n_ind, bkd)
    lik0 = GaussianLikelihood(1e-3, (1e-5, 1.0), bkd)
    layer0 = DGPLayer(
        kernel0, ZeroMean(bkd), ip0, vd0, bkd,
        likelihood=lik0, nugget=1e-6,
    )

    # Layer 1: Cutajar composite kernel, random inducing
    k_corr = SquaredExponentialKernel(
        lenscale=[0.1], lenscale_bounds=(0.01, 5.0), nvars=1, bkd=bkd,
    )
    k_prev = SquaredExponentialKernel(
        lenscale=[0.5], lenscale_bounds=(0.01, 5.0), nvars=1, bkd=bkd,
    )
    k_linear = LinearKernel(
        signal_variance=1.0, signal_variance_bounds=(0.01, 10.0),
        nvars=1, bkd=bkd,
    )
    k_delta = SquaredExponentialKernel(
        lenscale=[0.1], lenscale_bounds=(0.01, 5.0), nvars=1, bkd=bkd,
    )
    kernel1 = _CutajarMFKernel(1, k_corr, k_prev, k_linear, k_delta, bkd)

    init1 = RandomUniformInitializer(bounds=(-1.5, 1.5))
    Z1_rand = init1.initialize(n_ind, 2, bkd, rng)
    ip1 = InducingPoints(2, n_ind, bkd, Z1_rand, (-1.5, 1.5))
    vd1 = GaussianVariationalDistribution(n_ind, bkd)
    lik1 = GaussianLikelihood(1e-3, (1e-5, 1.0), bkd)
    mean1 = ParentPassthroughMean(parent_start=1, bkd=bkd)
    layer1 = DGPLayer(
        kernel1, mean1, ip1, vd1, bkd,
        likelihood=lik1, nugget=1e-6,
    )

    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1])
    dag.add_edge(0, 1)

    propagator = LayerPropagator(bkd)
    dgp = DeepGaussianProcess(
        dag, {0: layer0, 1: layer1}, propagator, bkd, n_propagation=25,
    )

    seq_opt = AdamOptimizer(lr=1e-2, maxiter=300, verbosity=0)
    joint_opt = AdamOptimizer(lr=1e-3, maxiter=300, verbosity=0)

    chain = DGPFitterChain([
        MFDGPSequentialFitter(bkd, optimizer=seq_opt),
        DGPMaximumLikelihoodFitter(
            bkd, optimizer=joint_opt, n_propagation=25,
        ),
    ])

    data = {0: (X_low_t, y_low_t), 1: (X_high_t, y_high_t)}
    result = chain.fit(dgp, data)
    fitted_dgp = result.surrogate()

    gh_rule = TensorProductGHRule(order=5)
    fitted_dgp.set_propagator(LayerPropagator(bkd, rule=gh_rule))

    _FIT_CACHE["mfdgp_random"] = (fitted_dgp, bkd)
    return _FIT_CACHE["mfdgp_random"]


def _fit_mfdgp():
    """MF-DGP with fixed inducing at training data + var mean init.

    Follows the Cutajar 2019 pattern: inducing points are placed at
    training locations (fixed, not optimized), and the variational mean
    is initialized from a least-squares fit to targets. Uses a single
    Matern 5/2 kernel on the augmented input [x, f] at layer 1.
    """
    if "mfdgp" in _FIT_CACHE:
        return _FIT_CACHE["mfdgp"]

    from pyapprox.optimization.minimize.adam.adam_optimizer import (
        AdamOptimizer,
    )
    from pyapprox.surrogates.gaussianprocess.deep.builders import (
        build_multilevel_dgp,
    )
    from pyapprox.surrogates.gaussianprocess.deep.initializers import (
        CustomInitializer,
    )
    from pyapprox.surrogates.gaussianprocess.deep.propagator import (
        LayerPropagator,
    )
    from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
        TensorProductGHRule,
    )
    from pyapprox.surrogates.gaussianprocess.fitters.composition import (
        DGPFitterChain,
    )
    from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
        DGPMaximumLikelihoodFitter,
        MFDGPSequentialFitter,
    )
    from pyapprox.util.backends.torch import TorchBkd

    bkd = TorchBkd()

    X_low, y_low, X_high, y_high, _, _, _ = _nonlinear_A_dataset()
    X_low_t = bkd.array(X_low)
    y_low_t = bkd.array(y_low)
    X_high_t = bkd.array(X_high)
    y_high_t = bkd.array(y_high)

    Z0, Z1 = _training_inducing_points(bkd)

    dgp = build_multilevel_dgp(
        level_nvars=[1, 1],
        num_inducing=[Z0.shape[1], Z1.shape[1]],
        kernel_factory=_se_kernel_factory,
        bkd=bkd,
        noise_std=1e-3,
        noise_bounds=(1e-5, 1.0),
        inducing_bounds=[(0.0, 1.0), (-1.5, 1.5)],
        nugget=1e-6,
        n_propagation=25,
        seed=0,
        initializer=[CustomInitializer(Z0), CustomInitializer(Z1)],
    )

    for layer in dgp.layers().values():
        layer.inducing_points().hyp_list().set_all_inactive()

    seq_opt = AdamOptimizer(lr=1e-2, maxiter=300, verbosity=0)
    joint_opt = AdamOptimizer(lr=1e-3, maxiter=300, verbosity=0)

    chain = DGPFitterChain([
        MFDGPSequentialFitter(
            bkd, optimizer=seq_opt, init_variational_mean=True,
        ),
        DGPMaximumLikelihoodFitter(
            bkd, optimizer=joint_opt, n_propagation=25,
        ),
    ])

    data = {0: (X_low_t, y_low_t), 1: (X_high_t, y_high_t)}
    result = chain.fit(dgp, data)
    fitted_dgp = result.surrogate()

    gh_rule = TensorProductGHRule(order=5)
    fitted_dgp.set_propagator(LayerPropagator(bkd, rule=gh_rule))

    _FIT_CACHE["mfdgp"] = (fitted_dgp, bkd)
    return _FIT_CACHE["mfdgp"]


# ---------------------------------------------------------------------------
# Cutajar composite kernel
# ---------------------------------------------------------------------------


class _CutajarMFKernel(Kernel):
    """Cutajar 2019 Eq. 11 composite kernel for non-root DGP layers.

    k([x,f], [x',f']) = k_corr(x,x') * [Linear(f,f') + k_prev(f,f')]
                       + k_delta(x,x')

    Operates on augmented input [x, f] where x has nvars_x dims
    and f has 1 dim (scalar parent output).
    """

    def __init__(self, nvars_x, k_corr, k_prev, k_linear, k_delta, bkd):
        super().__init__(bkd)
        self._nvars_x = nvars_x
        self._k_corr = k_corr
        self._k_prev = k_prev
        self._k_linear = k_linear
        self._k_delta = k_delta
        hyps = (
            k_corr.hyp_list().hyperparameters()
            + k_prev.hyp_list().hyperparameters()
            + k_linear.hyp_list().hyperparameters()
            + k_delta.hyp_list().hyperparameters()
        )
        self._hyp_list = HyperParameterList(hyps)

    def nvars(self):
        return self._nvars_x + 1

    def hyp_list(self):
        return self._hyp_list

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        d = self._nvars_x
        x1, f1 = X1[:d, :], X1[d:, :]
        x2, f2 = X2[:d, :], X2[d:, :]
        return (
            self._k_corr(x1, x2) * (self._k_linear(f1, f2)
                                     + self._k_prev(f1, f2))
            + self._k_delta(x1, x2)
        )

    def diag(self, X):
        d = self._nvars_x
        x, f = X[:d, :], X[d:, :]
        return (
            self._k_corr.diag(x) * (self._k_linear.diag(f)
                                     + self._k_prev.diag(f))
            + self._k_delta.diag(x)
        )


def _fit_mfdgp_cutajar_kernel():
    """MF-DGP with Cutajar composite kernel + fixed inducing + var init.

    Layer 1 uses the kernel structure from Cutajar 2019 Eq. 11:
      k_corr(x) * [Linear(f) + k_prev(f)] + k_delta(x)
    which explicitly decomposes the inter-fidelity relationship into
    correlation, linear transfer, nonlinear transfer, and discrepancy.
    """
    if "mfdgp_cutajar" in _FIT_CACHE:
        return _FIT_CACHE["mfdgp_cutajar"]

    import networkx as nx
    from pyapprox.optimization.minimize.adam.adam_optimizer import (
        AdamOptimizer,
    )
    from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
        DeepGaussianProcess,
    )
    from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
    from pyapprox.surrogates.gaussianprocess.deep.propagator import (
        LayerPropagator,
    )
    from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
        TensorProductGHRule,
    )
    from pyapprox.surrogates.gaussianprocess.fitters.composition import (
        DGPFitterChain,
    )
    from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
        DGPMaximumLikelihoodFitter,
        MFDGPSequentialFitter,
    )
    from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
        InducingPoints,
    )
    from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
        GaussianVariationalDistribution,
    )
    from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
        GaussianLikelihood,
    )
    from pyapprox.surrogates.gaussianprocess.mean_functions import (
        ParentPassthroughMean,
        ZeroMean,
    )
    from pyapprox.surrogates.kernels.linear import LinearKernel
    from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
    from pyapprox.util.backends.torch import TorchBkd

    bkd = TorchBkd()

    X_low, y_low, X_high, y_high, _, _, _ = _nonlinear_A_dataset()
    X_low_t = bkd.array(X_low)
    y_low_t = bkd.array(y_low)
    X_high_t = bkd.array(X_high)
    y_high_t = bkd.array(y_high)

    Z0, Z1 = _training_inducing_points(bkd)

    # Layer 0: SE kernel on x
    kernel0 = _se_kernel_factory(1, bkd)
    ip0 = InducingPoints(1, Z0.shape[1], bkd, Z0, (0.0, 1.0))
    vd0 = GaussianVariationalDistribution(Z0.shape[1], bkd)
    lik0 = GaussianLikelihood(1e-3, (1e-5, 1.0), bkd)
    layer0 = DGPLayer(
        kernel0, ZeroMean(bkd), ip0, vd0, bkd,
        likelihood=lik0, nugget=1e-6,
    )

    # Layer 1: Cutajar composite kernel on [x, f]
    k_corr = SquaredExponentialKernel(
        lenscale=[0.1], lenscale_bounds=(0.01, 5.0), nvars=1, bkd=bkd,
    )
    k_prev = SquaredExponentialKernel(
        lenscale=[0.5], lenscale_bounds=(0.01, 5.0), nvars=1, bkd=bkd,
    )
    k_linear = LinearKernel(
        signal_variance=1.0, signal_variance_bounds=(0.01, 10.0),
        nvars=1, bkd=bkd,
    )
    k_delta = SquaredExponentialKernel(
        lenscale=[0.1], lenscale_bounds=(0.01, 5.0), nvars=1, bkd=bkd,
    )
    kernel1 = _CutajarMFKernel(1, k_corr, k_prev, k_linear, k_delta, bkd)

    ip1 = InducingPoints(2, Z1.shape[1], bkd, Z1, (-1.5, 1.5))
    vd1 = GaussianVariationalDistribution(Z1.shape[1], bkd)
    lik1 = GaussianLikelihood(1e-3, (1e-5, 1.0), bkd)
    mean1 = ParentPassthroughMean(parent_start=1, bkd=bkd)
    layer1 = DGPLayer(
        kernel1, mean1, ip1, vd1, bkd,
        likelihood=lik1, nugget=1e-6,
    )

    # Fix inducing points
    layer0.inducing_points().hyp_list().set_all_inactive()
    layer1.inducing_points().hyp_list().set_all_inactive()

    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1])
    dag.add_edge(0, 1)

    propagator = LayerPropagator(bkd)
    dgp = DeepGaussianProcess(
        dag, {0: layer0, 1: layer1}, propagator, bkd, n_propagation=25,
    )

    seq_opt = AdamOptimizer(lr=1e-2, maxiter=300, verbosity=0)
    joint_opt = AdamOptimizer(lr=1e-3, maxiter=300, verbosity=0)

    chain = DGPFitterChain([
        MFDGPSequentialFitter(
            bkd, optimizer=seq_opt, init_variational_mean=True,
        ),
        DGPMaximumLikelihoodFitter(
            bkd, optimizer=joint_opt, n_propagation=25,
        ),
    ])

    data = {0: (X_low_t, y_low_t), 1: (X_high_t, y_high_t)}
    result = chain.fit(dgp, data)
    fitted_dgp = result.surrogate()

    gh_rule = TensorProductGHRule(order=5)
    fitted_dgp.set_propagator(LayerPropagator(bkd, rule=gh_rule))

    _FIT_CACHE["mfdgp_cutajar"] = (fitted_dgp, bkd)
    return _FIT_CACHE["mfdgp_cutajar"]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def _predict_baseline_high(gp, bkd, x_grid_np):
    """Single-fidelity GP prediction at x_grid_np."""
    x_grid_t = bkd.array(x_grid_np)
    mean = bkd.to_numpy(gp.predict(x_grid_t)).ravel()
    std = bkd.to_numpy(gp.predict_std(x_grid_t)).ravel()
    return mean, std


def _predict_ar1_high(ar1_gp, bkd, x_grid_np):
    """AR1 prediction on the high-fidelity output at x_grid_np.

    MultiOutputGP.predict_with_uncertainty returns lists of arrays,
    one entry per output. We want the high-fidelity entry (index 1).
    """
    x_grid_t = bkd.array(x_grid_np)
    # MultiOutputGP wants a list of test inputs, one per output
    mean_list, std_list = ar1_gp.predict_with_uncertainty(
        [x_grid_t, x_grid_t],
    )
    mean_high = bkd.to_numpy(mean_list[1]).ravel()
    std_high = bkd.to_numpy(std_list[1]).ravel()
    mean_low = bkd.to_numpy(mean_list[0]).ravel()
    std_low = bkd.to_numpy(std_list[0]).ravel()
    return mean_low, std_low, mean_high, std_high


def _predict_mfdgp_high(mfdgp, bkd, x_grid_np, n_propagation=25):
    """MF-DGP prediction at the high-fidelity (top) layer."""
    x_grid_t = bkd.array(x_grid_np)
    mean = mfdgp.predict(x_grid_t, n_propagation=n_propagation)
    std = mfdgp.predict_std(x_grid_t, n_propagation=n_propagation)
    return bkd.to_numpy(mean).ravel(), bkd.to_numpy(std).ravel()


def _compute_test_metrics(mean, std, truth):
    """Test-set RMSE and mean Gaussian log-likelihood."""
    residual = truth - mean
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    var = std ** 2 + 1e-12
    mean_loglik = float(np.mean(
        -0.5 * np.log(2.0 * np.pi * var) - 0.5 * residual ** 2 / var
    ))
    return rmse, mean_loglik


# ---------------------------------------------------------------------------
# Figure 1: the benchmark dataset and the nonlinear correlation
# ---------------------------------------------------------------------------


def plot_nonlinear_A_dataset(axes):
    """dgp_multifidelity_concept.qmd -> fig-nonlinear-a-dataset

    Two panels: (left) low and high-fidelity functions over the input
    domain with training points marked; (right) high-fidelity output
    vs low-fidelity output, showing the parabolic (nonlinear)
    correlation that breaks the AR1 assumption.
    """
    from ._style import apply_style

    (X_low, y_low, X_high, y_high,
     x_grid, y_low_truth, y_high_truth) = _nonlinear_A_dataset()

    # ---- Left panel: functions over the input domain ----
    ax = axes[0]
    ax.plot(
        x_grid[0], y_low_truth, color="#2C7FB8", lw=1.6,
        label=r"$y_{\mathrm{low}}(x) = \sin(8\pi x)$",
    )
    ax.plot(
        x_grid[0], y_high_truth, color="#C0392B", lw=1.6,
        label=r"$y_{\mathrm{high}}(x) = (x-\sqrt{2}) y_{\mathrm{low}}(x)^2$",
    )
    ax.scatter(
        X_low[0], y_low[0], s=16, color="#2C7FB8",
        edgecolor="k", linewidth=0.4, zorder=5,
        label=f"Low-fidelity ($N_L={X_low.shape[1]}$)",
    )
    ax.scatter(
        X_high[0], y_high[0], s=42, color="#C0392B",
        edgecolor="k", linewidth=0.6, zorder=6,
        label=f"High-fidelity ($N_H={X_high.shape[1]}$)",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel(r"$y$", fontsize=11)
    ax.set_title("Two fidelities on the unit interval", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    apply_style(ax)

    # ---- Right panel: y_high vs y_low (parametric trace by x) ----
    ax = axes[1]
    ax.plot(
        y_low_truth, y_high_truth, color="#7D3C98", lw=2.0,
        label=r"$y_{\mathrm{high}}$ vs $y_{\mathrm{low}}$",
    )
    ax.axhline(0, color="k", lw=0.6, alpha=0.3)
    ax.axvline(0, color="k", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"$y_{\mathrm{low}}$", fontsize=11)
    ax.set_ylabel(r"$y_{\mathrm{high}}$", fontsize=11)
    ax.set_title(
        "Correlation between fidelities is nonlinear",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower center")
    apply_style(ax)


# ---------------------------------------------------------------------------
# Figure 2: three-panel comparison (baseline, AR1, MF-DGP)
# ---------------------------------------------------------------------------


def _plot_model_panel(ax, x_grid, mean, std, y_high_truth,
                      X_high, y_high, color, label, title,
                      X_low=None, y_low=None):
    """Shared plotting logic for one model's fit panel."""
    from ._style import apply_style

    rmse, mean_ll = _compute_test_metrics(mean, std, y_high_truth)

    ax.plot(
        x_grid[0], y_high_truth, color="k", lw=1.2, ls="--", alpha=0.7,
        label="True high-fidelity",
    )
    ax.fill_between(
        x_grid[0],
        mean - 1.96 * std, mean + 1.96 * std,
        color=color, alpha=0.18, label=f"{label} 95% band",
    )
    ax.plot(
        x_grid[0], mean, color=color, lw=1.8, label=f"{label} mean",
    )
    if X_low is not None and y_low is not None:
        ax.scatter(
            X_low[0], y_low[0], s=12, color="#2C7FB8",
            edgecolor="k", linewidth=0.3, zorder=5, alpha=0.7,
            label="Low-fidelity data",
        )
    ax.scatter(
        X_high[0], y_high[0], s=40, color="k", zorder=6,
        label="High-fidelity data",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel(r"$f(x)$", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.text(
        0.04, 0.04,
        f"test RMSE = {rmse:.3f}\nmean log-lik = {mean_ll:.1f}",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )
    ax.legend(fontsize=7, loc="upper right")
    apply_style(ax)


def plot_baseline_fit(ax):
    """Single-fidelity GP baseline on high-fidelity data only."""
    (_, _, X_high, y_high,
     x_grid, _, y_high_truth) = _nonlinear_A_dataset()

    gp, bkd = _fit_baseline()
    mean, std = _predict_baseline_high(gp, bkd, x_grid)

    _plot_model_panel(
        ax, x_grid, mean, std, y_high_truth, X_high, y_high,
        color="#7F8C8D", label="Single-fidelity GP",
        title="Baseline: GP on 13 high-fidelity points only",
    )


def plot_ar1_fit(ax):
    """AR1 multi-fidelity GP prediction on the high-fidelity output."""
    (X_low, y_low, X_high, y_high,
     x_grid, _, y_high_truth) = _nonlinear_A_dataset()

    ar1_gp, bkd = _fit_ar1()
    _, _, mean_high, std_high = _predict_ar1_high(ar1_gp, bkd, x_grid)

    _plot_model_panel(
        ax, x_grid, mean_high, std_high, y_high_truth, X_high, y_high,
        color="#C0392B", label="AR1",
        title="AR1: constant scaling cannot follow the parabola",
        X_low=X_low, y_low=y_low,
    )


def plot_nargp_fit(ax):
    """Exact NARGP prediction on the high-fidelity output."""
    (X_low, y_low, X_high, y_high,
     x_grid, _, y_high_truth) = _nonlinear_A_dataset()

    model, bkd = _fit_nargp()
    mean, std = _predict_nargp_high(model, bkd, x_grid)

    _plot_model_panel(
        ax, x_grid, mean, std, y_high_truth, X_high, y_high,
        color="#D4AC0D", label="NARGP",
        title="NARGP: exact GP chain captures nonlinearity",
        X_low=X_low, y_low=y_low,
    )


def plot_mfdgp_fit(ax):
    """Multi-fidelity DGP prediction on the high-fidelity output."""
    (X_low, y_low, X_high, y_high,
     x_grid, _, y_high_truth) = _nonlinear_A_dataset()

    mfdgp, bkd = _fit_mfdgp_cutajar_kernel()
    mean_high, std_high = _predict_mfdgp_high(mfdgp, bkd, x_grid)

    _plot_model_panel(
        ax, x_grid, mean_high, std_high, y_high_truth, X_high, y_high,
        color="#117A65", label="MF-DGP",
        title="MF-DGP: GP composition captures the parabola",
        X_low=X_low, y_low=y_low,
    )


# ---------------------------------------------------------------------------
# Figure 3: learned correlation between fidelities
# ---------------------------------------------------------------------------


def plot_learned_correlation(axes):
    """dgp_multifidelity_concept.qmd -> fig-learned-correlation

    Two panels: (left) AR1 learned correlation vs true,
    (right) MF-DGP learned correlation vs true.
    """
    from ._style import apply_style

    (X_low, y_low, X_high, y_high,
     x_grid, y_low_truth, y_high_truth) = _nonlinear_A_dataset()

    # AR1's learned correlation
    ar1_gp, bkd_ar1 = _fit_ar1()
    mean_low_ar1, _, mean_high_ar1, _ = _predict_ar1_high(
        ar1_gp, bkd_ar1, x_grid,
    )

    # MF-DGP's learned correlation
    mfdgp, bkd_dgp = _fit_mfdgp_cutajar_kernel()
    layer_0 = mfdgp.layers()[0]
    x_grid_t = bkd_dgp.array(x_grid)
    mean_low_dgp_t, _ = layer_0.predict_marginal(x_grid_t)
    mean_low_dgp = bkd_dgp.to_numpy(mean_low_dgp_t).ravel()
    mean_high_dgp, _ = _predict_mfdgp_high(mfdgp, bkd_dgp, x_grid)

    # Left panel: AR1 vs true
    ax = axes[0]
    ax.plot(
        y_low_truth, y_high_truth,
        color="#7D3C98", lw=2.0, label="True correlation",
    )
    ax.plot(
        mean_low_ar1, mean_high_ar1,
        color="#C0392B", lw=2.0, ls="--", label="AR1 learned",
    )
    ax.axhline(0, color="k", lw=0.6, alpha=0.3)
    ax.axvline(0, color="k", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"$y_{\mathrm{low}}$", fontsize=11)
    ax.set_ylabel(r"$y_{\mathrm{high}}$", fontsize=11)
    ax.set_title("AR1: constant $\\rho$ gives a straight line", fontsize=10)
    ax.legend(fontsize=9, loc="lower center")
    apply_style(ax)

    # Right panel: MF-DGP vs true
    ax = axes[1]
    ax.plot(
        y_low_truth, y_high_truth,
        color="#7D3C98", lw=2.0, label="True correlation",
    )
    ax.plot(
        mean_low_dgp, mean_high_dgp,
        color="#117A65", lw=2.0, ls="--", label="MF-DGP learned",
    )
    ax.axhline(0, color="k", lw=0.6, alpha=0.3)
    ax.axvline(0, color="k", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"$y_{\mathrm{low}}$", fontsize=11)
    ax.set_ylabel(r"$y_{\mathrm{high}}$", fontsize=11)
    ax.set_title("MF-DGP: GP composition follows the parabola", fontsize=10)
    ax.legend(fontsize=9, loc="lower center")
    apply_style(ax)


# ---------------------------------------------------------------------------
# Figure 4: random vs fixed inducing points
# ---------------------------------------------------------------------------


def plot_mfdgp_initialization(axes):
    """Side-by-side: random inducing vs fixed inducing at training data."""
    (X_low, y_low, X_high, y_high,
     x_grid, _, y_high_truth) = _nonlinear_A_dataset()

    mfdgp_rand, bkd_rand = _fit_mfdgp_random_inducing()
    mean_rand, std_rand = _predict_mfdgp_high(mfdgp_rand, bkd_rand, x_grid)
    _plot_model_panel(
        axes[0], x_grid, mean_rand, std_rand, y_high_truth, X_high, y_high,
        color="#7F8C8D", label="MF-DGP",
        title="Random inducing points",
        X_low=X_low, y_low=y_low,
    )

    mfdgp, bkd = _fit_mfdgp_cutajar_kernel()
    mean_fix, std_fix = _predict_mfdgp_high(mfdgp, bkd, x_grid)
    _plot_model_panel(
        axes[1], x_grid, mean_fix, std_fix, y_high_truth, X_high, y_high,
        color="#117A65", label="MF-DGP",
        title="Fixed inducing at training data + var mean init",
        X_low=X_low, y_low=y_low,
    )


# ---------------------------------------------------------------------------
# Figure 5: single SE kernel vs Cutajar composite kernel
# ---------------------------------------------------------------------------


def plot_mfdgp_kernel_comparison(axes):
    """Side-by-side: single SE kernel vs Cutajar composite kernel."""
    (X_low, y_low, X_high, y_high,
     x_grid, _, y_high_truth) = _nonlinear_A_dataset()

    mfdgp, bkd = _fit_mfdgp()
    mean_se, std_se = _predict_mfdgp_high(mfdgp, bkd, x_grid)
    _plot_model_panel(
        axes[0], x_grid, mean_se, std_se, y_high_truth, X_high, y_high,
        color="#117A65", label="MF-DGP",
        title=r"Single SE kernel on $[\mathbf{x}, f]$",
        X_low=X_low, y_low=y_low,
    )

    mfdgp_c, bkd_c = _fit_mfdgp_cutajar_kernel()
    mean_c, std_c = _predict_mfdgp_high(mfdgp_c, bkd_c, x_grid)
    _plot_model_panel(
        axes[1], x_grid, mean_c, std_c, y_high_truth, X_high, y_high,
        color="#2471A3", label="MF-DGP",
        title=r"Cutajar composite kernel",
        X_low=X_low, y_low=y_low,
    )
