"""Quadrature convergence tests for OED utility estimators.

Verifies that numerical OED utilities match analytical values to high
accuracy when evaluated with tensor-product quadrature over the joint
(θ, ε) space.

Most utilities use Gauss-Hermite quadrature (rtol=1e-6). The U3
lognormal AVaR utility uses piecewise cubic quadrature on a bounded
domain with PDF reweighting (rtol=2e-4), because AVaR is a
quantile-based functional whose convergence rate is limited by point
density rather than polynomial exactness.

Tested utilities and their analytical references:
- KL (EIG): ConjugateGaussianOEDExpectedInformationGain
- U1 linear: sqrt(B·Σ_post·B^T) (closed-form for Gaussian QoI)
- U1 lognormal: ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev
- U3 linear: AVaR of constant = constant = U1 linear
- U3 lognormal: ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev
- U4 lognormal: ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev
"""

from itertools import product as iterproduct

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDExpectedInformationGain,
    ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev,
)
from pyapprox.expdesign.objective import create_prediction_oed_objective
from pyapprox.expdesign.objective.factory import (
    create_kl_oed_objective_from_data,
)
from pyapprox.risk.exact_avar import ExactAVaR
from pyapprox.surrogates.affine.univariate.piecewisepoly.cubic import (
    PiecewiseCubic,
)
from pyapprox.surrogates.affine.univariate.piecewisepoly.dynamic import (
    DynamicPiecewiseBasis,
    EquidistantNodeGenerator,
)
from pyapprox.util.backends.protocols import Array, Backend


def _gauss_hermite_oed_data(
    bkd: Backend[Array],
    nparams: int,
    nobs: int,
    prior_std: float,
    noise_std: float,
    ngauss: int,
):
    """Build tensor-product Gauss-Hermite quadrature data for OED."""
    noise_var = noise_std**2
    ndim = nparams + nobs

    nodes_1d, weights_1d = hermegauss(ngauss)
    weights_1d /= np.sqrt(2 * np.pi)

    tp = list(iterproduct(range(ngauss), repeat=ndim))
    joint_nodes = np.array(
        [[nodes_1d[idx[d]] for d in range(ndim)] for idx in tp]
    ).T
    joint_weights = np.array(
        [np.prod([weights_1d[idx[d]] for d in range(ndim)]) for idx in tp]
    )

    param_std = joint_nodes[:nparams, :]
    latent_std = joint_nodes[nparams:, :]

    theta = prior_std * param_std

    obs_locs = np.linspace(-1, 1, nobs) if nobs > 1 else np.array([0.0])
    A = np.column_stack([obs_locs**p for p in range(nparams)])
    B = np.ones((1, nparams)) / nparams

    outer_shapes = bkd.asarray(A @ theta)
    inner_shapes = bkd.asarray(A @ theta)
    latent = bkd.asarray(latent_std)
    qoi_linear = bkd.asarray((B @ theta).T)
    qoi_lognormal = bkd.asarray(np.exp(B @ theta).T)
    quad_weights = bkd.asarray(joint_weights)

    noise_variances = bkd.full((nobs,), noise_var)
    w_uniform = np.full(nobs, 1.0 / nobs)
    w = bkd.full((nobs, 1), 1.0 / nobs)

    prior_cov = bkd.asarray(np.eye(nparams) * prior_std**2)
    prior_mean = bkd.zeros((nparams, 1))
    noise_cov = bkd.diag(bkd.asarray(noise_var / w_uniform))
    A_bkd = bkd.asarray(A)
    B_bkd = bkd.asarray(B)

    # Analytical EIG
    eig_obj = ConjugateGaussianOEDExpectedInformationGain(prior_cov, bkd)
    eig_obj.set_observation_matrix(A_bkd)
    eig_obj.set_noise_covariance(noise_cov)
    ref_eig = eig_obj.value()

    # Analytical U1 linear: sqrt(B·Σ_post·B^T)
    prior_prec = np.linalg.inv(np.eye(nparams) * prior_std**2)
    noise_prec = np.diag(w_uniform / noise_var)
    post_cov = np.linalg.inv(prior_prec + A.T @ noise_prec @ A)
    ref_u1_linear = float(np.sqrt((B @ post_cov @ B.T).item()))

    # Analytical U1 lognormal
    u1_ln = ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev(
        prior_mean, prior_cov, B_bkd, bkd,
    )
    u1_ln.set_observation_matrix(A_bkd)
    u1_ln.set_noise_covariance(noise_cov)
    ref_u1_lognormal = u1_ln.value()

    # Analytical U3 lognormal: AVaR_beta[Std(W|y)]
    avar_alpha = 0.9
    u3_ln = ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev(
        prior_mean, prior_cov, B_bkd, avar_alpha, bkd,
    )
    u3_ln.set_observation_matrix(A_bkd)
    u3_ln.set_noise_covariance(noise_cov)
    ref_u3_lognormal = u3_ln.value()

    # Analytical U4 lognormal: E[Std] + c*Std_y[Std]
    safety_c = 1.0
    u4_ln = ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev(
        prior_mean, prior_cov, B_bkd, safety_c, bkd,
    )
    u4_ln.set_observation_matrix(A_bkd)
    u4_ln.set_noise_covariance(noise_cov)
    ref_u4_lognormal = u4_ln.value()

    return {
        "outer_shapes": outer_shapes,
        "inner_shapes": inner_shapes,
        "latent": latent,
        "qoi_linear": qoi_linear,
        "qoi_lognormal": qoi_lognormal,
        "quad_weights": quad_weights,
        "noise_variances": noise_variances,
        "w": w,
        "ref_eig": ref_eig,
        "ref_u1_linear": ref_u1_linear,
        "ref_u1_lognormal": ref_u1_lognormal,
        "ref_u3_lognormal": ref_u3_lognormal,
        "ref_u4_lognormal": ref_u4_lognormal,
        "avar_alpha": avar_alpha,
        "safety_c": safety_c,
    }


_CONFIGS = [(1, 1), (2, 1), (1, 2)]
_NGAUSS = 20
_PRIOR_STD = 0.5
_NOISE_STD = 0.5


@pytest.mark.parametrize("nparams,nobs", _CONFIGS)
class TestGaussHermiteOEDConvergence:
    def test_kl_matches_analytical_eig(
        self, numpy_bkd: Backend[Array], nparams: int, nobs: int,
    ) -> None:
        d = _gauss_hermite_oed_data(
            numpy_bkd, nparams, nobs, _PRIOR_STD, _NOISE_STD, _NGAUSS,
        )
        obj = create_kl_oed_objective_from_data(
            d["noise_variances"], d["outer_shapes"], d["inner_shapes"],
            d["latent"], numpy_bkd,
            outer_quad_weights=d["quad_weights"],
            inner_quad_weights=d["quad_weights"],
        )
        val = -float(numpy_bkd.to_numpy(obj(d["w"])).flat[0])
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([val]),
            numpy_bkd.asarray([d["ref_eig"]]),
            rtol=1e-6,
        )

    def test_u1_linear_matches_analytical(
        self, numpy_bkd: Backend[Array], nparams: int, nobs: int,
    ) -> None:
        d = _gauss_hermite_oed_data(
            numpy_bkd, nparams, nobs, _PRIOR_STD, _NOISE_STD, _NGAUSS,
        )
        obj = create_prediction_oed_objective(
            d["noise_variances"], d["outer_shapes"], d["inner_shapes"],
            d["latent"], d["qoi_linear"], numpy_bkd,
            deviation_type="stdev",
            risk_type="mean",
            noise_stat_type="mean",
            outer_quad_weights=d["quad_weights"],
            inner_quad_weights=d["quad_weights"],
        )
        val = float(numpy_bkd.to_numpy(obj(d["w"])).flat[0])
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([val]),
            numpy_bkd.asarray([d["ref_u1_linear"]]),
            rtol=1e-6,
        )

    def test_u1_lognormal_matches_analytical(
        self, numpy_bkd: Backend[Array], nparams: int, nobs: int,
    ) -> None:
        d = _gauss_hermite_oed_data(
            numpy_bkd, nparams, nobs, _PRIOR_STD, _NOISE_STD, _NGAUSS,
        )
        obj = create_prediction_oed_objective(
            d["noise_variances"], d["outer_shapes"], d["inner_shapes"],
            d["latent"], d["qoi_lognormal"], numpy_bkd,
            deviation_type="stdev",
            risk_type="mean",
            noise_stat_type="mean",
            outer_quad_weights=d["quad_weights"],
            inner_quad_weights=d["quad_weights"],
        )
        val = float(numpy_bkd.to_numpy(obj(d["w"])).flat[0])
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([val]),
            numpy_bkd.asarray([d["ref_u1_lognormal"]]),
            rtol=1e-6,
        )

    def test_u3_linear_matches_analytical(
        self, numpy_bkd: Backend[Array], nparams: int, nobs: int,
    ) -> None:
        # For linear Gaussian QoI, Std(B·θ|y) is data-independent so
        # AVaR of a constant = the constant = U1 linear.
        d = _gauss_hermite_oed_data(
            numpy_bkd, nparams, nobs, _PRIOR_STD, _NOISE_STD, _NGAUSS,
        )
        obj = create_prediction_oed_objective(
            d["noise_variances"], d["outer_shapes"], d["inner_shapes"],
            d["latent"], d["qoi_linear"], numpy_bkd,
            deviation_type="stdev",
            risk_type="mean",
            noise_stat_type="avar",
            noise_stat_kwargs={
                "alpha": d["avar_alpha"],
                "delta": 1e8,
            },
            outer_quad_weights=d["quad_weights"],
            inner_quad_weights=d["quad_weights"],
        )
        val = float(numpy_bkd.to_numpy(obj(d["w"])).flat[0])
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([val]),
            numpy_bkd.asarray([d["ref_u1_linear"]]),
            rtol=1e-6,
        )

    def test_u3_lognormal_matches_analytical(
        self, numpy_bkd: Backend[Array], nparams: int, nobs: int,
    ) -> None:
        # Smoothed AVaR on lognormal QoI tails: the lognormal Std(W|y)
        # has high dynamic range across data realizations, so the discrete
        # smoothed AVaR converges slowly. Use rtol=1e-2.
        d = _gauss_hermite_oed_data(
            numpy_bkd, nparams, nobs, _PRIOR_STD, _NOISE_STD, _NGAUSS,
        )
        obj = create_prediction_oed_objective(
            d["noise_variances"], d["outer_shapes"], d["inner_shapes"],
            d["latent"], d["qoi_lognormal"], numpy_bkd,
            deviation_type="stdev",
            risk_type="mean",
            noise_stat_type="avar",
            noise_stat_kwargs={
                "alpha": d["avar_alpha"],
                "delta": 1e8,
            },
            outer_quad_weights=d["quad_weights"],
            inner_quad_weights=d["quad_weights"],
        )
        val = float(numpy_bkd.to_numpy(obj(d["w"])).flat[0])
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([val]),
            numpy_bkd.asarray([d["ref_u3_lognormal"]]),
            rtol=1e-2,
        )

    def test_u4_lognormal_matches_analytical(
        self, numpy_bkd: Backend[Array], nparams: int, nobs: int,
    ) -> None:
        d = _gauss_hermite_oed_data(
            numpy_bkd, nparams, nobs, _PRIOR_STD, _NOISE_STD, _NGAUSS,
        )
        obj = create_prediction_oed_objective(
            d["noise_variances"], d["outer_shapes"], d["inner_shapes"],
            d["latent"], d["qoi_lognormal"], numpy_bkd,
            deviation_type="stdev",
            risk_type="mean",
            noise_stat_type="mean_stdev",
            noise_stat_kwargs={"safety_factor": d["safety_c"]},
            outer_quad_weights=d["quad_weights"],
            inner_quad_weights=d["quad_weights"],
        )
        val = float(numpy_bkd.to_numpy(obj(d["w"])).flat[0])
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([val]),
            numpy_bkd.asarray([d["ref_u4_lognormal"]]),
            rtol=1e-6,
        )


def _piecewise_cubic_oed_data(
    bkd: Backend[Array],
    nparams: int,
    nobs: int,
    prior_std: float,
    noise_std: float,
    npts_1d: int,
    domain_half_width: float,
):
    """Build tensor-product piecewise cubic quadrature data for OED.

    Uses equidistant nodes on [-L, L] with N(0,1) PDF reweighting.
    Better than Gauss-Hermite for quantile-based functionals (AVaR)
    because equidistant points give uniform CDF resolution.
    """
    from scipy.stats import norm

    noise_var = noise_std**2
    ndim = nparams + nobs
    L = domain_half_width

    node_gen = EquidistantNodeGenerator(bkd, (-L, L))
    basis = DynamicPiecewiseBasis(bkd, PiecewiseCubic, node_gen)
    basis.set_nterms(npts_1d)
    pts_1d, wts_1d = basis.quadrature_rule()
    pts_1d_np = bkd.to_numpy(pts_1d).ravel()
    wts_1d_np = bkd.to_numpy(wts_1d).ravel()

    wts_reweighted = wts_1d_np * norm.pdf(pts_1d_np)

    tp = list(iterproduct(range(len(pts_1d_np)), repeat=ndim))
    joint_nodes = np.array(
        [[pts_1d_np[idx[d]] for d in range(ndim)] for idx in tp]
    ).T
    joint_weights = np.array(
        [np.prod([wts_reweighted[idx[d]] for d in range(ndim)]) for idx in tp]
    )

    param_std = joint_nodes[:nparams, :]
    latent_std = joint_nodes[nparams:, :]
    theta = prior_std * param_std

    obs_locs = np.linspace(-1, 1, nobs) if nobs > 1 else np.array([0.0])
    A = np.column_stack([obs_locs**p for p in range(nparams)])
    B = np.ones((1, nparams)) / nparams

    outer_shapes = bkd.asarray(A @ theta)
    latent = bkd.asarray(latent_std)
    qoi_lognormal = bkd.asarray(np.exp(B @ theta).T)
    quad_weights = bkd.asarray(joint_weights)

    noise_variances = bkd.full((nobs,), noise_var)
    w_uniform = np.full(nobs, 1.0 / nobs)
    w = bkd.full((nobs, 1), 1.0 / nobs)

    prior_cov = bkd.asarray(np.eye(nparams) * prior_std**2)
    prior_mean = bkd.zeros((nparams, 1))
    noise_cov = bkd.diag(bkd.asarray(noise_var / w_uniform))
    B_bkd = bkd.asarray(B)
    A_bkd = bkd.asarray(A)

    avar_alpha = 0.5
    u3_ln = ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev(
        prior_mean, prior_cov, B_bkd, avar_alpha, bkd,
    )
    u3_ln.set_observation_matrix(A_bkd)
    u3_ln.set_noise_covariance(noise_cov)
    ref_u3_lognormal = u3_ln.value()

    return {
        "outer_shapes": outer_shapes,
        "inner_shapes": outer_shapes,
        "latent": latent,
        "qoi_lognormal": qoi_lognormal,
        "quad_weights": quad_weights,
        "noise_variances": noise_variances,
        "w": w,
        "ref_u3_lognormal": ref_u3_lognormal,
        "avar_alpha": avar_alpha,
    }


_PW_NPTS_1D = 85
_PW_DOMAIN_HALF_WIDTH = 5.0


class TestPiecewiseCubicU3Convergence:
    def test_u3_lognormal_matches_analytical(
        self, numpy_bkd: Backend[Array],
    ) -> None:
        d = _piecewise_cubic_oed_data(
            numpy_bkd, 1, 1, _PRIOR_STD, _NOISE_STD,
            _PW_NPTS_1D, _PW_DOMAIN_HALF_WIDTH,
        )
        exact_avar = ExactAVaR(d["avar_alpha"], numpy_bkd)
        obj = create_prediction_oed_objective(
            d["noise_variances"], d["outer_shapes"], d["inner_shapes"],
            d["latent"], d["qoi_lognormal"], numpy_bkd,
            deviation_type="stdev",
            risk_type="mean",
            noise_stat=exact_avar,
            outer_quad_weights=d["quad_weights"],
            inner_quad_weights=d["quad_weights"],
        )
        val = float(numpy_bkd.to_numpy(obj(d["w"])).flat[0])
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([val]),
            numpy_bkd.asarray([d["ref_u3_lognormal"]]),
            rtol=2e-4,
        )
