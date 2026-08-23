"""Cantilever beam multi-fidelity ensemble problem.

Four-model partially ordered ensemble sharing one ``(xi, v)``
parameterization:

- f0: 2D Neo-Hookean (nonlinear), fine mesh — HF reference
- f1: 2D linear elasticity, same fine mesh — reduced physics
- f2: 2D linear elasticity, coarse mesh — reduced physics + mesh
- f3: PCE surrogate of f0 — data-fit with analytic (known) mean

The shared KLE field ensures every model sees the same Young's modulus
realization for a given ``xi``.  The random wind speed ``v`` enters as
dynamic pressure ``q0 = v^2``, making all models nonlinear in the
input.  Models are wrapped with ``make_parallel`` for multi-core
evaluation and ``timed`` for cost measurement.
"""

from typing import Dict, Generic, List, Optional, Tuple

import numpy as np
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.interface.functions.timing import TimedFunction
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.scipy_continuous import (
    ScipyContinuousMarginal,
)
from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
)
from pyapprox.util.backends.protocols import Array, Backend
from scipy.stats import lognorm

from pyapprox_benchmarks.pde.cantilever_beam_ensemble import (
    MESH_PATHS,
    build_shared_field_beam,
)
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)


class CantileverBeamEnsembleProblem(Generic[Array]):
    """Four-model cantilever beam ensemble for multi-fidelity estimation.

    Builds FEM models f0-f2 on construction, wraps them with
    ``make_parallel`` + ``timed`` for parallel evaluation and cost
    measurement.  The PCE surrogate (f3) is fitted separately via
    ``fit_pce_surrogate``.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    num_kle_terms : int
        Number of KLE terms for the shared E field.
    sigma : float
        Standard deviation of log(E) field.
    correlation_length : float
        Correlation length for KLE kernel.
    E_mean : float
        Mean Young's modulus.
    poisson_ratio : float
        Poisson ratio (constant across all models).
    length : float
        Beam length.
    height : float
        Beam height.
    fine_mesh_h : float
        Mesh size key for fine mesh (0.5 or 1).
    coarse_mesh_h : float
        Mesh size key for coarse mesh (2 or 4).
    wind_s : float
        Shape parameter for lognormal wind speed distribution.
    wind_scale : float
        Scale parameter for lognormal wind speed distribution.
    n_jobs : int
        Number of parallel workers for model evaluation (-1 = all CPUs).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        num_kle_terms: int = 5,
        sigma: float = 0.3,
        correlation_length: float = 0.3,
        E_mean: float = 1e4,
        poisson_ratio: float = 0.3,
        length: float = 100.0,
        height: float = 30.0,
        fine_mesh_h: float = 0.5,
        coarse_mesh_h: float = 2,
        wind_s: float = 0.5,
        wind_scale: float = 2.5,
        n_jobs: int = -1,
    ) -> None:
        from pyapprox.interface.functions.timing import timed
        from pyapprox.interface.parallel import make_parallel

        self._bkd = bkd
        self._num_kle_terms = num_kle_terms

        f0_raw = build_shared_field_beam(
            bkd, MESH_PATHS[fine_mesh_h], "neohookean",
            num_kle_terms=num_kle_terms, sigma=sigma,
            correlation_length=correlation_length, E_mean=E_mean,
            poisson_ratio=poisson_ratio, length=length, height=height,
        )
        f1_raw = build_shared_field_beam(
            bkd, MESH_PATHS[fine_mesh_h], "linear",
            num_kle_terms=num_kle_terms, sigma=sigma,
            correlation_length=correlation_length, E_mean=E_mean,
            poisson_ratio=poisson_ratio, length=length, height=height,
        )
        f2_raw = build_shared_field_beam(
            bkd, MESH_PATHS[coarse_mesh_h], "linear",
            num_kle_terms=num_kle_terms, sigma=sigma,
            correlation_length=correlation_length, E_mean=E_mean,
            poisson_ratio=poisson_ratio, length=length, height=height,
        )

        self._models_raw = [f0_raw, f1_raw, f2_raw]
        self._fem_models: List[TimedFunction[Array]] = [
            timed(make_parallel(m, n_jobs=n_jobs)) for m in self._models_raw
        ]
        self._models: List[FunctionProtocol[Array]] = list(self._fem_models)
        self._model_names = [
            "f0: NeoHookean fine",
            "f1: Linear fine",
            "f2: Linear coarse",
        ]

        self._wind_s = wind_s
        self._wind_scale = wind_scale
        self._prior = IndependentJoint(
            [GaussianMarginal(0.0, 1.0, bkd)] * num_kle_terms
            + [ScipyContinuousMarginal(
                lognorm(s=wind_s, scale=wind_scale), bkd,
            )],
            bkd,
        )

        self._pce: Optional[PolynomialChaosExpansion[Array]] = None
        self._pce_mean: Optional[Array] = None
        self._costs: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def prior(self) -> IndependentJoint[Array]:
        return self._prior

    def models(self) -> List[FunctionProtocol[Array]]:
        return list(self._models)

    def model_names(self) -> List[str]:
        return list(self._model_names)

    def nmodels(self) -> int:
        return len(self._models)

    def extract_costs(self) -> Array:
        """Extract measured per-sample costs from timed wrappers.

        Must be called after evaluating all models (e.g. after pilot).
        Normalizes so f0 cost = 1.0.  If a PCE surrogate has been
        fitted, appends a negligible cost for f3.
        """
        bkd = self._bkd
        raw_costs = []
        for m in self._fem_models:
            raw_costs.append(m.timer().get("__call__").median())

        costs_np = np.array(raw_costs)
        costs_np = costs_np / costs_np[0]
        if self._pce is not None:
            costs_np = np.append(costs_np, 1e-5)

        self._costs = bkd.array(costs_np)
        return self._costs

    def costs(self) -> Array:
        if self._costs is None:
            raise RuntimeError(
                "Costs not yet measured. Call extract_costs() after "
                "evaluating models (e.g. after pilot study)."
            )
        return self._costs

    def fit_pce_surrogate(
        self,
        train_samples: Array,
        train_values_f0: Array,
        max_level: int = 4,
        cv_levels: Optional[List[int]] = None,
    ) -> PolynomialChaosExpansion[Array]:
        """Fit a PCE surrogate of f0 and add it as model 3.

        Training data should be drawn independently from the pilot
        to avoid inflating the f3-f0 correlation.

        Parameters
        ----------
        train_samples : Array
            Training inputs, shape ``(nvars, ntrain)``.
        train_values_f0 : Array
            f0 outputs at training inputs, shape ``(nqoi, ntrain)``.
        max_level : int
            Maximum PCE polynomial level for index set.
        cv_levels : list of int, optional
            Candidate levels for LOO CV degree selection.
            Default: ``[1, 2, 3]``.

        Returns
        -------
        pce : PolynomialChaosExpansion
            The fitted PCE.
        """
        from pyapprox.surrogates.affine.expansions.fitters.pce_cv import (
            PCEDegreeSelectionFitter,
        )
        from pyapprox.surrogates.affine.expansions.pce import (
            create_pce_from_marginals,
        )
        from pyapprox.surrogates.affine.indices.generators import (
            HyperbolicIndexSequence,
        )

        if cv_levels is None:
            cv_levels = [1, 2, 3]

        bkd = self._bkd
        nvars = self._num_kle_terms + 1
        nqoi = 2

        marginals = (
            [GaussianMarginal(0.0, 1.0, bkd)] * self._num_kle_terms
            + [ScipyContinuousMarginal(
                lognorm(s=self._wind_s, scale=self._wind_scale), bkd,
            )]
        )
        pce = create_pce_from_marginals(
            marginals, max_level, bkd, nqoi=nqoi,
        )

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, cv_levels, index_seq,
        )
        cv_result = fitter.fit(pce, train_samples, train_values_f0)
        fitted = cv_result.surrogate()
        if not isinstance(fitted, PolynomialChaosExpansion):
            raise TypeError(
                f"Expected PolynomialChaosExpansion from CV fitter, "
                f"got {type(fitted).__name__}"
            )
        self._pce = fitted

        self._pce_mean = self._pce.mean()

        if len(self._models) == 3:
            self._models.append(self._pce)
            self._model_names.append("f3: PCE surrogate")
        else:
            self._models[3] = self._pce
            self._model_names[3] = "f3: PCE surrogate"

        self._costs = None

        return self._pce

    def pce_mean(self) -> Array:
        if self._pce_mean is None:
            raise RuntimeError(
                "PCE not yet fitted. Call fit_pce_surrogate() first."
            )
        return self._pce_mean

    def known_quantities(self) -> Dict[Tuple[int, str], Array]:
        return {(3, "mean"): self.pce_mean()}

    def problem(
        self,
    ) -> MultifidelityForwardUQProblem[FunctionProtocol[Array], Array]:
        return MultifidelityForwardUQProblem(
            "cantilever_beam_ensemble",
            self._models,
            self.costs(),
            self._prior,
            description=(
                "4-model partially ordered cantilever beam ensemble "
                "(NeoHookean fine, linear fine, linear coarse, PCE surrogate)"
            ),
        )


