"""
Prediction OED objective function.

The prediction OED objective minimizes expected deviation in QoI predictions:
    objective(w) = noise_stat[risk_measure[deviation(qoi | obs, w)]]

where:
- deviation: Measures spread of QoI predictions (StdDev, Entropic, AVaR)
- risk_measure: Aggregates deviations over prediction space
- noise_stat: Averages over data realizations
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.typing.expdesign.evidence import Evidence
from pyapprox.typing.expdesign.deviation.base import DeviationMeasure
from pyapprox.typing.expdesign.statistics.base import SampleStatistic


class PredictionOEDObjective(Generic[Array]):
    """
    Prediction-based OED objective.

    Computes:
        objective(w) = noise_stat[risk_measure[deviation(qoi | obs, w)]]

    The objective returns positive values (minimization reduces deviation).

    Parameters
    ----------
    inner_likelihood : GaussianOEDInnerLoopLikelihood[Array]
        Inner loop likelihood for evidence computation.
    outer_shapes : Array
        Model outputs for outer samples. Shape: (nobs, nouter)
    latent_samples : Array
        Latent noise samples for reparameterization. Shape: (nobs, nouter)
    inner_shapes : Array
        Model outputs for inner samples. Shape: (nobs, ninner)
    qoi_vals : Array
        QoI values at inner samples. Shape: (ninner, npred)
    deviation_measure : DeviationMeasure[Array]
        Deviation measure (StdDev, Entropic, or AVaR).
    risk_measure : SampleStatistic[Array]
        Risk measure to aggregate deviations over predictions.
    noise_stat : SampleStatistic[Array]
        Statistic to average over data realizations.
    outer_quad_weights : Array, optional
        Quadrature weights for outer expectation. Shape: (nouter,)
    inner_quad_weights : Array, optional
        Quadrature weights for evidence integration. Shape: (ninner,)
    qoi_quad_weights : Array, optional
        Quadrature weights for prediction aggregation. Shape: (1, npred)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        inner_likelihood: GaussianOEDInnerLoopLikelihood[Array],
        outer_shapes: Array,
        latent_samples: Array,
        inner_shapes: Array,
        qoi_vals: Array,
        deviation_measure: DeviationMeasure[Array],
        risk_measure: SampleStatistic[Array],
        noise_stat: SampleStatistic[Array],
        outer_quad_weights: Optional[Array],
        inner_quad_weights: Optional[Array],
        qoi_quad_weights: Optional[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._inner_loglike = inner_likelihood

        self._nobs = inner_likelihood.nobs()
        self._nouter = outer_shapes.shape[1]
        self._ninner = inner_shapes.shape[1]
        self._npred = qoi_vals.shape[1]

        # Validate qoi_vals shape
        if qoi_vals.shape[0] != self._ninner:
            raise ValueError(
                f"qoi_vals first dimension {qoi_vals.shape[0]} must match "
                f"ninner {self._ninner}"
            )

        # Store shapes and QoI values
        self._outer_shapes = outer_shapes
        self._latent_samples = latent_samples
        self._inner_shapes = inner_shapes
        self._qoi_vals = qoi_vals

        # Set up inner likelihood with shapes
        self._inner_loglike.set_shapes(inner_shapes)

        # Set quadrature weights
        if outer_quad_weights is None:
            outer_quad_weights = bkd.ones((self._nouter,)) / self._nouter
        if inner_quad_weights is None:
            inner_quad_weights = bkd.ones((self._ninner,)) / self._ninner
        if qoi_quad_weights is None:
            qoi_quad_weights = bkd.ones((1, self._npred)) / self._npred

        self._outer_quad_weights = outer_quad_weights
        self._inner_quad_weights = inner_quad_weights
        self._qoi_quad_weights = qoi_quad_weights

        # Store measures
        self._deviation_measure = deviation_measure
        self._risk_measure = risk_measure
        self._noise_stat = noise_stat

        # Evidence will be created when observations are set
        self._evidence: Optional[Evidence[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of design variables (= nobs)."""
        return self._nobs

    def nqoi(self) -> int:
        """Number of outputs (= 1 for scalar objective)."""
        return 1

    def nobs(self) -> int:
        """Number of observation locations."""
        return self._nobs

    def ninner(self) -> int:
        """Number of inner (prior) samples."""
        return self._ninner

    def nouter(self) -> int:
        """Number of outer (observation) samples."""
        return self._nouter

    def npred(self) -> int:
        """Number of prediction QoI locations."""
        return self._npred

    def deviation_measure(self) -> DeviationMeasure[Array]:
        """Get the deviation measure."""
        return self._deviation_measure

    def risk_measure(self) -> SampleStatistic[Array]:
        """Get the risk measure."""
        return self._risk_measure

    def noise_stat(self) -> SampleStatistic[Array]:
        """Get the noise statistic."""
        return self._noise_stat

    def _generate_observations(self, design_weights: Array) -> Array:
        """Generate artificial observations using reparameterization trick.

        obs = shapes + sqrt(variance / weights) * latent_samples
        """
        base_var = self._inner_loglike._base_variances
        effective_std = self._bkd.sqrt(base_var[:, None] / design_weights)
        return self._outer_shapes + effective_std * self._latent_samples

    def _update_observations(self, design_weights: Array) -> None:
        """Update observations for the current design weights."""
        obs = self._generate_observations(design_weights)

        # Set observations on inner likelihood
        self._inner_loglike.set_observations(obs)

        # Set latent samples for reparameterization trick
        self._inner_loglike.set_latent_samples(self._latent_samples)

        # Create Evidence with current inner likelihood
        self._evidence = Evidence(
            self._inner_loglike, self._inner_quad_weights, self._bkd
        )

        # Set up deviation measure with evidence and QoI data
        self._deviation_measure.set_evidence(self._evidence)
        self._deviation_measure.set_qoi_data(self._qoi_vals)

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate the prediction OED objective.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Objective value. Shape: (1, 1)
        """
        self._update_observations(design_weights)

        # Compute deviations: Shape (1, npred * nouter)
        deviations_flat = self._deviation_measure(design_weights)

        # Reshape to (npred, nouter)
        # deviations[q, o] = deviation of prediction q for outer sample o
        deviations = self._bkd.reshape(
            deviations_flat, (self._npred, self._nouter)
        )

        # Apply risk measure over predictions
        # risk_measure expects (nqoi, nsamples) with weights (1, nsamples)
        # Here: nqoi = nouter (outer samples), nsamples = npred (predictions)
        # deviations shape: (npred, nouter) -> transpose to (nouter, npred)
        # qoi_quad_weights shape: (1, npred)
        risk_values = self._risk_measure(deviations.T, self._qoi_quad_weights)
        # risk_values shape: (nouter, 1)

        # Apply noise stat over data realizations
        # noise_stat expects (nqoi, nsamples) with weights (1, nsamples)
        # Here: nqoi = 1, nsamples = nouter
        # risk_values.T shape: (1, nouter), outer_weights shape: (1, nouter)
        outer_weights = self._bkd.reshape(
            self._outer_quad_weights, (1, self._nouter)
        )
        objective = self._noise_stat(risk_values.T, outer_weights)

        return objective

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of objective w.r.t. design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nobs)
        """
        self._update_observations(design_weights)

        # Compute deviations and their Jacobian
        deviations_flat = self._deviation_measure(design_weights)
        deviation_jac_flat = self._deviation_measure.jacobian(design_weights)
        # deviation_jac_flat shape: (npred * nouter, nobs)

        # Reshape deviations to (npred, nouter)
        deviations = self._bkd.reshape(
            deviations_flat, (self._npred, self._nouter)
        )

        # Reshape Jacobian to (npred, nouter, nobs)
        deviation_jac = self._bkd.reshape(
            deviation_jac_flat, (self._npred, self._nouter, self._nobs)
        )

        # Apply risk measure over predictions
        # risk_measure expects (nqoi, nsamples) with weights (1, nsamples)
        # nqoi = nouter, nsamples = npred
        # deviations.T: (nouter, npred), qoi_quad_weights: (1, npred)
        risk_values = self._risk_measure(deviations.T, self._qoi_quad_weights)
        # risk_values shape: (nouter, 1)

        # For jacobian: need (nqoi, nsamples, nvars) = (nouter, npred, nobs)
        # deviation_jac shape: (npred, nouter, nobs)
        # transpose to (nouter, npred, nobs)
        deviation_jac_transposed = self._bkd.einsum(
            "ijk->jik", deviation_jac
        )
        risk_jac = self._risk_measure.jacobian(
            deviations.T, deviation_jac_transposed, self._qoi_quad_weights
        )
        # risk_jac shape: (nouter, nobs)

        # Apply noise stat over data realizations
        # noise_stat expects (nqoi, nsamples) with weights (1, nsamples)
        # nqoi = 1, nsamples = nouter
        outer_weights = self._bkd.reshape(
            self._outer_quad_weights, (1, self._nouter)
        )
        # For noise_stat jacobian: values (1, nouter), jac (1, nouter, nobs)
        risk_jac_3d = self._bkd.reshape(
            risk_jac, (self._nouter, 1, self._nobs)
        )
        # transpose to (nqoi=1, nsamples=nouter, nvars=nobs)
        risk_jac_3d = self._bkd.einsum("ijk->jik", risk_jac_3d)
        objective_jac = self._noise_stat.jacobian(
            risk_values.T, risk_jac_3d, outer_weights
        )
        # objective_jac shape: (1, nobs)

        return objective_jac

    def evaluate(self, design_weights: Array) -> Array:
        """
        Single-sample evaluation (alias for __call__).

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Objective value. Shape: (1, 1)
        """
        return self(design_weights)

    def __repr__(self) -> str:
        return (
            f"PredictionOEDObjective("
            f"deviation={self._deviation_measure.label()}, "
            f"risk={self._risk_measure}, "
            f"noise_stat={self._noise_stat})"
        )
