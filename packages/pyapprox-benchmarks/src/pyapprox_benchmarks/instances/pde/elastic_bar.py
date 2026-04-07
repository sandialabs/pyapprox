"""1D elastic bar benchmark instances for UQ workflows.

Wraps the zoo factories (linear and hyperelastic 1D bar) into benchmark
instances with standard normal KLE priors and configurable QoI functionals.
"""

from __future__ import annotations

from typing import Any, Tuple, Union

from pyapprox_benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox_benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox_benchmarks.registry import BenchmarkRegistry
from pyapprox.pde.collocation.functionals.point_evaluation import (
    PointEvaluationFunctional,
)
from pyapprox.pde.collocation.functionals.strain_energy_1d import (
    StrainEnergyFunctional1D,
    create_linear_strain_energy_1d,
    create_neo_hookean_strain_energy_1d,
)
from pyapprox.pde.collocation.functionals.subdomain_integral import (
    SubdomainIntegralFunctional,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)
from pyapprox.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.zoo.elastic_bar_1d import (
    create_linear_elastic_bar_1d,
)
from pyapprox.pde.zoo.hyperelastic_bar_1d import (
    create_hyperelastic_bar_1d,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend


class PDEBenchmarkWrapper:
    """PDE benchmark wrapper.

    Satisfies: HasForwardModel, HasPrior, HasEstimatedEvaluationCost.
    Conditionally satisfies HasJacobian if the forward model exposes
    a jacobian method.
    """

    def __init__(self, inner: Any, estimated_cost: float) -> None:
        self._inner = inner
        self._estimated_cost = estimated_cost
        if hasattr(inner.function(), "jacobian"):
            self.jacobian = self._jacobian

    def name(self) -> str:
        return self._inner.name()

    def function(self) -> Any:
        return self._inner.function()

    def domain(self) -> Any:
        return self._inner.domain()

    def prior(self) -> Any:
        return self._inner.prior()

    def ground_truth(self) -> Any:
        return self._inner.ground_truth()

    def estimated_evaluation_cost(self) -> float:
        return self._estimated_cost

    def _jacobian(self, sample: Any) -> Any:
        return self._inner.function().jacobian(sample)


def _make_kle_field_map(
    bkd: Backend[Array],
    mesh: TransformedMesh1D[Array],
    num_kle_terms: int,
    sigma: float,
    correlation_length: float,
) -> TransformedFieldMap[Array]:
    """Create lognormal KLE field map on mesh nodes."""
    physical_pts = mesh.points()  # shape (1, npts)
    npts = physical_pts.shape[1]
    x_min = physical_pts[0, 0]
    x_max = physical_pts[0, -1]
    length = x_max - x_min
    mesh_coords = (physical_pts - x_min) / length  # normalize to [0, 1]
    mean_log = bkd.zeros((npts,))
    return create_lognormal_kle_field_map(
        mesh_coords,
        mean_log,
        bkd,
        num_kle_terms=num_kle_terms,
        sigma=sigma,
        correlation_length=correlation_length,
    )


def _lame_from_E(
    E_mean: float,
    poisson_ratio: float,
) -> Tuple[float, float]:
    """Convert Young's modulus and Poisson ratio to Lame parameters."""
    mu = E_mean / (2.0 * (1.0 + poisson_ratio))
    lamda = (
        E_mean * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    return lamda, mu


def _make_functional(
    basis: ChebyshevBasis1D[Array],
    nparams: int,
    bkd: Backend[Array],
    qoi: str,
    constitutive: str,
    length: float,
    E_mean: float,
    poisson_ratio: float,
) -> Union[
    PointEvaluationFunctional[Array],
    SubdomainIntegralFunctional[Array],
    StrainEnergyFunctional1D[Array],
]:
    """Create QoI functional from string identifier."""
    npts = basis.nodes().shape[0]

    if qoi == "tip_displacement":
        return PointEvaluationFunctional(basis, length, nparams, bkd)

    if qoi == "average_displacement":
        coeff = bkd.ones((npts,)) / length
        return SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            coefficient=coeff,
        )

    if qoi == "average_stress":
        return _make_average_stress_functional(
            basis,
            nparams,
            bkd,
            constitutive,
            length,
            E_mean,
            poisson_ratio,
        )

    if qoi == "strain_energy":
        return _make_strain_energy_functional(
            basis,
            nparams,
            bkd,
            constitutive,
            E_mean,
            poisson_ratio,
        )

    raise ValueError(
        f"Unknown qoi: {qoi!r}. Must be one of: "
        "'tip_displacement', 'average_displacement', "
        "'average_stress', 'strain_energy'"
    )


def _make_average_stress_functional(
    basis: ChebyshevBasis1D[Array],
    nparams: int,
    bkd: Backend[Array],
    constitutive: str,
    length: float,
    E_mean: float,
    poisson_ratio: float,
) -> StrainEnergyFunctional1D[Array]:
    """Create average stress functional: (1/L) integral sigma dx."""
    if constitutive == "linear":
        inv_L = 1.0 / length
        E_over_L = E_mean * inv_L

        def energy_density_linear(
            epsilon: Array,
            bkd: Backend[Array],
        ) -> Tuple[Array, Array]:
            return E_over_L * epsilon, E_over_L * bkd.ones_like(epsilon)

        return StrainEnergyFunctional1D(
            basis,
            nparams,
            bkd,
            energy_density_linear,
            deformation_gradient=False,
        )

    # hyperelastic
    lamda, mu = _lame_from_E(E_mean, poisson_ratio)
    stress_model = NeoHookeanStress(lamda=lamda, mu=mu)
    inv_L = 1.0 / length

    def energy_density_hyperelastic(
        F: Array,
        bkd: Backend[Array],
    ) -> Tuple[Array, Array]:
        P = stress_model.compute_stress_1d(F, bkd)
        C = stress_model.compute_tangent_1d(F, bkd)
        return inv_L * P, inv_L * C

    return StrainEnergyFunctional1D(
        basis,
        nparams,
        bkd,
        energy_density_hyperelastic,
        deformation_gradient=True,
    )


def _make_strain_energy_functional(
    basis: ChebyshevBasis1D[Array],
    nparams: int,
    bkd: Backend[Array],
    constitutive: str,
    E_mean: float,
    poisson_ratio: float,
) -> StrainEnergyFunctional1D[Array]:
    """Create strain energy functional."""
    if constitutive == "linear":
        return create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E_mean,
        )
    lamda, mu = _lame_from_E(E_mean, poisson_ratio)
    return create_neo_hookean_strain_energy_1d(
        basis,
        nparams,
        bkd,
        lamda,
        mu,
    )


def elastic_bar_1d(
    bkd: Backend[Array],
    constitutive: str = "linear",
    qoi: str = "tip_displacement",
    npts: int = 25,
    length: float = 1.0,
    E_mean: float = 1.0,
    poisson_ratio: float = 0.3,
    traction: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth[Array]]:
    """Create a 1D elastic bar benchmark for UQ workflows.

    Maps KLE coefficients (standard normal) to a scalar QoI via a
    collocation PDE solve. Supports linear and hyperelastic (Neo-Hookean)
    constitutive models with four QoI options.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    constitutive : str
        ``"linear"`` or ``"hyperelastic"`` (Neo-Hookean).
    qoi : str
        Quantity of interest:
        - ``"tip_displacement"``: u(x=L) via interpolation
        - ``"average_displacement"``: (1/L) integral u dx
        - ``"average_stress"``: (1/L) integral sigma dx
        - ``"strain_energy"``: integral psi(epsilon) dx
    npts : int
        Number of Chebyshev collocation points.
    length : float
        Bar length L.
    E_mean : float
        Mean Young's modulus (at KLE coefficients = 0, E = exp(0) = 1
        regardless; E_mean controls the mean of the log-field and the
        reference modulus used in QoI functionals).
    poisson_ratio : float
        Fixed Poisson ratio (hyperelastic only; ignored for linear).
    traction : float
        Applied traction at x=L.
    num_kle_terms : int
        Number of KLE terms (= number of input parameters).
    sigma : float
        Standard deviation of the log-field for KLE.
    correlation_length : float
        Correlation length for the KLE kernel.

    Returns
    -------
    BenchmarkWithPrior
        Benchmark with ``function()`` (the forward model), ``prior()``
        (iid standard normal), and ``domain()`` ([-4, 4]^n).
    """
    if constitutive not in ("linear", "hyperelastic"):
        raise ValueError(
            f"constitutive must be 'linear' or 'hyperelastic', got {constitutive!r}"
        )

    # Mesh and basis
    transform = AffineTransform1D((0.0, length), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)
    basis = ChebyshevBasis1D(mesh, bkd)

    # KLE field map
    field_map = _make_kle_field_map(
        bkd,
        mesh,
        num_kle_terms,
        sigma,
        correlation_length,
    )

    # QoI functional
    functional = _make_functional(
        basis,
        num_kle_terms,
        bkd,
        qoi,
        constitutive,
        length,
        E_mean,
        poisson_ratio,
    )

    # Forward model via zoo factory
    forcing = lambda t: bkd.ones((npts,))  # noqa: E731
    if constitutive == "linear":
        fwd = create_linear_elastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean_field=E_mean,
            forcing=forcing,
            field_map=field_map,
            traction=traction,
            functional=functional,
        )
    else:
        fwd = create_hyperelastic_bar_1d(
            bkd=bkd,
            npts=npts,
            length=length,
            E_mean=E_mean,
            poisson_ratio=poisson_ratio,
            forcing=forcing,
            field_map=field_map,
            traction=traction,
            functional=functional,
        )

    # Prior: iid standard normal
    prior = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(num_kle_terms)],
        bkd,
    )

    # Domain: [-4, 4]^n (4-sigma truncation)
    bounds = bkd.array([[-4.0, 4.0]] * num_kle_terms)
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    name = f"elastic_bar_1d_{constitutive}_{qoi}"
    description = (
        f"1D {constitutive} elastic bar, QoI={qoi}, "
        f"npts={npts}, {num_kle_terms} KLE terms"
    )

    inner = BenchmarkWithPrior(
        _name=name,
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=description,
    )
    estimated_cost = 2.4e-04 if constitutive == "linear" else 1.1e-03
    return PDEBenchmarkWrapper(inner, estimated_cost=estimated_cost)


@BenchmarkRegistry.register(
    "elastic_bar_1d_linear",
    category="pde",
    description="1D linear elastic bar with KLE-parameterized Young's modulus",
)
def _elastic_bar_1d_linear_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return elastic_bar_1d(bkd, constitutive="linear")


@BenchmarkRegistry.register(
    "elastic_bar_1d_hyperelastic",
    category="pde",
    description=(
        "1D hyperelastic (Neo-Hookean) bar with KLE-parameterized Young's modulus"
    ),
)
def _elastic_bar_1d_hyperelastic_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return elastic_bar_1d(bkd, constitutive="hyperelastic")
