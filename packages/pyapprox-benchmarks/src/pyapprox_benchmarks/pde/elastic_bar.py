"""1D elastic bar forward UQ problems.

Wraps the collocation bar builders (linear and hyperelastic 1D bar) into
ForwardUQProblem instances with standard normal KLE priors. The
constitutive law and the QoI are objects passed in, each satisfying a
protocol of this module (:class:`BarConstitutiveProtocol`,
:class:`BarQoIProtocol`), so a new law or QoI needs no change here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, runtime_checkable

from pyapprox.implicitfunction.functional_terms import StateTermProtocol
from pyapprox.implicitfunction.observation_terms import ObservedValue
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.functionals.density_terms import (
    NodalMaterialMapProtocol,
    gradient_density_term_1d,
)
from pyapprox.pde.collocation.functionals.observations import (
    point_observation,
    subdomain_integral_observation,
)
from pyapprox.pde.collocation.functionals.strain_energy_1d import (
    GradientDensity1DProtocol,
    LinearStrainEnergyDensity1D,
    LinearStressDensity1D,
    NeoHookeanStrainEnergyDensity1D,
    NeoHookeanStressDensity1D,
)
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeAwareCallableProtocol,
    TimeIndependent,
)
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.lame import FixedPoissonRatioLameMap
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.models.collocation.steady_function import (
    ParameterizedCollocationSystem,
    collocation_steady_function,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox_benchmarks.functions.pde.collocation.elastic_bar_1d import (
    build_linear_elastic_bar_1d,
)
from pyapprox_benchmarks.functions.pde.collocation.hyperelastic_bar_1d import (
    build_hyperelastic_bar_1d,
)
from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem


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


class _UnitLoad(Generic[Array]):
    """The body force f = 1."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, coords: Array) -> Array:
        return self._bkd.ones((coords.shape[1],))


@runtime_checkable
class BarConstitutiveProtocol(Protocol):
    """A constitutive law of the bar: its configured system and the
    material and densities its stress and energy QoIs read."""

    def label(self) -> str:
        """The name used in the problem's name."""
        ...

    def configure(
        self,
        mesh: TransformedMesh1D[Array],
        forcing: TimeAwareCallableProtocol[Array],
        field_map: FieldMapProtocol[Array],
        traction: float,
    ) -> ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]]:
        """The bar, fixed at the left end, ``traction`` at the right, with
        Young's modulus ``field_map``."""
        ...

    def material_map(
        self, field_map: FieldMapProtocol[Array], npts: int, bkd: Backend[Array]
    ) -> NodalMaterialMapProtocol[Array]:
        """The material the densities read, from the parameters: the same
        map the configured parameterization applies."""
        ...

    def stress_density(self, bkd: Backend[Array]) -> GradientDensity1DProtocol[Array]:
        """The pointwise stress."""
        ...

    def strain_energy_density(
        self, bkd: Backend[Array]
    ) -> GradientDensity1DProtocol[Array]:
        """The pointwise strain energy density."""
        ...


@dataclass(frozen=True)
class LinearElastic:
    """Linear elasticity, :math:`\\sigma = E u'`; the material is E."""

    def label(self) -> str:
        return "linear"

    def configure(
        self,
        mesh: TransformedMesh1D[Array],
        forcing: TimeAwareCallableProtocol[Array],
        field_map: FieldMapProtocol[Array],
        traction: float,
    ) -> ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]]:
        return build_linear_elastic_bar_1d(mesh, forcing, field_map, traction)

    def material_map(
        self, field_map: FieldMapProtocol[Array], npts: int, bkd: Backend[Array]
    ) -> NodalMaterialMapProtocol[Array]:
        return field_map

    def stress_density(self, bkd: Backend[Array]) -> GradientDensity1DProtocol[Array]:
        return LinearStressDensity1D(bkd)

    def strain_energy_density(
        self, bkd: Backend[Array]
    ) -> GradientDensity1DProtocol[Array]:
        return LinearStrainEnergyDensity1D(bkd)


@dataclass(frozen=True)
class NeoHookean:
    """A Neo-Hookean law with a fixed Poisson ratio; the material is the
    stacked Lame fields ``[mu; lamda]`` of E."""

    poisson_ratio: float = 0.3

    def label(self) -> str:
        return "hyperelastic"

    def configure(
        self,
        mesh: TransformedMesh1D[Array],
        forcing: TimeAwareCallableProtocol[Array],
        field_map: FieldMapProtocol[Array],
        traction: float,
    ) -> ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]]:
        return build_hyperelastic_bar_1d(
            mesh, forcing, field_map, self.poisson_ratio, traction
        )

    def material_map(
        self, field_map: FieldMapProtocol[Array], npts: int, bkd: Backend[Array]
    ) -> NodalMaterialMapProtocol[Array]:
        return FixedPoissonRatioLameMap(field_map, self.poisson_ratio, npts, bkd)

    def stress_density(self, bkd: Backend[Array]) -> GradientDensity1DProtocol[Array]:
        return NeoHookeanStressDensity1D(bkd)

    def strain_energy_density(
        self, bkd: Backend[Array]
    ) -> GradientDensity1DProtocol[Array]:
        return NeoHookeanStrainEnergyDensity1D(bkd)


@dataclass(frozen=True)
class BarQoIContext(Generic[Array]):
    """What a QoI builds its term from.

    Attributes
    ----------
    basis : ChebyshevBasis1D
        The bar's basis.
    parameters : ParameterState
        The parameters, shared with the forward model.
    material_map : NodalMaterialMapProtocol
        The material from the parameters, as the constitutive law reads it.
    stress, strain_energy : GradientDensity1DProtocol
        The law's pointwise stress and strain energy densities.
    length : float
        The bar length L.
    bkd : Backend
        Computational backend.
    """

    basis: ChebyshevBasis1D[Array]
    parameters: ParameterState[Array]
    material_map: NodalMaterialMapProtocol[Array]
    stress: GradientDensity1DProtocol[Array]
    strain_energy: GradientDensity1DProtocol[Array]
    length: float
    bkd: Backend[Array]


@runtime_checkable
class BarQoIProtocol(Protocol):
    """A scalar QoI of the bar's displacement."""

    def label(self) -> str:
        """The name used in the problem's name."""
        ...

    def term(self, context: BarQoIContext[Array]) -> StateTermProtocol[Array, Array]:
        """The QoI's term, of one entry."""
        ...


@dataclass(frozen=True)
class TipDisplacement:
    """u(x=L) via interpolation."""

    def label(self) -> str:
        return "tip_displacement"

    def term(self, context: BarQoIContext[Array]) -> StateTermProtocol[Array, Array]:
        bkd = context.bkd
        return ObservedValue(point_observation(context.basis, [context.length], bkd))


@dataclass(frozen=True)
class AverageDisplacement:
    """(1/L) integral u dx."""

    def label(self) -> str:
        return "average_displacement"

    def term(self, context: BarQoIContext[Array]) -> StateTermProtocol[Array, Array]:
        bkd = context.bkd
        coefficient = bkd.ones((context.basis.npts(),)) / context.length
        return ObservedValue(
            subdomain_integral_observation(context.basis, bkd, coefficient)
        )


@dataclass(frozen=True)
class AverageStress:
    """(1/L) integral sigma dx, sigma the law's stress at the material the
    state was solved with."""

    def label(self) -> str:
        return "average_stress"

    def term(self, context: BarQoIContext[Array]) -> StateTermProtocol[Array, Array]:
        return gradient_density_term_1d(
            context.basis,
            context.stress,
            context.material_map,
            context.parameters,
            context.bkd,
            scale=1.0 / context.length,
        )


@dataclass(frozen=True)
class StrainEnergy:
    """integral psi(epsilon) dx, psi the law's strain energy density at
    the material the state was solved with."""

    def label(self) -> str:
        return "strain_energy"

    def term(self, context: BarQoIContext[Array]) -> StateTermProtocol[Array, Array]:
        return gradient_density_term_1d(
            context.basis,
            context.strain_energy,
            context.material_map,
            context.parameters,
            context.bkd,
        )


def build_elastic_bar_1d(
    bkd: Backend[Array],
    constitutive: BarConstitutiveProtocol = LinearElastic(),
    qoi: BarQoIProtocol = TipDisplacement(),
    npts: int = 25,
    length: float = 1.0,
    traction: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> ForwardUQProblem[SteadyImplicitFunction[Array, DenseFormat[Array]], Array]:
    """Create a 1D elastic bar forward UQ problem.

    Maps KLE coefficients (standard normal) to a scalar QoI via a
    collocation PDE solve. The bar is fixed at x=0, carries the traction
    at x=L and a unit body force, and has Young's modulus E = exp(KLE),
    so E = 1 at KLE coefficients = 0.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    constitutive : BarConstitutiveProtocol
        The law: ``LinearElastic()`` or ``NeoHookean(poisson_ratio)``.
    qoi : BarQoIProtocol
        Quantity of interest:
        - ``TipDisplacement()``: u(x=L) via interpolation
        - ``AverageDisplacement()``: (1/L) integral u dx
        - ``AverageStress()``: (1/L) integral sigma dx
        - ``StrainEnergy()``: integral psi(epsilon) dx
    npts : int
        Number of Chebyshev collocation points.
    length : float
        Bar length L.
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
    ForwardUQProblem
        Problem with ``function()`` (the forward model) and ``prior()``
        (iid standard normal).

    Raises
    ------
    TypeError
        If ``constitutive`` or ``qoi`` does not satisfy its protocol.
    """
    if not isinstance(constitutive, BarConstitutiveProtocol):
        raise TypeError(
            "constitutive must satisfy BarConstitutiveProtocol, got "
            f"{type(constitutive).__name__}"
        )
    if not isinstance(qoi, BarQoIProtocol):
        raise TypeError(
            f"qoi must satisfy BarQoIProtocol, got {type(qoi).__name__}"
        )

    # Mesh
    transform = AffineTransform1D((0.0, length), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)

    # KLE field map
    field_map = _make_kle_field_map(
        bkd,
        mesh,
        num_kle_terms,
        sigma,
        correlation_length,
    )

    # The configured bar, and the parameters its residual and the QoI
    # share
    configured = constitutive.configure(
        mesh, TimeIndependent(_UnitLoad(bkd)), field_map, traction
    )
    parameters = ParameterState(configured.parameterization)

    # QoI functional, reading the material from the parameters with the
    # map the configured parameterization applies to the same field map
    context = BarQoIContext(
        configured.basis,
        parameters,
        constitutive.material_map(field_map, npts, bkd),
        constitutive.stress_density(bkd),
        constitutive.strain_energy_density(bkd),
        length,
        bkd,
    )
    functional = StateFunctional.from_terms(
        bkd, 1, state_terms=(qoi.term(context),)
    )

    # Forward model
    fwd = collocation_steady_function(configured.system, parameters, functional)

    # Prior: iid standard normal
    prior = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(num_kle_terms)],
        bkd,
    )

    name = f"elastic_bar_1d_{constitutive.label()}_{qoi.label()}"
    description = (
        f"1D {constitutive.label()} elastic bar, QoI={qoi.label()}, "
        f"npts={npts}, {num_kle_terms} KLE terms"
    )

    return ForwardUQProblem(
        name=name,
        function=fwd,
        prior=prior,
        description=description,
    )
