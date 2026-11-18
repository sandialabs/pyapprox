"""2D pressurized cylinder forward UQ problems.

Wraps the collocation cylinder builders (linear and hyperelastic 2D
pressurized cylinder) into ForwardUQProblem instances with standard
normal KLE priors. The constitutive law and the QoI are objects passed
in, each satisfying a protocol of this module
(:class:`CylinderConstitutiveProtocol`, :class:`CylinderQoIProtocol`), so
a new law or QoI needs no change here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Protocol, runtime_checkable

from pyapprox.implicitfunction.functional_terms import StateTermProtocol
from pyapprox.implicitfunction.observation_terms import ObservedValue
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.functionals.density_terms import (
    HoopStressDensity,
    IntegratedDensity,
    NodalDensityProtocol,
    NodalMaterialMapProtocol,
    StrainEnergyDensity2D,
)
from pyapprox.pde.collocation.functionals.observations import (
    average_radial_displacement_observation,
)
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.post_processing.stress import (
    HyperelasticStressPostProcessor2D,
    StressPostProcessor2D,
)
from pyapprox.pde.collocation.quadrature import (
    CollocationQuadrature2D,
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
from pyapprox_benchmarks.functions.pde.collocation.hyperelastic_cylinder_2d import (
    build_hyperelastic_pressurized_cylinder_2d,
)
from pyapprox_benchmarks.functions.pde.collocation.pressurized_cylinder_2d import (
    build_linear_pressurized_cylinder_2d,
)
from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem


def _make_kle_field_map_2d(
    bkd: Backend[Array],
    mesh: TransformedMesh2D[Array],
    num_kle_terms: int,
    sigma: float,
) -> TransformedFieldMap[Array]:
    """Create lognormal KLE field map on 2D mesh nodes."""
    physical_pts = mesh.points()  # (2, npts)
    npts = physical_pts.shape[1]
    x = physical_pts[0, :]
    y = physical_pts[1, :]
    x_min = bkd.to_float(bkd.min(x))
    x_max = bkd.to_float(bkd.max(x))
    y_min = bkd.to_float(bkd.min(y))
    y_max = bkd.to_float(bkd.max(y))
    x_range = max(x_max - x_min, 1e-12)
    y_range = max(y_max - y_min, 1e-12)
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    mesh_coords = bkd.stack([x_norm, y_norm], axis=0)
    mean_log = bkd.zeros((npts,))
    return create_lognormal_kle_field_map(
        mesh_coords,
        mean_log,
        bkd,
        num_kle_terms=num_kle_terms,
        sigma=sigma,
    )


@runtime_checkable
class CylinderConstitutiveProtocol(Protocol):
    """A constitutive law of the cylinder: its configured system and the
    densities its stress and energy QoIs read, of the material
    ``[mu; lamda]``."""

    def label(self) -> str:
        """The name used in the problem's name."""
        ...

    def configure(
        self,
        mesh: TransformedMesh2D[Array],
        poisson_ratio: float,
        inner_pressure: float,
        field_map: FieldMapProtocol[Array],
    ) -> ParameterizedCollocationSystem[Array, ChebyshevBasis2D[Array]]:
        """The cylinder under ``inner_pressure``, with Young's modulus
        ``field_map`` and the fixed Poisson ratio."""
        ...

    def hoop_stress_density(
        self, basis: ChebyshevBasis2D[Array], curvilinear_basis: Array
    ) -> NodalDensityProtocol[Array]:
        """The pointwise hoop stress."""
        ...

    def strain_energy_density(
        self, basis: ChebyshevBasis2D[Array], curvilinear_basis: Array
    ) -> NodalDensityProtocol[Array]:
        """The pointwise strain energy density."""
        ...


@dataclass(frozen=True)
class LinearElastic:
    """Linear elasticity, the Cauchy stress of the small strain."""

    def label(self) -> str:
        return "linear"

    def configure(
        self,
        mesh: TransformedMesh2D[Array],
        poisson_ratio: float,
        inner_pressure: float,
        field_map: FieldMapProtocol[Array],
    ) -> ParameterizedCollocationSystem[Array, ChebyshevBasis2D[Array]]:
        return build_linear_pressurized_cylinder_2d(
            mesh, poisson_ratio, inner_pressure, field_map
        )

    def _processor(
        self, basis: ChebyshevBasis2D[Array], curvilinear_basis: Array
    ) -> StressPostProcessor2D[Array]:
        return StressPostProcessor2D(
            basis.derivative_matrix(1, 0),
            basis.derivative_matrix(1, 1),
            bkd=basis.bkd(),
            curvilinear_basis=curvilinear_basis,
        )

    def hoop_stress_density(
        self, basis: ChebyshevBasis2D[Array], curvilinear_basis: Array
    ) -> NodalDensityProtocol[Array]:
        return HoopStressDensity(
            basis.bkd(), self._processor(basis, curvilinear_basis)
        )

    def strain_energy_density(
        self, basis: ChebyshevBasis2D[Array], curvilinear_basis: Array
    ) -> NodalDensityProtocol[Array]:
        return StrainEnergyDensity2D(
            basis.bkd(), self._processor(basis, curvilinear_basis)
        )


@dataclass(frozen=True)
class NeoHookean:
    """A plane-strain Neo-Hookean law; the stresses are the Cauchy
    stresses of its first Piola-Kirchhoff stress."""

    def label(self) -> str:
        return "hyperelastic"

    def configure(
        self,
        mesh: TransformedMesh2D[Array],
        poisson_ratio: float,
        inner_pressure: float,
        field_map: FieldMapProtocol[Array],
    ) -> ParameterizedCollocationSystem[Array, ChebyshevBasis2D[Array]]:
        return build_hyperelastic_pressurized_cylinder_2d(
            mesh, poisson_ratio, inner_pressure, field_map
        )

    def _processor(
        self, basis: ChebyshevBasis2D[Array], curvilinear_basis: Array
    ) -> HyperelasticStressPostProcessor2D[Array]:
        bkd = basis.bkd()
        return HyperelasticStressPostProcessor2D(
            basis.derivative_matrix(1, 0),
            basis.derivative_matrix(1, 1),
            NeoHookeanLaw(bkd),
            bkd=bkd,
            curvilinear_basis=curvilinear_basis,
        )

    def hoop_stress_density(
        self, basis: ChebyshevBasis2D[Array], curvilinear_basis: Array
    ) -> NodalDensityProtocol[Array]:
        return HoopStressDensity(
            basis.bkd(), self._processor(basis, curvilinear_basis)
        )

    def strain_energy_density(
        self, basis: ChebyshevBasis2D[Array], curvilinear_basis: Array
    ) -> NodalDensityProtocol[Array]:
        return StrainEnergyDensity2D(
            basis.bkd(), self._processor(basis, curvilinear_basis)
        )


@dataclass(frozen=True)
class CylinderQoIContext(Generic[Array]):
    """What a QoI builds its term from.

    Attributes
    ----------
    mesh : TransformedMesh2D
        The quarter-annulus; its RIGHT side is the outer wall r=R_o.
    basis : ChebyshevBasis2D
        The cylinder's basis.
    parameters : ParameterState
        The parameters, shared with the forward model.
    material_map : NodalMaterialMapProtocol
        The Lame fields ``[mu; lamda]`` from the parameters.
    hoop_stress, strain_energy : NodalDensityProtocol
        The law's pointwise hoop stress and strain energy densities.
    bkd : Backend
        Computational backend.
    """

    mesh: TransformedMesh2D[Array]
    basis: ChebyshevBasis2D[Array]
    parameters: ParameterState[Array]
    material_map: NodalMaterialMapProtocol[Array]
    hoop_stress: NodalDensityProtocol[Array]
    strain_energy: NodalDensityProtocol[Array]
    bkd: Backend[Array]


@runtime_checkable
class CylinderQoIProtocol(Protocol):
    """A scalar QoI of the cylinder's displacement."""

    def label(self) -> str:
        """The name used in the problem's name."""
        ...

    def term(
        self, context: CylinderQoIContext[Array]
    ) -> StateTermProtocol[Array, Array]:
        """The QoI's term, of one entry."""
        ...


@dataclass(frozen=True)
class OuterRadialDisplacement:
    """The average radial displacement u_r over the outer wall nodes."""

    def label(self) -> str:
        return "outer_radial_displacement"

    def term(
        self, context: CylinderQoIContext[Array]
    ) -> StateTermProtocol[Array, Array]:
        bkd, mesh = context.bkd, context.mesh
        outer_idx = mesh.boundary_indices(BoundarySide.RIGHT)
        pts = mesh.points()
        outer_pts_x = pts[0, :][outer_idx]
        outer_pts_y = pts[1, :][outer_idx]
        r_outer = bkd.sqrt(outer_pts_x**2 + outer_pts_y**2)
        return ObservedValue(
            average_radial_displacement_observation(
                bkd,
                [bkd.to_int(outer_idx[k]) for k in range(outer_idx.shape[0])],
                outer_pts_x / r_outer,
                outer_pts_y / r_outer,
                mesh.npts(),
            )
        )


@dataclass(frozen=True)
class AverageHoopStress:
    """The average hoop stress sigma_tt over the weld zone, the inner
    ``weld_r_fraction`` of the wall thickness over the full angle, at the
    material the state was solved with."""

    weld_r_fraction: float = 0.25

    def label(self) -> str:
        return "average_hoop_stress"

    def term(
        self, context: CylinderQoIContext[Array]
    ) -> StateTermProtocol[Array, Array]:
        bkd = context.bkd
        quad = CollocationQuadrature2D(context.basis, bkd)
        # Weld zone: inner fraction of wall in r-direction, full theta
        # In reference coords: x in [-1, -1 + 2*fraction], y in [-1, 1]
        xi_b = -1.0 + 2.0 * self.weld_r_fraction
        w_sub = quad.weights(x_bounds=(-1.0, xi_b))
        return IntegratedDensity(
            bkd,
            context.hoop_stress,
            context.material_map,
            context.parameters,
            w_sub / bkd.sum(w_sub),
        )


@dataclass(frozen=True)
class StrainEnergy:
    """The total strain energy, the integral of the law's strain energy
    density at the material the state was solved with."""

    def label(self) -> str:
        return "strain_energy"

    def term(
        self, context: CylinderQoIContext[Array]
    ) -> StateTermProtocol[Array, Array]:
        bkd = context.bkd
        quad = CollocationQuadrature2D(context.basis, bkd)
        return IntegratedDensity(
            bkd,
            context.strain_energy,
            context.material_map,
            context.parameters,
            quad.full_domain_weights(),
        )


def build_pressurized_cylinder_2d(
    bkd: Backend[Array],
    constitutive: CylinderConstitutiveProtocol = LinearElastic(),
    qoi: CylinderQoIProtocol = OuterRadialDisplacement(),
    npts_r: int = 12,
    npts_theta: int = 12,
    r_inner: float = 1.0,
    r_outer: float = 2.0,
    poisson_ratio: float = 0.3,
    inner_pressure: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
) -> ForwardUQProblem[SteadyImplicitFunction[Array, DenseFormat[Array]], Array]:
    """Create a 2D pressurized cylinder forward UQ problem.

    Maps KLE coefficients (standard normal) to a scalar QoI via a 2D
    collocation PDE solve on a quarter-annulus. Young's modulus is
    E = exp(KLE), so E = 1 at KLE coefficients = 0.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    constitutive : CylinderConstitutiveProtocol
        The law: ``LinearElastic()`` or ``NeoHookean()``.
    qoi : CylinderQoIProtocol
        Quantity of interest:
        - ``OuterRadialDisplacement()``: average u_r on outer wall
        - ``AverageHoopStress(weld_r_fraction)``: average sigma_tt over
          the weld zone
        - ``StrainEnergy()``: total strain energy integral
    npts_r : int
        Number of Chebyshev points in radial direction.
    npts_theta : int
        Number of Chebyshev points in angular direction.
    r_inner : float
        Inner radius R_i.
    r_outer : float
        Outer radius R_o.
    poisson_ratio : float
        Fixed Poisson ratio.
    inner_pressure : float
        Internal pressure.
    num_kle_terms : int
        Number of KLE terms (= number of input parameters).
    sigma : float
        Standard deviation of the log-field for KLE.

    Returns
    -------
    ForwardUQProblem
        Problem with ``function()`` (forward model) and ``prior()``
        (iid standard normal).

    Raises
    ------
    TypeError
        If ``constitutive`` or ``qoi`` does not satisfy its protocol.
    """
    if not isinstance(constitutive, CylinderConstitutiveProtocol):
        raise TypeError(
            "constitutive must satisfy CylinderConstitutiveProtocol, got "
            f"{type(constitutive).__name__}"
        )
    if not isinstance(qoi, CylinderQoIProtocol):
        raise TypeError(
            f"qoi must satisfy CylinderQoIProtocol, got {type(qoi).__name__}"
        )

    # Mesh
    transform = PolarTransform(
        (r_inner, r_outer),
        (0.0, math.pi / 2.0),
        bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)

    # KLE field map
    field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms, sigma)

    # The configured cylinder, and the parameters its residual and the
    # QoI share
    configured = constitutive.configure(
        mesh, poisson_ratio, inner_pressure, field_map
    )
    parameters = ParameterState(configured.parameterization)

    # QoI functional. The stress QoIs evaluate the material from the
    # parameters with the map the configured parameterization builds from
    # the same field map.
    basis = configured.basis
    curvilinear_basis = transform.unit_curvilinear_basis(mesh.reference_points())
    context = CylinderQoIContext(
        mesh,
        basis,
        parameters,
        FixedPoissonRatioLameMap(field_map, poisson_ratio, basis.npts(), bkd),
        constitutive.hoop_stress_density(basis, curvilinear_basis),
        constitutive.strain_energy_density(basis, curvilinear_basis),
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

    name = f"pressurized_cylinder_2d_{constitutive.label()}_{qoi.label()}"
    description = (
        f"2D {constitutive.label()} pressurized cylinder, QoI={qoi.label()}, "
        f"npts_r={npts_r}, npts_theta={npts_theta}, "
        f"{num_kle_terms} KLE terms"
    )

    return ForwardUQProblem(
        name=name,
        function=fwd,
        prior=prior,
        description=description,
    )
