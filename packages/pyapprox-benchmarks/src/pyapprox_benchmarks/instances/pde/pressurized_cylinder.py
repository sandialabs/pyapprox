"""2D pressurized cylinder benchmark instances for UQ workflows.

Wraps the zoo factory (linear elastic 2D pressurized cylinder) into
benchmark instances with standard normal KLE priors and configurable
QoI functionals: outer radial displacement, average hoop stress, and
strain energy.
"""

# TODO: This anot the other elasticity benchmarks in this submodule
# seem to use a lot of custom code
# can we make general testable utilities that we can then resuse to
# reduce code bloat and prouce more reliable code

from __future__ import annotations

import math
from typing import Union

from pyapprox.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.benchmarks.instances.pde.elastic_bar import (
    PDEBenchmarkWrapper,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.pde.collocation.functionals.elasticity_2d import (
    AverageHoopStressFunctional,
    HyperelasticAverageHoopStressFunctional,
    OuterWallRadialDisplacementFunctional,
    StrainEnergyFunctional2D,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.collocation.physics.stress_models import (
    NeoHookeanStress,
)
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
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.zoo.hyperelastic_cylinder_2d import (
    create_hyperelastic_pressurized_cylinder_2d,
)
from pyapprox.pde.zoo.pressurized_cylinder_2d import (
    create_linear_pressurized_cylinder_2d,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend


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


def _make_functional(
    bkd: Backend[Array],
    qoi: str,
    physics: LinearElasticityPhysics[Array],
    basis: ChebyshevBasis2D[Array],
    mesh: TransformedMesh2D[Array],
    transform: PolarTransform[Array],
    nparams: int,
    weld_r_fraction: float,
) -> Union[
    OuterWallRadialDisplacementFunctional[Array],
    AverageHoopStressFunctional[Array],
    StrainEnergyFunctional2D[Array],
]:
    """Create QoI functional from string identifier."""
    npts = basis.npts()
    Dx = basis.derivative_matrix(1, 0)
    Dy = basis.derivative_matrix(1, 1)
    curv_basis = transform.unit_curvilinear_basis(mesh.reference_points())

    proc = StressPostProcessor2D(
        Dx,
        Dy,
        get_lamda=lambda: physics._lambda_array,
        get_mu=lambda: physics._mu_array,
        bkd=bkd,
        curvilinear_basis=curv_basis,
    )

    if qoi == "outer_radial_displacement":
        outer_idx = mesh.boundary_indices(1)
        pts = mesh.points()
        outer_pts_x = pts[0, :][outer_idx]
        outer_pts_y = pts[1, :][outer_idx]
        r_outer = bkd.sqrt(outer_pts_x**2 + outer_pts_y**2)
        cos_theta = outer_pts_x / r_outer
        sin_theta = outer_pts_y / r_outer
        return OuterWallRadialDisplacementFunctional(
            outer_idx,
            cos_theta,
            sin_theta,
            npts,
            nparams,
            bkd,
        )

    if qoi == "average_hoop_stress":
        quad = CollocationQuadrature2D(basis, bkd)
        # Weld zone: inner fraction of wall in r-direction, full theta
        # In reference coords: x in [-1, -1 + 2*fraction], y in [-1, 1]
        xi_b = -1.0 + 2.0 * weld_r_fraction
        w_sub = quad.weights(x_bounds=(-1.0, xi_b))
        area = bkd.to_float(bkd.sum(w_sub))
        return AverageHoopStressFunctional(
            proc,
            w_sub,
            area,
            nparams,
            bkd,
        )

    if qoi == "strain_energy":
        quad = CollocationQuadrature2D(basis, bkd)
        w_full = quad.full_domain_weights()
        return StrainEnergyFunctional2D(proc, w_full, nparams, bkd)

    raise ValueError(
        f"Unknown qoi: {qoi!r}. Must be one of: "
        "'outer_radial_displacement', 'average_hoop_stress', "
        "'strain_energy'"
    )


def _make_hyperelastic_functional(
    bkd: Backend[Array],
    qoi: str,
    stress_model: NeoHookeanStress[Array],
    basis: ChebyshevBasis2D[Array],
    mesh: TransformedMesh2D[Array],
    transform: PolarTransform[Array],
    nparams: int,
    weld_r_fraction: float,
) -> Union[
    OuterWallRadialDisplacementFunctional[Array],
    HyperelasticAverageHoopStressFunctional[Array],
    StrainEnergyFunctional2D[Array],
]:
    """Create QoI functional for hyperelastic problems."""
    npts = basis.npts()
    Dx = basis.derivative_matrix(1, 0)
    Dy = basis.derivative_matrix(1, 1)
    curv_basis = transform.unit_curvilinear_basis(mesh.reference_points())

    proc = HyperelasticStressPostProcessor2D(
        Dx,
        Dy,
        stress_model=stress_model,
        bkd=bkd,
        curvilinear_basis=curv_basis,
    )

    if qoi == "outer_radial_displacement":
        outer_idx = mesh.boundary_indices(1)
        pts = mesh.points()
        outer_pts_x = pts[0, :][outer_idx]
        outer_pts_y = pts[1, :][outer_idx]
        r_outer = bkd.sqrt(outer_pts_x**2 + outer_pts_y**2)
        cos_theta = outer_pts_x / r_outer
        sin_theta = outer_pts_y / r_outer
        return OuterWallRadialDisplacementFunctional(
            outer_idx,
            cos_theta,
            sin_theta,
            npts,
            nparams,
            bkd,
        )

    if qoi == "average_hoop_stress":
        quad = CollocationQuadrature2D(basis, bkd)
        xi_b = -1.0 + 2.0 * weld_r_fraction
        w_sub = quad.weights(x_bounds=(-1.0, xi_b))
        area = bkd.to_float(bkd.sum(w_sub))
        return HyperelasticAverageHoopStressFunctional(
            proc,
            w_sub,
            area,
            nparams,
            bkd,
        )

    if qoi == "strain_energy":
        quad = CollocationQuadrature2D(basis, bkd)
        w_full = quad.full_domain_weights()
        return StrainEnergyFunctional2D(proc, w_full, nparams, bkd)

    raise ValueError(
        f"Unknown qoi: {qoi!r}. Must be one of: "
        "'outer_radial_displacement', 'average_hoop_stress', "
        "'strain_energy'"
    )


def pressurized_cylinder_2d(
    bkd: Backend[Array],
    qoi: str = "outer_radial_displacement",
    npts_r: int = 12,
    npts_theta: int = 12,
    r_inner: float = 1.0,
    r_outer: float = 2.0,
    E_mean: float = 1.0,
    poisson_ratio: float = 0.3,
    inner_pressure: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    weld_r_fraction: float = 0.25,
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth[Array]]:
    """Create a 2D pressurized cylinder benchmark for UQ workflows.

    Maps KLE coefficients (standard normal) to a scalar QoI via a
    2D linear elastic collocation PDE solve on a quarter-annulus.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    qoi : str
        Quantity of interest:
        - ``"outer_radial_displacement"``: average u_r on outer wall
        - ``"average_hoop_stress"``: average sigma_tt over weld zone
        - ``"strain_energy"``: total strain energy integral
    npts_r : int
        Number of Chebyshev points in radial direction.
    npts_theta : int
        Number of Chebyshev points in angular direction.
    r_inner : float
        Inner radius R_i.
    r_outer : float
        Outer radius R_o.
    E_mean : float
        Mean Young's modulus.
    poisson_ratio : float
        Fixed Poisson ratio.
    inner_pressure : float
        Internal pressure.
    num_kle_terms : int
        Number of KLE terms (= number of input parameters).
    sigma : float
        Standard deviation of the log-field for KLE.
    weld_r_fraction : float
        Fraction of wall thickness for weld zone (average_hoop_stress QoI).

    Returns
    -------
    BenchmarkWithPrior
        Benchmark with ``function()`` (forward model), ``prior()``
        (iid standard normal), and ``domain()`` ([-4, 4]^n).
    """
    # Mesh and basis
    transform = PolarTransform(
        (r_inner, r_outer),
        (0.0, math.pi / 2.0),
        bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)

    # KLE field map
    field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms, sigma)

    # We need physics to create the functional (for derivative matrices
    # and Lame params). The zoo factory creates its own physics internally,
    # so we create a temporary physics just to pass to _make_functional.
    # The real physics is created inside the zoo factory with the same
    # initial Lame parameters.
    dmu_dE = 1.0 / (2.0 * (1.0 + poisson_ratio))
    dlam_dE = poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu_init = E_mean * dmu_dE
    lamda_init = E_mean * dlam_dE
    physics = LinearElasticityPhysics(
        basis,
        bkd,
        lamda=lamda_init,
        mu=mu_init,
    )

    # QoI functional
    functional = _make_functional(
        bkd,
        qoi,
        physics,
        basis,
        mesh,
        transform,
        num_kle_terms,
        weld_r_fraction,
    )

    # Forward model via zoo factory
    fwd = create_linear_pressurized_cylinder_2d(
        bkd=bkd,
        npts_r=npts_r,
        npts_theta=npts_theta,
        r_inner=r_inner,
        r_outer=r_outer,
        E_mean=E_mean,
        poisson_ratio=poisson_ratio,
        inner_pressure=inner_pressure,
        field_map=field_map,
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

    name = f"pressurized_cylinder_2d_linear_{qoi}"
    description = (
        f"2D linear elastic pressurized cylinder, QoI={qoi}, "
        f"npts_r={npts_r}, npts_theta={npts_theta}, "
        f"{num_kle_terms} KLE terms"
    )

    inner = BenchmarkWithPrior(
        _name=name,
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=description,
    )
    return PDEBenchmarkWrapper(inner, estimated_cost=3.6e-02)


@BenchmarkRegistry.register(
    "pressurized_cylinder_2d_linear",
    category="pde",
    description=(
        "2D linear elastic pressurized cylinder with KLE-parameterized Young's modulus"
    ),
)
def _pressurized_cylinder_2d_linear_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return pressurized_cylinder_2d(bkd)


def hyperelastic_pressurized_cylinder_2d(
    bkd: Backend[Array],
    qoi: str = "outer_radial_displacement",
    npts_r: int = 12,
    npts_theta: int = 12,
    r_inner: float = 1.0,
    r_outer: float = 2.0,
    E_mean: float = 1.0,
    poisson_ratio: float = 0.3,
    inner_pressure: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    weld_r_fraction: float = 0.25,
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth[Array]]:
    """Create a 2D hyperelastic pressurized cylinder benchmark for UQ.

    Maps KLE coefficients (standard normal) to a scalar QoI via a
    2D Neo-Hookean hyperelastic collocation PDE solve on a quarter-annulus.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    qoi : str
        Quantity of interest:
        - ``"outer_radial_displacement"``: average u_r on outer wall
        - ``"average_hoop_stress"``: average sigma_tt over weld zone
        - ``"strain_energy"``: total strain energy integral
    npts_r : int
        Number of Chebyshev points in radial direction.
    npts_theta : int
        Number of Chebyshev points in angular direction.
    r_inner : float
        Inner radius R_i.
    r_outer : float
        Outer radius R_o.
    E_mean : float
        Mean Young's modulus.
    poisson_ratio : float
        Fixed Poisson ratio.
    inner_pressure : float
        Internal pressure.
    num_kle_terms : int
        Number of KLE terms (= number of input parameters).
    sigma : float
        Standard deviation of the log-field for KLE.
    weld_r_fraction : float
        Fraction of wall thickness for weld zone (average_hoop_stress QoI).

    Returns
    -------
    BenchmarkWithPrior
        Benchmark with ``function()`` (forward model), ``prior()``
        (iid standard normal), and ``domain()`` ([-4, 4]^n).
    """
    # Mesh and basis
    transform = PolarTransform(
        (r_inner, r_outer),
        (0.0, math.pi / 2.0),
        bkd,
    )
    mesh = TransformedMesh2D(npts_r, npts_theta, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)

    # KLE field map
    field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms, sigma)

    # Create stress model for functionals (same Lame params as zoo factory)
    dmu_dE = 1.0 / (2.0 * (1.0 + poisson_ratio))
    dlam_dE = poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu_init = E_mean * dmu_dE
    lamda_init = E_mean * dlam_dE
    stress_model = NeoHookeanStress(lamda=lamda_init, mu=mu_init)

    # QoI functional
    functional = _make_hyperelastic_functional(
        bkd,
        qoi,
        stress_model,
        basis,
        mesh,
        transform,
        num_kle_terms,
        weld_r_fraction,
    )

    # Forward model via zoo factory
    fwd = create_hyperelastic_pressurized_cylinder_2d(
        bkd=bkd,
        npts_r=npts_r,
        npts_theta=npts_theta,
        r_inner=r_inner,
        r_outer=r_outer,
        E_mean=E_mean,
        poisson_ratio=poisson_ratio,
        inner_pressure=inner_pressure,
        field_map=field_map,
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

    name = f"pressurized_cylinder_2d_hyperelastic_{qoi}"
    description = (
        f"2D hyperelastic pressurized cylinder, QoI={qoi}, "
        f"npts_r={npts_r}, npts_theta={npts_theta}, "
        f"{num_kle_terms} KLE terms"
    )

    inner = BenchmarkWithPrior(
        _name=name,
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=description,
    )
    return PDEBenchmarkWrapper(inner, estimated_cost=4.0e-01)


@BenchmarkRegistry.register(
    "pressurized_cylinder_2d_hyperelastic",
    category="pde",
    description=(
        "2D hyperelastic pressurized cylinder with KLE-parameterized Young's modulus"
    ),
)
def _pressurized_cylinder_2d_hyperelastic_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return hyperelastic_pressurized_cylinder_2d(bkd)
