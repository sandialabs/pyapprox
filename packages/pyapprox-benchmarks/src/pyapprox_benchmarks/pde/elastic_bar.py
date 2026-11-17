"""1D elastic bar forward UQ problems.

Wraps the zoo factories (linear and hyperelastic 1D bar) into
ForwardUQProblem instances with standard normal KLE priors and
configurable QoI functionals.
"""

from __future__ import annotations

from typing import Union

from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.functionals.point_evaluation import (
    PointEvaluationFunctional,
)
from pyapprox.pde.collocation.functionals.strain_energy_1d import (
    LinearStressDensity1D,
    MaterialMapProtocol,
    NeoHookeanStressDensity1D,
    StrainEnergyFunctional1D,
    create_linear_strain_energy_1d,
    create_neo_hookean_strain_energy_1d,
)
from pyapprox.pde.collocation.functionals.subdomain_integral import (
    SubdomainIntegralFunctional,
)
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.lame import FixedPoissonRatioLameMap
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.parameterizations.fields import ConstantInTimeField
from pyapprox.pde.zoo.elastic_bar_1d import (
    create_linear_elastic_bar_1d,
)
from pyapprox.pde.zoo.hyperelastic_bar_1d import (
    create_hyperelastic_bar_1d,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend
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


def _make_functional(
    basis: ChebyshevBasis1D[Array],
    nparams: int,
    bkd: Backend[Array],
    qoi: str,
    constitutive: str,
    length: float,
    material_map: MaterialMapProtocol[Array],
) -> Union[
    PointEvaluationFunctional[Array],
    SubdomainIntegralFunctional[Array],
    StrainEnergyFunctional1D[Array],
]:
    """Create QoI functional from string identifier.

    The stress and energy QoIs read the material ``material_map`` gives
    at the forward model's parameters: E for the linear bar,
    ``[mu; lamda]`` for the hyperelastic one.
    """
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
        # (1/L) integral sigma dx
        stress: Union[LinearStressDensity1D[Array], NeoHookeanStressDensity1D[Array]]
        stress = (
            LinearStressDensity1D(bkd)
            if constitutive == "linear"
            else NeoHookeanStressDensity1D(bkd)
        )
        return StrainEnergyFunctional1D(
            basis, stress, material_map, bkd, scale=1.0 / length
        )

    if qoi == "strain_energy":
        if constitutive == "linear":
            return create_linear_strain_energy_1d(basis, material_map, bkd)
        return create_neo_hookean_strain_energy_1d(basis, material_map, bkd)

    raise ValueError(
        f"Unknown qoi: {qoi!r}. Must be one of: "
        "'tip_displacement', 'average_displacement', "
        "'average_stress', 'strain_energy'"
    )


def build_elastic_bar_1d(
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
) -> ForwardUQProblem[FunctionProtocol[Array], Array]:
    """Create a 1D elastic bar forward UQ problem.

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
        The modulus the physics is built with; the parameterization
        replaces it with the KLE field, so at KLE coefficients = 0 the
        modulus is E = exp(0) = 1 regardless.
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
    ForwardUQProblem
        Problem with ``function()`` (the forward model) and ``prior()``
        (iid standard normal).
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

    # QoI functional, reading the material from the parameters with the
    # map the zoo's parameterization builds from the same field map
    material_map: MaterialMapProtocol[Array] = (
        field_map
        if constitutive == "linear"
        else FixedPoissonRatioLameMap(field_map, poisson_ratio, npts, bkd)
    )
    functional = _make_functional(
        basis,
        num_kle_terms,
        bkd,
        qoi,
        constitutive,
        length,
        material_map,
    )

    # Forward model via zoo factory
    forcing = ConstantInTimeField(bkd.ones((npts,)))
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

    name = f"elastic_bar_1d_{constitutive}_{qoi}"
    description = (
        f"1D {constitutive} elastic bar, QoI={qoi}, "
        f"npts={npts}, {num_kle_terms} KLE terms"
    )

    return ForwardUQProblem(
        name=name,
        function=fwd,
        prior=prior,
        description=description,
    )
