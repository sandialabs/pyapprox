"""Elasticity post-processing: stress and strain recovery.

Computes strain, stress, and von Mises stress at quadrature points from
displacement solutions on any mesh/element supported by
:class:`~pyapprox.pde.galerkin.basis.VectorLagrangeBasis` (2D quads and
triangles, 3D hexahedra and tetrahedra, degree 1 or 2). Supports
composite materials with per-element or per-quadrature Lame parameters.

Strain and stress use Voigt ordering with TENSOR (not engineering)
shear components:

- 2D: ``[e_xx, e_yy, e_xy]``
- 3D: ``[e_xx, e_yy, e_zz, e_xy, e_xz, e_yz]``

Stress is always returned with the six 3D Voigt components
``[s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]``: for two-dimensional strain
the out-of-plane components are populated according to the stated
``assumption`` ("plane_stress" or "plane_strain"), which previous
versions of this module left implicit (and inconsistent between the
Hooke's law and the von Mises formula).

Functions
---------
strain_from_displacement
    Strain in Voigt order at quadrature points.
stress_from_strain
    Cauchy stress (six Voigt components) via Hooke's law under an
    explicit plane_stress/plane_strain/3d assumption.
von_mises_stress
    Von Mises stress at quadrature points from a displacement solution.
integrate
    Domain integral of a quadrature-point field.
"""

from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    # only used in annotations; importing it would require skfem, which
    # must stay optional for this module to be importable without it
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )

HookeAssumption = Literal["plane_stress", "plane_strain", "3d"]


def strain_from_displacement(
    basis: "VectorLagrangeBasis[Array]",
    displacement: Array,
) -> NDArray[np.floating[Any]]:
    """Compute the strain tensor at quadrature points.

    Uses the skfem interpolation of the displacement gradient, so it
    works for every element type and degree the basis supports.

    Parameters
    ----------
    basis : VectorLagrangeBasis
        Vector finite element basis the displacement is defined on.
    displacement : Array
        Interleaved displacement DOF values. Shape: (ndofs,)

    Returns
    -------
    np.ndarray
        Voigt-ordered strain with tensor shear components.
        Shape: (3, nelems, nquad) in 2D, (6, nelems, nquad) in 3D.
    """
    ndim = basis.ncomponents()
    if ndim not in (2, 3):
        raise ValueError(f"strain recovery requires 2D or 3D, got {ndim}D")
    skfem_basis = basis.skfem_basis()
    u_np = basis.bkd().to_numpy(displacement)
    grad = np.asarray(skfem_basis.interpolate(u_np).grad)
    # grad[i, j] = du_i/dx_j at quadrature points, shape (nelems, nquad)
    if ndim == 2:
        rows = [
            grad[0, 0],
            grad[1, 1],
            0.5 * (grad[0, 1] + grad[1, 0]),
        ]
    else:
        rows = [
            grad[0, 0],
            grad[1, 1],
            grad[2, 2],
            0.5 * (grad[0, 1] + grad[1, 0]),
            0.5 * (grad[0, 2] + grad[2, 0]),
            0.5 * (grad[1, 2] + grad[2, 1]),
        ]
    return np.stack(rows)


def _broadcast_lame(
    param: Union[float, NDArray[np.floating[Any]]],
    shape: tuple[int, ...],
) -> NDArray[np.floating[Any]]:
    """Broadcast a scalar, per-element, or per-quadrature Lame field."""
    arr = np.asarray(param, dtype=np.float64)
    if arr.ndim == 1:
        # per-element values: broadcast across quadrature points
        arr = arr[:, None]
    return np.broadcast_to(arr, shape)


def stress_from_strain(
    strain: NDArray[np.floating[Any]],
    lam: Union[float, NDArray[np.floating[Any]]],
    mu: Union[float, NDArray[np.floating[Any]]],
    assumption: HookeAssumption,
) -> NDArray[np.floating[Any]]:
    """Compute Cauchy stress from strain via Hooke's law.

    Parameters
    ----------
    strain : np.ndarray
        Voigt strain from :func:`strain_from_displacement`.
        Shape: (3, ...) for 2D (requires a plane assumption) or
        (6, ...) for 3D (requires assumption="3d").
    lam : float or np.ndarray
        First Lame parameter of the 3D material: scalar, per-element
        (nelems,), or per-quadrature (nelems, nquad).
    mu : float or np.ndarray
        Shear modulus; same accepted shapes as ``lam``.
    assumption : {"plane_stress", "plane_strain", "3d"}
        Constitutive assumption. "plane_strain" (e_zz = 0) applies the
        3D law to the in-plane strain and yields s_zz = lam*tr(eps).
        "plane_stress" (s_zz = 0) uses the effective in-plane parameter
        lam_eff = 2*lam*mu/(lam + 2*mu).

    Returns
    -------
    np.ndarray
        The six 3D Voigt stress components
        [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]; out-of-plane components
        follow the stated assumption for 2D input.
        Shape: (6,) + strain.shape[1:]
    """
    nvoigt = strain.shape[0]
    if assumption == "3d":
        if nvoigt != 6:
            raise ValueError(
                f"assumption '3d' requires 6 strain components, got {nvoigt}"
            )
    elif assumption in ("plane_stress", "plane_strain"):
        if nvoigt != 3:
            raise ValueError(
                f"assumption '{assumption}' requires 3 strain components, "
                f"got {nvoigt}"
            )
    else:
        raise ValueError(
            "assumption must be 'plane_stress', 'plane_strain', or '3d', "
            f"got '{assumption}'"
        )

    batch_shape = strain.shape[1:]
    lam_b = _broadcast_lame(lam, batch_shape)
    mu_b = _broadcast_lame(mu, batch_shape)
    zero = np.zeros(batch_shape)

    if assumption == "3d":
        trace = strain[0] + strain[1] + strain[2]
        return np.stack(
            [
                lam_b * trace + 2.0 * mu_b * strain[0],
                lam_b * trace + 2.0 * mu_b * strain[1],
                lam_b * trace + 2.0 * mu_b * strain[2],
                2.0 * mu_b * strain[3],
                2.0 * mu_b * strain[4],
                2.0 * mu_b * strain[5],
            ]
        )

    trace2 = strain[0] + strain[1]
    if assumption == "plane_strain":
        # e_zz = 0: the 3D law applied to in-plane strain
        return np.stack(
            [
                lam_b * trace2 + 2.0 * mu_b * strain[0],
                lam_b * trace2 + 2.0 * mu_b * strain[1],
                lam_b * trace2,
                2.0 * mu_b * strain[2],
                zero,
                zero,
            ]
        )

    # plane stress (s_zz = 0): effective in-plane first Lame parameter
    lam_eff = 2.0 * lam_b * mu_b / (lam_b + 2.0 * mu_b)
    return np.stack(
        [
            lam_eff * trace2 + 2.0 * mu_b * strain[0],
            lam_eff * trace2 + 2.0 * mu_b * strain[1],
            zero,
            2.0 * mu_b * strain[2],
            zero,
            zero,
        ]
    )


def von_mises_stress(
    basis: "VectorLagrangeBasis[Array]",
    displacement: Array,
    lam: Union[float, NDArray[np.floating[Any]]],
    mu: Union[float, NDArray[np.floating[Any]]],
    assumption: HookeAssumption,
) -> NDArray[np.floating[Any]]:
    """Compute von Mises stress at quadrature points.

    Uses the full deviatoric stress, so it is correct for plane strain
    and 3D as well as plane stress (unlike the plane-stress-only
    formula this module previously hard-coded).

    Parameters
    ----------
    basis : VectorLagrangeBasis
        Vector finite element basis the displacement is defined on.
    displacement : Array
        Interleaved displacement DOF values. Shape: (ndofs,)
    lam, mu : float or np.ndarray
        Lame parameters (see :func:`stress_from_strain`).
    assumption : {"plane_stress", "plane_strain", "3d"}
        Constitutive assumption; must match the basis dimension.

    Returns
    -------
    np.ndarray
        Von Mises stress at quadrature points. Shape: (nelems, nquad)
    """
    strain = strain_from_displacement(basis, displacement)
    stress = stress_from_strain(strain, lam, mu, assumption)
    sxx, syy, szz, sxy, sxz, syz = stress
    result: NDArray[np.floating[Any]] = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + sxz**2 + syz**2)
    )
    return result


def integrate(
    basis: "VectorLagrangeBasis[Array]",
    field: NDArray[np.floating[Any]],
) -> float:
    """Integrate a quadrature-point field over the domain.

    Parameters
    ----------
    basis : VectorLagrangeBasis
        Basis whose quadrature rule the field is sampled on.
    field : np.ndarray
        Field values at quadrature points. Shape: (nelems, nquad)

    Returns
    -------
    float
        The domain integral (quadrature weights include the Jacobian).
    """
    dx = np.asarray(basis.skfem_basis().dx)
    if field.shape != dx.shape:
        raise ValueError(
            f"field shape {field.shape} does not match quadrature "
            f"shape {dx.shape}"
        )
    return float(np.einsum("eq,eq->", field, dx))
