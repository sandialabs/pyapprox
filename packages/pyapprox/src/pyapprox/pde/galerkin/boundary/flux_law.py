r"""Pointwise normal-flux laws for Galerkin natural boundary conditions.

A flux law evaluates the quantity a natural boundary condition prescribes,

.. math::

    g = q(x, u, \nabla u, t) \cdot n,

from the state, its gradient and the outward normal. ``g`` carries the sign
it has in the Galerkin residual :math:`F`: a Neumann term with data ``g``
adds :math:`\int_\Gamma g\, v` to :math:`F`. For diffusion that is
:math:`\kappa \nabla u \cdot n`; for elasticity the traction
:math:`\sigma n`.

A law takes pointwise NumPy arrays, as the skfem assembly seam does, and
knows nothing of meshes or bases. Collocation does not use it: its natural
conditions apply normal operators to the discrete state, on backend arrays.
A physics builds the
law its weak form integrates by parts (``FluxLawProviderProtocol``), so
manufactured boundary data computed from that law always matches what the
physics enforces. A manufactured test passes the exact :math:`u` and
:math:`\nabla u`, and builds the physics from exact coefficients, so the
data carries no discretization error.

Shapes, for ``ncomp`` solution components in ``ndim`` dimensions at
``npts`` points:

- ``coords``: ``(ndim, npts)``
- ``u``: ``(ncomp, npts)``
- ``grad_u``: ``(ncomp, ndim, npts)``, with ``grad_u[i, j]`` the derivative
  of component ``i`` along coordinate ``j``
- ``normal``: ``(ndim, npts)``, unit outward normals
- result: ``(ncomp, npts)``
"""

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.constitutive.coefficient_functions import (
    DiffusionFunctionProtocol,
    VelocityFunctionProtocol,
)
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.constitutive.protocols import StressModelProtocol
from pyapprox.util.backends.numpy import NumpyBkd

_Pts = NDArray[np.floating[Any]]


@runtime_checkable
class NormalFluxLawProtocol(Protocol):
    """A pointwise law for the natural boundary quantity ``q . n``."""

    def normal_flux(
        self,
        coords: _Pts,
        u: _Pts,
        grad_u: _Pts,
        normal: _Pts,
        time: float,
    ) -> _Pts:
        """Evaluate ``q . n``. Returns shape ``(ncomp, npts)``."""
        ...

    def is_time_dependent(self) -> bool:
        """Whether any coefficient of the law varies with time."""
        ...


@runtime_checkable
class FluxLawProviderProtocol(Protocol):
    """A physics that states the flux its weak form integrates by parts.

    Optional: a physics used only with essential boundary conditions need
    not provide one.
    """

    def flux_law(self) -> NormalFluxLawProtocol:
        """Return the law built from the physics' current coefficients."""
        ...


class DiffusiveFlux:
    r"""Diffusive flux :math:`a(x, t)\,\kappa(u)\,\nabla u \cdot n`.

    Parameters
    ----------
    diffusivity : DiffusionFunctionProtocol
        The spatial (and possibly temporal) diffusivity :math:`a(x, t)`.
    state_factor : Callable, optional
        A state-dependent factor :math:`\kappa(u)`, evaluated pointwise.
        Omitted, the flux is linear in :math:`\nabla u`.
    """

    def __init__(
        self,
        diffusivity: DiffusionFunctionProtocol,
        state_factor: Optional[Callable[[_Pts], _Pts]] = None,
    ) -> None:
        if not isinstance(diffusivity, DiffusionFunctionProtocol):
            raise TypeError(
                "diffusivity must satisfy DiffusionFunctionProtocol, got "
                f"{type(diffusivity).__name__}"
            )
        self._diffusivity = diffusivity
        self._state_factor = state_factor

    def normal_flux(
        self,
        coords: _Pts,
        u: _Pts,
        grad_u: _Pts,
        normal: _Pts,
        time: float,
    ) -> _Pts:
        coefficient = self._diffusivity.values(coords, time)
        if self._state_factor is not None:
            coefficient = coefficient * self._state_factor(u[0])
        ret: _Pts = (coefficient * np.sum(grad_u[0] * normal, axis=0))[
            np.newaxis, :
        ]
        return ret

    def is_time_dependent(self) -> bool:
        return self._diffusivity.is_time_dependent()

    def __repr__(self) -> str:
        return (
            f"DiffusiveFlux({self._diffusivity!r}, "
            f"state_factor={self._state_factor!r})"
        )


class AdvectiveFlux:
    r"""Advective boundary term :math:`-(v \cdot n)\, u`.

    It is the boundary term of integrating :math:`-\nabla\cdot(v u)` by
    parts, so it belongs to the conservative advection form only; the
    non-conservative form :math:`-v\cdot\nabla u` has no boundary term.

    Parameters
    ----------
    velocity : VelocityFunctionProtocol
        The advecting velocity :math:`v(x, t)`.
    """

    def __init__(self, velocity: VelocityFunctionProtocol) -> None:
        if not isinstance(velocity, VelocityFunctionProtocol):
            raise TypeError(
                "velocity must satisfy VelocityFunctionProtocol, got "
                f"{type(velocity).__name__}"
            )
        self._velocity = velocity

    def normal_flux(
        self,
        coords: _Pts,
        u: _Pts,
        grad_u: _Pts,
        normal: _Pts,
        time: float,
    ) -> _Pts:
        velocity = self._velocity.values(coords, time)
        ret: _Pts = -(np.sum(velocity * normal, axis=0) * u[0])[np.newaxis, :]
        return ret

    def is_time_dependent(self) -> bool:
        return self._velocity.is_time_dependent()

    def __repr__(self) -> str:
        return f"AdvectiveFlux({self._velocity!r})"


class SumFlux:
    """The sum of flux laws, composed the way the residual sums its terms.

    Parameters
    ----------
    terms : Sequence[NormalFluxLawProtocol]
        The laws to sum. At least one.
    """

    def __init__(self, terms: Sequence[NormalFluxLawProtocol]) -> None:
        if len(terms) == 0:
            raise ValueError("SumFlux needs at least one term")
        for term in terms:
            if not isinstance(term, NormalFluxLawProtocol):
                raise TypeError(
                    "every term must satisfy NormalFluxLawProtocol, got "
                    f"{type(term).__name__}"
                )
        self._terms: List[NormalFluxLawProtocol] = list(terms)

    def normal_flux(
        self,
        coords: _Pts,
        u: _Pts,
        grad_u: _Pts,
        normal: _Pts,
        time: float,
    ) -> _Pts:
        total = self._terms[0].normal_flux(coords, u, grad_u, normal, time)
        for term in self._terms[1:]:
            total = total + term.normal_flux(coords, u, grad_u, normal, time)
        return total

    def is_time_dependent(self) -> bool:
        return any(term.is_time_dependent() for term in self._terms)

    def __repr__(self) -> str:
        return f"SumFlux({self._terms!r})"


class _ConstantField:
    """A spatially constant field, evaluated pointwise."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def __call__(self, coords: _Pts) -> _Pts:
        return np.full(coords.shape[1], self._value)

    def __repr__(self) -> str:
        return repr(self._value)


PointwiseField = Callable[[_Pts], _Pts]
"""A material field: coordinates ``(ndim, npts)`` to values ``(npts,)``."""


def _as_field(value: Union[float, PointwiseField]) -> PointwiseField:
    """A constant becomes a constant field; a field passes through."""
    if callable(value):
        return value
    return _ConstantField(value)


class LinearElasticTraction:
    r"""Small-strain traction :math:`\sigma n`.

    :math:`\sigma = \lambda\,\mathrm{tr}(\varepsilon)\,I + 2\mu\,\varepsilon`
    with :math:`\varepsilon = (\nabla u + \nabla u^T)/2`. In one dimension
    this is :math:`(\lambda + 2\mu)\,u'\,n`.

    Parameters
    ----------
    lamda : float or PointwiseField
        Lame's first parameter, constant or a field of the coordinates.
    mu : float or PointwiseField
        Shear modulus, constant or a field of the coordinates.
    """

    def __init__(
        self,
        lamda: Union[float, PointwiseField],
        mu: Union[float, PointwiseField],
    ) -> None:
        self._lamda = _as_field(lamda)
        self._mu = _as_field(mu)

    def normal_flux(
        self,
        coords: _Pts,
        u: _Pts,
        grad_u: _Pts,
        normal: _Pts,
        time: float,
    ) -> _Pts:
        strain = 0.5 * (grad_u + np.swapaxes(grad_u, 0, 1))
        trace = np.trace(strain, axis1=0, axis2=1)
        stress = 2.0 * self._mu(coords) * strain
        lamda_trace = self._lamda(coords) * trace
        ndim = grad_u.shape[1]
        for ii in range(ndim):
            stress[ii, ii] = stress[ii, ii] + lamda_trace
        ret: _Pts = np.einsum("ijp,jp->ip", stress, normal)
        return ret

    def is_time_dependent(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"LinearElasticTraction(lamda={self._lamda!r}, mu={self._mu!r})"


def _pk1_traction(
    stress_model: StressModelProtocol[NDArray[Any]],
    grad_u: _Pts,
    normal: _Pts,
) -> _Pts:
    r"""The nominal traction :math:`P(I + \nabla u)\,N`, ``(ndim, npts)``."""
    bkd = NumpyBkd()
    ndim = grad_u.shape[1]
    F = grad_u + np.eye(ndim)[:, :, np.newaxis]
    if ndim == 1:
        stress = np.asarray(stress_model.compute_stress_1d(F[0, 0], bkd))[
            np.newaxis, np.newaxis, :
        ]
    elif ndim == 2:
        P11, P12, P21, P22 = stress_model.compute_stress_2d(
            F[0, 0], F[0, 1], F[1, 0], F[1, 1], bkd
        )
        stress = np.array([[P11, P12], [P21, P22]])
    elif ndim == 3:
        P3 = stress_model.compute_stress_3d(
            tuple(tuple(F[ii, jj] for jj in range(3)) for ii in range(3)),
            bkd,
        )
        stress = np.array([[np.asarray(Pij) for Pij in row] for row in P3])
    else:
        raise ValueError(f"unsupported dimension {ndim}")
    ret: _Pts = np.einsum("ijp,jp->ip", stress, normal)
    return ret


class PK1Traction:
    r"""Nominal traction :math:`P(F)\, N` of a hyperelastic stress model.

    :math:`F = I + \nabla u` is the deformation gradient and :math:`P` the
    first Piola-Kirchhoff stress, so ``normal`` is the reference-
    configuration normal :math:`N`.

    Parameters
    ----------
    stress_model : StressModelProtocol
        Pointwise PK1 stress, evaluated with NumPy arrays. Its material
        parameters are uniform; see ``NeoHookeanTraction`` for fields.
    """

    def __init__(self, stress_model: StressModelProtocol[NDArray[Any]]) -> None:
        if not isinstance(stress_model, StressModelProtocol):
            raise TypeError(
                "stress_model must satisfy StressModelProtocol, got "
                f"{type(stress_model).__name__}"
            )
        self._stress_model = stress_model

    def normal_flux(
        self,
        coords: _Pts,
        u: _Pts,
        grad_u: _Pts,
        normal: _Pts,
        time: float,
    ) -> _Pts:
        return _pk1_traction(self._stress_model, grad_u, normal)

    def is_time_dependent(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"PK1Traction({self._stress_model!r})"


class NeoHookeanTraction:
    r"""Neo-Hookean nominal traction with Lame fields.

    :math:`P = \mu F + (\lambda \ln J - \mu) F^{-T}`, with :math:`\lambda`
    and :math:`\mu` evaluated at each point, as in a multi-material body.

    Parameters
    ----------
    lamda : float or PointwiseField
        Lame's first parameter, constant or a field of the coordinates.
    mu : float or PointwiseField
        Shear modulus, constant or a field of the coordinates.
    """

    def __init__(
        self,
        lamda: Union[float, PointwiseField],
        mu: Union[float, PointwiseField],
    ) -> None:
        self._lamda = _as_field(lamda)
        self._mu = _as_field(mu)

    def normal_flux(
        self,
        coords: _Pts,
        u: _Pts,
        grad_u: _Pts,
        normal: _Pts,
        time: float,
    ) -> _Pts:
        model: NeoHookeanStress[NDArray[Any]] = NeoHookeanStress(
            self._lamda(coords), self._mu(coords)
        )
        return _pk1_traction(model, grad_u, normal)

    def is_time_dependent(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"NeoHookeanTraction(lamda={self._lamda!r}, mu={self._mu!r})"
