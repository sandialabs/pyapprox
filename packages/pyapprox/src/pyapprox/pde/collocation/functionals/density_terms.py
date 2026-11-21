r"""Integrals of nodal densities of a collocation state, as state terms.

A strain energy, an average stress or any integral of a pointwise density
:math:`\psi(u, m)` of the state and the material is

.. math::

    Q(u;\, p) = \sum_i w_i\, \psi_i(u, m(p)),

for quadrature weights :math:`w` at the nodes (scaled by the caller, e.g.
by the inverse area for an average) and the material :math:`m(p)` that
the forward model's parameter map gives. :class:`IntegratedDensity` is
that term, satisfying
:class:`~pyapprox.implicitfunction.functional_terms.StateTermProtocol`
with first derivatives

.. math::

    Q_u = w^\top \psi_u, \qquad
    Q_p = (w \otimes \psi_m)^\top m_p,

so ``StateFunctional`` and the trajectory terms take it as they take any
term. It reads the parameters currently set through a
:class:`CurrentParametersProtocol`, the same object the forward model's
residual reads, so the material it evaluates is the one the state was
solved with.

The densities are adapters, each satisfying :class:`NodalDensityProtocol`:
:class:`GradientDensity1D` over a 1D gradient density
(:class:`GradientDensity1DProtocol`: the linear and neo-Hookean energy
and stress densities here), and :class:`HoopStressDensity` and
:class:`StrainEnergyDensity2D` over a 2D stress post-processor
(:class:`HoopStressSourceProtocol`, :class:`StrainEnergySourceProtocol`).
"""

from typing import Generic, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.implicitfunction.functional_terms import (
    BackendScalarGradient,
    LinearizationProtocol,
    StateTermDerivatives,
)
from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.util.backends.protocols import Array, Array_co, Backend


@runtime_checkable
class CurrentParametersProtocol(Protocol[Array_co]):
    """The parameters currently set, as the forward model's residual reads
    them; ``ParameterState`` satisfies it."""

    def current(self) -> Array_co:
        """The parameters. Shape: (nparams,)"""
        ...


@runtime_checkable
class NodalMaterialMapProtocol(Protocol[Array]):
    """The parameters-to-material map: ``nfields`` stacked nodal fields
    (``nfields * npts`` values, field-major) and their Jacobian. A Young's
    modulus field map and ``FixedPoissonRatioLameMap`` satisfy it."""

    def nvars(self) -> int:
        """The number of parameters."""
        ...

    def __call__(self, params_1d: Array) -> Array:
        """The material. Shape: (nfields*npts,)"""
        ...

    def jacobian(self, params_1d: Array) -> Array:
        """Shape: (nfields*npts, nvars)"""
        ...


@runtime_checkable
class NodalDensityProtocol(Protocol[Array]):
    r"""A pointwise density :math:`\psi(u, m)` at the nodes, with its state
    Jacobian and its pointwise material derivative."""

    def npts(self) -> int:
        """The number of nodes."""
        ...

    def nfields(self) -> int:
        """The number of material fields the density reads."""
        ...

    def value(self, state: Array, material: Array) -> Array:
        r""":math:`\psi` at the nodes, for a state of shape (nstates,).
        Shape: (npts,)"""
        ...

    def state_jacobian(self, state: Array, material: Array) -> Array:
        r""":math:`\partial \psi_i/\partial u_j`. Shape: (npts, nstates)"""
        ...

    def material_derivative(self, state: Array, material: Array) -> Array:
        r""":math:`\partial \psi_i/\partial m_{k,i}` for each field
        :math:`k`, stacked as the material is. Shape: (nfields*npts,)"""
        ...


@runtime_checkable
class GradientDensity1DProtocol(Protocol, Generic[Array]):
    """A pointwise density of the 1D displacement gradient and the
    material, with its derivatives.

    The argument is the strain du/dx or the deformation gradient
    1 + du/dx. ``material`` stacks ``nfields()`` nodal fields,
    field-major, as the material map gives them.
    """

    def nfields(self) -> int:
        """The number of material fields the density reads."""
        ...

    def uses_deformation_gradient(self) -> bool:
        """True for ``arg = 1 + du/dx``, False for ``arg = du/dx``."""
        ...

    def value(self, arg: Array, material: Array) -> Array:
        """The density at the nodes. Shape: (npts,)"""
        ...

    def arg_derivative(self, arg: Array, material: Array) -> Array:
        """The pointwise derivative in ``arg``. Shape: (npts,)"""
        ...

    def material_derivative(self, arg: Array, material: Array) -> Array:
        """The pointwise derivative in each field, stacked as
        ``material`` is. Shape: (nfields*npts,)"""
        ...


class LinearStrainEnergyDensity1D(Generic[Array]):
    """:math:`\\psi = \\tfrac12 E \\varepsilon^2` with material ``[E]``."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nfields(self) -> int:
        return 1

    def uses_deformation_gradient(self) -> bool:
        return False

    def value(self, arg: Array, material: Array) -> Array:
        return 0.5 * material * arg**2

    def arg_derivative(self, arg: Array, material: Array) -> Array:
        return material * arg

    def material_derivative(self, arg: Array, material: Array) -> Array:
        return 0.5 * arg**2


class LinearStressDensity1D(Generic[Array]):
    """:math:`\\sigma = E \\varepsilon` with material ``[E]``."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nfields(self) -> int:
        return 1

    def uses_deformation_gradient(self) -> bool:
        return False

    def value(self, arg: Array, material: Array) -> Array:
        return material * arg

    def arg_derivative(self, arg: Array, material: Array) -> Array:
        return material

    def material_derivative(self, arg: Array, material: Array) -> Array:
        return arg


class NeoHookeanStrainEnergyDensity1D(Generic[Array]):
    """The 1D neo-Hookean stored energy
    :math:`\\psi = \\tfrac{\\mu}{2}(F^2 - 1 - 2\\ln F) +
    \\tfrac{\\lambda}{2}(\\ln F)^2` with material ``[mu; lamda]``.
    Its derivative in F is the first Piola-Kirchhoff stress."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nfields(self) -> int:
        return 2

    def uses_deformation_gradient(self) -> bool:
        return True

    def value(self, arg: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, material.shape[0] // 2)
        ln_F = self._bkd.log(arg)
        return 0.5 * mu * (arg**2 - 1.0 - 2.0 * ln_F) + 0.5 * lamda * ln_F**2

    def arg_derivative(self, arg: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, material.shape[0] // 2)
        return mu * (arg - 1.0 / arg) + lamda * self._bkd.log(arg) / arg

    def material_derivative(self, arg: Array, material: Array) -> Array:
        ln_F = self._bkd.log(arg)
        return self._bkd.concatenate([0.5 * (arg**2 - 1.0 - 2.0 * ln_F), 0.5 * ln_F**2])


class NeoHookeanStressDensity1D(Generic[Array]):
    """The 1D neo-Hookean first Piola-Kirchhoff stress
    :math:`P = \\mu (F - 1/F) + \\lambda \\ln F / F` with material
    ``[mu; lamda]``."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nfields(self) -> int:
        return 2

    def uses_deformation_gradient(self) -> bool:
        return True

    def value(self, arg: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, material.shape[0] // 2)
        return mu * (arg - 1.0 / arg) + lamda * self._bkd.log(arg) / arg

    def arg_derivative(self, arg: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, material.shape[0] // 2)
        inv_F2 = 1.0 / arg**2
        return mu * (1.0 + inv_F2) + lamda * (1.0 - self._bkd.log(arg)) * inv_F2

    def material_derivative(self, arg: Array, material: Array) -> Array:
        return self._bkd.concatenate([arg - 1.0 / arg, self._bkd.log(arg) / arg])


@runtime_checkable
class HoopStressSourceProtocol(Protocol, Generic[Array]):
    """What the hoop-stress density reads: the hoop stress for given
    Lame parameters, its state Jacobian and its pointwise Lame
    derivatives. Both 2D stress post-processors satisfy it."""

    def npts(self) -> int: ...

    def hoop_stress(self, state: Array, mu: Array, lamda: Array) -> Array: ...

    def hoop_stress_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array: ...

    def hoop_stress_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]: ...


@runtime_checkable
class StrainEnergySourceProtocol(Protocol, Generic[Array]):
    """What the strain-energy density reads, as for
    :class:`HoopStressSourceProtocol`."""

    def npts(self) -> int: ...

    def strain_energy_density(self, state: Array, mu: Array, lamda: Array) -> Array: ...

    def strain_energy_density_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array: ...

    def strain_energy_density_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]: ...


class IntegratedDensity(Generic[Array]):
    r""":math:`Q(u;\, p) = \sum_i w_i \psi_i(u, m(p))`, a one-state term of
    one entry with first derivatives.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    density : NodalDensityProtocol
        :math:`\psi`.
    material_map : NodalMaterialMapProtocol
        :math:`m`, the forward model's parameters-to-material map.
    parameters : CurrentParametersProtocol
        The parameters currently set, shared with the forward model.
    weights : Array
        :math:`w`, shape ``(npts,)``.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If ``weights`` is not of shape ``(npts,)``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        density: NodalDensityProtocol[Array],
        material_map: NodalMaterialMapProtocol[Array],
        parameters: CurrentParametersProtocol[Array],
        weights: Array,
    ) -> None:
        for name, value, protocol in (
            ("density", density, NodalDensityProtocol),
            ("material_map", material_map, NodalMaterialMapProtocol),
            ("parameters", parameters, CurrentParametersProtocol),
        ):
            if not isinstance(value, protocol):
                raise TypeError(
                    f"{name} must satisfy {protocol.__name__}, got "
                    f"{type(value).__name__}"
                )
        npts = density.npts()
        if tuple(weights.shape) != (npts,):
            raise ValueError(
                f"weights must have shape ({npts},), got {tuple(weights.shape)}"
            )
        self._bkd, self._density, self._map = bkd, density, material_map
        self._parameters, self._weights = parameters, weights
        self._derivatives: StateTermDerivatives[Array, Array] = (
            StateTermDerivatives.first_order(self._wrt_state, self._wrt_param)
        )

    def nqoi(self) -> int:
        return 1

    def _material(self) -> Array:
        material = self._map(self._parameters.current())
        expected = self._density.nfields() * self._density.npts()
        if tuple(material.shape) != (expected,):
            raise ValueError(
                f"the material map must give nfields*npts = {expected} values, "
                f"got shape {tuple(material.shape)}"
            )
        return material

    def value(self, state: Array) -> Array:
        """:math:`Q`, shape (1, 1), for a state of shape (nstates, 1)."""
        bkd = self._bkd
        psi = self._density.value(state[:, 0], self._material())
        return bkd.reshape(bkd.sum(self._weights * psi), (1, 1))

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return self._derivatives

    def _wrt_state(self, state: Array) -> LinearizationProtocol[Array, Array]:
        jacobian = self._density.state_jacobian(state[:, 0], self._material())
        gradient = self._bkd.reshape(self._weights @ jacobian, (-1, 1))
        return BackendScalarGradient(gradient, self._bkd)

    def _wrt_param(self, state: Array) -> Optional[LinearizationProtocol[Array, Array]]:
        bkd = self._bkd
        current = self._parameters.current()
        derivative = self._density.material_derivative(state[:, 0], self._material())
        weights = bkd.concatenate([self._weights] * self._density.nfields())
        gradient = bkd.reshape(
            (weights * derivative) @ self._map.jacobian(current), (-1, 1)
        )
        return BackendScalarGradient(gradient, bkd)


class GradientDensity1D(Generic[Array]):
    r"""A density of the 1D displacement gradient as a nodal density:
    :math:`\psi(a(u), m)` with :math:`a = D_x u` or :math:`1 + D_x u`, as
    ``density`` declares.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    density : GradientDensity1DProtocol
        The pointwise density of the gradient.

    Raises
    ------
    TypeError
        If ``density`` does not satisfy ``GradientDensity1DProtocol``.
    """

    def __init__(
        self, basis: ChebyshevBasis1D[Array], density: GradientDensity1DProtocol[Array]
    ) -> None:
        if not isinstance(density, GradientDensity1DProtocol):
            raise TypeError(
                "density must satisfy GradientDensity1DProtocol, got "
                f"{type(density).__name__}"
            )
        self._density = density
        self._dx = basis.derivative_matrix(1, 0)

    def npts(self) -> int:
        return int(self._dx.shape[0])

    def nfields(self) -> int:
        return self._density.nfields()

    def _arg(self, state: Array) -> Array:
        gradient = self._dx @ state
        if self._density.uses_deformation_gradient():
            return 1.0 + gradient
        return gradient

    def value(self, state: Array, material: Array) -> Array:
        return self._density.value(self._arg(state), material)

    def state_jacobian(self, state: Array, material: Array) -> Array:
        derivative = self._density.arg_derivative(self._arg(state), material)
        return derivative[:, None] * self._dx

    def material_derivative(self, state: Array, material: Array) -> Array:
        return self._density.material_derivative(self._arg(state), material)


def _split_lame(material: Array, npts: int) -> Tuple[Array, Array]:
    """``(mu, lamda)`` from the stacked ``[mu; lamda]``."""
    return material[:npts], material[npts:]


class HoopStressDensity(Generic[Array]):
    r"""The hoop stress :math:`\sigma_{tt}` of a 2D stress post-processor
    as a nodal density, with material ``[mu; lamda]``.

    Raises
    ------
    TypeError
        If ``source`` does not satisfy ``HoopStressSourceProtocol``.
    """

    def __init__(
        self, bkd: Backend[Array], source: HoopStressSourceProtocol[Array]
    ) -> None:
        if not isinstance(source, HoopStressSourceProtocol):
            raise TypeError(
                "source must satisfy HoopStressSourceProtocol, got "
                f"{type(source).__name__}"
            )
        self._bkd, self._source = bkd, source

    def npts(self) -> int:
        return self._source.npts()

    def nfields(self) -> int:
        return 2

    def value(self, state: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, self.npts())
        return self._source.hoop_stress(state, mu, lamda)

    def state_jacobian(self, state: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, self.npts())
        return self._source.hoop_stress_state_jacobian(state, mu, lamda)

    def material_derivative(self, state: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, self.npts())
        dmu, dlamda = self._source.hoop_stress_lame_jacobian(state, mu, lamda)
        return self._bkd.concatenate([dmu, dlamda])


class StrainEnergyDensity2D(Generic[Array]):
    r"""The strain energy density :math:`\psi` of a 2D stress
    post-processor as a nodal density, with material ``[mu; lamda]``.

    Raises
    ------
    TypeError
        If ``source`` does not satisfy ``StrainEnergySourceProtocol``.
    """

    def __init__(
        self, bkd: Backend[Array], source: StrainEnergySourceProtocol[Array]
    ) -> None:
        if not isinstance(source, StrainEnergySourceProtocol):
            raise TypeError(
                "source must satisfy StrainEnergySourceProtocol, got "
                f"{type(source).__name__}"
            )
        self._bkd, self._source = bkd, source

    def npts(self) -> int:
        return self._source.npts()

    def nfields(self) -> int:
        return 2

    def value(self, state: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, self.npts())
        return self._source.strain_energy_density(state, mu, lamda)

    def state_jacobian(self, state: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, self.npts())
        return self._source.strain_energy_density_state_jacobian(state, mu, lamda)

    def material_derivative(self, state: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material, self.npts())
        dmu, dlamda = self._source.strain_energy_density_lame_jacobian(state, mu, lamda)
        return self._bkd.concatenate([dmu, dlamda])


def gradient_density_term_1d(
    basis: ChebyshevBasis1D[Array],
    density: GradientDensity1DProtocol[Array],
    material_map: NodalMaterialMapProtocol[Array],
    parameters: CurrentParametersProtocol[Array],
    bkd: Backend[Array],
    bounds: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
) -> IntegratedDensity[Array]:
    r""":math:`s \int_a^b \psi(a(u), m(p))\, dx` by the collocation
    quadrature: the strain energy for a strain-energy density, an average
    stress for a stress density and :math:`s = 1/L`.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    density : GradientDensity1DProtocol
        :math:`\psi`.
    material_map : NodalMaterialMapProtocol
        The forward model's parameters-to-material map.
    parameters : CurrentParametersProtocol
        The parameters currently set, shared with the forward model.
    bkd : Backend
        Computational backend.
    bounds : tuple of float, optional
        :math:`(a, b)` in physical coordinates, :math:`a < b`; default the
        whole domain.
    scale : float, default 1
        :math:`s`.

    Raises
    ------
    ValueError
        If ``bounds`` are not increasing.
    """
    quadrature = CollocationQuadrature1D(basis, bkd)
    if bounds is None:
        weights = quadrature.full_domain_weights()
    else:
        lower, upper = bounds
        if not lower < upper:
            raise ValueError(f"bounds must be increasing, got {bounds}")
        weights = quadrature.weights(lower, upper)
    return IntegratedDensity(
        bkd,
        GradientDensity1D(basis, density),
        material_map,
        parameters,
        scale * weights,
    )
