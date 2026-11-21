"""Pointwise densities of the 1D displacement gradient and the
material: the argument is the strain du/dx or the deformation gradient
1 + du/dx, and the material stacks nodal fields. Each density gives its
value and its pointwise derivatives in the argument and the material.
"""

# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class GradientDensity1DProtocol(Protocol, Generic[Array]):
    """A pointwise density of the displacement gradient and the
    material, with its derivatives.

    ``material`` stacks ``nfields()`` nodal fields, field-major, as the
    material map gives them.
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
        mu, lamda = _split_lame(material)
        ln_F = self._bkd.log(arg)
        return 0.5 * mu * (arg**2 - 1.0 - 2.0 * ln_F) + 0.5 * lamda * ln_F**2

    def arg_derivative(self, arg: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material)
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
        mu, lamda = _split_lame(material)
        return mu * (arg - 1.0 / arg) + lamda * self._bkd.log(arg) / arg

    def arg_derivative(self, arg: Array, material: Array) -> Array:
        mu, lamda = _split_lame(material)
        inv_F2 = 1.0 / arg**2
        return mu * (1.0 + inv_F2) + lamda * (1.0 - self._bkd.log(arg)) * inv_F2

    def material_derivative(self, arg: Array, material: Array) -> Array:
        return self._bkd.concatenate([arg - 1.0 / arg, self._bkd.log(arg) / arg])


def _split_lame(material: Array) -> Tuple[Array, Array]:
    """``(mu, lamda)`` from the stacked ``[mu; lamda]``."""
    npts = material.shape[0] // 2
    return material[:npts], material[npts:]
