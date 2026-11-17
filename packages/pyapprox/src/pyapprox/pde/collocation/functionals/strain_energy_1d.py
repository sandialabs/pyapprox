"""Integrals of pointwise densities of the 1D displacement gradient.

Computes Q(u, p) = s * integral psi(arg(u), m(p)) dx, where arg is the
strain du/dx or the deformation gradient 1 + du/dx, and m(p) the
material at the nodes that a map of the parameters gives. The material
map is the one the forward model's parameterization uses, so Q reads
the material the state was solved with. The derivative matrix Dx maps
the state to its gradient, so dQ/du = s (w * dpsi/darg) @ Dx, and
dQ/dp = s (w * dpsi/dm) @ dm/dp.
"""

# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module


from typing import Generic, Literal, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)
from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class MaterialMapProtocol(Protocol, Generic[Array]):
    """The parameters-to-material map: the stacked material fields at
    the nodes (``nfields * npts`` values, field-major) and their
    Jacobian. A Young's modulus field map and ``FixedPoissonRatioLameMap``
    satisfy it."""

    def nvars(self) -> int:
        """The number of parameters."""
        ...

    def __call__(self, params_1d: Array) -> Array:
        """Shape: (nfields*npts,)"""
        ...

    def jacobian(self, params_1d: Array) -> Array:
        """Shape: (nfields*npts, nvars)"""
        ...


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
        return self._bkd.concatenate(
            [0.5 * (arg**2 - 1.0 - 2.0 * ln_F), 0.5 * ln_F**2]
        )


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
        return self._bkd.concatenate(
            [arg - 1.0 / arg, self._bkd.log(arg) / arg]
        )


def _split_lame(material: Array) -> Tuple[Array, Array]:
    """``(mu, lamda)`` from the stacked ``[mu; lamda]``."""
    npts = material.shape[0] // 2
    return material[:npts], material[npts:]


class StrainEnergyFunctional1D(Generic[Array]):
    """Integral of a pointwise density of the 1D displacement gradient.

    Computes Q(u, p) = s * integral_a^b psi(arg(u), m(p)) dx for the
    density ``density`` (the strain energy, or a stress for an average
    stress) and the material ``m = material_map(p)``.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    density : GradientDensity1DProtocol
        The pointwise density and its derivatives.
    material_map : MaterialMapProtocol
        The parameters-to-material map of the forward model's
        parameterization; its output stacks ``density.nfields()``
        nodal fields.
    bkd : Backend
        Computational backend.
    a_sub : float, optional
        Left endpoint of integration domain (physical coordinates).
        If None, integrates over the full domain.
    b_sub : float, optional
        Right endpoint of integration domain (physical coordinates).
        If None, integrates over the full domain.
    scale : float
        The factor s, e.g. 1/L for an average over a bar of length L.

    Raises
    ------
    TypeError
        If ``density`` or ``material_map`` does not satisfy its protocol.
    ValueError
        If only one of ``a_sub`` and ``b_sub`` is given.
    """

    def __init__(
        self,
        basis: ChebyshevBasis1D[Array],
        density: GradientDensity1DProtocol[Array],
        material_map: MaterialMapProtocol[Array],
        bkd: Backend[Array],
        a_sub: Optional[float] = None,
        b_sub: Optional[float] = None,
        scale: float = 1.0,
    ) -> None:
        if not isinstance(density, GradientDensity1DProtocol):
            raise TypeError(
                "density must satisfy GradientDensity1DProtocol, got "
                f"{type(density).__name__}"
            )
        if not isinstance(material_map, MaterialMapProtocol):
            raise TypeError(
                "material_map must satisfy MaterialMapProtocol, got "
                f"{type(material_map).__name__}"
            )
        self._bkd = bkd
        self._density = density
        self._map = material_map
        self._scale = scale

        # Physical derivative matrix
        self._Dx = basis.derivative_matrix(1, 0)  # shape (npts, npts)
        self._nstates = self._Dx.shape[0]

        # Quadrature weights
        quad = CollocationQuadrature1D(basis, bkd)
        if a_sub is None and b_sub is None:
            weights = quad.full_domain_weights()
        else:
            if a_sub is None or b_sub is None:
                raise ValueError("Both a_sub and b_sub must be provided, or both None")
            weights = quad.weights(a_sub, b_sub)
        self._weights = scale * weights

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return self._nstates

    def nparams(self) -> int:
        return self._map.nvars()

    def nunique_params(self) -> int:
        return 0

    def time_dependence(self) -> Literal["steady"]:
        """Return the kind of state this functional reads.

        One steady state, shape ``(nstates, 1)``.
        """
        return "steady"

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        return SteadyFunctionalDerivatives.first_order(
            self.state_jacobian, self.param_jacobian
        )

    def _compute_arg(self, state: Array) -> Array:
        """Compute the argument to the density from state."""
        u = state[:, 0]  # shape (npts,)
        grad_u = self._Dx @ u  # du/dx, shape (npts,)
        if self._density.uses_deformation_gradient():
            return 1.0 + grad_u
        return grad_u

    def _material(self, param: Array) -> Array:
        """The stacked material fields at ``param`` (shape (nparams, 1))."""
        material = self._map(param[:, 0])
        expected = self._density.nfields() * self._nstates
        if material.shape[0] != expected:
            raise ValueError(
                f"material_map must give nfields*npts = {expected} values, "
                f"got {material.shape[0]}"
            )
        return material

    def __call__(self, state: Array, param: Array) -> Array:
        """Evaluate Q(u, p).

        Parameters
        ----------
        state : Array, shape (nstates, 1)
            Displacement field at collocation nodes.
        param : Array, shape (nparams, 1)
            Parameters, which set the material.

        Returns
        -------
        Array, shape (1, 1)
        """
        bkd = self._bkd
        psi = self._density.value(self._compute_arg(state), self._material(param))
        return bkd.reshape(bkd.sum(self._weights * psi), (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/du via chain rule: s (w * dpsi/darg) @ Dx.

        Returns
        -------
        Array, shape (1, nstates)
        """
        dpsi = self._density.arg_derivative(
            self._compute_arg(state), self._material(param)
        )
        # dQ/du_j = sum_i s w_i * dpsi_i * Dx_{ij}
        return (self._weights * dpsi)[None, :] @ self._Dx

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/dp at fixed state, through the material:
        s (w * dpsi/dm) @ dm/dp.

        Returns
        -------
        Array, shape (1, nparams)
        """
        bkd = self._bkd
        dpsi_dm = self._density.material_derivative(
            self._compute_arg(state), self._material(param)
        )
        weights = bkd.concatenate([self._weights] * self._density.nfields())
        return (weights * dpsi_dm)[None, :] @ self._map.jacobian(param[:, 0])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"density={type(self._density).__name__}, "
            f"nstates={self._nstates}, "
            f"nparams={self.nparams()}, "
            f"bkd={type(self._bkd).__name__})"
        )


def create_linear_strain_energy_1d(
    basis: ChebyshevBasis1D[Array],
    youngs_modulus_map: MaterialMapProtocol[Array],
    bkd: Backend[Array],
    a_sub: Optional[float] = None,
    b_sub: Optional[float] = None,
) -> StrainEnergyFunctional1D[Array]:
    """Create strain energy functional for a linear elastic 1D bar.

    Computes W = integral (1/2)*E*epsilon^2 dx where epsilon = du/dx and
    E = youngs_modulus_map(p).

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    youngs_modulus_map : MaterialMapProtocol
        The parameters-to-E map of the forward model.
    bkd : Backend
        Computational backend.
    a_sub, b_sub : float, optional
        Integration bounds. None for full domain.
    """
    return StrainEnergyFunctional1D(
        basis,
        LinearStrainEnergyDensity1D(bkd),
        youngs_modulus_map,
        bkd,
        a_sub=a_sub,
        b_sub=b_sub,
    )


def create_neo_hookean_strain_energy_1d(
    basis: ChebyshevBasis1D[Array],
    lame_map: MaterialMapProtocol[Array],
    bkd: Backend[Array],
    a_sub: Optional[float] = None,
    b_sub: Optional[float] = None,
) -> StrainEnergyFunctional1D[Array]:
    """Create strain energy functional for a 1D Neo-Hookean bar.

    Computes W = integral psi(F) dx where F = 1 + du/dx and
    psi(F) = mu/2*(F^2 - 1 - 2*ln(F)) + lamda/2*(ln(F))^2, with
    [mu; lamda] = lame_map(p).

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    lame_map : MaterialMapProtocol
        The parameters-to-``[mu; lamda]`` map of the forward model, e.g.
        ``FixedPoissonRatioLameMap``.
    bkd : Backend
        Computational backend.
    a_sub, b_sub : float, optional
        Integration bounds. None for full domain.
    """
    return StrainEnergyFunctional1D(
        basis,
        NeoHookeanStrainEnergyDensity1D(bkd),
        lame_map,
        bkd,
        a_sub=a_sub,
        b_sub=b_sub,
    )
