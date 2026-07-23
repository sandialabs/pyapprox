"""GalerkinLameParameterization: maps (E, nu) per material to Lame parameters.

For Galerkin composite elasticity physics with per-element material properties.
The parameter vector is [E1, nu1, E2, nu2, ...] of length 2*nmaterials.

Satisfies ParameterizationProtocol. The ``apply`` method calls
``physics.set_lame_parameters()`` with per-element Lame arrays. When the
physics provides ``residual_lam_sensitivity()``/
``residual_mu_sensitivity()``, the derivative bundle carries a parameter
jacobian computed from them via the chain rule; otherwise the bundle is
empty.
"""

from typing import (
    Dict,
    Generic,
    List,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np

from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class _GalerkinLamePhysicsProtocol(Protocol):
    """Galerkin elasticity physics members ``apply`` calls
    (interim; the typed facades of the parameterization redesign
    replace it). Array-free, hence non-generic."""

    def nstates(self) -> int: ...

    def set_lame_parameters(
        self,
        lam_per_elem: "np.ndarray[Tuple[int], np.dtype[np.float64]]",
        mu_per_elem: "np.ndarray[Tuple[int], np.dtype[np.float64]]",
    ) -> None: ...


@runtime_checkable
class _GalerkinLameSensitivityPhysicsProtocol(
    _GalerkinLamePhysicsProtocol, Protocol, Generic[Array]
):
    """Adds the residual sensitivities ``param_jacobian`` needs.

    Physics lacking these (e.g. hyperelastic composites) still support
    ``apply``; the derivative bundle is simply empty."""

    def residual_lam_sensitivity(
        self, state: Array, material_index: int
    ) -> Array: ...

    def residual_mu_sensitivity(
        self, state: Array, material_index: int
    ) -> Array: ...


@runtime_checkable
class _GalerkinLameFactoryPhysicsProtocol(
    _GalerkinLamePhysicsProtocol, Protocol
):
    """Additional members the convenience factory reads at construction."""

    def basis(self) -> "_SkfemBasisHolderProtocol": ...

    def material_names(self) -> List[str]: ...

    def element_materials(self) -> Dict[str, np.ndarray]: ...


class _SkfemMeshProtocol(Protocol):
    """Mesh member the factory reads (skfem is untyped)."""

    nelements: int


class _SkfemBasisProtocol(Protocol):
    """skfem basis member the factory reads."""

    mesh: _SkfemMeshProtocol


@runtime_checkable
class _SkfemBasisHolderProtocol(Protocol):
    """Basis member the factory reads (skfem mesh is untyped)."""

    def skfem_basis(self) -> _SkfemBasisProtocol: ...


def _lame_from_E_nu(E: float, nu: float) -> Tuple[float, float]:
    """Compute Lame parameters from Young's modulus and Poisson ratio."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def _lame_hessians_E_nu(
    E: float, nu: float
) -> Tuple[
    "np.ndarray[Tuple[int, int], np.dtype[np.float64]]",
    "np.ndarray[Tuple[int, int], np.dtype[np.float64]]",
]:
    """Hessians of lambda(E, nu) and mu(E, nu) in the (E, nu) ordering.

    Both maps are linear in E, so the (E, E) entries vanish and the
    mixed entries equal the nu-derivatives of the E-slopes.
    """
    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    d2lam_dEdnu = (1.0 + 2.0 * nu**2) / denom**2
    d2lam_dnu2 = E * (
        4.0 * nu / denom**2
        + 2.0 * (1.0 + 2.0 * nu**2) * (1.0 + 4.0 * nu) / denom**3
    )
    d2mu_dEdnu = -1.0 / (2.0 * (1.0 + nu) ** 2)
    d2mu_dnu2 = E / (1.0 + nu) ** 3
    lam_hess = np.array(
        [[0.0, d2lam_dEdnu], [d2lam_dEdnu, d2lam_dnu2]]
    )
    mu_hess = np.array([[0.0, d2mu_dEdnu], [d2mu_dEdnu, d2mu_dnu2]])
    return lam_hess, mu_hess


class GalerkinLameParameterization(Generic[Array]):
    """Maps [E1, nu1, E2, nu2, ...] to per-element Lame parameters.

    Satisfies ``ParameterizationProtocol``. The physics is bound at
    construction and must provide ``set_lame_parameters()`` and
    ``nstates()`` (validated eagerly). If it additionally provides
    ``residual_lam_sensitivity()``/``residual_mu_sensitivity()`` the
    derivative bundle carries first-order capability; otherwise the
    bundle is empty. One instance serves one physics — ensembles
    construct one parameterization per physics.

    Parameters
    ----------
    physics : _GalerkinLamePhysicsProtocol
        Galerkin elasticity physics to bind.
    material_names : List[str]
        Ordered list of material names.
    element_materials : Dict[str, np.ndarray]
        Mapping from material name to element index arrays.
    nelems : int
        Total number of elements in the mesh.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        physics: _GalerkinLamePhysicsProtocol,
        material_names: List[str],
        element_materials: Dict[str, np.ndarray],
        nelems: int,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, _GalerkinLamePhysicsProtocol):
            raise TypeError(
                f"physics must provide set_lame_parameters/nstates, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._material_names = list(material_names)
        self._element_materials = {
            k: np.asarray(v) for k, v in element_materials.items()
        }
        self._nelems = nelems
        self._bkd = bkd
        if isinstance(physics, _GalerkinLameSensitivityPhysicsProtocol):
            self._derivs: ParamDerivatives[Array] = (
                ParamDerivatives.second_order(
                    self._param_jacobian,
                    self.initial_param_jacobian,
                    self._param_param_hvp,
                    self._state_param_hvp,
                    self._param_state_hvp,
                )
            )
        else:
            self._derivs = ParamDerivatives.none()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def physics(self) -> _GalerkinLamePhysicsProtocol:
        """Return the bound physics instance."""
        return self._physics

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the derivative capability bundle."""
        return self._derivs

    def nparams(self) -> int:
        """Return number of parameters (2 per material: E, nu)."""
        return 2 * len(self._material_names)

    def apply(self, params_1d: Array) -> None:
        """Convert [E1, nu1, ...] to per-element Lame arrays.

        Calls ``physics.set_lame_parameters(lam_per_elem, mu_per_elem)``.

        Parameters
        ----------
        params_1d : Array
            Parameter vector [E1, nu1, E2, nu2, ...].
            Shape: ``(2*nmaterials,)``.
        """
        params_np = self._bkd.to_numpy(params_1d)
        lam_per_elem = np.zeros(self._nelems)
        mu_per_elem = np.zeros(self._nelems)

        for i, name in enumerate(self._material_names):
            E = float(params_np[2 * i])
            nu = float(params_np[2 * i + 1])
            if not (-1.0 < nu < 0.5):
                raise ValueError(
                    f"Poisson ratio for material '{name}' must satisfy "
                    f"-1 < nu < 0.5, got {nu}"
                )
            lam, mu = _lame_from_E_nu(E, nu)
            elem_idx = self._element_materials[name]
            lam_per_elem[elem_idx] = lam
            mu_per_elem[elem_idx] = mu

        self._physics.set_lame_parameters(lam_per_elem, mu_per_elem)

    def _param_jacobian(
        self,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Compute dF/dp via chain rule through Lame parameters.

        For each material *i* with parameters (E_i, nu_i):

        .. math::

            dF/dE_i  = d\\lambda/dE  \\cdot dF/d\\lambda_i
                     + d\\mu/dE     \\cdot dF/d\\mu_i

            dF/d\\nu_i = d\\lambda/d\\nu \\cdot dF/d\\lambda_i
                      + d\\mu/d\\nu    \\cdot dF/d\\mu_i

        where ``dF/d(lambda_i)`` and ``dF/d(mu_i)`` come from
        ``physics.residual_lam_sensitivity`` and
        ``physics.residual_mu_sensitivity``.

        Parameters
        ----------
        state : Array
            Current displacement. Shape: ``(nstates,)``.
        time : float
            Current time (unused for time-independent materials).
        params_1d : Array
            Parameter vector [E1, nu1, E2, nu2, ...].

        Returns
        -------
        Array
            Parameter Jacobian. Shape: ``(nstates, 2*nmaterials)``.
        """
        if not isinstance(
            self._physics, _GalerkinLameSensitivityPhysicsProtocol
        ):
            raise RuntimeError(
                "param_jacobian is unavailable; check param_derivatives() "
                "before calling"
            )
        params_np = self._bkd.to_numpy(params_1d)
        cols = []

        for i, name in enumerate(self._material_names):
            E = float(params_np[2 * i])
            nu = float(params_np[2 * i + 1])
            denom = (1.0 + nu) * (1.0 - 2.0 * nu)

            dLambda_dE = nu / denom
            dMu_dE = 1.0 / (2.0 * (1.0 + nu))
            dLambda_dnu = E * (1.0 + 2.0 * nu**2) / denom**2
            dMu_dnu = -E / (2.0 * (1.0 + nu) ** 2)

            lam_sens = self._physics.residual_lam_sensitivity(state, i)
            mu_sens = self._physics.residual_mu_sensitivity(state, i)

            col_E = dLambda_dE * lam_sens + dMu_dE * mu_sens
            col_nu = dLambda_dnu * lam_sens + dMu_dnu * mu_sens

            cols.extend([col_E, col_nu])

        return self._bkd.stack(cols, axis=1)

    def _param_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute adj^T (d^2F/dp^2) v. Shape: ``(nparams,)``.

        F depends on p = (E_i, nu_i) only through the per-material Lame
        pair, and is LINEAR in (lam_i, mu_i), so the parameter Hessian
        of F is the Lame-map Hessian weighted by the (state-dependent,
        p-independent) residual sensitivities:

        .. math::

            adj^T \\partial^2 F/\\partial p_a \\partial p_b =
            H^{\\lambda_i}_{ab} (adj^T S_{\\lambda_i}(u))
            + H^{\\mu_i}_{ab} (adj^T S_{\\mu_i}(u))

        with no cross-material coupling.
        """
        physics = self._require_sensitivity_physics()
        params_np = self._bkd.to_numpy(params_1d)
        vvec_np = self._bkd.to_numpy(vvec)
        out = np.zeros(self.nparams())
        for i in range(len(self._material_names)):
            E = float(params_np[2 * i])
            nu = float(params_np[2 * i + 1])
            lam_hess, mu_hess = _lame_hessians_E_nu(E, nu)
            lam_sens = physics.residual_lam_sensitivity(state, i)
            mu_sens = physics.residual_mu_sensitivity(state, i)
            adj_dot_lam = float(
                self._bkd.to_numpy(self._bkd.dot(adj_state, lam_sens))
            )
            adj_dot_mu = float(
                self._bkd.to_numpy(self._bkd.dot(adj_state, mu_sens))
            )
            block = lam_hess * adj_dot_lam + mu_hess * adj_dot_mu
            out[2 * i : 2 * i + 2] = block @ vvec_np[2 * i : 2 * i + 2]
        return self._bkd.asarray(out)

    def _state_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute adj^T (d^2F/dy dp) v. Shape: ``(nstates,)``.

        dF/dp is LINEAR in the state with zero constant part (the
        sensitivities are -dK/d(lame) u contractions), and each
        dK/d(lame) is SYMMETRIC, so the mixed second derivative
        contracted with adj is the parameter Jacobian evaluated at the
        adjoint in place of the state:

        .. math::

            adj^T \\partial^2 F/\\partial y \\partial p \\, v
            = (dF/dp)(u{=}adj) \\, v
        """
        return self._bkd.dot(
            self._param_jacobian(adj_state, time, params_1d), vvec
        )

    def _param_state_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute adj^T (d^2F/dp dy) w. Shape: ``(nparams,)``.

        Transpose contraction of ``_state_param_hvp`` (same linearity +
        symmetry argument): the parameter Jacobian evaluated at the
        direction w, transposed onto the adjoint.
        """
        return self._bkd.dot(
            self._param_jacobian(wvec, time, params_1d).T, adj_state
        )

    def _require_sensitivity_physics(
        self,
    ) -> "_GalerkinLameSensitivityPhysicsProtocol[Array]":
        """Narrow the physics to the sensitivity protocol, or raise."""
        if not isinstance(
            self._physics, _GalerkinLameSensitivityPhysicsProtocol
        ):
            raise RuntimeError(
                "parameter derivatives are unavailable; check "
                "param_derivatives() before calling"
            )
        return self._physics

    def initial_param_jacobian(
        self,
        params_1d: Array,
    ) -> Array:
        """Return d(u_0)/dp = 0 (IC does not depend on material params).

        Parameters
        ----------
        params_1d : Array
            Parameter vector (unused).

        Returns
        -------
        Array
            Zero matrix. Shape: ``(nstates, 2*nmaterials)``.
        """
        return self._bkd.asarray(
            np.zeros((self._physics.nstates(), self.nparams()))
        )


def create_galerkin_lame_parameterization(
    physics: _GalerkinLameFactoryPhysicsProtocol,
    bkd: Backend[Array],
) -> "GalerkinLameParameterization[Array]":
    """Create a GalerkinLameParameterization from a Galerkin elasticity physics.

    Reads material geometry data (names, element indices, nelems) from the
    physics object at construction time.

    Parameters
    ----------
    physics : _GalerkinLameFactoryPhysicsProtocol
        Galerkin elasticity physics with ``material_names()``,
        ``element_materials()``, and a skfem basis.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    GalerkinLameParameterization
    """
    skfem_basis = physics.basis().skfem_basis()
    nelems = int(skfem_basis.mesh.nelements)
    return GalerkinLameParameterization(
        physics=physics,
        material_names=physics.material_names(),
        element_materials=physics.element_materials(),
        nelems=nelems,
        bkd=bkd,
    )
