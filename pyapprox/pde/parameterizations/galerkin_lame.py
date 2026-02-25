"""GalerkinLameParameterization: maps (E, nu) per material to Lame parameters.

For Galerkin composite elasticity physics with per-element material properties.
The parameter vector is [E1, nu1, E2, nu2, ...] of length 2*nmaterials.

Satisfies ParameterizationProtocol. The ``apply`` method calls
``physics.set_lame_parameters()`` with per-element Lame arrays. The
``param_jacobian`` method uses ``physics.residual_lam_sensitivity()`` and
``physics.residual_mu_sensitivity()`` via the chain rule.
"""

from typing import Dict, Generic, List, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def _lame_from_E_nu(E: float, nu: float) -> Tuple[float, float]:
    """Compute Lame parameters from Young's modulus and Poisson ratio."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


class GalerkinLameParameterization(Generic[Array]):
    """Maps [E1, nu1, E2, nu2, ...] to per-element Lame parameters.

    Satisfies ``ParameterizationProtocol``. Always exposes ``param_jacobian``
    and ``initial_param_jacobian``, but ``param_jacobian`` requires the
    physics to have ``residual_lam_sensitivity()`` and
    ``residual_mu_sensitivity()`` methods (raises ``NotImplementedError``
    if absent).

    Parameters
    ----------
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
        material_names: List[str],
        element_materials: Dict[str, np.ndarray],
        nelems: int,
        bkd: Backend[Array],
    ) -> None:
        self._material_names = list(material_names)
        self._element_materials = {
            k: np.asarray(v) for k, v in element_materials.items()
        }
        self._nelems = nelems
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return number of parameters (2 per material: E, nu)."""
        return 2 * len(self._material_names)

    def apply(self, physics: object, params_1d: Array) -> None:
        """Convert [E1, nu1, ...] to per-element Lame arrays.

        Calls ``physics.set_lame_parameters(lam_per_elem, mu_per_elem)``.

        Parameters
        ----------
        physics : object
            Galerkin elasticity physics with ``set_lame_parameters`` method.
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

        physics.set_lame_parameters(lam_per_elem, mu_per_elem)

    def param_jacobian(
        self,
        physics: object,
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
        physics : object
            Galerkin elasticity physics with sensitivity methods.
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

        Raises
        ------
        NotImplementedError
            If physics lacks ``residual_lam_sensitivity`` or
            ``residual_mu_sensitivity``.
        """
        if not hasattr(physics, "residual_lam_sensitivity"):
            raise NotImplementedError(
                f"Physics {type(physics).__name__} does not support "
                f"residual_lam_sensitivity — param_jacobian not available"
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

            lam_sens = physics.residual_lam_sensitivity(state, i)
            mu_sens = physics.residual_mu_sensitivity(state, i)

            col_E = dLambda_dE * lam_sens + dMu_dE * mu_sens
            col_nu = dLambda_dnu * lam_sens + dMu_dnu * mu_sens

            cols.extend([col_E, col_nu])

        return self._bkd.stack(cols, axis=1)

    def initial_param_jacobian(
        self, physics: object, params_1d: Array,
    ) -> Array:
        """Return d(u_0)/dp = 0 (IC does not depend on material params).

        Parameters
        ----------
        physics : object
            Galerkin elasticity physics.
        params_1d : Array
            Parameter vector (unused).

        Returns
        -------
        Array
            Zero matrix. Shape: ``(nstates, 2*nmaterials)``.
        """
        return self._bkd.asarray(
            np.zeros((physics.nstates(), self.nparams()))
        )


def create_galerkin_lame_parameterization(
    physics: object,
    bkd: Backend[Array],
) -> "GalerkinLameParameterization[Array]":
    """Create a GalerkinLameParameterization from a Galerkin elasticity physics.

    Reads material geometry data (names, element indices, nelems) from the
    physics object at construction time.

    Parameters
    ----------
    physics : object
        Galerkin elasticity physics with ``material_names()``,
        ``element_materials()``, and a skfem basis.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    GalerkinLameParameterization
    """
    nelems = physics._basis.skfem_basis().mesh.nelements
    return GalerkinLameParameterization(
        material_names=physics.material_names(),
        element_materials=physics.element_materials(),
        nelems=nelems,
        bkd=bkd,
    )
