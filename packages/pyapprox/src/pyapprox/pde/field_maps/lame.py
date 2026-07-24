"""ENuToLameFieldMap: per-material (E, nu) to interleaved Lame values."""

from typing import Generic, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def _lame_from_E_nu(E: float, nu: float) -> Tuple[float, float]:
    """Compute Lame parameters from Young's modulus and Poisson ratio.

    .. math::

        \\lambda = \\frac{E \\nu}{(1 + \\nu)(1 - 2\\nu)}, \\qquad
        \\mu = \\frac{E}{2 (1 + \\nu)}
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def _lame_jacobian_E_nu(
    E: float, nu: float
) -> "np.ndarray[Tuple[int, int], np.dtype[np.float64]]":
    """Jacobian of :math:`(\\lambda, \\mu)` w.r.t. :math:`(E, \\nu)`.

    Rows are :math:`(\\lambda, \\mu)`, columns :math:`(E, \\nu)`.
    """
    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    dlam_dE = nu / denom
    dlam_dnu = E * (1.0 + 2.0 * nu**2) / denom**2
    dmu_dE = 1.0 / (2.0 * (1.0 + nu))
    dmu_dnu = -E / (2.0 * (1.0 + nu) ** 2)
    return np.array([[dlam_dE, dlam_dnu], [dmu_dE, dmu_dnu]])


def _lame_hessians_E_nu(
    E: float, nu: float
) -> Tuple[
    "np.ndarray[Tuple[int, int], np.dtype[np.float64]]",
    "np.ndarray[Tuple[int, int], np.dtype[np.float64]]",
]:
    """Hessians of :math:`\\lambda(E, \\nu)` and :math:`\\mu(E, \\nu)`.

    Both maps are linear in E, so the (E, E) entries vanish and the
    mixed entries equal the :math:`\\nu`-derivatives of the E-slopes.
    """
    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    d2lam_dEdnu = (1.0 + 2.0 * nu**2) / denom**2
    d2lam_dnu2 = E * (
        4.0 * nu / denom**2
        + 2.0 * (1.0 + 2.0 * nu**2) * (1.0 + 4.0 * nu) / denom**3
    )
    d2mu_dEdnu = -1.0 / (2.0 * (1.0 + nu) ** 2)
    d2mu_dnu2 = E / (1.0 + nu) ** 3
    lam_hess = np.array([[0.0, d2lam_dEdnu], [d2lam_dEdnu, d2lam_dnu2]])
    mu_hess = np.array([[0.0, d2mu_dEdnu], [d2mu_dEdnu, d2mu_dnu2]])
    return lam_hess, mu_hess


class ENuToLameFieldMap(Generic[Array]):
    """Map ``[E_1, nu_1, E_2, nu_2, ...]`` to ``[lam_1, mu_1, ...]``.

    Satisfies ``FieldMapWithHVPProtocol`` with closed-form Jacobian and
    adjoint-weighted HVP. Each material's :math:`(\\lambda_i, \\mu_i)`
    pair depends only on its own :math:`(E_i, \\nu_i)`, so the Jacobian
    is block-diagonal with 2-by-2 blocks and the map has no
    cross-material curvature. ``__call__`` validates
    :math:`-1 < \\nu < 0.5` per material.

    Computes with numpy internally (the numpy seam of the skfem-backed
    elasticity stack), converting to backend arrays at the boundary.

    Parameters
    ----------
    nmaterials : int
        Number of materials (``nvars() == 2 * nmaterials``).
    bkd : Backend
        Computational backend.
    """

    def __init__(self, nmaterials: int, bkd: Backend[Array]) -> None:
        if nmaterials < 1:
            raise ValueError(
                f"nmaterials must be positive, got {nmaterials}"
            )
        self._nmaterials = nmaterials
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of parameters (2 per material)."""
        return 2 * self._nmaterials

    def __call__(self, params_1d: Array) -> Array:
        """Evaluate the interleaved per-material Lame values.

        Parameters
        ----------
        params_1d : Array
            Parameters ``[E_1, nu_1, ...]``. Shape: ``(2*nmaterials,)``.

        Returns
        -------
        Array
            Lame values ``[lam_1, mu_1, ...]``.
            Shape: ``(2*nmaterials,)``.
        """
        params_np = self._validated_params(params_1d)
        out = np.zeros(self.nvars())
        for i in range(self._nmaterials):
            lam, mu = _lame_from_E_nu(
                float(params_np[2 * i]), float(params_np[2 * i + 1])
            )
            out[2 * i] = lam
            out[2 * i + 1] = mu
        return self._bkd.asarray(out)

    def jacobian(self, params_1d: Array) -> Array:
        """Return the block-diagonal Jacobian.

        Shape: ``(2*nmaterials, 2*nmaterials)``.
        """
        params_np = self._validated_params(params_1d)
        out = np.zeros((self.nvars(), self.nvars()))
        for i in range(self._nmaterials):
            out[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = _lame_jacobian_E_nu(
                float(params_np[2 * i]), float(params_np[2 * i + 1])
            )
        return self._bkd.asarray(out)

    def hvp(self, params_1d: Array, adj_state: Array, vvec: Array) -> Array:
        """Adjoint-weighted Hessian-vector product.

        For each material the field Hessian is the pair of 2-by-2 Lame
        Hessians weighted by the adjoint entries of that material's
        :math:`(\\lambda_i, \\mu_i)` outputs; there is no cross-material
        coupling.

        Parameters
        ----------
        params_1d : Array
            Parameters ``[E_1, nu_1, ...]``. Shape: ``(2*nmaterials,)``.
        adj_state : Array
            Field-space weights. Shape: ``(2*nmaterials,)``.
        vvec : Array
            Parameter direction. Shape: ``(2*nmaterials,)``.

        Returns
        -------
        Array
            HVP. Shape: ``(2*nmaterials,)``.
        """
        params_np = self._validated_params(params_1d)
        adj_np = self._bkd.to_numpy(adj_state)
        vvec_np = self._bkd.to_numpy(vvec)
        out = np.zeros(self.nvars())
        for i in range(self._nmaterials):
            lam_hess, mu_hess = _lame_hessians_E_nu(
                float(params_np[2 * i]), float(params_np[2 * i + 1])
            )
            block = (
                lam_hess * float(adj_np[2 * i])
                + mu_hess * float(adj_np[2 * i + 1])
            )
            out[2 * i : 2 * i + 2] = block @ vvec_np[2 * i : 2 * i + 2]
        return self._bkd.asarray(out)

    def _validated_params(
        self, params_1d: Array
    ) -> "np.ndarray[Tuple[int], np.dtype[np.float64]]":
        """Convert to numpy and validate shape and Poisson ratios."""
        params_np = self._bkd.to_numpy(params_1d)
        if params_np.shape != (self.nvars(),):
            raise ValueError(
                f"params_1d must have shape ({self.nvars()},), got "
                f"{params_np.shape}"
            )
        for i in range(self._nmaterials):
            nu = float(params_np[2 * i + 1])
            if not (-1.0 < nu < 0.5):
                raise ValueError(
                    f"Poisson ratio for material {i} must satisfy "
                    f"-1 < nu < 0.5, got {nu}"
                )
        return params_np
