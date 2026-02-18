"""Analytical cantilever beam model.

Closed-form solutions for a cantilever beam with composite cross-section
under linearly increasing load q(x) = q0*x/L.

The effective bending stiffness is computed from the rule-of-mixtures:
    E_eff = (A_skin * E1 + A_core * E2) / (A_skin + A_core)
    EI = E_eff * I

where I = h^3/12 for a rectangular cross-section of height h.

Under linearly increasing load q(x) = q0*x/L:
    tip_deflection    = 11 * q0 * L^4 / (120 * EI)
    integrated_stress = 3 * q0 * L^3 / (4 * H^2)   (independent of EI)
    max_curvature     = q0 * L^2 / (3 * EI)         (at x=0)

Note: For a statically determinate beam with uniform cross-section, the
bending moment M(x) depends only on the load, not on EI. Therefore the
integrated bending stress is constant regardless of material properties.

Derivation: M(x) = q0/(6L) * (2L^3 - 3xL^2 + x^3), sigma = M*6/H^2,
integral_0^L M dx = q0*L^3/8, so integral_0^L sigma dx = 3*q0*L^3/(4*H^2).
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class CantileverBeam1DAnalytical(Generic[Array]):
    """Analytical cantilever beam: (E1, E2) -> [tip_deflection, integrated_stress, max_curvature].

    Composite beam with uniform effective bending stiffness:
        E_eff = (A_skin * E1 + A_core * E2) / (A_skin + A_core)
        EI = E_eff * I

    Under linearly increasing load q(x) = q0*x/L:
        tip_deflection    = 11 * q0 * L^4 / (120 * EI)
        integrated_stress = q0 * L^3 / (2 * H^2)
        max_curvature     = q0 * L^2 / (3 * EI)

    Parameters
    ----------
    length : float
        Beam length L.
    height : float
        Total cross-section height h.
    skin_thickness : float
        Thickness of each skin layer (two symmetric skins).
    q0 : float
        Load magnitude.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        length: float,
        height: float,
        skin_thickness: float,
        q0: float,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._length = length
        self._height = height
        self._q0 = q0
        self._A_skin = 2 * skin_thickness * 1.0
        self._A_core = (height - 2 * skin_thickness) * 1.0
        self._I_rect = height**3 / 12.0

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 3

    def __call__(self, samples: Array) -> Array:
        """Evaluate: (E1, E2) -> [tip_deflection, integrated_stress, max_curvature].

        Parameters
        ----------
        samples : Array
            Shape (2, nsamples). Row 0: skin modulus E1,
            Row 1: core modulus E2.

        Returns
        -------
        Array
            QoIs. Shape: (3, nsamples).
            Row 0: tip deflection.
            Row 1: integrated bending stress over beam length.
            Row 2: max absolute curvature.
        """
        bkd = self._bkd
        E1 = samples[0:1, :]
        E2 = samples[1:2, :]
        E_eff = (
            (self._A_skin * E1 + self._A_core * E2)
            / (self._A_skin + self._A_core)
        )
        EI = E_eff * self._I_rect
        L = self._length
        H = self._height
        q0 = self._q0
        tip_defl = 11.0 * q0 * L**4 / (120.0 * EI)
        int_stress = 3.0 * q0 * L**3 / (4.0 * H**2) + 0.0 * EI
        max_curv = q0 * L**2 / (3.0 * EI)
        return bkd.concatenate([tip_defl, int_stress, max_curv], axis=0)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian d(QoI)/d(E1, E2).

        Parameters
        ----------
        sample : Array
            Shape (2, 1). Single sample.

        Returns
        -------
        Array
            Jacobian. Shape: (3, 2).
        """
        bkd = self._bkd
        E1 = sample[0, 0]
        E2 = sample[1, 0]
        A_s = self._A_skin
        A_c = self._A_core
        I = self._I_rect
        L = self._length
        q0 = self._q0

        E_eff = (A_s * E1 + A_c * E2) / (A_s + A_c)
        EI = E_eff * I

        dEeff_dE1 = A_s / (A_s + A_c)
        dEeff_dE2 = A_c / (A_s + A_c)

        dtip_dEI = -11.0 * q0 * L**4 / (120.0 * EI**2)
        dcurv_dEI = -q0 * L**2 / (3.0 * EI**2)

        # Integrated stress is independent of EI for uniform beam
        zero = 0.0 * E1  # keep as array for autograd

        return bkd.asarray([
            [dtip_dEI * dEeff_dE1 * I, dtip_dEI * dEeff_dE2 * I],
            [zero, zero],
            [dcurv_dEI * dEeff_dE1 * I, dcurv_dEI * dEeff_dE2 * I],
        ])
