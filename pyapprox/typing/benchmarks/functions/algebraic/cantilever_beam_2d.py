"""Analytical 2D cantilever beam model.

Closed-form solutions for a cantilever beam under tip point loads,
based on Euler-Bernoulli beam theory with superposition of bending
in two orthogonal planes.

Variables (6): X, Y, E, R, w, t
    X : horizontal tip load (bending about the weak axis)
    Y : vertical tip load (bending about the strong axis)
    E : Young's modulus
    R : yield stress (carried through but not used in raw QoIs)
    w : beam width
    t : beam depth (thickness)

Constants:
    L : beam length (default 100)

QoIs (2):
    stress = 6*L * (X/(w^2*t) + Y/(w*t^2))
    displacement = 4*L^3/(E*w*t) * sqrt(X^2/w^4 + Y^2/t^4)

This matches the legacy ``CantileverBeamModel`` formulas but returns
the raw stress and displacement rather than constraint ratios.

Regime of validity:
    - Euler-Bernoulli (slender) beam theory: L/w >> 1 and L/t >> 1
    - Linear elastic, small deformations
    - Rectangular cross-section w x t
    - Superposition of bending in two orthogonal planes (3D beam)
    - Not directly comparable to 2D plane stress FEM, which captures
      only in-plane deformation and includes shear/Poisson effects
      absent from Euler-Bernoulli theory
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class CantileverBeam2DAnalytical(Generic[Array]):
    """Analytical cantilever beam: (X,Y,E,R,w,t) -> [stress, displacement].

    Parameters
    ----------
    length : float
        Beam length L.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        length: float,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._length = length

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 6

    def nqoi(self) -> int:
        return 2

    def __call__(self, samples: Array) -> Array:
        """Evaluate: (X,Y,E,R,w,t) -> [stress, displacement].

        Parameters
        ----------
        samples : Array
            Shape (6, nsamples).

        Returns
        -------
        Array
            Shape (2, nsamples).
            Row 0: max bending stress.
            Row 1: resultant tip displacement.
        """
        bkd = self._bkd
        L = self._length

        X = samples[0:1, :]
        Y = samples[1:2, :]
        E = samples[2:3, :]
        # R = samples[3:4, :]  # not used in raw QoIs
        w = samples[4:5, :]
        t = samples[5:6, :]

        stress = 6.0 * L * (X / (w**2 * t) + Y / (w * t**2))

        displacement = (
            4.0 * L**3 / (E * w * t)
            * bkd.sqrt(X**2 / w**4 + Y**2 / t**4)
        )

        return bkd.concatenate([stress, displacement], axis=0)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian d(QoI)/d(X,Y,E,R,w,t).

        Parameters
        ----------
        sample : Array
            Shape (6, 1). Single sample.

        Returns
        -------
        Array
            Jacobian. Shape: (2, 6).
        """
        bkd = self._bkd
        L = self._length

        X = sample[0, 0]
        Y = sample[1, 0]
        E = sample[2, 0]
        # R = sample[3, 0]
        w = sample[4, 0]
        t = sample[5, 0]

        # ---- stress = 6*L * (X/(w^2*t) + Y/(w*t^2)) ----
        ds_dX = 6.0 * L / (w**2 * t)
        ds_dY = 6.0 * L / (w * t**2)
        ds_dE = 0.0 * E  # keep as array for autograd
        ds_dR = 0.0 * E
        ds_dw = 6.0 * L * (-2.0 * X / (w**3 * t) - Y / (w**2 * t**2))
        ds_dt = 6.0 * L * (-X / (w**2 * t**2) - 2.0 * Y / (w * t**3))

        # ---- displacement = 4*L^3/(E*w*t) * sqrt(X^2/w^4 + Y^2/t^4) ----
        S = X**2 / w**4 + Y**2 / t**4
        sqrtS = bkd.sqrt(bkd.asarray([S]))[0]
        coeff = 4.0 * L**3

        dd_dX = coeff / (E * w * t) * X / (w**4 * sqrtS)
        dd_dY = coeff / (E * w * t) * Y / (t**4 * sqrtS)
        dd_dE = -coeff / (E**2 * w * t) * sqrtS
        dd_dR = 0.0 * E
        dd_dw = coeff * (
            -sqrtS / (E * w**2 * t)
            + (-4.0 * X**2 / w**5) / (E * w * t * 2.0 * sqrtS)
        )
        dd_dt = coeff * (
            -sqrtS / (E * w * t**2)
            + (-4.0 * Y**2 / t**5) / (E * w * t * 2.0 * sqrtS)
        )

        row0 = bkd.asarray([ds_dX, ds_dY, ds_dE, ds_dR, ds_dw, ds_dt])
        row1 = bkd.asarray([dd_dX, dd_dY, dd_dE, dd_dR, dd_dw, dd_dt])
        return bkd.stack([row0, row1], axis=0)
