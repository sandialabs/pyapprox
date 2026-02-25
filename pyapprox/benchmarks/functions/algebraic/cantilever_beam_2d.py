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

from pyapprox.util.backends.protocols import Array, Backend


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


class CantileverBeam2DConstraints(Generic[Array]):
    """Safety margin constraints: [1 - stress/R, 1 - disp/D0].

    Wraps CantileverBeam2DAnalytical and returns constraint values that
    should be >= 0 for feasibility.

    Parameters
    ----------
    beam : CantileverBeam2DAnalytical[Array]
        The analytical beam model.
    yield_stress : float
        Yield stress threshold R.
    max_displacement : float
        Maximum allowable displacement D0.
    """

    def __init__(
        self,
        beam: CantileverBeam2DAnalytical[Array],
        yield_stress: float,
        max_displacement: float,
    ):
        self._beam = beam
        self._bkd = beam.bkd()
        self._R = yield_stress
        self._D0 = max_displacement

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._beam.nvars()

    def nqoi(self) -> int:
        return 2

    def __call__(self, samples: Array) -> Array:
        """Evaluate constraints: [1 - stress/R, 1 - disp/D0].

        Parameters
        ----------
        samples : Array
            Shape (6, nsamples).

        Returns
        -------
        Array
            Shape (2, nsamples). Values >= 0 means feasible.
        """
        qoi = self._beam(samples)  # (2, nsamples)
        stress = qoi[0:1, :]
        disp = qoi[1:2, :]
        c1 = 1.0 - stress / self._R
        c2 = 1.0 - disp / self._D0
        return self._bkd.concatenate([c1, c2], axis=0)

    def jacobian(self, sample: Array) -> Array:
        """Jacobian d(constraints)/d(vars).

        Parameters
        ----------
        sample : Array
            Shape (6, 1).

        Returns
        -------
        Array
            Shape (2, 6).
        """
        jac_beam = self._beam.jacobian(sample)  # (2, 6)
        jac_stress = jac_beam[0:1, :]  # (1, 6)
        jac_disp = jac_beam[1:2, :]  # (1, 6)
        return self._bkd.concatenate(
            [-jac_stress / self._R, -jac_disp / self._D0], axis=0
        )


class CantileverBeam2DObjective(Generic[Array]):
    """Cross-sectional area objective: w * t.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 6

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate area = w * t.

        Parameters
        ----------
        samples : Array
            Shape (6, nsamples).

        Returns
        -------
        Array
            Shape (1, nsamples).
        """
        w = samples[4:5, :]
        t = samples[5:6, :]
        return w * t

    def jacobian(self, sample: Array) -> Array:
        """Jacobian d(area)/d(X,Y,E,R,w,t).

        Parameters
        ----------
        sample : Array
            Shape (6, 1).

        Returns
        -------
        Array
            Shape (1, 6).
        """
        w = sample[4, 0]
        t = sample[5, 0]
        zero = 0.0 * w  # keep as array for autograd
        return self._bkd.asarray([[zero, zero, zero, zero, t, w]])
