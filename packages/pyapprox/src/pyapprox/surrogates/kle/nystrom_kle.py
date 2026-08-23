r"""Karhunen-Loeve expansion evaluable away from its collocation points.

:class:`MeshKLE` ties its basis to the coordinates it was built on:
``eigenvectors()`` is ``(ncoords, nterms)`` and there is no way to ask
for the field anywhere else. That is the binding limitation for a large
mesh, quite apart from the cost of the eigensolve -- a field built on a
subset cannot be evaluated at, say, a quadrature rule.

The Nystrom extension supplies a basis at arbitrary points from
eigenpairs computed on a small landmark set.
"""

import math
from typing import Callable, Generic, Optional, Tuple, Union

import numpy as np

from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.surrogates.kle.eigensolvers import (
    PivotedCholeskyEigenSolver,
    finalize_eigenpairs,
)
from pyapprox.util.backends.protocols import Array, Backend

MeanFieldFn = Callable[[Array], Array]


class NystromKLE(Generic[Array]):
    r"""A KLE whose basis can be evaluated at arbitrary points.

    Satisfies the KLE protocol at the landmark set and additionally
    supports out-of-sample evaluation, which is different *behaviour*
    rather than a different algorithm for the same behaviour -- hence a
    class rather than another eigensolver.

    The basis at any point is one matrix product,

    .. math:: \phi(x) = C(x, S)\, T

    for a precomputed ``(m, k)`` matrix :math:`T` and landmarks
    :math:`S`. Two properties of that form are easy to get wrong:

    **No quadrature weight at** :math:`x` **appears.** The
    :math:`\sqrt{w}` introduced by symmetrizing and the
    :math:`1/\sqrt{w}` removing it cancel; only landmark weights
    survive, inside :math:`T`. That cancellation is what lets
    :meth:`evaluate_at` work on an arbitrary quadrature rule without the
    caller supplying weights for the evaluation points.

    **Landmark weights are used as-is, never rescaled.** Treating the
    landmarks as a coarse quadrature rule and scaling
    :math:`w_j \to w_j N/m` measured 3.8e-02 against 5.1e-09 for the
    algebraic treatment.

    Build with :func:`create_nystrom_kle` rather than directly, unless
    the eigenpairs come from elsewhere.

    Parameters
    ----------
    kernel : KernelProtocol[Array]
        The covariance kernel.
    landmark_coords : Array
        Landmark points, shape ``(ndim, m)``.
    extension : Array
        The ``(m, nterms)`` matrix ``T`` above.
    eigenvalues : Array
        Shape ``(nterms,)``, descending.
    landmark_eigenvectors : Array
        Basis at the landmarks, shape ``(m, nterms)``.
    sigma : float
        Standard deviation scaling applied to the basis.
    mean_field : float or callable
        A scalar, or ``mean_field(coords) -> (npts,)``. An array is
        rejected: it has no value at a point outside the set it was
        tabulated on, which is exactly what this class exists to
        evaluate. In practice the mean is known analytically or through
        a field map that can already interpolate.
    use_log : bool
        If True the field is ``exp(mean + basis @ coef)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        kernel: KernelProtocol[Array],
        landmark_coords: Array,
        extension: Array,
        eigenvalues: Array,
        landmark_eigenvectors: Array,
        sigma: float = 1.0,
        mean_field: Union[float, MeanFieldFn[Array]] = 0.0,
        use_log: bool = False,
        bkd: Optional[Backend[Array]] = None,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        if not isinstance(kernel, KernelProtocol):
            raise TypeError(
                "kernel must satisfy KernelProtocol, got "
                f"{type(kernel).__name__}"
            )
        if not callable(mean_field) and not isinstance(
            mean_field, (int, float)
        ):
            raise TypeError(
                "mean_field must be a scalar or a callable "
                "mean_field(coords) -> (npts,). An array cannot be used "
                "here: it has no value at points outside the set it was "
                "tabulated on, which is what this class evaluates."
            )
        self._bkd = bkd
        self._kernel = kernel
        self._landmark_coords = landmark_coords
        self._extension = extension
        self._eig_vals = eigenvalues
        self._landmark_eig_vecs = landmark_eigenvectors
        self._sigma = sigma
        self._mean_field = mean_field
        self._use_log = use_log
        self._nterms = int(eigenvalues.shape[0])
        self._sqrt_eig_vals = bkd.sqrt(eigenvalues)

    # --- KLE protocol, at the landmark set -------------------------

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nterms(self) -> int:
        """Return the number of KLE terms."""
        return self._nterms

    def nvars(self) -> int:
        """Return the number of landmark points."""
        return int(self._landmark_coords.shape[1])

    def eigenvalues(self) -> Array:
        """Return the eigenvalues, shape ``(nterms,)``."""
        return self._eig_vals

    def eigenvectors(self) -> Array:
        """Unweighted basis at the landmarks, ``(m, nterms)``."""
        return self._landmark_eig_vecs

    def weighted_eigenvectors(self) -> Array:
        """Landmark basis scaled by ``sqrt(eigenvalue)`` and sigma."""
        return self._landmark_eig_vecs * self._sqrt_eig_vals * self._sigma

    def landmark_coords(self) -> Array:
        """Return the landmark points, shape ``(ndim, m)``."""
        return self._landmark_coords

    def __call__(self, coef: Array) -> Array:
        """Evaluate the field at the landmarks.

        Parameters
        ----------
        coef : Array
            Shape ``(nterms, nsamples)``.

        Returns
        -------
        Array
            Shape ``(m, nsamples)``.
        """
        return self.evaluate_at(self._landmark_coords, coef)

    # --- the capability MeshKLE lacks ------------------------------

    def eigenvectors_at(self, coords: Array) -> Array:
        """Unweighted basis at arbitrary points.

        Parameters
        ----------
        coords : Array
            Shape ``(ndim, npts)``. No quadrature weights are needed or
            accepted; see the class docstring for why they cancel.

        Returns
        -------
        Array
            Shape ``(npts, nterms)``, built on demand and not stored.
        """
        if coords.ndim != 2:
            raise ValueError(
                f"coords must be 2D (ndim, npts), got ndim={coords.ndim}"
            )
        return self._kernel(coords, self._landmark_coords) @ self._extension

    def weighted_eigenvectors_at(self, coords: Array) -> Array:
        """Basis at arbitrary points, scaled as the field uses it."""
        return (
            self.eigenvectors_at(coords) * self._sqrt_eig_vals * self._sigma
        )

    def evaluate_at(self, coords: Array, coef: Array) -> Array:
        """Field values at arbitrary points.

        Never stores a basis over the whole evaluation set: peak memory
        is ``npts * nterms`` for whatever block the caller passes, so a
        large quadrature rule can be walked a block at a time.

        Parameters
        ----------
        coords : Array
            Shape ``(ndim, npts)``.
        coef : Array
            Shape ``(nterms, nsamples)``.

        Returns
        -------
        Array
            Shape ``(npts, nsamples)``.
        """
        if coef.ndim != 2:
            raise ValueError(f"coef.ndim={coef.ndim} but should be 2")
        if coef.shape[0] != self._nterms:
            raise ValueError(
                f"coef.shape[0]={coef.shape[0]} != nterms={self._nterms}"
            )
        basis = self.weighted_eigenvectors_at(coords)
        mean = self.mean_field_at(coords)
        if self._use_log:
            return self._bkd.exp(mean[:, None] + basis @ coef)
        return mean[:, None] + basis @ coef

    def mean_field(self) -> Array:
        """Mean field at the landmarks, shape ``(m,)``.

        Every KLE has a mean; this one holds it as a scalar or a
        callable rather than an array, because it evaluates at
        arbitrary points and an array would be tied to one point set.
        Reported at the landmarks, matching ``eigenvectors()`` and
        ``__call__``, so the three agree on where "here" is.
        Use :meth:`mean_field_at` for anywhere else.
        """
        return self.mean_field_at(self._landmark_coords)

    def mean_field_at(self, coords: Array) -> Array:
        """Mean field at ``coords``, shape ``(npts,)``."""
        npts = int(coords.shape[1])
        if callable(self._mean_field):
            mean = self._mean_field(coords)
            if mean.shape != (npts,):
                raise ValueError(
                    "mean_field(coords) must return shape "
                    f"({npts},), got {tuple(mean.shape)}"
                )
            return mean
        return self._bkd.full((npts,), 1.0) * self._mean_field


def create_nystrom_kle(
    kernel: KernelProtocol[Array],
    coords: Array,
    nterms: int,
    bkd: Backend[Array],
    quad_weights: Optional[Array] = None,
    nlandmarks: Optional[int] = None,
    sigma: float = 1.0,
    mean_field: Union[float, MeanFieldFn[Array]] = 0.0,
    use_log: bool = False,
    landmark_multiplier: float = 8.0,
) -> NystromKLE[Array]:
    r"""Build a :class:`NystromKLE` by pivoted-Cholesky landmark selection.

    The pivots of a pivoted Cholesky factorization *are* the landmarks,
    so one factorization yields both the landmark set and the data the
    extension needs. Reusing
    :class:`PivotedCholeskyEigenSolver` for that rather than repeating
    the selection keeps a single numerical path: two implementations of
    one algorithm drift, and a reproduce-the-dense-solver test cannot
    catch the drift because both would pass it.

    The landmark eigenproblem is solved densely. ``m`` is chosen to be
    dense-solvable -- that is what makes it a landmark set -- so at
    ``m = 2000`` the block is 32 MB and a dense ``eigh`` is correct
    rather than a regression. It deliberately takes no eigensolver: the
    construction needs the *full* positive spectrum of the ``m x m``
    block to form :math:`U_S \Lambda_S^{-1/2}`, while the solver
    protocol returns only the leading ``nterms``, so threading one
    through would silently truncate the basis.

    Parameters
    ----------
    kernel : KernelProtocol[Array]
    coords : Array
        Candidate points, shape ``(ndim, N)``. Landmarks are chosen
        from these.
    nterms : int
        KLE terms to retain.
    bkd : Backend[Array]
    quad_weights : Array, optional
        Quadrature weights over ``coords``, shape ``(N,)``.
    nlandmarks : int, optional
        Landmarks to select. Defaults to
        ``landmark_multiplier * nterms``, capped at ``N``.
    sigma, mean_field, use_log
        Passed to :class:`NystromKLE`.
    landmark_multiplier : float
        Default landmark count as a multiple of ``nterms``.

    Raises
    ------
    ValueError
        If the landmark block cannot supply ``nterms`` directions. The
        binding constraint is the *effective rank after filtering*
        non-positive eigenvalues, not ``nlandmarks``: measured, 80
        landmarks on a squared exponential yielded only 26 usable
        directions, so comparing ``nterms`` against ``nlandmarks``
        would look comfortable while the real margin is far smaller.
    """
    if nterms < 1:
        raise ValueError(f"nterms must be >= 1, got {nterms}")
    npoints = int(coords.shape[1])
    if nlandmarks is None:
        nlandmarks = int(math.ceil(landmark_multiplier * nterms))
    nlandmarks = min(max(nlandmarks, nterms), npoints)

    sqrt_weights = None
    if quad_weights is not None:
        if quad_weights.ndim != 1:
            raise ValueError(
                f"quad_weights must be 1D, got ndim={quad_weights.ndim}"
            )
        sqrt_weights = bkd.sqrt(quad_weights)

    # the pivots are the landmarks
    selector = PivotedCholeskyEigenSolver(bkd, rank=nlandmarks)
    pivots = selector.factorize(kernel, coords, nterms, sqrt_weights).pivots()
    landmark_coords = coords[:, pivots]

    extension, eig_vals, landmark_vecs = _nystrom_extension(
        kernel, coords, landmark_coords, pivots, nterms, sqrt_weights, bkd
    )
    return NystromKLE(
        kernel,
        landmark_coords,
        extension,
        eig_vals,
        landmark_vecs,
        sigma=sigma,
        mean_field=mean_field,
        use_log=use_log,
        bkd=bkd,
    )


def _nystrom_extension(
    kernel: KernelProtocol[Array],
    coords: Array,
    landmark_coords: Array,
    pivots: Array,
    nterms: int,
    sqrt_weights: Optional[Array],
    bkd: Backend[Array],
) -> Tuple[Array, Array, Array]:
    r"""Eigenpairs of the full operator, and the extension matrix.

    Approximates the *full* ``N``-point operator by a low-rank stand-in
    built from the landmarks, then takes its eigenpairs. The textbook
    Nystrom formula instead eigendecomposes the ``m x m`` landmark block
    alone and extends those eigenfunctions; that is a correctly-solved
    *different* problem, whose accuracy is the quadrature error of the
    ``m``-point rule rather than a low-rank truncation error. It is the
    right choice when the landmarks carry a genuine quadrature rule, and
    the wrong one here, because pivoted landmarks are chosen for
    approximation quality and carry no measure: measured, 9.3e-01
    relative eigenvalue error against 5.6e-11 for the form below.

    The Rayleigh-Ritz rotation is what separates the two, and cannot be
    applied to the textbook form afterwards -- its eigenpairs satisfy an
    equation summed over all ``N`` points, while the textbook extension
    restricts the sum to the landmarks.
    """
    nlandmarks = int(landmark_coords.shape[1])
    kmat_nm = kernel(coords, landmark_coords)
    if sqrt_weights is not None:
        kmat_nm = (sqrt_weights[:, None] * kmat_nm) * sqrt_weights[pivots][
            None, :
        ]
    kmat_mm = kmat_nm[pivots, :]

    block_vals, block_vecs = bkd.eigh(kmat_mm)
    # drop non-positive directions: Lam_S^{-1/2} amplifies them without
    # bound, and they carry no information. This filter is what keeps
    # the subsequent QR well conditioned.
    threshold = (
        bkd.to_float(bkd.max(block_vals)) * nlandmarks * _MACHINE_EPS
    )
    keep = bkd.to_numpy(block_vals) > threshold
    nkept = int(keep.sum())
    if nkept < nterms:
        raise ValueError(
            f"the landmark block supplies only {nkept} usable directions "
            f"but {nterms} terms were requested. The binding constraint "
            "is the effective rank after dropping non-positive "
            f"eigenvalues, not the landmark count ({nlandmarks}): reduce "
            "nterms, raise nlandmarks, or use a kernel whose spectrum "
            "decays more slowly."
        )
    idx = bkd.asarray(keep.nonzero()[0], dtype=bkd.int64_dtype())
    block_vals = block_vals[idx]
    block_vecs = block_vecs[:, idx]

    gmat = block_vecs / bkd.sqrt(block_vals)[None, :]
    bmat = kmat_nm @ gmat
    qmat, rmat = bkd.qr(bmat)
    del qmat  # only R is needed; Q would be the N-row basis we avoid
    sig, vmat = bkd.eigh(rmat @ rmat.T)
    order = bkd.arange(sig.shape[0] - 1, -1, -1, dtype=bkd.int64_dtype())[
        :nterms
    ]
    sig = sig[order]
    vmat = vmat[:, order]

    # T maps kernel columns at any x to the basis there
    sqrt_w_landmarks = (
        sqrt_weights[pivots]
        if sqrt_weights is not None
        else bkd.full((nlandmarks,), 1.0)
    )
    extension = (
        sqrt_w_landmarks[:, None] * (gmat @ (rmat.T @ vmat)) / sig[None, :]
    )

    landmark_vecs = kernel(landmark_coords, landmark_coords) @ extension
    raw_landmark_vecs = landmark_vecs
    eig_vals, landmark_vecs = finalize_eigenpairs(
        sig, landmark_vecs, None, nterms, bkd
    )
    # finalize_eigenpairs canonicalizes signs, but it sees only the
    # landmark basis -- the extension matrix that produces the basis
    # everywhere else is untouched by it. Left there, the same object
    # reports one basis from eigenvectors() and computes with another
    # from eigenvectors_at(): measured, four of six columns came back
    # negated, so a caller reading the basis to reproduce a realization
    # got a different field with no error and matching column norms.
    #
    # Carry the same flips into the extension, so every basis this
    # object produces shares one convention. The relation is pure
    # per-column sign -- verified to a residual of exactly 0.0 -- so
    # recovering it by projection is exact rather than a fit.
    column_signs = bkd.sign(
        bkd.sum(landmark_vecs * raw_landmark_vecs, axis=0)
    )
    # A zero column cannot indicate a sign; leave it as it is rather
    # than multiplying the extension by zero.
    column_signs = bkd.where(
        bkd.equal(column_signs, 0.0),
        bkd.full(column_signs.shape, 1.0),
        column_signs,
    )
    extension = extension * column_signs[None, :]
    return extension, eig_vals, landmark_vecs


_MACHINE_EPS = float(np.finfo(float).eps)
