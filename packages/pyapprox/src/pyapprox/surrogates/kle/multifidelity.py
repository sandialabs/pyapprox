r"""One random field, evaluated consistently on several meshes.

Multifidelity estimators and multifidelity surrogates need every model in
the ensemble to see the *same* realization of a random field. Solving a
separate KLE per mesh does not give that: each eigensolve produces its own
basis, with its own ordering, sign convention and truncation, so one
coefficient vector :math:`z` would denote a different field on each mesh
and the estimator's correlations would reflect basis mismatch rather than
genuine discretization error.

The fix is to let one eigensolve define the field and *evaluate* that basis
everywhere else. A :class:`~pyapprox.surrogates.kle.nystrom_kle.NystromKLE`
extends its eigenfunctions to arbitrary points,

.. math:: \phi(x) = C(x, S)\, T,

so a basis solved once -- typically on the highest-fidelity mesh -- can be
sampled on any other point set without a second solve. Passing the same
:math:`z` to the results then yields the same field, sampled differently.

:func:`nystrom_kles_on_meshes` performs that extension and hands back one
ordinary :class:`~pyapprox.surrogates.kle.precomputed_kle.PrecomputedKLE`
per mesh. Returning plain KLEs rather than a container is deliberate: each
satisfies ``KLEProtocol``, so it drops into field maps, parameterizations
and :func:`~pyapprox.surrogates.kle.save_kle` unchanged, and callers keep
whatever per-fidelity bookkeeping their method already uses instead of
adopting one imposed here. Consistency does not need policing by a
wrapper -- it follows from the bases sharing one eigensolve and one
extension.

Meshes are supplied as bare coordinate arrays. Nothing here knows what a
mesh *is* -- no connectivity, no element type, no library type -- so any
format works once the caller can produce ``(ndim, npts)`` coordinates.
Nodes and quadrature points need no distinction either: both are point
sets, and evaluation points carry no quadrature weights at all (the
:math:`\sqrt{w}` introduced by symmetrizing the eigenproblem and the
:math:`1/\sqrt{w}` removing it cancel, leaving only landmark weights inside
:math:`T`).

Because the source basis may equally come from a fresh solve or from
:func:`~pyapprox.surrogates.kle.load_nystrom_kle`, a field solved weeks
earlier can be extended to a mesh that did not exist at solve time. That
covers both ways a multifidelity study acquires meshes: all known up front,
or one added part-way through when an estimator needs another fidelity.
"""

from typing import List, Sequence

from pyapprox.surrogates.kle.nystrom_kle import NystromKLE
from pyapprox.surrogates.kle.precomputed_kle import PrecomputedKLE
from pyapprox.util.backends.protocols import Array


def nystrom_kle_on_mesh(
    nystrom: NystromKLE[Array], coords: Array
) -> PrecomputedKLE[Array]:
    r"""Extend a Nystrom basis to one mesh.

    Parameters
    ----------
    nystrom : NystromKLE[Array]
        The basis defining the field. May come from a fresh solve or from
        :func:`~pyapprox.surrogates.kle.load_nystrom_kle`; an extensible
        basis is an extensible basis either way.
    coords : Array
        Shape ``(ndim, npts)``. Node coordinates, quadrature points, or any
        other evaluation points -- the eigenproblem does not distinguish
        them. No quadrature weights are needed or accepted.

    Returns
    -------
    PrecomputedKLE[Array]
        The same field, evaluable on ``coords``. Frozen there: extending
        further needs the kernel, so start again from ``nystrom``.

    Raises
    ------
    TypeError
        If ``nystrom`` is not a
        :class:`~pyapprox.surrogates.kle.nystrom_kle.NystromKLE`.
    ValueError
        If ``coords`` is not 2D or its spatial dimension disagrees with
        the landmarks'.
    """
    if not isinstance(nystrom, NystromKLE):
        raise TypeError(
            f"nystrom must be a NystromKLE, got {type(nystrom).__name__}. "
            "A basis already frozen at a point set cannot be evaluated on "
            "another mesh, which is the capability this composes."
        )
    if coords.ndim != 2:
        raise ValueError(
            f"coords must be 2D (ndim, npts), got ndim={coords.ndim}"
        )
    ndim = int(nystrom.landmark_coords().shape[0])
    if int(coords.shape[0]) != ndim:
        raise ValueError(
            f"coords has {int(coords.shape[0])} spatial dimensions but the "
            f"basis was built in {ndim}"
        )
    # eigenvectors_at returns the *unweighted* basis, which is what
    # PrecomputedKLE takes; it applies sqrt(eigenvalue) and sigma itself.
    # Handing it weighted vectors would square both.
    return PrecomputedKLE(
        nystrom.eigenvalues(),
        nystrom.eigenvectors_at(coords),
        nystrom.mean_field_at(coords),
        sigma=nystrom.sigma(),
        use_log=nystrom.use_log(),
        bkd=nystrom.bkd(),
    )


def nystrom_kles_on_meshes(
    nystrom: NystromKLE[Array], meshes: Sequence[Array]
) -> List[PrecomputedKLE[Array]]:
    r"""Extend a Nystrom basis to several meshes at once.

    Equivalent to calling :func:`nystrom_kle_on_mesh` per mesh; provided
    because extending a shared field to a set of fidelities is the whole
    reason the extension exists, and a list comprehension at every call
    site obscures that.

    Parameters
    ----------
    nystrom : NystromKLE[Array]
        The basis defining the field.
    meshes : sequence of Array
        Each of shape ``(ndim, npts_i)``. Point counts may differ; the
        spatial dimension may not.

    Returns
    -------
    list of PrecomputedKLE[Array]
        One per entry of ``meshes``, in the same order. Every element
        expands the same field, so passing one ``coef`` to all of them
        gives one realization sampled at several resolutions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.kernels.matern import ExponentialKernel
    >>> from pyapprox.surrogates.kle import (
    ...     create_nystrom_kle, nystrom_kles_on_meshes)
    >>> bkd = NumpyBkd()
    >>> kernel = ExponentialKernel(
    ...     bkd.full((1,), 0.2), (0.01, 100.0), 1, bkd)
    >>> fine = bkd.linspace(0, 1, 33)[None, :]
    >>> nystrom = create_nystrom_kle(kernel, fine, 4, bkd)
    >>> coarse = bkd.linspace(0, 1, 9)[None, :]
    >>> kles = nystrom_kles_on_meshes(nystrom, [fine, coarse])
    >>> coef = bkd.ones((4, 1))
    >>> [kle(coef).shape for kle in kles]
    [(33, 1), (9, 1)]
    """
    return [nystrom_kle_on_mesh(nystrom, coords) for coords in meshes]
