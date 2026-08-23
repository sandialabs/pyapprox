"""Store a KLE basis and read it back without re-solving.

Computing a KLE costs an eigendecomposition; evaluating one is a
matmul. These two functions let the expensive half happen once.

The format is a plain ``.npz`` archive: standard library, no optional
dependency, and readable by anything that reads numpy archives rather
than only by the class that wrote it. That last property is the reason
to prefer it over pickling a KLE object, which round-trips correctly
today but ties every stored basis to the current class layout and
carries the kernel and coordinates along with it.

Saving and loading are deliberately asymmetric. ``save_kle`` accepts
any :class:`KLEProtocol`, since it needs only the protocol's own
members; ``load_kle`` always returns a :class:`PrecomputedKLE`,
because the build inputs are exactly what is being discarded and
there would be nothing to rebuild a :class:`MeshKLE` from.
"""

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from pyapprox.surrogates.kle.precomputed_kle import PrecomputedKLE
from pyapprox.surrogates.kle.protocols import KLEProtocol
from pyapprox.util.backends.protocols import Array, Backend

# Bump when the set of stored arrays or their meaning changes. Stored
# from the first release so that a later change can be detected and
# reported, rather than read as though it were the current format and
# producing wrong numbers.
SCHEMA_VERSION = 1

PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class _StoredKLE:
    """The on-disk form of a KLE basis.

    The schema declared once, rather than as two dicts of string
    literals in :func:`save_kle` and :func:`load_kle` that have to be
    kept in step. Each entry is typed individually, so storing a
    ``bool`` where an array belongs is a type error rather than an
    archive that fails on read.

    Private: callers use :func:`save_kle` and :func:`load_kle`. The
    layout is an implementation detail of the format, and naming it
    here is what allows it to change in one place.
    """

    eigenvalues: NDArray[np.floating[Any]]
    eigenvectors: NDArray[np.floating[Any]]
    mean_field: NDArray[np.floating[Any]]
    sigma: float
    use_log: bool
    schema_version: int = SCHEMA_VERSION

    def to_npz(self, path: PathLike) -> None:
        """Write every field as an entry of a ``.npz`` archive."""
        np.savez(path, **asdict(self))

    @classmethod
    def from_npz(cls, path: PathLike) -> "_StoredKLE":
        """Read an archive back, checking it before trusting it.

        Scalars come out of a ``.npz`` as zero-dimensional arrays, so
        each is converted to the Python type the field declares.
        """
        with np.load(path) as archive:
            present = set(archive.files)
            if "schema_version" not in present:
                raise ValueError(
                    f"{path} has no schema_version entry, so it was not "
                    "written by save_kle"
                )
            stored = int(archive["schema_version"])
            if stored != SCHEMA_VERSION:
                raise ValueError(
                    f"{path} was written with KLE schema version {stored}, "
                    f"but this version of PyApprox reads version "
                    f"{SCHEMA_VERSION}. The stored arrays cannot be "
                    "interpreted safely."
                )
            missing = {f.name for f in fields(cls)} - present
            if missing:
                raise ValueError(
                    f"{path} is missing required entries: "
                    f"{', '.join(sorted(missing))}"
                )
            return cls(
                eigenvalues=archive["eigenvalues"],
                eigenvectors=archive["eigenvectors"],
                mean_field=archive["mean_field"],
                sigma=float(archive["sigma"]),
                use_log=bool(archive["use_log"]),
                schema_version=stored,
            )


def save_kle(
    path: PathLike,
    kle: KLEProtocol[Array],
    sigma: float = 1.0,
    use_log: bool = False,
) -> None:
    """Write a KLE basis to a ``.npz`` archive.

    Stores the eigenvalues, the *unweighted* eigenvectors, the mean
    field, and the two scalars needed to interpret them.

    The unweighted basis is stored rather than
    ``weighted_eigenvectors()``, though either recovers the other and
    both directions were measured accurate to 1e-16 even on a spectrum
    spanning 1.2e+02 down to 6.0e-12. Two reasons decide it: the
    unweighted vectors are what the solver produces and what
    ``finalize_eigenpairs`` canonicalizes, so storing them keeps the
    primitive rather than a derived quantity; and they leave ``sigma``
    separable, so one stored basis can serve several field definitions.

    ``sigma`` and ``use_log`` are parameters here rather than read off
    ``kle``, because they are not part of the KLE -- they scale and
    transform a realization, and the classes disagree about which of
    them they model. Reading them off whatever accessors happened to
    exist gave a KLE built with ``sigma=2.0`` a stored value of 1.0 and
    halved every reloaded realization. State them the same way they
    were stated when the KLE was built.

    Parameters
    ----------
    path : str or PathLike
        Destination. ``.npz`` is appended by numpy if absent.
    kle : KLEProtocol[Array]
        Any KLE; only protocol members are read.

        A :class:`NystromKLE` stores its basis *at the landmarks*. To
        store one evaluated elsewhere, build a
        :class:`PrecomputedKLE` from ``eigenvectors_at(points)`` and
        ``mean_field_at(points)`` first; a loaded basis cannot be
        extended, since extension needs the kernel.
    sigma : float
        Standard deviation scaling to record with the basis.
    use_log : bool
        Whether realizations from this basis are exponentiated.

    Raises
    ------
    TypeError
        If ``kle`` does not satisfy :class:`KLEProtocol`.
    """
    if not isinstance(kle, KLEProtocol):
        raise TypeError(
            f"kle must satisfy KLEProtocol, got {type(kle).__name__}"
        )
    bkd = kle.bkd()
    _StoredKLE(
        eigenvalues=bkd.to_numpy(kle.eigenvalues()),
        eigenvectors=bkd.to_numpy(kle.eigenvectors()),
        mean_field=bkd.to_numpy(kle.mean_field()),
        sigma=float(sigma),
        use_log=bool(use_log),
    ).to_npz(path)


def load_kle(path: PathLike, bkd: Backend[Array]) -> PrecomputedKLE[Array]:
    """Read a basis written by :func:`save_kle`.

    Parameters
    ----------
    path : str or PathLike
        A ``.npz`` archive written by :func:`save_kle`.
    bkd : Backend[Array]
        Backend the returned arrays are created under. The archive is
        backend-neutral, so a basis saved from one backend loads into
        another.

    Returns
    -------
    PrecomputedKLE[Array]
        A KLE carrying the stored basis. It evaluates identically to
        the one that was saved, but cannot be extended to new points.

    Raises
    ------
    ValueError
        If the archive was written by an incompatible schema version,
        or is missing an array the format requires.
    """
    stored = _StoredKLE.from_npz(path)
    return PrecomputedKLE(
        bkd.asarray(stored.eigenvalues),
        bkd.asarray(stored.eigenvectors),
        bkd.asarray(stored.mean_field),
        sigma=stored.sigma,
        use_log=stored.use_log,
        bkd=bkd,
    )


