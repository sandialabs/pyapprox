"""Every ResultStore must round-trip what it was given.

The tests are written once and run against every implementation, because
what is being checked is the protocol rather than any one store. A store
that passes here can be swapped for another without the calling code
noticing, which is the whole claim of injecting them.

The failure these guard against is quiet. A store silently dropping an
optional field, or returning a jacobian with its axes transposed, does
not raise on save or on load -- it produces a resumed run whose numbers
are wrong in a way nothing announces.
"""

import os
from pathlib import Path
from typing import Tuple, Union

import pytest

from pyapprox.interface.evaluation.protocols import ResultStore
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    Decoded,
)
from pyapprox.interface.evaluation.stores import (
    InMemoryResultStore,
    NpzResultStore,
    PickleResultStore,
    decode_key,
    encode_key,
)
from pyapprox.util.backends.protocols import Array, Backend


def _cost() -> Cost:
    return Cost(
        wall_clock=1.5, compute=6.0, provenance=ComputeProvenance.MEASURED
    )


@pytest.fixture(params=["memory", "npz", "pickle"])
def store(
    request: pytest.FixtureRequest, bkd: Backend[Array], tmp_path: Path
) -> ResultStore[Array]:
    """Each implementation, behind the one protocol."""
    if request.param == "memory":
        return InMemoryResultStore()
    if request.param == "npz":
        return NpzResultStore(str(tmp_path / "npz"), bkd)
    return PickleResultStore(str(tmp_path / "pkl"))


# The stores that write files, which alone have a directory to reopen.
_FileStore = Union[NpzResultStore[Array], PickleResultStore[Array]]


def _load(store: ResultStore[Array], key: str) -> Tuple[Decoded[Array], Cost]:
    """Load ``key``, failing the test if it is absent."""
    record = store.load(key)
    assert record is not None, f"nothing stored under {key!r}"
    return record


class TestProtocolConformance:
    def test_every_store_satisfies_the_protocol(
        self, store: ResultStore[Array]
    ) -> None:
        assert isinstance(store, ResultStore)


class TestRoundTrip:
    """What goes in comes back, field for field."""

    def test_values_and_indices(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        decoded = Decoded(
            values=bkd.array([[1.0, 2.0]]), indices=[3, 7]
        )
        store.save("k", decoded, _cost())
        loaded, cost = _load(store, "k")
        bkd.assert_allclose(loaded.values, decoded.values)
        assert list(loaded.indices) == [3, 7]
        assert cost.wall_clock == pytest.approx(1.5)
        assert cost.compute == pytest.approx(6.0)
        assert cost.provenance is ComputeProvenance.MEASURED

    def test_indices_survive_as_ints(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """Indices index arrays, so they must not come back as floats.

        A numpy int64 would still index correctly, which is why this
        asserts the Python type rather than the value: a store that
        widens them passes every value check and then fails wherever an
        index is used as a dict key.
        """
        store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[5]), _cost()
        )
        loaded, _ = _load(store, "k")
        assert all(isinstance(i, int) for i in loaded.indices)

    def test_jacobians(self, store: ResultStore[Array], bkd: Backend[Array]) -> None:
        # sample-first (n, nqoi, nvars)
        jacobians = bkd.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        decoded = Decoded(
            values=bkd.array([[1.0, 2.0]]),
            indices=[0, 1],
            jacobians=jacobians,
        )
        store.save("k", decoded, _cost())
        loaded, _ = _load(store, "k")
        assert loaded.jacobians is not None
        assert loaded.jacobians.shape == (2, 1, 2)
        bkd.assert_allclose(loaded.jacobians, jacobians)

    def test_directional_fields_keep_their_axes(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """jvps are sample-last, hvps sample-first -- easy to transpose.

        The two directional fields use opposite conventions, so a store
        that normalizes them to one layout would corrupt exactly one of
        them, and only for non-square shapes.
        """
        jvps = bkd.array([[1.0, 2.0]])  # (nqoi, n) = (1, 2)
        hvps = bkd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (n, nvars)
        decoded = Decoded(
            values=bkd.array([[1.0, 2.0]]),
            indices=[0, 1],
            jvps=jvps,
            hvps=hvps,
        )
        store.save("k", decoded, _cost())
        loaded, _ = _load(store, "k")
        assert loaded.jvps is not None and loaded.hvps is not None
        assert loaded.jvps.shape == (1, 2)
        assert loaded.hvps.shape == (2, 3)
        bkd.assert_allclose(loaded.jvps, jvps)
        bkd.assert_allclose(loaded.hvps, hvps)

    def test_absent_fields_stay_none(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """None must not become an empty array on the way back.

        An empty array is truthy-shaped and would make a consumer
        believe the capability was served and returned nothing.
        """
        store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        loaded, _ = _load(store, "k")
        assert loaded.jacobians is None
        assert loaded.hessians is None
        assert loaded.jvps is None
        assert loaded.hvps is None
        assert loaded.hvp_weights is None
        assert loaded.wall_time is None

    def test_wall_time_round_trips(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        store.save(
            "k",
            Decoded(
                values=bkd.array([[1.0]]), indices=[0], wall_time=12.25
            ),
            _cost(),
        )
        loaded, _ = _load(store, "k")
        assert loaded.wall_time == pytest.approx(12.25)

    @pytest.mark.parametrize(
        "provenance",
        [
            ComputeProvenance.MEASURED,
            ComputeProvenance.ESTIMATED,
            ComputeProvenance.NOT_APPLICABLE,
        ],
    )
    def test_every_provenance_round_trips(
        self,
        store: ResultStore[Array],
        bkd: Backend[Array],
        provenance: ComputeProvenance,
    ) -> None:
        """Provenance travels with the number or the number is a lie."""
        compute = 0.0 if provenance is ComputeProvenance.NOT_APPLICABLE else 2.0
        store.save(
            "k",
            Decoded(values=bkd.array([[1.0]]), indices=[0]),
            Cost(wall_clock=1.0, compute=compute, provenance=provenance),
        )
        _, cost = _load(store, "k")
        assert cost.provenance is provenance


class TestKeys:
    def test_absent_key_loads_as_none(self, store: ResultStore[Array]) -> None:
        assert store.load("never-written") is None

    def test_keys_lists_what_was_saved(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        for key in ("a", "b", "c"):
            store.save(
                key, Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
            )
        assert sorted(store.keys()) == ["a", "b", "c"]

    def test_keys_is_empty_before_any_save(self, store: ResultStore[Array]) -> None:
        assert list(store.keys()) == []

    def test_resaving_a_key_replaces_it(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """A resumed run may recompute a sample whose save was cut short."""
        store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        store.save(
            "k", Decoded(values=bkd.array([[9.0]]), indices=[0]), _cost()
        )
        loaded, _ = _load(store, "k")
        bkd.assert_allclose(loaded.values, bkd.array([[9.0]]))
        assert list(store.keys()) == ["k"]

    def test_keys_with_separators_round_trip(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """Keys are the caller's, and "sweep7/0" is a natural one.

        A filesystem-backed store must not let a separator create a
        directory or escape upwards, and must return the key it was
        given rather than its on-disk spelling.
        """
        key = "sweep7/run 2:sample=3"
        store.save(
            key, Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        assert list(store.keys()) == [key]
        assert store.load(key) is not None


class TestHeader:
    """What the keys mean, recorded beside them."""

    def test_absent_before_any_save(self, store: ResultStore[Array]) -> None:
        assert store.load_header() is None

    def test_entries_of_different_lengths_round_trip(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """Ragged entries need no padding: each is its own array."""
        header = {
            "sequence_0": bkd.array([0.5, 1.0, 0.0]),
            "sequence_1": bkd.array([0.5]),
        }
        store.save_header(header)
        loaded = store.load_header()
        assert loaded is not None
        assert sorted(loaded) == ["sequence_0", "sequence_1"]
        for name, value in header.items():
            assert loaded[name].shape == value.shape
            bkd.assert_allclose(loaded[name], value)

    def test_saving_again_replaces_every_entry(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """An entry dropped from the new header must not survive."""
        store.save_header({"a": bkd.array([1.0]), "b": bkd.array([2.0])})
        store.save_header({"a": bkd.array([3.0])})
        loaded = store.load_header()
        assert loaded is not None
        assert list(loaded) == ["a"]
        bkd.assert_allclose(loaded["a"], bkd.array([3.0]))

    def test_is_not_a_record(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """A header must not appear among, or replace, the records."""
        store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        store.save_header({"k": bkd.array([9.0])})
        assert list(store.keys()) == ["k"]
        loaded, _ = _load(store, "k")
        bkd.assert_allclose(loaded.values, bkd.array([[1.0]]))

    def test_entry_named_like_a_savez_keyword(
        self, store: ResultStore[Array], bkd: Backend[Array]
    ) -> None:
        """np.savez takes allow_pickle as a keyword; an entry may not."""
        store.save_header({"allow_pickle": bkd.array([1.0])})
        loaded = store.load_header()
        assert loaded is not None
        bkd.assert_allclose(loaded["allow_pickle"], bkd.array([1.0]))


class TestKeyEncoding:
    @pytest.mark.parametrize(
        "key", ["pilot:top-up:1", "sweep7/0", "températures", "cost in €"]
    )
    def test_round_trips(self, key: str) -> None:
        """Past U+00FF too: a character there has more than two hex
        digits, so the escape is of its UTF-8 bytes."""
        assert decode_key(encode_key(key)) == key

    def test_encoded_keys_are_plain_filenames(self) -> None:
        encoded = encode_key("../a/€:b")
        assert set(encoded) <= set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_.%"
        )
        assert "/" not in encoded


class TestDurability:
    """Behavior specific to the stores that write files."""

    @pytest.fixture(params=["npz", "pickle"])
    def file_store(
        self, request: pytest.FixtureRequest, bkd: Backend[Array], tmp_path: Path
    ) -> _FileStore[Array]:
        directory = str(tmp_path / request.param)
        if request.param == "npz":
            return NpzResultStore(directory, bkd)
        return PickleResultStore(directory)

    def test_directory_is_created(
        self, bkd: Backend[Array], tmp_path: Path
    ) -> None:
        nested = str(tmp_path / "does" / "not" / "exist")
        NpzResultStore(nested, bkd)
        assert os.path.isdir(nested)

    def test_a_second_store_reads_the_first(
        self, file_store: _FileStore[Array], bkd: Backend[Array]
    ) -> None:
        """The point of a file-backed store: it outlives its object."""
        file_store.save(
            "k", Decoded(values=bkd.array([[4.0]]), indices=[2]), _cost()
        )
        directory = file_store.directory()
        reopened: _FileStore[Array] = (
            NpzResultStore(directory, bkd)
            if isinstance(file_store, NpzResultStore)
            else PickleResultStore(directory)
        )
        loaded, _ = _load(reopened, "k")
        bkd.assert_allclose(loaded.values, bkd.array([[4.0]]))

    def test_header_survives_reopening(
        self, file_store: _FileStore[Array], bkd: Backend[Array]
    ) -> None:
        file_store.save_header({"sequence_0": bkd.array([0.5, 1.0])})
        directory = file_store.directory()
        reopened: _FileStore[Array] = (
            NpzResultStore(directory, bkd)
            if isinstance(file_store, NpzResultStore)
            else PickleResultStore(directory)
        )
        loaded = reopened.load_header()
        assert loaded is not None
        bkd.assert_allclose(loaded["sequence_0"], bkd.array([0.5, 1.0]))

    def test_store_written_without_a_header_stays_readable(
        self, file_store: _FileStore[Array], bkd: Backend[Array]
    ) -> None:
        """A store from before headers existed: records load, header None."""
        file_store.save(
            "k", Decoded(values=bkd.array([[4.0]]), indices=[0]), _cost()
        )
        assert file_store.load_header() is None
        loaded, _ = _load(file_store, "k")
        bkd.assert_allclose(loaded.values, bkd.array([[4.0]]))

    def test_no_temporary_files_are_left(
        self, file_store: _FileStore[Array], bkd: Backend[Array]
    ) -> None:
        """A completed save leaves the result and nothing else."""
        file_store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        file_store.save_header({"a": bkd.array([1.0])})
        leftovers = [
            name
            for name in os.listdir(file_store.directory())
            if name.endswith(".tmp")
        ]
        assert leftovers == []

    def test_stray_temporaries_are_not_reported_as_keys(
        self, file_store: _FileStore[Array], bkd: Backend[Array]
    ) -> None:
        """A crash mid-save must not look like a stored result.

        This is the case the store exists for, so the half-written file
        left behind by an interrupted process is the expected state, not
        an unlucky one.
        """
        file_store.save(
            "good", Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        stray = os.path.join(file_store.directory(), "interrupted.npz.tmp")
        with open(stray, "wb") as handle:
            handle.write(b"not a complete file")
        assert list(file_store.keys()) == ["good"]
