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
)


def _cost() -> Cost:
    return Cost(
        wall_clock=1.5, compute=6.0, provenance=ComputeProvenance.MEASURED
    )


@pytest.fixture(params=["memory", "npz", "pickle"])
def store(request, bkd, tmp_path):
    """Each implementation, behind the one protocol."""
    if request.param == "memory":
        return InMemoryResultStore()
    if request.param == "npz":
        return NpzResultStore(str(tmp_path / "npz"), bkd)
    return PickleResultStore(str(tmp_path / "pkl"))


def _values(bkd, n=2):
    return bkd.array([[float(i), float(i) + 0.5] for i in range(n)]).T[:1]


class TestProtocolConformance:
    def test_every_store_satisfies_the_protocol(self, store):
        assert isinstance(store, ResultStore)


class TestRoundTrip:
    """What goes in comes back, field for field."""

    def test_values_and_indices(self, store, bkd):
        decoded = Decoded(
            values=bkd.array([[1.0, 2.0]]), indices=[3, 7]
        )
        store.save("k", decoded, _cost())
        loaded, cost = store.load("k")
        bkd.assert_allclose(loaded.values, decoded.values)
        assert list(loaded.indices) == [3, 7]
        assert cost.wall_clock == pytest.approx(1.5)
        assert cost.compute == pytest.approx(6.0)
        assert cost.provenance is ComputeProvenance.MEASURED

    def test_indices_survive_as_ints(self, store, bkd):
        """Indices index arrays, so they must not come back as floats.

        A numpy int64 would still index correctly, which is why this
        asserts the Python type rather than the value: a store that
        widens them passes every value check and then fails wherever an
        index is used as a dict key.
        """
        store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[5]), _cost()
        )
        loaded, _ = store.load("k")
        assert all(isinstance(i, int) for i in loaded.indices)

    def test_jacobians(self, store, bkd):
        decoded = Decoded(
            values=bkd.array([[1.0, 2.0]]),
            indices=[0, 1],
            # sample-first (n, nqoi, nvars)
            jacobians=bkd.array(
                [[[1.0, 2.0]], [[3.0, 4.0]]]
            ),
        )
        store.save("k", decoded, _cost())
        loaded, _ = store.load("k")
        assert loaded.jacobians is not None
        assert loaded.jacobians.shape == (2, 1, 2)
        bkd.assert_allclose(loaded.jacobians, decoded.jacobians)

    def test_directional_fields_keep_their_axes(self, store, bkd):
        """jvps are sample-last, hvps sample-first -- easy to transpose.

        The two directional fields use opposite conventions, so a store
        that normalizes them to one layout would corrupt exactly one of
        them, and only for non-square shapes.
        """
        decoded = Decoded(
            values=bkd.array([[1.0, 2.0]]),
            indices=[0, 1],
            jvps=bkd.array([[1.0, 2.0]]),  # (nqoi, n) = (1, 2)
            hvps=bkd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # (n, nvars)
        )
        store.save("k", decoded, _cost())
        loaded, _ = store.load("k")
        assert loaded.jvps is not None and loaded.hvps is not None
        assert loaded.jvps.shape == (1, 2)
        assert loaded.hvps.shape == (2, 3)
        bkd.assert_allclose(loaded.jvps, decoded.jvps)
        bkd.assert_allclose(loaded.hvps, decoded.hvps)

    def test_absent_fields_stay_none(self, store, bkd):
        """None must not become an empty array on the way back.

        An empty array is truthy-shaped and would make a consumer
        believe the capability was served and returned nothing.
        """
        store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        loaded, _ = store.load("k")
        assert loaded.jacobians is None
        assert loaded.hessians is None
        assert loaded.jvps is None
        assert loaded.hvps is None
        assert loaded.hvp_weights is None
        assert loaded.wall_time is None

    def test_wall_time_round_trips(self, store, bkd):
        store.save(
            "k",
            Decoded(
                values=bkd.array([[1.0]]), indices=[0], wall_time=12.25
            ),
            _cost(),
        )
        loaded, _ = store.load("k")
        assert loaded.wall_time == pytest.approx(12.25)

    @pytest.mark.parametrize(
        "provenance",
        [
            ComputeProvenance.MEASURED,
            ComputeProvenance.ESTIMATED,
            ComputeProvenance.NOT_APPLICABLE,
        ],
    )
    def test_every_provenance_round_trips(self, store, bkd, provenance):
        """Provenance travels with the number or the number is a lie."""
        compute = 0.0 if provenance is ComputeProvenance.NOT_APPLICABLE else 2.0
        store.save(
            "k",
            Decoded(values=bkd.array([[1.0]]), indices=[0]),
            Cost(wall_clock=1.0, compute=compute, provenance=provenance),
        )
        _, cost = store.load("k")
        assert cost.provenance is provenance


class TestKeys:
    def test_absent_key_loads_as_none(self, store):
        assert store.load("never-written") is None

    def test_keys_lists_what_was_saved(self, store, bkd):
        for key in ("a", "b", "c"):
            store.save(
                key, Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
            )
        assert sorted(store.keys()) == ["a", "b", "c"]

    def test_keys_is_empty_before_any_save(self, store):
        assert list(store.keys()) == []

    def test_resaving_a_key_replaces_it(self, store, bkd):
        """A resumed run may recompute a sample whose save was cut short."""
        store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        store.save(
            "k", Decoded(values=bkd.array([[9.0]]), indices=[0]), _cost()
        )
        loaded, _ = store.load("k")
        bkd.assert_allclose(loaded.values, bkd.array([[9.0]]))
        assert list(store.keys()) == ["k"]

    def test_keys_with_separators_round_trip(self, store, bkd):
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


class TestDurability:
    """Behavior specific to the stores that write files."""

    @pytest.fixture(params=["npz", "pickle"])
    def file_store(self, request, bkd, tmp_path):
        directory = str(tmp_path / request.param)
        if request.param == "npz":
            return NpzResultStore(directory, bkd)
        return PickleResultStore(directory)

    def test_directory_is_created(self, bkd, tmp_path):
        nested = str(tmp_path / "does" / "not" / "exist")
        NpzResultStore(nested, bkd)
        assert os.path.isdir(nested)

    def test_a_second_store_reads_the_first(self, file_store, bkd):
        """The point of a file-backed store: it outlives its object."""
        file_store.save(
            "k", Decoded(values=bkd.array([[4.0]]), indices=[2]), _cost()
        )
        directory = file_store.directory()
        reopened = (
            NpzResultStore(directory, bkd)
            if isinstance(file_store, NpzResultStore)
            else PickleResultStore(directory)
        )
        loaded, _ = reopened.load("k")
        bkd.assert_allclose(loaded.values, bkd.array([[4.0]]))

    def test_no_temporary_files_are_left(self, file_store, bkd):
        """A completed save leaves the result and nothing else."""
        file_store.save(
            "k", Decoded(values=bkd.array([[1.0]]), indices=[0]), _cost()
        )
        leftovers = [
            name
            for name in os.listdir(file_store.directory())
            if name.endswith(".tmp")
        ]
        assert leftovers == []

    def test_stray_temporaries_are_not_reported_as_keys(
        self, file_store, bkd
    ):
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
