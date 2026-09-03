"""The record of what happened, independent of what wrote it.

These drive ``ManifestWriter`` and the record builders directly. The
marshaller's own manifest behavior -- one record per directory, the
status derivation over several invocations -- is tested where that
behavior lives, in the marshaller's tests.

No backend fixture: nothing here touches an array.
"""

import json
import os

from pyapprox.interface.evaluation.manifest import (
    KIND_PREPARED,
    KIND_RELEASED,
    KIND_RUN,
    MAX_RECORD_BYTES,
    ManifestWriter,
    manifest_filename,
    prepared_record,
    read_records,
    released_record,
    run_record,
    task_record,
)


def _writer(tmp_path):
    return ManifestWriter(str(tmp_path / "manifest.host.1.jsonl"))


class TestFilename:
    def test_it_names_host_and_pid(self) -> None:
        """Two writers that must not collide are in different
        processes, often on different machines."""
        assert manifest_filename("node042", 31337) == (
            "manifest.node042.31337.jsonl"
        )

    def test_a_dotted_hostname_keeps_only_its_first_label(self) -> None:
        assert manifest_filename("node042.cluster.example", 7) == (
            "manifest.node042.7.jsonl"
        )

    def test_it_defaults_to_this_process(self) -> None:
        assert manifest_filename().endswith(f".{os.getpid()}.jsonl")


class TestAppending:
    def test_a_record_round_trips(self, tmp_path) -> None:
        writer = _writer(tmp_path)
        writer.append({"kind": "run", "run_id": "abc"})
        assert read_records(writer.path()) == [
            {"kind": "run", "run_id": "abc"}
        ]

    def test_records_accumulate_in_order(self, tmp_path) -> None:
        writer = _writer(tmp_path)
        for i in range(5):
            writer.append({"kind": "prepared", "index": i})
        assert [r["index"] for r in read_records(writer.path())] == [
            0,
            1,
            2,
            3,
            4,
        ]

    def test_the_file_is_created_on_first_write(self, tmp_path) -> None:
        writer = _writer(tmp_path)
        assert not os.path.exists(writer.path())
        writer.append({"kind": "run"})
        assert os.path.exists(writer.path())

    def test_a_second_writer_appends_rather_than_truncating(
        self, tmp_path
    ) -> None:
        """A resumed run opens the same path if host and pid repeat."""
        _writer(tmp_path).append({"kind": "run", "n": 1})
        _writer(tmp_path).append({"kind": "run", "n": 2})
        records = read_records(str(tmp_path / "manifest.host.1.jsonl"))
        assert [r["n"] for r in records] == [1, 2]

    def test_one_line_per_record(self, tmp_path) -> None:
        """A record must not split across writes, or two processes
        would interleave halves of a line."""
        writer = _writer(tmp_path)
        writer.append({"kind": "prepared", "index": 1})
        writer.append({"kind": "prepared", "index": 2})
        with open(writer.path(), encoding="utf-8") as handle:
            assert len(handle.readlines()) == 2


class TestWritingNeverRaises:
    """A record about a run must not be able to end the run."""

    def test_an_unwritable_path_is_counted_not_raised(
        self, tmp_path
    ) -> None:
        writer = ManifestWriter(str(tmp_path / "absent" / "m.jsonl"))
        writer.append({"kind": "run"})
        assert writer.nfailed() == 1

    def test_an_unencodable_record_is_counted_not_raised(
        self, tmp_path
    ) -> None:
        writer = _writer(tmp_path)
        writer.append({"kind": "run", "bad": {1, 2, 3}})
        assert writer.nfailed() == 0
        assert len(read_records(writer.path())) == 1


class TestTheRecordCap:
    def test_a_long_record_sheds_optional_fields(self, tmp_path) -> None:
        writer = _writer(tmp_path)
        writer.append(
            prepared_record(
                submission=0,
                index=1,
                workdir="sub-000/sample-000001",
                sample=[float(i) for i in range(2000)],
            )
        )
        records = read_records(writer.path())
        assert len(records) == 1
        assert records[0]["truncated"] is True
        assert "sample" not in records[0]
        # The identity fields are what a reader cannot do without.
        assert records[0]["index"] == 1
        assert records[0]["workdir"] == "sub-000/sample-000001"

    def test_every_written_line_is_within_the_cap(self, tmp_path) -> None:
        writer = _writer(tmp_path)
        writer.append(
            prepared_record(
                submission=0,
                index=1,
                workdir="sub-000/sample-000001",
                sample=[float(i) for i in range(5000)],
            )
        )
        with open(writer.path(), "rb") as handle:
            for line in handle:
                assert len(line) <= MAX_RECORD_BYTES

    def test_a_small_sample_is_kept(self, tmp_path) -> None:
        """Self-describing runs are worth having when they fit."""
        writer = _writer(tmp_path)
        writer.append(
            prepared_record(
                submission=0, index=1, workdir="w", sample=[1.5, 2.5]
            )
        )
        assert read_records(writer.path())[0]["sample"] == [1.5, 2.5]


class TestPartialFiles:
    def test_a_trailing_partial_line_is_skipped(self, tmp_path) -> None:
        """A run killed mid-write is readable up to its last full line.

        This is the whole reason for line-delimited JSON rather than one
        document.
        """
        path = tmp_path / "m.jsonl"
        path.write_text(
            json.dumps({"kind": "run", "n": 1})
            + "\n"
            + json.dumps({"kind": "prepared", "n": 2})
            + "\n"
            + '{"kind": "prepared", "n":'
        )
        records = read_records(str(path))
        assert [r["n"] for r in records] == [1, 2]

    def test_an_absent_file_reads_as_empty(self, tmp_path) -> None:
        assert read_records(str(tmp_path / "nothing.jsonl")) == []

    def test_a_corrupt_line_does_not_stop_the_rest(self, tmp_path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text('not json\n{"kind": "run", "n": 2}\n')
        assert [r["n"] for r in read_records(str(path))] == [2]


class TestRecordShapes:
    def test_the_header_records_the_frame(self) -> None:
        """Retention is what tells absent-by-policy from absent-by-loss."""
        record = run_record(
            run_id="r",
            retention="on_failure",
            command=["solver", "-i", "params.in"],
            link_files=["/proj/mesh.exo"],
            created="2026-09-02T14:30:11Z",
            host="node042",
            pid=31337,
        )
        assert record["kind"] == KIND_RUN
        assert record["retention"] == "on_failure"
        assert record["command"] == ["solver", "-i", "params.in"]
        assert record["host"] == "node042"

    def test_the_header_omits_nsubmitted(self) -> None:
        """A marshaller never learns it; a count here would be a guess."""
        record = run_record(
            run_id="r",
            retention="never",
            command=["s"],
            link_files=[],
            created="now",
        )
        assert "nsubmitted" not in record

    def test_a_prepared_record_can_carry_a_join_key(self) -> None:
        """Batch-local indices cannot link a resubmission to its retry."""
        record = prepared_record(
            submission=1, index=0, workdir="sub-001/sample-000000",
            key="sweep:7",
        )
        assert record["kind"] == KIND_PREPARED
        assert record["key"] == "sweep:7"

    def test_a_released_record_carries_every_invocation(self) -> None:
        record = released_record(
            submission=0,
            index=12,
            workdir="sub-000/sample-000012",
            status="FAILED",
            any_failed=True,
            tasks=[
                task_record("SUCCEEDED", None, 1.0),
                task_record("FAILED", "exit code 2", 0.5),
            ],
            retained=True,
        )
        assert record["kind"] == KIND_RELEASED
        assert record["any_failed"] is True
        assert len(record["tasks"]) == 2
        assert record["tasks"][1]["detail"] == "exit code 2"
        assert record["retained"] is True
