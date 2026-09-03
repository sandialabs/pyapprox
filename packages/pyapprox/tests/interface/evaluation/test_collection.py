"""Gathering one working directory, against directories made by hand.

Built with ``tmp_path`` rather than by running a solver: what is under
test is selection and transfer, and driving a subprocess to produce the
files would make the failures harder to read without testing anything
more.

No backend fixture. Nothing here touches an array -- these are files,
symlinks and bytes.

The extensions below are arbitrary. Nothing in the module knows one
format from another: selection is by glob and by an injected collector,
and no matched file is ever opened. ``TestNoFormatIsPrivileged`` is the
assertion of that, so the fixtures elsewhere can use a stand-in
extension without implying one.
"""

import inspect
import os
import shutil

import pytest
from pyapprox.interface.evaluation.collection import (
    DEFAULT_SKIP,
    AnomalyKind,
    CollectionError,
    OutputCollectorProtocol,
    OutputSpec,
    SpecCollector,
    TransferMode,
    gather_into,
    gather_run,
    reconcile,
)
from pyapprox.interface.evaluation.manifest import (
    RUN_DONE_FILENAME,
    ManifestWriter,
    prepared_record,
    released_record,
    run_record,
    stream_records,
    task_record,
)
from pyapprox.interface.evaluation.records import JobStatus


@pytest.fixture
def workdir(tmp_path):
    """A working directory shaped like one a solver leaves behind."""
    work = tmp_path / "work"
    (work / "fields").mkdir(parents=True)
    (work / "out.fld").write_text("mesh data")
    (work / "fields" / "step_0400.fld").write_text("field data")
    (work / "params.in").write_text("1.0 2.0")
    (work / "results.out").write_text("4.2")
    (work / "solver.stderr").write_text("a warning")
    return work


@pytest.fixture
def shared_mesh(tmp_path, workdir):
    """A mesh symlinked in, as ``link_files`` does for every sample."""
    mesh = tmp_path / "shared.fld"
    mesh.write_text("x" * 1000)
    (workdir / "shared.fld").symlink_to(mesh)
    return mesh


def _gather(workdir, dest, patterns, **kwargs):
    specs = [
        p if isinstance(p, OutputSpec) else OutputSpec(pattern=p)
        for p in patterns
    ]
    return gather_into(
        str(workdir), SpecCollector(specs), str(dest), **kwargs
    )


class TestSelection:
    def test_a_glob_selects_matching_files(self, workdir, tmp_path) -> None:
        result = _gather(workdir, tmp_path / "dest", ["*.fld"])
        assert [os.path.basename(p) for p in result.paths] == ["out.fld"]

    def test_a_recursive_glob_reaches_subdirectories(
        self, workdir, tmp_path
    ) -> None:
        result = _gather(workdir, tmp_path / "dest", ["**/*.fld"])
        assert len(result.paths) == 2

    def test_relative_structure_is_preserved(
        self, workdir, tmp_path
    ) -> None:
        """Flattening would collide: every sample writes out.fld."""
        dest = tmp_path / "dest"
        _gather(workdir, dest, ["fields/*.fld"])
        assert (dest / "fields" / "step_0400.fld").exists()

    def test_a_file_matched_twice_is_collected_once(
        self, workdir, tmp_path
    ) -> None:
        result = _gather(workdir, tmp_path / "dest", ["*.fld", "out.*"])
        assert len(result.paths) == 1

    def test_framework_files_are_skipped_by_default(
        self, workdir, tmp_path
    ) -> None:
        """A pattern as ordinary as *.out must not take the inputs."""
        result = _gather(workdir, tmp_path / "dest", ["*"])
        names = {os.path.basename(p) for p in result.paths}
        assert names.isdisjoint(DEFAULT_SKIP)
        assert "out.fld" in names

    def test_a_skipped_file_can_still_be_asked_for(
        self, workdir, tmp_path
    ) -> None:
        result = gather_into(
            str(workdir),
            SpecCollector([OutputSpec("results.out")], skip=()),
            str(tmp_path / "dest"),
        )
        assert len(result.paths) == 1

    def test_nothing_matching_is_not_an_error(
        self, workdir, tmp_path
    ) -> None:
        result = _gather(workdir, tmp_path / "dest", ["*.nope"])
        assert result.paths == ()
        assert result.ok()


class TestTheSymlinkTrap:
    """A linked mesh must never be copied.

    ``link_files`` symlinks shared input into every working directory,
    so following links turns one deliberately shared 2 GB file into
    2 GB times nsamples -- silently, as a copy that appears to work.
    """

    def test_a_symlinked_file_is_not_collected(
        self, workdir, shared_mesh, tmp_path
    ) -> None:
        result = _gather(workdir, tmp_path / "dest", ["*.fld"])
        names = {os.path.basename(p) for p in result.paths}
        assert "shared.fld" not in names
        assert "out.fld" in names

    def test_is_file_alone_would_have_admitted_it(
        self, workdir, shared_mesh
    ) -> None:
        """Why the order of the two checks matters.

        ``is_file`` resolves the link, so asking it first admits exactly
        what must be excluded.
        """
        link = workdir / "shared.fld"
        assert link.is_file()
        assert link.is_symlink()

    def test_a_symlinked_directory_is_not_descended(
        self, workdir, tmp_path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "huge.fld").write_text("y" * 1000)
        (workdir / "linkdir").symlink_to(outside, target_is_directory=True)
        result = _gather(workdir, tmp_path / "dest", ["**/*.fld"])
        names = {os.path.basename(p) for p in result.paths}
        assert "huge.fld" not in names

    def test_a_solver_written_convenience_link_is_also_skipped(
        self, workdir, tmp_path
    ) -> None:
        """``latest.fld -> step_0400.fld`` is the solver's own link."""
        (workdir / "latest.fld").symlink_to(
            workdir / "fields" / "step_0400.fld"
        )
        result = _gather(workdir, tmp_path / "dest", ["*.fld"])
        names = {os.path.basename(p) for p in result.paths}
        assert names == {"out.fld"}


class TestTransferModes:
    def test_copy_leaves_the_source_in_place(
        self, workdir, tmp_path
    ) -> None:
        _gather(workdir, tmp_path / "dest", ["out.fld"])
        assert (workdir / "out.fld").exists()

    def test_move_drains_the_source(self, workdir, tmp_path) -> None:
        _gather(
            workdir, tmp_path / "dest", ["out.fld"], mode=TransferMode.MOVE
        )
        assert not (workdir / "out.fld").exists()
        assert (tmp_path / "dest" / "out.fld").exists()

    def test_hardlink_shares_the_inode(self, workdir, tmp_path) -> None:
        _gather(
            workdir,
            tmp_path / "dest",
            ["out.fld"],
            mode=TransferMode.HARDLINK,
        )
        assert (
            (workdir / "out.fld").stat().st_ino
            == (tmp_path / "dest" / "out.fld").stat().st_ino
        )

    def test_content_survives(self, workdir, tmp_path) -> None:
        _gather(workdir, tmp_path / "dest", ["out.fld"])
        assert (
            tmp_path / "dest" / "out.fld"
        ).read_text() == "mesh data"

    def test_bytes_are_reported(self, workdir, tmp_path) -> None:
        result = _gather(workdir, tmp_path / "dest", ["out.fld"])
        assert result.nbytes == len("mesh data")


class TestAtomicity:
    def test_no_partial_file_is_left_behind(
        self, workdir, tmp_path
    ) -> None:
        dest = tmp_path / "dest"
        _gather(workdir, dest, ["**/*.fld"])
        assert list(dest.rglob("*.part")) == []

    def test_a_failed_transfer_leaves_no_scratch_name(
        self, workdir, tmp_path, monkeypatch
    ) -> None:
        """A truncated file at the real path reads as a complete one."""
        dest = tmp_path / "dest"

        def explode(src, dst, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            "pyapprox.interface.evaluation.collection.shutil.copy2",
            explode,
        )
        result = _gather(workdir, dest, ["out.fld"])
        assert not result.ok()
        assert not (dest / "out.fld").exists()
        assert list(dest.rglob("*.part")) == []


class TestCollisions:
    def test_an_existing_destination_is_refused(
        self, workdir, tmp_path
    ) -> None:
        dest = tmp_path / "dest"
        _gather(workdir, dest, ["out.fld"])
        result = _gather(workdir, dest, ["out.fld"])
        assert not result.ok()
        assert "exists" in result.failures[0][1]

    def test_overwrite_allows_replacement(
        self, workdir, tmp_path
    ) -> None:
        dest = tmp_path / "dest"
        _gather(workdir, dest, ["out.fld"])
        result = _gather(workdir, dest, ["out.fld"], overwrite=True)
        assert result.ok()

    def test_two_matches_renamed_onto_one_name_is_refused(
        self, workdir, tmp_path
    ) -> None:
        """A rename mapping two files onto one is invisible data loss."""
        result = _gather(
            workdir,
            tmp_path / "dest",
            [OutputSpec("*.fld", rename=lambda p: "same.fld")],
        )
        (workdir / "second.fld").write_text("more")
        result = _gather(
            workdir,
            tmp_path / "dest2",
            [OutputSpec("*.fld", rename=lambda p: "same.fld")],
        )
        assert len(result.paths) == 1
        assert any("already claimed" in why for _, why in result.failures)


class TestRenaming:
    def test_rename_applies_to_the_basename(
        self, workdir, tmp_path
    ) -> None:
        dest = tmp_path / "dest"
        _gather(
            workdir,
            dest,
            [OutputSpec("fields/*.fld", rename=lambda p: "final.fld")],
        )
        # The directory part survives; only the name changed.
        assert (dest / "fields" / "final.fld").exists()


class TestRequiredSpecs:
    def test_an_unmatched_required_spec_is_reported(
        self, workdir
    ) -> None:
        """The case the whole feature exists for.

        A solver that exits zero having written no field file raises
        nothing anywhere else, because the decoder reads results.out
        and finds it perfectly well formed.
        """
        collector = SpecCollector([OutputSpec("*.h5", required=True)])
        assert len(collector.unmatched_required(workdir)) == 1

    def test_a_matched_required_spec_is_not_reported(
        self, workdir
    ) -> None:
        collector = SpecCollector([OutputSpec("*.fld", required=True)])
        assert collector.unmatched_required(workdir) == []

    def test_an_optional_spec_matching_nothing_is_silent(
        self, workdir
    ) -> None:
        """Everything mandatory would flag every missing optional log."""
        collector = SpecCollector([OutputSpec("*.log")])
        assert collector.unmatched_required(workdir) == []


class TestNoFormatIsPrivileged:
    """Selection is by pattern; no extension means anything here.

    Worth pinning because the motivating case is a PDE solver writing a
    mesh, and it would be easy to grow a special case for one format.
    The only names this module knows are the ones the framework itself
    writes into a working directory.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "out.h5",
            "solution.vtu",
            "grid.cgns",
            "values.csv",
            "restart.bin",
            "checkpoint",
            "field.exo",
            "TAPE.13",
        ],
    )
    def test_any_name_is_collectable(
        self, tmp_path, name
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / name).write_text("payload")
        result = _gather(work, tmp_path / "dest", [name])
        assert [os.path.basename(p) for p in result.paths] == [name]

    def test_a_directory_per_timestep_is_collectable(
        self, tmp_path
    ) -> None:
        """Some codes write a directory per step, not a file."""
        work = tmp_path / "work"
        for step in range(3):
            step_dir = work / f"step_{step:04d}"
            step_dir.mkdir(parents=True)
            (step_dir / "data").write_text(f"step {step}")
        result = _gather(work, tmp_path / "dest", ["step_*/data"])
        assert len(result.paths) == 3

    def test_no_matched_file_is_opened(
        self, tmp_path, monkeypatch
    ) -> None:
        """Unreadable content must not stop a transfer.

        The module copies bytes; anything that parsed them would fail
        on the first format it did not expect.
        """
        work = tmp_path / "work"
        work.mkdir()
        (work / "binary.dat").write_bytes(b"\x00\xff\xfe not text")
        result = _gather(work, tmp_path / "dest", ["*.dat"])
        assert result.ok()
        assert (
            tmp_path / "dest" / "binary.dat"
        ).read_bytes() == b"\x00\xff\xfe not text"


class TestGatheringAWholeRun:
    """The deferred case: a run directory, and nothing else alive.

    No evaluator, no marshaller, no process that produced it. These
    build the run directory by hand for the same reason the rest of the
    module does -- what is under test is reading a manifest and walking
    to the directories it names.
    """

    def _run(self, tmp_path, nsamples=2, submission=0, done=True):
        run = tmp_path / "scratch" / "20260903T101500Z-abc123def456"
        writer = ManifestWriter(str(run / "manifest.host.1.jsonl"))
        run.mkdir(parents=True, exist_ok=True)
        writer.append(
            run_record(
                run_id=run.name,
                retention="always",
                command=["solver"],
                link_files=[],
                created="2026-09-03T10:15:00Z",
            )
        )
        for index in range(nsamples):
            relative = f"sub-{submission:03d}/sample-{index:06d}"
            workdir = run / relative
            workdir.mkdir(parents=True)
            (workdir / "out.fld").write_text(f"sample {index}")
            (workdir / "params.in").write_text("inputs")
            writer.append(
                prepared_record(
                    submission=submission, index=index, workdir=relative
                )
            )
        if done:
            (run / RUN_DONE_FILENAME).touch()
        return run

    def test_every_recorded_directory_is_gathered(
        self, tmp_path
    ) -> None:
        run = self._run(tmp_path, nsamples=3)
        report = gather_run(
            str(run),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "dest"),
        )
        assert sorted(report.gathered) == [0, 1, 2]
        assert report.nfiles() == 3
        assert report.ok()

    def test_the_destination_keeps_run_and_submission_levels(
        self, tmp_path
    ) -> None:
        """Two submissions both number from zero; they must stay apart."""
        run = self._run(tmp_path, nsamples=1)
        dest = tmp_path / "dest"
        gather_run(
            str(run), SpecCollector([OutputSpec("*.fld")]), str(dest)
        )
        assert (
            dest / run.name / "sub-000" / "sample-000000" / "out.fld"
        ).exists()

    def test_content_reaches_the_destination(self, tmp_path) -> None:
        run = self._run(tmp_path, nsamples=2)
        dest = tmp_path / "dest"
        gather_run(
            str(run), SpecCollector([OutputSpec("*.fld")]), str(dest)
        )
        landed = dest / run.name / "sub-000" / "sample-000001" / "out.fld"
        assert landed.read_text() == "sample 1"

    def test_framework_inputs_are_still_skipped(self, tmp_path) -> None:
        run = self._run(tmp_path, nsamples=1)
        report = gather_run(
            str(run),
            SpecCollector([OutputSpec("*")]),
            str(tmp_path / "dest"),
        )
        names = {os.path.basename(p) for p in report.gathered[0]}
        assert "params.in" not in names

    def test_indices_restrict_what_is_gathered(self, tmp_path) -> None:
        run = self._run(tmp_path, nsamples=4)
        report = gather_run(
            str(run),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "dest"),
            indices=[1, 3],
        )
        assert sorted(report.gathered) == [1, 3]

    def test_a_recorded_directory_that_is_gone_is_reported(
        self, tmp_path
    ) -> None:
        """Retention may explain it; gathering cannot say on its own."""
        run = self._run(tmp_path, nsamples=2)
        shutil.rmtree(run / "sub-000" / "sample-000001")
        report = gather_run(
            str(run),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "dest"),
        )
        assert report.missing == (1,)
        assert sorted(report.gathered) == [0]

    def test_several_submissions_are_all_gathered(
        self, tmp_path
    ) -> None:
        run = self._run(tmp_path, nsamples=2, submission=0)
        self._run(tmp_path, nsamples=2, submission=1)
        dest = tmp_path / "dest"
        gather_run(
            str(run), SpecCollector([OutputSpec("*.fld")]), str(dest)
        )
        assert (dest / run.name / "sub-000" / "sample-000000").exists()
        assert (dest / run.name / "sub-001" / "sample-000000").exists()

    def test_several_manifests_are_all_read(self, tmp_path) -> None:
        """Each writing process has its own file; a resume adds one."""
        run = self._run(tmp_path, nsamples=1)
        second = ManifestWriter(str(run / "manifest.host.2.jsonl"))
        relative = "sub-001/sample-000000"
        (run / relative).mkdir(parents=True)
        (run / relative / "out.fld").write_text("from the resume")
        second.append(
            prepared_record(submission=1, index=0, workdir=relative)
        )
        report = gather_run(
            str(run),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "dest"),
        )
        assert report.nfiles() == 2

    def test_an_unfinished_run_is_flagged(self, tmp_path) -> None:
        """Gathering a live run may race writes still in progress."""
        run = self._run(tmp_path, nsamples=1, done=False)
        report = gather_run(
            str(run),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "dest"),
        )
        assert report.complete is False

    def test_a_finished_run_is_not_flagged(self, tmp_path) -> None:
        run = self._run(tmp_path, nsamples=1, done=True)
        report = gather_run(
            str(run),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "dest"),
        )
        assert report.complete is True

    def test_move_drains_the_run(self, tmp_path) -> None:
        run = self._run(tmp_path, nsamples=2)
        gather_run(
            str(run),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "dest"),
            mode=TransferMode.MOVE,
        )
        assert list(run.rglob("*.fld")) == []

    def test_an_empty_run_directory_yields_nothing(
        self, tmp_path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        report = gather_run(
            str(empty),
            SpecCollector([OutputSpec("*.fld")]),
            str(tmp_path / "dest"),
        )
        assert report.gathered == {}
        assert report.ok()

    def test_the_manifest_is_streamed_not_materialized(
        self, tmp_path
    ) -> None:
        """A large sweep records one line per sample.

        Building an index-to-record map before copying anything would
        hold the whole sweep in memory to move one file at a time.
        """
        assert inspect.isgeneratorfunction(stream_records)


class TestReconciliation:
    """Either a sample produced its output, or it was excused.

    Checked against what the run recorded, never against a directory
    listing: a listing cannot tell "the solver wrote nothing" from "the
    pattern did not match", and cannot see a sample whose directory
    retention has already removed.
    """

    def _run(self, tmp_path):
        run = tmp_path / "scratch" / "20260903T120000Z-feedfacecafe"
        run.mkdir(parents=True)
        return run, ManifestWriter(str(run / "manifest.host.1.jsonl"))

    def _header(self, writer, run, retention="always"):
        writer.append(
            run_record(
                run_id=run.name,
                retention=retention,
                command=["solver"],
                link_files=[],
                created="2026-09-03T12:00:00Z",
            )
        )

    def _sample(
        self,
        run,
        writer,
        index,
        status="SUCCEEDED",
        any_failed=False,
        retained=True,
        files=("out.fld",),
        make_dir=True,
        release=True,
    ):
        relative = f"sub-000/sample-{index:06d}"
        if make_dir:
            workdir = run / relative
            workdir.mkdir(parents=True)
            for name in files:
                (workdir / name).write_text("payload")
        writer.append(
            prepared_record(submission=0, index=index, workdir=relative)
        )
        if release:
            writer.append(
                released_record(
                    submission=0,
                    index=index,
                    workdir=relative,
                    status=status,
                    any_failed=any_failed,
                    tasks=[task_record(status, None, 1.0)],
                    retained=retained,
                )
            )

    def test_a_sample_that_produced_its_output_is_ok(
        self, tmp_path
    ) -> None:
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0)
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.ok_indices == (0,)
        assert report.ok()

    def test_a_solver_that_exited_zero_and_wrote_nothing(
        self, tmp_path
    ) -> None:
        """The case the whole feature exists for.

        An mpirun line skipped in a batch script, or a write that failed
        on a full filesystem: the scalar is there, results.out parses,
        and nothing anywhere else ever notices.
        """
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0, files=("results.out",))
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert not report.ok()
        assert report.anomalies[0].kind is AnomalyKind.NO_OUTPUT
        assert report.anomalies[0].index == 0

    def test_no_required_spec_means_the_check_cannot_fire(
        self, tmp_path
    ) -> None:
        """Nothing mandatory, so nothing to be missing."""
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0, files=("results.out",))
        report = reconcile(
            str(run), collector=SpecCollector([OutputSpec("*.fld")])
        )
        assert report.ok()

    def test_a_failed_sample_is_explained_not_flagged(
        self, tmp_path
    ) -> None:
        """The run said it failed; no output is expected."""
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(
            run, writer, 0, status="FAILED", any_failed=True, files=()
        )
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.ok()
        assert 0 in report.explained

    @pytest.mark.parametrize("status", ["TIMED_OUT", "CANCELLED"])
    def test_a_retryable_failure_is_reported_separately(
        self, tmp_path, status
    ) -> None:
        """Resubmit this, versus this parameter point is bad."""
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(
            run, writer, 0, status=status, any_failed=True, files=()
        )
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.retryable == (0,)
        assert report.ok()

    def test_a_plain_failure_is_not_retryable(self, tmp_path) -> None:
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(
            run, writer, 0, status="FAILED", any_failed=True, files=()
        )
        report = reconcile(str(run))
        assert report.retryable == ()

    def test_a_partial_sample_is_not_reported_as_writing_nothing(
        self, tmp_path
    ) -> None:
        """Row ordering, and the reason it matters.

        Success is per quantity, so a sample whose values succeeded and
        whose jacobian failed is legitimately missing that quantity's
        file. Checking the output rule first would flag every
        partially-failed derivative sample as a solver that wrote
        nothing.
        """
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(
            run,
            writer,
            0,
            status="SUCCEEDED",
            any_failed=True,
            files=("results.out",),
        )
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.ok()
        assert 0 in report.partial

    def test_a_directory_removed_by_retention_is_explained(
        self, tmp_path
    ) -> None:
        run, writer = self._run(tmp_path)
        self._header(writer, run, retention="never")
        self._sample(run, writer, 0, retained=False, make_dir=False)
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.ok()
        assert "retention" in report.explained[0]

    def test_a_directory_that_should_have_been_kept_and_is_gone(
        self, tmp_path
    ) -> None:
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0, retained=True, make_dir=False)
        report = reconcile(str(run))
        assert report.anomalies[0].kind is AnomalyKind.VANISHED

    def test_prepared_but_never_released(self, tmp_path) -> None:
        """The run stopped in between, so the directory is still there."""
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0, release=False)
        report = reconcile(str(run))
        assert report.anomalies[0].kind is AnomalyKind.NEVER_RELEASED

    def test_submitted_but_never_prepared(self, tmp_path) -> None:
        """Needs the caller's statuses; the manifest cannot know."""
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0)
        report = reconcile(
            str(run),
            statuses={0: JobStatus.SUCCEEDED, 7: JobStatus.SUCCEEDED},
        )
        kinds = {a.index: a.kind for a in report.anomalies}
        assert kinds == {7: AnomalyKind.NEVER_PREPARED}

    def test_two_submissions_using_one_index_are_not_a_duplicate(
        self, tmp_path
    ) -> None:
        """Indices are batch-local and restart at zero.

        One marshaller submitted twice produces two samples numbered 0,
        in different directories, doing different work. Keying by index
        alone would report every second submission as a duplicate --
        and two submissions through one marshaller is ordinary, not an
        edge case.
        """
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0)
        for relative, submission in (("sub-001/sample-000000", 1),):
            workdir = run / relative
            workdir.mkdir(parents=True)
            (workdir / "out.fld").write_text("payload")
            writer.append(
                prepared_record(
                    submission=submission, index=0, workdir=relative
                )
            )
            writer.append(
                released_record(
                    submission=submission,
                    index=0,
                    workdir=relative,
                    status="SUCCEEDED",
                    any_failed=False,
                    tasks=[task_record("SUCCEEDED", None, 1.0)],
                    retained=True,
                )
            )
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert report.ok()
        assert report.ok_indices == (0,)

    def test_an_anomaly_in_either_submission_still_surfaces(
        self, tmp_path
    ) -> None:
        """A later clean run must not paper over an earlier problem."""
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        # Submission 0 wrote no field file; submission 1 did.
        self._sample(run, writer, 0, files=("results.out",))
        relative = "sub-001/sample-000000"
        (run / relative).mkdir(parents=True)
        (run / relative / "out.fld").write_text("payload")
        writer.append(
            prepared_record(submission=1, index=0, workdir=relative)
        )
        writer.append(
            released_record(
                submission=1,
                index=0,
                workdir=relative,
                status="SUCCEEDED",
                any_failed=False,
                tasks=[task_record("SUCCEEDED", None, 1.0)],
                retained=True,
            )
        )
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert not report.ok()
        assert report.anomalies[0].kind is AnomalyKind.NO_OUTPUT

    def test_a_duplicate_release_is_flagged(self, tmp_path) -> None:
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0)
        writer.append(
            released_record(
                submission=0,
                index=0,
                workdir="sub-000/sample-000000",
                status="SUCCEEDED",
                any_failed=False,
                tasks=[],
                retained=True,
            )
        )
        report = reconcile(str(run))
        assert report.anomalies[0].kind is AnomalyKind.DUPLICATE

    def test_reconciliation_copies_nothing(self, tmp_path) -> None:
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0)
        before = sorted(p.name for p in (run).rglob("*"))
        reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert sorted(p.name for p in (run).rglob("*")) == before

    def test_anomalies_are_returned_not_raised(self, tmp_path) -> None:
        """1994 good samples and a list beats a traceback."""
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        for index in range(4):
            self._sample(
                run,
                writer,
                index,
                files=("out.fld",) if index else ("results.out",),
            )
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        assert len(report.anomalies) == 1
        assert report.ok_indices == (1, 2, 3)

    def test_raise_if_anomalous_is_available(self, tmp_path) -> None:
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0, files=("results.out",))
        report = reconcile(
            str(run),
            collector=SpecCollector([OutputSpec("*.fld", required=True)]),
        )
        with pytest.raises(CollectionError, match="no_output"):
            report.raise_if_anomalous()

    def test_raise_if_anomalous_is_silent_when_clean(
        self, tmp_path
    ) -> None:
        run, writer = self._run(tmp_path)
        self._header(writer, run)
        self._sample(run, writer, 0)
        reconcile(str(run)).raise_if_anomalous()


class TestTheCollectorIsAProtocol:
    def test_the_default_collector_satisfies_it(self) -> None:
        assert isinstance(SpecCollector([]), OutputCollectorProtocol)

    def test_a_custom_collector_is_accepted(
        self, workdir, tmp_path
    ) -> None:
        """For output whose names no glob expresses."""

        class Newest:
            def matches(self, workdir):
                candidates = [
                    p for p in workdir.glob("**/*.fld") if p.is_file()
                ]
                return sorted(candidates)[-1:]

        result = gather_into(
            str(workdir), Newest(), str(tmp_path / "dest")
        )
        assert len(result.paths) == 1

    def test_something_that_is_not_a_collector_is_refused(
        self, workdir, tmp_path
    ) -> None:
        with pytest.raises(TypeError, match="OutputCollectorProtocol"):
            gather_into(str(workdir), object(), str(tmp_path / "dest"))
