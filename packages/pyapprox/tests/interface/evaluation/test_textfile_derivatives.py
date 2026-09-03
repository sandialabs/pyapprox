"""An external solver that also produces derivatives.

The library ships no derivative-reading marshaller, deliberately: the
file layout, the command, and which invocation produces what are all
properties of the solver, and a class guessing at them would ship an API
a real code has to fight.

What it does ship is everything that is *not* solver-specific --
directory creation, symlinking, retention, grouping, and three pieces
that bite: slicing a request's direction vectors by the sample's batch
column, reshaping decoded numbers onto each quantity's own axis, and
reading which quantities a request asked for. This module is the
evidence that split is right: the subclass below supplies only the
format and the commands, and everything else is inherited.

Two quantities, chosen because they exercise different seams:

- ``jacobian_batch`` needs no input beyond the sample, so it tests
  ``commands_for`` (a second invocation) and ``read_numbers`` (a second
  file, with a different count);
- ``hvp_batch`` is computed *with* a direction vector, so it
  additionally tests ``write_inputs`` -- the seam that exists precisely
  because a Hessian-vector product takes its direction as an input
  rather than being contracted out of a materialized Hessian. It is
  also the common case: gradient-based optimization and Newton-type
  methods want this product, and forming the Hessian to get it would
  convert a matrix-free method into an O(nvars^2) one.

The rest differ only in shape -- a hessian from a jacobian by its count,
a weighted product from a plain one by carrying weights -- so covering
all ten would be repetition rather than coverage.

The model is ``f(x) = sum(x_i^2)``, whose derivatives are known exactly:
the jacobian is ``2x`` and the Hessian is ``2I``, so a Hessian-vector
product is ``2v``. Assertions compare against those rather than against
finite differences, which would only agree to a few digits and need a
step-size tolerance. An exact answer is the stronger check.
"""

import os
import sys
import textwrap
from pathlib import Path
from typing import Sequence

import pytest
from pyapprox.interface.evaluation.collection import (
    AnomalyKind,
    OutputSpec,
    SpecCollector,
    gather_run,
    reconcile,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.manifest import (
    KIND_RELEASED,
    read_records,
)
from pyapprox.interface.evaluation.protocols import MarshalError
from pyapprox.interface.evaluation.records import (
    Decoded,
    Outcome,
    Request,
)
from pyapprox.interface.evaluation.subprocess_dispatcher import (
    ShellPayload,
    ShellTask,
    SubprocessDispatcher,
)
from pyapprox.interface.evaluation.textfile_marshaller import (
    QUANTITY_LAYOUT,
    Retention,
    TextFileMarshaller,
    _Layout,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array

#: A solver over f(x) = sum(x_i^2). Its jacobian is 2x and its Hessian
#: is 2I, so a Hessian-vector product is 2v -- independent of x, which
#: makes a direction that never arrived immediately visible.
SOLVER = '''
import sys
from pathlib import Path

mode = sys.argv[1]
values = [float(x) for x in Path("params.in").read_text().split()]

if mode == "values":
    Path("results.out").write_text(repr(sum(v * v for v in values)) + "\\n")
elif mode == "jacobian":
    # One row of nvars entries: d(sum x^2)/dx_i = 2 x_i.
    Path("jacobian.out").write_text(
        " ".join(repr(2.0 * v) for v in values) + "\\n"
    )
elif mode == "hvp":
    # The direction is an input, read from the file the marshaller
    # wrote. The Hessian of sum(x^2) is 2I, so the product is 2v.
    direction = [float(x) for x in Path("direction.in").read_text().split()]
    Path("hvp.out").write_text(
        " ".join(repr(2.0 * d) for d in direction) + "\\n"
    )
'''


class DerivativeMarshaller(TextFileMarshaller[Array]):
    """Reads values, a jacobian and a Hessian-vector product.

    A worked example of the three seams, and the shape a real solver
    wrapper would take. Everything not overridden -- working
    directories, symlinks, retention, one task per sample -- is
    inherited unchanged, which is the point.
    """

    def __init__(self, solver: Path, **kwargs: object) -> None:
        self._solver = solver
        super().__init__(
            command=[sys.executable, str(solver), "values"], **kwargs
        )

    def derivatives(self) -> Derivatives[Array]:
        """What this solver produces beyond its values.

        The callables are never invoked: the evaluator dispatches these
        quantities as tasks. The bundle is how the capability is
        *declared*, which is the same discovery mechanism every other
        model in the library uses.
        """
        return Derivatives(
            jacobian_batch=self._not_called,
            hvp_batch=self._not_called,
        )

    def write_inputs(
        self,
        layout: _Layout,
        sample: Array,
        index: int,
        request: Request[Array],
    ) -> None:
        """Write the sample, and any direction the request carries.

        The direction has to reach the solver as an *input*, because a
        Hessian-vector product is evaluated with it rather than
        contracted out of a Hessian afterwards -- which is what keeps a
        matrix-free method matrix-free. ``index`` is the sample's column
        in the submitted batch, which is what indexes those vectors:
        they span the submission rather than this one task.
        """
        super().write_inputs(layout, sample, index, request)
        # The slice comes from the base class, which knows that the
        # request's vectors span the submission and must be indexed by
        # the sample's batch column rather than its position in a task.
        direction = self.direction_for(request, index)
        if direction is not None:
            (layout.workdir / "direction.in").write_text(
                "\n".join(
                    repr(self.bkd().to_float(direction[row, 0]))
                    for row in range(self.nvars())
                )
                + "\n"
            )

    def commands_for(
        self, layout: _Layout, index: int, request: Request[Array]
    ) -> Sequence[ShellTask]:
        """One invocation per quantity asked for.

        A code computing everything in one solve would return a single
        task here; this one has a mode per quantity, which is the other
        shape the protocol has to support.
        """
        # Which quantities were asked for -- and restricted to what this
        # marshaller's bundle advertises -- comes from the base class.
        # Only the mapping onto this solver's mode names is local.
        mode_for = {
            "values": "values",
            "jacobian_batch": "jacobian",
            "hvp_batch": "hvp",
        }
        modes = [
            mode_for[quantity]
            for quantity in self.quantities_in(request)
            if quantity in mode_for
        ]
        if not modes:
            raise MarshalError(
                "this solver produces values, a jacobian and a "
                "Hessian-vector product; the request asked for none"
            )
        return [
            ShellTask(
                indices=(index,),
                argv=[sys.executable, str(self._solver), mode],
                workdir=str(layout.workdir),
            )
            for mode in modes
        ]

    def values(
        self, outcome: Outcome[ShellTask, ShellPayload]
    ) -> Decoded[Array]:
        """Decode whichever quantity this task computed.

        Each task ran one mode, so each decodes one file. The evaluator
        merges the pieces by index, which is what makes success
        per-quantity rather than per-sample.
        """
        payload = outcome.payload
        if payload is None:
            return super().values(outcome)
        mode = outcome.task.argv[-1]
        workdir = Path(payload.workdir)
        bkd = self.bkd()

        # Which quantity this invocation produced, and where its numbers
        # live, is the solver's business. The count to expect and the
        # axis to reshape onto are not: those come from the base class,
        # which is the point of shipping them -- the conventions are not
        # uniform and a wrong axis transposes rather than raising.
        quantity = {"jacobian": "jacobian_batch", "hvp": "hvp_batch"}.get(
            mode
        )
        if quantity is not None:
            layout = QUANTITY_LAYOUT[quantity]
            numbers = self.read_numbers(
                workdir / f"{mode}.out",
                layout.count(self.nqoi(), self.nvars()),
            )
            field, array = self.decode_quantity(quantity, numbers)
            return Decoded(
                values=bkd.zeros((self.nqoi(), 0)),
                indices=outcome.indices,
                **{field: array},
            )
        return super().values(outcome)

    def _not_called(self, *args: Array) -> Array:
        raise NotImplementedError(
            "declared so the bundle advertises the capability; the "
            "evaluator dispatches it as a task rather than calling this"
        )


@pytest.fixture
def solver(tmp_path):
    path = tmp_path / "solver.py"
    path.write_text(textwrap.dedent(SOLVER))
    return path


def _evaluator(
    solver,
    tmp_path,
    numpy_bkd,
    concurrency=2,
    retention=Retention.NEVER,
):
    marshaller = DerivativeMarshaller(
        solver,
        bkd=numpy_bkd,
        nvars=2,
        nqoi=1,
        scratch_root=str(tmp_path / "scratch"),
        retention=retention,
    )
    return marshaller, Evaluator(
        marshaller, SubprocessDispatcher(concurrency=concurrency)
    )


class TestCapabilityIsDeclared:
    def test_the_bundle_advertises_what_the_solver_produces(
        self, solver, tmp_path, numpy_bkd
    ):
        """Discovery is bundle inspection, as everywhere else."""
        marshaller, ev = _evaluator(solver, tmp_path, numpy_bkd)
        assert marshaller.derivatives().jacobian_batch is not None
        assert ev.derivatives().hvp_batch is not None
        assert ev.derivatives().hessian_batch is None

    def test_requesting_an_absent_capability_is_refused(
        self, solver, tmp_path, numpy_bkd
    ):
        """At submit, where the caller can still act on it."""
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        with pytest.raises(ValueError, match="hessians"):
            ev.submit(numpy_bkd.ones((2, 1)), Request(hessians=True))


class TestJacobian:
    """No extra input: the seam is a second command and a second file."""

    def test_a_second_invocation_reads_a_second_file(
        self, solver, tmp_path, numpy_bkd
    ):
        """d(sum x^2)/dx = 2x, known exactly."""
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        X = numpy_bkd.array([[1.0, 2.0], [3.0, 4.0]])
        result = ev.submit(
            X, Request(values=False, jacobians=True)
        ).collect()
        assert result.jacobians is not None
        numpy_bkd.assert_allclose(
            result.jacobians, numpy_bkd.reshape(2.0 * X.T, (2, 1, 2))
        )

    def test_values_and_jacobian_from_one_request(
        self, solver, tmp_path, numpy_bkd
    ):
        """Two invocations, one request, merged by index.

        The split-code shape: the caller states the ask and the
        marshaller decides how many commands that is.
        """
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        X = numpy_bkd.array([[1.0, 2.0], [3.0, 4.0]])
        result = ev.submit(
            X, Request(values=True, jacobians=True)
        ).collect()
        expected = numpy_bkd.array(
            [[float(X[0, j]) ** 2 + float(X[1, j]) ** 2 for j in range(2)]]
        )
        numpy_bkd.assert_allclose(result.values, expected)
        numpy_bkd.assert_allclose(
            result.jacobians, numpy_bkd.reshape(2.0 * X.T, (2, 1, 2))
        )

    def test_ordering_holds_across_processes(
        self, solver, tmp_path, numpy_bkd
    ):
        """Derivatives follow their indices, not their completion order."""
        _, ev = _evaluator(solver, tmp_path, numpy_bkd, concurrency=4)
        X = numpy_bkd.array([[float(j) for j in range(6)], [1.0] * 6])
        result = ev.submit(
            X, Request(values=False, jacobians=True)
        ).collect()
        assert result.jacobians is not None
        numpy_bkd.assert_allclose(
            result.jacobians, numpy_bkd.reshape(2.0 * X.T, (6, 1, 2))
        )


class TestHessianVectorProduct:
    """The direction is an input, which is why write_inputs exists.

    The Hessian of sum(x^2) is 2I, so the product is 2v regardless of
    x. That independence is deliberate: a direction that never reached
    the solver, or reached it for the wrong sample, gives a visibly
    wrong answer rather than one that happens to look plausible.
    """

    def test_the_direction_reaches_the_solver(
        self, solver, tmp_path, numpy_bkd
    ):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        X = numpy_bkd.array([[1.0, 2.0], [3.0, 4.0]])
        vecs = numpy_bkd.array([[1.0, 0.0], [0.0, 1.0]])
        result = ev.submit(
            X, Request(values=False, hvp_vecs=vecs)
        ).collect()
        assert result.hvps is not None
        numpy_bkd.assert_allclose(result.hvps, 2.0 * vecs.T)

    def test_each_sample_gets_its_own_direction(
        self, solver, tmp_path, numpy_bkd
    ):
        """The vectors span the batch, so a task must take its own slice.

        Passing the whole array to a one-sample task would compute the
        wrong width, which the decoded record then rejects.
        """
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        X = numpy_bkd.ones((2, 3))
        vecs = numpy_bkd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ev.submit(
            X, Request(values=False, hvp_vecs=vecs)
        ).collect()
        assert result.hvps is not None
        numpy_bkd.assert_allclose(result.hvps, 2.0 * vecs.T)

    def test_hvps_are_sample_first(self, solver, tmp_path, numpy_bkd):
        """(n, nvars), with no nqoi axis -- unlike jacobians.

        The batch form is scalar-implicit, which the derivative bundle
        calls out as a recurring trap when reshaping.
        """
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        vecs = numpy_bkd.ones((2, 4))
        result = ev.submit(
            numpy_bkd.ones((2, 4)), Request(values=False, hvp_vecs=vecs)
        ).collect()
        assert result.hvps is not None
        assert result.hvps.shape == (4, 2)

    def test_values_and_a_product_from_one_request(
        self, solver, tmp_path, numpy_bkd
    ):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        X = numpy_bkd.array([[1.0, 2.0], [3.0, 4.0]])
        vecs = numpy_bkd.array([[1.0, 1.0], [0.0, 2.0]])
        result = ev.submit(
            X, Request(values=True, hvp_vecs=vecs)
        ).collect()
        expected = numpy_bkd.array(
            [[float(X[0, j]) ** 2 + float(X[1, j]) ** 2 for j in range(2)]]
        )
        numpy_bkd.assert_allclose(result.values, expected)
        numpy_bkd.assert_allclose(result.hvps, 2.0 * vecs.T)


class TestSharedDirectoryLifetime:
    """A directory belongs to a sample; release is called per task.

    When one request becomes several invocations they share a working
    directory, so deleting it on the first release removes the inputs
    the second still has to read. That failure is silent: the second
    task finds no output where it expected one, and the caller gets a
    result with a missing derivative and nothing reported wrong.

    The other tests in this module would fail if this broke, but for
    reasons that read as "the jacobian is missing" rather than "the
    directory was deleted early", so the behavior is worth naming.
    """

    def _scratch(self, tmp_path):
        """Sample directories, wherever the run/submission levels put them.

        Globbing the scratch root directly would match nothing now that
        samples live under ``<run>/sub-NNN/``, and the assertions that
        expect an empty list would pass without testing anything.
        """
        return list((tmp_path / "scratch").glob("*/sub-*/sample-*"))

    def test_a_shared_directory_outlives_its_first_task(
        self, solver, tmp_path, numpy_bkd
    ):
        """Two invocations, one directory, discarded only at the end."""
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        result = ev.submit(
            numpy_bkd.ones((2, 2)), Request(values=True, jacobians=True)
        ).collect()
        # Both quantities survived, so neither task lost its directory.
        assert result.values.shape == (1, 2)
        assert result.jacobians is not None
        # And it was still cleaned up once every task had finished.
        assert self._scratch(tmp_path) == []

    def test_three_invocations_share_one_directory(
        self, solver, tmp_path, numpy_bkd
    ):
        """Values, jacobian and a product, all from one request."""
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        vecs = numpy_bkd.array([[1.0, 2.0], [3.0, 4.0]])
        result = ev.submit(
            numpy_bkd.ones((2, 2)),
            Request(values=True, jacobians=True, hvp_vecs=vecs),
        ).collect()
        assert result.values.shape == (1, 2)
        assert result.jacobians is not None
        assert result.hvps is not None
        numpy_bkd.assert_allclose(result.hvps, 2.0 * vecs.T)
        assert self._scratch(tmp_path) == []

    def test_a_failure_in_any_invocation_keeps_the_evidence(
        self, tmp_path, numpy_bkd
    ):
        """Retained on failure means *any* task failing, not the last.

        A solver whose values succeed and whose derivative fails has
        left the evidence in that directory, and releasing the
        successful task last must not discard it.
        """
        partial = tmp_path / "partial_solver.py"
        partial.write_text(
            textwrap.dedent(SOLVER).replace(
                'elif mode == "jacobian":',
                'elif mode == "jacobian":\n    sys.exit(3)\nelif False:',
            )
        )
        marshaller = DerivativeMarshaller(
            partial,
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
            retention=Retention.ON_FAILURE,
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=2))
        ev.submit(
            numpy_bkd.ones((2, 1)), Request(values=True, jacobians=True)
        ).collect()
        assert len(self._scratch(tmp_path)) == 1


class TestManifestWithSeveralInvocations:
    """One directory, several tasks, one record.

    ``release`` fires per task, so the naive write gives a
    three-invocation sample three release lines -- which a reader can
    only see as a duplicate -- and lets whichever task finished last
    decide what the run says happened.
    """

    def _released(self, marshaller):
        return [
            record
            for record in read_records(marshaller.manifest_path())
            if record["kind"] == KIND_RELEASED
        ]

    def test_three_invocations_produce_one_release_record(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(solver, tmp_path, numpy_bkd)
        vecs = numpy_bkd.array([[1.0], [3.0]])
        ev.submit(
            numpy_bkd.ones((2, 1)),
            Request(values=True, jacobians=True, hvp_vecs=vecs),
        ).collect()
        released = self._released(marshaller)
        assert len(released) == 1
        assert len(released[0]["tasks"]) == 3

    def test_each_invocation_keeps_its_own_outcome(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        marshaller, ev = _evaluator(solver, tmp_path, numpy_bkd)
        ev.submit(
            numpy_bkd.ones((2, 1)),
            Request(values=True, jacobians=True),
        ).collect()
        tasks = self._released(marshaller)[0]["tasks"]
        assert [task["status"] for task in tasks] == [
            "SUCCEEDED",
            "SUCCEEDED",
        ]

    def test_a_failure_in_one_invocation_marks_the_sample(
        self, tmp_path, numpy_bkd
    ) -> None:
        """Status is derived, and agrees with what the caller was told.

        The values invocation succeeds and the jacobian one does not.
        Success is per quantity, so the evaluator reports this sample
        SUCCEEDED and puts it in ``succeeded`` -- a manifest recording
        FAILED would contradict the result handed back. ``any_failed``
        and the per-task list carry what actually went wrong, and
        neither depends on which invocation finished first.
        """
        partial = tmp_path / "partial_solver.py"
        partial.write_text(
            textwrap.dedent(SOLVER).replace(
                'elif mode == "jacobian":',
                'elif mode == "jacobian":\n    sys.exit(3)\nelif False:',
            )
        )
        marshaller = DerivativeMarshaller(
            partial,
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
            retention=Retention.ON_FAILURE,
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=2))
        ev.submit(
            numpy_bkd.ones((2, 1)), Request(values=True, jacobians=True)
        ).collect()
        record = self._released(marshaller)[0]
        assert record["any_failed"] is True
        # SUCCEEDED, matching what the evaluator reported: the values
        # invocation produced its quantity. The failure is not hidden --
        # any_failed and the task list below carry it.
        assert record["status"] == "SUCCEEDED"
        assert record["retained"] is True
        # Both invocations are visible, so "which part failed" is
        # answerable rather than collapsed into one word.
        assert sorted(task["status"] for task in record["tasks"]) == [
            "FAILED",
            "SUCCEEDED",
        ]


class TestTheManifestAgreesWithTheResult:
    """One sample must not be SUCCEEDED in one place and FAILED in another.

    The evaluator reports success per quantity, so a sample whose
    values decoded lands in ``succeeded`` even when its jacobian
    invocation failed. A manifest calling the same sample FAILED would
    make the run's own record contradict what the caller was handed,
    and reconciliation -- which reads the manifest and takes the
    caller's statuses -- would see the two disagree.
    """

    def test_status_matches_what_the_batch_reported(
        self, tmp_path, numpy_bkd
    ) -> None:
        partial = tmp_path / "partial_solver.py"
        partial.write_text(
            textwrap.dedent(SOLVER).replace(
                'elif mode == "jacobian":',
                'elif mode == "jacobian":\n    sys.exit(3)\nelif False:',
            )
        )
        marshaller = DerivativeMarshaller(
            partial,
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
            retention=Retention.ALWAYS,
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=2))
        batch = ev.submit(
            numpy_bkd.ones((2, 1)), Request(values=True, jacobians=True)
        )
        batch.collect()
        record = [
            r
            for r in read_records(marshaller.manifest_path())
            if r["kind"] == KIND_RELEASED
        ][0]
        assert record["status"] == batch.statuses()[0].name


class TestReconcilingSeveralInvocations:
    """A partly-failed sample must not read as one that wrote nothing.

    Success is per quantity. A sample whose values succeeded and whose
    jacobian failed is legitimately missing that quantity's file, so
    checking the required-output rule before the partial-failure rule
    would report every such sample as a solver that exited zero having
    written nothing -- the exact false positive that would make the
    check untrustworthy on any run asking for derivatives.
    """

    def _partial_solver(self, tmp_path):
        """A solver whose jacobian invocation fails."""
        path = tmp_path / "partial_solver.py"
        path.write_text(
            textwrap.dedent(SOLVER).replace(
                'elif mode == "jacobian":',
                'elif mode == "jacobian":\n    sys.exit(3)\nelif False:',
            )
        )
        return path

    def test_a_partial_sample_is_reported_as_partial(
        self, tmp_path, numpy_bkd
    ) -> None:
        marshaller = DerivativeMarshaller(
            self._partial_solver(tmp_path),
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
            retention=Retention.ALWAYS,
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=2))
        batch = ev.submit(
            numpy_bkd.ones((2, 1)), Request(values=True, jacobians=True)
        )
        batch.collect()
        report = reconcile(
            marshaller.run_dir(),
            statuses=batch.statuses(),
            collector=SpecCollector(
                [OutputSpec("jacobian.out", required=True)]
            ),
        )
        assert report.ok()
        assert 0 in report.partial
        assert not any(
            a.kind is AnomalyKind.NO_OUTPUT for a in report.anomalies
        )

    def test_a_fully_successful_run_still_reconciles(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """The rule above must not excuse everything."""
        marshaller, ev = _evaluator(
            solver, tmp_path, numpy_bkd, retention=Retention.ALWAYS
        )
        batch = ev.submit(
            numpy_bkd.ones((2, 2)), Request(values=True, jacobians=True)
        )
        batch.collect()
        report = reconcile(
            marshaller.run_dir(),
            statuses=batch.statuses(),
            collector=SpecCollector(
                [OutputSpec("jacobian.out", required=True)]
            ),
        )
        assert report.ok()
        assert report.partial == {}
        assert sorted(report.ok_indices) == [0, 1]

    def test_a_missing_output_is_still_caught_for_derivatives(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """Row 7 excuses a partial sample, not a silent one."""
        marshaller, ev = _evaluator(
            solver, tmp_path, numpy_bkd, retention=Retention.ALWAYS
        )
        batch = ev.submit(
            numpy_bkd.ones((2, 2)), Request(values=True, jacobians=True)
        )
        batch.collect()
        report = reconcile(
            marshaller.run_dir(),
            statuses=batch.statuses(),
            collector=SpecCollector([OutputSpec("*.nope", required=True)]),
        )
        assert not report.ok()
        assert all(
            a.kind is AnomalyKind.NO_OUTPUT for a in report.anomalies
        )

    def test_gathering_a_shared_directory_does_not_double_count(
        self, solver, tmp_path, numpy_bkd
    ) -> None:
        """Three invocations, one directory, gathered once."""
        marshaller, ev = _evaluator(
            solver, tmp_path, numpy_bkd, retention=Retention.ALWAYS
        )
        vecs = numpy_bkd.array([[1.0], [3.0]])
        ev.submit(
            numpy_bkd.ones((2, 1)),
            Request(values=True, jacobians=True, hvp_vecs=vecs),
        ).collect()
        report = gather_run(
            marshaller.run_dir(),
            SpecCollector([OutputSpec("*.out")], skip=("params.in",)),
            str(tmp_path / "archive"),
        )
        assert sorted(report.gathered) == [0]
        names = sorted(
            os.path.basename(path) for path in report.gathered[0]
        )
        assert names == ["hvp.out", "jacobian.out", "results.out"]


class TestInheritedBehaviourIsUnchanged:
    """What a subclass does not have to restate.

    The seams are worth having only if overriding them leaves
    everything else alone.
    """

    def test_one_task_per_sample_still_holds(
        self, solver, tmp_path, numpy_bkd
    ):
        marshaller, _ = _evaluator(solver, tmp_path, numpy_bkd)
        assert marshaller.max_samples_per_task() == 1

    def test_retention_still_applies(self, solver, tmp_path, numpy_bkd):
        _, ev = _evaluator(solver, tmp_path, numpy_bkd)
        ev.submit(
            numpy_bkd.ones((2, 2)), Request(jacobians=True)
        ).collect()
        assert list((tmp_path / "scratch").glob("*/sub-*/sample-*")) == []

    def test_failures_are_still_per_sample(
        self, solver, tmp_path, numpy_bkd
    ):
        """A derivative task that cannot run fails only its own sample."""
        marshaller = DerivativeMarshaller(
            tmp_path / "absent_solver.py",
            bkd=numpy_bkd,
            nvars=2,
            nqoi=1,
            scratch_root=str(tmp_path / "scratch"),
        )
        ev = Evaluator(marshaller, SubprocessDispatcher(concurrency=2))
        result = ev.submit(
            numpy_bkd.ones((2, 2)), Request(values=False, jacobians=True)
        ).collect()
        assert result.nfailed() == 2
