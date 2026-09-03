"""Non-blocking model evaluation: submit work, ask about it, collect it.

A model that takes hours per sample is not merely slow to call. While it
runs, the workflow has state worth asking about, and a blocking call has
none. This subpackage separates the three concerns that a blocking
external-solver wrapper fuses together:

- **Dispatch** -- how work is launched, checked and throttled. Varies
  with the *machine*. A dispatcher never sees an ``Array``, a
  ``Backend``, or a file format; it launches opaque tasks and reports on
  them, so it is not generic in ``Array`` at all.
- **Marshalling** -- sample to input, output to values. Varies with the
  *code being wrapped*. This is also where the mathematics lives:
  ``MarshallerProtocol`` carries ``bkd()``, ``nvars()`` and ``nqoi()``
  because for an external solver "the problem" and "the code being
  wrapped" are the same axis. You do not change the input format without
  changing the solver, and you do not change the solver without changing
  the mathematics.
- **Composition** -- ``Evaluator`` and ``Batch``, which group samples
  into tasks, record failures, buffer what has already been returned,
  and accumulate cost.

The currency between dispatch and marshalling is two type variables.
``Task`` is what the marshaller builds and the dispatcher runs;
``Payload`` is what a finished job produced and the marshaller decodes.
Both erase at the consumer-facing seam -- the concrete evaluator is
``Generic[Array, Task, Payload]`` but satisfies
``EvaluatorProtocol[Array]`` -- so callers never name either.

**Implementations are injected, not registered.** There is no registry,
no name-to-class table and no if/elif factory here, and nothing in this
package enumerates the implementations that exist. Pairing a marshaller
with a dispatcher is checked *statically*::

    evaluator = Evaluator(
        TextFileMarshaller(bkd, nqoi=2, command="./solver"),
        SubprocessDispatcher(concurrency=8),
    )

Handing a marshaller that produces one task family to a dispatcher that
accepts another fails at the call site with ``Cannot infer value of type
parameter "Task"``. A registry cannot do this: ``create(name, **kwargs)``
erases both the constructor arguments and the task family, so the same
mistake either raises at runtime or silently mispairs. Nor can the
``isinstance`` backstop, since a subscripted protocol cannot be used with
``isinstance`` at all.
"""

from pyapprox.interface.evaluation.caching import (
    CachedObjective,
    RoundedHashLookup,
)
from pyapprox.interface.evaluation.collection import (
    DEFAULT_SKIP,
    Anomaly,
    AnomalyKind,
    CollectionError,
    CollectionReport,
    GatherReport,
    OutputCollectorProtocol,
    OutputSpec,
    SpecCollector,
    TransferMode,
    TransferResult,
    gather_into,
    gather_run,
    reconcile,
)
from pyapprox.interface.evaluation.ensemble import (
    Ensemble,
    EnsembleBatch,
    EnsembleProgress,
    ModelId,
)
from pyapprox.interface.evaluation.manifest import (
    RUN_DONE_FILENAME,
    ManifestWriter,
    read_records,
)
from pyapprox.interface.evaluation.protocols import (
    BatchProtocol,
    DispatcherProtocol,
    EvaluatorProtocol,
    JobHandle,
    MarshalError,
    MarshallerProtocol,
    ResultStore,
    SampleLookup,
    SubmissionAware,
)
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    CostLedger,
    Decoded,
    EvalProgress,
    EvalResult,
    JobStatus,
    Outcome,
    Request,
    Resources,
    TimeSource,
)
from pyapprox.interface.evaluation.stores import (
    InMemoryResultStore,
    NpzResultStore,
    PickleResultStore,
    StoreWriter,
    restore_columns,
    stored_indices,
)
from pyapprox.interface.evaluation.subprocess_dispatcher import (
    ShellPayload,
    ShellTask,
    SubprocessDispatcher,
)
from pyapprox.interface.evaluation.textfile_marshaller import (
    OnExisting,
    Retention,
    TextFileMarshaller,
)

__all__ = [
    # Ensemble
    "Ensemble",
    "EnsembleBatch",
    "EnsembleProgress",
    "ModelId",
    # Records
    "ComputeProvenance",
    "Cost",
    "CostLedger",
    "Decoded",
    "EvalProgress",
    "EvalResult",
    "JobStatus",
    "Outcome",
    "Request",
    "Resources",
    "TimeSource",
    # Protocols -- exported because writing a dispatcher or a marshaller
    # needs all of them.
    "BatchProtocol",
    "DispatcherProtocol",
    "EvaluatorProtocol",
    "JobHandle",
    "MarshalError",
    "MarshallerProtocol",
    "ResultStore",
    "SampleLookup",
    "SubmissionAware",
    # Caching -- serving a quantity from a store rather than recomputing
    "CachedObjective",
    "RoundedHashLookup",
    # Result stores -- results that outlive the process
    "InMemoryResultStore",
    "NpzResultStore",
    "PickleResultStore",
    "StoreWriter",
    "restore_columns",
    "stored_indices",
    # External solvers
    "OnExisting",
    "Retention",
    "ShellPayload",
    "ShellTask",
    "SubprocessDispatcher",
    "TextFileMarshaller",
    # Run manifests -- what happened to each working directory
    "RUN_DONE_FILENAME",
    "ManifestWriter",
    "read_records",
    # Output collection -- getting a solver's own files off scratch
    "DEFAULT_SKIP",
    "Anomaly",
    "AnomalyKind",
    "CollectionError",
    "CollectionReport",
    "GatherReport",
    "OutputCollectorProtocol",
    "OutputSpec",
    "SpecCollector",
    "TransferMode",
    "TransferResult",
    "gather_into",
    "gather_run",
    "reconcile",
]
