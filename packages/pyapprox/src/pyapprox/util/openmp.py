"""One OpenMP runtime per process on macOS.

The PyTorch pip wheel ships its own OpenMP runtime (``torch/lib/libomp.dylib``)
and header (``torch/include/omp.h``). An environment may hold a second
runtime, most often conda's ``llvm-openmp``, which many conda-forge packages
depend on. Two runtimes initialized in one process abort it with
``OMP: Error #15``. Two consumers can load the second one:

- ``torch.compile``'s CPU kernels, which link whichever OpenMP inductor finds
  first. Inductor reads ``OMP_PREFIX`` before the conda environment and
  Homebrew, so pointing it at torch's own install folder makes compiled
  kernels use torch's runtime.
- numba's ``omp`` threading layer, which links the environment's runtime.
  Ranking it last keeps numba on ``tbb`` (when installed) or ``workqueue``.

Both functions apply only on macOS, only when torch is installed, and never
override a setting the user made; calling them again does nothing.
"""

import importlib.util
import os
import sys

NUMBA_LAYER_PRIORITY = ["tbb", "workqueue", "omp"]


def _torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def use_torch_openmp() -> None:
    """Point ``torch.compile`` at the OpenMP runtime in torch's own wheel.

    Sets ``OMP_PREFIX`` to the torch package folder when it holds
    ``include/omp.h``, on macOS and only if ``OMP_PREFIX`` is unset. Call
    before the first ``torch.compile`` call compiles a kernel.
    """
    if sys.platform != "darwin" or "OMP_PREFIX" in os.environ:
        return
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return
    prefix = os.path.dirname(spec.origin)
    if os.path.exists(os.path.join(prefix, "include", "omp.h")):
        os.environ["OMP_PREFIX"] = prefix


def keep_numba_off_openmp() -> None:
    """Rank numba's ``omp`` threading layer last when torch is installed.

    On macOS, unless ``NUMBA_THREADING_LAYER`` or
    ``NUMBA_THREADING_LAYER_PRIORITY`` is set. numba chooses its layer at
    the first parallel launch, so call this before any parallel kernel runs.

    The order goes into ``NUMBA_THREADING_LAYER_PRIORITY`` as well as
    numba's config: numba's compiler rebuilds its config from the
    environment whenever a ``NUMBA_*`` variable changes, which would drop
    a config-only setting.
    """
    if sys.platform != "darwin" or not _torch_installed():
        return
    if (
        "NUMBA_THREADING_LAYER" in os.environ
        or "NUMBA_THREADING_LAYER_PRIORITY" in os.environ
    ):
        return
    import numba

    os.environ["NUMBA_THREADING_LAYER_PRIORITY"] = " ".join(NUMBA_LAYER_PRIORITY)
    numba.config.THREADING_LAYER_PRIORITY = list(NUMBA_LAYER_PRIORITY)
