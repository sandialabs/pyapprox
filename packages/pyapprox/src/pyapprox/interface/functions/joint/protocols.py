"""Protocol for evaluating observations and targets together."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.interface.functions.joint.outputs import JointOutputs
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class JointEvaluatorProtocol(Protocol, Generic[Array]):
    """Evaluates observations and target blocks at the same samples.

    Whether this takes one model run or several is the implementation's
    business; consumers see only the outputs.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of input variables.
    nobs()
        Number of observations.
    target_sizes()
        Number of rows in each target block.
    evaluate(samples)
        Observations and targets at the samples.
    """

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        ...

    def nvars(self) -> int:
        """Number of input variables."""
        ...

    def nobs(self) -> int:
        """Number of observations."""
        ...

    def target_sizes(self) -> tuple[int, ...]:
        """Number of rows in each target block, in order."""
        ...

    def evaluate(self, samples: Array) -> JointOutputs[Array]:
        """Evaluate observations and targets.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        JointOutputs[Array]
            Observations of shape (nobs, nsamples) and one block per
            target of shape (target_sizes[k], nsamples).
        """
        ...
