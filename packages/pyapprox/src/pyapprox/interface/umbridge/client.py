"""UMBridge model client.

This module provides a client for UMBridge (Unified Model Bridge) models,
which allows calling external models via HTTP.

UMBridge is a unified interface for numerical models accessible from
virtually any programming language or framework. It is primarily intended
for coupling advanced models (e.g., simulations of complex physical
processes) to advanced statistical or optimization methods.

See: https://um-bridge-benchmarks.readthedocs.io/
"""

import os
import signal
import subprocess
import time
from typing import IO, Any, Dict, Generic, List, Optional, Tuple

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available

UMBRIDGE_AVAILABLE = package_available("umbridge")


class UMBridgeModel(Generic[Array]):
    """HTTP client for UMBridge models.

    This class provides an interface for evaluating UMBridge models via HTTP.
    UMBridge models can run in Docker containers, on remote servers, or locally.

    Parameters
    ----------
    url : str
        The URL of the UMBridge server (e.g., "http://localhost:4242").
    model_name : str
        The name of the model to use (as registered with the server).
    bkd : Backend[Array]
        The backend for array operations.
    config : dict, optional
        Configuration dictionary for the UMBridge model. Default is {}.

    Raises
    ------
    ImportError
        If the umbridge package is not installed.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.interface.umbridge import UMBridgeModel
    >>> bkd = NumpyBkd()
    >>> # Start a UMBridge server first, then:
    >>> model = UMBridgeModel("http://localhost:4242", "forward", bkd)
    >>> samples = bkd.asarray([[0.5], [0.5]])
    >>> values = model(samples)
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        bkd: Backend[Array],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the UMBridge model client.

        Parameters
        ----------
        url : str
            The URL of the UMBridge server.
        model_name : str
            The name of the model to use.
        bkd : Backend[Array]
            The backend for array operations.
        config : dict, optional
            Configuration dictionary for the UMBridge model.
        """
        from pyapprox.util.optional_deps import import_optional_dependency

        import_optional_dependency(
            "umbridge", feature_name="UMBridgeModel", extra_name="umbridge"
        )
        import umbridge

        self._bkd = bkd
        self._url = url
        self._model_name = model_name
        self._config: Dict[str, Any] = config if config is not None else {}

        # Create HTTP model connection
        self._model: umbridge.HTTPModel = umbridge.HTTPModel(url, model_name)

        # Capability is whatever the server advertises, resolved once
        # here rather than per call. A server may support any subset --
        # apply_jacobian without gradient is legal -- so the raw
        # constructor is used rather than a named one, which would
        # require a jacobian that this server need not provide.
        self._derivs: Derivatives[Array] = Derivatives(
            jacobian=(
                self._jacobian if self._model.supports_gradient() else None
            ),
            jvp=(
                self._jvp
                if self._model.supports_apply_jacobian()
                else None
            ),
            hvp=(
                self._hvp
                if self._model.supports_apply_hessian()
                else None
            ),
        )

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables.

        Returns
        -------
        int
            The number of input variables.
        """
        return int(self._model.get_input_sizes(self._config)[0])

    def nqoi(self) -> int:
        """Return the number of output quantities of interest.

        Returns
        -------
        int
            The number of output QoIs.
        """
        return int(self._model.get_output_sizes(self._config)[0])

    def config(self) -> Dict[str, Any]:
        """Return the model configuration."""
        return self._config

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set the model configuration.

        Parameters
        ----------
        config : dict
            The new configuration dictionary.
        """
        self._config = config

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative capability the server advertises.

        The bundle is built once at construction from the server's
        ``supports_*`` queries, so capability is fixed for the lifetime
        of the client and each query costs no HTTP round trip.

        Returns
        -------
        Derivatives[Array]
            Populated for each capability the server advertises; every
            other field is None.
        """
        return self._derivs

    def _to_parameters(self, sample: Array) -> List[List[float]]:
        """Convert a sample to UMBridge parameter format.

        Parameters
        ----------
        sample : Array
            Input sample of shape (nvars, 1).

        Returns
        -------
        list
            Parameters in UMBridge format [[p1, p2, ...]].
        """
        if sample.ndim != 2 or sample.shape[1] != 1:
            raise ValueError(f"Expected sample shape (nvars, 1), got {sample.shape}")
        return [self._bkd.to_numpy(sample[:, 0]).tolist()]

    def __call__(self, samples: Array) -> Array:
        """Evaluate the model at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples of shape (nvars, nsamples).

        Returns
        -------
        Array
            Output values of shape (nqoi, nsamples).
        """
        if samples.ndim != 2:
            raise ValueError(
                f"Expected samples shape (nvars, nsamples), got {samples.shape}"
            )

        nsamples = samples.shape[1]
        results = []

        for ii in range(nsamples):
            sample = samples[:, ii : ii + 1]
            parameters = self._to_parameters(sample)
            result = self._model(parameters, config=self._config)[0]
            results.append(result)

        # Stack results: each result is a list of length nqoi
        values = self._bkd.asarray(results).T  # (nqoi, nsamples)
        return values

    def _jacobian(self, sample: Array) -> Array:
        """Compute the Jacobian at a single sample.

        Reached through ``derivatives().jacobian``, which is None unless
        the server advertises gradient support.

        Parameters
        ----------
        sample : Array
            Input sample of shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix of shape (nqoi, nvars).
        """
        parameters = self._to_parameters(sample)
        # UMBridge gradient returns sens^T J, a vector of length nvars.
        # Seeding sens with the qq-th unit vector selects row qq, so the
        # jacobian is assembled one row per request.
        #
        # out_wrt and in_wrt index output and input *blocks*, not
        # components within a block. This client exposes a single block
        # of each, so both stay 0 for every row; passing qq there asks
        # for a block that does not exist.
        nqoi = self.nqoi()
        nvars = self.nvars()
        jacobian = self._bkd.zeros((nqoi, nvars))

        for qq in range(nqoi):
            sens = [0.0] * nqoi
            sens[qq] = 1.0
            grad = self._model.gradient(
                0, 0, parameters, sens, config=self._config
            )
            jacobian[qq, :] = self._bkd.asarray(grad)

        return jacobian

    def _jvp(self, sample: Array, vec: Array) -> Array:
        """Compute Jacobian-vector product.

        Reached through ``derivatives().jvp``, which is None unless the
        server advertises apply_jacobian support.

        Parameters
        ----------
        sample : Array
            Input sample of shape (nvars, 1).
        vec : Array
            Direction vector of shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian-vector product of shape (nqoi, 1).
        """
        parameters = self._to_parameters(sample)
        vec_list = self._bkd.to_numpy(vec[:, 0]).tolist()

        # out_wrt/in_wrt are output and input block indices, not optional
        # flags; the server rejects None. This client exposes a single
        # input and a single output block, so both are 0.
        result = self._model.apply_jacobian(
            0, 0, parameters, vec_list, config=self._config
        )
        return self._bkd.reshape(self._bkd.asarray(result), (-1, 1))

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product.

        This is only valid for scalar functions (nqoi == 1). Reached
        through ``derivatives().hvp``, which is None unless the server
        advertises apply_hessian support.

        Parameters
        ----------
        sample : Array
            Input sample of shape (nvars, 1).
        vec : Array
            Direction vector of shape (nvars, 1).

        Returns
        -------
        Array
            Hessian-vector product of shape (nvars, 1).

        Raises
        ------
        ValueError
            If nqoi != 1, or if the server returns a vector whose length
            is not nvars.

        Notes
        -----
        The Hessian is nvars x nvars, so H v has length nvars. UM-Bridge
        specifies this, and the C++ reference server does not constrain
        the length. The Python reference server (as of umbridge 1.2.4)
        instead validates the reply against the *output* size, so a
        conformant model serving nvars values through it is rejected
        with an InvalidOutput error whenever nvars != nqoi. That is a
        defect in that server, not a different convention: honoring it
        would mean returning nqoi numbers for a quantity that has nvars
        of them. This client sends and expects the specified length, so
        it works against C++ servers and against a fixed Python one.
        """
        if self.nqoi() != 1:
            raise ValueError(f"HVP only defined for nqoi=1, got nqoi={self.nqoi()}")

        parameters = self._to_parameters(sample)
        vec_list = self._bkd.to_numpy(vec[:, 0]).tolist()

        # Signature is (out_wrt, in_wrt1, in_wrt2, parameters, sens, vec):
        # sens precedes vec. The block indices are 0 for this client's
        # single input and output block. sens seeds the adjoint of the
        # scalar output, so it is [1.0].
        result = self._model.apply_hessian(
            0, 0, 0, parameters, [1.0], vec_list, config=self._config
        )
        # Checked rather than reshaped blindly: a server returning nqoi
        # numbers would otherwise yield a plausible (1, 1) array that is
        # not an HVP.
        if len(result) != self.nvars():
            raise ValueError(
                f"Server returned {len(result)} values for a "
                f"Hessian-vector product; expected nvars="
                f"{self.nvars()}. The model may be returning a vector "
                "sized by its output rather than its input."
            )
        return self._bkd.reshape(self._bkd.asarray(result), (-1, 1))

    @staticmethod
    def start_server(
        run_command: str,
        url: str = "http://localhost:4242",
        out: Optional[IO[Any]] = None,
        max_wait_time: int = 20,
    ) -> Tuple[subprocess.Popen[bytes], Optional[IO[Any]]]:
        """Start a UMBridge server subprocess.

        This is a utility method for testing. It starts a server subprocess
        and waits for it to become available.

        Parameters
        ----------
        run_command : str
            Command to start the server (e.g., "python my_server.py").
        url : str, optional
            URL of the server. Default is "http://localhost:4242".
        out : file-like object, optional
            Output stream for server logs. If None, uses os.devnull.
        max_wait_time : int, optional
            Maximum seconds to wait for server. Default is 20.

        Returns
        -------
        process : subprocess.Popen
            The server process.
        out : file-like object
            The output stream (for cleanup).

        Raises
        ------
        RuntimeError
            If the server does not start within max_wait_time.
        """
        from pyapprox.util.optional_deps import import_optional_dependency

        import_optional_dependency(
            "umbridge", feature_name="UMBridgeModel", extra_name="umbridge"
        )
        import requests

        if out is None:
            out = open(os.devnull, "w")

        import shlex

        process = subprocess.Popen(
            shlex.split(run_command),
            stdout=out,
            stderr=out,
            preexec_fn=os.setsid,
        )

        print(f"Starting server: {run_command}")
        t0 = time.time()

        while True:
            try:
                requests.get(os.path.join(url, "Info"))
                print("Server running")
                break
            except requests.exceptions.ConnectionError:
                if time.time() - t0 > max_wait_time:
                    UMBridgeModel.kill_server(process, out)
                    raise RuntimeError(
                        f"Could not connect to server at {url}"
                    ) from None
                time.sleep(0.5)

        return process, out

    @staticmethod
    def kill_server(
        process: subprocess.Popen[bytes],
        out: Optional[IO[Any]] = None,
    ) -> None:
        """Kill a UMBridge server subprocess.

        Parameters
        ----------
        process : subprocess.Popen
            The server process to kill.
        out : file-like object, optional
            The output stream to close.
        """
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already terminated
        if out is not None:
            out.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"UMBridgeModel(url={self._url!r}, model={self._model_name!r}, "
            f"nvars={self.nvars()}, nqoi={self.nqoi()})"
        )
