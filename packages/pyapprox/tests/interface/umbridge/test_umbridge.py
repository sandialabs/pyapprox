"""Tests for UMBridgeModel client.

These tests spawn a real UMBridge server subprocess and exercise the
client against it. The server is a module-scoped fixture rather than
setup_class so that a failure part way through startup still tears down
what was created, and it binds a port claimed from the OS rather than a
fixed one so that parallel test runs do not collide.
"""

import os
import socket
import subprocess
import sys
from typing import Iterator

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("umbridge"):
    pytest.skip("umbridge not installed", allow_module_level=True)

from pyapprox.interface.umbridge.client import UMBridgeModel
from pyapprox.util.backends.numpy import NumpyBkd


def _free_port() -> int:
    """Claim a port from the OS and release it for the server to bind.

    Binding to port 0 makes the kernel pick an unused port. There is a
    race between closing here and the server binding, but it is far
    narrower than the certain collision of a hardcoded port under
    parallel test execution.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    """Start the test server, yield its URL, and always kill it."""
    port = _free_port()
    url = f"http://localhost:{port}"
    server_script = os.path.join(
        os.path.dirname(__file__), "test_umbridge_server.py"
    )
    # sys.executable, not "python": the interpreter running the tests is
    # the one with umbridge installed, which whatever "python" resolves
    # to on PATH need not be.
    run_command = f"{sys.executable} {server_script} {port}"

    process, out = UMBridgeModel.start_server(
        run_command, url=url, max_wait_time=30
    )
    try:
        yield url
    finally:
        UMBridgeModel.kill_server(process, out)


@pytest.fixture
def bkd() -> NumpyBkd:
    """UMBridge speaks JSON lists, so the backend is NumPy only."""
    return NumpyBkd()


def _model(url: str, name: str, bkd: NumpyBkd, nvars: int = 2) -> UMBridgeModel:
    return UMBridgeModel(url, name, bkd, config={"nvars": nvars})


class TestEvaluation:
    """Values, shapes and the sample-as-column convention."""

    def test_single_sample(self, server_url, bkd) -> None:
        model = _model(server_url, "quadratic", bkd)
        # f([1, 2]) = 1^2 + 2^2 = 5
        values = model(bkd.asarray([[1.0], [2.0]]))
        assert values.shape == (1, 1)
        bkd.assert_allclose(values, bkd.asarray([[5.0]]))

    def test_batch_preserves_sample_order(self, server_url, bkd) -> None:
        """Column j of the result must be f of column j of the input.

        The client loops over samples and stacks, so a transpose slip
        would still produce the right shape with the values permuted.
        Distinct values per sample are what catch that.
        """
        model = _model(server_url, "quadratic", bkd)
        samples = bkd.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        values = model(samples)
        assert values.shape == (1, 3)
        # 1+4=5, 4+9=13, 9+16=25
        bkd.assert_allclose(values, bkd.asarray([[5.0, 13.0, 25.0]]))

    def test_vector_valued_output(self, server_url, bkd) -> None:
        """nqoi > 1 fixes the (nqoi, nsamples) orientation."""
        model = _model(server_url, "vector", bkd)
        samples = bkd.asarray([[1.0, 3.0], [2.0, 4.0]])
        values = model(samples)
        assert values.shape == (2, 2)
        # rows: [sum x^2, sum x] for each of the two samples
        bkd.assert_allclose(
            values, bkd.asarray([[5.0, 25.0], [3.0, 7.0]])
        )

    def test_rejects_one_dimensional_samples(self, server_url, bkd) -> None:
        model = _model(server_url, "quadratic", bkd)
        with pytest.raises(ValueError, match="nvars, nsamples"):
            model(bkd.asarray([1.0, 2.0]))

    def test_nvars_nqoi(self, server_url, bkd) -> None:
        model = _model(server_url, "quadratic", bkd, nvars=3)
        assert model.nvars() == 3
        assert model.nqoi() == 1

    def test_config_update_changes_nvars(self, server_url, bkd) -> None:
        model = _model(server_url, "quadratic", bkd)
        assert model.nvars() == 2
        model.set_config({"nvars": 3})
        assert model.nvars() == 3

    def test_repr(self, server_url, bkd) -> None:
        repr_str = repr(_model(server_url, "quadratic", bkd))
        assert "UMBridgeModel" in repr_str
        assert "quadratic" in repr_str


class TestCapabilityBundle:
    """Capability is read from derivatives(); absence is None."""

    def test_full_capability_populated(self, server_url, bkd) -> None:
        d = _model(server_url, "quadratic", bkd).derivatives()
        assert d.jacobian is not None
        assert d.jvp is not None
        assert d.hvp is not None

    def test_absent_capability_is_none(self, server_url, bkd) -> None:
        """A server advertising no derivatives yields None, not a raise.

        The whole point of the bundle is that a caller can branch on the
        field instead of calling and catching.
        """
        d = _model(server_url, "linear", bkd).derivatives()
        assert d.jacobian is None
        assert d.jvp is None
        assert d.hvp is None

    def test_partial_capability(self, server_url, bkd) -> None:
        """gradient without apply_jacobian/apply_hessian is legal."""
        d = _model(server_url, "vector", bkd).derivatives()
        assert d.jacobian is not None
        assert d.jvp is None
        assert d.hvp is None

    def test_batch_fields_absent(self, server_url, bkd) -> None:
        """The client evaluates one sample per request, so no batch form.

        Populating these would claim a vectorized path the HTTP protocol
        does not provide.
        """
        d = _model(server_url, "quadratic", bkd).derivatives()
        assert d.jacobian_batch is None
        assert d.hvp_batch is None
        assert d.whvp_batch is None
        assert d.hessian_batch is None

    def test_bundle_is_stable(self, server_url, bkd) -> None:
        """Capability is resolved once, not re-queried per access."""
        model = _model(server_url, "quadratic", bkd)
        assert model.derivatives() is model.derivatives()


class TestDerivativeValues:
    """Derivatives are checked against closed-form values."""

    def test_jacobian(self, server_url, bkd) -> None:
        model = _model(server_url, "quadratic", bkd)
        jacobian = model.derivatives().jacobian
        assert jacobian is not None
        # d/dx sum(x_i^2) = 2x, so at [1, 2] it is [2, 4]
        result = jacobian(bkd.asarray([[1.0], [2.0]]))
        assert result.shape == (1, 2)
        bkd.assert_allclose(result, bkd.asarray([[2.0, 4.0]]))

    def test_jacobian_multiple_outputs(self, server_url, bkd) -> None:
        """Each output row is fetched with its own seed vector."""
        model = _model(server_url, "vector", bkd)
        jacobian = model.derivatives().jacobian
        assert jacobian is not None
        result = jacobian(bkd.asarray([[1.0], [2.0]]))
        assert result.shape == (2, 2)
        # row 0: d(sum x^2) = 2x = [2, 4]; row 1: d(sum x) = [1, 1]
        bkd.assert_allclose(
            result, bkd.asarray([[2.0, 4.0], [1.0, 1.0]])
        )

    def test_jvp_matches_jacobian_product(self, server_url, bkd) -> None:
        """J v computed by the server equals the assembled J times v."""
        model = _model(server_url, "quadratic", bkd)
        d = model.derivatives()
        assert d.jvp is not None and d.jacobian is not None
        sample = bkd.asarray([[1.0], [2.0]])
        vec = bkd.asarray([[3.0], [-1.0]])
        result = d.jvp(sample, vec)
        assert result.shape == (1, 1)
        # J = [2, 4]; J v = 6 - 4 = 2
        bkd.assert_allclose(result, bkd.asarray([[2.0]]))
        bkd.assert_allclose(result, d.jacobian(sample) @ vec)

    def test_hvp_single_variable(self, server_url, bkd) -> None:
        """H v end to end, restricted to nvars == 1 by an upstream bug.

        An HVP is defined for a scalar output, so nqoi must be 1. The
        Python reference server has a bug (see
        ``test_upstream_hessian_length_bug_still_present``) that also
        demands nvars == nqoi for this call. Together they leave only
        nvars == nqoi == 1, which is why this test is narrower than the
        capability it covers.

        Against a C++ server, or once the bug is fixed, this path works
        for any nvars; widen this test to a multi-variable model at that
        point.
        """
        model = _model(server_url, "quadratic", bkd, nvars=1)
        hvp = model.derivatives().hvp
        assert hvp is not None
        # Hessian of x^2 is 2, so H v = 2 * 3 = 6
        result = hvp(bkd.asarray([[2.0]]), bkd.asarray([[3.0]]))
        assert result.shape == (1, 1)
        bkd.assert_allclose(result, bkd.asarray([[6.0]]))

    def test_hvp_reply_length_is_checked(self, server_url, bkd) -> None:
        """A reply that is not nvars long raises instead of reshaping.

        Guards the silent-corruption case: reshaping a wrong-length
        reply to (-1, 1) yields a plausible array that is not an HVP.
        The model here returns an output-sized vector, which is what a
        model written against the Python server's check would do.
        """
        model = _model(server_url, "short_hessian", bkd, nvars=3)
        hvp = model.derivatives().hvp
        assert hvp is not None
        with pytest.raises(ValueError, match="expected nvars=3"):
            hvp(
                bkd.asarray([[1.0], [2.0], [3.0]]),
                bkd.asarray([[1.0], [0.0], [0.0]]),
            )

    def test_hvp_rejects_vector_valued_model(self, server_url, bkd) -> None:
        """hvp is scalar-only; nqoi > 1 must raise, not return nonsense.

        The quadratic model advertises apply_hessian and reports nqoi=1,
        so the guard is reached by pointing the same client at a config
        the server answers with two outputs.
        """
        model = _model(server_url, "quadratic", bkd)
        hvp = model.derivatives().hvp
        assert hvp is not None
        model._model = UMBridgeModel(
            server_url, "vector", bkd, config={"nvars": 2}
        )._model
        assert model.nqoi() == 2
        with pytest.raises(ValueError, match="nqoi=2"):
            hvp(
                bkd.asarray([[1.0], [2.0]]),
                bkd.asarray([[1.0], [0.0]]),
            )

    def test_derivative_rejects_batch_input(self, server_url, bkd) -> None:
        """Single-sample fields take (nvars, 1), not (nvars, nsamples)."""
        model = _model(server_url, "quadratic", bkd)
        jacobian = model.derivatives().jacobian
        assert jacobian is not None
        with pytest.raises(ValueError, match=r"nvars, 1"):
            jacobian(bkd.asarray([[1.0, 2.0], [2.0, 3.0]]))


class TestUpstreamBug:
    """Watches an umbridge defect that constrains the tests above."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "umbridge Python server validates the ApplyHessian reply "
            "against the output size, but H v is input-sized. When this "
            "test starts passing the bug is fixed: drop this class, "
            "widen test_hvp_single_variable to several variables, and "
            "revisit the length check in UMBridgeModel._hvp."
        ),
    )
    def test_upstream_hessian_length_bug_still_present(
        self, server_url, bkd
    ) -> None:
        """A conformant nvars-long H v must round trip; today it cannot.

        The specification defines H v as input-sized -- H is nvars x
        nvars -- and the C++ reference server imposes no length check.
        The Python server checks ``len(output) != output_sizes[out_wrt]``
        in its ApplyHessian handler, so it rejects a correct reply from
        any model with nvars != nqoi.

        This is written as a strict xfail rather than a comment so that
        the workarounds elsewhere in this file cannot outlive the defect
        that motivates them: the suite fails once upstream is fixed.
        """
        model = _model(server_url, "quadratic", bkd, nvars=3)
        hvp = model.derivatives().hvp
        assert hvp is not None
        result = hvp(
            bkd.asarray([[1.0], [2.0], [3.0]]),
            bkd.asarray([[1.0], [0.0], [0.0]]),
        )
        # Hessian of sum(x_i^2) is 2I, so H v = 2 v
        bkd.assert_allclose(
            result, bkd.asarray([[2.0], [0.0], [0.0]])
        )


class TestServerLifecycle:
    """The start/kill helpers used by these tests and by users."""

    def test_start_server_reports_unreachable_url(self, bkd) -> None:
        """A server that never answers raises rather than hanging."""
        port = _free_port()
        with pytest.raises(RuntimeError, match="Could not connect"):
            UMBridgeModel.start_server(
                f"{sys.executable} -c 'import time; time.sleep(30)'",
                url=f"http://localhost:{port}",
                max_wait_time=2,
            )

    def test_kill_server_tolerates_dead_process(self) -> None:
        """Killing an already-exited process is not an error."""
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"], preexec_fn=os.setsid
        )
        process.wait()
        UMBridgeModel.kill_server(process, None)
