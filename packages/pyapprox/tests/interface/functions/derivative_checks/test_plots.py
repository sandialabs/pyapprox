"""Tests for finite-difference error-sweep plotting."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivative_checks.plots import (
    plot_fd_error_sweep,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)

_NVARS = 3


def _fd_eps(bkd):
    return bkd.flip(bkd.logspace(-13, -1, 13))


class TestPlotFDErrorSweep:
    def test_single_curve(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        eps = _fd_eps(bkd)
        errors = bkd.asarray(np.abs(np.sin(np.arange(13.0))) + 1e-8)
        _, ax = plt.subplots()
        out = plot_fd_error_sweep(eps, errors, bkd, ax)
        assert out is ax
        assert len(ax.get_lines()) == 1
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close("all")

    def test_multiple_curves_with_labels_and_guides(
        self, numpy_bkd
    ) -> None:
        bkd = numpy_bkd
        eps = _fd_eps(bkd)
        eps_np = bkd.to_numpy(eps)
        curves = [
            bkd.asarray(eps_np**1 + 1e-10),
            bkd.asarray(np.full(13, 1e-2)),
        ]
        _, ax = plt.subplots()
        plot_fd_error_sweep(
            eps,
            curves,
            bkd,
            ax,
            labels=["correct", "broken"],
            slope_guides=[1],
        )
        # Two curves + one guide line.
        assert len(ax.get_lines()) == 3
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "correct" in legend_texts and "broken" in legend_texts
        # The guide is anchored at the largest eps and decays at the
        # declared order.
        guide_y = ax.get_lines()[-1].get_ydata()
        anchor = np.argmax(eps_np)
        ratio = guide_y / (eps_np / eps_np[anchor]) ** 1
        np.testing.assert_allclose(ratio, ratio[anchor], rtol=1e-12)
        plt.close("all")

    def test_validation(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        eps = _fd_eps(bkd)
        errors = bkd.ones((13,))
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="labels"):
            plot_fd_error_sweep(eps, [errors], bkd, ax, labels=["a", "b"])
        with pytest.raises(ValueError, match="shape"):
            plot_fd_error_sweep(eps, bkd.ones((7,)), bkd, ax)
        plt.close("all")

    def test_end_to_end_with_derivative_checker(self, numpy_bkd) -> None:
        """The intended pattern: pass the SAME fd_eps to the checker
        and the plot; a correct gradient's sweep dips well below its
        largest-eps truncation error (the V), which a plateau would
        not."""
        bkd = numpy_bkd

        def eval_fn(samples):
            return bkd.sum(samples**3, axis=0, keepdims=True)

        def jac_fn(sample):
            return 3.0 * (sample.T ** 2)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=_NVARS, fun=eval_fn, jacobian=jac_fn, bkd=bkd
        )
        checker = DerivativeChecker(wrapped)
        eps = _fd_eps(bkd)
        sample = bkd.asarray(
            np.random.default_rng(3).normal(0.0, 1.0, (_NVARS, 1))
        )
        errors = checker.check_derivatives(
            sample, fd_eps=eps, relative=True
        )[0]
        _, ax = plt.subplots()
        plot_fd_error_sweep(eps, errors, bkd, ax, labels=["gradient"])
        errors_np = bkd.to_numpy(errors)
        assert errors_np.min() <= 1e-6 * errors_np.max()
        plt.close("all")
