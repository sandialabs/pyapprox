r"""POD-DeepONet end to end, on an operator a linear fit cannot represent.

The assembly under test is fields in, fields out: project with a fixed
isometric basis, learn the latent map with a network, decode. Every
piece has its own tests; what only appears here is whether the pieces
together produce a *useful* surrogate on a problem where usefulness is
not automatic.

The operator is pointwise squaring, :math:`u \mapsto u^2`, with
:math:`u` drawn from a sine basis with a decaying spectrum. Three
properties make it the right choice. It is genuinely nonlinear in the
latent coordinates, so a linear latent map is provably insufficient
rather than merely worse -- the degree-1 baseline below measures how
insufficient. Squaring a sine mode produces the difference and sum
harmonics, so the *output* needs more modes than the input, which is
why the two encoders differ in width and why an output basis as narrow
as the input would cap the achievable error at the projection floor
regardless of the latent map. And its exact quadratic structure is what
makes the degree-2 comparison meaningful rather than a tuned baseline.

Every test seeds before constructing a latent map, never only via the
fitter's ``seed``. The weights are drawn in ``MLPLatentMap.__init__``,
which runs before the fitter sees the map, so the fitter's seed cannot
reach them: measured across three identical runs, seeding the fitter
alone gave test errors 0.049, 0.081, 0.081 while seeding first gave
0.081 three times.

Errors are measured on held-out samples throughout. On the training set
this operator shows no trend at all -- the error there is flat to
slightly rising as samples are added, because more of them are harder
to interpolate -- so a trend asserted on training error would be
asserting nothing.

The trend tests use a short optimizer budget and hold it *equal* across
the points they compare, rather than running each to convergence. That
is not a shortcut: L-BFGS reports its budget exhausted on this problem
at every width and budget tried, so convergence is not available to
require, and equality is what keeps a comparison about the quantity
being varied. Only the absolute-accuracy test pays for a large budget,
and it is the slow one in the file for exactly that reason.
"""

from typing import Any, Tuple

import numpy as np
import pytest
import torch
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis.orthonormal_poly import (
    OrthonormalPolynomialBasis,
)
from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.indices.utils import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate.factory import create_bases_1d
from pyapprox.surrogates.operatorlearning import (
    GramProjectionEncoder,
    IdentityFieldEncoder,
    WeightedLeastSquaresOperatorFitter,
    bochner_error,
)
from pyapprox.surrogates.operatorlearning.latent_maps.gradient_fitter import (
    TorchGradientLatentMapFitter,
)
from pyapprox.surrogates.operatorlearning.latent_maps.mlp import (
    MLPLatentMap,
)
from pyapprox.surrogates.operatorlearning.latent_maps.torch_optimizers import (
    torch_adam_then_lbfgs,
)
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.inner_product import DiagonalInnerProduct

from tests._helpers.markers import slow_test, slowest_test

NINPUT_MODES = 4
NOUTPUT_MODES = 10


def _sine_encoder(bkd: Backend, nmodes: int, ngrid: int = 64) -> Any:
    """Orthonormal sine basis with the quadrature inner product.

    Isometric in that inner product, which is what lets the tests below
    report a Bochner error rather than a coefficient proxy.
    """
    x = np.linspace(0.0, 1.0, ngrid + 2)[1:-1]
    modes = np.arange(1, nmodes + 1)
    sine = np.sqrt(2.0) * np.sin(np.outer(x, modes) * np.pi)
    mass = DiagonalInnerProduct(bkd.full((ngrid,), 1.0 / (ngrid + 1)), bkd)
    return GramProjectionEncoder(bkd.asarray(sine), mass, bkd)


def _squaring_data(
    bkd: Backend, encoder: Any, nsamples: int, seed: int
) -> Tuple[Any, Any, Any]:
    r"""Realizations of :math:`u` and of :math:`u^2`, plus the raw coefficients.

    The :math:`1/k` scaling gives the decaying spectrum a
    Karhunen-Loeve expansion produces, so the inputs are smooth fields
    rather than white noise. The raw unscaled coefficients come back
    too because they are uniform on :math:`[-1, 1]`, which is what the
    polynomial baseline needs as its expansion variables.
    """
    rng = np.random.RandomState(seed)
    raw = rng.uniform(-1.0, 1.0, (NINPUT_MODES, nsamples))
    decay = 1.0 / np.arange(1, NINPUT_MODES + 1)
    inputs = encoder.decode(bkd.asarray(decay[:, None] * raw))
    return bkd.asarray(raw), inputs, inputs**2


def _fit_mlp(
    bkd: Backend,
    input_encoder: Any,
    output_encoder: Any,
    inputs: Any,
    outputs: Any,
    width: int = 16,
    nepochs: int = 300,
    npolish: int = 600,
) -> Any:
    """Seed, build the map, fit. Returns the fit result, not the surrogate.

    Callers need the stage records as well as the surrogate: an error
    that is large because the optimizer ran out of iterations says
    something different from the same error at convergence.

    The default budget is deliberately short -- around two seconds --
    because the trend tests below need it only to be *equal* across the
    points they compare, not to be converged. The tests that make an
    absolute accuracy claim pass a larger one explicitly.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    latent_map = MLPLatentMap(
        input_encoder.latent_dim(),
        output_encoder.latent_dim(),
        [width, width],
        bkd,
    )
    fitter = TorchGradientLatentMapFitter(
        input_encoder,
        output_encoder,
        bkd,
        optimizer=torch_adam_then_lbfgs(
            nepochs=nepochs, npolish_iterations=npolish
        ),
        seed=0,
    )
    return fitter.fit(latent_map, inputs, outputs)


class TestPODDeepONetOnASquaringOperator:
    r"""What the assembled surrogate achieves, and against what.

    Accuracy alone would be a weak claim: a number like 1e-3 means
    nothing without something to compare it against and without knowing
    which ingredient set it. So these tests pin an achievable error,
    show it falls as data and capacity are added, and measure it against
    the best linear map -- while separately establishing that the
    operator really is quadratic, so that the linear map's failure is
    algebraic rather than a tuning artifact.
    """

    @pytest.fixture
    def encoders(self, torch_bkd: Backend) -> Tuple[Any, Any]:
        """Input and output bases, of deliberately different width.

        Squaring a sine mode generates sum and difference harmonics, so
        the output lives in a wider space than the input. Using one
        basis for both would be the easy mistake, and it would cap the
        error at the projection floor rather than at the latent map's
        accuracy -- with four output modes that floor is 16%.
        """
        return (
            _sine_encoder(torch_bkd, NINPUT_MODES),
            _sine_encoder(torch_bkd, NOUTPUT_MODES),
        )

    @slowest_test
    def test_reaches_a_useful_error(
        self, torch_bkd: Backend, encoders: Tuple[Any, Any]
    ) -> None:
        """An absolute floor, so "it converges" cannot mean "to anything".

        A trend test alone passes on a sequence descending to a useless
        plateau, so one test has to say where the plateau is: about 1e-3
        relative Bochner error on held-out data, which for this operator
        is the accuracy that makes the surrogate worth having at all.

        This is the expensive test in the file and unavoidably so -- the
        trend tests need only a budget held equal across their points,
        while an absolute claim needs a budget large enough to actually
        arrive. L-BFGS still reports its budget exhausted here rather
        than converging, which means 1e-3 is a lower bound on what this
        architecture reaches and not its limit.
        """
        input_encoder, output_encoder = encoders
        _, train_in, train_out = _squaring_data(
            torch_bkd, input_encoder, 800, seed=0
        )
        _, test_in, test_out = _squaring_data(
            torch_bkd, input_encoder, 400, seed=7
        )
        result = _fit_mlp(
            torch_bkd,
            input_encoder,
            output_encoder,
            train_in,
            train_out,
            width=32,
            nepochs=2000,
            npolish=8000,
        )
        surrogate = result.surrogate()
        test_error = bochner_error(
            output_encoder, surrogate(test_in), test_out, torch_bkd
        )
        assert test_error < 1.5e-3

    @slow_test
    def test_the_error_falls_as_data_is_added(
        self, torch_bkd: Backend, encoders: Tuple[Any, Any]
    ) -> None:
        """Convergence in the sample size, at fixed capacity and budget.

        The property an error threshold cannot express: that the
        surrogate is actually learning the operator rather than
        happening to land near it. Width, depth and both optimizer
        budgets are held fixed, so the data is the only thing varying
        and the comparison is not secretly about the optimizer.

        Measured on held-out samples, which is the only place this
        converges: the training error is flat to slightly *rising* over
        the same sweep, because more samples are harder to interpolate,
        not easier. A convergence test on training error would assert
        nothing.

        Endpoints with a margin rather than each adjacent pair, because
        adjacent points sit within the run-to-run spread of the initial
        weights -- measured across five initializations, neighbours
        invert but the 50-to-400 ratio stays in 2.7 to 4.7. Asserting
        strict monotonicity here would be asserting a property of one
        seed.
        """
        input_encoder, output_encoder = encoders
        _, test_in, test_out = _squaring_data(
            torch_bkd, input_encoder, 400, seed=7
        )

        def error_with(nsamples: int) -> float:
            _, train_in, train_out = _squaring_data(
                torch_bkd, input_encoder, nsamples, seed=0
            )
            surrogate = _fit_mlp(
                torch_bkd,
                input_encoder,
                output_encoder,
                train_in,
                train_out,
            ).surrogate()
            return bochner_error(
                output_encoder, surrogate(test_in), test_out, torch_bkd
            )

        assert error_with(50) > 2.0 * error_with(400)

    @slow_test
    def test_the_error_falls_as_the_network_widens(
        self, torch_bkd: Backend, encoders: Tuple[Any, Any]
    ) -> None:
        """Convergence in capacity, the other axis, at fixed data.

        Separate from the data sweep because the two fail independently:
        a surrogate can be starved of samples at ample capacity, or
        starved of capacity with ample samples, and only varying one at
        a time says which. Here the sample count and both optimizer
        budgets are fixed and the width alone moves.

        Strictly monotone, unlike the data sweep, and it can be asserted
        that way: each doubling cuts the error by roughly a factor of
        five, which is far outside the spread across initializations.
        Width 4 cannot represent the target at all, which is what makes
        the first step so large.
        """
        input_encoder, output_encoder = encoders
        _, train_in, train_out = _squaring_data(
            torch_bkd, input_encoder, 200, seed=0
        )
        _, test_in, test_out = _squaring_data(
            torch_bkd, input_encoder, 400, seed=7
        )
        errors = []
        for width in (4, 8, 16):
            surrogate = _fit_mlp(
                torch_bkd,
                input_encoder,
                output_encoder,
                train_in,
                train_out,
                width=width,
            ).surrogate()
            errors.append(
                bochner_error(
                    output_encoder,
                    surrogate(test_in),
                    test_out,
                    torch_bkd,
                )
            )
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        assert errors[0] > 5.0 * errors[2]

    @slow_test
    def test_beats_the_linear_baseline_it_must(
        self, torch_bkd: Backend, encoders: Tuple[Any, Any]
    ) -> None:
        """The pairing that stops a dead nonlinearity from passing.

        A network whose activations were broken would still reproduce a
        linear map, and would still pass an accuracy threshold on a
        linear problem. Here the best linear map there is -- the
        least-squares degree-1 fit -- leaves ~75% relative error, and
        that is a limit of representation rather than of tuning, which
        the degree-two test below establishes separately.

        The margin asserted is 40x, against a measured 52x at this
        budget. A linear map cannot close that gap by being fitted
        better, so the assertion is about the nonlinearity being alive --
        and it is deliberately close to the measurement, since a margin
        far below what is achieved would keep passing while the
        nonlinearity degraded.
        """
        input_encoder, output_encoder = encoders
        raw_train, train_in, train_out = _squaring_data(
            torch_bkd, input_encoder, 400, seed=0
        )
        raw_test, test_in, test_out = _squaring_data(
            torch_bkd, input_encoder, 400, seed=7
        )

        linear_error = _polynomial_baseline_error(
            torch_bkd,
            output_encoder,
            degree=1,
            raw_train=raw_train,
            train_out=train_out,
            raw_test=raw_test,
            test_out=test_out,
        )
        mlp = _fit_mlp(
            torch_bkd,
            input_encoder,
            output_encoder,
            train_in,
            train_out,
        ).surrogate()
        mlp_error = bochner_error(
            output_encoder, mlp(test_in), test_out, torch_bkd
        )

        assert linear_error > 0.5
        assert mlp_error < linear_error / 40.0

    def test_degree_two_solves_it_exactly_and_degree_one_cannot(
        self, torch_bkd: Backend, encoders: Tuple[Any, Any]
    ) -> None:
        """The problem is exactly quadratic, which is what makes the pairing work.

        Squaring is degree 2 in the latent coordinates, so a degree-2
        expansion does not approximate this operator, it *represents* it:
        the least-squares fit lands at machine precision, measured 7e-16.
        Degree 1 on the same data leaves 75%. The two numbers together
        say the gap the network is asked to close is a gap in
        representation, not one in tuning -- a linear latent map is
        excluded by algebra, and a quadratic one is exactly enough.

        The network is deliberately *not* compared against the degree-2
        fit. At 7e-16 that fit is a ceiling no gradient-trained network
        reaches, so asking it to come within any constant factor would
        assert something false; the useful comparison for the network is
        the degree-1 floor, next to this one.

        Cheap: two least-squares solves, no network.
        """
        input_encoder, output_encoder = encoders
        raw_train, _, train_out = _squaring_data(
            torch_bkd, input_encoder, 400, seed=0
        )
        raw_test, _, test_out = _squaring_data(
            torch_bkd, input_encoder, 400, seed=7
        )

        def baseline(degree: int) -> float:
            return _polynomial_baseline_error(
                torch_bkd,
                output_encoder,
                degree=degree,
                raw_train=raw_train,
                train_out=train_out,
                raw_test=raw_test,
                test_out=test_out,
            )

        assert baseline(2) < 1e-12
        assert baseline(1) > 0.5

    @slow_test
    def test_says_the_budget_rather_than_the_model_set_the_error(
        self, torch_bkd: Backend, encoders: Tuple[Any, Any]
    ) -> None:
        """What every error in this file has to be read against.

        On this operator L-BFGS reports its budget exhausted at every
        width and budget tried, up to 20000 iterations -- so every error
        here is a lower bound on the architecture rather than its limit,
        and the fit result is what says so. A test suite that reported
        such numbers as converged accuracy would be overstating them.

        Also the reason the trend tests fix their budget instead of
        asking for convergence: convergence is not available to ask for,
        so equality across compared points is what keeps those trends
        about data and capacity.
        """
        input_encoder, output_encoder = encoders
        _, train_in, train_out = _squaring_data(
            torch_bkd, input_encoder, 200, seed=0
        )
        starved = _fit_mlp(
            torch_bkd,
            input_encoder,
            output_encoder,
            train_in,
            train_out,
            nepochs=50,
            npolish=50,
        )
        assert starved.stages()[-1].exhausted_budget()
        assert not starved.success()

        longer = _fit_mlp(
            torch_bkd,
            input_encoder,
            output_encoder,
            train_in,
            train_out,
            nepochs=600,
            npolish=1200,
        )
        # Still not converged, and reporting it -- which is the point.
        assert longer.stages()[-1].exhausted_budget()
        # The extra budget was spent to some purpose all the same.
        assert longer.fun() < starved.fun()


def _polynomial_baseline_error(
    bkd: Backend,
    output_encoder: Any,
    degree: int,
    raw_train: Any,
    train_out: Any,
    raw_test: Any,
    test_out: Any,
) -> float:
    """Relative Bochner error of a least-squares polynomial latent map.

    The same encoders and the same data as the network fit, differing
    only in the latent map, so the comparison isolates that map. The
    expansion variables are the *raw* coefficients, which are uniform
    on :math:`[-1, 1]` as the orthonormal basis assumes; feeding it the
    decayed coefficients would shrink the inputs into part of the
    domain and understate the baseline.
    """
    marginals = [
        UniformMarginal(-1.0, 1.0, bkd) for _ in range(NINPUT_MODES)
    ]
    basis = OrthonormalPolynomialBasis(create_bases_1d(marginals, bkd), bkd)
    basis.set_indices(
        compute_hyperbolic_indices(NINPUT_MODES, degree, 1.0, bkd)
    )
    expansion = PolynomialChaosExpansion(
        basis, bkd, nqoi=output_encoder.latent_dim()
    )
    fitter = WeightedLeastSquaresOperatorFitter(
        IdentityFieldEncoder(NINPUT_MODES, bkd), output_encoder, bkd
    )
    surrogate = fitter.fit_encoded(
        expansion, raw_train, output_encoder.encode(train_out)
    ).surrogate()
    return bochner_error(
        output_encoder, surrogate(raw_test), test_out, bkd
    )
