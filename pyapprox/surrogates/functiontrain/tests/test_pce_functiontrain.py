"""Tests for PCEFunctionTrain and create_pce_functiontrain."""


import numpy as np
import pytest

from pyapprox.probability import (
    BetaMarginal,
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import (
    FunctionTrain,
    PCEFunctionTrain,
    PCEFunctionTrainCore,
    create_pce_functiontrain,
)
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore


class TestPCEFunctionTrain:
    """Base class for PCEFunctionTrain tests."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_pce(self, bkd, max_level, nqoi=1):
        """Create a univariate PCE expansion with Legendre basis."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_rank1_ft(self, bkd, nvars, max_level):
        """Create a rank-1 FunctionTrain with random PCE coefficients."""
        nterms = max_level + 1
        cores = []

        for _ in range(nvars):
            pce = self._create_univariate_pce(bkd, max_level)
            coef = bkd.asarray(np.random.randn(nterms, 1))
            pce = pce.with_params(coef)
            core = FunctionTrainCore([[pce]], bkd)
            cores.append(core)

        return FunctionTrain(cores, bkd, nqoi=1)

    def test_construction_success(self, bkd) -> None:
        """Test PCEFunctionTrain construction succeeds for valid FT."""
        ft = self._create_rank1_ft(bkd, nvars=3, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        assert pce_ft.nvars() == 3
        assert pce_ft.nqoi() == 1
        assert len(pce_ft.pce_cores()) == 3

    def test_construction_rejects_non_functiontrain(self, bkd) -> None:
        """Test construction fails for non-FunctionTrain input."""
        with pytest.raises(TypeError) as ctx:
            PCEFunctionTrain("not a function train")  # type: ignore
        assert "Expected FunctionTrain" in str(ctx.value)

    def test_construction_rejects_multi_qoi(self, bkd) -> None:
        """Test construction fails for nqoi > 1."""
        # Create FT with nqoi=2
        pce = self._create_univariate_pce(bkd, 2, nqoi=2)
        core = FunctionTrainCore([[pce]], bkd)
        ft = FunctionTrain([core], bkd, nqoi=2)

        with pytest.raises(ValueError) as ctx:
            PCEFunctionTrain(ft)
        assert "nqoi=1" in str(ctx.value)

    def test_ft_accessor(self, bkd) -> None:
        """Test ft() returns underlying FunctionTrain."""
        ft = self._create_rank1_ft(bkd, nvars=2, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        assert pce_ft.ft() is ft

    def test_pce_cores_type(self, bkd) -> None:
        """Test pce_cores returns list of PCEFunctionTrainCore."""
        ft = self._create_rank1_ft(bkd, nvars=3, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        cores = pce_ft.pce_cores()
        assert len(cores) == 3
        for core in cores:
            assert isinstance(core, PCEFunctionTrainCore)

    def test_evaluation_delegates_to_ft(self, bkd) -> None:
        """Test __call__ delegates to underlying FunctionTrain."""
        ft = self._create_rank1_ft(bkd, nvars=3, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        samples = bkd.asarray(np.random.uniform(-1, 1, (3, 5)))

        ft_result = ft(samples)
        pce_ft_result = pce_ft(samples)

        bkd.assert_allclose(pce_ft_result, ft_result, rtol=1e-12)

    def test_bkd_accessor(self, bkd) -> None:
        """Test bkd() returns correct backend."""
        ft = self._create_rank1_ft(bkd, nvars=2, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        assert pce_ft.bkd() is bkd

    def test_repr(self, bkd) -> None:
        """Test __repr__ provides useful info."""
        ft = self._create_rank1_ft(bkd, nvars=3, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        repr_str = repr(pce_ft)
        assert "PCEFunctionTrain" in repr_str
        assert "nvars=3" in repr_str
        assert "nqoi=1" in repr_str


class TestCreatePceFunctiontrain:
    """Tests for create_pce_functiontrain factory."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_uniform_marginals_construction(self, bkd) -> None:
        """Test construction with uniform marginals (Legendre)."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        ft = create_pce_functiontrain(marginals, max_level=5, ranks=[2, 2], bkd=bkd)

        assert ft.nvars() == 3
        assert ft.nqoi() == 1
        # All cores should have nterms = max_level + 1 = 6 for each entry
        for core in ft.cores():
            assert core.get_nterms(0, 0) == 6

    def test_mixed_marginals_construction(self, bkd) -> None:
        """Test construction with mixed marginals (Legendre, Hermite, Jacobi)."""
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),  # Legendre
            GaussianMarginal(0.0, 1.0, bkd),  # Hermite
            BetaMarginal(2.0, 5.0, bkd),  # Jacobi
        ]
        ft = create_pce_functiontrain(marginals, max_level=5, ranks=[2, 2], bkd=bkd)

        assert ft.nvars() == 3
        assert ft.nqoi() == 1

    def test_mixed_marginals_evaluation(self, bkd) -> None:
        """Test FT with mixed marginals can be evaluated."""
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
            BetaMarginal(2.0, 5.0, bkd),
        ]
        ft = create_pce_functiontrain(marginals, max_level=3, ranks=[2, 2], bkd=bkd)

        # Create samples in each variable's domain
        nsamples = 10
        samples = bkd.vstack(
            [
                bkd.asarray(np.random.uniform(-1, 1, nsamples)),  # Uniform [-1, 1]
                bkd.asarray(np.random.randn(nsamples)),  # Gaussian
                bkd.asarray(np.random.beta(2.0, 5.0, nsamples)),  # Beta [0, 1]
            ]
        )

        result = ft(samples)
        assert result.shape == (1, nsamples)

    def test_per_core_max_level(self, bkd) -> None:
        """Test per-core polynomial degrees."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        ft = create_pce_functiontrain(
            marginals, max_level=[3, 5, 4], ranks=[2, 2], bkd=bkd
        )

        # Core 0: degree 3 -> 4 terms
        # Core 1: degree 5 -> 6 terms
        # Core 2: degree 4 -> 5 terms
        assert ft.cores()[0].get_nterms(0, 0) == 4
        assert ft.cores()[1].get_nterms(0, 0) == 6
        assert ft.cores()[2].get_nterms(0, 0) == 5

    def test_pce_functiontrain_wrapping(self, bkd) -> None:
        """Test FT from mixed marginals can be wrapped in PCEFunctionTrain."""
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
        ]
        ft = create_pce_functiontrain(marginals, max_level=4, ranks=[2], bkd=bkd)

        # Should not raise
        pce_ft = PCEFunctionTrain(ft)
        assert pce_ft.nvars() == 2
        assert len(pce_ft.pce_cores()) == 2

    def test_error_empty_marginals(self, bkd) -> None:
        """Test error on empty marginals."""
        with pytest.raises(ValueError) as ctx:
            create_pce_functiontrain([], max_level=3, ranks=[], bkd=bkd)
        assert "empty" in str(ctx.value).lower()

    def test_error_wrong_ranks_length(self, bkd) -> None:
        """Test error on wrong ranks length."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        with pytest.raises(ValueError) as ctx:
            create_pce_functiontrain(marginals, max_level=3, ranks=[2], bkd=bkd)
        assert "ranks" in str(ctx.value).lower()

    def test_error_wrong_max_level_length(self, bkd) -> None:
        """Test error on wrong max_level sequence length."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        with pytest.raises(ValueError) as ctx:
            create_pce_functiontrain(marginals, max_level=[3, 4], ranks=[2, 2], bkd=bkd)
        assert "max_level" in str(ctx.value).lower()

    def test_init_scale_zero(self, bkd) -> None:
        """Test zero initialization."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        ft = create_pce_functiontrain(
            marginals, max_level=3, ranks=[2], bkd=bkd, init_scale=0.0
        )

        # With zero init, all coefficients should be zero
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 5)))
        result = ft(samples)
        bkd.assert_allclose(result, bkd.zeros((1, 5)), atol=1e-12)
