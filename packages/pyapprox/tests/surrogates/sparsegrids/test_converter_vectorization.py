"""The subspace-to-PCE conversion reproduces the interpolant.

``convert_subspace`` re-expresses a tensor product Lagrange interpolant
in an orthonormal polynomial basis. Both bases span the same space, so
the conversion is exact and the defining property is that the expansion
and the subspace agree everywhere, not merely at the nodes.

That property is what these tests assert. It pins the mathematics rather
than any particular way of computing it, so it holds whatever
contraction order the implementation uses.
"""

from typing import List

import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.converters.pce import (
    TensorProductSubspaceToPCEConverter,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.util.cartesian import cartesian_product_indices


def _build(bkd, npts_1d, basis_type, nqoi, lower=0.0, upper=1.0):
    """Build a subspace with the given per-dimension point counts.

    LinearGrowthRule(1, 1) gives level l exactly l + 1 points, so the
    multi-index is npts - 1 per dimension.
    """
    nvars = len(npts_1d)
    marginals = [UniformMarginal(lower, upper, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, basis_type)
    tp_factory = TensorProductSubspaceFactory(
        bkd, factories, LinearGrowthRule(scale=1, shift=1)
    )
    idx = bkd.asarray([n - 1 for n in npts_1d], dtype=bkd.int64_dtype())
    subspace = tp_factory(idx)
    samples = subspace.get_samples()
    rows = [
        bkd.sum(samples**2, axis=0) + (q + 1) * samples[0]
        for q in range(nqoi)
    ]
    subspace.set_values(bkd.stack(rows, axis=0))
    converter = TensorProductSubspaceToPCEConverter(
        bkd, create_bases_1d(marginals, bkd)
    )
    return converter, subspace, marginals


def _as_pce(bkd, marginals, indices, coefficients):
    """Assemble an expansion from converted indices and coefficients."""
    bases_1d = create_bases_1d(marginals, bkd)
    for dim, basis in enumerate(bases_1d):
        basis.set_nterms(int(bkd.to_int(bkd.max(indices[dim]))) + 1)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    pce = PolynomialChaosExpansion(basis, bkd, coefficients.shape[0])
    pce.set_coefficients(coefficients.T)
    return pce


def _test_points(bkd, nvars, lower=0.0, upper=1.0, npoints=40):
    """Points away from the nodes, where agreement is not automatic."""
    import numpy as np

    np.random.seed(7)
    unit = np.random.uniform(0.0, 1.0, (nvars, npoints))
    return bkd.asarray(lower + (upper - lower) * unit)


class TestConversionReproducesTheInterpolant:
    """The expansion and the subspace are the same function."""

    @pytest.mark.parametrize(
        "npts_1d",
        [[3], [3, 4], [4, 2], [2, 3, 4], [3, 3, 3]],
    )
    @pytest.mark.parametrize("basis_type", ["gauss", "leja"])
    @pytest.mark.parametrize("nqoi", [1, 2])
    def test_agrees_away_from_the_nodes(
        self, bkd, npts_1d: List[int], basis_type: str, nqoi: int
    ) -> None:
        converter, subspace, marginals = _build(
            bkd, npts_1d, basis_type, nqoi
        )
        indices, coefficients = converter.convert_subspace(subspace)
        pce = _as_pce(bkd, marginals, indices, coefficients)

        samples = _test_points(bkd, len(npts_1d))
        bkd.assert_allclose(
            pce(samples), subspace(samples), rtol=1e-10, atol=1e-12
        )

    def test_agrees_on_a_non_canonical_domain(self, bkd) -> None:
        """The domain transform must not shift the projection."""
        converter, subspace, marginals = _build(
            bkd, [3, 4], "gauss", 1, lower=-2.0, upper=5.0
        )
        indices, coefficients = converter.convert_subspace(subspace)
        pce = _as_pce(bkd, marginals, indices, coefficients)

        samples = _test_points(bkd, 2, lower=-2.0, upper=5.0)
        bkd.assert_allclose(
            pce(samples), subspace(samples), rtol=1e-10, atol=1e-12
        )

    def test_agrees_at_the_nodes(self, bkd) -> None:
        """Interpolation is exact there, so the expansion must be too."""
        converter, subspace, marginals = _build(bkd, [3, 4], "gauss", 2)
        indices, coefficients = converter.convert_subspace(subspace)
        pce = _as_pce(bkd, marginals, indices, coefficients)

        nodes = subspace.get_samples()
        bkd.assert_allclose(
            pce(nodes), subspace(nodes), rtol=1e-10, atol=1e-12
        )

    @pytest.mark.parametrize(
        "npts_1d", [[3, 4], [4, 2], [2, 3, 4], [3, 3, 3]]
    )
    @pytest.mark.parametrize("basis_type", ["gauss", "leja"])
    def test_l2_error_is_at_machine_precision(
        self, bkd, npts_1d: List[int], basis_type: str
    ) -> None:
        """The relative L2 difference over the domain is ~eps.

        Pointwise agreement can be satisfied by luck at scattered
        points. A change of basis is exact, so the error integrated
        over the whole domain must be at rounding level, which a
        systematically wrong coefficient would not be.
        """
        converter, subspace, marginals = _build(
            bkd, npts_1d, basis_type, 1
        )
        indices, coefficients = converter.convert_subspace(subspace)
        pce = _as_pce(bkd, marginals, indices, coefficients)

        samples = _test_points(bkd, len(npts_1d), npoints=2000)
        diff = pce(samples) - subspace(samples)
        rel_l2 = bkd.to_float(
            bkd.sqrt(bkd.sum(diff**2))
            / bkd.sqrt(bkd.sum(subspace(samples) ** 2))
        )
        assert rel_l2 < 1e-13, f"relative L2 error {rel_l2:.3e}"


class TestConvertedCoefficients:
    """Shape and ordering of what the conversion returns."""

    @pytest.mark.parametrize("npts_1d", [[3, 4], [2, 3, 4]])
    def test_index_ordering_matches_the_samples(
        self, bkd, npts_1d: List[int]
    ) -> None:
        """Indices use the C-order the subspace's samples use."""
        converter, subspace, _ = _build(bkd, npts_1d, "gauss", 1)
        indices, _ = converter.convert_subspace(subspace)
        bkd.assert_allclose(
            indices, cartesian_product_indices(npts_1d, bkd)
        )

    @pytest.mark.parametrize("nqoi", [1, 3])
    def test_shapes(self, bkd, nqoi: int) -> None:
        npts_1d = [3, 4]
        converter, subspace, _ = _build(bkd, npts_1d, "gauss", nqoi)
        indices, coefficients = converter.convert_subspace(subspace)
        nterms = npts_1d[0] * npts_1d[1]
        assert indices.shape == (2, nterms)
        assert coefficients.shape == (nqoi, nterms)

    def test_constant_function_has_one_nonzero_coefficient(
        self, bkd
    ) -> None:
        """A constant projects onto the constant polynomial alone."""
        nvars = 2
        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
        factories = create_basis_factories(marginals, bkd, "gauss")
        tp_factory = TensorProductSubspaceFactory(
            bkd, factories, LinearGrowthRule(scale=1, shift=1)
        )
        idx = bkd.asarray([2, 3], dtype=bkd.int64_dtype())
        subspace = tp_factory(idx)
        subspace.set_values(
            bkd.full((1, subspace.nsamples()), 3.5)
        )
        converter = TensorProductSubspaceToPCEConverter(
            bkd, create_bases_1d(marginals, bkd)
        )
        indices, coefficients = converter.convert_subspace(subspace)

        # The constant term is the all-zero multi-index, which C-order
        # puts first.
        bkd.assert_allclose(
            coefficients[:, 0], bkd.asarray([3.5]), rtol=1e-12
        )
        bkd.assert_allclose(
            coefficients[:, 1:],
            bkd.zeros((1, coefficients.shape[1] - 1)),
            atol=1e-12,
        )


class TestBasisValidation:
    """Spectral projection is only valid for polynomial bases."""

    @pytest.mark.parametrize(
        "basis_type", ["piecewise_linear", "piecewise_quadratic"]
    )
    def test_rejects_piecewise_bases(self, bkd, basis_type: str) -> None:
        """The projection would return a wrong number, not an error.

        Piecewise-quadratic is included deliberately: it can agree with
        the true value on a given target, which makes an unguarded
        conversion harder to catch rather than safe.
        """
        from pyapprox.surrogates.affine.indices import (
            ClenshawCurtisGrowthRule,
        )

        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
        factories = create_basis_factories(marginals, bkd, basis_type)
        tp_factory = TensorProductSubspaceFactory(
            bkd, factories, ClenshawCurtisGrowthRule()
        )
        subspace = tp_factory(
            bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        )
        samples = subspace.get_samples()
        subspace.set_values(
            bkd.reshape(bkd.sum(samples**2, axis=0), (1, -1))
        )
        converter = TensorProductSubspaceToPCEConverter(
            bkd, create_bases_1d(marginals, bkd)
        )
        with pytest.raises(ValueError, match="globally polynomial"):
            converter.convert_subspace(subspace)

    @pytest.mark.parametrize(
        "basis_type", ["gauss", "leja", "clenshaw_curtis"]
    )
    def test_accepts_polynomial_bases(self, bkd, basis_type: str) -> None:
        from pyapprox.surrogates.affine.indices import (
            ClenshawCurtisGrowthRule,
        )

        growth = (
            ClenshawCurtisGrowthRule()
            if basis_type == "clenshaw_curtis"
            else LinearGrowthRule(scale=1, shift=1)
        )
        marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
        factories = create_basis_factories(marginals, bkd, basis_type)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        subspace = tp_factory(
            bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        )
        samples = subspace.get_samples()
        subspace.set_values(
            bkd.reshape(bkd.sum(samples**2, axis=0), (1, -1))
        )
        converter = TensorProductSubspaceToPCEConverter(
            bkd, create_bases_1d(marginals, bkd)
        )
        indices, coefficients = converter.convert_subspace(subspace)
        assert indices.shape[0] == 2
        assert coefficients.shape[0] == 1


class TestSizeSmokeTest:
    """A grid large enough to matter."""

    def test_three_dimensions_twelve_points(self, numpy_bkd) -> None:
        """3D with n_d = 12 is 1728 terms over 1728 samples."""
        converter, subspace, marginals = _build(
            numpy_bkd, [12, 12, 12], "gauss", 1
        )
        indices, coefficients = converter.convert_subspace(subspace)
        assert indices.shape == (3, 12**3)
        assert coefficients.shape == (1, 12**3)

        pce = _as_pce(numpy_bkd, marginals, indices, coefficients)
        samples = _test_points(numpy_bkd, 3, npoints=20)
        numpy_bkd.assert_allclose(
            pce(samples), subspace(samples), rtol=1e-8, atol=1e-10
        )


class TestAutograd:
    """Coefficients stay differentiable with respect to the values."""

    def test_gradient_flows_to_the_values(self, torch_bkd) -> None:
        """The conversion is linear, so the gradient is the projection.

        Summing the coefficients and differentiating gives, for each
        value, the summed product of its per-dimension projection
        coefficients --- the row sum of the Kronecker matrix.
        """
        import torch

        npts_1d = [3, 4]
        converter, subspace, _ = _build(torch_bkd, npts_1d, "gauss", 1)
        values = torch.ones(
            (1, subspace.nsamples()), dtype=torch.double, requires_grad=True
        )
        subspace._interpolant._values = values

        _, coefficients = converter.convert_subspace(subspace)
        coefficients.sum().backward()
        assert values.grad is not None

        proj = [
            converter._get_projection_coefficients(
                dim, torch_bkd.flatten(subspace.get_samples_1d(dim))
            )
            for dim in range(2)
        ]
        expected = torch_bkd.flatten(
            torch_bkd.sum(proj[0], axis=1)[:, None]
            * torch_bkd.sum(proj[1], axis=1)[None, :]
        )
        torch_bkd.assert_allclose(
            torch_bkd.flatten(values.grad), expected, rtol=1e-10
        )
