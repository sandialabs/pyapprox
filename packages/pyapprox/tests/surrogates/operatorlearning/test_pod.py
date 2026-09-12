r"""Building a POD basis over a domain, and the ways it goes wrong quietly.

There is no ``pod_encoder`` wrapper: a basis over a domain is
``fit_kle_encoder(snapshots, domain.bkd(), latent_dim=...,
metric=domain.inner_product())``, and a wrapper around that would add a
name without adding content. What needs testing is not the call but the
two properties nobody can see by reading it -- that the basis is optimal
in the *metric* rather than in the Euclidean norm, and that composing
several fields does not starve one of them.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kle.encoder import fit_kle_encoder
from pyapprox.surrogates.operatorlearning import (
    FieldEncoderProtocol,
    GramProjectionEncoder,
    ProductFieldEncoder,
    orthonormalize_basis,
)
from pyapprox.surrogates.operatorlearning.domains import UniformGridDomain
from pyapprox.util.backends.protocols import Backend


def _graded_axis(bkd: Backend):
    """An axis refined near zero, so Euclidean and M norms disagree.

    Half the points sit in the first tenth of the interval. A mode
    concentrated there has large Euclidean energy and small integral
    energy, which is what separates the two decompositions.
    """
    return bkd.asarray(
        np.concatenate([np.linspace(0.0, 0.1, 12), np.linspace(0.2, 1.0, 9)])
    )


def _snapshots_concentrated_in_the_refined_region(bkd: Backend, x, nsamples):
    """Fields whose variation is mostly where the grid is fine.

    The spike's *width and position* vary, not merely its amplitude, so
    the snapshot matrix has full numerical rank rather than the rank of
    however many fixed shapes were summed. A basis truncated to three
    modes is then a genuine approximation.
    """
    rng = np.random.RandomState(0)
    xs = np.asarray(bkd.to_numpy(x))
    columns = []
    for _ in range(nsamples):
        width = 0.02 * (1.0 + 0.8 * rng.uniform(0.0, 1.0))
        center = 0.03 * rng.uniform(0.0, 1.0)
        spike = np.exp(-(((xs - center) / width) ** 2))
        smooth = rng.uniform(-1.0, 1.0) * np.sin(np.pi * xs)
        columns.append(spike + 0.35 * smooth)
    return bkd.asarray(np.array(columns).T)


def _relative_projection_error(encoder, snapshots, domain) -> float:
    r"""Squared M-error of the basis's reconstruction, relative to total.

    .. math::

        \frac{\|S_c - P S_c\|_M^2}{\|S_c\|_M^2}

    on the mean-removed snapshots, in the field norm the domain defines
    -- the quantity a POD basis is built to minimize. Uses the encoder's
    own round trip, so it measures the basis as a consumer would use it.
    """
    bkd = domain.bkd()
    metric = domain.inner_product()
    mean = bkd.reshape(
        bkd.mean(snapshots, axis=1), (int(snapshots.shape[0]), 1)
    )
    centered = snapshots - mean
    residual = centered - (
        encoder.decode(encoder.encode(snapshots)) - mean
    )
    total = float(bkd.sum(centered * metric.apply(centered)))
    return float(bkd.sum(residual * metric.apply(residual))) / total


def _discarded_spectrum_fraction(snapshots, domain, nmodes: int) -> float:
    r"""The error an optimal rank-``nmodes`` basis must attain.

    .. math:: \sum_{k>p}\lambda_k \big/ \sum_k \lambda_k

    from the full spectrum in the same metric. This is what makes the
    tolerances below *derived* rather than chosen: POD's optimality says
    the projection error equals this exactly, so a test can assert the
    mathematical property instead of a constant someone picked by
    running the code once.
    """
    bkd = domain.bkd()
    full = fit_kle_encoder(snapshots, bkd, metric=domain.inner_product())
    eigenvalues = np.asarray(bkd.to_numpy(full.kle().eigenvalues()))
    return float(
        eigenvalues[nmodes:].sum() / eigenvalues.sum()
    )


class TestMPODProvenance:
    r"""A regression guard against substituting a repaired Euclidean POD.

    The eigensolvers solve the weighted problem directly. Someone
    "simplifying" that to a plain SVD plus ``orthonormalize_basis``
    would get a basis that is M-orthonormal and spans the *wrong*
    subspace -- and every shape test, plus ``is_isometry`` itself, would
    still pass. This is the test that would not.
    """

    def _setup(self, bkd: Backend):
        x = _graded_axis(bkd)
        domain = UniformGridDomain([x], bkd)
        snapshots = _snapshots_concentrated_in_the_refined_region(
            bkd, x, 40
        )
        return domain, snapshots

    def _euclidean_then_repaired(self, bkd: Backend, domain, snapshots, p):
        """The tempting shortcut: Euclidean SVD, then re-orthonormalize."""
        centered = snapshots - bkd.reshape(
            bkd.mean(snapshots, axis=1), (int(snapshots.shape[0]), 1)
        )
        left, _, _ = bkd.svd(centered, full_matrices=False)
        repaired = orthonormalize_basis(
            left[:, :p], domain.inner_product(), bkd
        )
        return GramProjectionEncoder(
            repaired, domain.inner_product(), bkd
        )

    def test_m_pod_attains_the_optimal_projection_error(
        self, numpy_bkd: Backend
    ) -> None:
        r"""POD is optimal, so its error *equals* the discarded spectrum.

        Not "is small" -- equals, to solver tolerance. That is the
        defining property of the decomposition, and asserting it needs no
        invented constant: both sides are computed from the data.
        """
        bkd = numpy_bkd
        domain, snapshots = self._setup(bkd)
        for nmodes in (2, 3, 4):
            encoder = fit_kle_encoder(
                snapshots,
                bkd,
                latent_dim=nmodes,
                metric=domain.inner_product(),
            )
            assert _relative_projection_error(
                encoder, snapshots, domain
            ) == pytest.approx(
                _discarded_spectrum_fraction(snapshots, domain, nmodes),
                rel=1e-10,
            )

    def test_a_repaired_euclidean_basis_misses_that_optimum(
        self, numpy_bkd: Backend
    ) -> None:
        """The substitution this guard exists to catch.

        A Euclidean SVD re-orthonormalized against M is M-orthonormal
        and spans the wrong subspace, so it cannot attain the optimum
        the test above pins. The comparison is against POD's own error
        rather than a threshold, so the margin is whatever the grading
        makes it.
        """
        bkd = numpy_bkd
        domain, snapshots = self._setup(bkd)
        nmodes = 3
        m_pod = fit_kle_encoder(
            snapshots,
            bkd,
            latent_dim=nmodes,
            metric=domain.inner_product(),
        )
        repaired = self._euclidean_then_repaired(
            bkd, domain, snapshots, nmodes
        )
        assert _relative_projection_error(
            repaired, snapshots, domain
        ) > _relative_projection_error(m_pod, snapshots, domain)

    def test_orthonormality_is_not_the_discriminator(
        self, numpy_bkd: Backend
    ) -> None:
        """Both bases pass ``is_isometry``, which is the whole point.

        The cheap check cannot distinguish them, so a reader who
        concludes it suffices will substitute the wrong decomposition
        and see nothing fail. Asserted explicitly so that conclusion is
        unavailable.
        """
        bkd = numpy_bkd
        domain, snapshots = self._setup(bkd)
        m_pod = fit_kle_encoder(
            snapshots, bkd, latent_dim=3, metric=domain.inner_product()
        )
        repaired = self._euclidean_then_repaired(bkd, domain, snapshots, 3)
        assert m_pod.is_isometry()
        assert repaired.is_isometry()


class TestBasisOverADomain:
    def test_is_an_isometry_in_the_domain_metric(
        self, bkd: Backend
    ) -> None:
        """What lets a fitter read its coefficient residual as a field error."""
        domain = UniformGridDomain([_graded_axis(bkd)], bkd)
        snapshots = _snapshots_concentrated_in_the_refined_region(
            bkd, domain.sample_points()[0], 30
        )
        encoder = fit_kle_encoder(
            snapshots, bkd, latent_dim=4, metric=domain.inner_product()
        )
        assert encoder.is_isometry()
        assert isinstance(encoder, FieldEncoderProtocol)

    def test_centering_reaches_the_affine_set(self, bkd: Backend) -> None:
        r"""A centered basis reconstructs the mean; a linear one cannot.

        ``fit_kle_encoder`` centers by default, so ``decode`` adds the
        mean back and the reachable set is :math:`\{\bar u + Vz\}`. With
        a mean well outside the span of the modes -- the Darcy shape --
        that is the difference between a usable basis and a useless one.
        """
        domain = UniformGridDomain([bkd.asarray(np.linspace(0, 1, 12))], bkd)
        x = np.asarray(bkd.to_numpy(domain.sample_points()[0]))
        rng = np.random.RandomState(1)
        # Large constant offset plus small variation.
        snapshots = bkd.asarray(
            np.array(
                [
                    50.0 + 0.1 * a * np.sin(np.pi * x)
                    for a in rng.uniform(-1.0, 1.0, 25)
                ]
            ).T
        )
        centered = fit_kle_encoder(
            snapshots, bkd, latent_dim=1, metric=domain.inner_product()
        )
        uncentered = fit_kle_encoder(
            snapshots,
            bkd,
            latent_dim=1,
            center=False,
            metric=domain.inner_product(),
        )

        def relative_error(encoder) -> float:
            return float(
                bkd.norm(
                    encoder.decode(encoder.encode(snapshots)) - snapshots
                )
                / bkd.norm(snapshots)
            )

        # One mode plus the mean spans these snapshots exactly: the mean
        # carries the offset, the mode carries the single sine.
        assert relative_error(centered) < 1e-10
        # One mode alone cannot, because it must spend itself on the
        # offset and has nothing left for the variation. Compared against
        # the centered basis rather than a threshold, so the claim is
        # the ordering rather than a number.
        assert relative_error(uncentered) > relative_error(centered)


class TestMultiFieldComposition:
    """Several fields: one basis each, composed.

    No joint decomposition and no per-component scalings. Each basis is
    orthonormal in its own metric, so fields with wildly different units
    do not compete -- which is what a joint POD would need the scalings
    to repair.
    """

    def _two_fields(self, bkd: Backend):
        d_fine = UniformGridDomain([bkd.asarray(np.linspace(0, 1, 9))], bkd)
        d_coarse = UniformGridDomain([bkd.asarray(np.linspace(0, 1, 5))], bkd)
        x9 = np.asarray(bkd.to_numpy(d_fine.sample_points()[0]))
        x5 = np.asarray(bkd.to_numpy(d_coarse.sample_points()[0]))
        rng = np.random.RandomState(0)
        # The shape parameter varies, not only the amplitude, so each
        # snapshot matrix has full numerical rank and a truncation to
        # three or two modes is a real approximation.
        freqs = 1.0 + 0.6 * rng.uniform(0.0, 1.0, 40)
        shifts = 0.2 * rng.uniform(-1.0, 1.0, 40)
        small = bkd.asarray(
            np.array(
                [np.sin(np.pi * f * x9 + s) for f, s in zip(freqs, shifts)]
            ).T
        )
        # Five orders of magnitude larger, as pressure is beside velocity.
        large = bkd.asarray(
            np.array(
                [
                    1e5 * np.cos(np.pi * f * x5 + s)
                    for f, s in zip(freqs, shifts)
                ]
            ).T
        )
        return (d_fine, small), (d_coarse, large)

    def _product(self, bkd: Backend, first, second, nmodes):
        encoders = [
            fit_kle_encoder(
                snapshots,
                bkd,
                latent_dim=count,
                metric=domain.inner_product(),
            )
            for (domain, snapshots), count in zip((first, second), nmodes)
        ]
        return ProductFieldEncoder(encoders, bkd)

    def test_product_of_per_field_bases_is_an_isometry(
        self, bkd: Backend
    ) -> None:
        first, second = self._two_fields(bkd)
        product = self._product(bkd, first, second, (3, 2))
        assert product.is_isometry()
        assert isinstance(product, FieldEncoderProtocol)
        assert product.latent_dim() == 5
        assert product.full_dim() == 14

    def test_neither_field_is_starved_despite_unit_disparity(
        self, bkd: Backend
    ) -> None:
        """The units problem a joint decomposition has to repair.

        Joint POD maximizes a sum over quantities with different units,
        so the field at 1e5 would take every leading mode. Per-field
        bases are each orthonormal in their own norm, so both reconstruct
        well on their own few modes.
        """
        (d_small, small), (d_large, large) = self._two_fields(bkd)
        product = self._product(
            bkd, (d_small, small), (d_large, large), (3, 2)
        )
        stacked = bkd.concatenate([small, large], axis=0)
        round_trip = product.decode(product.encode(stacked))
        # Each component must reconstruct as well through the product as
        # its own basis does alone -- that is, attain its own optimum.
        # The bounds come from each field's discarded spectrum, so
        # nothing here is a constant chosen by running the code.
        for values, reconstructed, domain, nmodes in (
            (small, round_trip[:9], d_small, 3),
            (large, round_trip[9:], d_large, 2),
        ):
            mean = bkd.reshape(
                bkd.mean(values, axis=1), (int(values.shape[0]), 1)
            )
            metric = domain.inner_product()
            centered = values - mean
            residual = centered - (reconstructed - mean)
            relative = float(
                bkd.sum(residual * metric.apply(residual))
            ) / float(bkd.sum(centered * metric.apply(centered)))
            assert relative == pytest.approx(
                _discarded_spectrum_fraction(values, domain, nmodes),
                rel=1e-10,
            )

    def test_fields_may_live_on_different_grids(
        self, bkd: Backend
    ) -> None:
        """Each component carries its own metric.

        The assumption a shared-metric joint decomposition makes
        silently: that one mass matrix spans every field. Here the two
        grids have different sizes, so no single metric could.
        """
        (d_small, _), (d_large, _) = self._two_fields(bkd)
        assert d_small.nsites() != d_large.nsites()
        assert (
            d_small.inner_product().nstates()
            != d_large.inner_product().nstates()
        )
