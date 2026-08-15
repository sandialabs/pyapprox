"""The evaluation layer must not silently drift from the Derivatives bundle.

``Request``, ``Decoded`` and ``EvalResult`` name their capabilities in
typed fields rather than deriving them from the bundle at runtime. That
is deliberate -- it keeps mypy checking call sites, which is where this
design does most of its work -- but it means the two can drift apart
with nothing to notice: add a field to ``Derivatives`` and no code here
breaks, nothing fails to typecheck, and a user eventually finds that
their capability vanished on the way through an evaluator.

``BUNDLE_FIELD_CARRIERS`` is the single place stating the
correspondence, and this module fails if it stops covering the bundle.

**What this does NOT catch:** changed *semantics*. If ``hvp_batch``'s
shape contract were altered, every assertion here would still pass.
Shape agreement is the marshaller tests' job.
"""

from dataclasses import fields

import pytest
from pyapprox.interface.evaluation.records import (
    BUNDLE_FIELD_CARRIERS,
    Decoded,
    EvalResult,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives

BUNDLE_FIELDS = {f.name for f in fields(Derivatives)}


class TestBundleCoverage:
    def test_every_bundle_field_is_accounted_for(self):
        """A new capability upstream must be triaged here, not ignored.

        If this fails, add the field to ``BUNDLE_FIELD_CARRIERS``: map it
        to the record field that carries it, or to ``None`` with a
        comment saying why it is deliberately not marshalled.
        """
        missing = BUNDLE_FIELDS - set(BUNDLE_FIELD_CARRIERS)
        assert not missing, (
            f"Derivatives grew {sorted(missing)} but the evaluation layer "
            "does not say how (or whether) it carries them. Add each to "
            "BUNDLE_FIELD_CARRIERS in interface/evaluation/records.py."
        )

    def test_no_stale_entries(self):
        """A renamed or removed bundle field must not linger here."""
        stale = set(BUNDLE_FIELD_CARRIERS) - BUNDLE_FIELDS
        assert not stale, (
            f"BUNDLE_FIELD_CARRIERS names {sorted(stale)}, which no longer "
            "exist on Derivatives. Remove or rename them."
        )

    def test_carriers_name_real_record_fields(self):
        """Every non-None carrier must exist on Decoded and EvalResult."""
        decoded_fields = {f.name for f in fields(Decoded)}
        result_fields = {f.name for f in fields(EvalResult)}
        for bundle_field, carrier in BUNDLE_FIELD_CARRIERS.items():
            if carrier is None:
                continue
            assert carrier in decoded_fields, (
                f"{bundle_field} claims to be carried by Decoded."
                f"{carrier}, which does not exist"
            )
            assert carrier in result_fields, (
                f"{bundle_field} claims to be carried by EvalResult."
                f"{carrier}, which does not exist"
            )

    @pytest.mark.parametrize(
        "bundle_field,carrier",
        [
            ("jacobian", "jacobians"),
            ("jacobian_batch", "jacobians"),
            ("hessian", "hessians"),
            ("hessian_batch", "hessians"),
            ("jvp", "jvps"),
            ("hvp", "hvps"),
            ("whvp", "hvps"),
        ],
    )
    def test_expected_correspondences_hold(self, bundle_field, carrier):
        """Pin the mapping so a silent re-pointing is visible in review."""
        assert BUNDLE_FIELD_CARRIERS[bundle_field] == carrier

    def test_hvp_and_whvp_share_a_carrier(self):
        """Same tensor, different contraction -- one field, not two.

        The bundle itself treats them as convertible via resolved_hvp /
        resolved_whvp rather than as unrelated capabilities.
        """
        assert (
            BUNDLE_FIELD_CARRIERS["hvp"] == BUNDLE_FIELD_CARRIERS["whvp"]
        )

    def test_inexact_is_deliberately_not_marshalled(self):
        """``tol`` is a per-call input, not a property of a sample."""
        assert BUNDLE_FIELD_CARRIERS["inexact"] is None

    def test_directional_carriers_differ_by_output_space(self):
        """jvp lands in QoI space, hvp in parameter space.

        Splitting on output space is what keeps a record field's shape
        independent of any discriminator a consumer would have to read
        first.
        """
        assert (
            BUNDLE_FIELD_CARRIERS["jvp"] != BUNDLE_FIELD_CARRIERS["hvp"]
        )


class TestRequestSurface:
    def test_request_can_ask_for_every_marshalled_capability(self):
        """Anything with a carrier must be requestable.

        A capability the layer claims to carry but offers no way to ask
        for is carried in name only.
        """
        askable = {
            "jacobians": Request(jacobians=True),
            "hessians": Request(hessians=True),
        }
        for carrier in askable:
            assert carrier in {f.name for f in fields(Decoded)}

    def test_values_only_is_the_default(self):
        request = Request.values_only()
        assert request.values
        assert not request.jacobians
        assert not request.hessians
        assert not request.wants_jvp()
        assert not request.wants_hvp()

    def test_empty_request_is_rejected(self):
        with pytest.raises(ValueError, match="at least one quantity"):
            Request(values=False)

    def test_weights_without_vectors_is_rejected(self, numpy_bkd):
        """Weights only mean anything applied to a Hessian-vector product."""
        with pytest.raises(ValueError, match="hvp_weights"):
            Request(hvp_weights=numpy_bkd.ones((1, 1)))

    def test_weighted_hvp_is_distinguishable(self, numpy_bkd):
        plain = Request(values=False, hvp_vecs=numpy_bkd.ones((2, 1)))
        weighted = Request(
            values=False,
            hvp_vecs=numpy_bkd.ones((2, 1)),
            hvp_weights=numpy_bkd.ones((1, 1)),
        )
        assert plain.wants_hvp()
        assert not plain.is_weighted_hvp()
        assert weighted.is_weighted_hvp()
