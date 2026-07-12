"""Runtime guarantees of the Derivatives bundle (promoted Phase-0 spike).

Covers: deepcopy/pickle of self-referential bundles (the GP fitter
deep-copies surrogates), construction-time TypeErrors on the untyped-caller
path, frozen semantics, with_ composition, opt-in shape validation,
resolver lift/synthesis in both directions, protocol isinstance behavior
(including the AutodiffBackend semantic-contract negative on the REAL
backends), and autograd composition.
"""

import copy
import dataclasses
import pickle

import pytest

from pyapprox.interface.functions.autograd import (
    OverrideDerivatives,
    WithAutogradJacobian,
    autograd_derivatives,
)
from pyapprox.interface.functions.derivatives import (
    Derivatives,
    InexactSuite,
    with_shape_validation,
)
from pyapprox.interface.functions.protocols import (
    NonlinearConstraintProtocol,
    ObjectiveProtocol,
)
from pyapprox.util.backends.autodiff import AutodiffBackend


class QuadraticObjective:
    """f(x) = sum_i x_i^2 with the full analytic second-order set."""

    def __init__(self, bkd, nvars):
        self._bkd = bkd
        self._nvars = nvars
        self._derivs = Derivatives(
            jacobian=self.jacobian, hvp=self.hvp, whvp=self.whvp
        )

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._nvars

    def nqoi(self):
        return 1

    def __call__(self, samples):
        return self._bkd.sum(samples * samples, axis=0)[None, :]

    def jacobian(self, sample):
        return (sample * 2.0).T

    def hvp(self, sample, vec):
        return vec * 2.0

    def whvp(self, sample, vec, weights):
        return vec * 2.0 * weights[0, 0]

    def derivatives(self):
        return self._derivs


class QuadraticNoDerivatives(QuadraticObjective):
    """Same evaluation, empty bundle: the autograd-composition target."""

    def derivatives(self):
        return Derivatives.none()


class ScalarConstraintWithHVP(QuadraticObjective):
    """Scalar constraint exposing only plain hvp; whvp comes via the
    resolver lift (strictly more capable than the status quo)."""

    def __init__(self, bkd, nvars):
        super().__init__(bkd, nvars)
        self._derivs = Derivatives.second_order(self.jacobian, self.hvp)

    def lb(self):
        return self._bkd.full((1, 1), -1.0)

    def ub(self):
        return self._bkd.full((1, 1), 1.0)


class CycleObj:
    """Objective whose bundle holds a bound method to itself (cycle)."""

    def __init__(self):
        self._x = 3.0
        self._derivs = Derivatives(jacobian=self.jacobian)

    def jacobian(self, s):
        return s * self._x

    def derivatives(self):
        return self._derivs


class TestCopySemantics:
    def test_deepcopy_repoints_cycle(self, numpy_bkd):
        obj = CycleObj()
        assert obj._derivs.jacobian.__self__ is obj
        clone = copy.deepcopy(obj)
        assert clone._derivs.jacobian.__self__ is clone
        assert clone._derivs.jacobian.__self__ is not obj

    def test_deepcopy_is_live_not_stale(self, numpy_bkd):
        obj = CycleObj()
        clone = copy.deepcopy(obj)
        clone._x = 5.0
        assert clone._derivs.jacobian(2.0) == 10.0
        assert obj._derivs.jacobian(2.0) == 6.0

    def test_pickle_roundtrip_repoints_cycle(self, numpy_bkd):
        obj = CycleObj()
        restored = pickle.loads(pickle.dumps(obj))
        assert restored._derivs.jacobian.__self__ is restored
        assert restored._derivs.jacobian(2.0) == 6.0

    def test_synthesized_resolver_fields_pickle(self, numpy_bkd):
        # resolver synthesis/lift must be picklable (multiprocessing):
        # module-level callable objects, never closures
        objective = QuadraticObjective(numpy_bkd, 2)
        hessp = objective.derivatives().with_(hvp=None).resolved_hvp(
            1, numpy_bkd
        )
        whvp = objective.derivatives().with_(whvp=None).resolved_whvp(1)
        x0 = numpy_bkd.array([[0.5], [0.5]])
        vec = numpy_bkd.array([[1.0], [2.0]])
        restored_hessp = pickle.loads(pickle.dumps(hessp))
        numpy_bkd.assert_allclose(
            restored_hessp(x0, vec), numpy_bkd.array([[2.0], [4.0]])
        )
        restored_whvp = pickle.loads(pickle.dumps(whvp))
        numpy_bkd.assert_allclose(
            restored_whvp(x0, vec, numpy_bkd.full((1, 1), 4.0)),
            numpy_bkd.array([[8.0], [16.0]]),
        )

    def test_shape_validated_bundle_pickles(self, numpy_bkd):
        objective = QuadraticObjective(numpy_bkd, 2)
        checked = with_shape_validation(
            objective.derivatives(), nvars=2, nqoi=1
        )
        restored = pickle.loads(pickle.dumps(checked))
        x0 = numpy_bkd.array([[0.5], [0.5]])
        numpy_bkd.assert_allclose(
            restored.jacobian(x0), numpy_bkd.array([[1.0, 1.0]])
        )

    def test_deepcopy_of_array_producer(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        clone = copy.deepcopy(objective)
        assert clone.derivatives().jacobian.__self__ is clone
        x0 = bkd.array([[0.5], [2.0]])
        bkd.assert_allclose(
            clone.derivatives().jacobian(x0), bkd.array([[1.0, 4.0]])
        )


class TestConstructionValidation:
    def test_non_callable_field_raises(self, numpy_bkd):
        with pytest.raises(TypeError, match="'jacobian' must be callable"):
            Derivatives(jacobian=42)

    def test_computed_value_mistake_raises(self, numpy_bkd):
        obj = CycleObj()
        with pytest.raises(TypeError, match="computed value"):
            Derivatives(jacobian=obj.jacobian(2.0))

    def test_batch_fields_validated(self, numpy_bkd):
        with pytest.raises(TypeError, match="'hvp_batch' must be callable"):
            Derivatives(hvp_batch=1.0)
        with pytest.raises(
            TypeError, match="'hessian_batch' must be callable"
        ):
            Derivatives(hessian_batch=1.0)

    def test_second_order_none_hvp_raises(self, numpy_bkd):
        obj = CycleObj()
        with pytest.raises(TypeError, match="second_order requires"):
            Derivatives.second_order(obj.jacobian, None)

    def test_second_order_weighted_none_whvp_raises(self, numpy_bkd):
        obj = CycleObj()
        with pytest.raises(TypeError, match="second_order_weighted"):
            Derivatives.second_order_weighted(obj.jacobian, None)

    def test_first_order_none_raises(self, numpy_bkd):
        with pytest.raises(TypeError, match="first_order requires"):
            Derivatives.first_order(None)

    def test_inexact_wrong_type_raises(self, numpy_bkd):
        with pytest.raises(TypeError, match="'inexact' must be"):
            Derivatives(inexact=lambda x, tol: x)

    def test_frozen_at_runtime(self, numpy_bkd):
        d = Derivatives.none()
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.jacobian = None


class TestWith:
    def test_override_and_removal(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        d = objective.derivatives()
        stripped = d.with_(hvp=None, whvp=None)
        assert stripped.hvp is None and stripped.whvp is None
        assert stripped.jacobian is d.jacobian
        restored = stripped.with_(hvp=objective.hvp)
        assert restored.hvp is not None
        assert d.hvp is not None  # original bundle untouched

    def test_with_batch_fields(self, bkd):
        objective = QuadraticObjective(bkd, 2)

        def hessian(sample):
            return bkd.eye(2) * 2.0

        d = objective.derivatives().with_(hessian=hessian)
        assert d.hessian is hessian
        assert d.with_(hessian=None).hessian is None


class TestShapeValidation:
    def test_passes_correct_shapes(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        checked = with_shape_validation(
            objective.derivatives(), nvars=2, nqoi=1
        )
        x0 = bkd.array([[0.5], [0.5]])
        assert checked.jacobian(x0).shape == (1, 2)
        assert checked.hvp(x0, x0).shape == (2, 1)
        assert checked.whvp(x0, x0, bkd.ones((1, 1))).shape == (2, 1)

    def test_rejects_wrong_jacobian_shape(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        checked = with_shape_validation(
            objective.derivatives(), nvars=2, nqoi=4
        )
        with pytest.raises(ValueError):
            checked.jacobian(bkd.array([[0.5], [0.5]]))

    def test_rejects_wrong_input_shape(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        checked = with_shape_validation(
            objective.derivatives(), nvars=2, nqoi=1
        )
        with pytest.raises(ValueError):
            checked.jacobian(bkd.array([[0.5]]))

    def test_rejects_wrong_whvp_weights_shape(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        checked = with_shape_validation(
            objective.derivatives(), nvars=2, nqoi=1
        )
        x0 = bkd.array([[0.5], [0.5]])
        with pytest.raises(ValueError, match="whvp weights"):
            checked.whvp(x0, x0, bkd.ones((3, 1)))

    def test_batch_fields_checked(self, bkd):
        def hvp_batch(samples, vecs):
            # WRONG: returns (nvars, nsamples) instead of (nsamples, nvars)
            return vecs * 2.0

        d = Derivatives(hvp_batch=hvp_batch)
        checked = with_shape_validation(d, nvars=3, nqoi=1)
        samples = bkd.ones((3, 2))
        with pytest.raises(ValueError, match="hvp_batch output"):
            checked.hvp_batch(samples, samples)


class TestResolvers:
    def test_synthesizes_hvp_from_whvp_when_scalar(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        whvp_only = objective.derivatives().with_(hvp=None)
        hessp = whvp_only.resolved_hvp(1, bkd)
        assert hessp is not None
        x0 = bkd.array([[0.5], [0.5]])
        vec = bkd.array([[1.0], [2.0]])
        # whvp with w=[1] IS the plain hvp: H v = 2 v
        bkd.assert_allclose(hessp(x0, vec), bkd.array([[2.0], [4.0]]))

    def test_lifts_whvp_from_hvp_when_scalar(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        hvp_only = objective.derivatives().with_(whvp=None)
        whvp = hvp_only.resolved_whvp(1)
        assert whvp is not None
        x0 = bkd.array([[0.5], [0.5]])
        vec = bkd.array([[1.0], [2.0]])
        weights = bkd.full((1, 1), 4.0)
        # w[0] * hvp = 4 * 2 v
        bkd.assert_allclose(
            whvp(x0, vec, weights), bkd.array([[8.0], [16.0]])
        )

    def test_prefers_native_fields(self, bkd):
        d = QuadraticObjective(bkd, 2).derivatives()
        assert d.resolved_hvp(1, bkd) is d.hvp
        assert d.resolved_whvp(1) is d.whvp

    def test_no_synthesis_when_vector_valued(self, bkd):
        d = QuadraticObjective(bkd, 2).derivatives()
        assert d.with_(hvp=None).resolved_hvp(3, bkd) is None
        assert d.with_(whvp=None).resolved_whvp(3) is None

    def test_empty_bundle_resolves_to_none(self, bkd):
        d = Derivatives.none()
        assert d.resolved_hvp(1, bkd) is None
        assert d.resolved_whvp(1) is None

    def test_no_synthesis_from_materialized_hessian(self, bkd):
        # matrix-free must not silently become O(nvars^2): a bundle with
        # only hessian populated resolves NO hvp
        d = Derivatives(hessian=lambda sample: bkd.eye(2) * 2.0)
        assert d.resolved_hvp(1, bkd) is None
        assert d.resolved_whvp(1) is None

    def test_constraint_scalar_lift_uses_owning_nqoi(self, bkd):
        con = ScalarConstraintWithHVP(bkd, 2)
        whvp = con.derivatives().resolved_whvp(con.nqoi())
        assert whvp is not None
        x0 = bkd.array([[0.5], [0.5]])
        vec = bkd.array([[1.0], [1.0]])
        bkd.assert_allclose(
            whvp(x0, vec, bkd.full((1, 1), 3.0)),
            bkd.array([[6.0], [6.0]]),
        )


class TestInexactSuite:
    def test_carried_in_bundle(self, bkd):
        def value(sample, tol):
            return sample * (1.0 + tol)

        suite = InexactSuite(value=value)
        d = Derivatives.none().with_(inexact=suite)
        assert d.inexact is suite
        assert d.inexact.jacobian is None

    def test_value_required_callable(self, numpy_bkd):
        with pytest.raises(TypeError, match="'value' must be callable"):
            InexactSuite(value=1.0)


class TestProtocolChecks:
    def test_objective_isinstance(self, bkd):
        assert isinstance(QuadraticObjective(bkd, 2), ObjectiveProtocol)
        assert not isinstance(object(), ObjectiveProtocol)

    def test_constraint_isinstance(self, bkd):
        assert isinstance(
            ScalarConstraintWithHVP(bkd, 2), NonlinearConstraintProtocol
        )
        # objectives lack lb/ub: they are NOT constraints
        assert not isinstance(
            QuadraticObjective(bkd, 2), NonlinearConstraintProtocol
        )

    def test_numpy_backend_is_not_autodiff(self, numpy_bkd):
        # Semantic contract: only backends with jacobian/hvp methods opt
        # into autograd dispatch. NumpyBkd must NEVER pass.
        assert not isinstance(numpy_bkd, AutodiffBackend)

    def test_torch_backend_is_autodiff(self, torch_bkd):
        assert isinstance(torch_bkd, AutodiffBackend)


class TestAutogradComposition:
    def test_wrapper_rejects_non_autodiff_backend(self, numpy_bkd):
        bare = QuadraticNoDerivatives(numpy_bkd, 2)
        with pytest.raises(TypeError, match="AutodiffBackend"):
            WithAutogradJacobian(bare, numpy_bkd)

    def test_helper_rejects_non_autodiff_backend(self, numpy_bkd):
        objective = QuadraticObjective(numpy_bkd, 2)
        with pytest.raises(TypeError, match="AutodiffBackend"):
            autograd_derivatives(objective, numpy_bkd)

    def test_wrapper_fills_jacobian(self, torch_bkd):
        bare = QuadraticNoDerivatives(torch_bkd, 2)
        assert bare.derivatives().jacobian is None
        wrapped = WithAutogradJacobian(bare, torch_bkd)
        assert isinstance(wrapped, ObjectiveProtocol)
        jac = wrapped.derivatives().jacobian
        assert jac is not None
        x0 = torch_bkd.array([[0.5], [2.0]])
        torch_bkd.assert_allclose(
            jac(x0), torch_bkd.array([[1.0, 4.0]]), rtol=1e-10
        )

    def test_wrapper_preserves_inner_capabilities(self, torch_bkd):
        objective = QuadraticObjective(torch_bkd, 2)
        stripped = objective.derivatives().with_(jacobian=None)

        class _NoJac(QuadraticObjective):
            def derivatives(self):
                return stripped

        wrapped = WithAutogradJacobian(_NoJac(torch_bkd, 2), torch_bkd)
        d = wrapped.derivatives()
        assert d.jacobian is not None
        # untouched pass-through: same stored field object, not a re-access
        assert d.hvp is objective.derivatives().hvp

    def test_helper_first_order(self, torch_bkd):
        objective = QuadraticNoDerivatives(torch_bkd, 2)
        d = autograd_derivatives(objective, torch_bkd)
        assert d.hvp is None
        x0 = torch_bkd.array([[0.5], [2.0]])
        torch_bkd.assert_allclose(
            d.jacobian(x0), torch_bkd.array([[1.0, 4.0]]), rtol=1e-10
        )

    def test_helper_fill_hvp(self, torch_bkd):
        objective = QuadraticNoDerivatives(torch_bkd, 2)
        d = autograd_derivatives(objective, torch_bkd, fill_hvp=True)
        assert d.hvp is not None
        x0 = torch_bkd.array([[0.5], [2.0]])
        vec = torch_bkd.array([[1.0], [3.0]])
        torch_bkd.assert_allclose(
            d.hvp(x0, vec), torch_bkd.array([[2.0], [6.0]]), rtol=1e-10
        )


class TestOverrideDerivatives:
    def test_masks_capability_without_touching_producer(self, bkd):
        objective = QuadraticObjective(bkd, 2)
        jac_only = OverrideDerivatives(
            objective, objective.derivatives().with_(hvp=None, whvp=None)
        )
        assert isinstance(jac_only, ObjectiveProtocol)
        d = jac_only.derivatives()
        assert d.jacobian is not None
        assert d.hvp is None and d.whvp is None
        # producer untouched
        assert objective.derivatives().hvp is not None
        # evaluation delegates
        x0 = bkd.array([[1.0], [2.0]])
        bkd.assert_allclose(jac_only(x0), objective(x0))

    def test_rejects_non_bundle(self, numpy_bkd):
        objective = QuadraticObjective(numpy_bkd, 2)
        with pytest.raises(TypeError, match="Derivatives bundle"):
            OverrideDerivatives(objective, {"jacobian": None})
