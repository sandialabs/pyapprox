"""Tests for shared search strategies."""

import unittest

from pyapprox.statest.strategies import (
    AllModelsStrategy,
    AllQoIStrategy,
    AllQoISubsetsStrategy,
    AllSubsetsStrategy,
    FixedQoIStrategy,
    FixedSubsetStrategy,
    ListQoIStrategy,
    ListSubsetStrategy,
)
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestModelSubsetStrategy(unittest.TestCase):
    """Tests for ModelSubsetStrategy implementations."""

    def test_all_models_strategy(self) -> None:
        """Returns all models."""
        strategy = AllModelsStrategy()
        subsets = strategy.subsets(nmodels=4)
        self.assertEqual(subsets, [[0, 1, 2, 3]])

    def test_fixed_subset_strategy(self) -> None:
        """Returns specified subset."""
        strategy = FixedSubsetStrategy(model_indices=(0, 1, 3))
        subsets = strategy.subsets(nmodels=4)
        self.assertEqual(subsets, [[0, 1, 3]])

    def test_fixed_subset_strategy_no_zero(self) -> None:
        """Raises ValueError if 0 not included."""
        with self.assertRaises(ValueError):
            FixedSubsetStrategy(model_indices=(1, 2, 3))

    def test_all_subsets_strategy(self) -> None:
        """Generates correct subsets."""
        strategy = AllSubsetsStrategy(min_models=2)
        subsets = strategy.subsets(nmodels=4)
        # 2-model: [0,1], [0,2], [0,3] = 3
        # 3-model: [0,1,2], [0,1,3], [0,2,3] = 3
        # 4-model: [0,1,2,3] = 1
        # Total = 7
        self.assertEqual(len(subsets), 7)
        for subset in subsets:
            self.assertIn(0, subset)

    def test_all_subsets_strategy_max_models(self) -> None:
        """Respects max_models."""
        strategy = AllSubsetsStrategy(min_models=2, max_models=3)
        subsets = strategy.subsets(nmodels=5)
        for subset in subsets:
            self.assertLessEqual(len(subset), 3)

    def test_all_subsets_strategy_min_equals_max(self) -> None:
        """Works when min_models equals max_models."""
        strategy = AllSubsetsStrategy(min_models=3, max_models=3)
        subsets = strategy.subsets(nmodels=4)
        # Only 3-model subsets: [0,1,2], [0,1,3], [0,2,3] = 3
        self.assertEqual(len(subsets), 3)
        for subset in subsets:
            self.assertEqual(len(subset), 3)

    def test_list_subset_strategy(self) -> None:
        """Returns custom list."""
        strategy = ListSubsetStrategy(model_subsets=((0, 1), (0, 2, 3)))
        subsets = strategy.subsets(nmodels=4)
        self.assertEqual(len(subsets), 2)
        self.assertEqual(subsets[0], [0, 1])
        self.assertEqual(subsets[1], [0, 2, 3])

    def test_list_subset_strategy_no_zero(self) -> None:
        """Raises ValueError if subset missing 0."""
        with self.assertRaises(ValueError):
            ListSubsetStrategy(model_subsets=((0, 1), (1, 2)))


class TestQoISubsetStrategy(unittest.TestCase):
    """Tests for QoISubsetStrategy implementations."""

    def test_all_qoi_strategy(self) -> None:
        """Returns all QoI."""
        strategy = AllQoIStrategy()
        subsets = strategy.subsets(nqoi=3)
        self.assertEqual(subsets, [[0, 1, 2]])

    def test_fixed_qoi_strategy(self) -> None:
        """Returns specified subset."""
        strategy = FixedQoIStrategy(qoi_indices=(0, 2))
        subsets = strategy.subsets(nqoi=4)
        self.assertEqual(subsets, [[0, 2]])

    def test_all_qoi_subsets_strategy(self) -> None:
        """Generates correct subsets."""
        strategy = AllQoISubsetsStrategy(min_qoi=1, max_qoi=2)
        subsets = strategy.subsets(nqoi=3)
        # 1-qoi: [0], [1], [2] = 3
        # 2-qoi: [0,1], [0,2], [1,2] = 3
        # Total = 6
        self.assertEqual(len(subsets), 6)

    def test_all_qoi_subsets_with_required(self) -> None:
        """AllQoISubsetsStrategy respects required_qoi."""
        strategy = AllQoISubsetsStrategy(min_qoi=2, required_qoi=(0,))
        subsets = strategy.subsets(nqoi=4)
        # All subsets must contain 0
        for subset in subsets:
            self.assertIn(0, subset)
        # All subsets have at least 2 elements
        for subset in subsets:
            self.assertGreaterEqual(len(subset), 2)

    def test_all_qoi_subsets_required_larger_than_min(self) -> None:
        """When required_qoi has more elements than min_qoi."""
        strategy = AllQoISubsetsStrategy(min_qoi=1, required_qoi=(0, 1, 2))
        subsets = strategy.subsets(nqoi=5)
        # Minimum size is len(required_qoi) = 3, not min_qoi = 1
        for subset in subsets:
            self.assertGreaterEqual(len(subset), 3)
            # All required indices present
            for req in [0, 1, 2]:
                self.assertIn(req, subset)

    def test_all_qoi_subsets_multiple_required(self) -> None:
        """AllQoISubsetsStrategy with multiple required QoI."""
        strategy = AllQoISubsetsStrategy(min_qoi=2, required_qoi=(0, 2))
        subsets = strategy.subsets(nqoi=4)
        for subset in subsets:
            self.assertIn(0, subset)
            self.assertIn(2, subset)
            self.assertGreaterEqual(len(subset), 2)

    def test_all_qoi_subsets_max_qoi(self) -> None:
        """AllQoISubsetsStrategy respects max_qoi."""
        strategy = AllQoISubsetsStrategy(min_qoi=1, max_qoi=2)
        subsets = strategy.subsets(nqoi=5)
        for subset in subsets:
            self.assertLessEqual(len(subset), 2)

    def test_list_qoi_strategy(self) -> None:
        """Returns custom list."""
        strategy = ListQoIStrategy(qoi_subsets=((0,), (1, 2)))
        subsets = strategy.subsets(nqoi=3)
        self.assertEqual(len(subsets), 2)
        self.assertEqual(subsets[0], [0])
        self.assertEqual(subsets[1], [1, 2])

    def test_strategy_descriptions(self) -> None:
        """All strategies have description."""
        strategies = [
            AllModelsStrategy(),
            FixedSubsetStrategy(model_indices=(0, 1)),
            AllSubsetsStrategy(min_models=2),
            ListSubsetStrategy(model_subsets=((0, 1),)),
            AllQoIStrategy(),
            FixedQoIStrategy(qoi_indices=(0,)),
            AllQoISubsetsStrategy(min_qoi=1),
            ListQoIStrategy(qoi_subsets=((0,),)),
        ]
        for s in strategies:
            self.assertIsInstance(s.description(), str)
            self.assertGreater(len(s.description()), 0)

    def test_all_qoi_subsets_description_with_required(self) -> None:
        """Description includes required_qoi when present."""
        strategy = AllQoISubsetsStrategy(min_qoi=2, required_qoi=(0, 1))
        desc = strategy.description()
        self.assertIn("required", desc)
        self.assertIn("(0, 1)", desc)


if __name__ == "__main__":
    unittest.main()
