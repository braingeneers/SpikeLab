"""Tests for spikedata/decoding.py — classifier-based decoding."""

import numpy as np
import pytest

from spikelab.spikedata.decoding import (
    cross_validated_decode,
    regularization_sweep,
    latency_dependent_decoding,
)


def _separable_dataset(n_per_class=20, n_features=10, n_classes=3, seed=0):
    """Linearly separable synthetic dataset: per-class mean shift in feature space."""
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for cls in range(n_classes):
        center = np.zeros(n_features)
        center[cls % n_features] = 5.0  # large shift in one feature per class
        X.append(rng.normal(center, 1.0, (n_per_class, n_features)))
        y.extend([cls] * n_per_class)
    return np.vstack(X), np.asarray(y)


def _random_dataset(n_samples=40, n_features=10, n_classes=4, seed=1):
    """Pure-noise dataset — labels are random, no signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    y = rng.integers(0, n_classes, n_samples)
    return X, y


class TestCrossValidatedDecode:
    """Tests for cross_validated_decode."""

    def test_separable_high_accuracy(self):
        """
        On a linearly separable dataset, ridge decoding achieves high accuracy.

        Tests:
            (Test Case 1) Accuracy > 0.9 on simple separable data.
        """
        X, y = _separable_dataset(n_per_class=20, n_classes=3)
        result = cross_validated_decode(X, y, classifier="ridge", cv=5, random_state=0)
        assert result["accuracy"] > 0.9

    def test_random_chance_level(self):
        """
        On pure-noise data, accuracy is near 1/K chance level.

        Tests:
            (Test Case 1) Accuracy < 0.5 on 4-class random data (chance = 0.25).
        """
        X, y = _random_dataset(n_samples=60, n_features=20, n_classes=4, seed=2)
        result = cross_validated_decode(X, y, classifier="ridge", cv=5, random_state=0)
        assert result["accuracy"] < 0.5

    def test_returns_expected_keys(self):
        """
        Output dict contains all documented keys with correct shapes.

        Tests:
            (Test Case 1) accuracy is a float.
            (Test Case 2) predictions has shape (n_samples,).
            (Test Case 3) confusion_matrix has shape (K, K).
            (Test Case 4) classes contains all unique labels.
            (Test Case 5) classifier_name matches input.
        """
        X, y = _separable_dataset(n_per_class=10, n_classes=3)
        result = cross_validated_decode(X, y, classifier="ridge", cv=3, random_state=0)
        assert isinstance(result["accuracy"], float)
        assert result["predictions"].shape == y.shape
        assert result["confusion_matrix"].shape == (3, 3)
        assert sorted(result["classes"]) == [0, 1, 2]
        assert result["classifier_name"] == "ridge"

    def test_loo_cv(self):
        """
        Leave-One-Out CV runs and returns one prediction per sample.

        Tests:
            (Test Case 1) per_fold_accuracy length equals n_samples.
        """
        X, y = _separable_dataset(n_per_class=8, n_classes=2)
        result = cross_validated_decode(
            X, y, classifier="ridge", cv="loo", random_state=0
        )
        assert len(result["per_fold_accuracy"]) == len(y)

    def test_mlp_backend(self):
        """
        MLPClassifier backend runs end-to-end.

        Tests:
            (Test Case 1) Returns valid accuracy in [0, 1].
        """
        X, y = _separable_dataset(n_per_class=15, n_classes=2)
        result = cross_validated_decode(
            X,
            y,
            classifier="mlp",
            cv=3,
            classifier_kwargs={"hidden_layer_sizes": (16,), "max_iter": 200},
            random_state=0,
        )
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_random_forest_backend(self):
        """
        RandomForestClassifier backend runs end-to-end.

        Tests:
            (Test Case 1) Returns valid accuracy in [0, 1].
        """
        X, y = _separable_dataset(n_per_class=10, n_classes=3)
        result = cross_validated_decode(
            X,
            y,
            classifier="random_forest",
            cv=3,
            classifier_kwargs={"n_estimators": 25},
            random_state=0,
        )
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_unknown_classifier_raises(self):
        """
        Unknown classifier raises ValueError.

        Tests:
            (Test Case 1) ValueError for bogus classifier name.
        """
        X, y = _separable_dataset()
        with pytest.raises(ValueError, match="classifier must be one of"):
            cross_validated_decode(X, y, classifier="bogus", cv=3)

    def test_unknown_cv_raises(self):
        """
        Invalid cv raises ValueError.

        Tests:
            (Test Case 1) ValueError for unknown cv string.
            (Test Case 2) ValueError for cv < 2.
        """
        X, y = _separable_dataset()
        with pytest.raises(ValueError, match="cv string must be 'loo'"):
            cross_validated_decode(X, y, cv="bogus")
        with pytest.raises(ValueError, match="must be >= 2"):
            cross_validated_decode(X, y, cv=1)

    def test_shape_mismatch_raises(self):
        """
        Mismatched X / y lengths raise ValueError.

        Tests:
            (Test Case 1) ValueError when X and y have different lengths.
        """
        X = np.zeros((10, 5))
        y = np.zeros(8)
        with pytest.raises(ValueError, match="same number of samples"):
            cross_validated_decode(X, y, cv=3)

    def test_single_class_raises(self):
        """
        Only one class present raises ValueError.

        Tests:
            (Test Case 1) ValueError when all y are identical.
        """
        X = np.zeros((10, 5))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            cross_validated_decode(X, y, cv=3)


class TestRegularizationSweep:
    """Tests for regularization_sweep."""

    def test_returns_per_alpha_accuracy(self):
        """
        Sweep returns one accuracy per alpha and identifies the best.

        Tests:
            (Test Case 1) mean_accuracy has shape (n_alphas,).
            (Test Case 2) best_alpha is among the input alphas.
            (Test Case 3) best_accuracy equals max(mean_accuracy).
        """
        X, y = _separable_dataset(n_per_class=15, n_classes=3)
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        result = regularization_sweep(
            X, y, alphas, classifier="ridge", cv=3, random_state=0
        )
        assert result["mean_accuracy"].shape == (len(alphas),)
        assert result["best_alpha"] in alphas
        assert result["best_accuracy"] == result["mean_accuracy"].max()

    def test_per_alpha_predictions_shape(self):
        """
        Per-alpha predictions has shape (n_alphas, n_samples).

        Tests:
            (Test Case 1) Shape matches expected.
        """
        X, y = _separable_dataset(n_per_class=10, n_classes=2)
        alphas = [0.1, 1.0]
        result = regularization_sweep(
            X, y, alphas, classifier="ridge", cv=3, random_state=0
        )
        assert result["per_alpha_predictions"].shape == (2, len(y))

    def test_empty_alphas_raises(self):
        """
        Empty alphas raises ValueError.

        Tests:
            (Test Case 1) ValueError for empty alpha list.
        """
        X, y = _separable_dataset()
        with pytest.raises(ValueError, match="non-empty"):
            regularization_sweep(X, y, [], cv=3)


class TestLatencyDependentDecoding:
    """Tests for latency_dependent_decoding."""

    def _stim_response_stack(self, n_classes=3, n_per_class=8, U=12, T=40, seed=0):
        """Build a (U, T, S) stack where class identity drives a specific
        latency band of activity. Class c puts response in bins [c*10, c*10+10]."""
        rng = np.random.default_rng(seed)
        S = n_classes * n_per_class
        stack = rng.poisson(0.5, (U, T, S)).astype(float)
        labels = []
        for c in range(n_classes):
            for k in range(n_per_class):
                idx = c * n_per_class + k
                response_start = c * 10
                response_end = response_start + 10
                # Strong activity in the class-specific band, top half of units
                stack[U // 2 :, response_start:response_end, idx] += rng.poisson(
                    5, (U - U // 2, 10)
                )
                labels.append(c)
        return stack, np.asarray(labels)

    def test_class_specific_window_decodes(self):
        """
        Decoding accuracy is highest in the latency window that carries the
        class-specific signal.

        Tests:
            (Test Case 1) Class-specific window accuracy is well above chance.
        """
        stack, labels = self._stim_response_stack(
            n_classes=3, n_per_class=12, U=12, T=40, seed=0
        )
        windows = [(0, 10), (10, 20), (20, 30), (30, 40)]
        result = latency_dependent_decoding(
            stack,
            labels,
            windows,
            bin_size=1.0,
            classifier="ridge",
            cv=4,
            random_state=0,
        )
        assert result["accuracies"].shape == (4,)
        # The first three windows carry the class signal; the last is empty.
        assert result["accuracies"][:3].mean() > 0.6  # well above chance (1/3)

    def test_returns_expected_keys(self):
        """
        Output contains all documented keys.

        Tests:
            (Test Case 1) windows, accuracies, per_window_predictions,
                classifier_name present.
        """
        stack, labels = self._stim_response_stack(
            n_classes=2, n_per_class=10, U=8, T=30, seed=1
        )
        windows = [(0, 10), (10, 20)]
        result = latency_dependent_decoding(
            stack, labels, windows, bin_size=1.0, cv=3, random_state=0
        )
        for k in ("windows", "accuracies", "per_window_predictions", "classifier_name"):
            assert k in result
        assert result["per_window_predictions"].shape == (2, len(labels))

    def test_bad_stack_shape_raises(self):
        """
        Non-3D stack raises ValueError.

        Tests:
            (Test Case 1) 2-D input raises.
        """
        with pytest.raises(ValueError, match="3-D"):
            latency_dependent_decoding(
                np.zeros((5, 10)),
                np.array([0, 1, 0, 1, 0]),
                [(0, 5)],
                bin_size=1.0,
                cv=2,
            )

    def test_bad_window_raises(self):
        """
        Empty / malformed window raises ValueError.

        Tests:
            (Test Case 1) end <= start raises.
            (Test Case 2) Wrong tuple form raises.
            (Test Case 3) Window mapped to empty bin range raises.
        """
        stack = np.zeros((4, 20, 10))
        labels = np.array([0, 1] * 5)
        with pytest.raises(ValueError, match="end must be greater"):
            latency_dependent_decoding(stack, labels, [(10, 5)], bin_size=1.0, cv=2)
        with pytest.raises(ValueError, match="tuple"):
            latency_dependent_decoding(stack, labels, [5.0], bin_size=1.0, cv=2)
        with pytest.raises(ValueError, match="empty bin range"):
            latency_dependent_decoding(stack, labels, [(100, 200)], bin_size=1.0, cv=2)

    def test_labels_length_mismatch_raises(self):
        """
        Mismatched labels length raises ValueError.

        Tests:
            (Test Case 1) ValueError when labels length != S.
        """
        stack = np.zeros((4, 20, 10))
        labels = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="length S"):
            latency_dependent_decoding(stack, labels, [(0, 10)], bin_size=1.0, cv=2)
