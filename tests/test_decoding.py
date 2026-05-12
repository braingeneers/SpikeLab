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


class TestCrossEntropy:
    """Tests for log_loss / predicted_probabilities additions to cross_validated_decode."""

    def test_logistic_returns_log_loss(self):
        """
        LogisticRegression backend returns a finite log_loss and per-fold log-loss.

        Tests:
            (Test Case 1) log_loss is finite.
            (Test Case 2) per_fold_log_loss has expected length.
            (Test Case 3) predicted_probabilities has shape (n_samples, K).
        """
        X, y = _separable_dataset(n_per_class=15, n_classes=3)
        result = cross_validated_decode(
            X,
            y,
            classifier="logistic",
            cv=3,
            classifier_kwargs={"max_iter": 200},
            random_state=0,
        )
        assert np.isfinite(result["log_loss"])
        assert result["per_fold_log_loss"].shape == (3,)
        assert result["predicted_probabilities"].shape == (len(y), 3)

    def test_log_loss_lower_for_separable(self):
        """
        Separable data yields lower log_loss than random-noise data.

        Tests:
            (Test Case 1) log_loss(separable) < log_loss(random).
        """
        X1, y1 = _separable_dataset(n_per_class=20, n_classes=3)
        X2, y2 = _random_dataset(n_samples=60, n_features=10, n_classes=3, seed=0)
        r1 = cross_validated_decode(X1, y1, classifier="logistic", cv=5, random_state=0)
        r2 = cross_validated_decode(X2, y2, classifier="logistic", cv=5, random_state=0)
        assert r1["log_loss"] < r2["log_loss"]

    def test_ridge_uses_decision_function_softmax(self):
        """
        Ridge classifier (no predict_proba) falls back to softmax over
        decision_function and still yields a finite log_loss.

        Tests:
            (Test Case 1) Probabilities are present.
            (Test Case 2) log_loss is finite.
        """
        X, y = _separable_dataset(n_per_class=15, n_classes=3)
        result = cross_validated_decode(X, y, classifier="ridge", cv=3, random_state=0)
        assert result["predicted_probabilities"] is not None
        assert np.isfinite(result["log_loss"])

    def test_probabilities_sum_to_one(self):
        """
        Per-row probabilities sum to ~1.

        Tests:
            (Test Case 1) Row sums are within 1e-6 of 1.0.
        """
        X, y = _separable_dataset(n_per_class=10, n_classes=3)
        result = cross_validated_decode(
            X, y, classifier="logistic", cv=3, random_state=0
        )
        np.testing.assert_allclose(
            result["predicted_probabilities"].sum(axis=1), 1.0, atol=1e-6
        )


class TestTrainTestDecoding:
    """Tests for train_test_decoding."""

    def test_perfect_train_test(self):
        """
        Train and test on identical separable data yield high accuracy.

        Tests:
            (Test Case 1) Accuracy > 0.9.
            (Test Case 2) log_loss is finite.
        """
        from spikelab.spikedata.decoding import train_test_decoding

        X, y = _separable_dataset(n_per_class=15, n_classes=3)
        n = len(X)
        idx = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        train_idx = idx[: n // 2]
        test_idx = idx[n // 2 :]
        result = train_test_decoding(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
            classifier="logistic",
            classifier_kwargs={"max_iter": 200},
            random_state=0,
        )
        assert result["accuracy"] > 0.9
        assert np.isfinite(result["log_loss"])
        assert result["predictions"].shape == (len(test_idx),)

    def test_returns_expected_keys(self):
        """
        Result dict has all documented keys.

        Tests:
            (Test Case 1) Required keys present.
        """
        from spikelab.spikedata.decoding import train_test_decoding

        X, y = _separable_dataset(n_per_class=10, n_classes=2)
        order = np.random.default_rng(0).permutation(len(X))
        X = X[order]
        y = y[order]
        result = train_test_decoding(
            X[:10],
            y[:10],
            X[10:],
            y[10:],
            classifier="logistic",
            random_state=0,
        )
        for k in (
            "accuracy",
            "log_loss",
            "predictions",
            "predicted_probabilities",
            "true_labels",
            "confusion_matrix",
            "classes",
            "classifier_name",
        ):
            assert k in result

    def test_feature_mismatch_raises(self):
        """
        Different feature counts in train and test raise ValueError.

        Tests:
            (Test Case 1) ValueError for different n_features.
        """
        from spikelab.spikedata.decoding import train_test_decoding

        with pytest.raises(ValueError, match="number of features"):
            train_test_decoding(
                np.zeros((5, 4)),
                np.array([0, 1, 0, 1, 0]),
                np.zeros((3, 5)),
                np.array([0, 1, 0]),
            )


class TestTemporalDecodingDecay:
    """Tests for temporal_decoding_decay."""

    def test_decay_on_drifting_data(self):
        """
        When test groups drift away from training distribution, accuracy drops
        and log_loss rises.

        Tests:
            (Test Case 1) accuracies has shape (n_groups,).
            (Test Case 2) Drifted group has higher log_loss than matched group.
            (Test Case 3) Matched group has higher accuracy than drifted group.
        """
        from spikelab.spikedata.decoding import temporal_decoding_decay

        rng = np.random.default_rng(0)
        n = 60
        X = np.zeros((n, 2))
        y = np.zeros(n, dtype=int)
        for i in range(n):
            cls = i % 2
            center = np.array([0.0, 0.0]) if cls == 0 else np.array([5.0, 0.0])
            X[i] = center + rng.normal(0, 0.5, 2)
            y[i] = cls

        train_idx = np.arange(20)
        identical_group = np.arange(20, 40)
        drifted_group = np.arange(40, 60)
        for i in drifted_group:
            if y[i] == 1:
                X[i] = np.array([0.5, 0.0]) + rng.normal(0, 0.5, 2)

        result = temporal_decoding_decay(
            X,
            y,
            train_idx,
            [identical_group, drifted_group],
            classifier="logistic",
            classifier_kwargs={"max_iter": 200},
            random_state=0,
        )
        assert result["accuracies"].shape == (2,)
        assert result["log_losses"].shape == (2,)
        assert result["log_losses"][1] > result["log_losses"][0]
        assert result["accuracies"][0] > result["accuracies"][1]

    def test_returns_expected_keys(self):
        """
        Result dict has all documented keys.

        Tests:
            (Test Case 1) Required keys present.
            (Test Case 2) per_group_predictions length matches n_groups.
        """
        from spikelab.spikedata.decoding import temporal_decoding_decay

        X, y = _separable_dataset(n_per_class=10, n_classes=2)
        order = np.random.default_rng(0).permutation(len(X))
        X = X[order]
        y = y[order]
        result = temporal_decoding_decay(
            X,
            y,
            train_indices=np.arange(10),
            test_index_groups=[np.arange(10, 15), np.arange(15, 20)],
            classifier="logistic",
            random_state=0,
        )
        for k in (
            "train_indices",
            "test_index_groups",
            "accuracies",
            "log_losses",
            "per_group_predictions",
            "per_group_probabilities",
            "classes",
            "classifier_name",
        ):
            assert k in result
        assert len(result["per_group_predictions"]) == 2

    def test_empty_train_raises(self):
        """
        Empty train_indices raises ValueError.

        Tests:
            (Test Case 1) ValueError for empty train.
        """
        from spikelab.spikedata.decoding import temporal_decoding_decay

        X, y = _separable_dataset(n_per_class=5, n_classes=2)
        with pytest.raises(ValueError, match="train_indices must be non-empty"):
            temporal_decoding_decay(
                X,
                y,
                np.array([], dtype=int),
                [np.arange(5)],
                classifier="logistic",
                random_state=0,
            )

    def test_empty_group_raises(self):
        """
        Empty test group raises ValueError.

        Tests:
            (Test Case 1) ValueError for empty test group.
        """
        from spikelab.spikedata.decoding import temporal_decoding_decay

        X, y = _separable_dataset(n_per_class=5, n_classes=2)
        order = np.random.default_rng(0).permutation(len(X))
        X = X[order]
        y = y[order]
        with pytest.raises(ValueError, match="is empty"):
            temporal_decoding_decay(
                X,
                y,
                np.arange(8),
                [np.array([], dtype=int)],
                classifier="logistic",
                random_state=0,
            )

    def test_no_test_groups_raises(self):
        """
        Empty test_index_groups list raises ValueError.

        Tests:
            (Test Case 1) ValueError for empty list.
        """
        from spikelab.spikedata.decoding import temporal_decoding_decay

        X, y = _separable_dataset(n_per_class=5, n_classes=2)
        with pytest.raises(ValueError, match="non-empty list"):
            temporal_decoding_decay(
                X,
                y,
                np.arange(5),
                [],
                classifier="logistic",
                random_state=0,
            )


class TestNoveltyPerCycle:
    """Tests for novelty_per_cycle."""

    def _make_drifting(self, n_classes=3, samples_per_cycle=12, n_cycles=8, seed=0):
        """Build cycles 0..2 stable; cycles 3..7 drift away from training."""
        rng = np.random.default_rng(seed)
        per_cycle_per_class = samples_per_cycle // n_classes
        n_features = 6
        X = []
        y = []
        cyc = []
        for c in range(n_cycles):
            drift = max(0, c - 2) * 0.7
            for cls in range(n_classes):
                center = np.zeros(n_features)
                center[cls] = 4.0 - drift
                pts = rng.normal(center, 0.6, (per_cycle_per_class, n_features))
                X.append(pts)
                y.extend([cls] * per_cycle_per_class)
                cyc.extend([c] * per_cycle_per_class)
        return np.vstack(X), np.asarray(y), np.asarray(cyc)

    def test_novelty_increases_with_drift(self):
        """
        Cycles further from the training distribution have larger log_loss.

        Tests:
            (Test Case 1) Mean log_loss in late cycles > mean in early test cycles.
        """
        from spikelab.spikedata.decoding import novelty_per_cycle

        X, y, cyc = self._make_drifting()
        result = novelty_per_cycle(
            X,
            y,
            cyc,
            train_cycles=[0, 1, 2],
            classifier="logistic",
            classifier_kwargs={"max_iter": 200},
            random_state=0,
        )
        early = np.nanmean(result["log_losses"][result["cycles"] <= 4])
        late = np.nanmean(result["log_losses"][result["cycles"] >= 6])
        assert late > early

    def test_normalize_to_subtracts_baseline(self):
        """
        normalize_to subtracts the baseline cycle mean from all values.

        Tests:
            (Test Case 1) Cycles in normalize_to have ~0 normalized log_loss.
            (Test Case 2) raw_log_losses preserves the unnormalized values.
        """
        from spikelab.spikedata.decoding import novelty_per_cycle

        X, y, cyc = self._make_drifting()
        result = novelty_per_cycle(
            X,
            y,
            cyc,
            train_cycles=[0, 1, 2],
            normalize_to=[3, 4],
            classifier="logistic",
            classifier_kwargs={"max_iter": 200},
            random_state=0,
        )
        baseline_mask = np.isin(result["cycles"], [3, 4])
        assert abs(np.nanmean(result["log_losses"][baseline_mask])) < 1e-9
        assert not np.allclose(result["raw_log_losses"], result["log_losses"])

    def test_returns_expected_keys(self):
        """
        Result dict has all documented keys.

        Tests:
            (Test Case 1) Required keys present.
        """
        from spikelab.spikedata.decoding import novelty_per_cycle

        X, y, cyc = self._make_drifting()
        result = novelty_per_cycle(
            X,
            y,
            cyc,
            train_cycles=[0, 1, 2],
            classifier="logistic",
            classifier_kwargs={"max_iter": 200},
            random_state=0,
        )
        for k in (
            "cycles",
            "log_losses",
            "accuracies",
            "raw_log_losses",
            "raw_accuracies",
            "train_cycles",
            "normalize_to",
            "classes",
            "classifier_name",
        ):
            assert k in result

    def test_no_train_match_raises(self):
        """
        train_cycles missing from cycle_labels raises ValueError.

        Tests:
            (Test Case 1) ValueError when no samples match train_cycles.
        """
        from spikelab.spikedata.decoding import novelty_per_cycle

        X, y, cyc = self._make_drifting()
        with pytest.raises(ValueError, match="No samples match"):
            novelty_per_cycle(
                X,
                y,
                cyc,
                train_cycles=[99, 100],
                classifier="logistic",
                random_state=0,
            )


class TestDistinctnessPerCycle:
    """Tests for distinctness_per_cycle."""

    def _make_increasing_distinctness(
        self, n_classes=3, samples_per_cycle=18, n_cycles=6, seed=0
    ):
        """Cycles 0..2 noisy; cycles 3..5 well-separated."""
        rng = np.random.default_rng(seed)
        per_cycle_per_class = samples_per_cycle // n_classes
        n_features = 6
        X, y, cyc = [], [], []
        for c in range(n_cycles):
            spread = 4.0 if c < 3 else 0.5
            for cls in range(n_classes):
                center = np.zeros(n_features)
                center[cls] = 5.0
                pts = rng.normal(center, spread, (per_cycle_per_class, n_features))
                X.append(pts)
                y.extend([cls] * per_cycle_per_class)
                cyc.extend([c] * per_cycle_per_class)
        return np.vstack(X), np.asarray(y), np.asarray(cyc)

    def test_distinctness_higher_in_separable_cycles(self):
        """
        Cycles with well-separated classes have lower log_loss.

        Tests:
            (Test Case 1) Mean log_loss in late (separable) cycles < early (noisy).
        """
        from spikelab.spikedata.decoding import distinctness_per_cycle

        X, y, cyc = self._make_increasing_distinctness()
        result = distinctness_per_cycle(
            X,
            y,
            cyc,
            classifier="logistic",
            classifier_kwargs={"max_iter": 500},
            cv=3,
            random_state=0,
        )
        early = np.nanmean(result["log_losses"][result["cycles"] < 3])
        late = np.nanmean(result["log_losses"][result["cycles"] >= 3])
        assert late < early

    def test_returns_expected_keys(self):
        """
        Result dict has all documented keys.

        Tests:
            (Test Case 1) Required keys present.
        """
        from spikelab.spikedata.decoding import distinctness_per_cycle

        X, y, cyc = self._make_increasing_distinctness()
        result = distinctness_per_cycle(
            X,
            y,
            cyc,
            classifier="logistic",
            classifier_kwargs={"max_iter": 500},
            cv=3,
            random_state=0,
        )
        for k in (
            "cycles",
            "log_losses",
            "accuracies",
            "raw_log_losses",
            "raw_accuracies",
            "normalize_to",
            "classifier_name",
        ):
            assert k in result

    def test_normalize_to(self):
        """
        normalize_to subtracts baseline mean.

        Tests:
            (Test Case 1) Baseline cycles have ~0 normalized log_loss.
        """
        from spikelab.spikedata.decoding import distinctness_per_cycle

        X, y, cyc = self._make_increasing_distinctness()
        result = distinctness_per_cycle(
            X,
            y,
            cyc,
            normalize_to=[0, 1, 2],
            classifier="logistic",
            classifier_kwargs={"max_iter": 500},
            cv=3,
            random_state=0,
        )
        baseline_mask = np.isin(result["cycles"], [0, 1, 2])
        assert abs(np.nanmean(result["log_losses"][baseline_mask])) < 1e-9


class TestCountFunctions:
    """Tests for count_evoked_spikes and count_active_units."""

    def test_count_evoked_spikes_shape(self):
        """
        Returns one total per stim electrode.

        Tests:
            (Test Case 1) Output shape equals (n_stim,).
        """
        from spikelab.spikedata.decoding import count_evoked_spikes

        rng = np.random.default_rng(0)
        # (stim=4, units=10, time=100, iter=20)
        arr = rng.integers(0, 3, (4, 10, 100, 20))
        result = count_evoked_spikes(arr)
        assert result.shape == (4,)
        # Verify against direct sum
        np.testing.assert_array_equal(result, arr.sum(axis=(1, 2, 3)))

    def test_count_active_units_threshold(self):
        """
        With spikes_per_iteration=1 and 10 iterations, units must average
        at least 1 spike per iteration to count as active.

        Tests:
            (Test Case 1) Unit firing 10 spikes total over 10 iter is active.
            (Test Case 2) Unit firing 5 spikes total over 10 iter is not.
        """
        from spikelab.spikedata.decoding import count_active_units

        # (stim=2, units=3, time=50, iter=10)
        arr = np.zeros((2, 3, 50, 10), dtype=int)
        # Stim 0, unit 0: 10 total spikes (active)
        arr[0, 0, 0, :] = 1
        # Stim 0, unit 1: 5 total spikes (not active)
        arr[0, 1, 0, :5] = 1
        # Stim 0, unit 2: 20 total spikes (active)
        arr[0, 2, 0, :] = 2
        # Stim 1: nothing
        result = count_active_units(arr, spikes_per_iteration=1)
        assert result.shape == (2,)
        assert result[0] == 2  # units 0 and 2 active
        assert result[1] == 0

    def test_count_active_units_higher_threshold(self):
        """
        Higher spikes_per_iteration tightens the activity threshold.

        Tests:
            (Test Case 1) With spikes_per_iteration=2, only the 20-spike unit qualifies.
        """
        from spikelab.spikedata.decoding import count_active_units

        arr = np.zeros((1, 3, 50, 10), dtype=int)
        arr[0, 0, 0, :] = 1  # 10 total
        arr[0, 1, 0, :5] = 1  # 5 total
        arr[0, 2, 0, :] = 2  # 20 total
        result = count_active_units(arr, spikes_per_iteration=2)
        assert result[0] == 1

    def test_count_evoked_spikes_wrong_shape_raises(self):
        """
        Non-4D input raises ValueError.

        Tests:
            (Test Case 1) ValueError for 3-D input.
        """
        from spikelab.spikedata.decoding import count_evoked_spikes

        with pytest.raises(ValueError, match="4-D"):
            count_evoked_spikes(np.zeros((4, 10, 20)))
