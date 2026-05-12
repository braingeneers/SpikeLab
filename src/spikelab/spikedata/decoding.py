"""Classifier-based decoding of categorical labels from spike response features.

Provides cross-validated decoding, regularization sweeps, and latency-dependent
decoding. Designed for the stimulus-identity decoding pattern from the Maxwell
collaborator scripts (``fit_model_stim_response.py``,
``regularization_sensitivity_analysis.py``, ``model_predictions_analysis.py``):
take a per-slice response amplitude matrix ``(S, U)`` and decode the per-slice
stimulus label.

All three classifier backends — ``RidgeClassifier``, ``MLPClassifier``,
``RandomForestClassifier`` — come from scikit-learn (optional dependency). The
module raises a clear ``ImportError`` at first use when sklearn is missing.
"""

import importlib

import numpy as np

__all__ = [
    "cross_validated_decode",
    "regularization_sweep",
    "latency_dependent_decoding",
    "train_test_decoding",
    "temporal_decoding_decay",
    "novelty_per_cycle",
    "distinctness_per_cycle",
    "count_evoked_spikes",
    "count_active_units",
]


_CLASSIFIER_REGISTRY = {
    "ridge": ("sklearn.linear_model", "RidgeClassifier"),
    "logistic": ("sklearn.linear_model", "LogisticRegression"),
    "mlp": ("sklearn.neural_network", "MLPClassifier"),
    "random_forest": ("sklearn.ensemble", "RandomForestClassifier"),
}


def _compute_predicted_proba(clf, X, classes):
    """Probabilistic predictions for a fitted classifier.

    Tries ``predict_proba`` first; falls back to softmax over
    ``decision_function`` for classifiers that don't expose probabilities
    (e.g. ``RidgeClassifier``). Returns ``None`` if neither is available.

    The output is always aligned to ``classes`` (sorted unique labels)
    so that ``proba[:, i]`` corresponds to ``classes[i]``.
    """
    if hasattr(clf, "predict_proba"):
        proba = np.asarray(clf.predict_proba(X), dtype=float)
        # Align columns to `classes` ordering.
        clf_classes = list(getattr(clf, "classes_", classes))
        if list(clf_classes) == list(classes):
            return proba
        col_index = [clf_classes.index(c) for c in classes]
        return proba[:, col_index]

    if hasattr(clf, "decision_function"):
        scores = np.asarray(clf.decision_function(X), dtype=float)
        if scores.ndim == 1:
            # Binary case: reshape to (n_samples, 2)
            scores = np.column_stack([-scores, scores])
        # Softmax row-wise
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
        clf_classes = list(getattr(clf, "classes_", classes))
        if list(clf_classes) == list(classes):
            return proba
        col_index = [clf_classes.index(c) for c in classes]
        return proba[:, col_index]

    return None


def _safe_log_loss(y_true, proba, classes, eps=1e-12):
    """Cross-entropy / log-loss for multiclass labels.

    Returns NaN if ``proba`` is None.
    """
    if proba is None:
        return np.nan
    proba = np.clip(proba, eps, 1.0 - eps)
    # Build (n, K) one-hot of y_true
    classes = list(classes)
    idx = np.array([classes.index(y) for y in y_true])
    correct_proba = proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(correct_proba)))


def _import_sklearn():
    try:
        from sklearn import metrics, model_selection  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Classifier decoding requires 'scikit-learn'. "
            "Install with: pip install scikit-learn"
        ) from e
    return metrics, model_selection


def _build_classifier(name, classifier_kwargs, random_state):
    if name not in _CLASSIFIER_REGISTRY:
        raise ValueError(
            f"classifier must be one of {sorted(_CLASSIFIER_REGISTRY)}, got {name!r}"
        )
    module_path, class_name = _CLASSIFIER_REGISTRY[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            "Classifier decoding requires 'scikit-learn'. "
            "Install with: pip install scikit-learn"
        ) from e
    cls = getattr(module, class_name)
    kwargs = dict(classifier_kwargs or {})
    if random_state is not None and "random_state" not in kwargs:
        # RidgeClassifier accepts random_state when solver='sag'/'saga' only,
        # but always accepts the kwarg (ignored otherwise). MLP / RF use it.
        kwargs["random_state"] = random_state
    return cls(**kwargs)


def _build_cv_splitter(cv, y, random_state):
    _, model_selection = _import_sklearn()
    if isinstance(cv, str):
        if cv == "loo":
            return model_selection.LeaveOneOut()
        raise ValueError(f"cv string must be 'loo'; got {cv!r}")
    if isinstance(cv, int):
        if cv < 2:
            raise ValueError(f"cv (k-fold) must be >= 2; got {cv}.")
        return model_selection.StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
    raise ValueError(f"cv must be 'loo' or an int >= 2; got {cv!r}")


def cross_validated_decode(
    X,
    y,
    *,
    classifier="ridge",
    cv="loo",
    classifier_kwargs=None,
    random_state=None,
):
    """Train a classifier via cross-validation and report accuracy + predictions.

    Parameters:
        X (np.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
            For per-slice decoding, ``n_samples = S`` (one per slice) and the
            features are e.g. per-unit response amplitudes or flattened
            ``(U, T)`` rasters.
        y (array-like): Labels per sample, shape ``(n_samples,)``. Categorical.
        classifier (str): ``"ridge"`` (default), ``"mlp"``, or
            ``"random_forest"``.
        cv (str or int): ``"loo"`` (default) for Leave-One-Out CV, or an int
            ``>= 2`` for stratified k-fold.
        classifier_kwargs (dict or None): Forwarded to the underlying sklearn
            classifier constructor. Use e.g. ``{"alpha": 1.0}`` for ridge or
            ``{"hidden_layer_sizes": (50,), "max_iter": 200}`` for MLP.
        random_state (int or None): For reproducibility (k-fold shuffling +
            classifier).

    Returns:
        result (dict):
            - ``accuracy`` (float): Overall out-of-fold accuracy.
            - ``log_loss`` (float or NaN): Mean out-of-fold cross-entropy.
              NaN when the classifier exposes neither ``predict_proba`` nor
              ``decision_function`` (e.g. some custom estimators).
            - ``predicted_probabilities`` (np.ndarray or None): Out-of-fold
              probabilities, shape ``(n_samples, K)``, columns aligned to
              ``classes``. None when probabilities are unavailable.
            - ``predictions`` (np.ndarray): Out-of-fold predicted labels,
              shape ``(n_samples,)``.
            - ``true_labels`` (np.ndarray): Copy of ``y``.
            - ``confusion_matrix`` (np.ndarray): ``(K, K)`` confusion matrix
              with rows = true, cols = predicted.
            - ``per_fold_accuracy`` (np.ndarray): Per-fold accuracy.
            - ``per_fold_log_loss`` (np.ndarray): Per-fold log-loss; NaN
              entries where probabilities were unavailable.
            - ``classes`` (np.ndarray): Unique label values in sorted order.
            - ``classifier_name`` (str): Resolved classifier name.

    Notes:
        - Requires ``scikit-learn`` (optional dependency).
        - LOO yields fold size 1, so each fold's accuracy is 0 or 1; the
          overall ``accuracy`` is the mean across all single-sample folds.
        - ``log_loss`` requires probabilistic predictions. ``logistic``,
          ``mlp``, ``random_forest`` provide them natively;
          ``ridge`` falls back to a softmax over ``decision_function``
          (useful as a relative score, not as a calibrated probability).
    """
    metrics, _ = _import_sklearn()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples; got {len(X)} and {len(y)}."
        )
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 distinct classes to train a classifier.")

    splitter = _build_cv_splitter(cv, y, random_state)
    predictions = np.empty_like(y)
    classes = np.array(sorted(np.unique(y)))
    proba_out = np.full((len(y), len(classes)), np.nan, dtype=float)
    per_fold_acc = []
    per_fold_ll = []
    any_proba = False

    for train_idx, test_idx in splitter.split(X, y):
        clf = _build_classifier(classifier, classifier_kwargs, random_state)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        predictions[test_idx] = y_pred
        per_fold_acc.append(float(np.mean(y_pred == y[test_idx])))

        proba = _compute_predicted_proba(clf, X[test_idx], classes)
        if proba is not None:
            any_proba = True
            proba_out[test_idx] = proba
            per_fold_ll.append(_safe_log_loss(y[test_idx], proba, classes))
        else:
            per_fold_ll.append(np.nan)

    accuracy = float(np.mean(predictions == y))
    cm = metrics.confusion_matrix(y, predictions, labels=classes)

    if any_proba:
        log_loss = _safe_log_loss(y, proba_out, classes)
        predicted_probabilities = proba_out
    else:
        log_loss = float("nan")
        predicted_probabilities = None

    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "predicted_probabilities": predicted_probabilities,
        "predictions": predictions,
        "true_labels": y.copy(),
        "confusion_matrix": cm,
        "per_fold_accuracy": np.asarray(per_fold_acc, dtype=float),
        "per_fold_log_loss": np.asarray(per_fold_ll, dtype=float),
        "classes": classes,
        "classifier_name": classifier,
    }


def regularization_sweep(
    X,
    y,
    alphas,
    *,
    classifier="ridge",
    cv="loo",
    classifier_kwargs=None,
    random_state=None,
):
    """Sweep classifier regularization strength and report per-alpha CV accuracy.

    For ``ridge``, ``alpha`` is the L2 penalty (``alpha`` kwarg). For ``mlp``,
    ``alpha`` is also the L2 penalty (``alpha`` kwarg). For ``random_forest``,
    ``alpha`` is interpreted as ``ccp_alpha`` (minimal cost-complexity
    pruning); pass an explicit ``classifier_kwargs`` if you want different
    semantics.

    Parameters:
        X (np.ndarray): Feature matrix ``(n_samples, n_features)``.
        y (array-like): Labels ``(n_samples,)``.
        alphas (array-like): 1-D sequence of regularization strengths.
        classifier (str): ``"ridge"`` (default), ``"mlp"``, or
            ``"random_forest"``.
        cv (str or int): ``"loo"`` (default) or an int ``>= 2``.
        classifier_kwargs (dict or None): Base classifier kwargs; ``alpha`` /
            ``ccp_alpha`` is overridden per iteration.
        random_state (int or None): Reproducibility seed.

    Returns:
        result (dict):
            - ``alphas`` (np.ndarray): Input alphas.
            - ``mean_accuracy`` (np.ndarray): Per-alpha CV accuracy,
              shape ``(len(alphas),)``.
            - ``per_alpha_predictions`` (np.ndarray): Per-alpha out-of-fold
              predictions, shape ``(len(alphas), n_samples)``.
            - ``best_alpha`` (float): Alpha with highest accuracy.
            - ``best_accuracy`` (float): Accuracy at ``best_alpha``.

    Notes:
        - Requires ``scikit-learn``.
    """
    alphas = np.asarray(alphas, dtype=float).ravel()
    if alphas.size == 0:
        raise ValueError("alphas must be non-empty.")

    base_kwargs = dict(classifier_kwargs or {})
    alpha_kw = "ccp_alpha" if classifier == "random_forest" else "alpha"

    mean_acc = np.empty(alphas.size, dtype=float)
    preds = np.empty((alphas.size, len(y)), dtype=np.asarray(y).dtype)

    for i, a in enumerate(alphas):
        kw = dict(base_kwargs)
        kw[alpha_kw] = float(a)
        result = cross_validated_decode(
            X,
            y,
            classifier=classifier,
            cv=cv,
            classifier_kwargs=kw,
            random_state=random_state,
        )
        mean_acc[i] = result["accuracy"]
        preds[i] = result["predictions"]

    best_idx = int(np.argmax(mean_acc))
    return {
        "alphas": alphas,
        "mean_accuracy": mean_acc,
        "per_alpha_predictions": preds,
        "best_alpha": float(alphas[best_idx]),
        "best_accuracy": float(mean_acc[best_idx]),
    }


def latency_dependent_decoding(
    response_stack,
    y,
    latency_bins_ms,
    *,
    bin_size,
    slice_start_time_ms=0.0,
    classifier="ridge",
    cv="loo",
    classifier_kwargs=None,
    random_state=None,
):
    """Decode per-slice labels using progressively wider latency windows.

    For each latency window ``(start_ms, end_ms)``, builds a feature matrix
    from ``response_stack[:, bins_in_window, :].sum(axis=1).T`` (shape
    ``(S, U)``) and runs cross-validated decoding. Useful for asking
    "from when does the population encode stimulus identity?".

    Parameters:
        response_stack (np.ndarray): Per-slice raster of shape ``(U, T, S)``.
            Typically produced by ``SpikeSliceStack.to_raster_array(bin_size)``
            or ``baseline_normalized_raster`` (subtract mode).
        y (array-like): Per-slice labels of length ``S``.
        latency_bins_ms (list[tuple[float, float]]): Sequence of latency
            windows ``(start_ms, end_ms)`` relative to slice origin.
        bin_size (float): Bin size of ``response_stack`` in ms.
        slice_start_time_ms (float): Time-axis offset of bin 0 in ms (slice
            ``start_time``). 0.0 for 0-based slices; negative ``pre_ms`` for
            event-centered slices.
        classifier (str): ``"ridge"`` (default), ``"mlp"``, or
            ``"random_forest"``.
        cv (str or int): ``"loo"`` (default) or int ``>= 2``.
        classifier_kwargs (dict or None): Forwarded.
        random_state (int or None): Reproducibility seed.

    Returns:
        result (dict):
            - ``windows`` (list[tuple[float, float]]): Input windows.
            - ``accuracies`` (np.ndarray): Per-window CV accuracy.
            - ``per_window_predictions`` (np.ndarray): Per-window
              out-of-fold predictions, shape ``(len(windows), S)``.
            - ``classifier_name`` (str).

    Notes:
        - Requires ``scikit-learn``.
        - Windows that map to an empty bin range raise ``ValueError``.
    """
    response_stack = np.asarray(response_stack, dtype=float)
    if response_stack.ndim != 3:
        raise ValueError(
            f"response_stack must be 3-D (U, T, S); got shape {response_stack.shape}."
        )
    U, T, S = response_stack.shape
    y = np.asarray(y).ravel()
    if len(y) != S:
        raise ValueError(f"y must have length S={S}; got {len(y)}.")

    accuracies = np.empty(len(latency_bins_ms), dtype=float)
    preds = np.empty((len(latency_bins_ms), S), dtype=y.dtype)

    for i, win in enumerate(latency_bins_ms):
        if not isinstance(win, (tuple, list)) or len(win) != 2:
            raise ValueError(
                f"Each latency window must be a (start_ms, end_ms) tuple; "
                f"got {win!r}."
            )
        r_start, r_end = float(win[0]), float(win[1])
        if r_end <= r_start:
            raise ValueError(
                f"Latency window end must be greater than start; got {win!r}."
            )
        bin_start = int(np.floor((r_start - slice_start_time_ms) / bin_size))
        bin_end = int(np.ceil((r_end - slice_start_time_ms) / bin_size))
        bin_start = max(0, bin_start)
        bin_end = min(T, bin_end)
        if bin_end <= bin_start:
            raise ValueError(
                f"Latency window {win!r} maps to an empty bin range "
                f"given bin_size={bin_size} and stack T={T}."
            )

        # Feature matrix: per-slice sum of counts over the window → (U, S) → (S, U)
        X = response_stack[:, bin_start:bin_end, :].sum(axis=1).T  # (S, U)
        out = cross_validated_decode(
            X,
            y,
            classifier=classifier,
            cv=cv,
            classifier_kwargs=classifier_kwargs,
            random_state=random_state,
        )
        accuracies[i] = out["accuracy"]
        preds[i] = out["predictions"]

    return {
        "windows": list(latency_bins_ms),
        "accuracies": accuracies,
        "per_window_predictions": preds,
        "classifier_name": classifier,
    }


def train_test_decoding(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    classifier="logistic",
    classifier_kwargs=None,
    random_state=None,
):
    """Fit a classifier once on training data and evaluate on a held-out test set.

    Use this when you want to ask "does a model trained on data X transfer
    to data Y?" — for example, training on early stimulus cycles and asking
    whether evoked responses in later cycles still match.

    Parameters:
        X_train (np.ndarray): Training feature matrix ``(n_train, n_features)``.
        y_train (array-like): Training labels ``(n_train,)``.
        X_test (np.ndarray): Test feature matrix ``(n_test, n_features)``.
        y_test (array-like): Test labels ``(n_test,)``.
        classifier (str): ``"logistic"`` (default; supports probabilities),
            ``"ridge"``, ``"mlp"``, or ``"random_forest"``.
        classifier_kwargs (dict or None): Forwarded to the sklearn class.
        random_state (int or None): Reproducibility seed.

    Returns:
        result (dict):
            - ``accuracy`` (float): Test-set accuracy.
            - ``log_loss`` (float or NaN): Cross-entropy on the test set.
            - ``predictions`` (np.ndarray): Test predictions ``(n_test,)``.
            - ``predicted_probabilities`` (np.ndarray or None): Test
              probabilities ``(n_test, K)`` aligned to ``classes``.
            - ``true_labels`` (np.ndarray): Copy of ``y_test``.
            - ``confusion_matrix`` (np.ndarray): ``(K, K)``.
            - ``classes`` (np.ndarray): Unique labels (union of train+test).
            - ``classifier_name`` (str).

    Notes:
        - Requires ``scikit-learn``.
        - Test labels not seen during training are scored against zero
          probability (yielding large log-loss) — this is the desired
          behavior when asking whether the trained model can generalize.
    """
    metrics, _ = _import_sklearn()

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"X_train and X_test must be 2-D; got shapes {X_train.shape} and {X_test.shape}."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train and X_test must have the same number of features; "
            f"got {X_train.shape[1]} and {X_test.shape[1]}."
        )
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have the same number of samples; "
            f"got {len(X_train)} and {len(y_train)}."
        )
    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test must have the same number of samples; "
            f"got {len(X_test)} and {len(y_test)}."
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError("Need at least 2 distinct training classes.")

    classes = np.array(sorted(set(np.unique(y_train)) | set(np.unique(y_test))))
    clf = _build_classifier(classifier, classifier_kwargs, random_state)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = float(np.mean(predictions == y_test))

    proba = _compute_predicted_proba(clf, X_test, classes)
    log_loss = _safe_log_loss(y_test, proba, classes)
    cm = metrics.confusion_matrix(y_test, predictions, labels=classes)

    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "predictions": predictions,
        "predicted_probabilities": proba,
        "true_labels": y_test.copy(),
        "confusion_matrix": cm,
        "classes": classes,
        "classifier_name": classifier,
    }


def temporal_decoding_decay(
    X,
    y,
    train_indices,
    test_index_groups,
    *,
    classifier="logistic",
    classifier_kwargs=None,
    random_state=None,
):
    """Train once on a subset of samples and evaluate on later sample groups.

    Designed for the "is the evoked response drifting?" workflow: train a
    classifier on early stimulus cycles, then watch how its accuracy
    decreases (and cross-entropy increases) on later cycle groups.

    Parameters:
        X (np.ndarray): Feature matrix ``(n_samples, n_features)``.
        y (array-like): Labels ``(n_samples,)``.
        train_indices (array-like): Sample indices used for training.
        test_index_groups (list[array-like]): A list of arrays of sample
            indices, one per test group. Each group is evaluated
            independently against the single trained model.
        classifier (str): ``"logistic"`` (default; supports probabilities),
            ``"ridge"``, ``"mlp"``, or ``"random_forest"``.
        classifier_kwargs (dict or None): Forwarded to the sklearn class.
        random_state (int or None): Reproducibility seed.

    Returns:
        result (dict):
            - ``train_indices`` (np.ndarray): Echoed.
            - ``test_index_groups`` (list[np.ndarray]): Echoed.
            - ``accuracies`` (np.ndarray): Per-group accuracy,
              shape ``(n_groups,)``.
            - ``log_losses`` (np.ndarray): Per-group cross-entropy.
            - ``per_group_predictions`` (list[np.ndarray]): Per-group
              predictions.
            - ``per_group_probabilities`` (list[np.ndarray] or list[None]):
              Per-group predicted probabilities (None when not supported).
            - ``classes`` (np.ndarray): Unique labels.
            - ``classifier_name`` (str).

    Notes:
        - Requires ``scikit-learn``.
        - All overlap is the caller's responsibility: a sample present in
          both ``train_indices`` and a test group will be evaluated as
          training data, leaking into the score.
    """
    metrics, _ = _import_sklearn()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples; got {len(X)} and {len(y)}."
        )

    train_indices = np.asarray(train_indices, dtype=int).ravel()
    if train_indices.size == 0:
        raise ValueError("train_indices must be non-empty.")
    if not isinstance(test_index_groups, (list, tuple)) or len(test_index_groups) == 0:
        raise ValueError("test_index_groups must be a non-empty list of index arrays.")

    classes = np.array(sorted(np.unique(y)))
    clf = _build_classifier(classifier, classifier_kwargs, random_state)
    clf.fit(X[train_indices], y[train_indices])

    accuracies = np.empty(len(test_index_groups), dtype=float)
    log_losses = np.empty(len(test_index_groups), dtype=float)
    per_group_preds = []
    per_group_probas = []

    for i, group in enumerate(test_index_groups):
        group = np.asarray(group, dtype=int).ravel()
        if group.size == 0:
            raise ValueError(f"test_index_groups[{i}] is empty.")
        X_test = X[group]
        y_test = y[group]
        preds = clf.predict(X_test)
        accuracies[i] = float(np.mean(preds == y_test))
        proba = _compute_predicted_proba(clf, X_test, classes)
        log_losses[i] = _safe_log_loss(y_test, proba, classes)
        per_group_preds.append(preds)
        per_group_probas.append(proba)

    return {
        "train_indices": train_indices,
        "test_index_groups": [np.asarray(g, dtype=int) for g in test_index_groups],
        "accuracies": accuracies,
        "log_losses": log_losses,
        "per_group_predictions": per_group_preds,
        "per_group_probabilities": per_group_probas,
        "classes": classes,
        "classifier_name": classifier,
    }


def novelty_per_cycle(
    X,
    y,
    cycle_labels,
    train_cycles,
    *,
    test_cycles=None,
    classifier="logistic",
    classifier_kwargs=None,
    normalize_to=None,
    random_state=None,
):
    """Per-cycle cross-entropy of a model trained on baseline cycles.

    Implements the "novelty" measure from the Maxwell project
    (``fit_linear_reg_stim_resp.py``): train a single classifier on all
    samples whose ``cycle_labels`` falls in ``train_cycles``, then evaluate
    cross-entropy and accuracy on each ``test_cycles`` value separately.
    A rising log-loss across later cycles indicates that the evoked
    response patterns are drifting away from the trained baseline (i.e.
    becoming "novel").

    Parameters:
        X (np.ndarray): Feature matrix ``(n_samples, n_features)``.
        y (array-like): Labels ``(n_samples,)``.
        cycle_labels (array-like): Cycle index per sample ``(n_samples,)``.
        train_cycles (array-like): Cycle indices used for training.
        test_cycles (array-like or None): Cycles to evaluate. When None
            (default), uses every cycle that is not in ``train_cycles``.
        classifier (str): ``"logistic"`` (default; supports probabilities)
            or any other registered classifier name.
        classifier_kwargs (dict or None): Forwarded to the sklearn class.
        normalize_to (array-like or None): Optional reference cycles whose
            mean log-loss / accuracy is subtracted from every per-cycle
            value. None (default) returns raw log-loss / accuracy.
        random_state (int or None): Reproducibility seed.

    Returns:
        result (dict):
            - ``cycles`` (np.ndarray): Cycle indices evaluated, shape
              ``(n_test_cycles,)``.
            - ``log_losses`` (np.ndarray): Per-cycle cross-entropy
              ("novelty"). NaN where the cycle has no samples.
            - ``accuracies`` (np.ndarray): Per-cycle accuracy.
            - ``raw_log_losses`` (np.ndarray): Always-unnormalized
              log-losses (== ``log_losses`` when ``normalize_to`` is None).
            - ``raw_accuracies`` (np.ndarray): Same for accuracy.
            - ``train_cycles`` (np.ndarray): Echoed.
            - ``normalize_to`` (np.ndarray or None): Echoed.
            - ``classes`` (np.ndarray): Unique training labels.
            - ``classifier_name`` (str).

    Notes:
        - Requires ``scikit-learn``.
        - Cycles in ``test_cycles`` that overlap ``train_cycles`` are
          evaluated as in-distribution data — the caller is responsible
          for any train/test separation desired.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    cycle_labels = np.asarray(cycle_labels).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}.")
    if not (len(X) == len(y) == len(cycle_labels)):
        raise ValueError(
            f"X, y, and cycle_labels must all have the same length; got "
            f"{len(X)}, {len(y)}, {len(cycle_labels)}."
        )

    train_cycles = np.asarray(train_cycles).ravel()
    train_mask = np.isin(cycle_labels, train_cycles)
    if not train_mask.any():
        raise ValueError(
            "No samples match train_cycles; check that cycle_labels and "
            "train_cycles use the same indexing."
        )

    if test_cycles is None:
        all_cycles = np.unique(cycle_labels)
        test_cycles = np.array([c for c in all_cycles if c not in train_cycles])
    else:
        test_cycles = np.asarray(test_cycles).ravel()

    classes = np.array(sorted(np.unique(y[train_mask])))
    if len(classes) < 2:
        raise ValueError("Need at least 2 distinct classes in the training cycles.")

    clf = _build_classifier(classifier, classifier_kwargs, random_state)
    clf.fit(X[train_mask], y[train_mask])

    log_losses = np.full(len(test_cycles), np.nan)
    accuracies = np.full(len(test_cycles), np.nan)

    for i, c in enumerate(test_cycles):
        mask = cycle_labels == c
        if not mask.any():
            continue
        preds = clf.predict(X[mask])
        accuracies[i] = float(np.mean(preds == y[mask]))
        proba = _compute_predicted_proba(clf, X[mask], classes)
        log_losses[i] = _safe_log_loss(y[mask], proba, classes)

    raw_log_losses = log_losses.copy()
    raw_accuracies = accuracies.copy()

    if normalize_to is not None:
        normalize_to = np.asarray(normalize_to).ravel()
        baseline_mask = np.isin(test_cycles, normalize_to)
        if not baseline_mask.any():
            raise ValueError(
                "normalize_to does not overlap test_cycles; cannot compute "
                "baseline-normalized values."
            )
        baseline_ll = float(np.nanmean(raw_log_losses[baseline_mask]))
        baseline_acc = float(np.nanmean(raw_accuracies[baseline_mask]))
        log_losses = raw_log_losses - baseline_ll
        accuracies = raw_accuracies - baseline_acc

    return {
        "cycles": test_cycles,
        "log_losses": log_losses,
        "accuracies": accuracies,
        "raw_log_losses": raw_log_losses,
        "raw_accuracies": raw_accuracies,
        "train_cycles": train_cycles,
        "normalize_to": normalize_to,
        "classes": classes,
        "classifier_name": classifier,
    }


def distinctness_per_cycle(
    X,
    y,
    cycle_labels,
    *,
    cycles=None,
    classifier="logistic",
    classifier_kwargs=None,
    cv="loo",
    normalize_to=None,
    random_state=None,
):
    """Per-cycle within-cycle cross-validation cross-entropy ("distinctness").

    Implements the "distinctness" measure from the Maxwell project: for
    each cycle, train and cross-validate a classifier using only that
    cycle's samples. Lower log-loss / higher accuracy means the evoked
    response patterns within that cycle are more separable across stim
    classes (i.e. more "distinct").

    Parameters:
        X (np.ndarray): Feature matrix ``(n_samples, n_features)``.
        y (array-like): Labels ``(n_samples,)``.
        cycle_labels (array-like): Cycle index per sample ``(n_samples,)``.
        cycles (array-like or None): Cycles to evaluate. When None
            (default), uses every unique cycle in ``cycle_labels``.
        classifier (str): ``"logistic"`` (default) or any registered name.
        classifier_kwargs (dict or None): Forwarded to the sklearn class.
        cv (str or int): ``"loo"`` (default) or stratified k-fold int.
        normalize_to (array-like or None): Optional reference cycles whose
            mean log-loss / accuracy is subtracted from every per-cycle
            value.
        random_state (int or None): Reproducibility seed.

    Returns:
        result (dict):
            - ``cycles`` (np.ndarray): Evaluated cycle indices.
            - ``log_losses`` (np.ndarray): Per-cycle within-cycle CV
              cross-entropy ("distinctness"). NaN where insufficient data.
            - ``accuracies`` (np.ndarray): Per-cycle CV accuracy.
            - ``raw_log_losses`` / ``raw_accuracies`` (np.ndarray):
              Always-unnormalized variants.
            - ``normalize_to`` (np.ndarray or None): Echoed.
            - ``classifier_name`` (str).

    Notes:
        - Requires ``scikit-learn``.
        - Cycles where the within-cycle data has only one class, or fewer
          samples than required by ``cv``, return NaN.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    cycle_labels = np.asarray(cycle_labels).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}.")
    if not (len(X) == len(y) == len(cycle_labels)):
        raise ValueError("X, y, and cycle_labels must all have the same length.")

    if cycles is None:
        cycles = np.unique(cycle_labels)
    else:
        cycles = np.asarray(cycles).ravel()

    log_losses = np.full(len(cycles), np.nan)
    accuracies = np.full(len(cycles), np.nan)

    for i, c in enumerate(cycles):
        mask = cycle_labels == c
        if not mask.any():
            continue
        Xc = X[mask]
        yc = y[mask]
        if len(np.unique(yc)) < 2:
            continue
        if cv == "loo" and len(yc) < 2:
            continue
        if isinstance(cv, int) and any(np.bincount(yc.astype(int)) < cv):
            continue
        try:
            res = cross_validated_decode(
                Xc,
                yc,
                classifier=classifier,
                cv=cv,
                classifier_kwargs=classifier_kwargs,
                random_state=random_state,
            )
        except ValueError:
            continue
        log_losses[i] = res["log_loss"]
        accuracies[i] = res["accuracy"]

    raw_log_losses = log_losses.copy()
    raw_accuracies = accuracies.copy()

    if normalize_to is not None:
        normalize_to = np.asarray(normalize_to).ravel()
        baseline_mask = np.isin(cycles, normalize_to)
        if not baseline_mask.any():
            raise ValueError(
                "normalize_to does not overlap cycles; cannot compute "
                "baseline-normalized values."
            )
        baseline_ll = float(np.nanmean(raw_log_losses[baseline_mask]))
        baseline_acc = float(np.nanmean(raw_accuracies[baseline_mask]))
        log_losses = raw_log_losses - baseline_ll
        accuracies = raw_accuracies - baseline_acc

    return {
        "cycles": cycles,
        "log_losses": log_losses,
        "accuracies": accuracies,
        "raw_log_losses": raw_log_losses,
        "raw_accuracies": raw_accuracies,
        "normalize_to": normalize_to,
        "classifier_name": classifier,
    }


def count_evoked_spikes(response_stack, *, axis_units=1, axis_time=2, axis_iter=3):
    """Total spikes per stim electrode from a 4-D detection array.

    Sums spike counts over the response-units, response-time and stim-
    iteration axes to give a per-stim spike total. Mirrors the
    ``count_resp_per_cyc`` helper in ``fit_linear_reg_stim_resp.py``.

    Parameters:
        response_stack (np.ndarray): 4-D array of detections with shape
            ``(stim, units, time, iterations)`` (the default axis layout
            in the Maxwell pipeline). Use the ``axis_*`` kwargs to remap
            if your array has a different ordering.
        axis_units (int): Axis index for response units. Default 1.
        axis_time (int): Axis index for response time. Default 2.
        axis_iter (int): Axis index for stim iterations. Default 3.

    Returns:
        spikes (np.ndarray): 1-D array of total evoked spikes per
            stimulus electrode, shape ``(n_stim,)``.

    Notes:
        - For a 3-D ``(units, time, iter)`` per-stim stack, pass the
          stack indexed by stim and call this helper with the appropriate
          axes.
    """
    arr = np.asarray(response_stack)
    if arr.ndim != 4:
        raise ValueError(
            f"response_stack must be 4-D (stim, units, time, iter); got shape {arr.shape}."
        )
    return arr.sum(axis=(axis_units, axis_time, axis_iter))


def count_active_units(
    response_stack,
    *,
    spikes_per_iteration=1,
    axis_stim=0,
    axis_units=1,
    axis_time=2,
    axis_iter=3,
):
    """Number of "active" units per stim electrode in a 4-D detection array.

    Mirrors the activity threshold from ``fit_linear_reg_stim_resp.py``: a
    unit is active if its total evoked spike count divided by
    ``spikes_per_iteration`` is at least the number of stim iterations,
    i.e. the unit averages at least ``spikes_per_iteration`` spikes per
    stim. With the default ``spikes_per_iteration=1`` this is equivalent
    to "fired at least once per stim on average".

    Parameters:
        response_stack (np.ndarray): 4-D detection array, default layout
            ``(stim, units, time, iterations)``.
        spikes_per_iteration (float): Activity threshold (default 1.0).
            A unit must average at least this many spikes per stim
            iteration to count as active.
        axis_stim (int): Axis index for stim electrodes. Default 0.
        axis_units (int): Axis index for response units. Default 1.
        axis_time (int): Axis index for response time. Default 2.
        axis_iter (int): Axis index for stim iterations. Default 3.

    Returns:
        active (np.ndarray): 1-D integer array of shape ``(n_stim,)``,
            count of active response units per stim electrode.
    """
    arr = np.asarray(response_stack)
    if arr.ndim != 4:
        raise ValueError(
            f"response_stack must be 4-D (stim, units, time, iter); got shape {arr.shape}."
        )
    n_iter = arr.shape[axis_iter]
    # Sum over time and iter for each (stim, unit).
    total_spikes = arr.sum(axis=(axis_time, axis_iter))
    # Bring stim axis to front
    if axis_stim != 0:
        total_spikes = np.moveaxis(total_spikes, axis_stim, 0)
    threshold = float(spikes_per_iteration) * n_iter
    return (total_spikes >= threshold).sum(axis=1)
