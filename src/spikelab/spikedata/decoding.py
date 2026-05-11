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
]


_CLASSIFIER_REGISTRY = {
    "ridge": ("sklearn.linear_model", "RidgeClassifier"),
    "mlp": ("sklearn.neural_network", "MLPClassifier"),
    "random_forest": ("sklearn.ensemble", "RandomForestClassifier"),
}


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
            - ``predictions`` (np.ndarray): Out-of-fold predicted labels,
              shape ``(n_samples,)``.
            - ``true_labels`` (np.ndarray): Copy of ``y``.
            - ``confusion_matrix`` (np.ndarray): ``(K, K)`` confusion matrix
              with rows = true, cols = predicted.
            - ``per_fold_accuracy`` (np.ndarray): Per-fold accuracy.
            - ``classes`` (np.ndarray): Unique label values in sorted order.
            - ``classifier_name`` (str): Resolved classifier name.

    Notes:
        - Requires ``scikit-learn`` (optional dependency).
        - LOO yields fold size 1, so each fold's accuracy is 0 or 1; the
          overall ``accuracy`` is the mean across all single-sample folds.
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
    per_fold = []

    for train_idx, test_idx in splitter.split(X, y):
        clf = _build_classifier(classifier, classifier_kwargs, random_state)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        predictions[test_idx] = y_pred
        per_fold.append(float(np.mean(y_pred == y[test_idx])))

    accuracy = float(np.mean(predictions == y))
    classes = np.array(sorted(np.unique(y)))
    cm = metrics.confusion_matrix(y, predictions, labels=classes)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "true_labels": y.copy(),
        "confusion_matrix": cm,
        "per_fold_accuracy": np.asarray(per_fold, dtype=float),
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
