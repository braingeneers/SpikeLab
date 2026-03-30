"""Statistical utilities for SpikeLab.

Provides reusable statistical functions (regression, confidence intervals,
pairwise group comparisons) that can be used independently of plotting.
"""

import numpy as np


def linear_regression(x, y, ci_level=0.95):
    """Compute ordinary least-squares linear regression with optional confidence interval.

    Parameters:
        x (np.ndarray): 1-D array of predictor values.
        y (np.ndarray): 1-D array of response values (same length as *x*).
        ci_level (float): Confidence level for the interval (default 0.95).

    Returns:
        result (dict): Dictionary with keys:
            - ``slope`` (float): Fitted slope.
            - ``intercept`` (float): Fitted intercept.
            - ``r_squared`` (float): Coefficient of determination.
            - ``x_fit`` (np.ndarray): Sorted x values for plotting the fit line.
            - ``y_fit`` (np.ndarray): Predicted y values along *x_fit*.
            - ``ci_lower`` (np.ndarray): Lower confidence bound along *x_fit*.
            - ``ci_upper`` (np.ndarray): Upper confidence bound along *x_fit*.

    Notes:
        - Uses pure numpy (no scipy/sklearn dependency).
        - NaN pairs are dropped automatically.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Drop NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 non-NaN data points for regression.")

    # OLS via normal equations
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        raise ValueError("All x values are identical; regression is undefined.")
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # Predictions and R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Confidence interval (t-distribution approximation via normal for large n)
    # For small n we use a simple approximation; scipy is not required.
    se = np.sqrt(ss_res / (n - 2)) if n > 2 else 0.0
    # Approximate t critical value using normal quantile (good for n > 10,
    # conservative for smaller n)
    alpha = 1.0 - ci_level
    # Rational approximation of the normal quantile (Abramowitz & Stegun 26.2.23)
    p = 1.0 - alpha / 2.0
    t_val = _approx_normal_quantile(p)

    x_fit = np.sort(x)
    y_fit = slope * x_fit + intercept
    se_fit = se * np.sqrt(1.0 / n + (x_fit - x_mean) ** 2 / ss_xx)
    ci_lower = y_fit - t_val * se_fit
    ci_upper = y_fit + t_val * se_fit

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _approx_normal_quantile(p):
    """Approximate the standard normal quantile for *p* in (0.5, 1).

    Uses the rational approximation from Abramowitz & Stegun (26.2.23).
    Accurate to ~4.5e-4 for typical confidence levels.
    """
    if p <= 0.5:
        raise ValueError("p must be > 0.5")
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)


def pairwise_tests(
    groups,
    test="welch_t",
    correction="bonferroni",
    alpha=0.05,
    labels=None,
):
    """Run pairwise statistical tests across groups with multiple-comparison correction.

    Parameters:
        groups (dict[str, np.ndarray] or list[np.ndarray]): Per-group data
            arrays. Dict keys are used as labels; for list input supply
            ``labels`` separately.
        test (str): Statistical test to use. ``"welch_t"`` (default) for
            Welch's unequal-variance t-test, ``"student_t"`` for Student's
            equal-variance t-test, ``"mann_whitney"`` for the Mann-Whitney U
            test. All require ``scipy``.
        correction (str or None): Multiple-comparison correction.
            ``"bonferroni"`` (default) or ``None`` for uncorrected p-values.
        alpha (float): Significance threshold applied after correction.
            Default 0.05.
        labels (list[str] or None): Group labels. Required for list input;
            ignored for dict input (keys are used).

    Returns:
        result (dict): Dictionary with keys:
            - ``pval_matrix`` (np.ndarray): (K, K) corrected p-values.
              Diagonal entries are NaN.
            - ``sig_matrix`` (np.ndarray): (K, K) boolean — True where
              corrected p < ``alpha``.
            - ``n_comparisons`` (int): Number of pairwise comparisons.
            - ``labels`` (list[str]): Ordered group labels.

    Notes:
        - Requires ``scipy`` (optional dependency). Raises ``ImportError``
          with installation instructions if not available.
    """
    try:
        from scipy import stats as sp_stats
    except ImportError as e:
        raise ImportError(
            "pairwise_tests requires 'scipy'. Install with: pip install scipy"
        ) from e

    # --- Normalise input --------------------------------------------------
    if isinstance(groups, dict):
        ordered_labels = list(groups.keys())
        data = [np.asarray(groups[k]).ravel() for k in ordered_labels]
    else:
        data = [np.asarray(a).ravel() for a in groups]
        if labels is not None:
            ordered_labels = list(labels)
        else:
            ordered_labels = [str(i) for i in range(len(data))]

    # Strip NaNs
    data = [d[~np.isnan(d)] for d in data]

    K = len(data)
    n_comparisons = K * (K - 1) // 2
    pval_matrix = np.full((K, K), np.nan)

    # --- Select test function ---------------------------------------------
    if test == "welch_t":

        def _test(a, b):
            _, p = sp_stats.ttest_ind(a, b, equal_var=False)
            return p

    elif test == "student_t":

        def _test(a, b):
            _, p = sp_stats.ttest_ind(a, b, equal_var=True)
            return p

    elif test == "mann_whitney":

        def _test(a, b):
            _, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
            return p

    else:
        raise ValueError(
            f"Unknown test '{test}'. Use 'welch_t', 'student_t', or 'mann_whitney'."
        )

    # --- Run pairwise tests -----------------------------------------------
    for i in range(K):
        for j in range(i + 1, K):
            p = _test(data[i], data[j])

            if correction == "bonferroni":
                p = min(p * n_comparisons, 1.0)
            elif correction is not None:
                raise ValueError(
                    f"Unknown correction '{correction}'. " "Use 'bonferroni' or None."
                )

            pval_matrix[i, j] = p
            pval_matrix[j, i] = p

    sig_matrix = pval_matrix < alpha

    return {
        "pval_matrix": pval_matrix,
        "sig_matrix": sig_matrix,
        "n_comparisons": n_comparisons,
        "labels": ordered_labels,
    }
