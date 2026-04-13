"""Statistical utilities for SpikeLab.

Provides reusable statistical functions (regression, confidence intervals,
pairwise group comparisons, paired tests, omnibus tests) that can be used
independently of plotting.
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


def paired_test(
    a,
    b,
    test="wilcoxon",
    alternative="two-sided",
):
    """Run a paired statistical test on two matched samples.

    Parameters:
        a (array-like): First sample (1-D).
        b (array-like): Second sample (1-D, same length as *a*).
        test (str): ``"wilcoxon"`` (default) for the Wilcoxon signed-rank
            test, or ``"paired_t"`` for a paired Student's t-test.
        alternative (str): ``"two-sided"`` (default), ``"less"``, or
            ``"greater"``. Passed directly to the underlying scipy test.

    Returns:
        result (dict): Dictionary with keys:
            - ``statistic`` (float): Test statistic (W for Wilcoxon, t for
              paired t).
            - ``p_value`` (float): p-value for the test.
            - ``n`` (int): Number of valid (non-NaN, non-zero-difference)
              pairs used.

    Notes:
        - NaN pairs (where either *a* or *b* is NaN) are dropped
          automatically.
        - Requires ``scipy`` (optional dependency).
    """
    try:
        from scipy import stats as sp_stats
    except ImportError as e:
        raise ImportError(
            "paired_test requires 'scipy'. Install with: pip install scipy"
        ) from e

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(
            f"a and b must have the same length, got {len(a)} and {len(b)}."
        )

    valid = ~(np.isnan(a) | np.isnan(b))
    a = a[valid]
    b = b[valid]

    if len(a) < 1:
        raise ValueError("No valid (non-NaN) pairs to test.")

    if test == "wilcoxon":
        stat, p = sp_stats.wilcoxon(a, b, alternative=alternative)
    elif test == "paired_t":
        stat, p = sp_stats.ttest_rel(a, b, alternative=alternative)
    else:
        raise ValueError(f"Unknown test '{test}'. Use 'wilcoxon' or 'paired_t'.")

    return {"statistic": float(stat), "p_value": float(p), "n": len(a)}


def omnibus_test(
    groups,
    test="anova",
    posthoc="tukey",
    labels=None,
):
    """Run an omnibus test across groups with optional post-hoc comparisons.

    Parameters:
        groups (dict[str, array-like] or list[array-like]): Per-group data.
            Dict keys are used as labels; for list input supply *labels*
            separately.
        test (str): ``"anova"`` (default) for one-way ANOVA
            (``scipy.stats.f_oneway``), or ``"kruskal"`` for the
            Kruskal-Wallis H test.
        posthoc (str or None): Post-hoc test to run when the omnibus test is
            significant. ``"tukey"`` (default) for Tukey HSD,
            ``"none"``/``None`` to skip post-hoc comparisons.
        labels (list[str] or None): Group labels for list input. Ignored for
            dict input (keys are used).

    Returns:
        result (dict): Dictionary with keys:
            - ``statistic`` (float): F-statistic (ANOVA) or H-statistic
              (Kruskal-Wallis).
            - ``p_value`` (float): Omnibus p-value.
            - ``n_groups`` (int): Number of groups.
            - ``group_ns`` (list[int]): Sample sizes per group.
            - ``labels`` (list[str]): Ordered group labels.
            - ``posthoc`` (list[dict] or None): Post-hoc comparison results
              when *posthoc* is not None. Each dict contains:
              ``"group_a"``, ``"group_b"``, ``"p_value"``, ``"significant"``
              (at alpha=0.05).

    Notes:
        - NaN values are stripped from each group before testing.
        - Requires ``scipy`` (optional dependency).
    """
    try:
        from scipy import stats as sp_stats
    except ImportError as e:
        raise ImportError(
            "omnibus_test requires 'scipy'. Install with: pip install scipy"
        ) from e

    if isinstance(groups, dict):
        ordered_labels = list(groups.keys())
        data = [np.asarray(groups[k], dtype=float).ravel() for k in ordered_labels]
    else:
        data = [np.asarray(a, dtype=float).ravel() for a in groups]
        ordered_labels = (
            list(labels) if labels is not None else [str(i) for i in range(len(data))]
        )

    data = [d[~np.isnan(d)] for d in data]
    K = len(data)
    if K < 2:
        raise ValueError("Need at least 2 groups for an omnibus test.")

    if test == "anova":
        stat, p = sp_stats.f_oneway(*data)
    elif test == "kruskal":
        stat, p = sp_stats.kruskal(*data)
    else:
        raise ValueError(f"Unknown test '{test}'. Use 'anova' or 'kruskal'.")

    result = {
        "statistic": float(stat),
        "p_value": float(p),
        "n_groups": K,
        "group_ns": [len(d) for d in data],
        "labels": ordered_labels,
        "posthoc": None,
    }

    if posthoc is not None and posthoc != "none":
        if posthoc == "tukey":
            tukey = sp_stats.tukey_hsd(*data)
            posthoc_results = []
            for i in range(K):
                for j in range(i + 1, K):
                    pv = float(tukey.pvalue[i, j])
                    posthoc_results.append(
                        {
                            "group_a": ordered_labels[i],
                            "group_b": ordered_labels[j],
                            "p_value": pv,
                            "significant": pv < 0.05,
                        }
                    )
            result["posthoc"] = posthoc_results
        else:
            raise ValueError(f"Unknown posthoc '{posthoc}'. Use 'tukey' or None.")

    return result
