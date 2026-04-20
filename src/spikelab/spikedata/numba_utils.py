"""
Optional numba-accelerated kernels for computationally expensive operations.

When ``numba`` is installed, the functions in this module provide parallelised
implementations of pairwise STTC, pairwise latencies, and spike-triggered
population rate.  All kernels use a **flat array + offset** representation for
spike trains so that ``prange`` can distribute pair/unit work across threads.

When ``numba`` is *not* installed, a ``NUMBA_AVAILABLE`` flag is set to False
and the calling code falls back to the existing pure-numpy loops.
"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    # Provide no-op decorators so the module can still be imported (the
    # functions below will never be called when NUMBA_AVAILABLE is False).
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def _decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return _decorator

    def prange(*args, **kwargs):  # type: ignore[no-redef]
        return range(*args)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def flatten_spike_trains(trains, start_time=0.0):
    """Convert a list of spike-time arrays to a flat array + offset vector.

    Parameters:
        trains (list[np.ndarray]): Per-unit spike time arrays (ms).
        start_time (float): Value subtracted from all spike times so that
            the recording starts at 0.

    Returns:
        flat (np.ndarray): Concatenated, 0-based spike times (float64).
        offsets (np.ndarray): Int64 array of length ``len(trains) + 1``.
            Unit *i*'s spikes are ``flat[offsets[i]:offsets[i+1]]``.
    """
    n = len(trains)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, t in enumerate(trains):
        offsets[i + 1] = offsets[i] + len(t)
    total = int(offsets[-1])
    if total == 0:
        return np.empty(0, dtype=np.float64), offsets
    flat = np.empty(total, dtype=np.float64)
    for i, t in enumerate(trains):
        arr = np.asarray(t, dtype=np.float64)
        flat[offsets[i] : offsets[i + 1]] = arr - start_time
    return flat, offsets


# ===================================================================
# STTC kernels
# ===================================================================


@njit
def _nb_sttc_ta(spike_times, delt, tmax):
    """Total time within ±delt of any spike (Cutts & Eglen formula).

    Mirrors ``utils._sttc_ta`` so that results are identical to the
    pure-numpy path.  Accumulates inter-spike contributions into a
    temporary array and sums once (matching numpy's vectorised
    ``np.minimum(np.diff(t), 2*delt).sum()`` reduction order).
    """
    n = len(spike_times)
    if n == 0:
        return 0.0
    # Edge contribution: min(delt, first_spike) + min(delt, tmax - last_spike)
    first = spike_times[0]
    last = spike_times[n - 1]
    base = min(delt, first) + min(delt, max(0.0, tmax - last))
    if n == 1:
        return base
    # Accumulate inter-spike contributions left-to-right
    total = 0.0
    for i in range(n - 1):
        d = spike_times[i + 1] - spike_times[i]
        total += d if d < 2.0 * delt else 2.0 * delt
    return base + total


@njit
def _nb_sttc_na(tA, tB, delt):
    """Count spikes in tA within ±delt of any spike in tB.

    Mirrors ``utils._sttc_na`` using binary search per spike in tA,
    matching the numpy searchsorted logic exactly.
    """
    nA = len(tA)
    nB = len(tB)
    if nA == 0 or nB == 0:
        return 0

    if nB == 1:
        count = 0
        for i in range(nA):
            if abs(tA[i] - tB[0]) <= delt:
                count += 1
        return count

    count = 0
    for i in range(nA):
        t = tA[i]
        # Binary search: find first index in tB where tB[idx] >= t
        lo = 0
        hi = nB
        while lo < hi:
            mid = (lo + hi) // 2
            if tB[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        # lo is the insertion point; clip to [1, nB-1] like numpy
        iB = lo
        if iB < 1:
            iB = 1
        if iB > nB - 1:
            iB = nB - 1
        # Check both tB[iB] and tB[iB-1], take the closer one
        dt_right = abs(tB[iB] - t)
        dt_left = abs(tB[iB - 1] - t)
        min_dt = dt_right if dt_right < dt_left else dt_left
        if min_dt <= delt:
            count += 1
    return count


@njit
def _nb_sttc_pair(spike_a, spike_b, delt, length):
    """Compute STTC for a single pair.  Returns 0.0 for degenerate cases."""
    nA = len(spike_a)
    nB = len(spike_b)
    if nA == 0 or nB == 0:
        return 0.0

    TA = _nb_sttc_ta(spike_a, delt, length) / length
    TB = _nb_sttc_ta(spike_b, delt, length) / length

    PA = _nb_sttc_na(spike_a, spike_b, delt) / nA
    PB = _nb_sttc_na(spike_b, spike_a, delt) / nB

    denom1 = 1.0 - PA * TB
    denom2 = 1.0 - PB * TA
    aa = (PA - TB) / denom1 if abs(denom1) > 1e-12 else 0.0
    bb = (PB - TA) / denom2 if abs(denom2) > 1e-12 else 0.0
    return (aa + bb) / 2.0


@njit(parallel=True)
def nb_sttc_all_pairs(flat, offsets, n_units, delt, length):
    """Compute STTC for all upper-triangle pairs in parallel.

    Parameters:
        flat (np.ndarray): Concatenated 0-based spike times.
        offsets (np.ndarray): Offset vector (length n_units + 1).
        n_units (int): Number of units.
        delt (float): STTC time window (ms).
        length (float): Recording duration (ms).

    Returns:
        result (np.ndarray): Upper-triangle STTC values, length
            ``n_units * (n_units - 1) // 2``.
    """
    n_pairs = n_units * (n_units - 1) // 2
    # Pre-build pair index arrays so prange can iterate over a flat index.
    pairs_i = np.empty(n_pairs, dtype=np.int64)
    pairs_j = np.empty(n_pairs, dtype=np.int64)
    k = 0
    for i in range(n_units):
        for j in range(i + 1, n_units):
            pairs_i[k] = i
            pairs_j[k] = j
            k += 1

    result = np.empty(n_pairs, dtype=np.float64)
    for k in prange(n_pairs):
        i = pairs_i[k]
        j = pairs_j[k]
        a = flat[offsets[i] : offsets[i + 1]]
        b = flat[offsets[j] : offsets[j + 1]]
        result[k] = _nb_sttc_pair(a, b, delt, length)
    return result


# ===================================================================
# Pairwise latency kernels
# ===================================================================


@njit
def _nb_latencies_pair(train_i, train_j, window_ms, has_window):
    """Compute mean and std of signed nearest-spike latencies from i→j.

    Returns (mean, std, count).  If count == 0, mean and std are 0.0.
    """
    nI = len(train_i)
    nJ = len(train_j)
    if nI == 0 or nJ == 0:
        return 0.0, 0.0, 0

    # For each spike in i, find nearest spike in j using binary search
    sum_lat = 0.0
    sum_sq = 0.0
    count = 0
    for s in range(nI):
        t = train_i[s]
        # Binary search for insertion point
        lo = 0
        hi = nJ
        while lo < hi:
            mid = (lo + hi) // 2
            if train_j[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        # lo is the index of first spike in j >= t
        # Check lo and lo-1 to find closest
        best_lat = 1e30
        best_abs = 1e30
        if lo < nJ:
            lat = train_j[lo] - t
            if abs(lat) < best_abs:
                best_abs = abs(lat)
                best_lat = lat
        if lo > 0:
            lat = train_j[lo - 1] - t
            if abs(lat) < best_abs:
                best_abs = abs(lat)
                best_lat = lat

        if has_window and best_abs > window_ms:
            continue

        sum_lat += best_lat
        sum_sq += best_lat * best_lat
        count += 1

    if count == 0:
        return 0.0, 0.0, 0
    mean = sum_lat / count
    variance = sum_sq / count - mean * mean
    std = np.sqrt(max(variance, 0.0))
    return mean, std, count


@njit(parallel=True)
def nb_latencies_all_pairs(flat, offsets, n_units, window_ms, has_window):
    """Compute pairwise nearest-spike latencies for all ordered pairs.

    Parameters:
        flat (np.ndarray): Concatenated 0-based spike times.
        offsets (np.ndarray): Offset vector (length n_units + 1).
        n_units (int): Number of units.
        window_ms (float): Maximum absolute latency to include.
        has_window (bool): Whether to apply the window filter.

    Returns:
        mean_matrix (np.ndarray): (N, N) mean latencies.
        std_matrix (np.ndarray): (N, N) std latencies.
    """
    mean_matrix = np.zeros((n_units, n_units), dtype=np.float64)
    std_matrix = np.zeros((n_units, n_units), dtype=np.float64)

    # Total number of ordered off-diagonal pairs
    n_tasks = n_units * (n_units - 1)
    task_i = np.empty(n_tasks, dtype=np.int64)
    task_j = np.empty(n_tasks, dtype=np.int64)
    k = 0
    for i in range(n_units):
        for j in range(n_units):
            if i != j:
                task_i[k] = i
                task_j[k] = j
                k += 1

    for k in prange(n_tasks):
        i = task_i[k]
        j = task_j[k]
        ti = flat[offsets[i] : offsets[i + 1]]
        tj = flat[offsets[j] : offsets[j + 1]]
        m, s, _ = _nb_latencies_pair(ti, tj, window_ms, has_window)
        mean_matrix[i, j] = m
        std_matrix[i, j] = s

    return mean_matrix, std_matrix


# ===================================================================
# Spike-triggered population rate kernel
# ===================================================================


@njit(parallel=True)
def nb_spike_trig_pop_rate(spike_matrix, lags):
    """Compute raw (unfiltered) spike-triggered population rate.

    Parameters:
        spike_matrix (np.ndarray): (N, T) binned spike counts.
        lags (np.ndarray): 1-D array of lag values.

    Returns:
        stPR (np.ndarray): (N, len(lags)) raw coupling curves.
    """
    num_neurons = spike_matrix.shape[0]
    num_bins = spike_matrix.shape[1]
    n_lags = len(lags)

    # Pre-compute population sum and per-neuron stats
    pop_sum = np.zeros(num_bins, dtype=np.float64)
    for t in range(num_bins):
        s = 0.0
        for i in range(num_neurons):
            s += spike_matrix[i, t]
        pop_sum[t] = s

    mu = np.zeros(num_neurons, dtype=np.float64)
    total_spikes = np.zeros(num_neurons, dtype=np.float64)
    mu_sum = 0.0
    for i in range(num_neurons):
        s = 0.0
        for t in range(num_bins):
            s += spike_matrix[i, t]
        total_spikes[i] = s
        mu[i] = s / num_bins
        mu_sum += mu[i]

    stPR = np.zeros((num_neurons, n_lags), dtype=np.float64)

    for i in prange(num_neurons):
        if total_spikes[i] == 0 or mu[i] == 0:
            continue

        # Σ_{j≠i} μ_j = leave-one-out sum of mean rates
        mu_loo = mu_sum - mu[i]

        # Leave-one-out population rate mean
        P_loo_mean = 0.0
        for t in range(num_bins):
            P_loo_mean += pop_sum[t] - spike_matrix[i, t]
        P_loo_mean /= num_bins

        # Find spike times for neuron i
        n_spikes_i = int(total_spikes[i])
        spike_times_i = np.empty(n_spikes_i, dtype=np.int64)
        idx = 0
        for t in range(num_bins):
            if spike_matrix[i, t] > 0:
                spike_times_i[idx] = t
                idx += 1

        # Compute coupling for each lag
        for tau_idx in range(n_lags):
            tau = lags[tau_idx]
            sum_dev = 0.0
            for s_idx in range(n_spikes_i):
                t = spike_times_i[s_idx] + tau
                if 0 <= t < num_bins:
                    P_loo_t = pop_sum[t] - spike_matrix[i, t]
                    sum_dev += P_loo_t - P_loo_mean
            stPR[i, tau_idx] = sum_dev / (total_spikes[i] * mu_loo)

    return stPR


# ===================================================================
# Sorter comparison kernels
# ===================================================================


@njit(nogil=True)
def _nb_count_matching_spikes(times1, times2, delta):
    """Numba-accelerated greedy spike matching (single pair).

    Equivalent to ``_count_matching_spikes`` in ``utils.py`` but compiled
    with ``nogil=True`` so that ThreadPoolExecutor can achieve true
    parallelism across pairs.

    Parameters:
        times1 (np.ndarray): Sorted float64 spike times for train 1.
        times2 (np.ndarray): Sorted float64 spike times for train 2.
        delta (float): Maximum temporal distance for a match.

    Returns:
        n_matches (int): Number of matched spike pairs.
    """
    n1 = len(times1)
    n2 = len(times2)
    if n1 == 0 or n2 == 0:
        return 0

    i = 0
    j = 0
    n_matches = 0

    while i < n1 and j < n2:
        dt = times1[i] - times2[j]
        if abs(dt) <= delta:
            n_matches += 1
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1

    return n_matches


@njit(parallel=True)
def nb_agreement_all_pairs(flat1, offsets1, n_units1, flat2, offsets2, n_units2, delta):
    """Compute spike-time agreement matrix for all unit pairs across two sorters.

    Uses the flat-array + offset representation for both sorter outputs.
    Parallelises over rows (units in sorter 1) using ``prange``.

    Parameters:
        flat1 (np.ndarray): Concatenated spike times for sorter 1 (float64).
        offsets1 (np.ndarray): Offset array for sorter 1 (int64, length n_units1+1).
        n_units1 (int): Number of units in sorter 1.
        flat2 (np.ndarray): Concatenated spike times for sorter 2 (float64).
        offsets2 (np.ndarray): Offset array for sorter 2 (int64, length n_units2+1).
        n_units2 (int): Number of units in sorter 2.
        delta (float): Maximum temporal distance for a spike match.

    Returns:
        agreement (np.ndarray): (n_units1, n_units2) Jaccard agreement scores.
        frac_1 (np.ndarray): (n_units1, n_units2) fraction of sorter 1 spikes matched.
        frac_2 (np.ndarray): (n_units1, n_units2) fraction of sorter 2 spikes matched.
    """
    agreement = np.zeros((n_units1, n_units2), dtype=np.float64)
    frac_1 = np.zeros((n_units1, n_units2), dtype=np.float64)
    frac_2 = np.zeros((n_units1, n_units2), dtype=np.float64)

    for i in prange(n_units1):
        t1 = flat1[offsets1[i] : offsets1[i + 1]]
        n1 = len(t1)
        for j in range(n_units2):
            t2 = flat2[offsets2[j] : offsets2[j + 1]]
            n2 = len(t2)

            if n1 == 0 and n2 == 0:
                continue

            # Greedy matching (same algorithm as _count_matching_spikes)
            ii = 0
            jj = 0
            n_matches = 0
            while ii < n1 and jj < n2:
                dt = t1[ii] - t2[jj]
                if abs(dt) <= delta:
                    n_matches += 1
                    ii += 1
                    jj += 1
                elif dt < 0:
                    ii += 1
                else:
                    jj += 1

            denom = n1 + n2 - n_matches
            if denom > 0:
                agreement[i, j] = n_matches / denom
            if n1 > 0:
                frac_1[i, j] = n_matches / n1
            if n2 > 0:
                frac_2[i, j] = n_matches / n2

    return agreement, frac_1, frac_2
