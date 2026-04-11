"""Unit curation methods for SpikeData objects.

Each public function accepts a SpikeData as its first argument and returns
``(SpikeData, result_dict)`` where *result_dict* always contains:

- ``metric`` — ``np.ndarray (N,)`` with the per-unit metric value
  (computed over **all** original units).
- ``passed`` — ``np.ndarray (N,)`` boolean mask indicating which units
  passed the curation criterion.

The returned SpikeData contains only the passing units (via ``subset``).

These functions are bound as methods on ``SpikeData`` by
``spikedata.py`` so they can be called as ``sd.curate_by_*(…)``.
"""

import numpy as np

from .utils import compute_cosine_similarity_with_lag


def curate_by_min_spikes(sd, min_spikes=30):
    """Remove units with fewer than *min_spikes* spikes.

    Parameters:
        sd (SpikeData): Source spike data.
        min_spikes (int): Minimum spike count threshold.

    Returns:
        sd_out (SpikeData): SpikeData with only passing units.
        result (dict): ``{"metric": np.ndarray (N,), "passed": np.ndarray (N,)}``.
            Metric is the spike count per unit.
    """
    metric = np.array([len(t) for t in sd.train], dtype=float)
    passed = metric >= min_spikes
    return sd.subset(np.where(passed)[0]), {"metric": metric, "passed": passed}


def curate_by_firing_rate(sd, min_rate_hz=0.05):
    """Remove units whose firing rate is below *min_rate_hz*.

    Parameters:
        sd (SpikeData): Source spike data.
        min_rate_hz (float): Minimum firing rate in Hz.

    Returns:
        sd_out (SpikeData): SpikeData with only passing units.
        result (dict): ``{"metric": np.ndarray (N,), "passed": np.ndarray (N,)}``.
            Metric is the firing rate in Hz per unit.
    """
    duration_s = sd.length / 1000.0
    if duration_s <= 0:
        metric = np.zeros(sd.N, dtype=float)
    else:
        metric = np.array([len(t) / duration_s for t in sd.train], dtype=float)
    passed = metric >= min_rate_hz
    return sd.subset(np.where(passed)[0]), {"metric": metric, "passed": passed}


def curate_by_isi_violations(
    sd, max_violation=1.0, threshold_ms=1.5, min_isi_ms=0.0, method="percent"
):
    """Remove units with excessive inter-spike-interval violations.

    Two methods are available:

    - ``"percent"`` — violation count divided by total spike count,
      expressed as a percentage.
    - ``"hill"`` — violation rate ratio from Hill et al. (2011)
      J Neurosci 31:8699-8705.  Values above 1 indicate highly
      contaminated units.

    Parameters:
        sd (SpikeData): Source spike data.
        max_violation (float): Maximum allowed violation metric.
        threshold_ms (float): Refractory period threshold in ms.
        min_isi_ms (float): Minimum possible ISI enforced by hardware or
            post-processing, in ms.
        method (str): ``"percent"`` or ``"hill"``.

    Returns:
        sd_out (SpikeData): SpikeData with only passing units.
        result (dict): ``{"metric": np.ndarray (N,), "passed": np.ndarray (N,)}``.
            Metric is the violation percentage or ratio per unit.
    """
    if method not in ("percent", "hill"):
        raise ValueError(f"method must be 'percent' or 'hill', got '{method}'")

    duration_s = sd.length / 1000.0
    threshold_s = threshold_ms / 1000.0
    min_isi_s = min_isi_ms / 1000.0

    metric = np.zeros(sd.N, dtype=float)
    for i, train in enumerate(sd.train):
        n_spikes = len(train)
        if n_spikes < 2:
            metric[i] = 0.0
            continue
        isis = np.diff(train)  # already in ms
        violation_count = np.sum(isis < threshold_ms)

        if method == "hill":
            violation_time = 2 * n_spikes * (threshold_s - min_isi_s)
            total_rate = n_spikes / duration_s if duration_s > 0 else 0.0
            violation_rate = (
                violation_count / violation_time if violation_time > 0 else 0.0
            )
            metric[i] = violation_rate / total_rate if total_rate > 0 else 0.0
        else:
            metric[i] = (violation_count / n_spikes) * 100.0

    passed = metric <= max_violation
    return sd.subset(np.where(passed)[0]), {"metric": metric, "passed": passed}


def curate_by_snr(sd, min_snr=5.0, ms_before=1.0, ms_after=2.0):
    """Remove units whose signal-to-noise ratio is below *min_snr*.

    SNR is defined as ``peak_amplitude / noise_level`` where peak
    amplitude is the absolute maximum of the average waveform on the
    channel with the largest amplitude, and noise level is estimated
    via the median absolute deviation (MAD) of the raw trace on that
    channel.

    The method first checks for a precomputed ``"snr"`` value in
    ``neuron_attributes``.  If not found, it computes SNR from
    ``raw_data`` (using ``get_waveform_traces``).  If neither is
    available a ``ValueError`` is raised.

    Parameters:
        sd (SpikeData): Source spike data.
        min_snr (float): Minimum SNR threshold.
        ms_before (float): ms before spike for waveform extraction
            (only used when computing from raw_data).
        ms_after (float): ms after spike for waveform extraction
            (only used when computing from raw_data).

    Returns:
        sd_out (SpikeData): SpikeData with only passing units.
        result (dict): ``{"metric": np.ndarray (N,), "passed": np.ndarray (N,)}``.
            Metric is the SNR per unit.
    """
    metric = _get_or_compute_waveform_metric(sd, "snr", ms_before, ms_after)
    passed = metric >= min_snr
    return sd.subset(np.where(passed)[0]), {"metric": metric, "passed": passed}


def curate_by_std_norm(
    sd,
    max_std_norm=1.0,
    at_peak=True,
    window_ms_before=0.5,
    window_ms_after=1.5,
    ms_before=1.0,
    ms_after=2.0,
):
    """Remove units whose normalized waveform standard deviation exceeds
    *max_std_norm*.

    Normalized STD is ``|std| / |amplitude|`` on the channel with the
    largest amplitude.  When *at_peak* is True, STD is measured at the
    single peak sample; otherwise it is averaged over a window around
    the peak.

    The method first checks for a precomputed ``"std_norm"`` value in
    ``neuron_attributes``.  If not found, it computes the metric from
    ``raw_data``.  If neither is available a ``ValueError`` is raised.

    Parameters:
        sd (SpikeData): Source spike data.
        max_std_norm (float): Maximum allowed normalized STD.
        at_peak (bool): Measure STD at peak sample only.
        window_ms_before (float): Window before peak for averaging STD
            (only used when *at_peak* is False).
        window_ms_after (float): Window after peak for averaging STD
            (only used when *at_peak* is False).
        ms_before (float): ms before spike for waveform extraction
            (only used when computing from raw_data).
        ms_after (float): ms after spike for waveform extraction
            (only used when computing from raw_data).

    Returns:
        sd_out (SpikeData): SpikeData with only passing units.
        result (dict): ``{"metric": np.ndarray (N,), "passed": np.ndarray (N,)}``.
            Metric is the normalized STD per unit.
    """
    metric = _get_or_compute_waveform_metric(
        sd,
        "std_norm",
        ms_before,
        ms_after,
        at_peak=at_peak,
        window_ms_before=window_ms_before,
        window_ms_after=window_ms_after,
    )
    passed = metric <= max_std_norm
    return sd.subset(np.where(passed)[0]), {"metric": metric, "passed": passed}


def compute_waveform_metrics(
    sd,
    ms_before=1.0,
    ms_after=2.0,
    at_peak=True,
    window_ms_before=0.5,
    window_ms_after=1.5,
):
    """Compute average waveforms, SNR, and normalized STD for every unit.

    Results are stored in ``neuron_attributes`` under the keys
    ``"snr"`` and ``"std_norm"``.  Average waveforms are stored by
    ``get_waveform_traces`` (called internally with ``store=True``).

    Parameters:
        sd (SpikeData): Source spike data.  Must have non-empty
            ``raw_data``.
        ms_before (float): ms before spike for waveform extraction.
        ms_after (float): ms after spike for waveform extraction.
        at_peak (bool): Measure STD at peak sample only.
        window_ms_before (float): Window before peak for averaging STD
            (only used when *at_peak* is False).
        window_ms_after (float): Window after peak for averaging STD
            (only used when *at_peak* is False).

    Returns:
        sd (SpikeData): The same SpikeData object (modified in place
            with updated ``neuron_attributes``).
        metrics (dict): ``{"snr": np.ndarray (N,),
            "std_norm": np.ndarray (N,)}``.
    """
    if sd.raw_data.size == 0:
        raise ValueError(
            "raw_data is empty. Attach raw voltage traces before calling "
            "compute_waveform_metrics."
        )

    if sd.neuron_attributes is None:
        sd.neuron_attributes = [{} for _ in range(sd.N)]

    # Extract waveforms for all units (stores avg_waveform in neuron_attributes)
    sd.get_waveform_traces(
        unit=None,
        ms_before=ms_before,
        ms_after=ms_after,
        store=True,
        return_avg_waveform=True,
    )

    # Compute noise levels via MAD on raw_data
    noise_levels = _estimate_noise_levels(sd.raw_data)

    snr_arr = np.zeros(sd.N, dtype=float)
    std_norm_arr = np.zeros(sd.N, dtype=float)

    # Determine sampling rate for window conversion
    if np.ndim(sd.raw_time) == 0 or sd.raw_time.shape == ():
        fs_kHz = float(sd.raw_time)
    else:
        fs_kHz = 1.0 / np.median(np.diff(sd.raw_time))

    for i in range(sd.N):
        attrs = sd.neuron_attributes[i]
        waveforms = attrs.get("waveforms")  # (channels, samples, spikes)
        if waveforms is None or waveforms.size == 0:
            snr_arr[i] = 0.0
            std_norm_arr[i] = np.inf
            continue

        avg_wf = attrs.get("avg_waveform")  # (channels, samples)
        if avg_wf is None:
            avg_wf = np.mean(waveforms, axis=2)

        # Find channel with max amplitude
        peak_per_chan = np.max(np.abs(avg_wf), axis=1)
        chan_max = int(np.argmax(peak_per_chan))

        # Peak amplitude and index on best channel
        chan_wf = avg_wf[chan_max, :]
        peak_ind = int(np.argmax(np.abs(chan_wf)))
        amplitude = np.abs(chan_wf[peak_ind])

        # SNR = amplitude / noise
        noise = noise_levels[chan_max] if chan_max < len(noise_levels) else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_arr[i] = amplitude / noise if noise > 0 else 0.0

        # Normalized STD
        wf_std = np.std(waveforms, axis=2)  # (channels, samples)
        chan_std = wf_std[chan_max, :]

        if at_peak:
            std_val = chan_std[peak_ind]
        else:
            n_before = max(1, int(round(window_ms_before * fs_kHz)))
            n_after = max(1, int(round(window_ms_after * fs_kHz))) + 1
            win_start = max(0, peak_ind - n_before)
            win_end = min(len(chan_std), peak_ind + n_after)
            std_val = np.mean(chan_std[win_start:win_end])

        with np.errstate(divide="ignore", invalid="ignore"):
            std_norm_arr[i] = np.abs(std_val / amplitude) if amplitude > 0 else np.inf

        # Store in neuron_attributes
        attrs["snr"] = float(snr_arr[i])
        attrs["std_norm"] = float(std_norm_arr[i])

    return sd, {"snr": snr_arr, "std_norm": std_norm_arr}


def curate(
    sd,
    min_spikes=None,
    min_rate_hz=None,
    isi_max=None,
    isi_threshold_ms=1.5,
    isi_min_ms=0.0,
    isi_method="percent",
    min_snr=None,
    max_std_norm=None,
    std_at_peak=True,
    std_window_ms_before=0.5,
    std_window_ms_after=1.5,
    snr_ms_before=1.0,
    snr_ms_after=2.0,
):
    """Apply multiple curation criteria in sequence (intersection).

    Only criteria whose threshold is not None are applied.  Returns the
    filtered SpikeData and a dict of per-criterion results.

    Parameters:
        sd (SpikeData): Source spike data.
        min_spikes (int or None): Minimum spike count.
        min_rate_hz (float or None): Minimum firing rate in Hz.
        isi_max (float or None): Maximum ISI violation metric.
        isi_threshold_ms (float): Refractory period for ISI check.
        isi_min_ms (float): Minimum possible ISI for ISI check.
        isi_method (str): ``"percent"`` or ``"hill"`` for ISI check.
        min_snr (float or None): Minimum SNR.
        max_std_norm (float or None): Maximum normalized STD.
        std_at_peak (bool): Measure STD at peak only.
        std_window_ms_before (float): Window before peak for STD averaging.
        std_window_ms_after (float): Window after peak for STD averaging.
        snr_ms_before (float): ms before spike for waveform extraction.
        snr_ms_after (float): ms after spike for waveform extraction.

    Returns:
        sd_out (SpikeData): SpikeData with only units passing all criteria.
        results (dict): ``{criterion_name: {"metric": (N,), "passed": (N,)}}``.
            Metrics are computed on the **original** units (before any
            filtering).  Only requested criteria are included.
    """
    results = {}
    current = sd

    if min_spikes is not None:
        current, res = curate_by_min_spikes(current, min_spikes=min_spikes)
        results["spike_count"] = res

    if min_rate_hz is not None:
        current, res = curate_by_firing_rate(current, min_rate_hz=min_rate_hz)
        results["firing_rate"] = res

    if isi_max is not None:
        current, res = curate_by_isi_violations(
            current,
            max_violation=isi_max,
            threshold_ms=isi_threshold_ms,
            min_isi_ms=isi_min_ms,
            method=isi_method,
        )
        results["isi_violation"] = res

    if min_snr is not None:
        current, res = curate_by_snr(
            current,
            min_snr=min_snr,
            ms_before=snr_ms_before,
            ms_after=snr_ms_after,
        )
        results["snr"] = res

    if max_std_norm is not None:
        current, res = curate_by_std_norm(
            current,
            max_std_norm=max_std_norm,
            at_peak=std_at_peak,
            window_ms_before=std_window_ms_before,
            window_ms_after=std_window_ms_after,
            ms_before=snr_ms_before,
            ms_after=snr_ms_after,
        )
        results["std_norm"] = res

    return current, results


def build_curation_history(sd_original, sd_curated, results, parameters=None):
    """Translate curation results into a serializable history dict.

    The output format mirrors the curation history produced by the
    Kilosort2 pipeline, making it suitable for saving as JSON.

    Parameters:
        sd_original (SpikeData): The SpikeData **before** curation.
        sd_curated (SpikeData): The SpikeData **after** curation.
        results (dict): Results dict returned by ``curate()`` or
            assembled manually from individual ``curate_by_*`` calls.
            Keys are criterion names, values are dicts with ``"metric"``
            and ``"passed"`` arrays.
        parameters (dict or None): Curation parameter values to record.
            If None, an empty dict is stored.

    Returns:
        history (dict): Serializable curation history with keys:
            ``curation_parameters``, ``initial``, ``curations``,
            ``curated``, ``failed``, ``metrics``, ``curated_final``.
    """

    # Resolve unit IDs: use neuron_attributes["unit_id"] if available,
    # otherwise fall back to positional indices.
    def _unit_ids(sd):
        if sd.neuron_attributes is not None:
            ids = [a.get("unit_id") for a in sd.neuron_attributes]
            if all(uid is not None for uid in ids):
                return [int(uid) for uid in ids]
        return list(range(sd.N))

    original_ids = _unit_ids(sd_original)
    final_ids = _unit_ids(sd_curated)

    curations = []
    curated = {}
    failed = {}
    metrics = {}

    # Walk through results in insertion order.  Each result was computed
    # on the SpikeData that entered that stage (after previous filters),
    # but the metric and passed arrays are indexed relative to that
    # stage's input.  We need to map back to the original unit IDs.
    #
    # Because curate() applies criteria sequentially, each stage's input
    # is a subset of the original.  We track the surviving ID list to
    # perform the mapping.
    surviving_ids = list(original_ids)

    for criterion, res in results.items():
        curations.append(criterion)
        metric_arr = res["metric"]
        passed_arr = res["passed"]

        stage_curated = []
        stage_failed = []
        stage_metrics = {}

        for j, uid in enumerate(surviving_ids):
            stage_metrics[uid] = float(metric_arr[j])
            if passed_arr[j]:
                stage_curated.append(uid)
            else:
                stage_failed.append(uid)

        curated[criterion] = stage_curated
        failed[criterion] = stage_failed
        metrics[criterion] = stage_metrics

        # Update survivors for the next stage
        surviving_ids = stage_curated

    return {
        "curation_parameters": parameters if parameters is not None else {},
        "initial": original_ids,
        "curations": curations,
        "curated": curated,
        "failed": failed,
        "metrics": metrics,
        "curated_final": final_ids,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_noise_levels(raw_data, num_chunks=20, chunk_size=10000, seed=0):
    """Estimate per-channel noise via MAD on random chunks of *raw_data*.

    Parameters:
        raw_data (np.ndarray): Shape ``(channels, time)``.
        num_chunks (int): Number of random chunks to sample.
        chunk_size (int): Samples per chunk.
        seed (int): Random seed.

    Returns:
        noise (np.ndarray): Shape ``(channels,)``.
    """
    rng = np.random.default_rng(seed)
    n_channels, n_samples = raw_data.shape
    max_start = n_samples - chunk_size
    if max_start <= 0:
        # Recording shorter than one chunk — use all data
        data = raw_data
    else:
        starts = rng.integers(0, max_start, size=num_chunks)
        chunks = [raw_data[:, s : s + chunk_size] for s in starts]
        data = np.concatenate(chunks, axis=1)

    # MAD-based noise estimate: median(|x - median(x)|) / 0.6745
    medians = np.median(data, axis=1, keepdims=True)
    noise = np.median(np.abs(data - medians), axis=1) / 0.6745
    return noise


def _get_or_compute_waveform_metric(sd, metric_name, ms_before, ms_after, **kwargs):
    """Try to read a precomputed metric from neuron_attributes, fall back
    to computing from raw_data, or raise if neither is available.

    Returns:
        metric (np.ndarray): Shape ``(N,)``.
    """
    # 1. Check neuron_attributes for precomputed values
    if sd.neuron_attributes is not None:
        values = []
        for attrs in sd.neuron_attributes:
            val = attrs.get(metric_name)
            if val is None:
                break
            values.append(float(val))
        if len(values) == sd.N:
            return np.array(values, dtype=float)

    # 2. Fall back to computing from raw_data
    if sd.raw_data.size > 0:
        at_peak = kwargs.get("at_peak", True)
        window_ms_before = kwargs.get("window_ms_before", 0.5)
        window_ms_after = kwargs.get("window_ms_after", 1.5)
        _, metrics = compute_waveform_metrics(
            sd,
            ms_before=ms_before,
            ms_after=ms_after,
            at_peak=at_peak,
            window_ms_before=window_ms_before,
            window_ms_after=window_ms_after,
        )
        return metrics[metric_name]

    # 3. Neither available
    raise ValueError(
        f"Cannot compute '{metric_name}': no precomputed values in "
        "neuron_attributes and raw_data is empty. Call "
        "compute_waveform_metrics() first, or attach raw voltage traces."
    )


# ---------------------------------------------------------------------------
# Merge-based deduplication
# ---------------------------------------------------------------------------


def _find_nearby_unit_pairs(sd, dist_um=24.8):
    """Return all pairs of units whose electrode positions are within distance.

    Uses ``sd.unit_locations`` (which normalizes across ``"location"``,
    ``"x"/"y"``, and ``"position"`` keys) so this works regardless of
    which loader populated the SpikeData object.

    Parameters:
        sd (SpikeData): spike data
        dist_um (float): Maximum inter-electrode distance in um.
            Default 24.8 accounts for the 24.7 µm electrode neighbourhood
            radius plus floating-point tolerance.

    Returns:
        pairs (set[tuple[int, int]]): Set of (i, j) index tuples with i < j.
    """
    locations = sd.unit_locations
    if locations is None:
        raise ValueError(
            "sd.unit_locations is None. Position data is required to find nearby pairs."
        )
    pairs = set()
    for i in range(sd.N):
        for j in range(i + 1, sd.N):
            pos_i, pos_j = locations[i], locations[j]
            dist = np.sqrt(np.sum((pos_i[:2] - pos_j[:2]) ** 2))
            if dist <= dist_um:
                pairs.add((i, j))
    return pairs


def _filter_pairs_by_isi_violations(
    sd, pairs, max_violation_rate=0.04, threshold_ms=1.5
):
    """Remove pairs where either unit exceeds the ISI violation rate threshold.

    ISI violation rate is n_violations / n_spikes where a violation is any
    inter-spike interval shorter than threshold_ms.

    Parameters:
        sd (SpikeData): spike data
        pairs (set[tuple[int, int]]): Candidate unit-index pairs.
        max_violation_rate (float): Maximum allowed violation rate as a
            fraction (not percent).  Default 0.04 (4 %).
        threshold_ms (float): Refractory period threshold in ms.

    Returns:
        filtered_pairs (set[tuple[int, int]]): Pairs where both units pass.
        violation_rates (dict[int, float]): Per-unit violation rates for
            every unit that appeared in pairs.
    """
    if not pairs:
        return set(), {}

    units_in_pairs = {u for pair in pairs for u in pair}
    violation_rates = {
        u: _isi_violation_fraction(sd.train[u], threshold_ms) for u in units_in_pairs
    }
    filtered_pairs = {
        (i, j)
        for i, j in pairs
        if violation_rates[i] <= max_violation_rate
        and violation_rates[j] <= max_violation_rate
    }
    return filtered_pairs, violation_rates


def _compute_pairwise_similarity(sd, pairs, max_lag=10):
    """Compute cosine similarity for candidate pairs using flat concatenated waveforms.

    Each unit is represented as a single 1-D array built by concatenating
    avg_waveform in a globally consistent channel order (sorted numerically).
    Channels absent for a unit are zero-padded.  No spatial weighting or channel
    selection is applied.

    Parameters:
        sd (SpikeData): Source spike data with neuron_attributes.
        pairs (set[tuple[int, int]]): Candidate unit-index pairs to evaluate.
        max_lag (int): Maximum lag in samples for cosine similarity alignment.
            Pairs whose best match falls at the boundary (abs(lag) == max_lag)
            are assigned 0 similarity.  Default 10.

    Returns:
        similarity_matrix (np.ndarray): Shape (N, N).  NaN for unevaluated
            pairs; 1.0 on the diagonal.
        lag_matrix (np.ndarray): Shape (N, N).  Best lag in samples; NaN
            for unevaluated pairs; 0 on the diagonal.
        unit_ids (list): neuron_attributes["unit_id"] or index fallback,
            one entry per unit.
    """
    if sd.neuron_attributes is None:
        raise ValueError(
            "neuron_attributes is None. Waveform data is required for similarity computation."
        )

    n = sd.N
    sim_mat = np.full((n, n), np.nan)
    lag_mat = np.full((n, n), np.nan)
    np.fill_diagonal(sim_mat, 1.0)
    np.fill_diagonal(lag_mat, 0.0)

    unit_ids = [attrs.get("unit_id", i) for i, attrs in enumerate(sd.neuron_attributes)]

    if not pairs:
        return sim_mat, lag_mat, unit_ids

    # Build a single global channel list from all units, sorted numerically.
    all_channels = set()
    wf_lengths = []
    for attrs in sd.neuron_attributes:
        avg_wf = attrs.get("avg_waveform")
        if avg_wf is None:
            continue
        wf_lengths.append(avg_wf.shape[1])
        traces_meta = attrs.get("traces_meta", {})
        for ch in traces_meta.get("channels", []):
            all_channels.add(int(ch))

    if not wf_lengths:
        raise ValueError(
            "No units have 'avg_waveform' in neuron_attributes. "
            "Load waveform data before calling _compute_pairwise_similarity()."
        )

    if not all_channels:
        raise ValueError(
            "avg_waveform found in neuron_attributes but no unit has "
            "traces_meta['channels']. Call compute_waveform_metrics() or "
            "get_waveform_traces(store=True) to populate both keys together."
        )

    global_channels = sorted(all_channels)
    template_len = max(wf_lengths)

    # Pre-build a 1-D array for every unit using the shared channel order.
    unit_arrays = {
        i: _build_1d_array_for_channels(sd, i, global_channels, template_len)
        for i in range(n)
    }

    for i, j in pairs:
        arr_i = unit_arrays[i]
        arr_j = unit_arrays[j]

        if not np.any(arr_i) or not np.any(arr_j):
            continue

        sim, best_lag = compute_cosine_similarity_with_lag(
            arr_i, arr_j, max_lag=max_lag
        )
        if abs(best_lag) == max_lag:
            sim = 0.0

        sim_mat[i, j] = sim_mat[j, i] = sim
        lag_mat[i, j] = best_lag
        lag_mat[j, i] = -best_lag

    return sim_mat, lag_mat, unit_ids


def _filter_by_cosine_sim(pairs, similarity_matrix, threshold=0.9):
    """Return the subset of pairs whose cosine similarity meets threshold.

    Parameters:
        pairs (set[tuple[int, int]]): Candidate unit-index pairs.
        similarity_matrix (np.ndarray): Shape (N, N) similarity values,
            e.g. from _compute_pairwise_similarity().
        threshold (float): Minimum cosine similarity to retain a pair.

    Returns:
        filtered_pairs (set[tuple[int, int]]): Pairs passing the threshold.
    """
    return {
        (i, j)
        for i, j in pairs
        if not np.isnan(similarity_matrix[i, j])
        and similarity_matrix[i, j] >= threshold
    }


def curate_by_merge_duplicates(
    sd,
    dist_um=24.8,
    max_violation_rate=0.04,
    isi_threshold_ms=1.5,
    cosine_threshold=0.9,
    max_lag=10,
    delta_ms=0.4,
    max_isi_increase=0.04,
    verbose=False,
):
    """Remove duplicate units by merging nearby pairs with similar waveforms.

    Runs the full merge-based deduplication pipeline:

    1. Find spatially nearby unit pairs within dist_um.
    2. Discard pairs where either unit exceeds the ISI violation threshold.
    3. Compute pairwise cosine waveform similarity.
    4. Discard pairs below cosine_threshold.
    5. Greedily merge accepted pairs; a merge is rejected if the ISI
       violation fraction increases by more than max_isi_increase.

    Requires neuron_attributes with position and avg_waveform
    entries.  Unlike other curate_by_* functions this merges spike
    trains rather than simply removing units.

    Parameters:
        sd (SpikeData): spike data.
        dist_um (float): Maximum inter-electrode distance in µm to consider
            a pair as candidate duplicates.
        max_violation_rate (float): Maximum ISI violation rate (fraction,
            not percent) for a unit to participate in a merge.
        isi_threshold_ms (float): Refractory period threshold in ms.
        cosine_threshold (float): Minimum cosine similarity to merge a pair.
        max_lag (int): Maximum lag in samples for cosine similarity alignment.
        delta_ms (float): Spike deduplication window in ms when merging trains.
        max_isi_increase (float): Maximum allowable absolute increase in ISI
            violation fraction after merging.
        verbose (bool): Print per-pair merge decisions.

    Returns:
        sd_out (SpikeData): SpikeData with merged units.
        result (dict): {"metric": np.ndarray (N,), "passed": np.ndarray (N,)}.
            metric[i] is the cosine similarity to the partner unit in the
            accepted merge, or 0.0 if unit i was not involved in any merge.
            passed[i] is True if unit i was retained (primary or
            standalone), False if it was absorbed into another unit.
    """
    metric = np.zeros(sd.N, dtype=float)
    passed = np.ones(sd.N, dtype=bool)

    pairs = _find_nearby_unit_pairs(sd, dist_um=dist_um)
    if not pairs:
        return sd.subset(np.arange(sd.N)), {"metric": metric, "passed": passed}

    pairs, _ = _filter_pairs_by_isi_violations(
        sd, pairs, max_violation_rate=max_violation_rate, threshold_ms=isi_threshold_ms
    )
    if not pairs:
        return sd.subset(np.arange(sd.N)), {"metric": metric, "passed": passed}

    sim_mat, lag_mat, _ = _compute_pairwise_similarity(sd, pairs, max_lag=max_lag)
    pairs = _filter_by_cosine_sim(pairs, sim_mat, threshold=cosine_threshold)
    if not pairs:
        return sd.subset(np.arange(sd.N)), {"metric": metric, "passed": passed}

    sd_out, merge_result = _merge_redundant_units(
        sd,
        pairs,
        sim_mat,
        lag_matrix=lag_mat,
        delta_ms=delta_ms,
        max_isi_increase=max_isi_increase,
        isi_threshold_ms=isi_threshold_ms,
        verbose=verbose,
    )

    for primary, secondary, sim in merge_result["merged_pairs"]:
        passed[secondary] = False
        metric[secondary] = sim
        metric[primary] = max(metric[primary], sim)

    return sd_out, {"metric": metric, "passed": passed}


def _merge_redundant_units(
    sd,
    pairs,
    similarity_matrix,
    lag_matrix=None,
    delta_ms=0.4,
    max_isi_increase=0.04,
    isi_threshold_ms=1.5,
    verbose=False,
):
    """Merge pre-filtered candidate duplicate unit pairs into a new SpikeData.

    Pairs are processed in descending similarity order (greedy).  For each
    pair the unit with more spikes is kept as primary; the unit with fewer
    spikes is merged into it.  A merge is accepted only if the ISI violation
    rate after merging does not exceed the pre-merge maximum by more than
    max_isi_increase.  Units can be involved in multiple merges (e.g., A→B
    then B→C results in a final unit containing spikes from A, B, and C).

    Parameters:
        sd (SpikeData): Source spike data.
        pairs (set[tuple[int, int]] or list[tuple[int, int]]): Candidate
            duplicate pairs, e.g. from _filter_by_cosine_sim().
        similarity_matrix (np.ndarray): Shape (N, N) similarity values
            used to sort pairs and record scores.
        lag_matrix (np.ndarray, optional): Shape (N, N) lag values in samples.
            If provided, the secondary unit's spikes are shifted by the lag
            before merging to correct for timing offsets.  Default None.
        delta_ms (float): Spike deduplication window in ms.
        max_isi_increase (float): Maximum allowable absolute increase in ISI
            violation fraction after merging.  Default 0.04 (4 percentage points).
        isi_threshold_ms (float): ISI violation threshold in ms.
        verbose (bool): Print a line for each pair decision.

    Returns:
        sd_out (SpikeData): New SpikeData with merged units.
        result (dict): {"merged_pairs": list[tuple], "n_removed": int}.
            merged_pairs is a list of (primary, secondary, similarity)
            tuples that were accepted.
    """
    if not pairs:
        raise ValueError(
            "pairs must be a non-empty collection. "
            "Run _filter_by_cosine_sim() first."
        )

    sorted_pairs = sorted(
        ((i, j, float(similarity_matrix[i, j])) for i, j in pairs),
        key=lambda x: x[2],
        reverse=True,
    )

    merge_chain: dict = (
        {}
    )  # unit_idx → primary_idx (maps each unit to its final primary)
    current_train: dict = {}  # tracks merged-so-far train for each primary
    accepted_pairs = []

    def _resolve(unit):
        """Follow merge_chain to its root (with path compression)."""
        while unit in merge_chain:
            nxt = merge_chain[unit]
            if nxt in merge_chain:
                merge_chain[unit] = merge_chain[nxt]
            unit = nxt
        return unit

    for i, j, sim in sorted_pairs:

        # Resolve which primary each unit is currently merged into (full chain)
        prim_i = _resolve(i)
        prim_j = _resolve(j)

        # Skip if both units are already merged into the same primary
        if prim_i == prim_j:
            continue

        primary, secondary = _choose_primary_unit(sd, i, j)
        prim_primary = _resolve(primary)
        prim_secondary = _resolve(secondary)

        # Skip if already chained together via different paths
        if prim_primary == prim_secondary:
            continue

        # Use the already-merged primary train if it has prior merges accepted
        primary_train = current_train.get(prim_primary, sd.train[prim_primary])
        secondary_train = current_train.get(prim_secondary, sd.train[prim_secondary])

        # Apply lag correction only when secondary is fresh (not yet merged into another chain)
        if lag_matrix is not None and prim_secondary == secondary:
            lag_val = lag_matrix[primary, secondary]
            if not np.isnan(lag_val) and lag_val != 0:
                fs = sd.metadata.get("fs_Hz", 30000) if sd.metadata else 30000
                secondary_train = secondary_train + float(lag_val) / (fs / 1000.0)

        before_max = max(
            _isi_violation_fraction(primary_train, isi_threshold_ms),
            _isi_violation_fraction(secondary_train, isi_threshold_ms),
        )
        merged_train, _ = _merge_two_trains(primary_train, secondary_train, delta_ms)
        after_rate = _isi_violation_fraction(merged_train, isi_threshold_ms)
        isi_increase = after_rate - before_max

        if isi_increase <= max_isi_increase:
            # Accept the merge: prim_secondary is merged into prim_primary
            merge_chain[prim_secondary] = prim_primary
            current_train[prim_primary] = merged_train
            accepted_pairs.append((primary, secondary, sim))
            if verbose:
                print(
                    f"  Merge [{i},{j}]: sim={sim:.3f}, "
                    f"ISI {before_max:.3f}→{after_rate:.3f} (Δ={isi_increase:+.3f})"
                )
        else:
            if verbose:
                print(
                    f"  Skip  [{i},{j}]: sim={sim:.3f}, "
                    f"ISI increase too high (Δ={isi_increase:+.3f} > {max_isi_increase})"
                )

    # Build a mapping: final_primary → list of original units that merged into it.
    # Follow the chain for merges (e.g., A→B, B→C yields A,B→C).
    primary_groups: dict = {}
    for orig_unit in range(sd.N):
        final_primary = merge_chain.get(orig_unit, orig_unit)
        while final_primary in merge_chain:
            final_primary = merge_chain[final_primary]
        primary_groups.setdefault(final_primary, []).append(orig_unit)

    new_trains = []
    new_attrs = []
    for primary in sorted(primary_groups.keys()):
        constituent_units = primary_groups[primary]
        # Reuse the pre-merged train from current_train if available
        merged = current_train.get(primary, sd.train[primary])
        original_spike_count = sum(len(sd.train[u]) for u in constituent_units)
        total_dup = original_spike_count - len(merged)
        new_trains.append(merged)
        attrs = sd.neuron_attributes[primary].copy() if sd.neuron_attributes else {}
        attrs["merged_from"] = [
            (sd.neuron_attributes[u].get("unit_id", u) if sd.neuron_attributes else u)
            for u in constituent_units
        ]
        attrs["n_duplicates_removed"] = total_dup
        new_attrs.append(attrs)

    from .spikedata import SpikeData

    sd_out = SpikeData(
        new_trains,
        length=sd.length,
        start_time=sd.start_time,
        neuron_attributes=new_attrs,
        metadata=sd.metadata.copy() if sd.metadata else {},
        raw_data=sd.raw_data,
        raw_time=sd.raw_time,
    )

    n_removed = sd.N - len(new_trains)
    if verbose:
        print(f"  {n_removed} units merged; " f"{sd.N} → {sd_out.N} units")

    return sd_out, {"merged_pairs": accepted_pairs, "n_removed": n_removed}


# ---------------------------------------------------------------------------
# Internal helpers (merge-based deduplication)
# ---------------------------------------------------------------------------


def _isi_violation_fraction(train, threshold_ms):
    """Return the ISI violation rate as a fraction for a spike train.

    ISI violation rate is n_violations / n_spikes.

    Parameters:
        train (np.ndarray): Spike times in milliseconds.
        threshold_ms (float): Refractory period threshold in ms.

    Returns:
        rate (float): Violation rate as a fraction (0.0-1.0), or 0.0 for
            fewer than 2 spikes.
    """
    if len(train) < 2:
        return 0.0
    isis = np.diff(train)
    violation_count = np.sum(isis < threshold_ms)
    return float(violation_count / len(train))


def _build_1d_array_for_channels(sd, unit_idx, channels, template_len):
    """Build a 1-D waveform vector for unit_idx on an explicit channel list.

    Channels present in the unit's avg_waveform are copied in; missing
    channels are zero-padded.

    Parameters:
        sd (SpikeData): Source spike data with neuron_attributes.
        unit_idx (int): Index of the unit.
        channels (list[int]): Ordered channel list (defines output layout).
        template_len (int): Samples per channel slot.

    Returns:
        arr (np.ndarray): 1-D array of length len(channels) * template_len.
    """
    attrs = sd.neuron_attributes[unit_idx]
    avg_wf = attrs.get("avg_waveform")
    traces_meta = attrs.get("traces_meta", {})
    channel_list = [int(c) for c in traces_meta.get("channels", [])]
    ch_to_row = {ch: idx for idx, ch in enumerate(channel_list)}

    arr = np.zeros(len(channels) * template_len)
    if avg_wf is None:
        return arr
    for k, ch in enumerate(channels):
        row = ch_to_row.get(ch)
        if row is not None:
            wf = avg_wf[row, :]
            n = min(len(wf), template_len)
            arr[k * template_len : k * template_len + n] = wf[:n]
    return arr


def _merge_two_trains(train1, train2, delta_ms=0.4):
    """Merge two spike trains, removing duplicates within delta_ms.

    Parameters:
        train1 (np.ndarray): First spike train (ms).
        train2 (np.ndarray): Second spike train (ms).
        delta_ms (float): Deduplication window in ms.

    Returns:
        merged (np.ndarray): Sorted merged spike train.
        n_duplicates (int): Number of spikes removed as duplicates.
    """
    if len(train1) == 0 and len(train2) == 0:
        return np.array([]), 0
    if len(train1) == 0:
        return np.sort(train2), 0
    if len(train2) == 0:
        return np.sort(train1), 0

    times = np.concatenate([train1, train2])
    membership = np.concatenate(
        [np.zeros(len(train1), dtype=np.int8), np.ones(len(train2), dtype=np.int8)]
    )
    idx = np.argsort(times, kind="mergesort")
    times = times[idx]
    membership = membership[idx]

    diffs = np.diff(times)
    cross_train = np.diff(membership) != 0
    dup_mask = (diffs <= delta_ms) & cross_train

    keep = np.ones(len(times), dtype=bool)
    keep[1:][dup_mask] = False

    merged = times[keep]
    return merged, int(np.sum(~keep))


def _choose_primary_unit(sd, i, j):
    """Return (primary, secondary) based on spike count.

    The unit with the larger number of spikes is kept as primary.
    """
    spike_count_i = len(sd.train[i])
    spike_count_j = len(sd.train[j])
    return (i, j) if spike_count_i >= spike_count_j else (j, i)
