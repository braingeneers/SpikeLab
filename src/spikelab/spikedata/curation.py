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

from spikelab.spike_sorting._exceptions import EmptyWaveformMetricsError


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
        raise EmptyWaveformMetricsError(
            "raw_data is empty. Attach raw voltage traces before calling "
            "compute_waveform_metrics.",
            metric_name="waveform_metrics",
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
    raise EmptyWaveformMetricsError(
        f"Cannot compute '{metric_name}': no precomputed values in "
        "neuron_attributes and raw_data is empty. Call "
        "compute_waveform_metrics() first, or attach raw voltage traces.",
        metric_name=metric_name,
    )
