import warnings

import numpy as np

__all__ = ["SpikeSliceStack"]

from .pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
from .spikedata import SpikeData
from concurrent.futures import ThreadPoolExecutor

from .utils import (
    _validate_time_start_to_end,
    _get_attr,
    get_sttc,
    compute_cross_correlation_with_lag,
    _resolve_n_jobs,
    _slice_to_slice_similarity_matrix,
)


class SpikeSliceStack:
    """A list of SpikeData objects, one per slice, with spike-based comparison capabilities.

    U is units (neurons) and S is slices (bursts, events, etc). Construct from
    either a single SpikeData with time specifications, or directly from a
    pre-built list of SpikeData objects.

    Parameters:
        data_obj (SpikeData or None): A SpikeData object to slice. Provide
            either this or spike_stack, not both.
        times_start_to_end (list or None): Each entry is a tuple (start, end)
            representing the start and end times of a desired slice. Each
            tuple must have the same duration.
        time_peaks (list or None): List of times as int or float where there
            is a burst peak or stimulation event. Must be paired with
            time_bounds. Alternative to times_start_to_end.
        time_bounds (tuple or None): Single tuple (left_bound, right_bound).
            For example, (250, 500) means 250 ms before peak and 500 ms
            after peak. Must be paired with time_peaks.
        spike_stack (list or None): List of SpikeData objects, one per slice.
            All must have the same number of units. Spike times must be
            relative to the slice (0-based or event-centered via
            start_time), not absolute recording times. Provide either this
            or data_obj, not both.
        neuron_attributes (list or None): List of attribute dicts, one per
            unit. If None, inherited from data_obj when available.
        drop_slice_attributes (bool): If True (default), neuron_attributes
            are removed from individual SpikeData slices after construction.
            The shared copy is stored at neuron_attributes. This avoids
            duplicating large per-unit data (e.g. waveform templates) across
            every slice. Set to False to keep per-slice attributes.

    Attributes:
        spike_stack (list): List of SpikeData objects, one per slice. Spike
            times are relative to the slice window. For 0-based slices,
            times run from 0 to duration. For event-centered slices, times
            run from -pre_ms to +post_ms with t=0 at the event. Use
            self.times for absolute recording time positions.
        times (list): List of (start, end) time bounds for each slice in
            absolute recording time, sorted chronologically. Length equals S.
            Example: [(100, 350), (500, 750), (1000, 1250)].
        N (int): Number of units.
        neuron_attributes (list or None): List of attribute dicts, one per
            unit. None if not provided. When drop_slice_attributes is True
            (default), this is the only copy and individual slices will have
            neuron_attributes set to None.
    """

    def __init__(
        self,
        data_obj=None,
        times_start_to_end=None,
        time_peaks=None,
        time_bounds=None,
        spike_stack=None,
        neuron_attributes=None,
        drop_slice_attributes=True,
    ):
        if data_obj is None and spike_stack is None:
            raise TypeError(
                "Must input either a SpikeData as data_obj (option 1) or spike_stack (option 2)"
            )
        if data_obj is not None and spike_stack is not None:
            warnings.warn(
                "User input both data_obj and spike_stack. "
                "Ignoring data_obj and using spike_stack instead.",
                UserWarning,
            )
            data_obj = None

        # Option 1: Using data_obj
        if data_obj is not None:
            if not isinstance(data_obj, SpikeData):
                raise TypeError("data_obj must be a SpikeData object")

            if times_start_to_end is None:
                if time_peaks is None or time_bounds is None:
                    raise ValueError(
                        "Must provide either times_start_to_end or "
                        "both time_peaks and time_bounds"
                    )
                if not isinstance(time_bounds, tuple) or len(time_bounds) != 2:
                    raise TypeError(
                        "time_bounds must be a tuple of (before, after) durations"
                    )
                before, after = time_bounds
                time_peaks = sorted(time_peaks)
                times_start_to_end = []
                for t in time_peaks:
                    times_start_to_end.append((t - before, t + after))

            rec_range = (
                data_obj.start_time,
                data_obj.start_time + data_obj.length,
            )
            times_start_to_end = _validate_time_start_to_end(
                times_start_to_end, recording_range=rec_range
            )

            self.times = times_start_to_end
            self.spike_stack = []
            if time_peaks is not None:
                # Event-centered: shift_to=peak so t=0 is the event
                for peak, (start, end) in zip(time_peaks, times_start_to_end):
                    self.spike_stack.append(data_obj.subtime(start, end, shift_to=peak))
            else:
                # Standard: shift_to=start so t=0 is the window start
                for start, end in times_start_to_end:
                    self.spike_stack.append(data_obj.subtime(start, end))

            if neuron_attributes is None:
                neuron_attributes = data_obj.neuron_attributes

        # Option 2: Using spike_stack directly
        if spike_stack is not None:
            if not isinstance(spike_stack, list):
                raise TypeError("spike_stack must be a list of SpikeData objects")
            for s in spike_stack:
                if not isinstance(s, SpikeData):
                    raise TypeError("spike_stack must be a list of SpikeData objects")
            if len(spike_stack) == 0:
                raise ValueError("spike_stack must not be empty")

            N = spike_stack[0].N
            for s in spike_stack:
                if s.N != N:
                    raise ValueError(
                        "All SpikeData objects in spike_stack must have the same number of units"
                    )

            if times_start_to_end is None:
                t = 0.0
                times_start_to_end = []
                for s in spike_stack:
                    times_start_to_end.append((t, t + s.length))
                    t += s.length
            else:
                warn_neg = spike_stack[0].start_time >= 0
                times_start_to_end = _validate_time_start_to_end(
                    times_start_to_end, warn_negative_start=warn_neg
                )
                if len(times_start_to_end) != len(spike_stack):
                    raise ValueError(
                        "times_start_to_end must have the same length as spike_stack"
                    )

            self.spike_stack = list(spike_stack)
            self.times = times_start_to_end

            # Validate that all slices share the same ``start_time``
            # convention. Mixing 0-based slices (start_time=0) with
            # event-centered slices (start_time=-pre) — or two event-
            # centered stacks with different ``pre`` values — silently
            # mis-aligns downstream raster outputs. Require uniformity.
            if len(self.spike_stack) > 1:
                start_times = [sd.start_time for sd in self.spike_stack]
                if len(set(start_times)) > 1:
                    raise ValueError(
                        "All slices in spike_stack must share the same "
                        f"start_time convention; got {sorted(set(start_times))}. "
                        "Mixing 0-based and event-centered slices (or two "
                        "event-centered stacks with different pre-windows) "
                        "would silently mis-align downstream raster outputs."
                    )

            # Validate that spike times are consistent with the slice
            # duration. Spike times must be relative to the slice (0-based
            # or event-centered), not absolute recording times.
            for i, (sd, (start, end)) in enumerate(zip(self.spike_stack, self.times)):
                duration = end - start
                expected_start = sd.start_time
                expected_end = sd.start_time + duration
                for u, train in enumerate(sd.train):
                    if len(train) == 0:
                        continue
                    if train[0] < expected_start or train[-1] > expected_end:
                        raise ValueError(
                            f"Slice {i}, unit {u}: spike times "
                            f"[{train[0]:.1f}, {train[-1]:.1f}] ms fall outside "
                            f"expected range [{expected_start:.1f}, "
                            f"{expected_end:.1f}] ms. "
                            "Spike times must be relative to the slice (0-based "
                            "or event-centered), not absolute recording times."
                        )

        self.N = self.spike_stack[0].N

        self.neuron_attributes = None
        if neuron_attributes is not None:
            self.neuron_attributes = neuron_attributes.copy()
            if len(self.neuron_attributes) != self.N:
                raise ValueError(
                    f"neuron_attributes has {len(self.neuron_attributes)} items "
                    f"but spike_stack has {self.N} units"
                )

        # Strip per-slice neuron_attributes to avoid duplicating large data
        # (e.g. waveform templates) across every slice.
        if drop_slice_attributes:
            for sd in self.spike_stack:
                sd.neuron_attributes = None

    def __repr__(self) -> str:
        S = len(self.spike_stack)
        return f"SpikeSliceStack(N={self.N}, S={S})"

    def __len__(self) -> int:
        return len(self.spike_stack)

    def __iter__(self):
        return iter(self.spike_stack)

    def subslice(self, slices):
        """Extract a subset of slices from the spike stack.

        Parameters:
            slices (int or list): Slice index or list of slice indices to
                extract.

        Returns:
            result (SpikeSliceStack): New SpikeSliceStack containing only the
                specified slices. Shape changes from S to S_trimmed. All
                units and neuron_attributes are carried over.
        """
        S = len(self.spike_stack)
        if isinstance(slices, int):
            slices = [slices]
        for s in slices:
            if s >= S or s < -S:
                raise ValueError(f"One or more slice indices out of range for S={S}")
        slices = sorted(slices)
        new_spike_stack = []
        new_times = []
        for s in slices:
            new_spike_stack.append(self.spike_stack[s])
            new_times.append(self.times[s])
        return SpikeSliceStack(
            spike_stack=new_spike_stack,
            times_start_to_end=new_times,
            neuron_attributes=self.neuron_attributes,
        )

    def subset(self, units, by=None, preserve_order=False):
        """Extract a subset of units from every slice in the spike stack.

        Parameters:
            units (int, str, or list): Unit indices to extract. If by is None,
                must be int(s). If by is set, values to match in
                neuron_attributes.
            by (str or None): If set, select units by this neuron_attribute
                key instead of by index.
            preserve_order (bool): When False (default), output is
                sorted ascending by index — consistent with the other
                SpikeLab data classes. When True, output respects the
                order of the input ``units`` list. Duplicates are
                deduplicated either way.

        Returns:
            result (SpikeSliceStack): New SpikeSliceStack containing only the
                specified units across all slices. All slices and
                neuron_attributes are carried over.

        Notes:
            - If IDs are not unique (when using by), every matching neuron is
              included.
        """
        if isinstance(units, (int, str)):
            units = [units]

        # Resolve which indices will be kept so we can update neuron_attributes
        if by is not None:
            # ``by`` resolves to whichever units carry the matching
            # attribute, in self.train order — caller-supplied order
            # cannot be honoured because the value list has no
            # positional correspondence to unit indices.
            if self.neuron_attributes is None:
                raise ValueError("can't use `by` without `neuron_attributes`")
            _missing = object()
            unit_set = set(units)
            kept_indices = []
            for i in range(self.N):
                if _get_attr(self.neuron_attributes[i], by, _missing) in unit_set:
                    kept_indices.append(i)
        else:
            for u in units:
                ui = int(u)
                if ui < 0 or ui >= self.N:
                    raise ValueError(
                        f"Unit index {ui} out of range for {self.N} units."
                    )
            if preserve_order:
                seen: set = set()
                ordered = []
                for u in units:
                    ui = int(u)
                    if ui not in seen:
                        seen.add(ui)
                        ordered.append(ui)
                kept_indices = ordered
            else:
                kept_indices = sorted({int(u) for u in units})

        new_spike_stack = []
        for sd in self.spike_stack:
            # Forward preserve_order so per-slice subsets agree with
            # the SpikeSliceStack-level ordering decision.
            new_spike_stack.append(sd.subset(kept_indices, preserve_order=True))

        new_neuron_attributes = None
        if self.neuron_attributes is not None:
            new_neuron_attributes = []
            for i in kept_indices:
                new_neuron_attributes.append(self.neuron_attributes[i])

        return SpikeSliceStack(
            spike_stack=new_spike_stack,
            times_start_to_end=self.times,
            neuron_attributes=new_neuron_attributes,
        )

    def subtime_by_index(self, start_idx, end_idx):
        """Trim each slice to a sub-window specified by millisecond indices.

        Indices are measured from the start of each slice (1 index = 1 ms).
        Trims along the time axis while preserving all slices and units.

        Parameters:
            start_idx (int): Start index in ms from slice start (inclusive).
                Supports negative indexing.
            end_idx (int): End index in ms from slice start (exclusive).
                Supports negative indexing.

        Returns:
            result (SpikeSliceStack): New SpikeSliceStack where each slice is
                trimmed to the corresponding absolute time window. Absolute
                spike times are preserved (not shifted). self.times is updated
                to reflect the new absolute time bounds.

        Raises:
            ValueError: If the underlying slice duration (``times[0][1] -
                times[0][0]``) is not an integer number of milliseconds.
                Use ``SpikeData.subtime()`` with explicit ms bounds for
                non-integer windows.

        Notes:
            - Indices are relative to each slice's own start (index 0 = slice
              start ms). They are converted to absolute recording times
              internally before trimming.
            - Original absolute timestamps are preserved. To get
              shifted-to-zero timestamps, create a new SpikeSliceStack.
            - All slices and neuron_attributes are carried over from the
              original.
        """
        slice_duration_ms = self.times[0][1] - self.times[0][0]
        # 1 index = 1 ms; non-integer durations would silently drop the
        # sub-ms tail. Push the rounding decision back to the caller.
        if abs(slice_duration_ms - round(slice_duration_ms)) > 1e-9:
            raise ValueError(
                f"slice_duration_ms ({slice_duration_ms}) must be an "
                f"integer number of milliseconds for subtime_by_index "
                f"(1 index = 1 ms). For non-integer windows, call "
                f"SpikeData.subtime() with explicit ms bounds, or "
                f"reconstruct the SpikeSliceStack with an integer slice "
                f"duration."
            )
        T = int(round(slice_duration_ms))

        if start_idx < 0:
            start_idx += T
        if end_idx < 0:
            end_idx += T
        if start_idx < 0 or start_idx >= T:
            raise ValueError(f"start_idx {start_idx} out of range for T={T}")
        if end_idx <= start_idx or end_idx > T:
            raise ValueError(f"end_idx {end_idx} invalid for T={T}")

        new_spike_stack = []
        new_times = []
        for sd, t in zip(self.spike_stack, self.times):
            new_spike_stack.append(
                sd.subtime(
                    sd.start_time + float(start_idx), sd.start_time + float(end_idx)
                )
            )
            abs_start = t[0] + float(start_idx)
            abs_end = t[0] + float(end_idx)
            new_times.append((abs_start, abs_end))

        return SpikeSliceStack(
            spike_stack=new_spike_stack,
            times_start_to_end=new_times,
            neuron_attributes=self.neuron_attributes,
        )

    def to_raster_array(self, bin_size=1.0, absolute_times=False):
        """Convert the spike stack into a 3D raster array of shape (N, T, S).

        Each slice is rasterized with the given bin size, producing a spike
        count matrix where entry (n, t, s) is the number of spikes unit n
        fired in time bin t of slice s.

        Parameters:
            bin_size (float): Time bin size in ms (default 1.0).
            absolute_times (bool): If False (default), time bin 0 corresponds
                to the start of each slice (0-based). If True, each slice's
                spikes are offset by its absolute start time from self.times,
                so bin indices reflect the original recording position. The T
                dimension is sized to cover the full time span from the
                earliest slice start to the latest slice end. **Caution:**
                this can produce very large arrays when the recording span is
                long and bin_size is small.

        Returns:
            raster_stack (np.ndarray): 3D array of shape (N, T, S) with
                non-negative integer spike counts. When absolute_times is
                True, T covers the full recording span and all slices share
                the same time axis.
        """
        if bin_size <= 0:
            raise ValueError(f"bin_size must be > 0, got {bin_size}")

        if not absolute_times:
            dense_list = []
            for sd in self.spike_stack:
                # Spike times are relative to each slice (0-based or event-centered).
                # sparse_raster handles start_time internally.
                dense_list.append(sd.sparse_raster(bin_size=bin_size).toarray())
            return np.stack(dense_list, axis=2)

        # Absolute times: offset each slice by its start time so bin indices
        # reflect original recording position. All slices share the same
        # time axis spanning [min(start), max(end)].
        global_start = min(start for start, _ in self.times)
        global_end = max(end for _, end in self.times)
        total_bins = int(np.ceil((global_end - global_start) / bin_size))

        raster_stack = np.zeros((self.N, total_bins, len(self.spike_stack)), dtype=int)
        for s_idx, (sd, (start, _)) in enumerate(zip(self.spike_stack, self.times)):
            offset = start - global_start
            r = sd.sparse_raster(bin_size=bin_size, time_offset=offset).toarray()
            # Clamp r.shape[1] to total_bins. Both shapes come from
            # independent np.ceil calls on float arithmetic, so a
            # ULP-level difference between (global_end - global_start)
            # and (slice_length + offset) can leave r one bin larger
            # than the buffer — the unclamped assignment would raise
            # a broadcasting error. The reverse case (r smaller than
            # total_bins) is benign — trailing bins stay zero.
            n = min(r.shape[1], total_bins)
            raster_stack[:, :n, s_idx] = r[:, :n]

        return raster_stack

    def baseline_normalized_raster(
        self, bin_size, baseline_window_ms, *, mode="subtract"
    ):
        """Per-slice raster normalized against a per-slice baseline rate.

        Wraps ``to_raster_array(bin_size)`` and converts each bin into a
        baseline-normalized response value. The baseline rate is computed
        from spikes inside ``baseline_window_ms`` (in milliseconds relative
        to each slice's time origin) and projected to each bin via
        ``rate * bin_size``. Output shape matches the raster: ``(U, T, S)``.

        Parameters:
            bin_size (float): Raster bin size in milliseconds. Passed to
                ``to_raster_array``.
            baseline_window_ms (tuple[float, float]): ``(start_ms, end_ms)``
                window relative to slice origin used to estimate the
                per-slice baseline rate.
            mode (str): Normalization mode:

                - ``"subtract"`` (default) — counts above baseline expectation.
                - ``"ratio"`` — counts / expected_counts (NaN where expected
                  is 0).
                - ``"zscore"`` — (counts - expected) / sqrt(expected), the
                  Poisson z-score (NaN where expected is 0).

        Returns:
            normalized (np.ndarray): Float array of shape ``(U, T, S)``.

        Notes:
            - Baseline window is validated against each slice's actual time
              range. ``ValueError`` if any slice doesn't contain it.
            - For uniform-bin response counts (no normalization), use
              ``to_raster_array(bin_size)`` directly; this method adds the
              per-slice baseline correction on top.
        """
        if mode not in ("subtract", "ratio", "zscore"):
            raise ValueError(
                f"mode must be 'subtract', 'ratio', or 'zscore', got {mode!r}"
            )
        if (
            not isinstance(baseline_window_ms, (tuple, list))
            or len(baseline_window_ms) != 2
        ):
            raise ValueError("baseline_window_ms must be a (start_ms, end_ms) tuple.")
        b_start, b_end = float(baseline_window_ms[0]), float(baseline_window_ms[1])
        if b_end <= b_start:
            raise ValueError("baseline_window_ms end must be greater than start.")

        for s_idx, sd in enumerate(self.spike_stack):
            slice_start = sd.start_time
            slice_end = sd.start_time + (self.times[s_idx][1] - self.times[s_idx][0])
            if b_start < slice_start - 1e-9 or b_end > slice_end + 1e-9:
                raise ValueError(
                    f"baseline_window_ms ({b_start}, {b_end}) falls outside "
                    f"slice {s_idx} time range [{slice_start}, {slice_end}]."
                )

        counts = self.to_raster_array(bin_size=bin_size).astype(float)  # (U, T, S)

        baseline_width = b_end - b_start
        baseline_counts = np.zeros((self.N, len(self.spike_stack)), dtype=float)
        for s_idx, sd in enumerate(self.spike_stack):
            for u in range(self.N):
                train = np.asarray(sd.train[u], dtype=float)
                if train.size == 0:
                    continue
                baseline_counts[u, s_idx] = float(
                    np.sum((train >= b_start) & (train < b_end))
                )

        # Per-slice expected counts per bin = rate * bin_size
        expected_per_bin = baseline_counts * (bin_size / baseline_width)  # (U, S)
        expected = expected_per_bin[:, np.newaxis, :]  # broadcasts over T

        if mode == "subtract":
            return counts - expected
        if mode == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(expected > 0, counts / expected, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                expected > 0, (counts - expected) / np.sqrt(expected), np.nan
            )

    def responsive_units(
        self,
        bin_size,
        baseline_window_ms,
        *,
        response_window_ms=None,
        z_threshold=2.0,
        aggregator="mean",
    ):
        """Identify units that show a significant evoked response.

        Builds the Poisson-z-scored baseline-normalized raster, optionally
        restricts to a response time window, aggregates across slices
        (mean or max), and returns a unit mask where any time bin's
        aggregated z-score exceeds ``z_threshold``.

        Parameters:
            bin_size (float): Raster bin size in milliseconds.
            baseline_window_ms (tuple[float, float]): Baseline window
                ``(start_ms, end_ms)`` relative to slice origin used to
                estimate the per-slice baseline rate.
            response_window_ms (tuple[float, float] or None): Optional
                response window ``(start_ms, end_ms)`` relative to slice
                origin. When None (default), the full slice is searched.
            z_threshold (float): Z-score threshold (default 2.0).
            aggregator (str): How to combine z-scores across slices before
                thresholding. ``"mean"`` (default) or ``"max"``.

        Returns:
            mask (np.ndarray): Boolean array of shape ``(U,)``. True for
                responsive units.

        Notes:
            - Units with no baseline spikes in any slice are flagged
              non-responsive (z-scores are NaN there).
        """
        if aggregator not in ("mean", "max"):
            raise ValueError(f"aggregator must be 'mean' or 'max', got {aggregator!r}")
        z = self.baseline_normalized_raster(
            bin_size, baseline_window_ms, mode="zscore"
        )  # (U, T, S)

        if response_window_ms is not None:
            if (
                not isinstance(response_window_ms, (tuple, list))
                or len(response_window_ms) != 2
            ):
                raise ValueError(
                    "response_window_ms must be a (start_ms, end_ms) tuple or None."
                )
            r_start = float(response_window_ms[0])
            r_end = float(response_window_ms[1])
            if r_end <= r_start:
                raise ValueError("response_window_ms end must be greater than start.")
            sd0 = self.spike_stack[0]
            bin_start = int(np.floor((r_start - sd0.start_time) / bin_size))
            bin_end = int(np.ceil((r_end - sd0.start_time) / bin_size))
            bin_start = max(0, bin_start)
            bin_end = min(z.shape[1], bin_end)
            if bin_end <= bin_start:
                raise ValueError(
                    f"response_window_ms ({r_start}, {r_end}) maps to an empty "
                    f"bin range; check it against the slice duration and bin_size."
                )
            z = z[:, bin_start:bin_end, :]

        if aggregator == "mean":
            agg = np.nanmean(z, axis=2)
        else:
            agg = np.nanmax(z, axis=2)
        with np.errstate(invalid="ignore"):
            return np.any(agg > z_threshold, axis=1)

    def decode_slice_labels(
        self,
        labels,
        response_window_ms,
        *,
        bin_size,
        baseline_window_ms=None,
        classifier="ridge",
        cv="loo",
        classifier_kwargs=None,
        random_state=None,
    ):
        """Decode per-slice labels (e.g. stim identity) from population responses.

        Builds an ``(S, U)`` feature matrix by summing per-unit spike counts
        in ``response_window_ms`` (optionally with baseline subtraction) and
        runs cross-validated classifier decoding via
        :func:`spikelab.spikedata.decoding.cross_validated_decode`.

        Parameters:
            labels (array-like): Per-slice labels of length ``S``
                (e.g. stim electrode index, treatment category).
            response_window_ms (tuple[float, float]): Window relative to
                slice origin over which response counts are summed.
            bin_size (float): Raster bin size in ms.
            baseline_window_ms (tuple[float, float] or None): Optional
                per-slice baseline window; when provided, counts are
                baseline-subtracted via ``baseline_normalized_raster``.
            classifier (str): ``"ridge"`` (default), ``"mlp"``, or
                ``"random_forest"``.
            cv (str or int): ``"loo"`` (default) or int ``>= 2``.
            classifier_kwargs (dict or None): Forwarded to the sklearn
                classifier constructor.
            random_state (int or None): Reproducibility seed.

        Returns:
            result (dict): Same shape as
                :func:`spikelab.spikedata.decoding.cross_validated_decode` —
                ``accuracy``, ``predictions``, ``true_labels``,
                ``confusion_matrix``, ``per_fold_accuracy``, ``classes``,
                ``classifier_name``.

        Notes:
            - Requires ``scikit-learn`` (optional dependency).
            - For decoding from the full ``(U, T)`` raster (not just summed
              counts), call ``decoding.cross_validated_decode`` directly on
              ``self.to_raster_array(bin_size).reshape(U * T, S).T``.
        """
        from .decoding import cross_validated_decode

        if (
            not isinstance(response_window_ms, (tuple, list))
            or len(response_window_ms) != 2
        ):
            raise ValueError("response_window_ms must be a (start_ms, end_ms) tuple.")
        r_start = float(response_window_ms[0])
        r_end = float(response_window_ms[1])
        if r_end <= r_start:
            raise ValueError("response_window_ms end must be greater than start.")

        if baseline_window_ms is None:
            counts = self.to_raster_array(bin_size=bin_size).astype(float)
        else:
            counts = self.baseline_normalized_raster(
                bin_size, baseline_window_ms, mode="subtract"
            )

        sd0 = self.spike_stack[0]
        bin_start = int(np.floor((r_start - sd0.start_time) / bin_size))
        bin_end = int(np.ceil((r_end - sd0.start_time) / bin_size))
        bin_start = max(0, bin_start)
        bin_end = min(counts.shape[1], bin_end)
        if bin_end <= bin_start:
            raise ValueError(
                f"response_window_ms ({r_start}, {r_end}) maps to an empty "
                f"bin range; check it against the slice duration and bin_size."
            )

        # (U, S) per-unit summed response amplitude per slice -> (S, U) features
        X = counts[:, bin_start:bin_end, :].sum(axis=1).T

        labels = np.asarray(labels).ravel()
        if len(labels) != X.shape[0]:
            raise ValueError(
                f"labels must have length S={X.shape[0]}; got {len(labels)}."
            )

        return cross_validated_decode(
            X,
            labels,
            classifier=classifier,
            cv=cv,
            classifier_kwargs=classifier_kwargs,
            random_state=random_state,
        )

    def group_pair_similarity(
        self,
        stim_labels,
        *,
        metric="cosine",
        bin_size=1.0,
        slice_indices=None,
    ):
        """Pairwise similarity between mean response vectors for each stim class.

        For each unique stimulus label, averages the per-slice
        ``(U, T)`` raster across all slices that share that label, then
        computes a ``(K, K)`` similarity matrix between the resulting
        per-stim mean response vectors. Lets you ask: "how distinguishable
        are responses to different stims?".

        Parameters:
            stim_labels (array-like): Per-slice stim label of length ``S``.
            metric (str): ``"cosine"`` (default), ``"pearson"``,
                ``"euclidean"`` (distance), or ``"cross_entropy"``.
            bin_size (float): Raster bin size in ms (default 1.0).
            slice_indices (array-like or None): Optional subset of slice
                indices to use (e.g. an "early-cycle" or "late-cycle"
                window). When None (default), uses all slices.

        Returns:
            sim (PairwiseCompMatrix): ``(K, K)`` similarity matrix with
                ``labels`` set to the unique stim classes (sorted).

        Notes:
            - Stim classes that have no slices in ``slice_indices`` are
              dropped from the output.
        """
        from .utils import _slice_to_slice_similarity_matrix

        stim_labels = np.asarray(stim_labels).ravel()
        if len(stim_labels) != len(self):
            raise ValueError(
                f"stim_labels must have length S={len(self)}; got {len(stim_labels)}."
            )

        if slice_indices is None:
            slice_indices = np.arange(len(self))
        else:
            slice_indices = np.asarray(slice_indices, dtype=int).ravel()
            if (slice_indices < 0).any() or (slice_indices >= len(self)).any():
                raise IndexError(f"slice_indices out of range for S={len(self)}.")

        unique_labels = np.array(sorted(np.unique(stim_labels[slice_indices])))
        if len(unique_labels) < 2:
            raise ValueError(
                "Need at least 2 distinct stim classes in the selected slices."
            )

        # (U, T, S_subset)
        raster = self.subslice(list(slice_indices)).to_raster_array(bin_size=bin_size)
        sub_labels = stim_labels[slice_indices]

        # Per-class mean across slices -> (U, T, K)
        mean_per_class = np.stack(
            [raster[:, :, sub_labels == cls].mean(axis=2) for cls in unique_labels],
            axis=2,
        )
        sim = _slice_to_slice_similarity_matrix(mean_per_class, metric)
        return PairwiseCompMatrix(
            matrix=sim,
            labels=list(unique_labels),
            metadata={"metric": metric, "n_classes": len(unique_labels)},
        )

    def responsive_units_per_group(
        self,
        group_labels,
        bin_size,
        baseline_window_ms,
        *,
        response_window_ms=None,
        z_threshold=2.0,
        aggregator="mean",
    ):
        """Per-cycle responsive-unit mask for tracking responsiveness over time.

        For each unique cycle, runs ``responsive_units`` on the slices
        belonging to that cycle and returns a ``(U, n_cycles)`` boolean
        matrix. Use the per-cycle masks to compute gained / lost /
        preserved responsive units across cycle groups, or to correlate
        responsiveness changes with intrinsic activity changes per unit.

        Parameters:
            group_labels (array-like): Per-slice cycle index of length ``S``.
            bin_size (float): Raster bin size in ms.
            baseline_window_ms (tuple[float, float]): Baseline window for
                Poisson z-score normalization.
            response_window_ms (tuple[float, float] or None): Optional
                response window (default: full slice).
            z_threshold (float): Per-cycle z-threshold (default 2.0).
            aggregator (str): ``"mean"`` (default) or ``"max"`` across
                slices within each cycle.

        Returns:
            result (dict):
                - ``cycles`` (np.ndarray): Sorted unique cycle indices.
                - ``mask`` (np.ndarray): ``(U, n_cycles)`` boolean
                  responsiveness mask.
                - ``responsive_count`` (np.ndarray): Per-cycle responsive
                  unit count, shape ``(n_cycles,)``.
        """
        group_labels = np.asarray(group_labels).ravel()
        if len(group_labels) != len(self):
            raise ValueError(
                f"group_labels must have length S={len(self)}; got {len(group_labels)}."
            )

        groups = np.array(sorted(np.unique(group_labels)))
        mask = np.zeros((self.N, len(groups)), dtype=bool)
        for j, c in enumerate(groups):
            slice_idx = np.where(group_labels == c)[0]
            if slice_idx.size == 0:
                continue
            sub = self.subslice(slice_idx.tolist())
            mask[:, j] = sub.responsive_units(
                bin_size=bin_size,
                baseline_window_ms=baseline_window_ms,
                response_window_ms=response_window_ms,
                z_threshold=z_threshold,
                aggregator=aggregator,
            )
        return {
            "groups": groups,
            "mask": mask,
            "responsive_count": mask.sum(axis=0),
        }

    def responsiveness_change(
        self,
        group_labels,
        early_groups,
        late_groups,
        bin_size,
        baseline_window_ms,
        *,
        response_window_ms=None,
        z_threshold=2.0,
        aggregator="mean",
    ):
        """Gained / lost / preserved responsive units between two cycle groups.

        Computes responsive-unit masks separately for slices in
        ``early_groups`` and ``late_groups`` (any iterables of cycle
        indices), and reports which units become responsive ("gained"),
        stop being responsive ("lost"), or stay responsive ("preserved").

        Parameters:
            group_labels (array-like): Per-slice cycle index of length ``S``.
            early_groups (array-like): Cycle indices for the early group.
            late_groups (array-like): Cycle indices for the late group.
            bin_size (float): Raster bin size in ms.
            baseline_window_ms (tuple[float, float]): Baseline window.
            response_window_ms (tuple[float, float] or None): Response
                window (default: full slice).
            z_threshold (float): Per-group z-threshold.
            aggregator (str): ``"mean"`` (default) or ``"max"``.

        Returns:
            result (dict):
                - ``early_mask`` (np.ndarray ``(U,)`` bool): Responsive in
                  early.
                - ``late_mask`` (np.ndarray ``(U,)`` bool): Responsive in
                  late.
                - ``gained`` (np.ndarray ``(U,)`` bool): NOT responsive
                  in early AND responsive in late.
                - ``lost`` (np.ndarray ``(U,)`` bool): Responsive in
                  early AND NOT responsive in late.
                - ``preserved`` (np.ndarray ``(U,)`` bool): Responsive
                  in BOTH.
                - ``early_count``, ``late_count``, ``gained_count``,
                  ``lost_count``, ``preserved_count`` (int).

        Notes:
            - Pair this with intrinsic-activity changes per unit (e.g.
              ``cv_isi`` differences) and correlate via
              ``stat_utils.linear_regression`` to ask whether
              responsiveness changes track changes in baseline activity.
        """
        group_labels = np.asarray(group_labels).ravel()
        if len(group_labels) != len(self):
            raise ValueError(
                f"group_labels must have length S={len(self)}; got {len(group_labels)}."
            )
        early_groups = np.asarray(early_groups).ravel()
        late_groups = np.asarray(late_groups).ravel()

        early_idx = np.where(np.isin(group_labels, early_groups))[0]
        late_idx = np.where(np.isin(group_labels, late_groups))[0]
        if early_idx.size == 0:
            raise ValueError("No slices match early_groups.")
        if late_idx.size == 0:
            raise ValueError("No slices match late_groups.")

        kwargs = dict(
            bin_size=bin_size,
            baseline_window_ms=baseline_window_ms,
            response_window_ms=response_window_ms,
            z_threshold=z_threshold,
            aggregator=aggregator,
        )
        early_mask = self.subslice(early_idx.tolist()).responsive_units(**kwargs)
        late_mask = self.subslice(late_idx.tolist()).responsive_units(**kwargs)

        gained = (~early_mask) & late_mask
        lost = early_mask & (~late_mask)
        preserved = early_mask & late_mask

        return {
            "early_mask": early_mask,
            "late_mask": late_mask,
            "gained": gained,
            "lost": lost,
            "preserved": preserved,
            "early_count": int(early_mask.sum()),
            "late_count": int(late_mask.sum()),
            "gained_count": int(gained.sum()),
            "lost_count": int(lost.sum()),
            "preserved_count": int(preserved.sum()),
        }

    def slice_to_slice_similarity(self, metric="cosine", *, bin_size=1.0):
        """Pairwise similarity between slice-wise population response vectors.

        Each slice is converted to a ``(U * T)`` flat vector via
        ``to_raster_array(bin_size).reshape(U*T, S).T`` and a square
        ``(S, S)`` similarity matrix is computed using the requested metric.

        Parameters:
            metric (str): One of ``"cosine"`` (default), ``"pearson"``,
                ``"euclidean"`` (distance), or ``"cross_entropy"``
                (symmetric KL on normalized bin distributions).
            bin_size (float): Raster bin size in ms (default 1.0).

        Returns:
            sim (PairwiseCompMatrix): ``(S, S)`` similarity matrix. For
                cosine and pearson, higher = more similar (diagonal ~1.0);
                for euclidean and cross_entropy, lower = more similar
                (diagonal 0).

        Notes:
            - ``cosine`` and ``pearson`` return values in ``[-1, 1]``.
            - ``euclidean`` returns raw L2 distance.
            - ``cross_entropy`` returns symmetric KL divergence (i.e.
              ``(KL(p||q) + KL(q||p)) / 2``) between bin distributions
              normalized to sum to 1.
            - Use ``PairwiseCompMatrix.extract_lower_triangle()`` for
              feature extraction.
        """
        stack = self.to_raster_array(bin_size=bin_size)  # (U, T, S)
        sim = _slice_to_slice_similarity_matrix(stack, metric)
        return PairwiseCompMatrix(matrix=sim, metadata={"metric": metric})

    def per_unit_response_regression(
        self,
        bin_size,
        response_window_ms,
        *,
        x_values=None,
        baseline_window_ms=None,
        min_valid_slices=3,
    ):
        """Per-unit OLS regression of evoked response amplitude across slices.

        For each slice and unit, computes response amplitude as the sum of
        spike counts in ``response_window_ms`` — optionally with a per-slice
        baseline subtraction. Then fits a linear regression of amplitude
        against ``x_values`` for every unit. Use this to detect facilitation
        / depression of the evoked response across cycles or stimulus
        intensities.

        Parameters:
            bin_size (float): Raster bin size in milliseconds (passed to
                ``to_raster_array``).
            response_window_ms (tuple[float, float]): ``(start_ms, end_ms)``
                window relative to slice origin over which response counts
                are summed.
            x_values (array-like or None): Per-slice x values for the
                regression (e.g. cycle index, stimulus intensity). Length
                must equal the number of slices ``S``. When None (default),
                uses ``np.arange(S)``.
            baseline_window_ms (tuple[float, float] or None): Optional
                baseline window for per-slice subtraction. When None
                (default), uses raw response counts; otherwise subtracts the
                expected count per bin (``baseline_rate * bin_size``) before
                summing.
            min_valid_slices (int): Minimum number of valid (non-NaN)
                ``(x, y)`` pairs required to fit a regression. Units with
                fewer return NaN for all coefficients. Default 3.

        Returns:
            result (dict): Dictionary with keys:
                - ``slope`` (np.ndarray ``(U,)``): Slope per unit.
                - ``intercept`` (np.ndarray ``(U,)``): Intercept per unit.
                - ``r_squared`` (np.ndarray ``(U,)``): R² per unit.
                - ``p_value`` (np.ndarray ``(U,)``): Two-sided p-value of
                  the slope per unit.
                - ``stderr`` (np.ndarray ``(U,)``): Standard error of the
                  slope per unit.
                - ``amplitudes`` (np.ndarray ``(U, S)``): Per-slice response
                  amplitude (raw or baseline-subtracted).
                - ``x_values`` (np.ndarray ``(S,)``): The x values used.

        Notes:
            - Requires ``scipy`` (optional dependency); raises
              ``ImportError`` with installation instructions if missing.
            - Units with constant amplitudes get ``r_squared = 0``,
              ``slope = 0`` and ``p_value = 1.0``.
        """
        try:
            from scipy import stats as sp_stats
        except ImportError as e:
            raise ImportError(
                "per_unit_response_regression requires 'scipy'. "
                "Install with: pip install scipy"
            ) from e

        if (
            not isinstance(response_window_ms, (tuple, list))
            or len(response_window_ms) != 2
        ):
            raise ValueError("response_window_ms must be a (start_ms, end_ms) tuple.")
        r_start = float(response_window_ms[0])
        r_end = float(response_window_ms[1])
        if r_end <= r_start:
            raise ValueError("response_window_ms end must be greater than start.")

        S = len(self.spike_stack)
        if x_values is None:
            x_values = np.arange(S, dtype=float)
        else:
            x_values = np.asarray(x_values, dtype=float).ravel()
            if x_values.size != S:
                raise ValueError(
                    f"x_values must have length S={S}, got {x_values.size}."
                )

        if baseline_window_ms is None:
            counts = self.to_raster_array(bin_size=bin_size).astype(float)  # (U, T, S)
        else:
            counts = self.baseline_normalized_raster(
                bin_size, baseline_window_ms, mode="subtract"
            )

        sd0 = self.spike_stack[0]
        bin_start = int(np.floor((r_start - sd0.start_time) / bin_size))
        bin_end = int(np.ceil((r_end - sd0.start_time) / bin_size))
        bin_start = max(0, bin_start)
        bin_end = min(counts.shape[1], bin_end)
        if bin_end <= bin_start:
            raise ValueError(
                f"response_window_ms ({r_start}, {r_end}) maps to an empty "
                f"bin range; check it against the slice duration and bin_size."
            )

        amplitudes = np.nansum(counts[:, bin_start:bin_end, :], axis=1)  # (U, S)

        slope = np.full(self.N, np.nan)
        intercept = np.full(self.N, np.nan)
        r_squared = np.full(self.N, np.nan)
        p_value = np.full(self.N, np.nan)
        stderr = np.full(self.N, np.nan)

        for u in range(self.N):
            y = amplitudes[u, :]
            valid = np.isfinite(x_values) & np.isfinite(y)
            if int(np.sum(valid)) < min_valid_slices:
                continue
            xv = x_values[valid]
            yv = y[valid]
            # Constant predictor or response: linregress would emit a warning
            # and return NaN; handle explicitly.
            if np.ptp(xv) == 0:
                continue
            try:
                res = sp_stats.linregress(xv, yv)
            except (ValueError, FloatingPointError):
                continue
            slope[u] = float(res.slope)
            intercept[u] = float(res.intercept)
            r_squared[u] = float(res.rvalue) ** 2
            p_value[u] = float(res.pvalue)
            stderr[u] = float(res.stderr)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "p_value": p_value,
            "stderr": stderr,
            "amplitudes": amplitudes,
            "x_values": x_values,
        }

    def compute_frac_active(self, min_spikes=2):
        """Compute the fraction of slices each unit is active in.

        A unit counts as active in a slice if it has at least min_spikes
        spikes within that slice's time window.

        Parameters:
            min_spikes (int): Minimum number of spikes for a unit to count as
                active in a slice (default: 2).

        Returns:
            frac_active (np.ndarray): 1-D array of shape ``(U,)`` with the
                fraction of slices each unit is active in (values in [0, 1]).

        Notes:
            - The returned array can be passed as ``frac_active`` to
              ``RateSliceStack.order_units_across_slices``,
              ``RateSliceStack.get_slice_to_slice_unit_corr_from_stack``,
              ``SpikeSliceStack.order_units_across_slices``, or
              ``SpikeSliceStack.get_slice_to_slice_unit_comparison``
              to override their internal activity calculation.
            - ``SpikeData.get_frac_active`` produces a compatible ``(U,)``
              array based on burst edges and can be used in the same way.
        """
        num_units = self.N
        num_slices = len(self.spike_stack)
        active_count = np.zeros(num_units, dtype=int)

        for sd, (start, end) in zip(self.spike_stack, self.times):
            for u in range(num_units):
                spikes = np.asarray(sd.train[u])
                n_valid = np.sum(
                    (spikes >= sd.start_time) & (spikes < sd.start_time + (end - start))
                )
                if n_valid >= min_spikes:
                    active_count[u] += 1

        return active_count / num_slices if num_slices > 0 else np.zeros(num_units)

    def order_units_across_slices(
        self,
        agg_func="median",
        timing="median",
        min_spikes=2,
        min_frac_active=0.0,
        frac_active=None,
        timing_matrix=None,
    ):
        """Reorder units by their typical spike timing across slices.

        For each unit in each slice, computes a representative spike time
        (median, mean, or first spike) relative to the slice's time origin. These
        per-slice values are aggregated across slices to obtain a single
        typical timing per unit. Units are then sorted by this value from
        earliest to latest and optionally split into a highly-active group
        and a low-activity group.

        Parameters:
            agg_func (str): How to aggregate per-slice timing values across
                slices. ``"median"`` (default) or ``"mean"``.
            timing (str): Which spike time to extract per unit per slice.
                ``"median"`` — median spike time within the slice (default).
                ``"mean"`` — mean spike time within the slice.
                ``"first"`` — first spike time (onset latency).
                Ignored when ``timing_matrix`` is provided.
            min_spikes (int): Minimum number of spikes for a unit to count as
                active in a slice (default: 2). Ignored when ``timing_matrix``
                is provided.
            min_frac_active (float or None): Minimum fraction of slices a unit
                must be active in to be placed in the highly-active group.
                ``0.0`` or ``None`` (default: 0.0) skips the split entirely
                and places all units in the highly-active group without
                computing activity fractions.
            frac_active (np.ndarray or None): Optional pre-computed
                fraction-active array of shape ``(U,)`` to override the
                internal calculation for the group split. Only used when
                ``min_frac_active > 0``. Compatible sources:
                ``SpikeSliceStack.compute_frac_active`` and
                ``SpikeData.get_frac_active`` (``frac_per_unit`` output).
            timing_matrix (np.ndarray or None): Optional pre-computed ``(U, S)``
                timing matrix from ``get_unit_timing_per_slice``. When provided,
                ``timing`` and ``min_spikes`` are ignored and this matrix is
                used directly.

        Returns:
            reordered_stacks (tuple): Two ``SpikeSliceStack`` objects
                ``(highly_active, low_active)`` with units reordered by typical
                timing. The low-activity stack is ``None`` when the group is
                empty.
            unit_ids_in_order (tuple): Two ``ndarray``
                ``(highly_active, low_active)`` of original unit indices in the
                reordered sequence.
            unit_std (tuple): Two ``ndarray`` ``(highly_active, low_active)``
                of standard deviation of per-slice timing values. Lower values
                indicate more consistent timing across slices.
            unit_peak_times_ms (tuple): Two ``ndarray``
                ``(highly_active, low_active)`` of the aggregated typical
                timing in milliseconds relative to slice start. NaN for units
                with no active slices.
            unit_frac_active (tuple): Two ``ndarray``
                ``(highly_active, low_active)`` of the fraction of slices each
                unit was active in.

        Notes:
            - Call ``get_unit_timing_per_slice`` first to pre-compute the
              timing matrix if you want to reuse it across multiple calls
              (e.g. ``rank_order_correlation`` and this method).
            - When ``frac_active`` is None and ``min_frac_active > 0``,
              activity fraction is computed via ``compute_frac_active``.
            - Analogous to ``RateSliceStack.order_units_across_slices`` but
              operates on raw spike trains instead of firing rate curves.
        """
        if agg_func not in ("median", "mean"):
            raise ValueError(f"agg_func must be 'median' or 'mean', got {agg_func!r}")

        num_units = self.N
        num_slices = len(self.spike_stack)

        if timing_matrix is not None:
            timing_matrix = np.asarray(timing_matrix, dtype=float)
            if timing_matrix.shape != (num_units, num_slices):
                raise ValueError(
                    f"timing_matrix must have shape ({num_units}, {num_slices}), "
                    f"got {timing_matrix.shape}"
                )
        else:
            timing_matrix = self.get_unit_timing_per_slice(
                timing=timing, min_spikes=min_spikes
            )

        # Standard deviation across slices
        unit_std_values = np.nanstd(timing_matrix, axis=1)

        # Aggregate across slices
        if agg_func == "median":
            unit_timing = np.nanmedian(timing_matrix, axis=1)
        else:
            unit_timing = np.nanmean(timing_matrix, axis=1)

        # Compute or validate frac_active only when splitting is requested
        skip_split = not min_frac_active
        if skip_split:
            frac_active = np.ones(num_units)
            ha_units = np.arange(num_units)
            la_units = np.array([], dtype=int)
        else:
            if frac_active is not None:
                frac_active = np.asarray(frac_active, dtype=float)
                if frac_active.shape != (num_units,):
                    raise ValueError(
                        f"frac_active must have shape ({num_units},), "
                        f"got {frac_active.shape}"
                    )
            else:
                frac_active = self.compute_frac_active(min_spikes=min_spikes)
            highly_active_mask = frac_active >= min_frac_active
            ha_units = np.where(highly_active_mask)[0]
            la_units = np.where(~highly_active_mask)[0]

        # Sort within each group by typical timing
        ha_order = ha_units[np.argsort(unit_timing[ha_units])]
        la_order = la_units[np.argsort(unit_timing[la_units])]

        # Build reordered SpikeSliceStacks
        def _reorder_stack(unit_indices):
            if len(unit_indices) == 0:
                return None
            return self.subset(list(unit_indices))

        ha_stack = _reorder_stack(ha_order)
        la_stack = _reorder_stack(la_order)

        return (
            (ha_stack, la_stack),
            (ha_order, la_order),
            (unit_std_values[ha_order], unit_std_values[la_order]),
            (unit_timing[ha_order], unit_timing[la_order]),
            (frac_active[ha_order], frac_active[la_order]),
        )

    def apply(self, func, *args, **kwargs):
        """Apply a function to each SpikeData in the stack and return stacked results.

        Calls ``func(sd, *args, **kwargs)`` on every slice and stacks the
        outputs into a single numpy array with a new leading axis of size S
        (number of slices).

        Parameters:
            func (callable): Function that accepts a SpikeData as its first
                argument and returns a numeric value (scalar, 1-D, or 2-D
                array). Output shape must be consistent across all slices.
            *args: Additional positional arguments forwarded to func.
            **kwargs: Additional keyword arguments forwarded to func.

        Returns:
            result (np.ndarray): Stacked results with shape ``(S, ...)``.

        Notes:
            - Intended for use with stacks built by ``SpikeData.frames``,
              ``SpikeData.align_to_events``, ``SpikeData.spike_shuffle_stack``,
              or ``SpikeData.subset_stack``. Pair with ``shuffle_z_score``,
              ``shuffle_percentile``, ``slice_trend``, or ``slice_stability``
              from ``utils`` to interpret the results.
        """
        results = [func(sd, *args, **kwargs) for sd in self.spike_stack]
        return np.stack(results, axis=0)

    def unit_to_unit_comparison(
        self,
        metric="ccg",
        delt=20.0,
        bin_size=1.0,
        max_lag=350,
        n_jobs=-1,
    ):
        """Compute pairwise unit-to-unit similarity within each slice using spike-based metrics.

        For each slice, computes a (U, U) similarity matrix between all unit pairs,
        then stacks the results into a ``PairwiseCompMatrixStack (U, U, S)``.

        Parameters:
            metric (str): Similarity metric to use. ``"ccg"`` for cross-correlogram
                on binned rasters (default), ``"sttc"`` for spike time tiling coefficient.
            delt (float): STTC time window in milliseconds (default: 20.0).
                Only used when metric is ``"sttc"``.
            bin_size (float): Bin size in milliseconds for the binary raster
                (default: 1.0). Only used when metric is ``"ccg"``.
            max_lag (float): Maximum lag in milliseconds to search for the peak
                correlation (default: 350). Only used when metric is ``"ccg"``.

        Returns:
            corr_stack (PairwiseCompMatrixStack): Pairwise similarity scores between
                all unit pairs for each slice. Shape is ``(U, U, S)``.
            lag_stack (PairwiseCompMatrixStack or None): Lag at which maximum
                similarity occurs for each pair per slice. Shape is ``(U, U, S)``.
                ``None`` when metric is ``"sttc"`` (STTC has no lag).
            av_corr (np.ndarray): Average similarity per slice across all unit
                pairs in the lower triangle. Shape is ``(S,)``.
            av_lag (np.ndarray or None): Average lag per slice. Shape is ``(S,)``.
                ``None`` when metric is ``"sttc"``.

        Notes:
            - Analogous to ``RateSliceStack.unit_to_unit_correlation`` but operates
              on raw spike trains instead of firing rate time series.
        """
        if metric not in ("sttc", "ccg"):
            raise ValueError(f"metric must be 'sttc' or 'ccg', got {metric!r}")

        num_units = self.N
        num_slices = len(self.spike_stack)

        if num_units < 2:
            warnings.warn(
                "Cannot compute unit-to-unit comparison with fewer than "
                "2 units. Returning NaN.",
                RuntimeWarning,
            )
            nan_stack = np.full((num_units, num_units, num_slices), np.nan)
            nan_avgs = np.full(num_slices, np.nan)
            return (
                PairwiseCompMatrixStack(stack=nan_stack, times=self.times),
                (
                    PairwiseCompMatrixStack(stack=nan_stack.copy(), times=self.times)
                    if metric == "ccg"
                    else None
                ),
                nan_avgs,
                nan_avgs.copy() if metric == "ccg" else None,
            )

        corr_matrices = []
        lag_matrices = []

        for sd in self.spike_stack:
            if metric == "sttc":
                pcm = sd.spike_time_tilings(delt=delt)
                corr_matrices.append(pcm.matrix)
            else:  # ccg
                corr_pcm, lag_pcm = sd.get_pairwise_ccg(
                    bin_size=bin_size, max_lag=max_lag, n_jobs=n_jobs
                )
                corr_matrices.append(corr_pcm.matrix)
                lag_matrices.append(lag_pcm.matrix)

        # Stack: list of (U, U) -> (S, U, U) -> transpose to (U, U, S)
        corr_array = np.moveaxis(np.stack(corr_matrices, axis=0), 0, 2)

        lower_tri = np.tril_indices(num_units, k=-1)
        av_corr = np.nanmean(corr_array[lower_tri[0], lower_tri[1], :], axis=0)

        corr_stack = PairwiseCompMatrixStack(stack=corr_array, times=self.times)

        if metric == "ccg":
            lag_array = np.moveaxis(np.stack(lag_matrices, axis=0), 0, 2)
            av_lag = np.nanmean(lag_array[lower_tri[0], lower_tri[1], :], axis=0)
            lag_stack = PairwiseCompMatrixStack(stack=lag_array, times=self.times)
        else:
            lag_stack = None
            av_lag = None

        return corr_stack, lag_stack, av_corr, av_lag

    def get_slice_to_slice_unit_comparison(
        self,
        metric="ccg",
        delt=20.0,
        bin_size=1.0,
        max_lag=350,
        min_spikes=2,
        min_frac=0.3,
        frac_active=None,
        n_jobs=-1,
    ):
        """Compute slice-to-slice similarity for each unit using spike-based metrics.

        For each unit independently, compares its spike train across every pair of
        slices. Asks: "Does unit X fire in the same temporal pattern across repeated
        events?" Returns a ``PairwiseCompMatrixStack (S, S, U)``.

        Parameters:
            metric (str): Similarity metric to use. ``"ccg"`` for cross-correlogram
                on binned rasters (default), ``"sttc"`` for spike time tiling coefficient.
            delt (float): STTC time window in milliseconds (default: 20.0).
                Only used when metric is ``"sttc"``.
            bin_size (float): Bin size in milliseconds for the binary raster
                (default: 1.0). Only used when metric is ``"ccg"``.
            max_lag (float): Maximum lag in milliseconds to search for the peak
                correlation (default: 350). Only used when metric is ``"ccg"``.
            min_spikes (int): Minimum number of spikes in a slice for a unit to
                be considered valid in that slice (default: 2).
            min_frac (float): Maximum fraction of slices that can be invalid before
                a unit's average is set to NaN (default: 0.3).
            frac_active (np.ndarray or None): Optional pre-computed
                fraction-active array of shape ``(U,)`` to override the
                internal per-unit validity check for computing averages.
                When provided, a unit's average is set to NaN if
                ``frac_active[u] < (1 - min_frac)``. ``min_spikes`` still
                controls which individual slice pairs are computed.
                Compatible sources: ``SpikeSliceStack.compute_frac_active``
                and ``SpikeData.get_frac_active`` (``frac_per_unit`` output).
            n_jobs (int): Number of threads for parallel computation. -1 uses
                all cores (default), 1 disables parallelism, None is serial.

        Returns:
            all_corr (PairwiseCompMatrixStack): Pairwise similarity between all
                slice pairs for each unit. Shape is ``(S, S, U)``.
            all_lag (PairwiseCompMatrixStack or None): Lag at which maximum
                similarity occurs for each slice pair per unit. Shape is ``(S, S, U)``.
                ``None`` when metric is ``"sttc"``.
            av_corr (np.ndarray): Average similarity per unit across all valid
                slice pairs. Shape is ``(U,)``.
            av_lag (np.ndarray or None): Average lag per unit. Shape is ``(U,)``.
                ``None`` when metric is ``"sttc"``.

        Notes:
            - Analogous to ``RateSliceStack.get_slice_to_slice_unit_corr_from_stack``
              but operates on raw spike trains.
            - Spike times within each slice are relative to the slice time
              origin (0-based or event-centered) for aligned comparison.
        """
        if metric not in ("sttc", "ccg"):
            raise ValueError(f"metric must be 'sttc' or 'ccg', got {metric!r}")

        num_units = self.N
        num_slices = len(self.spike_stack)

        if num_slices < 2:
            warnings.warn(
                "Cannot compute slice-to-slice unit comparison with fewer than "
                "2 slices. Returning NaN.",
                RuntimeWarning,
            )
            av_corr = np.full(num_units, np.nan)
            nan_stack = np.full((num_slices, num_slices, num_units), np.nan)
            return (
                PairwiseCompMatrixStack(stack=nan_stack),
                (
                    PairwiseCompMatrixStack(stack=nan_stack.copy())
                    if metric == "ccg"
                    else None
                ),
                av_corr,
                av_corr.copy() if metric == "ccg" else None,
            )

        # Pre-compute spike trains per slice (relative to slice time origin)
        # and per-slice rasters for CCG
        shifted_trains = []  # list of S lists, each containing U spike arrays
        slice_durations = []
        slice_rasters = []  # only populated for CCG

        for sd, (start, end) in zip(self.spike_stack, self.times):
            duration = end - start
            slice_durations.append(duration)
            trains = []
            for u in range(num_units):
                trains.append(np.asarray(sd.train[u]))
            shifted_trains.append(trains)

            if metric == "ccg":
                # Build shifted SpikeData for raster computation
                temp_sd = SpikeData(
                    trains, length=duration, start_time=sd.start_time, N=num_units
                )
                slice_rasters.append(temp_sd.raster(bin_size))

        max_lag_bins = int(round(max_lag / bin_size)) if metric == "ccg" else 0

        # Warn when a positive max_lag rounds down to zero bins for the
        # ccg path — the user's lag request is silently discarded
        # otherwise. Matches the guard in ``SpikeData.get_pairwise_ccg``.
        if metric == "ccg" and max_lag > 0 and max_lag_bins == 0:
            warnings.warn(
                f"max_lag={max_lag} ms is smaller than bin_size={bin_size} ms; "
                f"max_lag_bins collapsed to 0 (zero-lag only). To resolve "
                f"sub-bin lags, decrease bin_size.",
                UserWarning,
                stacklevel=2,
            )

        # Validate frac_active override if provided
        if frac_active is not None:
            frac_active = np.asarray(frac_active, dtype=float)
            if frac_active.shape != (num_units,):
                raise ValueError(
                    f"frac_active must have shape ({num_units},), "
                    f"got {frac_active.shape}"
                )

        # Initialize result arrays: (U, S, S), will transpose to (S, S, U) at end
        all_corr_scores = np.full((num_units, num_slices, num_slices), np.nan)
        all_lag_scores = (
            np.full((num_units, num_slices, num_slices), np.nan)
            if metric == "ccg"
            else None
        )
        av_corr = np.full(num_units, np.nan)
        av_lag = np.full(num_units, np.nan) if metric == "ccg" else None

        lower_tri = np.tril_indices(num_slices, k=-1)

        start_times = [sd.start_time for sd in self.spike_stack]

        def _process_unit(unit):
            unit_corr = np.full((num_slices, num_slices), np.nan)
            unit_lag = (
                np.full((num_slices, num_slices), np.nan) if metric == "ccg" else None
            )
            invalid_count = 0

            for ref_s in range(num_slices):
                ref_train = shifted_trains[ref_s][unit]
                if len(ref_train) < min_spikes:
                    invalid_count += 1
                    continue
                for comp_s in range(ref_s, num_slices):
                    comp_train = shifted_trains[comp_s][unit]
                    if len(comp_train) < min_spikes:
                        continue
                    if metric == "sttc":
                        length = max(slice_durations[ref_s], slice_durations[comp_s])
                        # start_time from the ref slice is correct here:
                        # all slices share the same start_time (event-centered
                        # data has -pre_ms, frames() produces 0-based slices).
                        score = get_sttc(
                            ref_train,
                            comp_train,
                            delt=delt,
                            length=length,
                            start_time=start_times[ref_s],
                        )
                        unit_corr[ref_s, comp_s] = score
                        unit_corr[comp_s, ref_s] = score
                    else:
                        ref_signal = slice_rasters[ref_s][unit, :]
                        comp_signal = slice_rasters[comp_s][unit, :]
                        score, lag = compute_cross_correlation_with_lag(
                            ref_signal, comp_signal, max_lag=max_lag_bins
                        )
                        unit_corr[ref_s, comp_s] = score
                        unit_corr[comp_s, ref_s] = score
                        unit_lag[ref_s, comp_s] = lag
                        unit_lag[comp_s, ref_s] = -lag

            if frac_active is not None:
                unit_valid = frac_active[unit] >= (1 - min_frac)
            else:
                unit_valid = invalid_count / num_slices <= min_frac

            av_c = (
                np.nanmean(unit_corr[lower_tri[0], lower_tri[1]])
                if unit_valid
                else np.nan
            )
            av_l = np.nan
            if metric == "ccg" and unit_valid:
                av_l = np.nanmean(unit_lag[lower_tri[0], lower_tri[1]])

            return unit, unit_corr, unit_lag, av_c, av_l

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and num_units > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = pool.map(_process_unit, range(num_units))
        else:
            results = map(_process_unit, range(num_units))

        for unit, unit_corr, unit_lag, av_c, av_l in results:
            all_corr_scores[unit] = unit_corr
            av_corr[unit] = av_c
            if metric == "ccg":
                all_lag_scores[unit] = unit_lag
                av_lag[unit] = av_l

        # Transpose from (U, S, S) to (S, S, U)
        all_corr_scores = np.moveaxis(all_corr_scores, 0, 2)
        all_corr_stack = PairwiseCompMatrixStack(stack=all_corr_scores)

        if metric == "ccg":
            all_lag_scores = np.moveaxis(all_lag_scores, 0, 2)
            all_lag_stack = PairwiseCompMatrixStack(stack=all_lag_scores)
        else:
            all_lag_stack = None

        return all_corr_stack, all_lag_stack, av_corr, av_lag

    def get_unit_timing_per_slice(
        self,
        timing="median",
        min_spikes=2,
    ):
        """Compute a representative spike time for each unit in each slice.

        Returns a ``(U, S)`` matrix where entry ``[u, s]`` is the timing
        value (in milliseconds relative to the slice's time origin) for unit u
        in slice s. For event-centered slices, t=0 is the event. Units with
        fewer than min_spikes spikes in a slice are marked NaN.

        Parameters:
            timing (str): Which spike time to extract per unit per slice.
                ``"median"`` (default), ``"mean"``, or ``"first"`` (onset
                latency).
            min_spikes (int): Minimum number of spikes for a unit to count
                as active in a slice (default: 2).

        Returns:
            timing_matrix (np.ndarray): Array of shape ``(U, S)`` with timing
                values in milliseconds relative to each slice's time origin.
                NaN where the unit is inactive.

        Notes:
            - Values are in milliseconds, not bin indices. This differs from
              ``RateSliceStack.get_unit_timing_per_slice`` which returns bin
              indices (suitable for direct indexing into the event stack).
              Both representations preserve rank order, so
              ``rank_order_correlation`` produces identical results either way.
            - The returned matrix can be passed to ``rank_order_correlation``
              to compute Spearman rank correlations between slice pairs, or
              used as input to ``order_units_across_slices`` for manual
              inspection of per-slice timing values.
        """
        if timing not in ("median", "mean", "first"):
            raise ValueError(
                f"timing must be 'median', 'mean', or 'first', got {timing!r}"
            )

        num_units = self.N
        num_slices = len(self.spike_stack)
        timing_matrix = np.full((num_units, num_slices), np.nan)

        for s_idx, (sd, (start, end)) in enumerate(zip(self.spike_stack, self.times)):
            for u in range(num_units):
                spikes = np.asarray(sd.train[u])
                duration = end - start
                spikes = spikes[
                    (spikes >= sd.start_time) & (spikes < sd.start_time + duration)
                ]
                if len(spikes) < min_spikes:
                    continue
                if timing == "median":
                    timing_matrix[u, s_idx] = np.median(spikes)
                elif timing == "mean":
                    timing_matrix[u, s_idx] = np.mean(spikes)
                else:
                    timing_matrix[u, s_idx] = spikes[0]

        return timing_matrix

    def rank_order_correlation(
        self,
        timing_matrix=None,
        timing="median",
        min_spikes=2,
        min_overlap=3,
        n_shuffles=100,
        min_overlap_frac=None,
        seed=1,
        n_jobs=-1,
    ):
        """Compute Spearman rank-order correlation of unit timing between all slice pairs.

        For each pair of slices, only units active in both slices (non-NaN in
        both columns of the timing matrix) are included. If the overlap falls
        below the required minimum, the pair is set to NaN.

        When ``n_shuffles > 0``, the rank orders are shuffled n_shuffles
        times for each pair to build a null distribution, and the raw
        correlation is z-score normalised against it.

        Parameters:
            timing_matrix (np.ndarray or None): Array of shape ``(U, S)`` with
                timing values per unit per slice. NaN entries mark inactive
                units. Typically produced by ``get_unit_timing_per_slice``.
                When None, computed automatically using timing and
                min_spikes.
            timing (str): Which spike time to extract per unit per slice.
                ``"median"`` (default), ``"mean"``, or ``"first"``. Only used
                when timing_matrix is None.
            min_spikes (int): Minimum spikes for activity (default: 2). Only
                used when timing_matrix is None.
            min_overlap (int): Minimum number of units that must be active in
                both slices (default: 3).
            min_overlap_frac (float or None): Minimum fraction of total units
                that must be active in both slices (default: None). When
                provided, the effective threshold is
                ``max(min_overlap, ceil(min_overlap_frac * U))``.
            n_shuffles (int): Number of shuffle iterations for z-scoring
                (default: 100). Set to 0 to return raw Spearman correlations.
                Values between 1 and 4 are rejected (minimum 5 required for
                a meaningful null distribution).
            seed (int or None): Random seed for reproducibility of the shuffle
                (default: 1).
            n_jobs (int): Number of threads for parallel computation. -1 uses
                all cores (default), 1 disables parallelism, None is serial.

        Returns:
            corr_matrix (PairwiseCompMatrix): Spearman correlation matrix of
                shape ``(S, S)``. When ``n_shuffles > 0``, values are z-scores.
                When ``n_shuffles == 0``, values are raw Spearman correlations.
            av_corr (float): Average correlation (or z-score) across all valid
                lower-triangle pairs.
            overlap_matrix (PairwiseCompMatrix): Matrix of shape ``(S, S)``
                with fraction of units active in both slices.
        """
        from .utils import _rank_order_correlation_from_timing

        if timing_matrix is None:
            timing_matrix = self.get_unit_timing_per_slice(
                timing=timing, min_spikes=min_spikes
            )

        return _rank_order_correlation_from_timing(
            timing_matrix,
            min_overlap=min_overlap,
            min_overlap_frac=min_overlap_frac,
            n_shuffles=n_shuffles,
            seed=seed,
            n_jobs=n_jobs,
        )

    def plot_aligned_slice_single_unit(
        self,
        unit_idx,
        ax=None,
        color_vals=None,
        color_label="",
        cmap="viridis",
        time_offset=0,
        xlabel="Rel. time (ms)",
        ylabel="Burst",
        x_range=None,
        vlines=None,
        show_colorbar=True,
        marker_size=20,
        font_size=None,
        style="scatter",
        invert_y=False,
        linewidths=0.5,
    ):
        """Plot a single unit's spike times across all slices as a raster.

        Extracts the spike train for unit_idx from every slice and delegates
        to :func:`~SpikeLab.spikedata.plot_utils.plot_aligned_slice_single_unit`.

        Parameters:
            unit_idx (int): Index of the unit to plot.
            ax (matplotlib.axes.Axes or None): Target axes. If None, a new
                figure and axes are created.
            color_vals (np.ndarray or None): Per-slice colour values.
            color_label (str): Colorbar label.
            cmap (str): Matplotlib colormap name.
            time_offset (float): Value subtracted from every spike time
                before plotting. Slices from ``align_to_events`` are
                already event-centered (spike times in
                ``[-pre_ms, +post_ms]``), so use the default
                ``time_offset=0``. Only set a non-zero value when spike
                times are not already centered on the event.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            x_range (tuple or None): ``(xmin, xmax)`` for the x-axis.
            vlines (list[dict] or None): Vertical reference lines. Each dict
                must contain ``'x'`` and may optionally include ``'color'``,
                ``'linestyle'``, ``'linewidth'``.
            show_colorbar (bool): Add a colorbar when color_vals is provided.
            marker_size (float): Scatter marker size.
            font_size (int or None): Font size for labels/ticks.
            style (str): ``"scatter"`` for dot markers, ``"eventplot"`` for
                vertical line markers.
            invert_y (bool): If True, first slice at top, last at bottom.
            linewidths (float): Line width for eventplot markers.

        Returns:
            result: ``(fig, ax, sc)`` when ax is None, otherwise just sc.
                sc is the scatter ``PathCollection`` (or None if no colour
                coding).
        """
        from .plot_utils import (
            plot_aligned_slice_single_unit as _plot_aligned_slice_single_unit,
        )

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "plot_aligned_slice_single_unit requires 'matplotlib'. "
                "Install with: pip install matplotlib"
            ) from e

        if unit_idx < 0 or unit_idx >= self.N:
            raise IndexError(f"unit_idx {unit_idx} out of range for {self.N} units.")

        # Extract per-slice spike times for this unit
        spike_times_per_slice = [sd.train[unit_idx] for sd in self.spike_stack]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 6))

        sc = _plot_aligned_slice_single_unit(
            ax,
            spike_times_per_slice,
            color_vals=color_vals,
            color_label=color_label,
            cmap=cmap,
            time_offset=time_offset,
            xlabel=xlabel,
            ylabel=ylabel,
            x_range=x_range,
            vlines=vlines,
            show_colorbar=show_colorbar,
            marker_size=marker_size,
            font_size=font_size,
            style=style,
            invert_y=invert_y,
            linewidths=linewidths,
        )

        if standalone:
            plt.tight_layout()
            return fig, ax, sc

        return sc
