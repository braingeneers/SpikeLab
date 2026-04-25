"""Tests for the stim_sorting preprocessing helper and BaseRecording plumbing.

Covers six recent fixes:

1. ``recording_io.load_recording`` must accept a pre-loaded
   ``BaseRecording`` without crashing on ``Path(rec_path)``.
2. ``pipeline.sort_recording`` must be able to auto-generate intermediate
   and results folder paths from ``BinaryRecordingExtractor`` inputs
   (previously it crashed because ``Path(BinaryRecordingExtractor)`` raises).
3. ``stim_sorting.preprocess_stim_artifacts`` wraps ``recenter_stim_times``
   and ``remove_stim_artifacts`` and returns a SpikeInterface recording
   whose channel IDs/locations/gains/offsets are inherited from the input.
4. ``remove_stim_artifacts(method="polynomial")`` must clamp divergent
   fits (≥ ``poly_clamp_factor * saturation_threshold``) by blanking
   the segment and emitting one warning per call.
5. ``maxwell_io.load_maxwell_native`` must dedupe duplicate channel IDs
   in mxw v25.x ``settings/mapping`` tables.
6. ``recenter_stim_times`` must warn when the median ``|offset|`` shift
   exceeds ``warn_offset_ms`` — usually a fixed log/hardware delay.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

try:
    import spikeinterface  # noqa: F401

    _has_spikeinterface = True
except Exception:
    _has_spikeinterface = False

try:
    import h5py  # noqa: F401

    _has_h5py = True
except Exception:
    _has_h5py = False

skip_no_h5py = pytest.mark.skipif(not _has_h5py, reason="h5py not installed")

skip_no_spikeinterface = pytest.mark.skipif(
    not _has_spikeinterface, reason="spikeinterface not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_numpy_recording(
    num_samples: int = 20000,
    num_channels: int = 4,
    fs: float = 20000.0,
    seed: int = 0,
):
    """Build a small NumpyRecording with channel metadata set."""
    from spikeinterface.core import NumpyRecording

    rng = np.random.default_rng(seed)
    traces = rng.standard_normal((num_samples, num_channels)).astype(np.float32) * 30.0
    rec = NumpyRecording(
        traces_list=[traces],
        sampling_frequency=fs,
        channel_ids=np.array([f"ch{i}" for i in range(num_channels)]),
    )
    rec.set_channel_locations(
        np.column_stack(
            [np.arange(num_channels) * 20.0, np.zeros(num_channels)]
        ).astype(np.float32)
    )
    rec.set_channel_gains(np.ones(num_channels, dtype=np.float32))
    rec.set_channel_offsets(np.zeros(num_channels, dtype=np.float32))
    return rec


def _make_binary_recording(
    tmp_path: Path, num_samples=2000, num_channels=4, fs=20000.0
):
    """Build an on-disk BinaryRecordingExtractor of random float32 data."""
    from spikeinterface.core import BinaryRecordingExtractor

    rng = np.random.default_rng(1)
    traces = rng.standard_normal((num_samples, num_channels)).astype(np.float32)
    bin_path = tmp_path / "rec.dat"
    traces.tofile(bin_path)
    return BinaryRecordingExtractor(
        file_paths=[str(bin_path)],
        sampling_frequency=fs,
        num_channels=num_channels,
        dtype="float32",
    )


def _inject_artifacts(
    traces_ch_first: np.ndarray,
    stim_samples: np.ndarray,
    width_samples: int = 40,
    amp: float = 2000.0,
):
    """Stamp a fat saturation-like pulse at each stim sample, in-place."""
    n_ch, n_samp = traces_ch_first.shape
    rng = np.random.default_rng(123)
    for s in stim_samples:
        lo = max(0, int(s) - 2)
        hi = min(n_samp, int(s) + width_samples)
        if hi <= lo:
            continue
        sign = rng.choice([-1.0, 1.0], size=(n_ch, hi - lo))
        traces_ch_first[:, lo:hi] = amp * sign


# ===========================================================================
# Fix 1: load_recording with a pre-loaded BaseRecording
# ===========================================================================


@skip_no_spikeinterface
class TestLoadRecordingBaseRecording:
    """
    Tests for the BaseRecording short-circuit in
    ``spikelab.spike_sorting.recording_io.load_recording``.

    Tests:
        (Test Case 1) NumpyRecording passes through without crashing.
        (Test Case 2) BinaryRecordingExtractor passes through without crashing.
        (Test Case 3) FIRST_N_MINS truncation still applies to BaseRecording.
    """

    @pytest.fixture(autouse=True)
    def _reset_recording_globals(self, monkeypatch):
        """``load_recording`` reads several module-level globals that earlier
        tests in the same pytest process may have populated.  Reset them to
        their declared defaults via ``monkeypatch`` so each test sees a clean
        state and the globals are restored at teardown."""
        from spikelab.spike_sorting import _globals

        for name, default in [
            ("STREAM_ID", None),
            ("FIRST_N_MINS", None),
            ("MEA_Y_MAX", None),
            ("GAIN_TO_UV", None),
            ("OFFSET_TO_UV", None),
            ("REC_CHUNKS", []),
            ("REC_CHUNKS_S", []),
            ("START_TIME_S", None),
            ("END_TIME_S", None),
            ("FREQ_MIN", 300),
            ("FREQ_MAX", 6000),
        ]:
            monkeypatch.setattr(_globals, name, default)

    def test_numpy_recording_passes_through(self):
        from spikeinterface.core import BaseRecording

        from spikelab.spike_sorting.recording_io import load_recording

        rec = _make_numpy_recording(num_samples=4000, num_channels=4)
        out = load_recording(rec)
        assert isinstance(out, BaseRecording)
        # load_single_recording wraps in ScaleRecording + bandpass_filter, so
        # channel count is preserved but the object identity won't be.
        assert out.get_num_channels() == rec.get_num_channels()
        assert out.get_sampling_frequency() == rec.get_sampling_frequency()

    def test_binary_recording_passes_through(self, tmp_path):
        from spikeinterface.core import BaseRecording

        from spikelab.spike_sorting.recording_io import load_recording

        rec = _make_binary_recording(tmp_path)
        out = load_recording(rec)
        assert isinstance(out, BaseRecording)
        assert out.get_num_channels() == rec.get_num_channels()

    def test_first_n_mins_truncation_applies(self, monkeypatch):
        """Post-loading truncation (FIRST_N_MINS) must still fire for
        BaseRecording inputs — the early-return branch hands control back
        to the same post-processing pipeline as the file-path branch."""
        from spikelab.spike_sorting import _globals
        from spikelab.spike_sorting.recording_io import load_recording

        # 4 s recording, truncate to 1/30 min (2 s).
        fs = 20000.0
        rec = _make_numpy_recording(num_samples=int(4 * fs), fs=fs)
        monkeypatch.setattr(_globals, "FIRST_N_MINS", 2 / 60)
        out = load_recording(rec)
        assert out.get_total_duration() == pytest.approx(2.0, rel=0.01)


# ===========================================================================
# Fix 2: sort_recording auto-folder generation from BaseRecording
# ===========================================================================


@skip_no_spikeinterface
class TestSortRecordingAutoFolderFromBaseRecording:
    """
    Tests for BaseRecording handling in
    ``pipeline.sort_recording``'s auto-folder generation.

    Tests:
        (Test Case 1) BinaryRecordingExtractor input without explicit
            intermediate_folders/results_folders resolves to the backing
            file's parent directory rather than raising TypeError.
        (Test Case 2) NumpyRecording (no backing file) requires explicit
            folders — the validation error is about the folder path, not
            ``Path(NumpyRecording)``.
    """

    def test_binary_recording_auto_folder_resolves(self, tmp_path, monkeypatch):
        """Exercise the folder auto-generation path end-to-end.

        We don't actually want to run a sorter; we want to verify that
        auto-folder generation no longer raises ``TypeError: expected
        str/bytes/PathLike, not BinaryRecordingExtractor``.  We intercept
        ``get_backend_class`` (the first heavy call after the folder
        block) and raise a sentinel.  If the test reaches that sentinel,
        the auto-folder code survived a BinaryRecordingExtractor input.
        """
        from spikelab.spike_sorting import backends as _backends
        from spikelab.spike_sorting import pipeline as _pipeline

        rec = _make_binary_recording(tmp_path)

        class _Sentinel(RuntimeError):
            pass

        def _raise_after_folder_block(sorter_name):
            raise _Sentinel("folders resolved")

        monkeypatch.setattr(_backends, "get_backend_class", _raise_after_folder_block)

        with pytest.raises(_Sentinel):
            _pipeline.sort_recording(
                recording_files=[rec],
                sorter="kilosort2",
                # deliberately omit intermediate_folders / results_folders
            )

    def test_numpy_recording_requires_explicit_folders(self):
        """A NumpyRecording has no backing file; the auto-folder helper
        should raise a clear ValueError rather than a cryptic
        ``Path(<NumpyRecording>)`` TypeError."""
        from spikelab.spike_sorting import pipeline as _pipeline

        rec = _make_numpy_recording(num_samples=1000, num_channels=4)
        with pytest.raises(ValueError, match="backing file path"):
            _pipeline.sort_recording(
                recording_files=[rec],
                sorter="kilosort2",
            )


# ===========================================================================
# Fix 3: preprocess_stim_artifacts end-to-end
# ===========================================================================


@skip_no_spikeinterface
class TestPreprocessStimArtifacts:
    """
    Tests for ``spikelab.spike_sorting.stim_sorting.preprocess_stim_artifacts``.

    Tests:
        (Test Case 1) With no output_path, returns a NumpyRecording and
            injected artifact windows are blanked.
        (Test Case 2) With output_path, returns a BinaryRecordingExtractor
            whose file size matches num_samples * num_channels * 4 bytes.
        (Test Case 3) Channel IDs, locations, gains, and offsets are
            propagated from the input recording.
        (Test Case 4) Non-BaseRecording inputs raise TypeError.
        (Test Case 5) Multi-segment recordings raise ValueError.
        (Test Case 6) Empty stim_times_ms is a no-op: output traces equal
            input traces and blanked_fraction is 0.
        (Test Case 7) recenter=False preserves the stim times as passed.
    """

    @pytest.fixture()
    def recording_with_artifacts(self):
        """A NumpyRecording with injected stim artifacts at known samples."""
        fs = 20000.0
        n_ch, n_samp = 4, int(fs)  # 1 s
        rng = np.random.default_rng(2)
        traces = (
            rng.standard_normal((n_samp, n_ch)).astype(np.float32) * 30.0
        )  # (samples, channels)
        # Inject artifacts at 200 ms and 600 ms
        stim_samples = np.array([int(0.2 * fs), int(0.6 * fs)])
        _inject_artifacts(traces.T, stim_samples, width_samples=60, amp=3000.0)
        stim_ms = stim_samples.astype(np.float64) / fs * 1000.0

        from spikeinterface.core import NumpyRecording

        rec = NumpyRecording(
            traces_list=[traces],
            sampling_frequency=fs,
            channel_ids=np.array(["a", "b", "c", "d"]),
        )
        rec.set_channel_locations(
            np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=np.float32)
        )
        rec.set_channel_gains(np.full(n_ch, 2.5, dtype=np.float32))
        rec.set_channel_offsets(np.full(n_ch, 7.0, dtype=np.float32))
        return rec, stim_ms

    def test_numpy_output_no_path(self, recording_with_artifacts):
        from spikeinterface.core import BaseRecording, NumpyRecording

        from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts

        rec, stim_ms = recording_with_artifacts
        out, meta = preprocess_stim_artifacts(
            rec, stim_ms, method="blank", recenter=False, artifact_window_ms=5.0
        )
        assert isinstance(out, BaseRecording)
        assert isinstance(out, NumpyRecording)
        assert out.get_num_channels() == rec.get_num_channels()
        assert out.get_sampling_frequency() == rec.get_sampling_frequency()
        # Some fraction of samples must have been blanked.
        assert 0.0 < meta["blanked_fraction"] < 1.0
        assert meta["blanked_fraction_per_channel"].shape == (rec.get_num_channels(),)

    def test_binary_output_with_path(self, recording_with_artifacts, tmp_path):
        from spikeinterface.core import BaseRecording, BinaryRecordingExtractor

        from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts

        rec, stim_ms = recording_with_artifacts
        bin_path = tmp_path / "cleaned.dat"
        out, meta = preprocess_stim_artifacts(
            rec,
            stim_ms,
            output_path=bin_path,
            method="blank",
            recenter=False,
            artifact_window_ms=5.0,
        )
        assert isinstance(out, BaseRecording)
        assert isinstance(out, BinaryRecordingExtractor)
        assert bin_path.exists()
        expected_bytes = (
            rec.get_num_channels()
            * rec.get_num_samples()
            * np.dtype("float32").itemsize
        )
        assert bin_path.stat().st_size == expected_bytes
        # BinaryRecordingExtractor must be dumpable — this is the whole
        # reason we prefer file-backed output for Docker-based sorters.
        out.to_dict(recursive=True)

    def test_channel_metadata_propagation(self, recording_with_artifacts, tmp_path):
        from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts

        rec, stim_ms = recording_with_artifacts
        out, _ = preprocess_stim_artifacts(
            rec,
            stim_ms,
            output_path=tmp_path / "clean.dat",
            method="blank",
            recenter=False,
        )
        np.testing.assert_array_equal(out.get_channel_ids(), rec.get_channel_ids())
        np.testing.assert_array_equal(
            out.get_channel_locations(), rec.get_channel_locations()
        )
        np.testing.assert_array_equal(out.get_channel_gains(), rec.get_channel_gains())
        np.testing.assert_array_equal(
            out.get_channel_offsets(), rec.get_channel_offsets()
        )

    def test_non_baserecording_raises(self):
        from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts

        with pytest.raises(TypeError, match="BaseRecording"):
            preprocess_stim_artifacts(
                np.zeros((4, 1000), dtype=np.float32),
                np.array([100.0]),
            )

    def test_multi_segment_raises(self):
        from spikeinterface.core import NumpyRecording

        from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts

        rng = np.random.default_rng(0)
        seg = rng.standard_normal((500, 3)).astype(np.float32)
        rec = NumpyRecording(traces_list=[seg, seg], sampling_frequency=20000.0)
        with pytest.raises(ValueError, match="single-segment"):
            preprocess_stim_artifacts(rec, np.array([10.0]))

    def test_empty_stim_times_is_noop(self, recording_with_artifacts):
        from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts

        rec, _ = recording_with_artifacts
        out, meta = preprocess_stim_artifacts(
            rec, np.array([]), method="blank", recenter=False
        )
        # No stims -> no blanking
        assert meta["blanked_fraction"] == 0.0
        # Output traces equal input traces (both in sample-major layout)
        np.testing.assert_array_equal(out.get_traces(), rec.get_traces())

    def test_recenter_false_preserves_times(self, recording_with_artifacts):
        from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts

        rec, stim_ms = recording_with_artifacts
        _, meta = preprocess_stim_artifacts(
            rec, stim_ms, method="blank", recenter=False
        )
        np.testing.assert_array_equal(
            meta["stim_times_ms_corrected"], meta["stim_times_ms_logged"]
        )
        np.testing.assert_array_equal(
            meta["recenter_offsets_ms"], np.zeros_like(stim_ms)
        )


# ===========================================================================
# stim_sorting __init__ re-export
# ===========================================================================


class TestStimSortingInitExport:
    """
    Tests for the lazy re-export of ``preprocess_stim_artifacts`` from
    ``spikelab.spike_sorting.stim_sorting``.

    Tests:
        (Test Case 1) The attribute is in __all__.
        (Test Case 2) The attribute is importable and callable.
    """

    def test_in_all(self):
        from spikelab.spike_sorting import stim_sorting

        assert "preprocess_stim_artifacts" in stim_sorting.__all__

    @skip_no_spikeinterface
    def test_importable(self):
        from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts

        assert callable(preprocess_stim_artifacts)


# ===========================================================================
# Fix 4: polynomial-fit divergence sanity clamp
# ===========================================================================


class TestPolynomialClamp:
    """
    Tests for the divergence sanity clamp in
    ``remove_stim_artifacts(method="polynomial")``.

    The clamp blanks any segment where the post-subtraction signal
    exceeds ``poly_clamp_factor * saturation_threshold`` and emits one
    warning per call.

    Tests:
        (Test Case 1) Well-behaved polynomial fits don't trigger the
            clamp and no warning is emitted.
        (Test Case 2) A divergent polynomial fit (forced by an
            extra-physiological residual) is blanked and a warning is
            emitted.
        (Test Case 3) ``poly_clamp_factor=None`` disables the clamp
            even on divergent fits.
        (Test Case 4) ``method="blank"`` doesn't engage the clamp.
        (Test Case 5) ``saturation_threshold=+inf`` (no clipping
            detected) disables the clamp.
    """

    @staticmethod
    def _make_traces(fs=20000.0, n_ch=4, dur_s=0.5, seed=0):
        rng = np.random.default_rng(seed)
        n_samp = int(dur_s * fs)
        return (rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 30.0), n_samp

    def test_clean_fit_no_warning(self):
        from spikelab.spike_sorting.stim_sorting import remove_stim_artifacts

        traces, n_samp = self._make_traces()
        # Smooth saturation step at 100 ms — cubic fit handles cleanly.
        s0 = int(0.1 * 20000.0)
        traces[:, s0 : s0 + 80] = 5000.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned, _ = remove_stim_artifacts(
                traces.copy(),
                np.array([100.0]),
                fs_Hz=20000.0,
                method="polynomial",
                artifact_window_ms=10.0,
                saturation_threshold=4500.0,
                poly_clamp_factor=10.0,
            )
        assert not any("polynomial fit diverged" in str(w.message) for w in caught)
        assert float(np.max(np.abs(cleaned))) < 45000.0

    def test_divergent_fit_clamped_and_warns(self):
        """Force divergence by stuffing a huge non-saturated outlier
        into the polynomial-fit window so the cubic over-shoots."""
        from spikelab.spike_sorting.stim_sorting import remove_stim_artifacts

        fs = 20000.0
        traces, n_samp = self._make_traces(fs=fs, dur_s=0.5)
        s0 = int(0.1 * fs)
        # Saturated tail (gets blanked) followed by a huge unsaturated
        # outlier inside the fit window — the polynomial fit will swing
        # wildly trying to follow it, producing residuals far above any
        # plausible neural amplitude.
        traces[:, s0 : s0 + 40] = 5000.0
        traces[:, s0 + 50 : s0 + 60] = -800000.0  # 800 mV outlier
        sat_thr = 4500.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned, blanked = remove_stim_artifacts(
                traces.copy(),
                np.array([100.0]),
                fs_Hz=fs,
                method="polynomial",
                artifact_window_ms=10.0,
                saturation_threshold=sat_thr,
                poly_clamp_factor=10.0,
            )
        assert any("polynomial fit diverged" in str(w.message) for w in caught), [
            str(w.message) for w in caught
        ]
        # Clamp must keep the segment-level cleaned amplitudes bounded.
        assert float(np.max(np.abs(cleaned))) <= 10.0 * sat_thr + 1.0
        # Blanked mask must include the divergent region.
        assert blanked[:, s0 : s0 + 60].any()

    def test_disable_clamp_via_none(self):
        from spikelab.spike_sorting.stim_sorting import remove_stim_artifacts

        fs = 20000.0
        traces, _ = self._make_traces(fs=fs, dur_s=0.5)
        s0 = int(0.1 * fs)
        traces[:, s0 : s0 + 40] = 5000.0
        traces[:, s0 + 50 : s0 + 60] = -800000.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            remove_stim_artifacts(
                traces.copy(),
                np.array([100.0]),
                fs_Hz=fs,
                method="polynomial",
                artifact_window_ms=10.0,
                saturation_threshold=4500.0,
                poly_clamp_factor=None,
            )
        assert not any("polynomial fit diverged" in str(w.message) for w in caught)

    def test_blank_method_not_affected(self):
        from spikelab.spike_sorting.stim_sorting import remove_stim_artifacts

        fs = 20000.0
        traces, _ = self._make_traces(fs=fs)
        s0 = int(0.1 * fs)
        traces[:, s0 : s0 + 40] = 5000.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            remove_stim_artifacts(
                traces.copy(),
                np.array([100.0]),
                fs_Hz=fs,
                method="blank",
                artifact_window_ms=10.0,
                saturation_threshold=4500.0,
                poly_clamp_factor=10.0,
            )
        assert not any("polynomial fit diverged" in str(w.message) for w in caught)

    def test_inf_saturation_disables_clamp(self):
        """When no clipping is detected, saturation_threshold = +inf
        means the clamp threshold is also +inf and nothing is blanked
        on divergence grounds."""
        from spikelab.spike_sorting.stim_sorting import remove_stim_artifacts

        fs = 20000.0
        traces, _ = self._make_traces(fs=fs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            remove_stim_artifacts(
                traces.copy(),
                np.array([100.0]),
                fs_Hz=fs,
                method="polynomial",
                artifact_window_ms=10.0,
                saturation_threshold=float("inf"),
                poly_clamp_factor=10.0,
            )
        assert not any("polynomial fit diverged" in str(w.message) for w in caught)


# ===========================================================================
# Fix 5: native MaxWell HDF5 loader for mxw v25.x
# ===========================================================================


@skip_no_spikeinterface
@skip_no_h5py
class TestMaxwellNativeLoader:
    """
    Tests for ``spikelab.spike_sorting.maxwell_io.load_maxwell_native``.

    Tests:
        (Test Case 1) Reads a synthetic single-well MaxWell HDF5 file
            and returns a single-segment BaseRecording with the
            expected channel count, sampling rate, and locations.
        (Test Case 2) Dedupes duplicate entries in settings/mapping
            (mxw v25.x bug) — the returned recording reflects the
            unique routed channels only.
        (Test Case 3) ``output_path`` produces a dumpable
            BinaryRecordingExtractor of the correct file size.
        (Test Case 4) ``list_maxwell_wells`` enumerates available
            well/rec pairs.
    """

    @staticmethod
    def _write_mxw_file(
        tmp_path: Path,
        n_unique_channels: int = 6,
        n_duplicates: int = 2,
        n_samples: int = 200,
        fs_Hz: float = 20000.0,
        lsb_volts: float = 6.29e-6,
    ) -> Path:
        import h5py

        path = tmp_path / "synthetic.raw.h5"
        rng = np.random.default_rng(0)
        raw = rng.integers(
            0, 2**14, size=(n_unique_channels, n_samples), dtype=np.uint16
        )
        routed = np.arange(n_unique_channels, dtype=np.uint16)
        # Mapping table: include n_duplicates extra rows for some
        # channels so the dedupe logic has work to do.
        mapping_dtype = np.dtype(
            [
                ("channel", np.int32),
                ("electrode", np.int32),
                ("x", np.float32),
                ("y", np.float32),
            ]
        )
        rows = []
        for c in range(n_unique_channels):
            rows.append((c, c * 10, c * 17.5, c * 7.5))
        for d in range(n_duplicates):
            rows.append((d, d * 10, d * 17.5, d * 7.5))
        mapping = np.array(rows, dtype=mapping_dtype)

        with h5py.File(path, "w") as f:
            grp = f.create_group("wells/well000/rec0000")
            grp.create_dataset("settings/mapping", data=mapping)
            grp.create_dataset("settings/sampling", data=np.array([fs_Hz]))
            grp.create_dataset("settings/lsb", data=np.array([lsb_volts]))
            grp.create_dataset("groups/routed/raw", data=raw)
            grp.create_dataset("groups/routed/channels", data=routed)
        return path

    def test_load_returns_baserecording(self, tmp_path):
        from spikeinterface.core import BaseRecording

        from spikelab.spike_sorting.maxwell_io import load_maxwell_native

        path = self._write_mxw_file(tmp_path, n_unique_channels=6)
        rec = load_maxwell_native(path)
        assert isinstance(rec, BaseRecording)
        assert rec.get_num_channels() == 6
        assert rec.get_num_segments() == 1
        assert rec.get_sampling_frequency() == 20000.0
        assert rec.get_channel_locations().shape == (6, 2)

    def test_dedupes_mapping_table(self, tmp_path):
        from spikelab.spike_sorting.maxwell_io import load_maxwell_native

        # Create a file whose mapping table has 6 unique IDs but 8 rows
        # (2 duplicates for channels 0 and 1) — neo would reject this.
        path = self._write_mxw_file(tmp_path, n_unique_channels=6, n_duplicates=2)
        rec = load_maxwell_native(path)
        # Returned recording reflects the routed_channels list (unique
        # by construction), not the inflated mapping table.
        assert rec.get_num_channels() == 6
        assert len(set(rec.get_channel_ids())) == 6

    def test_output_path_returns_binary_recording(self, tmp_path):
        from spikeinterface.core import BinaryRecordingExtractor

        from spikelab.spike_sorting.maxwell_io import load_maxwell_native

        path = self._write_mxw_file(tmp_path, n_unique_channels=4, n_samples=300)
        bin_path = tmp_path / "cleaned.dat"
        rec = load_maxwell_native(path, output_path=bin_path)
        assert isinstance(rec, BinaryRecordingExtractor)
        assert bin_path.exists()
        assert bin_path.stat().st_size == 4 * 300 * np.dtype("float32").itemsize
        # Dumpability is the whole reason BinaryRecording exists for
        # this loader — Docker-based sorters require it.
        rec.to_dict(recursive=True)

    def test_list_wells_enumerates(self, tmp_path):
        from spikelab.spike_sorting.maxwell_io import list_maxwell_wells

        path = self._write_mxw_file(tmp_path)
        pairs = list_maxwell_wells(path)
        assert pairs == [("well000", "rec0000")]


# ===========================================================================
# Fix 6: large-shift warning in recenter_stim_times
# ===========================================================================


class TestRecenterShiftWarning:
    """
    Tests for the median-offset warning in
    ``recenter_stim_times``.

    Tests:
        (Test Case 1) Small shifts (< warn_offset_ms) are silent.
        (Test Case 2) Large systematic shifts emit a UserWarning that
            names the median offset.
        (Test Case 3) ``warn_offset_ms=None`` silences the warning.
        (Test Case 4) Empty stim_times_ms is silent.
    """

    @staticmethod
    def _traces_with_artifacts_at_samples(
        artifact_samples, fs=20000.0, n_ch=4, dur_s=2.0
    ):
        rng = np.random.default_rng(2)
        n_samp = int(dur_s * fs)
        traces = rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 30.0
        for s in artifact_samples:
            lo = max(0, int(s) - 2)
            hi = min(n_samp, int(s) + 60)
            traces[:, lo:hi] = 5000.0
        return traces

    def test_small_shift_silent(self):
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        fs = 20000.0
        # Artifact 1 ms after the logged time — under the 3 ms default.
        artifact_samples = [int(0.101 * fs), int(0.501 * fs)]
        traces = self._traces_with_artifacts_at_samples(artifact_samples, fs=fs)
        logged_ms = np.array([100.0, 500.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recenter_stim_times(traces, logged_ms, fs_Hz=fs, max_offset_ms=20.0)
        assert not any("median |offset|" in str(w.message) for w in caught)

    def test_large_shift_warns(self):
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        fs = 20000.0
        # Logged times claim 100 ms / 500 ms but artifacts are at
        # 117 ms / 517 ms — a fixed +17 ms hardware delay.
        artifact_samples = [int(0.117 * fs), int(0.517 * fs)]
        traces = self._traces_with_artifacts_at_samples(artifact_samples, fs=fs)
        logged_ms = np.array([100.0, 500.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recenter_stim_times(traces, logged_ms, fs_Hz=fs, max_offset_ms=50.0)
        msgs = [str(w.message) for w in caught if "median |offset|" in str(w.message)]
        assert msgs, [str(w.message) for w in caught]

    def test_warn_offset_ms_none_silences(self):
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        fs = 20000.0
        artifact_samples = [int(0.117 * fs), int(0.517 * fs)]
        traces = self._traces_with_artifacts_at_samples(artifact_samples, fs=fs)
        logged_ms = np.array([100.0, 500.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recenter_stim_times(
                traces,
                logged_ms,
                fs_Hz=fs,
                max_offset_ms=50.0,
                warn_offset_ms=None,
            )
        assert not any("median |offset|" in str(w.message) for w in caught)

    def test_empty_stim_times_silent(self):
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        traces = self._traces_with_artifacts_at_samples([])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recenter_stim_times(traces, np.array([]), fs_Hz=20000.0, max_offset_ms=50.0)
        assert not any("median |offset|" in str(w.message) for w in caught)
