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


# ===========================================================================
# Multi-peak recentering (PR #126) — _multi_peak_anchor + multi_peak path
# ===========================================================================


class TestMultiPeakAnchor:
    """Direct unit tests for the private _multi_peak_anchor helper."""

    def test_empty_segment_returns_lo(self):
        """
        _multi_peak_anchor with hi <= lo (empty segment) returns lo.

        Tests:
            (Test Case 1) An empty search window short-circuits to lo.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        reference = np.zeros(100, dtype=np.float64)
        result = _multi_peak_anchor(
            reference,
            lo=50,
            hi=50,
            peak_mode="abs_max",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        assert result == 50

    def test_all_zero_window_falls_back_argmax_for_unsigned(self):
        """
        _multi_peak_anchor with an all-zero search window falls back
        to argmax/argmin per peak_mode.

        Tests:
            (Test Case 1) abs_max + all-zero -> argmax in the segment
                (returns lo since np.argmax of zeros is 0).
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        reference = np.zeros(100, dtype=np.float64)
        result = _multi_peak_anchor(
            reference,
            lo=10,
            hi=20,
            peak_mode="abs_max",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        # All zeros -> search.max() == 0 -> argmax fallback returns lo.
        assert result == 10

    def test_neg_peak_all_positive_window_falls_back_to_argmin(self):
        """
        _multi_peak_anchor with peak_mode='neg_peak' and an all-positive
        window degrades via argmin(segment).

        Tests:
            (Test Case 1) When the search signal -minimum(segment, 0)
                is identically zero (no negative samples), the helper
                falls back to argmin of the original segment.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        reference = np.array([1.0, 2.0, 3.0, 0.5, 4.0], dtype=np.float64)
        result = _multi_peak_anchor(
            reference,
            lo=0,
            hi=5,
            peak_mode="neg_peak",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        # argmin of the segment is index 3 (value 0.5).
        assert result == 3

    def test_first_vs_last_select_returns_different_pulses(self):
        """
        _multi_peak_anchor with multi_peak_select='first' vs 'last'
        returns the first vs the last pulse in a multi-pulse train.

        Tests:
            (Test Case 1) 'first' returns the index of the earliest pulse.
            (Test Case 2) 'last' returns the index of the latest pulse.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        # Three positive pulses at samples 100, 200, 300 in a 400-sample
        # window, pulse width ~5 samples, 5x noise floor.
        reference = np.zeros(400, dtype=np.float64)
        for center in (100, 200, 300):
            reference[center - 2 : center + 3] = 100.0

        first = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="pos_peak",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        last = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="pos_peak",
            multi_peak_select="last",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        assert first < last
        # The first chosen peak should be near sample 100, last near 300.
        assert abs(first - 100) <= 3
        assert abs(last - 300) <= 3

    def test_monotonic_ramp_no_interior_peak_falls_back(self):
        """
        _multi_peak_anchor on a monotonic ramp (no interior peaks above
        threshold) falls back to argmax/argmin.

        Tests:
            (Test Case 1) Linear ramp up + abs_max returns the last index
                (largest value) via argmax fallback.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        reference = np.linspace(0.0, 10.0, 50, dtype=np.float64)
        result = _multi_peak_anchor(
            reference,
            lo=0,
            hi=50,
            peak_mode="abs_max",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        # With no interior peaks, argmax fallback returns the global max
        # at index 49.
        assert result == 49


class TestRecenterStimTimesMultiPeak:
    """Tests for recenter_stim_times with multi_peak=True."""

    def test_invalid_multi_peak_select_raises(self):
        """
        recenter_stim_times with multi_peak=True and an invalid
        multi_peak_select value raises ValueError.

        Tests:
            (Test Case 1) multi_peak_select='middle' raises.
        """
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        traces = np.zeros((4, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="multi_peak_select"):
            recenter_stim_times(
                traces,
                np.array([10.0]),
                fs_Hz=20000.0,
                multi_peak=True,
                multi_peak_select="middle",
            )

    def test_invalid_multi_peak_threshold_raises(self):
        """
        recenter_stim_times with multi_peak=True and a threshold
        outside (0, 1] raises ValueError.

        Tests:
            (Test Case 1) multi_peak_threshold=0.0 raises.
            (Test Case 2) multi_peak_threshold=1.5 raises.
        """
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        traces = np.zeros((4, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="multi_peak_threshold"):
            recenter_stim_times(
                traces,
                np.array([10.0]),
                fs_Hz=20000.0,
                multi_peak=True,
                multi_peak_threshold=0.0,
            )
        with pytest.raises(ValueError, match="multi_peak_threshold"):
            recenter_stim_times(
                traces,
                np.array([10.0]),
                fs_Hz=20000.0,
                multi_peak=True,
                multi_peak_threshold=1.5,
            )

    def test_multi_peak_first_picks_first_pulse(self):
        """
        recenter_stim_times with multi_peak=True and select='first'
        aligns the corrected time to the first pulse in a 3-pulse train.

        Tests:
            (Test Case 1) The corrected time is closer to the first pulse
                than to the median or last pulse.
        """
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        fs = 20000.0
        n_ch = 4
        n_samp = int(0.5 * fs)  # 500 ms
        rng = np.random.default_rng(0)
        traces = rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 5.0
        # Three positive pulses (5 ms apart) starting at 100 ms.
        pulse_times_samples = [int(0.100 * fs), int(0.105 * fs), int(0.110 * fs)]
        for s in pulse_times_samples:
            traces[:, s : s + 8] = 1000.0

        # Logged stim time at the start of the train (within window).
        logged_ms = np.array([100.0])
        corrected = recenter_stim_times(
            traces,
            logged_ms,
            fs_Hz=fs,
            max_offset_ms=20.0,
            peak_mode="pos_peak",
            multi_peak=True,
            multi_peak_select="first",
            warn_offset_ms=None,
        )
        # First pulse onset is at 100 ms; corrected should be very close.
        assert abs(corrected[0] - 100.0) < 1.0

    def test_multi_peak_last_picks_last_pulse(self):
        """
        recenter_stim_times with multi_peak=True and select='last'
        aligns the corrected time to the last pulse in a 3-pulse train.

        Tests:
            (Test Case 1) The corrected time is closer to the last pulse
                than to the first.
        """
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        fs = 20000.0
        n_ch = 4
        n_samp = int(0.5 * fs)
        rng = np.random.default_rng(0)
        traces = rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 5.0
        pulse_times_samples = [int(0.100 * fs), int(0.105 * fs), int(0.110 * fs)]
        for s in pulse_times_samples:
            traces[:, s : s + 8] = 1000.0

        logged_ms = np.array([105.0])  # middle of the train
        corrected = recenter_stim_times(
            traces,
            logged_ms,
            fs_Hz=fs,
            max_offset_ms=20.0,
            peak_mode="pos_peak",
            multi_peak=True,
            multi_peak_select="last",
            warn_offset_ms=None,
        )
        # Last pulse onset at 110 ms; corrected should be near it.
        assert abs(corrected[0] - 110.0) < 1.0
        # And further from the first pulse than from the last.
        assert abs(corrected[0] - 110.0) < abs(corrected[0] - 100.0)


# ===========================================================================
# recording_io._patch_neo_maxwell_hdf5_plugin_path_handling
# ===========================================================================


class TestPatchNeoMaxwellHdf5PluginPathHandling:
    """
    Tests for the import-time monkey-patch of neo's Maxwell HDF5 plugin
    path handling.
    """

    def test_patch_runs_without_error(self):
        """
        _patch_neo_maxwell_hdf5_plugin_path_handling runs without error.

        Tests:
            (Test Case 1) Calling the patch is a no-op when neo is
                missing and idempotent when neo is present (no exception).
        """
        from spikelab.spike_sorting.recording_io import (
            _patch_neo_maxwell_hdf5_plugin_path_handling,
        )

        # Should not raise on any platform with or without neo installed.
        _patch_neo_maxwell_hdf5_plugin_path_handling()

    def test_patch_is_idempotent(self):
        """
        _patch_neo_maxwell_hdf5_plugin_path_handling can be called
        repeatedly without crashing or causing side-effects on
        subsequent calls.

        Tests:
            (Test Case 1) Two calls in a row produce the same end state
                (no exception, patched attribute remains set if neo is
                installed).
        """
        from spikelab.spike_sorting.recording_io import (
            _patch_neo_maxwell_hdf5_plugin_path_handling,
        )

        _patch_neo_maxwell_hdf5_plugin_path_handling()
        _patch_neo_maxwell_hdf5_plugin_path_handling()

        # If neo is installed, the auto_install_maxwell_hdf5_compression_plugin
        # attribute should now be patched.
        try:
            import neo.rawio.maxwellrawio as _mwrawio
        except ImportError:
            return  # neo missing -> patch is a no-op
        assert hasattr(_mwrawio, "auto_install_maxwell_hdf5_compression_plugin")
        assert callable(_mwrawio.auto_install_maxwell_hdf5_compression_plugin)


# ===========================================================================
# Multi-peak peak_mode x select matrix and parameter propagation
# ===========================================================================


def _make_multi_peak_traces(
    pulse_centers_samples,
    *,
    fs=20000.0,
    n_ch=4,
    dur_s=0.5,
    polarity="pos",
    pulse_width_samples=8,
    pulse_amp=1000.0,
    noise_std=5.0,
    seed=0,
):
    """Build (channels, samples) traces with synthetic pulses at given samples.

    Parameters:
        pulse_centers_samples: Sample indices of pulse onsets.
        polarity: ``"pos"`` (positive pulses), ``"neg"`` (negative pulses),
            or ``"alt"`` (alternating, biphasic-like).
    """
    rng = np.random.default_rng(seed)
    n_samp = int(dur_s * fs)
    traces = rng.standard_normal((n_ch, n_samp)).astype(np.float32) * noise_std
    for i, s in enumerate(pulse_centers_samples):
        if polarity == "pos":
            sign = 1.0
        elif polarity == "neg":
            sign = -1.0
        else:  # "alt"
            sign = 1.0 if i % 2 == 0 else -1.0
        lo = max(0, int(s))
        hi = min(n_samp, int(s) + pulse_width_samples)
        traces[:, lo:hi] = sign * pulse_amp
    return traces


class TestMultiPeakAnchorPeakModeMatrix:
    """
    Tests covering the (peak_mode x multi_peak_select) matrix in
    ``_multi_peak_anchor`` for the previously-untested combinations
    (``neg_peak``/``abs_max`` x ``first``/``last``).
    """

    def test_neg_peak_first_picks_earliest_negative_pulse(self):
        """
        With ``peak_mode='neg_peak'`` and ``select='first'``, the helper
        anchors on the earliest negative pulse in a 3-pulse train.

        Tests:
            (Test Case 1) ``-reference`` is searched; the first peak is
                near sample 100 in a 3-pulse train at 100/200/300.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        # Three negative-going pulses in a positive-noise reference.
        reference = np.zeros(400, dtype=np.float64)
        for center in (100, 200, 300):
            reference[center : center + 5] = -100.0

        result = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="neg_peak",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        assert abs(result - 100) <= 5

    def test_neg_peak_last_picks_latest_negative_pulse(self):
        """
        With ``peak_mode='neg_peak'`` and ``select='last'``, the helper
        anchors on the latest negative pulse in a 3-pulse train.

        Tests:
            (Test Case 1) The chosen peak is near the last (sample 300) pulse.
            (Test Case 2) It is strictly later than the first-select result.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        reference = np.zeros(400, dtype=np.float64)
        for center in (100, 200, 300):
            reference[center : center + 5] = -100.0

        first = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="neg_peak",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        last = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="neg_peak",
            multi_peak_select="last",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        assert last > first
        assert abs(last - 300) <= 5

    def test_abs_max_first_picks_earliest_pulse_regardless_of_polarity(self):
        """
        With ``peak_mode='abs_max'`` and ``select='first'``, the helper
        anchors on the earliest pulse regardless of polarity (the search
        signal is ``|reference|``).

        Tests:
            (Test Case 1) An alternating pos/neg/pos pulse train:
                the chosen anchor is near the first pulse.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        reference = np.zeros(400, dtype=np.float64)
        # Alternating polarity pulses; |reference| treats them all equally.
        reference[100:105] = 100.0
        reference[200:205] = -100.0
        reference[300:305] = 100.0

        result = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="abs_max",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        assert abs(result - 100) <= 5

    def test_abs_max_last_picks_latest_pulse_regardless_of_polarity(self):
        """
        With ``peak_mode='abs_max'`` and ``select='last'``, the helper
        anchors on the latest pulse regardless of polarity.

        Tests:
            (Test Case 1) On a pos/neg/pos train at 100/200/300, the
                last-select anchor is near sample 300.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        reference = np.zeros(400, dtype=np.float64)
        reference[100:105] = 100.0
        reference[200:205] = -100.0
        reference[300:305] = 100.0

        result = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="abs_max",
            multi_peak_select="last",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=2.0,
            fs_Hz=20000.0,
        )
        assert abs(result - 300) <= 5


class TestRecenterStimTimesMultiPeakPropagation:
    """
    Tests that ``recenter_stim_times`` forwards multi_peak parameters
    correctly into the underlying ``_multi_peak_anchor`` call.
    """

    def test_multi_peak_select_changes_recentered_time(self):
        """
        The same 3-pulse train recentered with ``select='first'`` vs
        ``select='last'`` produces different corrected times.

        Tests:
            (Test Case 1) ``select='first'`` gives the earliest pulse
                onset; ``select='last'`` gives the latest.
            (Test Case 2) The two values differ by approximately the
                inter-pulse interval (10 ms).
        """
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        fs = 20000.0
        pulse_samples = [int(0.100 * fs), int(0.105 * fs), int(0.110 * fs)]
        traces = _make_multi_peak_traces(
            pulse_samples,
            fs=fs,
            polarity="pos",
            pulse_amp=1000.0,
        )

        logged_ms = np.array([105.0])
        first = recenter_stim_times(
            traces,
            logged_ms,
            fs_Hz=fs,
            max_offset_ms=20.0,
            peak_mode="pos_peak",
            multi_peak=True,
            multi_peak_select="first",
            warn_offset_ms=None,
        )
        last = recenter_stim_times(
            traces,
            logged_ms,
            fs_Hz=fs,
            max_offset_ms=20.0,
            peak_mode="pos_peak",
            multi_peak=True,
            multi_peak_select="last",
            warn_offset_ms=None,
        )
        # First aligns near 100 ms; last near 110 ms; inter-pulse gap = 10 ms.
        assert abs(first[0] - 100.0) < 1.0
        assert abs(last[0] - 110.0) < 1.0
        assert (last[0] - first[0]) == pytest.approx(10.0, abs=1.0)

    def test_multi_peak_threshold_filters_weak_pulses(self):
        """
        A strong primary pulse and a weaker secondary pulse: with the
        default threshold (0.6) only the strong pulse qualifies.

        Tests:
            (Test Case 1) With a weak (10% amplitude) trailing pulse
                and threshold=0.6, the strong primary is selected by
                both 'first' and 'last' (no second qualifying peak).
            (Test Case 2) Lowering threshold to 0.05 admits the weak
                pulse — the 'last' result then differs from the first.
        """
        from spikelab.spike_sorting.stim_sorting import recenter_stim_times

        fs = 20000.0
        n_ch = 4
        n_samp = int(0.5 * fs)
        rng = np.random.default_rng(0)
        traces = rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 5.0
        # Strong pulse at 100 ms (1000 uV), weak pulse at 110 ms (100 uV).
        s_strong = int(0.100 * fs)
        s_weak = int(0.110 * fs)
        traces[:, s_strong : s_strong + 8] = 1000.0
        traces[:, s_weak : s_weak + 8] = 100.0

        logged_ms = np.array([105.0])

        # Tight threshold: only the strong pulse counts.
        tight = recenter_stim_times(
            traces,
            logged_ms,
            fs_Hz=fs,
            max_offset_ms=20.0,
            peak_mode="pos_peak",
            multi_peak=True,
            multi_peak_select="last",
            multi_peak_threshold=0.6,
            warn_offset_ms=None,
        )
        # Loose threshold: both pulses count → 'last' picks the weak one.
        loose = recenter_stim_times(
            traces,
            logged_ms,
            fs_Hz=fs,
            max_offset_ms=20.0,
            peak_mode="pos_peak",
            multi_peak=True,
            multi_peak_select="last",
            multi_peak_threshold=0.05,
            warn_offset_ms=None,
        )
        assert abs(tight[0] - 100.0) < 1.0
        assert abs(loose[0] - 110.0) < 1.5

    def test_multi_peak_min_separation_merges_close_pulses(self):
        """
        Two pulses spaced below the ``multi_peak_min_separation_ms``
        threshold are merged by ``find_peaks`` (which keeps the
        strongest within the distance window).

        Tests:
            (Test Case 1) Two equal-amplitude pulses 1 ms apart, with
                ``multi_peak_min_separation_ms=10.0``: the helper sees
                only one peak (find_peaks distance window suppresses
                the other), so 'first' and 'last' return the same
                anchor.
            (Test Case 2) The same pulses with separation=0.2 ms
                allow both peaks: 'last' anchors at the second pulse,
                which is later than 'first'.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _multi_peak_anchor,
        )

        fs = 20000.0
        # Two pulses 20 samples apart = 1 ms at 20 kHz.
        reference = np.zeros(400, dtype=np.float64)
        reference[100:103] = 100.0
        reference[120:123] = 100.0

        # Large min-separation (10 ms = 200 samples) → forces find_peaks
        # to keep at most one peak in the window, so first==last.
        merged_first = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="pos_peak",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=10.0,
            fs_Hz=fs,
        )
        merged_last = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="pos_peak",
            multi_peak_select="last",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=10.0,
            fs_Hz=fs,
        )
        assert merged_first == merged_last

        # Small min-separation (0.2 ms = 4 samples) → both peaks survive.
        split_first = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="pos_peak",
            multi_peak_select="first",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=0.2,
            fs_Hz=fs,
        )
        split_last = _multi_peak_anchor(
            reference,
            lo=0,
            hi=400,
            peak_mode="pos_peak",
            multi_peak_select="last",
            multi_peak_threshold=0.6,
            multi_peak_min_separation_ms=0.2,
            fs_Hz=fs,
        )
        assert split_last > split_first
        assert abs(split_first - 100) <= 2
        assert abs(split_last - 120) <= 2


# ===========================================================================
# Multi-peak chunked vs full-recording dispatch parity
# ===========================================================================


@skip_no_spikeinterface
class TestSortStimRecordingMultiPeakDispatch:
    """
    Tests that ``sort_stim_recording`` forwards the multi-peak parameters
    to the underlying ``recenter_stim_times`` call on both the chunked
    (BaseRecording) and full-recording (ndarray) dispatch paths.

    Tests use mocks to short-circuit RT-Sort itself — the goal is to
    verify the multi-peak parameter plumbing, not to run a real sort.
    """

    @staticmethod
    def _mock_rt_sort_obj():
        """A SimpleNamespace standing in for an RTSort with sort_offline."""
        from types import SimpleNamespace

        class _StubSorting:
            def get_unit_ids(self):
                return []

            def get_unit_spike_train(self, uid):
                return np.array([], dtype=np.int64)

            def get_num_samples(self):
                return 0

        ns = SimpleNamespace()
        ns.sort_offline = lambda recording, **kwargs: _StubSorting()
        return ns

    def test_full_recording_path_forwards_multi_peak_params(self, monkeypatch):
        """
        When ``stim_recording`` is an ndarray, the full-recording branch
        forwards ``multi_peak``, ``multi_peak_select``,
        ``multi_peak_threshold``, and ``multi_peak_min_separation_ms``
        to ``recenter_stim_times``.

        Tests:
            (Test Case 1) The captured kwargs match what was passed in.
        """
        from spikelab.spike_sorting.stim_sorting import pipeline as stim_pipeline

        captured = {}

        def fake_recenter(*args, **kwargs):
            captured.update(kwargs)
            # Return ms times corresponding to the input stim times.
            return np.asarray(args[1], dtype=np.float64)

        def fake_remove(*args, **kwargs):
            return args[0], np.zeros(args[0].shape, dtype=bool)

        monkeypatch.setattr(
            stim_pipeline, "_load_rt_sort", lambda *a, **kw: self._mock_rt_sort_obj()
        )
        # Patch the lazily-imported helpers *inside* the per-call import
        # by monkeypatching the source modules used by `from .recentering import recenter_stim_times`.
        from spikelab.spike_sorting.stim_sorting import recentering as _recentering
        from spikelab.spike_sorting.stim_sorting import artifact_removal as _ar

        monkeypatch.setattr(_recentering, "recenter_stim_times", fake_recenter)
        monkeypatch.setattr(_ar, "remove_stim_artifacts", fake_remove)

        traces = np.zeros((4, 20000), dtype=np.float32)
        stim_times_ms = np.array([100.0])
        stim_pipeline.sort_stim_recording(
            traces,
            rt_sort=object(),  # bypassed via _load_rt_sort patch
            stim_times_ms=stim_times_ms,
            pre_ms=10.0,
            post_ms=20.0,
            fs_Hz=20000.0,
            peak_mode="pos_peak",
            multi_peak=True,
            multi_peak_select="last",
            multi_peak_threshold=0.42,
            multi_peak_min_separation_ms=3.5,
            verbose=False,
        )

        assert captured.get("multi_peak") is True
        assert captured.get("multi_peak_select") == "last"
        assert captured.get("multi_peak_threshold") == pytest.approx(0.42)
        assert captured.get("multi_peak_min_separation_ms") == pytest.approx(3.5)
        assert captured.get("peak_mode") == "pos_peak"

    def test_chunked_path_forwards_multi_peak_params(self, monkeypatch):
        """
        When ``stim_recording`` is a BaseRecording, the chunked branch
        forwards multi-peak parameters to ``recenter_stim_times``.

        Tests:
            (Test Case 1) Captured kwargs match what was passed in.
        """
        from spikelab.spike_sorting.stim_sorting import pipeline as stim_pipeline

        captured = {}

        def fake_recenter(*args, **kwargs):
            captured.update(kwargs)
            return np.asarray(args[1], dtype=np.float64)

        def fake_remove(*args, **kwargs):
            return args[0], np.zeros(args[0].shape, dtype=bool)

        monkeypatch.setattr(
            stim_pipeline, "_load_rt_sort", lambda *a, **kw: self._mock_rt_sort_obj()
        )
        from spikelab.spike_sorting.stim_sorting import recentering as _recentering
        from spikelab.spike_sorting.stim_sorting import artifact_removal as _ar

        monkeypatch.setattr(_recentering, "recenter_stim_times", fake_recenter)
        monkeypatch.setattr(_ar, "remove_stim_artifacts", fake_remove)

        # Build a BaseRecording (NumpyRecording), 1 s of synthetic data.
        rec = _make_numpy_recording(num_samples=20000, num_channels=4, fs=20000.0)
        stim_times_ms = np.array([500.0])
        stim_pipeline.sort_stim_recording(
            rec,
            rt_sort=object(),
            stim_times_ms=stim_times_ms,
            pre_ms=20.0,
            post_ms=20.0,
            peak_mode="abs_max",
            multi_peak=True,
            multi_peak_select="first",
            multi_peak_threshold=0.55,
            multi_peak_min_separation_ms=4.5,
            verbose=False,
        )

        assert captured.get("multi_peak") is True
        assert captured.get("multi_peak_select") == "first"
        assert captured.get("multi_peak_threshold") == pytest.approx(0.55)
        assert captured.get("multi_peak_min_separation_ms") == pytest.approx(4.5)
        assert captured.get("peak_mode") == "abs_max"


# ===========================================================================
# _find_down_edge / _find_up_edge with caller-supplied neg_peak / pos_peak
# (the multi-peak recentering anchor-override path).
# ===========================================================================


class TestFindEdgeWithExplicitAnchor:
    """
    Tests for ``_find_down_edge(neg_peak=...)`` and
    ``_find_up_edge(pos_peak=...)`` — the optional caller-supplied
    anchor used by the multi-peak recentering helper.
    """

    def _make_biphasic_pulse(
        self,
        n_samples: int,
        pulse_centers: list[int],
        polarity: str = "down",
        amp: float = 50.0,
        half_width: int = 5,
    ) -> np.ndarray:
        """Synthesize a reference trace with one or more biphasic pulses.

        Each pulse has a positive lobe followed by a negative lobe (for
        ``polarity='down'``) or the inverse (``polarity='up'``).  The
        sample at ``center`` carries the trough/peak amplitude of the
        second lobe; the two lobes are joined by a zero-crossing one
        sample before the trough.
        """
        ref = np.zeros(n_samples, dtype=np.float64)
        for c in pulse_centers:
            for k in range(-half_width, half_width + 1):
                idx = c + k
                if idx < 0 or idx >= n_samples:
                    continue
                # Positive lobe to the left of c, negative lobe at and
                # to the right (or inverted, for 'up').
                if polarity == "down":
                    val = amp if k < 0 else -amp
                else:
                    val = -amp if k < 0 else amp
                ref[idx] = val
        return ref

    def test_find_down_edge_uses_supplied_neg_peak(self):
        """
        ``_find_down_edge`` with an explicit ``neg_peak`` overrides the
        internal ``argmin`` and computes the zero-crossing relative to
        the caller's anchor.

        Tests:
            (Test Case 1) Two pulses in window: default argmin picks
                the larger-amplitude pulse; supplying ``neg_peak`` of
                the smaller pulse moves the returned edge to that
                pulse's zero-crossing.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_down_edge,
        )

        # Two negative-going pulses; second is larger so argmin picks it.
        ref = self._make_biphasic_pulse(
            200, pulse_centers=[60, 130], polarity="down", amp=50.0
        )
        ref[125:131] *= 1.5  # boost the second pulse so argmin prefers it

        default_edge = _find_down_edge(
            ref, lo=0, hi=200, prewindow_ms=1.0, fs_Hz=20000.0
        )
        # Override: anchor on the FIRST pulse's negative peak (sample 60).
        override_edge = _find_down_edge(
            ref, lo=0, hi=200, prewindow_ms=1.0, fs_Hz=20000.0, neg_peak=60
        )
        # Default attaches to the second pulse near sample 130.
        assert default_edge >= 120
        # Override attaches near the first pulse's transition (~sample 60).
        assert 50 <= override_edge < 70
        # Different from the default — the anchor really did shift it.
        assert override_edge != default_edge

    def test_find_up_edge_uses_supplied_pos_peak(self):
        """
        ``_find_up_edge`` with an explicit ``pos_peak`` overrides the
        internal ``argmax`` and computes the zero-crossing relative
        to the caller's anchor.

        Tests:
            (Test Case 1) Two positive-going pulses in window: an
                explicit ``pos_peak`` on the smaller pulse shifts the
                returned edge to that pulse.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_up_edge,
        )

        ref = self._make_biphasic_pulse(
            200, pulse_centers=[60, 130], polarity="up", amp=50.0
        )
        ref[125:131] *= 1.5  # boost the second pulse so argmax prefers it

        default_edge = _find_up_edge(ref, lo=0, hi=200, prewindow_ms=1.0, fs_Hz=20000.0)
        override_edge = _find_up_edge(
            ref, lo=0, hi=200, prewindow_ms=1.0, fs_Hz=20000.0, pos_peak=60
        )
        assert default_edge >= 120
        assert 50 <= override_edge < 70
        assert override_edge != default_edge

    def test_find_down_edge_anchor_at_lo_returns_anchor(self):
        """
        ``_find_down_edge`` with ``neg_peak == lo`` produces an empty
        pre-window (``pre_hi <= pre_lo``) and falls through to the
        early-return branch, returning the supplied anchor verbatim.

        Tests:
            (Test Case 1) ``neg_peak=lo`` returns ``lo`` regardless of
                the surrounding signal.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_down_edge,
        )

        ref = np.linspace(-1.0, 1.0, 100, dtype=np.float64)
        result = _find_down_edge(
            ref, lo=20, hi=80, prewindow_ms=1.0, fs_Hz=20000.0, neg_peak=20
        )
        assert result == 20

    def test_find_up_edge_anchor_at_lo_returns_anchor(self):
        """
        ``_find_up_edge`` with ``pos_peak == lo`` produces an empty
        pre-window and returns the anchor verbatim.

        Tests:
            (Test Case 1) ``pos_peak=lo`` returns ``lo``.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_up_edge,
        )

        ref = np.linspace(1.0, -1.0, 100, dtype=np.float64)
        result = _find_up_edge(
            ref, lo=20, hi=80, prewindow_ms=1.0, fs_Hz=20000.0, pos_peak=20
        )
        assert result == 20

    def test_find_down_edge_anchor_near_hi_does_not_crash(self):
        """
        ``_find_down_edge`` with ``neg_peak`` near (but inside)
        ``hi`` continues to look for a zero-crossing in the
        ``[pos_peak, neg_peak+1)`` segment without indexing past
        the end of ``reference``.

        Tests:
            (Test Case 1) ``neg_peak = hi - 1`` returns a sample in
                ``[lo, hi]`` without raising IndexError.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_down_edge,
        )

        ref = self._make_biphasic_pulse(
            200, pulse_centers=[60], polarity="down", amp=30.0
        )
        # Anchor at hi-1; zero-crossing must still be inside [lo, hi].
        result = _find_down_edge(
            ref, lo=0, hi=70, prewindow_ms=1.0, fs_Hz=20000.0, neg_peak=69
        )
        assert 0 <= result <= 69

    def test_find_up_edge_anchor_near_hi_does_not_crash(self):
        """
        ``_find_up_edge`` with ``pos_peak`` near ``hi`` does not
        crash when computing the segment ``[neg_peak, pos_peak+1)``.

        Tests:
            (Test Case 1) ``pos_peak = hi - 1`` returns a sample in
                ``[lo, hi]`` without raising IndexError.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_up_edge,
        )

        ref = self._make_biphasic_pulse(
            200, pulse_centers=[60], polarity="up", amp=30.0
        )
        result = _find_up_edge(
            ref, lo=0, hi=70, prewindow_ms=1.0, fs_Hz=20000.0, pos_peak=69
        )
        assert 0 <= result <= 69

    def test_find_down_edge_default_vs_explicit_match_when_equal(self):
        """
        ``_find_down_edge`` with ``neg_peak`` set to the same value
        the default branch would compute returns identical results.

        Tests:
            (Test Case 1) Single-pulse trace: default and explicit
                anchor produce the same edge.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_down_edge,
        )

        ref = self._make_biphasic_pulse(
            200, pulse_centers=[100], polarity="down", amp=50.0
        )
        default_neg_peak = int(np.argmin(ref[0:200]))
        default_edge = _find_down_edge(
            ref, lo=0, hi=200, prewindow_ms=1.0, fs_Hz=20000.0
        )
        explicit_edge = _find_down_edge(
            ref,
            lo=0,
            hi=200,
            prewindow_ms=1.0,
            fs_Hz=20000.0,
            neg_peak=default_neg_peak,
        )
        assert default_edge == explicit_edge


# ===========================================================================
# Polynomial detrend NaN-coefficient propagation
# ===========================================================================


class TestPolyfitAndSubtractNanCoefficients:
    """
    Tests for ``_polyfit_and_subtract``'s behaviour when
    ``np.polyfit`` returns NaN coefficients (e.g. from a singular
    linear system or NaN-contaminated upstream data).

    Current behaviour (pinned to flag the gap):
      - ``np.polyval(NaN_coeffs, x)`` yields NaN-valued samples.
      - In-place subtraction propagates NaN into ``channel_trace``.
      - The clamp guard ``float(np.max(np.abs(seg))) > clamp_threshold``
        evaluates ``NaN > threshold == False``, so the segment is NOT
        blanked and the clamp counter is NOT incremented.
      - As a result, NaN-corrupted samples are shipped downstream
        (silent data corruption flagged in REVIEW.md).
    """

    def test_nan_coefficients_propagate_through_segment(self, monkeypatch):
        """
        With ``np.polyfit`` patched to return NaN coefficients,
        ``_polyfit_and_subtract`` writes NaN samples into
        ``channel_trace[lo:hi]``.

        Tests:
            (Test Case 1) After the call, the post-fit segment is
                all-NaN.
            (Test Case 2) Samples outside [lo, hi) are unchanged.

        Notes:
            - This documents a bug. The recommended fix is to detect
              NaN/inf in the post-subtraction segment, blank in that
              case, and increment the clamp counter; the test should
              be updated to assert blanking once the source is hardened.
        """
        from spikelab.spike_sorting.stim_sorting import (
            artifact_removal as ar_mod,
        )

        n_samples = 200
        channel_trace = np.linspace(-1.0, 1.0, n_samples).astype(np.float64)
        original_outside = channel_trace[:50].copy()
        blanked = np.zeros((1, n_samples), dtype=bool)
        clamp_counter = [0]

        def fake_polyfit(x, y, deg):
            return np.full(deg + 1, np.nan, dtype=np.float64)

        monkeypatch.setattr(ar_mod.np, "polyfit", fake_polyfit)

        ar_mod._polyfit_and_subtract(
            channel_trace=channel_trace,
            blanked=blanked,
            ch_idx=0,
            lo=50,
            hi=150,
            poly_order=3,
            clamp_threshold=1000.0,
            clamp_counter=clamp_counter,
        )

        # Pinned: post-fit segment is all-NaN.
        assert np.all(np.isnan(channel_trace[50:150]))
        # Outside the window is unchanged.
        np.testing.assert_array_equal(channel_trace[:50], original_outside)

    def test_nan_segment_not_blanked_by_clamp_check(self, monkeypatch):
        """
        With NaN coefficients, the segment-level clamp check
        (``float(np.max(np.abs(seg))) > clamp_threshold``) is False
        because every comparison against NaN is False. The segment
        is therefore NOT blanked and the clamp counter is NOT
        incremented — the silent-corruption path.

        Tests:
            (Test Case 1) ``clamp_counter[0]`` is still 0.
            (Test Case 2) ``blanked[ch_idx, lo:hi]`` is still False
                for all samples (the bug).

        Notes:
            - Pins current behaviour. When the source is hardened
              to detect NaN, this test must be updated to assert
              that the segment IS blanked and the counter IS
              incremented.
        """
        from spikelab.spike_sorting.stim_sorting import (
            artifact_removal as ar_mod,
        )

        n_samples = 200
        channel_trace = np.zeros(n_samples, dtype=np.float64)
        blanked = np.zeros((1, n_samples), dtype=bool)
        clamp_counter = [0]

        monkeypatch.setattr(
            ar_mod.np,
            "polyfit",
            lambda x, y, deg: np.full(deg + 1, np.nan, dtype=np.float64),
        )

        ar_mod._polyfit_and_subtract(
            channel_trace=channel_trace,
            blanked=blanked,
            ch_idx=0,
            lo=20,
            hi=80,
            poly_order=3,
            clamp_threshold=10.0,
            clamp_counter=clamp_counter,
        )

        # Bug: NaN propagates but the segment is not blanked.
        assert clamp_counter[0] == 0
        assert not blanked[0, 20:80].any()
        # Confirm the NaN actually landed (the bug premise).
        assert np.all(np.isnan(channel_trace[20:80]))

    def test_finite_clamp_threshold_with_nan_segment_is_no_op(self, monkeypatch):
        """
        Even with a generous (small) ``clamp_threshold``, a NaN
        segment is not detected as out-of-bounds. Confirms the bug
        does not depend on the threshold magnitude.

        Tests:
            (Test Case 1) clamp_threshold=0.0 with NaN segment:
                still no blanking, still no counter increment.
        """
        from spikelab.spike_sorting.stim_sorting import (
            artifact_removal as ar_mod,
        )

        n_samples = 100
        channel_trace = np.zeros(n_samples, dtype=np.float64)
        blanked = np.zeros((1, n_samples), dtype=bool)
        clamp_counter = [0]

        monkeypatch.setattr(
            ar_mod.np,
            "polyfit",
            lambda x, y, deg: np.full(deg + 1, np.nan, dtype=np.float64),
        )

        ar_mod._polyfit_and_subtract(
            channel_trace=channel_trace,
            blanked=blanked,
            ch_idx=0,
            lo=10,
            hi=60,
            poly_order=3,
            clamp_threshold=0.0,
            clamp_counter=clamp_counter,
        )

        assert clamp_counter[0] == 0
        assert not blanked[0, 10:60].any()
        assert np.all(np.isnan(channel_trace[10:60]))
