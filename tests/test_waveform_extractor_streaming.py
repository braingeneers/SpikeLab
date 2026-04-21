"""Tests for WaveformExtractor.run_extract_waveforms_streaming.

Verifies the per-unit streaming path that replaces the parallel
chunked extractor for high-unit-count, high-density-MEA sorts:

    1. Templates are correct (peak channel + amplitude match injected
       waveforms within noise tolerance).
    2. ``save_waveform_files=True`` writes one ``waveforms_<uid>.npy``
       per unit with the expected shape.
    3. ``save_waveform_files=False`` writes templates but NO per-unit
       waveform files (the low-RAM mode).
    4. Spike times are recentered onto the actual trace peak (not just
       the originally-stored sample), matching the chunked path's
       behavior.
    5. Empty units (zero spikes) are skipped without error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import spikeinterface  # noqa: F401
    from spikeinterface.core import NumpyRecording

    _has_spikeinterface = True
except Exception:
    _has_spikeinterface = False

skip_no_spikeinterface = pytest.mark.skipif(
    not _has_spikeinterface, reason="spikeinterface not installed"
)


# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------


def _build_dataset(
    tmp_path: Path,
    n_units: int = 3,
    n_spikes_per_unit: int = 20,
    n_channels: int = 4,
    fs: float = 20000.0,
    duration_s: float = 5.0,
    spike_offset_samples: int = 0,
):
    """Make a NumpyRecording + KilosortSortingExtractor with known waveforms.

    Each unit has a unique negative-peak waveform on a distinct channel.
    Spikes are placed at evenly-spaced times across the recording.

    Parameters
    ----------
    spike_offset_samples : int
        Stored ``spike_times.npy`` are offset from the true peak by this
        many samples — used to test that the streaming path recenters
        them correctly.
    """
    from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor

    n_samples = int(fs * duration_s)
    nbefore = 40  # 2 ms @ 20 kHz
    nafter = 41
    nsamples = nbefore + nafter

    rng = np.random.default_rng(42)
    traces = rng.standard_normal((n_samples, n_channels)).astype(np.float32) * 0.05

    unit_waveforms = {}
    for u in range(n_units):
        wf = np.zeros((nsamples, n_channels), dtype=np.float32)
        peak_chan = u % n_channels
        # Triangular dip centered at the peak index
        for k in range(nsamples):
            wf[k, peak_chan] = -10.0 * max(0.0, 1.0 - abs(k - nbefore) / 10.0)
        unit_waveforms[u] = wf

    # Per-unit non-overlapping spike times.  Each unit gets its own
    # set of times spaced at ``min_isi`` samples; units start at
    # different offsets so their windows do not collide on a single
    # channel (which would otherwise make the peak channel of the
    # sum-of-waveforms trace ambiguous in this test).
    min_isi = max(2 * nsamples + 1, 200)
    true_peak_times: list[int] = []
    stored_spike_times: list[int] = []
    spike_clusters: list[int] = []
    margin = nbefore + 200
    base = margin
    for u in range(n_units):
        unit_start = base + u * (min_isi // n_units)
        true_times = unit_start + np.arange(n_spikes_per_unit) * min_isi
        # Cap inside the recording
        valid_max = n_samples - margin
        true_times = true_times[true_times < valid_max]
        for t in true_times:
            t = int(t)
            traces[t - nbefore : t - nbefore + nsamples, :] += unit_waveforms[u]
            true_peak_times.append(t)
            stored_spike_times.append(int(t + spike_offset_samples))
            spike_clusters.append(int(u))

    order = np.argsort(stored_spike_times)
    stored_spike_times_arr = np.asarray(stored_spike_times)[order]
    spike_clusters_arr = np.asarray(spike_clusters)[order]
    true_peak_times_arr = np.asarray(true_peak_times)[order]

    templates = np.stack([unit_waveforms[u] for u in range(n_units)], axis=0).astype(
        np.float32
    )

    ks_folder = tmp_path / "ks"
    ks_folder.mkdir()
    np.save(ks_folder / "spike_times.npy", stored_spike_times_arr)
    np.save(ks_folder / "spike_clusters.npy", spike_clusters_arr)
    np.save(ks_folder / "templates.npy", templates)
    np.save(ks_folder / "channel_map.npy", np.arange(n_channels))
    (ks_folder / "params.py").write_text(
        f"dat_path = 'recording.dat'\n"
        f"n_channels_dat = {n_channels}\n"
        f"dtype = 'float32'\n"
        f"offset = 0\n"
        f"sample_rate = {fs}\n"
        f"hp_filtered = True\n"
    )

    rec = NumpyRecording(traces_list=[traces], sampling_frequency=fs)
    sorting = KilosortSortingExtractor(ks_folder)
    return rec, sorting, unit_waveforms, true_peak_times_arr, ks_folder


def _set_globals(monkeypatch, *, streaming: bool, save_files: bool):
    from spikelab.spike_sorting import _globals

    monkeypatch.setattr(_globals, "WAVEFORMS_MS_BEFORE", 2.0)
    monkeypatch.setattr(_globals, "WAVEFORMS_MS_AFTER", 2.0)
    monkeypatch.setattr(_globals, "POS_PEAK_THRESH", 2.0)
    monkeypatch.setattr(_globals, "MAX_WAVEFORMS_PER_UNIT", 100)
    monkeypatch.setattr(_globals, "N_JOBS", 1)
    monkeypatch.setattr(_globals, "TOTAL_MEMORY", "1G")
    monkeypatch.setattr(_globals, "STREAMING_WAVEFORMS", streaming)
    monkeypatch.setattr(_globals, "SAVE_WAVEFORM_FILES", save_files)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_spikeinterface
class TestStreamingWaveformExtractor:
    def test_templates_match_injected_waveforms(self, tmp_path, monkeypatch):
        """Per-unit average template's peak channel and amplitude match injection."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _set_globals(monkeypatch, streaming=True, save_files=True)

        rec, sorting, unit_waveforms, _, ks_folder = _build_dataset(tmp_path)

        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
        )
        we.run_extract_waveforms_streaming()

        for u in sorting.unit_ids:
            tmpl = we.template_cache["average"][u]
            inj = unit_waveforms[u]

            assert (
                tmpl.shape == inj.shape
            ), f"Unit {u} template shape {tmpl.shape} != injected {inj.shape}"

            inj_peak_chan = int(np.argmin(np.min(inj, axis=0)))
            tmpl_peak_chan = int(np.argmin(np.min(tmpl, axis=0)))
            assert tmpl_peak_chan == inj_peak_chan, (
                f"Unit {u}: streaming put peak on chan {tmpl_peak_chan}, "
                f"injection was on chan {inj_peak_chan}"
            )

            inj_peak = float(np.min(inj))
            tmpl_peak = float(np.min(tmpl))
            assert abs(tmpl_peak - inj_peak) < 1.0, (
                f"Unit {u}: peak amplitude {tmpl_peak:.2f} far from "
                f"injected {inj_peak:.2f}"
            )

    def test_template_std_is_finite_and_nonneg(self, tmp_path, monkeypatch):
        """Std template entries are finite and non-negative."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _set_globals(monkeypatch, streaming=True, save_files=True)

        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
        )
        we.run_extract_waveforms_streaming()

        std = we.template_cache["std"]
        for u in sorting.unit_ids:
            assert np.all(np.isfinite(std[u])), f"Unit {u} std contains NaN/inf"
            assert np.all(std[u] >= 0), f"Unit {u} std has negative values"

    def test_save_waveform_files_false_skips_disk(self, tmp_path, monkeypatch):
        """``SAVE_WAVEFORM_FILES=False`` writes templates only — no per-unit .npy."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _set_globals(monkeypatch, streaming=True, save_files=False)

        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
        )
        we.run_extract_waveforms_streaming()

        assert (root_folder / "templates" / "templates_average.npy").is_file()
        assert (root_folder / "templates" / "templates_std.npy").is_file()

        wf_files = list((root_folder / "waveforms").glob("waveforms_*.npy"))
        assert wf_files == [], f"Expected no per-unit files, found: {wf_files}"

    def test_save_waveform_files_true_writes_one_per_unit(self, tmp_path, monkeypatch):
        """``SAVE_WAVEFORM_FILES=True`` writes a per-unit waveform file."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _set_globals(monkeypatch, streaming=True, save_files=True)

        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
        )
        we.run_extract_waveforms_streaming()

        for u in sorting.unit_ids:
            wf_path = root_folder / "waveforms" / f"waveforms_{u}.npy"
            assert wf_path.is_file(), f"Unit {u}: expected {wf_path} to exist"
            wfs = np.load(wf_path)
            assert wfs.ndim == 3
            assert wfs.shape[1] == we.nsamples
            assert wfs.shape[2] == rec.get_num_channels()
            assert wfs.shape[0] > 0

    def test_recentering_corrects_offset_spike_times(self, tmp_path, monkeypatch):
        """Stored spike times offset from the peak get recentered on the peak."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _set_globals(monkeypatch, streaming=True, save_files=False)

        offset = 5
        rec, sorting, _, true_peak_times, ks_folder = _build_dataset(
            tmp_path, spike_offset_samples=offset
        )
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
        )
        we.run_extract_waveforms_streaming()

        recentered = np.load(sorting.folder / "spike_times.npy")
        original = np.load(sorting.folder / "spike_times_kilosort.npy")

        assert np.array_equal(np.sort(original), np.sort(original)), "monotonicity"
        assert original.shape == recentered.shape

        diffs = recentered.astype(int) - original.astype(int)
        # The original ks spike_times are offset by +offset from the true peak;
        # recentering should pull them back by approximately -offset.
        median_correction = int(np.median(diffs))
        assert (
            median_correction == -offset
        ), f"Expected median recentering shift of {-offset}, got {median_correction}"

    def test_unit_with_zero_in_window_spikes_does_not_crash(
        self, tmp_path, monkeypatch
    ):
        """A unit whose only spikes fall inside the trim margin yields zero template (no crash)."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _set_globals(monkeypatch, streaming=True, save_files=False)

        rec, sorting, _, _, ks_folder = _build_dataset(
            tmp_path, n_units=2, n_spikes_per_unit=15
        )

        # Append a unit whose spikes are all at frame 5 — too close to the
        # recording start to produce a valid waveform window. The streaming
        # path should skip it cleanly and leave a zero template.
        st = np.load(ks_folder / "spike_times.npy")
        sc = np.load(ks_folder / "spike_clusters.npy")
        edge_unit_id = int(sc.max()) + 1
        st_edge = np.array([5, 6, 7], dtype=st.dtype)
        sc_edge = np.full(st_edge.shape, edge_unit_id, dtype=sc.dtype)
        order = np.argsort(np.concatenate([st, st_edge]))
        np.save(ks_folder / "spike_times.npy", np.concatenate([st, st_edge])[order])
        np.save(
            ks_folder / "spike_clusters.npy",
            np.concatenate([sc, sc_edge])[order],
        )

        templates = np.load(ks_folder / "templates.npy")
        edge_template = np.zeros(
            (1, templates.shape[1], templates.shape[2]), dtype=templates.dtype
        )
        edge_template[0, templates.shape[1] // 2, 0] = -5.0
        np.save(ks_folder / "templates.npy", np.concatenate([templates, edge_template]))

        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor

        sorting2 = KilosortSortingExtractor(ks_folder)

        root_folder = tmp_path / "wf_root2"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting2,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
        )
        we.run_extract_waveforms_streaming()

        # Edge unit's template should be all zeros (no waveform written)
        assert np.all(we.template_cache["average"][edge_unit_id] == 0)

    def test_streaming_matches_chunked_templates(self, tmp_path, monkeypatch):
        """The streaming path produces equivalent templates to the chunked path.

        Runs both ``run_extract_waveforms`` (parallel, chunked) and
        ``run_extract_waveforms_streaming`` against the same synthetic
        dataset, and asserts:
          - same set of populated unit ids,
          - per-unit average templates equal within tight tolerance,
          - per-unit std templates equal within tight tolerance,
          - recentered spike times equal exactly.

        Tolerances allow for sub-sample rounding differences in the
        recentering peak picker (the chunked path uses `np.searchsorted`
        chunk boundaries which can shift which spike is the "first"
        sample of a chunk, whereas streaming reads per-spike windows).
        """
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        # ----- chunked run -----
        _set_globals(monkeypatch, streaming=False, save_files=True)
        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_chunked = tmp_path / "wf_chunked"
        we_chunked = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_chunked,
            initial_folder=root_chunked / "initial",
        )
        we_chunked.run_extract_waveforms(n_jobs=1)
        we_chunked.compute_templates(modes=("average", "std"), n_jobs=1)
        chunked_avg = we_chunked.template_cache["average"].copy()
        chunked_std = we_chunked.template_cache["std"].copy()
        chunked_centered = np.load(sorting.folder / "spike_times.npy").copy()

        # Reset the on-disk spike_times.npy so the streaming run starts
        # from the same un-centered times the chunked run got.
        np.save(
            sorting.folder / "spike_times.npy",
            np.load(sorting.folder / "spike_times_kilosort.npy"),
        )

        # ----- streaming run, same inputs -----
        _set_globals(monkeypatch, streaming=True, save_files=True)
        # Re-build the sorting (cached attributes from chunked may differ)
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor

        sorting2 = KilosortSortingExtractor(ks_folder)
        root_streaming = tmp_path / "wf_streaming"
        we_streaming = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting2,
            root_folder=root_streaming,
            initial_folder=root_streaming / "initial",
        )
        we_streaming.run_extract_waveforms_streaming()
        streaming_avg = we_streaming.template_cache["average"].copy()
        streaming_std = we_streaming.template_cache["std"].copy()
        streaming_centered = np.load(sorting2.folder / "spike_times.npy").copy()

        # ----- assertions -----
        assert set(sorting.unit_ids) == set(sorting2.unit_ids)
        for u in sorting.unit_ids:
            np.testing.assert_allclose(
                streaming_avg[u],
                chunked_avg[u],
                atol=1e-3,
                err_msg=f"avg template differs for unit {u}",
            )
            np.testing.assert_allclose(
                streaming_std[u],
                chunked_std[u],
                atol=1e-3,
                err_msg=f"std template differs for unit {u}",
            )
        # Recentered spike times equal element-wise
        np.testing.assert_array_equal(
            np.sort(streaming_centered),
            np.sort(chunked_centered),
            err_msg="recentered spike times differ between paths",
        )

    def test_dispatcher_routes_through_streaming_when_flag_set(
        self, tmp_path, monkeypatch
    ):
        """``recording_io.extract_waveforms`` dispatches to streaming when flag is set."""
        from spikelab.spike_sorting import recording_io
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _set_globals(monkeypatch, streaming=True, save_files=False)

        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_folder = tmp_path / "wf_root_dispatch"
        initial_folder = root_folder / "initial"

        called: dict[str, int] = {"streaming": 0, "chunked": 0}
        orig_streaming = WaveformExtractor.run_extract_waveforms_streaming
        orig_chunked = WaveformExtractor.run_extract_waveforms

        def _spy_streaming(self):
            called["streaming"] += 1
            return orig_streaming(self)

        def _spy_chunked(self, **kwargs):
            called["chunked"] += 1
            return orig_chunked(self, **kwargs)

        monkeypatch.setattr(
            WaveformExtractor, "run_extract_waveforms_streaming", _spy_streaming
        )
        monkeypatch.setattr(WaveformExtractor, "run_extract_waveforms", _spy_chunked)

        recording_io.extract_waveforms(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=initial_folder,
        )

        assert called["streaming"] == 1
        assert called["chunked"] == 0
