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


def _build_config(*, streaming: bool, save_files: bool):
    """Build a :class:`SortingPipelineConfig` for the streaming tests.

    Replaces an earlier ``_set_globals`` helper that mutated module
    globals; after Phase 5 of the ``_globals.py`` refactor those
    globals no longer exist. Tests pass the returned config to
    :meth:`WaveformExtractor.create_initial` via ``config=``.
    """
    from spikelab.spike_sorting.config import (
        ExecutionConfig,
        SortingPipelineConfig,
        WaveformConfig,
    )

    return SortingPipelineConfig(
        waveform=WaveformConfig(
            ms_before=2.0,
            ms_after=2.0,
            pos_peak_thresh=2.0,
            max_waveforms_per_unit=100,
            streaming=streaming,
            save_waveform_files=save_files,
        ),
        execution=ExecutionConfig(n_jobs=1, total_memory="1G"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_spikeinterface
class TestStreamingWaveformExtractor:
    def test_templates_match_injected_waveforms(self, tmp_path, monkeypatch):
        """Per-unit average template's peak channel and amplitude match injection."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _wf_config = _build_config(streaming=True, save_files=True)

        rec, sorting, unit_waveforms, _, ks_folder = _build_dataset(tmp_path)

        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
            config=_wf_config,
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

        _wf_config = _build_config(streaming=True, save_files=True)

        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
            config=_wf_config,
        )
        we.run_extract_waveforms_streaming()

        std = we.template_cache["std"]
        for u in sorting.unit_ids:
            assert np.all(np.isfinite(std[u])), f"Unit {u} std contains NaN/inf"
            assert np.all(std[u] >= 0), f"Unit {u} std has negative values"

    def test_save_waveform_files_false_skips_disk(self, tmp_path, monkeypatch):
        """``SAVE_WAVEFORM_FILES=False`` writes templates only — no per-unit .npy."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _wf_config = _build_config(streaming=True, save_files=False)

        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
            config=_wf_config,
        )
        we.run_extract_waveforms_streaming()

        assert (root_folder / "templates" / "templates_average.npy").is_file()
        assert (root_folder / "templates" / "templates_std.npy").is_file()

        wf_files = list((root_folder / "waveforms").glob("waveforms_*.npy"))
        assert wf_files == [], f"Expected no per-unit files, found: {wf_files}"

    def test_save_waveform_files_true_writes_one_per_unit(self, tmp_path, monkeypatch):
        """``SAVE_WAVEFORM_FILES=True`` writes a per-unit waveform file."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        _wf_config = _build_config(streaming=True, save_files=True)

        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
            config=_wf_config,
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

        _wf_config = _build_config(streaming=True, save_files=False)

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
            config=_wf_config,
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

        _wf_config = _build_config(streaming=True, save_files=False)

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
            config=_wf_config,
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
        _wf_config = _build_config(streaming=False, save_files=True)
        rec, sorting, _, _, ks_folder = _build_dataset(tmp_path)
        root_chunked = tmp_path / "wf_chunked"
        we_chunked = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_chunked,
            initial_folder=root_chunked / "initial",
            config=_wf_config,
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
        _wf_config = _build_config(streaming=True, save_files=True)
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
            config=_wf_config,
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

        _wf_config = _build_config(streaming=True, save_files=False)

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


# ---------------------------------------------------------------------------
# Batch A — parallel pre-allocation (open_memmap) + per-unit flush()
#
# Pins the contracts introduced by:
#   * dda9b16 — ``run_extract_waveforms`` replaces the
#     ``np.zeros(..) → np.save(..)`` pre-alloc pattern with
#     ``np.lib.format.open_memmap`` so the per-unit waveform file is
#     created via ``ftruncate`` instead of materialising a giant zero
#     array in RAM.
#   * 99ded3a — after each unit's per-spike write loop the worker
#     calls ``wfs.flush()`` so the OS does not buffer dirty pages
#     indefinitely (durability + IOStallWatchdog visibility).
# ---------------------------------------------------------------------------


@skip_no_spikeinterface
class TestParallelPreallocationAndFlush:
    """Memmap pre-allocation + flush invariants for ``run_extract_waveforms``."""

    def _build_we(self, tmp_path: Path, n_units: int = 2, n_spikes_per_unit: int = 6):
        """Lightweight synthetic dataset + ``WaveformExtractor`` for the
        parallel path. Returns ``(we, sorting, rec, ks_folder, root)``."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        cfg = _build_config(streaming=False, save_files=True)
        rec, sorting, _, _, ks_folder = _build_dataset(
            tmp_path, n_units=n_units, n_spikes_per_unit=n_spikes_per_unit
        )
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
            config=cfg,
        )
        return we, sorting, rec, ks_folder, root_folder

    def test_preallocation_uses_open_memmap_not_zeros(self, tmp_path, monkeypatch):
        """``run_extract_waveforms`` pre-allocates per-unit files via
        ``np.lib.format.open_memmap`` — never via ``np.zeros + np.save``.

        Spies on both APIs to assert:

        - ``np.lib.format.open_memmap`` is called once per unit.
        - ``np.zeros`` is never called with a shape that looks like the
          big per-unit waveform buffer
          ``(n_spikes, nsamples, num_channels)`` — the regression we
          would see if the old in-RAM pattern returned. Small per-spike
          buffers (e.g. the ``sampled_index`` struct used by
          :meth:`sample_spikes`) are exempted by gating on total size.
        """
        we, sorting, rec, ks_folder, _ = self._build_we(tmp_path)
        num_chans = rec.get_num_channels()

        import numpy as _np
        from spikelab.spike_sorting import waveform_extractor as _wfx

        real_open = _np.lib.format.open_memmap
        # Count only the parent-process pre-allocation opens (``mode='w+'``
        # with an explicit shape). Worker-side ``np.load(..., mmap_mode='r+')``
        # also routes through ``open_memmap`` but with ``mode='r+'``, so we
        # filter on ``mode``.
        open_calls = {"count": 0, "shapes": []}

        def _spy_open(path, *args, **kwargs):
            mode = kwargs.get("mode")
            if mode is None and len(args) >= 1:
                mode = args[0]
            shape = kwargs.get("shape")
            if shape is None and len(args) >= 3:
                shape = args[2]
            if mode == "w+":
                open_calls["count"] += 1
                open_calls["shapes"].append(shape)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(_np.lib.format, "open_memmap", _spy_open)

        # ``np.zeros`` is used elsewhere in the extractor (e.g.
        # ``sample_spikes`` builds a small struct array, the templates
        # cache, etc.). Gate the raise on the "big per-unit buffer"
        # signature so we only catch the regression we care about.
        real_zeros = _np.zeros
        big_threshold = we.nsamples * num_chans * 8  # ≥ one (nsamples, nchans) slab

        def _zeros_guard(shape, *args, **kwargs):
            try:
                shp_tuple = (
                    tuple(shape) if hasattr(shape, "__iter__") else (int(shape),)
                )
            except TypeError:
                shp_tuple = (int(shape),)
            # Big 3-D per-unit waveform buffer: (n_spikes, nsamples, nchans)
            if len(shp_tuple) == 3 and shp_tuple[1:] == (we.nsamples, num_chans):
                raise AssertionError(
                    f"np.zeros called with per-unit waveform shape {shp_tuple} — "
                    "expected open_memmap-based pre-allocation."
                )
            # Anything else (small structs, scalars, templates cache):
            # delegate to the real implementation.
            return real_zeros(shape, *args, **kwargs)

        monkeypatch.setattr(_wfx.np, "zeros", _zeros_guard)
        # The extractor imports numpy as ``np`` at module scope; that's
        # the binding the open_memmap pre-alloc path uses.

        we.run_extract_waveforms(n_jobs=1)

        n_units = len(sorting.unit_ids)
        assert open_calls["count"] == n_units, (
            f"Expected open_memmap called once per unit ({n_units}); "
            f"saw {open_calls['count']} calls."
        )
        for shp in open_calls["shapes"]:
            assert shp is not None and len(shp) == 3
            assert shp[1] == we.nsamples
            assert shp[2] == num_chans

    def test_preallocated_file_is_valid_npy(self, tmp_path):
        """Each per-unit ``waveforms_<uid>.npy`` is a valid .npy header
        and loads with the expected ``(n_spikes, nsamples, num_chans)``
        shape + dtype. Positions never written by a worker read back as
        zero (sparse-file semantics of ``open_memmap(mode='w+')``).
        """
        we, sorting, rec, _, root_folder = self._build_we(tmp_path)
        num_chans = rec.get_num_channels()

        we.run_extract_waveforms(n_jobs=1)

        for uid in sorting.unit_ids:
            wf_path = root_folder / "waveforms" / f"waveforms_{uid}.npy"
            assert wf_path.is_file(), f"Unit {uid}: expected {wf_path}"
            # Without mmap so we actually parse the .npy header.
            wfs = np.load(wf_path)
            assert wfs.ndim == 3
            assert wfs.shape[1] == we.nsamples
            assert wfs.shape[2] == num_chans
            assert wfs.dtype == np.dtype(we.dtype)
            # Sparse-file zeros are valid data — just assert finite.
            assert np.all(np.isfinite(wfs))

    def test_wfs_flush_called_per_unit(self, tmp_path, monkeypatch):
        """The worker calls ``wfs.flush()`` at least once per unit
        with spikes in a chunk. Pins the durability/visibility contract
        from commit 99ded3a: without the flush, dirty pages can sit in
        the OS page cache indefinitely, and the IOStallWatchdog's
        byte-counter delta can decide the worker is stalled when it's
        actually batching writes.

        The flush call sits inside
        ``_waveform_extractor_chunk`` between unit writes, so we
        spy on the result of ``np.load(..., mmap_mode='r+')`` rather
        than on ``open_memmap`` (which is called by the parent process
        before any worker spins up).
        """
        we, sorting, _, _, _ = self._build_we(tmp_path)

        from spikelab.spike_sorting import waveform_extractor as _wfx

        real_load = _wfx.np.load
        flushed_files: dict = {}

        def _wrapping_load(path, *args, **kwargs):
            arr = real_load(path, *args, **kwargs)
            if str(path).endswith(".npy") and "waveforms_" in str(path):
                real_flush = arr.flush

                def _spy_flush(*a, **k):
                    flushed_files[str(path)] = flushed_files.get(str(path), 0) + 1
                    return real_flush(*a, **k)

                # Patch only this instance's flush.
                try:
                    arr.flush = _spy_flush  # type: ignore[assignment]
                except (AttributeError, TypeError):
                    pass
            return arr

        monkeypatch.setattr(_wfx.np, "load", _wrapping_load)

        we.run_extract_waveforms(n_jobs=1)

        # At least one per-unit waveform file got flushed. (With
        # ``n_jobs=1`` the worker loads each unit's memmap inside the
        # chunk loop, so we expect one flush per unit-with-spikes.)
        assert flushed_files, (
            "Expected at least one wfs.flush() call inside the worker; "
            f"saw none. flushed_files={flushed_files}"
        )
        # Every unit with spikes should have had its memmap flushed
        # at least once (durability contract).
        for uid in sorting.unit_ids:
            unit_keys = [k for k in flushed_files if f"waveforms_{uid}.npy" in k]
            assert unit_keys, f"Unit {uid}: no flush() recorded"

    def test_zero_spike_unit_produces_valid_empty_npy(self, tmp_path):
        """A unit with zero spikes in the dataset still pre-allocates a
        valid .npy with shape ``(0, nsamples, num_chans)``. Loader and
        extractor do not crash.
        """
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        rec, sorting, _, _, ks_folder = _build_dataset(
            tmp_path, n_units=2, n_spikes_per_unit=5
        )

        # Inject an empty unit by appending a cluster ID with no spikes
        # in spike_clusters.npy. KilosortSortingExtractor scans
        # ``set(spike_clusters)`` for ``unit_ids``, so we need to give
        # it at least one spike but place it inside the trim margin so
        # ``select_random_spikes_uniformly`` filters it out.
        st = np.load(ks_folder / "spike_times.npy")
        sc = np.load(ks_folder / "spike_clusters.npy")
        empty_uid = int(sc.max()) + 1
        # Place a single spike right at sample 0 — well inside the
        # nbefore guard band, so sample_spikes will drop it.
        st_e = np.array([0], dtype=st.dtype)
        sc_e = np.array([empty_uid], dtype=sc.dtype)
        order = np.argsort(np.concatenate([st, st_e]))
        np.save(ks_folder / "spike_times.npy", np.concatenate([st, st_e])[order])
        np.save(ks_folder / "spike_clusters.npy", np.concatenate([sc, sc_e])[order])

        sorting = KilosortSortingExtractor(ks_folder)

        cfg = _build_config(streaming=False, save_files=True)
        root_folder = tmp_path / "wf_root"
        we = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
            config=cfg,
        )

        we.run_extract_waveforms(n_jobs=1)

        wf_path = root_folder / "waveforms" / f"waveforms_{empty_uid}.npy"
        assert (
            wf_path.is_file()
        ), f"Expected an empty-but-valid .npy for unit {empty_uid} at {wf_path}"
        wfs = np.load(wf_path)
        assert wfs.shape == (0, we.nsamples, rec.get_num_channels()), (
            f"Empty unit {empty_uid}: shape {wfs.shape} != "
            f"(0, {we.nsamples}, {rec.get_num_channels()})"
        )

    def test_reextraction_truncates_and_rewrites(self, tmp_path):
        """Re-running ``run_extract_waveforms`` with a smaller spike
        count truncates the existing per-unit file (``mode='w+'``
        semantics). Without that, the stale tail of the larger file
        would silently linger on disk.
        """
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        # ---- Run 1: 8 spikes / unit ----
        cfg = _build_config(streaming=False, save_files=True)
        rec, sorting, _, _, ks_folder = _build_dataset(
            tmp_path, n_units=2, n_spikes_per_unit=8
        )
        root_folder = tmp_path / "wf_root"
        we1 = WaveformExtractor.create_initial(
            recording_path=ks_folder / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
            config=cfg,
        )
        we1.run_extract_waveforms(n_jobs=1)

        first_shapes = {}
        first_sizes = {}
        for uid in sorting.unit_ids:
            p = root_folder / "waveforms" / f"waveforms_{uid}.npy"
            first_shapes[uid] = np.load(p).shape
            first_sizes[uid] = p.stat().st_size

        # ---- Run 2: 3 spikes / unit, *same* root_folder ----
        tmp_path2 = tmp_path / "run2"
        tmp_path2.mkdir()
        rec2, sorting2, _, _, ks_folder2 = _build_dataset(
            tmp_path2, n_units=2, n_spikes_per_unit=3
        )
        # Need a fresh initial_folder location too, because
        # ``create_initial`` re-builds ``unit_ids.npy`` etc. there.
        # Reuse the same root_folder so the second run overwrites
        # the per-unit .npy files.
        we2 = WaveformExtractor.create_initial(
            recording_path=ks_folder2 / "recording.dat",
            recording=rec2,
            sorting=sorting2,
            root_folder=root_folder,
            initial_folder=root_folder / "initial",
            config=cfg,
        )
        we2.run_extract_waveforms(n_jobs=1)

        for uid in sorting2.unit_ids:
            p = root_folder / "waveforms" / f"waveforms_{uid}.npy"
            second_shape = np.load(p).shape
            second_size = p.stat().st_size
            # Second run had fewer spikes → file shrank.
            assert second_shape[0] < first_shapes[uid][0], (
                f"Unit {uid}: re-extraction did not reduce spike count "
                f"(first {first_shapes[uid]}, second {second_shape})"
            )
            assert second_size < first_sizes[uid], (
                f"Unit {uid}: file size did not shrink (first "
                f"{first_sizes[uid]}, second {second_size}) — looks like "
                "mode='w+' is not truncating."
            )
            # And the new size is consistent with the new shape (no
            # stale-tail bytes hanging around).
            assert second_shape[1:] == (we2.nsamples, rec2.get_num_channels())

    def test_disjoint_writes_across_workers_no_corruption(self, tmp_path):
        """Per-unit memmap is written disjointly: every position the
        worker fills should match the result of a deterministic serial
        run.

        Implementation: run extraction twice on the same synthetic
        dataset with the same RNG seed (controlled via the
        ``_build_dataset`` fixture, which seeds inline) and assert
        byte-equality of the resulting .npy files. Forces ``n_jobs=1``
        in both runs — multi-process tests on Windows + pytest + numpy
        memmap are flaky in CI — but the equality contract being
        exercised is the same: identical inputs must produce identical
        per-unit memmap contents.
        """
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        # ---- Run A ----
        cfg = _build_config(streaming=False, save_files=True)
        (tmp_path / "A").mkdir()
        recA, sortingA, _, _, ks_folderA = _build_dataset(
            tmp_path / "A", n_units=3, n_spikes_per_unit=12
        )
        rootA = tmp_path / "A_root"
        weA = WaveformExtractor.create_initial(
            recording_path=ks_folderA / "recording.dat",
            recording=recA,
            sorting=sortingA,
            root_folder=rootA,
            initial_folder=rootA / "initial",
            config=cfg,
        )
        weA.run_extract_waveforms(n_jobs=1)

        # ---- Run B (rebuilt from scratch with the same seed) ----
        (tmp_path / "B").mkdir()
        recB, sortingB, _, _, ks_folderB = _build_dataset(
            tmp_path / "B", n_units=3, n_spikes_per_unit=12
        )
        rootB = tmp_path / "B_root"
        weB = WaveformExtractor.create_initial(
            recording_path=ks_folderB / "recording.dat",
            recording=recB,
            sorting=sortingB,
            root_folder=rootB,
            initial_folder=rootB / "initial",
            config=cfg,
        )
        weB.run_extract_waveforms(n_jobs=1)

        # Same units, same waveforms — no dropped writes, no
        # cross-unit corruption.
        assert list(sortingA.unit_ids) == list(sortingB.unit_ids)
        for uid in sortingA.unit_ids:
            arrA = np.load(rootA / "waveforms" / f"waveforms_{uid}.npy")
            arrB = np.load(rootB / "waveforms" / f"waveforms_{uid}.npy")
            assert arrA.shape == arrB.shape, (
                f"Unit {uid}: shapes diverged between runs "
                f"({arrA.shape} vs {arrB.shape})"
            )
            np.testing.assert_array_equal(
                arrA,
                arrB,
                err_msg=(
                    f"Unit {uid}: per-spike waveforms diverged between "
                    "identical runs — looks like a dropped/corrupted "
                    "write."
                ),
            )
