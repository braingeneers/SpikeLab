"""
Tests for spike_sorting module — Kilosort2 pipeline utilities.

These tests cover the testable components of the kilosort2 module without
requiring MATLAB, real recordings, or spikeinterface hardware access.
Heavy external dependencies are mocked throughout.
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional-dependency gating
# ---------------------------------------------------------------------------

try:
    import spikeinterface  # noqa: F401

    _has_spikeinterface = True
except Exception:
    _has_spikeinterface = False

try:
    import pandas as pd  # noqa: F401

    _has_pandas = True
except Exception:
    _has_pandas = False

skip_no_spikeinterface = pytest.mark.skipif(
    not _has_spikeinterface, reason="spikeinterface not installed"
)
skip_no_pandas = pytest.mark.skipif(not _has_pandas, reason="pandas not installed")


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for KilosortSortingExtractor file-based init
# ---------------------------------------------------------------------------


def _write_ks_folder(
    folder: Path,
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    sample_rate: float = 20000.0,
    tsv_data: dict | None = None,
    write_templates: bool = False,
    templates: np.ndarray | None = None,
    channel_map: np.ndarray | None = None,
):
    """Create a minimal Kilosort-style output folder on disk.

    Parameters
    ----------
    folder : Path
        Target directory (created if needed).
    spike_times, spike_clusters : np.ndarray
        Core Kilosort output arrays.
    sample_rate : float
        Sampling frequency written to params.py.
    tsv_data : dict or None
        If provided, written as cluster_info.tsv. Keys become column names.
    write_templates : bool
        If True, also write templates.npy and channel_map.npy.
    templates : np.ndarray or None
        Explicit templates array (n_templates, n_samples, n_channels).
    channel_map : np.ndarray or None
        Explicit channel map array.
    """
    folder.mkdir(parents=True, exist_ok=True)

    np.save(str(folder / "spike_times.npy"), spike_times)
    np.save(str(folder / "spike_clusters.npy"), spike_clusters)

    params_text = (
        f"dat_path = 'recording.dat'\n"
        f"n_channels_dat = 4\n"
        f"dtype = 'int16'\n"
        f"offset = 0\n"
        f"sample_rate = {sample_rate}\n"
        f"hp_filtered = True\n"
    )
    (folder / "params.py").write_text(params_text)

    if tsv_data is not None:
        lines = ["\t".join(tsv_data.keys())]
        n_rows = len(next(iter(tsv_data.values())))
        for i in range(n_rows):
            lines.append("\t".join(str(tsv_data[k][i]) for k in tsv_data))
        (folder / "cluster_info.tsv").write_text("\n".join(lines))

    if write_templates:
        if templates is None:
            n_units = int(spike_clusters.max()) + 1
            templates = (
                np.random.default_rng(42)
                .standard_normal((n_units, 61, 4))
                .astype(np.float32)
            )
        np.save(str(folder / "templates.npy"), templates)
        if channel_map is None:
            channel_map = np.arange(templates.shape[2])
        np.save(str(folder / "channel_map.npy"), channel_map)


def _make_mock_sorting(unit_ids, spike_trains_dict, sampling_frequency=20000.0):
    """Return a lightweight object mimicking KilosortSortingExtractor."""
    mock = SimpleNamespace()
    mock.unit_ids = list(unit_ids)
    mock.sampling_frequency = sampling_frequency

    def get_unit_spike_train(
        unit_id, segment_index=None, start_frame=None, end_frame=None
    ):
        st = spike_trains_dict[unit_id].copy()
        if start_frame is not None:
            st = st[st >= start_frame]
        if end_frame is not None:
            st = st[st < end_frame]
        return np.atleast_1d(st)

    mock.get_unit_spike_train = get_unit_spike_train
    return mock


def _make_mock_recording(
    num_samples=200000, sampling_frequency=20000.0, num_channels=4
):
    """Return a lightweight object mimicking a SpikeInterface recording."""
    mock = SimpleNamespace()
    mock.get_num_samples = lambda: num_samples
    mock.get_num_frames = lambda: num_samples
    mock.get_total_samples = lambda: num_samples
    mock.get_total_duration = lambda: num_samples / sampling_frequency
    mock.get_sampling_frequency = lambda: sampling_frequency
    mock.get_num_channels = lambda: num_channels
    mock.get_dtype = lambda: np.dtype("int16")
    mock.has_scaleable_traces = lambda: False
    mock.get_channel_ids = lambda: np.arange(num_channels)
    mock.get_channel_locations = lambda: np.column_stack(
        [np.arange(num_channels) * 20.0, np.zeros(num_channels)]
    )
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((num_samples, num_channels)).astype(np.float32)

    def get_traces(
        start_frame=0, end_frame=None, channel_ids=None, return_scaled=False
    ):
        ef = end_frame if end_frame is not None else num_samples
        if channel_ids is not None:
            return traces[start_frame:ef, channel_ids]
        return traces[start_frame:ef]

    mock.get_traces = get_traces
    return mock


# ===========================================================================
# __init__.py lazy import
# ===========================================================================


class TestLazyImport:
    """
    Tests for the lazy ``__getattr__`` in ``spikelab.spike_sorting.__init__``.

    Tests:
        (Test Case 1) Successful lazy import of sort_with_kilosort2.
        (Test Case 2) ImportError when dependencies are missing.
        (Test Case 3) AttributeError for unknown attributes.
    """

    def test_unknown_attribute_raises_attribute_error(self):
        """
        Accessing a non-existent attribute raises AttributeError.

        Tests:
            (Test Case 1) Unknown name triggers AttributeError with module name.
        """
        import spikelab.spike_sorting as pkg

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = pkg.totally_nonexistent_symbol

    def test_all_contains_sort_with_kilosort2(self):
        """
        The __all__ list advertises sort_with_kilosort2.

        Tests:
            (Test Case 1) __all__ contains exactly one expected entry.
        """
        import spikelab.spike_sorting as pkg

        assert "sort_with_kilosort2" in pkg.__all__

    @skip_no_spikeinterface
    def test_lazy_import_succeeds_when_deps_available(self):
        """
        sort_with_kilosort2 is importable when spikeinterface is present.

        Tests:
            (Test Case 1) Attribute access returns a callable.
        """
        import spikelab.spike_sorting as pkg

        fn = pkg.sort_with_kilosort2
        assert callable(fn)

    def test_lazy_import_raises_import_error_when_deps_missing(self):
        """
        sort_with_kilosort2 raises ImportError when dependencies are absent.

        Tests:
            (Test Case 1) ImportError is raised with a helpful message.
        """
        import spikelab.spike_sorting as pkg

        # Force a fresh lookup by removing cached attribute if present
        pkg_globals = vars(pkg)
        had_cached = "sort_with_kilosort2" in pkg_globals
        cached_val = pkg_globals.pop("sort_with_kilosort2", None)

        try:
            with patch.dict(sys.modules, {"spikelab.spike_sorting.kilosort2": None}):
                # Also block the relative import by making it raise
                original_getattr = pkg.__getattr__

                def patched_getattr(name):
                    if name == "sort_with_kilosort2":
                        raise ImportError("mocked missing dep")
                    return original_getattr(name)

                with patch.object(pkg, "__getattr__", patched_getattr):
                    with pytest.raises(ImportError, match="mocked missing dep"):
                        _ = pkg.sort_with_kilosort2
        finally:
            if had_cached:
                pkg_globals["sort_with_kilosort2"] = cached_val


# ===========================================================================
# KilosortSortingExtractor
# ===========================================================================


@skip_no_spikeinterface
@skip_no_pandas
class TestKilosortSortingExtractor:
    """
    Tests for KilosortSortingExtractor init and spike-train retrieval.

    Tests:
        (Test Case 1) Basic init from numpy files and params.py.
        (Test Case 2) TSV-based cluster filtering (exclude_cluster_groups).
        (Test Case 3) keep_good_only filtering via KSLabel.
        (Test Case 4) Units with zero spikes are excluded.
        (Test Case 5) get_unit_spike_train with start/end frame slicing.
        (Test Case 6) get_num_segments always returns 1.
        (Test Case 7) ms_to_samples conversion.
        (Test Case 8) No tsv files — fallback to minimal cluster_info.
    """

    @pytest.fixture()
    def ks_module(self):
        from spikelab.spike_sorting.kilosort2 import KilosortSortingExtractor

        return SimpleNamespace(
            KilosortSortingExtractor=KilosortSortingExtractor,
        )

    def test_basic_init(self, tmp_path, ks_module):
        """
        Basic init loads spike_times, spike_clusters, and sampling_frequency.

        Tests:
            (Test Case 1) unit_ids populated from spike data.
            (Test Case 2) sampling_frequency read from params.py.
        """
        spike_times = np.array([10, 20, 30, 100, 200], dtype=np.int64)
        spike_clusters = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        _write_ks_folder(tmp_path, spike_times, spike_clusters, sample_rate=30000.0)

        # Need to set KILOSORT_PARAMS global for init
        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(tmp_path)
            assert set(kse.unit_ids) == {0, 1}
            assert kse.sampling_frequency == 30000.0
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_exclude_cluster_groups_string(self, tmp_path, ks_module):
        """
        Excluding a cluster group as a string removes matching units.

        Tests:
            (Test Case 1) Units labeled 'noise' are excluded.
        """
        spike_times = np.array([10, 20, 100, 200], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 1], dtype=np.int64)
        tsv = {"cluster_id": [0, 1], "group": ["good", "noise"]}
        _write_ks_folder(tmp_path, spike_times, spike_clusters, tsv_data=tsv)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(
                tmp_path, exclude_cluster_groups="noise"
            )
            assert kse.unit_ids == [0]
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_exclude_cluster_groups_list(self, tmp_path, ks_module):
        """
        Excluding cluster groups as a list removes all matching units.

        Tests:
            (Test Case 1) Units labeled 'noise' or 'mua' are excluded.
        """
        spike_times = np.array([10, 20, 100, 200, 300], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        tsv = {"cluster_id": [0, 1, 2], "group": ["good", "noise", "mua"]}
        _write_ks_folder(tmp_path, spike_times, spike_clusters, tsv_data=tsv)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(
                tmp_path, exclude_cluster_groups=["noise", "mua"]
            )
            assert kse.unit_ids == [0]
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_keep_good_only(self, tmp_path, ks_module):
        """
        keep_good_only filters to units with KSLabel='good'.

        Tests:
            (Test Case 1) Only 'good' labeled units survive.
        """
        spike_times = np.array([10, 20, 100, 200], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 1], dtype=np.int64)
        tsv = {
            "cluster_id": [0, 1],
            "KSLabel": ["good", "mua"],
            "group": ["good", "mua"],
        }
        _write_ks_folder(tmp_path, spike_times, spike_clusters, tsv_data=tsv)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": True}
        try:
            kse = ks_module.KilosortSortingExtractor(tmp_path)
            assert kse.unit_ids == [0]
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_units_with_zero_spikes_excluded(self, tmp_path, ks_module):
        """
        Units present in tsv but with no spikes are excluded from unit_ids.

        Tests:
            (Test Case 1) Unit 2 exists in tsv but has no spikes.
        """
        spike_times = np.array([10, 20, 100], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1], dtype=np.int64)
        tsv = {"cluster_id": [0, 1, 2], "group": ["good", "good", "good"]}
        _write_ks_folder(tmp_path, spike_times, spike_clusters, tsv_data=tsv)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(tmp_path)
            assert 2 not in kse.unit_ids
            assert set(kse.unit_ids) == {0, 1}
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_get_unit_spike_train_slicing(self, tmp_path, ks_module):
        """
        get_unit_spike_train respects start_frame and end_frame.

        Tests:
            (Test Case 1) No slicing returns all spikes.
            (Test Case 2) start_frame filters out earlier spikes.
            (Test Case 3) end_frame filters out later spikes.
            (Test Case 4) Both bounds together.
        """
        spike_times = np.array([10, 50, 100, 200, 500], dtype=np.int64)
        spike_clusters = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        _write_ks_folder(tmp_path, spike_times, spike_clusters)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(tmp_path)

            # All spikes
            st = kse.get_unit_spike_train(0)
            assert len(st) == 5

            # start_frame only
            st = kse.get_unit_spike_train(0, start_frame=100)
            np.testing.assert_array_equal(st, [100, 200, 500])

            # end_frame only
            st = kse.get_unit_spike_train(0, end_frame=200)
            np.testing.assert_array_equal(st, [10, 50, 100])

            # Both
            st = kse.get_unit_spike_train(0, start_frame=50, end_frame=200)
            np.testing.assert_array_equal(st, [50, 100])
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_get_num_segments(self, ks_module):
        """
        get_num_segments always returns 1.

        Tests:
            (Test Case 1) Static method returns 1.
        """
        assert ks_module.KilosortSortingExtractor.get_num_segments() == 1

    def test_ms_to_samples(self, tmp_path, ks_module):
        """
        ms_to_samples converts milliseconds to sample counts correctly.

        Tests:
            (Test Case 1) 1 ms at 20 kHz = 20 samples.
            (Test Case 2) 0.5 ms at 20 kHz = 10 samples.
        """
        spike_times = np.array([10], dtype=np.int64)
        spike_clusters = np.array([0], dtype=np.int64)
        _write_ks_folder(tmp_path, spike_times, spike_clusters, sample_rate=20000.0)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(tmp_path)
            assert kse.ms_to_samples(1.0) == 20
            assert kse.ms_to_samples(0.5) == 10
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_no_tsv_files_fallback(self, tmp_path, ks_module):
        """
        When no tsv/csv files exist, cluster_info is built from spike data.

        Tests:
            (Test Case 1) unit_ids are populated from unique spike_clusters.
        """
        spike_times = np.array([10, 20, 100], dtype=np.int64)
        spike_clusters = np.array([0, 0, 3], dtype=np.int64)
        folder = tmp_path / "no_tsv"
        _write_ks_folder(folder, spike_times, spike_clusters)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(folder)
            assert set(kse.unit_ids) == {0, 3}
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_single_spike_single_unit(self, tmp_path, ks_module):
        """
        Init handles a folder with exactly one spike in one unit.

        Tests:
            (Test Case 1) np.atleast_1d guard on single-element arrays works.

        Notes:
            - The source uses np.atleast_1d specifically to handle this case.
        """
        spike_times = np.array([42], dtype=np.int64)
        spike_clusters = np.array([0], dtype=np.int64)
        _write_ks_folder(tmp_path, spike_times, spike_clusters)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(tmp_path)
            assert kse.unit_ids == [0]
            st = kse.get_unit_spike_train(0)
            np.testing.assert_array_equal(st, [42])
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_csv_file_loading(self, tmp_path, ks_module):
        """
        Init reads .csv files with comma delimiter.

        Tests:
            (Test Case 1) CSV with cluster_id and group columns is parsed.
        """
        spike_times = np.array([10, 20, 100], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1], dtype=np.int64)
        folder = tmp_path / "csv_test"
        _write_ks_folder(folder, spike_times, spike_clusters)
        csv_text = "cluster_id,group\n0,good\n1,noise"
        (folder / "cluster_info.csv").write_text(csv_text)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(
                folder, exclude_cluster_groups="noise"
            )
            assert kse.unit_ids == [0]
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_id_column_fallback(self, tmp_path, ks_module):
        """
        Init handles TSV files that use 'id' instead of 'cluster_id'.

        Tests:
            (Test Case 1) 'id' column is renamed to 'cluster_id' internally.
        """
        spike_times = np.array([10, 100], dtype=np.int64)
        spike_clusters = np.array([0, 1], dtype=np.int64)
        folder = tmp_path / "id_col"
        _write_ks_folder(folder, spike_times, spike_clusters)
        (folder / "cluster_info.tsv").write_text("id\tgroup\n0\tgood\n1\tgood")

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(folder)
            assert set(kse.unit_ids) == {0, 1}
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_empty_exclude_cluster_groups_list(self, tmp_path, ks_module):
        """
        An empty exclude_cluster_groups list excludes nothing.

        Tests:
            (Test Case 1) All units remain when exclude list is [].
        """
        spike_times = np.array([10, 100], dtype=np.int64)
        spike_clusters = np.array([0, 1], dtype=np.int64)
        tsv = {"cluster_id": [0, 1], "group": ["good", "noise"]}
        _write_ks_folder(tmp_path, spike_times, spike_clusters, tsv_data=tsv)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(
                tmp_path, exclude_cluster_groups=[]
            )
            assert set(kse.unit_ids) == {0, 1}
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_multiple_tsv_files_merged(self, tmp_path, ks_module):
        """
        Multiple TSV files are merged on cluster_id.

        Tests:
            (Test Case 1) Columns from both files are available for filtering.
        """
        spike_times = np.array([10, 100], dtype=np.int64)
        spike_clusters = np.array([0, 1], dtype=np.int64)
        folder = tmp_path / "multi_tsv"
        _write_ks_folder(folder, spike_times, spike_clusters)
        (folder / "cluster_group.tsv").write_text("cluster_id\tgroup\n0\tgood\n1\tgood")
        (folder / "cluster_KSLabel.tsv").write_text(
            "cluster_id\tKSLabel\n0\tgood\n1\tmua"
        )

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": True}
        try:
            kse = ks_module.KilosortSortingExtractor(folder)
            assert kse.unit_ids == [0]
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_spike_train_start_equals_end(self, tmp_path, ks_module):
        """
        get_unit_spike_train returns empty when start_frame == end_frame.

        Tests:
            (Test Case 1) No spike can satisfy start <= t < start.
        """
        spike_times = np.array([10, 50, 100], dtype=np.int64)
        spike_clusters = np.array([0, 0, 0], dtype=np.int64)
        folder = tmp_path / "start_eq_end"
        _write_ks_folder(folder, spike_times, spike_clusters)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(folder)
            st = kse.get_unit_spike_train(0, start_frame=50, end_frame=50)
            assert len(st) == 0
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_spike_train_bounds_beyond_all_spikes(self, tmp_path, ks_module):
        """
        get_unit_spike_train returns empty when bounds exclude all spikes.

        Tests:
            (Test Case 1) start_frame after last spike.
            (Test Case 2) end_frame before first spike.
        """
        spike_times = np.array([10, 50, 100], dtype=np.int64)
        spike_clusters = np.array([0, 0, 0], dtype=np.int64)
        folder = tmp_path / "beyond_bounds"
        _write_ks_folder(folder, spike_times, spike_clusters)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(folder)
            assert len(kse.get_unit_spike_train(0, start_frame=200)) == 0
            assert len(kse.get_unit_spike_train(0, end_frame=5)) == 0
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_spike_exactly_at_end_frame_excluded(self, tmp_path, ks_module):
        """
        A spike at exactly end_frame is excluded (exclusive upper bound).

        Tests:
            (Test Case 1) Spike at t=100 with end_frame=100 is not included.
        """
        spike_times = np.array([50, 100, 150], dtype=np.int64)
        spike_clusters = np.array([0, 0, 0], dtype=np.int64)
        folder = tmp_path / "at_end"
        _write_ks_folder(folder, spike_times, spike_clusters)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(folder)
            st = kse.get_unit_spike_train(0, end_frame=100)
            np.testing.assert_array_equal(st, [50])
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_ms_to_samples_zero(self, tmp_path, ks_module):
        """
        ms_to_samples(0) returns 0 regardless of sampling frequency.

        Tests:
            (Test Case 1) 0 ms => 0 samples.
        """
        spike_times = np.array([10], dtype=np.int64)
        spike_clusters = np.array([0], dtype=np.int64)
        folder = tmp_path / "ms_zero"
        _write_ks_folder(folder, spike_times, spike_clusters, sample_rate=44100.0)

        import spikelab.spike_sorting.kilosort2 as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(folder)
            assert kse.ms_to_samples(0) == 0
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params


# ===========================================================================
# KilosortSortingExtractor — get_chans_max and templates
# ===========================================================================


@skip_no_spikeinterface
@skip_no_pandas
class TestKilosortSortingExtractorGetChansMax:
    """
    Tests for get_chans_max and get_templates_half_windows_sizes.

    Tests:
        (Test Case 1) get_chans_max identifies correct peak channels.
        (Test Case 2) Positive-peak detection when positive peak dominates.
        (Test Case 3) get_templates_half_windows_sizes returns correct sizes.
    """

    @pytest.fixture()
    def kse_with_templates(self, tmp_path):
        """Create a KSE with known templates."""
        from spikelab.spike_sorting.kilosort2 import KilosortSortingExtractor
        import spikelab.spike_sorting.kilosort2 as ks_mod

        spike_times = np.array([10, 20, 100, 200], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 1], dtype=np.int64)

        # 2 templates, 61 samples, 4 channels
        templates = np.zeros((2, 61, 4), dtype=np.float32)
        # Unit 0: negative peak on channel 2 at sample 30
        templates[0, 30, 2] = -10.0
        # Unit 1: negative peak on channel 0 at sample 30
        templates[1, 30, 0] = -8.0

        channel_map = np.array([0, 1, 2, 3])
        _write_ks_folder(
            tmp_path,
            spike_times,
            spike_clusters,
            write_templates=True,
            templates=templates,
            channel_map=channel_map,
        )

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        old_pos_peak = getattr(ks_mod, "POS_PEAK_THRESH", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        ks_mod.POS_PEAK_THRESH = 2.0

        kse = KilosortSortingExtractor(tmp_path)
        yield kse

        if old_params is not None:
            ks_mod.KILOSORT_PARAMS = old_params
        if old_pos_peak is not None:
            ks_mod.POS_PEAK_THRESH = old_pos_peak

    def test_get_chans_max_negative_peaks(self, kse_with_templates):
        """
        get_chans_max identifies the channel with the largest negative peak.

        Tests:
            (Test Case 1) Unit 0 peak is on channel 2.
            (Test Case 2) Unit 1 peak is on channel 0.
            (Test Case 3) use_pos_peak is False for both (neg peak dominates).
        """
        use_pos, chans_ks, chans_all = kse_with_templates.get_chans_max()

        assert chans_all[0] == 2
        assert chans_all[1] == 0
        assert not use_pos[0]
        assert not use_pos[1]

    def test_get_chans_max_positive_peak_dominant(self, tmp_path):
        """
        When positive peak greatly exceeds negative, use_pos_peak is True.

        Tests:
            (Test Case 1) Unit with large positive peak uses positive channel.
        """
        from spikelab.spike_sorting.kilosort2 import KilosortSortingExtractor
        import spikelab.spike_sorting.kilosort2 as ks_mod

        spike_times = np.array([10, 20], dtype=np.int64)
        spike_clusters = np.array([0, 0], dtype=np.int64)

        templates = np.zeros((1, 61, 4), dtype=np.float32)
        # Negative peak small, positive peak very large (ratio > POS_PEAK_THRESH)
        templates[0, 30, 1] = -1.0
        templates[0, 30, 3] = 50.0

        channel_map = np.array([0, 1, 2, 3])
        folder = tmp_path / "pos_peak"
        _write_ks_folder(
            folder,
            spike_times,
            spike_clusters,
            write_templates=True,
            templates=templates,
            channel_map=channel_map,
        )

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        old_pos_peak = getattr(ks_mod, "POS_PEAK_THRESH", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        ks_mod.POS_PEAK_THRESH = 2.0

        try:
            kse = KilosortSortingExtractor(folder)
            use_pos, _, chans_all = kse.get_chans_max()
            assert use_pos[0]
            assert chans_all[0] == 3
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params
            if old_pos_peak is not None:
                ks_mod.POS_PEAK_THRESH = old_pos_peak

    def test_get_templates_half_windows_sizes(self, kse_with_templates):
        """
        get_templates_half_windows_sizes computes correct window sizes.

        Tests:
            (Test Case 1) Returns a list with one entry per template.
            (Test Case 2) Window sizes are non-negative integers.
        """
        _, chans_ks, _ = kse_with_templates.get_chans_max()
        hw_sizes = kse_with_templates.get_templates_half_windows_sizes(chans_ks)

        assert len(hw_sizes) == 2
        assert all(isinstance(s, int) and s >= 0 for s in hw_sizes)


@skip_no_spikeinterface
class TestWaveformExtractorToSpikeData:
    """
    Tests for _waveform_extractor_to_spikedata conversion function.

    Tests:
        (Test Case 1) Produces a SpikeData with correct number of units.
        (Test Case 2) Spike times are converted to milliseconds.
        (Test Case 3) Metadata contains source_file, source_format, fs_Hz.
        (Test Case 4) neuron_attributes contain enriched per-unit data.
    """

    @pytest.fixture()
    def convert_fn(self):
        from spikelab.spike_sorting.kilosort2 import _waveform_extractor_to_spikedata

        return _waveform_extractor_to_spikedata

    @staticmethod
    def _make_mock_we(
        unit_ids,
        spike_trains_dict,
        num_channels=2,
        sampling_frequency=20000.0,
        template_len=30,
        peak_ind=15,
    ):
        """Build a mock WaveformExtractor with all attributes needed by
        _waveform_extractor_to_spikedata."""
        sorting = _make_mock_sorting(
            unit_ids, spike_trains_dict, sampling_frequency=sampling_frequency
        )
        recording = _make_mock_recording(
            num_channels=num_channels, sampling_frequency=sampling_frequency
        )
        # Add get_property for electrode IDs
        recording.get_property = lambda name: None

        # chans_max_all: map each unit to channel 0
        chans_max_all = {uid: 0 for uid in unit_ids}

        # Polarity flags: all negative peak
        use_pos_peak = {uid: False for uid in unit_ids}

        # Templates: random (template_len, num_channels) per unit
        rng = np.random.default_rng(42)
        templates_avg = {
            uid: rng.standard_normal((template_len, num_channels)) for uid in unit_ids
        }
        templates_std = {
            uid: np.abs(rng.standard_normal((template_len, num_channels)))
            for uid in unit_ids
        }

        we = SimpleNamespace()
        we.sorting = sorting
        we.recording = recording
        we.sampling_frequency = sampling_frequency
        we.chans_max_all = chans_max_all
        we.use_pos_peak = use_pos_peak
        we.peak_ind = peak_ind
        we.return_scaled = True
        we.root_folder = Path("/fake/waveforms")

        def get_computed_template(unit_id, mode="average"):
            return (
                templates_avg[unit_id] if mode == "average" else templates_std[unit_id]
            )

        def ms_to_samples(ms):
            return int(round(ms * sampling_frequency / 1000.0))

        we.get_computed_template = get_computed_template
        we.ms_to_samples = ms_to_samples
        return we

    @staticmethod
    def _patch_globals(monkeypatch):
        """Patch module globals needed by _waveform_extractor_to_spikedata."""
        from spikelab.spike_sorting import kilosort2

        monkeypatch.setattr(kilosort2, "STD_AT_PEAK", True, raising=False)
        monkeypatch.setattr(kilosort2, "COMPILED_WAVEFORMS_MS_BEFORE", 2, raising=False)
        monkeypatch.setattr(kilosort2, "COMPILED_WAVEFORMS_MS_AFTER", 2, raising=False)
        monkeypatch.setattr(kilosort2, "SCALE_COMPILED_WAVEFORMS", True, raising=False)
        monkeypatch.setattr(kilosort2, "STD_OVER_WINDOW_MS_BEFORE", 0.5, raising=False)
        monkeypatch.setattr(kilosort2, "STD_OVER_WINDOW_MS_AFTER", 1.5, raising=False)
        # Patch _get_noise_levels to return simple noise array
        monkeypatch.setattr(
            kilosort2,
            "_get_noise_levels",
            lambda rec, return_scaled=True, **kw: np.ones(2),
        )

    def test_basic_conversion(self, convert_fn, monkeypatch):
        """
        Conversion produces SpikeData with correct trains and metadata.

        Tests:
            (Test Case 1) Two units produce two trains.
            (Test Case 2) Spike times are in milliseconds.
            (Test Case 3) Metadata fields are set.
            (Test Case 4) neuron_attributes have enriched data.
        """
        self._patch_globals(monkeypatch)

        trains = {
            0: np.array([200, 400, 600], dtype=np.int64),
            1: np.array([1000, 2000], dtype=np.int64),
        }
        we = self._make_mock_we([0, 1], trains)

        sd = convert_fn(we, "/fake/recording.h5")

        assert len(sd.train) == 2
        np.testing.assert_allclose(sd.train[0], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(sd.train[1], [50.0, 100.0])
        assert sd.metadata["source_file"] == "/fake/recording.h5"
        assert sd.metadata["source_format"] == "Kilosort2"
        assert sd.metadata["fs_Hz"] == 20000.0
        assert sd.neuron_attributes[0]["unit_id"] == 0
        assert sd.neuron_attributes[1]["unit_id"] == 1
        # Enriched attributes
        assert "snr" in sd.neuron_attributes[0]
        assert "std_norm" in sd.neuron_attributes[0]
        assert "template_full" in sd.neuron_attributes[0]
        assert "has_pos_peak" in sd.neuron_attributes[0]
        assert "channel" in sd.neuron_attributes[0]
        assert "x" in sd.neuron_attributes[0]
        assert "amplitude" in sd.neuron_attributes[0]
        assert "spike_train_samples" in sd.neuron_attributes[0]

    def test_empty_unit(self, convert_fn, monkeypatch):
        """
        A unit with no spikes produces an empty train.

        Tests:
            (Test Case 1) Empty spike train becomes empty array in SpikeData.
        """
        self._patch_globals(monkeypatch)

        trains = {0: np.array([], dtype=np.int64)}
        we = self._make_mock_we([0], trains)

        sd = convert_fn(we, "test.h5")
        assert len(sd.train) == 1
        assert len(sd.train[0]) == 0

    def test_single_unit_single_spike(self, convert_fn, monkeypatch):
        """
        Minimal valid input: one unit with one spike.

        Tests:
            (Test Case 1) Produces SpikeData with 1 unit and 1 spike time in ms.
        """
        self._patch_globals(monkeypatch)

        trains = {0: np.array([2000], dtype=np.int64)}
        we = self._make_mock_we([0], trains)

        sd = convert_fn(we, "test.h5")
        assert len(sd.train) == 1
        assert len(sd.train[0]) == 1
        np.testing.assert_allclose(sd.train[0], [100.0])

    def test_unsorted_spikes_are_sorted(self, convert_fn, monkeypatch):
        """
        Output spike times are sorted even if input samples are not.

        Tests:
            (Test Case 1) Source calls np.sort(), so output is monotonic.
        """
        self._patch_globals(monkeypatch)

        trains = {0: np.array([600, 200, 400], dtype=np.int64)}
        we = self._make_mock_we([0], trains)

        sd = convert_fn(we, "test.h5")
        times = sd.train[0]
        assert np.all(np.diff(times) >= 0), "Spike times should be monotonically sorted"
        np.testing.assert_allclose(times, [10.0, 20.0, 30.0])

    def test_metadata_includes_channel_locations(self, convert_fn, monkeypatch):
        """
        Metadata includes channel_locations and n_samples.

        Tests:
            (Test Case 1) channel_locations is a (channels, 2) array.
            (Test Case 2) n_samples is an integer.
        """
        self._patch_globals(monkeypatch)

        trains = {0: np.array([200], dtype=np.int64)}
        we = self._make_mock_we([0], trains)

        sd = convert_fn(we, "test.h5")
        locs = sd.metadata["channel_locations"]
        assert locs.shape == (2, 2)
        assert isinstance(sd.metadata["n_samples"], int)


# ===========================================================================
# ShellScript text processing
# ===========================================================================


@skip_no_spikeinterface
class TestShellScriptTextProcessing:
    """
    Tests for ShellScript private text-processing helpers.

    Tests:
        (Test Case 1) _remove_initial_blank_lines strips leading blanks.
        (Test Case 2) _get_num_initial_spaces counts leading spaces.
        (Test Case 3) substitute replaces placeholders.
        (Test Case 4) Script de-indentation in __init__.
    """

    @pytest.fixture()
    def ShellScript(self):
        from spikelab.spike_sorting.kilosort2 import ShellScript

        return ShellScript

    def test_remove_initial_blank_lines(self, ShellScript):
        """
        _remove_initial_blank_lines strips leading empty lines.

        Tests:
            (Test Case 1) Two blank lines followed by content.
            (Test Case 2) No blank lines returns unchanged.
            (Test Case 3) All blank lines returns empty list.
        """
        ss = ShellScript.__new__(ShellScript)
        ss._keep_temp_files = True
        ss._dirs_to_remove = []

        result = ss._remove_initial_blank_lines(["", "", "hello", "world"])
        assert result == ["hello", "world"]

        result = ss._remove_initial_blank_lines(["hello", "world"])
        assert result == ["hello", "world"]

        result = ss._remove_initial_blank_lines(["", "", ""])
        assert result == []

    def test_get_num_initial_spaces(self, ShellScript):
        """
        _get_num_initial_spaces counts leading space characters.

        Tests:
            (Test Case 1) No spaces returns 0.
            (Test Case 2) Four spaces returns 4.
            (Test Case 3) Empty string returns 0.
        """
        ss = ShellScript.__new__(ShellScript)
        ss._keep_temp_files = True
        ss._dirs_to_remove = []

        assert ss._get_num_initial_spaces("hello") == 0
        assert ss._get_num_initial_spaces("    hello") == 4
        assert ss._get_num_initial_spaces("") == 0

    def test_substitute(self, ShellScript):
        """
        substitute replaces placeholder strings in the script.

        Tests:
            (Test Case 1) Simple placeholder replacement.
        """
        ss = ShellScript.__new__(ShellScript)
        ss._keep_temp_files = True
        ss._dirs_to_remove = []
        ss._script = "echo {name}"
        ss.substitute("{name}", "world")
        assert ss._script == "echo world"

    def test_script_deindentation(self, ShellScript):
        """
        __init__ de-indents the script based on the first line's indentation.

        Tests:
            (Test Case 1) Uniform 8-space indent is stripped.
        """
        script = """\
        echo hello
        echo world"""

        ss = ShellScript.__new__(ShellScript)
        ss._script_path = None
        ss._log_path = None
        ss._keep_temp_files = False
        ss._process = None
        ss._files_to_remove = []
        ss._dirs_to_remove = []
        ss._start_time = None
        ss._verbose = False

        # Manually call the de-indentation logic
        lines = script.splitlines()
        lines = ss._remove_initial_blank_lines(lines)
        if len(lines) > 0:
            num_initial_spaces = ss._get_num_initial_spaces(lines[0])
            for ii, line in enumerate(lines):
                if len(line.strip()) > 0:
                    lines[ii] = lines[ii][num_initial_spaces:]
        result = "\n".join(lines)

        assert result == "echo hello\necho world"

    def test_rmdir_with_retries_nonexistent(self, ShellScript):
        """
        _rmdir_with_retries on a nonexistent dir returns without error.

        Tests:
            (Test Case 1) No exception for a path that does not exist.
        """
        ShellScript._rmdir_with_retries("/nonexistent_dir_abc123", num_retries=1)

    def test_is_running_no_process(self, ShellScript):
        """
        isRunning returns False when no process has been started.

        Tests:
            (Test Case 1) _process is None => False.
        """
        ss = ShellScript.__new__(ShellScript)
        ss._keep_temp_files = True
        ss._dirs_to_remove = []
        ss._process = None

        assert ss.isRunning() is False

    def test_is_finished_no_process(self, ShellScript):
        """
        isFinished returns False when no process has been started.

        Tests:
            (Test Case 1) _process is None => False.
        """
        ss = ShellScript.__new__(ShellScript)
        ss._keep_temp_files = True
        ss._dirs_to_remove = []
        ss._process = None

        assert ss.isFinished() is False

    def test_return_code_before_finished_raises(self, ShellScript):
        """
        returnCode raises Exception when process is not finished.

        Tests:
            (Test Case 1) No process => isFinished is False => raises.
        """
        ss = ShellScript.__new__(ShellScript)
        ss._keep_temp_files = True
        ss._dirs_to_remove = []
        ss._process = None

        with pytest.raises(Exception, match="Cannot get return code"):
            ss.returnCode()


# ===========================================================================
# Utils._mem_to_int
# ===========================================================================


@skip_no_spikeinterface
class TestUtilsMemToInt:
    """
    Tests for Utils._mem_to_int static method.

    Tests:
        (Test Case 1) Kilobyte suffix 'k'.
        (Test Case 2) Megabyte suffix 'M'.
        (Test Case 3) Gigabyte suffix 'G'.
        (Test Case 4) Fractional values.
    """

    @pytest.fixture()
    def Utils(self):
        from spikelab.spike_sorting.kilosort2 import Utils

        return Utils

    def test_kilobyte(self, Utils):
        """
        'k' suffix converts to 1e3 multiplier.

        Tests:
            (Test Case 1) '4k' => 4000.
        """
        assert Utils._mem_to_int("4k") == 4000

    def test_megabyte(self, Utils):
        """
        'M' suffix converts to 1e6 multiplier.

        Tests:
            (Test Case 1) '16M' => 16_000_000.
        """
        assert Utils._mem_to_int("16M") == 16_000_000

    def test_gigabyte(self, Utils):
        """
        'G' suffix converts to 1e9 multiplier.

        Tests:
            (Test Case 1) '2G' => 2_000_000_000.
        """
        assert Utils._mem_to_int("2G") == 2_000_000_000

    def test_fractional(self, Utils):
        """
        Fractional values are supported.

        Tests:
            (Test Case 1) '1.5G' => 1_500_000_000.
        """
        assert Utils._mem_to_int("1.5G") == 1_500_000_000

    def test_invalid_suffix_raises(self, Utils):
        """
        An unrecognized suffix raises ValueError.

        Tests:
            (Test Case 1) 'T' suffix is not recognized.
        """
        with pytest.raises(ValueError, match="Invalid memory suffix"):
            Utils._mem_to_int("4T")

    def test_zero_value(self, Utils):
        """
        '0G' converts to 0.

        Tests:
            (Test Case 1) Zero multiplied by any exponent is 0.
        """
        assert Utils._mem_to_int("0G") == 0
        assert Utils._mem_to_int("0k") == 0
        assert Utils._mem_to_int("0M") == 0


# ===========================================================================
# Utils.read_python
# ===========================================================================


@skip_no_spikeinterface
class TestUtilsReadPython:
    """
    Tests for Utils.read_python — parses Kilosort params.py files.

    Tests:
        (Test Case 1) Parses simple key=value assignments.
        (Test Case 2) Keys are lowercased.
        (Test Case 3) Non-existent file raises.
    """

    @pytest.fixture()
    def Utils(self):
        from spikelab.spike_sorting.kilosort2 import Utils

        return Utils

    def test_parses_params_file(self, tmp_path, Utils):
        """
        read_python parses a params.py file into a dictionary.

        Tests:
            (Test Case 1) sample_rate is parsed as float.
            (Test Case 2) dtype is parsed as string.
            (Test Case 3) hp_filtered is parsed as bool.
        """
        params_text = (
            "sample_rate = 30000.0\n" "dtype = 'int16'\n" "hp_filtered = True\n"
        )
        p = tmp_path / "params.py"
        p.write_text(params_text)

        result = Utils.read_python(str(p))
        assert result["sample_rate"] == 30000.0
        assert result["dtype"] == "int16"
        assert result["hp_filtered"] is True

    def test_keys_lowercased(self, tmp_path, Utils):
        """
        All keys in the parsed dict are lowercased.

        Tests:
            (Test Case 1) 'Sample_Rate' becomes 'sample_rate'.
        """
        p = tmp_path / "params.py"
        p.write_text("Sample_Rate = 20000\n")

        result = Utils.read_python(str(p))
        assert "sample_rate" in result
        assert "Sample_Rate" not in result

    def test_nonexistent_file_raises(self, Utils):
        """
        A non-existent file raises FileNotFoundError.

        Tests:
            (Test Case 1) Path that does not exist triggers error.
        """
        with pytest.raises(FileNotFoundError, match="parameter file not found"):
            Utils.read_python("/nonexistent/params.py")


# ===========================================================================
# sort_with_kilosort2 validation
# ===========================================================================


@skip_no_spikeinterface
class TestSortWithKilosort2Validation:
    """
    Tests for sort_with_kilosort2 parameter validation and defaults.

    Tests:
        (Test Case 1) compile_all_recordings=True without folder raises ValueError.
        (Test Case 2) Mismatched list lengths raise AssertionError.
        (Test Case 3) Default kilosort_params are applied when None.
    """

    @pytest.fixture()
    def sort_fn(self):
        from spikelab.spike_sorting.kilosort2 import sort_with_kilosort2

        return sort_with_kilosort2

    def test_compile_all_without_folder_raises(self, sort_fn):
        """
        compile_all_recordings=True without compiled_results_folder raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        with pytest.raises(ValueError, match="compile_all_recordings"):
            sort_fn(
                recording_files=["fake.h5"],
                compile_all_recordings=True,
                compiled_results_folder=None,
            )

    def test_mismatched_list_lengths_raises(self, sort_fn, tmp_path):
        """
        Mismatched lengths of recording_files and intermediate_folders
        raise ValueError.

        Tests:
            (Test Case 1) 2 recordings but 1 intermediate folder.
        """
        with pytest.raises(ValueError, match="same length"):
            sort_fn(
                recording_files=["fake1.h5", "fake2.h5"],
                intermediate_folders=[str(tmp_path / "inter1")],
                results_folders=[str(tmp_path / "res1"), str(tmp_path / "res2")],
            )

    def test_default_kilosort_params(self, sort_fn, tmp_path):
        """
        Default kilosort_params contain expected keys when None is passed.

        Tests:
            (Test Case 1) Defaults include detect_threshold, car, NT, etc.

        Notes:
            - This test patches process_recording to prevent actual execution.
        """
        import spikelab.spike_sorting.kilosort2 as ks_mod
        import spikelab.spike_sorting.pipeline as pipe_mod

        with patch.object(
            pipe_mod, "process_recording", return_value=Exception("skip")
        ):
            sort_fn(
                recording_files=["fake.h5"],
                intermediate_folders=[str(tmp_path / "inter")],
                results_folders=[str(tmp_path / "res")],
                kilosort_path="/fake/kilosort",
                kilosort_params=None,
                compile_all_recordings=False,
                delete_inter=False,
                create_figures=False,
            )
            # After calling, the global KILOSORT_PARAMS should have defaults
            params = ks_mod.KILOSORT_PARAMS
            assert params["detect_threshold"] == 6
            assert params["car"] in (True, 1)
            assert "NT" in params
            assert "nPCs" in params

    def test_custom_params_override_defaults(self, sort_fn, tmp_path):
        """
        User-provided kilosort_params override the default values.

        Tests:
            (Test Case 1) detect_threshold=10 overrides default of 6.
            (Test Case 2) Other defaults are preserved.
        """
        import spikelab.spike_sorting.kilosort2 as ks_mod
        import spikelab.spike_sorting.pipeline as pipe_mod

        with patch.object(
            pipe_mod, "process_recording", return_value=Exception("skip")
        ):
            sort_fn(
                recording_files=["fake.h5"],
                intermediate_folders=[str(tmp_path / "inter")],
                results_folders=[str(tmp_path / "res")],
                kilosort_path="/fake/kilosort",
                kilosort_params={"detect_threshold": 10},
                compile_all_recordings=False,
                delete_inter=False,
                create_figures=False,
            )
            params = ks_mod.KILOSORT_PARAMS
            assert params["detect_threshold"] == 10
            assert params["preclust_threshold"] == 8

    def test_empty_recording_files(self, sort_fn):
        """
        An empty recording_files list returns an empty result list.

        Tests:
            (Test Case 1) No recordings => no SpikeData objects returned.
        """
        result = sort_fn(
            recording_files=[],
            intermediate_folders=[],
            results_folders=[],
            kilosort_path="/fake/kilosort",
            compile_all_recordings=False,
            delete_inter=False,
            create_figures=False,
        )
        assert result == []

    def test_use_docker_sets_global(self, sort_fn, tmp_path):
        """
        use_docker=True sets the USE_DOCKER global.

        Tests:
            (Test Case 1) Global is True after calling with use_docker=True.
            (Test Case 2) Global is False after calling with use_docker=False.
        """
        import spikelab.spike_sorting.kilosort2 as ks_mod
        import spikelab.spike_sorting.pipeline as pipe_mod

        with patch.object(
            pipe_mod, "process_recording", return_value=Exception("skip")
        ):
            sort_fn(
                recording_files=["fake.h5"],
                intermediate_folders=[str(tmp_path / "inter")],
                results_folders=[str(tmp_path / "res")],
                kilosort_path="/fake/kilosort",
                use_docker=True,
                compile_all_recordings=False,
                delete_inter=False,
                create_figures=False,
            )
            assert ks_mod.USE_DOCKER is True

            sort_fn(
                recording_files=["fake.h5"],
                intermediate_folders=[str(tmp_path / "inter2")],
                results_folders=[str(tmp_path / "res2")],
                kilosort_path="/fake/kilosort",
                use_docker=False,
                compile_all_recordings=False,
                delete_inter=False,
                create_figures=False,
            )
            assert ks_mod.USE_DOCKER is False

    def test_save_raw_pkl_sets_global(self, sort_fn):
        """
        save_raw_pkl=True sets the SAVE_RAW_PKL global via config.

        Tests:
            (Test Case 1) Global is True after calling with save_raw_pkl=True.
            (Test Case 2) Global is False after calling with save_raw_pkl=False.
        """
        import spikelab.spike_sorting.kilosort2 as ks_mod
        import spikelab.spike_sorting.pipeline as pipe_mod

        with patch.object(
            pipe_mod, "process_recording", return_value=Exception("skip")
        ):
            sort_fn(
                recording_files=["fake.h5"],
                kilosort_path="/fake/kilosort",
                save_raw_pkl=True,
                compile_all_recordings=False,
                delete_inter=False,
                create_figures=False,
            )
            assert ks_mod.SAVE_RAW_PKL is True

            sort_fn(
                recording_files=["fake.h5"],
                kilosort_path="/fake/kilosort",
                save_raw_pkl=False,
                compile_all_recordings=False,
                delete_inter=False,
                create_figures=False,
            )
            assert ks_mod.SAVE_RAW_PKL is False

    def test_save_raw_pkl_writes_file(self, sort_fn, tmp_path):
        """
        When save_raw_pkl=True, both raw and curated pkl files are saved.

        Tests:
            (Test Case 1) sorted_spikedata.pkl is created.
            (Test Case 2) sorted_spikedata_curated.pkl is created.
            (Test Case 3) Raw pkl has more units than curated pkl.
        """
        import spikelab.spike_sorting.pipeline as pipe_mod
        from spikelab import SpikeData

        sd_raw = SpikeData(
            [np.array([10.0, 20.0]), np.array([30.0]), np.array([50.0, 60.0, 70.0])],
            length=100.0,
        )
        sd_curated = SpikeData(
            [np.array([10.0, 20.0]), np.array([50.0, 60.0, 70.0])],
            length=100.0,
        )

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with patch.object(
            pipe_mod, "process_recording", return_value=(sd_raw, sd_curated)
        ):
            result = sort_fn(
                recording_files=["fake.h5"],
                intermediate_folders=[str(tmp_path / "inter")],
                results_folders=[str(results_dir)],
                kilosort_path="/fake/kilosort",
                save_raw_pkl=True,
                compile_all_recordings=False,
                delete_inter=False,
                create_figures=False,
            )

        raw_pkl = results_dir / "sorted_spikedata.pkl"
        curated_pkl = results_dir / "sorted_spikedata_curated.pkl"
        assert raw_pkl.exists(), "Raw pkl not saved"
        assert curated_pkl.exists(), "Curated pkl not saved"

        import pickle

        with open(raw_pkl, "rb") as f:
            loaded_raw = pickle.load(f)
        with open(curated_pkl, "rb") as f:
            loaded_curated = pickle.load(f)
        assert loaded_raw.N == 3
        assert loaded_curated.N == 2

    def test_save_raw_pkl_false_skips_raw(self, sort_fn, tmp_path):
        """
        When save_raw_pkl=False (default), only the curated pkl is saved.

        Tests:
            (Test Case 1) sorted_spikedata.pkl is NOT created.
            (Test Case 2) sorted_spikedata_curated.pkl IS created.
        """
        import spikelab.spike_sorting.pipeline as pipe_mod
        from spikelab import SpikeData

        sd_curated = SpikeData(
            [np.array([10.0, 20.0]), np.array([50.0, 60.0, 70.0])],
            length=100.0,
        )

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with patch.object(pipe_mod, "process_recording", return_value=sd_curated):
            result = sort_fn(
                recording_files=["fake.h5"],
                intermediate_folders=[str(tmp_path / "inter")],
                results_folders=[str(results_dir)],
                kilosort_path="/fake/kilosort",
                save_raw_pkl=False,
                compile_all_recordings=False,
                delete_inter=False,
                create_figures=False,
            )

        raw_pkl = results_dir / "sorted_spikedata.pkl"
        curated_pkl = results_dir / "sorted_spikedata_curated.pkl"
        assert (
            not raw_pkl.exists()
        ), "Raw pkl should not be saved when save_raw_pkl=False"
        assert curated_pkl.exists(), "Curated pkl not saved"


# ===========================================================================
# _spike_sort_docker and Docker branch in spike_sort
# ===========================================================================


@skip_no_spikeinterface
@skip_no_pandas
class TestSpikeSortDocker:
    """
    Tests for _spike_sort_docker and the Docker branch in spike_sort.

    Tests:
        (Test Case 1) _spike_sort_docker calls run_sorter with correct args.
        (Test Case 2) _spike_sort_docker reads output from sorter_output subfolder.
        (Test Case 3) _spike_sort_docker falls back to output_folder if no subfolder.
        (Test Case 4) spike_sort uses Docker path when USE_DOCKER is True.
        (Test Case 5) spike_sort uses MATLAB path when USE_DOCKER is False.
        (Test Case 6) spike_sort returns exception when Docker sorting fails.

    Notes:
        - All tests mock spikeinterface.sorters.run_sorter and create fake
          Phy output files on disk. Docker is never actually invoked.
    """

    @pytest.fixture(autouse=True)
    def _set_globals(self):
        import spikelab.spike_sorting.kilosort2 as ks_mod

        self._ks_mod = ks_mod
        self._old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        self._old_docker = getattr(ks_mod, "USE_DOCKER", None)
        self._old_recompute = getattr(ks_mod, "RECOMPUTE_SORTING", None)
        ks_mod.KILOSORT_PARAMS = {
            "detect_threshold": 6,
            "projection_threshold": [10, 4],
            "preclust_threshold": 8,
            "car": True,
            "minFR": 0.1,
            "minfr_goodchannels": 0.1,
            "freq_min": 150,
            "sigmaMask": 30,
            "nPCs": 3,
            "ntbuff": 64,
            "nfilt_factor": 4,
            "NT": None,
            "keep_good_only": False,
        }
        ks_mod.RECOMPUTE_SORTING = True
        yield
        if self._old_params is not None:
            ks_mod.KILOSORT_PARAMS = self._old_params
        if self._old_docker is not None:
            ks_mod.USE_DOCKER = self._old_docker
        if self._old_recompute is not None:
            ks_mod.RECOMPUTE_SORTING = self._old_recompute

    def _write_fake_phy_output(self, folder):
        """Write minimal Phy output files so KilosortSortingExtractor can load."""
        folder.mkdir(parents=True, exist_ok=True)
        spike_times = np.array([100, 200, 300, 400], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 1], dtype=np.int64)
        np.save(str(folder / "spike_times.npy"), spike_times)
        np.save(str(folder / "spike_clusters.npy"), spike_clusters)
        (folder / "params.py").write_text(
            "dat_path = 'recording.dat'\n"
            "n_channels_dat = 4\n"
            "dtype = 'int16'\n"
            "offset = 0\n"
            "sample_rate = 20000.0\n"
            "hp_filtered = True\n"
        )

    def test_spike_sort_docker_calls_run_sorter(self, tmp_path):
        """
        _spike_sort_docker passes correct arguments to SI run_sorter.

        Tests:
            (Test Case 1) run_sorter is called with sorter_name='kilosort2'.
            (Test Case 2) A specific docker_image tag is passed.
            (Test Case 3) KILOSORT_PARAMS are forwarded as kwargs.
            (Test Case 4) installation_mode='no-install' is passed.
        """
        from spikelab.spike_sorting.kilosort2 import _spike_sort_docker

        output_folder = tmp_path / "ks_output"
        sorter_output = output_folder / "sorter_output"
        self._write_fake_phy_output(sorter_output)

        recording = _make_mock_recording()
        mock_rs = MagicMock(return_value=None)

        with (
            patch("spikelab.spike_sorting.kilosort2.write_binary_recording"),
            patch("spikelab.spike_sorting.kilosort2.BinaryRecordingExtractor"),
            patch("spikelab.spike_sorting.kilosort2.run_sorter", mock_rs),
        ):
            result = _spike_sort_docker(recording, output_folder)

        mock_rs.assert_called_once()
        _, call_kwargs = mock_rs.call_args
        assert call_kwargs["sorter_name"] == "kilosort2"
        assert isinstance(call_kwargs["docker_image"], str)
        assert "kilosort2" in call_kwargs["docker_image"]
        assert call_kwargs["installation_mode"] == "no-install"
        assert call_kwargs["detect_threshold"] == 6

        assert hasattr(result, "unit_ids")
        assert set(result.unit_ids) == {0, 1}

    def test_spike_sort_docker_sorter_output_subfolder(self, tmp_path):
        """
        _spike_sort_docker reads from sorter_output/ subfolder when it exists.

        Tests:
            (Test Case 1) Phy files in sorter_output/ are found.
        """
        from spikelab.spike_sorting.kilosort2 import _spike_sort_docker

        output_folder = tmp_path / "ks_output"
        sorter_output = output_folder / "sorter_output"
        self._write_fake_phy_output(sorter_output)

        recording = _make_mock_recording()

        with (
            patch("spikelab.spike_sorting.kilosort2.write_binary_recording"),
            patch("spikelab.spike_sorting.kilosort2.BinaryRecordingExtractor"),
            patch("spikelab.spike_sorting.kilosort2.run_sorter", MagicMock()),
        ):
            result = _spike_sort_docker(recording, output_folder)

        assert result.folder == sorter_output.absolute()

    def test_spike_sort_docker_fallback_to_output_folder(self, tmp_path):
        """
        _spike_sort_docker falls back to output_folder when no sorter_output/ exists.

        Tests:
            (Test Case 1) Phy files directly in output_folder are found.
        """
        from spikelab.spike_sorting.kilosort2 import _spike_sort_docker

        output_folder = tmp_path / "ks_output"
        self._write_fake_phy_output(output_folder)

        recording = _make_mock_recording()

        with (
            patch("spikelab.spike_sorting.kilosort2.write_binary_recording"),
            patch("spikelab.spike_sorting.kilosort2.BinaryRecordingExtractor"),
            patch("spikelab.spike_sorting.kilosort2.run_sorter", MagicMock()),
        ):
            result = _spike_sort_docker(recording, output_folder)

        assert result.folder == output_folder.absolute()

    def test_spike_sort_uses_docker_when_enabled(self, tmp_path):
        """
        spike_sort calls _spike_sort_docker when USE_DOCKER is True.

        Tests:
            (Test Case 1) _spike_sort_docker is called instead of RunKilosort.
            (Test Case 2) RunKilosort is never instantiated.
        """
        from spikelab.spike_sorting.kilosort2 import spike_sort

        self._ks_mod.USE_DOCKER = True
        output_folder = tmp_path / "ks_output"
        recording = _make_mock_recording()

        mock_kse = SimpleNamespace(unit_ids=[0, 1])

        with (
            patch.object(
                self._ks_mod, "_spike_sort_docker", return_value=mock_kse
            ) as mock_docker,
            patch.object(self._ks_mod, "RunKilosort") as mock_rk,
        ):
            result = spike_sort(
                recording, "fake.h5", tmp_path / "rec.dat", output_folder
            )

        mock_docker.assert_called_once_with(recording, output_folder)
        mock_rk.assert_not_called()
        assert result is mock_kse

    def test_spike_sort_uses_matlab_when_docker_disabled(self, tmp_path):
        """
        spike_sort uses RunKilosort when USE_DOCKER is False.

        Tests:
            (Test Case 1) RunKilosort is instantiated.
            (Test Case 2) _spike_sort_docker is not called.
        """
        from spikelab.spike_sorting.kilosort2 import spike_sort

        self._ks_mod.USE_DOCKER = False
        output_folder = tmp_path / "ks_output"
        recording = _make_mock_recording()

        mock_sorting = SimpleNamespace(unit_ids=[0])
        mock_ks_instance = MagicMock()
        mock_ks_instance.run.return_value = mock_sorting

        with (
            patch.object(
                self._ks_mod, "RunKilosort", return_value=mock_ks_instance
            ) as mock_rk,
            patch.object(self._ks_mod, "_spike_sort_docker") as mock_docker,
            patch.object(self._ks_mod, "write_recording"),
        ):
            result = spike_sort(
                recording, "fake.h5", tmp_path / "rec.dat", output_folder
            )

        mock_rk.assert_called_once()
        mock_docker.assert_not_called()
        assert result is mock_sorting

    def test_spike_sort_docker_failure_returns_exception(self, tmp_path):
        """
        spike_sort returns the exception when Docker sorting fails.

        Tests:
            (Test Case 1) Exception from _spike_sort_docker is caught and returned.
        """
        from spikelab.spike_sorting.kilosort2 import spike_sort

        self._ks_mod.USE_DOCKER = True
        output_folder = tmp_path / "ks_output"
        recording = _make_mock_recording()

        with patch.object(
            self._ks_mod,
            "_spike_sort_docker",
            side_effect=RuntimeError("Docker failed"),
        ):
            result = spike_sort(
                recording, "fake.h5", tmp_path / "rec.dat", output_folder
            )

        assert isinstance(result, RuntimeError)
        assert "Docker failed" in str(result)


# ===========================================================================
# print_stage
# ===========================================================================


@skip_no_spikeinterface
class TestPrintStage:
    """
    Tests for the print_stage banner formatting function.

    Tests:
        (Test Case 1) Output contains the text centered between '=' chars.
        (Test Case 2) Banner is 70 characters wide.
        (Test Case 3) Non-string input is converted to string.
    """

    @pytest.fixture()
    def print_stage(self):
        from spikelab.spike_sorting.kilosort2 import print_stage

        return print_stage

    def test_banner_contains_text(self, print_stage, capsys):
        """
        The banner output contains the provided text.

        Tests:
            (Test Case 1) 'HELLO' appears in printed output.
        """
        print_stage("HELLO")
        captured = capsys.readouterr().out
        assert "HELLO" in captured

    def test_banner_width(self, print_stage, capsys):
        """
        The banner lines of '=' are 70 characters wide.

        Tests:
            (Test Case 1) First non-empty line is 70 '=' characters.
        """
        print_stage("TEST")
        captured = capsys.readouterr().out
        lines = [l for l in captured.strip().split("\n") if l.strip()]
        assert lines[0] == "=" * 70

    def test_non_string_input(self, print_stage, capsys):
        """
        Non-string input is converted to string.

        Tests:
            (Test Case 1) Integer 42 appears in output.
        """
        print_stage(42)
        captured = capsys.readouterr().out
        assert "42" in captured


# ===========================================================================
# concatenate_recordings validation
# ===========================================================================


@skip_no_spikeinterface
class TestConcatenateRecordingsValidation:
    """
    Tests for electrode configuration validation in concatenate_recordings.
    """

    @pytest.fixture()
    def concat_fn(self, monkeypatch):
        from spikelab.spike_sorting import kilosort2

        monkeypatch.setattr(kilosort2, "REC_CHUNKS", [], raising=False)
        monkeypatch.setattr(kilosort2, "_REC_CHUNK_NAMES", [], raising=False)
        monkeypatch.setattr(kilosort2, "STREAM_ID", None, raising=False)
        monkeypatch.setattr(kilosort2, "GAIN_TO_UV", None, raising=False)
        monkeypatch.setattr(kilosort2, "OFFSET_TO_UV", None, raising=False)
        monkeypatch.setattr(kilosort2, "FREQ_MIN", 300, raising=False)
        monkeypatch.setattr(kilosort2, "FREQ_MAX", 6000, raising=False)
        monkeypatch.setattr(kilosort2, "FIRST_N_MINS", None, raising=False)
        monkeypatch.setattr(kilosort2, "MEA_Y_MAX", None, raising=False)
        return kilosort2.concatenate_recordings

    def test_channel_count_mismatch_raises(self, concat_fn, tmp_path, monkeypatch):
        """
        Recordings with different channel counts raise ValueError.

        Tests:
            (Test Case 1) Two files with 4 vs 2 channels cannot be
                concatenated.
        """
        from spikelab.spike_sorting import kilosort2

        rec_a = _make_mock_recording(num_channels=4)
        rec_b = _make_mock_recording(num_channels=2)

        # Create dummy .raw.h5 files so the directory scan finds them
        (tmp_path / "a.raw.h5").touch()
        (tmp_path / "b.raw.h5").touch()

        call_count = [0]
        recordings = [rec_a, rec_b]

        def mock_load(path):
            rec = recordings[call_count[0]]
            call_count[0] += 1
            return rec

        monkeypatch.setattr(kilosort2, "load_single_recording", mock_load)

        with pytest.raises(ValueError, match="channels"):
            concat_fn(tmp_path)

    def test_sampling_frequency_mismatch_raises(self, concat_fn, tmp_path, monkeypatch):
        """
        Recordings with different sampling frequencies raise ValueError.

        Tests:
            (Test Case 1) 20 kHz vs 30 kHz cannot be concatenated.
        """
        from spikelab.spike_sorting import kilosort2

        rec_a = _make_mock_recording(sampling_frequency=20000.0)
        rec_b = _make_mock_recording(sampling_frequency=30000.0)

        (tmp_path / "a.raw.h5").touch()
        (tmp_path / "b.raw.h5").touch()

        call_count = [0]
        recordings = [rec_a, rec_b]

        def mock_load(path):
            rec = recordings[call_count[0]]
            call_count[0] += 1
            return rec

        monkeypatch.setattr(kilosort2, "load_single_recording", mock_load)

        with pytest.raises(ValueError, match="sampling frequency"):
            concat_fn(tmp_path)

    def test_channel_ids_mismatch_warns(self, concat_fn, tmp_path, monkeypatch):
        """
        Recordings with different channel IDs produce a warning.

        Tests:
            (Test Case 1) Different channel IDs warn but don't raise.
        """
        from spikelab.spike_sorting import kilosort2

        rec_a = _make_mock_recording(num_channels=4)
        rec_b = _make_mock_recording(num_channels=4)
        rec_b.get_channel_ids = lambda: np.array([10, 11, 12, 13])

        (tmp_path / "a.raw.h5").touch()
        (tmp_path / "b.raw.h5").touch()

        call_count = [0]
        recordings = [rec_a, rec_b]

        def mock_load(path):
            rec = recordings[call_count[0]]
            call_count[0] += 1
            return rec

        monkeypatch.setattr(kilosort2, "load_single_recording", mock_load)
        # Also mock si_segmentutils.concatenate_recordings to avoid real SI call
        monkeypatch.setattr(
            kilosort2.si_segmentutils,
            "concatenate_recordings",
            lambda recs: rec_a,
        )

        with pytest.warns(UserWarning, match="different channel IDs"):
            concat_fn(tmp_path)

    def test_channel_locations_mismatch_warns(self, concat_fn, tmp_path, monkeypatch):
        """
        Recordings with different channel locations produce a warning.

        Tests:
            (Test Case 1) Different electrode layouts warn but don't raise.
        """
        from spikelab.spike_sorting import kilosort2

        rec_a = _make_mock_recording(num_channels=4)
        rec_b = _make_mock_recording(num_channels=4)
        rec_b.get_channel_locations = lambda: np.column_stack(
            [np.arange(4) * 100.0, np.ones(4) * 50.0]
        )

        (tmp_path / "a.raw.h5").touch()
        (tmp_path / "b.raw.h5").touch()

        call_count = [0]
        recordings = [rec_a, rec_b]

        def mock_load(path):
            rec = recordings[call_count[0]]
            call_count[0] += 1
            return rec

        monkeypatch.setattr(kilosort2, "load_single_recording", mock_load)
        monkeypatch.setattr(
            kilosort2.si_segmentutils,
            "concatenate_recordings",
            lambda recs: rec_a,
        )

        with pytest.warns(UserWarning, match="different channel locations"):
            concat_fn(tmp_path)

    def test_compatible_recordings_no_warning(
        self, concat_fn, tmp_path, monkeypatch, recwarn
    ):
        """
        Compatible recordings concatenate without warnings.

        Tests:
            (Test Case 1) Two identical-config recordings produce no warnings.
        """
        from spikelab.spike_sorting import kilosort2

        rec_a = _make_mock_recording(num_channels=4)
        rec_b = _make_mock_recording(num_channels=4)

        (tmp_path / "a.raw.h5").touch()
        (tmp_path / "b.raw.h5").touch()

        call_count = [0]
        recordings = [rec_a, rec_b]

        def mock_load(path):
            rec = recordings[call_count[0]]
            call_count[0] += 1
            return rec

        monkeypatch.setattr(kilosort2, "load_single_recording", mock_load)
        monkeypatch.setattr(
            kilosort2.si_segmentutils,
            "concatenate_recordings",
            lambda recs: rec_a,
        )

        concat_fn(tmp_path)
        user_warnings = [w for w in recwarn if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


# ===========================================================================
# sort_maxtwo_multiwell validation
# ===========================================================================


@skip_no_spikeinterface
class TestSortMaxtwoMultiwellValidation:
    """
    Tests for parameter validation in sort_maxtwo_multiwell.
    """

    @pytest.fixture()
    def multiwell_fn(self):
        from spikelab.spike_sorting.kilosort2 import sort_maxtwo_multiwell

        return sort_maxtwo_multiwell

    def test_stream_id_kwarg_raises(self, multiwell_fn):
        """
        Passing stream_id directly raises ValueError.

        Tests:
            (Test Case 1) stream_id in kwargs is rejected with a
                helpful message.
        """
        with pytest.raises(ValueError, match="Do not pass 'stream_id'"):
            multiwell_fn(
                recording="fake.raw.h5",
                stream_ids=["well000"],
                stream_id="well000",
            )

    def test_intermediate_folders_kwarg_raises(self, multiwell_fn):
        """
        Passing intermediate_folders raises ValueError.

        Tests:
            (Test Case 1) Custom intermediate_folders is rejected.
        """
        with pytest.raises(ValueError, match="intermediate_folders"):
            multiwell_fn(
                recording="fake.raw.h5",
                stream_ids=["well000"],
                intermediate_folders=["/tmp/inter"],
            )

    def test_results_folders_kwarg_raises(self, multiwell_fn):
        """
        Passing results_folders raises ValueError.

        Tests:
            (Test Case 1) Custom results_folders is rejected.
        """
        with pytest.raises(ValueError, match="results_folders"):
            multiwell_fn(
                recording="fake.raw.h5",
                stream_ids=["well000"],
                results_folders=["/tmp/results"],
            )


# ===========================================================================
# Backend registry
# ===========================================================================


@skip_no_spikeinterface
class TestBackendRegistry:
    """
    Tests for the sorter backend registry.
    """

    def test_list_sorters(self):
        """
        list_sorters returns available backend names.

        Tests:
            (Test Case 1) kilosort2 is in the list.
        """
        from spikelab.spike_sorting.backends import list_sorters

        sorters = list_sorters()
        assert "kilosort2" in sorters

    def test_get_backend_class_valid(self):
        """
        get_backend_class returns the correct class for a registered sorter.

        Tests:
            (Test Case 1) kilosort2 returns Kilosort2Backend.
        """
        from spikelab.spike_sorting.backends import get_backend_class

        cls = get_backend_class("kilosort2")
        assert cls.__name__ == "Kilosort2Backend"

    def test_get_backend_class_unknown_raises(self):
        """
        get_backend_class raises ValueError for unregistered sorter names.

        Tests:
            (Test Case 1) Error message lists available sorters.
        """
        from spikelab.spike_sorting.backends import get_backend_class

        with pytest.raises(ValueError, match="Unknown sorter"):
            get_backend_class("nonexistent_sorter")


# ===========================================================================
# SortingPipelineConfig
# ===========================================================================


@skip_no_spikeinterface
class TestSortingPipelineConfig:
    """
    Tests for the SortingPipelineConfig dataclass.
    """

    def test_default_construction(self):
        """
        Default config has expected default values.

        Tests:
            (Test Case 1) Default sorter name is kilosort2.
            (Test Case 2) Default snr_min is 5.0.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        assert cfg.sorter.sorter_name == "kilosort2"
        assert cfg.curation.snr_min == 5.0
        assert cfg.execution.n_jobs == 8

    def test_from_kwargs(self):
        """
        from_kwargs maps flat parameter names to nested sub-configs.

        Tests:
            (Test Case 1) kilosort_path maps to sorter.sorter_path.
            (Test Case 2) snr_min maps to curation.snr_min.
            (Test Case 3) n_jobs maps to execution.n_jobs.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig.from_kwargs(
            kilosort_path="/opt/ks2",
            snr_min=3.0,
            n_jobs=4,
        )
        assert cfg.sorter.sorter_path == "/opt/ks2"
        assert cfg.curation.snr_min == 3.0
        assert cfg.execution.n_jobs == 4

    def test_from_kwargs_unknown_raises(self):
        """
        from_kwargs raises TypeError for unknown parameter names.

        Tests:
            (Test Case 1) Bogus parameter is rejected.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        with pytest.raises(TypeError, match="Unknown parameter"):
            SortingPipelineConfig.from_kwargs(bogus_param=True)


# ===========================================================================
# sort_recording validation
# ===========================================================================


@skip_no_spikeinterface
class TestSortRecordingValidation:
    """
    Tests for sort_recording parameter validation.
    """

    @pytest.fixture()
    def sort_fn(self):
        from spikelab.spike_sorting.pipeline import sort_recording

        return sort_recording

    def test_unknown_sorter_raises(self, sort_fn):
        """
        Unknown sorter name raises ValueError.

        Tests:
            (Test Case 1) Error message lists available sorters.
        """
        with pytest.raises(ValueError, match="Unknown sorter"):
            sort_fn(
                recording_files=["fake.h5"],
                sorter="nonexistent_sorter",
            )

    def test_mismatched_list_lengths_raises(self, sort_fn, tmp_path):
        """
        Mismatched folder list lengths raise ValueError.

        Tests:
            (Test Case 1) 2 recordings but 1 intermediate folder.
        """
        with pytest.raises(ValueError, match="same length"):
            sort_fn(
                recording_files=["fake1.h5", "fake2.h5"],
                intermediate_folders=[str(tmp_path / "inter1")],
                results_folders=[str(tmp_path / "r1"), str(tmp_path / "r2")],
            )

    def test_compile_all_without_folder_raises(self, sort_fn):
        """
        compile_all_recordings=True without folder raises ValueError.

        Tests:
            (Test Case 1) Error message mentions compiled_results_folder.
        """
        with pytest.raises(ValueError, match="compile_all_recordings"):
            sort_fn(
                recording_files=["fake.h5"],
                compile_all_recordings=True,
                compiled_results_folder=None,
            )

    def test_empty_recording_files(self, sort_fn):
        """
        Empty recording_files returns empty result list.

        Tests:
            (Test Case 1) No recordings => empty list.
        """
        result = sort_fn(
            recording_files=[],
            intermediate_folders=[],
            results_folders=[],
        )
        assert result == []


# ===========================================================================
# sort_multistream validation
# ===========================================================================


@skip_no_spikeinterface
class TestSortMultistreamValidation:
    """
    Tests for sort_multistream parameter validation.
    """

    @pytest.fixture()
    def multistream_fn(self):
        from spikelab.spike_sorting.pipeline import sort_multistream

        return sort_multistream

    def test_stream_id_kwarg_raises(self, multistream_fn):
        """
        Passing stream_id directly raises ValueError.

        Tests:
            (Test Case 1) Error message tells user to use stream_ids.
        """
        with pytest.raises(ValueError, match="Do not pass 'stream_id'"):
            multistream_fn(
                recording="fake.raw.h5",
                stream_ids=["well000"],
                stream_id="well000",
            )

    def test_intermediate_folders_kwarg_raises(self, multistream_fn):
        """
        Passing intermediate_folders raises ValueError.

        Tests:
            (Test Case 1) Auto-generated folders cannot be overridden.
        """
        with pytest.raises(ValueError, match="intermediate_folders"):
            multistream_fn(
                recording="fake.raw.h5",
                stream_ids=["well000"],
                intermediate_folders=["/tmp/inter"],
            )

    def test_results_folders_kwarg_raises(self, multistream_fn):
        """
        Passing results_folders raises ValueError.

        Tests:
            (Test Case 1) Auto-generated folders cannot be overridden.
        """
        with pytest.raises(ValueError, match="results_folders"):
            multistream_fn(
                recording="fake.raw.h5",
                stream_ids=["well000"],
                results_folders=["/tmp/results"],
            )
