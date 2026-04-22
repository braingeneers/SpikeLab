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

try:
    import torch  # noqa: F401

    _has_torch = True
except Exception:
    _has_torch = False

skip_no_spikeinterface = pytest.mark.skipif(
    not _has_spikeinterface, reason="spikeinterface not installed"
)
skip_no_pandas = pytest.mark.skipif(not _has_pandas, reason="pandas not installed")
skip_no_torch = pytest.mark.skipif(not _has_torch, reason="torch not installed")


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

    def test_all_contains_sort_recording(self):
        """
        The __all__ list advertises sort_recording and sort_multistream.

        Tests:
            (Test Case 1) sort_recording is in __all__.
            (Test Case 2) sort_multistream is in __all__.
        """
        import spikelab.spike_sorting as pkg

        assert "sort_recording" in pkg.__all__
        assert "sort_multistream" in pkg.__all__

    @skip_no_spikeinterface
    def test_lazy_import_succeeds_when_deps_available(self):
        """
        sort_recording is importable when spikeinterface is present.

        Tests:
            (Test Case 1) Attribute access returns a callable.
        """
        import spikelab.spike_sorting as pkg

        fn = pkg.sort_recording
        assert callable(fn)


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
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor

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
        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

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

        import spikelab.spike_sorting._globals as ks_mod

        old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {"keep_good_only": False}
        try:
            kse = ks_module.KilosortSortingExtractor(folder)
            assert kse.ms_to_samples(0) == 0
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params

    def test_missing_params_py(self, tmp_path):
        """Missing params.py raises FileNotFoundError."""
        folder = tmp_path / "ks_out"
        folder.mkdir()
        np.save(str(folder / "spike_times.npy"), np.array([0, 1]))
        np.save(str(folder / "spike_clusters.npy"), np.array([0, 0]))
        # No params.py

        from spikelab.spike_sorting.sorting_extractor import (
            KilosortSortingExtractor,
        )

        with pytest.raises(FileNotFoundError):
            KilosortSortingExtractor(folder)

    def test_missing_both_cluster_id_columns(self, tmp_path):
        """TSV with neither 'cluster_id' nor 'id' column raises ValueError."""
        folder = tmp_path / "ks_out"
        _write_ks_folder(
            folder,
            spike_times=np.array([0, 10, 20]),
            spike_clusters=np.array([0, 0, 1]),
        )
        # Overwrite TSV with wrong column name
        (folder / "cluster_info.tsv").write_text("unit\tgroup\n0\tgood\n1\tgood\n")

        from spikelab.spike_sorting.sorting_extractor import (
            KilosortSortingExtractor,
        )

        with pytest.raises(ValueError, match="cluster_id"):
            KilosortSortingExtractor(folder)

    def test_nonexistent_unit_spike_train(self, tmp_path):
        """get_unit_spike_train for a non-existent unit returns empty array."""
        folder = tmp_path / "ks_out"
        _write_ks_folder(
            folder,
            spike_times=np.array([0, 10, 20]),
            spike_clusters=np.array([0, 0, 0]),
        )

        from spikelab.spike_sorting.sorting_extractor import (
            KilosortSortingExtractor,
        )

        kse = KilosortSortingExtractor(folder)
        result = kse.get_unit_spike_train(unit_id=999)
        assert len(result) == 0 or (len(result) == 1 and result[0] == 999)


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
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor
        import spikelab.spike_sorting._globals as ks_mod

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
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor
        import spikelab.spike_sorting._globals as ks_mod

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
    Tests for build_spikedata conversion function (pipeline.py).

    Tests:
        (Test Case 1) Produces a SpikeData with correct number of units.
        (Test Case 2) Spike times are converted to milliseconds.
        (Test Case 3) Metadata contains source_file, source_format, fs_Hz.
        (Test Case 4) neuron_attributes contain enriched per-unit data.
    """

    @pytest.fixture()
    def convert_fn(self):
        from spikelab.spike_sorting.pipeline import build_spikedata

        return build_spikedata

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
    def _make_mock_config():
        """Build a minimal mock config for build_spikedata."""
        waveform = SimpleNamespace(
            compiled_ms_before=2,
            compiled_ms_after=2,
            scale_compiled_waveforms=True,
            std_at_peak=True,
            std_over_window_ms_before=0.5,
            std_over_window_ms_after=1.5,
        )
        sorter = SimpleNamespace(sorter_name="kilosort2")
        return SimpleNamespace(waveform=waveform, sorter=sorter)

    @staticmethod
    def _patch_globals(monkeypatch):
        """Patch _get_noise_levels in pipeline to return simple noise array."""
        from spikelab.spike_sorting import pipeline

        monkeypatch.setattr(
            pipeline,
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

        sd = convert_fn(we, "/fake/recording.h5", self._make_mock_config())

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

        sd = convert_fn(we, "test.h5", self._make_mock_config())
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

        sd = convert_fn(we, "test.h5", self._make_mock_config())
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

        sd = convert_fn(we, "test.h5", self._make_mock_config())
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

        sd = convert_fn(we, "test.h5", self._make_mock_config())
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
        from spikelab.spike_sorting.ks2_runner import ShellScript

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
        from spikelab.spike_sorting.waveform_extractor import Utils

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
        from spikelab.spike_sorting.waveform_extractor import Utils

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
        import spikelab.spike_sorting._globals as ks_mod
        import spikelab.spike_sorting.ks2_runner as ks_runner_mod

        self._ks_mod = ks_mod
        self._ks_runner_mod = ks_runner_mod
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
        from spikelab.spike_sorting.ks2_runner import _spike_sort_docker

        output_folder = tmp_path / "ks_output"
        sorter_output = output_folder / "sorter_output"
        self._write_fake_phy_output(sorter_output)

        recording = _make_mock_recording()
        mock_rs = MagicMock(return_value=None)

        with (
            patch("spikelab.spike_sorting.ks2_runner.write_binary_recording"),
            patch("spikelab.spike_sorting.ks2_runner.BinaryRecordingExtractor"),
            patch("spikelab.spike_sorting.ks2_runner.run_sorter", mock_rs),
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
        from spikelab.spike_sorting.ks2_runner import _spike_sort_docker

        output_folder = tmp_path / "ks_output"
        sorter_output = output_folder / "sorter_output"
        self._write_fake_phy_output(sorter_output)

        recording = _make_mock_recording()

        with (
            patch("spikelab.spike_sorting.ks2_runner.write_binary_recording"),
            patch("spikelab.spike_sorting.ks2_runner.BinaryRecordingExtractor"),
            patch("spikelab.spike_sorting.ks2_runner.run_sorter", MagicMock()),
        ):
            result = _spike_sort_docker(recording, output_folder)

        assert result.folder == sorter_output.absolute()

    def test_spike_sort_docker_fallback_to_output_folder(self, tmp_path):
        """
        _spike_sort_docker falls back to output_folder when no sorter_output/ exists.

        Tests:
            (Test Case 1) Phy files directly in output_folder are found.
        """
        from spikelab.spike_sorting.ks2_runner import _spike_sort_docker

        output_folder = tmp_path / "ks_output"
        self._write_fake_phy_output(output_folder)

        recording = _make_mock_recording()

        with (
            patch("spikelab.spike_sorting.ks2_runner.write_binary_recording"),
            patch("spikelab.spike_sorting.ks2_runner.BinaryRecordingExtractor"),
            patch("spikelab.spike_sorting.ks2_runner.run_sorter", MagicMock()),
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
        from spikelab.spike_sorting.ks2_runner import spike_sort

        self._ks_mod.USE_DOCKER = True
        output_folder = tmp_path / "ks_output"
        recording = _make_mock_recording()

        mock_kse = SimpleNamespace(unit_ids=[0, 1])

        with (
            patch.object(
                self._ks_runner_mod, "_spike_sort_docker", return_value=mock_kse
            ) as mock_docker,
            patch.object(self._ks_runner_mod, "RunKilosort") as mock_rk,
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
        from spikelab.spike_sorting.ks2_runner import spike_sort

        self._ks_mod.USE_DOCKER = False
        output_folder = tmp_path / "ks_output"
        recording = _make_mock_recording()

        mock_sorting = SimpleNamespace(unit_ids=[0])
        mock_ks_instance = MagicMock()
        mock_ks_instance.run.return_value = mock_sorting

        with (
            patch.object(
                self._ks_runner_mod, "RunKilosort", return_value=mock_ks_instance
            ) as mock_rk,
            patch.object(self._ks_runner_mod, "_spike_sort_docker") as mock_docker,
            patch.object(self._ks_runner_mod, "write_recording"),
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
        from spikelab.spike_sorting.ks2_runner import spike_sort

        self._ks_mod.USE_DOCKER = True
        output_folder = tmp_path / "ks_output"
        recording = _make_mock_recording()

        with patch.object(
            self._ks_runner_mod,
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
        from spikelab.spike_sorting.sorting_utils import print_stage

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
# _time_chunks_to_frames
# ===========================================================================


@skip_no_spikeinterface
class TestTimeChunksToFrames:
    """
    Tests for the time-to-frame conversion helper used by load_recording.

    Tests:
        (Test Case 1) start_time_s/end_time_s produces a single frame range.
        (Test Case 2) end_time_s exceeding duration is clipped.
        (Test Case 3) Invalid range (start >= end) raises ValueError.
        (Test Case 4) rec_chunks_s produces multiple frame ranges.
        (Test Case 5) Empty inputs produce an empty list.
        (Test Case 6) start_time_s defaults to 0 when only end_time_s given.
        (Test Case 7) end_time_s defaults to total_duration_s when only start_time_s given.
    """

    @pytest.fixture()
    def helper(self):
        from spikelab.spike_sorting.recording_io import _time_chunks_to_frames

        return _time_chunks_to_frames

    def test_single_range(self, helper):
        chunks = helper(
            start_time_s=3 * 60,
            end_time_s=5 * 60,
            rec_chunks_s=[],
            fs=20000.0,
            total_duration_s=600.0,
        )
        assert chunks == [(3 * 60 * 20000, 5 * 60 * 20000)]

    def test_end_time_clipped_to_duration(self, helper, capsys):
        chunks = helper(
            start_time_s=0.0,
            end_time_s=100.0,
            rec_chunks_s=[],
            fs=1000.0,
            total_duration_s=50.0,
        )
        assert chunks == [(0, 50000)]
        assert "clipping" in capsys.readouterr().out

    def test_invalid_range_raises(self, helper):
        with pytest.raises(ValueError, match="Invalid time range"):
            helper(
                start_time_s=10.0,
                end_time_s=5.0,
                rec_chunks_s=[],
                fs=1000.0,
                total_duration_s=100.0,
            )

    def test_negative_start_raises(self, helper):
        with pytest.raises(ValueError, match="Invalid time range"):
            helper(
                start_time_s=-1.0,
                end_time_s=5.0,
                rec_chunks_s=[],
                fs=1000.0,
                total_duration_s=100.0,
            )

    def test_rec_chunks_s_multiple_ranges(self, helper):
        chunks = helper(
            start_time_s=None,
            end_time_s=None,
            rec_chunks_s=[(0.0, 10.0), (30.0, 40.0)],
            fs=1000.0,
            total_duration_s=60.0,
        )
        assert chunks == [(0, 10000), (30000, 40000)]

    def test_empty_inputs_return_empty(self, helper):
        chunks = helper(
            start_time_s=None,
            end_time_s=None,
            rec_chunks_s=[],
            fs=1000.0,
            total_duration_s=60.0,
        )
        assert chunks == []

    def test_start_time_defaults_to_zero(self, helper):
        chunks = helper(
            start_time_s=None,
            end_time_s=5.0,
            rec_chunks_s=[],
            fs=1000.0,
            total_duration_s=60.0,
        )
        assert chunks == [(0, 5000)]

    def test_end_time_defaults_to_duration(self, helper):
        chunks = helper(
            start_time_s=10.0,
            end_time_s=None,
            rec_chunks_s=[],
            fs=1000.0,
            total_duration_s=60.0,
        )
        assert chunks == [(10000, 60000)]

    def test_combined_start_end_and_rec_chunks_s(self, helper):
        """start_time_s/end_time_s and rec_chunks_s can coexist — both are included."""
        chunks = helper(
            start_time_s=0.0,
            end_time_s=10.0,
            rec_chunks_s=[(20.0, 30.0)],
            fs=1000.0,
            total_duration_s=60.0,
        )
        assert chunks == [(0, 10000), (20000, 30000)]


# ===========================================================================
# SortingPipelineConfig time-slicing kwargs
# ===========================================================================


@skip_no_spikeinterface
class TestSortingConfigTimeSlicingKwargs:
    """
    Tests that start_time_s / end_time_s / rec_chunks_s are accepted by
    SortingPipelineConfig.from_kwargs() and stored on the recording config.
    """

    def test_start_and_end_time_s_accepted(self):
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig.from_kwargs(start_time_s=60.0, end_time_s=120.0)
        assert cfg.recording.start_time_s == 60.0
        assert cfg.recording.end_time_s == 120.0

    def test_rec_chunks_s_accepted(self):
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig.from_kwargs(
            rec_chunks_s=[(0.0, 30.0), (60.0, 90.0)]
        )
        assert cfg.recording.rec_chunks_s == [(0.0, 30.0), (60.0, 90.0)]

    def test_defaults_are_none_and_empty(self):
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        assert cfg.recording.start_time_s is None
        assert cfg.recording.end_time_s is None
        assert cfg.recording.rec_chunks_s == []


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
        from spikelab.spike_sorting import _globals, recording_io

        monkeypatch.setattr(_globals, "REC_CHUNKS", [], raising=False)
        monkeypatch.setattr(_globals, "_REC_CHUNK_NAMES", [], raising=False)
        monkeypatch.setattr(_globals, "STREAM_ID", None, raising=False)
        monkeypatch.setattr(_globals, "GAIN_TO_UV", None, raising=False)
        monkeypatch.setattr(_globals, "OFFSET_TO_UV", None, raising=False)
        monkeypatch.setattr(_globals, "FREQ_MIN", 300, raising=False)
        monkeypatch.setattr(_globals, "FREQ_MAX", 6000, raising=False)
        monkeypatch.setattr(_globals, "FIRST_N_MINS", None, raising=False)
        monkeypatch.setattr(_globals, "MEA_Y_MAX", None, raising=False)
        return recording_io.concatenate_recordings

    def test_channel_count_mismatch_raises(self, concat_fn, tmp_path, monkeypatch):
        """
        Recordings with different channel counts raise ValueError.

        Tests:
            (Test Case 1) Two files with 4 vs 2 channels cannot be
                concatenated.
        """
        from spikelab.spike_sorting import recording_io

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

        monkeypatch.setattr(recording_io, "load_single_recording", mock_load)

        with pytest.raises(ValueError, match="channels"):
            concat_fn(tmp_path)

    def test_sampling_frequency_mismatch_raises(self, concat_fn, tmp_path, monkeypatch):
        """
        Recordings with different sampling frequencies raise ValueError.

        Tests:
            (Test Case 1) 20 kHz vs 30 kHz cannot be concatenated.
        """
        from spikelab.spike_sorting import recording_io

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

        monkeypatch.setattr(recording_io, "load_single_recording", mock_load)

        with pytest.raises(ValueError, match="sampling frequency"):
            concat_fn(tmp_path)

    def test_channel_ids_mismatch_warns(self, concat_fn, tmp_path, monkeypatch):
        """
        Recordings with different channel IDs produce a warning.

        Tests:
            (Test Case 1) Different channel IDs warn but don't raise.
        """
        from spikelab.spike_sorting import recording_io

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

        monkeypatch.setattr(recording_io, "load_single_recording", mock_load)
        # Also mock si_segmentutils.concatenate_recordings to avoid real SI call
        monkeypatch.setattr(
            recording_io.si_segmentutils,
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
        from spikelab.spike_sorting import recording_io

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

        monkeypatch.setattr(recording_io, "load_single_recording", mock_load)
        monkeypatch.setattr(
            recording_io.si_segmentutils,
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
        from spikelab.spike_sorting import recording_io

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

        monkeypatch.setattr(recording_io, "load_single_recording", mock_load)
        monkeypatch.setattr(
            recording_io.si_segmentutils,
            "concatenate_recordings",
            lambda recs: rec_a,
        )

        concat_fn(tmp_path)
        user_warnings = [w for w in recwarn if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


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

    def test_override(self):
        """
        override returns a new config with selected fields changed.

        Tests:
            (Test Case 1) Override changes the specified field.
            (Test Case 2) Original config is unchanged.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        original = SortingPipelineConfig()
        modified = original.override(snr_min=8.0, use_docker=True)
        assert modified.curation.snr_min == 8.0
        assert modified.sorter.use_docker is True
        assert original.curation.snr_min == 5.0
        assert original.sorter.use_docker is False

    def test_presets_exist(self):
        """
        Preset configs are importable and have correct sorter names.

        Tests:
            (Test Case 1) KILOSORT2 has sorter_name kilosort2.
            (Test Case 2) KILOSORT2_DOCKER has use_docker=True.
        """
        from spikelab.spike_sorting.config import (
            KILOSORT2,
            KILOSORT2_DOCKER,
        )

        assert KILOSORT2.sorter.sorter_name == "kilosort2"
        assert KILOSORT2.sorter.use_docker is False
        assert KILOSORT2_DOCKER.sorter.use_docker is True

    def test_sort_recording_with_config(self):
        """
        sort_recording accepts a config= parameter.

        Tests:
            (Test Case 1) Empty recording list with config returns empty.
        """
        from spikelab.spike_sorting.config import KILOSORT2
        from spikelab.spike_sorting.pipeline import sort_recording

        result = sort_recording(
            recording_files=[],
            config=KILOSORT2,
            intermediate_folders=[],
            results_folders=[],
        )
        assert result == []

    def test_from_kwargs_empty_dict(self):
        """Empty kwargs produces default config."""
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig.from_kwargs()
        default = SortingPipelineConfig()
        assert cfg.recording.freq_min == default.recording.freq_min
        assert cfg.sorter.sorter_name == default.sorter.sorter_name

    def test_from_kwargs_unknown_key_raises(self):
        """Unknown flat key raises TypeError."""
        from spikelab.spike_sorting.config import SortingPipelineConfig

        with pytest.raises(TypeError, match="Unknown parameter"):
            SortingPipelineConfig.from_kwargs(nonexistent_key=42)

    def test_override_empty_kwargs_deep_copy(self):
        """Override with empty kwargs returns a deep copy."""
        from spikelab.spike_sorting.config import SortingPipelineConfig

        original = SortingPipelineConfig()
        original.recording.freq_min = 999
        copy = original.override()
        assert copy.recording.freq_min == 999
        # Mutating the copy must not affect the original
        copy.recording.freq_min = 123
        assert original.recording.freq_min == 999

    def test_override_unknown_key_raises(self):
        """Override with unknown key raises TypeError."""
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        with pytest.raises(TypeError, match="Unknown parameter"):
            cfg.override(bogus_param="x")


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


# ===========================================================================
# Docker utilities
# ===========================================================================


class TestDockerUtilsGetHostCudaDriverVersion:
    """
    Tests for get_host_cuda_driver_version.

    Tests:
        (Test Case 1) Parses typical nvidia-smi output.
        (Test Case 2) Returns None when nvidia-smi is not found.
        (Test Case 3) Returns None when nvidia-smi times out.
        (Test Case 4) Parses multi-GPU output (first line).
    """

    def test_parses_typical_output(self):
        """
        Parses a typical nvidia-smi driver version string.

        Tests:
            (Test Case 1) "590.44.01" → 590.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_driver_version

        with patch(
            "spikelab.spike_sorting.docker_utils.subprocess.check_output",
            return_value="590.44.01\n",
        ):
            assert get_host_cuda_driver_version() == 590

    def test_returns_none_when_nvidia_smi_missing(self):
        """
        Returns None when nvidia-smi is not installed.

        Tests:
            (Test Case 1) FileNotFoundError → None.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_driver_version

        with patch(
            "spikelab.spike_sorting.docker_utils.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert get_host_cuda_driver_version() is None

    def test_returns_none_on_timeout(self):
        """
        Returns None when nvidia-smi times out.

        Tests:
            (Test Case 1) subprocess.TimeoutExpired → None.
        """
        import subprocess as sp

        from spikelab.spike_sorting.docker_utils import get_host_cuda_driver_version

        with patch(
            "spikelab.spike_sorting.docker_utils.subprocess.check_output",
            side_effect=sp.TimeoutExpired(cmd="nvidia-smi", timeout=10),
        ):
            assert get_host_cuda_driver_version() is None

    def test_returns_none_on_malformed_output(self):
        """
        Returns None when nvidia-smi output is not parseable.

        Tests:
            (Test Case 1) Non-numeric output → None.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_driver_version

        with patch(
            "spikelab.spike_sorting.docker_utils.subprocess.check_output",
            return_value="N/A\n",
        ):
            assert get_host_cuda_driver_version() is None


class TestDockerUtilsGetHostCudaTag:
    """
    Tests for get_host_cuda_tag.

    Tests:
        (Test Case 1) Driver version maps to the correct CUDA tag.
        (Test Case 2) Returns None when driver is too old.
        (Test Case 3) Returns None when no GPU detected.
    """

    def test_driver_560_maps_to_cu130(self):
        """
        Driver 560+ maps to cu130.

        Tests:
            (Test Case 1) Exact boundary at 560.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_tag

        with patch(
            "spikelab.spike_sorting.docker_utils.get_host_cuda_driver_version",
            return_value=560,
        ):
            assert get_host_cuda_tag() == "cu130"

    def test_driver_550_maps_to_cu126(self):
        """
        Driver 550-559 maps to cu126.

        Tests:
            (Test Case 1) Driver 550 is below cu130 threshold.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_tag

        with patch(
            "spikelab.spike_sorting.docker_utils.get_host_cuda_driver_version",
            return_value=550,
        ):
            assert get_host_cuda_tag() == "cu126"

    def test_driver_525_maps_to_cu118(self):
        """
        Driver 525 maps to cu118 (lowest supported).

        Tests:
            (Test Case 1) Boundary at 525.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_tag

        with patch(
            "spikelab.spike_sorting.docker_utils.get_host_cuda_driver_version",
            return_value=525,
        ):
            assert get_host_cuda_tag() == "cu118"

    def test_driver_too_old_returns_none(self):
        """
        Driver below all thresholds returns None.

        Tests:
            (Test Case 1) Driver 500 has no match.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_tag

        with patch(
            "spikelab.spike_sorting.docker_utils.get_host_cuda_driver_version",
            return_value=500,
        ):
            assert get_host_cuda_tag() is None

    def test_no_gpu_returns_none(self):
        """
        Returns None when no GPU is detected.

        Tests:
            (Test Case 1) get_host_cuda_driver_version returns None.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_tag

        with patch(
            "spikelab.spike_sorting.docker_utils.get_host_cuda_driver_version",
            return_value=None,
        ):
            assert get_host_cuda_tag() is None


class TestDockerUtilsGetDockerImage:
    """
    Tests for get_docker_image.

    Tests:
        (Test Case 1) KS2 returns default image regardless of CUDA tag.
        (Test Case 2) KS4 exact CUDA tag match.
        (Test Case 3) KS4 auto-detects CUDA from host.
        (Test Case 4) KS4 falls back to newest tag with warning.
        (Test Case 5) Unknown sorter raises ValueError.
        (Test Case 6) No compatible image raises RuntimeError.
        (Test Case 7) Auto-detect fails (no GPU) raises RuntimeError.
    """

    def test_ks2_returns_default(self):
        """
        KS2 always returns its single default image.

        Tests:
            (Test Case 1) CUDA tag is ignored for KS2.
        """
        from spikelab.spike_sorting.docker_utils import get_docker_image

        image = get_docker_image("kilosort2", cuda_tag="cu118")
        assert "kilosort2" in image
        assert image == get_docker_image("kilosort2")

    def test_ks4_exact_cuda_match(self):
        """
        KS4 with an exact CUDA tag returns the matching image.

        Tests:
            (Test Case 1) cu130 matches the registered image.
        """
        from spikelab.spike_sorting.docker_utils import get_docker_image

        image = get_docker_image("kilosort4", cuda_tag="cu130")
        assert "kilosort4" in image

    def test_ks4_auto_detects_cuda(self):
        """
        KS4 auto-detects CUDA when no tag is provided.

        Tests:
            (Test Case 1) Host with driver 560 → cu130 image.
        """
        from spikelab.spike_sorting.docker_utils import get_docker_image

        with patch(
            "spikelab.spike_sorting.docker_utils.get_host_cuda_tag",
            return_value="cu130",
        ):
            image = get_docker_image("kilosort4")
            assert "kilosort4" in image

    def test_ks4_fallback_to_newest_with_warning(self):
        """
        KS4 falls back to newest available image when exact match missing.

        Tests:
            (Test Case 1) cu118 has no KS4 image; falls back to cu130 with warning.
        """
        from spikelab.spike_sorting.docker_utils import get_docker_image

        with pytest.warns(UserWarning, match="No Docker image.*cu118"):
            image = get_docker_image("kilosort4", cuda_tag="cu118")
        assert "kilosort4" in image

    def test_unknown_sorter_raises(self):
        """
        Unknown sorter name raises ValueError.

        Tests:
            (Test Case 1) "mountainsort" is not registered.
        """
        from spikelab.spike_sorting.docker_utils import get_docker_image

        with pytest.raises(ValueError, match="No Docker images registered"):
            get_docker_image("mountainsort")

    def test_no_gpu_auto_detect_raises(self):
        """
        Auto-detect raises RuntimeError when no GPU is found.

        Tests:
            (Test Case 1) get_host_cuda_tag returns None → RuntimeError.
        """
        from spikelab.spike_sorting.docker_utils import get_docker_image

        with patch(
            "spikelab.spike_sorting.docker_utils.get_host_cuda_tag",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="Could not detect CUDA"):
                get_docker_image("kilosort4")

    def test_no_compatible_image_raises(self):
        """
        Raises RuntimeError when no image matches and newest fallback is absent.

        Tests:
            (Test Case 1) Registry with no matching or fallback tags.
        """
        from spikelab.spike_sorting import docker_utils

        original_registry = docker_utils._IMAGE_REGISTRY
        try:
            # Registry where newest tag (cu130) has no entry
            docker_utils._IMAGE_REGISTRY = {
                "test_sorter": {"cu118": "test:cu118"},
            }
            with pytest.raises(RuntimeError, match="No compatible Docker image"):
                docker_utils.get_docker_image("test_sorter", cuda_tag="cu121")
        finally:
            docker_utils._IMAGE_REGISTRY = original_registry


# ===========================================================================
# KS4 Docker branch
# ===========================================================================


@skip_no_spikeinterface
class TestKilosort4BackendDockerBranch:
    """
    Tests for Kilosort4Backend._run_sorting Docker branch.

    Tests:
        (Test Case 1) Docker kwargs are constructed correctly when USE_DOCKER=True.
        (Test Case 2) Custom docker image string is passed through.
        (Test Case 3) No docker kwargs when USE_DOCKER is falsy.
    """

    @pytest.fixture(autouse=True)
    def _set_globals(self):
        import spikelab.spike_sorting._globals as ks_mod

        self._ks_mod = ks_mod
        self._old_docker = getattr(ks_mod, "USE_DOCKER", None)
        self._old_recompute = getattr(ks_mod, "RECOMPUTE_SORTING", None)
        self._old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {}
        ks_mod.RECOMPUTE_SORTING = True
        yield
        if self._old_docker is not None:
            ks_mod.USE_DOCKER = self._old_docker
        if self._old_recompute is not None:
            ks_mod.RECOMPUTE_SORTING = self._old_recompute
        if self._old_params is not None:
            ks_mod.KILOSORT_PARAMS = self._old_params

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

    @pytest.fixture()
    def ks4_backend(self):
        """Create a Kilosort4Backend with a default config."""
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        config = SortingPipelineConfig()
        return Kilosort4Backend(config)

    def test_use_docker_true_passes_image_and_no_install(self, tmp_path, ks4_backend):
        """
        When USE_DOCKER=True, run_sorter receives docker_image and installation_mode.

        Tests:
            (Test Case 1) docker_image is auto-detected via get_docker_image.
            (Test Case 2) installation_mode is 'pypi' (SI 0.104 workaround).
        """
        self._ks_mod.USE_DOCKER = True
        output_folder = tmp_path / "ks4_output"
        sorter_output = output_folder / "sorter_output"
        self._write_fake_phy_output(sorter_output)

        recording = _make_mock_recording()
        mock_rs = MagicMock(return_value=None)

        with (
            patch(
                "spikeinterface.sorters.run_sorter",
                mock_rs,
            ),
            patch(
                "spikelab.spike_sorting.docker_utils.get_docker_image",
                return_value="spikeinterface/kilosort4-base:test",
            ),
        ):
            ks4_backend.sort(recording, "fake.raw", "fake.dat", output_folder)

        _, call_kwargs = mock_rs.call_args
        assert call_kwargs["docker_image"] == "spikeinterface/kilosort4-base:test"
        assert call_kwargs["installation_mode"] == "pypi"

    def test_use_docker_custom_string(self, tmp_path, ks4_backend):
        """
        When USE_DOCKER is a string, it is passed directly as docker_image.

        Tests:
            (Test Case 1) Custom image string bypasses auto-detection.
        """
        self._ks_mod.USE_DOCKER = "my-custom-image:latest"
        output_folder = tmp_path / "ks4_output"
        sorter_output = output_folder / "sorter_output"
        self._write_fake_phy_output(sorter_output)

        recording = _make_mock_recording()
        mock_rs = MagicMock(return_value=None)

        with patch(
            "spikeinterface.sorters.run_sorter",
            mock_rs,
        ):
            ks4_backend.sort(recording, "fake.raw", "fake.dat", output_folder)

        _, call_kwargs = mock_rs.call_args
        assert call_kwargs["docker_image"] == "my-custom-image:latest"

    def test_no_docker_kwargs_when_disabled(self, tmp_path, ks4_backend):
        """
        When USE_DOCKER is falsy, no docker_image or installation_mode is passed.

        Tests:
            (Test Case 1) docker_image not in kwargs.
            (Test Case 2) installation_mode not in kwargs.
        """
        self._ks_mod.USE_DOCKER = False
        output_folder = tmp_path / "ks4_output"
        sorter_output = output_folder / "sorter_output"
        self._write_fake_phy_output(sorter_output)

        recording = _make_mock_recording()
        mock_rs = MagicMock(return_value=None)

        with patch(
            "spikeinterface.sorters.run_sorter",
            mock_rs,
        ):
            ks4_backend.sort(recording, "fake.raw", "fake.dat", output_folder)

        _, call_kwargs = mock_rs.call_args
        assert "docker_image" not in call_kwargs
        assert "installation_mode" not in call_kwargs


# ===========================================================================
# Updated half-window-sizes logic (KS4-style dense templates)
# ===========================================================================


class TestTemplateHalfWindowDenseTemplates:
    """
    Tests for _get_templates_half_windows_sizes with dense (non-zero-edge) templates.

    The updated logic uses a 1%-of-peak threshold instead of searching for
    exact zeros. This test covers the KS4-style case where template edges
    are non-zero.

    Tests:
        (Test Case 1) Dense template with no zeros still produces a valid window.
        (Test Case 2) Template with uniform small values gives full half-window.
    """

    def test_dense_template_nonzero_edges(self, tmp_path):
        """
        Dense template with all pre-mid amplitudes above 1% of peak.

        Tests:
            (Test Case 1) No small_indices found → size = template_mid.
            (Test Case 2) Result is int(template_mid * 0.75) = 22.
        """
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor
        import spikelab.spike_sorting._globals as ks_mod

        spike_times = np.array([10, 20], dtype=np.int64)
        spike_clusters = np.array([0, 0], dtype=np.int64)

        # 1 template, 61 samples, 2 channels — dense non-zero background
        # Background of 2.0 is above 1% of peak (100 * 0.01 = 1.0)
        templates = np.full((1, 61, 2), 2.0, dtype=np.float32)
        # Large peak at midpoint on channel 0
        templates[0, 30, 0] = -100.0

        channel_map = np.array([0, 1])
        folder = tmp_path / "dense_template"
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
            _, chans_ks, _ = kse.get_chans_max()
            hw_sizes = kse.get_templates_half_windows_sizes(chans_ks)
            assert len(hw_sizes) == 1
            # All pre-mid values (abs=2.0) are above threshold (1.0),
            # so no small_indices → size = template_mid = 30
            # Result: int(30 * 0.75) = 22
            assert hw_sizes[0] == 22
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params
            if old_pos_peak is not None:
                ks_mod.POS_PEAK_THRESH = old_pos_peak

    def test_template_with_small_nonzero_edges(self, tmp_path):
        """
        Template with small but non-zero edges produces a tight window.

        Tests:
            (Test Case 1) Edges below 1% of peak are treated like zeros.
            (Test Case 2) Window is smaller than template_mid.
        """
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor
        import spikelab.spike_sorting._globals as ks_mod

        spike_times = np.array([10, 20], dtype=np.int64)
        spike_clusters = np.array([0, 0], dtype=np.int64)

        # 1 template, 61 samples, 2 channels
        # Small background with a sharp peak — mimics a real waveform
        templates = np.full((1, 61, 2), 0.001, dtype=np.float32)
        templates[0, 30, 0] = -10.0  # peak at mid
        # Ramp up to peak in last 5 samples before mid
        templates[0, 25:30, 0] = np.linspace(-0.5, -10.0, 5)

        channel_map = np.array([0, 1])
        folder = tmp_path / "small_edge_template"
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
            _, chans_ks, _ = kse.get_chans_max()
            hw_sizes = kse.get_templates_half_windows_sizes(chans_ks)
            assert len(hw_sizes) == 1
            assert hw_sizes[0] > 0
            # Edge values (0.001) are below 1% of 10.0 = 0.1, so they're "small".
            # The ramp starts at index 25 with -0.5 which is above threshold.
            # So the last small index should be 24, giving size = 30 - 24 = 6.
            assert hw_sizes[0] < 30  # tighter than full half
        finally:
            if old_params is not None:
                ks_mod.KILOSORT_PARAMS = old_params
            if old_pos_peak is not None:
                ks_mod.POS_PEAK_THRESH = old_pos_peak


# ===========================================================================
# Edge case tests — docker_utils
# ===========================================================================


class TestDockerUtils:
    """
    Edge case tests for docker_utils functions.

    Tests:
        (Test Case 1) Multi-GPU nvidia-smi output with multiple lines.
        (Test Case 2) Two-part driver version string (e.g. "470.82").
        (Test Case 3) Empty string sorter name raises ValueError.
        (Test Case 4) High future driver version maps to newest tag.
    """

    def test_multi_gpu_output_parses_first_line(self):
        """
        Multi-GPU nvidia-smi output returns the first GPU's driver version.

        Tests:
            (Test Case 1) Two-line output "590.44.01\\n590.44.01" → 590.

        Notes:
            - nvidia-smi returns one line per GPU. strip() removes trailing
              newline, and split(".")[0] takes the major from the first line.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_driver_version

        with patch(
            "spikelab.spike_sorting.docker_utils.subprocess.check_output",
            return_value="590.44.01\n590.44.01\n",
        ):
            assert get_host_cuda_driver_version() == 590

    def test_two_part_driver_version(self):
        """
        Two-part driver version string (no patch number) parses correctly.

        Tests:
            (Test Case 1) "470.82" → 470.
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_driver_version

        with patch(
            "spikelab.spike_sorting.docker_utils.subprocess.check_output",
            return_value="470.82\n",
        ):
            assert get_host_cuda_driver_version() == 470

    def test_empty_string_sorter_raises(self):
        """
        Empty string sorter name raises ValueError.

        Tests:
            (Test Case 1) get_docker_image("") raises ValueError.
        """
        from spikelab.spike_sorting.docker_utils import get_docker_image

        with pytest.raises(ValueError, match="No Docker images registered"):
            get_docker_image("")

    def test_high_future_driver_maps_to_newest(self):
        """
        A very high driver version (future hardware) maps to the newest CUDA tag.

        Tests:
            (Test Case 1) Driver 700 → cu130 (newest entry).
        """
        from spikelab.spike_sorting.docker_utils import get_host_cuda_tag

        with patch(
            "spikelab.spike_sorting.docker_utils.get_host_cuda_driver_version",
            return_value=700,
        ):
            assert get_host_cuda_tag() == "cu130"


# ===========================================================================
# Edge case tests — _get_templates_half_windows_sizes
# ===========================================================================


class TestTemplateHalfWindow:
    """
    Edge case tests for _get_templates_half_windows_sizes.

    Tests:
        (Test Case 1) Zero-amplitude (flat) template.
        (Test Case 2) Single-sample template.
        (Test Case 3) window_size_scale = 0.
    """

    def _make_kse_with_templates(self, tmp_path, templates, folder_name="ec_template"):
        """Helper to create a KSE from given templates array."""
        from spikelab.spike_sorting.sorting_extractor import KilosortSortingExtractor
        import spikelab.spike_sorting._globals as ks_mod

        n_templates = templates.shape[0]
        n_channels = templates.shape[2]
        spike_times = np.array([10, 20], dtype=np.int64)
        spike_clusters = np.array([0, 0], dtype=np.int64)

        channel_map = np.arange(n_channels)
        folder = tmp_path / folder_name
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

        kse = KilosortSortingExtractor(folder)

        return kse, ks_mod, old_params, old_pos_peak

    def _restore(self, ks_mod, old_params, old_pos_peak):
        if old_params is not None:
            ks_mod.KILOSORT_PARAMS = old_params
        if old_pos_peak is not None:
            ks_mod.POS_PEAK_THRESH = old_pos_peak

    def test_zero_amplitude_template(self, tmp_path):
        """
        Flat zero template produces non-zero window size (full half-window).

        Tests:
            (Test Case 1) All-zero template → peak_amp=0, threshold=0,
                no indices below threshold → size=template_mid.

        Notes:
            - A dead channel with a flat zero template gets a full half-window
              because abs(0) < 0 is always False. This is arguably wrong but
              matches the current implementation.
        """
        templates = np.zeros((1, 61, 2), dtype=np.float32)
        kse, ks_mod, old_p, old_pp = self._make_kse_with_templates(
            tmp_path, templates, "zero_amp"
        )
        try:
            _, chans_ks, _ = kse.get_chans_max()
            hw_sizes = kse.get_templates_half_windows_sizes(chans_ks)
            assert len(hw_sizes) == 1
            # All-zero template: peak_amp=0, threshold=0,
            # abs(0) < 0 is False for all → no small_indices → size = template_mid
            assert hw_sizes[0] == int(30 * 0.75)  # 22
        finally:
            self._restore(ks_mod, old_p, old_pp)

    def test_single_sample_template(self, tmp_path):
        """
        Template with a single time sample produces window size 0.

        Tests:
            (Test Case 1) 1-sample template → template_mid=0,
                template[:0] is empty → size=0 → result 0.
        """
        # 1 template, 1 sample, 2 channels
        templates = np.array([[[5.0, 0.0]]], dtype=np.float32)
        kse, ks_mod, old_p, old_pp = self._make_kse_with_templates(
            tmp_path, templates, "single_sample"
        )
        try:
            _, chans_ks, _ = kse.get_chans_max()
            hw_sizes = kse.get_templates_half_windows_sizes(chans_ks)
            assert len(hw_sizes) == 1
            assert hw_sizes[0] == 0
        finally:
            self._restore(ks_mod, old_p, old_pp)

    def test_window_size_scale_zero(self, tmp_path):
        """
        window_size_scale=0 produces all-zero window sizes.

        Tests:
            (Test Case 1) Non-trivial template with scale=0 → size 0.
        """
        templates = np.zeros((1, 61, 2), dtype=np.float32)
        templates[0, 30, 0] = -10.0
        kse, ks_mod, old_p, old_pp = self._make_kse_with_templates(
            tmp_path, templates, "scale_zero"
        )
        try:
            _, chans_ks, _ = kse.get_chans_max()
            hw_sizes = kse.get_templates_half_windows_sizes(
                chans_ks, window_size_scale=0.0
            )
            assert len(hw_sizes) == 1
            assert hw_sizes[0] == 0
        finally:
            self._restore(ks_mod, old_p, old_pp)


# ===========================================================================
# Edge case tests — KS4 Docker branch
# ===========================================================================


@skip_no_spikeinterface
class TestKilosort4BackendDocker:
    """
    Edge case tests for Kilosort4Backend.sort() Docker branch.

    Tests:
        (Test Case 1) get_docker_image raises RuntimeError → returned as object.
        (Test Case 2) run_sorter raises → exception returned as object.
    """

    @pytest.fixture(autouse=True)
    def _set_globals(self):
        import spikelab.spike_sorting._globals as ks_mod

        self._ks_mod = ks_mod
        self._old_docker = getattr(ks_mod, "USE_DOCKER", None)
        self._old_recompute = getattr(ks_mod, "RECOMPUTE_SORTING", None)
        self._old_params = getattr(ks_mod, "KILOSORT_PARAMS", None)
        ks_mod.KILOSORT_PARAMS = {}
        ks_mod.RECOMPUTE_SORTING = True
        yield
        if self._old_docker is not None:
            ks_mod.USE_DOCKER = self._old_docker
        if self._old_recompute is not None:
            ks_mod.RECOMPUTE_SORTING = self._old_recompute
        if self._old_params is not None:
            ks_mod.KILOSORT_PARAMS = self._old_params

    @pytest.fixture()
    def ks4_backend(self):
        """Create a Kilosort4Backend with a default config."""
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        return Kilosort4Backend(SortingPipelineConfig())

    def test_get_docker_image_failure_returned_as_object(self, tmp_path, ks4_backend):
        """
        When get_docker_image raises RuntimeError, it is caught and returned.

        Tests:
            (Test Case 1) USE_DOCKER=True with no GPU → RuntimeError returned,
                not raised.

        Notes:
            - The KS4 sort() method wraps run_sorter in try/except Exception
              and returns the exception. A failure in get_docker_image (called
              before run_sorter) is caught by the same handler.
        """
        self._ks_mod.USE_DOCKER = True
        output_folder = tmp_path / "ks4_fail"
        output_folder.mkdir()

        recording = _make_mock_recording()

        with patch(
            "spikelab.spike_sorting.ks4_runner.get_docker_image",
            side_effect=RuntimeError("Could not detect CUDA"),
        ):
            result = ks4_backend.sort(recording, "fake.raw", "fake.dat", output_folder)

        assert isinstance(result, RuntimeError)
        assert "CUDA" in str(result)

    def test_run_sorter_failure_returned_as_object(self, tmp_path, ks4_backend):
        """
        When run_sorter raises an exception, it is returned as an object.

        Tests:
            (Test Case 1) run_sorter raises ValueError → returned, not raised.
        """
        self._ks_mod.USE_DOCKER = False
        output_folder = tmp_path / "ks4_sorter_fail"
        output_folder.mkdir()

        recording = _make_mock_recording()

        with patch(
            "spikeinterface.sorters.run_sorter",
            side_effect=ValueError("Sorting failed"),
        ):
            result = ks4_backend.sort(recording, "fake.raw", "fake.dat", output_folder)

        assert isinstance(result, ValueError)
        assert "Sorting failed" in str(result)


# ---------------------------------------------------------------------------
# Spike-sorting figure tests
# ---------------------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

from spikelab.spike_sorting.figures import (
    plot_curation_bar,
    plot_std_scatter,
    plot_templates,
)


class TestSpikeSortingFigures:
    """Tests for plot_curation_bar, plot_std_scatter, and plot_templates."""

    @pytest.fixture(autouse=True)
    def close_figs(self):
        """Close all matplotlib figures after each test."""
        yield
        plt.close("all")

    # -- plot_curation_bar ---------------------------------------------------

    def test_curation_bar_returns_figure(self):
        """Happy path: valid data returns a matplotlib Figure."""
        fig = plot_curation_bar(
            rec_names=["rec1", "rec2", "rec3"],
            n_total=[100, 80, 120],
            n_selected=[60, 50, 90],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_curation_bar_on_existing_axes(self):
        """Passing an existing Axes draws onto it and returns its Figure."""
        fig_ext, ax_ext = plt.subplots()
        fig = plot_curation_bar(
            rec_names=["a", "b"],
            n_total=[10, 20],
            n_selected=[5, 15],
            ax=ax_ext,
        )
        assert fig is fig_ext

    def test_curation_bar_custom_labels(self):
        """Custom axis labels and legend labels are applied."""
        fig = plot_curation_bar(
            rec_names=["r1"],
            n_total=[50],
            n_selected=[30],
            total_label="All",
            selected_label="Good",
            x_label="Rec",
            y_label="Count",
            label_rotation=45,
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Rec"
        assert ax.get_ylabel() == "Count"

    def test_curation_bar_single_recording(self):
        """Works with a single recording."""
        fig = plot_curation_bar(
            rec_names=["only"],
            n_total=[10],
            n_selected=[5],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # -- plot_std_scatter ----------------------------------------------------

    def test_std_scatter_returns_figure(self):
        """Happy path: valid nested dicts produce a Figure."""
        n_spikes = {"rec1": {"u1": 500, "u2": 300}}
        std_norms = {"rec1": {"u1": 0.3, "u2": 0.5}}
        fig = plot_std_scatter(n_spikes, std_norms)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_std_scatter_on_existing_axes(self):
        """Drawing onto a pre-existing Axes returns its Figure."""
        fig_ext, ax_ext = plt.subplots()
        n_spikes = {"rec1": {"u1": 100}}
        std_norms = {"rec1": {"u1": 0.2}}
        fig = plot_std_scatter(n_spikes, std_norms, ax=ax_ext)
        assert fig is fig_ext

    def test_std_scatter_with_thresholds(self):
        """Threshold lines do not cause errors."""
        n_spikes = {"rec1": {"u1": 500, "u2": 200}}
        std_norms = {"rec1": {"u1": 0.3, "u2": 0.6}}
        fig = plot_std_scatter(
            n_spikes,
            std_norms,
            spikes_thresh=250,
            std_thresh=0.5,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_std_scatter_multiple_recordings(self):
        """Multiple recordings get different colours and a legend."""
        n_spikes = {
            "rec1": {"u1": 100},
            "rec2": {"u2": 200},
        }
        std_norms = {
            "rec1": {"u1": 0.4},
            "rec2": {"u2": 0.3},
        }
        fig = plot_std_scatter(n_spikes, std_norms)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None

    def test_std_scatter_empty_data(self):
        """Empty dicts produce a figure without errors."""
        fig = plot_std_scatter({}, {})
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_std_scatter_custom_colors(self):
        """Custom colour list is accepted."""
        n_spikes = {"rec1": {"u1": 100}}
        std_norms = {"rec1": {"u1": 0.2}}
        fig = plot_std_scatter(n_spikes, std_norms, colors=["#00FF00"])
        assert isinstance(fig, matplotlib.figure.Figure)

    # -- plot_templates ------------------------------------------------------

    def _make_template_data(self, n_units=5, n_samples=60, fs_Hz=20000.0):
        """Create minimal template data for testing."""
        rng = np.random.default_rng(42)
        templates = [rng.standard_normal(n_samples) for _ in range(n_units)]
        peak_indices = [n_samples // 2] * n_units
        is_curated = [True, False, True, True, False][:n_units]
        has_pos_peak = [False, False, True, False, True][:n_units]
        return templates, peak_indices, fs_Hz, is_curated, has_pos_peak

    def test_templates_returns_figure(self):
        """Happy path: valid template data returns a Figure."""
        templates, peaks, fs, curated, pos_peak = self._make_template_data()
        fig = plot_templates(templates, peaks, fs, curated, pos_peak)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_templates_on_existing_axes(self):
        """Drawing onto a pre-existing Axes returns its Figure."""
        fig_ext, ax_ext = plt.subplots()
        templates, peaks, fs, curated, pos_peak = self._make_template_data()
        fig = plot_templates(templates, peaks, fs, curated, pos_peak, ax=ax_ext)
        assert fig is fig_ext

    def test_templates_sort_by_amplitude(self):
        """sort_by_amplitude=True does not error."""
        templates, peaks, fs, curated, pos_peak = self._make_template_data()
        fig = plot_templates(
            templates,
            peaks,
            fs,
            curated,
            pos_peak,
            sort_by_amplitude=True,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_templates_all_negative_peak(self):
        """All units with negative peaks produce a single-polarity layout."""
        templates, peaks, fs, curated, _ = self._make_template_data()
        has_pos_peak = [False] * len(templates)
        fig = plot_templates(templates, peaks, fs, curated, has_pos_peak)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_templates_all_positive_peak(self):
        """All units with positive peaks produce a single-polarity layout."""
        templates, peaks, fs, curated, _ = self._make_template_data()
        has_pos_peak = [True] * len(templates)
        fig = plot_templates(templates, peaks, fs, curated, has_pos_peak)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_templates_no_reference_lines(self):
        """Setting line_ms_before/after to None skips reference lines."""
        templates, peaks, fs, curated, pos_peak = self._make_template_data()
        fig = plot_templates(
            templates,
            peaks,
            fs,
            curated,
            pos_peak,
            line_ms_before=None,
            line_ms_after=None,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_templates_single_unit(self):
        """Works with a single unit."""
        templates, peaks, fs, curated, pos_peak = self._make_template_data(n_units=1)
        fig = plot_templates(templates, peaks, fs, curated, pos_peak)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_templates_empty_units(self):
        """Empty template list produces a figure without errors."""
        fig = plot_templates([], [], 20000.0, [], [])
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# center_spike_times tests
# ---------------------------------------------------------------------------


def _make_fake_recording(traces: np.ndarray, n_samples: int | None = None):
    """Build a minimal mock recording for center_spike_times.

    Parameters
    ----------
    traces : np.ndarray
        2-D array (n_samples, n_channels) returned by ``get_traces``.
    n_samples : int or None
        Total number of samples; defaults to ``traces.shape[0]``.
    """
    if n_samples is None:
        n_samples = traces.shape[0]
    rec = MagicMock()
    rec.get_num_samples.return_value = n_samples

    def _get_traces(start_frame, end_frame, segment_index=0):
        return traces[start_frame:end_frame]

    rec.get_traces.side_effect = _get_traces
    return rec


class TestCenterSpikeTimes:
    """Tests for center_spike_times in waveform_utils.

    Uses a mock recording that returns a controlled voltage trace so the
    expected peak positions are deterministic.
    """

    def _import_fn(self):
        from spikelab.spike_sorting.waveform_utils import center_spike_times

        return center_spike_times

    # -- Happy path -----------------------------------------------------------

    def test_basic_negative_peak_centering(self):
        """Spike is shifted to the negative peak within the search window."""
        center_spike_times = self._import_fn()

        # 100-sample trace on 1 channel, baseline 0
        traces = np.zeros((100, 1), dtype=np.float32)
        # True negative peak at sample 52 (sorter reports 50)
        traces[52, 0] = -10.0

        rec = _make_fake_recording(traces)
        spike_times_by_unit = {0: np.array([50])}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert 0 in result
        assert len(result[0]) == 1
        # The spike should move toward sample 52
        assert result[0][0] != 50

    def test_basic_positive_peak_centering(self):
        """Spike is shifted to the positive peak for a positive-peak unit."""
        center_spike_times = self._import_fn()

        traces = np.zeros((100, 1), dtype=np.float32)
        # True positive peak at sample 48 (sorter reports 50)
        traces[48, 0] = 15.0

        rec = _make_fake_recording(traces)
        spike_times_by_unit = {0: np.array([50])}
        chans_max = {0: 0}
        use_pos_peak = {0: True}
        half_window_sizes = {0: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert 0 in result
        assert len(result[0]) == 1
        assert result[0][0] != 50

    def test_multiple_units(self):
        """Each unit is centered independently."""
        center_spike_times = self._import_fn()

        traces = np.zeros((200, 2), dtype=np.float32)
        # Unit 0 (neg peak) — true peak at 52 on channel 0
        traces[52, 0] = -20.0
        # Unit 1 (pos peak) — true peak at 103 on channel 1
        traces[103, 1] = 30.0

        rec = _make_fake_recording(traces)
        spike_times_by_unit = {0: np.array([50]), 1: np.array([100])}
        chans_max = {0: 0, 1: 1}
        use_pos_peak = {0: False, 1: True}
        half_window_sizes = {0: 5, 1: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert set(result.keys()) == {0, 1}
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    def test_multiple_spikes_per_unit(self):
        """Multiple spikes within one unit are each corrected independently."""
        center_spike_times = self._import_fn()

        traces = np.zeros((200, 1), dtype=np.float32)
        traces[52, 0] = -10.0  # peak near spike at 50
        traces[101, 0] = -15.0  # peak near spike at 100

        rec = _make_fake_recording(traces)
        spike_times_by_unit = {0: np.array([50, 100])}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert len(result[0]) == 2

    # -- Edge cases -----------------------------------------------------------

    def test_empty_spike_array(self):
        """Unit with zero spikes returns an empty array."""
        center_spike_times = self._import_fn()

        traces = np.zeros((100, 1), dtype=np.float32)
        rec = _make_fake_recording(traces)

        spike_times_by_unit = {0: np.array([], dtype=np.int64)}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert 0 in result
        assert len(result[0]) == 0

    def test_single_spike(self):
        """Single spike is handled without error."""
        center_spike_times = self._import_fn()

        traces = np.zeros((100, 1), dtype=np.float32)
        traces[50, 0] = -5.0

        rec = _make_fake_recording(traces)
        spike_times_by_unit = {0: np.array([50])}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 3}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert len(result[0]) == 1

    def test_no_shift_needed(self):
        """Spike already at peak position stays unchanged."""
        center_spike_times = self._import_fn()

        traces = np.zeros((100, 1), dtype=np.float32)
        # Peak exactly at the reported spike time
        traces[50, 0] = -10.0

        rec = _make_fake_recording(traces)
        spike_times_by_unit = {0: np.array([50])}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        # When the peak is at the center of the window, no correction needed
        assert result[0][0] == 50

    def test_spike_near_recording_start(self):
        """Spike near sample 0 is handled without out-of-bounds errors."""
        center_spike_times = self._import_fn()

        traces = np.zeros((100, 1), dtype=np.float32)
        traces[2, 0] = -10.0

        rec = _make_fake_recording(traces)
        spike_times_by_unit = {0: np.array([2])}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert 0 in result
        assert len(result[0]) == 1

    def test_spike_near_recording_end(self):
        """Spike near the last sample is handled without out-of-bounds errors."""
        center_spike_times = self._import_fn()

        n_total = 100
        traces = np.zeros((n_total, 1), dtype=np.float32)
        traces[97, 0] = -10.0

        rec = _make_fake_recording(traces, n_samples=n_total)
        spike_times_by_unit = {0: np.array([97])}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert 0 in result
        assert len(result[0]) == 1

    def test_preserves_dict_keys(self):
        """Output dict has the same keys as input, including non-contiguous IDs."""
        center_spike_times = self._import_fn()

        traces = np.zeros((200, 1), dtype=np.float32)
        rec = _make_fake_recording(traces)

        spike_times_by_unit = {
            3: np.array([50]),
            7: np.array([100]),
            12: np.array([], dtype=np.int64),
        }
        chans_max = {3: 0, 7: 0, 12: 0}
        use_pos_peak = {3: False, 7: False, 12: False}
        half_window_sizes = {3: 5, 7: 5, 12: 5}

        result = center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        assert set(result.keys()) == {3, 7, 12}
        assert len(result[12]) == 0

    def test_segment_index_passed_through(self):
        """The segment_index argument is forwarded to the recording."""
        center_spike_times = self._import_fn()

        traces = np.zeros((100, 1), dtype=np.float32)
        rec = _make_fake_recording(traces)

        spike_times_by_unit = {0: np.array([50])}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 3}

        center_spike_times(
            rec,
            spike_times_by_unit,
            chans_max,
            use_pos_peak,
            half_window_sizes,
            segment_index=2,
        )

        rec.get_num_samples.assert_called_with(segment_index=2)
        # get_traces should also have received segment_index=2
        call_kwargs = rec.get_traces.call_args[1]
        assert call_kwargs["segment_index"] == 2

    def test_original_spike_times_not_mutated(self):
        """The input arrays are not modified in place."""
        center_spike_times = self._import_fn()

        traces = np.zeros((100, 1), dtype=np.float32)
        traces[52, 0] = -10.0

        rec = _make_fake_recording(traces)
        original = np.array([50])
        spike_times_by_unit = {0: original.copy()}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 5}

        center_spike_times(
            rec, spike_times_by_unit, chans_max, use_pos_peak, half_window_sizes
        )

        np.testing.assert_array_equal(spike_times_by_unit[0], original)

    def test_half_window_zero(self):
        """half_window_sizes[unit_id] = 0 produces a zero-width window;
        the spike time should remain unchanged."""
        from spikelab.spike_sorting.waveform_utils import center_spike_times

        recording = _make_mock_recording(num_samples=1000, num_channels=2)
        # Add segment_index support
        recording.get_num_samples = lambda segment_index=0: 1000
        recording.get_traces = (
            lambda start_frame=0, end_frame=None, segment_index=0, **kw: np.random.default_rng(
                0
            )
            .standard_normal((end_frame - start_frame, 2))
            .astype(np.float32)
        )

        spike_times = {0: np.array([100, 200, 300])}
        chans_max = {0: 0}
        use_pos_peak = {0: False}
        half_window_sizes = {0: 0}

        result = center_spike_times(
            recording,
            spike_times,
            chans_max,
            use_pos_peak,
            half_window_sizes,
        )
        # With hw=0, the window has 1 sample, so the offset is always 0
        assert 0 in result
        assert len(result[0]) == 3


# ===========================================================================
# RT-Sort backend integration
# ===========================================================================
#
# These tests cover the SpikeLab RT-Sort wrapper code (config, registry,
# backend dependency check + globals sync, runner caching helpers, lazy
# subpackage public API).  They do not exercise the vendored RT-Sort
# algorithm itself (detect_sequences / RTSort.sort_offline / the DL
# detection model), which require torch + a GPU + a real recording and
# are validated separately on a GPU machine via the HANDOVER guide.


class TestRTSortConfig:
    """
    Tests for the RTSortConfig sub-config and the RT_SORT_* presets.
    """

    def test_default_construction(self):
        """
        RTSortConfig has the documented default values.

        Tests:
            (Test Case 1) probe defaults to "mea".
            (Test Case 2) device defaults to "cuda".
            (Test Case 3) save_rt_sort_pickle defaults to True.
            (Test Case 4) verbose defaults to True.
            (Test Case 5) model_path / num_processes / recording_window_ms / params default to None.
            (Test Case 6) delete_inter defaults to False.
        """
        from spikelab.spike_sorting.config import RTSortConfig

        cfg = RTSortConfig()
        assert cfg.probe == "mea"
        assert cfg.device == "cuda"
        assert cfg.save_rt_sort_pickle is True
        assert cfg.verbose is True
        assert cfg.model_path is None
        assert cfg.num_processes is None
        assert cfg.recording_window_ms is None
        assert cfg.params is None
        assert cfg.delete_inter is False

    def test_pipeline_config_includes_rt_sort_field(self):
        """
        SortingPipelineConfig has an rt_sort sub-config field.

        Tests:
            (Test Case 1) Default pipeline config has an RTSortConfig instance attached.
        """
        from spikelab.spike_sorting.config import RTSortConfig, SortingPipelineConfig

        cfg = SortingPipelineConfig()
        assert isinstance(cfg.rt_sort, RTSortConfig)

    def test_from_kwargs_maps_rt_sort_flat_keys(self):
        """
        from_kwargs maps the rt_sort_* prefixed flat keys to the RTSortConfig sub-config.

        Tests:
            (Test Case 1) rt_sort_probe maps to rt_sort.probe.
            (Test Case 2) rt_sort_device maps to rt_sort.device.
            (Test Case 3) rt_sort_num_processes maps to rt_sort.num_processes.
            (Test Case 4) rt_sort_recording_window_ms maps to rt_sort.recording_window_ms.
            (Test Case 5) rt_sort_save_pickle maps to rt_sort.save_rt_sort_pickle.
            (Test Case 6) rt_sort_delete_inter maps to rt_sort.delete_inter.
            (Test Case 7) rt_sort_verbose maps to rt_sort.verbose.
            (Test Case 8) rt_sort_params maps to rt_sort.params.
            (Test Case 9) rt_sort_model_path maps to rt_sort.model_path.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig.from_kwargs(
            rt_sort_probe="neuropixels",
            rt_sort_device="cpu",
            rt_sort_num_processes=4,
            rt_sort_recording_window_ms=(0, 60_000),
            rt_sort_save_pickle=False,
            rt_sort_delete_inter=True,
            rt_sort_verbose=False,
            rt_sort_params={"stringent_thresh": 0.2},
            rt_sort_model_path="/tmp/model",
        )
        assert cfg.rt_sort.probe == "neuropixels"
        assert cfg.rt_sort.device == "cpu"
        assert cfg.rt_sort.num_processes == 4
        assert cfg.rt_sort.recording_window_ms == (0, 60_000)
        assert cfg.rt_sort.save_rt_sort_pickle is False
        assert cfg.rt_sort.delete_inter is True
        assert cfg.rt_sort.verbose is False
        assert cfg.rt_sort.params == {"stringent_thresh": 0.2}
        assert cfg.rt_sort.model_path == "/tmp/model"

    def test_override_changes_rt_sort_fields(self):
        """
        override accepts rt_sort_* flat keys and returns a new config.

        Tests:
            (Test Case 1) Override changes the specified RT-Sort field.
            (Test Case 2) Original config is not mutated.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        original = SortingPipelineConfig()
        modified = original.override(rt_sort_probe="neuropixels", rt_sort_device="cpu")
        assert modified.rt_sort.probe == "neuropixels"
        assert modified.rt_sort.device == "cpu"
        assert original.rt_sort.probe == "mea"
        assert original.rt_sort.device == "cuda"

    def test_rt_sort_mea_preset(self):
        """
        RT_SORT_MEA preset selects the rt_sort backend with the MEA probe.

        Tests:
            (Test Case 1) sorter_name is "rt_sort".
            (Test Case 2) probe is "mea".
            (Test Case 3) params dict is None (no Neuropixels overrides).
        """
        from spikelab.spike_sorting.config import RT_SORT_MEA

        assert RT_SORT_MEA.sorter.sorter_name == "rt_sort"
        assert RT_SORT_MEA.rt_sort.probe == "mea"
        assert RT_SORT_MEA.rt_sort.params is None

    def test_rt_sort_neuropixels_preset(self):
        """
        RT_SORT_NEUROPIXELS preset hard-codes the paper-tuned Neuropixels parameters.

        Tests:
            (Test Case 1) sorter_name is "rt_sort".
            (Test Case 2) probe is "neuropixels".
            (Test Case 3) params dict carries the paper-tuned threshold values.
        """
        from spikelab.spike_sorting.config import RT_SORT_NEUROPIXELS

        assert RT_SORT_NEUROPIXELS.sorter.sorter_name == "rt_sort"
        assert RT_SORT_NEUROPIXELS.rt_sort.probe == "neuropixels"
        params = RT_SORT_NEUROPIXELS.rt_sort.params
        assert params is not None
        assert params["stringent_thresh"] == 0.175
        assert params["loose_thresh"] == 0.075
        assert params["inference_scaling_numerator"] == 15.4
        assert params["max_latency_diff_spikes"] == 2.5
        assert params["max_amp_median_diff_spikes"] == 0.45


class TestRTSortBackendRegistry:
    """
    Tests that the RTSortBackend is registered in the backend registry.
    """

    def test_rt_sort_in_list_sorters(self):
        """
        list_sorters includes "rt_sort" alongside the Kilosort backends.

        Tests:
            (Test Case 1) "rt_sort" is in the registered sorter list.
        """
        from spikelab.spike_sorting.backends import list_sorters

        assert "rt_sort" in list_sorters()

    def test_get_backend_class_returns_rtsort_backend(self):
        """
        get_backend_class("rt_sort") returns the RTSortBackend class.

        Tests:
            (Test Case 1) Returned class name is RTSortBackend.
            (Test Case 2) Returned class is a SorterBackend subclass.
        """
        from spikelab.spike_sorting.backends import get_backend_class
        from spikelab.spike_sorting.backends.base import SorterBackend

        cls = get_backend_class("rt_sort")
        assert cls.__name__ == "RTSortBackend"
        assert issubclass(cls, SorterBackend)


@skip_no_spikeinterface
class TestRTSortBackendDependencyCheck:
    """
    Tests for RTSortBackend._check_dependencies() — the upfront missing-dep
    error raised at backend construction time.
    """

    def test_missing_torch_raises_import_error(self, monkeypatch):
        """
        _check_dependencies raises ImportError when torch is unavailable.

        Tests:
            (Test Case 1) Import error message names torch.
            (Test Case 2) Import error message points to pytorch.org.
        """
        from spikelab.spike_sorting.backends.rt_sort import RTSortBackend
        from spikelab.spike_sorting.config import RT_SORT_MEA

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        with pytest.raises(ImportError) as exc_info:
            RTSortBackend(RT_SORT_MEA)
        assert "torch" in str(exc_info.value)
        assert "pytorch.org" in str(exc_info.value)

    def test_multiple_missing_packages_listed(self, monkeypatch):
        """
        _check_dependencies lists every missing package, not just the first one.

        Tests:
            (Test Case 1) Both torch and diptest appear in the error message
                          when both are unavailable.
        """
        from spikelab.spike_sorting.backends.rt_sort import RTSortBackend
        from spikelab.spike_sorting.config import RT_SORT_MEA

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name in ("torch", "diptest"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        with pytest.raises(ImportError) as exc_info:
            RTSortBackend(RT_SORT_MEA)
        msg = str(exc_info.value)
        assert "torch" in msg
        assert "diptest" in msg


@skip_no_spikeinterface
class TestRTSortBackendSyncGlobals:
    """
    Tests for RTSortBackend._sync_globals() — the config → _globals bridge.
    """

    @pytest.fixture
    def make_backend(self, monkeypatch):
        """Skip the dep check so we can construct the backend without torch."""
        from spikelab.spike_sorting.backends.rt_sort import RTSortBackend

        monkeypatch.setattr(RTSortBackend, "_check_dependencies", lambda self: None)
        return RTSortBackend

    def test_sync_writes_rt_sort_globals(self, make_backend):
        """
        _sync_globals mirrors the RT-Sort sub-config fields to module globals.

        Tests:
            (Test Case 1) RT_SORT_DEVICE is taken from config.rt_sort.device.
            (Test Case 2) RT_SORT_NUM_PROCESSES mirrors config.rt_sort.num_processes.
            (Test Case 3) RT_SORT_RECORDING_WINDOW_MS mirrors the window.
            (Test Case 4) RT_SORT_SAVE_PICKLE mirrors save_rt_sort_pickle.
            (Test Case 5) RT_SORT_DELETE_INTER mirrors delete_inter.
            (Test Case 6) RT_SORT_VERBOSE mirrors verbose.
            (Test Case 7) RT_SORT_MODEL_PATH mirrors model_path.
        """
        from spikelab.spike_sorting import _globals
        from spikelab.spike_sorting.config import (
            RTSortConfig,
            SortingPipelineConfig,
            SorterConfig,
        )

        cfg = SortingPipelineConfig(
            sorter=SorterConfig(sorter_name="rt_sort"),
            rt_sort=RTSortConfig(
                probe="mea",
                device="cpu",
                num_processes=2,
                recording_window_ms=(0, 30_000),
                save_rt_sort_pickle=False,
                delete_inter=True,
                verbose=False,
                model_path="/tmp/m",
            ),
        )

        make_backend(cfg)

        assert _globals.RT_SORT_DEVICE == "cpu"
        assert _globals.RT_SORT_NUM_PROCESSES == 2
        assert _globals.RT_SORT_RECORDING_WINDOW_MS == (0, 30_000)
        assert _globals.RT_SORT_SAVE_PICKLE is False
        assert _globals.RT_SORT_DELETE_INTER is True
        assert _globals.RT_SORT_VERBOSE is False
        assert _globals.RT_SORT_MODEL_PATH == "/tmp/m"

    def test_sync_merges_probe_into_params(self, make_backend):
        """
        _sync_globals merges the probe into RT_SORT_PARAMS so the runner can
        read both from a single dict.

        Tests:
            (Test Case 1) RT_SORT_PARAMS["probe"] reflects the configured probe.
            (Test Case 2) Override params from RTSortConfig.params are merged in.
        """
        from spikelab.spike_sorting import _globals
        from spikelab.spike_sorting.config import (
            RTSortConfig,
            SortingPipelineConfig,
            SorterConfig,
        )

        cfg = SortingPipelineConfig(
            sorter=SorterConfig(sorter_name="rt_sort"),
            rt_sort=RTSortConfig(
                probe="neuropixels",
                params={"stringent_thresh": 0.175, "loose_thresh": 0.075},
            ),
        )

        make_backend(cfg)

        assert _globals.RT_SORT_PARAMS["probe"] == "neuropixels"
        assert _globals.RT_SORT_PARAMS["stringent_thresh"] == 0.175
        assert _globals.RT_SORT_PARAMS["loose_thresh"] == 0.075

    def test_sync_writes_recording_globals_including_time_slicing(self, make_backend):
        """
        _sync_globals also writes the standard recording globals.

        Tests:
            (Test Case 1) FREQ_MIN / FREQ_MAX mirror RecordingConfig.
            (Test Case 2) REC_CHUNKS_S, START_TIME_S, END_TIME_S are mirrored.
        """
        from spikelab.spike_sorting import _globals
        from spikelab.spike_sorting.config import (
            RecordingConfig,
            RTSortConfig,
            SortingPipelineConfig,
            SorterConfig,
        )

        cfg = SortingPipelineConfig(
            recording=RecordingConfig(
                freq_min=250,
                freq_max=5000,
                rec_chunks_s=[(0.0, 60.0), (120.0, 180.0)],
                start_time_s=0.0,
                end_time_s=200.0,
            ),
            sorter=SorterConfig(sorter_name="rt_sort"),
            rt_sort=RTSortConfig(),
        )

        make_backend(cfg)

        assert _globals.FREQ_MIN == 250
        assert _globals.FREQ_MAX == 5000
        assert _globals.REC_CHUNKS_S == [(0.0, 60.0), (120.0, 180.0)]
        assert _globals.START_TIME_S == 0.0
        assert _globals.END_TIME_S == 200.0


@skip_no_spikeinterface
class TestNumpySortingToKsExtractor:
    """
    Tests for backends.rt_sort._numpy_sorting_to_ks_extractor — the adapter
    that writes Kilosort-format files from a NumpySorting so the shared
    waveform extractor can consume RT-Sort output.
    """

    @pytest.fixture
    def fake_recording(self):
        """Minimal SpikeInterface-like recording with the methods used by the adapter."""
        rec = MagicMock()
        rec.get_sampling_frequency.return_value = 30_000.0
        rec.get_num_channels.return_value = 8
        return rec

    def _make_numpy_sorting(self, spikes_by_unit, fs=30_000.0):
        from spikeinterface.extractors import NumpySorting

        return NumpySorting.from_unit_dict([spikes_by_unit], sampling_frequency=fs)

    def test_writes_kilosort_format_files(self, fake_recording, tmp_path):
        """
        Adapter writes the expected Kilosort-format files into the output folder.

        Tests:
            (Test Case 1) spike_times.npy is created.
            (Test Case 2) spike_clusters.npy is created.
            (Test Case 3) templates.npy is created.
            (Test Case 4) channel_map.npy is created.
            (Test Case 5) params.py is created.
        """
        from spikelab.spike_sorting.backends.rt_sort import (
            _numpy_sorting_to_ks_extractor,
        )

        sorting = self._make_numpy_sorting(
            {0: np.array([10, 20, 30]), 1: np.array([15, 25])}
        )
        out = tmp_path / "ks_out"

        _numpy_sorting_to_ks_extractor(sorting, fake_recording, out)

        assert (out / "spike_times.npy").exists()
        assert (out / "spike_clusters.npy").exists()
        assert (out / "templates.npy").exists()
        assert (out / "channel_map.npy").exists()
        assert (out / "params.py").exists()

    def test_spike_times_are_globally_sorted(self, fake_recording, tmp_path):
        """
        spike_times.npy is sorted in ascending order across all units.

        Tests:
            (Test Case 1) Output spike times are monotonic non-decreasing.
            (Test Case 2) Output cluster IDs follow the time ordering.
        """
        from spikelab.spike_sorting.backends.rt_sort import (
            _numpy_sorting_to_ks_extractor,
        )

        sorting = self._make_numpy_sorting(
            {0: np.array([100, 300]), 1: np.array([50, 200, 400])}
        )
        out = tmp_path / "ks_out"

        _numpy_sorting_to_ks_extractor(sorting, fake_recording, out)

        times = np.load(out / "spike_times.npy")
        clusters = np.load(out / "spike_clusters.npy")
        assert np.all(np.diff(times) >= 0)
        # Check the cluster ordering follows the time sort
        expected_times = np.array([50, 100, 200, 300, 400])
        expected_clusters = np.array([1, 0, 1, 0, 1])
        np.testing.assert_array_equal(times, expected_times)
        np.testing.assert_array_equal(clusters, expected_clusters)

    def test_root_elecs_set_template_peak_channel(self, fake_recording, tmp_path):
        """
        When root_elecs is provided, the synthesized templates have their peak
        on the corresponding channel for each unit.

        Tests:
            (Test Case 1) Unit 0 with root_elec=3 has its peak on channel 3.
            (Test Case 2) Unit 1 with root_elec=5 has its peak on channel 5.
        """
        from spikelab.spike_sorting.backends.rt_sort import (
            _numpy_sorting_to_ks_extractor,
        )

        sorting = self._make_numpy_sorting(
            {0: np.array([10, 20]), 1: np.array([15, 25])}
        )
        out = tmp_path / "ks_out"

        _numpy_sorting_to_ks_extractor(sorting, fake_recording, out, root_elecs=[3, 5])

        templates = np.load(out / "templates.npy")
        # templates shape: (n_units, n_samples, n_channels)
        assert templates.shape[0] == 2
        assert templates.shape[2] == 8
        # Find the channel where each unit has its largest absolute value
        unit_0_peak_channel = int(np.argmax(np.max(np.abs(templates[0]), axis=0)))
        unit_1_peak_channel = int(np.argmax(np.max(np.abs(templates[1]), axis=0)))
        assert unit_0_peak_channel == 3
        assert unit_1_peak_channel == 5

    def test_empty_sorting(self, fake_recording, tmp_path):
        """
        Adapter handles a sorting with zero units without raising.

        Tests:
            (Test Case 1) Empty input still produces all expected files.
            (Test Case 2) spike_times.npy is empty.
        """
        from spikelab.spike_sorting.backends.rt_sort import (
            _numpy_sorting_to_ks_extractor,
        )

        sorting = self._make_numpy_sorting({})
        out = tmp_path / "ks_out"

        _numpy_sorting_to_ks_extractor(sorting, fake_recording, out)

        assert (out / "spike_times.npy").exists()
        times = np.load(out / "spike_times.npy")
        assert len(times) == 0


@skip_no_spikeinterface
class TestRTSortRunnerHelpers:
    """
    Tests for the helper functions in rt_sort_runner — model loading,
    cache round-trip, and the load_rt_sort error paths.
    """

    @skip_no_torch
    def test_load_detection_model_unknown_probe_raises(self):
        """
        _load_detection_model rejects an unknown probe name.

        Tests:
            (Test Case 1) Passing an unknown probe raises ValueError naming the probe.

        Notes:
            - The runner imports ModelSpikeSorter from rt_sort.model, which
              requires torch at module-import time, so this test is skipped
              in environments without torch.
        """
        from spikelab.spike_sorting.rt_sort_runner import _load_detection_model

        with pytest.raises(ValueError, match="Unknown probe"):
            _load_detection_model(model_path=None, probe="bogus_probe")

    def test_save_and_load_sorting_cache_round_trip(self, tmp_path):
        """
        _save_sorting_cache + _load_cached_sorting round-trip a NumpySorting
        without losing per-unit spike trains or sampling frequency.

        Tests:
            (Test Case 1) Sampling frequency is preserved.
            (Test Case 2) Per-unit spike trains are preserved exactly.
            (Test Case 3) Unit IDs are preserved.
        """
        from spikeinterface.extractors import NumpySorting
        from spikelab.spike_sorting.rt_sort_runner import (
            _save_sorting_cache,
            _load_cached_sorting,
        )

        unit_ids = [0, 1, 7]
        spikes = {
            0: np.array([10, 20, 30, 100], dtype=np.int64),
            1: np.array([5, 15], dtype=np.int64),
            7: np.array([50], dtype=np.int64),
        }
        sorting = NumpySorting.from_unit_dict([spikes], sampling_frequency=30_000.0)

        cache_path = tmp_path / "sorting.npz"
        _save_sorting_cache(sorting, cache_path)
        assert cache_path.exists()

        loaded = _load_cached_sorting(cache_path, recording=None)
        assert loaded.get_sampling_frequency() == 30_000.0
        loaded_ids = sorted(loaded.get_unit_ids())
        assert loaded_ids == sorted(unit_ids)
        for uid in unit_ids:
            np.testing.assert_array_equal(loaded.get_unit_spike_train(uid), spikes[uid])

    @skip_no_torch
    def test_detection_window_s_narrows_only_detect_sequences(
        self, monkeypatch, tmp_path
    ):
        """``rt_sort.detection_window_s`` narrows the detect_sequences window
        but leaves sort_offline running across the full recording window.

        Captures the ``recording_window_ms`` argument passed to each of the
        two RT-Sort entry points, then asserts they differ as expected.
        """
        from spikelab.spike_sorting import _globals
        from spikelab.spike_sorting import rt_sort_runner as runner

        # --- Stub heavy dependencies -----------------------------------
        captured = {"detect": None, "sort_offline": None}

        class FakeRTSort:
            _seq_root_elecs = []

            def sort_offline(self, recording, inter_path, **kw):
                captured["sort_offline"] = kw.get("recording_window_ms")
                return object()  # opaque — _save_sorting_cache is stubbed

        def fake_detect_sequences(recording, inter_path, detection_model, **kw):
            captured["detect"] = kw.get("recording_window_ms")
            return FakeRTSort()

        # Replace the module-level imports the runner pulls in lazily.
        monkeypatch.setattr(
            "spikelab.spike_sorting.rt_sort_runner._load_detection_model",
            lambda *a, **k: object(),
        )
        # detect_sequences is imported via `from .rt_sort import detect_sequences`
        # inside spike_sort, so patch the symbol on the source module.
        import spikelab.spike_sorting.rt_sort as rt_sort_pkg

        monkeypatch.setattr(
            rt_sort_pkg, "detect_sequences", fake_detect_sequences, raising=False
        )
        # And the conditional model-cache reuse path needs to skip
        monkeypatch.setattr(_globals, "RECOMPUTE_SORTING", True)
        monkeypatch.setattr(_globals, "RT_SORT_RECORDING_WINDOW_MS", (0.0, 600_000.0))
        monkeypatch.setattr(_globals, "RT_SORT_DETECTION_WINDOW_S", 60.0)
        monkeypatch.setattr(_globals, "RT_SORT_DEVICE", "cpu")
        monkeypatch.setattr(_globals, "RT_SORT_NUM_PROCESSES", 1)
        monkeypatch.setattr(_globals, "RT_SORT_DELETE_INTER", False)
        monkeypatch.setattr(_globals, "RT_SORT_VERBOSE", False)
        monkeypatch.setattr(_globals, "RT_SORT_PARAMS", {"probe": "mea"})
        monkeypatch.setattr(_globals, "RT_SORT_SAVE_PICKLE", False)

        # _save_sorting_cache writes to disk; stub it
        monkeypatch.setattr(
            "spikelab.spike_sorting.rt_sort_runner._save_sorting_cache",
            lambda *a, **k: None,
        )

        runner.spike_sort(
            rec_cache=object(),
            rec_path=tmp_path / "fake.h5",
            recording_dat_path=None,
            output_folder=tmp_path / "out",
        )

        # Detection should be narrowed to (0, 60_000) ms
        assert captured["detect"] == (0.0, 60_000.0), (
            f"Expected detect window narrowed to (0, 60_000) ms, "
            f"got {captured['detect']!r}"
        )
        # sort_offline should still cover the full configured window
        assert captured["sort_offline"] == (0.0, 600_000.0), (
            f"Expected sort_offline to cover full (0, 600_000) ms, "
            f"got {captured['sort_offline']!r}"
        )

    @skip_no_torch
    def test_detection_window_s_unset_uses_full_window_for_both(
        self, monkeypatch, tmp_path
    ):
        """When ``detection_window_s`` is None (default), both phases see the
        same window — preserves legacy behavior."""
        from spikelab.spike_sorting import _globals
        from spikelab.spike_sorting import rt_sort_runner as runner

        captured = {"detect": None, "sort_offline": None}

        class FakeRTSort:
            _seq_root_elecs = []

            def sort_offline(self, recording, inter_path, **kw):
                captured["sort_offline"] = kw.get("recording_window_ms")
                return object()  # opaque — _save_sorting_cache is stubbed

        def fake_detect_sequences(recording, inter_path, detection_model, **kw):
            captured["detect"] = kw.get("recording_window_ms")
            return FakeRTSort()

        monkeypatch.setattr(
            "spikelab.spike_sorting.rt_sort_runner._load_detection_model",
            lambda *a, **k: object(),
        )
        import spikelab.spike_sorting.rt_sort as rt_sort_pkg

        monkeypatch.setattr(
            rt_sort_pkg, "detect_sequences", fake_detect_sequences, raising=False
        )
        monkeypatch.setattr(_globals, "RECOMPUTE_SORTING", True)
        monkeypatch.setattr(_globals, "RT_SORT_RECORDING_WINDOW_MS", (0.0, 120_000.0))
        monkeypatch.setattr(_globals, "RT_SORT_DETECTION_WINDOW_S", None)
        monkeypatch.setattr(_globals, "RT_SORT_DEVICE", "cpu")
        monkeypatch.setattr(_globals, "RT_SORT_NUM_PROCESSES", 1)
        monkeypatch.setattr(_globals, "RT_SORT_DELETE_INTER", False)
        monkeypatch.setattr(_globals, "RT_SORT_VERBOSE", False)
        monkeypatch.setattr(_globals, "RT_SORT_PARAMS", {"probe": "mea"})
        monkeypatch.setattr(_globals, "RT_SORT_SAVE_PICKLE", False)
        monkeypatch.setattr(
            "spikelab.spike_sorting.rt_sort_runner._save_sorting_cache",
            lambda *a, **k: None,
        )

        runner.spike_sort(
            rec_cache=object(),
            rec_path=tmp_path / "fake.h5",
            recording_dat_path=None,
            output_folder=tmp_path / "out",
        )

        assert captured["detect"] == (0.0, 120_000.0)
        assert captured["sort_offline"] == (0.0, 120_000.0)

    @skip_no_torch
    def test_load_rt_sort_missing_pickle_raises(self, tmp_path):
        """
        load_rt_sort raises a clear error when the pickle file does not exist.

        Tests:
            (Test Case 1) Nonexistent path raises FileNotFoundError or OSError.

        Notes:
            - load_rt_sort imports RTSort from rt_sort._algorithm, which
              transitively imports rt_sort.model and therefore requires
              torch at module-import time.
        """
        from spikelab.spike_sorting.rt_sort_runner import load_rt_sort

        bogus = tmp_path / "does_not_exist.pickle"
        with pytest.raises((FileNotFoundError, OSError)):
            load_rt_sort(bogus)


class TestRTSortSubpackagePublicAPI:
    """
    Tests for the rt_sort subpackage __init__.py — bundled model paths,
    lazy attribute resolution, and load_detection_model error handling.
    """

    def test_bundled_mea_model_path_exists(self):
        """
        DEFAULT_MEA_MODEL_PATH points at a folder containing the bundled
        MEA detection model files.

        Tests:
            (Test Case 1) Path exists and is a directory.
            (Test Case 2) init_dict.json exists in the folder.
            (Test Case 3) state_dict.pt exists in the folder.
        """
        from spikelab.spike_sorting.rt_sort import DEFAULT_MEA_MODEL_PATH

        assert DEFAULT_MEA_MODEL_PATH.is_dir()
        assert (DEFAULT_MEA_MODEL_PATH / "init_dict.json").exists()
        assert (DEFAULT_MEA_MODEL_PATH / "state_dict.pt").exists()

    def test_bundled_neuropixels_model_path_exists(self):
        """
        DEFAULT_NEUROPIXELS_MODEL_PATH points at a folder containing the
        bundled Neuropixels detection model files.

        Tests:
            (Test Case 1) Path exists and is a directory.
            (Test Case 2) init_dict.json exists in the folder.
            (Test Case 3) state_dict.pt exists in the folder.
        """
        from spikelab.spike_sorting.rt_sort import DEFAULT_NEUROPIXELS_MODEL_PATH

        assert DEFAULT_NEUROPIXELS_MODEL_PATH.is_dir()
        assert (DEFAULT_NEUROPIXELS_MODEL_PATH / "init_dict.json").exists()
        assert (DEFAULT_NEUROPIXELS_MODEL_PATH / "state_dict.pt").exists()

    def test_unknown_attribute_raises_attribute_error(self):
        """
        rt_sort.__getattr__ raises AttributeError for unknown attributes
        rather than silently returning None.

        Tests:
            (Test Case 1) Bogus attribute name raises AttributeError.
        """
        from spikelab.spike_sorting import rt_sort

        with pytest.raises(AttributeError):
            rt_sort.this_attribute_does_not_exist

    def test_all_advertises_public_symbols(self):
        """
        __all__ on the rt_sort subpackage advertises the documented public API.

        Tests:
            (Test Case 1) detect_sequences, RTSort, load_detection_model, and the
                          DEFAULT_*_MODEL_PATH constants are listed.
        """
        from spikelab.spike_sorting import rt_sort

        for name in (
            "detect_sequences",
            "RTSort",
            "load_detection_model",
            "DEFAULT_MEA_MODEL_PATH",
            "DEFAULT_NEUROPIXELS_MODEL_PATH",
            "NEUROPIXELS_PARAMS",
        ):
            assert name in rt_sort.__all__


class TestRTSortGlobals:
    """
    Tests that the RT_SORT_* globals exist with their documented default types.
    """

    def test_rt_sort_globals_present(self):
        """
        _globals declares the RT_SORT_* names referenced by the runner and backend.

        Tests:
            (Test Case 1) All documented RT_SORT_* attributes exist on the module.
            (Test Case 2) Default booleans are bools and string defaults are strings.
        """
        from spikelab.spike_sorting import _globals

        # Names exist
        for name in (
            "RT_SORT_MODEL_PATH",
            "RT_SORT_DEVICE",
            "RT_SORT_NUM_PROCESSES",
            "RT_SORT_RECORDING_WINDOW_MS",
            "RT_SORT_PARAMS",
            "RT_SORT_SAVE_PICKLE",
            "RT_SORT_DELETE_INTER",
            "RT_SORT_VERBOSE",
        ):
            assert hasattr(_globals, name), f"_globals is missing {name}"

        # Default types match the documented contract
        assert isinstance(_globals.RT_SORT_DEVICE, str)
        assert isinstance(_globals.RT_SORT_SAVE_PICKLE, bool)
        assert isinstance(_globals.RT_SORT_DELETE_INTER, bool)
        assert isinstance(_globals.RT_SORT_VERBOSE, bool)


# ===========================================================================
# Stim sorting — recentering
# ===========================================================================


class TestRecenterStimTimes:
    """
    Tests for recenter_stim_times (stim artifact peak detection).

    Tests:
        (Test Case 1) Basic recentering to a planted artifact peak.
        (Test Case 2) Multiple stim events recentered independently.
        (Test Case 3) Stim near recording start (window clipped).
        (Test Case 4) Stim near recording end (window clipped).
        (Test Case 5) No offset needed — logged time already at peak.
        (Test Case 6) Single channel recording.
        (Test Case 7) Empty stim_times array.
        (Test Case 8) Large max_offset_ms covering full recording.
    """

    @staticmethod
    def _make_traces(
        n_channels, n_samples, fs_Hz, artifact_positions, artifact_amp=100.0
    ):
        """Create zero traces with large spikes at given sample positions."""
        traces = np.random.default_rng(42).normal(0, 0.1, (n_channels, n_samples))
        for pos in artifact_positions:
            traces[:, pos] = artifact_amp
        return traces

    def test_basic_recentering(self):
        """
        A single stim event is recentered to the planted artifact peak.

        Tests:
            (Test Case 1) Corrected time matches the true artifact sample.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times

        fs_Hz = 20000.0
        n_samples = 40000  # 2 seconds
        true_sample = 10000
        traces = self._make_traces(4, n_samples, fs_Hz, [true_sample])

        # Logged time is 5 ms off from true artifact
        logged_ms = true_sample / fs_Hz * 1000.0 + 5.0
        corrected = recenter_stim_times(traces, [logged_ms], fs_Hz, max_offset_ms=50.0)

        expected_ms = true_sample / fs_Hz * 1000.0
        assert len(corrected) == 1
        assert abs(corrected[0] - expected_ms) < 1e-6

    def test_multiple_stim_events(self):
        """
        Multiple stim events are each recentered independently.

        Tests:
            (Test Case 2) Each corrected time matches its planted peak.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times

        fs_Hz = 20000.0
        n_samples = 80000
        true_samples = [10000, 30000, 50000]
        traces = self._make_traces(2, n_samples, fs_Hz, true_samples)

        # Offsets: -3 ms, +2 ms, -1 ms
        logged_ms = [
            true_samples[0] / fs_Hz * 1000.0 - 3.0,
            true_samples[1] / fs_Hz * 1000.0 + 2.0,
            true_samples[2] / fs_Hz * 1000.0 - 1.0,
        ]
        corrected = recenter_stim_times(traces, logged_ms, fs_Hz, max_offset_ms=50.0)

        assert len(corrected) == 3
        for i, ts in enumerate(true_samples):
            expected_ms = ts / fs_Hz * 1000.0
            assert abs(corrected[i] - expected_ms) < 1e-6

    def test_stim_near_recording_start(self):
        """
        Stim event near recording start clips the search window.

        Tests:
            (Test Case 3) Artifact at sample 5 is found despite clipped window.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times

        fs_Hz = 20000.0
        n_samples = 40000
        true_sample = 5
        traces = self._make_traces(2, n_samples, fs_Hz, [true_sample])

        logged_ms = true_sample / fs_Hz * 1000.0 + 1.0
        corrected = recenter_stim_times(traces, [logged_ms], fs_Hz, max_offset_ms=50.0)

        expected_ms = true_sample / fs_Hz * 1000.0
        assert abs(corrected[0] - expected_ms) < 1e-6

    def test_stim_near_recording_end(self):
        """
        Stim event near recording end clips the search window.

        Tests:
            (Test Case 4) Artifact near the last sample is found.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times

        fs_Hz = 20000.0
        n_samples = 40000
        true_sample = n_samples - 5
        traces = self._make_traces(2, n_samples, fs_Hz, [true_sample])

        logged_ms = true_sample / fs_Hz * 1000.0 - 1.0
        corrected = recenter_stim_times(traces, [logged_ms], fs_Hz, max_offset_ms=50.0)

        expected_ms = true_sample / fs_Hz * 1000.0
        assert abs(corrected[0] - expected_ms) < 1e-6

    def test_no_offset_needed(self):
        """
        Logged time already at the peak — corrected time is unchanged.

        Tests:
            (Test Case 5) Corrected time equals the logged time.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times

        fs_Hz = 20000.0
        n_samples = 40000
        true_sample = 10000
        traces = self._make_traces(2, n_samples, fs_Hz, [true_sample])

        logged_ms = true_sample / fs_Hz * 1000.0
        corrected = recenter_stim_times(traces, [logged_ms], fs_Hz, max_offset_ms=50.0)

        assert abs(corrected[0] - logged_ms) < 1e-6

    def test_single_channel(self):
        """
        Works with a single-channel recording.

        Tests:
            (Test Case 6) Recentering succeeds with shape (1, samples).
        """
        from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times

        fs_Hz = 20000.0
        n_samples = 40000
        true_sample = 10000
        traces = self._make_traces(1, n_samples, fs_Hz, [true_sample])

        logged_ms = true_sample / fs_Hz * 1000.0 + 2.0
        corrected = recenter_stim_times(traces, [logged_ms], fs_Hz, max_offset_ms=50.0)

        expected_ms = true_sample / fs_Hz * 1000.0
        assert abs(corrected[0] - expected_ms) < 1e-6

    def test_empty_stim_times(self):
        """
        Empty stim_times array returns an empty array.

        Tests:
            (Test Case 7) Output is empty with length 0.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times

        fs_Hz = 20000.0
        traces = np.random.default_rng(42).normal(0, 1, (2, 40000))

        corrected = recenter_stim_times(traces, [], fs_Hz)
        assert len(corrected) == 0

    def test_large_max_offset_covering_full_recording(self):
        """
        max_offset_ms larger than recording duration still finds the peak.

        Tests:
            (Test Case 8) Artifact found when search window spans entire trace.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times

        fs_Hz = 20000.0
        n_samples = 40000  # 2 seconds
        true_sample = 10000
        traces = self._make_traces(2, n_samples, fs_Hz, [true_sample])

        logged_ms = true_sample / fs_Hz * 1000.0 + 500.0  # way off
        corrected = recenter_stim_times(
            traces, [logged_ms], fs_Hz, max_offset_ms=5000.0
        )

        expected_ms = true_sample / fs_Hz * 1000.0
        assert abs(corrected[0] - expected_ms) < 1e-6


# ===========================================================================
# Stim sorting — artifact removal
# ===========================================================================


class TestRemoveStimArtifacts:
    """
    Tests for remove_stim_artifacts (polynomial detrend and blanking).

    Tests:
        (Test Case 1) Polynomial method removes a slow exponential artifact.
        (Test Case 2) Polynomial method preserves a spike in the artifact tail.
        (Test Case 3) Blank method zeros out the artifact window.
        (Test Case 4) Saturated samples are blanked in both methods.
        (Test Case 5) Sequential stims produce merged blanking region.
        (Test Case 6) Empty stim_times returns traces unchanged.
        (Test Case 7) Unknown method raises ValueError.
        (Test Case 8) copy=True does not modify input; copy=False does.
        (Test Case 9) Auto-detection of saturation threshold.
        (Test Case 10) Single channel, single stim event.
    """

    @staticmethod
    def _make_artifact_traces(
        n_channels,
        n_samples,
        fs_Hz,
        stim_sample,
        saturation_samples=5,
        saturation_amp=1000.0,
        decay_samples=200,
        decay_amp=50.0,
    ):
        """Create traces with a synthetic artifact: saturation + exponential decay."""
        rng = np.random.default_rng(42)
        traces = rng.normal(0, 0.5, (n_channels, n_samples))

        for ch in range(n_channels):
            # Saturation region
            end_sat = min(stim_sample + saturation_samples, n_samples)
            traces[ch, stim_sample:end_sat] = saturation_amp

            # Exponential decay tail
            t = np.arange(decay_samples, dtype=np.float64)
            decay = decay_amp * np.exp(-t / 50.0)
            start_decay = end_sat
            end_decay = min(start_decay + decay_samples, n_samples)
            actual_len = end_decay - start_decay
            traces[ch, start_decay:end_decay] += decay[:actual_len]

        return traces

    def test_polynomial_removes_exponential_artifact(self):
        """
        Polynomial method removes a slow exponential decay artifact.

        Tests:
            (Test Case 1) Residual in the artifact tail is much smaller
            than the original artifact amplitude.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        n_samples = 40000
        stim_sample = 10000
        decay_amp = 50.0

        traces = self._make_artifact_traces(
            2, n_samples, fs_Hz, stim_sample, decay_amp=decay_amp
        )
        stim_ms = [stim_sample / fs_Hz * 1000.0]

        cleaned, blanked = remove_stim_artifacts(
            traces,
            stim_ms,
            fs_Hz,
            method="polynomial",
            artifact_window_ms=15.0,
            saturation_threshold=500.0,
            baseline_threshold=5.0,
        )

        # The artifact tail region (after saturation)
        tail_start = stim_sample + 5
        tail_end = stim_sample + 205
        residual_max = np.max(np.abs(cleaned[:, tail_start:tail_end]))

        # Residual should be much smaller than original decay amplitude
        assert (
            residual_max < decay_amp * 0.5
        ), f"Residual {residual_max:.1f} not much smaller than artifact {decay_amp}"

    def test_polynomial_preserves_spike_in_tail(self):
        """
        A spike planted in the artifact tail survives polynomial subtraction.

        Tests:
            (Test Case 2) Spike amplitude in cleaned trace is at least 50%
            of the planted amplitude.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        n_samples = 40000
        stim_sample = 10000
        spike_amp = 20.0

        traces = self._make_artifact_traces(
            2, n_samples, fs_Hz, stim_sample, decay_amp=30.0
        )

        # Plant a fast spike (1 sample wide) in the tail on channel 0
        spike_sample = stim_sample + 80  # well after saturation
        traces[0, spike_sample] += spike_amp

        stim_ms = [stim_sample / fs_Hz * 1000.0]

        cleaned, _ = remove_stim_artifacts(
            traces,
            stim_ms,
            fs_Hz,
            method="polynomial",
            artifact_window_ms=15.0,
            saturation_threshold=500.0,
            baseline_threshold=5.0,
        )

        # The spike should survive — polynomial is too smooth to capture it
        # Check a small window around the spike
        window = cleaned[0, spike_sample - 2 : spike_sample + 3]
        peak_val = np.max(np.abs(window))
        assert (
            peak_val > spike_amp * 0.5
        ), f"Spike peak {peak_val:.1f} is less than 50% of planted {spike_amp}"

    def test_blank_method_zeros_window(self):
        """
        Blank method zeros out the artifact window.

        Tests:
            (Test Case 3) All samples in the blanked region are zero.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        n_samples = 40000
        stim_sample = 10000

        traces = self._make_artifact_traces(2, n_samples, fs_Hz, stim_sample)
        stim_ms = [stim_sample / fs_Hz * 1000.0]

        cleaned, blanked = remove_stim_artifacts(
            traces,
            stim_ms,
            fs_Hz,
            method="blank",
            artifact_window_ms=10.0,
            saturation_threshold=500.0,
            baseline_threshold=5.0,
        )

        # All blanked samples should be zero
        assert np.all(cleaned[blanked] == 0.0)
        # There should be some blanked samples
        assert np.any(blanked)

    def test_saturated_samples_blanked_polynomial(self):
        """
        Saturated samples are blanked (zeroed) in polynomial method.

        Tests:
            (Test Case 4) Samples at or above saturation threshold are zero
            in cleaned output.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        n_samples = 40000
        stim_sample = 10000

        traces = self._make_artifact_traces(
            2, n_samples, fs_Hz, stim_sample, saturation_amp=1000.0
        )
        stim_ms = [stim_sample / fs_Hz * 1000.0]

        cleaned, blanked = remove_stim_artifacts(
            traces,
            stim_ms,
            fs_Hz,
            method="polynomial",
            saturation_threshold=500.0,
            baseline_threshold=5.0,
        )

        # Saturation region should be blanked
        sat_region = blanked[:, stim_sample : stim_sample + 5]
        assert np.all(sat_region), "Saturation region not fully blanked"

        # Those samples should be zero
        assert np.all(cleaned[:, stim_sample : stim_sample + 5] == 0.0)

    def test_sequential_stims_merged_blanking(self):
        """
        Two stims close together merge their blanking regions.

        Tests:
            (Test Case 5) Blanked region spans from first stim through
            second stim's artifact window.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        n_samples = 40000
        stim1 = 10000
        stim2 = 10010  # 0.5 ms apart — signal can't recover

        rng = np.random.default_rng(42)
        traces = rng.normal(0, 0.5, (2, n_samples))
        # Plant saturation at both stims
        for s in [stim1, stim2]:
            traces[:, s : s + 5] = 1000.0

        stim_ms = [s / fs_Hz * 1000.0 for s in [stim1, stim2]]

        cleaned, blanked = remove_stim_artifacts(
            traces,
            stim_ms,
            fs_Hz,
            method="blank",
            artifact_window_ms=5.0,
            saturation_threshold=500.0,
            baseline_threshold=5.0,
        )

        # The region from stim1 through stim2+artifact should be blanked
        # At minimum, the gap between stim1 and stim2 should be blanked
        assert np.all(blanked[:, stim1 : stim2 + 5])

    def test_empty_stim_times_returns_unchanged(self):
        """
        Empty stim_times returns a copy of traces unchanged.

        Tests:
            (Test Case 6) Output equals input and blanked mask is all False.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        traces = np.random.default_rng(42).normal(0, 1, (2, 40000))

        cleaned, blanked = remove_stim_artifacts(traces, [], fs_Hz)

        np.testing.assert_array_equal(cleaned, traces)
        assert not np.any(blanked)

    def test_unknown_method_raises(self):
        """
        Unknown method name raises ValueError.

        Tests:
            (Test Case 7) ValueError with informative message.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        traces = np.random.default_rng(42).normal(0, 1, (2, 40000))

        with pytest.raises(ValueError, match="Unknown artifact removal method"):
            remove_stim_artifacts(traces, [100.0], fs_Hz, method="magic")

    def test_copy_true_does_not_modify_input(self):
        """
        copy=True returns a new array; copy=False modifies in place.

        Tests:
            (Test Case 8a) copy=True: input array unchanged after call.
            (Test Case 8b) copy=False: input array is modified.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        n_samples = 40000
        stim_sample = 10000

        # --- copy=True ---
        traces = self._make_artifact_traces(2, n_samples, fs_Hz, stim_sample)
        original = traces.copy()
        _ = remove_stim_artifacts(
            traces,
            [stim_sample / fs_Hz * 1000.0],
            fs_Hz,
            method="blank",
            saturation_threshold=500.0,
            baseline_threshold=5.0,
            copy=True,
        )
        np.testing.assert_array_equal(traces, original)

        # --- copy=False ---
        traces2 = self._make_artifact_traces(2, n_samples, fs_Hz, stim_sample)
        original2 = traces2.copy()
        _ = remove_stim_artifacts(
            traces2,
            [stim_sample / fs_Hz * 1000.0],
            fs_Hz,
            method="blank",
            saturation_threshold=500.0,
            baseline_threshold=5.0,
            copy=False,
        )
        assert not np.array_equal(
            traces2, original2
        ), "copy=False should modify traces in place"

    def test_auto_saturation_threshold(self):
        """
        Auto-detection of saturation threshold finds obvious saturation.

        Tests:
            (Test Case 9) With saturation_threshold=None, saturated samples
            are still blanked.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        n_samples = 40000
        stim_sample = 10000

        traces = self._make_artifact_traces(
            2, n_samples, fs_Hz, stim_sample, saturation_amp=1000.0
        )
        stim_ms = [stim_sample / fs_Hz * 1000.0]

        # Let both thresholds auto-detect
        cleaned, blanked = remove_stim_artifacts(
            traces,
            stim_ms,
            fs_Hz,
            method="polynomial",
            saturation_threshold=None,
            baseline_threshold=None,
        )

        # At least some samples around the stim should be blanked
        assert np.any(blanked[:, stim_sample : stim_sample + 10])

    def test_single_channel_single_stim(self):
        """
        Works with a single channel and single stim event.

        Tests:
            (Test Case 10) No errors; output shapes match input.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        fs_Hz = 20000.0
        n_samples = 40000
        stim_sample = 10000

        traces = self._make_artifact_traces(1, n_samples, fs_Hz, stim_sample)
        stim_ms = [stim_sample / fs_Hz * 1000.0]

        cleaned, blanked = remove_stim_artifacts(
            traces,
            stim_ms,
            fs_Hz,
            method="polynomial",
            saturation_threshold=500.0,
            baseline_threshold=5.0,
        )

        assert cleaned.shape == (1, n_samples)
        assert blanked.shape == (1, n_samples)
        assert np.any(blanked)


# ===========================================================================
# Stim sorting — pipeline input validation
# ===========================================================================


class TestSortStimRecordingValidation:
    """
    Tests for sort_stim_recording input validation (no torch/SI required).

    Tests:
        (Test Case 1) numpy array without fs_Hz raises ValueError.
        (Test Case 2) Wrong dimensionality array raises ValueError.
        (Test Case 3) stim_times as list (not array) works.
    """

    def test_numpy_array_without_fs_raises(self):
        """
        Passing a numpy array without fs_Hz raises ValueError.

        Tests:
            (Test Case 1) ValueError mentions fs_Hz is required.
        """
        from spikelab.spike_sorting.stim_sorting.pipeline import sort_stim_recording

        traces = np.zeros((2, 1000))
        with pytest.raises(ValueError, match="fs_Hz is required"):
            sort_stim_recording(
                traces,
                rt_sort=MagicMock(),
                stim_times_ms=[100.0],
                pre_ms=10.0,
                post_ms=50.0,
                fs_Hz=None,
            )

    def test_wrong_ndim_raises(self):
        """
        A 1-D or 3-D array raises ValueError about shape.

        Tests:
            (Test Case 2) ValueError for 1-D array.
        """
        from spikelab.spike_sorting.stim_sorting.pipeline import sort_stim_recording

        traces_1d = np.zeros(1000)
        with pytest.raises(ValueError, match="Expected 2-D array"):
            sort_stim_recording(
                traces_1d,
                rt_sort=MagicMock(),
                stim_times_ms=[100.0],
                pre_ms=10.0,
                post_ms=50.0,
                fs_Hz=20000.0,
            )

    def test_stim_times_as_list(self):
        """
        stim_times_ms as a plain Python list is accepted by input
        validation.  The pipeline may run to completion with a
        MagicMock ``rt_sort`` (the mock's sort_offline pretends to
        return an empty sorting) — the key assertion is that no
        ``TypeError`` / ``ValueError`` is raised complaining about
        the ``stim_times_ms`` argument shape or dtype.
        """
        from spikelab.spike_sorting.stim_sorting.pipeline import sort_stim_recording

        traces = np.zeros((2, 40000))

        try:
            sort_stim_recording(
                traces,
                rt_sort=MagicMock(),
                stim_times_ms=[100.0, 200.0, 300.0],
                pre_ms=10.0,
                post_ms=50.0,
                fs_Hz=20000.0,
            )
        except Exception as exc:
            # The list input itself should not be what broke things.
            assert "stim_times" not in str(exc).lower()


class TestSaturationThresholdFromRecording:
    """``_saturation_threshold_from_recording`` semantics:

    * Returns ``+inf`` when no clipping is detected (single-spike maxima
      are not saturation), so non-saturated recordings get *no* blanking
      and the polynomial detrend handles every event.
    * Returns a finite gain-anchored threshold when clipping IS detected
      — the threshold is rounded to a whole number of raw ADC bits so
      it's hardware-meaningful and reproducible.
    * Falls back to a 1.0 µV/bit assumption when no recording metadata
      is available.
    """

    def test_returns_inf_when_no_clipping(self):
        from unittest.mock import MagicMock

        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _saturation_threshold_from_recording,
        )

        # Random noise + a single big spike. Max is unique to that spike.
        rng = np.random.default_rng(0)
        traces = rng.standard_normal((4, 100_000)).astype(np.float32) * 5.0
        traces[1, 50_000] = 800.0  # unique large peak

        rec = MagicMock()
        rec.get_channel_gains.return_value = np.full(4, 3.14, dtype=np.float32)

        thresh = _saturation_threshold_from_recording(rec, traces)
        assert thresh == float(
            "inf"
        ), f"Expected +inf for non-saturated recording, got {thresh}"

    def test_returns_finite_threshold_when_clipped(self):
        from unittest.mock import MagicMock

        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _saturation_threshold_from_recording,
        )

        # Many samples pinned at +-rail (simulating a clipped ADC).
        rng = np.random.default_rng(0)
        traces = rng.standard_normal((4, 100_000)).astype(np.float32) * 5.0
        # Pin 200 samples to ±10000 µV across two channels = clear clipping
        traces[2, 1000:1100] = 10_000.0
        traces[3, 2000:2100] = -10_000.0

        rec = MagicMock()
        rec.get_channel_gains.return_value = np.full(4, 3.14, dtype=np.float32)

        thresh = _saturation_threshold_from_recording(rec, traces)
        # Threshold should be just below the rail (frac=0.95 default).
        assert 9_000.0 < thresh < 10_000.0, f"Expected threshold ~9500 µV, got {thresh}"
        # And it should be a whole number of raw bits times the gain.
        rail_bits = round(10_000.0 / 3.14)
        expected = 0.95 * rail_bits * 3.14
        assert abs(thresh - expected) < 1e-3

    def test_min_clip_samples_threshold(self):
        """Tunable ``min_clip_samples`` decides when "many samples at max"
        counts as a clip vs. just spikes."""
        from unittest.mock import MagicMock

        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _saturation_threshold_from_recording,
        )

        rec = MagicMock()
        rec.get_channel_gains.return_value = np.array([3.14], dtype=np.float32)

        # 5 samples at the rail — below default min_clip_samples=10, so
        # the recording is treated as unsaturated.
        traces_sparse = np.zeros((1, 1000), dtype=np.float32)
        traces_sparse[0, :5] = 5_000.0
        assert _saturation_threshold_from_recording(rec, traces_sparse) == float("inf")

        # 50 samples at the rail — above default, so threshold is finite.
        traces_dense = np.zeros((1, 1000), dtype=np.float32)
        traces_dense[0, :50] = 5_000.0
        thresh_default = _saturation_threshold_from_recording(rec, traces_dense)
        assert thresh_default < 5_000.0
        assert thresh_default > 4_500.0

        # Raising ``min_clip_samples`` above 50 pushes the same dense
        # traces back to the non-saturated branch.
        assert _saturation_threshold_from_recording(
            rec, traces_dense, min_clip_samples=100
        ) == float("inf")

    def test_no_recording_falls_back(self):
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _saturation_threshold_from_recording,
        )

        traces = np.zeros((1, 1000), dtype=np.float32)
        traces[0, :200] = 7_777.0

        # Without a recording, gain defaults to 1.0 — but saturation
        # detection still works (200 samples at 7777 µV passes the
        # default min_clip_samples=100).
        thresh = _saturation_threshold_from_recording(None, traces)
        assert 7_000.0 < thresh < 7_777.0

    def test_remove_stim_artifacts_uses_inf_threshold_no_blanking(self):
        """End-to-end: remove_stim_artifacts called with a recording on
        non-saturated traces produces ZERO blanked samples."""
        from unittest.mock import MagicMock

        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        rng = np.random.default_rng(0)
        fs = 20_000.0
        n_samples = int(0.5 * fs)
        # Background noise + small spikes — never reaches a "rail"
        traces = rng.standard_normal((4, n_samples)).astype(np.float32) * 10.0
        traces[1, 5_000] = 200.0  # one peak

        rec = MagicMock()
        rec.get_channel_gains.return_value = np.full(4, 3.14, dtype=np.float32)

        cleaned, blanked = remove_stim_artifacts(
            traces.copy(),
            stim_times_ms=[100.0, 250.0, 400.0],
            fs_Hz=fs,
            method="polynomial",
            recording=rec,
        )

        assert blanked.sum() == 0, (
            f"Expected 0 blanked samples on non-saturated recording, "
            f"got {int(blanked.sum())}"
        )


@skip_no_spikeinterface
class TestSortStimRecordingPassesNdarray:
    """sort_stim_recording must hand the cleaned traces to sort_offline as
    a raw ndarray, not as a NumpyRecording wrapper.

    Why this matters: ``RTSort.sort_offline`` short-circuits past its
    entire ``save_traces`` pipeline when it receives an ndarray.  If
    we wrap the cleaned 5.6 GB Phase-2 array in a NumpyRecording,
    ``save_traces_si`` instead opens a ``multiprocessing.Manager``
    that pickles the whole array into a separate proxy process and
    forks 16 workers off the parent — the OOM-kill pattern observed
    in the lowmem-rt-sort investigation.  This test pins the fix.
    """

    def test_sort_offline_receives_ndarray_not_recording(self):
        from unittest.mock import MagicMock

        from spikelab.spike_sorting.stim_sorting.pipeline import sort_stim_recording

        traces = (
            np.random.default_rng(0).standard_normal((4, 40000)).astype(np.float32)
            * 10.0
        )

        captured = {"recording": None}

        def fake_sort_offline(*, recording, **kw):
            captured["recording"] = recording
            ns = MagicMock()
            ns.get_unit_ids.return_value = [0]
            ns.get_unit_spike_train.return_value = np.array([100, 200], dtype=np.int64)
            ns.get_num_samples.return_value = traces.shape[1]
            return ns

        fake_rt_sort = MagicMock()
        fake_rt_sort.sort_offline = fake_sort_offline

        sort_stim_recording(
            traces,
            rt_sort=fake_rt_sort,
            stim_times_ms=[100.0, 500.0],
            pre_ms=10.0,
            post_ms=50.0,
            fs_Hz=20000.0,
            verbose=False,
        )

        assert captured["recording"] is not None, "sort_offline was never called"
        # Critical: must be an ndarray, NOT a SpikeInterface BaseRecording
        from spikeinterface.core import NumpyRecording
        from spikeinterface.core.baserecording import BaseRecording

        assert isinstance(captured["recording"], np.ndarray), (
            f"sort_offline received {type(captured['recording']).__name__}; "
            "expected ndarray to take the fast path"
        )
        assert not isinstance(captured["recording"], (NumpyRecording, BaseRecording)), (
            "sort_offline received a recording wrapper — this would route "
            "through save_traces_si and trigger the multiprocessing OOM"
        )
        assert captured["recording"].shape == (4, 40000)


@skip_no_spikeinterface
class TestSaveTracesSiFastPath:
    """save_traces_si fast-paths in-memory recordings.

    For a NumpyRecording the data is already in RAM; the parallel
    chunked extractor (``Manager`` + ``Pool``) is pure overhead and
    can OOM on multi-GB arrays.  The fast path writes the array
    directly via ``np.save`` and never instantiates a Pool.

    For non-in-memory recordings (e.g. lazy filter chains over a disk
    extractor), the parallel path is still useful and must remain in
    place.
    """

    @skip_no_torch
    def test_numpy_recording_skips_multiprocessing_pool(self, tmp_path, monkeypatch):
        from spikeinterface.core import NumpyRecording

        from spikelab.spike_sorting.rt_sort import _algorithm

        rng = np.random.default_rng(0)
        # NumpyRecording expects (samples, channels)
        traces_in = (rng.standard_normal((10_000, 4)) * 5.0).astype(np.float32)
        rec = NumpyRecording(traces_list=[traces_in], sampling_frequency=20_000.0)

        pool_instantiations = []

        def fail_if_called(*a, **kw):
            pool_instantiations.append((a, kw))
            raise AssertionError(
                "save_traces_si instantiated a multiprocessing Pool/threadpool "
                "for an in-memory NumpyRecording — fast path failed"
            )

        # Fast path must avoid every parallelism mechanism.
        monkeypatch.setattr(_algorithm, "_thread_map", fail_if_called)
        monkeypatch.setattr(_algorithm, "Pool", fail_if_called)
        monkeypatch.setattr(_algorithm, "Manager", fail_if_called)

        out_path = tmp_path / "scaled_traces.npy"
        _algorithm.save_traces_si(
            rec,
            out_path,
            start_ms=0,
            end_ms=None,
            num_processes=16,
            dtype="float32",
            verbose=False,
        )

        assert pool_instantiations == []

        saved = np.load(out_path)
        assert saved.shape == (4, 10_000)
        np.testing.assert_allclose(saved, traces_in.T, atol=1e-6)

    @skip_no_torch
    def test_non_numpy_recording_uses_time_chunked_bulk_read(
        self, tmp_path, monkeypatch
    ):
        """Lazy chains go through a time-chunked bulk read — one
        ``get_traces(start_frame, end_frame)`` call per chunk, covering
        *all* channels at once.

        Previous implementations walked channel-by-channel, which on a
        ``BandpassFilter → Maxwell`` chain triggered one full-duration
        filter pass per channel (1018 passes for a 1018-channel probe).
        Time-chunking replaces that with one filter pass per time chunk
        (e.g. ~36 passes for 180 s / 5 s-chunks) and is typically 1-2
        orders of magnitude faster.  No multiprocessing Pool or Manager
        is needed; the filter itself vectorises over channels.
        """
        from unittest.mock import MagicMock

        from spikelab.spike_sorting.rt_sort import _algorithm

        n_channels = 4
        n_samples = 10_000
        rng = np.random.default_rng(0)
        fake_full = (rng.standard_normal((n_samples, n_channels)) * 10.0).astype(
            np.float32
        )

        get_traces_calls = []

        def fake_get_traces(start_frame=None, end_frame=None, return_scaled=False):
            get_traces_calls.append((start_frame, end_frame))
            sf = 0 if start_frame is None else start_frame
            ef = n_samples if end_frame is None else end_frame
            return fake_full[sf:ef]

        rec = MagicMock()
        rec.get_sampling_frequency.return_value = 20_000.0
        rec.get_num_channels.return_value = n_channels
        rec.get_total_samples.return_value = n_samples
        rec.get_channel_ids.return_value = np.arange(n_channels)
        rec.has_scaleable_traces.return_value = True
        rec.get_traces.side_effect = fake_get_traces

        from spikeinterface.core import NumpyRecording

        assert not isinstance(rec, NumpyRecording)

        bare_pool_calls = []
        bare_manager_calls = []

        def fake_bare_pool(*a, **kw):
            bare_pool_calls.append(kw)
            raise AssertionError("bare Pool used — should be time-chunked")

        def fake_bare_manager(*a, **kw):
            bare_manager_calls.append(True)
            raise AssertionError("bare Manager used — should be time-chunked")

        monkeypatch.setattr(_algorithm, "Pool", fake_bare_pool)
        monkeypatch.setattr(_algorithm, "Manager", fake_bare_manager)

        out_path = tmp_path / "scaled_traces.npy"
        # 10 000 samples @ 20 kHz = 0.5 s total — with default 5 s
        # chunk it becomes a single chunk; force smaller chunks to
        # exercise the loop.
        _algorithm.save_traces_si(
            rec,
            out_path,
            start_ms=0,
            end_ms=None,
            num_processes=4,
            dtype="float32",
            verbose=False,
            chunk_seconds=0.1,  # 2 000-sample chunks → 5 chunks
        )

        # Should have been called once per chunk, all channels per call
        assert len(get_traces_calls) == 5
        for sf, ef in get_traces_calls:
            assert (ef - sf) <= 2_000
        assert bare_pool_calls == []
        assert bare_manager_calls == []

        saved = np.load(out_path)
        assert saved.shape == (n_channels, n_samples)
        np.testing.assert_allclose(saved, fake_full.T, atol=1e-6)

    @skip_no_torch
    def test_save_traces_si_chunked_matches_single_pass(self, tmp_path):
        """Chunked output on a realistic lazy chain (bandpass over noise)
        matches the single-pass output to within a small filter-
        boundary tolerance.

        ``BandpassFilterRecording`` applies a fixed margin at chunk
        edges to approximate continuity; differences are
        sub-quantization (~1 µV) and irrelevant for spike sorting,
        where the noise floor is ~10 µV.
        """
        from spikeinterface.core import NumpyRecording
        from spikeinterface.preprocessing import bandpass_filter

        from spikelab.spike_sorting.rt_sort import _algorithm

        rng = np.random.default_rng(42)
        fs = 20_000.0
        n_samples = int(fs)
        raw = (rng.standard_normal((n_samples, 4)) * 20.0).astype(np.float32)
        base = NumpyRecording(traces_list=[raw], sampling_frequency=fs)
        filtered = bandpass_filter(base, freq_min=300.0, freq_max=6000.0)

        big_path = tmp_path / "big.npy"
        _algorithm.save_traces_si(
            filtered, big_path, dtype="float32", verbose=False, chunk_seconds=10.0
        )
        small_path = tmp_path / "small.npy"
        _algorithm.save_traces_si(
            filtered, small_path, dtype="float32", verbose=False, chunk_seconds=0.2
        )

        big = np.load(big_path)
        small = np.load(small_path)
        # Well below the 3.147 µV/bit gain quantization and the typical
        # ~10 µV recording noise floor.
        np.testing.assert_allclose(small, big, atol=1.0)


class TestChunkedStimSort:
    """Unit tests for the per-event-chunked ``sort_stim_recording``
    helpers — event grouping + shape of the chunked path.

    Full end-to-end chunked-vs-full equivalence is too heavy for a
    unit test (needs a real RTSort object); here we verify just the
    grouping and chunk-boundary logic in isolation.
    """

    def test_group_events_isolated_non_overlapping(self):
        from spikelab.spike_sorting.stim_sorting.pipeline import (
            _group_stim_events_into_chunks,
        )

        # Events 2 s apart; chunk window 500 ms / 500 ms → no overlap.
        times_ms = np.array([1000.0, 3000.0, 5000.0, 7000.0])
        groups = _group_stim_events_into_chunks(
            times_ms, chunk_pre_ms=500.0, chunk_post_ms=500.0
        )
        assert len(groups) == 4
        assert all(len(g) == 1 for g in groups)
        flat = [i for g in groups for i in g]
        assert flat == [0, 1, 2, 3]

    def test_group_events_burst_merged(self):
        from spikelab.spike_sorting.stim_sorting.pipeline import (
            _group_stim_events_into_chunks,
        )

        # A burst of 4 events at 100 ms spacing — chunk window 500 ms
        # so every pair's windows overlap → single group.
        times_ms = np.array([1000.0, 1100.0, 1200.0, 1300.0])
        groups = _group_stim_events_into_chunks(
            times_ms, chunk_pre_ms=500.0, chunk_post_ms=500.0
        )
        assert len(groups) == 1
        assert groups[0] == [0, 1, 2, 3]

    def test_group_events_mixed_groups(self):
        from spikelab.spike_sorting.stim_sorting.pipeline import (
            _group_stim_events_into_chunks,
        )

        # Two bursts of 2 events each, far apart.
        times_ms = np.array(
            [
                1000.0,
                1200.0,  # burst 1 at +200 ms
                5000.0,
                5200.0,  # burst 2 far away
            ]
        )
        groups = _group_stim_events_into_chunks(
            times_ms, chunk_pre_ms=400.0, chunk_post_ms=400.0
        )
        assert len(groups) == 2
        assert groups[0] == [0, 1]
        assert groups[1] == [2, 3]

    def test_group_events_unsorted_input(self):
        from spikelab.spike_sorting.stim_sorting.pipeline import (
            _group_stim_events_into_chunks,
        )

        # Input out of order — grouping sorts, but returns indices into
        # the original array.  Indices within a group come out in time
        # order, so we compare against a time-sorted expectation.
        times_ms = np.array([5000.0, 1000.0, 1100.0, 5200.0])
        groups = _group_stim_events_into_chunks(
            times_ms, chunk_pre_ms=400.0, chunk_post_ms=400.0
        )
        assert len(groups) == 2
        # First group in sort order = events at 1000, 1100 (indices 1, 2)
        assert groups[0] == [1, 2]
        # Second group = events at 5000, 5200 (indices 0, 3)
        assert groups[1] == [0, 3]

    def test_group_events_empty(self):
        from spikelab.spike_sorting.stim_sorting.pipeline import (
            _group_stim_events_into_chunks,
        )

        groups = _group_stim_events_into_chunks(
            np.array([]), chunk_pre_ms=500.0, chunk_post_ms=500.0
        )
        assert groups == []


class TestRecenterStimTimesPeakModes:
    """Tests for ``peak_mode`` in ``recenter_stim_times``.

    The down-edge algorithm: find the negative peak first, then the
    positive peak within a preceding ``prewindow_ms`` window, then the
    first + → − zero-crossing between them (or steepest negative slope
    if no zero crossing).
    """

    @staticmethod
    def _synth_biphasic(
        n_channels=16,
        n_samples=40000,
        fs_Hz=20000.0,
        stim_sample=10000,
        up_duration_samples=4,
        down_duration_samples=4,
        up_amp=3000.0,
        down_amp=-10000.0,
        n_affected_channels=8,
    ):
        """Inject a biphasic (anodic-first) artifact at ``stim_sample``.

        Channels 0..``n_affected_channels``-1 see the full artifact;
        the rest see only noise.  The up phase runs for
        ``up_duration_samples`` starting at ``stim_sample``, followed
        immediately by the down phase of ``down_duration_samples``.

        The true transition (first + → − zero crossing) lands at
        ``stim_sample + up_duration_samples``.
        """
        rng = np.random.default_rng(0)
        traces = (rng.standard_normal((n_channels, n_samples)) * 5.0).astype(np.float32)
        up_start = stim_sample
        down_start = stim_sample + up_duration_samples
        down_end = down_start + down_duration_samples
        for ch in range(min(n_affected_channels, n_channels)):
            traces[ch, up_start:down_start] += up_amp
            traces[ch, down_start:down_end] += down_amp
        return traces, down_start  # true transition sample

    def test_invalid_peak_mode_raises(self):
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        traces = np.zeros((2, 1000))
        with pytest.raises(ValueError, match="Unknown peak_mode"):
            recenter_stim_times(traces, [10.0], fs_Hz=20000.0, peak_mode="garbage")

    def test_abs_max_backward_compatible(self):
        """Without ``peak_mode``, behavior matches the pre-patch API."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        fs_Hz = 20000.0
        traces, transition = self._synth_biphasic()
        # Logged time is a few ms off
        stim_ms = (transition / fs_Hz * 1000.0) - 3.0

        # Default is abs_max → should land on the largest |V|, i.e. the
        # negative peak of the down phase, NOT the transition.
        corrected = recenter_stim_times(
            traces, [stim_ms], fs_Hz=fs_Hz, max_offset_ms=10.0
        )
        corrected_sample = int(np.round(corrected[0] * fs_Hz / 1000.0))

        # Down phase starts at ``transition``, length 4 samples.
        # abs_max picks some sample in the down phase (amplitude 10000 > 3000).
        assert transition <= corrected_sample < transition + 4

    def test_neg_peak_finds_negative_extremum(self):
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        fs_Hz = 20000.0
        traces, transition = self._synth_biphasic()
        stim_ms = (transition / fs_Hz * 1000.0) - 3.0

        corrected = recenter_stim_times(
            traces,
            [stim_ms],
            fs_Hz=fs_Hz,
            max_offset_ms=10.0,
            peak_mode="neg_peak",
            n_reference_channels=8,
        )
        corrected_sample = int(np.round(corrected[0] * fs_Hz / 1000.0))
        # Should land somewhere inside the down phase (all 4 samples equal)
        assert transition <= corrected_sample < transition + 4

    def test_pos_peak_finds_positive_extremum(self):
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        fs_Hz = 20000.0
        traces, transition = self._synth_biphasic()
        stim_ms = (transition / fs_Hz * 1000.0) - 3.0

        corrected = recenter_stim_times(
            traces,
            [stim_ms],
            fs_Hz=fs_Hz,
            max_offset_ms=10.0,
            peak_mode="pos_peak",
            n_reference_channels=8,
        )
        corrected_sample = int(np.round(corrected[0] * fs_Hz / 1000.0))
        # Up phase spans [transition - 4, transition)
        assert transition - 4 <= corrected_sample < transition

    def test_down_edge_finds_up_to_down_transition(self):
        """``down_edge`` lands at the first + → − zero crossing between
        the positive peak and the negative peak."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        fs_Hz = 20000.0
        traces, transition = self._synth_biphasic(up_amp=3000.0, down_amp=-10000.0)
        stim_ms = (transition / fs_Hz * 1000.0) - 3.0

        corrected = recenter_stim_times(
            traces,
            [stim_ms],
            fs_Hz=fs_Hz,
            max_offset_ms=10.0,
            peak_mode="down_edge",
            n_reference_channels=8,
            prewindow_ms=1.0,
        )
        corrected_sample = int(np.round(corrected[0] * fs_Hz / 1000.0))
        # Transition is at ``transition``.  Tolerance ±1 sample for
        # zero-crossing discretisation.
        assert abs(corrected_sample - transition) <= 1

    def test_down_edge_uses_top_k_summed_reference(self):
        """Top-K summing should be robust when most channels see no
        artifact — only the top ``n_reference_channels`` contribute."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        fs_Hz = 20000.0
        # 8 channels see the artifact; 24 others are pure noise.  The
        # top-K sum (K=8) should isolate just the artifact channels.
        traces, transition = self._synth_biphasic(n_channels=32, n_affected_channels=8)
        stim_ms = (transition / fs_Hz * 1000.0) - 3.0

        corrected = recenter_stim_times(
            traces,
            [stim_ms],
            fs_Hz=fs_Hz,
            max_offset_ms=10.0,
            peak_mode="down_edge",
            n_reference_channels=8,
            prewindow_ms=1.0,
        )
        corrected_sample = int(np.round(corrected[0] * fs_Hz / 1000.0))
        assert abs(corrected_sample - transition) <= 1

    def test_down_edge_fallback_when_no_zero_crossing(self):
        """When the reference between the positive and negative peaks
        never crosses zero (e.g. DC-offset biphasic), the fallback
        returns the sample of steepest negative slope in that interval.

        Tests the ``_find_down_edge`` helper directly with a constructed
        reference so the test is decoupled from top-K selection and
        pulse synthesis.
        """
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_down_edge,
        )

        # Reference trace where indices 2-3 are the positive peak (10),
        # indices 6-7 are the negative "peak" (3), and nothing crosses
        # zero between them.  Steepest negative slope is between index
        # 3 (value 10) and index 4 (value 6): diff = -4 at index 3.
        reference = np.array([5.0, 5.0, 10.0, 10.0, 6.0, 4.0, 3.0, 3.0])

        # Search whole array.  fs_Hz=1000 gives prewindow_samples=4 for
        # prewindow_ms=4.0, enough to reach the positive peak.
        result = _find_down_edge(
            reference, lo=0, hi=len(reference), prewindow_ms=4.0, fs_Hz=1000.0
        )

        # argmin(diff) returns the first index of the steepest-drop
        # pair; diff[3] = -4 is the steepest, so ``pos_peak + 3``.
        # pos_peak itself is in [2, 3] (both equal to 10); argmax picks
        # the first, index 2.  Expected = 2 + 3 = 5, or nearby.
        assert (
            3 <= result <= 6
        ), f"Expected steepest-slope sample in [3, 6], got {result}"

    def test_up_edge_symmetric_to_down_edge(self):
        """For a cathodic-first biphasic pulse (first down then up),
        ``up_edge`` lands at the - -> + transition."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        fs_Hz = 20000.0
        # Cathodic-first: swap signs
        traces, transition = self._synth_biphasic(up_amp=-3000.0, down_amp=10000.0)
        stim_ms = (transition / fs_Hz * 1000.0) - 3.0

        corrected = recenter_stim_times(
            traces,
            [stim_ms],
            fs_Hz=fs_Hz,
            max_offset_ms=10.0,
            peak_mode="up_edge",
            n_reference_channels=8,
            prewindow_ms=1.0,
        )
        corrected_sample = int(np.round(corrected[0] * fs_Hz / 1000.0))
        assert abs(corrected_sample - transition) <= 1


# ===========================================================================
# Edge Case Tests -- SortingPipelineConfig
# ===========================================================================




# ===========================================================================
# Edge Case Tests -- Tee
# ===========================================================================


class TestTee:
    """Edge cases for the Tee context manager."""

    def test_stdout_restored_on_exception(self, tmp_path):
        """stdout is restored even when an exception occurs inside Tee."""
        from spikelab.spike_sorting.sorting_utils import Tee

        original_stdout = sys.stdout
        log_file = tmp_path / "test.log"
        with pytest.raises(RuntimeError, match="deliberate"):
            with Tee(log_file, "w"):
                raise RuntimeError("deliberate error")
        assert sys.stdout is original_stdout

    def test_write_skips_newline_and_space(self, tmp_path):
        """Tee._write does not echo newlines or spaces to stdout."""
        from spikelab.spike_sorting.sorting_utils import Tee
        from unittest.mock import MagicMock

        log_file = tmp_path / "test.log"
        with Tee(log_file, "w") as f:
            mock_stdout = MagicMock()
            f.stdout = mock_stdout
            f.write("\n")
            f.write(" ")
            # Neither should be echoed to stdout (print calls .write)
            mock_stdout.write.assert_not_called()
            # But a real message should be echoed via print(s, file=stdout)
            f.write("hello")
            mock_stdout.write.assert_called()


# ===========================================================================
# Edge Case Tests -- Utils._mem_to_int
# ===========================================================================


class TestMemToInt:
    """Edge cases for Utils._mem_to_int."""

    def test_empty_string_raises(self):
        """Empty string input raises an error."""
        from spikelab.spike_sorting.waveform_extractor import Utils

        with pytest.raises((IndexError, ValueError)):
            Utils._mem_to_int("")

    def test_invalid_suffix_raises(self):
        """Invalid memory suffix raises ValueError."""
        from spikelab.spike_sorting.waveform_extractor import Utils

        with pytest.raises(ValueError, match="Invalid memory suffix"):
            Utils._mem_to_int("4T")


# ===========================================================================
# Edge Case Tests -- Utils.ensure_chunk_size
# ===========================================================================


class TestEnsureChunkSize:
    """Edge cases for Utils.ensure_chunk_size."""

    def test_both_chunk_memory_and_total_memory_raises(self):
        """Providing both chunk_memory and total_memory raises ValueError."""
        from spikelab.spike_sorting.waveform_extractor import Utils

        recording = _make_mock_recording()
        with pytest.raises(ValueError, match="Cannot specify both"):
            Utils.ensure_chunk_size(recording, chunk_memory="500M", total_memory="16G")

    def test_n_jobs_greater_than_1_no_memory_raises(self):
        """n_jobs > 1 without any memory specification raises ValueError."""
        from spikelab.spike_sorting.waveform_extractor import Utils

        recording = _make_mock_recording()
        with pytest.raises(ValueError, match="TOTAL_MEMORY"):
            Utils.ensure_chunk_size(recording, n_jobs=4)


# ===========================================================================
# Edge Case Tests -- Utils.ensure_n_jobs
# ===========================================================================


class TestEnsureNJobs:
    """Edge cases for Utils.ensure_n_jobs."""

    def test_n_jobs_zero_becomes_one(self):
        """n_jobs=0 is treated as 1."""
        from spikelab.spike_sorting.waveform_extractor import Utils

        recording = _make_mock_recording()
        result = Utils.ensure_n_jobs(recording, n_jobs=0)
        assert result == 1

    def test_n_jobs_negative_one_returns_cpu_count(self):
        """n_jobs=-1 returns the number of CPUs."""
        from spikelab.spike_sorting.waveform_extractor import Utils

        recording = _make_mock_recording()
        result = Utils.ensure_n_jobs(recording, n_jobs=-1)
        assert result >= 1

    def test_n_jobs_none_becomes_one(self):
        """n_jobs=None is treated as 1."""
        from spikelab.spike_sorting.waveform_extractor import Utils

        recording = _make_mock_recording()
        result = Utils.ensure_n_jobs(recording, n_jobs=None)
        assert result == 1


# ===========================================================================
# Edge Case Tests -- classify_polarity (waveform_utils)
# ===========================================================================


class TestClassifyPolarity:
    """Edge cases for classify_polarity in waveform_utils."""

    def test_all_zero_templates(self):
        """All-zero templates are classified as positive-peak (edge case)."""
        from spikelab.spike_sorting.waveform_utils import classify_polarity

        templates = np.zeros((3, 61, 4), dtype=np.float32)
        result = classify_polarity(templates, pos_peak_thresh=2.0)
        # 0 >= 2.0 * |0| -> 0 >= 0 -> True
        assert result.shape == (3,)
        assert np.all(result), "All-zero templates should classify as positive-peak"

    def test_single_unit_template(self):
        """Single-unit template array works correctly."""
        from spikelab.spike_sorting.waveform_utils import classify_polarity

        templates = np.zeros((1, 61, 4), dtype=np.float32)
        templates[0, 30, 0] = -10.0
        result = classify_polarity(templates, pos_peak_thresh=2.0)
        assert result.shape == (1,)
        assert not result[0], "Negative-peak unit should be classified as such"

    def test_nan_in_templates(self):
        """NaN in templates: comparison with NaN is False, defaults to
        negative-peak classification."""
        from spikelab.spike_sorting.waveform_utils import classify_polarity

        templates = np.zeros((1, 61, 4), dtype=np.float32)
        templates[0, 30, 0] = np.nan
        result = classify_polarity(templates, pos_peak_thresh=2.0)
        # np.max with nan -> nan, nan >= thresh * |nan| -> False
        assert result.shape == (1,)
        assert not result[0], "NaN templates should default to negative-peak"

    def test_pos_peak_thresh_zero(self):
        """pos_peak_thresh=0 means all non-negative templates classify
        as positive-peak."""
        from spikelab.spike_sorting.waveform_utils import classify_polarity

        templates = np.zeros((2, 61, 4), dtype=np.float32)
        templates[0, 30, 0] = -5.0  # neg only: pos_val=0, neg_val=-5
        templates[1, 30, 0] = 0.1  # tiny positive
        result = classify_polarity(templates, pos_peak_thresh=0.0)
        # unit 0: 0 >= 0 * 5 -> True (even though it's a negative-peak unit)
        assert result[0]
        assert result[1]


# ===========================================================================
# Edge Case Tests -- get_max_channels (waveform_utils)
# ===========================================================================


class TestGetMaxChannels:
    """Edge cases for get_max_channels in waveform_utils."""

    def test_tied_peak_values(self):
        """When two channels have identical peak values, argmin/argmax
        picks the first one -- verify the function returns a valid index."""
        from spikelab.spike_sorting.waveform_utils import (
            classify_polarity,
            get_max_channels,
        )

        templates = np.zeros((1, 61, 4), dtype=np.float32)
        # Two channels with the same negative peak
        templates[0, 30, 0] = -10.0
        templates[0, 30, 2] = -10.0
        use_pos = classify_polarity(templates)
        chans = get_max_channels(templates, use_pos)
        assert chans[0] in (0, 2), "Should pick one of the tied channels"

    def test_single_channel_templates(self):
        """Single-channel templates always return channel 0."""
        from spikelab.spike_sorting.waveform_utils import (
            classify_polarity,
            get_max_channels,
        )

        templates = np.zeros((2, 61, 1), dtype=np.float32)
        templates[0, 30, 0] = -5.0
        templates[1, 30, 0] = 5.0
        use_pos = classify_polarity(templates)
        chans = get_max_channels(templates, use_pos)
        assert np.all(chans == 0)


# ===========================================================================
# Edge Case Tests -- compute_half_window_sizes (waveform_utils)
# ===========================================================================


class TestComputeHalfWindowSizes:
    """Edge cases for compute_half_window_sizes in waveform_utils."""

    def test_very_short_template(self):
        """A 2-sample template produces a valid (small) window size."""
        from spikelab.spike_sorting.waveform_utils import (
            classify_polarity,
            get_max_channels,
            compute_half_window_sizes,
        )

        templates = np.zeros((1, 2, 1), dtype=np.float32)
        templates[0, 1, 0] = -5.0
        use_pos = classify_polarity(templates)
        chans = get_max_channels(templates, use_pos)
        hw = compute_half_window_sizes(templates, chans)
        assert hw.shape == (1,)
        assert hw[0] >= 0


# ===========================================================================
# Edge Case Tests -- center_spike_times (waveform_utils)
# ===========================================================================




# ===========================================================================
# Edge Case Tests -- ChunkRecordingExecutor.divide_segment_into_chunks
# ===========================================================================


class TestDivideSegmentIntoChunks:
    """Edge cases for ChunkRecordingExecutor.divide_segment_into_chunks."""

    def test_chunk_size_none_returns_full_segment(self):
        """chunk_size=None returns the full segment as a single chunk."""
        from spikelab.spike_sorting.waveform_extractor import (
            ChunkRecordingExecutor,
        )

        chunks = ChunkRecordingExecutor.divide_segment_into_chunks(1000, None)
        assert chunks == [(0, 1000)]

    def test_chunk_size_larger_than_frames(self):
        """chunk_size > num_frames: produces one chunk with a remainder."""
        from spikelab.spike_sorting.waveform_extractor import (
            ChunkRecordingExecutor,
        )

        chunks = ChunkRecordingExecutor.divide_segment_into_chunks(100, 500)
        # 100 // 500 = 0 full chunks, remainder = 100
        assert len(chunks) == 1
        assert chunks[0] == (0, 100)

    def test_chunk_size_zero_raises(self):
        """chunk_size=0 causes division by zero."""
        from spikelab.spike_sorting.waveform_extractor import (
            ChunkRecordingExecutor,
        )

        with pytest.raises((ZeroDivisionError, ValueError)):
            ChunkRecordingExecutor.divide_segment_into_chunks(1000, 0)


# ===========================================================================
# Edge Case Tests -- _get_noise_levels (pipeline.py)
# ===========================================================================


class TestGetNoiseLevels:
    """Edge cases for _get_noise_levels in pipeline.py."""

    def test_short_recording_raises(self):
        """Recording shorter than chunk_size raises ValueError
        from rng.randint(0, negative)."""
        from spikelab.spike_sorting.pipeline import _get_noise_levels

        # Recording with only 100 samples, chunk_size default is 10000
        recording = _make_mock_recording(num_samples=100)
        with pytest.raises(ValueError):
            _get_noise_levels(recording)

    def test_single_channel_recording(self):
        """Single-channel recording produces a 1-element noise array."""
        from spikelab.spike_sorting.pipeline import _get_noise_levels

        recording = _make_mock_recording(num_samples=50000, num_channels=1)
        noise = _get_noise_levels(recording)
        assert noise.shape == (1,)
        assert np.isfinite(noise[0])


# ===========================================================================
# Edge Case Tests -- get_paths
# ===========================================================================


class TestGetPaths:
    """Edge cases for get_paths in sorting_utils."""

    def test_results_path_equals_inter_path(self, tmp_path):
        """When results_path == inter_path, a 'results' subdirectory
        is appended."""
        from spikelab.spike_sorting.sorting_utils import get_paths

        rec = tmp_path / "recording.h5"
        rec.touch()
        inter = tmp_path / "inter"
        result = get_paths(rec, inter, inter)
        # Last element is results_path
        assert result[-1] == inter / "results"

    def test_execution_config_none(self, tmp_path):
        """execution_config=None skips all recompute logic without error."""
        from spikelab.spike_sorting.sorting_utils import get_paths

        rec = tmp_path / "recording.h5"
        rec.touch()
        inter = tmp_path / "inter"
        results = tmp_path / "results"
        paths = get_paths(rec, inter, results, execution_config=None)
        assert len(paths) == 9


# ===========================================================================
# Edge Case Tests -- WaveformExtractor.get_computed_template
# ===========================================================================


class TestGetComputedTemplate:
    """Edge cases for WaveformExtractor.get_computed_template."""

    def test_invalid_mode_raises(self):
        """Invalid mode string raises ValueError."""
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        # Create a minimal mock
        we = object.__new__(WaveformExtractor)
        we.template_cache = {}
        we._waveforms = {}

        with pytest.raises(ValueError, match="mode must be one of"):
            we.get_computed_template(0, "invalid_mode")


# ===========================================================================
# Edge Case Tests -- Classifier (_classifier.py)
# ===========================================================================


class TestClassifier:
    """Edge cases for failure classifiers in _classifier.py."""

    def test_hdf5_marker_in_log_not_chain(self):
        """HDF5 marker present in log_text but not in chain_text
        should still detect the issue."""
        from spikelab.spike_sorting._classifier import (
            _classify_hdf5_plugin_missing,
        )

        result = _classify_hdf5_plugin_missing(
            chain_text="some generic error",
            log_text="HDF5_PLUGIN_PATH is not set, filter decoding failed",
        )
        assert result is not None

    def test_insufficient_activity_ks2_cuda_marker_no_metrics(self):
        """Log with CUDA marker but no parseable metrics returns None."""
        from spikelab.spike_sorting._classifier import (
            _classify_insufficient_activity_ks2,
        )

        log = "something something invalid configuration argument something"
        # No threshold crossings, no template-optimization lines
        exc = RuntimeError("CUDA error")
        result = _classify_insufficient_activity_ks2(log, None, exc)
        # No metrics -> none of the conditions are met -> returns None
        assert result is None

    def test_insufficient_activity_ks2_multiple_template_lines(self):
        """When multiple template-optimization lines exist, the last
        one is used."""
        from spikelab.spike_sorting._classifier import (
            _classify_insufficient_activity_ks2,
        )

        log = (
            "invalid configuration argument\n"
            "found 100 threshold crossings\n"
            "1/10 batches, 50 units, nspks: 100.0\n"
            "5/10 batches, 3 units, nspks: 2.0\n"  # last line
        )
        exc = RuntimeError("CUDA crash")
        result = _classify_insufficient_activity_ks2(log, None, exc)
        assert result is not None
        assert result.units_at_failure == 3
        assert result.nspks_at_failure == 2.0

    def test_classify_ks2_failure_output_folder_not_exist(self, tmp_path):
        """classify_ks2_failure with non-existent output folder does not
        crash (returns None when no signatures match)."""
        from spikelab.spike_sorting._classifier import classify_ks2_failure

        result = classify_ks2_failure(
            tmp_path / "nonexistent",
            RuntimeError("generic error"),
        )
        assert result is None

    def test_classify_ks4_failure_output_folder_not_exist(self, tmp_path):
        """classify_ks4_failure with non-existent output folder does not
        crash (returns None when no signatures match)."""
        from spikelab.spike_sorting._classifier import classify_ks4_failure

        result = classify_ks4_failure(
            tmp_path / "nonexistent",
            RuntimeError("generic error"),
        )
        assert result is None


# ===========================================================================
# Tests -- classify_rt_sort_failure
# ===========================================================================


class TestClassifyRTSortFailure:
    """Tests for the RT-Sort failure classifier."""

    def test_no_match_returns_none(self, tmp_path):
        """Unrecognised exception returns None."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure

        result = classify_rt_sort_failure(
            tmp_path, RuntimeError("something unexpected")
        )
        assert result is None

    def test_nonexistent_folder_returns_none(self, tmp_path):
        """Non-existent output folder does not crash."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure

        result = classify_rt_sort_failure(
            tmp_path / "nonexistent", RuntimeError("generic error")
        )
        assert result is None

    # -- Environment: model loading --

    def test_torch_missing(self, tmp_path):
        """PyTorch ImportError is classified as ModelLoadingError."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import ModelLoadingError

        exc = ImportError("PyTorch is required for RT-Sort's spike detection model")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, ModelLoadingError)
        assert result.sorter == "rt_sort"

    def test_torch_module_not_found(self, tmp_path):
        """ModuleNotFoundError for torch is classified as ModelLoadingError."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import ModelLoadingError

        exc = ModuleNotFoundError("No module named 'torch'")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, ModelLoadingError)

    def test_model_files_missing(self, tmp_path):
        """Missing init_dict.json / state_dict.pt is classified as
        ModelLoadingError with model_path extracted."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import ModelLoadingError

        exc = ValueError(
            "The folder /models/mea does not contain init_dict.json and "
            "state_dict.pt for loading a model"
        )
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, ModelLoadingError)
        assert result.model_path == "/models/mea"

    def test_state_dict_load_error(self, tmp_path):
        """Corrupt state_dict is classified as ModelLoadingError."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import ModelLoadingError

        exc = RuntimeError("Error(s) in loading state_dict for ModelSpikeSorter")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, ModelLoadingError)

    # -- Environment: HDF5 --

    def test_hdf5_plugin_missing(self, tmp_path):
        """HDF5 plugin marker in chain is classified as HDF5PluginMissingError."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import HDF5PluginMissingError

        exc = OSError("HDF5_PLUGIN_PATH filter decoding failed")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, HDF5PluginMissingError)

    # -- Resource: GPU OOM --

    def test_gpu_oom(self, tmp_path):
        """CUDA OOM is classified as GPUOutOfMemoryError."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import GPUOutOfMemoryError

        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, GPUOutOfMemoryError)
        assert result.sorter == "rt_sort"

    # -- Biology: insufficient activity --

    def test_zero_sequences_in_log(self, tmp_path):
        """Log text with '0 preliminary propagation sequences' triggers
        InsufficientActivityError."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import InsufficientActivityError

        # Write a log file with the zero-sequences message
        log_file = tmp_path / "rt_sort.log"
        log_file.write_text(
            "0 preliminary propagation sequences remain after "
            "reassigning spikes and filtering\n"
        )
        exc = AttributeError("'NoneType' object has no attribute 'sort_offline'")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, InsufficientActivityError)
        assert result.sorter == "rt_sort"
        assert result.log_path == log_file

    def test_nonetype_sort_offline(self, tmp_path):
        """AttributeError from calling sort_offline on None is classified
        as InsufficientActivityError (chain-based, no log)."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import InsufficientActivityError

        exc = AttributeError("'NoneType' object has no attribute 'sort_offline'")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, InsufficientActivityError)
        assert result.log_path is None

    def test_zero_sequences_after_merging(self, tmp_path):
        """'0 sequences remain' in log triggers InsufficientActivityError."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import InsufficientActivityError

        log_file = tmp_path / "rt_sort.log"
        log_file.write_text("0 sequences remain first merging\n")
        exc = AttributeError("'NoneType' object has no attribute 'sort_offline'")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, InsufficientActivityError)

    # -- Priority ordering --

    def test_model_error_takes_priority_over_oom(self, tmp_path):
        """Environment errors (model loading) take priority over resource
        errors (GPU OOM) when both signatures are present."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import ModelLoadingError

        # Chain contains both model loading and OOM signatures
        inner = RuntimeError("CUDA out of memory")
        exc = ValueError(
            "The folder /m does not contain init_dict.json and state_dict.pt"
        )
        exc.__cause__ = inner
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, ModelLoadingError)

    def test_oom_takes_priority_over_biology(self, tmp_path):
        """Resource errors (GPU OOM) take priority over biology errors."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import GPUOutOfMemoryError

        log_file = tmp_path / "rt_sort.log"
        log_file.write_text("0 sequences remain second merging\n")
        exc = RuntimeError("CUDA out of memory")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, GPUOutOfMemoryError)

    # -- Exception hierarchy --

    def test_model_loading_error_is_environment_failure(self):
        """ModelLoadingError is a subclass of EnvironmentSortFailure."""
        from spikelab.spike_sorting._exceptions import (
            EnvironmentSortFailure,
            ModelLoadingError,
        )

        assert issubclass(ModelLoadingError, EnvironmentSortFailure)

    def test_model_loading_error_attributes(self):
        """ModelLoadingError stores sorter and model_path."""
        from spikelab.spike_sorting._exceptions import ModelLoadingError

        err = ModelLoadingError("msg", sorter="rt_sort", model_path="/models/mea")
        assert err.sorter == "rt_sort"
        assert err.model_path == "/models/mea"
        assert str(err) == "msg"

    def test_model_loading_error_default_attributes(self):
        """ModelLoadingError defaults: sorter='rt_sort', model_path=None."""
        from spikelab.spike_sorting._exceptions import ModelLoadingError

        err = ModelLoadingError("msg")
        assert err.sorter == "rt_sort"
        assert err.model_path is None

    # -- Log file discovery --

    def test_log_file_found(self, tmp_path):
        """_find_rt_sort_log locates rt_sort.log in the output folder."""
        from spikelab.spike_sorting._classifier import _find_rt_sort_log

        log_file = tmp_path / "rt_sort.log"
        log_file.write_text("test log content")
        assert _find_rt_sort_log(tmp_path) == log_file

    def test_log_file_not_found(self, tmp_path):
        """_find_rt_sort_log returns None when no log exists."""
        from spikelab.spike_sorting._classifier import _find_rt_sort_log

        assert _find_rt_sort_log(tmp_path) is None

    def test_classifier_reads_log_for_signatures(self, tmp_path):
        """Classifier detects a signature present only in the log file,
        not in the exception chain."""
        from spikelab.spike_sorting._classifier import classify_rt_sort_failure
        from spikelab.spike_sorting._exceptions import InsufficientActivityError

        log_file = tmp_path / "rt_sort.log"
        log_file.write_text(
            "0 preliminary propagation sequences remain after filtering\n"
        )
        # Exception chain is generic — signature is only in the log
        exc = RuntimeError("sort failed")
        result = classify_rt_sort_failure(tmp_path, exc)
        assert isinstance(result, InsufficientActivityError)


# ===========================================================================
# Edge Case Tests -- Recentering (stim_sorting/recentering.py)
# ===========================================================================


class TestRecentering:
    """Edge cases for stim_sorting recentering functions."""

    def test_max_offset_zero(self):
        """max_offset_ms=0 produces a 1-sample search window; should
        return the center sample itself."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        fs_Hz = 20000.0
        traces = np.random.default_rng(0).standard_normal((4, 10000)).astype(np.float32)
        stim_ms = [100.0]
        result = recenter_stim_times(traces, stim_ms, fs_Hz, max_offset_ms=0.0)
        # The search window is [center, center+1), so it picks the center
        center_sample = int(np.round(100.0 * fs_Hz / 1000.0))
        result_sample = int(np.round(result[0] * fs_Hz / 1000.0))
        assert result_sample == center_sample

    def test_stim_at_recording_boundary(self):
        """Stim time at the very end of the recording clips the search
        window and does not crash."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            recenter_stim_times,
        )

        fs_Hz = 20000.0
        n_samples = 10000
        traces = (
            np.random.default_rng(0).standard_normal((4, n_samples)).astype(np.float32)
        )
        # Stim at the very last sample
        stim_ms = [(n_samples - 1) / fs_Hz * 1000.0]
        result = recenter_stim_times(traces, stim_ms, fs_Hz, max_offset_ms=5.0)
        assert len(result) == 1
        result_sample = int(np.round(result[0] * fs_Hz / 1000.0))
        assert 0 <= result_sample < n_samples

    def test_build_reference_trace_n_ref_larger_than_channels(self):
        """n_reference_channels larger than total channels is clamped."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _build_reference_trace,
        )

        traces = np.random.default_rng(0).standard_normal((3, 100)).astype(np.float32)
        ref = _build_reference_trace(traces, n_reference_channels=100)
        assert ref.shape == (100,)

    def test_build_reference_trace_n_ref_zero(self):
        """n_reference_channels=0 is clamped to 1."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _build_reference_trace,
        )

        traces = np.random.default_rng(0).standard_normal((3, 100)).astype(np.float32)
        ref = _build_reference_trace(traces, n_reference_channels=0)
        assert ref.shape == (100,)

    def test_find_down_edge_constant_signal(self):
        """Constant-value signal: no zero crossing, falls back to
        steepest slope or neg_peak."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_down_edge,
        )

        reference = np.ones(100, dtype=np.float64) * 5.0
        result = _find_down_edge(
            reference, lo=10, hi=50, prewindow_ms=1.0, fs_Hz=20000.0
        )
        assert 10 <= result < 50

    def test_find_down_edge_lo_equals_hi(self):
        """lo == hi produces an empty window; argmin on empty array
        should be handled gracefully or raise ValueError."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_down_edge,
        )

        reference = np.ones(100, dtype=np.float64)
        # This may raise due to empty slice
        with pytest.raises((ValueError, IndexError)):
            _find_down_edge(reference, lo=50, hi=50, prewindow_ms=1.0, fs_Hz=20000.0)

    def test_find_up_edge_constant_signal(self):
        """Constant-value signal for up_edge."""
        from spikelab.spike_sorting.stim_sorting.recentering import (
            _find_up_edge,
        )

        reference = np.ones(100, dtype=np.float64) * 5.0
        result = _find_up_edge(reference, lo=10, hi=50, prewindow_ms=1.0, fs_Hz=20000.0)
        assert 10 <= result < 50


# ===========================================================================
# Edge Case Tests -- Artifact Removal (stim_sorting/artifact_removal.py)
# ===========================================================================


class TestArtifactRemoval:
    """Edge cases for stim_sorting artifact removal functions."""

    def test_zero_channels(self):
        """traces with shape (0, n_samples) raises ValueError.

        Tests:
            (Test Case 1) Zero-channel traces are rejected with a clear message.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        traces = np.zeros((0, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="at least one channel"):
            remove_stim_artifacts(traces, [100.0], fs_Hz=20000.0)

    def test_zero_samples(self):
        """traces with shape (n_channels, 0) raises ValueError.

        Tests:
            (Test Case 1) Zero-sample traces are rejected with a clear message.
        """
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        traces = np.zeros((4, 0), dtype=np.float32)
        with pytest.raises(ValueError, match="at least one channel"):
            remove_stim_artifacts(traces, [100.0], fs_Hz=20000.0)

    def test_artifact_window_zero(self):
        """artifact_window_ms=0 produces 0-sample windows; should not crash."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        traces = np.random.default_rng(0).standard_normal((2, 1000)).astype(np.float32)
        cleaned, blanked = remove_stim_artifacts(
            traces,
            [10.0],
            fs_Hz=20000.0,
            artifact_window_ms=0.0,
            saturation_threshold=1e6,
        )
        assert cleaned.shape == (2, 1000)

    def test_stim_times_outside_recording(self):
        """Stim times outside the recording range are filtered out."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        traces = np.random.default_rng(0).standard_normal((2, 1000)).astype(np.float32)
        original = traces.copy()
        cleaned, blanked = remove_stim_artifacts(
            traces,
            [-100.0, 999999.0],
            fs_Hz=20000.0,
            saturation_threshold=1e6,
        )
        # No valid stim times -> traces unchanged
        np.testing.assert_array_equal(cleaned, original)

    def test_artifact_window_only_false_with_blank(self):
        """artifact_window_only=False with method='blank' blanks only
        saturated samples globally."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        traces = np.ones((1, 100), dtype=np.float32) * 5.0
        # Set threshold so everything is "saturated"
        cleaned, blanked = remove_stim_artifacts(
            traces,
            [1.0],
            fs_Hz=20000.0,
            method="blank",
            artifact_window_only=False,
            saturation_threshold=3.0,
        )
        assert np.all(blanked)
        assert np.all(cleaned == 0.0)

    def test_auto_saturation_threshold_all_zeros(self):
        """All-zero traces: threshold is 0.0, blanking covers entire
        recording (edge case)."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _auto_saturation_threshold,
        )

        traces = np.zeros((2, 100), dtype=np.float32)
        thresh = _auto_saturation_threshold(traces)
        assert thresh == 0.0

    def test_auto_baseline_threshold_empty_stim_times(self):
        """Empty stim_times_ms: falls back to the first 2 ms of recording."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _auto_baseline_threshold,
        )

        traces = np.random.default_rng(0).standard_normal((2, 1000)).astype(np.float32)
        thresh = _auto_baseline_threshold(traces, np.array([]), 20000.0)
        assert np.isfinite(thresh)

    def test_auto_baseline_threshold_first_stim_at_zero(self):
        """First stim at time 0: pre-stim segment has 0 samples,
        but code clamps end to max(1, 0) = 1."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _auto_baseline_threshold,
        )

        traces = np.random.default_rng(0).standard_normal((2, 1000)).astype(np.float32)
        thresh = _auto_baseline_threshold(traces, np.array([0.0]), 20000.0)
        assert np.isfinite(thresh)

    def test_find_saturation_end_start_past_end(self):
        """start >= n_samples: the while loop never executes so the function
        returns ``start`` unchanged (not clamped to n_samples)."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _find_saturation_end,
        )

        trace = np.array([1.0, 2.0, 3.0])
        result = _find_saturation_end(
            trace, start=10, saturation_threshold=1.0, n_samples=3
        )
        assert result == 10

    def test_signal_reached_baseline_window_zero(self):
        """window_samples=0: the consecutive count can only reach 0 when a
        sample is actually below threshold. With all values above threshold
        the function returns False because the increment branch is never
        entered."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _signal_reached_baseline,
        )

        trace = np.array([100.0, 100.0, 100.0])
        reached, idx = _signal_reached_baseline(
            trace,
            start=0,
            baseline_threshold=1.0,
            window_samples=0,
            n_samples=3,
        )
        assert not reached

    def test_signal_reached_baseline_start_past_end(self):
        """start >= n_samples returns False immediately."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _signal_reached_baseline,
        )

        trace = np.array([1.0, 2.0])
        reached, idx = _signal_reached_baseline(
            trace,
            start=10,
            baseline_threshold=100.0,
            window_samples=1,
            n_samples=2,
        )
        assert not reached
        assert idx == 2

    def test_saturation_threshold_from_recording_zero_gains(self):
        """Recording with zero gains: gain is clamped to 1e-9."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _saturation_threshold_from_recording,
        )

        recording = SimpleNamespace()
        recording.get_channel_gains = lambda: np.array([0.0, 0.0])
        traces = np.ones((2, 100), dtype=np.float32) * 50.0
        thresh = _saturation_threshold_from_recording(recording, traces)
        # gain_uV_per_bit = max(1e-9, 0.0) = 1e-9; many samples at rail
        assert np.isfinite(thresh)

    def test_poly_order_larger_than_samples(self):
        """poly_order larger than available samples: polynomial fit is
        skipped (not enough points for the fit)."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            remove_stim_artifacts,
        )

        # Very short trace with a single stim event
        fs_Hz = 20000.0
        traces = np.random.default_rng(0).standard_normal((1, 50)).astype(np.float32)
        cleaned, blanked = remove_stim_artifacts(
            traces,
            [0.5],
            fs_Hz=fs_Hz,
            poly_order=20,
            saturation_threshold=1e6,
        )
        assert cleaned.shape == (1, 50)


# ===========================================================================
# Edge Case Tests -- KilosortSortingExtractor
# ===========================================================================


@skip_no_pandas


# ===========================================================================
# Edge Case Tests -- Figures
# ===========================================================================


class TestFigures:
    """Edge cases for spike sorting figure functions."""

    def test_plot_curation_bar_empty(self):
        """Empty rec_names list should not crash."""
        from spikelab.spike_sorting.figures import plot_curation_bar

        fig = plot_curation_bar([], [], [])
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_curation_bar_mismatched_lengths(self):
        """Mismatched lengths of input lists should raise or degrade
        gracefully."""
        from spikelab.spike_sorting.figures import plot_curation_bar

        # Matplotlib may raise or just plot mismatched bars
        try:
            fig = plot_curation_bar(["A", "B"], [10], [5, 3])
            import matplotlib.pyplot as plt

            plt.close(fig)
        except (ValueError, IndexError):
            pass  # Expected for mismatched lengths


# ===========================================================================
# Edge Case Tests -- _global_polynomial_detrend
# ===========================================================================


class TestGlobalPolynomialDetrend:
    """Edge cases for _global_polynomial_detrend."""

    def test_overlap_samples_equals_window_samples(self):
        """overlap_samples >= window_samples: step is clamped to 1."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _global_polynomial_detrend,
        )

        trace = np.random.default_rng(0).standard_normal(100).astype(np.float64)
        blanked = np.zeros((1, 100), dtype=bool)
        # overlap == window -> step = max(1, 0) = 1
        _global_polynomial_detrend(
            trace,
            window_samples=10,
            overlap_samples=10,
            saturation_threshold=1e6,
            poly_order=3,
            n_samples=100,
            blanked=blanked,
            ch_idx=0,
        )
        # Should complete without error

    def test_window_samples_zero(self):
        """window_samples=0: empty windows -- step is clamped to 1,
        each segment has 0 samples."""
        from spikelab.spike_sorting.stim_sorting.artifact_removal import (
            _global_polynomial_detrend,
        )

        trace = np.random.default_rng(0).standard_normal(50).astype(np.float64)
        blanked = np.zeros((1, 50), dtype=bool)
        _global_polynomial_detrend(
            trace,
            window_samples=0,
            overlap_samples=0,
            saturation_threshold=1e6,
            poly_order=3,
            n_samples=50,
            blanked=blanked,
            ch_idx=0,
        )
        # Should complete without crash
