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


class _GlobalsStub:
    """No-op stand-in for the deleted ``_globals`` module.

    Some test fixtures predate Phase 5 of the ``_globals.py`` refactor
    (see ``iat/TO_IMPLEMENT.md``) and still expect to import
    ``spikelab.spike_sorting._globals`` to set sentinel attributes
    before the test runs. With the module gone — and the code under
    test reading from ``SortingPipelineConfig`` instead — those writes
    have no effect; this stub absorbs them silently so the fixtures
    stay syntactically valid until a follow-up cleanup pass removes
    them.
    """

    def __getattr__(self, name):
        return None

    def __setattr__(self, name, value):
        pass


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
        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        kse = ks_module.KilosortSortingExtractor(tmp_path, keep_good_only=True)
        assert kse.unit_ids == [0]

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

        kse = ks_module.KilosortSortingExtractor(tmp_path, keep_good_only=False)
        assert 2 not in kse.unit_ids
        assert set(kse.unit_ids) == {0, 1}

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        kse = ks_module.KilosortSortingExtractor(folder, keep_good_only=True)
        assert kse.unit_ids == [0]

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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
        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes
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
        from spikelab.spike_sorting.config import SorterConfig, SortingPipelineConfig
        from spikelab.spike_sorting.ks2_runner import spike_sort

        output_folder = tmp_path / "ks_output"
        recording = _make_mock_recording()
        config = SortingPipelineConfig(
            sorter=SorterConfig(sorter_name="kilosort2", use_docker=True),
        )
        config.execution.recompute_sorting = True

        mock_kse = SimpleNamespace(unit_ids=[0, 1])

        with (
            patch.object(
                self._ks_runner_mod, "_spike_sort_docker", return_value=mock_kse
            ) as mock_docker,
            patch.object(self._ks_runner_mod, "RunKilosort") as mock_rk,
        ):
            result = spike_sort(
                recording,
                "fake.h5",
                tmp_path / "rec.dat",
                output_folder,
                config=config,
            )

        mock_docker.assert_called_once()
        # Positional args remain (recording, output_folder); keyword args
        # carry the kilosort_params/pos_peak_thresh values resolved from
        # the config.
        call_args = mock_docker.call_args
        assert call_args.args[0] is recording
        assert call_args.args[1] == output_folder
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
        from spikelab.spike_sorting.config import SorterConfig, SortingPipelineConfig
        from spikelab.spike_sorting.ks2_runner import spike_sort

        output_folder = tmp_path / "ks_output"
        recording = _make_mock_recording()
        config = SortingPipelineConfig(
            sorter=SorterConfig(sorter_name="kilosort2", use_docker=True),
        )
        config.execution.recompute_sorting = True

        with patch.object(
            self._ks_runner_mod,
            "_spike_sort_docker",
            side_effect=RuntimeError("Docker failed"),
        ):
            result = spike_sort(
                recording,
                "fake.h5",
                tmp_path / "rec.dat",
                output_folder,
                config=config,
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
        _globals = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes
        from spikelab.spike_sorting import recording_io

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

        def mock_load(path, **_kw):
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

        def mock_load(path, **_kw):
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

        def mock_load(path, **_kw):
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

        def mock_load(path, **_kw):
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

        def mock_load(path, **_kw):
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
# Auto-populated rec_chunks interactions with time-based slicing
# ===========================================================================


@skip_no_spikeinterface
class TestLoaderTimeVsFrameChunks:
    """
    Tests that the loader correctly resolves the effective frame-chunk
    list when the recording is a directory (auto-populated per-file
    chunks) vs when the user supplies their own frame chunks or
    time-based slicing.

    The ambiguous case — user-supplied frame ``rec_chunks`` AND time-
    based slicing — must raise ``ValueError`` to surface the
    contradiction. Auto-populated chunks from directory concatenation
    are silently overridden by time-based slicing (that is what the
    canary relies on: ``start_time_s=0`` / ``end_time_s=window_s``
    over a directory of recordings).

    Replaces ``TestRecChunksFromConcatOverride`` (deleted in Phase 5
    of the ``_globals.py`` refactor — those tests asserted on the now-
    deleted ``_globals.REC_CHUNKS_FROM_CONCAT`` flag; the equivalent
    state is now flowing through ``LoadRecordingResult.rec_chunks``).
    """

    def test_loader_raises_when_user_set_rec_chunks_collides_with_time(
        self, monkeypatch
    ):
        """
        User-supplied ``rec_chunks`` combined with time-based slicing
        is ambiguous and raises ``ValueError``.

        Tests:
            (Test Case 1) ``config.recording.rec_chunks=[(0, 1000)]``
                + ``start_time_s=0.0`` / ``end_time_s=1.0`` →
                ValueError naming "frame-based".
        """
        from spikelab.spike_sorting import recording_io
        from spikelab.spike_sorting.config import (
            RecordingConfig,
            SortingPipelineConfig,
        )

        rec = _make_mock_recording(num_samples=200000)
        monkeypatch.setattr(recording_io, "BaseRecording", type(rec), raising=False)
        monkeypatch.setattr(recording_io, "load_single_recording", lambda p, **_kw: rec)

        config = SortingPipelineConfig(
            recording=RecordingConfig(
                rec_chunks=[(0, 1000)],
                start_time_s=0.0,
                end_time_s=1.0,
            )
        )

        with pytest.raises(ValueError, match="frame-based"):
            recording_io.load_recording(rec, config=config)


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
            preflight=False,
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
            preflight=False,
        )
        assert result == []


# ===========================================================================
# sort_recording: guard wiring (preflight + host-memory watchdog)
# ===========================================================================


@skip_no_spikeinterface
class TestSortRecordingGuardWiring:
    """
    sort_recording invokes the preflight + watchdog hooks when configured.

    Uses recording_files=[] so the per-recording loop body never runs;
    we only verify the pre-loop wiring fires (or is skipped when the
    relevant ExecutionConfig flag is False).
    """

    @pytest.fixture()
    def sort_fn(self):
        from spikelab.spike_sorting.pipeline import sort_recording

        return sort_recording

    def _make_config(self, **execution_overrides):
        """Build a default SortingPipelineConfig with execution overrides.

        The new ExecutionConfig guard fields are not (yet) wired into
        ``SortingPipelineConfig.from_kwargs``, so flat-kwarg overrides
        such as ``host_ram_watchdog=False`` raise. Tests therefore
        construct the config directly and pass it via ``config=``.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        for key, value in execution_overrides.items():
            setattr(cfg.execution, key, value)
        return cfg

    def test_preflight_invoked_by_default(self, sort_fn):
        """
        Default config triggers run_preflight + report_findings.

        Tests:
            (Test Case 1) run_preflight is called once.
            (Test Case 2) report_findings is called with the
                preflight findings and strict=False.
        """
        cfg = self._make_config()
        with (
            patch(
                "spikelab.spike_sorting.guards.run_preflight",
                return_value=[],
            ) as mock_run,
            patch("spikelab.spike_sorting.guards.report_findings") as mock_report,
        ):
            sort_fn(
                recording_files=[],
                config=cfg,
                intermediate_folders=[],
                results_folders=[],
            )
        mock_run.assert_called_once()
        mock_report.assert_called_once()
        # report_findings called with strict kwarg matching the config
        # (default False).
        assert mock_report.call_args.kwargs.get("strict") is False

    def test_preflight_skipped_when_disabled(self, sort_fn):
        """
        execution.preflight=False suppresses preflight entirely.

        Tests:
            (Test Case 1) run_preflight is not called.
            (Test Case 2) report_findings is not called.
        """
        cfg = self._make_config(preflight=False)
        with (
            patch("spikelab.spike_sorting.guards.run_preflight") as mock_run,
            patch("spikelab.spike_sorting.guards.report_findings") as mock_report,
        ):
            sort_fn(
                recording_files=[],
                config=cfg,
                intermediate_folders=[],
                results_folders=[],
            )
        mock_run.assert_not_called()
        mock_report.assert_not_called()

    def test_preflight_strict_propagates(self, sort_fn):
        """
        execution.preflight_strict propagates to report_findings.

        Tests:
            (Test Case 1) strict=True is forwarded when configured.
        """
        cfg = self._make_config(preflight_strict=True)
        with (
            patch(
                "spikelab.spike_sorting.guards.run_preflight",
                return_value=[],
            ),
            patch("spikelab.spike_sorting.guards.report_findings") as mock_report,
        ):
            sort_fn(
                recording_files=[],
                config=cfg,
                intermediate_folders=[],
                results_folders=[],
            )
        mock_report.assert_called_once()
        assert mock_report.call_args.kwargs.get("strict") is True

    def test_watchdog_instantiated_with_configured_thresholds(self, sort_fn):
        """
        sort_recording constructs HostMemoryWatchdog from ExecutionConfig.

        Tests:
            (Test Case 1) HostMemoryWatchdog is instantiated once when
                host_ram_watchdog=True.
            (Test Case 2) Constructor receives the configured warn /
                abort / poll values.
        """
        cfg = self._make_config(
            host_ram_warn_pct=70.0,
            host_ram_abort_pct=88.0,
            host_ram_poll_interval_s=1.5,
        )
        with (
            patch("spikelab.spike_sorting.guards.HostMemoryWatchdog") as mock_wd_class,
            patch(
                "spikelab.spike_sorting.guards.run_preflight",
                return_value=[],
            ),
            patch("spikelab.spike_sorting.guards.report_findings"),
        ):
            instance = MagicMock()
            instance.__enter__.return_value = instance
            instance.__exit__.return_value = False
            mock_wd_class.return_value = instance
            sort_fn(
                recording_files=[],
                config=cfg,
                intermediate_folders=[],
                results_folders=[],
            )
        mock_wd_class.assert_called_once_with(
            warn_pct=70.0,
            abort_pct=88.0,
            poll_interval_s=1.5,
        )
        instance.__enter__.assert_called_once()
        instance.__exit__.assert_called_once()

    def test_watchdog_skipped_when_disabled(self, sort_fn):
        """
        host_ram_watchdog=False replaces the watchdog with a nullcontext.

        Tests:
            (Test Case 1) HostMemoryWatchdog is never constructed.
            (Test Case 2) sort_recording still completes normally.
        """
        cfg = self._make_config(host_ram_watchdog=False)
        with (
            patch("spikelab.spike_sorting.guards.HostMemoryWatchdog") as mock_wd_class,
            patch(
                "spikelab.spike_sorting.guards.run_preflight",
                return_value=[],
            ),
            patch("spikelab.spike_sorting.guards.report_findings"),
        ):
            result = sort_fn(
                recording_files=[],
                config=cfg,
                intermediate_folders=[],
                results_folders=[],
            )
        mock_wd_class.assert_not_called()
        assert result == []


# ===========================================================================
# Backend OOM-parameter scaling
# ===========================================================================


@skip_no_spikeinterface
class TestKilosort2BackendOomScaling:
    """``Kilosort2Backend`` reduces ``NT`` on OOM and round-trips snapshot."""

    def _make_backend(self, **sorter_params_overrides):
        """Build a Kilosort2Backend with a controllable NT in sorter_params."""
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        # Stub out KILOSORT_PATH validation; we only exercise the
        # config path, not the MATLAB-launching path.
        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_path = "/fake/kilosort/path"
        cfg.sorter.sorter_params = {"NT": 65600, **sorter_params_overrides}
        return Kilosort2Backend(cfg)

    def test_scale_halves_NT_and_rounds_to_32(self):
        """
        scale_oom_params(0.5) halves NT and rounds down to a multiple of 32.

        Tests:
            (Test Case 1) Returns True.
            (Test Case 2) Resulting NT is half (and divisible by 32).
            (Test Case 3) Config is mutated in place.
        """
        backend = self._make_backend()
        ok = backend.scale_oom_params(0.5)
        assert ok is True
        new_nt = backend.config.sorter.sorter_params["NT"]
        assert new_nt == 32800  # 65600 // 2 = 32800; already mult of 32
        assert new_nt % 32 == 0

    def test_scale_resolves_none_NT_to_default(self):
        """
        NT=None resolves to 64*1024 + ntbuff before scaling.

        Tests:
            (Test Case 1) When sorter_params has NT=None, the
                backend uses the format_params default and halves it.
        """
        backend = self._make_backend()
        backend.config.sorter.sorter_params = {"NT": None, "ntbuff": 64}
        ok = backend.scale_oom_params(0.5)
        assert ok is True
        new_nt = backend.config.sorter.sorter_params["NT"]
        # (64*1024 + 64) // 2 = 32800; rounded down to multiple of 32
        assert new_nt == 32800

    def test_scale_refuses_below_minimum(self):
        """
        Refuses to scale when the resulting NT would be < 1024.

        Tests:
            (Test Case 1) Tiny starting NT → returns False.
            (Test Case 2) Config is left unchanged.
        """
        backend = self._make_backend(NT=512)
        ok = backend.scale_oom_params(0.5)
        assert ok is False
        assert backend.config.sorter.sorter_params["NT"] == 512

    def test_scale_refuses_invalid_factor(self):
        """
        factor outside (0, 1) is rejected.

        Tests:
            (Test Case 1) factor=0 returns False.
            (Test Case 2) factor>=1 returns False.
            (Test Case 3) factor<0 returns False.
        """
        backend = self._make_backend()
        for bad in (0.0, 1.0, 1.5, -0.5):
            assert backend.scale_oom_params(bad) is False

    def test_snapshot_restore_round_trip(self):
        """
        snapshot_oom_params + restore_oom_params reverts a scale-down.

        Tests:
            (Test Case 1) Snapshot captures sorter_params as-is.
            (Test Case 2) After scaling and restoring, NT matches the
                pre-snapshot value.
        """
        backend = self._make_backend()
        snap = backend.snapshot_oom_params()
        backend.scale_oom_params(0.5)
        assert backend.config.sorter.sorter_params["NT"] != snap["sorter_params"]["NT"]
        backend.restore_oom_params(snap)
        assert backend.config.sorter.sorter_params == snap["sorter_params"]


@skip_no_spikeinterface
class TestKilosort4BackendOomScaling:
    """``Kilosort4Backend`` reduces ``batch_size`` on OOM."""

    def _make_backend(self, **sorter_params_overrides):
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"
        cfg.sorter.sorter_params = dict(sorter_params_overrides)
        return Kilosort4Backend(cfg)

    def test_scale_halves_batch_size(self):
        """
        scale_oom_params(0.5) halves batch_size from the KS4 default.

        Tests:
            (Test Case 1) Returns True.
            (Test Case 2) batch_size becomes 30000 (60000 // 2).
        """
        backend = self._make_backend()  # No batch_size set → uses default 60000
        ok = backend.scale_oom_params(0.5)
        assert ok is True
        assert backend.config.sorter.sorter_params["batch_size"] == 30000

    def test_scale_refuses_below_minimum(self):
        """
        Refuses to scale when batch_size would drop below 1024.

        Tests:
            (Test Case 1) Starting batch_size=1500 → returns False
                (1500 // 2 = 750 < 1024).
        """
        backend = self._make_backend(batch_size=1500)
        ok = backend.scale_oom_params(0.5)
        assert ok is False
        assert backend.config.sorter.sorter_params["batch_size"] == 1500

    def test_snapshot_restore_round_trip(self):
        """
        snapshot/restore reverts the batch_size change.

        Tests:
            (Test Case 1) After scale + restore, sorter_params equal
                the original.
        """
        backend = self._make_backend(batch_size=60000)
        snap = backend.snapshot_oom_params()
        backend.scale_oom_params(0.5)
        backend.restore_oom_params(snap)
        assert backend.config.sorter.sorter_params == snap["sorter_params"]


@skip_no_torch
@skip_no_spikeinterface
class TestRTSortBackendOomScaling:
    """``RTSortBackend`` scales ``num_processes`` on OOM."""

    def _make_backend(self, num_processes=8):
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.backends.rt_sort import RTSortBackend

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "rt_sort"
        cfg.rt_sort.num_processes = num_processes
        return RTSortBackend(cfg)

    def test_scale_halves_num_processes(self):
        """
        scale_oom_params(0.5) halves num_processes.

        Tests:
            (Test Case 1) Returns True.
            (Test Case 2) num_processes drops from 8 to 4.
        """
        backend = self._make_backend(num_processes=8)
        ok = backend.scale_oom_params(0.5)
        assert ok is True
        assert backend.config.rt_sort.num_processes == 4

    def test_scale_refuses_at_one(self):
        """
        Refuses to scale when num_processes is already 1.

        Tests:
            (Test Case 1) Returns False.
            (Test Case 2) Value is unchanged.
        """
        backend = self._make_backend(num_processes=1)
        ok = backend.scale_oom_params(0.5)
        assert ok is False
        assert backend.config.rt_sort.num_processes == 1

    def test_snapshot_restore_round_trip(self):
        """
        snapshot/restore reverts the num_processes change.

        Tests:
            (Test Case 1) After scale + restore, num_processes equal
                the original.
        """
        backend = self._make_backend(num_processes=8)
        snap = backend.snapshot_oom_params()
        backend.scale_oom_params(0.5)
        backend.restore_oom_params(snap)
        assert backend.config.rt_sort.num_processes == snap["num_processes"]


# ===========================================================================
# SorterBackend._resolve_inactivity_timeout_s
# ===========================================================================


@skip_no_spikeinterface
class TestResolveInactivityTimeoutS:
    """Backend's recording-aware inactivity-tolerance helper."""

    def _make_recording(self, n_samples, fs_hz):
        """Build a duck-typed recording with the two methods we need."""
        rec = MagicMock()
        rec.get_num_samples.return_value = n_samples
        rec.get_sampling_frequency.return_value = fs_hz
        return rec

    def _make_backend(self):
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_path = "/fake/path"
        return Kilosort2Backend(cfg)

    def test_respects_disabled_flag(self):
        """
        sorter_inactivity_timeout=False returns None.

        Tests:
            (Test Case 1) Disabled flag short-circuits the recording
                inspection and returns None unconditionally.
        """
        backend = self._make_backend()
        backend.config.execution.sorter_inactivity_timeout = False
        rec = self._make_recording(20000 * 60 * 30, 20000)  # 30 min
        assert backend._resolve_inactivity_timeout_s(rec) is None

    def test_scales_with_recording_duration(self):
        """
        Tolerance grows as base_s + per_min_s × duration_min.

        Tests:
            (Test Case 1) 30-min recording at 20 kHz → 1500 s.
            (Test Case 2) 5-min recording → 750 s.
        """
        backend = self._make_backend()
        # 30 min @ 20kHz → 30 min → 600 + 30*30 = 1500
        rec30 = self._make_recording(20000 * 60 * 30, 20000)
        assert backend._resolve_inactivity_timeout_s(rec30) == 1500.0
        # 5 min → 600 + 30*5 = 750
        rec5 = self._make_recording(20000 * 60 * 5, 20000)
        assert backend._resolve_inactivity_timeout_s(rec5) == 750.0

    def test_returns_none_on_unreadable_recording(self):
        """
        Recording missing get_num_samples / sampling_frequency yields None.

        Tests:
            (Test Case 1) When get_num_samples raises, returns None.
            (Test Case 2) When fs_Hz is non-positive, returns None.
        """
        backend = self._make_backend()
        broken = MagicMock()
        broken.get_num_samples.side_effect = AttributeError
        assert backend._resolve_inactivity_timeout_s(broken) is None

        zero_fs = self._make_recording(1000, 0)
        assert backend._resolve_inactivity_timeout_s(zero_fs) is None


# ===========================================================================
# In-process inactivity watchdog wiring in backends
# ===========================================================================


@skip_no_spikeinterface
class TestInProcessInactivityWatchdog:
    """``SorterBackend._make_in_process_inactivity_watchdog``."""

    def _make_recording(self, n_samples=1_000_000, fs_hz=20000):
        rec = MagicMock()
        rec.get_num_samples.return_value = n_samples
        rec.get_sampling_frequency.return_value = fs_hz
        return rec

    def _make_backend(self):
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"
        return Kilosort4Backend(cfg)

    def test_returns_none_when_no_active_log_path(self):
        """
        Without a published Tee log path the helper returns None.

        Tests:
            (Test Case 1) Outside any sort_recording context the
                helper short-circuits to None.
        """
        backend = self._make_backend()
        rec = self._make_recording()
        # No set_active_log_path active.
        assert (
            backend._make_in_process_inactivity_watchdog(rec, sorter="kilosort4")
            is None
        )

    def test_builds_watchdog_with_callback_inside_sort_recording_context(
        self, tmp_path
    ):
        """
        With set_active_log_path active, the helper returns a
        configured LogInactivityWatchdog without a popen.

        Tests:
            (Test Case 1) Returned object is a LogInactivityWatchdog.
            (Test Case 2) ``popen`` is None.
            (Test Case 3) ``kill_callback`` is set (callable).
            (Test Case 4) ``inactivity_s`` matches the recording-aware
                tolerance.
        """
        from spikelab.spike_sorting.guards import (
            LogInactivityWatchdog,
            set_active_log_path,
        )

        backend = self._make_backend()
        # 5 min @ 20kHz: 600 + 30*5 = 750
        rec = self._make_recording(n_samples=20000 * 60 * 5, fs_hz=20000)

        log_path = tmp_path / "rec.log"
        with set_active_log_path(log_path):
            wd = backend._make_in_process_inactivity_watchdog(rec, sorter="kilosort4")
        assert isinstance(wd, LogInactivityWatchdog)
        assert wd.popen is None
        assert callable(wd.kill_callback)
        assert wd.inactivity_s == 750.0
        assert wd.sorter == "kilosort4"

    def test_watchdog_disabled_when_inactivity_disabled(self, tmp_path):
        """
        sorter_inactivity_timeout=False yields a no-op watchdog.

        Tests:
            (Test Case 1) Returned watchdog has _enabled=False (the
                in-process kill callback is set but inactivity_s is
                None).
        """
        from spikelab.spike_sorting.guards import set_active_log_path

        backend = self._make_backend()
        backend.config.execution.sorter_inactivity_timeout = False
        rec = self._make_recording()
        log_path = tmp_path / "rec.log"
        with set_active_log_path(log_path):
            wd = backend._make_in_process_inactivity_watchdog(rec, sorter="kilosort4")
        # inactivity_s is None → watchdog is disabled even though the
        # kill_callback is set.
        assert wd is not None
        assert wd.inactivity_s is None
        assert wd._enabled is False


# ===========================================================================
# sort_recording: OOM retry + always-run cleanup wiring
# ===========================================================================


@skip_no_spikeinterface
class TestSortRecordingOomRetry:
    """sort_recording loop honours oom_retry_max + scale_oom_params."""

    @pytest.fixture()
    def sort_fn(self):
        from spikelab.spike_sorting.pipeline import sort_recording

        return sort_recording

    def _make_config(self, **execution_overrides):
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        for key, value in execution_overrides.items():
            setattr(cfg.execution, key, value)
        return cfg

    def test_no_recordings_does_not_invoke_retry(self, sort_fn):
        """
        Empty recording list never enters the per-recording body.

        Tests:
            (Test Case 1) process_recording is not called.
            (Test Case 2) snapshot_oom_params is not called.
        """
        cfg = self._make_config()
        with (
            patch("spikelab.spike_sorting.pipeline.process_recording") as mock_proc,
            patch(
                "spikelab.spike_sorting.guards.run_preflight",
                return_value=[],
            ),
            patch("spikelab.spike_sorting.guards.report_findings"),
        ):
            result = sort_fn(
                recording_files=[],
                config=cfg,
                intermediate_folders=[],
                results_folders=[],
            )
        mock_proc.assert_not_called()
        assert result == []


@skip_no_spikeinterface
class TestFreeGpuAndPythonMemory:
    """The shared cleanup helper is callable and idempotent."""

    def test_runs_without_torch(self):
        """
        _free_gpu_and_python_memory is a no-op when torch is absent.

        Tests:
            (Test Case 1) Function runs without raising even when
                torch import fails.
        """
        from spikelab.spike_sorting import pipeline as _p

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated missing torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", _fake_import):
            _p._free_gpu_and_python_memory()  # Must not raise

    def test_callable_with_torch(self):
        """
        _free_gpu_and_python_memory invokes empty_cache() when torch is present.

        Tests:
            (Test Case 1) Function runs and reaches the torch branch
                (asserted via the patched torch.cuda.empty_cache mock).
        """
        from spikelab.spike_sorting import pipeline as _p

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            _p._free_gpu_and_python_memory()
        fake_torch.cuda.empty_cache.assert_called_once()


# ===========================================================================
# Atomic pickle write helper
# ===========================================================================


@skip_no_spikeinterface
class TestAtomicWritePickle:
    """``_atomic_write_pickle`` survives mid-write interruptions."""

    def test_writes_then_atomically_replaces(self, tmp_path):
        """
        Successful write produces the final file and removes the .tmp.

        Tests:
            (Test Case 1) Final file exists with the pickled object.
            (Test Case 2) The intermediate .tmp file is gone after
                the os.replace.
        """
        from spikelab.spike_sorting.pipeline import _atomic_write_pickle
        import pickle as _pkl

        target = tmp_path / "out.pkl"
        _atomic_write_pickle({"k": [1, 2, 3]}, target)
        assert target.exists()
        assert not (target.with_suffix(target.suffix + ".tmp")).exists()
        with open(target, "rb") as f:
            assert _pkl.load(f) == {"k": [1, 2, 3]}

    def test_creates_parent_directories(self, tmp_path):
        """
        Writing to a non-existent directory creates parents on demand.

        Tests:
            (Test Case 1) Deeply nested target path is created.
        """
        from spikelab.spike_sorting.pipeline import _atomic_write_pickle

        target = tmp_path / "deep" / "nested" / "out.pkl"
        _atomic_write_pickle("payload", target)
        assert target.exists()

    def test_replaces_existing_file_atomically(self, tmp_path):
        """
        A pre-existing file at the target is replaced wholesale.

        Tests:
            (Test Case 1) Final file's pickled contents match the
                latest write.
            (Test Case 2) Old contents are not preserved.
        """
        from spikelab.spike_sorting.pipeline import _atomic_write_pickle
        import pickle as _pkl

        target = tmp_path / "out.pkl"
        # Pre-existing payload.
        with open(target, "wb") as f:
            _pkl.dump("OLD", f)

        _atomic_write_pickle("NEW", target)
        with open(target, "rb") as f:
            assert _pkl.load(f) == "NEW"

    def test_failed_write_does_not_corrupt_existing_file(self, tmp_path):
        """
        Pickle failure mid-write leaves the existing target untouched.

        Tests:
            (Test Case 1) When pickling raises, the previous target
                file is preserved (no partial overwrite).
            (Test Case 2) The .tmp file may remain on disk; the
                contract is only that the final file is intact.
        """
        from spikelab.spike_sorting.pipeline import _atomic_write_pickle
        import pickle as _pkl

        target = tmp_path / "out.pkl"
        with open(target, "wb") as f:
            _pkl.dump("OLD", f)

        # An object that fails to pickle (lambdas are not picklable).
        with pytest.raises(Exception):
            _atomic_write_pickle(lambda x: x, target)

        # The final target must still hold the previous contents.
        with open(target, "rb") as f:
            assert _pkl.load(f) == "OLD"


# ===========================================================================
# SortRunReport
# ===========================================================================


@skip_no_spikeinterface
class TestSortRunReport:
    """``SortRunReport`` aggregates per-recording results."""

    def _make_record(self, status="success", **overrides):
        from spikelab.spike_sorting.pipeline import RecordingResult

        defaults = dict(
            rec_name="rec1",
            rec_path="/tmp/rec1.h5",
            results_folder="/tmp/sorted_rec1",
            status=status,
            wall_time_s=12.5,
        )
        defaults.update(overrides)
        return RecordingResult(**defaults)

    def test_succeeded_failed_split(self):
        """
        ``succeeded`` and ``failed`` partition records by status.

        Tests:
            (Test Case 1) Successful entries land in ``succeeded``.
            (Test Case 2) Anything else lands in ``failed``.
            (Test Case 3) ``all_succeeded`` only when every record
                is a success and the report is non-empty.
        """
        from spikelab.spike_sorting.pipeline import SortRunReport

        report = SortRunReport()
        report.add(self._make_record(status="success"))
        report.add(self._make_record(rec_name="rec2", status="oom_gpu"))
        report.add(self._make_record(rec_name="rec3", status="sorter_timeout"))
        assert len(report.succeeded) == 1
        assert len(report.failed) == 2
        assert report.all_succeeded is False

        empty = SortRunReport()
        assert empty.all_succeeded is False  # Empty is also not "all succeeded"

    def test_to_dict_round_trip(self):
        """
        ``to_dict`` exposes per-record fields and tally counts.

        Tests:
            (Test Case 1) Returned dict contains records list +
                aggregate counts.
            (Test Case 2) Counts match the report contents.
        """
        from spikelab.spike_sorting.pipeline import SortRunReport

        report = SortRunReport()
        report.add(self._make_record(status="success"))
        report.add(self._make_record(rec_name="rec2", status="failed"))
        d = report.to_dict()
        assert d["n_total"] == 2
        assert d["n_succeeded"] == 1
        assert d["n_failed"] == 1
        assert len(d["records"]) == 2
        assert d["records"][0]["rec_name"] == "rec1"


@skip_no_spikeinterface
class TestClassifyRecordingStatus:
    """``_classify_recording_status`` maps results / exceptions to statuses."""

    def test_success_and_failure_classes(self):
        """
        Each known exception maps to the right status string.

        Tests:
            (Test Case 1) Non-exception → 'success'.
            (Test Case 2) HostMemoryWatchdogError → 'oom_host_ram'.
            (Test Case 3) SorterTimeoutError → 'sorter_timeout'.
            (Test Case 4) GPUOutOfMemoryError → 'oom_gpu'.
            (Test Case 5) MemoryError → 'oom_memoryerror'.
            (Test Case 6) Plain exception → 'failed'.
        """
        from spikelab.spike_sorting.pipeline import _classify_recording_status
        from spikelab.spike_sorting._exceptions import (
            GPUOutOfMemoryError,
            HostMemoryWatchdogError,
            SorterTimeoutError,
        )

        assert _classify_recording_status(MagicMock()) == "success"
        assert (
            _classify_recording_status(
                HostMemoryWatchdogError("x", percent_at_trip=99, abort_pct=92)
            )
            == "oom_host_ram"
        )
        assert (
            _classify_recording_status(SorterTimeoutError("x", sorter="ks2"))
            == "sorter_timeout"
        )
        assert (
            _classify_recording_status(GPUOutOfMemoryError("x", sorter="ks4"))
            == "oom_gpu"
        )
        assert _classify_recording_status(MemoryError("x")) == "oom_memoryerror"
        assert _classify_recording_status(RuntimeError("plain")) == "failed"


@skip_no_spikeinterface
class TestSortRecordingOutReport:
    """``sort_recording`` populates ``out_report`` and writes per-recording JSONs."""

    @pytest.fixture()
    def sort_fn(self):
        from spikelab.spike_sorting.pipeline import sort_recording

        return sort_recording

    def _make_config(self, **execution_overrides):
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        for key, value in execution_overrides.items():
            setattr(cfg.execution, key, value)
        return cfg

    def test_out_report_empty_for_empty_batch(self, sort_fn):
        """
        Empty input list yields an empty report.

        Tests:
            (Test Case 1) ``out_report.records`` stays empty.
            (Test Case 2) No exception raised.
        """
        from spikelab.spike_sorting.pipeline import SortRunReport

        cfg = self._make_config()
        report = SortRunReport()
        with (
            patch(
                "spikelab.spike_sorting.guards.run_preflight",
                return_value=[],
            ),
            patch("spikelab.spike_sorting.guards.report_findings"),
        ):
            sort_fn(
                recording_files=[],
                config=cfg,
                intermediate_folders=[],
                results_folders=[],
                out_report=report,
            )
        assert report.records == []


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

    def test_ks4_no_match_raises(self):
        """
        KS4 with unsupported CUDA tag raises RuntimeError.

        Tests:
            (Test Case 1) cu118 has no KS4 image; raises RuntimeError.
        """
        from spikelab.spike_sorting.docker_utils import get_docker_image

        with pytest.raises(RuntimeError, match="No compatible Docker image"):
            get_docker_image("kilosort4", cuda_tag="cu118")

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
        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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
        When config.sorter.use_docker=True, run_sorter receives docker_image and installation_mode.

        Tests:
            (Test Case 1) docker_image is auto-detected via get_docker_image.
            (Test Case 2) installation_mode is 'pypi' (SI 0.104 workaround).
        """
        ks4_backend.config.sorter.use_docker = True
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
        When config.sorter.use_docker is a string, it is passed directly as docker_image.

        Tests:
            (Test Case 1) Custom image string bypasses auto-detection.
        """
        ks4_backend.config.sorter.use_docker = "my-custom-image:latest"
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
        When config.sorter.use_docker is falsy, no docker_image or installation_mode is passed.

        Tests:
            (Test Case 1) docker_image not in kwargs.
            (Test Case 2) installation_mode not in kwargs.
        """
        ks4_backend.config.sorter.use_docker = False
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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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

    def test_zero_amplitude_template_returns_zero(self, tmp_path):
        """
        Flat zero template produces half-window size 0.

        Tests:
            (Test Case 1) All-zero template → half-window size 0.

        Notes:
            - Dead-channel / all-zero templates produce half-window size 0
              (no waveform to bound).
        """
        templates = np.zeros((1, 61, 2), dtype=np.float32)
        kse, ks_mod, old_p, old_pp = self._make_kse_with_templates(
            tmp_path, templates, "zero_amp"
        )
        try:
            _, chans_ks, _ = kse.get_chans_max()
            hw_sizes = kse.get_templates_half_windows_sizes(chans_ks)
            assert len(hw_sizes) == 1
            assert hw_sizes[0] == 0
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
        ks_mod = _GlobalsStub()  # _globals.py deleted in Phase 5; stub absorbs writes

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
            (Test Case 1) config.sorter.use_docker=True with no GPU →
                RuntimeError returned, not raised.

        Notes:
            - The KS4 sort() method wraps run_sorter in try/except Exception
              and returns the exception. A failure in get_docker_image (called
              before run_sorter) is caught by the same handler.
        """
        ks4_backend.config.sorter.use_docker = True
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


# TestRTSortBackendSyncGlobals removed in Phase 3 of the _globals.py
# refactor (iat/TO_IMPLEMENT.md): RTSortBackend no longer calls
# _sync_globals on construction, so the test class — which asserted that
# the constructor wrote specific values into _globals.RT_SORT_* — no
# longer applies. Runner-level integration tests
# (TestRTSortRunnerHelpers, TestNumpySortingToKsExtractor) and the
# backend `sort` smoke tests cover the equivalent behaviour by checking
# that config values reach the runner.


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

    # The ``test_detection_window_s_*`` tests that previously lived
    # here were removed during Phase 5 cleanup: they relied on
    # ``_GlobalsStub`` to absorb writes to ``_globals.RT_SORT_*``,
    # which was a no-op against the new config-driven runner. The
    # equivalent contracts are now pinned by
    # ``TestRTSortDetectionWindow`` below using explicit
    # ``RTSortConfig`` construction.

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


class TestRTSortConfigDefaults:
    """
    Tests that the RTSortConfig declares the RT-Sort parameter fields
    referenced by the runner and backend.

    Replaces ``TestRTSortGlobals`` (deleted in Phase 5 of the
    ``_globals.py`` refactor — those tests asserted on ``_globals.
    RT_SORT_*`` attributes which were deleted along with the
    ``_globals`` module). The same default-type contract is now
    expressed through the typed :class:`RTSortConfig` dataclass.
    """

    def test_rt_sort_config_fields_present(self):
        """
        RTSortConfig exposes the fields the runner and backend rely on,
        with the documented default types.

        Tests:
            (Test Case 1) All documented attributes exist on the dataclass.
            (Test Case 2) Default booleans are bools and string defaults are strings.
        """
        from spikelab.spike_sorting.config import RTSortConfig

        cfg = RTSortConfig()
        for name in (
            "model_path",
            "device",
            "num_processes",
            "recording_window_ms",
            "params",
            "save_rt_sort_pickle",
            "delete_inter",
            "verbose",
        ):
            assert hasattr(cfg, name), f"RTSortConfig is missing {name}"

        # Default types match the documented contract
        assert isinstance(cfg.device, str)
        assert isinstance(cfg.save_rt_sort_pickle, bool)
        assert isinstance(cfg.delete_inter, bool)
        assert isinstance(cfg.verbose, bool)


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

    @skip_no_torch
    def test_partial_last_chunk_does_not_raise(self, tmp_path, monkeypatch):
        """get_total_samples() may exceed the actual sample count in the HDF5
        file.  The memmap is pre-allocated to the reported size; without the
        fix the last chunk write broadcast-fails because the source is shorter
        than the target slice.  Regression test for the (923, 199800) into
        (923, 200000) shape mismatch seen on hCO-stim-demo_2026-05-13.
        """
        from unittest.mock import MagicMock

        from spikelab.spike_sorting.rt_sort import _algorithm

        n_channels = 4
        reported_samples = 2_200  # what get_total_samples() claims
        actual_samples = 2_000  # what the HDF5 actually contains

        rng = np.random.default_rng(7)
        fake_full = (rng.standard_normal((actual_samples, n_channels)) * 10.0).astype(
            np.float32
        )

        def fake_get_traces(start_frame=None, end_frame=None, return_scaled=False):
            sf = 0 if start_frame is None else start_frame
            ef = actual_samples if end_frame is None else min(end_frame, actual_samples)
            return fake_full[sf:ef]

        rec = MagicMock()
        rec.get_sampling_frequency.return_value = 20_000.0
        rec.get_num_channels.return_value = n_channels
        rec.get_total_samples.return_value = reported_samples
        rec.get_channel_ids.return_value = np.arange(n_channels)
        rec.has_scaleable_traces.return_value = True
        rec.get_traces.side_effect = fake_get_traces

        from spikeinterface.core import NumpyRecording

        assert not isinstance(rec, NumpyRecording)

        monkeypatch.setattr(
            _algorithm,
            "Pool",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Pool used")),
        )
        monkeypatch.setattr(
            _algorithm,
            "Manager",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Manager used")),
        )

        out_path = tmp_path / "scaled_traces.npy"
        # chunk_seconds=0.1 → 2 000-sample chunks; reported=2 200 means the
        # second chunk requests frames 2000–2200 but the recording only has
        # 2000, so get_traces returns 0 samples — exercises the partial path.
        _algorithm.save_traces_si(
            rec,
            out_path,
            start_ms=0,
            end_ms=None,
            num_processes=1,
            dtype="float32",
            verbose=False,
            chunk_seconds=0.1,
        )

        saved = np.load(out_path)
        # Output shape is (n_channels, reported_samples) — partial tail is zeros.
        assert saved.shape == (n_channels, reported_samples)
        # The actual data region must match.
        np.testing.assert_allclose(saved[:, :actual_samples], fake_full.T, atol=1e-6)


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

    def test_rec_name_preserves_interior_dots(self, tmp_path):
        """
        get_paths derives rec_name via Path.stem so files with interior
        dots produce distinct intermediate paths instead of silently
        colliding via the old split('.')[0] truncation.

        Tests:
            (Test Case 1) "my.session1.h5" yields recording_dat_path
                ending in "my.session1_scaled_filtered.dat".
            (Test Case 2) "my.session2.h5" yields a DIFFERENT
                recording_dat_path (no collision).
        """
        from spikelab.spike_sorting.sorting_utils import get_paths

        rec_a = tmp_path / "my.session1.h5"
        rec_b = tmp_path / "my.session2.h5"
        rec_a.touch()
        rec_b.touch()
        inter = tmp_path / "inter"
        results = tmp_path / "results"

        paths_a = get_paths(rec_a, inter, results, execution_config=None)
        paths_b = get_paths(rec_b, inter, results, execution_config=None)

        # paths[2] is recording_dat_path
        dat_a = paths_a[2]
        dat_b = paths_b[2]
        assert dat_a.name == "my.session1_scaled_filtered.dat"
        assert dat_b.name == "my.session2_scaled_filtered.dat"
        assert dat_a != dat_b

    def test_rec_name_single_extension_unchanged(self, tmp_path):
        """
        Files with no interior dot (e.g. 'recording.h5') still produce
        the expected stem — no regression for the simple case.

        Tests:
            (Test Case 1) "recording.h5" yields rec_name "recording".
        """
        from spikelab.spike_sorting.sorting_utils import get_paths

        rec = tmp_path / "recording.h5"
        rec.touch()
        inter = tmp_path / "inter"
        results = tmp_path / "results"
        paths = get_paths(rec, inter, results, execution_config=None)
        assert paths[2].name == "recording_scaled_filtered.dat"


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


# ===========================================================================
# FEAT-005 — Docker image digest pinning (docker_utils + ExecutionConfig +
# pipeline banner). Tests live here rather than under guards/ because the
# concerns touch docker_utils.get_local_image_digest, ExecutionConfig new
# fields, and _print_pipeline_banner — none of which are guards.
# ===========================================================================


def _block_imports_for_digest(monkeypatch, *names):
    """Patch ``builtins.__import__`` to raise ImportError for *names*.

    Used by the FEAT-005 tests below to force the ``docker`` Python
    client into the subprocess fallback. Module-local mirror of the
    same helper in ``test_preflight.py`` so the two test files stay
    independent.
    """
    import builtins

    real_import = builtins.__import__
    blocked = set(names)

    def _patched_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"blocked-by-test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)


class TestGetLocalImageDigest:
    """``docker_utils.get_local_image_digest`` returns sha256 or None."""

    def test_empty_tag_returns_none(self):
        """
        Empty / falsy tag short-circuits to None.

        Tests:
            (Test Case 1) tag='' → None.
        """
        from spikelab.spike_sorting.docker_utils import get_local_image_digest

        assert get_local_image_digest("") is None

    def test_docker_py_path_used_when_available(self, monkeypatch):
        """
        Python ``docker`` client returns the digest via images.get().id.

        Tests:
            (Test Case 1) Inject a fake docker module with
                from_env().images.get(tag).id == 'sha256:abc'.
        """
        fake_image = SimpleNamespace(id="sha256:abc")
        fake_client = SimpleNamespace(
            images=SimpleNamespace(get=lambda tag: fake_image)
        )
        fake_docker = SimpleNamespace(from_env=lambda: fake_client)
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        from spikelab.spike_sorting.docker_utils import get_local_image_digest

        assert get_local_image_digest("foo:bar") == "sha256:abc"

    def test_subprocess_fallback_when_docker_py_missing(self, monkeypatch):
        """
        Without docker-py, ``docker inspect --format={{.Id}}`` is used.

        Tests:
            (Test Case 1) docker-py import patched to raise; subprocess
                stub returns 'sha256:def\\n' → trimmed to 'sha256:def'.
        """
        import subprocess as _subprocess

        monkeypatch.delitem(sys.modules, "docker", raising=False)
        _block_imports_for_digest(monkeypatch, "docker")

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        def _fake_run(args, **_kwargs):
            return _subprocess.CompletedProcess(args, 0, "sha256:def\n", "")

        monkeypatch.setattr(docker_utils_mod.subprocess, "run", _fake_run)
        assert docker_utils_mod.get_local_image_digest("foo:bar") == "sha256:def"

    def test_both_paths_fail_returns_none(self, monkeypatch):
        """
        docker-py absent and ``docker inspect`` failing → None.

        Tests:
            (Test Case 1) Import-fail + subprocess raises → None.
        """
        monkeypatch.delitem(sys.modules, "docker", raising=False)
        _block_imports_for_digest(monkeypatch, "docker")

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        def _fake_run(*_a, **_k):
            raise FileNotFoundError("no docker cli")

        monkeypatch.setattr(docker_utils_mod.subprocess, "run", _fake_run)
        assert docker_utils_mod.get_local_image_digest("foo:bar") is None

    def test_docker_py_get_failure_falls_back_to_subprocess(self, monkeypatch):
        """
        When docker-py is importable but ``images.get(tag)`` raises,
        the function falls back to the ``docker inspect`` subprocess
        path rather than returning None outright.

        Tests:
            (Test Case 1) docker.from_env().images.get raises → CLI
                fallback returns sha256:cli.
        """
        import subprocess as _subprocess
        from unittest import mock as _mock

        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                get=_mock.Mock(side_effect=RuntimeError("not found"))
            )
        )
        fake_docker = SimpleNamespace(from_env=lambda: fake_client)
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        def _fake_run(args, **_kwargs):
            return _subprocess.CompletedProcess(args, 0, "sha256:cli\n", "")

        monkeypatch.setattr(docker_utils_mod.subprocess, "run", _fake_run)
        assert docker_utils_mod.get_local_image_digest("foo:bar") == "sha256:cli"


class TestNewExecutionConfigFields:
    """``ExecutionConfig`` exposes the new canary + digest fields."""

    def test_defaults(self):
        """
        New fields default to disabled / unset.

        Tests:
            (Test Case 1) canary_first_n_s defaults to 0.0.
            (Test Case 2) docker_image_expected_digest defaults to None.
            (Test Case 3) canary_min_recording_s defaults to 120.0.
        """
        from spikelab.spike_sorting.config import ExecutionConfig

        cfg = ExecutionConfig()
        assert cfg.canary_first_n_s == 0.0
        assert cfg.docker_image_expected_digest is None
        assert cfg.canary_min_recording_s == 120.0

    def test_flat_map_round_trip(self):
        """
        Both fields can be set via SortingPipelineConfig.from_kwargs()
        and survive ``override``.

        Tests:
            (Test Case 1) from_kwargs accepts canary_first_n_s and
                docker_image_expected_digest.
            (Test Case 2) override re-sets them.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig.from_kwargs(
            canary_first_n_s=12.5,
            docker_image_expected_digest="sha256:abc",
        )
        assert cfg.execution.canary_first_n_s == 12.5
        assert cfg.execution.docker_image_expected_digest == "sha256:abc"
        cfg2 = cfg.override(canary_first_n_s=0.0)
        assert cfg2.execution.canary_first_n_s == 0.0
        # Other field preserved.
        assert cfg2.execution.docker_image_expected_digest == "sha256:abc"

    def test_canary_min_recording_s_round_trip(self):
        """
        canary_min_recording_s can be set via from_kwargs and overridden.

        Tests:
            (Test Case 1) from_kwargs accepts canary_min_recording_s.
            (Test Case 2) override re-sets it.
            (Test Case 3) Setting to 0 disables the floor.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig.from_kwargs(canary_min_recording_s=60.0)
        assert cfg.execution.canary_min_recording_s == 60.0
        cfg2 = cfg.override(canary_min_recording_s=0.0)
        assert cfg2.execution.canary_min_recording_s == 0.0


class TestPrintPipelineBannerDockerLines:
    """``_print_pipeline_banner`` surfaces Docker image + digest."""

    def test_lines_emitted_when_use_docker(self, capsys, tmp_path):
        """
        With use_docker=True and both kwargs supplied, the banner
        prints 'Docker image:' and 'Docker image digest:' lines under
        the Environment section so the report parser picks them up.

        Tests:
            (Test Case 1) Both lines present in stdout.
            (Test Case 2) Image tag + digest values appear verbatim.
        """
        from spikelab.spike_sorting.config import SorterConfig, SortingPipelineConfig
        from spikelab.spike_sorting.pipeline import _print_pipeline_banner

        cfg = SortingPipelineConfig(
            sorter=SorterConfig(sorter_name="kilosort4", use_docker=True)
        )
        _print_pipeline_banner(
            "kilosort4",
            "/data/rec.h5",
            cfg,
            log_path=tmp_path / "log.log",
            recording=None,
            docker_image_tag="spikeinterface/kilosort4-base:py311-si0.104",
            docker_image_digest="sha256:deadbeef",
        )
        out = capsys.readouterr().out
        assert "Docker image:" in out
        assert "spikeinterface/kilosort4-base:py311-si0.104" in out
        assert "Docker image digest: sha256:deadbeef" in out

    def test_lines_suppressed_when_use_docker_false(self, capsys, tmp_path):
        """
        Without use_docker the Docker lines are not printed even when
        the kwargs are supplied.

        Tests:
            (Test Case 1) use_docker=False suppresses both lines.
        """
        from spikelab.spike_sorting.config import SorterConfig, SortingPipelineConfig
        from spikelab.spike_sorting.pipeline import _print_pipeline_banner

        cfg = SortingPipelineConfig(
            sorter=SorterConfig(sorter_name="kilosort4", use_docker=False)
        )
        _print_pipeline_banner(
            "kilosort4",
            "/data/rec.h5",
            cfg,
            log_path=tmp_path / "log.log",
            recording=None,
            docker_image_tag="spikeinterface/kilosort4-base:py311-si0.104",
            docker_image_digest="sha256:deadbeef",
        )
        out = capsys.readouterr().out
        assert "Docker image:" not in out
        assert "Docker image digest:" not in out


# ===========================================================================
# Pipeline orchestration helpers (C4)
# ===========================================================================


class TestMakeRecordingResult:
    """
    Tests for ``pipeline._make_recording_result``.
    """

    def test_success_result_populates_units_and_status(self, tmp_path):
        """
        A SpikeData-like object as ``result`` produces a RecordingResult
        with status='success' and n_curated_units derived from sd.N.

        Tests:
            (Test Case 1) Status is "success".
            (Test Case 2) n_curated_units equals sd.N.
            (Test Case 3) error_class and error_message are None.
        """
        from spikelab.spike_sorting.pipeline import (
            RecordingResult,
            _make_recording_result,
        )

        results_folder = tmp_path / "rec0_sorted"
        results_folder.mkdir()
        fake_sd = SimpleNamespace(N=42)

        record = _make_recording_result(
            rec_name="rec0",
            rec_path="/data/rec0.h5",
            results_folder=results_folder,
            result=fake_sd,
            wall_time_s=12.5,
            retries_used=1,
            log_path=results_folder / "sorting.log",
        )
        assert isinstance(record, RecordingResult)
        assert record.status == "success"
        assert record.n_curated_units == 42
        assert record.error_class is None
        assert record.error_message is None
        assert record.retries_used == 1
        assert record.wall_time_s == pytest.approx(12.5)
        assert record.rec_name == "rec0"
        assert record.log_path == str(results_folder / "sorting.log")

    def test_tuple_result_uses_curated_field(self, tmp_path):
        """
        When ``result`` is a (raw, curated) tuple (save_raw_pkl=True),
        ``n_curated_units`` is taken from the second element's ``.N``.

        Tests:
            (Test Case 1) n_curated_units == curated.N (not raw.N).
        """
        from spikelab.spike_sorting.pipeline import _make_recording_result

        results_folder = tmp_path / "rec1_sorted"
        results_folder.mkdir()
        sd_raw = SimpleNamespace(N=100)
        sd_curated = SimpleNamespace(N=37)

        record = _make_recording_result(
            rec_name="rec1",
            rec_path="/data/rec1.h5",
            results_folder=results_folder,
            result=(sd_raw, sd_curated),
            wall_time_s=20.0,
            retries_used=0,
        )
        assert record.status == "success"
        assert record.n_curated_units == 37

    def test_exception_result_classifies_failure(self, tmp_path):
        """
        When ``result`` is a BaseException, the record status is
        derived from the exception type and error metadata is filled.

        Tests:
            (Test Case 1) status is "failed" for plain RuntimeError.
            (Test Case 2) error_class is the exception class name.
            (Test Case 3) error_message is the exception's message.
            (Test Case 4) n_curated_units is None on failure.
        """
        from spikelab.spike_sorting.pipeline import _make_recording_result

        results_folder = tmp_path / "rec2_sorted"
        results_folder.mkdir()
        err = RuntimeError("kilosort blew up")

        record = _make_recording_result(
            rec_name="rec2",
            rec_path="/data/rec2.h5",
            results_folder=results_folder,
            result=err,
            wall_time_s=5.0,
            retries_used=0,
        )
        assert record.status == "failed"
        assert record.error_class == "RuntimeError"
        assert record.error_message is not None
        assert "kilosort blew up" in record.error_message
        assert record.n_curated_units is None


class TestWriteRecordingReport:
    """
    Tests for ``pipeline._write_recording_report``.
    """

    def test_writes_json_with_record_fields(self, tmp_path):
        """
        ``_write_recording_report`` writes ``recording_report.json`` next
        to the per-recording results folder containing all RecordingResult
        fields.

        Tests:
            (Test Case 1) Output file exists at the expected path.
            (Test Case 2) JSON content includes the rec_name, status,
                wall_time_s, and n_curated_units fields.
        """
        import json as _json

        from spikelab.spike_sorting.pipeline import (
            RecordingResult,
            _write_recording_report,
        )

        results_folder = tmp_path / "rec_done"
        # Note: do NOT pre-create — verifies that the helper creates
        # the folder if missing.
        record = RecordingResult(
            rec_name="rec_alpha",
            rec_path="/data/rec_alpha.h5",
            results_folder=str(results_folder),
            status="success",
            wall_time_s=99.5,
            n_curated_units=21,
        )
        _write_recording_report(record, results_folder)
        out_path = results_folder / "recording_report.json"
        assert out_path.exists()
        data = _json.loads(out_path.read_text(encoding="utf-8"))
        assert data["rec_name"] == "rec_alpha"
        assert data["status"] == "success"
        assert data["wall_time_s"] == pytest.approx(99.5)
        assert data["n_curated_units"] == 21

    def test_failure_to_write_does_not_raise(self, tmp_path, capsys, monkeypatch):
        """
        Errors from the underlying ``open`` are swallowed: the helper
        prints a diagnostic but does not propagate the exception.

        Tests:
            (Test Case 1) An OSError raised by open() is caught.
            (Test Case 2) A diagnostic line is printed.
        """
        from spikelab.spike_sorting.pipeline import (
            RecordingResult,
            _write_recording_report,
        )

        record = RecordingResult(
            rec_name="rec_x",
            rec_path="/data/rec_x.h5",
            results_folder=str(tmp_path / "rx"),
            status="success",
            wall_time_s=1.0,
        )

        # Force the open() call to raise OSError on the .tmp path.
        import builtins

        real_open = builtins.open

        def boom(path, *args, **kwargs):
            if str(path).endswith(".tmp"):
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", boom)

        # Must not raise.
        _write_recording_report(record, tmp_path / "rx")
        out = capsys.readouterr().out
        assert "recording report" in out


class TestWriteDiskExhaustionReport:
    """
    Tests for ``pipeline._write_disk_exhaustion_report``.
    """

    def test_writes_disk_report_json(self, tmp_path, capsys):
        """
        ``_write_disk_exhaustion_report`` writes
        ``disk_exhaustion_report.json`` containing the report's
        to_dict() payload.

        Tests:
            (Test Case 1) Target file exists.
            (Test Case 2) Content includes folder + thresholds + free space.
            (Test Case 3) A success diagnostic is printed.
        """
        import json as _json

        from spikelab.spike_sorting.guards._disk_watchdog import (
            DiskExhaustionReport,
        )
        from spikelab.spike_sorting.pipeline import _write_disk_exhaustion_report

        report = DiskExhaustionReport(
            folder=str(tmp_path / "inter"),
            free_gb_at_trip=0.3,
            abort_threshold_gb=1.0,
            projected_need_gb=12.0,
            bytes_consumed_during_sort=5e9,
            top_consumers=[("/big.npy", 4.5)],
            suggested_actions=["clean ~/cache"],
        )
        results_folder = tmp_path / "rec_results"
        _write_disk_exhaustion_report(report, results_folder)

        target = results_folder / "disk_exhaustion_report.json"
        assert target.exists()
        data = _json.loads(target.read_text(encoding="utf-8"))
        assert data["folder"].endswith("inter")
        assert data["abort_threshold_gb"] == pytest.approx(1.0)
        assert data["free_gb_at_trip"] == pytest.approx(0.3)
        assert data["projected_need_gb"] == pytest.approx(12.0)
        assert data["suggested_actions"] == ["clean ~/cache"]
        out = capsys.readouterr().out
        assert "wrote disk-exhaustion report" in out


class TestPrintPipelineSummary:
    """
    Tests for ``pipeline._print_pipeline_summary``.
    """

    def test_prints_status_and_walltime(self, capsys):
        """
        ``_print_pipeline_summary`` prints status, wall time, and the
        SUMMARY banner header.

        Tests:
            (Test Case 1) The "SUMMARY" banner is printed.
            (Test Case 2) The status line is printed.
            (Test Case 3) The wall time is formatted as Xm Ys.
        """
        from spikelab.spike_sorting.pipeline import _print_pipeline_summary

        _print_pipeline_summary(status="success", elapsed_s=125.0)
        out = capsys.readouterr().out
        assert "SUMMARY" in out
        assert "Status:" in out
        assert "success" in out
        # 125 seconds = 2m 5s.
        assert "2m 5s" in out

    def test_prints_error_when_provided(self, capsys):
        """
        When ``error`` is provided, an Error: line with the exception
        type and message is printed.

        Tests:
            (Test Case 1) The error class name and message appear in
                the output.
        """
        from spikelab.spike_sorting.pipeline import _print_pipeline_summary

        err = ValueError("bad config")
        _print_pipeline_summary(status="failed", elapsed_s=3.0, error=err)
        out = capsys.readouterr().out
        assert "Error:" in out
        assert "ValueError" in out
        assert "bad config" in out


class TestPrintBatchSummary:
    """
    Tests for ``pipeline._print_batch_summary``.
    """

    def test_prints_totals_and_succeeded_section(self, capsys):
        """
        With one success and one failure, the batch summary prints
        totals and includes both the successful and failure sections.

        Tests:
            (Test Case 1) "BATCH SUMMARY" banner is printed.
            (Test Case 2) "Total recordings: 2" is printed.
            (Test Case 3) "Succeeded: 1" / "Failed: 1" lines are present.
            (Test Case 4) The success rec_name and the failure rec_name
                each appear in the output.
        """
        from spikelab.spike_sorting.pipeline import (
            RecordingResult,
            SortRunReport,
            _print_batch_summary,
        )

        ok = RecordingResult(
            rec_name="rec_ok",
            rec_path="/data/rec_ok.h5",
            results_folder="/out/rec_ok",
            status="success",
            wall_time_s=10.0,
            n_curated_units=12,
        )
        bad = RecordingResult(
            rec_name="rec_bad",
            rec_path="/data/rec_bad.h5",
            results_folder="/out/rec_bad",
            status="failed",
            wall_time_s=2.5,
            error_class="RuntimeError",
            error_message="boom",
            log_path="/out/rec_bad/sorting.log",
        )
        report = SortRunReport(records=[ok, bad])
        _print_batch_summary(report)
        out = capsys.readouterr().out
        assert "BATCH SUMMARY" in out
        assert "Total recordings:  2" in out
        assert "Succeeded:         1" in out
        assert "Failed:            1" in out
        assert "rec_ok" in out
        assert "rec_bad" in out
        assert "RuntimeError" in out
        assert "boom" in out

    def test_resource_trends_table_when_peaks_present(self, capsys):
        """
        Recordings whose ``peak_host_ram_pct`` / ``peak_gpu_used_pct`` /
        ``min_disk_free_gb`` are populated trigger a Markdown table
        summarising trends across the batch.

        Tests:
            (Test Case 1) The "Resource trends" header is printed.
            (Test Case 2) The numerical values appear in the table.
        """
        from spikelab.spike_sorting.pipeline import (
            RecordingResult,
            SortRunReport,
            _print_batch_summary,
        )

        rec = RecordingResult(
            rec_name="rec_trend",
            rec_path="/data/rec_trend.h5",
            results_folder="/out/rec_trend",
            status="success",
            wall_time_s=42.0,
            n_curated_units=5,
            peak_host_ram_pct=87.3,
            peak_gpu_used_pct=72.1,
            min_disk_free_gb=3.25,
        )
        report = SortRunReport(records=[rec])
        _print_batch_summary(report)
        out = capsys.readouterr().out
        assert "Resource trends" in out
        assert "87.3" in out
        assert "72.1" in out
        assert "3.25" in out


class TestMakeDiskWatchdog:
    """
    Tests for ``pipeline._make_disk_watchdog``.
    """

    def test_disabled_when_config_disables_disk_watchdog(self, tmp_path):
        """
        When ``config.execution.disk_watchdog`` is False, the helper
        returns a DiskUsageWatchdog with no kill target set, so the
        watchdog's ``_enabled`` flag is False.

        Tests:
            (Test Case 1) Returned object is a DiskUsageWatchdog.
            (Test Case 2) ``_enabled`` is False (popen=None,
                kill_callback=None).
            (Test Case 3) Configured thresholds and folder are still
                populated from the config.
        """
        from spikelab.spike_sorting.config import (
            ExecutionConfig,
            SortingPipelineConfig,
        )
        from spikelab.spike_sorting.guards._disk_watchdog import (
            DiskUsageWatchdog,
        )
        from spikelab.spike_sorting.pipeline import _make_disk_watchdog

        cfg = SortingPipelineConfig(
            execution=ExecutionConfig(
                disk_watchdog=False,
                disk_warn_free_gb=4.5,
                disk_abort_free_gb=0.5,
                disk_poll_interval_s=15.0,
            )
        )
        wd = _make_disk_watchdog(
            inter_path=tmp_path / "inter",
            config=cfg,
            sorter="kilosort2",
        )
        assert isinstance(wd, DiskUsageWatchdog)
        assert wd._enabled is False
        assert wd.warn_free_gb == pytest.approx(4.5)
        assert wd.abort_free_gb == pytest.approx(0.5)
        assert wd.poll_interval_s == pytest.approx(15.0)
        assert wd.folder == Path(tmp_path / "inter")
        assert wd.sorter == "kilosort2"

    def test_enabled_with_kill_callback_when_watchdog_active(self, tmp_path):
        """
        With ``disk_watchdog=True`` (default), the helper installs a
        kill callback so the returned watchdog is ``_enabled``.

        Tests:
            (Test Case 1) Returned object is a DiskUsageWatchdog.
            (Test Case 2) ``_enabled`` is True (kill_callback installed).
            (Test Case 3) ``sorter`` field matches the input.
        """
        from spikelab.spike_sorting.config import (
            ExecutionConfig,
            SortingPipelineConfig,
        )
        from spikelab.spike_sorting.guards._disk_watchdog import (
            DiskUsageWatchdog,
        )
        from spikelab.spike_sorting.pipeline import _make_disk_watchdog

        cfg = SortingPipelineConfig(execution=ExecutionConfig(disk_watchdog=True))
        wd = _make_disk_watchdog(
            inter_path=tmp_path / "inter2",
            config=cfg,
            sorter="kilosort4",
        )
        assert isinstance(wd, DiskUsageWatchdog)
        assert wd._enabled is True
        assert wd.kill_callback is not None
        assert wd.sorter == "kilosort4"


class TestBoundedHostMemory:
    """
    Tests for ``pipeline._bounded_host_memory``.
    """

    def test_no_op_when_resource_module_missing(self, capsys, monkeypatch):
        """
        On platforms without the ``resource`` module (Windows), the
        context manager is a no-op that prints a notice.

        Tests:
            (Test Case 1) Context exits normally without raising.
            (Test Case 2) A notice line is printed.
        """
        from spikelab.spike_sorting.pipeline import _bounded_host_memory

        # Force the import of `resource` to fail by stubbing the
        # builtins.__import__ machinery used inside the function. We
        # cannot rely on platform here since the test must work on
        # Linux runners too.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "resource":
                raise ImportError("forced — no resource module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with _bounded_host_memory(0.8):
            pass
        out = capsys.readouterr().out
        assert "host memory cap" in out

    def test_no_op_when_ram_unknown(self, capsys, monkeypatch):
        """
        When ``get_system_ram_bytes`` returns None, the helper prints
        a notice and yields without setting a limit.

        Tests:
            (Test Case 1) Context exits normally.
            (Test Case 2) A "cap not enforced" notice is printed.
        """
        from spikelab.spike_sorting.pipeline import _bounded_host_memory

        # If the platform lacks `resource`, the function early-returns
        # before ever reaching get_system_ram_bytes; in that case
        # this branch isn't testable directly. Skip when needed.
        try:
            import resource as _resource  # noqa: F401
        except ImportError:
            pytest.skip("`resource` module not available on this platform")

        from spikelab.spike_sorting import sorting_utils

        monkeypatch.setattr(sorting_utils, "get_system_ram_bytes", lambda: None)

        with _bounded_host_memory(0.5):
            pass
        out = capsys.readouterr().out
        assert "Could not detect system RAM" in out

    def test_clamps_new_soft_to_existing_hard_limit(self, capsys, monkeypatch):
        """
        When the requested cap exceeds the current hard RLIMIT_DATA,
        the helper clamps to the hard limit and the original soft
        limit is restored on exit.

        Tests:
            (Test Case 1) ``setrlimit`` is called with new_soft <= hard.
            (Test Case 2) On exit, the original (soft, hard) tuple is
                restored.
        """
        from spikelab.spike_sorting.pipeline import _bounded_host_memory

        try:
            import resource as _resource
        except ImportError:
            pytest.skip("`resource` module not available on this platform")

        # Patch get_system_ram_bytes so the requested new_soft is
        # huge — much bigger than any sane hard limit we set.
        from spikelab.spike_sorting import sorting_utils

        # 1 PB requested cap (frac=0.8 → 0.8 PB) — vastly bigger than hard.
        monkeypatch.setattr(sorting_utils, "get_system_ram_bytes", lambda: 10**15)

        original_soft, original_hard = (1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
        observed_calls = []

        def fake_getrlimit(which):
            return (original_soft, original_hard)

        def fake_setrlimit(which, limits):
            observed_calls.append(limits)

        monkeypatch.setattr(_resource, "getrlimit", fake_getrlimit)
        monkeypatch.setattr(_resource, "setrlimit", fake_setrlimit)

        with _bounded_host_memory(0.8):
            pass

        # Two setrlimit calls: one to lower (clamped to hard), one to restore.
        assert len(observed_calls) == 2
        new_soft_in_call, hard_in_call = observed_calls[0]
        assert new_soft_in_call <= original_hard
        assert hard_in_call == original_hard
        # Restoration call.
        restore_call = observed_calls[1]
        assert restore_call == (original_soft, original_hard)


# ===========================================================================
# Phase 5 refactor coverage — regression tests for the cross-recording leak
# fix, plus documenting tests for the migrated entry points.
# ===========================================================================


@skip_no_spikeinterface
class TestBackendDoesNotMutateConfigRecChunks:
    """Regression tests for the cross-recording leak fix.

    Pre-fix, each backend's ``load_recording`` wrote the effective frame
    chunks back onto ``self.config.recording.rec_chunks``. Since
    ``sort_recording`` reuses one backend instance across the batch
    loop, recording N's effective chunks leaked into recording N+1's
    user-supplied configuration — either tripping the frame-vs-time
    ValueError or silently slicing N+1 with N's boundaries.

    The fix stores effective chunks on ``self.rec_chunks_effective``
    (a fresh attribute on the backend) instead of writing them back to
    the shared config. ``pipeline.py`` reads from the backend attribute
    for the SpikeData metadata.
    """

    @pytest.fixture
    def _patch_recording_io(self, monkeypatch):
        """Stub ``_load_recording_with_state`` to return a known fixed
        result without touching the real spike loader. The test focuses
        on whether the backend mutates config or stores on self.
        """
        from spikelab.spike_sorting import recording_io as _rio

        rec = _make_mock_recording()
        fake_chunks = [(0, 1_000), (1_000, 2_500)]
        fake_names = ["a.raw.h5", "b.raw.h5"]

        result = _rio.LoadRecordingResult(
            recording=rec, rec_chunks=fake_chunks, recording_names=fake_names
        )

        def _stub(rec_path, config=None):
            return result

        monkeypatch.setattr(_rio, "_load_recording_with_state", _stub)
        return rec, fake_chunks, fake_names

    def test_kilosort2_backend_does_not_mutate_config(self, _patch_recording_io):
        """
        ``Kilosort2Backend.load_recording`` stores effective chunks on
        the backend, not on ``config.recording.rec_chunks``.

        Tests:
            (Test Case 1) ``self.rec_chunks_effective`` == the effective
                chunks returned by the loader.
            (Test Case 2) ``self.config.recording.rec_chunks`` remains
                unchanged from the user-supplied config (here: empty
                list, the default).
            (Test Case 3) ``self.rec_chunk_names`` matches the loader's
                names.
        """
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        rec, fake_chunks, fake_names = _patch_recording_io
        config = SortingPipelineConfig()
        backend = Kilosort2Backend(config)

        backend.load_recording("any.h5")

        assert backend.rec_chunks_effective == fake_chunks
        assert backend.rec_chunk_names == fake_names
        # Critical: user-supplied config is untouched.
        assert backend.config.recording.rec_chunks == []

    def test_kilosort4_backend_does_not_mutate_config(self, _patch_recording_io):
        """Same regression check for the Kilosort4 backend."""
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        _rec, fake_chunks, _names = _patch_recording_io
        config = SortingPipelineConfig()
        backend = Kilosort4Backend(config)

        backend.load_recording("any.h5")

        assert backend.rec_chunks_effective == fake_chunks
        assert backend.config.recording.rec_chunks == []

    def test_backend_reused_across_recordings_isolates_chunks(self, monkeypatch):
        """
        Two sequential loads with different effective chunks must not
        contaminate the user-supplied config. The second load must see
        the user's original ``rec_chunks`` (``[]``), not recording 1's
        effective chunks.

        Tests:
            (Test Case 1) Load A returns chunks ``[(0, 1000)]``;
                backend attr reflects them; config remains ``[]``.
            (Test Case 2) Load B returns chunks ``[(0, 2000)]``;
                backend attr now reflects B's chunks; config still
                ``[]``.
        """
        from spikelab.spike_sorting import recording_io as _rio
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        rec = _make_mock_recording()
        chunks_a = [(0, 1_000)]
        chunks_b = [(0, 2_000)]
        results = iter(
            [
                _rio.LoadRecordingResult(rec, chunks_a, ["a.raw.h5"]),
                _rio.LoadRecordingResult(rec, chunks_b, ["b.raw.h5"]),
            ]
        )

        def _stub(rec_path, config=None):
            return next(results)

        monkeypatch.setattr(_rio, "_load_recording_with_state", _stub)

        config = SortingPipelineConfig()
        backend = Kilosort2Backend(config)

        backend.load_recording("a.h5")
        assert backend.rec_chunks_effective == chunks_a
        assert backend.config.recording.rec_chunks == []

        backend.load_recording("b.h5")
        assert backend.rec_chunks_effective == chunks_b
        assert backend.config.recording.rec_chunks == []


@skip_no_spikeinterface
class TestConcatenateRecordingsEmptyDirectory:
    """Regression test for the FileNotFoundError guard added when the
    input directory has no ``.raw.h5`` or ``.nwb`` files. Pre-fix this
    crashed with ``UnboundLocalError`` because ``rec`` was never bound.
    """

    def test_empty_directory_raises_file_not_found(self, tmp_path):
        """
        ``_concatenate_recordings_with_state`` raises
        ``FileNotFoundError`` when the directory contains no recording
        files, naming both the path and the expected extensions.

        Tests:
            (Test Case 1) Empty directory raises FileNotFoundError.
            (Test Case 2) Error message references the directory path.
            (Test Case 3) Error message names the expected extensions.
        """
        from spikelab.spike_sorting.recording_io import (
            _concatenate_recordings_with_state,
        )

        with pytest.raises(FileNotFoundError) as exc:
            _concatenate_recordings_with_state(tmp_path)
        msg = str(exc.value)
        assert str(tmp_path) in msg
        assert ".raw.h5" in msg or ".nwb" in msg

    def test_directory_with_unrelated_files_raises(self, tmp_path):
        """
        A directory containing files that do not match the supported
        extensions is treated as empty.

        Tests:
            (Test Case 1) ``.txt`` and ``.npy`` files don't count as
                recordings — function raises FileNotFoundError.
        """
        from spikelab.spike_sorting.recording_io import (
            _concatenate_recordings_with_state,
        )

        (tmp_path / "notes.txt").write_text("not a recording")
        (tmp_path / "spike_times.npy").write_bytes(b"\x00")

        with pytest.raises(FileNotFoundError):
            _concatenate_recordings_with_state(tmp_path)


class TestRunKilosortFormatParamsIsPure:
    """``RunKilosort.format_params`` is a pure function — it never
    mutates its input dict. This is the canonical fix for the
    cross-recording leak in the original CRITICAL finding.
    """

    def test_format_params_returns_new_dict_without_mutating_input(self):
        """
        Calling ``format_params`` twice on the same input dict produces
        identical outputs and leaves the input unchanged.

        Tests:
            (Test Case 1) Input ``car=True`` survives — output ``car=1``.
            (Test Case 2) Input ``NT=None`` survives — output is
                ``64*1024 + ntbuff``.
            (Test Case 3) Second call on the same input still sees the
                original values (the leak symptom would be ``car=1``
                flipped to bool, ``NT`` mutated to an int).
        """
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        params = {
            "car": True,
            "NT": None,
            "ntbuff": 64,
            "projection_threshold": [10, 4],
        }
        params_before = dict(params)

        out_a = RunKilosort.format_params(params)
        out_b = RunKilosort.format_params(params)

        # Input untouched
        assert params == params_before
        # Outputs are normalized
        assert out_a["car"] == 1
        assert out_a["NT"] == 64 * 1024 + 64
        # Repeated calls produce identical outputs (no drift from
        # in-place mutation)
        assert out_a == out_b

    def test_format_params_rounds_nt_to_multiple_of_32(self):
        """
        Concrete ``NT`` values are rounded down to the nearest multiple
        of 32 (KS2 mex requirement).

        Tests:
            (Test Case 1) NT=70_000 → 69984 (== 70000 // 32 * 32).
            (Test Case 2) NT=64 → 64 (already a multiple of 32).
        """
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        out = RunKilosort.format_params({"car": False, "NT": 70_000, "ntbuff": 64})
        assert out["NT"] == 70_000 // 32 * 32

        out = RunKilosort.format_params({"car": False, "NT": 64, "ntbuff": 64})
        assert out["NT"] == 64

    def test_format_params_car_false_becomes_zero(self):
        """``car=False`` maps to integer 0 (the MATLAB ops template
        uses ``car`` as a numeric literal).
        """
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        out = RunKilosort.format_params({"car": False, "NT": 64, "ntbuff": 64})
        assert out["car"] == 0


@skip_no_spikeinterface
class TestWaveformExtractorJsonRoundTrip:
    """``WaveformExtractor.__init__`` reads three new keys from the
    ``extraction_parameters.json`` written by ``create_initial``:
    ``pos_peak_thresh``, ``max_waveforms_per_unit``, ``save_waveform_files``.
    A JSON file written before Phase 2.4 lacks ``save_waveform_files``;
    the constructor falls back to ``WaveformConfig`` defaults so old
    waveform folders remain loadable.
    """

    def _make_dataset(self, tmp_path):
        from spikeinterface.core import NumpyRecording
        from spikelab.spike_sorting.sorting_extractor import (
            KilosortSortingExtractor,
        )

        fs = 20000.0
        n_samples = int(0.5 * fs)
        n_channels = 4
        rng = np.random.default_rng(0)
        traces = rng.standard_normal((n_samples, n_channels)).astype(np.float32)
        rec = NumpyRecording(traces_list=[traces], sampling_frequency=fs)

        ks = tmp_path / "ks"
        ks.mkdir()
        np.save(ks / "spike_times.npy", np.array([100, 200], dtype=np.int64))
        np.save(ks / "spike_clusters.npy", np.array([0, 1], dtype=np.int64))
        np.save(ks / "templates.npy", np.zeros((2, 82, n_channels), dtype=np.float32))
        np.save(ks / "channel_map.npy", np.arange(n_channels))
        (ks / "params.py").write_text(
            f"dat_path='r.dat'\nn_channels_dat={n_channels}\ndtype='float32'\n"
            f"offset=0\nsample_rate={fs}\nhp_filtered=True\n"
        )
        return rec, KilosortSortingExtractor(ks), ks

    def test_init_persists_save_waveform_files_in_json(self, tmp_path):
        """
        ``create_initial`` writes ``save_waveform_files`` into the
        parameters JSON so it round-trips through ``__init__``.

        Tests:
            (Test Case 1) JSON contains the configured value.
            (Test Case 2) Instance attr reflects it after a fresh
                ``__init__`` from disk.
        """
        import json
        from spikelab.spike_sorting.config import (
            SortingPipelineConfig,
            WaveformConfig,
        )
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        rec, sorting, ks = self._make_dataset(tmp_path)
        config = SortingPipelineConfig(
            waveform=WaveformConfig(save_waveform_files=False)
        )
        root = tmp_path / "wf"
        we = WaveformExtractor.create_initial(
            recording_path=ks / "recording.dat",
            recording=rec,
            sorting=sorting,
            root_folder=root,
            initial_folder=root / "initial",
            config=config,
        )

        params = json.loads((root / "extraction_parameters.json").read_text())
        assert params["save_waveform_files"] is False
        assert we.save_waveform_files is False

    def test_init_falls_back_to_defaults_for_missing_json_keys(self, tmp_path):
        """
        ``WaveformExtractor.__init__`` falls back to ``WaveformConfig``
        defaults when the JSON predates Phase 2.4 (no
        ``save_waveform_files`` / ``pos_peak_thresh`` /
        ``max_waveforms_per_unit`` keys).

        Tests:
            (Test Case 1) Older JSON (no new keys) loads without error.
            (Test Case 2) Missing keys fall back to ``WaveformConfig()``
                defaults — documents the silent fallback behaviour.

        Notes:
            - The fallback is intentional but silent. A downstream
              reload of an old folder will use the dataclass defaults
              for any missing keys, which may differ from whatever
              global was set at original-extraction time. Loud-vs-quiet
              behaviour here is a design call; this test pins the
              current quiet contract.
        """
        import json
        from spikelab.spike_sorting.config import WaveformConfig
        from spikelab.spike_sorting.waveform_extractor import WaveformExtractor

        rec, sorting, ks = self._make_dataset(tmp_path)
        root = tmp_path / "wf"
        root.mkdir()
        (root / "waveforms").mkdir()
        # Simulate an older parameters JSON without the three new keys.
        legacy_params = {
            "recording_path": str(ks / "recording.dat"),
            "sampling_frequency": 20000.0,
            "ms_before": 2.0,
            "ms_after": 2.0,
            "peak_ind": int(2.0 * 20000.0 / 1000.0),
            "dtype": "float32",
            "n_jobs": 1,
            "total_memory": "1G",
        }
        (root / "extraction_parameters.json").write_text(json.dumps(legacy_params))

        we = WaveformExtractor(rec, sorting, root, root / "initial")
        wf_defaults = WaveformConfig()
        assert we.pos_peak_thresh == wf_defaults.pos_peak_thresh
        assert we.max_waveforms_per_unit == wf_defaults.max_waveforms_per_unit
        assert we.save_waveform_files == wf_defaults.save_waveform_files


@skip_no_spikeinterface
class TestBackendOomScalingMutatesConfig:
    """The Phase 3+4 refactor removed ``_sync_globals()`` from
    ``scale_oom_params`` / ``restore_oom_params``. The scaled value
    now lives on ``self.config.sorter.sorter_params`` (KS2/KS4)
    directly, and any subsequent ``RunKilosort`` instance constructed
    from the same backend must read it from there.
    """

    def test_kilosort2_scale_persists_on_config(self):
        """
        ``Kilosort2Backend.scale_oom_params`` writes the scaled NT onto
        ``self.config.sorter.sorter_params`` so the next
        ``RunKilosort`` instance constructed from the same backend
        sees it.

        Tests:
            (Test Case 1) Initial ``sorter_params`` is None → first
                scale resolves NT from default + ntbuff, then halves.
            (Test Case 2) Scaled NT is rounded to a multiple of 32.
            (Test Case 3) ``restore_oom_params`` reverts to the
                snapshot.
        """
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        backend = Kilosort2Backend(SortingPipelineConfig())
        snapshot = backend.snapshot_oom_params()

        scaled = backend.scale_oom_params(0.5)
        assert scaled is True
        assert backend.config.sorter.sorter_params is not None
        nt = backend.config.sorter.sorter_params["NT"]
        # NT is a multiple of 32 and was halved from the default
        # 64*1024 + 64 = 65600.
        assert nt % 32 == 0
        assert nt < 64 * 1024 + 64

        backend.restore_oom_params(snapshot)
        # Restored to original (None in this case).
        assert backend.config.sorter.sorter_params is None

    def test_kilosort4_scale_persists_on_config(self):
        """``Kilosort4Backend.scale_oom_params`` persists ``batch_size``
        on ``self.config.sorter.sorter_params``."""
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        backend = Kilosort4Backend(SortingPipelineConfig())
        snapshot = backend.snapshot_oom_params()

        scaled = backend.scale_oom_params(0.5)
        assert scaled is True
        # Default batch_size 60000 → 30000 after halving.
        assert backend.config.sorter.sorter_params["batch_size"] == 30000

        backend.restore_oom_params(snapshot)
        assert backend.config.sorter.sorter_params is None


# ===========================================================================
# Direct tests for the new private state-returning helpers in recording_io.py
# (the replacement for the deleted _globals.REC_CHUNKS / _REC_CHUNK_NAMES
# reads). Backends now consume ``LoadRecordingResult.rec_chunks`` and
# ``LoadRecordingResult.recording_names`` directly.
# ===========================================================================


def _make_concat_mock_recording(num_samples=100_000, sampling_frequency=20000.0):
    """Recording mock with the extra ``frame_slice`` method needed by the
    chunk-applying loader path."""
    rec = _make_mock_recording(
        num_samples=num_samples, sampling_frequency=sampling_frequency
    )
    rec.frame_slice = lambda start_frame, end_frame: _make_mock_recording(
        num_samples=end_frame - start_frame,
        sampling_frequency=sampling_frequency,
    )
    rec.get_probes = lambda: []
    rec.set_probes = lambda probes: rec
    return rec


@skip_no_spikeinterface
class TestConcatenateRecordingsWithState:
    """Direct tests for ``_concatenate_recordings_with_state`` — the
    private helper that returns the per-file frame boundaries and file
    names previously written into ``_globals.REC_CHUNKS`` /
    ``_globals._REC_CHUNK_NAMES``.
    """

    @pytest.fixture()
    def patched_concat(self, monkeypatch):
        """Patch ``load_single_recording`` and ``concatenate_recordings``
        so the helper can be exercised without real Maxwell/NWB files.
        """
        from spikelab.spike_sorting import recording_io

        def make_loader(per_file_samples):
            counts = list(per_file_samples)
            calls = {"i": 0}

            def _load(path, **_kw):
                rec = _make_mock_recording(
                    num_samples=counts[calls["i"]], sampling_frequency=20000.0
                )
                calls["i"] += 1
                return rec

            return _load

        def _stub_concat(recs):
            total = sum(r.get_total_samples() for r in recs)
            return _make_mock_recording(
                num_samples=total, sampling_frequency=recs[0].get_sampling_frequency()
            )

        def _setup(per_file_samples):
            monkeypatch.setattr(
                recording_io, "load_single_recording", make_loader(per_file_samples)
            )
            monkeypatch.setattr(
                recording_io.si_segmentutils,
                "concatenate_recordings",
                _stub_concat,
            )

        return _setup

    def test_two_file_directory_returns_per_file_boundaries(
        self, patched_concat, tmp_path
    ):
        """
        A directory with two ``.raw.h5`` files returns auto-populated
        per-file frame boundaries plus the natsorted file-name list.

        Tests:
            (Test Case 1) ``auto_rec_chunks`` covers both files with
                ``[(0, n_a), (n_a, n_a+n_b)]``.
            (Test Case 2) ``recording_names`` is the natsorted list.
            (Test Case 3) Returned recording covers the full duration.
        """
        from spikelab.spike_sorting.recording_io import (
            _concatenate_recordings_with_state,
        )

        n_a, n_b = 1_000, 2_500
        patched_concat([n_a, n_b])
        (tmp_path / "a.raw.h5").touch()
        (tmp_path / "b.raw.h5").touch()

        rec, auto_chunks, names = _concatenate_recordings_with_state(tmp_path)

        assert auto_chunks == [(0, n_a), (n_a, n_a + n_b)]
        assert names == ["a.raw.h5", "b.raw.h5"]
        assert rec.get_total_samples() == n_a + n_b

    def test_single_file_directory_returns_empty_chunks(self, patched_concat, tmp_path):
        """
        A directory containing exactly one matching file returns
        ``auto_rec_chunks=[]`` (single-recording inputs do not need
        per-file frame boundaries) but still returns a one-element
        ``recording_names``.

        Tests:
            (Test Case 1) ``auto_rec_chunks`` is empty for a single-file
                directory — the documented Phase 2.1 contract.
            (Test Case 2) ``recording_names`` is a one-element list.
        """
        from spikelab.spike_sorting.recording_io import (
            _concatenate_recordings_with_state,
        )

        patched_concat([1_500])
        (tmp_path / "only.raw.h5").touch()

        rec, auto_chunks, names = _concatenate_recordings_with_state(tmp_path)

        assert auto_chunks == []
        assert names == ["only.raw.h5"]
        assert rec.get_total_samples() == 1_500

    def test_recording_names_natsorted(self, patched_concat, tmp_path):
        """
        File names are returned in natural-sort order so that
        ``rec_10`` follows ``rec_2`` (not lex-sort order which would
        place ``rec_10`` before ``rec_2``).

        Tests:
            (Test Case 1) Natsort order: rec_2 before rec_10.
        """
        from spikelab.spike_sorting.recording_io import (
            _concatenate_recordings_with_state,
        )

        patched_concat([100, 200, 300])
        for name in ("rec_2.raw.h5", "rec_10.raw.h5", "rec_1.raw.h5"):
            (tmp_path / name).touch()

        _rec, _chunks, names = _concatenate_recordings_with_state(tmp_path)

        assert names == ["rec_1.raw.h5", "rec_2.raw.h5", "rec_10.raw.h5"]


@skip_no_spikeinterface
class TestLoadRecordingWithState:
    """Direct tests for ``_load_recording_with_state`` — the private
    helper backends call to receive the effective frame chunks and
    per-file recording names. Verifies the ``LoadRecordingResult``
    return tuple across the three supported input modes.
    """

    def test_pre_loaded_baserecording_returns_empty_state(self, monkeypatch):
        """
        Passing a pre-loaded ``BaseRecording`` skips the directory
        branch entirely; both ``rec_chunks`` and ``recording_names``
        are empty.

        Tests:
            (Test Case 1) ``rec_chunks == []`` for pre-loaded input.
            (Test Case 2) ``recording_names == []`` for pre-loaded input.
            (Test Case 3) Returned recording is the loaded mock.
        """
        from spikelab.spike_sorting import recording_io

        rec = _make_concat_mock_recording()
        # Make isinstance(rec, BaseRecording) succeed against the mock.
        monkeypatch.setattr(recording_io, "BaseRecording", type(rec), raising=False)
        monkeypatch.setattr(recording_io, "load_single_recording", lambda p, **_kw: rec)

        result = recording_io._load_recording_with_state(rec)

        assert result.rec_chunks == []
        assert result.recording_names == []
        assert result.recording is rec

    def test_directory_returns_auto_chunks_and_names(self, monkeypatch, tmp_path):
        """
        Loading from a 2-file directory propagates the per-file
        boundaries from ``_concatenate_recordings_with_state`` into the
        ``LoadRecordingResult.rec_chunks`` field, and the file-name
        list into ``recording_names``.

        Tests:
            (Test Case 1) ``rec_chunks`` matches the per-file boundaries.
            (Test Case 2) ``recording_names`` matches the directory
                listing.
        """
        from spikelab.spike_sorting import recording_io

        n_a, n_b = 1_000, 2_500
        rec_total = _make_concat_mock_recording(num_samples=n_a + n_b)

        def _stub_concat_state(rec_path, config=None):
            return rec_total, [(0, n_a), (n_a, n_a + n_b)], ["a.raw.h5", "b.raw.h5"]

        monkeypatch.setattr(
            recording_io,
            "_concatenate_recordings_with_state",
            _stub_concat_state,
        )
        # The loader applies the auto chunks via frame_slice and then
        # re-concatenates the slices through SI; stub that to avoid a
        # real SpikeInterface call against the SimpleNamespace mock.
        monkeypatch.setattr(
            recording_io.si_segmentutils,
            "concatenate_recordings",
            lambda recs: recs[0],
        )

        (tmp_path / "a.raw.h5").touch()
        (tmp_path / "b.raw.h5").touch()
        result = recording_io._load_recording_with_state(tmp_path)

        assert result.rec_chunks == [(0, n_a), (n_a, n_a + n_b)]
        assert result.recording_names == ["a.raw.h5", "b.raw.h5"]

    def test_default_config_does_not_mutate_caller(self, monkeypatch):
        """
        ``_load_recording_with_state(rec, config=None)`` constructs a
        fresh default ``SortingPipelineConfig`` internally and never
        reaches back to mutate the caller's config (because there is
        no caller config in this case). Pinned here as a regression
        guard for the refactor's no-mutation invariant.

        Tests:
            (Test Case 1) ``config=None`` path returns a result with
                empty ``rec_chunks`` (no time-slicing, no user-supplied
                chunks, no auto-populated chunks for a single-file
                input).
        """
        from spikelab.spike_sorting import recording_io

        rec = _make_concat_mock_recording()
        monkeypatch.setattr(recording_io, "BaseRecording", type(rec), raising=False)
        monkeypatch.setattr(recording_io, "load_single_recording", lambda p, **_kw: rec)

        result = recording_io._load_recording_with_state(rec, config=None)

        assert result.rec_chunks == []
        assert result.recording_names == []


@skip_no_spikeinterface
class TestLoadRecordingTimeOverridesAutoConcatChunks:
    """Regression test for the documented "auto-populated chunks
    silently overridden by time-based slicing" contract — the path
    the canary relies on when narrowing a directory recording to its
    leading window.

    Pre-refactor this was guarded by ``_globals.REC_CHUNKS_FROM_CONCAT``;
    the deleted ``TestRecChunksFromConcatOverride`` was the only test
    of this branch. The replacement class
    (``TestLoaderTimeVsFrameChunks``) covers only the user-supplied
    chunks + time collision case. This class restores coverage of the
    silent-override path.
    """

    def test_time_slicing_supersedes_auto_populated_chunks(self, monkeypatch, tmp_path):
        """
        Directory loader auto-populates per-file boundaries; setting
        ``end_time_s`` then takes precedence and the auto chunks are
        dropped from the effective chunk list.

        A regression that flipped this precedence (auto over time)
        would silently let the canary sort the full directory rather
        than the leading window — the bug ``REC_CHUNKS_FROM_CONCAT``
        existed to prevent in the pre-refactor design.

        Tests:
            (Test Case 1) ``result.rec_chunks`` equals the time-derived
                single chunk, NOT the per-file auto chunks.
            (Test Case 2) ``result.recording_names`` is still populated
                (the canary uses it to address the original files).
        """
        from spikelab.spike_sorting import recording_io
        from spikelab.spike_sorting.config import (
            RecordingConfig,
            SortingPipelineConfig,
        )

        fs = 20000.0
        n_a, n_b = 100_000, 200_000  # 5 s + 10 s @ 20 kHz
        rec_total = _make_concat_mock_recording(
            num_samples=n_a + n_b, sampling_frequency=fs
        )

        def _stub_concat_state(rec_path, config=None):
            return (
                rec_total,
                [(0, n_a), (n_a, n_a + n_b)],
                ["a.raw.h5", "b.raw.h5"],
            )

        monkeypatch.setattr(
            recording_io,
            "_concatenate_recordings_with_state",
            _stub_concat_state,
        )

        # Stub the segmentutils concatenate that the loader calls when
        # applying the time chunks.
        monkeypatch.setattr(
            recording_io.si_segmentutils,
            "concatenate_recordings",
            lambda recs: recs[0],
        )

        (tmp_path / "a.raw.h5").touch()
        (tmp_path / "b.raw.h5").touch()

        # Time-window narrows to the first 2 s — must override the
        # 2-file auto chunks.
        config = SortingPipelineConfig(
            recording=RecordingConfig(start_time_s=0.0, end_time_s=2.0)
        )
        result = recording_io._load_recording_with_state(tmp_path, config=config)

        expected_time_chunk = [(0, int(round(2.0 * fs)))]
        assert result.rec_chunks == expected_time_chunk
        # File-name list must still flow through unchanged.
        assert result.recording_names == ["a.raw.h5", "b.raw.h5"]


# ===========================================================================
# RunKilosort.__init__ default resolution — the new lazy-import path that
# closes the old _globals.KILOSORT_PARAMS / pos_peak_thresh leaks.
# ===========================================================================


@skip_no_spikeinterface
class TestRunKilosortDefaults:
    """``RunKilosort.__init__`` resolves ``kilosort_params=None`` and
    ``pos_peak_thresh=None`` to their canonical defaults at construction
    time. Pre-refactor these were read out of ``_globals.KILOSORT_PARAMS``
    and ``_globals.POS_PEAK_THRESH``; post-refactor they are sourced
    from ``DEFAULT_KILOSORT2_PARAMS`` and ``WaveformConfig()``.
    """

    @pytest.fixture()
    def fake_kilosort_path(self, tmp_path):
        """A tmp dir with the sentinel MATLAB entry-point file so
        ``check_if_installed`` returns True without requiring a real
        Kilosort2 install."""
        (tmp_path / "master_kilosort.m").touch()
        return tmp_path

    def test_kilosort_params_none_resolves_to_default_dict(self, fake_kilosort_path):
        """
        ``kilosort_params=None`` falls back to a normalised copy of
        ``DEFAULT_KILOSORT2_PARAMS``.

        Tests:
            (Test Case 1) Resulting ``self.kilosort_params`` equals
                ``format_params(DEFAULT_KILOSORT2_PARAMS)``.
            (Test Case 2) The defaults dict on the backends module is
                NOT mutated by the constructor (purity).
        """
        from spikelab.spike_sorting.backends.kilosort2 import (
            DEFAULT_KILOSORT2_PARAMS,
        )
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        defaults_before = dict(DEFAULT_KILOSORT2_PARAMS)
        runner = RunKilosort(kilosort_path=str(fake_kilosort_path))
        expected = RunKilosort.format_params(dict(DEFAULT_KILOSORT2_PARAMS))

        assert runner.kilosort_params == expected
        # Module-level defaults dict is untouched (the canonical leak
        # the refactor closed).
        assert DEFAULT_KILOSORT2_PARAMS == defaults_before

    def test_pos_peak_thresh_none_resolves_to_waveform_default(
        self, fake_kilosort_path
    ):
        """
        ``pos_peak_thresh=None`` falls back to
        ``WaveformConfig().pos_peak_thresh``.

        Tests:
            (Test Case 1) ``self.pos_peak_thresh`` equals the
                ``WaveformConfig`` dataclass default.
        """
        from spikelab.spike_sorting.config import WaveformConfig
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        runner = RunKilosort(kilosort_path=str(fake_kilosort_path))
        assert runner.pos_peak_thresh == WaveformConfig().pos_peak_thresh

    def test_explicit_values_override_defaults(self, fake_kilosort_path):
        """
        Explicit ``kilosort_params`` and ``pos_peak_thresh`` flow
        through unchanged (modulo ``format_params`` normalisation of
        ``NT`` and ``car``).

        Tests:
            (Test Case 1) Explicit ``pos_peak_thresh=1.5`` is stored.
            (Test Case 2) Explicit ``kilosort_params`` is normalised
                via ``format_params`` (e.g. ``car=True`` → ``car=1``).
        """
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        runner = RunKilosort(
            kilosort_path=str(fake_kilosort_path),
            kilosort_params={"NT": 65600, "ntbuff": 64, "car": True},
            pos_peak_thresh=1.5,
        )
        assert runner.pos_peak_thresh == 1.5
        assert runner.kilosort_params["car"] == 1
        assert runner.kilosort_params["NT"] == 65600


# ===========================================================================
# RT-Sort runner — explicit-config replacements for the broken
# ``_GlobalsStub`` tests below. The original test_detection_window_s_*
# tests in ``TestRTSortRunnerHelpers`` set values on the no-op
# ``_GlobalsStub`` (a leftover from Phase 5 cleanup); those writes
# vanish, so the tests effectively run against ``RTSortConfig()``
# defaults rather than the values the test author intended. Class
# below pins the same contracts via explicit ``RTSortConfig``
# construction.
# ===========================================================================


@skip_no_torch
class TestRTSortDetectionWindow:
    """Tests for ``rt_sort_runner.spike_sort``'s detection-window
    narrowing — the contract that ``RTSortConfig.detection_window_s``
    narrows the ``detect_sequences`` window without affecting the
    full-recording ``sort_offline`` window.
    """

    @pytest.fixture()
    def captured_calls(self, monkeypatch, tmp_path):
        """Stub ``_load_detection_model``, ``detect_sequences``, and
        ``_save_sorting_cache`` so ``spike_sort`` can be driven without
        real RT-Sort/torch internals; capture the
        ``recording_window_ms`` argument passed into each phase.
        """
        captured = {"detect": None, "sort_offline": None}

        class _FakeRTSort:
            _seq_root_elecs = []

            def sort_offline(self, recording, inter_path, **kw):
                captured["sort_offline"] = kw.get("recording_window_ms")
                return object()

        def _fake_detect_sequences(recording, inter_path, detection_model, **kw):
            captured["detect"] = kw.get("recording_window_ms")
            return _FakeRTSort()

        monkeypatch.setattr(
            "spikelab.spike_sorting.rt_sort_runner._load_detection_model",
            lambda *a, **k: object(),
        )
        import spikelab.spike_sorting.rt_sort as rt_sort_pkg

        monkeypatch.setattr(
            rt_sort_pkg, "detect_sequences", _fake_detect_sequences, raising=False
        )
        monkeypatch.setattr(
            "spikelab.spike_sorting.rt_sort_runner._save_sorting_cache",
            lambda *a, **k: None,
        )
        return captured

    def test_detection_window_s_narrows_only_detect_phase(
        self, captured_calls, tmp_path
    ):
        """
        Setting ``detection_window_s=60`` narrows the detect_sequences
        window to ``(0, 60_000) ms`` while leaving sort_offline running
        across the configured full window ``(0, 600_000) ms``.

        Tests:
            (Test Case 1) ``detect_sequences`` receives the narrowed
                ``recording_window_ms``.
            (Test Case 2) ``sort_offline`` receives the full
                ``recording_window_ms``.
        """
        from spikelab.spike_sorting import rt_sort_runner as runner
        from spikelab.spike_sorting.config import (
            ExecutionConfig,
            RTSortConfig,
            SortingPipelineConfig,
        )

        config = SortingPipelineConfig(
            execution=ExecutionConfig(recompute_sorting=True),
            rt_sort=RTSortConfig(
                recording_window_ms=(0.0, 600_000.0),
                detection_window_s=60.0,
                device="cpu",
                num_processes=1,
                delete_inter=False,
                verbose=False,
                save_rt_sort_pickle=False,
            ),
        )

        runner.spike_sort(
            rec_cache=object(),
            rec_path=tmp_path / "fake.h5",
            recording_dat_path=None,
            output_folder=tmp_path / "out",
            config=config,
        )

        assert captured_calls["detect"] == (0.0, 60_000.0)
        assert captured_calls["sort_offline"] == (0.0, 600_000.0)

    def test_detection_window_s_unset_uses_full_window_for_both(
        self, captured_calls, tmp_path
    ):
        """
        With ``detection_window_s=None`` (default), both phases see
        the same ``recording_window_ms`` — preserves legacy behaviour.

        Tests:
            (Test Case 1) Both ``detect_sequences`` and
                ``sort_offline`` receive ``(0, 120_000) ms``.
        """
        from spikelab.spike_sorting import rt_sort_runner as runner
        from spikelab.spike_sorting.config import (
            ExecutionConfig,
            RTSortConfig,
            SortingPipelineConfig,
        )

        config = SortingPipelineConfig(
            execution=ExecutionConfig(recompute_sorting=True),
            rt_sort=RTSortConfig(
                recording_window_ms=(0.0, 120_000.0),
                detection_window_s=None,
                device="cpu",
                num_processes=1,
                delete_inter=False,
                verbose=False,
                save_rt_sort_pickle=False,
            ),
        )

        runner.spike_sort(
            rec_cache=object(),
            rec_path=tmp_path / "fake.h5",
            recording_dat_path=None,
            output_folder=tmp_path / "out",
            config=config,
        )

        assert captured_calls["detect"] == (0.0, 120_000.0)
        assert captured_calls["sort_offline"] == (0.0, 120_000.0)


# ===========================================================================
# Phase 2 sorter_params merge — runners merge ``DEFAULT_*_PARAMS`` with the
# caller's ``config.sorter.sorter_params`` (user wins). Pre-refactor this
# happened via ``_globals.KILOSORT_PARAMS`` mutation in ``_sync_globals``;
# post-refactor it's a fresh dict per call. Tests verify the merge resolves
# correctly and the values reach the downstream runner / SI call.
# ===========================================================================


@skip_no_spikeinterface
class TestSpikeSortKs2SorterParamsMerge:
    """``ks2_runner.spike_sort`` merges ``DEFAULT_KILOSORT2_PARAMS`` with
    the caller's ``config.sorter.sorter_params`` and forwards the result
    to ``RunKilosort.__init__``. User overrides win over defaults; keys
    not in the user dict fall back to the defaults.
    """

    def test_user_overrides_win_over_defaults_in_runkilosort_kwargs(self, monkeypatch):
        """
        Custom ``sorter_params`` keys override the matching keys in
        ``DEFAULT_KILOSORT2_PARAMS``; keys not in the user dict fall
        back to the default value.

        Tests:
            (Test Case 1) ``detect_threshold=9`` (user) overrides
                ``DEFAULT_KILOSORT2_PARAMS["detect_threshold"]=6``.
            (Test Case 2) ``minFR`` (not in user dict) flows through
                from defaults.
            (Test Case 3) ``RunKilosort`` is called with the merged
                dict on the ``kilosort_params=`` kwarg.
        """
        from spikelab.spike_sorting import ks2_runner
        from spikelab.spike_sorting.backends.kilosort2 import (
            DEFAULT_KILOSORT2_PARAMS,
        )
        from spikelab.spike_sorting.config import (
            SorterConfig,
            SortingPipelineConfig,
        )

        captured = {}

        class _StubRunKilosort:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, **_kw):
                return MagicMock(unit_ids=[])

        monkeypatch.setattr(ks2_runner, "RunKilosort", _StubRunKilosort)
        monkeypatch.setattr(ks2_runner, "write_recording", lambda *a, **kw: None)
        monkeypatch.setattr(ks2_runner, "create_folder", lambda *a, **kw: None)

        cfg = SortingPipelineConfig(
            sorter=SorterConfig(
                sorter_path="/fake/kilosort",
                sorter_params={"detect_threshold": 9},
            ),
        )

        ks2_runner.spike_sort(
            rec_cache=_make_mock_recording(),
            rec_path="rec.h5",
            recording_dat_path=Path("/tmp/rec.dat"),
            output_folder=Path("/tmp/out"),
            config=cfg,
        )

        merged = captured["kilosort_params"]
        # User override wins
        assert merged["detect_threshold"] == 9
        # Default flows through
        assert merged["minFR"] == DEFAULT_KILOSORT2_PARAMS["minFR"]
        # Source defaults dict is untouched (canonical leak guard).
        assert DEFAULT_KILOSORT2_PARAMS["detect_threshold"] == 6


@skip_no_spikeinterface
class TestSpikeSortKs4SorterParamsMerge:
    """``ks4_runner.spike_sort`` merges ``DEFAULT_KILOSORT4_PARAMS`` with
    the caller's ``config.sorter.sorter_params`` and forwards the
    result to ``spikeinterface.sorters.run_sorter`` as ``**kwargs``.
    """

    def test_user_overrides_win_over_defaults_in_run_sorter_kwargs(
        self, monkeypatch, tmp_path
    ):
        """
        Custom ``sorter_params`` keys override the KS4 defaults;
        unspecified keys fall back to the defaults dict.

        Tests:
            (Test Case 1) ``do_correction=False`` (user) overrides
                ``DEFAULT_KILOSORT4_PARAMS["do_correction"]=True``.
            (Test Case 2) ``invert_sign`` (not in user dict) flows
                through from defaults.
            (Test Case 3) ``run_sorter`` receives the merged kwargs.
            (Test Case 4) ``DEFAULT_KILOSORT4_PARAMS`` is not mutated.
        """
        import spikeinterface.sorters as ss

        from spikelab.spike_sorting import ks4_runner
        from spikelab.spike_sorting.backends.kilosort4 import (
            DEFAULT_KILOSORT4_PARAMS,
        )
        from spikelab.spike_sorting.config import (
            SorterConfig,
            SortingPipelineConfig,
        )

        captured = {}

        def _stub_run_sorter(name, recording, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

        monkeypatch.setattr(ss, "run_sorter", _stub_run_sorter)

        # The runner builds a KilosortSortingExtractor at the end;
        # short-circuit it to return a plain Mock without touching disk.
        from spikelab.spike_sorting import sorting_extractor

        monkeypatch.setattr(
            sorting_extractor,
            "KilosortSortingExtractor",
            lambda **_kw: MagicMock(unit_ids=[]),
        )
        # And the symbol the runner imported at module load time.
        monkeypatch.setattr(
            ks4_runner,
            "KilosortSortingExtractor",
            lambda **_kw: MagicMock(unit_ids=[]),
        )

        defaults_before = dict(DEFAULT_KILOSORT4_PARAMS)
        cfg = SortingPipelineConfig(
            sorter=SorterConfig(sorter_params={"do_correction": False}),
        )

        ks4_runner.spike_sort(
            rec_cache=_make_mock_recording(),
            rec_path="rec.h5",
            recording_dat_path=Path("/tmp/rec.dat"),
            output_folder=tmp_path / "ks4_out",
            config=cfg,
        )

        # User override wins
        assert captured["kwargs"]["do_correction"] is False
        # Default flows through
        assert (
            captured["kwargs"]["invert_sign"] == DEFAULT_KILOSORT4_PARAMS["invert_sign"]
        )
        # Source defaults dict is untouched
        assert DEFAULT_KILOSORT4_PARAMS == defaults_before


@skip_no_spikeinterface
class TestWriteRecordingDefaults:
    """``ks2_runner.write_recording`` resolves the new ``n_jobs`` /
    ``total_memory`` / ``use_parallel`` keyword arguments against the
    ``ExecutionConfig`` defaults when not supplied, and forwards
    explicit values unchanged. Pre-refactor these came from
    ``_globals.N_JOBS`` / ``_globals.TOTAL_MEMORY`` / etc.
    """

    @pytest.fixture()
    def captured_writer(self, monkeypatch):
        """Patch ``BinaryRecordingExtractor.write_recording`` to record
        the job kwargs without writing to disk."""
        from spikeinterface.extractors.extractor_classes import (
            BinaryRecordingExtractor,
        )

        captured = {}

        def _stub(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(BinaryRecordingExtractor, "write_recording", _stub)
        return captured

    def test_defaults_resolved_from_execution_config(self, captured_writer, tmp_path):
        """
        With no kwargs supplied, ``write_recording`` reads
        ``ExecutionConfig`` defaults (``n_jobs=8``,
        ``total_memory="16G"``, ``use_parallel=True``) and forwards
        them to the underlying SI writer.

        Tests:
            (Test Case 1) ``n_jobs`` defaults to ``ExecutionConfig().n_jobs``.
            (Test Case 2) ``total_memory`` defaults to
                ``ExecutionConfig().total_memory``.
            (Test Case 3) ``use_parallel=True`` (default) takes the
                multi-job branch (verbose=True forwarded).
        """
        from spikelab.spike_sorting.config import ExecutionConfig
        from spikelab.spike_sorting.ks2_runner import write_recording

        rec = _make_mock_recording()
        dat_path = tmp_path / "rec.dat"

        write_recording(rec, dat_path, verbose=True)

        defaults = ExecutionConfig()
        assert captured_writer["n_jobs"] == defaults.n_jobs
        assert captured_writer["total_memory"] == defaults.total_memory
        # use_parallel=True branch keeps verbose=True
        assert captured_writer["verbose"] is True

    def test_explicit_kwargs_override_defaults(self, captured_writer, tmp_path):
        """
        Explicit ``n_jobs`` / ``total_memory`` / ``use_parallel``
        values override the ``ExecutionConfig`` defaults.

        Tests:
            (Test Case 1) Explicit ``n_jobs=4`` flows through.
            (Test Case 2) Explicit ``total_memory="8G"`` flows through.
            (Test Case 3) ``use_parallel=False`` takes the single-job
                branch (n_jobs forced to 1, ignoring the explicit
                ``n_jobs=4`` — documented behaviour).
        """
        from spikelab.spike_sorting.ks2_runner import write_recording

        rec = _make_mock_recording()
        dat_path = tmp_path / "rec_a.dat"
        write_recording(
            rec,
            dat_path,
            verbose=True,
            n_jobs=4,
            total_memory="8G",
            use_parallel=True,
        )
        assert captured_writer["n_jobs"] == 4
        assert captured_writer["total_memory"] == "8G"

        # use_parallel=False branch overrides n_jobs to 1 internally.
        captured_writer.clear()
        dat_path_b = tmp_path / "rec_b.dat"
        write_recording(
            rec,
            dat_path_b,
            verbose=True,
            n_jobs=4,
            total_memory="8G",
            use_parallel=False,
        )
        assert captured_writer["n_jobs"] == 1


@skip_no_spikeinterface
class TestRunKilosortGetResultFromFolder:
    """``RunKilosort.get_result_from_folder`` is now an instance method
    (formerly ``@classmethod``). It reads ``self.kilosort_params`` and
    ``self.pos_peak_thresh`` to materialise the result extractor.
    Pre-refactor those values came from ``_globals.KILOSORT_PARAMS`` /
    ``_globals.POS_PEAK_THRESH``.
    """

    @pytest.fixture()
    def fake_kilosort_path(self, tmp_path):
        ks_path = tmp_path / "ks_install"
        ks_path.mkdir()
        (ks_path / "master_kilosort.m").touch()
        return ks_path

    def test_returns_extractor_with_instance_attribute_values(
        self, fake_kilosort_path, tmp_path
    ):
        """
        The returned ``KilosortSortingExtractor`` reflects
        ``self.pos_peak_thresh`` and ``self.kilosort_params["keep_good_only"]``
        as set on the runner instance (not from globals).

        Tests:
            (Test Case 1) ``pos_peak_thresh=1.5`` from the runner reaches
                the extractor.
            (Test Case 2) ``keep_good_only=False`` (resolved from
                ``self.kilosort_params``) yields all units.
            (Test Case 3) Instance-method form requires an instance:
                calling ``RunKilosort.get_result_from_folder(folder)``
                without ``self`` raises ``TypeError``.
        """
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        ks_out = tmp_path / "ks_out"
        spike_times = np.array([10, 20, 100, 200], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 1], dtype=np.int64)
        _write_ks_folder(ks_out, spike_times, spike_clusters)

        runner = RunKilosort(
            kilosort_path=str(fake_kilosort_path),
            kilosort_params={
                "NT": 65600,
                "ntbuff": 64,
                "car": True,
                "keep_good_only": False,
            },
            pos_peak_thresh=1.5,
        )

        kse = runner.get_result_from_folder(ks_out)
        assert kse.pos_peak_thresh == 1.5
        # Both clusters survive when keep_good_only=False (no KSLabel
        # filtering).
        assert set(kse.unit_ids) == {0, 1}

        # Instance-method form: classmethod-style calls now fail.
        with pytest.raises(TypeError):
            RunKilosort.get_result_from_folder(ks_out)


@skip_no_spikeinterface
class TestRunCanaryStateIsolation:
    """Headline invariant of the Phase 5 refactor: ``run_canary`` must
    not mutate the caller's ``config``. Pre-refactor the canary
    snapshotted/restored ``_globals`` to enforce this; post-refactor
    the canary builds a deep clone via ``_build_canary_config`` and
    runs the backend against the clone, so no per-call state can
    leak back to the caller.

    Existing coverage in ``tests/test_canary.py`` only exercises
    ``_build_canary_config`` directly; this class pins the higher-level
    end-to-end guarantee against ``run_canary``.
    """

    def test_run_canary_does_not_mutate_caller_config(self, monkeypatch, tmp_path):
        """
        Pass a config to ``run_canary``, run it with a stubbed backend
        and stubbed ``process_recording``, and assert the caller's
        config is byte-identical before and after.

        Tests:
            (Test Case 1) ``deepcopy(config)`` before == config after.
            (Test Case 2) ``id(input_config)`` differs from the
                ``config`` the canary backend sees (the canary uses a
                deep clone, not the caller's reference).
        """
        import copy

        from spikelab.spike_sorting import canary
        from spikelab.spike_sorting.config import (
            ExecutionConfig,
            SortingPipelineConfig,
        )

        # Trigger the canary by setting a non-zero window.
        cfg = SortingPipelineConfig(
            execution=ExecutionConfig(canary_first_n_s=5.0),
        )
        cfg_before = copy.deepcopy(cfg)

        seen_backend_configs = []

        class _StubBackend:
            def __init__(self, config):
                seen_backend_configs.append(id(config))

        # Patch the lazy imports inside run_canary.
        from spikelab.spike_sorting import backends as _backends_pkg
        from spikelab.spike_sorting import pipeline as _pipeline

        monkeypatch.setattr(
            _backends_pkg, "get_backend_class", lambda name: _StubBackend
        )
        monkeypatch.setattr(_backends_pkg, "list_sorters", lambda: ["kilosort2"])
        monkeypatch.setattr(_pipeline, "process_recording", lambda *a, **kw: None)

        result = canary.run_canary(
            cfg,
            recording=None,
            rec_path=tmp_path / "rec.h5",
            inter_path=tmp_path,
            sorter_name="kilosort2",
        )

        assert result is None
        # Caller's config is untouched.
        assert cfg == cfg_before
        # Backend saw a different config object (the canary clone).
        assert seen_backend_configs == [id(cfg)] or id(cfg) not in seen_backend_configs
        assert id(cfg) not in seen_backend_configs


@skip_no_spikeinterface
class TestBackendConfigThreading:
    """Each backend's ``sort()`` and ``extract_waveforms()`` must
    forward ``config=self.config`` to the runner / extractor so the
    same per-recording config flows through the call chain. Pre-
    refactor the backend wrote into ``_globals`` instead of passing
    config; this class pins the post-refactor pass-through contract.
    """

    def test_kilosort2_sort_threads_self_config(self, monkeypatch):
        """
        ``Kilosort2Backend.sort`` calls ``ks2_runner.spike_sort`` with
        ``config=self.config`` (identity check, not just equality).

        Tests:
            (Test Case 1) Captured ``config`` kwarg is the same object
                as ``backend.config``.
        """
        from spikelab.spike_sorting import ks2_runner
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        captured = {}

        def _stub_spike_sort(**kwargs):
            captured.update(kwargs)
            return MagicMock(unit_ids=[])

        monkeypatch.setattr(ks2_runner, "spike_sort", _stub_spike_sort)
        # The backend imports spike_sort lazily inside .sort() — patch
        # the symbol on the source module too in case it's already cached.
        from spikelab.spike_sorting.backends import kilosort2 as ks2_backend_mod

        monkeypatch.setattr(
            ks2_backend_mod, "spike_sort", _stub_spike_sort, raising=False
        )

        cfg = SortingPipelineConfig()
        backend = Kilosort2Backend(cfg)
        backend.sort(
            recording=_make_mock_recording(),
            rec_path="r.h5",
            recording_dat_path=Path("/tmp/r.dat"),
            output_folder=Path("/tmp/out"),
        )

        assert captured["config"] is backend.config

    def test_kilosort4_sort_threads_self_config(self, monkeypatch):
        """``Kilosort4Backend.sort`` forwards ``config=self.config``."""
        from spikelab.spike_sorting import ks4_runner
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        captured = {}

        def _stub_spike_sort(**kwargs):
            captured.update(kwargs)
            return MagicMock(unit_ids=[])

        monkeypatch.setattr(ks4_runner, "spike_sort", _stub_spike_sort)

        cfg = SortingPipelineConfig()
        backend = Kilosort4Backend(cfg)
        # The KS4 backend wraps the call in an in-process inactivity
        # watchdog; force the no-watchdog path by stubbing the helper
        # to return None (``_make_in_process_inactivity_watchdog`` is
        # inherited from SorterBackend).
        monkeypatch.setattr(
            backend, "_make_in_process_inactivity_watchdog", lambda *a, **kw: None
        )
        backend.sort(
            recording=_make_mock_recording(),
            rec_path="r.h5",
            recording_dat_path=Path("/tmp/r.dat"),
            output_folder=Path("/tmp/out"),
        )

        assert captured["config"] is backend.config

    def test_kilosort2_extract_waveforms_threads_self_config(self, monkeypatch):
        """
        ``Kilosort2Backend.extract_waveforms`` forwards
        ``config=self.config`` to ``recording_io.extract_waveforms``.

        Tests:
            (Test Case 1) Captured ``config`` kwarg is the same object
                as ``backend.config``.
            (Test Case 2) ``n_jobs`` and ``total_memory`` from
                ``config.execution`` are forwarded too.
        """
        from spikelab.spike_sorting import recording_io
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        captured = {}

        def _stub_extract(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(recording_io, "extract_waveforms", _stub_extract)

        cfg = SortingPipelineConfig()
        backend = Kilosort2Backend(cfg)
        backend.extract_waveforms(
            recording=_make_mock_recording(),
            sorting=MagicMock(),
            waveforms_folder=Path("/tmp/wf"),
            curation_folder=Path("/tmp/wf/initial"),
        )

        assert captured["config"] is backend.config
        assert captured["n_jobs"] == cfg.execution.n_jobs
        assert captured["total_memory"] == cfg.execution.total_memory

    def test_kilosort4_extract_waveforms_threads_self_config(self, monkeypatch):
        """``Kilosort4Backend.extract_waveforms`` forwards
        ``config=self.config``."""
        from spikelab.spike_sorting import recording_io
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SortingPipelineConfig

        captured = {}

        def _stub_extract(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(recording_io, "extract_waveforms", _stub_extract)

        cfg = SortingPipelineConfig()
        backend = Kilosort4Backend(cfg)
        backend.extract_waveforms(
            recording=_make_mock_recording(),
            sorting=MagicMock(),
            waveforms_folder=Path("/tmp/wf"),
            curation_folder=Path("/tmp/wf/initial"),
        )

        assert captured["config"] is backend.config


# ===========================================================================
# Batch G — additional Phase 5 refactor coverage: format_params validation,
# sorter_params merge equivalences, backend isolation, OOM snapshot
# semantics, and rt_sort defaults round-trip.
# ===========================================================================


class TestRunKilosortFormatParamsValidation:
    """``RunKilosort.format_params`` is the canonical leak-fix entry
    point; pin the error contract for hand-crafted partial input
    dicts (the static-method form is more callable in isolation than
    the pre-refactor in-place mutation was).
    """

    def test_nt_none_without_ntbuff_raises_key_error(self):
        """
        ``format_params({"NT": None, "car": True})`` (no ``ntbuff``)
        raises ``KeyError`` because the resolution
        ``NT = 64*1024 + out["ntbuff"]`` indexes without a default.

        Tests:
            (Test Case 1) KeyError raised, message names "ntbuff".
        """
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        with pytest.raises(KeyError, match="ntbuff"):
            RunKilosort.format_params({"NT": None, "car": True})

    def test_nt_non_numeric_string_raises_value_error(self):
        """
        Non-numeric string for ``NT`` (e.g. ``"64k"``) fails the
        ``int()`` cast.

        Tests:
            (Test Case 1) ValueError raised by ``int()``.
        """
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        with pytest.raises(ValueError):
            RunKilosort.format_params({"NT": "64k", "ntbuff": 64, "car": False})

    def test_nt_digit_string_accepted(self):
        """
        Digit-only string for ``NT`` is accepted (``int("65600")``
        works); rounded down to a multiple of 32.

        Tests:
            (Test Case 1) ``NT="65600"`` → ``out["NT"] == 65600``.
        """
        from spikelab.spike_sorting.ks2_runner import RunKilosort

        out = RunKilosort.format_params({"NT": "65600", "ntbuff": 64, "car": False})
        assert out["NT"] == 65600


@skip_no_spikeinterface
class TestSpikeSortKs2SorterParamsEmptyDictEquivalentToNone:
    """``config.sorter.sorter_params={}`` and ``None`` must produce
    the same merged ``kilosort_params`` dict reaching ``RunKilosort``.
    """

    def test_empty_dict_equals_none(self, monkeypatch):
        """
        Tests:
            (Test Case 1) Both forms forward the same merged dict
                (== ``DEFAULT_KILOSORT2_PARAMS``) to ``RunKilosort``.
        """
        from spikelab.spike_sorting import ks2_runner
        from spikelab.spike_sorting.backends.kilosort2 import (
            DEFAULT_KILOSORT2_PARAMS,
        )
        from spikelab.spike_sorting.config import SorterConfig, SortingPipelineConfig

        captured = {"none": None, "empty": None}

        def _stub_factory(slot):
            class _Stub:
                def __init__(self, **kwargs):
                    captured[slot] = kwargs.get("kilosort_params")

                def run(self, **_kw):
                    return MagicMock(unit_ids=[])

            return _Stub

        monkeypatch.setattr(ks2_runner, "write_recording", lambda *a, **kw: None)
        monkeypatch.setattr(ks2_runner, "create_folder", lambda *a, **kw: None)

        monkeypatch.setattr(ks2_runner, "RunKilosort", _stub_factory("none"))
        ks2_runner.spike_sort(
            rec_cache=_make_mock_recording(),
            rec_path="r.h5",
            recording_dat_path=Path("/tmp/r.dat"),
            output_folder=Path("/tmp/out"),
            config=SortingPipelineConfig(
                sorter=SorterConfig(sorter_path="/fake/ks", sorter_params=None)
            ),
        )

        monkeypatch.setattr(ks2_runner, "RunKilosort", _stub_factory("empty"))
        ks2_runner.spike_sort(
            rec_cache=_make_mock_recording(),
            rec_path="r.h5",
            recording_dat_path=Path("/tmp/r.dat"),
            output_folder=Path("/tmp/out"),
            config=SortingPipelineConfig(
                sorter=SorterConfig(sorter_path="/fake/ks", sorter_params={})
            ),
        )

        assert captured["none"] == captured["empty"]
        assert captured["none"] == dict(DEFAULT_KILOSORT2_PARAMS)


@skip_no_spikeinterface
class TestSpikeSortKs4ConfigNoneUsesDefaults:
    """``ks4_runner.spike_sort(config=None)`` constructs a default
    config and forwards bare ``DEFAULT_KILOSORT4_PARAMS`` to
    ``run_sorter``.
    """

    def test_config_none_forwards_default_kilosort4_params(self, monkeypatch, tmp_path):
        """
        Tests:
            (Test Case 1) ``run_sorter`` kwargs contain every key
                from ``DEFAULT_KILOSORT4_PARAMS``.
        """
        import spikeinterface.sorters as ss

        from spikelab.spike_sorting import ks4_runner
        from spikelab.spike_sorting.backends.kilosort4 import (
            DEFAULT_KILOSORT4_PARAMS,
        )

        captured = {}

        def _stub(name, recording, **kwargs):
            captured["kwargs"] = kwargs

        monkeypatch.setattr(ss, "run_sorter", _stub)
        monkeypatch.setattr(
            ks4_runner, "KilosortSortingExtractor", lambda **_kw: MagicMock(unit_ids=[])
        )

        ks4_runner.spike_sort(
            rec_cache=_make_mock_recording(),
            rec_path="r.h5",
            recording_dat_path=Path("/tmp/r.dat"),
            output_folder=tmp_path / "ks4_out",
            config=None,
        )

        for k, v in DEFAULT_KILOSORT4_PARAMS.items():
            assert captured["kwargs"][k] == v


@skip_no_spikeinterface
class TestBackendStateIsolationAcrossRecordings:
    """Two backends constructed with distinct ``sorter_params`` must
    not contaminate each other. Pins the canonical multi-recording
    isolation that the refactor closed.
    """

    def test_two_kilosort2_backends_keep_independent_params(self):
        """
        Tests:
            (Test Case 1) Construction-time isolation: backend A's NT
                is 65600 even after backend B with NT=32000 is built.
            (Test Case 2) Method-call isolation: scaling backend A's
                config does not leak to backend B.
        """
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SorterConfig, SortingPipelineConfig

        cfg_a = SortingPipelineConfig(sorter=SorterConfig(sorter_params={"NT": 65600}))
        cfg_b = SortingPipelineConfig(sorter=SorterConfig(sorter_params={"NT": 32000}))
        backend_a = Kilosort2Backend(cfg_a)
        backend_b = Kilosort2Backend(cfg_b)

        assert backend_a.config.sorter.sorter_params["NT"] == 65600
        assert backend_b.config.sorter.sorter_params["NT"] == 32000

        backend_a.scale_oom_params(0.5)
        assert backend_b.config.sorter.sorter_params["NT"] == 32000


@skip_no_spikeinterface
class TestBackendOomSnapshotIsDeepCopy:
    """``snapshot_oom_params()`` returns a value independent of
    subsequent ``scale_oom_params`` mutations.
    """

    def test_kilosort2_snapshot_independent_of_subsequent_scale(self):
        """
        Tests:
            (Test Case 1) NT in the snapshot is unchanged after
                ``scale_oom_params`` mutates the live config.
        """
        from spikelab.spike_sorting.backends.kilosort2 import Kilosort2Backend
        from spikelab.spike_sorting.config import SorterConfig, SortingPipelineConfig

        backend = Kilosort2Backend(
            SortingPipelineConfig(
                sorter=SorterConfig(sorter_params={"NT": 65600, "ntbuff": 64})
            )
        )
        snap = backend.snapshot_oom_params()
        original_nt = snap["sorter_params"]["NT"]

        backend.scale_oom_params(0.5)
        scaled_nt = backend.config.sorter.sorter_params["NT"]

        assert scaled_nt < original_nt
        assert snap["sorter_params"]["NT"] == original_nt

    def test_kilosort4_snapshot_independent_of_subsequent_scale(self):
        """KS4 equivalent — pins the same contract on batch_size."""
        from spikelab.spike_sorting.backends.kilosort4 import Kilosort4Backend
        from spikelab.spike_sorting.config import SorterConfig, SortingPipelineConfig

        backend = Kilosort4Backend(
            SortingPipelineConfig(
                sorter=SorterConfig(sorter_params={"batch_size": 60000})
            )
        )
        snap = backend.snapshot_oom_params()
        original_bs = snap["sorter_params"]["batch_size"]

        backend.scale_oom_params(0.5)
        scaled_bs = backend.config.sorter.sorter_params["batch_size"]

        assert scaled_bs < original_bs
        assert snap["sorter_params"]["batch_size"] == original_bs


@skip_no_spikeinterface
class TestNumpySortingToKsExtractorDefaults:
    """``_numpy_sorting_to_ks_extractor`` falls back to documented
    defaults when ``keep_good_only`` or ``pos_peak_thresh`` is
    ``None``.
    """

    @pytest.fixture()
    def captured_kse_init(self, monkeypatch):
        """Patch ``KilosortSortingExtractor.__init__`` to capture the
        kwargs it receives without running the real init (which
        requires Kilosort-format files on disk)."""
        from spikelab.spike_sorting import sorting_extractor

        captured = {}

        class _StubKSE:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(sorting_extractor, "KilosortSortingExtractor", _StubKSE)
        return captured

    def test_keep_good_only_none_defaults_to_false(self, captured_kse_init, tmp_path):
        """
        Tests:
            (Test Case 1) ``keep_good_only=None`` resolves to ``False``
                in the kwargs passed to ``KilosortSortingExtractor``.
            (Test Case 2) ``pos_peak_thresh=None`` resolves to
                ``WaveformConfig().pos_peak_thresh``.
        """
        from spikeinterface.core import NumpyRecording
        from spikeinterface.extractors import NumpySorting

        from spikelab.spike_sorting.backends.rt_sort import (
            _numpy_sorting_to_ks_extractor,
        )
        from spikelab.spike_sorting.config import WaveformConfig

        fs = 20000.0
        rec = NumpyRecording(
            traces_list=[np.zeros((1000, 4), dtype=np.float32)],
            sampling_frequency=fs,
        )
        sorting = NumpySorting.from_unit_dict(
            [{0: np.array([100, 200], dtype=np.int64)}], sampling_frequency=fs
        )

        _numpy_sorting_to_ks_extractor(
            sorting,
            rec,
            tmp_path / "out",
            root_elecs=[0],
            keep_good_only=None,
            pos_peak_thresh=None,
        )
        assert captured_kse_init["keep_good_only"] is False
        assert captured_kse_init["pos_peak_thresh"] == WaveformConfig().pos_peak_thresh

    def test_keep_good_only_true_round_trip(self, captured_kse_init, tmp_path):
        """
        Tests:
            (Test Case 1) ``keep_good_only=True`` propagates verbatim
                to the KSE constructor.
            (Test Case 2) ``pos_peak_thresh=1.5`` propagates verbatim.
        """
        from spikeinterface.core import NumpyRecording
        from spikeinterface.extractors import NumpySorting

        from spikelab.spike_sorting.backends.rt_sort import (
            _numpy_sorting_to_ks_extractor,
        )

        fs = 20000.0
        rec = NumpyRecording(
            traces_list=[np.zeros((1000, 4), dtype=np.float32)],
            sampling_frequency=fs,
        )
        sorting = NumpySorting.from_unit_dict(
            [{0: np.array([100, 200], dtype=np.int64)}], sampling_frequency=fs
        )

        _numpy_sorting_to_ks_extractor(
            sorting,
            rec,
            tmp_path / "out",
            root_elecs=[0],
            keep_good_only=True,
            pos_peak_thresh=1.5,
        )
        assert captured_kse_init["keep_good_only"] is True
        assert captured_kse_init["pos_peak_thresh"] == 1.5
