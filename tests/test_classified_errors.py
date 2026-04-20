"""Tests for spike_sorting._classifier and ._exceptions.

The classifiers inspect sorter logs and exception chains to re-raise
generic failures as specific classified exceptions. Tests exercise:

* The class hierarchy (subclass relationships).
* Each positive classifier branch with realistic signatures.
* Negative controls so real tooling bugs are not masked.
* Dispatcher priority (environment / resource / biology order).
* Cross-module behaviour: curation raises
  :class:`EmptyWaveformMetricsError` directly and preserves its
  historical ``ValueError`` identity.

All tests run without MATLAB, Docker, a GPU, or network access — the
classifier is pure log/exception inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from spikelab.spike_sorting._classifier import (
    _walk_exception_chain,
    classify_ks2_failure,
    classify_ks4_failure,
)
from spikelab.spike_sorting._exceptions import (
    BiologicalSortFailure,
    DockerEnvironmentError,
    EmptyWaveformMetricsError,
    EnvironmentSortFailure,
    GPUOutOfMemoryError,
    HDF5PluginMissingError,
    InsufficientActivityError,
    NoGoodChannelsError,
    ResourceSortFailure,
    SaturatedSignalError,
    SpikeSortingClassifiedError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_ks2_log_matlab(folder: Path, text: str) -> Path:
    """Write a kilosort2.log at the MATLAB-path location."""
    path = folder / "kilosort2.log"
    path.write_text(text)
    return path


def _write_ks2_log_docker(folder: Path, text: str) -> Path:
    """Write a kilosort2.log at the Docker-path location."""
    (folder / "sorter_output").mkdir(parents=True, exist_ok=True)
    path = folder / "sorter_output" / "kilosort2.log"
    path.write_text(text)
    return path


def _chain(messages: List[str]) -> BaseException:
    """Build a ``__cause__``-linked exception chain from a list of messages."""
    inner: Optional[BaseException] = None
    for msg in messages:
        new: BaseException = RuntimeError(msg) if inner is not None else ValueError(msg)
        if inner is not None:
            new.__cause__ = inner
        inner = new
    assert inner is not None
    return inner


_KS2_SPARSE_LOG = """Warning: X does not support locale C.UTF-8
Time   0s. Determining good channels..
Recording has 974 channels
found 1346 threshold crossings in 299.96 seconds of data
found 966 bad channels
Time 364s. Computing whitening matrix..
random seed for clusterSingleBatches: 1
time 1.80, Re-ordered 46 batches.
Time   2s. Optimizing templates ...
2.53 sec, 1 / 46 batches, 2 units, nspks: 1.3292, mu: 10.9917, nst0: 1, merges: 0.0000, 0.0000
----------------------------------------Error using indexing
An unexpected error occurred trying to launch a kernel. The CUDA error was:
invalid configuration argument
"""

_KS2_HEALTHY_LOG_WITH_CUDA_ERROR = """Time 0s.
Recording has 974 channels
found 1800000 threshold crossings in 300 seconds of data
found 12 bad channels
Time   2s. Optimizing templates ...
2.53 sec, 1 / 46 batches, 423 units, nspks: 123.45
CUDA kernel launched strangely
invalid configuration argument
"""

_KS2_ALL_BAD_CHANNELS_LOG = """Recording has 974 channels
found 974 bad channels
Aborting because no good channels remained.
"""

_KS2_ZERO_GOOD_CHANNELS_LOG = """Time 0s. Determining good channels..
found 0 good channels
some fatal error
"""

_KS2_PARTIAL_BAD_CHANNELS_LOG = """Recording has 974 channels
found 100 bad channels
Time 364s. Computing whitening matrix..
(recording sorted fine, eventually unrelated error)
"""


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    @pytest.mark.parametrize(
        "concrete, categorical",
        [
            (InsufficientActivityError, BiologicalSortFailure),
            (NoGoodChannelsError, BiologicalSortFailure),
            (SaturatedSignalError, BiologicalSortFailure),
            (EmptyWaveformMetricsError, BiologicalSortFailure),
            (HDF5PluginMissingError, EnvironmentSortFailure),
            (DockerEnvironmentError, EnvironmentSortFailure),
            (GPUOutOfMemoryError, ResourceSortFailure),
        ],
    )
    def test_concrete_subclasses_its_category(self, concrete, categorical):
        assert issubclass(concrete, categorical)
        assert issubclass(concrete, SpikeSortingClassifiedError)
        assert issubclass(concrete, RuntimeError)

    @pytest.mark.parametrize(
        "categorical",
        [BiologicalSortFailure, EnvironmentSortFailure, ResourceSortFailure],
    )
    def test_categorical_subclasses_base(self, categorical):
        assert issubclass(categorical, SpikeSortingClassifiedError)

    def test_empty_waveform_preserves_valueerror_identity(self):
        """Backward-compat: historical callers caught ValueError here."""
        err = EmptyWaveformMetricsError("boom", metric_name="snr")
        assert isinstance(err, ValueError)
        assert isinstance(err, BiologicalSortFailure)
        assert err.metric_name == "snr"


# ---------------------------------------------------------------------------
# KS2 classifier — InsufficientActivityError
# ---------------------------------------------------------------------------


class TestKs2InsufficientActivity:
    def test_sparse_log_classified(self, tmp_path):
        log_path = _write_ks2_log_matlab(tmp_path, _KS2_SPARSE_LOG)
        err = classify_ks2_failure(tmp_path, RuntimeError("ks2 exit"))
        assert isinstance(err, InsufficientActivityError)
        assert err.sorter == "kilosort2"
        assert err.threshold_crossings == 1346
        assert err.units_at_failure == 2
        assert err.nspks_at_failure == pytest.approx(1.3292)
        assert err.log_path == log_path

    def test_docker_path_also_found(self, tmp_path):
        """Dispatcher must locate the log at sorter_output/ when present."""
        log_path = _write_ks2_log_docker(tmp_path, _KS2_SPARSE_LOG)
        err = classify_ks2_failure(tmp_path, RuntimeError("ks2 exit"))
        assert isinstance(err, InsufficientActivityError)
        assert err.log_path == log_path

    def test_cuda_error_with_normal_activity_does_not_misclassify(self, tmp_path):
        """A real CUDA bug on an active recording must not be reclassified."""
        _write_ks2_log_matlab(tmp_path, _KS2_HEALTHY_LOG_WITH_CUDA_ERROR)
        err = classify_ks2_failure(tmp_path, RuntimeError("ks2 exit"))
        assert err is None

    def test_log_without_cuda_marker_is_not_classified(self, tmp_path):
        """No CUDA marker = not the insufficient-activity signature."""
        _write_ks2_log_matlab(
            tmp_path,
            "Recording has 974 channels\nfound 0 threshold crossings\n(no cuda error)\n",
        )
        err = classify_ks2_failure(tmp_path, RuntimeError("something else"))
        assert err is None

    def test_missing_log_returns_none(self, tmp_path):
        err = classify_ks2_failure(tmp_path, RuntimeError("no log written"))
        assert err is None

    def test_low_nspks_alone_is_sufficient_trigger(self, tmp_path):
        """Any single low-activity indicator is enough to trigger classification."""
        log = (
            "Recording has 974 channels\n"
            "found 500000 threshold crossings in 300 seconds of data\n"  # NOT low
            "found 2 bad channels\n"
            "Time 2s. Optimizing templates ...\n"
            "0.10 sec, 1 / 46 batches, 100 units, nspks: 1.0\n"  # low nspks
            "invalid configuration argument\n"
        )
        _write_ks2_log_matlab(tmp_path, log)
        err = classify_ks2_failure(tmp_path, RuntimeError("ks2 exit"))
        assert isinstance(err, InsufficientActivityError)
        assert err.nspks_at_failure == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# KS2 classifier — NoGoodChannelsError
# ---------------------------------------------------------------------------


class TestKs2NoGoodChannels:
    def test_all_channels_flagged_bad(self, tmp_path):
        log_path = _write_ks2_log_matlab(tmp_path, _KS2_ALL_BAD_CHANNELS_LOG)
        err = classify_ks2_failure(tmp_path, RuntimeError("ks2 exit"))
        assert isinstance(err, NoGoodChannelsError)
        assert err.sorter == "kilosort2"
        assert err.total_channels == 974
        assert err.bad_channels == 974
        assert err.log_path == log_path

    def test_explicit_zero_good_channels_marker(self, tmp_path):
        """'found 0 good channels' triggers even without a channel-count line."""
        _write_ks2_log_matlab(tmp_path, _KS2_ZERO_GOOD_CHANNELS_LOG)
        err = classify_ks2_failure(tmp_path, RuntimeError("ks2 exit"))
        assert isinstance(err, NoGoodChannelsError)

    def test_partial_bad_channels_not_classified(self, tmp_path):
        """Partial channel loss is tolerated by KS2 and must not misfire."""
        _write_ks2_log_matlab(tmp_path, _KS2_PARTIAL_BAD_CHANNELS_LOG)
        err = classify_ks2_failure(tmp_path, RuntimeError("unrelated"))
        assert err is None


# ---------------------------------------------------------------------------
# KS4 classifier — InsufficientActivityError via exception chain
# ---------------------------------------------------------------------------


class TestKs4InsufficientActivity:
    def test_truncated_svd_empty(self, tmp_path):
        inner = ValueError(
            "Found array with 0 sample(s) (shape=(0, 61)) while a minimum "
            "of 1 is required by TruncatedSVD."
        )
        outer = RuntimeError("run_sorter failed")
        outer.__cause__ = inner
        err = classify_ks4_failure(tmp_path, outer)
        assert isinstance(err, InsufficientActivityError)
        assert err.sorter == "kilosort4"
        assert err.units_at_failure == 0

    def test_kmeans_too_few_samples(self, tmp_path):
        inner = ValueError("n_samples=3 should be >= n_clusters=6.")
        outer = RuntimeError("SI wrapper failed")
        outer.__cause__ = inner
        err = classify_ks4_failure(tmp_path, outer)
        assert isinstance(err, InsufficientActivityError)
        assert err.units_at_failure == 3

    def test_deeply_nested_chain_walked(self, tmp_path):
        chain = _chain(
            [
                "n_samples=2 should be >= n_clusters=6.",
                "clustering step failed",
                "run_sorter wrapper",
                "outer batch runner",
            ]
        )
        err = classify_ks4_failure(tmp_path, chain)
        assert isinstance(err, InsufficientActivityError)
        assert err.units_at_failure == 2

    def test_unrelated_exception_returns_none(self, tmp_path):
        err = classify_ks4_failure(
            tmp_path, RuntimeError("some totally unrelated crash")
        )
        assert err is None

    def test_walk_exception_chain_handles_cycle(self):
        """_walk_exception_chain must not infinite-loop on a cyclic chain."""
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        text = _walk_exception_chain(a)
        # Both messages appear and the walk terminates.
        assert "a" in text and "b" in text


# ---------------------------------------------------------------------------
# Environment classifier — HDF5PluginMissingError
# ---------------------------------------------------------------------------


class TestHDF5PluginMissing:
    def test_env_var_message_classified(self, tmp_path):
        err = classify_ks2_failure(
            tmp_path,
            RuntimeError("HDF5_PLUGIN_PATH was set but no compression filter found"),
        )
        assert isinstance(err, HDF5PluginMissingError)

    def test_filter_keyword_classified(self, tmp_path):
        err = classify_ks4_failure(
            tmp_path,
            RuntimeError(
                "HDF5 filter plugin missing: Unable to synchronously read data"
            ),
        )
        assert isinstance(err, HDF5PluginMissingError)

    def test_configured_path_echoed_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HDF5_PLUGIN_PATH", "/some/deploy/specific/path")
        err = classify_ks4_failure(
            tmp_path,
            RuntimeError("HDF5_PLUGIN_PATH leads nowhere; compression filter missing"),
        )
        assert isinstance(err, HDF5PluginMissingError)
        assert err.configured_path == "/some/deploy/specific/path"

    def test_generic_cant_open_without_filter_keyword_not_classified(self, tmp_path):
        """Bare 'Can't open directory' on a non-filter context is not HDF5-plugin."""
        err = classify_ks4_failure(
            tmp_path,
            RuntimeError("Can't open directory: /nonexistent"),
        )
        assert err is None


# ---------------------------------------------------------------------------
# Environment classifier — DockerEnvironmentError (reason-coded)
# ---------------------------------------------------------------------------


class TestDockerEnvironment:
    @pytest.mark.parametrize(
        "message, expected_reason",
        [
            (
                "Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
                "daemon_down",
            ),
            (
                "Is the docker daemon running on this host?",
                "daemon_down",
            ),
            (
                "ModuleNotFoundError: No module named 'docker'",
                "client_missing",
            ),
            (
                "docker: Got permission denied while trying to connect",
                "permission_denied",
            ),
            (
                'manifest unknown: manifest tagged by "x" is not found',
                "image_pull_failed",
            ),
            (
                "pull access denied for foo/bar",
                "image_pull_failed",
            ),
            (
                "failed to pull and unpack image registry-1.docker.io/...",
                "image_pull_failed",
            ),
            (
                "dial tcp: lookup registry-1.docker.io: no such host",
                "image_pull_failed",
            ),
        ],
    )
    def test_reason_classification(self, tmp_path, message, expected_reason):
        err = classify_ks4_failure(tmp_path, RuntimeError(message))
        assert isinstance(err, DockerEnvironmentError)
        assert err.reason == expected_reason

    def test_permission_denied_precedes_daemon_down(self, tmp_path):
        """A message matching both permission-denied and daemon-down should
        be classified as permission_denied (the more specific reason)."""
        err = classify_ks4_failure(
            tmp_path,
            RuntimeError(
                "permission denied while trying to connect to the Docker "
                "daemon at unix:///var/run/docker.sock"
            ),
        )
        assert isinstance(err, DockerEnvironmentError)
        assert err.reason == "permission_denied"


# ---------------------------------------------------------------------------
# Resource classifier — GPUOutOfMemoryError
# ---------------------------------------------------------------------------


class TestGPUOutOfMemory:
    def test_torch_oom_marker_ks4(self, tmp_path):
        err = classify_ks4_failure(
            tmp_path,
            RuntimeError(
                "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to "
                "allocate 5.36 GiB"
            ),
        )
        assert isinstance(err, GPUOutOfMemoryError)
        assert err.sorter == "kilosort4"

    def test_matlab_oom_marker_ks2(self, tmp_path):
        err = classify_ks2_failure(
            tmp_path,
            RuntimeError("The CUDA error was: CUDA_ERROR_OUT_OF_MEMORY (201)"),
        )
        assert isinstance(err, GPUOutOfMemoryError)
        assert err.sorter == "kilosort2"


# ---------------------------------------------------------------------------
# Dispatcher priority — env / resource must win over biology
# ---------------------------------------------------------------------------


class TestDispatcherPriority:
    def test_docker_beats_ks2_insufficient_activity(self, tmp_path):
        """Sparse log + docker-daemon error must classify as docker."""
        _write_ks2_log_matlab(tmp_path, _KS2_SPARSE_LOG)
        err = classify_ks2_failure(
            tmp_path,
            RuntimeError("Cannot connect to the Docker daemon"),
        )
        assert isinstance(err, DockerEnvironmentError)
        assert err.reason == "daemon_down"

    def test_hdf5_beats_ks2_insufficient_activity(self, tmp_path):
        _write_ks2_log_matlab(tmp_path, _KS2_SPARSE_LOG)
        err = classify_ks2_failure(
            tmp_path,
            RuntimeError("HDF5_PLUGIN_PATH is not set; filter plugin missing"),
        )
        assert isinstance(err, HDF5PluginMissingError)

    def test_oom_beats_ks2_insufficient_activity(self, tmp_path):
        _write_ks2_log_matlab(tmp_path, _KS2_SPARSE_LOG)
        err = classify_ks2_failure(
            tmp_path,
            RuntimeError("CUDA out of memory while allocating"),
        )
        assert isinstance(err, GPUOutOfMemoryError)

    def test_hdf5_beats_ks4_insufficient_activity(self, tmp_path):
        inner = ValueError("Found array with 0 sample(s) required by TruncatedSVD")
        outer = RuntimeError("HDF5 filter plugin missing (HDF5_PLUGIN_PATH unset)")
        outer.__cause__ = inner
        err = classify_ks4_failure(tmp_path, outer)
        assert isinstance(err, HDF5PluginMissingError)


# ---------------------------------------------------------------------------
# Curation — EmptyWaveformMetricsError raises directly
# ---------------------------------------------------------------------------


class TestCurationRaisesEmptyWaveformMetrics:
    def _make_empty_sd(self):
        """SpikeData with no raw_data attached."""
        from spikelab.spikedata import SpikeData

        trains = [np.array([0.1, 0.2, 0.3]) for _ in range(3)]
        return SpikeData(trains, length=1.0)

    def test_compute_waveform_metrics_empty_raw_data(self):
        from spikelab.spikedata.curation import compute_waveform_metrics

        sd = self._make_empty_sd()
        with pytest.raises(EmptyWaveformMetricsError, match="raw_data is empty"):
            compute_waveform_metrics(sd)

    def test_get_or_compute_waveform_metric_empty_raw_data(self):
        from spikelab.spikedata.curation import _get_or_compute_waveform_metric

        sd = self._make_empty_sd()
        with pytest.raises(EmptyWaveformMetricsError) as excinfo:
            _get_or_compute_waveform_metric(sd, "snr", 1.0, 2.0)
        assert excinfo.value.metric_name == "snr"

    def test_raised_error_is_also_valueerror_and_biological(self):
        """Backward compat + category-aware both work."""
        from spikelab.spikedata.curation import _get_or_compute_waveform_metric

        sd = self._make_empty_sd()
        try:
            _get_or_compute_waveform_metric(sd, "std_norm", 1.0, 2.0)
        except ValueError as err:
            # Both identities hold simultaneously.
            assert isinstance(err, EmptyWaveformMetricsError)
            assert isinstance(err, BiologicalSortFailure)
            assert isinstance(err, SpikeSortingClassifiedError)
