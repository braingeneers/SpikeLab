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

    def test_model_loading_error_inherits_environment_sort_failure(self):
        """
        ``ModelLoadingError`` must remain under the ``EnvironmentSortFailure``
        category, and ``GPUOutOfMemoryError`` must remain under
        ``ResourceSortFailure`` — pins the section-header invariant from
        the ``_exceptions.py`` module's structure.

        Tests:
            (Test Case 1) ``ModelLoadingError.__mro__`` includes
                ``EnvironmentSortFailure``.
            (Test Case 2) ``GPUOutOfMemoryError.__mro__`` includes
                ``ResourceSortFailure``.
        """
        from spikelab.spike_sorting._exceptions import ModelLoadingError

        assert EnvironmentSortFailure in ModelLoadingError.__mro__
        assert ResourceSortFailure in GPUOutOfMemoryError.__mro__

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


class TestWalkExceptionChainDeduplicates:
    """
    Tests for the message-text dedup added in commit 0d91204.

    When SpikeInterface re-raises an inner sklearn/numpy error, the
    inner and outer exceptions are distinct Python objects but carry
    identical ``str(exc)`` text — a naive walk would emit the same line
    twice. The walker uses identity checks to break cycles AND a text
    dedup so duplicate-message chains collapse to a single line, while
    distinct messages still each appear.

    Tests:
        (Test Case 1) Two distinct exception objects with identical
            ``str(exc)`` text produce exactly one line.
        (Test Case 2) Two exceptions with different text still produce
            two lines.
        (Test Case 3) A three-exception chain with one duplicate and
            one unique tail produces two lines (one per unique message).
    """

    def test_duplicate_text_collapses_to_single_line(self):
        """
        Tests:
            (Test Case 1) Outer + inner with identical ``str`` produce
                a single line (not two).
        """
        inner = RuntimeError("identical message")
        outer = RuntimeError("identical message")
        outer.__cause__ = inner

        text = _walk_exception_chain(outer)
        # Single occurrence — dedup collapses the second.
        assert text.count("identical message") == 1
        # Single line (no newline since there's only one message).
        assert "\n" not in text

    def test_distinct_text_still_produces_two_lines(self):
        """
        Tests:
            (Test Case 1) Outer + inner with distinct ``str`` produce
                two lines.
            (Test Case 2) Both messages are present in the output.
        """
        inner = RuntimeError("inner failure")
        outer = RuntimeError("outer wrapper")
        outer.__cause__ = inner

        text = _walk_exception_chain(outer)
        lines = text.split("\n")
        assert len(lines) == 2
        assert "outer wrapper" in text
        assert "inner failure" in text

    def test_three_level_chain_with_one_duplicate(self):
        """
        Tests:
            (Test Case 1) A three-level chain (outer -> middle -> inner)
                where outer and middle carry identical text dedups to
                exactly two unique lines.
            (Test Case 2) The unique inner message is preserved.
        """
        inner = RuntimeError("inner failure")
        middle = RuntimeError("duplicate text")
        middle.__cause__ = inner
        outer = RuntimeError("duplicate text")
        outer.__cause__ = middle

        text = _walk_exception_chain(outer)
        lines = text.split("\n")
        # "duplicate text" appears once; "inner failure" appears once.
        assert len(lines) == 2
        assert text.count("duplicate text") == 1
        assert text.count("inner failure") == 1


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


class TestClassifierLogFinders:
    """``_find_ks2_log`` / ``_find_ks4_log`` / ``_find_rt_sort_log``
    each search a small list of candidate paths in priority order and
    return the first that ``is_file()``.
    """

    def test_ks2_log_prefers_root_over_sorter_output(self, tmp_path: Path):
        """
        Tests:
            (Test Case 1) When both ``output/kilosort2.log`` and
                ``output/sorter_output/kilosort2.log`` exist, the
                root-level file is returned (first candidate wins).
        """
        from spikelab.spike_sorting._classifier import _find_ks2_log

        (tmp_path / "kilosort2.log").write_text("root", encoding="utf-8")
        (tmp_path / "sorter_output").mkdir()
        (tmp_path / "sorter_output" / "kilosort2.log").write_text(
            "nested", encoding="utf-8"
        )
        result = _find_ks2_log(tmp_path)
        assert result == tmp_path / "kilosort2.log"

    def test_ks2_log_falls_back_to_sorter_output(self, tmp_path: Path):
        """
        Tests:
            (Test Case 1) Only ``output/sorter_output/kilosort2.log``
                exists — the search falls through to the second
                candidate.
        """
        from spikelab.spike_sorting._classifier import _find_ks2_log

        (tmp_path / "sorter_output").mkdir()
        (tmp_path / "sorter_output" / "kilosort2.log").write_text(
            "nested", encoding="utf-8"
        )
        result = _find_ks2_log(tmp_path)
        assert result == tmp_path / "sorter_output" / "kilosort2.log"

    def test_ks2_log_none_when_no_candidates(self, tmp_path: Path):
        """
        Tests:
            (Test Case 1) Neither candidate path exists → returns None.
        """
        from spikelab.spike_sorting._classifier import _find_ks2_log

        assert _find_ks2_log(tmp_path) is None

    def test_ks4_log_prefers_root_over_sorter_output(self, tmp_path: Path):
        """
        Tests:
            (Test Case 1) Root-level KS4 log wins over nested.
        """
        from spikelab.spike_sorting._classifier import _find_ks4_log

        (tmp_path / "kilosort4.log").write_text("root", encoding="utf-8")
        (tmp_path / "sorter_output").mkdir()
        (tmp_path / "sorter_output" / "kilosort4.log").write_text(
            "nested", encoding="utf-8"
        )
        assert _find_ks4_log(tmp_path) == tmp_path / "kilosort4.log"

    def test_ks4_log_none_when_no_candidates(self, tmp_path: Path):
        """
        Tests:
            (Test Case 1) No KS4 log → None.
        """
        from spikelab.spike_sorting._classifier import _find_ks4_log

        assert _find_ks4_log(tmp_path) is None

    def test_rt_sort_log_returns_path_when_present(self, tmp_path: Path):
        """
        Tests:
            (Test Case 1) ``rt_sort.log`` at the root → returned.
        """
        from spikelab.spike_sorting._classifier import _find_rt_sort_log

        (tmp_path / "rt_sort.log").write_text("ok", encoding="utf-8")
        assert _find_rt_sort_log(tmp_path) == tmp_path / "rt_sort.log"

    def test_rt_sort_log_none_when_missing(self, tmp_path: Path):
        """
        Tests:
            (Test Case 1) No ``rt_sort.log`` → None.
        """
        from spikelab.spike_sorting._classifier import _find_rt_sort_log

        assert _find_rt_sort_log(tmp_path) is None

    def test_ks2_log_skips_directories(self, tmp_path: Path):
        """
        ``is_file()`` rejects directories — a folder named
        ``kilosort2.log`` should not match.

        Tests:
            (Test Case 1) A directory named ``kilosort2.log`` is not
                returned as a log file.
        """
        from spikelab.spike_sorting._classifier import _find_ks2_log

        (tmp_path / "kilosort2.log").mkdir()
        assert _find_ks2_log(tmp_path) is None
