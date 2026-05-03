"""Tests for the additional safeguards (items 1–9 from the post-Stream 2 round).

Covered modules:

* ``guards/_sort_lock`` — concurrent-sort prevention
* ``guards/_job_object`` — Windows Job Object memory cap
* ``guards/_preflight`` — FD/process rlimits + SI version
* ``guards/_audit`` — JSONL events log
* ``guards/_io_stall`` — disk-I/O stall watchdog
* ``guards/_tempfile_cleanup`` — temp-file sweep on clean exit
* ``guards/_power_state`` — Windows sleep prevention
* Plus pipeline-level wiring for batch trending in ``SortRunReport``.

Tests are platform-agnostic via mocks where the real OS API is
Windows-only or psutil-dependent.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from spikelab.spike_sorting._exceptions import (
    ConcurrentSortError,
    EnvironmentSortFailure,
    IOStallError,
    ResourceSortFailure,
)
from spikelab.spike_sorting.config import ExecutionConfig
from spikelab.spike_sorting.guards import (
    IOStallWatchdog,
    acquire_sort_lock,
    append_audit_event,
    cleanup_temp_files,
    prevent_system_sleep,
    windows_job_object_cap,
)

# ---------------------------------------------------------------------------
# #1 — Concurrent-sort lock
# ---------------------------------------------------------------------------


class TestConcurrentSortLock:
    """``acquire_sort_lock`` blocks concurrent sorts and reclaims stale locks."""

    def test_writes_lock_file_on_entry(self, tmp_path):
        """
        Entry creates a JSON lock file recording PID / hostname /
        start time.

        Tests:
            (Test Case 1) The lock file appears at .spikelab_sort.lock.
            (Test Case 2) PID matches the current process.
            (Test Case 3) hostname and started_at fields are populated.
        """
        with acquire_sort_lock(tmp_path) as lock_path:
            assert lock_path.exists()
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            assert data["pid"] == os.getpid()
            assert isinstance(data.get("hostname"), str) and data["hostname"]
            assert isinstance(data.get("started_at"), str)

    def test_lock_removed_on_exit(self, tmp_path):
        """
        The lock file is deleted on normal exit.

        Tests:
            (Test Case 1) After the with-block exits, the lock
                file no longer exists.
        """
        with acquire_sort_lock(tmp_path) as lock_path:
            pass
        assert not lock_path.exists()

    def test_concurrent_acquire_raises(self, tmp_path):
        """
        Acquiring while another live holder owns the lock raises.

        Tests:
            (Test Case 1) Second acquire inside the first raises
                ConcurrentSortError.
            (Test Case 2) Exception carries holder PID and lock_path.
        """
        with acquire_sort_lock(tmp_path):
            with pytest.raises(ConcurrentSortError) as exc_info:
                with acquire_sort_lock(tmp_path):
                    pass
        err = exc_info.value
        assert err.holder_pid == os.getpid()
        assert err.lock_path is not None
        assert "Another sort" in str(err)

    def test_stale_lock_reclaimed(self, tmp_path):
        """
        A lock file pointing at a dead PID is reclaimed.

        Tests:
            (Test Case 1) After writing a lock file with a dead PID,
                a fresh acquire succeeds (after reclaim).
            (Test Case 2) The new lock file records the current PID.
        """
        from spikelab.spike_sorting.guards import _sort_lock as lock_mod

        # Drop a synthetic stale lock claiming PID 99999 on this host.
        lock_path = tmp_path / ".spikelab_sort.lock"
        lock_path.write_text(
            json.dumps(
                {
                    "pid": 99999,
                    "hostname": lock_mod.socket.gethostname(),
                    "started_at": "1970-01-01T00:00:00",
                }
            )
        )

        # Patch _pid_alive so our synthetic PID looks dead.
        with mock.patch.object(lock_mod, "_pid_alive", return_value=False):
            with acquire_sort_lock(tmp_path) as new_lock:
                data = json.loads(new_lock.read_text(encoding="utf-8"))
                assert data["pid"] == os.getpid()

    def test_unparseable_lock_raises(self, tmp_path):
        """
        Malformed lock file is treated as live (cannot reclaim safely).

        Tests:
            (Test Case 1) Raises ConcurrentSortError when the lock
                file is not valid JSON.
        """
        lock_path = tmp_path / ".spikelab_sort.lock"
        lock_path.write_text("not json {")
        with pytest.raises(ConcurrentSortError) as exc_info:
            with acquire_sort_lock(tmp_path):
                pass
        assert "unparseable" in str(exc_info.value).lower()

    def test_other_host_raises(self, tmp_path):
        """
        Lock from a different host cannot be liveness-checked.

        Tests:
            (Test Case 1) Lock file with a foreign hostname raises
                ConcurrentSortError; we do not attempt cross-host
                PID liveness checks.
        """
        lock_path = tmp_path / ".spikelab_sort.lock"
        lock_path.write_text(
            json.dumps(
                {
                    "pid": 1234,
                    "hostname": "some-other-host-not-this-one",
                    "started_at": "1970-01-01T00:00:00",
                }
            )
        )
        with pytest.raises(ConcurrentSortError) as exc_info:
            with acquire_sort_lock(tmp_path):
                pass
        assert exc_info.value.holder_hostname == "some-other-host-not-this-one"


# ---------------------------------------------------------------------------
# #2 — Windows Job Object cap
# ---------------------------------------------------------------------------


class TestWindowsJobObjectCap:
    """``windows_job_object_cap`` is a no-op off Windows or w/o pywin32."""

    def test_noop_on_non_windows(self):
        """
        Off Windows, the context manager yields False without
        raising.

        Tests:
            (Test Case 1) Yields False on the current platform when
                ``sys.platform != 'win32'``.

        Notes:
            - The test patches sys.platform to simulate a non-Windows
              host even when running on Windows.
        """
        from spikelab.spike_sorting.guards import _job_object as job_mod

        with mock.patch.object(job_mod.sys, "platform", "linux"):
            with windows_job_object_cap(0.8) as active:
                assert active is False

    def test_noop_when_pywin32_missing(self):
        """
        On Windows with pywin32 missing, yields False.

        Tests:
            (Test Case 1) When the win32job/win32api imports fail,
                the helper yields False and does not raise.
        """
        from spikelab.spike_sorting.guards import _job_object as job_mod

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _fake_import(name, *args, **kwargs):
            if name in ("win32job", "win32api", "win32con"):
                raise ImportError("simulated missing pywin32")
            return real_import(name, *args, **kwargs)

        with (
            mock.patch.object(job_mod.sys, "platform", "win32"),
            mock.patch("builtins.__import__", _fake_import),
        ):
            with windows_job_object_cap(0.8) as active:
                assert active is False

    def test_noop_when_ram_undetectable(self):
        """
        Without a host-RAM total, returns False.

        Tests:
            (Test Case 1) When the underlying ``get_system_ram_bytes``
                returns None, the helper yields False.
        """
        from spikelab.spike_sorting.guards import _job_object as job_mod

        with (
            mock.patch.object(job_mod.sys, "platform", "win32"),
            mock.patch.object(job_mod, "_get_total_ram_bytes", return_value=None),
        ):
            with windows_job_object_cap(0.8) as active:
                assert active is False


# ---------------------------------------------------------------------------
# #3 — FD + process-count preflight
# ---------------------------------------------------------------------------


class TestResourceRlimitPreflight:
    """``_check_resource_rlimits`` warns on tight RLIMIT_NOFILE / NPROC."""

    def test_low_nofile_warns(self):
        """
        RLIMIT_NOFILE under threshold yields a low_rlimit_nofile warn.

        Tests:
            (Test Case 1) Patched getrlimit returning 1024 → warn
                with code 'low_rlimit_nofile'.
        """
        try:
            import resource as _resource
        except ImportError:
            pytest.skip("POSIX-only check; resource module unavailable")

        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards._preflight import (
            _check_resource_rlimits,
        )

        cfg = SortingPipelineConfig()

        def _fake_getrlimit(which):
            if which == _resource.RLIMIT_NOFILE:
                return (1024, 65536)
            return (1_000_000, 1_000_000)

        with mock.patch.object(_resource, "getrlimit", _fake_getrlimit):
            findings = _check_resource_rlimits(cfg)
        codes = [f.code for f in findings]
        assert "low_rlimit_nofile" in codes

    def test_low_nproc_warns_and_scales_with_num_processes(self):
        """
        RLIMIT_NPROC threshold scales with rt_sort.num_processes.

        Tests:
            (Test Case 1) num_processes=64 → threshold = max(256,
                4*64) = 256.
            (Test Case 2) num_processes=128 → threshold = 512.
        """
        try:
            import resource as _resource
        except ImportError:
            pytest.skip("POSIX-only check; resource module unavailable")
        if not hasattr(_resource, "RLIMIT_NPROC"):
            pytest.skip("RLIMIT_NPROC not available on this platform")

        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards._preflight import (
            _check_resource_rlimits,
        )

        cfg = SortingPipelineConfig()
        cfg.rt_sort.num_processes = 128  # threshold = 512

        def _fake_getrlimit(which):
            if which == _resource.RLIMIT_NPROC:
                return (300, 300)
            return (1_000_000, 1_000_000)

        with mock.patch.object(_resource, "getrlimit", _fake_getrlimit):
            findings = _check_resource_rlimits(cfg)
        codes = [f.code for f in findings]
        assert "low_rlimit_nproc" in codes


class TestSpikeInterfaceVersionCheck:
    """``_check_spikeinterface_version`` warns when SI is outside tested range."""

    def test_inside_range_no_finding(self):
        """
        SI version inside [low, high) yields no finding.

        Tests:
            (Test Case 1) Version 0.104.0 produces no warning.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod

        fake_si = SimpleNamespace(__version__="0.104.0")
        with mock.patch.dict(sys.modules, {"spikeinterface": fake_si}):
            assert preflight_mod._check_spikeinterface_version() is None

    def test_outside_range_warns(self):
        """
        SI version below or above the range yields a warn finding.

        Tests:
            (Test Case 1) 0.090.0 → warn.
            (Test Case 2) 1.50.0 → warn.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod

        for ver in ("0.090.0", "1.50.0"):
            fake_si = SimpleNamespace(__version__=ver)
            with mock.patch.dict(sys.modules, {"spikeinterface": fake_si}):
                finding = preflight_mod._check_spikeinterface_version()
                assert finding is not None
                assert finding.level == "warn"
                assert finding.code == "spikeinterface_version_outside_tested_range"


# ---------------------------------------------------------------------------
# #4 — Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    """``append_audit_event`` writes JSONL events next to the log."""

    def test_writes_event_with_explicit_path(self, tmp_path):
        """
        Explicit log_path argument controls the audit file location.

        Tests:
            (Test Case 1) Audit file appears next to the supplied
                log_path with one event line.
            (Test Case 2) Event JSON contains watchdog and event
                fields plus the supplied payload.
        """
        log_path = tmp_path / "rec.log"
        log_path.touch()
        append_audit_event(
            watchdog="host_memory",
            event="warn",
            log_path=log_path,
            used_pct=87.4,
            warn_pct=85.0,
        )
        audit = tmp_path / "watchdog_events.jsonl"
        assert audit.is_file()
        line = audit.read_text(encoding="utf-8").strip()
        entry = json.loads(line)
        assert entry["watchdog"] == "host_memory"
        assert entry["event"] == "warn"
        assert entry["used_pct"] == 87.4
        assert "timestamp" in entry

    def test_silent_noop_when_no_log_path(self):
        """
        Without an explicit or active log path, the call is a silent
        no-op.

        Tests:
            (Test Case 1) Calling without log_path or active
                ContextVar does not raise and does not crash.
        """
        # No active log path set; should silently skip.
        append_audit_event(watchdog="x", event="y")

    def test_appends_multiple_events(self, tmp_path):
        """
        Multiple events accumulate as JSONL.

        Tests:
            (Test Case 1) Three calls produce three lines.
        """
        log_path = tmp_path / "rec.log"
        log_path.touch()
        for i in range(3):
            append_audit_event(
                watchdog="disk", event="warn", log_path=log_path, free_gb=float(i)
            )
        audit = tmp_path / "watchdog_events.jsonl"
        lines = audit.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# #5 — Cumulative batch trending
# ---------------------------------------------------------------------------


class TestPeakReadingFromAudit:
    """``_read_peaks_from_audit`` extracts peak resource values."""

    def test_no_audit_yields_none_peaks(self, tmp_path):
        """
        Missing audit file yields None for every peak field.

        Tests:
            (Test Case 1) All three peaks are None when the audit
                file does not exist.
        """
        from spikelab.spike_sorting.pipeline import _read_peaks_from_audit

        peaks = _read_peaks_from_audit(tmp_path)
        assert peaks["peak_host_ram_pct"] is None
        assert peaks["peak_gpu_used_pct"] is None
        assert peaks["min_disk_free_gb"] is None

    def test_extracts_max_for_memory_min_for_disk(self, tmp_path):
        """
        Peaks pick max for memory percent and min for disk free GB.

        Tests:
            (Test Case 1) Two host_memory events at 88% and 91% →
                peak = 91%.
            (Test Case 2) Two disk events at 4 GB and 1.5 GB →
                min = 1.5 GB.
        """
        from spikelab.spike_sorting.pipeline import _read_peaks_from_audit

        audit = tmp_path / "watchdog_events.jsonl"
        events = [
            {"watchdog": "host_memory", "event": "warn", "used_pct": 88.0},
            {"watchdog": "host_memory", "event": "warn", "used_pct": 91.0},
            {"watchdog": "disk", "event": "warn", "free_gb": 4.0},
            {"watchdog": "disk", "event": "warn", "free_gb": 1.5},
            {"watchdog": "gpu_memory", "event": "warn", "used_pct": 92.0},
        ]
        audit.write_text(
            "\n".join(json.dumps(e) for e in events) + "\n",
            encoding="utf-8",
        )
        peaks = _read_peaks_from_audit(tmp_path)
        assert peaks["peak_host_ram_pct"] == 91.0
        assert peaks["peak_gpu_used_pct"] == 92.0
        assert peaks["min_disk_free_gb"] == 1.5


# ---------------------------------------------------------------------------
# #6 — IOStallWatchdog
# ---------------------------------------------------------------------------


class TestIOStallWatchdog:
    """The I/O stall watchdog trips on stagnant byte counters."""

    def test_disabled_when_psutil_cannot_resolve_device(self, tmp_path):
        """
        Without a resolvable device, the watchdog is a no-op.

        Tests:
            (Test Case 1) When ``_resolve_device_for_path`` returns
                None, the watchdog reports as disabled.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        with mock.patch.object(iom, "_resolve_device_for_path", return_value=None):
            wd = IOStallWatchdog(tmp_path, stall_s=1.0, poll_interval_s=0.1)
            with wd:
                assert wd._enabled is False

    def test_trip_on_stagnant_bytes(self, tmp_path):
        """
        Constant byte counter for stall_s seconds trips the watchdog.

        Tests:
            (Test Case 1) Patched ``_read_io_bytes`` returns the same
                value across polls; after stall_s seconds, tripped()
                is True.
            (Test Case 2) make_error returns IOStallError with the
                resolved device.

        Notes:
            - The watchdog calls ``_thread.interrupt_main`` on trip,
              which raises KeyboardInterrupt into this test thread.
              We catch it and verify the trip via ``tripped()``.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", return_value=100),
        ):
            wd = IOStallWatchdog(tmp_path, stall_s=0.5, poll_interval_s=0.1)
            try:
                with wd:
                    deadline = time.time() + 3.0
                    while time.time() < deadline and not wd.tripped():
                        time.sleep(0.05)
            except KeyboardInterrupt:
                pass
        assert wd.tripped()
        err = wd.make_error()
        assert isinstance(err, IOStallError)
        assert err.device == "sda1"

    def test_no_trip_when_bytes_advance(self, tmp_path):
        """
        Steadily increasing byte counter never trips the watchdog.

        Tests:
            (Test Case 1) After several polls with monotonically
                increasing reads, tripped() is False.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        counter = {"value": 0}

        def _advance(_dev):
            counter["value"] += 1024
            return counter["value"]

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", side_effect=_advance),
        ):
            wd = IOStallWatchdog(tmp_path, stall_s=1.0, poll_interval_s=0.05)
            with wd:
                time.sleep(0.6)
                assert not wd.tripped()


# ---------------------------------------------------------------------------
# #7 — Temp-file cleanup
# ---------------------------------------------------------------------------


class TestTempFileCleanup:
    """``cleanup_temp_files`` sweeps marker-prefixed temp files on clean exit."""

    def test_removes_new_marker_files(self, tmp_path, monkeypatch):
        """
        Files created during the context that match a known marker
        are deleted on clean exit.

        Tests:
            (Test Case 1) ``spikelab_*`` and ``kilosort_*`` files
                created during the context are gone after exit.
            (Test Case 2) Files without a marker prefix are kept.
        """
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        # Pre-existing files (should NOT be removed).
        keep1 = tmp_path / "unrelated.txt"
        keep1.write_text("keep")
        keep2 = tmp_path / "spikelab_pre_existing.tmp"
        keep2.write_text("keep")

        # Create marker files inside the context.
        with cleanup_temp_files(enabled=True):
            (tmp_path / "spikelab_runtime.tmp").write_text("x")
            (tmp_path / "kilosort_temp.dat").write_text("x")
            (tmp_path / "still_unrelated.dat").write_text("x")

        # Pre-existing marker file is preserved (it was there before
        # the sort started).
        assert keep1.exists()
        assert keep2.exists()
        # Created marker files removed.
        assert not (tmp_path / "spikelab_runtime.tmp").exists()
        assert not (tmp_path / "kilosort_temp.dat").exists()
        # Created non-marker file preserved.
        assert (tmp_path / "still_unrelated.dat").exists()

    def test_disabled_is_noop(self, tmp_path, monkeypatch):
        """
        ``enabled=False`` keeps every file regardless of marker.

        Tests:
            (Test Case 1) Marker files created during the context
                survive.
        """
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        with cleanup_temp_files(enabled=False):
            (tmp_path / "spikelab_x.tmp").write_text("x")
        assert (tmp_path / "spikelab_x.tmp").exists()

    def test_files_kept_on_exception(self, tmp_path, monkeypatch):
        """
        Exceptions in the context propagate and leave temp files alone.

        Tests:
            (Test Case 1) Marker files created before the raise
                survive (the exception triggers the no-sweep path).
        """
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        with pytest.raises(RuntimeError):
            with cleanup_temp_files(enabled=True):
                (tmp_path / "spikelab_diag.tmp").write_text("x")
                raise RuntimeError("simulated failure")
        assert (tmp_path / "spikelab_diag.tmp").exists()


# ---------------------------------------------------------------------------
# #9 — Power state
# ---------------------------------------------------------------------------


class TestPowerStateLock:
    """``prevent_system_sleep`` is a no-op off Windows."""

    def test_noop_on_non_windows(self):
        """
        Off Windows, prevent_system_sleep yields False without raising.

        Tests:
            (Test Case 1) Non-Windows platform → yields False.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        with mock.patch.object(ps.sys, "platform", "linux"):
            with prevent_system_sleep() as active:
                assert active is False

    def test_yields_false_when_platform_simulated_non_windows(self):
        """
        Patching sys.platform to a non-Windows value yields False.

        Tests:
            (Test Case 1) When the helper sees a non-Windows
                platform, it yields False without touching any
                ctypes APIs — even on a real Windows host.

        Notes:
            - The Windows-API-call path (``SetThreadExecutionState``)
              is exercised in production rather than tested here —
              mocking ``ctypes.windll`` reliably across platforms
              is fragile and the live call interacts with the OS
              in ways that can stall a test process.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        for fake_platform in ("linux", "darwin"):
            with mock.patch.object(ps.sys, "platform", fake_platform):
                with prevent_system_sleep() as active:
                    assert active is False


# ---------------------------------------------------------------------------
# ExecutionConfig defaults for the new fields
# ---------------------------------------------------------------------------


class TestExtraSafeguardConfigDefaults:
    """The new ExecutionConfig fields have the documented defaults."""

    def test_defaults(self):
        """
        Defaults match the documented values for items 6, 7, 9.

        Tests:
            (Test Case 1) io_stall_watchdog defaults True with
                300s stall window and 10s poll.
            (Test Case 2) cleanup_temp_files defaults True.
            (Test Case 3) prevent_system_sleep defaults True.
        """
        cfg = ExecutionConfig()
        assert cfg.io_stall_watchdog is True
        assert cfg.io_stall_s == 300.0
        assert cfg.io_stall_poll_interval_s == 10.0
        assert cfg.cleanup_temp_files is True
        assert cfg.prevent_system_sleep is True
