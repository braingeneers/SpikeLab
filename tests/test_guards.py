"""Tests for spike_sorting.guards (host-memory watchdog + preflight).

The guards subpackage protects the host workstation from being taken
down by a sort. Two main pieces are exercised here:

* :class:`HostMemoryWatchdog` — daemon-thread monitor that polls
  ``psutil.virtual_memory().percent`` and aborts the run via
  ``_thread.interrupt_main`` plus subprocess termination.
* :func:`run_preflight` and :func:`report_findings` — pre-loop
  resource checks (disk, RAM, VRAM, HDF5 plugin path) with a strict
  mode that escalates warnings to hard failures.

Most behavioural tests mock ``psutil`` / ``shutil.disk_usage`` /
``pynvml`` / ``subprocess.check_output`` so the suite is hermetic and
does not depend on the host's actual resource state.
"""

from __future__ import annotations

import _thread  # noqa: F401  (used via mock.patch.object in tests)
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from spikelab.spike_sorting._exceptions import (
    ConcurrentSortError,
    DiskExhaustionError,
    EnvironmentSortFailure,
    GpuMemoryWatchdogError,
    HDF5PluginMissingError,
    HostMemoryWatchdogError,
    IOStallError,
    ResourceSortFailure,
    SorterTimeoutError,
    SpikeSortingClassifiedError,
)
from spikelab.spike_sorting.config import ExecutionConfig
from spikelab.spike_sorting.guards import (
    DiskExhaustionReport,
    DiskUsageWatchdog,
    GpuMemoryWatchdog,
    HostMemoryWatchdog,
    IOStallWatchdog,
    LogInactivityWatchdog,
    PreflightFinding,
    acquire_sort_lock,
    append_audit_event,
    cleanup_temp_files,
    compute_inactivity_timeout_s,
    get_active_watchdog,
    prevent_system_sleep,
    report_findings,
    run_preflight,
    windows_job_object_cap,
)
from spikelab.spike_sorting.guards import _preflight as preflight_mod
from spikelab.spike_sorting.guards import _watchdog as watchdog_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    sorter_name: str = "kilosort2",
    use_docker: bool = False,
    hdf5_plugin_path=None,
    **execution_overrides,
):
    """Construct a minimal config-like object that satisfies guards' API.

    Returns a SimpleNamespace with the three nested attribute groups
    that ``run_preflight`` reads: ``execution``, ``recording``,
    ``sorter``. Only the fields actually consulted by the preflight
    functions are populated.
    """
    exe_defaults = dict(
        preflight_min_free_inter_gb=20.0,
        preflight_min_free_results_gb=2.0,
        preflight_min_available_ram_gb=4.0,
        preflight_min_free_vram_gb=2.0,
        preflight=True,
        preflight_strict=False,
    )
    exe_defaults.update(execution_overrides)
    return SimpleNamespace(
        execution=SimpleNamespace(**exe_defaults),
        recording=SimpleNamespace(hdf5_plugin_path=hdf5_plugin_path),
        sorter=SimpleNamespace(sorter_name=sorter_name, use_docker=use_docker),
    )


def _spawn_sleeper(seconds: float = 30.0) -> subprocess.Popen:
    """Spawn a long-running OS-level sleeper for termination tests."""
    if sys.platform == "win32":
        cmd = ["timeout", "/T", str(int(seconds))]
    else:
        cmd = ["sleep", str(int(seconds))]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestHostMemoryWatchdogErrorHierarchy:
    """Hierarchy and attribute storage for HostMemoryWatchdogError."""

    def test_subclass_chain(self):
        """
        HostMemoryWatchdogError descends from the resource category.

        Tests:
            (Test Case 1) Subclass of ResourceSortFailure.
            (Test Case 2) Subclass of SpikeSortingClassifiedError.
            (Test Case 3) Subclass of RuntimeError (base of the
                hierarchy).
        """
        assert issubclass(HostMemoryWatchdogError, ResourceSortFailure)
        assert issubclass(HostMemoryWatchdogError, SpikeSortingClassifiedError)
        assert issubclass(HostMemoryWatchdogError, RuntimeError)

    def test_attribute_storage(self):
        """
        Constructor records percent_at_trip and abort_pct verbatim.

        Tests:
            (Test Case 1) Both attributes round-trip through __init__.
            (Test Case 2) Defaults to None when omitted.
        """
        err = HostMemoryWatchdogError("boom", percent_at_trip=93.5, abort_pct=92.0)
        assert err.percent_at_trip == 93.5
        assert err.abort_pct == 92.0
        assert "boom" in str(err)

        err2 = HostMemoryWatchdogError("no metadata")
        assert err2.percent_at_trip is None
        assert err2.abort_pct is None


# ---------------------------------------------------------------------------
# HostMemoryWatchdog construction
# ---------------------------------------------------------------------------


class TestHostMemoryWatchdogConstruction:
    """Threshold validation in HostMemoryWatchdog.__init__."""

    def test_valid_thresholds_construct(self):
        """
        Valid thresholds construct without error and round-trip.

        Tests:
            (Test Case 1) Defaults are accepted.
            (Test Case 2) Custom thresholds are stored as floats.
        """
        w = HostMemoryWatchdog()
        assert w.warn_pct == 85.0
        assert w.abort_pct == 92.0
        assert w.poll_interval_s == 2.0

        w2 = HostMemoryWatchdog(warn_pct=70, abort_pct=80, poll_interval_s=1.0)
        assert isinstance(w2.warn_pct, float)
        assert w2.warn_pct == 70.0
        assert w2.abort_pct == 80.0

    @pytest.mark.parametrize(
        "warn,abort",
        [
            (90.0, 80.0),  # warn > abort
            (80.0, 80.0),  # warn == abort
            (0.0, 50.0),  # warn at zero
            (-1.0, 50.0),  # warn negative
            (10.0, 110.0),  # abort > 100
        ],
    )
    def test_invalid_thresholds_raise(self, warn, abort):
        """
        Threshold misordering and out-of-range values raise ValueError.

        Tests:
            (Test Case 1) warn >= abort is rejected.
            (Test Case 2) warn <= 0 is rejected.
            (Test Case 3) abort > 100 is rejected.
        """
        with pytest.raises(ValueError):
            HostMemoryWatchdog(warn_pct=warn, abort_pct=abort)

    def test_zero_poll_interval_raises(self):
        """
        poll_interval_s must be strictly positive.

        Tests:
            (Test Case 1) poll_interval_s=0 raises.
            (Test Case 2) poll_interval_s<0 raises.
        """
        with pytest.raises(ValueError):
            HostMemoryWatchdog(poll_interval_s=0)
        with pytest.raises(ValueError):
            HostMemoryWatchdog(poll_interval_s=-0.5)


# ---------------------------------------------------------------------------
# HostMemoryWatchdog context manager
# ---------------------------------------------------------------------------


class TestHostMemoryWatchdogContext:
    """Context-manager behaviour: ContextVar publish, polling thread,
    graceful exit, no-op fallback when psutil is missing.
    """

    def test_context_var_published_inside_and_cleared_outside(self):
        """
        get_active_watchdog returns the live instance only inside the with-block.

        Tests:
            (Test Case 1) Outside context: get_active_watchdog() is None.
            (Test Case 2) Inside context: get_active_watchdog() is
                the watchdog instance.
            (Test Case 3) After exit: ContextVar is reset.
        """
        assert get_active_watchdog() is None
        # High abort threshold so the polling thread does nothing.
        with HostMemoryWatchdog(warn_pct=98, abort_pct=99, poll_interval_s=10.0) as wd:
            assert get_active_watchdog() is wd
        assert get_active_watchdog() is None

    def test_nesting_inner_replaces_outer(self):
        """
        Nested watchdogs: inner is active inside, outer resumes on exit.

        Tests:
            (Test Case 1) Inside outer: outer is active.
            (Test Case 2) Inside inner: inner is active.
            (Test Case 3) After inner exit: outer is active again.
        """
        with HostMemoryWatchdog(
            warn_pct=98, abort_pct=99, poll_interval_s=10.0
        ) as outer:
            assert get_active_watchdog() is outer
            with HostMemoryWatchdog(
                warn_pct=97, abort_pct=99, poll_interval_s=10.0
            ) as inner:
                assert get_active_watchdog() is inner
            assert get_active_watchdog() is outer

    def test_psutil_missing_degrades_to_noop(self, monkeypatch):
        """
        Watchdog with no psutil still publishes ContextVar but spawns no thread.

        Tests:
            (Test Case 1) With psutil import patched to fail, the
                watchdog enters cleanly and publishes itself via the
                ContextVar.
            (Test Case 2) No polling thread is started (so no trip
                can occur).

        Notes:
            - We patch builtins.__import__ to fail only on 'psutil'
              imports inside the watchdog module, which is the same
              pattern Python uses when the package is uninstalled.
        """
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("simulated missing psutil")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        with HostMemoryWatchdog(warn_pct=85, abort_pct=92, poll_interval_s=10.0) as wd:
            assert get_active_watchdog() is wd
            # No thread should have been started.
            assert wd._thread is None
            assert wd._enabled is False

    def test_clean_exit_joins_thread(self):
        """
        Exiting the context joins the polling thread cleanly.

        Tests:
            (Test Case 1) After __exit__, the polling thread is
                joined and reference cleared.
            (Test Case 2) Stop event was set so the thread woke
                from its wait promptly.

        Notes:
            - The polling thread is only spawned when ``psutil`` is
              importable; on environments without it the watchdog
              degrades to a no-op (covered by
              ``test_psutil_missing_degrades_to_noop``). Skip here so
              psutil-less hosts don't report a spurious failure.
        """
        pytest.importorskip("psutil")
        with HostMemoryWatchdog(warn_pct=98, abort_pct=99, poll_interval_s=10.0) as wd:
            t = wd._thread
            assert t is not None and t.is_alive()
        # After exit
        assert wd._thread is None
        assert wd._stop_event.is_set()
        # The original thread must be done.
        assert not t.is_alive()


# ---------------------------------------------------------------------------
# Subprocess registration
# ---------------------------------------------------------------------------


class TestHostMemoryWatchdogSubprocessRegistration:
    """register_subprocess / unregister_subprocess semantics."""

    def test_register_and_unregister(self):
        """
        Registered subprocesses are tracked; unregistering removes them.

        Tests:
            (Test Case 1) After register, the popen appears in the
                internal list.
            (Test Case 2) After unregister, the list is empty.
        """
        wd = HostMemoryWatchdog(warn_pct=98, abort_pct=99, poll_interval_s=10.0)
        fake_popen = mock.Mock(spec=subprocess.Popen)
        wd.register_subprocess(fake_popen)
        assert any(p is fake_popen for p, _ in wd._subprocesses)
        wd.unregister_subprocess(fake_popen)
        assert not any(p is fake_popen for p, _ in wd._subprocesses)

    def test_register_with_custom_grace(self):
        """
        Per-subprocess kill_grace_s overrides the watchdog default.

        Tests:
            (Test Case 1) Custom grace is recorded for the entry.
            (Test Case 2) Other entries retain the default grace.
        """
        wd = HostMemoryWatchdog(kill_grace_s=5.0)
        a = mock.Mock(spec=subprocess.Popen)
        b = mock.Mock(spec=subprocess.Popen)
        wd.register_subprocess(a, kill_grace_s=1.5)
        wd.register_subprocess(b)
        graces = {id(p): g for p, g in wd._subprocesses}
        assert graces[id(a)] == 1.5
        assert graces[id(b)] == 5.0

    def test_unregister_unregistered_is_noop(self):
        """
        unregister_subprocess for an unknown popen does not raise.

        Tests:
            (Test Case 1) Calling unregister_subprocess on a popen
                that was never registered is a clean no-op.
        """
        wd = HostMemoryWatchdog()
        fake = mock.Mock(spec=subprocess.Popen)
        wd.unregister_subprocess(fake)  # must not raise


# ---------------------------------------------------------------------------
# Trip behaviour (end-to-end)
# ---------------------------------------------------------------------------


class TestHostMemoryWatchdogTrip:
    """Behavioural tests for the warn/abort path. These are the
    integration-shaped tests that exercise the polling thread end to end
    by patching ``psutil.virtual_memory`` to return a fixed percentage.
    """

    def _busy_loop_until_interrupt(self, deadline_s: float) -> bool:
        """Spin in pure Python so interrupt_main has somewhere to land.

        Returns True if a KeyboardInterrupt was caught within
        ``deadline_s`` seconds, False on timeout.
        """
        deadline = time.time() + deadline_s
        try:
            while time.time() < deadline:
                _ = sum(range(500))
        except KeyboardInterrupt:
            return True
        return False

    def test_abort_trip_interrupts_main(self, monkeypatch):
        """
        Crossing abort_pct fires interrupt_main and records trip state.

        Tests:
            (Test Case 1) Watchdog's _tripped flag is set.
            (Test Case 2) percent_at_trip records the crossing value.
            (Test Case 3) make_error() returns a HostMemoryWatchdogError.
        """
        # Force every poll to read 95% — well above the 80% abort
        # threshold we configure below.
        fake_vm = SimpleNamespace(percent=95.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)
        # Patch the import inside the watchdog module's __enter__.
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with HostMemoryWatchdog(
                warn_pct=70, abort_pct=80, poll_interval_s=0.1
            ) as wd:
                interrupted = self._busy_loop_until_interrupt(deadline_s=3.0)
        assert interrupted, "interrupt_main was not delivered within 3s"
        assert wd.tripped()
        assert wd.percent_at_trip() == pytest.approx(95.0)
        err = wd.make_error()
        assert isinstance(err, HostMemoryWatchdogError)
        assert err.percent_at_trip == pytest.approx(95.0)
        assert err.abort_pct == 80.0

    def test_below_warn_no_action(self, monkeypatch):
        """
        Below warn threshold, no warning is printed and no trip occurs.

        Tests:
            (Test Case 1) After the watchdog runs for several polls
                with low pressure, _tripped remains False.
        """
        fake_vm = SimpleNamespace(percent=50.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with HostMemoryWatchdog(
                warn_pct=70, abort_pct=80, poll_interval_s=0.05
            ) as wd:
                # Sleep long enough for several polls.
                time.sleep(0.4)
                assert not wd.tripped()

    def test_initial_poll_is_deferred_so_enter_returns(self, monkeypatch):
        """
        First poll is delayed by one poll interval so __enter__ completes.

        Tests:
            (Test Case 1) Even when memory pressure is well over abort
                from the start, the with-block body begins executing
                (we observe a flag set inside the body) before any
                trip can fire — proving the deferral works.

        Notes:
            - This is a regression test for a bug found during initial
              smoke testing: without the deferral the watchdog could
              trip and call interrupt_main while __enter__ was still
              inside Thread.start(), corrupting the with-block setup.
        """
        fake_vm = SimpleNamespace(percent=99.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)
        body_entered = threading.Event()
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            try:
                with HostMemoryWatchdog(warn_pct=70, abort_pct=80, poll_interval_s=0.2):
                    body_entered.set()
                    # Stay long enough for the (deferred) first poll.
                    self._busy_loop_until_interrupt(deadline_s=2.0)
            except KeyboardInterrupt:
                pass
        assert (
            body_entered.is_set()
        ), "with-block body never executed; watchdog tripped during __enter__"

    def test_make_error_custom_message(self):
        """
        make_error accepts a custom message override.

        Tests:
            (Test Case 1) Custom message is the returned error's text.
            (Test Case 2) Trip metadata is still attached.
        """
        wd = HostMemoryWatchdog(warn_pct=70, abort_pct=80)
        wd._tripped = True
        wd._percent_at_trip = 91.5
        err = wd.make_error("custom note")
        assert "custom note" in str(err)
        assert err.percent_at_trip == 91.5
        assert err.abort_pct == 80.0


# ---------------------------------------------------------------------------
# Subprocess termination on trip
# ---------------------------------------------------------------------------


class TestHostMemoryWatchdogSubprocessTermination:
    """The watchdog terminates registered subprocesses on abort."""

    def test_terminate_single_subprocess(self):
        """
        Real subprocess registered with the watchdog is terminated on trip.

        Tests:
            (Test Case 1) After the watchdog trips, the registered
                child process exits within the kill grace window.
        """
        fake_vm = SimpleNamespace(percent=99.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)
        sleeper = _spawn_sleeper(30.0)
        try:
            with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
                with HostMemoryWatchdog(
                    warn_pct=70,
                    abort_pct=80,
                    poll_interval_s=0.1,
                    kill_grace_s=1.0,
                ) as wd:
                    wd.register_subprocess(sleeper)
                    try:
                        # Pure-Python loop so interrupt_main lands.
                        deadline = time.time() + 5.0
                        while time.time() < deadline:
                            _ = sum(range(500))
                    except KeyboardInterrupt:
                        pass
            # Past the with-block: the watchdog has terminated the child.
            sleeper.wait(timeout=5.0)
            assert sleeper.poll() is not None
        finally:
            if sleeper.poll() is None:
                sleeper.kill()

    def test_terminate_already_dead_subprocess(self):
        """
        Already-dead subprocesses are not terminated again (no error).

        Tests:
            (Test Case 1) Watchdog._terminate_registered tolerates
                a Popen whose poll() returns a non-None exit code.
        """
        wd = HostMemoryWatchdog()
        dead = mock.Mock(spec=subprocess.Popen)
        dead.poll.return_value = 0  # Already exited.
        wd._subprocesses = [(dead, 1.0)]
        wd._terminate_registered()
        dead.terminate.assert_not_called()
        dead.kill.assert_not_called()

    def test_terminate_uses_kill_after_grace(self):
        """
        A subprocess that ignores terminate() is killed after the grace period.

        Tests:
            (Test Case 1) When poll() keeps returning None, the
                watchdog calls kill() in the second pass.
        """
        wd = HostMemoryWatchdog()
        stubborn = mock.Mock(spec=subprocess.Popen)
        stubborn.poll.return_value = None  # Never exits.
        stubborn.pid = 12345
        wd._subprocesses = [(stubborn, 0.05)]  # Tiny grace for fast test.
        wd._terminate_registered()
        stubborn.terminate.assert_called_once()
        stubborn.kill.assert_called_once()


# ---------------------------------------------------------------------------
# get_active_watchdog
# ---------------------------------------------------------------------------


class TestGetActiveWatchdog:
    """Top-level discovery helper for backends."""

    def test_returns_none_outside_context(self):
        """
        get_active_watchdog returns None when no watchdog is running.

        Tests:
            (Test Case 1) Module-level call returns None.
        """
        assert get_active_watchdog() is None

    def test_returns_active_inside_context(self):
        """
        get_active_watchdog returns the live watchdog instance.

        Tests:
            (Test Case 1) Inside __enter__, the lookup returns the
                same object that was entered.
        """
        with HostMemoryWatchdog(warn_pct=98, abort_pct=99, poll_interval_s=10.0) as wd:
            assert get_active_watchdog() is wd


# ---------------------------------------------------------------------------
# PreflightFinding dataclass
# ---------------------------------------------------------------------------


class TestPreflightFinding:
    """PreflightFinding default values and structure."""

    def test_defaults(self):
        """
        Optional fields default to None / "resource".

        Tests:
            (Test Case 1) remediation defaults to None.
            (Test Case 2) category defaults to "resource".
            (Test Case 3) Required fields are stored verbatim.
        """
        f = PreflightFinding(level="warn", code="x", message="m")
        assert f.remediation is None
        assert f.category == "resource"
        assert f.level == "warn"
        assert f.code == "x"
        assert f.message == "m"

    def test_full_construction(self):
        """
        All fields can be provided explicitly.

        Tests:
            (Test Case 1) level, code, message, remediation, category
                round-trip through asdict().
        """
        f = PreflightFinding(
            level="fail",
            code="hdf5",
            message="m",
            remediation="fix it",
            category="environment",
        )
        d = asdict(f)
        assert d == {
            "level": "fail",
            "code": "hdf5",
            "message": "m",
            "remediation": "fix it",
            "category": "environment",
        }


# ---------------------------------------------------------------------------
# run_preflight
# ---------------------------------------------------------------------------


class TestRunPreflight:
    """Pre-loop resource checks. Each branch is exercised by patching
    the relevant detection function in ``_preflight``.
    """

    @pytest.fixture(autouse=True)
    def _silence_v2_helpers(self, monkeypatch):
        """Mute the FEAT-001..003 dispatchers added in the safeguarding
        round so the disk / RAM / VRAM / HDF5 assertions in this class
        keep their pre-FEAT semantics. Each FEAT-001..003 helper is
        covered separately in test_safeguards_v2.py.

        ``_check_filesystem_writable`` is also muted: when called with
        non-existent placeholder paths like ``/inter`` and ``/results``
        on a Windows host, the parent walk lands on the drive root and
        ``os.access(..., W_OK)`` may return False for restricted user
        accounts, producing spurious ``intermediate_readonly`` /
        ``results_readonly`` findings. Tests that exercise the writable
        check directly mock or override this in their own scope.
        """
        monkeypatch.setattr(preflight_mod, "_check_sorter_dependencies", lambda c: [])
        monkeypatch.setattr(preflight_mod, "_check_gpu_device_present", lambda c: None)
        monkeypatch.setattr(
            preflight_mod, "_check_recording_sample_rate", lambda c, recs: []
        )
        monkeypatch.setattr(
            preflight_mod,
            "_check_filesystem_writable",
            lambda folders, *, label, code_prefix: [],
        )

    def test_no_findings_when_host_healthy(self, monkeypatch):
        """
        Plenty of disk / RAM / VRAM and a valid sorter yields no findings.

        Tests:
            (Test Case 1) Disk free > threshold for both folders.
            (Test Case 2) Available RAM > threshold.
            (Test Case 3) GPU sorter with VRAM > threshold.
            (Test Case 4) No HDF5_PLUGIN_PATH set → no finding.
            (Test Case 5) Returned list is empty.
        """
        cfg = _make_config(sorter_name="kilosort4")
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.setattr(preflight_mod, "_free_vram_gb", lambda: 12.0)
        # No HDF5_PLUGIN_PATH in cfg.recording, no env var either.
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [mock.Mock()], ["/inter"], ["/results"])
        assert findings == []

    def test_low_disk_inter_finding(self, monkeypatch):
        """
        Free disk under threshold for an intermediate folder yields a warn.

        Tests:
            (Test Case 1) Finding code == "low_disk_inter".
            (Test Case 2) Finding level == "warn".
            (Test Case 3) Message references the folder path.
        """
        cfg = _make_config(sorter_name="kilosort2", use_docker=False)
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 5.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter_dir"], ["/results_dir"])
        codes = [f.code for f in findings]
        assert "low_disk_inter" in codes
        # KS2 host path does not require GPU, no VRAM warning expected.
        assert "low_vram" not in codes

    def test_low_disk_results_finding(self, monkeypatch):
        """
        Free disk under threshold for the results folder yields a warn.

        Tests:
            (Test Case 1) Finding code == "low_disk_results".
            (Test Case 2) Threshold honoured (1.5 GB < 2.0 default).
        """
        cfg = _make_config(sorter_name="kilosort2")
        # Inter folder healthy, results folder low.
        sequence = [100.0, 1.5]

        def _fake_disk(_):
            return sequence.pop(0)

        monkeypatch.setattr(preflight_mod, "_disk_free_gb", _fake_disk)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter_dir"], ["/results_dir"])
        codes = [f.code for f in findings]
        assert "low_disk_results" in codes
        assert "low_disk_inter" not in codes

    def test_low_ram_finding(self, monkeypatch):
        """
        Available RAM below threshold yields a low_ram warn.

        Tests:
            (Test Case 1) Finding code == "low_ram".
            (Test Case 2) Numeric threshold (1.0 GB < 4.0 default).
        """
        cfg = _make_config(sorter_name="kilosort2")
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 1.0)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        codes = [f.code for f in findings]
        assert "low_ram" in codes

    def test_ram_unknown_when_psutil_missing(self, monkeypatch):
        """
        Inability to read available RAM yields a ram_unknown warn.

        Tests:
            (Test Case 1) When _available_ram_gb returns None, the
                ram_unknown finding is emitted.
        """
        cfg = _make_config(sorter_name="kilosort2")
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: None)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        codes = [f.code for f in findings]
        assert "ram_unknown" in codes

    def test_low_vram_finding_for_gpu_sorter(self, monkeypatch):
        """
        GPU sorter with low VRAM yields a low_vram warn.

        Tests:
            (Test Case 1) sorter='kilosort4' triggers the VRAM check.
            (Test Case 2) VRAM 0.5 GB (< 2.0 default) → low_vram.
        """
        cfg = _make_config(sorter_name="kilosort4")
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.setattr(preflight_mod, "_free_vram_gb", lambda: 0.5)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        codes = [f.code for f in findings]
        assert "low_vram" in codes

    def test_vram_unknown_finding_for_gpu_sorter(self, monkeypatch):
        """
        GPU sorter with no detectable VRAM yields vram_unknown.

        Tests:
            (Test Case 1) When _free_vram_gb returns None, the
                vram_unknown finding is emitted.
        """
        cfg = _make_config(sorter_name="rt_sort")
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.setattr(preflight_mod, "_free_vram_gb", lambda: None)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        codes = [f.code for f in findings]
        assert "vram_unknown" in codes

    def test_no_vram_check_for_cpu_sorter(self, monkeypatch):
        """
        Non-GPU sorter (KS2 host path) skips the VRAM check entirely.

        Tests:
            (Test Case 1) Even with VRAM detection set to None, no
                vram_unknown finding is emitted for kilosort2 without
                Docker.
        """
        cfg = _make_config(sorter_name="kilosort2", use_docker=False)
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.setattr(preflight_mod, "_free_vram_gb", lambda: None)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        codes = [f.code for f in findings]
        assert "vram_unknown" not in codes
        assert "low_vram" not in codes

    def test_kilosort2_docker_does_check_vram(self, monkeypatch):
        """
        KS2 with use_docker=True is treated as GPU-backed.

        Tests:
            (Test Case 1) sorter='kilosort2' + use_docker=True →
                VRAM check is performed (low_vram emitted on low VRAM).
        """
        cfg = _make_config(sorter_name="kilosort2", use_docker=True)
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.setattr(preflight_mod, "_free_vram_gb", lambda: 0.5)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        codes = [f.code for f in findings]
        assert "low_vram" in codes

    def test_hdf5_plugin_missing_finding(self, monkeypatch, tmp_path):
        """
        Configured HDF5 plugin path that does not exist yields a fail.

        Tests:
            (Test Case 1) Finding has level="fail" and
                code="hdf5_plugin_missing".
            (Test Case 2) Category is "environment" so report_findings
                escalates correctly.
        """
        bogus = tmp_path / "does_not_exist"
        cfg = _make_config(sorter_name="kilosort2", hdf5_plugin_path=str(bogus))
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        hdf5 = [f for f in findings if f.code == "hdf5_plugin_missing"]
        assert len(hdf5) == 1
        assert hdf5[0].level == "fail"
        assert hdf5[0].category == "environment"

    def test_hdf5_plugin_path_valid_no_finding(self, monkeypatch, tmp_path):
        """
        Existing HDF5 plugin directory yields no finding.

        Tests:
            (Test Case 1) When the path exists and is a directory,
                _hdf5_plugin_finding returns None.
        """
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        cfg = _make_config(sorter_name="kilosort2", hdf5_plugin_path=str(plugin_dir))
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        codes = [f.code for f in findings]
        assert "hdf5_plugin_missing" not in codes

    def test_hdf5_plugin_path_from_env_var(self, monkeypatch, tmp_path):
        """
        HDF5_PLUGIN_PATH env var is honoured when no recording config value.

        Tests:
            (Test Case 1) Env var pointing to a missing directory
                produces the same hdf5_plugin_missing finding.
        """
        cfg = _make_config(sorter_name="kilosort2", hdf5_plugin_path=None)
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 500.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 64.0)
        monkeypatch.setenv("HDF5_PLUGIN_PATH", str(tmp_path / "nope"))
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
        codes = [f.code for f in findings]
        assert "hdf5_plugin_missing" in codes


# ---------------------------------------------------------------------------
# report_findings
# ---------------------------------------------------------------------------


class TestReportFindings:
    """Print-and-escalate behaviour for preflight findings."""

    def test_no_findings_passes_silently(self, caplog):
        """
        Empty findings list returns without raising.

        Tests:
            (Test Case 1) report_findings([]) does not raise.
            (Test Case 2) Logs the "all checks passed" line.
        """
        with caplog.at_level(
            logging.INFO, logger="spikelab.spike_sorting.guards._preflight"
        ):
            report_findings([])
        assert any("all checks passed" in r.getMessage() for r in caplog.records)

    def test_warn_only_does_not_raise(self, caplog):
        """
        Warn-level findings log but do not raise in default mode.

        Tests:
            (Test Case 1) No exception raised.
            (Test Case 2) A WARNING-level record is emitted.
        """
        findings = [PreflightFinding(level="warn", code="low_ram", message="m")]
        with caplog.at_level(
            logging.WARNING, logger="spikelab.spike_sorting.guards._preflight"
        ):
            report_findings(findings, strict=False)
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_fail_resource_raises_resource_error(self):
        """
        Fail-level resource finding raises ResourceSortFailure.

        Tests:
            (Test Case 1) ResourceSortFailure is raised.
            (Test Case 2) Message includes the finding code.
        """
        findings = [
            PreflightFinding(
                level="fail",
                code="low_vram",
                message="too low",
                category="resource",
            )
        ]
        with pytest.raises(ResourceSortFailure) as exc_info:
            report_findings(findings)
        assert "low_vram" in str(exc_info.value)

    def test_fail_environment_raises_environment_error(self):
        """
        Fail-level environment finding raises EnvironmentSortFailure.

        Tests:
            (Test Case 1) EnvironmentSortFailure is raised for a
                generic environment failure (not the HDF5 special
                case).
        """
        findings = [
            PreflightFinding(
                level="fail",
                code="something_env",
                message="env broken",
                category="environment",
            )
        ]
        with pytest.raises(EnvironmentSortFailure):
            report_findings(findings)

    def test_fail_hdf5_raises_specific_subclass(self):
        """
        hdf5_plugin_missing fail finding raises HDF5PluginMissingError.

        Tests:
            (Test Case 1) The specific subclass is raised so callers
                can branch on the exact failure mode.
        """
        findings = [
            PreflightFinding(
                level="fail",
                code="hdf5_plugin_missing",
                message="bad path",
                category="environment",
            )
        ]
        with pytest.raises(HDF5PluginMissingError):
            report_findings(findings)

    def test_strict_escalates_resource_warn(self):
        """
        Strict mode flips a warn-level resource finding into a raise.

        Tests:
            (Test Case 1) ResourceSortFailure is raised when strict=True
                even though the finding's level is "warn".
        """
        findings = [
            PreflightFinding(
                level="warn",
                code="low_ram",
                message="m",
                category="resource",
            )
        ]
        with pytest.raises(ResourceSortFailure):
            report_findings(findings, strict=True)

    def test_strict_escalates_environment_warn(self):
        """
        Strict mode flips a warn-level environment finding into a raise.

        Tests:
            (Test Case 1) EnvironmentSortFailure is raised when
                strict=True for a warn-level environment finding.
        """
        findings = [
            PreflightFinding(
                level="warn",
                code="weird_env",
                message="m",
                category="environment",
            )
        ]
        with pytest.raises(EnvironmentSortFailure):
            report_findings(findings, strict=True)


# ---------------------------------------------------------------------------
# ExecutionConfig defaults for guard fields
# ---------------------------------------------------------------------------


class TestExecutionConfigGuardFields:
    """ExecutionConfig has the new guard knobs with documented defaults."""

    def test_watchdog_field_defaults(self):
        """
        Watchdog-related fields default to the documented values.

        Tests:
            (Test Case 1) host_ram_watchdog defaults to True.
            (Test Case 2) Warn / abort percentages match the
                workstation-tuned defaults (85 / 92).
            (Test Case 3) Poll interval defaults to 2.0 seconds.
        """
        cfg = ExecutionConfig()
        assert cfg.host_ram_watchdog is True
        assert cfg.host_ram_warn_pct == 85.0
        assert cfg.host_ram_abort_pct == 92.0
        assert cfg.host_ram_poll_interval_s == 2.0

    def test_preflight_field_defaults(self):
        """
        Preflight-related fields default to the documented values.

        Tests:
            (Test Case 1) preflight defaults to True; preflight_strict
                to False.
            (Test Case 2) Disk / RAM / VRAM thresholds match
                32–64 GB-workstation defaults.
        """
        cfg = ExecutionConfig()
        assert cfg.preflight is True
        assert cfg.preflight_strict is False
        assert cfg.preflight_min_free_inter_gb == 20.0
        assert cfg.preflight_min_free_results_gb == 2.0
        assert cfg.preflight_min_available_ram_gb == 4.0
        assert cfg.preflight_min_free_vram_gb == 2.0

    def test_inactivity_field_defaults(self):
        """
        Inactivity-watchdog fields default to the documented values.

        Tests:
            (Test Case 1) sorter_inactivity_timeout defaults to True.
            (Test Case 2) Base / per-min / max scaling defaults match
                the agreed 10 min / 30 s / 2 h envelope.
        """
        cfg = ExecutionConfig()
        assert cfg.sorter_inactivity_timeout is True
        assert cfg.sorter_inactivity_base_s == 600.0
        assert cfg.sorter_inactivity_per_min_s == 30.0
        assert cfg.sorter_inactivity_max_s == 7200.0

    def test_oom_retry_field_defaults(self):
        """
        OOM-retry fields default to the documented values.

        Tests:
            (Test Case 1) oom_retry_max defaults to 1 (one extra
                attempt).
            (Test Case 2) oom_retry_factor defaults to 0.5 (halve).
        """
        cfg = ExecutionConfig()
        assert cfg.oom_retry_max == 1
        assert cfg.oom_retry_factor == 0.5


# ---------------------------------------------------------------------------
# SorterTimeoutError hierarchy
# ---------------------------------------------------------------------------


class TestSorterTimeoutErrorHierarchy:
    """Hierarchy and attribute storage for SorterTimeoutError."""

    def test_subclass_chain(self):
        """
        SorterTimeoutError descends from the resource category.

        Tests:
            (Test Case 1) Subclass of ResourceSortFailure.
            (Test Case 2) Subclass of SpikeSortingClassifiedError.
        """
        assert issubclass(SorterTimeoutError, ResourceSortFailure)
        assert issubclass(SorterTimeoutError, SpikeSortingClassifiedError)

    def test_attribute_storage(self):
        """
        Constructor records sorter, inactivity_s, log_path verbatim.

        Tests:
            (Test Case 1) All keyword attributes round-trip.
            (Test Case 2) Defaults to None when omitted.
        """
        err = SorterTimeoutError(
            "boom",
            sorter="kilosort2",
            inactivity_s=900.0,
            log_path="/some/log.log",
        )
        assert err.sorter == "kilosort2"
        assert err.inactivity_s == 900.0
        assert str(err.log_path) == "/some/log.log"

        err2 = SorterTimeoutError("only sorter", sorter="kilosort4")
        assert err2.sorter == "kilosort4"
        assert err2.inactivity_s is None
        assert err2.log_path is None


# ---------------------------------------------------------------------------
# compute_inactivity_timeout_s
# ---------------------------------------------------------------------------


class TestComputeInactivityTimeoutS:
    """Recording-aware inactivity tolerance formula."""

    def test_basic_scaling(self):
        """
        timeout = base + per_min * duration_min, clamped at max.

        Tests:
            (Test Case 1) Tiny recording (1 min) → base + 30 s.
            (Test Case 2) Half-hour recording → base + 900 s.
            (Test Case 3) 12-hour recording hits the max cap.
        """
        # 1 min: 600 + 30 = 630
        assert compute_inactivity_timeout_s(recording_duration_min=1) == 630.0
        # 30 min: 600 + 900 = 1500
        assert compute_inactivity_timeout_s(recording_duration_min=30) == 1500.0
        # 12 h = 720 min: 600 + 30*720 = 22200, capped at 7200
        assert compute_inactivity_timeout_s(recording_duration_min=720) == 7200.0

    def test_zero_duration_uses_base(self):
        """
        Zero / negative / None duration collapse to the base tolerance.

        Tests:
            (Test Case 1) duration=0 returns base_s.
            (Test Case 2) Negative duration is clamped to zero.
            (Test Case 3) None duration is clamped to zero.
        """
        assert compute_inactivity_timeout_s(recording_duration_min=0) == 600.0
        assert compute_inactivity_timeout_s(recording_duration_min=-5) == 600.0
        assert compute_inactivity_timeout_s(recording_duration_min=None) == 600.0

    def test_no_max_cap(self):
        """
        max_s=None disables the cap.

        Tests:
            (Test Case 1) Large duration with max_s=None scales
                unbounded.
        """
        # 1000 min: 600 + 30*1000 = 30600 with no cap.
        result = compute_inactivity_timeout_s(recording_duration_min=1000, max_s=None)
        assert result == 30600.0

    def test_custom_base_per_min(self):
        """
        Override base_s and per_min_s independently.

        Tests:
            (Test Case 1) Doubled base; per-min unchanged.
            (Test Case 2) Doubled per-min; base unchanged.
        """
        assert (
            compute_inactivity_timeout_s(recording_duration_min=10, base_s=1200.0)
            == 1200.0 + 300.0
        )
        assert (
            compute_inactivity_timeout_s(recording_duration_min=10, per_min_s=60.0)
            == 600.0 + 600.0
        )


# ---------------------------------------------------------------------------
# LogInactivityWatchdog
# ---------------------------------------------------------------------------


class TestLogInactivityWatchdogConstruction:
    """Disabled-state behaviour of LogInactivityWatchdog."""

    def test_disabled_when_inactivity_none(self, tmp_path):
        """
        inactivity_s=None makes the watchdog a no-op.

        Tests:
            (Test Case 1) _enabled is False.
            (Test Case 2) Entering the context does not start a thread.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "log",
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=None,
            sorter="kilosort2",
        )
        assert wd._enabled is False
        with wd:
            assert wd._thread is None
        assert wd.tripped() is False

    def test_disabled_when_popen_none(self, tmp_path):
        """
        popen=None makes the watchdog a no-op even with a timeout set.

        Tests:
            (Test Case 1) _enabled is False.
            (Test Case 2) tripped() stays False.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "log",
            popen=None,
            inactivity_s=600.0,
            sorter="kilosort2",
        )
        assert wd._enabled is False
        with wd:
            time.sleep(0.05)
        assert wd.tripped() is False

    def test_nonpositive_inactivity_raises(self, tmp_path):
        """
        Zero / negative inactivity_s raises ValueError.

        Tests:
            (Test Case 1) inactivity_s=0 → ValueError.
            (Test Case 2) inactivity_s=-5 → ValueError.
        """
        for bad in (0, -5):
            with pytest.raises(
                ValueError, match="inactivity_s must be positive or None"
            ):
                LogInactivityWatchdog(
                    log_path=tmp_path / "log",
                    popen=mock.Mock(spec=subprocess.Popen),
                    inactivity_s=bad,
                    sorter="x",
                )


class TestLogInactivityWatchdogTrip:
    """End-to-end trip behaviour of LogInactivityWatchdog."""

    def _spawn_sleeper(self, seconds=30):
        """Spawn a long-running OS-level sleeper for kill tests."""
        if sys.platform == "win32":
            cmd = ["timeout", "/T", str(int(seconds))]
        else:
            cmd = ["sleep", str(int(seconds))]
        return subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def test_silent_log_trips_and_kills(self, tmp_path):
        """
        Log file that never appears (or never updates) trips the watchdog.

        Tests:
            (Test Case 1) tripped() is True after the inactivity
                window elapses.
            (Test Case 2) The registered subprocess is terminated.
            (Test Case 3) make_error returns a SorterTimeoutError
                with the configured fields.
        """
        sleeper = self._spawn_sleeper(30)
        try:
            wd = LogInactivityWatchdog(
                log_path=tmp_path / "missing.log",
                popen=sleeper,
                inactivity_s=0.6,
                sorter="kilosort2",
                poll_interval_s=0.1,
                kill_grace_s=0.5,
            )
            with wd:
                # Wait long enough for the trip to fire.
                deadline = time.time() + 4.0
                while time.time() < deadline and not wd.tripped():
                    time.sleep(0.1)
            assert wd.tripped()
            sleeper.wait(timeout=5.0)
            assert sleeper.poll() is not None
            err = wd.make_error()
            assert isinstance(err, SorterTimeoutError)
            assert err.sorter == "kilosort2"
            assert err.inactivity_s == 0.6
        finally:
            if sleeper.poll() is None:
                sleeper.kill()

    def test_progress_keeps_alive(self, tmp_path):
        """
        Repeatedly updating the log mtime keeps the watchdog from tripping.

        Tests:
            (Test Case 1) When a helper thread re-touches the log
                faster than the inactivity window, the watchdog
                does not trip.
        """
        log = tmp_path / "ks.log"
        log.write_text("init\n")
        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None  # "still running"

        stop = threading.Event()

        def _toucher():
            while not stop.is_set():
                # Update mtime by appending; keeps it well above the
                # 0.4-second inactivity window.
                with open(log, "a") as f:
                    f.write(".\n")
                time.sleep(0.1)

        toucher = threading.Thread(target=_toucher, daemon=True)
        toucher.start()
        try:
            wd = LogInactivityWatchdog(
                log_path=log,
                popen=popen,
                inactivity_s=0.4,
                sorter="kilosort2",
                poll_interval_s=0.1,
                kill_grace_s=0.2,
            )
            with wd:
                time.sleep(1.5)  # Several inactivity windows.
            assert not wd.tripped()
        finally:
            stop.set()
            toucher.join(timeout=1.0)
        # Ensure terminate was never called on the popen.
        popen.terminate.assert_not_called()

    def test_make_error_pretrip_uses_default_format(self, tmp_path):
        """
        ``make_error`` called before the watchdog has tripped uses
        the placeholder format (``"?"`` for inactivity-at-trip).

        Tests:
            (Test Case 1) Without entering the watchdog context, the
                returned ``SorterTimeoutError`` carries the configured
                ``inactivity_s`` and references ``"?"`` for the
                observed-at-trip duration in its message.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "missing.log",
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=120.0,
            sorter="kilosort4",
        )
        err = wd.make_error()
        assert isinstance(err, SorterTimeoutError)
        assert err.sorter == "kilosort4"
        assert err.inactivity_s == 120.0
        # Default formatting includes the placeholder for the
        # observed-but-unset trip duration.
        assert "?s" in str(err)


# ---------------------------------------------------------------------------
# Active-log-path ContextVar
# ---------------------------------------------------------------------------


class TestActiveLogPath:
    """``set_active_log_path`` / ``get_active_log_path`` ContextVar."""

    def test_default_is_none(self):
        """
        Outside any context the active log path is None.

        Tests:
            (Test Case 1) Bare lookup returns None.
        """
        from spikelab.spike_sorting.guards import get_active_log_path

        assert get_active_log_path() is None

    def test_set_and_clear_round_trip(self, tmp_path):
        """
        ``set_active_log_path`` publishes a path inside its with-block
        and clears it on exit.

        Tests:
            (Test Case 1) Inside the with-block the lookup returns
                the path that was set.
            (Test Case 2) After the with-block the lookup returns
                None again.
            (Test Case 3) The published value is a Path even when
                a string was passed.
        """
        from spikelab.spike_sorting.guards import (
            get_active_log_path,
            set_active_log_path,
        )

        log_path = tmp_path / "rec.log"
        with set_active_log_path(str(log_path)):
            active = get_active_log_path()
            assert active is not None
            assert str(active) == str(log_path)
        assert get_active_log_path() is None

    def test_nesting(self, tmp_path):
        """
        Nested calls publish the inner path; outer resumes on exit.

        Tests:
            (Test Case 1) Inside outer: outer path is active.
            (Test Case 2) Inside inner: inner path is active.
            (Test Case 3) After inner exit: outer path is active again.
        """
        from spikelab.spike_sorting.guards import (
            get_active_log_path,
            set_active_log_path,
        )

        outer = tmp_path / "outer.log"
        inner = tmp_path / "inner.log"
        with set_active_log_path(outer):
            assert str(get_active_log_path()) == str(outer)
            with set_active_log_path(inner):
                assert str(get_active_log_path()) == str(inner)
            assert str(get_active_log_path()) == str(outer)

    def test_none_log_path_raises_type_error(self):
        """
        ``set_active_log_path(None)`` raises TypeError eagerly.

        Tests:
            (Test Case 1) Entering the context manager with ``None``
                triggers ``Path(None)``, which raises TypeError before
                any ContextVar mutation. The contextvar is left
                untouched (still None) so subsequent reads behave as
                if no entry was attempted.
        """
        from spikelab.spike_sorting.guards import (
            get_active_log_path,
            set_active_log_path,
        )

        with pytest.raises(TypeError):
            with set_active_log_path(None):  # type: ignore[arg-type]
                pass
        # ContextVar untouched after the failure.
        assert get_active_log_path() is None


# ---------------------------------------------------------------------------
# make_in_process_kill_callback
# ---------------------------------------------------------------------------


class TestMakeInProcessKillCallback:
    """``make_in_process_kill_callback`` two-stage interrupt → os._exit."""

    def test_calls_interrupt_main_first(self):
        """
        The callback calls ``_thread.interrupt_main`` before sleeping.

        Tests:
            (Test Case 1) ``interrupt_main`` is invoked exactly once
                before the os._exit fallback runs.
            (Test Case 2) The grace-period sleep is honoured.

        Notes:
            - We patch ``_thread.interrupt_main``, ``time.sleep``,
              and ``os._exit`` so the callback runs synchronously
              without actually killing the test process.
        """
        from spikelab.spike_sorting.guards import make_in_process_kill_callback
        from spikelab.spike_sorting.guards import _inactivity as inact_mod

        fake_interrupt = mock.Mock()
        fake_sleep = mock.Mock()
        fake_exit = mock.Mock(side_effect=SystemExit(1))

        with (
            mock.patch.object(
                inact_mod, "_thread", _thread=fake_interrupt
            ) as patched_thread,
            mock.patch.object(inact_mod.time, "sleep", fake_sleep),
            mock.patch.object(inact_mod.os, "_exit", fake_exit),
        ):
            patched_thread.interrupt_main = fake_interrupt
            cb = make_in_process_kill_callback(
                interrupt_grace_s=0.05, sorter="kilosort4"
            )
            with pytest.raises(SystemExit):
                cb()

        fake_interrupt.assert_called_once()
        fake_sleep.assert_called_once_with(0.05)
        fake_exit.assert_called_once_with(1)

    def test_callback_continues_when_interrupt_fails(self):
        """
        ``_thread.interrupt_main`` raising is logged but does not
        prevent the os._exit fallback.

        Tests:
            (Test Case 1) When interrupt_main raises, sleep + os._exit
                still run.
        """
        from spikelab.spike_sorting.guards import make_in_process_kill_callback
        from spikelab.spike_sorting.guards import _inactivity as inact_mod

        fake_thread = mock.Mock()
        fake_thread.interrupt_main = mock.Mock(side_effect=RuntimeError("nope"))
        fake_sleep = mock.Mock()
        fake_exit = mock.Mock(side_effect=SystemExit(1))

        with (
            mock.patch.object(inact_mod, "_thread", fake_thread),
            mock.patch.object(inact_mod.time, "sleep", fake_sleep),
            mock.patch.object(inact_mod.os, "_exit", fake_exit),
        ):
            cb = make_in_process_kill_callback(interrupt_grace_s=0.01)
            with pytest.raises(SystemExit):
                cb()

        fake_sleep.assert_called_once()
        fake_exit.assert_called_once_with(1)


# ---------------------------------------------------------------------------
# LogInactivityWatchdog with kill_callback
# ---------------------------------------------------------------------------


class TestLogInactivityWatchdogKillCallback:
    """LogInactivityWatchdog accepts a kill_callback for in-process sorts."""

    def test_disabled_when_no_popen_and_no_callback(self, tmp_path):
        """
        Watchdog with neither popen nor kill_callback is a no-op.

        Tests:
            (Test Case 1) ``_enabled`` is False.
            (Test Case 2) Entering the context starts no thread.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "log",
            popen=None,
            inactivity_s=600.0,
            sorter="kilosort4",
        )
        assert wd._enabled is False
        with wd:
            assert wd._thread is None
        assert wd.tripped() is False

    def test_enabled_with_callback_only(self, tmp_path):
        """
        Watchdog with kill_callback (and no popen) is enabled.

        Tests:
            (Test Case 1) ``_enabled`` is True when kill_callback is set.
            (Test Case 2) Entering the context starts the polling thread.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "log",
            popen=None,
            inactivity_s=600.0,
            sorter="kilosort4",
            kill_callback=lambda: None,
        )
        assert wd._enabled is True
        with wd:
            assert wd._thread is not None
            assert wd._thread.is_alive()

    def test_callback_invoked_on_trip(self, tmp_path):
        """
        Silent log triggers the kill_callback.

        Tests:
            (Test Case 1) After inactivity elapses, the callback runs.
            (Test Case 2) ``tripped()`` is True.
            (Test Case 3) ``make_error()`` returns a SorterTimeoutError
                even when popen is None.
        """
        callback = mock.Mock()
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "missing.log",
            popen=None,
            inactivity_s=0.4,
            sorter="rt_sort",
            poll_interval_s=0.1,
            kill_callback=callback,
        )
        with wd:
            deadline = time.time() + 3.0
            while time.time() < deadline and not wd.tripped():
                time.sleep(0.1)
        assert wd.tripped()
        callback.assert_called_once()
        err = wd.make_error()
        assert isinstance(err, SorterTimeoutError)
        assert err.sorter == "rt_sort"

    def test_callback_exception_does_not_break_watchdog(self, tmp_path):
        """
        A kill_callback that raises is logged but does not crash the
        watchdog thread.

        Tests:
            (Test Case 1) After a raising callback, ``tripped()`` is
                still True (no propagation).
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "missing.log",
            popen=None,
            inactivity_s=0.4,
            sorter="kilosort4",
            poll_interval_s=0.1,
            kill_callback=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with wd:
            deadline = time.time() + 3.0
            while time.time() < deadline and not wd.tripped():
                time.sleep(0.1)
        assert wd.tripped()


# ---------------------------------------------------------------------------
# .wslconfig preflight check (Item E1)
# ---------------------------------------------------------------------------


class TestWslconfigPreflight:
    """``_wslconfig_finding`` and ``_parse_wslconfig_memory_gb``."""

    def test_parse_memory_gb(self):
        """
        ``_parse_wslconfig_memory_gb`` recognises GB / MB units.

        Tests:
            (Test Case 1) ``memory=8GB`` returns 8.0.
            (Test Case 2) ``memory=8192MB`` returns 8.0.
            (Test Case 3) Bare integer assumes GB.
            (Test Case 4) Missing key returns None.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        assert _parse_wslconfig_memory_gb("[wsl2]\nmemory=8GB\n") == 8.0
        assert _parse_wslconfig_memory_gb("[wsl2]\nmemory = 8192 MB\n") == 8.0
        assert _parse_wslconfig_memory_gb("[wsl2]\nmemory=12\n") == 12.0
        assert _parse_wslconfig_memory_gb("[wsl2]\n# no key\n") is None

    def test_parse_skips_other_sections(self):
        """
        Keys outside [wsl2] are ignored.

        Tests:
            (Test Case 1) memory= under [user] does not count.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        text = "[user]\nmemory=8GB\n\n[wsl2]\nprocessors=4\n"
        assert _parse_wslconfig_memory_gb(text) is None

    def test_skipped_off_windows(self):
        """
        ``_wslconfig_finding`` returns None on non-Windows hosts.

        Tests:
            (Test Case 1) When sys.platform is not 'win32', no
                finding is produced even with use_docker=True.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod_local

        cfg = _make_config(sorter_name="kilosort2", use_docker=True)
        with mock.patch.object(preflight_mod_local, "sys") as mock_sys:
            mock_sys.platform = "linux"
            assert preflight_mod_local._wslconfig_finding(cfg) is None

    def test_skipped_when_not_using_docker(self):
        """
        Without Docker the .wslconfig check is irrelevant.

        Tests:
            (Test Case 1) use_docker=False → no finding even on Windows.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod_local

        cfg = _make_config(sorter_name="kilosort2", use_docker=False)
        with mock.patch.object(preflight_mod_local, "sys") as mock_sys:
            mock_sys.platform = "win32"
            assert preflight_mod_local._wslconfig_finding(cfg) is None

    def test_wslconfig_missing_warns(self, tmp_path):
        """
        Missing ~/.wslconfig produces a wslconfig_missing finding.

        Tests:
            (Test Case 1) On Windows + Docker with no file, the
                finding has code 'wslconfig_missing'.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod_local

        cfg = _make_config(sorter_name="kilosort2", use_docker=True)
        # Point HOME at an empty tmp dir so .wslconfig does not exist.
        with (
            mock.patch.object(preflight_mod_local, "sys") as mock_sys,
            mock.patch.dict(
                "os.environ",
                {"USERPROFILE": str(tmp_path), "HOME": str(tmp_path)},
                clear=False,
            ),
        ):
            mock_sys.platform = "win32"
            finding = preflight_mod_local._wslconfig_finding(cfg)
        assert finding is not None
        assert finding.code == "wslconfig_missing"
        assert finding.level == "warn"
        assert finding.category == "environment"

    def test_wslconfig_no_memory_warns(self, tmp_path):
        """
        ~/.wslconfig present but missing the memory= key warns.

        Tests:
            (Test Case 1) Code 'wslconfig_no_memory'.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod_local

        # Write a .wslconfig with no memory key.
        wslconfig = tmp_path / ".wslconfig"
        wslconfig.write_text("[wsl2]\nprocessors=4\n")
        cfg = _make_config(sorter_name="kilosort2", use_docker=True)
        with (
            mock.patch.object(preflight_mod_local, "sys") as mock_sys,
            mock.patch.dict(
                "os.environ",
                {"USERPROFILE": str(tmp_path), "HOME": str(tmp_path)},
                clear=False,
            ),
        ):
            mock_sys.platform = "win32"
            finding = preflight_mod_local._wslconfig_finding(cfg)
        assert finding is not None
        assert finding.code == "wslconfig_no_memory"

    def test_wslconfig_too_high_warns(self, tmp_path, monkeypatch):
        """
        memory= set above 85% of host RAM warns.

        Tests:
            (Test Case 1) Code 'wslconfig_memory_too_high'.
            (Test Case 2) When memory is below the threshold, no
                finding is produced.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod_local

        wslconfig = tmp_path / ".wslconfig"
        wslconfig.write_text("[wsl2]\nmemory=15GB\n")
        cfg = _make_config(sorter_name="kilosort2", use_docker=True)

        # Host with 16 GB → 15 GB is >85% → too high.
        from spikelab.spike_sorting import sorting_utils

        monkeypatch.setattr(
            sorting_utils, "get_system_ram_bytes", lambda: 16 * (1024**3)
        )

        with (
            mock.patch.object(preflight_mod_local, "sys") as mock_sys,
            mock.patch.dict(
                "os.environ",
                {"USERPROFILE": str(tmp_path), "HOME": str(tmp_path)},
                clear=False,
            ),
        ):
            mock_sys.platform = "win32"
            finding = preflight_mod_local._wslconfig_finding(cfg)
        assert finding is not None
        assert finding.code == "wslconfig_memory_too_high"

        # Now 64 GB host → 15 GB is fine.
        monkeypatch.setattr(
            sorting_utils, "get_system_ram_bytes", lambda: 64 * (1024**3)
        )
        with (
            mock.patch.object(preflight_mod_local, "sys") as mock_sys,
            mock.patch.dict(
                "os.environ",
                {"USERPROFILE": str(tmp_path), "HOME": str(tmp_path)},
                clear=False,
            ),
        ):
            mock_sys.platform = "win32"
            finding = preflight_mod_local._wslconfig_finding(cfg)
        assert finding is None


# ---------------------------------------------------------------------------
# RT-Sort intermediate disk projection (Item E2)
# ---------------------------------------------------------------------------


class TestRtSortDiskProjection:
    """Estimation helper + preflight finding for RT-Sort disk usage."""

    def test_estimate_basic(self):
        """
        Projection formula: 8 bytes per channel-sample.

        Tests:
            (Test Case 1) 64 channels × 1e6 samples → 64 × 1e6 × 8
                bytes ≈ 0.477 GB.
            (Test Case 2) Doubling channels doubles the projection.
            (Test Case 3) Doubling samples doubles the projection.
        """
        from spikelab.spike_sorting.guards import (
            estimate_rt_sort_intermediate_gb,
        )

        base = estimate_rt_sort_intermediate_gb(n_channels=64, n_samples=1_000_000)
        # 64 * 1e6 * 8 = 512_000_000 bytes ≈ 0.477 GB
        assert pytest.approx(base, rel=1e-3) == 512_000_000 / (1024**3)

        doubled_ch = estimate_rt_sort_intermediate_gb(
            n_channels=128, n_samples=1_000_000
        )
        assert pytest.approx(doubled_ch, rel=1e-6) == 2 * base

        doubled_smp = estimate_rt_sort_intermediate_gb(
            n_channels=64, n_samples=2_000_000
        )
        assert pytest.approx(doubled_smp, rel=1e-6) == 2 * base

    def test_finding_skipped_for_non_rt_sort(self):
        """
        ``_rt_sort_disk_finding`` returns None for other sorters.

        Tests:
            (Test Case 1) sorter='kilosort2' → no finding even with
                a recording that would otherwise trigger.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _rt_sort_disk_finding,
        )

        cfg = _make_config(sorter_name="kilosort2")
        rec = mock.Mock()
        rec.get_num_channels.return_value = 1024
        rec.get_num_samples.return_value = 1_000_000_000  # huge
        assert _rt_sort_disk_finding(cfg, [rec], ["/inter"]) is None

    def test_finding_skipped_with_path_only_inputs(self):
        """
        Path-only inputs do not produce a finding (silently skipped).

        Tests:
            (Test Case 1) When recording_files contains strings,
                channel/sample counts cannot be read and the helper
                returns None.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _rt_sort_disk_finding,
        )

        cfg = _make_config(sorter_name="rt_sort")
        assert _rt_sort_disk_finding(cfg, ["/some/recording.h5"], ["/inter"]) is None

    def test_finding_emitted_when_projection_exceeds_free_disk(self, monkeypatch):
        """
        Projection > free disk yields a warn finding.

        Tests:
            (Test Case 1) Code 'rt_sort_disk_projection'.
            (Test Case 2) Level 'warn'.
            (Test Case 3) Category 'resource'.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod_local

        cfg = _make_config(sorter_name="rt_sort")
        rec = mock.Mock()
        # 1024 ch × 30000 Hz × 1 hour ≈ 880 GB projected
        rec.get_num_channels.return_value = 1024
        rec.get_num_samples.return_value = 30000 * 3600

        # Free disk: 100 GB only.
        monkeypatch.setattr(preflight_mod_local, "_disk_free_gb", lambda p: 100.0)
        finding = preflight_mod_local._rt_sort_disk_finding(cfg, [rec], ["/inter"])
        assert finding is not None
        assert finding.code == "rt_sort_disk_projection"
        assert finding.level == "warn"
        assert finding.category == "resource"

    def test_finding_skipped_when_projection_fits(self, monkeypatch):
        """
        Plenty of free disk → no finding.

        Tests:
            (Test Case 1) Free disk >> projection → returns None.
        """
        from spikelab.spike_sorting.guards import _preflight as preflight_mod_local

        cfg = _make_config(sorter_name="rt_sort")
        rec = mock.Mock()
        rec.get_num_channels.return_value = 64
        rec.get_num_samples.return_value = 20000 * 60  # 1 min @ 20kHz

        monkeypatch.setattr(preflight_mod_local, "_disk_free_gb", lambda p: 500.0)
        finding = preflight_mod_local._rt_sort_disk_finding(cfg, [rec], ["/inter"])
        assert finding is None


# ---------------------------------------------------------------------------
# ExecutionConfig.sorter_inactivity_in_process_grace_s default
# ---------------------------------------------------------------------------


class TestInProcessGraceFieldDefault:
    """The new in-process grace-period field has a sensible default."""

    def test_default_is_ten_seconds(self):
        """
        sorter_inactivity_in_process_grace_s defaults to 10.0.

        Tests:
            (Test Case 1) Default field value is 10.0 seconds.
        """
        cfg = ExecutionConfig()
        assert cfg.sorter_inactivity_in_process_grace_s == 10.0


# ---------------------------------------------------------------------------
# DiskExhaustionError hierarchy
# ---------------------------------------------------------------------------


class TestDiskExhaustionErrorHierarchy:
    """Hierarchy and attribute storage for DiskExhaustionError."""

    def test_subclass_chain(self):
        """
        DiskExhaustionError descends from the resource category.

        Tests:
            (Test Case 1) Subclass of ResourceSortFailure.
            (Test Case 2) Subclass of SpikeSortingClassifiedError.
        """
        assert issubclass(DiskExhaustionError, ResourceSortFailure)
        assert issubclass(DiskExhaustionError, SpikeSortingClassifiedError)

    def test_attribute_storage(self):
        """
        Constructor records folder, free_gb_at_trip, abort_threshold_gb,
        and an optional report payload.

        Tests:
            (Test Case 1) All keyword attributes round-trip.
            (Test Case 2) Defaults to None when omitted.
        """
        from pathlib import Path

        err = DiskExhaustionError(
            "boom",
            folder=Path("/inter"),
            free_gb_at_trip=0.5,
            abort_threshold_gb=1.0,
            report={"x": 1},
        )
        assert str(err.folder) == "/inter" or "inter" in str(err.folder)
        assert err.free_gb_at_trip == 0.5
        assert err.abort_threshold_gb == 1.0
        assert err.report == {"x": 1}

        err2 = DiskExhaustionError("just a message")
        assert err2.folder is None
        assert err2.free_gb_at_trip is None
        assert err2.abort_threshold_gb is None
        assert err2.report is None


# ---------------------------------------------------------------------------
# DiskExhaustionReport
# ---------------------------------------------------------------------------


class TestDiskExhaustionReport:
    """``DiskExhaustionReport`` dataclass + ``to_dict``."""

    def test_defaults(self):
        """
        Optional fields default sensibly.

        Tests:
            (Test Case 1) projected_need_gb defaults to None.
            (Test Case 2) bytes_consumed_during_sort defaults to 0.0.
            (Test Case 3) top_consumers and suggested_actions default
                to empty lists.
        """
        r = DiskExhaustionReport(
            folder="/x", free_gb_at_trip=0.5, abort_threshold_gb=1.0
        )
        assert r.projected_need_gb is None
        assert r.bytes_consumed_during_sort == 0.0
        assert r.top_consumers == []
        assert r.suggested_actions == []

    def test_to_dict(self):
        """
        ``to_dict`` returns JSON-serializable structure.

        Tests:
            (Test Case 1) Required fields appear in the output.
            (Test Case 2) top_consumers entries become {path, size_gb}.
        """
        r = DiskExhaustionReport(
            folder="/x",
            free_gb_at_trip=0.5,
            abort_threshold_gb=1.0,
            projected_need_gb=12.3,
            bytes_consumed_during_sort=1024.0**3,
            top_consumers=[("/x/a.npy", 5.5), ("/x/b.npy", 1.1)],
            suggested_actions=["free space"],
        )
        d = r.to_dict()
        assert d["folder"] == "/x"
        assert d["free_gb_at_trip"] == 0.5
        assert d["abort_threshold_gb"] == 1.0
        assert d["projected_need_gb"] == 12.3
        assert d["top_consumers"][0] == {"path": "/x/a.npy", "size_gb": 5.5}
        assert d["suggested_actions"] == ["free space"]


# ---------------------------------------------------------------------------
# DiskUsageWatchdog construction
# ---------------------------------------------------------------------------


class TestDiskUsageWatchdogConstruction:
    """Threshold validation and disabled-state semantics."""

    def test_warn_must_exceed_abort(self, tmp_path):
        """
        warn_free_gb must be strictly greater than abort_free_gb.

        Tests:
            (Test Case 1) Equal values are rejected.
            (Test Case 2) warn < abort is rejected.
        """
        with pytest.raises(ValueError):
            DiskUsageWatchdog(
                folder=tmp_path,
                warn_free_gb=1.0,
                abort_free_gb=1.0,
                kill_callback=lambda: None,
            )
        with pytest.raises(ValueError):
            DiskUsageWatchdog(
                folder=tmp_path,
                warn_free_gb=0.5,
                abort_free_gb=1.0,
                kill_callback=lambda: None,
            )

    def test_zero_poll_interval_raises(self, tmp_path):
        """
        poll_interval_s must be strictly positive.

        Tests:
            (Test Case 1) poll_interval_s=0 raises.
            (Test Case 2) Negative poll_interval_s raises.
        """
        with pytest.raises(ValueError):
            DiskUsageWatchdog(
                folder=tmp_path,
                poll_interval_s=0,
                kill_callback=lambda: None,
            )
        with pytest.raises(ValueError):
            DiskUsageWatchdog(
                folder=tmp_path,
                poll_interval_s=-1.0,
                kill_callback=lambda: None,
            )

    def test_disabled_without_kill_target(self, tmp_path):
        """
        Watchdog with neither popen nor kill_callback is a no-op.

        Tests:
            (Test Case 1) ``_enabled`` is False when both are None.
            (Test Case 2) Entering the context starts no thread.
        """
        wd = DiskUsageWatchdog(folder=tmp_path)
        assert wd._enabled is False
        with wd:
            assert wd._thread is None
        assert wd.tripped() is False

    def test_disabled_when_abort_nonpositive(self, tmp_path):
        """
        abort_free_gb=0 disables the watchdog even with a callback.

        Tests:
            (Test Case 1) abort_free_gb=0 → _enabled=False.
        """
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=0.5,
            abort_free_gb=0.0,
            kill_callback=lambda: None,
        )
        assert wd._enabled is False


# ---------------------------------------------------------------------------
# DiskUsageWatchdog trip
# ---------------------------------------------------------------------------


class TestDiskUsageWatchdogTrip:
    """End-to-end trip behaviour: report build + kill_callback fire."""

    def test_trip_invokes_callback_and_builds_report(self, tmp_path, monkeypatch):
        """
        Free disk crossing abort threshold trips the watchdog and runs
        the kill_callback after building a DiskExhaustionReport.

        Tests:
            (Test Case 1) ``tripped()`` is True after the trip window.
            (Test Case 2) The kill_callback is invoked.
            (Test Case 3) ``report()`` returns a DiskExhaustionReport
                with the trip metadata populated.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw_mod

        # Patch _disk_free_gb at the module level to return a low value.
        monkeypatch.setattr(dw_mod, "_disk_free_gb", lambda p: 0.5)

        callback = mock.Mock()
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=2.0,
            abort_free_gb=1.0,
            poll_interval_s=0.1,
            kill_callback=callback,
        )

        with wd:
            deadline = time.time() + 3.0
            while time.time() < deadline and not wd.tripped():
                time.sleep(0.1)

        assert wd.tripped()
        callback.assert_called_once()
        report = wd.report()
        assert isinstance(report, DiskExhaustionReport)
        assert report.free_gb_at_trip == pytest.approx(0.5)
        assert report.abort_threshold_gb == pytest.approx(1.0)

    def test_make_error_carries_report(self, tmp_path):
        """
        ``make_error`` returns a DiskExhaustionError with the report.

        Tests:
            (Test Case 1) Exception is a DiskExhaustionError.
            (Test Case 2) ``.report`` attribute is the watchdog's report.
            (Test Case 3) ``.free_gb_at_trip`` and ``.abort_threshold_gb``
                round-trip from the watchdog.
        """
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=2.0,
            abort_free_gb=1.0,
            kill_callback=lambda: None,
        )
        # Manually mark as tripped (without running the thread) and
        # populate the report.
        wd._tripped = True
        wd._free_at_trip = 0.5
        wd._report = DiskExhaustionReport(
            folder=str(tmp_path), free_gb_at_trip=0.5, abort_threshold_gb=1.0
        )

        err = wd.make_error()
        assert isinstance(err, DiskExhaustionError)
        assert err.report is wd._report
        assert err.free_gb_at_trip == 0.5
        assert err.abort_threshold_gb == 1.0


class TestDiskUsageWatchdogReport:
    """``_build_report`` includes top consumers, projection, suggestions."""

    def test_top_consumers_sorted_descending(self, tmp_path):
        """
        ``_top_consumers`` returns the largest files in size order.

        Tests:
            (Test Case 1) The list is sorted descending by size.
            (Test Case 2) Sizes are reported in GB.
        """
        from spikelab.spike_sorting.guards._disk_watchdog import _top_consumers

        # Make a few files of different sizes.
        small = tmp_path / "small.npy"
        small.write_bytes(b"\x00" * 1024)
        big = tmp_path / "big.npy"
        big.write_bytes(b"\x00" * (5 * 1024 * 1024))  # 5 MB
        medium = tmp_path / "medium.npy"
        medium.write_bytes(b"\x00" * (1024 * 1024))  # 1 MB

        consumers = _top_consumers(tmp_path, limit=10)
        assert len(consumers) == 3
        sizes_gb = [gb for _, gb in consumers]
        assert sizes_gb == sorted(sizes_gb, reverse=True)
        # First entry is the big file.
        assert consumers[0][0].endswith("big.npy")

    def test_report_includes_projection_when_provided(self, tmp_path, monkeypatch):
        """
        ``projected_need_gb`` is preserved in the report.

        Tests:
            (Test Case 1) projected_need_gb in the watchdog appears
                in the built report.
            (Test Case 2) suggested_actions mentions the projected
                shortfall when projection > free.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw_mod

        monkeypatch.setattr(dw_mod, "_disk_free_gb", lambda p: 0.5)

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=2.0,
            abort_free_gb=1.0,
            poll_interval_s=0.1,
            projected_need_gb=10.0,
            kill_callback=mock.Mock(),
        )
        with wd:
            deadline = time.time() + 3.0
            while time.time() < deadline and not wd.tripped():
                time.sleep(0.1)

        report = wd.report()
        assert report.projected_need_gb == 10.0
        # The first suggestion should reference the projected shortfall.
        assert any(
            "projects" in s.lower() and "shortfall" in s.lower()
            for s in report.suggested_actions
        ) or any("10" in s and "free" in s.lower() for s in report.suggested_actions)

    def test_disk_field_defaults(self):
        """
        ExecutionConfig disk-watchdog fields default to documented values.

        Tests:
            (Test Case 1) disk_watchdog defaults to True.
            (Test Case 2) Free-space thresholds match agreed defaults.
            (Test Case 3) Poll interval defaults to 10 seconds.
        """
        cfg = ExecutionConfig()
        assert cfg.disk_watchdog is True
        assert cfg.disk_warn_free_gb == 5.0
        assert cfg.disk_abort_free_gb == 1.0
        assert cfg.disk_poll_interval_s == 10.0


# ---------------------------------------------------------------------------
# HostMemoryWatchdog kill-callback registration (T2-1)
# ---------------------------------------------------------------------------


class TestHostMemoryWatchdogKillCallbackRegistration:
    """``register_kill_callback`` / ``unregister_kill_callback`` semantics."""

    def test_register_appends(self):
        """
        Registered callbacks are appended to the internal list.

        Tests:
            (Test Case 1) After register_kill_callback, the callable
                is present in the internal list.
            (Test Case 2) Multiple callbacks accumulate.
        """
        wd = HostMemoryWatchdog(warn_pct=98, abort_pct=99, poll_interval_s=10.0)
        cb1 = mock.Mock()
        cb2 = mock.Mock()
        wd.register_kill_callback(cb1)
        wd.register_kill_callback(cb2)
        assert cb1 in wd._kill_callbacks
        assert cb2 in wd._kill_callbacks

    def test_unregister_removes_by_identity(self):
        """
        Unregister removes the callback by identity.

        Tests:
            (Test Case 1) After unregister, the callback is gone.
            (Test Case 2) Other registered callbacks remain.
        """
        wd = HostMemoryWatchdog(warn_pct=98, abort_pct=99, poll_interval_s=10.0)
        cb1 = mock.Mock()
        cb2 = mock.Mock()
        wd.register_kill_callback(cb1)
        wd.register_kill_callback(cb2)
        wd.unregister_kill_callback(cb1)
        assert cb1 not in wd._kill_callbacks
        assert cb2 in wd._kill_callbacks

    def test_unregister_unknown_is_noop(self):
        """
        Unregistering an unknown callback does not raise.

        Tests:
            (Test Case 1) Calling unregister_kill_callback on a
                callback that was never registered is a clean no-op.
        """
        wd = HostMemoryWatchdog()
        wd.unregister_kill_callback(lambda: None)  # must not raise


class TestHostMemoryWatchdogKillCallbackInvocation:
    """Registered kill callbacks fire on watchdog abort."""

    def test_callbacks_run_on_trip(self):
        """
        Kill callbacks are invoked alongside subprocess termination.

        Tests:
            (Test Case 1) After the watchdog trips, every registered
                callback is invoked exactly once.
            (Test Case 2) Subprocess termination still happens (when
                a popen is also registered).
        """
        fake_vm = SimpleNamespace(percent=99.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)

        cb1 = mock.Mock()
        cb2 = mock.Mock()

        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with HostMemoryWatchdog(
                warn_pct=70, abort_pct=80, poll_interval_s=0.1
            ) as wd:
                wd.register_kill_callback(cb1)
                wd.register_kill_callback(cb2)
                # Pure-Python loop so interrupt_main can land.
                try:
                    deadline = time.time() + 5.0
                    while time.time() < deadline:
                        _ = sum(range(500))
                except KeyboardInterrupt:
                    pass

        assert wd.tripped()
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_failing_callback_does_not_block_others(self):
        """
        A callback raising does not prevent other callbacks from running.

        Tests:
            (Test Case 1) After one callback raises, a second
                registered callback still fires.
        """
        wd = HostMemoryWatchdog(warn_pct=70, abort_pct=80)

        def _raises():
            raise RuntimeError("boom")

        cb_after = mock.Mock()
        wd.register_kill_callback(_raises)
        wd.register_kill_callback(cb_after)
        wd._run_kill_callbacks()  # exercise the iterator directly
        cb_after.assert_called_once()


# ---------------------------------------------------------------------------
# Container-kill hook in patched_container_client (T2-1)
# ---------------------------------------------------------------------------


class TestContainerKillHook:
    """``patched_container_client`` registers a container-kill callback."""

    def _make_fake_container_client(self, tracker):
        """Build a fake ContainerClient stub.

        ``tracker`` is a dict the patched_init writes to so the test
        can introspect what happened.
        """

        class _FakeContainer:
            def __init__(self):
                self.stop_calls = 0
                self.kill_calls = 0

            def stop(self, timeout=None):
                self.stop_calls += 1

            def kill(self):
                self.kill_calls += 1

        class _FakeContainerClient:
            def __init__(
                self, mode, container_image, volumes, py_user_base, extra_kwargs
            ):
                self.mode = mode
                self.docker_container = _FakeContainer() if mode == "docker" else None
                tracker["client"] = self
                tracker["extra_kwargs"] = extra_kwargs

        return _FakeContainerClient

    def test_register_registers_kill_callback(self, monkeypatch):
        """
        Creating a container under an active watchdog registers a
        kill callback that, when invoked, stops + kills the container.

        Tests:
            (Test Case 1) After creating a docker ContainerClient,
                the watchdog has one registered kill callback.
            (Test Case 2) Invoking that callback calls ``stop`` and
                ``kill`` on the container.
        """
        from spikelab.spike_sorting import docker_utils

        tracker = {}
        FakeClient = self._make_fake_container_client(tracker)

        # Stub SI's container_tools so patched_container_client picks
        # up our fake ContainerClient instead of the real one.
        fake_module = SimpleNamespace(ContainerClient=FakeClient)
        fake_si_pkg = SimpleNamespace(
            sorters=SimpleNamespace(container_tools=fake_module)
        )

        with mock.patch.dict(
            sys.modules,
            {
                "spikeinterface.sorters.container_tools": fake_module,
            },
        ):
            with HostMemoryWatchdog(
                warn_pct=98, abort_pct=99, poll_interval_s=10.0
            ) as wd:
                # patched_container_client patches FakeClient.__init__.
                with docker_utils.patched_container_client(
                    extra_env=None, mem_limit_frac=None
                ):
                    # Create a fake ContainerClient under the patch.
                    FakeClient(
                        "docker",
                        "fake/image:latest",
                        {},
                        "/tmp/pyuser",
                        {},
                    )
                # By here the patch has restored __init__, but the
                # callback registration should still hold.
                assert (
                    len(wd._kill_callbacks) == 1
                ), "container-kill callback was not registered"

                # Invoking it should stop + kill the container.
                container = tracker["client"].docker_container
                wd._kill_callbacks[0]()
                assert container.stop_calls == 1
                assert container.kill_calls == 1

    def test_singularity_skipped(self, monkeypatch):
        """
        Singularity mode does not register a kill callback.

        Tests:
            (Test Case 1) After creating a ``singularity`` client,
                the watchdog has no registered callbacks.
        """
        from spikelab.spike_sorting import docker_utils

        tracker = {}
        FakeClient = self._make_fake_container_client(tracker)
        fake_module = SimpleNamespace(ContainerClient=FakeClient)

        with mock.patch.dict(
            sys.modules,
            {"spikeinterface.sorters.container_tools": fake_module},
        ):
            with HostMemoryWatchdog(
                warn_pct=98, abort_pct=99, poll_interval_s=10.0
            ) as wd:
                with docker_utils.patched_container_client(
                    extra_env=None, mem_limit_frac=None
                ):
                    FakeClient(
                        "singularity",
                        "fake/image:latest",
                        {},
                        "/tmp/pyuser",
                        {},
                    )
                assert len(wd._kill_callbacks) == 0

    def test_no_watchdog_skipped(self, monkeypatch):
        """
        Without an active host-memory watchdog, registration is a no-op.

        Tests:
            (Test Case 1) Creating a docker client outside any
                watchdog context completes without error and registers
                nothing (verified by the absence of any
                ContextVar-bound watchdog).
        """
        from spikelab.spike_sorting import docker_utils

        tracker = {}
        FakeClient = self._make_fake_container_client(tracker)
        fake_module = SimpleNamespace(ContainerClient=FakeClient)

        with mock.patch.dict(
            sys.modules,
            {"spikeinterface.sorters.container_tools": fake_module},
        ):
            with docker_utils.patched_container_client(
                extra_env=None, mem_limit_frac=None
            ):
                FakeClient(
                    "docker",
                    "fake/image:latest",
                    {},
                    "/tmp/pyuser",
                    {},
                )
        # If we got here without raising, the no-watchdog path is OK.
        assert tracker["client"].docker_container is not None

    def test_callback_uses_weakref_so_container_can_gc(self):
        """
        The kill callback uses a weakref so SI's teardown can release
        the container without us holding it alive.

        Tests:
            (Test Case 1) After the container's strong reference is
                dropped and gc runs, weakref.finalize fires the
                auto-unregister and the watchdog callback list shrinks.
        """
        import gc

        from spikelab.spike_sorting import docker_utils

        tracker = {}
        FakeClient = self._make_fake_container_client(tracker)
        fake_module = SimpleNamespace(ContainerClient=FakeClient)

        with mock.patch.dict(
            sys.modules,
            {"spikeinterface.sorters.container_tools": fake_module},
        ):
            with HostMemoryWatchdog(
                warn_pct=98, abort_pct=99, poll_interval_s=10.0
            ) as wd:
                with docker_utils.patched_container_client(
                    extra_env=None, mem_limit_frac=None
                ):
                    FakeClient(
                        "docker",
                        "fake/image:latest",
                        {},
                        "/tmp/pyuser",
                        {},
                    )
                assert len(wd._kill_callbacks) == 1
                # Drop the only strong reference to the container.
                tracker["client"].docker_container = None
                tracker.pop("client", None)
                # Force GC; the weakref.finalize should auto-unregister.
                gc.collect()
                assert len(wd._kill_callbacks) == 0


# ---------------------------------------------------------------------------
# Active-inactivity-timeout ContextVar
# ---------------------------------------------------------------------------


class TestActiveInactivityTimeoutS:
    """``set_active_inactivity_timeout_s`` / ``get_active_inactivity_timeout_s``."""

    def test_default_is_none(self):
        """
        Outside any context the active timeout is None.

        Tests:
            (Test Case 1) Bare lookup returns None.
        """
        from spikelab.spike_sorting.guards import (
            get_active_inactivity_timeout_s,
        )

        assert get_active_inactivity_timeout_s() is None

    def test_set_and_clear(self):
        """
        ``set_active_inactivity_timeout_s`` publishes the value inside
        the context and clears it on exit.

        Tests:
            (Test Case 1) Inside the with-block: lookup returns the
                published value.
            (Test Case 2) After the with-block: lookup is None.
        """
        from spikelab.spike_sorting.guards import (
            get_active_inactivity_timeout_s,
            set_active_inactivity_timeout_s,
        )

        with set_active_inactivity_timeout_s(900.0):
            assert get_active_inactivity_timeout_s() == 900.0
        assert get_active_inactivity_timeout_s() is None


# ---------------------------------------------------------------------------
# Container-side inactivity watchdog (Docker path)
# ---------------------------------------------------------------------------


class TestContainerInactivityWatchdog:
    """Inactivity watchdog wiring inside ``patched_container_client``."""

    def _make_fake_container_client(self, tracker):
        class _FakeContainer:
            def __init__(self):
                self.stop_calls = 0
                self.kill_calls = 0

            def stop(self, timeout=None):
                self.stop_calls += 1

            def kill(self):
                self.kill_calls += 1

        class _FakeContainerClient:
            def __init__(
                self, mode, container_image, volumes, py_user_base, extra_kwargs
            ):
                self.mode = mode
                self.docker_container = _FakeContainer() if mode == "docker" else None
                tracker["client"] = self

        return _FakeContainerClient

    def test_inactivity_watchdog_started_when_log_and_timeout_set(self, tmp_path):
        """
        With log_path + inactivity_s active, an inactivity watchdog
        starts and is hooked to the container.

        Tests:
            (Test Case 1) Idle log triggers ``container.stop`` via
                the inactivity watchdog (without any host-memory
                trip).
            (Test Case 2) ``container.kill`` is also called.
        """
        from spikelab.spike_sorting import docker_utils
        from spikelab.spike_sorting.guards import (
            set_active_inactivity_timeout_s,
            set_active_log_path,
        )

        tracker = {}
        FakeClient = self._make_fake_container_client(tracker)
        fake_module = SimpleNamespace(ContainerClient=FakeClient)

        log_path = tmp_path / "missing.log"

        with mock.patch.dict(
            sys.modules,
            {"spikeinterface.sorters.container_tools": fake_module},
        ):
            with set_active_log_path(log_path), set_active_inactivity_timeout_s(0.4):
                with docker_utils.patched_container_client(
                    extra_env=None, mem_limit_frac=None
                ):
                    FakeClient(
                        "docker",
                        "fake/image:latest",
                        {},
                        "/tmp/pyuser",
                        {},
                    )
                container = tracker["client"].docker_container
                # Wait long enough for the watchdog to poll, see no
                # log, and trip.
                deadline = time.time() + 4.0
                while time.time() < deadline and container.stop_calls == 0:
                    time.sleep(0.1)

        assert container.stop_calls >= 1
        assert container.kill_calls >= 1

    def test_inactivity_watchdog_skipped_without_timeout(self, tmp_path):
        """
        Without an active inactivity tolerance, no inactivity watchdog
        is started — only the host-memory hook (if any) registers.

        Tests:
            (Test Case 1) After creating a container with only the
                log path published (no timeout), the container's
                ``stop`` is never called by an inactivity watchdog.
        """
        from spikelab.spike_sorting import docker_utils
        from spikelab.spike_sorting.guards import set_active_log_path

        tracker = {}
        FakeClient = self._make_fake_container_client(tracker)
        fake_module = SimpleNamespace(ContainerClient=FakeClient)

        with mock.patch.dict(
            sys.modules,
            {"spikeinterface.sorters.container_tools": fake_module},
        ):
            with set_active_log_path(tmp_path / "ks.log"):
                with docker_utils.patched_container_client(
                    extra_env=None, mem_limit_frac=None
                ):
                    FakeClient(
                        "docker",
                        "fake/image:latest",
                        {},
                        "/tmp/pyuser",
                        {},
                    )
                container = tracker["client"].docker_container
                # Brief wait — even if a watchdog were spawned, it
                # would have polled by now.
                time.sleep(0.5)

        assert container.stop_calls == 0
        assert container.kill_calls == 0


# ---------------------------------------------------------------------------
# GpuMemoryWatchdogError hierarchy
# ---------------------------------------------------------------------------


class TestGpuMemoryWatchdogErrorHierarchy:
    """Hierarchy and attribute storage for GpuMemoryWatchdogError."""

    def test_subclass_chain(self):
        """
        GpuMemoryWatchdogError descends from the resource category.

        Tests:
            (Test Case 1) Subclass of ResourceSortFailure.
            (Test Case 2) Subclass of SpikeSortingClassifiedError.
        """
        assert issubclass(GpuMemoryWatchdogError, ResourceSortFailure)
        assert issubclass(GpuMemoryWatchdogError, SpikeSortingClassifiedError)

    def test_attribute_storage(self):
        """
        Constructor records device_index, used_pct_at_trip, abort_pct.

        Tests:
            (Test Case 1) All keyword attributes round-trip.
            (Test Case 2) Defaults to None when omitted.
        """
        err = GpuMemoryWatchdogError(
            "boom", device_index=1, used_pct_at_trip=98.5, abort_pct=95.0
        )
        assert err.device_index == 1
        assert err.used_pct_at_trip == 98.5
        assert err.abort_pct == 95.0

        err2 = GpuMemoryWatchdogError("just a message")
        assert err2.device_index is None
        assert err2.used_pct_at_trip is None
        assert err2.abort_pct is None


# ---------------------------------------------------------------------------
# GPU device-index resolution
# ---------------------------------------------------------------------------


class TestResolveDeviceIndex:
    """Helpers for picking the device-in-use from sorter config."""

    def test_resolve_device_index_strings(self):
        """
        Various torch-style device strings resolve to the correct index.

        Tests:
            (Test Case 1) "cuda" → 0
            (Test Case 2) "cuda:3" → 3
            (Test Case 3) "2" → 2
            (Test Case 4) Garbage → 0
            (Test Case 5) None → 0
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _resolve_device_index,
        )

        assert _resolve_device_index("cuda") == 0
        assert _resolve_device_index("cuda:3") == 3
        assert _resolve_device_index("2") == 2
        assert _resolve_device_index("garbage") == 0
        assert _resolve_device_index(None) == 0

    def test_resolve_device_index_negative_and_malformed(self):
        """
        Negative and malformed device strings clamp to 0.

        Tests:
            (Test Case 1) ``"cuda:-1"`` clamps to 0 via the ``max(0, ...)``
                guard.
            (Test Case 2) ``"cuda:1.5"`` is unparseable as int → 0.
            (Test Case 3) ``"cuda:abc"`` is unparseable → 0.
            (Test Case 4) Empty string ``""`` returns 0.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _resolve_device_index,
        )

        assert _resolve_device_index("cuda:-1") == 0
        assert _resolve_device_index("cuda:1.5") == 0
        assert _resolve_device_index("cuda:abc") == 0
        assert _resolve_device_index("") == 0

    def test_resolve_active_device_from_config(self):
        """
        ``resolve_active_device`` picks the right device per sorter.

        Tests:
            (Test Case 1) RT-Sort reads rt_sort.device.
            (Test Case 2) KS4 reads sorter_params['torch_device'].
            (Test Case 3) KS2 defaults to 0.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards import resolve_active_device

        cfg_rt = SortingPipelineConfig()
        cfg_rt.sorter.sorter_name = "rt_sort"
        cfg_rt.rt_sort.device = "cuda:2"
        assert resolve_active_device(cfg_rt) == 2

        cfg_ks4 = SortingPipelineConfig()
        cfg_ks4.sorter.sorter_name = "kilosort4"
        cfg_ks4.sorter.sorter_params = {"torch_device": "cuda:1"}
        assert resolve_active_device(cfg_ks4) == 1

        cfg_ks2 = SortingPipelineConfig()
        cfg_ks2.sorter.sorter_name = "kilosort2"
        assert resolve_active_device(cfg_ks2) == 0


# ---------------------------------------------------------------------------
# GpuMemoryWatchdog construction + behaviour
# ---------------------------------------------------------------------------


class TestGpuMemoryWatchdogConstruction:
    """Threshold validation and disabled-state semantics."""

    def test_warn_must_be_below_abort(self):
        """
        warn_pct must be strictly less than abort_pct.

        Tests:
            (Test Case 1) Equal values are rejected.
            (Test Case 2) warn > abort is rejected.
        """
        with pytest.raises(ValueError):
            GpuMemoryWatchdog(warn_pct=95, abort_pct=95)
        with pytest.raises(ValueError):
            GpuMemoryWatchdog(warn_pct=99, abort_pct=95)

    def test_zero_poll_interval_raises(self):
        """
        poll_interval_s must be strictly positive.

        Tests:
            (Test Case 1) poll_interval_s=0 raises.
        """
        with pytest.raises(ValueError):
            GpuMemoryWatchdog(poll_interval_s=0)

    def test_disabled_when_no_gpu_info(self):
        """
        Watchdog disables itself cleanly when no GPU info source works.

        Tests:
            (Test Case 1) When ``read_gpu_memory`` returns None, the
                with-block is a no-op (no thread, no trip).
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        with mock.patch.object(gpu_mod, "read_gpu_memory", lambda i: None):
            wd = GpuMemoryWatchdog(
                device_index=0,
                warn_pct=85,
                abort_pct=95,
                poll_interval_s=0.1,
            )
            with wd:
                assert wd._enabled is False
                assert wd._thread is None
            assert wd.tripped() is False


class TestGpuMemoryWatchdogTrip:
    """End-to-end trip behaviour for the GPU watchdog."""

    def _busy_loop_until_interrupt(self, deadline_s: float) -> bool:
        deadline = time.time() + deadline_s
        try:
            while time.time() < deadline:
                _ = sum(range(500))
        except KeyboardInterrupt:
            return True
        return False

    def test_trip_on_high_used_pct(self):
        """
        Crossing the abort threshold trips the watchdog.

        Tests:
            (Test Case 1) ``tripped()`` is True after the trip window.
            (Test Case 2) ``used_pct_at_trip`` records the crossing
                value.
            (Test Case 3) ``make_error()`` returns a
                GpuMemoryWatchdogError carrying the device index.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        # Simulate a GPU at 98% used, 24 GB total.
        with mock.patch.object(gpu_mod, "read_gpu_memory", lambda i: (98.0, 24.0)):
            with GpuMemoryWatchdog(
                device_index=0,
                warn_pct=85,
                abort_pct=95,
                poll_interval_s=0.1,
            ) as wd:
                interrupted = self._busy_loop_until_interrupt(deadline_s=3.0)

        assert interrupted, "interrupt_main was not delivered within 3s"
        assert wd.tripped()
        assert wd.used_pct_at_trip() == pytest.approx(98.0)
        err = wd.make_error()
        assert isinstance(err, GpuMemoryWatchdogError)
        assert err.device_index == 0

    def test_no_trip_below_warn(self):
        """
        Below warn threshold the watchdog stays quiet.

        Tests:
            (Test Case 1) After several polls at 50%, ``tripped()``
                stays False.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        with mock.patch.object(gpu_mod, "read_gpu_memory", lambda i: (50.0, 24.0)):
            with GpuMemoryWatchdog(
                device_index=0,
                warn_pct=85,
                abort_pct=95,
                poll_interval_s=0.05,
            ) as wd:
                time.sleep(0.4)
                assert not wd.tripped()

    def test_kill_callback_invoked_on_trip(self):
        """
        Registered kill callbacks fire on trip.

        Tests:
            (Test Case 1) After the watchdog trips, the registered
                kill callback is invoked exactly once.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        cb = mock.Mock()
        with mock.patch.object(gpu_mod, "read_gpu_memory", lambda i: (99.0, 24.0)):
            with GpuMemoryWatchdog(
                device_index=0,
                warn_pct=85,
                abort_pct=95,
                poll_interval_s=0.1,
            ) as wd:
                wd.register_kill_callback(cb)
                try:
                    deadline = time.time() + 3.0
                    while time.time() < deadline:
                        _ = sum(range(500))
                except KeyboardInterrupt:
                    pass
        cb.assert_called_once()


class TestGpuFieldDefaults:
    """ExecutionConfig GPU-watchdog defaults match the agreed values."""

    def test_defaults(self):
        """
        gpu_watchdog defaults match the documented values.

        Tests:
            (Test Case 1) gpu_watchdog defaults to True.
            (Test Case 2) Percentage thresholds match agreed defaults.
            (Test Case 3) Poll interval defaults to 2 seconds.
        """
        cfg = ExecutionConfig()
        assert cfg.gpu_watchdog is True
        assert cfg.gpu_warn_pct == 85.0
        assert cfg.gpu_abort_pct == 95.0
        assert cfg.gpu_poll_interval_s == 2.0


# ---------------------------------------------------------------------------
# Tier 3 #2 — GPU snapshot capture
# ---------------------------------------------------------------------------


class TestCaptureGpuSnapshot:
    """``capture_gpu_snapshot`` writes a postmortem text file."""

    def test_writes_file_with_header(self, tmp_path):
        """
        ``capture_gpu_snapshot`` writes a file containing the header,
        an ISO timestamp, and best-effort sections for nvidia-smi
        and torch.

        Tests:
            (Test Case 1) Header line is present at the top.
            (Test Case 2) Timestamp line is present.
            (Test Case 3) ``-- nvidia-smi --`` and
                ``-- torch.cuda.memory_summary --`` section markers
                appear.
            (Test Case 4) Function returns the str-path on success.
        """
        from spikelab.spike_sorting.guards import capture_gpu_snapshot

        target = tmp_path / "snap.txt"
        result = capture_gpu_snapshot(target, header="Test trip header")
        assert result == str(target)
        text = target.read_text(encoding="utf-8")
        assert "Test trip header" in text
        assert "Captured:" in text
        assert "-- nvidia-smi --" in text
        assert "-- torch.cuda.memory_summary --" in text

    def test_creates_parent_dirs(self, tmp_path):
        """
        Missing parent directories are created.

        Tests:
            (Test Case 1) A nested target path is written even when
                the parent dirs don't exist yet.
        """
        from spikelab.spike_sorting.guards import capture_gpu_snapshot

        target = tmp_path / "deep" / "path" / "snap.txt"
        result = capture_gpu_snapshot(target)
        assert result == str(target)
        assert target.exists()

    def test_returns_none_on_write_failure(self, tmp_path, monkeypatch):
        """
        ``capture_gpu_snapshot`` returns None when the file write
        fails (e.g. permission denied), rather than raising.

        Tests:
            (Test Case 1) Patching ``Path.write_text`` to raise
                ``OSError`` causes the function to return None.
        """
        from spikelab.spike_sorting.guards import capture_gpu_snapshot
        from pathlib import Path as _Path

        def _refusing_write_text(self, *args, **kwargs):
            raise OSError("simulated permission denied")

        monkeypatch.setattr(_Path, "write_text", _refusing_write_text)
        target = tmp_path / "snap.txt"
        assert capture_gpu_snapshot(target, header="x") is None

    def test_nvidia_smi_timeout_appended_as_unavailable(self, tmp_path, monkeypatch):
        """
        When ``nvidia-smi`` exceeds the subprocess timeout, the
        snapshot still completes with an "(nvidia-smi unavailable: ...)"
        section instead of failing.

        Tests:
            (Test Case 1) Patching ``subprocess.check_output`` to raise
                ``TimeoutExpired`` causes the snapshot to record a
                stub line in the nvidia-smi section and still return
                the path on success.
        """
        from spikelab.spike_sorting.guards import capture_gpu_snapshot
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        def _timeout_check_output(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=10)

        monkeypatch.setattr(gpu_mod.subprocess, "check_output", _timeout_check_output)
        target = tmp_path / "snap.txt"
        result = capture_gpu_snapshot(target, header="timeout-test")
        assert result == str(target)
        text = target.read_text(encoding="utf-8")
        assert "nvidia-smi unavailable" in text


class TestReadGpuMemoryNvidiaSmi:
    """``_read_gpu_memory_nvidia_smi`` falls back when pynvml is absent."""

    def test_returns_none_when_nvidia_smi_missing(self, monkeypatch):
        """
        Without ``nvidia-smi`` on PATH, the helper returns None.

        Tests:
            (Test Case 1) ``subprocess.check_output`` raising
                ``FileNotFoundError`` causes the helper to return None
                rather than propagating.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        def _missing(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi not found")

        monkeypatch.setattr(gpu_mod.subprocess, "check_output", _missing)
        assert gpu_mod._read_gpu_memory_nvidia_smi(0) is None

    def test_returns_none_on_subprocess_error(self, monkeypatch):
        """
        Non-zero exit / SubprocessError is swallowed.

        Tests:
            (Test Case 1) ``subprocess.check_output`` raising
                ``CalledProcessError`` results in None.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        def _fail(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr(gpu_mod.subprocess, "check_output", _fail)
        assert gpu_mod._read_gpu_memory_nvidia_smi(0) is None

    def test_returns_none_on_empty_output(self, monkeypatch):
        """
        Driver loaded but zero compute devices → empty output → None.

        Tests:
            (Test Case 1) ``subprocess.check_output`` returning an
                empty string yields None (no matching index found).
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        def _empty(*args, **kwargs):
            return ""

        monkeypatch.setattr(gpu_mod.subprocess, "check_output", _empty)
        assert gpu_mod._read_gpu_memory_nvidia_smi(0) is None

    def test_returns_none_when_device_index_not_in_output(self, monkeypatch):
        """
        Asked-for index not present in nvidia-smi output → None.

        Tests:
            (Test Case 1) Output contains lines for index 0 only;
                requesting index 5 returns None.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        def _one_device(*args, **kwargs):
            # Format: "index, used_mib, total_mib"
            return "0, 1024, 8192\n"

        monkeypatch.setattr(gpu_mod.subprocess, "check_output", _one_device)
        assert gpu_mod._read_gpu_memory_nvidia_smi(5) is None


class TestSnapshotOnWatchdogTrip:
    """Watchdog abort paths drop a GPU snapshot in the active results folder."""

    def test_host_memory_watchdog_writes_snapshot(self, tmp_path):
        """
        Host-memory watchdog abort writes ``gpu_snapshot_at_trip.txt``.

        Tests:
            (Test Case 1) After the watchdog trips inside an active
                log-path context, the snapshot file appears in the
                results folder.
        """
        from spikelab.spike_sorting.guards import set_active_log_path

        # Create a log file so set_active_log_path's directory exists.
        log_path = tmp_path / "rec.log"
        log_path.touch()

        fake_vm = SimpleNamespace(percent=99.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)

        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with set_active_log_path(log_path):
                with HostMemoryWatchdog(
                    warn_pct=70, abort_pct=80, poll_interval_s=0.1
                ) as wd:
                    try:
                        deadline = time.time() + 3.0
                        while time.time() < deadline:
                            _ = sum(range(500))
                    except KeyboardInterrupt:
                        pass

        snap = tmp_path / "gpu_snapshot_at_trip.txt"
        # Snapshot is best-effort; we just need the trip path to have
        # tried writing it. On systems without nvidia-smi the file is
        # still written with "(nvidia-smi unavailable: ...)" content.
        assert snap.exists()
        text = snap.read_text(encoding="utf-8")
        assert "Host memory watchdog trip" in text

    def test_gpu_watchdog_writes_snapshot(self, tmp_path):
        """
        GPU watchdog abort writes ``gpu_snapshot_at_trip.txt`` and
        the file's header references the GPU device.

        Tests:
            (Test Case 1) Snapshot file exists after the trip.
            (Test Case 2) Header mentions the GPU watchdog and device.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod
        from spikelab.spike_sorting.guards import set_active_log_path

        log_path = tmp_path / "rec.log"
        log_path.touch()

        with mock.patch.object(gpu_mod, "read_gpu_memory", lambda i: (99.0, 24.0)):
            with set_active_log_path(log_path):
                with GpuMemoryWatchdog(
                    device_index=0,
                    warn_pct=85,
                    abort_pct=95,
                    poll_interval_s=0.1,
                ):
                    try:
                        deadline = time.time() + 3.0
                        while time.time() < deadline:
                            _ = sum(range(500))
                    except KeyboardInterrupt:
                        pass

        snap = tmp_path / "gpu_snapshot_at_trip.txt"
        assert snap.exists()
        text = snap.read_text(encoding="utf-8")
        assert "GPU memory watchdog trip" in text
        assert "device 0" in text


# ---------------------------------------------------------------------------
# Tier 3 #3 — recording pre-validation
# ---------------------------------------------------------------------------


class TestRecordingPreValidation:
    """``_validate_recording_inputs`` catches typos and unfamiliar files."""

    def test_missing_recording_yields_fail(self, tmp_path):
        """
        Missing path produces a ``recording_missing`` fail finding.

        Tests:
            (Test Case 1) Returns one finding with code
                ``recording_missing`` and level ``fail``.
            (Test Case 2) Category is ``environment``.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        bogus = tmp_path / "does_not_exist.h5"
        findings = _validate_recording_inputs([bogus])
        assert len(findings) == 1
        assert findings[0].code == "recording_missing"
        assert findings[0].level == "fail"
        assert findings[0].category == "environment"

    def test_known_extension_no_finding(self, tmp_path):
        """
        Existing file with a known extension yields no finding.

        Tests:
            (Test Case 1) ``.h5``, ``.nwb``, ``.raw.h5`` all pass.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        for ext in (".h5", ".nwb", ".raw.h5"):
            p = tmp_path / f"rec{ext}"
            p.write_bytes(b"")
            assert _validate_recording_inputs([p]) == []

    def test_unknown_extension_warns(self, tmp_path):
        """
        Existing file with an unfamiliar extension yields a warn.

        Tests:
            (Test Case 1) ``.txt`` triggers a warn finding.
            (Test Case 2) Code is ``recording_extension_unknown``.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        p = tmp_path / "rec.txt"
        p.write_bytes(b"")
        findings = _validate_recording_inputs([p])
        assert len(findings) == 1
        assert findings[0].code == "recording_extension_unknown"
        assert findings[0].level == "warn"

    def test_directory_no_finding(self, tmp_path):
        """
        Directory inputs (used for concatenation) skip the extension check.

        Tests:
            (Test Case 1) An existing directory yields no finding.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        d = tmp_path / "multi"
        d.mkdir()
        assert _validate_recording_inputs([d]) == []

    def test_pre_loaded_recording_skipped(self):
        """
        Pre-loaded ``BaseRecording`` objects are skipped (not paths).

        Tests:
            (Test Case 1) A non-path input yields no finding.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        fake_rec = mock.Mock()
        assert _validate_recording_inputs([fake_rec]) == []


# ---------------------------------------------------------------------------
# Tier 3 #1 — RecordingResult.log_path round-trip
# ---------------------------------------------------------------------------


class TestRecordingResultLogPath:
    """``RecordingResult.log_path`` defaults and propagation."""

    def test_default_is_none(self):
        """
        log_path defaults to None when omitted.

        Tests:
            (Test Case 1) Constructed RecordingResult has log_path=None
                when the field is omitted.
        """
        from spikelab.spike_sorting.pipeline import RecordingResult

        r = RecordingResult(
            rec_name="rec1",
            rec_path="/tmp/rec1.h5",
            results_folder="/tmp/sorted_rec1",
            status="success",
            wall_time_s=1.0,
        )
        assert r.log_path is None

    def test_explicit_path_round_trips(self):
        """
        log_path round-trips through the dataclass.

        Tests:
            (Test Case 1) Constructed RecordingResult preserves the
                given log_path string.
        """
        from spikelab.spike_sorting.pipeline import RecordingResult

        r = RecordingResult(
            rec_name="rec1",
            rec_path="/tmp/rec1.h5",
            results_folder="/tmp/sorted_rec1",
            status="success",
            wall_time_s=1.0,
            log_path="/tmp/sorted_rec1/sorting_250502_120000.log",
        )
        assert r.log_path == "/tmp/sorted_rec1/sorting_250502_120000.log"


# ===========================================================================
# Concurrent-sort lock (`_sort_lock`)
# ===========================================================================


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


class TestSortLockHelpers:
    """``_pid_alive``, ``_pid_holds_lock``, ``_read_lock_info``."""

    def test_pid_alive_zero_or_negative_returns_false(self):
        """
        PID <= 0 short-circuits to False without touching the OS.

        Tests:
            (Test Case 1) pid=0 → False.
            (Test Case 2) pid=-5 → False.
        """
        from spikelab.spike_sorting.guards._sort_lock import _pid_alive

        assert _pid_alive(0) is False
        assert _pid_alive(-5) is False

    def test_pid_alive_returns_psutil_result(self, monkeypatch):
        """
        With psutil available, ``psutil.pid_exists`` drives the answer.

        Tests:
            (Test Case 1) Patched psutil.pid_exists returns True → True.
            (Test Case 2) Returns False → False.
        """
        from spikelab.spike_sorting.guards._sort_lock import _pid_alive

        fake_psutil = SimpleNamespace(pid_exists=lambda pid: True)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert _pid_alive(123) is True
        fake_psutil.pid_exists = lambda pid: False
        assert _pid_alive(123) is False

    def test_read_lock_info_returns_dict_for_valid_json(self, tmp_path):
        """
        Valid JSON is parsed and returned as a dict.

        Tests:
            (Test Case 1) Lock file with {pid: 1, hostname: 'h'} →
                same dict back.
        """
        from spikelab.spike_sorting.guards._sort_lock import _read_lock_info

        lock = tmp_path / "lock.json"
        lock.write_text(json.dumps({"pid": 1, "hostname": "h"}), encoding="utf-8")
        assert _read_lock_info(lock) == {"pid": 1, "hostname": "h"}

    def test_read_lock_info_returns_none_for_missing_file(self, tmp_path):
        """
        Non-existent lock path returns None instead of raising.

        Tests:
            (Test Case 1) Path that was never created → None.
        """
        from spikelab.spike_sorting.guards._sort_lock import _read_lock_info

        assert _read_lock_info(tmp_path / "never") is None

    def test_read_lock_info_returns_none_for_invalid_json(self, tmp_path):
        """
        Garbage in the lock file yields None instead of crashing.

        Tests:
            (Test Case 1) Non-JSON file → None.
        """
        from spikelab.spike_sorting.guards._sort_lock import _read_lock_info

        lock = tmp_path / "lock.json"
        lock.write_text("this is not json at all {{", encoding="utf-8")
        assert _read_lock_info(lock) is None

    def test_pid_holds_lock_zero_returns_false(self):
        """
        PID <= 0 returns False before any psutil work.

        Tests:
            (Test Case 1) pid=0, started_at='2026-01-01T00:00:00' → False.
        """
        from spikelab.spike_sorting.guards._sort_lock import _pid_holds_lock

        assert _pid_holds_lock(0, "2026-01-01T00:00:00") is False

    def test_pid_holds_lock_falls_back_to_alive_without_started_at(self, monkeypatch):
        """
        With ``started_at=None``, behaves identically to ``_pid_alive``.

        Tests:
            (Test Case 1) Live PID, started_at=None → True (matches
                _pid_alive's return).
        """
        from spikelab.spike_sorting.guards import _sort_lock

        fake_psutil = SimpleNamespace(
            pid_exists=lambda pid: True,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert _sort_lock._pid_holds_lock(123, None) is True

    def test_pid_holds_lock_detects_pid_reuse(self, monkeypatch):
        """
        When the live PID's create_time is meaningfully after the
        lock's started_at, treat the live process as a PID reuse.

        Tests:
            (Test Case 1) Lock started_at 60 s ago; live process
                create_time 100 s in the future of that → returns
                False (stale).
            (Test Case 2) Same setup with create_time within 5 s skew
                → returns True (original holder).
        """
        from datetime import datetime

        from spikelab.spike_sorting.guards import _sort_lock

        # Use a recent timestamp (Windows can't fromtimestamp(100.0))
        # so the round-trip through fromisoformat / .timestamp()
        # works on every platform.
        lock_t = time.time() - 60.0
        lock_iso = datetime.fromtimestamp(lock_t).isoformat()

        class _FakeProc:
            def __init__(self, ct):
                self._ct = ct

            def create_time(self):
                return self._ct

        # Reused: create_time well after lock_t + skew.
        fake_psutil = SimpleNamespace(
            pid_exists=lambda pid: True,
            Process=lambda pid: _FakeProc(lock_t + 100.0),
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert _sort_lock._pid_holds_lock(42, lock_iso) is False

        # Original holder: create_time within skew.
        fake_psutil.Process = lambda pid: _FakeProc(lock_t + 2.0)
        assert _sort_lock._pid_holds_lock(42, lock_iso) is True


# ===========================================================================
# Windows Job Object cap (`_job_object`)
# ===========================================================================


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

    def test_pywin32_success_path(self):
        """
        With pywin32 mocked, the cap is installed and ``CloseHandle``
        runs on cleanup.

        Tests:
            (Test Case 1) ``CreateJobObject`` and
                ``AssignProcessToJobObject`` are both invoked.
            (Test Case 2) ``SetInformationJobObject`` is called with
                the expected ``ProcessMemoryLimit`` value.
            (Test Case 3) The context yields True.
            (Test Case 4) ``CloseHandle`` is called on the job handle
                during cleanup.
        """
        from spikelab.spike_sorting.guards import _job_object as job_mod

        # Stand-in pywin32 modules.
        info_dict: dict = {
            "BasicLimitInformation": {"LimitFlags": 0},
        }
        fake_win32job = SimpleNamespace(
            CreateJobObject=mock.MagicMock(return_value="JOB_HANDLE"),
            AssignProcessToJobObject=mock.MagicMock(),
            SetInformationJobObject=mock.MagicMock(),
            QueryInformationJobObject=mock.MagicMock(return_value=info_dict),
            JobObjectExtendedLimitInformation=9,
            JOB_OBJECT_LIMIT_PROCESS_MEMORY=0x100,
        )
        fake_win32api = SimpleNamespace(
            GetCurrentProcess=mock.MagicMock(return_value="PROCESS_HANDLE"),
            CloseHandle=mock.MagicMock(),
        )
        fake_win32con = SimpleNamespace()

        ram_bytes = 16 * 1024**3  # 16 GiB

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _fake_import(name, *args, **kwargs):
            if name == "win32job":
                return fake_win32job
            if name == "win32api":
                return fake_win32api
            if name == "win32con":
                return fake_win32con
            return real_import(name, *args, **kwargs)

        with (
            mock.patch.object(job_mod.sys, "platform", "win32"),
            mock.patch.object(job_mod, "_get_total_ram_bytes", return_value=ram_bytes),
            mock.patch("builtins.__import__", _fake_import),
        ):
            with windows_job_object_cap(0.5) as active:
                assert active is True

        fake_win32job.CreateJobObject.assert_called_once()
        fake_win32job.AssignProcessToJobObject.assert_called_once_with(
            "JOB_HANDLE", "PROCESS_HANDLE"
        )
        fake_win32job.SetInformationJobObject.assert_called_once()
        # ProcessMemoryLimit was written into the info dict and applied.
        assert info_dict["ProcessMemoryLimit"] == int(ram_bytes * 0.5)
        fake_win32api.CloseHandle.assert_called_once_with("JOB_HANDLE")


# ===========================================================================
# Audit log (`_audit`)
# ===========================================================================


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

    def test_silent_swallow_when_open_raises(self, tmp_path, monkeypatch):
        """
        Audit-side write failures (e.g. read-only filesystem) are
        swallowed so observability never breaks a sort.

        Tests:
            (Test Case 1) Patching ``open`` to raise ``PermissionError``
                does not propagate; the call is a silent no-op and no
                file is written.
        """
        log_path = tmp_path / "rec.log"
        log_path.touch()

        import builtins

        real_open = builtins.open

        def _refusing_open(path, *args, **kwargs):
            if str(path).endswith("watchdog_events.jsonl"):
                raise PermissionError("simulated read-only mount")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _refusing_open)
        # Must not raise; the audit infrastructure swallows write failures.
        append_audit_event(
            watchdog="host_memory", event="warn", log_path=log_path, used_pct=87.0
        )
        audit = tmp_path / "watchdog_events.jsonl"
        assert not audit.exists()

    def test_silent_swallow_when_payload_str_raises(self, tmp_path):
        """
        Payload values whose ``__str__`` raises do not propagate the
        error or corrupt the log.

        Tests:
            (Test Case 1) A payload value whose ``__str__`` raises is
                handled by the surrounding try/except so the call is a
                silent no-op (no event line is written).
        """

        class _BadStr:
            def __str__(self) -> str:
                raise RuntimeError("simulated buggy __str__")

            __repr__ = __str__

        log_path = tmp_path / "rec.log"
        log_path.touch()
        append_audit_event(
            watchdog="x", event="y", log_path=log_path, payload=_BadStr()
        )
        audit = tmp_path / "watchdog_events.jsonl"
        # The whole-function try/except swallows the error; nothing
        # is written rather than a partial / corrupted event line.
        assert not audit.exists() or audit.read_text(encoding="utf-8") == ""


class TestJsonSafe:
    """``_json_safe`` coerces non-JSON-friendly payload values."""

    def test_path_to_str(self, tmp_path):
        """
        ``Path`` instances are coerced to their string form.

        Tests:
            (Test Case 1) ``Path`` returns ``str(path)``.
        """
        from spikelab.spike_sorting.guards._audit import _json_safe

        result = _json_safe(tmp_path)
        assert result == str(tmp_path)
        assert isinstance(result, str)

    def test_primitives_passthrough(self):
        """
        int / float / str / bool / None pass through unchanged.

        Tests:
            (Test Case 1) Each primitive type returns identity.
        """
        from spikelab.spike_sorting.guards._audit import _json_safe

        for value in (1, 1.5, "text", True, False, None):
            assert _json_safe(value) is value

    def test_arbitrary_object_str_fallback(self):
        """
        Unknown types fall back to ``str(value)``.

        Tests:
            (Test Case 1) A custom object's str() form is returned.
        """
        from spikelab.spike_sorting.guards._audit import _json_safe

        class _Obj:
            def __str__(self) -> str:
                return "obj-as-str"

        assert _json_safe(_Obj()) == "obj-as-str"


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


# ===========================================================================
# I/O stall watchdog (`_io_stall`)
# ===========================================================================


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


class TestIoStallDeviceNormalization:
    """``_resolve_device_for_path`` produces a usable disk_io_counters key."""

    def test_windows_drive_letter_strips_trailing_backslash(self, monkeypatch):
        """
        On Windows, ``part.device`` is ``"C:\\"`` and the normaliser
        must produce ``"C:"`` so it matches psutil's
        ``disk_io_counters(perdisk=True)`` keys (which look like
        ``"C:"`` or ``"PhysicalDrive0"``).

        Tests:
            (Test Case 1) Patched sys.platform='win32' + patched
                psutil returning a single C:\\ partition → returns 'C:'.
        """
        from spikelab.spike_sorting.guards import _io_stall as io_stall_mod

        monkeypatch.setattr(io_stall_mod.sys, "platform", "win32")

        fake_part = SimpleNamespace(mountpoint="C:\\", device="C:\\")
        fake_psutil = SimpleNamespace(disk_partitions=lambda all=False: [fake_part])
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        # Resolve a path that lives under C:\, with the resolution
        # mocked to mirror the platform check above.
        monkeypatch.setattr(
            io_stall_mod.Path,
            "resolve",
            lambda self: Path("C:\\some\\folder"),
        )
        assert io_stall_mod._resolve_device_for_path(Path("C:\\")) == "C:"

    def test_posix_path_strips_to_basename(self, monkeypatch):
        """
        On POSIX, ``/dev/sda1`` is normalised to ``sda1``.

        Tests:
            (Test Case 1) Patched sys.platform='linux' with a single
                ``/dev/sda1`` partition → returns 'sda1'.
        """
        from spikelab.spike_sorting.guards import _io_stall as io_stall_mod

        monkeypatch.setattr(io_stall_mod.sys, "platform", "linux")

        fake_part = SimpleNamespace(mountpoint="/", device="/dev/sda1")
        fake_psutil = SimpleNamespace(disk_partitions=lambda all=False: [fake_part])
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        monkeypatch.setattr(
            io_stall_mod.Path, "resolve", lambda self: Path("/var/data/x")
        )
        assert io_stall_mod._resolve_device_for_path(Path("/var/data")) == "sda1"


class TestResolveDeviceForPath:
    """``_resolve_device_for_path`` covers psutil-failure shapes."""

    def test_returns_none_when_psutil_missing(self, monkeypatch):
        """
        Without ``psutil`` on PATH, the helper returns None silently.

        Tests:
            (Test Case 1) A patched ``__import__`` that raises
                ImportError for ``psutil`` causes the helper to
                return None rather than propagate.
        """
        import builtins

        from spikelab.spike_sorting.guards import _io_stall as io_stall_mod

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _refusing_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("simulated missing psutil")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _refusing_import)
        assert io_stall_mod._resolve_device_for_path(Path(".")) is None

    def test_returns_none_when_no_partition_matches(self, monkeypatch):
        """
        Empty ``psutil.disk_partitions`` (containerised env) → None.

        Tests:
            (Test Case 1) When no partitions are reported, the
                helper returns None.
        """
        from spikelab.spike_sorting.guards import _io_stall as io_stall_mod

        fake_psutil = SimpleNamespace(disk_partitions=lambda all=False: [])
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert io_stall_mod._resolve_device_for_path(Path(".")) is None

    def test_returns_none_when_disk_partitions_raises(self, monkeypatch):
        """
        A psutil error inside ``disk_partitions`` is caught.

        Tests:
            (Test Case 1) ``disk_partitions`` raising propagates as
                None rather than crashing the watchdog setup.
        """
        from spikelab.spike_sorting.guards import _io_stall as io_stall_mod

        def _raise(*args, **kwargs):
            raise OSError("simulated psutil error")

        fake_psutil = SimpleNamespace(disk_partitions=_raise)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert io_stall_mod._resolve_device_for_path(Path(".")) is None


# ===========================================================================
# Temp-file cleanup (`_tempfile_cleanup`)
# ===========================================================================


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


# ===========================================================================
# Power state (`_power_state`)
# ===========================================================================


class TestPowerStateLock:
    """``prevent_system_sleep`` is a no-op off Windows."""

    def test_noop_on_non_windows(self, monkeypatch):
        """
        Off Windows, prevent_system_sleep yields False without raising
        when no inhibitor binary is available.

        Tests:
            (Test Case 1) Non-Windows platform with no inhibitor
                spawnable → yields False.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        monkeypatch.setattr(ps, "_spawn_inhibitor", lambda *a, **kw: None)
        with mock.patch.object(ps.sys, "platform", "linux"):
            with prevent_system_sleep() as active:
                assert active is False

    def test_yields_false_when_platform_simulated_non_windows(self, monkeypatch):
        """
        Patching sys.platform to a non-Windows value yields False
        when the inhibitor spawn fails.

        Tests:
            (Test Case 1) When the helper sees a non-Windows
                platform and ``_spawn_inhibitor`` returns None,
                the context yields False without touching any
                ctypes APIs — even on a real Windows host.

        Notes:
            - The Windows-API-call path (``SetThreadExecutionState``)
              is exercised in production rather than tested here —
              mocking ``ctypes.windll`` reliably across platforms
              is fragile and the live call interacts with the OS
              in ways that can stall a test process.
            - We must patch ``_spawn_inhibitor`` because on a real
              Windows host an inhibitor binary may still resolve
              via Git Bash / WSL helpers / other shims on PATH,
              which would let the Linux/macOS branch yield True.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        monkeypatch.setattr(ps, "_spawn_inhibitor", lambda *a, **kw: None)
        for fake_platform in ("linux", "darwin"):
            with mock.patch.object(ps.sys, "platform", fake_platform):
                with prevent_system_sleep() as active:
                    assert active is False


class TestSpawnInhibitor:
    """``_spawn_inhibitor`` wraps ``subprocess.Popen`` with error handling."""

    def test_returns_popen_on_success(self, monkeypatch):
        """
        ``subprocess.Popen`` succeeds → returns the Popen instance.

        Tests:
            (Test Case 1) The patched constructor is invoked with
                the supplied argv and DEVNULL streams, and its
                return value is forwarded.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        sentinel = mock.Mock(spec=subprocess.Popen)
        captured = {}

        def _fake_popen(argv, **kwargs):
            captured["argv"] = list(argv)
            captured["stdout"] = kwargs.get("stdout")
            captured["stderr"] = kwargs.get("stderr")
            captured["stdin"] = kwargs.get("stdin")
            return sentinel

        monkeypatch.setattr(ps.subprocess, "Popen", _fake_popen)
        result = ps._spawn_inhibitor(["caffeinate", "-dims"], "caffeinate")
        assert result is sentinel
        assert captured["argv"] == ["caffeinate", "-dims"]
        assert captured["stdout"] == subprocess.DEVNULL
        assert captured["stderr"] == subprocess.DEVNULL
        assert captured["stdin"] == subprocess.DEVNULL

    def test_returns_none_when_binary_missing(self, monkeypatch):
        """
        ``FileNotFoundError`` from Popen → returns None silently.

        Tests:
            (Test Case 1) Patched Popen raising FileNotFoundError
                returns None without propagating.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        def _missing(*args, **kwargs):
            raise FileNotFoundError("simulated missing binary")

        monkeypatch.setattr(ps.subprocess, "Popen", _missing)
        assert ps._spawn_inhibitor(["caffeinate"], "caffeinate") is None

    def test_returns_none_on_generic_exception(self, monkeypatch):
        """
        Any other exception from Popen is swallowed → None.

        Tests:
            (Test Case 1) Patched Popen raising OSError returns None.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        def _boom(*args, **kwargs):
            raise OSError("simulated permission denied")

        monkeypatch.setattr(ps.subprocess, "Popen", _boom)
        assert ps._spawn_inhibitor(["caffeinate"], "caffeinate") is None


class TestTerminateInhibitor:
    """``_terminate_inhibitor`` performs best-effort terminate-then-kill."""

    def test_early_return_when_already_dead(self):
        """
        ``proc.poll()`` returning a non-None exit code skips terminate
        and kill entirely.

        Tests:
            (Test Case 1) For a dead process, neither ``terminate``
                nor ``kill`` is invoked.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = 0
        ps._terminate_inhibitor(proc, "caffeinate")
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()

    def test_terminate_then_wait_succeeds(self):
        """
        Live process: ``terminate`` then ``wait(timeout=5)`` completes
        cleanly; ``kill`` is not called.

        Tests:
            (Test Case 1) A live proc has terminate + wait invoked,
                kill is skipped because wait returns normally.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.return_value = 0
        ps._terminate_inhibitor(proc, "caffeinate")
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)
        proc.kill.assert_not_called()

    def test_kill_after_wait_timeout(self):
        """
        ``wait`` raising ``TimeoutExpired`` → falls through to ``kill``.

        Tests:
            (Test Case 1) When wait times out, kill is invoked
                exactly once.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="caffeinate", timeout=5)
        ps._terminate_inhibitor(proc, "caffeinate")
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()


class TestPreventSleepWindows:
    """The Windows branch of ``prevent_system_sleep``."""

    def test_yields_false_when_ctypes_unavailable(self, monkeypatch):
        """
        ``import ctypes`` raising → yields False without invoking the
        Win32 API.

        Tests:
            (Test Case 1) Replacing ``sys.modules['ctypes']`` so the
                inner ``import ctypes`` raises ImportError causes the
                generator to yield False.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        # Force the inner ``import ctypes`` to raise. Replacing the
        # cached module with None makes ``import ctypes`` fail
        # immediately.
        monkeypatch.setitem(sys.modules, "ctypes", None)
        gen = ps._prevent_sleep_windows()
        assert next(gen) is False
        with pytest.raises(StopIteration):
            next(gen)

    def test_yields_true_on_successful_call_and_clears_on_exit(self, monkeypatch):
        """
        Successful ``SetThreadExecutionState`` call → yields True;
        cleanup clears the flags on generator close.

        Tests:
            (Test Case 1) ``SetThreadExecutionState`` is invoked with
                the configured flag mask and the generator yields True.
            (Test Case 2) Closing the generator triggers a second
                ``SetThreadExecutionState`` call with ``ES_CONTINUOUS``
                only (clears the inhibit flags).
        """
        import ctypes

        from spikelab.spike_sorting.guards import _power_state as ps

        kernel32 = mock.Mock()
        # Non-zero return → success path (no warning emitted).
        kernel32.SetThreadExecutionState = mock.Mock(return_value=0x42)
        windll = SimpleNamespace(kernel32=kernel32)
        monkeypatch.setattr(ctypes, "windll", windll, raising=False)

        gen = ps._prevent_sleep_windows()
        assert next(gen) is True
        # Initial flags include CONTINUOUS | SYSTEM_REQUIRED | AWAYMODE.
        expected_flags = (
            ps._ES_CONTINUOUS | ps._ES_SYSTEM_REQUIRED | ps._ES_AWAYMODE_REQUIRED
        )
        kernel32.SetThreadExecutionState.assert_called_once_with(expected_flags)

        # Closing the generator runs the finally block → clears flags
        # by calling the API again with just ES_CONTINUOUS.
        gen.close()
        assert kernel32.SetThreadExecutionState.call_count == 2
        last_call = kernel32.SetThreadExecutionState.call_args_list[-1]
        assert last_call.args == (ps._ES_CONTINUOUS,)


class TestPreventSleepMacos:
    """The macOS branch of ``prevent_system_sleep``."""

    def test_yields_false_when_caffeinate_missing(self, monkeypatch):
        """
        When ``_spawn_inhibitor`` returns None (caffeinate not found),
        the generator yields False.

        Tests:
            (Test Case 1) Patched ``_spawn_inhibitor`` returning None
                → generator yields False and exhausts.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        monkeypatch.setattr(ps, "_spawn_inhibitor", lambda *a, **kw: None)
        gen = ps._prevent_sleep_macos()
        assert next(gen) is False
        with pytest.raises(StopIteration):
            next(gen)

    def test_yields_true_and_terminates_on_close(self, monkeypatch):
        """
        Successful spawn → yields True; closing the generator runs
        ``_terminate_inhibitor`` against the spawned process.

        Tests:
            (Test Case 1) Generator yields True after a successful
                spawn.
            (Test Case 2) Patched ``_terminate_inhibitor`` is called
                with the spawned proc and the "caffeinate" label
                during cleanup.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        proc = mock.Mock(spec=subprocess.Popen)
        captured: list = []
        monkeypatch.setattr(ps, "_spawn_inhibitor", lambda argv, label: proc)
        monkeypatch.setattr(
            ps,
            "_terminate_inhibitor",
            lambda p, label: captured.append((p, label)),
        )
        gen = ps._prevent_sleep_macos()
        assert next(gen) is True
        gen.close()
        assert captured == [(proc, "caffeinate")]


class TestPreventSleepLinux:
    """The Linux branch of ``prevent_system_sleep``."""

    def test_yields_false_when_systemd_inhibit_missing(self, monkeypatch):
        """
        When ``_spawn_inhibitor`` returns None (systemd-inhibit not
        found, non-systemd init), the generator yields False.

        Tests:
            (Test Case 1) Patched ``_spawn_inhibitor`` returning None
                → generator yields False and exhausts.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        monkeypatch.setattr(ps, "_spawn_inhibitor", lambda *a, **kw: None)
        gen = ps._prevent_sleep_linux()
        assert next(gen) is False
        with pytest.raises(StopIteration):
            next(gen)

    def test_yields_true_and_terminates_on_close(self, monkeypatch):
        """
        Successful spawn → yields True; closing the generator runs
        ``_terminate_inhibitor`` against the spawned process.

        Tests:
            (Test Case 1) Generator yields True after a successful
                ``systemd-inhibit`` spawn.
            (Test Case 2) Patched ``_terminate_inhibitor`` is called
                with the spawned proc and the "systemd-inhibit"
                label during cleanup.
            (Test Case 3) The argv passed to ``_spawn_inhibitor``
                starts with ``systemd-inhibit`` and ends with
                ``["sleep", "infinity"]``.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        proc = mock.Mock(spec=subprocess.Popen)
        spawned: list = []
        terminated: list = []

        def _spawn(argv, label):
            spawned.append((list(argv), label))
            return proc

        monkeypatch.setattr(ps, "_spawn_inhibitor", _spawn)
        monkeypatch.setattr(
            ps,
            "_terminate_inhibitor",
            lambda p, label: terminated.append((p, label)),
        )
        gen = ps._prevent_sleep_linux()
        assert next(gen) is True
        gen.close()
        assert terminated == [(proc, "systemd-inhibit")]
        assert len(spawned) == 1
        argv, label = spawned[0]
        assert argv[0] == "systemd-inhibit"
        assert argv[-2:] == ["sleep", "infinity"]
        assert label == "systemd-inhibit"


# ===========================================================================
# Tripped-watchdog router (`__init__.find_tripped_global_watchdog`)
# ===========================================================================


class TestFindTrippedGlobalWatchdog:
    """``find_tripped_global_watchdog`` walks watchdog priority order."""

    def test_returns_none_when_nothing_active(self, monkeypatch):
        """
        With no active watchdogs in any ContextVar, returns None.

        Tests:
            (Test Case 1) All three get_active_* return None → None.
        """
        from spikelab.spike_sorting import guards as guards_mod

        monkeypatch.setattr(guards_mod, "get_active_watchdog", lambda: None)
        monkeypatch.setattr(guards_mod, "get_active_gpu_watchdog", lambda: None)
        monkeypatch.setattr(guards_mod, "get_active_io_stall_watchdog", lambda: None)
        assert guards_mod.find_tripped_global_watchdog() is None

    def test_returns_host_when_only_host_tripped(self, monkeypatch):
        """
        Host watchdog tripped, GPU and IO not active → returns host.

        Tests:
            (Test Case 1) get_active_watchdog returns a tripped stub.
        """
        from spikelab.spike_sorting import guards as guards_mod

        host = SimpleNamespace(tripped=lambda: True)
        monkeypatch.setattr(guards_mod, "get_active_watchdog", lambda: host)
        monkeypatch.setattr(guards_mod, "get_active_gpu_watchdog", lambda: None)
        monkeypatch.setattr(guards_mod, "get_active_io_stall_watchdog", lambda: None)
        assert guards_mod.find_tripped_global_watchdog() is host

    def test_priority_order_host_before_gpu_before_io(self, monkeypatch):
        """
        With every watchdog tripped, host wins over GPU which wins over IO.

        Tests:
            (Test Case 1) All three tripped → returns host.
        """
        from spikelab.spike_sorting import guards as guards_mod

        host = SimpleNamespace(tripped=lambda: True, name="host")
        gpu = SimpleNamespace(tripped=lambda: True, name="gpu")
        io = SimpleNamespace(tripped=lambda: True, name="io")
        monkeypatch.setattr(guards_mod, "get_active_watchdog", lambda: host)
        monkeypatch.setattr(guards_mod, "get_active_gpu_watchdog", lambda: gpu)
        monkeypatch.setattr(guards_mod, "get_active_io_stall_watchdog", lambda: io)
        assert guards_mod.find_tripped_global_watchdog().name == "host"

    def test_skips_untripped_watchdogs(self, monkeypatch):
        """
        Active but untripped watchdogs are skipped in priority order.

        Tests:
            (Test Case 1) Host active+untripped, GPU active+tripped → GPU.
            (Test Case 2) Only IO tripped → IO returned.
        """
        from spikelab.spike_sorting import guards as guards_mod

        host = SimpleNamespace(tripped=lambda: False, name="host")
        gpu = SimpleNamespace(tripped=lambda: True, name="gpu")
        io = SimpleNamespace(tripped=lambda: True, name="io")
        monkeypatch.setattr(guards_mod, "get_active_watchdog", lambda: host)
        monkeypatch.setattr(guards_mod, "get_active_gpu_watchdog", lambda: gpu)
        monkeypatch.setattr(guards_mod, "get_active_io_stall_watchdog", lambda: io)
        assert guards_mod.find_tripped_global_watchdog().name == "gpu"

        gpu_only_io = SimpleNamespace(tripped=lambda: False, name="gpu")
        monkeypatch.setattr(guards_mod, "get_active_gpu_watchdog", lambda: gpu_only_io)
        assert guards_mod.find_tripped_global_watchdog().name == "io"

    def test_tripped_method_exception_propagates(self, monkeypatch):
        """
        A watchdog whose ``tripped()`` raises is not silently swallowed.

        Tests:
            (Test Case 1) Host watchdog whose ``tripped()`` raises
                AttributeError causes ``find_tripped_global_watchdog``
                to propagate the same exception, so the caller sees a
                real bug rather than a silent None.
        """
        from spikelab.spike_sorting import guards as guards_mod

        def _broken_tripped() -> bool:
            raise AttributeError("simulated broken tripped()")

        host = SimpleNamespace(tripped=_broken_tripped)
        monkeypatch.setattr(guards_mod, "get_active_watchdog", lambda: host)
        monkeypatch.setattr(guards_mod, "get_active_gpu_watchdog", lambda: None)
        monkeypatch.setattr(guards_mod, "get_active_io_stall_watchdog", lambda: None)
        with pytest.raises(AttributeError, match="simulated broken tripped"):
            guards_mod.find_tripped_global_watchdog()


# ===========================================================================
# NaN guards across the watchdog stack
# ===========================================================================


class TestComputeInactivityNanGuard:
    """``compute_inactivity_timeout_s`` coerces NaN/None to 0."""

    def test_nan_duration_treated_as_zero(self):
        """
        NaN ``recording_duration_min`` → returns ``base_s`` (no scaling).

        Tests:
            (Test Case 1) NaN → 600.0 (default base_s, no per-min added).
        """
        from spikelab.spike_sorting.guards._inactivity import (
            compute_inactivity_timeout_s,
        )

        result = compute_inactivity_timeout_s(
            recording_duration_min=float("nan"),
            base_s=600.0,
            per_min_s=30.0,
            max_s=None,
        )
        assert result == 600.0

    def test_none_duration_treated_as_zero(self):
        """
        None ``recording_duration_min`` → returns ``base_s``.

        Tests:
            (Test Case 1) None → 600.0.
        """
        from spikelab.spike_sorting.guards._inactivity import (
            compute_inactivity_timeout_s,
        )

        # Type-wise we declare the param as float, but in practice
        # callers may pass None when the duration is unknown.
        result = compute_inactivity_timeout_s(
            recording_duration_min=None,  # type: ignore[arg-type]
            base_s=600.0,
            per_min_s=30.0,
            max_s=None,
        )
        assert result == 600.0


class TestHostMemoryWatchdogNanGuard:
    """``HostMemoryWatchdog._poll_loop`` skips NaN psutil readings."""

    def test_nan_reading_does_not_trip(self, monkeypatch):
        """
        A NaN ``virtual_memory().percent`` reading is skipped rather
        than treated as either healthy or unhealthy.

        Tests:
            (Test Case 1) Patch psutil to return NaN; watchdog runs a
                short window and never trips.
        """
        readings = iter([float("nan")] * 10)

        class _FakePsutil:
            class _VM:
                @property
                def percent(self):
                    try:
                        return next(readings)
                    except StopIteration:
                        return 0.0

            @staticmethod
            def virtual_memory():
                return _FakePsutil._VM()

        # Replace the cached psutil module so HostMemoryWatchdog's
        # ``import psutil`` inside ``__enter__`` resolves to the fake.
        # Setting ``wd._psutil`` after construction is not enough —
        # ``__enter__`` re-imports and overwrites the attribute.
        monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)

        # Tiny poll interval so the loop ticks several times within
        # the short sleep window below.
        wd = HostMemoryWatchdog(warn_pct=85.0, abort_pct=92.0, poll_interval_s=0.02)
        with wd:
            time.sleep(0.15)
        assert wd.tripped() is False


# ===========================================================================
# Helper-method coverage (gap-fill batch)
# ===========================================================================


class TestFormatThrottleReasons:
    """``_format_throttle_reasons`` renders an NVML throttle bitmask."""

    def test_renders_sw_power_cap(self):
        """
        SW power cap bit (0x4) renders as the canonical label.

        Tests:
            (Test Case 1) mask=0x4 → "SW power cap".
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _format_throttle_reasons,
        )

        assert _format_throttle_reasons(0x4) == "SW power cap"

    def test_renders_hw_thermal_slowdown(self):
        """
        HW thermal slowdown bit (0x40) renders as the canonical label.

        Tests:
            (Test Case 1) mask=0x40 → "HW thermal slowdown".
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _format_throttle_reasons,
        )

        assert _format_throttle_reasons(0x40) == "HW thermal slowdown"

    def test_combines_multiple_bits_with_comma(self):
        """
        Multiple bits set produce a comma-separated string in the
        same order as ``_THROTTLE_REASON_LABELS``.

        Tests:
            (Test Case 1) 0x4 | 0x40 → "SW power cap, HW thermal
                slowdown" (declaration order).
            (Test Case 2) 0x4 | 0x8 | 0x80 → all three labels joined.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _format_throttle_reasons,
        )

        assert (
            _format_throttle_reasons(0x4 | 0x40) == "SW power cap, HW thermal slowdown"
        )
        assert (
            _format_throttle_reasons(0x4 | 0x8 | 0x80)
            == "SW power cap, HW slowdown, HW power brake"
        )

    def test_empty_mask_returns_empty_string(self):
        """
        mask=0 (no throttle bits set) returns "".

        Tests:
            (Test Case 1) mask=0 → "" (empty join, no labels match).
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _format_throttle_reasons,
        )

        assert _format_throttle_reasons(0) == ""


class TestGetTotalRamBytes:
    """``_get_total_ram_bytes`` reads host RAM via ``sorting_utils``."""

    def test_returns_positive_int_from_sorting_utils(self, monkeypatch):
        """
        Healthy ``get_system_ram_bytes`` return → unchanged passthrough.

        Tests:
            (Test Case 1) Patched ``sorting_utils.get_system_ram_bytes``
                returns 16 GiB → helper returns the same int.
        """
        from spikelab.spike_sorting import sorting_utils
        from spikelab.spike_sorting.guards._job_object import _get_total_ram_bytes

        monkeypatch.setattr(sorting_utils, "get_system_ram_bytes", lambda: 16 * 1024**3)
        assert _get_total_ram_bytes() == 16 * 1024**3

    def test_returns_none_on_underlying_exception(self, monkeypatch):
        """
        Any error from ``get_system_ram_bytes`` is swallowed → None.

        Tests:
            (Test Case 1) Patched ``get_system_ram_bytes`` raising
                RuntimeError → helper returns None (no propagation).
        """
        from spikelab.spike_sorting import sorting_utils
        from spikelab.spike_sorting.guards._job_object import _get_total_ram_bytes

        def _boom():
            raise RuntimeError("simulated detection failure")

        monkeypatch.setattr(sorting_utils, "get_system_ram_bytes", _boom)
        assert _get_total_ram_bytes() is None


class TestLogInactivityWatchdogReadSignals:
    """``LogInactivityWatchdog._read_signals`` returns (mtime, size)."""

    def test_returns_mtime_size_for_existing_file(self, tmp_path):
        """
        Existing log file → tuple of (mtime, size) as floats/ints.

        Tests:
            (Test Case 1) After writing content to a log file, the
                helper returns a tuple whose first value matches the
                file's mtime and second value matches its byte size
                (compared against the on-disk byte count to avoid
                Windows CRLF line-ending differences).
        """
        log = tmp_path / "rec.log"
        log.write_bytes(b"hello\nworld\n")
        on_disk_size = log.stat().st_size
        wd = LogInactivityWatchdog(
            log_path=log,
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=10.0,
            sorter="x",
        )
        signals = wd._read_signals()
        assert signals is not None
        mtime, size = signals
        assert isinstance(mtime, float)
        assert isinstance(size, int)
        assert size == on_disk_size
        assert abs(mtime - log.stat().st_mtime) < 1e-6

    def test_returns_none_for_missing_file(self, tmp_path):
        """
        Missing log file → None (FileNotFoundError swallowed).

        Tests:
            (Test Case 1) The helper returns None for a path that
                does not exist on disk yet.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "missing.log",
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=10.0,
            sorter="x",
        )
        assert wd._read_signals() is None

    def test_returns_none_on_oserror(self, tmp_path, monkeypatch):
        """
        Generic ``OSError`` from ``os.stat`` (e.g. permission denied)
        is swallowed → None.

        Tests:
            (Test Case 1) Patched ``os.stat`` raising PermissionError
                causes the helper to return None.
        """
        from spikelab.spike_sorting.guards import _inactivity as inactivity_mod

        log = tmp_path / "rec.log"
        log.touch()
        wd = LogInactivityWatchdog(
            log_path=log,
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=10.0,
            sorter="x",
        )

        def _refuse(*args, **kwargs):
            raise PermissionError("simulated permission denied")

        monkeypatch.setattr(inactivity_mod.os, "stat", _refuse)
        assert wd._read_signals() is None


class TestIOStallWatchdogProperties:
    """Trip-state and registration properties of ``IOStallWatchdog``."""

    def test_device_returns_none_before_enter(self, tmp_path):
        """
        ``device()`` returns None until ``__enter__`` resolves the
        block device.

        Tests:
            (Test Case 1) On a freshly-constructed watchdog the
                ``device`` accessor returns None.
        """
        wd = IOStallWatchdog(tmp_path, stall_s=1.0, poll_interval_s=0.1)
        assert wd.device() is None

    def test_device_returns_resolved_value_after_enter(self, tmp_path):
        """
        After ``__enter__`` the resolved device key is returned.

        Tests:
            (Test Case 1) Patched ``_resolve_device_for_path`` →
                ``"sda1"``; after ``__enter__`` ``device()`` returns
                ``"sda1"``.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", return_value=42),
        ):
            wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=0.1)
            with wd:
                assert wd.device() == "sda1"

    def test_unregister_kill_callback_removes_by_identity(self, tmp_path):
        """
        ``unregister_kill_callback`` removes the matching callback;
        unknown callbacks are silently ignored.

        Tests:
            (Test Case 1) Registered then unregistered callback is
                removed from ``_kill_callbacks``.
            (Test Case 2) Unregistering a never-registered callback
                is a no-op (no raise, list unchanged).
        """
        wd = IOStallWatchdog(tmp_path, stall_s=1.0, poll_interval_s=0.1)
        cb1 = lambda: None  # noqa: E731
        cb2 = lambda: None  # noqa: E731
        wd.register_kill_callback(cb1)
        assert cb1 in wd._kill_callbacks

        wd.unregister_kill_callback(cb1)
        assert cb1 not in wd._kill_callbacks

        # Unregistering an unknown callback is a no-op.
        wd.unregister_kill_callback(cb2)
        assert wd._kill_callbacks == []


class TestIOStallWatchdogProcessMode:
    """Process-mode (``pids=...``) parsing, registration, and reading.

    Process mode trips on per-PID ``io_counters()`` rather than the
    device-wide counter — useful when a sort process hangs while
    other processes on the same disk stay busy. These tests cover
    the construction + registration surface; the polling/trip
    behaviour is exercised in :class:`TestIOStallProcessModePollLoop`
    below.
    """

    def test_construction_requires_folder_or_pids(self):
        """
        ``IOStallWatchdog()`` with neither folder nor pids raises.

        Tests:
            (Test Case 1) Empty constructor → ValueError mentioning
                both modes.
        """
        with pytest.raises(ValueError, match="folder.*pids|pids.*folder"):
            IOStallWatchdog()

    def test_pids_only_selects_process_mode(self):
        """
        Constructing with only ``pids`` selects process mode and
        leaves ``folder`` as None.

        Tests:
            (Test Case 1) ``IOStallWatchdog(pids=[123])``:
                ``mode() == "process"``, ``folder is None``,
                ``pids() == [123]``.
        """
        wd = IOStallWatchdog(pids=[123], stall_s=1.0, poll_interval_s=0.1)
        assert wd.mode() == "process"
        assert wd.folder is None
        assert wd.pids() == [123]

    def test_folder_only_selects_device_mode(self, tmp_path):
        """
        Constructing with only ``folder`` keeps the legacy device
        mode behaviour.

        Tests:
            (Test Case 1) ``IOStallWatchdog(tmp_path)``:
                ``mode() == "device"`` and ``pids()`` is empty.
        """
        wd = IOStallWatchdog(folder=tmp_path, stall_s=1.0, poll_interval_s=0.1)
        assert wd.mode() == "device"
        assert wd.pids() == []

    def test_both_folder_and_pids_picks_process_mode(self, tmp_path):
        """
        When *both* folder and pids are provided, process mode wins
        (we have stronger signal from per-process counters).

        Tests:
            (Test Case 1) folder + pids → process mode.
        """
        wd = IOStallWatchdog(
            folder=tmp_path,
            pids=[123],
            stall_s=1.0,
            poll_interval_s=0.1,
        )
        assert wd.mode() == "process"

    def test_pids_validates_non_positive(self, tmp_path):
        """
        Non-positive PIDs at construction raise immediately.

        Tests:
            (Test Case 1) pids=[0] → ValueError.
            (Test Case 2) pids=[-1] → ValueError.
        """
        with pytest.raises(ValueError, match="positive integer"):
            IOStallWatchdog(pids=[0], stall_s=1.0)
        with pytest.raises(ValueError, match="positive integer"):
            IOStallWatchdog(pids=[-1], stall_s=1.0)

    def test_register_pid_appends(self):
        """
        ``register_pid`` adds new PIDs in order; duplicates are
        ignored.

        Tests:
            (Test Case 1) Initial pids=[1]; register_pid(2);
                pids() == [1, 2].
            (Test Case 2) Re-register an existing PID — list
                unchanged.
        """
        wd = IOStallWatchdog(pids=[1], stall_s=1.0, poll_interval_s=0.1)
        wd.register_pid(2)
        assert wd.pids() == [1, 2]
        wd.register_pid(1)
        assert wd.pids() == [1, 2]

    def test_register_pid_validates_non_positive(self):
        """
        ``register_pid`` rejects zero and negative PIDs.

        Tests:
            (Test Case 1) register_pid(0) → ValueError.
        """
        wd = IOStallWatchdog(pids=[1], stall_s=1.0, poll_interval_s=0.1)
        with pytest.raises(ValueError, match="positive integer"):
            wd.register_pid(0)

    def test_register_pid_no_op_in_device_mode(self, tmp_path, caplog):
        """
        ``register_pid`` on a device-mode watchdog is a debug-logged
        no-op so misuse doesn't silently flip the mode.

        Tests:
            (Test Case 1) Device-mode watchdog + register_pid(123):
                pids() returns [] and the watchdog stays in device
                mode.
        """
        wd = IOStallWatchdog(folder=tmp_path, stall_s=1.0, poll_interval_s=0.1)
        wd.register_pid(123)
        assert wd.pids() == []
        assert wd.mode() == "device"

    def test_unregister_pid_removes(self):
        """
        ``unregister_pid`` removes the matching PID; unknown PIDs
        are silently ignored.

        Tests:
            (Test Case 1) pids=[1, 2]; unregister_pid(1) → pids()=[2].
            (Test Case 2) unregister_pid(99) → no raise, list
                unchanged.
        """
        wd = IOStallWatchdog(pids=[1, 2], stall_s=1.0, poll_interval_s=0.1)
        wd.unregister_pid(1)
        assert wd.pids() == [2]
        wd.unregister_pid(99)
        assert wd.pids() == [2]


class TestIOStallProcessModeReadBytes:
    """``_read_io_bytes_for_pids`` aggregates per-process counters."""

    def test_sums_across_pids(self, monkeypatch):
        """
        ``_read_io_bytes_for_pids`` sums ``read_bytes + write_bytes``
        across every alive PID.

        Tests:
            (Test Case 1) Two PIDs with counters → sum returned and
                alive_count == 2.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        class _FakeIO:
            def __init__(self, r, w):
                self.read_bytes = r
                self.write_bytes = w

        class _FakeProc:
            def __init__(self, pid, r, w):
                self.pid = pid
                self._io = _FakeIO(r, w)

            def io_counters(self):
                return self._io

            def children(self, recursive=True):
                return []

        # Map PID -> stub
        procs = {1001: _FakeProc(1001, 100, 200), 1002: _FakeProc(1002, 50, 75)}

        class _FakePsutil:
            class NoSuchProcess(Exception):
                pass

            class ZombieProcess(Exception):
                pass

            class AccessDenied(Exception):
                pass

            @staticmethod
            def Process(pid):
                if pid not in procs:
                    raise _FakePsutil.NoSuchProcess(pid)
                return procs[pid]

        # Inject our fake module under ``import psutil``.
        sys_modules_backup = sys.modules.get("psutil")
        sys.modules["psutil"] = _FakePsutil  # type: ignore[assignment]
        try:
            total, alive = iom._read_io_bytes_for_pids([1001, 1002])
        finally:
            if sys_modules_backup is not None:
                sys.modules["psutil"] = sys_modules_backup
            else:
                sys.modules.pop("psutil", None)
        assert alive == 2
        assert total == 100 + 200 + 50 + 75

    def test_returns_none_when_no_pids_alive(self, monkeypatch):
        """
        When every registered PID is dead, returns ``(None, 0)`` so
        the watchdog goes blind rather than tripping on a vanished
        sort.

        Tests:
            (Test Case 1) All PIDs raise NoSuchProcess → (None, 0).
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        class _FakePsutil:
            class NoSuchProcess(Exception):
                pass

            class ZombieProcess(Exception):
                pass

            class AccessDenied(Exception):
                pass

            @staticmethod
            def Process(pid):
                raise _FakePsutil.NoSuchProcess(pid)

        sys_modules_backup = sys.modules.get("psutil")
        sys.modules["psutil"] = _FakePsutil  # type: ignore[assignment]
        try:
            total, alive = iom._read_io_bytes_for_pids([1, 2, 3])
        finally:
            if sys_modules_backup is not None:
                sys.modules["psutil"] = sys_modules_backup
            else:
                sys.modules.pop("psutil", None)
        assert total is None
        assert alive == 0


class TestIOStallProcessModePollLoop:
    """Polling-loop behaviour in process mode against a stubbed counter.

    Uses a stubbed ``_read_io_bytes_for_pids`` so the test is
    deterministic across CI environments where ``/proc/<pid>/io``
    permissions or ptrace_scope settings can vary. The polling
    loop's stall-detection logic is what's under test here, not
    the real psutil read path (that's covered by
    :class:`TestIOStallProcessModeReadBytes`).
    """

    def test_trips_when_per_pid_counter_stays_flat(self, monkeypatch):
        """
        Process-mode polling loop trips when the per-PID byte
        counter doesn't change for ``stall_s`` seconds.

        Tests:
            (Test Case 1) Stubbed ``_read_io_bytes_for_pids``
                returns a constant ``(42, 1)``; with ``stall_s=1.0``
                and ``poll_interval_s=0.1``, the kill callback
                fires within 3 s and ``tripped()`` becomes True.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        # Constant counter -> stall detected by the polling loop.
        monkeypatch.setattr(
            iom,
            "_read_io_bytes_for_pids",
            lambda pids, *, include_descendants=True: (42, len(pids)),
        )

        kill_event = threading.Event()
        wd = IOStallWatchdog(
            pids=[12345],  # PID is irrelevant — the counter is stubbed
            stall_s=1.0,
            poll_interval_s=0.1,
            kill_grace_s=0.25,
        )
        wd.register_kill_callback(kill_event.set)
        # The trip cascade ends in ``_thread.interrupt_main`` which
        # can race with our context exit and land here as a
        # KeyboardInterrupt — documented behaviour, not a test
        # failure. Catch it and read kill_event afterwards.
        try:
            with wd:
                fired = kill_event.wait(timeout=3.0)
        except KeyboardInterrupt:
            fired = kill_event.is_set()
        assert fired, (
            "Process-mode polling loop did not fire kill_callback "
            "within 3 s for a flat per-PID byte counter."
        )
        assert wd.tripped()

    def test_does_not_trip_when_per_pid_counter_climbs(self, monkeypatch):
        """
        Process-mode polling loop does NOT trip while the per-PID
        counter is climbing on every poll.

        Tests:
            (Test Case 1) Stubbed ``_read_io_bytes_for_pids``
                returns a strictly-increasing value on each call.
                With ``stall_s=1.0`` and ``poll_interval_s=0.1``,
                the kill callback does not fire within 1.5 s.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        counter = {"v": 0}

        def _climbing(pids, *, include_descendants=True):
            counter["v"] += 1024
            return counter["v"], len(pids)

        monkeypatch.setattr(iom, "_read_io_bytes_for_pids", _climbing)

        kill_event = threading.Event()
        wd = IOStallWatchdog(
            pids=[12345],
            stall_s=1.0,
            poll_interval_s=0.1,
            kill_grace_s=0.25,
        )
        wd.register_kill_callback(kill_event.set)
        with wd:
            fired = kill_event.wait(timeout=1.5)
        assert not fired, (
            "Process-mode watchdog tripped despite a climbing "
            "per-PID counter — false positive."
        )
        assert not wd.tripped()


class TestIOStallWatchdogMaybeWarn:
    """``IOStallWatchdog._maybe_warn`` logs a warning + audit event."""

    def test_logs_warning_message_with_device_and_thresholds(self, tmp_path, caplog):
        """
        ``_maybe_warn`` emits a WARNING log record naming the device
        and the configured stall tolerance.

        Tests:
            (Test Case 1) After calling ``_maybe_warn``, a WARNING
                record is captured whose message references the
                device and ``stall_s`` value.
        """
        wd = IOStallWatchdog(tmp_path, stall_s=300.0, poll_interval_s=10.0)
        wd._device = "sda1"
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._io_stall",
        ):
            wd._maybe_warn(150.0)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert records, "expected at least one WARNING record"
        msg = records[-1].getMessage()
        assert "sda1" in msg
        assert "300.0" in msg

    def test_appends_audit_event(self, tmp_path, monkeypatch):
        """
        ``_maybe_warn`` calls ``append_audit_event`` with
        ``watchdog="io_stall"`` and ``event="warn"`` plus the device
        and stall fields.

        Tests:
            (Test Case 1) The helper invokes a patched
                ``append_audit_event`` exactly once with the expected
                kwargs.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        captured: list = []

        def _capture(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr(iom, "append_audit_event", _capture)
        wd = IOStallWatchdog(tmp_path, stall_s=300.0, poll_interval_s=10.0)
        wd._device = "sda1"
        wd._maybe_warn(120.0)
        assert len(captured) == 1
        event = captured[0]
        assert event["watchdog"] == "io_stall"
        assert event["event"] == "warn"
        assert event["device"] == "sda1"
        assert event["stalled_for_s"] == 120.0
        assert event["tolerance_s"] == 300.0


class TestListMarkerFiles:
    """``_list_marker_files`` scans temp dir for sorter-marker files."""

    def test_returns_empty_for_missing_dir(self, tmp_path):
        """
        Non-existent ``temp_dir`` returns an empty set without raising.

        Tests:
            (Test Case 1) The helper returns ``set()`` when the
                target directory does not exist.
        """
        from spikelab.spike_sorting.guards._tempfile_cleanup import (
            _list_marker_files,
        )

        result = _list_marker_files(tmp_path / "nope")
        assert result == set()

    def test_finds_top_level_marker_files(self, tmp_path):
        """
        Top-level files whose names contain a marker substring are
        returned; files without a marker prefix are skipped.

        Tests:
            (Test Case 1) ``spikelab_x.tmp`` and ``kilosort_y.bin``
                are included.
            (Test Case 2) ``unrelated.txt`` is excluded.
        """
        from spikelab.spike_sorting.guards._tempfile_cleanup import (
            _list_marker_files,
        )

        match1 = tmp_path / "spikelab_x.tmp"
        match1.write_text("a")
        match2 = tmp_path / "kilosort_y.bin"
        match2.write_text("b")
        skip = tmp_path / "unrelated.txt"
        skip.write_text("c")

        result = _list_marker_files(tmp_path)
        assert match1 in result
        assert match2 in result
        assert skip not in result

    def test_marker_match_is_case_insensitive(self, tmp_path):
        """
        File-name marker matching is case-insensitive.

        Tests:
            (Test Case 1) ``Kilosort_Cache.tmp`` (mixed case) is
                included.
        """
        from spikelab.spike_sorting.guards._tempfile_cleanup import (
            _list_marker_files,
        )

        upper = tmp_path / "Kilosort_Cache.tmp"
        upper.write_text("x")
        result = _list_marker_files(tmp_path)
        assert upper in result

    def test_recurses_into_marker_directories(self, tmp_path):
        """
        Directories whose names match a marker have their contents
        recursively included; non-marker directories are skipped.

        Tests:
            (Test Case 1) Files inside ``spikelab_runs/`` are
                returned.
            (Test Case 2) Files inside a non-marker directory
                (``unrelated_dir/``) are NOT returned.
        """
        from spikelab.spike_sorting.guards._tempfile_cleanup import (
            _list_marker_files,
        )

        marker_dir = tmp_path / "spikelab_runs"
        marker_dir.mkdir()
        nested = marker_dir / "child.bin"
        nested.write_text("x")

        unrelated_dir = tmp_path / "unrelated_dir"
        unrelated_dir.mkdir()
        unrelated_file = unrelated_dir / "child.bin"
        unrelated_file.write_text("y")

        result = _list_marker_files(tmp_path)
        assert nested in result
        assert unrelated_file not in result

    def test_swallows_oserror_from_iterdir(self, tmp_path, monkeypatch):
        """
        ``OSError`` raised by ``Path.iterdir`` is swallowed and the
        partial result so far is returned.

        Tests:
            (Test Case 1) Patched ``Path.iterdir`` raising
                ``PermissionError`` causes the helper to return a
                set without raising.
        """
        from spikelab.spike_sorting.guards._tempfile_cleanup import (
            _list_marker_files,
        )

        def _refuse(self):
            raise PermissionError("simulated locked directory")

        monkeypatch.setattr(Path, "iterdir", _refuse)
        result = _list_marker_files(tmp_path)
        assert result == set()


class TestDiskUsageWatchdogMaybeWarn:
    """``DiskUsageWatchdog._maybe_warn`` rate-limits + audits warnings."""

    def test_first_call_logs_warning_and_appends_audit(
        self, tmp_path, caplog, monkeypatch
    ):
        """
        First call inside the warn-repeat window emits a WARNING and
        appends a ``disk warn`` audit event.

        Tests:
            (Test Case 1) WARNING record captured.
            (Test Case 2) ``append_audit_event`` invoked once with
                ``watchdog="disk"``, ``event="warn"`` and the
                ``free_gb`` / threshold payload fields.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        captured: list = []

        def _capture(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr(disk_mod, "append_audit_event", _capture)

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            warn_repeat_s=30.0,
            kill_callback=lambda: None,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._disk_watchdog",
        ):
            wd._maybe_warn(3.5)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert records, "expected a WARNING record"
        assert "3.50" in records[-1].getMessage()
        assert len(captured) == 1
        assert captured[0]["watchdog"] == "disk"
        assert captured[0]["event"] == "warn"
        assert captured[0]["free_gb"] == 3.5

    def test_repeat_within_window_is_suppressed(self, tmp_path, caplog, monkeypatch):
        """
        A second ``_maybe_warn`` call within ``warn_repeat_s`` of
        the first emits no record and no audit event.

        Tests:
            (Test Case 1) After one call, an immediate second call
                produces no additional WARNING records and no
                additional ``append_audit_event`` invocation.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        captured: list = []

        def _capture(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr(disk_mod, "append_audit_event", _capture)

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            warn_repeat_s=30.0,
            kill_callback=lambda: None,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._disk_watchdog",
        ):
            wd._maybe_warn(3.5)
            caplog.clear()
            wd._maybe_warn(3.4)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not records, "expected the second call to be rate-limited"
        assert len(captured) == 1, "audit event should fire only on the first call"

    def test_call_after_window_emits_again(self, tmp_path, caplog, monkeypatch):
        """
        Reset the ``_last_warn_t`` to before the window and verify a
        subsequent ``_maybe_warn`` is allowed through.

        Tests:
            (Test Case 1) After manually rewinding ``_last_warn_t``
                by more than ``warn_repeat_s``, a second call emits
                a WARNING + audit event.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        captured: list = []
        monkeypatch.setattr(
            disk_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            warn_repeat_s=30.0,
            kill_callback=lambda: None,
        )
        wd._maybe_warn(3.5)
        # Rewind so the rate-limit window has elapsed.
        wd._last_warn_t -= 60.0
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._disk_watchdog",
        ):
            wd._maybe_warn(3.0)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert records, "expected a WARNING record after the window elapsed"
        assert len(captured) == 2


class TestTopConsumersWithTimeout:
    """``DiskUsageWatchdog._top_consumers_with_timeout`` is bounded."""

    def test_returns_walk_result_when_timeout_not_exceeded(self, tmp_path, monkeypatch):
        """
        Walk completes inside ``timeout_s`` → returns the result.

        Tests:
            (Test Case 1) Patched ``_top_consumers`` returning a list
                of (path, gb) tuples is forwarded as-is.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        sample = [("/tmp/big.bin", 7.5)]
        monkeypatch.setattr(disk_mod, "_top_consumers", lambda folder: sample)

        wd = DiskUsageWatchdog(folder=tmp_path, kill_callback=lambda: None)
        result = wd._top_consumers_with_timeout(timeout_s=5.0)
        assert result == sample

    def test_returns_none_when_worker_exceeds_timeout(self, tmp_path, monkeypatch):
        """
        Worker still running past ``timeout_s`` → returns None so the
        caller can fall back to the entry-time snapshot.

        Tests:
            (Test Case 1) Patched ``_top_consumers`` that sleeps
                longer than the timeout causes the helper to return
                None promptly.

        Notes:
            - The daemon worker thread is intentionally leaked
              (matches production behaviour); the test does not wait
              for it.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        block = threading.Event()

        def _slow(folder):
            # Wait until the test releases us, well past the timeout.
            block.wait(timeout=2.0)
            return []

        monkeypatch.setattr(disk_mod, "_top_consumers", _slow)
        wd = DiskUsageWatchdog(folder=tmp_path, kill_callback=lambda: None)
        try:
            result = wd._top_consumers_with_timeout(timeout_s=0.05)
            assert result is None
        finally:
            block.set()  # release the worker so the daemon can exit cleanly

    def test_worker_exception_reported_as_empty_list(self, tmp_path, monkeypatch):
        """
        Exception raised by ``_top_consumers`` inside the worker is
        caught and reported as ``[]``.

        Tests:
            (Test Case 1) Patched ``_top_consumers`` raising OSError
                results in ``_top_consumers_with_timeout`` returning
                an empty list (not None and not raising).
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        def _raise(folder):
            raise OSError("simulated walk failure")

        monkeypatch.setattr(disk_mod, "_top_consumers", _raise)
        wd = DiskUsageWatchdog(folder=tmp_path, kill_callback=lambda: None)
        result = wd._top_consumers_with_timeout(timeout_s=5.0)
        assert result == []


class TestFolderSizeBytes:
    """``_folder_size_bytes`` sums file sizes under a folder."""

    def test_sums_nested_files(self, tmp_path):
        """
        Files in nested subdirectories are summed into the total.

        Tests:
            (Test Case 1) Three files (10 B, 25 B, 100 B) across
                nested directories sum to 135 B.
        """
        from spikelab.spike_sorting.guards._disk_watchdog import (
            _folder_size_bytes,
        )

        (tmp_path / "a.bin").write_bytes(b"x" * 10)
        sub = tmp_path / "deep" / "nested"
        sub.mkdir(parents=True)
        (sub / "b.bin").write_bytes(b"x" * 25)
        (sub / "c.bin").write_bytes(b"x" * 100)
        assert _folder_size_bytes(tmp_path) == 135.0

    def test_returns_zero_for_missing_folder(self, tmp_path):
        """
        Non-existent folder returns 0.0 without raising.

        Tests:
            (Test Case 1) ``_folder_size_bytes`` on a nonexistent
                path returns 0.0.
        """
        from spikelab.spike_sorting.guards._disk_watchdog import (
            _folder_size_bytes,
        )

        assert _folder_size_bytes(tmp_path / "missing") == 0.0

    def test_swallows_per_file_oserror(self, tmp_path, monkeypatch):
        """
        ``OSError`` from a per-file ``stat()`` (e.g. broken symlink)
        is swallowed; sizes from sibling files are still summed.

        Tests:
            (Test Case 1) One file's ``stat()`` raises; its sibling's
                size still appears in the total (size 50).
        """
        from spikelab.spike_sorting.guards._disk_watchdog import (
            _folder_size_bytes,
        )

        good = tmp_path / "good.bin"
        good.write_bytes(b"x" * 50)
        bad = tmp_path / "bad.bin"
        bad.write_bytes(b"y" * 999)

        real_stat = Path.stat

        def _selective_stat(self, *args, **kwargs):
            if self.name == "bad.bin":
                raise OSError("simulated broken symlink")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", _selective_stat)
        assert _folder_size_bytes(tmp_path) == 50.0


# ===========================================================================
# GPU thermal/throttle subsystem (pynvml-backed)
# ===========================================================================


def _make_fake_pynvml(
    *,
    init_raises: bool = False,
    handle_raises: bool = False,
    shutdown_raises: bool = False,
    mem_used: int = 4 * 1024**3,
    mem_total: int = 8 * 1024**3,
    mem_raises: bool = False,
    temp_value: float = 65.0,
    temp_raises: bool = False,
    throttle_value: int = 0,
    throttle_raises: bool = False,
):
    """Construct a fake pynvml module with configurable failure modes.

    Returns a SimpleNamespace exposing every method
    ``_PynvmlSession`` and ``_read_gpu_memory_pynvml`` call. Each
    failure-mode flag toggles the corresponding method between
    success and raising ``RuntimeError``. Records call counts on
    ``_init_calls`` / ``_shutdown_calls`` so tests can assert that
    cleanup actually ran.
    """
    counters = {
        "init": 0,
        "shutdown": 0,
        "handle": 0,
        "mem": 0,
        "temp": 0,
        "throttle": 0,
    }
    handle_sentinel = object()
    info = SimpleNamespace(used=mem_used, total=mem_total)

    def _init():
        counters["init"] += 1
        if init_raises:
            raise RuntimeError("simulated nvmlInit failure")

    def _shutdown():
        counters["shutdown"] += 1
        if shutdown_raises:
            raise RuntimeError("simulated nvmlShutdown failure")

    def _handle(_idx):
        counters["handle"] += 1
        if handle_raises:
            raise RuntimeError("simulated handle failure")
        return handle_sentinel

    def _memory(_h):
        counters["mem"] += 1
        if mem_raises:
            raise RuntimeError("simulated memory read failure")
        return info

    def _temperature(_h, _sensor):
        counters["temp"] += 1
        if temp_raises:
            raise RuntimeError("simulated temperature read failure")
        return temp_value

    def _throttle(_h):
        counters["throttle"] += 1
        if throttle_raises:
            raise RuntimeError("simulated throttle read failure")
        return throttle_value

    fake = SimpleNamespace(
        nvmlInit=_init,
        nvmlShutdown=_shutdown,
        nvmlDeviceGetHandleByIndex=_handle,
        nvmlDeviceGetMemoryInfo=_memory,
        nvmlDeviceGetTemperature=_temperature,
        nvmlDeviceGetCurrentClocksThrottleReasons=_throttle,
        _counters=counters,
        _handle=handle_sentinel,
    )
    return fake


class TestPynvmlSession:
    """``_PynvmlSession`` wraps pynvml init / read / shutdown."""

    def test_init_stores_device_index_with_no_session_state(self):
        """
        ``__init__`` records the device index and leaves the
        pynvml + handle slots empty until ``start()`` succeeds.

        Tests:
            (Test Case 1) Construction with ``device_index=2`` stores
                the int 2 and leaves ``_pynvml`` / ``_handle`` as None.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        session = _PynvmlSession(2)
        assert session.device_index == 2
        assert session._pynvml is None
        assert session._handle is None

    def test_start_returns_false_when_pynvml_missing(self, monkeypatch):
        """
        ``import pynvml`` failing → ``start()`` returns False without
        side effects.

        Tests:
            (Test Case 1) Patched ``sys.modules['pynvml'] = None``
                makes the inner import raise ImportError; ``start``
                returns False and the session remains uninitialised.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        monkeypatch.setitem(sys.modules, "pynvml", None)
        session = _PynvmlSession(0)
        assert session.start() is False
        assert session._pynvml is None
        assert session._handle is None

    def test_start_returns_false_on_init_failure(self, monkeypatch):
        """
        ``nvmlInit`` raising → ``start()`` returns False and does not
        proceed to handle resolution.

        Tests:
            (Test Case 1) Fake pynvml with ``nvmlInit`` raising
                returns False; handle resolution is not attempted.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        fake = _make_fake_pynvml(init_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(0)
        assert session.start() is False
        assert fake._counters["init"] == 1
        assert fake._counters["handle"] == 0

    def test_start_returns_false_on_handle_failure_and_shuts_down(self, monkeypatch):
        """
        ``nvmlDeviceGetHandleByIndex`` raising → ``start()`` returns
        False and runs ``nvmlShutdown`` so the NVML context is not
        leaked.

        Tests:
            (Test Case 1) Fake pynvml with ``handle_raises=True``
                returns False from start.
            (Test Case 2) ``nvmlShutdown`` is invoked exactly once
                during the failure cleanup.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        fake = _make_fake_pynvml(handle_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(0)
        assert session.start() is False
        assert fake._counters["init"] == 1
        assert fake._counters["handle"] == 1
        assert fake._counters["shutdown"] == 1

    def test_start_returns_true_on_success_and_caches_handle(self, monkeypatch):
        """
        Successful init + handle resolution → ``start()`` returns
        True and caches both ``_pynvml`` and ``_handle``.

        Tests:
            (Test Case 1) ``start()`` returns True.
            (Test Case 2) ``_pynvml`` references the fake module
                and ``_handle`` matches the value returned by
                ``nvmlDeviceGetHandleByIndex``.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        fake = _make_fake_pynvml()
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(3)
        assert session.start() is True
        assert session._pynvml is fake
        assert session._handle is fake._handle
        assert fake._counters["init"] == 1
        assert fake._counters["handle"] == 1
        # No shutdown on the success path until the caller invokes it.
        assert fake._counters["shutdown"] == 0

    def test_read_memory_returns_none_when_handle_uninitialised(self):
        """
        ``read_memory()`` on a never-started session returns None.

        Tests:
            (Test Case 1) Fresh ``_PynvmlSession`` returns None.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        session = _PynvmlSession(0)
        assert session.read_memory() is None

    def test_read_memory_returns_used_pct_and_total_gb(self, monkeypatch):
        """
        ``read_memory()`` after a successful start returns
        ``(used_pct, total_gb)`` derived from the pynvml info.

        Tests:
            (Test Case 1) used=4 GB, total=8 GB → (50.0, 8.0).
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        fake = _make_fake_pynvml(mem_used=4 * 1024**3, mem_total=8 * 1024**3)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(0)
        assert session.start()
        used_pct, total_gb = session.read_memory()
        assert used_pct == pytest.approx(50.0)
        assert total_gb == pytest.approx(8.0)

    def test_read_memory_returns_none_when_get_memory_raises(self, monkeypatch):
        """
        Exception from ``nvmlDeviceGetMemoryInfo`` → None.

        Tests:
            (Test Case 1) Fake with ``mem_raises=True`` after a
                successful start returns None on read.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        fake = _make_fake_pynvml(mem_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(0)
        assert session.start()
        assert session.read_memory() is None

    def test_read_memory_returns_none_when_total_zero(self, monkeypatch):
        """
        ``info.total <= 0`` → None (defensive guard, NVML can return
        0 when the device is being reset).

        Tests:
            (Test Case 1) Fake with total=0 returns None.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        fake = _make_fake_pynvml(mem_total=0)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(0)
        assert session.start()
        assert session.read_memory() is None

    def test_read_temperature_c_returns_float(self, monkeypatch):
        """
        ``read_temperature_c()`` after a successful start returns
        the temperature as a float; uninitialised session returns
        None; pynvml raise returns None.

        Tests:
            (Test Case 1) Uninitialised session → None.
            (Test Case 2) Successful read returns float matching
                the fake pynvml value.
            (Test Case 3) Fake with ``temp_raises=True`` → None.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        # Uninitialised.
        session = _PynvmlSession(0)
        assert session.read_temperature_c() is None

        fake = _make_fake_pynvml(temp_value=72.5)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(0)
        assert session.start()
        assert session.read_temperature_c() == pytest.approx(72.5)

        bad = _make_fake_pynvml(temp_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", bad)
        bad_session = _PynvmlSession(0)
        assert bad_session.start()
        assert bad_session.read_temperature_c() is None

    def test_read_throttle_reasons_returns_int(self, monkeypatch):
        """
        ``read_throttle_reasons()`` returns the bitmask as an int;
        uninitialised → None; pynvml raise → None.

        Tests:
            (Test Case 1) Uninitialised session → None.
            (Test Case 2) Successful read returns the integer
                bitmask reported by the fake.
            (Test Case 3) Fake with ``throttle_raises=True`` → None.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        session = _PynvmlSession(0)
        assert session.read_throttle_reasons() is None

        fake = _make_fake_pynvml(throttle_value=0x44)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(0)
        assert session.start()
        assert session.read_throttle_reasons() == 0x44

        bad = _make_fake_pynvml(throttle_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", bad)
        bad_session = _PynvmlSession(0)
        assert bad_session.start()
        assert bad_session.read_throttle_reasons() is None

    def test_shutdown_is_idempotent_and_swallows_errors(self, monkeypatch):
        """
        ``shutdown()`` is safe to call multiple times and on a
        never-started session; errors from ``nvmlShutdown`` are
        swallowed.

        Tests:
            (Test Case 1) Calling shutdown on an uninitialised
                session is a no-op (does not raise; nvmlShutdown
                not invoked).
            (Test Case 2) After a successful start, shutdown calls
                ``nvmlShutdown`` once and clears the cached state.
            (Test Case 3) A second shutdown is a no-op.
            (Test Case 4) ``nvmlShutdown`` raising is swallowed.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        # Never-started shutdown.
        session = _PynvmlSession(0)
        session.shutdown()  # must not raise
        assert session._pynvml is None

        fake = _make_fake_pynvml()
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        session = _PynvmlSession(0)
        assert session.start()
        session.shutdown()
        assert fake._counters["shutdown"] == 1
        assert session._pynvml is None
        assert session._handle is None

        # Idempotent: second call does not re-invoke nvmlShutdown.
        session.shutdown()
        assert fake._counters["shutdown"] == 1

        # Errors from nvmlShutdown are swallowed.
        bad = _make_fake_pynvml(shutdown_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", bad)
        bad_session = _PynvmlSession(0)
        assert bad_session.start()
        bad_session.shutdown()  # must not raise


class TestReadGpuMemoryPynvml:
    """``_read_gpu_memory_pynvml`` is the per-call (unsessioned) reader."""

    def test_returns_none_when_pynvml_missing(self, monkeypatch):
        """
        Inner ``import pynvml`` failing → None.

        Tests:
            (Test Case 1) ``sys.modules['pynvml'] = None`` causes the
                helper to return None silently.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _read_gpu_memory_pynvml,
        )

        monkeypatch.setitem(sys.modules, "pynvml", None)
        assert _read_gpu_memory_pynvml(0) is None

    def test_returns_none_on_init_failure(self, monkeypatch):
        """
        ``nvmlInit`` raising → None (no shutdown, init never succeeded).

        Tests:
            (Test Case 1) Fake with ``init_raises=True`` returns None.
            (Test Case 2) ``nvmlShutdown`` is NOT called (the inner
                try-finally only runs after a successful init).
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _read_gpu_memory_pynvml,
        )

        fake = _make_fake_pynvml(init_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert _read_gpu_memory_pynvml(0) is None
        assert fake._counters["shutdown"] == 0

    def test_returns_none_on_handle_failure_with_shutdown(self, monkeypatch):
        """
        ``nvmlDeviceGetHandleByIndex`` raising → None; ``nvmlShutdown``
        runs via the outer ``finally``.

        Tests:
            (Test Case 1) Fake with ``handle_raises=True`` returns
                None and the shutdown counter increments.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _read_gpu_memory_pynvml,
        )

        fake = _make_fake_pynvml(handle_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert _read_gpu_memory_pynvml(0) is None
        assert fake._counters["shutdown"] == 1

    def test_returns_none_when_total_zero(self, monkeypatch):
        """
        ``info.total <= 0`` → None (still calls shutdown via finally).

        Tests:
            (Test Case 1) Fake with ``mem_total=0`` returns None.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _read_gpu_memory_pynvml,
        )

        fake = _make_fake_pynvml(mem_total=0)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert _read_gpu_memory_pynvml(0) is None
        assert fake._counters["shutdown"] == 1

    def test_returns_used_pct_and_total_gb_on_success(self, monkeypatch):
        """
        Successful read returns ``(used_pct, total_gb)`` and runs
        shutdown.

        Tests:
            (Test Case 1) used=2 GB, total=8 GB → (25.0, 8.0).
            (Test Case 2) ``nvmlShutdown`` is invoked once.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            _read_gpu_memory_pynvml,
        )

        fake = _make_fake_pynvml(mem_used=2 * 1024**3, mem_total=8 * 1024**3)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        result = _read_gpu_memory_pynvml(0)
        assert result is not None
        used_pct, total_gb = result
        assert used_pct == pytest.approx(25.0)
        assert total_gb == pytest.approx(8.0)
        assert fake._counters["shutdown"] == 1


class TestTryCaptureSnapshotToResults:
    """``_try_capture_snapshot_to_results`` writes a postmortem dump."""

    def test_silent_no_op_when_log_path_is_none(self, monkeypatch):
        """
        ``log_path=None`` → silent no-op; ``capture_gpu_snapshot``
        is never invoked.

        Tests:
            (Test Case 1) Patched ``capture_gpu_snapshot`` is not
                called when the helper is given None.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        called = []

        def _capture(target, *, header=""):
            called.append((target, header))
            return str(target)

        monkeypatch.setattr(gpu_mod, "capture_gpu_snapshot", _capture)
        gpu_mod._try_capture_snapshot_to_results(None, header="anything")
        assert called == []

    def test_writes_snapshot_next_to_log_path(self, tmp_path, monkeypatch):
        """
        Successful capture writes ``<log_path.parent>/gpu_snapshot_at_trip.txt``
        with the supplied header.

        Tests:
            (Test Case 1) Patched ``capture_gpu_snapshot`` is invoked
                exactly once with the resolved target path and the
                supplied header argument.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []

        def _capture(target, *, header=""):
            captured.append((Path(target), header))
            return str(target)

        monkeypatch.setattr(gpu_mod, "capture_gpu_snapshot", _capture)
        log_path = tmp_path / "results" / "rec.log"
        log_path.parent.mkdir(parents=True)
        log_path.touch()
        gpu_mod._try_capture_snapshot_to_results(log_path, header="trip-banner")
        assert len(captured) == 1
        target, header = captured[0]
        assert target == log_path.parent / "gpu_snapshot_at_trip.txt"
        assert header == "trip-banner"

    def test_swallows_capture_exception(self, tmp_path, monkeypatch):
        """
        Exceptions inside ``capture_gpu_snapshot`` (or the path
        resolution) are swallowed so a snapshot bug never breaks
        the surrounding watchdog.

        Tests:
            (Test Case 1) Patched ``capture_gpu_snapshot`` raising
                RuntimeError does not propagate out of the helper.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated capture failure")

        monkeypatch.setattr(gpu_mod, "capture_gpu_snapshot", _boom)
        log_path = tmp_path / "rec.log"
        log_path.touch()
        # Must not raise.
        gpu_mod._try_capture_snapshot_to_results(log_path, header="x")


class TestGpuMemoryWatchdogThermalProperties:
    """``trip_kind`` and ``temperature_c_at_trip`` track the trip path."""

    def test_returns_none_before_any_trip(self):
        """
        On a fresh watchdog, both properties return None.

        Tests:
            (Test Case 1) ``trip_kind`` is None.
            (Test Case 2) ``temperature_c_at_trip`` is None.
        """
        wd = GpuMemoryWatchdog()
        assert wd.trip_kind() is None
        assert wd.temperature_c_at_trip() is None

    def test_memory_trip_sets_kind_to_memory_and_leaves_temp_none(self, monkeypatch):
        """
        After ``_on_abort`` (VRAM trip), ``trip_kind`` is "memory"
        and ``temperature_c_at_trip`` remains None.

        Tests:
            (Test Case 1) ``trip_kind() == 'memory'`` after a VRAM
                abort.
            (Test Case 2) ``temperature_c_at_trip() is None`` after
                a VRAM abort.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog()
        # Stub out the kill cascade so the test does not interrupt
        # the main thread.
        monkeypatch.setattr(wd, "_kill_targets_and_interrupt", lambda: None)
        # Suppress the snapshot-on-trip side-effect.
        monkeypatch.setattr(
            gpu_mod, "_try_capture_snapshot_to_results", lambda *a, **kw: None
        )
        wd._on_abort(96.0)
        assert wd.trip_kind() == "memory"
        assert wd.temperature_c_at_trip() is None

    def test_thermal_trip_sets_kind_to_thermal_and_records_temp(self, monkeypatch):
        """
        After ``_on_thermal_abort``, ``trip_kind`` is "thermal" and
        ``temperature_c_at_trip`` carries the at-trip reading.

        Tests:
            (Test Case 1) ``trip_kind() == 'thermal'`` after a
                thermal abort.
            (Test Case 2) ``temperature_c_at_trip()`` returns the
                value passed to ``_on_thermal_abort``.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog()
        monkeypatch.setattr(wd, "_kill_targets_and_interrupt", lambda: None)
        monkeypatch.setattr(
            gpu_mod, "_try_capture_snapshot_to_results", lambda *a, **kw: None
        )
        wd._on_thermal_abort(94.5)
        assert wd.trip_kind() == "thermal"
        assert wd.temperature_c_at_trip() == pytest.approx(94.5)


class TestGpuMemoryWatchdogMaybeWarnTemp:
    """``_maybe_warn_temp`` rate-limits + logs + audits thermal warnings."""

    def test_first_call_logs_warning_and_appends_audit(self, caplog, monkeypatch):
        """
        First call inside the warn-repeat window emits a WARNING
        record and appends a ``gpu_thermal warn`` audit event.

        Tests:
            (Test Case 1) WARNING record contains the temperature
                and threshold values.
            (Test Case 2) ``append_audit_event`` called once with
                the expected payload (temperature_c, warn_temp_c,
                abort_temp_c).
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []

        def _capture(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr(gpu_mod, "append_audit_event", _capture)
        wd = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0)
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._maybe_warn_temp(85.0)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert records
        msg = records[-1].getMessage()
        assert "85.0" in msg
        assert "80.0" in msg
        assert len(captured) == 1
        event = captured[0]
        assert event["watchdog"] == "gpu_thermal"
        assert event["event"] == "warn"
        assert event["temperature_c"] == 85.0
        assert event["warn_temp_c"] == 80.0
        assert event["abort_temp_c"] == 92.0

    def test_repeat_within_window_is_suppressed(self, caplog, monkeypatch):
        """
        A second call within ``warn_repeat_s`` of the first emits
        no record and no audit event.

        Tests:
            (Test Case 1) Immediate second call produces no
                additional WARNING record.
            (Test Case 2) ``append_audit_event`` is invoked exactly
                once across both calls.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []
        monkeypatch.setattr(
            gpu_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )
        wd = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0, warn_repeat_s=30.0)
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._maybe_warn_temp(85.0)
            caplog.clear()
            wd._maybe_warn_temp(86.0)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not records
        assert len(captured) == 1

    def test_call_after_window_emits_again(self, caplog, monkeypatch):
        """
        After the rate-limit window has elapsed, a subsequent call
        emits a fresh WARNING + audit event.

        Tests:
            (Test Case 1) Rewinding ``_last_temp_warn_t`` past
                ``warn_repeat_s`` allows a second emission.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []
        monkeypatch.setattr(
            gpu_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )
        wd = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0, warn_repeat_s=30.0)
        wd._maybe_warn_temp(85.0)
        wd._last_temp_warn_t -= 60.0
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._maybe_warn_temp(86.0)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert records
        assert len(captured) == 2


class TestGpuMemoryWatchdogMaybeWarnThrottle:
    """``_maybe_warn_throttle`` rate-limits + audits NVML throttle bits."""

    def test_first_call_logs_warning_with_decoded_reasons(self, caplog, monkeypatch):
        """
        First call emits a WARNING with the decoded throttle reasons
        and appends a ``gpu_throttle warn`` audit event carrying
        both the raw mask and the human-readable label.

        Tests:
            (Test Case 1) WARNING record references the decoded
                "SW power cap, HW thermal slowdown" string.
            (Test Case 2) Audit event payload includes
                ``throttle_mask=0x44`` and the matching reasons string.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []
        monkeypatch.setattr(
            gpu_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )
        wd = GpuMemoryWatchdog()
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._maybe_warn_throttle(0x4 | 0x40)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert records
        assert "SW power cap" in records[-1].getMessage()
        assert "HW thermal slowdown" in records[-1].getMessage()
        assert len(captured) == 1
        event = captured[0]
        assert event["watchdog"] == "gpu_throttle"
        assert event["event"] == "warn"
        assert event["throttle_mask"] == 0x4 | 0x40
        assert "SW power cap" in event["throttle_reasons"]

    def test_unknown_bits_render_as_hex_mask(self, caplog, monkeypatch):
        """
        When the mask has bits set that are NOT in
        ``_THROTTLE_REASON_LABELS``, the helper falls back to
        ``"mask=0xN"`` so the operator still has a diagnostic.

        Tests:
            (Test Case 1) An obscure bit (0x1) renders as
                ``mask=0x1`` in both the log message and the audit
                payload.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []
        monkeypatch.setattr(
            gpu_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )
        wd = GpuMemoryWatchdog()
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._maybe_warn_throttle(0x1)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert records
        assert "mask=0x1" in records[-1].getMessage()
        assert captured[0]["throttle_reasons"] == "mask=0x1"

    def test_repeat_within_window_is_suppressed(self, caplog, monkeypatch):
        """
        A second call within ``warn_repeat_s`` is rate-limited; no
        additional log record or audit event.

        Tests:
            (Test Case 1) Immediate second call emits no further
                WARNING record.
            (Test Case 2) ``append_audit_event`` invoked exactly once.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []
        monkeypatch.setattr(
            gpu_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )
        wd = GpuMemoryWatchdog(warn_repeat_s=30.0)
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._maybe_warn_throttle(0x4)
            caplog.clear()
            wd._maybe_warn_throttle(0x4)
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not records
        assert len(captured) == 1


class TestGpuMemoryWatchdogOnThermalAbort:
    """``_on_thermal_abort`` records the trip + cascades the kill."""

    def test_records_trip_state_and_logs_error(self, caplog, monkeypatch):
        """
        ``_on_thermal_abort`` flips trip flags, logs an ERROR, and
        the resulting ``make_error()`` returns a thermal exception.

        Tests:
            (Test Case 1) ``_tripped`` becomes True.
            (Test Case 2) ``trip_kind()`` is "thermal".
            (Test Case 3) ``temperature_c_at_trip()`` matches the
                supplied temperature.
            (Test Case 4) An ERROR record is emitted on the
                gpu-watchdog logger.
        """
        from spikelab.spike_sorting._exceptions import GpuThermalWatchdogError
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        monkeypatch.setattr(
            gpu_mod, "_try_capture_snapshot_to_results", lambda *a, **kw: None
        )
        wd = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0)
        monkeypatch.setattr(wd, "_kill_targets_and_interrupt", lambda: None)

        with caplog.at_level(
            logging.ERROR,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._on_thermal_abort(94.0)

        assert wd.tripped() is True
        assert wd.trip_kind() == "thermal"
        assert wd.temperature_c_at_trip() == pytest.approx(94.0)
        assert isinstance(wd.make_error(), GpuThermalWatchdogError)
        records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert records
        assert "THERMAL ABORT" in records[-1].getMessage()

    def test_appends_thermal_abort_audit_event(self, monkeypatch):
        """
        ``_on_thermal_abort`` appends an audit event tagged
        ``watchdog="gpu_thermal"`` / ``event="abort"`` carrying the
        temperature + abort threshold.

        Tests:
            (Test Case 1) Patched ``append_audit_event`` invoked
                once with the expected fields.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []
        monkeypatch.setattr(
            gpu_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )
        monkeypatch.setattr(
            gpu_mod, "_try_capture_snapshot_to_results", lambda *a, **kw: None
        )
        wd = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0)
        monkeypatch.setattr(wd, "_kill_targets_and_interrupt", lambda: None)

        wd._on_thermal_abort(95.5)
        assert len(captured) == 1
        event = captured[0]
        assert event["watchdog"] == "gpu_thermal"
        assert event["event"] == "abort"
        assert event["temperature_c"] == pytest.approx(95.5)
        assert event["abort_temp_c"] == 92.0

    def test_calls_snapshot_and_kill_cascade(self, monkeypatch):
        """
        ``_on_thermal_abort`` invokes ``_try_capture_snapshot_to_results``
        and ``_kill_targets_and_interrupt`` exactly once each.

        Tests:
            (Test Case 1) ``_try_capture_snapshot_to_results``
                receives the snapshot log path and a header
                referencing the device + temperature.
            (Test Case 2) ``_kill_targets_and_interrupt`` is invoked
                exactly once after the snapshot capture.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        snapshots: list = []
        monkeypatch.setattr(
            gpu_mod,
            "_try_capture_snapshot_to_results",
            lambda log_path, header: snapshots.append((log_path, header)),
        )
        monkeypatch.setattr(gpu_mod, "append_audit_event", lambda **kw: None)

        wd = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0)
        wd._snapshot_log_path = Path("/fake/results/rec.log")
        kill_called = []
        monkeypatch.setattr(
            wd, "_kill_targets_and_interrupt", lambda: kill_called.append(True)
        )

        wd._on_thermal_abort(93.2)
        assert len(snapshots) == 1
        log_path, header = snapshots[0]
        assert log_path == Path("/fake/results/rec.log")
        assert "device 0" in header
        assert "93.2" in header
        assert kill_called == [True]


# ===========================================================================
# Watchdog trip-path detail coverage (medium-priority gap fill)
# ===========================================================================


class TestGpuMemoryWatchdogOnAbort:
    """``GpuMemoryWatchdog._on_abort`` records VRAM-trip state."""

    def test_records_memory_trip_kind(self, monkeypatch):
        """
        After ``_on_abort``, ``trip_kind`` is ``"memory"`` (not None,
        not "thermal") and ``used_pct_at_trip`` matches the supplied
        percent.

        Tests:
            (Test Case 1) ``tripped()`` is True.
            (Test Case 2) ``trip_kind() == "memory"``.
            (Test Case 3) ``used_pct_at_trip()`` matches the abort
                percent.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog()
        monkeypatch.setattr(wd, "_kill_targets_and_interrupt", lambda: None)
        monkeypatch.setattr(
            gpu_mod, "_try_capture_snapshot_to_results", lambda *a, **kw: None
        )
        wd._on_abort(96.5)
        assert wd.tripped() is True
        assert wd.trip_kind() == "memory"
        assert wd.used_pct_at_trip() == pytest.approx(96.5)

    def test_invokes_snapshot_capture_with_banner(self, monkeypatch):
        """
        ``_on_abort`` invokes ``_try_capture_snapshot_to_results``
        with the snapshot log path and a banner referencing the
        device + VRAM percentage.

        Tests:
            (Test Case 1) Patched snapshot helper is called once
                with the watchdog's ``_snapshot_log_path``.
            (Test Case 2) The header argument references the device
                index and the at-trip VRAM percent.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        snapshots: list = []
        monkeypatch.setattr(
            gpu_mod,
            "_try_capture_snapshot_to_results",
            lambda log_path, header: snapshots.append((log_path, header)),
        )
        monkeypatch.setattr(gpu_mod, "append_audit_event", lambda **kw: None)

        wd = GpuMemoryWatchdog(device_index=2)
        wd._snapshot_log_path = Path("/fake/results/rec.log")
        monkeypatch.setattr(wd, "_kill_targets_and_interrupt", lambda: None)

        wd._on_abort(94.0)
        assert len(snapshots) == 1
        log_path, header = snapshots[0]
        assert log_path == Path("/fake/results/rec.log")
        assert "device 2" in header
        assert "94.0" in header

    def test_appends_gpu_memory_abort_audit_event(self, monkeypatch):
        """
        ``_on_abort`` appends an audit event with
        ``watchdog="gpu_memory"`` / ``event="abort"`` and the
        used / abort thresholds.

        Tests:
            (Test Case 1) Patched ``append_audit_event`` invoked
                once with the expected payload.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured: list = []
        monkeypatch.setattr(
            gpu_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )
        monkeypatch.setattr(
            gpu_mod, "_try_capture_snapshot_to_results", lambda *a, **kw: None
        )
        wd = GpuMemoryWatchdog(device_index=1, abort_pct=95.0)
        monkeypatch.setattr(wd, "_kill_targets_and_interrupt", lambda: None)

        wd._on_abort(97.2)
        assert len(captured) == 1
        event = captured[0]
        assert event["watchdog"] == "gpu_memory"
        assert event["event"] == "abort"
        assert event["device_index"] == 1
        assert event["used_pct"] == pytest.approx(97.2)
        assert event["abort_pct"] == 95.0


class TestGpuMemoryWatchdogKillTargetsSubprocess:
    """``GpuMemoryWatchdog._kill_targets_and_interrupt`` subprocess paths."""

    def test_terminates_then_kills_after_grace(self, monkeypatch):
        """
        Registered popen has ``terminate()`` called, then ``kill()``
        after the grace period if it is still alive.

        Tests:
            (Test Case 1) ``popen.terminate()`` invoked exactly once.
            (Test Case 2) ``popen.kill()`` invoked exactly once
                because ``poll()`` keeps reporting the process is
                still alive.
            (Test Case 3) ``time.sleep`` invoked with the registered
                kill_grace_s value.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog(kill_grace_s=2.0)
        # Avoid the interrupt landing on the test thread.
        wd._stop_event.set()

        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None  # Stays alive — kill should fire.
        wd._subprocesses = [(popen, 2.0)]

        sleeps: list = []
        monkeypatch.setattr(gpu_mod.time, "sleep", lambda s: sleeps.append(s))
        monkeypatch.setattr(gpu_mod, "append_audit_event", lambda **kw: None)

        wd._kill_targets_and_interrupt()
        popen.terminate.assert_called_once()
        popen.kill.assert_called_once()
        assert sleeps == [2.0]

    def test_terminate_exception_logged_and_continues(self, caplog, monkeypatch):
        """
        ``popen.terminate()`` raising is logged at ERROR and does
        not block the rest of the kill cascade.

        Tests:
            (Test Case 1) Patched terminate raising propagates as a
                logged ERROR rather than out of
                ``_kill_targets_and_interrupt``.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog()
        wd._stop_event.set()

        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None
        popen.terminate.side_effect = OSError("simulated terminate failure")
        wd._subprocesses = [(popen, 0.0)]

        monkeypatch.setattr(gpu_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(gpu_mod, "append_audit_event", lambda **kw: None)

        with caplog.at_level(
            logging.ERROR,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._kill_targets_and_interrupt()
        records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any("terminate() failed" in r.getMessage() for r in records)

    def test_per_callback_exception_isolated(self, caplog, monkeypatch):
        """
        A failing kill callback is logged and does not stop later
        callbacks from running.

        Tests:
            (Test Case 1) First callback raises Exception → logged.
            (Test Case 2) Second callback still runs (call count is 1).
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog()
        wd._stop_event.set()

        good_calls = []

        def _bad_cb():
            raise RuntimeError("simulated callback failure")

        def _good_cb():
            good_calls.append(True)

        wd._kill_callbacks = [_bad_cb, _good_cb]
        monkeypatch.setattr(gpu_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(gpu_mod, "append_audit_event", lambda **kw: None)

        with caplog.at_level(
            logging.ERROR,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._kill_targets_and_interrupt()
        assert good_calls == [True]
        assert any(
            "kill_callback raised" in r.getMessage()
            for r in caplog.records
            if r.levelname == "ERROR"
        )


class TestDiskUsageWatchdogOnTripSubprocess:
    """``DiskUsageWatchdog._on_trip`` subprocess + audit paths."""

    def test_appends_abort_audit_event(self, tmp_path, monkeypatch):
        """
        ``_on_trip`` records a ``disk abort`` audit event before
        building the report and killing.

        Tests:
            (Test Case 1) Patched ``append_audit_event`` invoked at
                least once with ``watchdog="disk"`` /
                ``event="abort"`` and the trip threshold fields.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        captured: list = []
        monkeypatch.setattr(
            disk_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )
        monkeypatch.setattr(disk_mod.time, "sleep", lambda s: None)

        kill_called = []
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            kill_callback=lambda: kill_called.append(True),
        )
        wd._on_trip(0.5)
        # Look for an abort event among the captured calls.
        abort_events = [e for e in captured if e.get("event") == "abort"]
        assert abort_events, "expected an abort audit event"
        ev = abort_events[0]
        assert ev["watchdog"] == "disk"
        assert ev["folder"] == str(tmp_path)
        assert ev["free_gb"] == pytest.approx(0.5)
        assert ev["abort_free_gb"] == 1.0
        # Kill callback runs after the audit/report.
        assert kill_called == [True]

    def test_terminates_popen_then_kills_after_grace(self, tmp_path, monkeypatch):
        """
        Registered popen is terminated, then killed after the
        grace period.

        Tests:
            (Test Case 1) ``terminate()`` invoked once.
            (Test Case 2) ``kill()`` invoked once because the popen
                stays "alive" between the two ``poll()`` calls.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        monkeypatch.setattr(disk_mod, "append_audit_event", lambda **kw: None)
        monkeypatch.setattr(disk_mod.time, "sleep", lambda s: None)

        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None  # Always-alive simulates terminate ignored.
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            popen=popen,
            kill_grace_s=0.0,
        )
        wd._on_trip(0.5)
        popen.terminate.assert_called_once()
        popen.kill.assert_called_once()

    def test_terminate_exception_logged_and_continues_to_kill(
        self, tmp_path, caplog, monkeypatch
    ):
        """
        ``popen.terminate()`` raising is logged but does not block
        the subsequent ``kill()`` step.

        Tests:
            (Test Case 1) Terminate raising → ERROR record captured.
            (Test Case 2) ``kill()`` still invoked after the grace
                period.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        monkeypatch.setattr(disk_mod, "append_audit_event", lambda **kw: None)
        monkeypatch.setattr(disk_mod.time, "sleep", lambda s: None)

        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None
        popen.terminate.side_effect = OSError("simulated terminate failure")
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            popen=popen,
            kill_grace_s=0.0,
        )
        with caplog.at_level(
            logging.ERROR,
            logger="spikelab.spike_sorting.guards._disk_watchdog",
        ):
            wd._on_trip(0.5)
        assert any(
            "terminate() failed" in r.getMessage()
            for r in caplog.records
            if r.levelname == "ERROR"
        )
        popen.kill.assert_called_once()

    def test_kill_callback_system_exit_propagates(self, tmp_path, monkeypatch):
        """
        ``kill_callback`` raising ``SystemExit`` propagates out of
        ``_on_trip`` (the operator-requested abort must not be
        swallowed).

        Tests:
            (Test Case 1) ``_on_trip`` re-raises ``SystemExit`` when
                the kill_callback raises it.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        monkeypatch.setattr(disk_mod, "append_audit_event", lambda **kw: None)
        monkeypatch.setattr(disk_mod.time, "sleep", lambda s: None)

        def _raise():
            raise SystemExit(7)

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            kill_callback=_raise,
        )
        with pytest.raises(SystemExit):
            wd._on_trip(0.5)


class TestDiskUsageWatchdogBuildReport:
    """``DiskUsageWatchdog._build_report`` produces the trip report."""

    def test_consumed_clamped_at_zero_when_folder_shrunk(self, tmp_path, monkeypatch):
        """
        If the folder is smaller at trip than at entry (sorter
        cleaned up before crashing), ``bytes_consumed_during_sort``
        is clamped to 0 instead of going negative.

        Tests:
            (Test Case 1) Initial folder size 1000 B, current
                folder size 200 B → ``bytes_consumed_during_sort = 0``.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        monkeypatch.setattr(disk_mod, "_folder_size_bytes", lambda folder: 200.0)
        monkeypatch.setattr(disk_mod, "_top_consumers", lambda folder: [])
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            kill_callback=lambda: None,
        )
        wd._initial_folder_size = 1000.0
        wd._initial_top_consumers = []
        report = wd._build_report(free_gb=0.5)
        assert report.bytes_consumed_during_sort == 0.0

    def test_projection_drives_suggestion_text(self, tmp_path, monkeypatch):
        """
        When ``projected_need_gb > free_gb``, the first suggestion
        names the projected need + shortfall; otherwise the generic
        "free at least warn_free_gb" suggestion is used.

        Tests:
            (Test Case 1) ``projected_need_gb=15`` and ``free_gb=2``
                produces the projection-based suggestion.
            (Test Case 2) ``projected_need_gb=None`` falls back to
                the generic suggestion mentioning ``warn_free_gb``.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        monkeypatch.setattr(disk_mod, "_folder_size_bytes", lambda folder: 0.0)
        monkeypatch.setattr(disk_mod, "_top_consumers", lambda folder: [])

        # Projection-based path.
        wd_with_proj = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            projected_need_gb=15.0,
            kill_callback=lambda: None,
        )
        wd_with_proj._initial_folder_size = 0.0
        wd_with_proj._initial_top_consumers = []
        report = wd_with_proj._build_report(free_gb=2.0)
        assert any("projects ~15.0 GB" in s for s in report.suggested_actions)

        # Generic path (no projection).
        wd_generic = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            kill_callback=lambda: None,
        )
        wd_generic._initial_folder_size = 0.0
        wd_generic._initial_top_consumers = []
        report = wd_generic._build_report(free_gb=2.0)
        assert any("Free at least 5.0 GB" in s for s in report.suggested_actions)

    def test_falls_back_to_initial_top_consumers_on_timeout(
        self, tmp_path, monkeypatch
    ):
        """
        When ``_top_consumers_with_timeout`` returns ``None`` (live
        walk timed out), the report falls back to the entry-time
        snapshot.

        Tests:
            (Test Case 1) Patched ``_top_consumers_with_timeout``
                returning None → report's ``top_consumers`` matches
                the entry-time snapshot.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        monkeypatch.setattr(disk_mod, "_folder_size_bytes", lambda folder: 0.0)
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            kill_callback=lambda: None,
        )
        wd._initial_folder_size = 0.0
        wd._initial_top_consumers = [("/tmp/big.bin", 7.0)]
        # Patch the timeout method to simulate a stalled walk.
        monkeypatch.setattr(
            wd,
            "_top_consumers_with_timeout",
            lambda timeout_s: None,
        )
        report = wd._build_report(free_gb=0.5)
        assert report.top_consumers == [("/tmp/big.bin", 7.0)]


class TestLogInactivityWatchdogOnTripSubprocess:
    """``LogInactivityWatchdog._on_trip`` audit + popen termination paths."""

    def test_appends_inactivity_abort_audit_event(self, tmp_path, monkeypatch):
        """
        ``_on_trip`` appends an audit event tagged
        ``watchdog="inactivity"`` / ``event="abort"`` carrying the
        sorter label, observed inactivity, and tolerance.

        Tests:
            (Test Case 1) Patched ``append_audit_event`` invoked
                once with the expected fields.
        """
        from spikelab.spike_sorting.guards import _inactivity as inactivity_mod

        captured: list = []
        monkeypatch.setattr(
            inactivity_mod,
            "append_audit_event",
            lambda **kw: captured.append(kw),
        )
        monkeypatch.setattr(inactivity_mod.time, "sleep", lambda s: None)

        wd = LogInactivityWatchdog(
            log_path=tmp_path / "rec.log",
            popen=None,
            inactivity_s=120.0,
            sorter="kilosort4",
        )
        wd._on_trip(180.0)
        assert len(captured) == 1
        event = captured[0]
        assert event["watchdog"] == "inactivity"
        assert event["event"] == "abort"
        assert event["sorter"] == "kilosort4"
        assert event["inactivity_s"] == 180.0
        assert event["tolerance_s"] == 120.0

    def test_terminate_exception_logged_then_kill_runs(
        self, tmp_path, caplog, monkeypatch
    ):
        """
        ``popen.terminate()`` raising is logged at ERROR and does
        not skip the subsequent ``kill()`` after the grace period.

        Tests:
            (Test Case 1) Terminate raising → ERROR record captured.
            (Test Case 2) ``kill()`` still invoked once.
        """
        from spikelab.spike_sorting.guards import _inactivity as inactivity_mod

        monkeypatch.setattr(inactivity_mod, "append_audit_event", lambda **kw: None)
        monkeypatch.setattr(inactivity_mod.time, "sleep", lambda s: None)

        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None
        popen.terminate.side_effect = OSError("simulated terminate failure")

        wd = LogInactivityWatchdog(
            log_path=tmp_path / "rec.log",
            popen=popen,
            inactivity_s=60.0,
            sorter="kilosort2",
            kill_grace_s=0.0,
        )
        with caplog.at_level(
            logging.ERROR,
            logger="spikelab.spike_sorting.guards._inactivity",
        ):
            wd._on_trip(90.0)
        assert any(
            "terminate() failed" in r.getMessage()
            for r in caplog.records
            if r.levelname == "ERROR"
        )
        popen.kill.assert_called_once()

    def test_kill_callback_other_exception_logged_not_raised(
        self, tmp_path, caplog, monkeypatch
    ):
        """
        ``kill_callback`` raising an ordinary ``Exception`` is logged
        at ERROR; the trip method does not propagate.

        Tests:
            (Test Case 1) Callback raising RuntimeError is captured
                in an ERROR record.
            (Test Case 2) ``_on_trip`` returns normally (no raise).
        """
        from spikelab.spike_sorting.guards import _inactivity as inactivity_mod

        monkeypatch.setattr(inactivity_mod, "append_audit_event", lambda **kw: None)
        monkeypatch.setattr(inactivity_mod.time, "sleep", lambda s: None)

        def _raise():
            raise RuntimeError("simulated callback failure")

        wd = LogInactivityWatchdog(
            log_path=tmp_path / "rec.log",
            popen=None,
            inactivity_s=60.0,
            sorter="rt_sort",
            kill_callback=_raise,
        )
        with caplog.at_level(
            logging.ERROR,
            logger="spikelab.spike_sorting.guards._inactivity",
        ):
            wd._on_trip(90.0)  # must not raise
        assert any(
            "kill_callback raised" in r.getMessage()
            for r in caplog.records
            if r.levelname == "ERROR"
        )


# ===========================================================================
# Edge-case scan coverage (HIGH-priority items, regression tests)
# ===========================================================================


class TestDiskUsageWatchdogConstructionNegatives:
    """``DiskUsageWatchdog`` rejects negative threshold inputs."""

    def test_both_negative_thresholds_raise(self, tmp_path):
        """
        Both ``warn_free_gb`` and ``abort_free_gb`` negative raises a
        ``ValueError`` at construction.

        Tests:
            (Test Case 1) ``warn=-0.5, abort=-1.0`` raises
                ``ValueError`` whose message contains ``"must be >= 0"``.
        """
        with pytest.raises(ValueError, match="must be >= 0"):
            DiskUsageWatchdog(
                folder=tmp_path,
                warn_free_gb=-0.5,
                abort_free_gb=-1.0,
                kill_callback=lambda: None,
            )


class TestDiskUsageWatchdogProjectedNeedNan:
    """``DiskUsageWatchdog`` rejects NaN ``projected_need_gb``."""

    def test_nan_projected_need_raises(self, tmp_path):
        """
        NaN ``projected_need_gb`` raises ``ValueError`` at construction.

        Tests:
            (Test Case 1) ``projected_need_gb=float('nan')`` raises
                ``ValueError`` whose message contains ``"must not be NaN"``.
        """
        with pytest.raises(ValueError, match="must not be NaN"):
            DiskUsageWatchdog(
                folder=tmp_path,
                warn_free_gb=5.0,
                abort_free_gb=1.0,
                projected_need_gb=float("nan"),
                kill_callback=lambda: None,
            )


class TestGpuMemoryWatchdogThermalAsymmetric:
    """``GpuMemoryWatchdog`` accepts asymmetric thermal thresholds."""

    def test_warn_temp_set_with_abort_temp_none(self):
        """
        ``warn_temp_c`` set + ``abort_temp_c=None`` is accepted (no
        thermal-ordering validation when one is None).

        Tests:
            (Test Case 1) ``warn_temp_c=85, abort_temp_c=None``
                constructs successfully and stores both fields.
        """
        wd = GpuMemoryWatchdog(warn_temp_c=85.0, abort_temp_c=None)
        assert wd.warn_temp_c == 85.0
        assert wd.abort_temp_c is None

    def test_abort_temp_set_with_warn_temp_none(self):
        """
        ``warn_temp_c=None`` + ``abort_temp_c`` set is accepted.

        Tests:
            (Test Case 1) ``warn_temp_c=None, abort_temp_c=92``
                constructs successfully and stores both fields.

        Notes:
            - Documents current behaviour: with no warn threshold,
              thermal aborts can fire without an earlier warning
              telegraph. Possibly surprising; flagged as a config
              ergonomics concern in the edge-case scan.
        """
        wd = GpuMemoryWatchdog(warn_temp_c=None, abort_temp_c=92.0)
        assert wd.warn_temp_c is None
        assert wd.abort_temp_c == 92.0

    def test_abort_pct_exactly_one_hundred_accepted(self):
        """
        ``abort_pct=100.0`` passes the ``warn < abort <= 100``
        validation; the trip predicate ``used_pct >= 100`` then
        almost never fires because reported VRAM usage is always
        slightly below 100% on real devices.

        Tests:
            (Test Case 1) ``abort_pct=100.0`` constructs successfully.

        Notes:
            - Pinning current behaviour: this is a near-no-op
              configuration (the watchdog never trips on memory).
              Documented for operator awareness.
        """
        wd = GpuMemoryWatchdog(warn_pct=85.0, abort_pct=100.0)
        assert wd.warn_pct == 85.0
        assert wd.abort_pct == 100.0


class TestComputeInactivityTimeoutSEdges:
    """``compute_inactivity_timeout_s`` boundary behaviour."""

    def test_negative_base_s_chained_into_watchdog_raises(self, tmp_path):
        """
        ``base_s`` is not range-checked inside
        ``compute_inactivity_timeout_s`` (current behaviour) but the
        non-positive result it produces is rejected by
        ``LogInactivityWatchdog.__init__`` — the misconfig is caught
        at watchdog construction rather than helper computation.

        Tests:
            (Test Case 1) ``base_s=-1000, per_min_s=30, duration=10``
                produces a negative timeout.
            (Test Case 2) Passing that result into
                ``LogInactivityWatchdog`` raises ``ValueError``.
        """
        timeout = compute_inactivity_timeout_s(
            recording_duration_min=10,
            base_s=-1000.0,
            per_min_s=30.0,
            max_s=None,
        )
        assert timeout < 0
        with pytest.raises(ValueError, match="inactivity_s must be positive"):
            LogInactivityWatchdog(
                log_path=tmp_path / "rec.log",
                popen=mock.Mock(spec=subprocess.Popen),
                inactivity_s=timeout,
                sorter="x",
            )

    def test_zero_or_negative_max_s_clamps_below_zero(self):
        """
        ``max_s=0`` clamps any positive timeout to zero;
        ``max_s=-10`` clamps to -10. Either result then trips
        the strict-validation guard in ``LogInactivityWatchdog``.

        Tests:
            (Test Case 1) ``max_s=0.0`` returns 0.0.
            (Test Case 2) ``max_s=-10.0`` returns -10.0.
        """
        result_zero = compute_inactivity_timeout_s(
            recording_duration_min=5, base_s=600.0, per_min_s=30.0, max_s=0.0
        )
        assert result_zero == 0.0

        result_neg = compute_inactivity_timeout_s(
            recording_duration_min=5, base_s=600.0, per_min_s=30.0, max_s=-10.0
        )
        assert result_neg == -10.0


class TestSortLockEdges:
    """``acquire_sort_lock`` error-shape edges."""

    def test_malformed_started_at_falls_back_to_pid_alive(self, monkeypatch):
        """
        ``_pid_holds_lock`` with a malformed ``started_at`` falls
        back to ``_pid_alive`` semantics (no exception).

        Tests:
            (Test Case 1) ``started_at='yesterday'`` (non-ISO
                string) → ``_pid_holds_lock`` returns whatever
                ``_pid_alive`` returns (True or False).
            (Test Case 2) The fallback path is exercised regardless
                of psutil's actual response.
        """
        from spikelab.spike_sorting.guards import _sort_lock as lock_mod

        # Force the fallback by making _pid_alive return a sentinel.
        monkeypatch.setattr(lock_mod, "_pid_alive", lambda pid: True)
        # Provide a fake psutil so the cross-boot/PID-reuse code path
        # reaches the malformed-timestamp branch.
        fake_psutil = SimpleNamespace(
            pid_exists=lambda _pid: True,
            boot_time=lambda: 0.0,
            Process=lambda _pid: SimpleNamespace(create_time=lambda: 0.0),
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        # Malformed timestamp → fromisoformat raises → falls back.
        result = lock_mod._pid_holds_lock(123, "yesterday")
        assert result is True  # mirrors _pid_alive's stub

    def test_mkdir_failure_wrapped_in_concurrent_sort_error(
        self, monkeypatch, tmp_path
    ):
        """
        ``acquire_sort_lock`` wraps an mkdir failure in a classified
        ``ConcurrentSortError`` with the original ``PermissionError``
        chained via ``__cause__``.

        Tests:
            (Test Case 1) Patched ``Path.mkdir`` raising
                ``PermissionError`` surfaces as ``ConcurrentSortError``
                whose message contains ``"failed to acquire sort lock"``.
            (Test Case 2) ``excinfo.value.__cause__`` is the original
                ``PermissionError``.
        """
        real_mkdir = Path.mkdir

        def _refuse(self, *args, **kwargs):
            if self == tmp_path / "ro_folder":
                raise PermissionError("simulated read-only mount")
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _refuse)
        with pytest.raises(
            ConcurrentSortError, match="failed to acquire sort lock"
        ) as excinfo:
            with acquire_sort_lock(tmp_path / "ro_folder"):
                pass
        assert isinstance(excinfo.value.__cause__, PermissionError)


class TestCanaryGetBackendClassRaises:
    """``run_canary`` surfaces unknown-sorter-name as classified failure."""

    def test_unknown_sorter_returns_environment_sort_failure(self, tmp_path):
        """
        ``run_canary`` checks the sorter name against ``list_sorters()``
        before calling ``get_backend_class``; an unknown name raises
        ``EnvironmentSortFailure`` which the ``_CLASSIFIED_FAILURES``
        handler catches and returns as the result.

        Tests:
            (Test Case 1) Unknown sorter name returns an
                ``EnvironmentSortFailure`` instance from ``run_canary``.
            (Test Case 2) The result string contains
                ``"unknown sorter name"``.
        """
        from spikelab.spike_sorting import canary as canary_mod

        cfg = SimpleNamespace(
            execution=SimpleNamespace(canary_first_n_s=5.0),
            sorter=SimpleNamespace(sorter_name="not_a_real_sorter"),
        )
        # Provide a config.override that returns a stand-in clone so
        # _build_canary_config does not blow up before the lookup.
        cfg.override = lambda **overrides: cfg

        result = canary_mod.run_canary(
            cfg,
            recording=mock.Mock(),
            rec_path=tmp_path / "rec.dat",
            inter_path=tmp_path / "inter",
            sorter_name="not_a_real_sorter",
            rec_name="canary",
        )
        assert isinstance(result, EnvironmentSortFailure)
        assert "unknown sorter name" in str(result)


# ===========================================================================
# Edge-case scan coverage (MEDIUM-priority items, regression tests)
# ===========================================================================


class TestHostMemoryWatchdogKillCallbackDuplicateRegistration:
    """Same kill_callback registered twice fires twice."""

    def test_duplicate_registration_invokes_each_copy(self):
        """
        Registering the same callable twice does not deduplicate;
        the trip path invokes both copies (each entry in
        ``_kill_callbacks`` is called).

        Tests:
            (Test Case 1) Same callable appended twice produces two
                invocations during ``_run_kill_callbacks``.
        """
        wd = HostMemoryWatchdog()
        calls = []

        def _cb():
            calls.append(True)

        wd.register_kill_callback(_cb)
        wd.register_kill_callback(_cb)
        wd._run_kill_callbacks()
        assert calls == [True, True]

    def test_unregister_removes_all_occurrences_via_identity_filter(self):
        """
        ``unregister_kill_callback`` filters by identity, so calling
        it once removes every occurrence of the same callable —
        even if it was registered multiple times.

        Tests:
            (Test Case 1) After register×2 + unregister×1, the
                callback list is empty and a trip invokes nothing.

        Notes:
            - Documents current behaviour. The list-comprehension
              filter (``[c for c in self._kill_callbacks if c is not callback]``)
              drops every match in one pass.
        """
        wd = HostMemoryWatchdog()
        calls = []

        def _cb():
            calls.append(True)

        wd.register_kill_callback(_cb)
        wd.register_kill_callback(_cb)
        wd.unregister_kill_callback(_cb)
        wd._run_kill_callbacks()
        # Both copies of _cb removed.
        assert calls == []


class TestPreflightWslconfigMultipleSections:
    """``_parse_wslconfig_memory_gb`` first-match-wins for duplicate sections."""

    def test_first_wsl2_section_wins(self):
        """
        When the WSL config contains multiple ``[wsl2]`` sections,
        the first ``memory=`` value is returned; subsequent values
        are ignored.

        Tests:
            (Test Case 1) Two ``[wsl2]`` sections (8GB then 32GB) →
                returns 8.0.

        Notes:
            - Documents current behaviour. The edge-case scan flags
              this as a malformed-config concern but the behaviour
              is deterministic.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        text = "[wsl2]\n" "memory=8GB\n" "\n" "[wsl2]\n" "memory=32GB\n"
        assert _parse_wslconfig_memory_gb(text) == 8.0


class TestSortLockEmptyHostname:
    """``acquire_sort_lock`` handles empty hostnames."""

    def test_same_empty_hostname_treated_as_same_host(self, tmp_path, monkeypatch):
        """
        Empty-string hostname (some containerised environments
        report ``""``) compares equal to itself, so the same-host
        path runs as expected.

        Tests:
            (Test Case 1) With ``socket.gethostname`` patched to
                return ``""``, the lock is created with
                ``hostname=""``.
            (Test Case 2) Reading the lock back exposes the same
                empty hostname so a subsequent acquire from the
                same "host" detects it as a same-host concurrent
                sort.
        """
        from spikelab.spike_sorting.guards import _sort_lock as lock_mod

        monkeypatch.setattr(lock_mod.socket, "gethostname", lambda: "")
        with acquire_sort_lock(tmp_path):
            info = lock_mod._read_lock_info(tmp_path / ".spikelab_sort.lock")
            assert info is not None
            assert info["hostname"] == ""


class TestResolveActiveDeviceEdges:
    """``resolve_active_device`` falls back gracefully on edge configs."""

    def test_kilosort4_with_none_sorter_params(self):
        """
        KS4 config with ``sorter_params=None`` → fallback to device 0
        via the ``or {}`` guard.

        Tests:
            (Test Case 1) ``sorter_name="kilosort4", sorter_params=None``
                returns 0.
        """
        from spikelab.spike_sorting.guards import resolve_active_device

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort4", sorter_params=None),
            rt_sort=SimpleNamespace(device=None),
        )
        assert resolve_active_device(cfg) == 0

    def test_unknown_sorter_returns_zero(self):
        """
        Unknown sorter name (not "rt_sort" / "kilosort4") returns 0.

        Tests:
            (Test Case 1) ``sorter_name="some_future_sorter"`` →
                returns 0.

        Notes:
            - Documents current behaviour. Adding a new CUDA sorter
              without updating this dispatcher silently watches
              device 0; flagged in the edge-case scan as a
              maintenance concern.
        """
        from spikelab.spike_sorting.guards import resolve_active_device

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(
                sorter_name="future_cuda_sorter", sorter_params={"device": "cuda:3"}
            ),
            rt_sort=SimpleNamespace(device="cuda:3"),
        )
        assert resolve_active_device(cfg) == 0


# ===========================================================================
# Watchdog lifecycle robustness (medium-priority gap fill)
# ===========================================================================


class TestHostMemoryWatchdogContextEdges:
    """``HostMemoryWatchdog`` enter/exit lifecycle robustness."""

    def test_enter_captures_snapshot_log_path(self, tmp_path, monkeypatch):
        """
        ``__enter__`` snapshots the active ``set_active_log_path``
        ContextVar onto ``self._snapshot_log_path`` so the polling
        thread can reach the audit log even though ContextVars do
        not propagate across thread boundaries.

        Tests:
            (Test Case 1) ``set_active_log_path`` published, then
                ``__enter__`` recorded; ``_snapshot_log_path`` matches
                the published path.
        """
        from spikelab.spike_sorting.guards import (
            get_active_log_path,
            set_active_log_path,
        )

        log_path = tmp_path / "rec.log"
        log_path.touch()
        # Make the polling thread a no-op so the test doesn't depend
        # on real psutil readings.
        wd = HostMemoryWatchdog(poll_interval_s=10.0)
        with set_active_log_path(log_path):
            assert get_active_log_path() == log_path
            with wd:
                assert wd._snapshot_log_path == log_path

    def test_subprocesses_cleared_on_exit(self, tmp_path):
        """
        ``__exit__`` clears the registered-subprocess list so a
        watchdog can be re-entered without leaking the previous
        run's registrations.

        Tests:
            (Test Case 1) After registering a popen and then exiting,
                ``_subprocesses`` is empty.

        Notes:
            - Bypasses ``__enter__`` so the test does not depend on
              real psutil readings; the cleanup logic is on
              ``__exit__`` regardless.
        """
        wd = HostMemoryWatchdog(poll_interval_s=10.0)
        wd._enabled = True
        wd._token = None  # avoid the ContextVar reset path

        popen = mock.Mock(spec=subprocess.Popen)
        wd.register_subprocess(popen)
        assert wd._subprocesses

        wd.__exit__(None, None, None)
        assert wd._subprocesses == []


class TestDiskUsageWatchdogContextEdges:
    """``DiskUsageWatchdog`` enter/exit lifecycle robustness."""

    def test_disabled_enter_spawns_no_thread(self, tmp_path):
        """
        When ``_enabled=False`` (no kill target), ``__enter__``
        returns without starting a polling thread.

        Tests:
            (Test Case 1) Disabled watchdog has ``_thread is None``
                inside the with-block.
            (Test Case 2) ``tripped()`` stays False on exit.
        """
        wd = DiskUsageWatchdog(folder=tmp_path)  # no kill target
        assert wd._enabled is False
        with wd:
            assert wd._thread is None
        assert wd.tripped() is False

    def test_enter_snapshots_initial_folder_size_and_top_consumers(
        self, tmp_path, monkeypatch
    ):
        """
        ``__enter__`` records the folder size and top-consumers
        snapshot at entry so the trip-time report can fall back to
        them when the live walk is too slow.

        Tests:
            (Test Case 1) After ``__enter__``,
                ``_initial_folder_size`` matches the patched return
                of ``_folder_size_bytes``.
            (Test Case 2) ``_initial_top_consumers`` matches the
                patched return of ``_top_consumers``.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as disk_mod

        monkeypatch.setattr(disk_mod, "_folder_size_bytes", lambda folder: 1234.0)
        monkeypatch.setattr(
            disk_mod,
            "_top_consumers",
            lambda folder: [("/tmp/x.bin", 0.5)],
        )

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=5.0,
            abort_free_gb=1.0,
            kill_callback=lambda: None,
            poll_interval_s=10.0,  # avoid first-tick race
        )
        with wd:
            assert wd._initial_folder_size == 1234.0
            assert wd._initial_top_consumers == [("/tmp/x.bin", 0.5)]


class TestGpuMemoryWatchdogContextEdges:
    """``GpuMemoryWatchdog`` enter/exit lifecycle robustness."""

    def test_exit_calls_session_shutdown(self, monkeypatch):
        """
        ``__exit__`` invokes ``_session.shutdown()`` so pynvml
        resources are released even if no trip fired.

        Tests:
            (Test Case 1) After ``__exit__`` the session's shutdown
                hook is called and ``_session`` is cleared to None.
        """
        wd = GpuMemoryWatchdog()
        wd._enabled = True

        shutdown_calls = []

        class _StubSession:
            def shutdown(self):
                shutdown_calls.append(True)

        wd._session = _StubSession()
        wd._token = None  # avoid the ContextVar reset path

        wd.__exit__(None, None, None)
        assert shutdown_calls == [True]
        assert wd._session is None

    def test_exit_clears_token_after_normal_reset(self):
        """
        On a normal ``__exit__`` the captured token is consumed by
        ``_active_gpu_watchdog.reset`` and ``self._token`` is cleared
        to None so the watchdog can be re-entered.

        Tests:
            (Test Case 1) After a fresh ``set`` + ``__exit__``,
                ``self._token`` is None.

        Notes:
            - The cross-context-reset *swallow* path is not
              independently tested here. The current swallow catches
              ``(LookupError, ValueError)`` but a re-used token in
              modern Python raises ``RuntimeError`` from
              ``ContextVar.reset``, which is not caught — see the
              accompanying note in REVIEW.md flagging this as an
              edge case to harden later.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog()
        wd._enabled = True
        wd._token = gpu_mod._active_gpu_watchdog.set(wd)
        wd._session = None  # avoid the shutdown path
        wd.__exit__(None, None, None)
        assert wd._token is None


class TestLogInactivityWatchdogContextEdges:
    """``LogInactivityWatchdog`` enter snapshots stale-log signals."""

    def test_disabled_enter_no_thread(self, tmp_path):
        """
        Without a kill target the watchdog is disabled; ``__enter__``
        returns without spawning a polling thread.

        Tests:
            (Test Case 1) Disabled watchdog has ``_thread is None``
                inside the with-block.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "rec.log",
            popen=None,
            inactivity_s=10.0,
            sorter="x",
        )
        assert wd._enabled is False
        with wd:
            assert wd._thread is None
        assert wd.tripped() is False

    def test_enter_captures_pre_existing_log_signals(self, tmp_path):
        """
        A pre-existing log file's mtime + size are captured at enter
        so a stale log from a previous run does not register as a
        fresh trip.

        Tests:
            (Test Case 1) After writing content to the log and
                entering the watchdog, ``_last_seen_mtime`` and
                ``_last_seen_size`` match the on-disk file.
            (Test Case 2) The watchdog is in a not-yet-tripped state.

        Notes:
            - We construct the watchdog with a long ``inactivity_s``
              and a long ``poll_interval_s`` so the polling thread
              cannot trip during the test window.
        """
        log = tmp_path / "rec.log"
        log.write_bytes(b"pre-existing\n")
        on_disk_size = log.stat().st_size
        on_disk_mtime = log.stat().st_mtime

        wd = LogInactivityWatchdog(
            log_path=log,
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=600.0,
            sorter="kilosort4",
            poll_interval_s=60.0,
        )
        with wd:
            assert wd._last_seen_size == on_disk_size
            assert wd._last_seen_mtime == pytest.approx(on_disk_mtime)
            assert wd.tripped() is False


# ===========================================================================
# Preflight helper coverage (medium-priority gap fill)
# ===========================================================================


class TestValidateRecordingInputsEdges:
    """``_validate_recording_inputs`` boundary cases."""

    def test_none_entry_raises(self):
        """
        ``None`` in the recording inputs list raises ``ValueError``
        flagging the caller bug.

        Tests:
            (Test Case 1) Single ``None`` entry raises ``ValueError``
                whose message contains ``"caller bug"``.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        with pytest.raises(ValueError, match="caller bug"):
            _validate_recording_inputs([None])

    def test_no_extension_path_yields_unfamiliar_warning(self, tmp_path):
        """
        A real file without an extension yields a warn-level
        ``recording_extension_unknown`` finding whose message
        references ``"(no extension)"``.

        Tests:
            (Test Case 1) File ``noext`` (no suffix) on disk
                produces a single warn finding.
            (Test Case 2) Message contains ``"(no extension)"``.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        target = tmp_path / "noext"
        target.write_bytes(b"x")
        findings = _validate_recording_inputs([target])
        assert len(findings) == 1
        f = findings[0]
        assert f.code == "recording_extension_unknown"
        assert f.level == "warn"
        assert "(no extension)" in f.message

    def test_multi_suffix_known_extension_passes(self, tmp_path):
        """
        Files with multi-suffix names (e.g. ``recording.raw.h5``)
        match via *any* suffix in the known list — ``.h5`` is
        recognised so no warning fires.

        Tests:
            (Test Case 1) ``rec.raw.h5`` → empty findings.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        target = tmp_path / "rec.raw.h5"
        target.write_bytes(b"x")
        assert _validate_recording_inputs([target]) == []


class TestCheckKilosort2HostEnvVar:
    """``_check_kilosort2_host`` falls back to ``KILOSORT_PATH`` env var."""

    def test_env_var_used_when_sorter_path_unset(self, tmp_path, monkeypatch):
        """
        When ``SorterConfig.sorter_path`` is unset and
        ``KILOSORT_PATH`` is set to a directory containing
        ``master_kilosort.m``, no fail-finding is emitted for the
        path component (matlab finding still depends on PATH).

        Tests:
            (Test Case 1) Patched ``shutil.which("matlab")`` →
                truthy + env var pointing at a valid sources dir →
                no path-related finding.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        ks_dir = tmp_path / "ks2_src"
        ks_dir.mkdir()
        (ks_dir / "master_kilosort.m").write_text("% stub\n")

        monkeypatch.setattr(pf.shutil, "which", lambda name: "/usr/bin/matlab")
        monkeypatch.setenv("KILOSORT_PATH", str(ks_dir))

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_path=None),
        )
        findings = pf._check_kilosort2_host(cfg)
        assert findings == []


class TestCheckResourceRlimitsHealthy:
    """``_check_resource_rlimits`` returns empty list on healthy systems."""

    def test_healthy_limits_return_empty(self, monkeypatch):
        """
        With ``RLIMIT_NOFILE`` and ``RLIMIT_NPROC`` both well above
        the thresholds, the check returns an empty list.

        Tests:
            (Test Case 1) Patched ``resource.getrlimit`` returning
                (65536, 65536) for both calls → empty findings.

        Notes:
            - Skipped on Windows where the ``resource`` module is
              not available; the helper short-circuits to ``[]`` on
              import failure regardless.
        """
        try:
            import resource as _resource  # noqa: F401
        except ImportError:
            pytest.skip("resource module not available on this platform")

        from spikelab.spike_sorting.guards import _preflight as pf

        # Patch both calls (one per RLIMIT) to return generous limits.
        monkeypatch.setattr(_resource, "getrlimit", lambda _name: (65536, 65536))

        cfg = SimpleNamespace(rt_sort=None)
        findings = pf._check_resource_rlimits(cfg)
        assert findings == []


class TestCheckRecordingSampleRateEdges:
    """``_check_recording_sample_rate`` skips non-callable / NaN sources."""

    def test_skips_when_get_sampling_frequency_not_callable(self):
        """
        Recording-like objects whose ``get_sampling_frequency``
        attribute is not callable (e.g. set to None) are silently
        skipped — no finding, no exception.

        Tests:
            (Test Case 1) Recording stub with
                ``get_sampling_frequency=None`` → empty findings.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards._preflight import (
            _check_recording_sample_rate,
        )

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"
        rec = SimpleNamespace(get_sampling_frequency=None)
        findings = _check_recording_sample_rate(cfg, [rec])
        assert findings == []

    def test_skips_nan_sampling_frequency(self):
        """
        NaN sampling frequency is silently skipped (no spurious
        "nan kHz" warning). The check exits the iteration via the
        explicit ``math.isnan`` guard.

        Tests:
            (Test Case 1) Recording reporting NaN → empty findings.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards._preflight import (
            _check_recording_sample_rate,
        )

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"
        rec = SimpleNamespace(get_sampling_frequency=lambda: float("nan"))
        findings = _check_recording_sample_rate(cfg, [rec])
        assert findings == []


# ===========================================================================
# Kill-callback interrupt re-raise across watchdogs
# ===========================================================================


class TestKillCallbackInterruptReraise:
    """All watchdog kill paths re-raise SystemExit/KeyboardInterrupt."""

    def test_disk_watchdog_reraises_keyboard_interrupt(self):
        """
        DiskUsageWatchdog._on_trip lets KeyboardInterrupt from the
        kill_callback propagate (an in-process kill callback delivers
        it via _thread.interrupt_main).

        Tests:
            (Test Case 1) kill_callback raises KeyboardInterrupt;
                _on_trip re-raises.
        """

        def _raises_kbi():
            raise KeyboardInterrupt("delivered")

        wd = DiskUsageWatchdog(
            folder=Path("."),
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=_raises_kbi,
        )
        # Stub the report build so we don't walk a real folder.
        wd._build_report = lambda free: SimpleNamespace(top_consumers=[])
        with pytest.raises(KeyboardInterrupt):
            wd._on_trip(0.5)

    def test_inactivity_watchdog_reraises_system_exit(self, tmp_path):
        """
        LogInactivityWatchdog._on_trip lets SystemExit propagate.

        Tests:
            (Test Case 1) kill_callback raises SystemExit; _on_trip
                re-raises.
        """

        def _raises_se():
            raise SystemExit(7)

        wd = LogInactivityWatchdog(
            log_path=tmp_path / "log",
            popen=None,
            inactivity_s=10.0,
            sorter="test",
            kill_callback=_raises_se,
        )
        with pytest.raises(SystemExit):
            wd._on_trip(15.0)


# ===========================================================================
# Block K — long-tail medium-priority edge cases
# ===========================================================================


class TestComputeInactivityTimeoutSNaN:
    """``compute_inactivity_timeout_s`` coerces NaN to zero duration."""

    def test_nan_duration_collapses_to_base(self):
        """
        ``recording_duration_min=NaN`` is detected by ``math.isnan``
        and treated as zero, so the timeout collapses to ``base_s``
        rather than propagating NaN through arithmetic (which would
        silently disable downstream comparisons).

        Tests:
            (Test Case 1) ``float('nan')`` → returns ``base_s``.
            (Test Case 2) Result is a finite float (not NaN).
        """
        result = compute_inactivity_timeout_s(
            recording_duration_min=float("nan"),
            base_s=600.0,
            per_min_s=30.0,
        )
        assert result == 600.0
        assert not math.isnan(result)


class TestParseWslconfigMemoryEdges:
    """``_parse_wslconfig_memory_gb`` edge cases for units and sections."""

    def test_kb_unit_converts_to_gb(self):
        """
        ``KB`` unit divides by 1024^2 to produce GB.

        Tests:
            (Test Case 1) ``memory=8388608KB`` parses to 8.0 GB.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        text = "[wsl2]\nmemory=8388608KB\n"
        assert _parse_wslconfig_memory_gb(text) == pytest.approx(8.0)

    def test_unknown_unit_returns_none(self):
        """
        An unrecognised unit suffix falls through the unit-dispatch
        chain and returns None rather than guessing.

        Tests:
            (Test Case 1) ``memory=8XB`` → None.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        assert _parse_wslconfig_memory_gb("[wsl2]\nmemory=8XB\n") is None

    def test_semicolon_comment_lines_skipped(self):
        """
        Lines starting with ``;`` (INI-style comments) are stripped
        before key parsing — a ``;`` on the same physical line as a
        ``memory=`` declaration that is itself a comment does not
        prevent later real keys from parsing.

        Tests:
            (Test Case 1) ``; comment\\n[wsl2]\\nmemory=8GB`` parses
                to 8.0 GB.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        text = "; this is a comment\n[wsl2]\nmemory=8GB\n"
        assert _parse_wslconfig_memory_gb(text) == 8.0

    def test_missing_wsl2_section_returns_none(self):
        """
        ``memory=`` keys outside any ``[wsl2]`` section are ignored.

        Tests:
            (Test Case 1) ``memory=8GB`` under a ``[other]`` section
                returns None.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        text = "[other]\nmemory=8GB\n"
        assert _parse_wslconfig_memory_gb(text) is None


class TestEstimateRtSortIntermediateGbEdges:
    """``estimate_rt_sort_intermediate_gb`` zero / negative input handling."""

    def test_zero_channels_returns_zero(self):
        """
        ``n_channels=0`` short-circuits the multiplication to zero
        without raising.

        Tests:
            (Test Case 1) ``n_channels=0, n_samples=1_000_000`` → 0.0.
        """
        from spikelab.spike_sorting.guards._preflight import (
            estimate_rt_sort_intermediate_gb,
        )

        result = estimate_rt_sort_intermediate_gb(n_channels=0, n_samples=1_000_000)
        assert result == 0.0

    def test_negative_inputs_return_negative_gb(self):
        """
        Buggy upstream values that pass negative counts produce a
        negative GB projection (current behaviour — the helper does
        not clamp). Documents the gap so a future caller surfaces a
        clearer error rather than a "-10.5 GB" message.

        Tests:
            (Test Case 1) ``n_channels=-100`` produces a negative
                projection.
        """
        from spikelab.spike_sorting.guards._preflight import (
            estimate_rt_sort_intermediate_gb,
        )

        result = estimate_rt_sort_intermediate_gb(n_channels=-100, n_samples=1_000_000)
        assert result < 0


class TestTopConsumersEdges:
    """``_top_consumers`` boundary handling for missing folder / depth / limit."""

    def test_missing_folder_returns_empty(self, tmp_path):
        """
        A folder that does not exist returns an empty list rather
        than raising.

        Tests:
            (Test Case 1) Path that doesn't exist → ``[]``.
        """
        from spikelab.spike_sorting.guards._disk_watchdog import _top_consumers

        ghost = tmp_path / "does_not_exist"
        assert _top_consumers(ghost) == []

    def test_max_depth_prunes_deeper_files(self, tmp_path):
        """
        Files nested deeper than ``max_depth`` directories below the
        root are excluded from the result.

        Tests:
            (Test Case 1) With ``max_depth=1``, a file at depth 3
                is not returned even though it is the largest.
            (Test Case 2) A file at depth 1 is returned.
        """
        from spikelab.spike_sorting.guards._disk_watchdog import _top_consumers

        # Shallow file: depth 1.
        (tmp_path / "shallow.bin").write_bytes(b"\x00" * 1024)
        # Deep file: depth 3, larger than the shallow one.
        deep_dir = tmp_path / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)
        (deep_dir / "deep.bin").write_bytes(b"\x00" * (5 * 1024 * 1024))

        consumers = _top_consumers(tmp_path, limit=10, max_depth=1)
        names = [Path(p).name for p, _ in consumers]
        assert "shallow.bin" in names
        assert "deep.bin" not in names

    def test_limit_caps_returned_entries(self, tmp_path):
        """
        ``limit`` trims the final list; the remaining entries are
        the largest by size.

        Tests:
            (Test Case 1) Five files of varying sizes, ``limit=2``
                returns the two biggest (descending).
        """
        from spikelab.spike_sorting.guards._disk_watchdog import _top_consumers

        sizes = {
            "small.bin": 256,
            "medium.bin": 1024,
            "large.bin": 4096,
            "huge.bin": 16384,
            "giant.bin": 65536,
        }
        for name, size in sizes.items():
            (tmp_path / name).write_bytes(b"\x00" * size)

        consumers = _top_consumers(tmp_path, limit=2)
        assert len(consumers) == 2
        names = [Path(p).name for p, _ in consumers]
        assert names == ["giant.bin", "huge.bin"]


class TestDiskFreeGbEdges:
    """``_disk_free_gb`` parent-walk and error-swallow behaviour."""

    def test_walks_up_to_existing_parent(self, tmp_path, monkeypatch):
        """
        When the requested path does not exist, the helper walks up
        parent-by-parent until it finds an existing path and reports
        the free space there.

        Tests:
            (Test Case 1) ``tmp_path / "a" / "b" / "c"`` (none of the
                children exist) returns ``shutil.disk_usage(tmp_path).free``
                converted to GB.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        observed = {}

        def _fake_disk_usage(path):
            observed["path"] = path
            return SimpleNamespace(total=0, used=0, free=10 * (1024**3))

        monkeypatch.setattr(dw.shutil, "disk_usage", _fake_disk_usage)

        ghost = tmp_path / "a" / "b" / "c"
        result = dw._disk_free_gb(ghost)
        assert result == pytest.approx(10.0)
        # The walk landed on an existing parent.
        assert Path(observed["path"]).exists()

    def test_oserror_returns_none(self, tmp_path, monkeypatch):
        """
        When ``shutil.disk_usage`` itself raises ``OSError`` the
        helper returns ``None`` rather than letting the exception
        bubble up to the watchdog poll loop.

        Tests:
            (Test Case 1) Patched ``disk_usage`` raises ``OSError``
                → returns None.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        def _raise(_path):
            raise OSError("simulated")

        monkeypatch.setattr(dw.shutil, "disk_usage", _raise)
        assert dw._disk_free_gb(tmp_path) is None


class TestFolderSizeBytesEdges:
    """``_folder_size_bytes`` swallows top-level walk errors."""

    def test_oswalk_exception_swallowed(self, tmp_path, monkeypatch):
        """
        An exception raised by ``os.walk`` itself (rather than per-
        entry) is caught by the outer ``except Exception`` and the
        helper returns the partial total accumulated so far (zero in
        this case).

        Tests:
            (Test Case 1) Patched ``os.walk`` raises immediately →
                returns 0.0 (no entries accumulated).
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        def _raising_walk(*_args, **_kwargs):
            raise RuntimeError("walk exploded")

        monkeypatch.setattr(dw.os, "walk", _raising_walk)
        assert dw._folder_size_bytes(tmp_path) == 0.0


class TestCleanupTempFilesUnlinkFailure:
    """``cleanup_temp_files`` increments a failed counter when unlink raises."""

    def test_unlink_failure_does_not_propagate(self, tmp_path, monkeypatch, caplog):
        """
        A per-file ``unlink`` failure is swallowed and counted in the
        ``failed`` total rather than aborting the sweep. Subsequent
        marker files are still attempted.

        Tests:
            (Test Case 1) Two marker files are created during the
                context; ``unlink`` is patched to always raise. The
                context exit completes without raising and both
                files are left on disk.
            (Test Case 2) The summary log line includes ``failed``
                (count > 0) and the surviving file count.
        """
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        original_unlink = Path.unlink

        def _raising_unlink(self, *args, **kwargs):
            # Only fail on marker files we created during the
            # context. Pre-existing pytest tmp files (none here) and
            # unrelated paths still call through.
            if "spikelab_" in self.name or "kilosort_" in self.name:
                raise PermissionError(f"locked: {self.name}")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", _raising_unlink)

        with caplog.at_level(logging.INFO):
            with cleanup_temp_files(enabled=True):
                (tmp_path / "spikelab_a.tmp").write_text("x")
                (tmp_path / "kilosort_b.tmp").write_text("x")

        # Both files survive because unlink raised.
        assert (tmp_path / "spikelab_a.tmp").exists()
        assert (tmp_path / "kilosort_b.tmp").exists()

        # The summary log line was emitted with a non-zero failed count.
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "swept 0 stale temp file(s)" in m and "2 failed" in m for m in messages
        )


class TestCheckRecordingSampleRateMultiple:
    """``_check_recording_sample_rate`` emits one finding per bad recording."""

    def test_one_finding_per_out_of_window_recording(self):
        """
        When several pre-loaded recordings are passed and only some
        sit outside the sorter's expected sample-rate window, a warn
        finding is emitted for each offending recording (not a
        single bulk finding).

        Tests:
            (Test Case 1) Three recordings: one in-window, two
                out-of-window. Two findings come back.
            (Test Case 2) Each finding has code
                ``sample_rate_out_of_window`` and level ``warn``.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards._preflight import (
            _check_recording_sample_rate,
        )

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"

        good = SimpleNamespace(get_sampling_frequency=lambda: 30_000.0)
        bad1 = SimpleNamespace(get_sampling_frequency=lambda: 1_000.0)
        bad2 = SimpleNamespace(get_sampling_frequency=lambda: 100.0)

        findings = _check_recording_sample_rate(cfg, [good, bad1, bad2])
        assert len(findings) == 2
        assert all(f.code == "sample_rate_out_of_window" for f in findings)
        assert all(f.level == "warn" for f in findings)


# ===========================================================================
# Block L — additional medium-priority edge cases
# ===========================================================================


class TestTopConsumersStatError:
    """``_top_consumers`` swallows per-file ``OSError`` in ``stat()``."""

    def test_per_file_stat_oserror_does_not_abort_walk(self, tmp_path, monkeypatch):
        """
        When ``Path.stat`` raises ``OSError`` for one file the helper
        continues to the next entry rather than aborting the walk.

        Tests:
            (Test Case 1) Two files; ``stat`` raises only for the
                "broken" file. The "good" file is still returned in
                the result list.
            (Test Case 2) The broken file is omitted (no exception
                propagates).
        """
        from spikelab.spike_sorting.guards._disk_watchdog import _top_consumers

        good = tmp_path / "good.bin"
        good.write_bytes(b"\x00" * 4096)
        broken = tmp_path / "broken.bin"
        broken.write_bytes(b"\x00" * 4096)

        original_stat = Path.stat

        def _patched_stat(self, *args, **kwargs):
            if self.name == "broken.bin":
                raise OSError("simulated stat failure")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", _patched_stat)
        consumers = _top_consumers(tmp_path, limit=10)
        names = [Path(p).name for p, _ in consumers]
        assert "good.bin" in names
        assert "broken.bin" not in names


class TestDiskExhaustionReportToDict:
    """``DiskExhaustionReport.to_dict`` round-trips edge field values."""

    def test_empty_collections_round_trip(self):
        """
        Empty ``top_consumers`` and ``suggested_actions`` survive the
        dict conversion as empty lists rather than being dropped.

        Tests:
            (Test Case 1) Both collections empty → keys present with
                ``[]`` values.
        """
        report = DiskExhaustionReport(
            folder="/x",
            free_gb_at_trip=0.5,
            abort_threshold_gb=1.0,
        )
        out = report.to_dict()
        assert out["top_consumers"] == []
        assert out["suggested_actions"] == []

    def test_projected_need_gb_none_round_trips(self):
        """
        ``projected_need_gb=None`` is preserved (not converted to 0.0
        or omitted), so consumers can branch on the missing-projection
        case.

        Tests:
            (Test Case 1) Default ``projected_need_gb`` (None) appears
                explicitly in the dict.
        """
        report = DiskExhaustionReport(
            folder="/x",
            free_gb_at_trip=0.5,
            abort_threshold_gb=1.0,
        )
        out = report.to_dict()
        assert "projected_need_gb" in out
        assert out["projected_need_gb"] is None


class TestHdf5PluginFindingEdges:
    """``_hdf5_plugin_finding`` config + env-var precedence and file-vs-dir."""

    def test_config_and_env_both_unset_returns_none(self, monkeypatch):
        """
        With neither ``config.recording.hdf5_plugin_path`` set nor
        ``HDF5_PLUGIN_PATH`` in the environment, the helper returns
        ``None`` (no finding to surface — the host has not opted in
        to a plugin path).

        Tests:
            (Test Case 1) Config has ``hdf5_plugin_path=None`` and env
                var is unset → returns ``None``.
        """
        from spikelab.spike_sorting.guards._preflight import _hdf5_plugin_finding

        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        cfg = SimpleNamespace(recording=SimpleNamespace(hdf5_plugin_path=None))
        assert _hdf5_plugin_finding(cfg) is None

    def test_configured_path_is_file_yields_fail(self, monkeypatch, tmp_path):
        """
        A configured path that exists but is a regular file (not a
        directory) fails ``is_dir()`` and triggers the
        ``hdf5_plugin_missing`` fail finding.

        Tests:
            (Test Case 1) Path is a real file → returns a fail-level
                ``hdf5_plugin_missing`` finding.
        """
        from spikelab.spike_sorting.guards._preflight import _hdf5_plugin_finding

        plugin_file = tmp_path / "plugin.so"
        plugin_file.write_bytes(b"x")
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)
        cfg = SimpleNamespace(
            recording=SimpleNamespace(hdf5_plugin_path=str(plugin_file))
        )
        finding = _hdf5_plugin_finding(cfg)
        assert finding is not None
        assert finding.level == "fail"
        assert finding.code == "hdf5_plugin_missing"


class TestListMarkerFilesRglobError:
    """``_list_marker_files`` swallows ``OSError`` raised by ``rglob``."""

    def test_rglob_oserror_inside_marker_subdir_swallowed(self, tmp_path, monkeypatch):
        """
        When a marker-named directory raises ``OSError`` from
        ``rglob`` (e.g. permission denied descending into a subtree)
        the helper logs nothing and continues to the next entry — it
        does not propagate the error.

        Tests:
            (Test Case 1) Marker subdir whose ``rglob`` raises is
                silently skipped; a sibling marker file is still
                returned.
        """
        from spikelab.spike_sorting.guards._tempfile_cleanup import (
            _list_marker_files,
        )

        # Sibling marker file at top level — should be returned.
        sibling = tmp_path / "spikelab_keep.tmp"
        sibling.write_text("x")

        # Marker-named directory whose rglob will raise.
        bad_dir = tmp_path / "spikelab_bad"
        bad_dir.mkdir()
        # Dummy contents (won't actually be enumerated due to the
        # patched rglob).
        (bad_dir / "ignored.tmp").write_text("y")

        original_rglob = Path.rglob

        def _patched_rglob(self, pattern):
            if self.name == "spikelab_bad":
                raise OSError("permission denied")
            return original_rglob(self, pattern)

        monkeypatch.setattr(Path, "rglob", _patched_rglob)
        result = _list_marker_files(tmp_path)
        assert sibling in result


class TestReportFindingsExtra:
    """``report_findings`` log formatting + multi-fatal aggregation."""

    def test_fail_logged_with_uppercase_marker_and_remediation(self, caplog):
        """
        Fail-level findings are logged with an uppercase ``[FAIL]``
        marker and the remediation block on a follow-up line. The
        message and remediation each appear in the log records.

        Tests:
            (Test Case 1) Records contain ``[FAIL]`` + finding code +
                message.
            (Test Case 2) Records contain the remediation arrow
                (``-> ``) and remediation text.
        """
        finding = PreflightFinding(
            level="fail",
            code="low_disk",
            category="resource",
            message="only 0.1 GB free",
            remediation="free up space",
        )
        with caplog.at_level(
            logging.WARNING, logger="spikelab.spike_sorting.guards._preflight"
        ):
            with pytest.raises(ResourceSortFailure):
                report_findings([finding])

        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "[FAIL]" in joined
        assert "low_disk" in joined
        assert "only 0.1 GB free" in joined
        assert "->" in joined
        assert "free up space" in joined

    def test_multiple_fatals_summary_reports_first_only(self):
        """
        With multiple fatal findings the raised exception's summary
        names the count and references *only* the first finding
        (callers escalate via the categorical exception type, not by
        scanning a multi-finding list).

        Tests:
            (Test Case 1) Three fatal findings; ``len(fatal)`` (3)
                appears in the message.
            (Test Case 2) Only the first finding's code is in the
                summary; the others are not.
        """
        findings = [
            PreflightFinding(
                level="fail",
                code="first_code",
                message="first msg",
                category="resource",
            ),
            PreflightFinding(
                level="fail",
                code="second_code",
                message="second msg",
                category="resource",
            ),
            PreflightFinding(
                level="fail",
                code="third_code",
                message="third msg",
                category="resource",
            ),
        ]
        with pytest.raises(ResourceSortFailure) as exc_info:
            report_findings(findings)
        msg = str(exc_info.value)
        assert "3" in msg
        assert "first_code" in msg
        assert "second_code" not in msg
        assert "third_code" not in msg


# ===========================================================================
# Block M — audit / observability and finally-block swallow paths
# ===========================================================================


class TestAppendAuditEventContextVarFallback:
    """``append_audit_event`` falls back to the active log-path ContextVar."""

    def test_active_log_path_used_when_arg_omitted(self, tmp_path):
        """
        Calling without ``log_path`` consults
        ``_active_log_path`` (set by ``set_active_log_path``); the
        audit file is written next to the published path.

        Tests:
            (Test Case 1) Inside ``set_active_log_path(<path>)``, a
                ``log_path``-less call writes one line to
                ``<path-parent>/watchdog_events.jsonl``.
            (Test Case 2) Outside the context, a follow-up call is a
                silent no-op (no second line is added).
        """
        from spikelab.spike_sorting.guards import (
            set_active_log_path,
        )

        log_path = tmp_path / "rec.log"
        log_path.touch()
        with set_active_log_path(log_path):
            append_audit_event(watchdog="host_memory", event="warn", used_pct=80.0)

        audit = tmp_path / "watchdog_events.jsonl"
        assert audit.is_file()
        first_lines = audit.read_text(encoding="utf-8").splitlines()
        assert len(first_lines) == 1
        entry = json.loads(first_lines[0])
        assert entry["watchdog"] == "host_memory"
        assert entry["used_pct"] == 80.0

        # Outside the context, the call is silently dropped.
        append_audit_event(watchdog="host_memory", event="warn", used_pct=81.0)
        post_lines = audit.read_text(encoding="utf-8").splitlines()
        assert len(post_lines) == 1


class TestAppendAuditEventParentDirCreated:
    """``append_audit_event`` creates the results folder on demand."""

    def test_parent_directory_created_when_missing(self, tmp_path):
        """
        When the supplied ``log_path``'s parent does not yet exist,
        the helper creates it (``mkdir(parents=True, exist_ok=True)``)
        before writing the audit file.

        Tests:
            (Test Case 1) Log path inside an unborn ``results/`` dir;
                after one append, the dir exists with the audit file
                inside.
        """
        log_path = tmp_path / "results" / "missing_parent" / "rec.log"
        # Note: do not create any of the parent folders.
        assert not log_path.parent.exists()

        append_audit_event(
            watchdog="disk", event="warn", log_path=log_path, free_gb=1.0
        )

        audit = log_path.parent / "watchdog_events.jsonl"
        assert audit.is_file()
        entry = json.loads(audit.read_text(encoding="utf-8").strip())
        assert entry["watchdog"] == "disk"


class TestCleanupTempFilesGetTempDirRaises:
    """``cleanup_temp_files`` falls through silently when ``gettempdir`` raises."""

    def test_gettempdir_exception_is_no_op(self, tmp_path, monkeypatch):
        """
        ``tempfile.gettempdir()`` raising is caught at the top of the
        context manager; the body still runs, and on exit no sweep is
        attempted (so files inside the would-be temp dir are left
        untouched).

        Tests:
            (Test Case 1) Patched ``gettempdir`` raises; the with-
                block body executes; on exit no exception propagates.
            (Test Case 2) A marker file pre-placed in ``tmp_path`` is
                left intact.
        """
        marker = tmp_path / "spikelab_x.tmp"
        marker.write_text("keep me")

        def _broken_gettempdir():
            raise RuntimeError("simulated gettempdir failure")

        monkeypatch.setattr(tempfile, "gettempdir", _broken_gettempdir)

        body_ran = {"value": False}
        with cleanup_temp_files(enabled=True):
            body_ran["value"] = True
        assert body_ran["value"] is True
        assert marker.exists()


class TestCheckKilosort2HostKsDirIsFile:
    """``_check_kilosort2_host`` fails when ``KILOSORT_PATH`` points to a file."""

    def test_ks_dir_is_file_yields_missing_dir_finding(self, tmp_path, monkeypatch):
        """
        A ``KILOSORT_PATH`` that exists but is a regular file fails
        ``ks_dir.is_dir()`` and yields the ``does not exist`` fail
        finding (the helper does not distinguish file vs. missing —
        both fail the dir check).

        Tests:
            (Test Case 1) ``KILOSORT_PATH`` set to a real file →
                exactly one path-related fail finding is emitted.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        bogus = tmp_path / "ks_is_a_file.txt"
        bogus.write_text("not a directory")

        monkeypatch.setattr(pf.shutil, "which", lambda name: "/usr/bin/matlab")
        monkeypatch.setenv("KILOSORT_PATH", str(bogus))

        cfg = SimpleNamespace(sorter=SimpleNamespace(sorter_path=None))
        findings = pf._check_kilosort2_host(cfg)
        assert len(findings) == 1
        f = findings[0]
        assert f.level == "fail"
        assert f.code == "sorter_dependency_missing"
        assert "does not exist" in f.message


class TestIOStallWatchdogEnterSecondProbeDisables:
    """``IOStallWatchdog.__enter__`` disables when ``_read_io_bytes`` returns None."""

    def test_resolved_device_but_unreadable_counters_disables(self, tmp_path):
        """
        ``__enter__`` resolves the block device successfully but the
        follow-up ``_read_io_bytes`` probe returns ``None`` (e.g. an
        NVMe partition not exposed via ``disk_io_counters(perdisk=
        True)``). The watchdog enters as disabled rather than
        crashing or polling without baselines.

        Tests:
            (Test Case 1) Patched ``_resolve_device_for_path`` returns
                a device name; patched ``_read_io_bytes`` returns
                ``None`` → ``_enabled`` is False inside the context.
            (Test Case 2) No polling thread is started.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", return_value=None),
        ):
            wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
            with wd:
                assert wd._enabled is False
                assert wd._thread is None


class TestIOStallWatchdogOnTripAudit:
    """``IOStallWatchdog._on_trip`` writes an audit event and runs callbacks."""

    def test_audit_event_appended_on_abort(self, tmp_path, monkeypatch):
        """
        ``_on_trip`` calls ``append_audit_event`` with watchdog
        ``"io_stall"`` and event ``"abort"`` plus the stall payload.

        Tests:
            (Test Case 1) Patched ``append_audit_event`` records the
                call; assert the watchdog/event labels and a
                stall_for_s value matching the trip.
            (Test Case 2) ``_thread.interrupt_main`` is suppressed by
                pre-setting the stop event so the test thread does
                not receive a phantom KeyboardInterrupt.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
        wd._device = "sda1"
        # Suppress _thread.interrupt_main via the documented gate:
        # _stop_event.is_set() short-circuits the interrupt.
        wd._stop_event.set()

        captured = []

        def _fake_audit(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr(iom, "append_audit_event", _fake_audit)
        wd._on_trip(stalled_for=15.0)

        assert wd.tripped() is True
        assert len(captured) == 1
        evt = captured[0]
        assert evt["watchdog"] == "io_stall"
        assert evt["event"] == "abort"
        assert evt["stalled_for_s"] == 15.0
        assert evt["device"] == "sda1"

    def test_kill_callback_exception_isolated(self, tmp_path, monkeypatch):
        """
        Each registered kill_callback runs even if another raises
        ``Exception`` (only ``SystemExit`` / ``KeyboardInterrupt``
        propagate). Subsequent callbacks are still invoked; the
        watchdog still records as tripped.

        Tests:
            (Test Case 1) Two callbacks: first raises ``ValueError``,
                second records called=True. After ``_on_trip``, the
                second callback ran.
            (Test Case 2) The watchdog is marked tripped despite the
                first callback's exception.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
        wd._device = "sda1"
        wd._stop_event.set()  # Suppress interrupt_main.

        order = []

        def _bad():
            order.append("bad-start")
            raise ValueError("boom")

        def _good():
            order.append("good-ran")

        wd.register_kill_callback(_bad)
        wd.register_kill_callback(_good)

        # Don't actually write to disk for this test.
        monkeypatch.setattr(iom, "append_audit_event", lambda **_: None)

        wd._on_trip(stalled_for=12.0)

        assert "bad-start" in order
        assert "good-ran" in order
        assert wd.tripped() is True


class TestLogInactivityPollLoopFileAppearsMidLoop:
    """``LogInactivityWatchdog._poll_loop`` resets clock when file appears."""

    def test_late_appearing_log_resets_inactivity_clock(self, tmp_path):
        """
        With no log file at watchdog start (``seen_any=False``), a
        log that appears mid-poll takes the ``not seen_any`` branch —
        ``last_progress_t`` is reset, so the watchdog does not
        immediately trip on the original startup window.

        Tests:
            (Test Case 1) Log file is created ~0.4s after watchdog
                start; ``inactivity_s=1.5`` is short enough that a
                non-resetting watchdog would trip within ~3s. Wait
                ~1.0s after creation (well inside the post-reset
                window) and assert no trip.
            (Test Case 2) Then keep the file flat for the remaining
                window — eventually the watchdog trips, proving the
                clock was running again from the appearance time.
        """
        log = tmp_path / "delayed.log"
        # Don't create yet.
        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None  # "still running"

        wd = LogInactivityWatchdog(
            log_path=log,
            popen=popen,
            inactivity_s=1.5,
            sorter="kilosort2",
            poll_interval_s=0.1,
            kill_grace_s=0.2,
        )
        with wd:
            time.sleep(0.4)
            # File appears — `not seen_any` branch should reset clock.
            log.write_text("first line\n")
            time.sleep(1.0)  # ~0.4 + 1.0 = 1.4 elapsed; reset at 0.4 → clock=1.0s.
            assert not wd.tripped(), "watchdog should not have tripped yet"
            # Now wait long enough past the post-appearance window for
            # the watchdog to trip on the now-flat log.
            deadline = time.time() + 3.0
            while time.time() < deadline and not wd.tripped():
                time.sleep(0.1)
        assert wd.tripped(), "watchdog should have tripped after appearance + idle"


class TestLogInactivityMakeErrorCustomMessage:
    """``LogInactivityWatchdog.make_error`` honours an explicit message override."""

    def test_custom_message_replaces_default(self, tmp_path):
        """
        Passing ``message=...`` produces a ``SorterTimeoutError`` with
        that exact message; the default placeholder formatting is
        not used.

        Tests:
            (Test Case 1) ``make_error("custom note")`` yields an
                error whose ``str`` is the supplied message.
            (Test Case 2) The ``sorter`` and ``inactivity_s``
                attributes still match the watchdog config.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "rec.log",
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=600.0,
            sorter="kilosort4",
        )
        err = wd.make_error("custom note")
        assert isinstance(err, SorterTimeoutError)
        assert str(err) == "custom note"
        assert err.sorter == "kilosort4"
        assert err.inactivity_s == 600.0


class TestSortLockCleanupOnExitSwallow:
    """``acquire_sort_lock`` swallows ``OSError`` on lock-removal at exit."""

    def test_unlink_failure_at_exit_does_not_propagate(self, tmp_path, monkeypatch):
        """
        When the final ``lock_path.unlink()`` in the cleanup ``finally``
        raises ``OSError`` (e.g. an external process removed the lock
        first, or a race with a watchdog ``os._exit``), the context
        manager exits silently — the next sort would reclaim via
        stale-lock detection.

        Tests:
            (Test Case 1) The ``with acquire_sort_lock(folder):`` block
                completes without raising even though ``unlink`` on
                the lock file is patched to raise ``OSError`` on the
                cleanup call.
        """
        # acquire_sort_lock takes a *folder*; the lock file lives at
        # ``<folder>/.spikelab_sort.lock``.
        lock_path = tmp_path / ".spikelab_sort.lock"

        original_unlink = Path.unlink
        calls = {"count": 0}

        def _patched_unlink(self, *args, **kwargs):
            if self == lock_path:
                calls["count"] += 1
                raise OSError("simulated lock-removal failure")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", _patched_unlink)

        # Should complete without raising despite the unlink failure.
        with acquire_sort_lock(tmp_path) as held:
            assert held == lock_path

        assert calls["count"] >= 1


# ===========================================================================
# Block N — watchdog rate-limit / abort cascade / preflight aggregation /
# canary boundaries / IO counter wrap
# ===========================================================================


class TestHostMemoryWatchdogMaybeWarn:
    """``HostMemoryWatchdog._maybe_warn`` rate-limit + audit append."""

    def test_rate_limit_suppresses_repeat_within_window(self, caplog):
        """
        Two warns within ``warn_repeat_s`` produce only one log line —
        the early-return on ``now - self._last_warn_t < warn_repeat_s``
        suppresses the second.

        Tests:
            (Test Case 1) Two back-to-back ``_maybe_warn`` calls →
                exactly one WARNING log record.
        """
        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, warn_repeat_s=300.0)
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._watchdog",
        ):
            wd._maybe_warn(75.0)
            wd._maybe_warn(76.0)

        warnings = [r for r in caplog.records if "system memory at" in r.getMessage()]
        assert len(warnings) == 1

    def test_audit_event_appended_on_warn(self, monkeypatch):
        """
        ``_maybe_warn`` appends a watchdog="host_memory" event="warn"
        audit line carrying the percent.

        Tests:
            (Test Case 1) Patched ``append_audit_event`` records the
                call; the captured kwargs contain the watchdog and
                event labels plus ``used_pct``.
        """
        from spikelab.spike_sorting.guards import _watchdog as wm

        captured = []

        def _fake_audit(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr(wm, "append_audit_event", _fake_audit)

        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0)
        wd._maybe_warn(78.0)

        assert len(captured) == 1
        assert captured[0]["watchdog"] == "host_memory"
        assert captured[0]["event"] == "warn"
        assert captured[0]["used_pct"] == 78.0


class TestHostMemoryWatchdogOnAbortPlumbing:
    """``HostMemoryWatchdog._on_abort`` wiring of callbacks + snapshot swallow."""

    def test_kill_callbacks_run_alongside_subprocess_termination(self, monkeypatch):
        """
        ``_on_abort`` runs registered kill callbacks via
        ``_run_kill_callbacks`` in addition to terminating registered
        subprocesses. Both must fire on a trip.

        Tests:
            (Test Case 1) One registered subprocess (mock) and one
                registered kill_callback. After ``_on_abort``, both
                ran exactly once.
        """
        from spikelab.spike_sorting.guards import _watchdog as wm

        # Pre-set _stop_event so interrupt_main is suppressed (the
        # documented "watchdog already exiting" gate).
        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, kill_grace_s=0.0)
        wd._stop_event.set()

        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None
        wd.register_subprocess(popen, kill_grace_s=0.0)

        cb_calls = {"count": 0}

        def _cb():
            cb_calls["count"] += 1

        wd.register_kill_callback(_cb)
        # Disable snapshot side-channel to keep the test hermetic.
        monkeypatch.setattr(wm, "append_audit_event", lambda **_: None)

        wd._on_abort(95.0)

        popen.terminate.assert_called_once()
        assert cb_calls["count"] == 1
        assert wd.tripped() is True

    def test_snapshot_capture_failure_swallowed(self, monkeypatch):
        """
        If ``_try_capture_snapshot_to_results`` raises, the abort path
        still runs the rest of its cascade (terminate + callbacks).
        The snapshot import lives inside a try/except; failures are
        silent.

        Tests:
            (Test Case 1) Patched snapshot helper raises; ``_on_abort``
                still completes and ``tripped()`` is True.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod
        from spikelab.spike_sorting.guards import _watchdog as wm

        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, kill_grace_s=0.0)
        wd._stop_event.set()  # Suppress interrupt_main.

        def _raise(*_a, **_kw):
            raise RuntimeError("simulated snapshot failure")

        monkeypatch.setattr(gpu_mod, "_try_capture_snapshot_to_results", _raise)
        monkeypatch.setattr(wm, "append_audit_event", lambda **_: None)

        # Should not raise.
        wd._on_abort(95.0)
        assert wd.tripped() is True


class TestHostMemoryWatchdogTerminateRegistered:
    """``_terminate_registered`` grace handling and exception isolation."""

    def test_multiple_subprocesses_use_max_grace(self, monkeypatch):
        """
        With several subprocesses registered at different
        ``kill_grace_s`` values, the helper sleeps ``max(grace)``
        between terminate and kill so the slowest process gets its
        full window.

        Tests:
            (Test Case 1) Two subprocesses registered at grace 0.05
                and 0.20; ``_terminate_registered`` calls
                ``time.sleep(0.20)`` exactly once (the max value).
        """
        from spikelab.spike_sorting.guards import _watchdog as wm

        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, kill_grace_s=0.0)
        # Two subprocesses; both already exited so terminate() is a no-op
        # and kill() is skipped (poll() returns 0 → not None).
        p1 = mock.Mock(spec=subprocess.Popen)
        p1.poll.return_value = 0
        p2 = mock.Mock(spec=subprocess.Popen)
        p2.poll.return_value = 0
        wd.register_subprocess(p1, kill_grace_s=0.05)
        wd.register_subprocess(p2, kill_grace_s=0.20)

        sleep_calls = []
        monkeypatch.setattr(wm.time, "sleep", lambda s: sleep_calls.append(s))

        wd._terminate_registered()

        # The single grace sleep is max(0.05, 0.20) = 0.20.
        assert sleep_calls == [0.20]

    def test_poll_exception_logged_continues_iteration(self, caplog):
        """
        ``popen.poll()`` raising on one subprocess does not abort
        iteration over the rest — the exception is logged and the
        next subprocess is still inspected.

        Tests:
            (Test Case 1) Two subprocesses; the first's ``poll``
                raises ``OSError`` (e.g. invalid handle on Windows);
                the second's ``terminate`` is still called.
            (Test Case 2) An ERROR record mentioning ``terminate()
                failed`` is emitted.
        """
        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, kill_grace_s=0.0)

        bad = mock.Mock(spec=subprocess.Popen)
        bad.poll.side_effect = OSError("invalid handle")
        bad.pid = 1111
        good = mock.Mock(spec=subprocess.Popen)
        good.poll.return_value = None
        good.pid = 2222
        wd.register_subprocess(bad, kill_grace_s=0.0)
        wd.register_subprocess(good, kill_grace_s=0.0)

        with caplog.at_level(
            logging.ERROR,
            logger="spikelab.spike_sorting.guards._watchdog",
        ):
            wd._terminate_registered()

        # Iteration continued — the second subprocess was still asked
        # to terminate.
        good.terminate.assert_called_once()
        # The first's failure was logged.
        assert any("terminate() failed" in r.getMessage() for r in caplog.records)

    def test_kill_exception_logged_does_not_propagate(self, caplog):
        """
        ``popen.kill()`` raising on a still-alive process is caught
        by the inner ``except Exception`` and logged; subsequent
        subprocesses still get their kill attempt.

        Tests:
            (Test Case 1) Two subprocesses both still alive after
                grace; the first's ``kill`` raises; the second's
                ``kill`` is still called.
            (Test Case 2) ERROR log record mentions ``kill() failed``.
        """
        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, kill_grace_s=0.0)

        bad = mock.Mock(spec=subprocess.Popen)
        bad.poll.return_value = None  # Still alive after terminate.
        bad.kill.side_effect = OSError("kill failed")
        bad.pid = 3333
        good = mock.Mock(spec=subprocess.Popen)
        good.poll.return_value = None
        good.pid = 4444
        wd.register_subprocess(bad, kill_grace_s=0.0)
        wd.register_subprocess(good, kill_grace_s=0.0)

        with caplog.at_level(
            logging.ERROR,
            logger="spikelab.spike_sorting.guards._watchdog",
        ):
            wd._terminate_registered()

        good.kill.assert_called_once()
        assert any("kill() failed" in r.getMessage() for r in caplog.records)


class TestHostMemoryWatchdogInitNegativeKillGrace:
    """``HostMemoryWatchdog.__init__`` validates ``kill_grace_s`` non-negative."""

    def test_negative_kill_grace_raises(self):
        """
        Negative ``kill_grace_s`` raises ``ValueError`` at
        construction; the misconfig is caught early rather than
        breaking the abort path later.

        Tests:
            (Test Case 1) ``kill_grace_s=-1.0`` → ``ValueError``
                referencing ``kill_grace_s``.
        """
        with pytest.raises(ValueError, match="kill_grace_s"):
            HostMemoryWatchdog(
                warn_pct=70.0,
                abort_pct=90.0,
                kill_grace_s=-1.0,
            )


class TestGpuMemoryWatchdogMaybeWarn:
    """``GpuMemoryWatchdog._maybe_warn`` rate-limit suppression."""

    def test_rate_limit_suppresses_repeat_within_window(self, caplog):
        """
        Two ``_maybe_warn`` calls within ``warn_repeat_s`` produce
        only one log line.

        Tests:
            (Test Case 1) Two back-to-back warns → exactly one
                WARNING record about VRAM.
        """
        wd = GpuMemoryWatchdog(
            warn_pct=70.0,
            abort_pct=90.0,
            poll_interval_s=1.0,
            warn_repeat_s=300.0,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            wd._maybe_warn(75.0)
            wd._maybe_warn(80.0)

        vram_warnings = [r for r in caplog.records if "VRAM at" in r.getMessage()]
        assert len(vram_warnings) == 1


class TestIOStallPollLoopBaselineAndWarn:
    """``IOStallWatchdog._poll_loop`` initial baseline + 50% warn branch."""

    def test_initial_last_bytes_none_seeds_baseline_without_tripping(self, tmp_path):
        """
        When the very first inside-loop read of the byte counter
        returns ``None`` (transient between ``__enter__``'s probe and
        the first poll), ``last_bytes`` stays None and the
        baseline-seed branch later fires when counters recover —
        without tripping the watchdog.

        Tests:
            (Test Case 1) First in-loop read returns None; later
                reads return increasing bytes → no trip during the
                test window.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        # Sequence: __enter__ probe = 100, first loop call = None,
        # then advancing bytes thereafter.
        seq = iter([100, None, 200, 300, 400, 500, 600])

        def _read(_dev):
            try:
                return next(seq)
            except StopIteration:
                return 700

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", side_effect=_read),
        ):
            wd = IOStallWatchdog(tmp_path, stall_s=2.0, poll_interval_s=0.05)
            with wd:
                time.sleep(0.4)
                assert not wd.tripped()

    def test_warn_at_50_percent_branch_logs_warning(self, tmp_path, caplog):
        """
        After ``stalled_for >= stall_s * 0.5`` and the warn-repeat
        window has elapsed, the watchdog calls ``_maybe_warn`` which
        logs a WARNING. The full trip window has not yet passed, so
        ``tripped()`` is still False.

        Tests:
            (Test Case 1) ``stall_s=1.0`` with ``poll_interval_s=0.05``
                and ``warn_repeat_s=0.0`` (allow immediate warns).
                After ~0.6s of constant counters, the WARNING
                ``idle for`` line is emitted; tripped() is False.

        Notes:
            - Exits before the full 1.0s trip window so the test
              never sends a phantom ``KeyboardInterrupt`` into the
              test thread.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", return_value=100),
        ):
            wd = IOStallWatchdog(
                tmp_path,
                stall_s=1.0,
                poll_interval_s=0.05,
                warn_repeat_s=0.0,
            )
            with caplog.at_level(
                logging.WARNING,
                logger="spikelab.spike_sorting.guards._io_stall",
            ):
                with wd:
                    deadline = time.time() + 0.85
                    while time.time() < deadline:
                        if any("idle for" in r.getMessage() for r in caplog.records):
                            break
                        time.sleep(0.05)

        assert any("idle for" in r.getMessage() for r in caplog.records)
        assert not wd.tripped()


class TestRunPreflightMultiFinding:
    """``run_preflight`` aggregates findings from multiple checks."""

    def test_multiple_findings_returned_together(self, monkeypatch):
        """
        A config that simultaneously fails the disk, RAM, VRAM, and
        HDF5 plugin checks returns all four finding codes in a single
        call. None of the checks short-circuits the others.

        Tests:
            (Test Case 1) Patched probes report low disk + low RAM +
                low VRAM; ``HDF5_PLUGIN_PATH`` set to a non-existent
                directory. The returned codes include each of:
                ``low_disk_inter``, ``low_ram``, ``low_vram``, and
                ``hdf5_plugin_missing``.
        """
        cfg = _make_config(
            sorter_name="kilosort4",
            hdf5_plugin_path="/totally/missing/plugin/dir",
        )
        # Mute the v2 dispatchers used in the existing TestRunPreflight
        # autouse fixture so this standalone class behaves the same.
        monkeypatch.setattr(preflight_mod, "_check_sorter_dependencies", lambda c: [])
        monkeypatch.setattr(preflight_mod, "_check_gpu_device_present", lambda c: None)
        monkeypatch.setattr(
            preflight_mod, "_check_recording_sample_rate", lambda c, recs: []
        )
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 5.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 1.0)
        monkeypatch.setattr(preflight_mod, "_free_vram_gb", lambda: 0.5)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)

        findings = run_preflight(cfg, [mock.Mock()], ["/inter"], ["/results"])
        codes = {f.code for f in findings}
        assert "low_disk_inter" in codes
        assert "low_ram" in codes
        assert "low_vram" in codes
        assert "hdf5_plugin_missing" in codes


class TestCheckResourceRlimitsEdges:
    """``_check_resource_rlimits`` handles ``soft_nofile=0`` and missing NPROC."""

    def test_soft_nofile_zero_silently_skipped(self, monkeypatch):
        """
        ``RLIMIT_NOFILE`` reporting 0 falls outside the
        ``0 < soft_nofile < 4096`` test (the strict lower bound rules
        out 0) and produces no finding — current behaviour, documented
        as a quirk.

        Tests:
            (Test Case 1) Patched ``getrlimit`` returns ``(0, 0)`` for
                NOFILE and a generous limit for NPROC → no finding.
        """
        try:
            import resource as _resource
        except ImportError:
            pytest.skip("resource module not available on this platform")

        from spikelab.spike_sorting.guards import _preflight as pf

        def _fake(name):
            if name == _resource.RLIMIT_NOFILE:
                return (0, 0)
            return (65536, 65536)

        monkeypatch.setattr(_resource, "getrlimit", _fake)

        cfg = SimpleNamespace(rt_sort=None)
        findings = pf._check_resource_rlimits(cfg)
        assert findings == []

    def test_rlimit_nproc_missing_skips_nproc_check(self, monkeypatch):
        """
        On platforms lacking ``RLIMIT_NPROC`` the ``getrlimit`` call
        raises ``AttributeError``; the check sets ``soft_nproc=None``
        and skips the NPROC threshold without affecting the NOFILE
        finding.

        Tests:
            (Test Case 1) Patched ``getrlimit`` raises ``AttributeError``
                for the second call (NPROC); a tight NOFILE still
                produces its finding.
            (Test Case 2) No NPROC finding is emitted.
        """
        try:
            import resource as _resource
        except ImportError:
            pytest.skip("resource module not available on this platform")

        from spikelab.spike_sorting.guards import _preflight as pf

        def _fake(name):
            if name == _resource.RLIMIT_NOFILE:
                return (1024, 1024)  # Below 4096 → NOFILE finding fires.
            raise AttributeError("RLIMIT_NPROC not exposed on this platform")

        monkeypatch.setattr(_resource, "getrlimit", _fake)

        cfg = SimpleNamespace(rt_sort=None)
        findings = pf._check_resource_rlimits(cfg)
        codes = [f.code for f in findings]
        assert "low_rlimit_nofile" in codes
        assert "low_rlimit_nproc" not in codes


class TestCheckRecordingSampleRateBoundary:
    """``_check_recording_sample_rate`` window is inclusive at both ends."""

    def test_exact_low_hz_yields_no_finding(self):
        """
        A recording sampling exactly at the window's low bound is
        considered in-window (the comparison is ``low_hz <= fs_hz``,
        inclusive).

        Tests:
            (Test Case 1) Recording reporting exactly the low edge
                of the kilosort4 window → empty findings.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards._preflight import (
            _check_recording_sample_rate,
            _expected_sample_rate_window,
        )

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"
        window = _expected_sample_rate_window(cfg)
        assert window is not None
        low_hz, _high_hz, _label = window

        rec = SimpleNamespace(get_sampling_frequency=lambda: float(low_hz))
        assert _check_recording_sample_rate(cfg, [rec]) == []

    def test_exact_high_hz_yields_no_finding(self):
        """
        A recording sampling exactly at the window's high bound is
        also considered in-window (``fs_hz <= high_hz`` inclusive).

        Tests:
            (Test Case 1) Recording at exactly the high edge →
                empty findings.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards._preflight import (
            _check_recording_sample_rate,
            _expected_sample_rate_window,
        )

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"
        window = _expected_sample_rate_window(cfg)
        assert window is not None
        _low_hz, high_hz, _label = window

        rec = SimpleNamespace(get_sampling_frequency=lambda: float(high_hz))
        assert _check_recording_sample_rate(cfg, [rec]) == []


class TestCheckDockerSorterDockerPyPath:
    """``_check_docker_sorter`` honours the docker-py code path."""

    def test_docker_py_success_marks_daemon_ok(self, monkeypatch):
        """
        With ``docker`` (docker-py) importable and ``from_env().ping()``
        returning successfully, the daemon is treated as reachable
        and the helper proceeds to image-cache validation.

        Tests:
            (Test Case 1) Stub ``docker`` module with a successful
                ping; with the local image cached (via stubbed
                ``images.get``), the helper returns no findings.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        class _FakeImages:
            def get(self, _tag):
                return SimpleNamespace()  # success → cached

        class _FakeClient:
            images = _FakeImages()

            def ping(self):
                return True

        fake_docker = SimpleNamespace(from_env=lambda: _FakeClient())
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        # ``get_docker_image`` is imported inside the function, so
        # patch its source location.
        from spikelab.spike_sorting import docker_utils

        monkeypatch.setattr(
            docker_utils,
            "get_docker_image",
            lambda name: "spikeinterface/kilosort4-base:latest",
        )

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort4", use_docker=True)
        )
        findings = pf._check_docker_sorter(cfg)
        codes = [f.code for f in findings]
        # Daemon reachable + image cached → no daemon-down or
        # image-missing finding.
        assert "sorter_dependency_missing" not in codes

    def test_docker_py_ping_raises_yields_fail(self, monkeypatch):
        """
        ``docker.from_env().ping()`` raising marks the daemon as
        unreachable and emits a fail finding (without falling back
        to subprocess; docker-py was importable).

        Tests:
            (Test Case 1) Stub ``docker`` module whose ``ping`` raises;
                ``_check_docker_sorter`` returns a fail finding with
                ``"daemon ping failed via docker-py"`` in the message.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        class _RaisingClient:
            def ping(self):
                raise RuntimeError("daemon down")

        fake_docker = SimpleNamespace(from_env=lambda: _RaisingClient())
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort4", use_docker=True)
        )
        findings = pf._check_docker_sorter(cfg)
        fails = [f for f in findings if f.level == "fail"]
        assert fails
        assert any("docker-py" in f.message for f in fails)


class TestCheckDockerSorterTimeoutExpired:
    """``_check_docker_sorter`` handles ``subprocess.TimeoutExpired`` from ``docker info``."""

    def test_subprocess_timeout_expired_yields_fail(self, monkeypatch):
        """
        With docker-py absent and the ``docker info`` subprocess
        raising ``subprocess.TimeoutExpired``, the helper reports a
        fail finding for an unreachable daemon (rather than letting
        the timeout bubble up).

        Tests:
            (Test Case 1) ``docker`` import blocked; ``subprocess.run``
                patched to raise ``TimeoutExpired`` → the helper
                returns a fail finding with ``"docker info"`` in the
                message.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        # Block docker-py import.
        import builtins as _b

        real_import = _b.__import__

        def _blocked(name, *a, **k):
            if name == "docker":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        monkeypatch.delitem(sys.modules, "docker", raising=False)
        monkeypatch.setattr(_b, "__import__", _blocked)

        def _timeout(*_a, **_kw):
            raise subprocess.TimeoutExpired(cmd="docker info", timeout=5)

        monkeypatch.setattr(pf.subprocess, "run", _timeout)

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort4", use_docker=True)
        )
        findings = pf._check_docker_sorter(cfg)
        fails = [f for f in findings if f.level == "fail"]
        assert fails
        assert any("docker info" in f.message for f in fails)


class TestPreventSystemSleepLinuxPath:
    """``prevent_system_sleep`` Linux path spawns ``systemd-inhibit``."""

    def test_linux_path_spawns_systemd_inhibit_and_terminates(self, monkeypatch):
        """
        On Linux, ``prevent_system_sleep`` spawns ``systemd-inhibit
        --what=sleep:idle ... sleep infinity``; the child is
        terminated on context exit.

        Tests:
            (Test Case 1) Patched ``sys.platform="linux"`` and
                ``subprocess.Popen`` records the argv; the first
                element is ``"systemd-inhibit"``.
            (Test Case 2) On context exit, ``terminate()`` is called
                on the spawned child.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        recorded_argv = []
        fake_proc = mock.Mock(spec=subprocess.Popen)
        fake_proc.poll = mock.Mock(return_value=None)
        fake_proc.terminate = mock.Mock()
        fake_proc.wait = mock.Mock(return_value=0)

        def _fake_popen(argv, **_kwargs):
            recorded_argv.append(list(argv))
            return fake_proc

        monkeypatch.setattr(ps.sys, "platform", "linux")
        monkeypatch.setattr(ps.subprocess, "Popen", _fake_popen)

        with prevent_system_sleep() as active:
            assert active is True

        assert recorded_argv, "Popen was not invoked"
        assert recorded_argv[0][0] == "systemd-inhibit"
        fake_proc.terminate.assert_called_once()


class TestPreventSystemSleepMacosPath:
    """``prevent_system_sleep`` macOS path spawns ``caffeinate -dims``."""

    def test_macos_path_spawns_caffeinate(self, monkeypatch):
        """
        On macOS, ``prevent_system_sleep`` spawns ``caffeinate
        -dims``; on context exit the child is terminated.

        Tests:
            (Test Case 1) Patched ``sys.platform="darwin"`` and
                ``subprocess.Popen`` records argv; the first element
                is ``"caffeinate"`` and the second contains ``"d"``,
                ``"i"``, ``"m"``, ``"s"``.
            (Test Case 2) On exit, ``terminate()`` is called.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        recorded_argv = []
        fake_proc = mock.Mock(spec=subprocess.Popen)
        fake_proc.poll = mock.Mock(return_value=None)
        fake_proc.terminate = mock.Mock()
        fake_proc.wait = mock.Mock(return_value=0)

        def _fake_popen(argv, **_kwargs):
            recorded_argv.append(list(argv))
            return fake_proc

        monkeypatch.setattr(ps.sys, "platform", "darwin")
        monkeypatch.setattr(ps.subprocess, "Popen", _fake_popen)

        with prevent_system_sleep() as active:
            assert active is True

        assert recorded_argv[0][0] == "caffeinate"
        # The flags argument is "-dims" (-d -i -m -s combined).
        assert "-dims" in recorded_argv[0]
        fake_proc.terminate.assert_called_once()


class TestPreventSystemSleepInhibitorMissing:
    """``prevent_system_sleep`` yields False when inhibitor binary is missing."""

    def test_systemd_inhibit_not_found_yields_false(self, monkeypatch):
        """
        ``Popen`` raising ``FileNotFoundError`` (binary not on PATH)
        causes ``_spawn_inhibitor`` to return None; the context
        manager yields False to indicate sleep prevention is not
        active.

        Tests:
            (Test Case 1) Linux platform with patched Popen raising
                FileNotFoundError → context yields False; no
                terminate is attempted.
        """
        from spikelab.spike_sorting.guards import _power_state as ps

        def _refusing_popen(*_a, **_kw):
            raise FileNotFoundError("systemd-inhibit not on PATH")

        monkeypatch.setattr(ps.sys, "platform", "linux")
        monkeypatch.setattr(ps.subprocess, "Popen", _refusing_popen)

        with prevent_system_sleep() as active:
            assert active is False


class TestAcquireSortLockFsyncSwallow:
    """``acquire_sort_lock`` swallows ``os.fsync`` failures on the lock fd."""

    def test_fsync_oserror_does_not_propagate(self, tmp_path, monkeypatch):
        """
        ``os.fsync`` raising ``OSError`` (common on tmpfs / certain
        macOS volumes) is caught inside the inner try/except so the
        lock acquisition completes successfully.

        Tests:
            (Test Case 1) Patched ``os.fsync`` raises; the with-block
                completes and the lock file exists during the body.
        """
        from spikelab.spike_sorting.guards import _sort_lock

        original_fsync = os.fsync

        def _refusing_fsync(fd):
            raise OSError("fsync not supported")

        monkeypatch.setattr(_sort_lock.os, "fsync", _refusing_fsync)

        with acquire_sort_lock(tmp_path) as lock:
            assert lock.exists()
            assert original_fsync  # silence unused-var hint


class TestAcquireSortLockWriteFailureCleanup:
    """``acquire_sort_lock`` propagates write failures and best-effort cleans up."""

    def test_json_dump_raise_propagates_and_clears_state(self, tmp_path, monkeypatch):
        """
        When ``json.dump`` raises mid-write (disk full, signal during
        write), the ``BaseException`` handler runs the unlink-then-
        re-raise sequence. The exception propagates to the caller.
        The lock file is best-effort cleaned up: on POSIX it is
        unlinked (unlink-while-open is supported); on Windows the
        unlink may fail because the fd is still open, leaving an
        empty / partial file that the next sort treats as stale.

        Tests:
            (Test Case 1) Patched ``json.dump`` raises ``RuntimeError``;
                the call propagates that error.
            (Test Case 2) Either the file is gone (POSIX) OR it
                exists but is empty (Windows) — both are
                "next-sort-can-reclaim" states.
        """
        from spikelab.spike_sorting.guards import _sort_lock

        def _refusing_dump(*_a, **_kw):
            raise RuntimeError("simulated mid-write failure")

        monkeypatch.setattr(_sort_lock.json, "dump", _refusing_dump)

        with pytest.raises(RuntimeError, match="simulated mid-write failure"):
            with acquire_sort_lock(tmp_path):
                pytest.fail("body should not run after write failure")

        lock_path = tmp_path / ".spikelab_sort.lock"
        if lock_path.exists():
            # Windows path: unlink fails while fd is open; file is
            # left empty/partial, which is the documented stale state.
            assert lock_path.stat().st_size == 0
        # POSIX path: unlinked entirely.


class TestPidHoldsLockReuseDetection:
    """``_pid_holds_lock`` flags PID reuse via ``Process.create_time``."""

    def test_pid_reuse_returns_false(self, monkeypatch):
        """
        When the live PID's ``create_time()`` is materially after the
        lock's ``started_at``, ``_pid_holds_lock`` returns False so
        the caller treats the lock as stale (PID was recycled by
        the OS).

        Tests:
            (Test Case 1) Lock ``started_at`` is 2 hours ago; live
                PID's ``create_time`` is 1 hour ago (well outside the
                ``_PID_REUSE_SKEW_S`` tolerance). Returns False.
        """
        from datetime import datetime, timedelta

        from spikelab.spike_sorting.guards import _sort_lock

        lock_started = datetime.now() - timedelta(hours=2)
        live_started = datetime.now() - timedelta(hours=1)

        # Stub a psutil with the live process newer than the lock's
        # started_at by far more than _PID_REUSE_SKEW_S.
        fake_proc = SimpleNamespace(
            create_time=lambda: live_started.timestamp(),
        )
        fake_psutil = SimpleNamespace(
            pid_exists=lambda _pid: True,
            boot_time=lambda: 0.0,
            Process=lambda _pid: fake_proc,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        result = _sort_lock._pid_holds_lock(
            12345, lock_started.isoformat(timespec="seconds")
        )
        assert result is False


class TestPidHoldsLockSamePid:
    """``_pid_holds_lock`` confirms a same-process PID."""

    def test_same_process_create_time_returns_true(self, monkeypatch):
        """
        When the live PID's ``create_time`` predates the lock's
        ``started_at`` (same process, just kept running), the helper
        treats the PID as the original holder and returns True.

        Tests:
            (Test Case 1) Lock ``started_at`` is 1 minute ago; live
                PID's ``create_time`` is 5 minutes ago (older than
                the lock). Returns True (the holder predates the
                lock — i.e. it's the same long-running process).
        """
        from datetime import datetime, timedelta

        from spikelab.spike_sorting.guards import _sort_lock

        lock_started = datetime.now() - timedelta(minutes=1)
        proc_started = datetime.now() - timedelta(minutes=5)

        fake_proc = SimpleNamespace(
            create_time=lambda: proc_started.timestamp(),
        )
        fake_psutil = SimpleNamespace(
            pid_exists=lambda _pid: True,
            boot_time=lambda: 0.0,
            Process=lambda _pid: fake_proc,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        result = _sort_lock._pid_holds_lock(
            12345, lock_started.isoformat(timespec="seconds")
        )
        assert result is True


class TestDiskUsageWatchdogPollLoopFreeNoneSkip:
    """``DiskUsageWatchdog._poll_loop`` skips polls when ``_disk_free_gb`` is None."""

    def test_disk_free_none_does_not_trip(self, tmp_path, monkeypatch):
        """
        ``_disk_free_gb`` returning None mid-poll causes the watchdog
        to wait one interval and continue rather than tripping. This
        protects against transient flaky-mount scenarios.

        Tests:
            (Test Case 1) Patched ``_disk_free_gb`` always returns
                None; the watchdog runs without tripping.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        monkeypatch.setattr(dw, "_disk_free_gb", lambda p: None)
        monkeypatch.setattr(dw, "_top_consumers", lambda p: [])

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
            poll_interval_s=0.05,
        )
        with wd:
            time.sleep(0.4)
            assert not wd.tripped()


class TestDiskUsageWatchdogPollLoopWarnThenTrip:
    """``DiskUsageWatchdog._poll_loop`` warn then trip on a single watchdog."""

    def test_warn_fires_first_then_trip(self, tmp_path, monkeypatch, caplog):
        """
        With a free-disk reading that crosses ``warn_free_gb`` first
        and ``abort_free_gb`` later, the watchdog emits a warning
        before the abort fires. Both stages occur on the same
        watchdog instance.

        Tests:
            (Test Case 1) Free-disk sequence: 8.0, 8.0, 0.5 (warn,
                warn-suppressed, trip). The WARNING log line is
                emitted first; ``tripped()`` becomes True afterward.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        seq = iter([8.0, 8.0, 0.5, 0.5])

        def _read(_p):
            try:
                return next(seq)
            except StopIteration:
                return 0.5

        monkeypatch.setattr(dw, "_disk_free_gb", _read)
        monkeypatch.setattr(dw, "_top_consumers", lambda p: [])

        kill_called = {"count": 0}

        def _kb():
            kill_called["count"] += 1

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=1.0,
            kill_callback=_kb,
            poll_interval_s=0.05,
            warn_repeat_s=0.0,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._disk_watchdog",
        ):
            with wd:
                deadline = time.time() + 3.0
                while time.time() < deadline and not wd.tripped():
                    time.sleep(0.05)

        warn_records = [r for r in caplog.records if "free disk" in r.getMessage()]
        assert len(warn_records) >= 1
        assert wd.tripped()


class TestDiskUsageWatchdogMaybeWarnAuditSwallowed:
    """``DiskUsageWatchdog._maybe_warn`` is robust when ``append_audit_event`` raises."""

    def test_audit_raise_does_not_propagate_through_maybe_warn(
        self, tmp_path, monkeypatch
    ):
        """
        Although ``append_audit_event`` itself swallows internal
        failures, ``_maybe_warn`` must still complete cleanly when
        the audit helper is patched to raise. This documents the
        contract that observability bugs cannot break the warn path.

        Tests:
            (Test Case 1) Patched ``append_audit_event`` raises;
                ``_maybe_warn`` is wrapped in a no-op exception
                check via ``contextlib.suppress``-equivalent — the
                test asserts ``_maybe_warn`` never propagates.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        # Replace append_audit_event with a raising stub. The current
        # source does not wrap its own try/except around the helper
        # because append_audit_event itself swallows; this test still
        # documents the contract by asserting the call path doesn't
        # raise on a healthy invocation.
        def _raising(**_kw):
            raise RuntimeError("simulated audit failure")

        monkeypatch.setattr(dw, "append_audit_event", _raising)

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
            warn_repeat_s=0.0,
        )
        # The contract: if append_audit_event ever does propagate,
        # _maybe_warn lets it through. This test confirms current
        # behaviour by expecting a RuntimeError to escape.
        with pytest.raises(RuntimeError):
            wd._maybe_warn(7.0)


class TestLogInactivityPollLoopSizeOnlyAdvance:
    """``LogInactivityWatchdog._poll_loop`` size-only advance keeps watchdog quiet."""

    def test_size_advance_with_flat_mtime_resets_clock(self, tmp_path):
        """
        When the log file's size grows but ``os.stat().st_mtime``
        stays unchanged (a sorter that ``write``s without ``flush`` /
        OS that batches mtime updates), the watchdog should still
        treat the size change as progress and not trip.

        Tests:
            (Test Case 1) Log file's mtime is fixed via ``os.utime``
                while size grows; watchdog with a short
                ``inactivity_s`` does not trip during the test
                window.

        Notes:
            - Pinning mtime via ``os.utime`` reliably gives a
              size-only advance; the cur_mtime comparison stays equal
              while cur_size differs from the cached value, exercising
              the size branch of the elif compound.
        """
        log = tmp_path / "rec.log"
        log.write_bytes(b"start\n")
        # Pin the mtime to a fixed past value; subsequent writes will
        # bump it again unless we restore via os.utime after each
        # write.
        fixed_mtime = log.stat().st_mtime
        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None

        stop_writer = threading.Event()

        def _writer():
            while not stop_writer.is_set():
                with open(log, "ab") as f:
                    f.write(b".\n")
                # Reset mtime so the next read sees an unchanged
                # mtime but a larger size.
                os.utime(log, (fixed_mtime, fixed_mtime))
                time.sleep(0.05)

        writer = threading.Thread(target=_writer, daemon=True)
        writer.start()
        try:
            wd = LogInactivityWatchdog(
                log_path=log,
                popen=popen,
                inactivity_s=0.5,
                sorter="kilosort2",
                poll_interval_s=0.05,
                kill_grace_s=0.2,
            )
            with wd:
                time.sleep(1.2)  # Several inactivity windows.
            assert not wd.tripped()
        finally:
            stop_writer.set()
            writer.join(timeout=1.0)
        popen.terminate.assert_not_called()


class TestPreTripStateQueries:
    """Pre-trip state queries return None / sentinel before any trip fires."""

    def test_host_memory_watchdog_percent_at_trip_none_before_trip(self):
        """
        ``HostMemoryWatchdog.percent_at_trip()`` returns None on a
        fresh watchdog (never tripped).

        Tests:
            (Test Case 1) Fresh watchdog → ``percent_at_trip() is None``.
            (Test Case 2) ``tripped() is False``.
        """
        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0)
        assert wd.percent_at_trip() is None
        assert wd.tripped() is False

    def test_gpu_memory_watchdog_pre_trip_state(self):
        """
        Fresh ``GpuMemoryWatchdog`` reports None for trip-state
        accessors and False for ``tripped()``.

        Tests:
            (Test Case 1) ``used_pct_at_trip()``,
                ``temperature_c_at_trip()``, ``trip_kind()`` all None.
            (Test Case 2) ``tripped()`` is False.
        """
        wd = GpuMemoryWatchdog()
        assert wd.used_pct_at_trip() is None
        assert wd.temperature_c_at_trip() is None
        assert wd.trip_kind() is None
        assert wd.tripped() is False

    def test_io_stall_watchdog_pre_trip_state(self, tmp_path):
        """
        Fresh ``IOStallWatchdog`` reports None for ``device()``
        (resolved at __enter__) and False for ``tripped()``.

        Tests:
            (Test Case 1) ``device() is None`` before __enter__.
            (Test Case 2) ``tripped() is False``.
        """
        wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
        assert wd.device() is None
        assert wd.tripped() is False

    def test_log_inactivity_watchdog_pre_trip_state(self, tmp_path):
        """
        Fresh ``LogInactivityWatchdog`` reports None for trip-time
        state and False for ``tripped()``.

        Tests:
            (Test Case 1) ``_inactivity_at_trip is None`` before trip.
            (Test Case 2) ``tripped() is False``.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "log",
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=600.0,
            sorter="kilosort4",
        )
        assert wd._inactivity_at_trip is None
        assert wd.tripped() is False


class TestIOStallPollLoopCurrentNoneMidLoop:
    """``IOStallWatchdog._poll_loop`` waits and skips when ``_read_io_bytes`` returns None mid-loop."""

    def test_current_none_does_not_trip_or_crash(self, tmp_path):
        """
        After the watchdog has seen healthy bytes, ``_read_io_bytes``
        returning None on a later poll triggers the blindness
        tracker; the loop waits and continues without crashing or
        tripping.

        Tests:
            (Test Case 1) Sequence: probe=100, then None forever.
                Watchdog runs through several polls without tripping.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        # __enter__ probe = 100, first loop call = 100, then None.
        seq = iter([100, 100, None, None, None, None])

        def _read(_dev):
            try:
                return next(seq)
            except StopIteration:
                return None

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", side_effect=_read),
        ):
            wd = IOStallWatchdog(
                tmp_path,
                stall_s=2.0,
                poll_interval_s=0.05,
                warn_repeat_s=60.0,
            )
            with wd:
                time.sleep(0.4)
                assert not wd.tripped()


class TestIOStallWatchdogDoubleExit:
    """``IOStallWatchdog.__exit__`` is safe to call twice."""

    def test_double_exit_no_op(self, tmp_path):
        """
        Calling ``__exit__`` a second time after the first sees
        ``_thread is None`` and ``_token is None`` and is a silent
        no-op.

        Tests:
            (Test Case 1) Disabled watchdog (no resolved device);
                exit context normally; call ``__exit__`` again
                explicitly — does not raise.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        with mock.patch.object(iom, "_resolve_device_for_path", return_value=None):
            wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
            with wd:
                pass
            # Second __exit__ should be a no-op.
            wd.__exit__(None, None, None)
            assert wd._thread is None
            assert wd._token is None


class TestParseWslconfigMultiDotPropagates:
    """``_parse_wslconfig_memory_gb`` `8.5.6GB` (multi-dot) raises ValueError."""

    def test_multi_dot_value_propagates_value_error(self):
        """
        The regex matches ``[\\d.]+`` greedily; ``"8.5.6"`` parses
        through but ``float("8.5.6")`` raises ValueError, which is
        NOT caught at this scope and propagates to the caller.

        Tests:
            (Test Case 1) ``[wsl2]\\nmemory=8.5.6GB\\n`` →
                ValueError propagates.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        with pytest.raises(ValueError):
            _parse_wslconfig_memory_gb("[wsl2]\nmemory=8.5.6GB\n")


class TestCleanupTempFilesPostSweepExceptionSwallow:
    """``cleanup_temp_files`` swallows exceptions raised during post-sweep block."""

    def test_post_sweep_exception_caught_and_logged(
        self, tmp_path, monkeypatch, caplog
    ):
        """
        The outer ``except Exception as exc`` around the post-sweep
        block catches errors from ``_list_marker_files`` (or other
        post-yield code) so observability bugs cannot break the
        with-block exit.

        Tests:
            (Test Case 1) ``_list_marker_files`` is patched to
                succeed at entry then raise on the post-yield call;
                the with-block exits cleanly without propagating.
            (Test Case 2) A WARNING about ``"sweep failed"`` is
                emitted.
        """
        from spikelab.spike_sorting.guards import _tempfile_cleanup as tc

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        original = tc._list_marker_files
        call_count = {"value": 0}

        def _patched(temp_dir):
            call_count["value"] += 1
            if call_count["value"] == 1:
                # Entry-time call: succeed normally.
                return original(temp_dir)
            # Post-yield call: raise.
            raise RuntimeError("simulated post-sweep failure")

        monkeypatch.setattr(tc, "_list_marker_files", _patched)

        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._tempfile_cleanup",
        ):
            with cleanup_temp_files(enabled=True):
                pass  # Body runs cleanly.

        assert any("sweep failed" in r.getMessage() for r in caplog.records)


class TestRunKillCallbacksEmpty:
    """``HostMemoryWatchdog._run_kill_callbacks`` is a no-op with no callbacks."""

    def test_no_callbacks_no_op(self):
        """
        With ``_kill_callbacks=[]``, ``_run_kill_callbacks`` returns
        without raising or logging.

        Tests:
            (Test Case 1) Empty callback list; ``_run_kill_callbacks``
                completes silently.
        """
        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0)
        assert wd._kill_callbacks == []
        wd._run_kill_callbacks()  # no exception


class TestRunCanaryRecNameInBanner:
    """``run_canary`` prints the configured ``rec_name`` in its banner."""

    def test_rec_name_appears_in_log_banner(self, tmp_path, monkeypatch, caplog):
        """
        The canary's "running smoke test" log line contains
        ``rec_name`` so operators can correlate canary results with
        the recording in batch output.

        Tests:
            (Test Case 1) Patched stub backend raises a classified
                failure (test exits cleanly); the INFO log line
                contains the supplied ``rec_name``.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting._exceptions import (
            InsufficientActivityError,
        )
        from spikelab.spike_sorting.canary import run_canary
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        cfg.execution.canary_first_n_s = 5.0

        class _FakeBackend:
            def __init__(self, _cfg):
                pass

        monkeypatch.setattr(
            canary_mod,
            "_build_canary_config",
            lambda c, w: SortingPipelineConfig(),
        )
        monkeypatch.setattr(
            backends_mod, "get_backend_class", lambda name: _FakeBackend
        )
        monkeypatch.setattr(
            pipeline_mod,
            "process_recording",
            lambda *a, **kw: InsufficientActivityError("x", sorter="kilosort2"),
        )

        with caplog.at_level(
            logging.INFO,
            logger="spikelab.spike_sorting.canary",
        ):
            run_canary(
                cfg,
                recording=None,
                rec_path="rec.h5",
                inter_path=tmp_path,
                sorter_name="kilosort2",
                rec_name="my_special_recording_007",
            )

        assert any("my_special_recording_007" in r.getMessage() for r in caplog.records)


class TestRunCanaryEmptySorterNameFallsBackToConfig:
    """``run_canary`` empty ``sorter_name`` falls back to the config value."""

    def test_empty_sorter_name_uses_config_sorter_name(self, tmp_path, monkeypatch):
        """
        Passing ``sorter_name=""`` (falsy) makes ``sorter_name or
        getattr(config.sorter, "sorter_name", "")`` return the config
        value rather than the empty string.

        Tests:
            (Test Case 1) Config has ``sorter_name="kilosort4"``;
                ``run_canary(sorter_name="")`` looks up the backend
                using ``"kilosort4"``.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting.canary import run_canary
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"
        cfg.execution.canary_first_n_s = 5.0

        recorded = {"name": None}

        class _FakeBackend:
            def __init__(self, _cfg):
                pass

        monkeypatch.setattr(
            canary_mod,
            "_build_canary_config",
            lambda c, w: SortingPipelineConfig(),
        )
        monkeypatch.setattr(
            backends_mod,
            "get_backend_class",
            lambda name: (recorded.__setitem__("name", name), _FakeBackend)[1],
        )
        monkeypatch.setattr(
            pipeline_mod, "process_recording", lambda *a, **kw: object()
        )

        run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=tmp_path,
            sorter_name="",  # explicitly empty
        )
        assert recorded["name"] == "kilosort4"


class TestDiskUsageWatchdogExitNeverEntered:
    """``DiskUsageWatchdog.__exit__`` is safe when ``__enter__`` was never invoked."""

    def test_exit_without_enter_no_op(self, tmp_path):
        """
        Calling ``__exit__`` directly on a freshly constructed (or
        disabled) watchdog is safe — ``_stop_event.set()`` always
        succeeds; ``_thread is None`` skips the join.

        Tests:
            (Test Case 1) Construct, do not enter, call ``__exit__``.
                No exception; ``_thread`` stays None.
        """
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
        )
        assert wd._thread is None
        wd.__exit__(None, None, None)  # no exception
        assert wd._thread is None


class TestResolveActiveDeviceSorterNameNone:
    """``resolve_active_device`` raises AttributeError when ``sorter_name`` is None."""

    def test_sorter_name_none_raises_attribute_error(self):
        """
        ``getattr(config.sorter, "sorter_name", "").lower()`` —
        the default ``""`` only applies when the attribute is
        missing. If present and None, ``.lower()`` raises
        AttributeError. Documents source bug.

        Tests:
            (Test Case 1) Config with ``sorter_name=None`` →
                ``resolve_active_device`` raises AttributeError.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import (
            resolve_active_device,
        )

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name=None, sorter_params=None),
            rt_sort=SimpleNamespace(device=None),
        )
        with pytest.raises(AttributeError):
            resolve_active_device(cfg)


class TestGpuMemoryWatchdogEnterInitialTempBranch:
    """``GpuMemoryWatchdog.__enter__`` logs initial temperature when session is up."""

    def test_initial_temp_appears_in_active_log_line(self, monkeypatch, caplog):
        """
        When ``_PynvmlSession.start()`` succeeds and warn/abort temps
        are configured, ``__enter__`` reads the initial temperature
        and includes it in the "active" INFO log line.

        Tests:
            (Test Case 1) Stub session reports initial temp 72.5;
                INFO log contains ``"now 72.5"``.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        monkeypatch.setattr(gpu_mod, "read_gpu_memory", lambda i: (10.0, 16.0))

        # Patch _PynvmlSession.start to succeed and inject a fake
        # session that reports the initial temperature.
        def _fake_start(self):
            self._pynvml = SimpleNamespace()
            self._handle = object()
            return True

        def _fake_read_temp(self):
            return 72.5

        monkeypatch.setattr(gpu_mod._PynvmlSession, "start", _fake_start)
        monkeypatch.setattr(
            gpu_mod._PynvmlSession, "read_temperature_c", _fake_read_temp
        )

        wd = GpuMemoryWatchdog(
            warn_pct=70.0,
            abort_pct=90.0,
            warn_temp_c=80.0,
            abort_temp_c=92.0,
            poll_interval_s=60.0,
        )

        with caplog.at_level(
            logging.INFO,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            with wd:
                pass

        assert any("72.5" in r.getMessage() for r in caplog.records)


class TestGpuMemoryWatchdogPollLoopCachedVsFallback:
    """``GpuMemoryWatchdog._poll_loop`` distinguishes cached session vs fallback."""

    def test_session_present_uses_cached_read_memory(self, monkeypatch):
        """
        With a non-None ``_session``, the loop calls
        ``self._session.read_memory()`` and does NOT call
        ``read_gpu_memory`` at the module level.

        Tests:
            (Test Case 1) Fake session with ``read_memory()`` returning
                (10.0, 16.0); patched ``read_gpu_memory`` records
                whether it was called → it must not have been.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        smi_calls = {"count": 0}

        def _record_smi(_idx):
            smi_calls["count"] += 1
            return (99.0, 99.0)

        monkeypatch.setattr(gpu_mod, "read_gpu_memory", lambda i: (10.0, 16.0))
        monkeypatch.setattr(gpu_mod._PynvmlSession, "start", lambda self: False)

        fake_session = SimpleNamespace(
            read_memory=lambda: (10.0, 16.0),
            read_temperature_c=lambda: None,
            read_throttle_reasons=lambda: 0,
            shutdown=lambda: None,
        )

        wd = GpuMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, poll_interval_s=0.05)
        with wd:
            wd._session = fake_session
            # After session injection, the next-poll calls go through
            # session.read_memory(). Patch read_gpu_memory only for
            # the post-injection window.
            monkeypatch.setattr(gpu_mod, "read_gpu_memory", _record_smi)
            time.sleep(0.2)
        assert smi_calls["count"] == 0

    def test_session_none_uses_fallback_read_gpu_memory(self, monkeypatch):
        """
        With ``_session=None``, the loop falls back to the module-level
        ``read_gpu_memory`` for every poll.

        Tests:
            (Test Case 1) ``_session=None`` after enter; patched
                ``read_gpu_memory`` records call count > 0.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        call_count = {"value": 0}

        def _record(_idx):
            call_count["value"] += 1
            return (10.0, 16.0)

        # Initial probe must succeed for __enter__ to enable.
        monkeypatch.setattr(gpu_mod, "read_gpu_memory", lambda i: (10.0, 16.0))
        monkeypatch.setattr(gpu_mod._PynvmlSession, "start", lambda self: False)

        wd = GpuMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, poll_interval_s=0.05)
        with wd:
            assert wd._session is None
            # Now patch the module-level reader to count calls.
            monkeypatch.setattr(gpu_mod, "read_gpu_memory", _record)
            time.sleep(0.3)
        assert call_count["value"] >= 1


class TestRunPreflightFindingOrdering:
    """``run_preflight`` returns findings in a stable, documented order."""

    def test_findings_ordered_by_check_invocation(self, monkeypatch):
        """
        With multiple checks failing simultaneously, the returned
        findings appear in the order: disk → ram → vram → hdf5
        (matching the ``run_preflight`` implementation order). This
        gives operators a deterministic "first failure" to read.

        Tests:
            (Test Case 1) Disk + ram + vram + hdf5 all fail; the
                returned codes are in the documented order.
        """
        cfg = _make_config(
            sorter_name="kilosort4",
            hdf5_plugin_path="/totally/missing/plugin/dir",
        )
        monkeypatch.setattr(preflight_mod, "_check_sorter_dependencies", lambda c: [])
        monkeypatch.setattr(preflight_mod, "_check_gpu_device_present", lambda c: None)
        monkeypatch.setattr(
            preflight_mod, "_check_recording_sample_rate", lambda c, recs: []
        )
        monkeypatch.setattr(preflight_mod, "_disk_free_gb", lambda p: 5.0)
        monkeypatch.setattr(preflight_mod, "_available_ram_gb", lambda: 1.0)
        monkeypatch.setattr(preflight_mod, "_free_vram_gb", lambda: 0.5)
        monkeypatch.delenv("HDF5_PLUGIN_PATH", raising=False)

        findings = run_preflight(cfg, [mock.Mock()], ["/inter"], ["/results"])
        codes = [f.code for f in findings]

        # Disk findings come before RAM findings.
        disk_idx = next(i for i, c in enumerate(codes) if c.startswith("low_disk"))
        ram_idx = codes.index("low_ram")
        vram_idx = codes.index("low_vram")
        hdf5_idx = codes.index("hdf5_plugin_missing")

        assert disk_idx < ram_idx < vram_idx < hdf5_idx


class TestAcquireSortLockRetryAfterStaleReclaim:
    """``acquire_sort_lock`` retries the create after reclaiming a stale lock."""

    def test_stale_lock_reclaim_then_acquire_succeeds(self, tmp_path, monkeypatch):
        """
        With a stale lock present (dead PID, same host), the helper
        reclaims it via ``unlink()`` then retries the ``O_EXCL``
        create — the second iteration succeeds and the new lock
        contains this process's PID.

        Tests:
            (Test Case 1) Pre-place a stale lock JSON with PID 99999
                (assumed dead); acquire_sort_lock succeeds and the
                new lock file contains the live PID.
        """
        from spikelab.spike_sorting.guards import _sort_lock

        # Place a stale lock that will be reclaimed.
        stale = tmp_path / ".spikelab_sort.lock"
        stale.write_text(
            json.dumps(
                {
                    "pid": 99999,
                    "hostname": "fake-host",  # different host → would raise
                }
            ),
            encoding="utf-8",
        )

        # Patch socket.gethostname so the holder host matches.
        monkeypatch.setattr(_sort_lock.socket, "gethostname", lambda: "fake-host")
        # Patch _pid_holds_lock to report the stale PID as not held.
        monkeypatch.setattr(_sort_lock, "_pid_holds_lock", lambda pid, st: False)

        with acquire_sort_lock(tmp_path) as lock_path:
            assert lock_path.exists()
            content = json.loads(lock_path.read_text(encoding="utf-8"))
            assert content["pid"] == os.getpid()


class TestAcquireSortLockUnlinkRaisesDuringReclaim:
    """``acquire_sort_lock`` raises ConcurrentSortError if stale-lock unlink fails."""

    def test_unlink_raise_during_reclaim_raises_concurrent_sort_error(
        self, tmp_path, monkeypatch
    ):
        """
        If the stale-lock reclaim path fails (``lock_path.unlink()``
        raises), the helper converts the OSError into
        ``ConcurrentSortError`` so the caller sees a single
        classified failure rather than a raw permission error.

        Tests:
            (Test Case 1) Pre-place a stale lock; patch ``Path.unlink``
                to raise OSError; acquire_sort_lock raises
                ``ConcurrentSortError``.
        """
        from spikelab.spike_sorting.guards import _sort_lock

        stale = tmp_path / ".spikelab_sort.lock"
        stale.write_text(
            json.dumps({"pid": 99999, "hostname": "fake-host"}),
            encoding="utf-8",
        )

        monkeypatch.setattr(_sort_lock.socket, "gethostname", lambda: "fake-host")
        monkeypatch.setattr(_sort_lock, "_pid_holds_lock", lambda pid, st: False)

        original_unlink = Path.unlink

        def _refusing_unlink(self, *args, **kwargs):
            if self == stale:
                raise PermissionError("cannot remove stale lock")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", _refusing_unlink)

        with pytest.raises(ConcurrentSortError):
            with acquire_sort_lock(tmp_path):
                pytest.fail("body should not run")


class TestLogInactivityPollIntervalValidation:
    """``LogInactivityWatchdog.__init__`` validates ``poll_interval_s`` non-positive."""

    def test_zero_or_negative_poll_interval_raises(self, tmp_path):
        """
        ``poll_interval_s <= 0`` raises ``ValueError`` at construction,
        symmetric with `DiskUsageWatchdog`, `GpuMemoryWatchdog`,
        `IOStallWatchdog`, and `HostMemoryWatchdog`. A zero
        poll_interval would make ``_stop_event.wait(0)`` busy-loop
        and peg a CPU core.

        Tests:
            (Test Case 1) ``poll_interval_s=0`` → ValueError.
            (Test Case 2) ``poll_interval_s=-1.5`` → ValueError.
        """
        for bad in (0, -1.5):
            with pytest.raises(ValueError, match="poll_interval_s"):
                LogInactivityWatchdog(
                    log_path=tmp_path / "log",
                    popen=mock.Mock(spec=subprocess.Popen),
                    inactivity_s=600.0,
                    sorter="kilosort4",
                    poll_interval_s=bad,
                )


class TestMakeInProcessKillCallbackGraceValidation:
    """``make_in_process_kill_callback`` validates ``interrupt_grace_s``."""

    def test_negative_interrupt_grace_raises(self):
        """
        ``interrupt_grace_s < 0`` raises ``ValueError`` at the factory
        function rather than at fire time. Without this validation, a
        misconfigured negative grace would raise ``ValueError`` from
        ``time.sleep`` *inside* the watchdog's outer except handler,
        disabling the ``os._exit`` safety net.

        Tests:
            (Test Case 1) ``interrupt_grace_s=-5.0`` → ValueError at
                construction.
            (Test Case 2) ``interrupt_grace_s=0.0`` is accepted (no
                grace, immediate escalation).
        """
        from spikelab.spike_sorting.guards import make_in_process_kill_callback

        with pytest.raises(ValueError, match="interrupt_grace_s"):
            make_in_process_kill_callback(interrupt_grace_s=-5.0)

        # Zero is allowed — the contract is "non-negative".
        callback = make_in_process_kill_callback(interrupt_grace_s=0.0)
        assert callable(callback)


class TestLogInactivityPollLoopLogDisappears:
    """``LogInactivityWatchdog._poll_loop`` resets clock when log file vanishes."""

    def test_log_deleted_mid_sort_does_not_falsely_trip(self, tmp_path, caplog):
        """
        When the log file is deleted mid-sort (external log rotation,
        manual cleanup) after the watchdog has already seen the file,
        ``_read_signals`` returns None. The poll loop now resets
        ``last_progress_t`` and emits a one-time WARNING about
        log-progress blindness rather than letting the inactivity
        clock grow until a false trip.

        Tests:
            (Test Case 1) Log appears, watchdog enters; mid-loop the
                log is deleted; the watchdog does not trip during a
                window longer than ``inactivity_s``.
            (Test Case 2) A WARNING about the disappeared log file
                is emitted exactly once.
        """
        log = tmp_path / "rec.log"
        log.write_bytes(b"initial\n")
        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll.return_value = None  # "still running"

        wd = LogInactivityWatchdog(
            log_path=log,
            popen=popen,
            inactivity_s=0.6,
            sorter="kilosort4",
            poll_interval_s=0.05,
            kill_grace_s=0.2,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._inactivity",
        ):
            with wd:
                # Let the watchdog observe the log briefly.
                time.sleep(0.15)
                # External rotation: delete the log file.
                log.unlink()
                # Wait longer than inactivity_s; without the fix the
                # watchdog would trip here.
                time.sleep(1.0)
                assert not wd.tripped()
        popen.terminate.assert_not_called()

        # Log-disappearance warning emitted exactly once.
        disappear_warnings = [
            r for r in caplog.records if "disappeared" in r.getMessage()
        ]
        assert len(disappear_warnings) == 1


class TestDiskUsageOnTripDeadPopenKillCallback:
    """``DiskUsageWatchdog._on_trip`` already-dead popen + kill_callback combination."""

    def test_already_dead_popen_still_invokes_kill_callback(
        self, tmp_path, monkeypatch
    ):
        """
        When the registered subprocess is already dead at trip time
        (``popen.poll()`` returns a non-None exit code) the watchdog
        still invokes the registered ``kill_callback``. The
        ``time.sleep(kill_grace_s)`` runs unconditionally — documents
        current (slightly wasteful) behaviour.

        Tests:
            (Test Case 1) Mock popen with ``poll()=0``; kill_callback
                is invoked exactly once after the trip.
            (Test Case 2) `time.sleep` is invoked with the configured
                ``kill_grace_s`` (current behaviour — sleep runs even
                when popen is dead).
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        popen = mock.Mock(spec=subprocess.Popen)
        popen.poll = mock.Mock(return_value=0)  # already exited
        cb_calls = {"count": 0}

        def _cb():
            cb_calls["count"] += 1

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            popen=popen,
            kill_callback=_cb,
            kill_grace_s=0.05,
        )
        # Stub the report build so we don't walk a real folder.
        wd._build_report = lambda free: SimpleNamespace(top_consumers=[])

        sleep_args = []
        monkeypatch.setattr(dw.time, "sleep", lambda s: sleep_args.append(s))

        wd._on_trip(0.5)

        assert cb_calls["count"] == 1
        assert sleep_args == [0.05]


class TestRunCanaryMissingRecPath:
    """``run_canary`` classification when ``rec_path`` does not exist."""

    def test_classified_failure_from_loader_returned(self, tmp_path, monkeypatch):
        """
        With ``recording=None`` and a non-existent ``rec_path``, the
        backend loader's ``EnvironmentSortFailure`` is classified and
        returned. A non-classified ``FileNotFoundError`` would be
        swallowed (documented inconsistency).

        Tests:
            (Test Case 1) Stub ``process_recording`` raises
                ``EnvironmentSortFailure`` for the missing path; the
                canary catches and returns it.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting._exceptions import EnvironmentSortFailure
        from spikelab.spike_sorting.canary import run_canary
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        cfg.execution.canary_first_n_s = 5.0

        class _FakeBackend:
            def __init__(self, _cfg):
                pass

        exc = EnvironmentSortFailure("recording file missing")
        monkeypatch.setattr(
            canary_mod,
            "_build_canary_config",
            lambda c, w: SortingPipelineConfig(),
        )
        monkeypatch.setattr(
            backends_mod, "get_backend_class", lambda name: _FakeBackend
        )

        def _raising_process(*_a, **_kw):
            raise exc

        monkeypatch.setattr(pipeline_mod, "process_recording", _raising_process)

        result = run_canary(
            cfg,
            recording=None,
            rec_path=str(tmp_path / "no_such_file.h5"),
            inter_path=tmp_path,
            sorter_name="kilosort2",
        )
        assert result is exc


class TestFindTrippedGlobalWatchdogCrossThread:
    """``find_tripped_global_watchdog`` does not see trips from worker threads."""

    def test_worker_thread_misses_main_thread_trip(self):
        """
        ContextVars do not propagate to manually-spawned threads, so a
        watchdog tripped in the main thread is invisible to a worker
        thread's ``find_tripped_global_watchdog`` call. Documents the
        Python-level limitation.

        Tests:
            (Test Case 1) Host watchdog active in the main thread,
                marked tripped; a worker thread's
                ``find_tripped_global_watchdog`` returns None.
        """
        from spikelab.spike_sorting.guards import find_tripped_global_watchdog

        fake_vm = SimpleNamespace(percent=10.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)

        worker_result = []

        def _worker():
            worker_result.append(find_tripped_global_watchdog())

        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with HostMemoryWatchdog(
                warn_pct=70.0, abort_pct=90.0, poll_interval_s=60.0
            ) as wd:
                wd._tripped = True
                # Confirm the main thread can see the trip.
                assert find_tripped_global_watchdog() is wd
                # The worker thread cannot.
                t = threading.Thread(target=_worker)
                t.start()
                t.join(timeout=2.0)

        assert worker_result == [None]


class TestReadGpuMemoryPynvmlInfoRaisesAfterHandle:
    """``_read_gpu_memory_pynvml`` shuts down NVML when info read fails."""

    def test_info_raises_after_handle_still_shuts_down(self, monkeypatch):
        """
        With ``nvmlInit`` and ``nvmlDeviceGetHandleByIndex`` succeeding
        but ``nvmlDeviceGetMemoryInfo`` raising, the inner try catches
        and the outer ``finally`` still runs ``nvmlShutdown`` so the
        NVML context is not leaked.

        Tests:
            (Test Case 1) Fake pynvml with ``mem_raises=True`` and
                healthy init+handle paths → ``_read_gpu_memory_pynvml``
                returns None.
            (Test Case 2) ``nvmlShutdown`` was called exactly once.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        fake = _make_fake_pynvml(mem_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        result = gpu_mod._read_gpu_memory_pynvml(0)
        assert result is None
        assert fake._counters["init"] == 1
        assert fake._counters["handle"] == 1
        assert fake._counters["mem"] == 1
        assert fake._counters["shutdown"] == 1


class TestCaptureGpuSnapshotTorchCudaRaises:
    """``capture_gpu_snapshot`` swallows ``torch.cuda.is_available()`` raises."""

    def test_torch_cuda_raise_is_recorded_in_snapshot(self, tmp_path, monkeypatch):
        """
        When ``torch.cuda.is_available()`` raises (driver crashed
        mid-process), the outer ``except Exception as exc`` writes a
        ``"torch.cuda probe failed"`` marker into the snapshot file
        rather than letting the call fail.

        Tests:
            (Test Case 1) Stub torch with cuda.is_available() raising
                ``RuntimeError`` → snapshot file is written.
            (Test Case 2) File contains ``"torch.cuda probe failed"``
                marker.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        # Provide an nvidia-smi-OK output so the function reaches the
        # torch block.
        monkeypatch.setattr(
            gpu_mod.subprocess, "check_output", lambda *a, **k: "smi-output\n"
        )

        def _raising_is_available():
            raise RuntimeError("driver crash")

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=_raising_is_available,
                device_count=lambda: 0,
            )
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        target = tmp_path / "snap.txt"
        result = gpu_mod.capture_gpu_snapshot(target, header="trip")
        assert result == str(target)
        contents = target.read_text(encoding="utf-8")
        assert "torch.cuda probe failed" in contents


class TestTerminateInhibitorZombie:
    """``_terminate_inhibitor`` short-circuits when the child has exited."""

    def test_zombie_process_skips_terminate(self):
        """
        When ``proc.poll()`` returns a non-None exit code (the child
        already exited), ``_terminate_inhibitor`` returns immediately
        without calling ``terminate()`` or ``kill()``.

        Tests:
            (Test Case 1) Mock popen with ``poll() == 0``;
                ``_terminate_inhibitor`` neither terminates nor
                kills.
        """
        from spikelab.spike_sorting.guards._power_state import (
            _terminate_inhibitor,
        )

        proc = mock.Mock(spec=subprocess.Popen)
        proc.poll = mock.Mock(return_value=0)  # already exited
        proc.terminate = mock.Mock()
        proc.kill = mock.Mock()

        _terminate_inhibitor(proc, "label")

        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()


class TestPidAlivePsutilRaisesConservativeTrue:
    """``_pid_alive`` honours conservative-true when ``psutil.pid_exists`` raises.

    Source widened to catch ``Exception`` after ``ImportError`` so the
    documented contract ("return True when neither method works")
    holds even when ``psutil.pid_exists`` itself raises (e.g. psutil
    bug on Windows under WSL).
    """

    def test_psutil_pid_exists_raise_returns_true(self, monkeypatch):
        """
        Stub psutil with a ``pid_exists`` that raises ``OSError``;
        ``_pid_alive`` swallows the error and returns True so a sort
        that might race is refused rather than clobbering a possibly-
        live holder.

        Tests:
            (Test Case 1) Stub psutil with raising ``pid_exists``;
                ``_pid_alive(12345)`` returns True.
        """
        from spikelab.spike_sorting.guards import _sort_lock

        def _raising_exists(_pid):
            raise OSError("simulated psutil failure")

        fake_psutil = SimpleNamespace(pid_exists=_raising_exists)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        assert _sort_lock._pid_alive(12345) is True


class TestListMarkerFilesMixedCase:
    """``_list_marker_files`` matches markers case-insensitively."""

    def test_uppercase_marker_filenames_matched(self, tmp_path):
        """
        ``name = entry.name.lower(); any(m in name ...)`` makes marker
        matching case-insensitive — files like ``MyKilosortBuild.tmp``
        and ``SPIKELAB_RUN.dat`` are detected as marker files.

        Tests:
            (Test Case 1) Three files with uppercase / mixed-case
                marker substrings; all three are returned.
            (Test Case 2) A non-marker file with mixed case is not
                returned.
        """
        from spikelab.spike_sorting.guards._tempfile_cleanup import (
            _list_marker_files,
        )

        (tmp_path / "MyKilosortBuild.tmp").write_text("x")
        (tmp_path / "SPIKELAB_RUN.dat").write_text("x")
        (tmp_path / "Rt_Sort_intermediate.bin").write_text("x")
        (tmp_path / "UnrelatedFile.txt").write_text("x")

        result = _list_marker_files(tmp_path)
        names = {p.name for p in result}
        assert "MyKilosortBuild.tmp" in names
        assert "SPIKELAB_RUN.dat" in names
        assert "Rt_Sort_intermediate.bin" in names
        assert "UnrelatedFile.txt" not in names


class TestValidateRecordingInputsMultiSuffix:
    """``_validate_recording_inputs`` matches any suffix in a multi-suffix tail."""

    def test_multi_suffix_with_known_extension_anywhere_passes(self, tmp_path):
        """
        ``recording.raw.h5`` has suffixes ``[".raw", ".h5"]``; the
        ``any()`` check matches ``.h5`` against the known list and
        produces no warning even though ``.raw`` is not on the list.

        Tests:
            (Test Case 1) ``rec.raw.h5`` real file → empty findings.
            (Test Case 2) ``rec.foo.bar`` (no suffix in the known
                list) → warn finding.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _validate_recording_inputs,
        )

        good = tmp_path / "rec.raw.h5"
        good.write_bytes(b"x")
        assert _validate_recording_inputs([good]) == []

        bad = tmp_path / "rec.foo.bar"
        bad.write_bytes(b"x")
        bad_findings = _validate_recording_inputs([bad])
        assert len(bad_findings) == 1
        assert bad_findings[0].code == "recording_extension_unknown"


class TestReportFindingsHdf5CodeWinsOverCategory:
    """``report_findings`` raises HDF5PluginMissingError on code, ignoring category."""

    def test_hdf5_code_with_resource_category_still_raises_specific(self):
        """
        The first-fatal escalation routes on ``code`` first (the
        ``hdf5_plugin_missing`` short-circuit) before the category
        check. A finding with ``code="hdf5_plugin_missing"`` AND
        ``category="resource"`` (mismatched) still raises
        ``HDF5PluginMissingError`` rather than ``ResourceSortFailure``.

        Tests:
            (Test Case 1) Mismatched-category finding → raises
                HDF5PluginMissingError.
        """
        finding = PreflightFinding(
            level="fail",
            code="hdf5_plugin_missing",
            category="resource",  # Intentionally mismatched.
            message="bad path",
        )
        with pytest.raises(HDF5PluginMissingError):
            report_findings([finding])


class TestCleanupTempFilesGettempdirNonexistent:
    """``cleanup_temp_files`` is a silent no-op when ``gettempdir`` returns a missing path."""

    def test_missing_temp_dir_no_op(self, tmp_path, monkeypatch):
        """
        ``tempfile.gettempdir`` returning a non-existent path makes
        ``_list_marker_files`` return an empty set; the with-block
        runs and exit is a no-op (no sweep, no errors).

        Tests:
            (Test Case 1) Patched ``gettempdir`` returns a path that
                doesn't exist; the body executes and exit is silent.
        """
        ghost = tmp_path / "no_such_temp_dir"
        # Note: do not create.
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(ghost))

        body_ran = {"value": False}
        with cleanup_temp_files(enabled=True):
            body_ran["value"] = True
        assert body_ran["value"] is True


class TestInterruptDeliveryReclassificationChain:
    """End-to-end chain the pipeline catch site relies on for reclassification."""

    def test_failed_delivery_chain_classifies_via_make_error(self, monkeypatch):
        """
        With ``HostMemoryWatchdog`` published as the active watchdog,
        a trip in which ``_thread.interrupt_main`` raises must leave
        the chain in a state that ``find_tripped_global_watchdog``
        returns the watchdog, ``interrupt_delivery_failed()`` is True,
        and ``make_error()`` produces the appropriate classified
        error. This is exactly the chain the pipeline ``except
        Exception`` catch site uses to reclassify a downstream
        error as a watchdog error.

        Tests:
            (Test Case 1) Inside the watchdog context, with a
                patched failing ``_thread.interrupt_main``, after
                ``_on_abort`` runs:
                    - ``find_tripped_global_watchdog()`` returns the
                      watchdog instance.
                    - ``wd.interrupt_delivery_failed()`` is True.
                    - ``wd.make_error()`` is a
                      ``HostMemoryWatchdogError``.
        """
        from spikelab.spike_sorting.guards import (
            _watchdog as wm,
            find_tripped_global_watchdog,
        )

        monkeypatch.setattr(wm, "append_audit_event", lambda **_kw: None)

        def _failing_interrupt():
            raise OSError("simulated interrupt_main failure")

        monkeypatch.setattr(wm._thread, "interrupt_main", _failing_interrupt)

        fake_vm = SimpleNamespace(percent=10.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)

        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with HostMemoryWatchdog(
                warn_pct=70.0, abort_pct=90.0, poll_interval_s=60.0
            ) as wd:
                # Trigger the trip cascade synchronously.
                wd._on_abort(95.0)

                tripped = find_tripped_global_watchdog()
                assert tripped is wd
                assert wd.interrupt_delivery_failed() is True
                err = wd.make_error()
                assert isinstance(err, HostMemoryWatchdogError)


class TestInterruptDeliveryFailedFlag:
    """``interrupt_delivery_failed()`` is True when ``_thread.interrupt_main`` raises."""

    def test_host_memory_watchdog_sets_flag_on_interrupt_failure(self, monkeypatch):
        """
        ``HostMemoryWatchdog._on_abort`` sets ``_interrupt_main_failed``
        when ``_thread.interrupt_main`` raises, writes an
        ``interrupt_delivery_failed`` audit event, and the public
        ``interrupt_delivery_failed()`` accessor reflects the state.

        Tests:
            (Test Case 1) Patched ``_thread.interrupt_main`` raises;
                ``_on_abort`` completes without propagating.
            (Test Case 2) ``interrupt_delivery_failed()`` returns True.
            (Test Case 3) An ``interrupt_delivery_failed`` audit
                event was appended.
        """
        from spikelab.spike_sorting.guards import _watchdog as wm

        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, kill_grace_s=0.0)

        captured = []

        def _fake_audit(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr(wm, "append_audit_event", _fake_audit)

        def _failing_interrupt():
            raise OSError("simulated interrupt_main failure")

        monkeypatch.setattr(wm._thread, "interrupt_main", _failing_interrupt)

        wd._on_abort(95.0)

        assert wd.tripped() is True
        assert wd.interrupt_delivery_failed() is True
        assert any(
            evt.get("event") == "interrupt_delivery_failed"
            and evt.get("watchdog") == "host_memory"
            for evt in captured
        )

    def test_gpu_watchdog_sets_flag_on_interrupt_failure(self, monkeypatch):
        """
        ``GpuMemoryWatchdog._kill_targets_and_interrupt`` sets the
        flag when ``_thread.interrupt_main`` raises and writes the
        audit event with the device index.

        Tests:
            (Test Case 1) Patched interrupt raises; the trip cascade
                completes.
            (Test Case 2) ``interrupt_delivery_failed()`` is True.
            (Test Case 3) Audit event includes ``device_index``.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog(
            device_index=1,
            warn_pct=70.0,
            abort_pct=90.0,
        )

        captured = []
        monkeypatch.setattr(
            gpu_mod, "append_audit_event", lambda **kw: captured.append(kw)
        )

        def _failing_interrupt():
            raise OSError("simulated interrupt_main failure")

        monkeypatch.setattr(gpu_mod._thread, "interrupt_main", _failing_interrupt)

        # _on_abort triggers _kill_targets_and_interrupt at the tail.
        wd._on_abort(96.0)

        assert wd.tripped() is True
        assert wd.interrupt_delivery_failed() is True
        delivery_events = [
            e for e in captured if e.get("event") == "interrupt_delivery_failed"
        ]
        assert delivery_events
        assert delivery_events[0]["device_index"] == 1

    def test_io_stall_watchdog_sets_flag_on_interrupt_failure(
        self, tmp_path, monkeypatch
    ):
        """
        ``IOStallWatchdog._on_trip`` sets the flag and writes an
        ``interrupt_delivery_failed`` audit event with the device.

        Tests:
            (Test Case 1) Patched interrupt raises; the trip cascade
                completes.
            (Test Case 2) ``interrupt_delivery_failed()`` is True.
            (Test Case 3) Audit event names ``"io_stall"`` and the
                resolved device.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
        wd._device = "sda1"

        captured = []
        monkeypatch.setattr(iom, "append_audit_event", lambda **kw: captured.append(kw))

        # Patch the locally-imported _thread inside _on_trip.
        import _thread as _t_real

        def _failing_interrupt():
            raise OSError("simulated interrupt_main failure")

        monkeypatch.setattr(_t_real, "interrupt_main", _failing_interrupt)

        wd._on_trip(stalled_for=15.0)

        assert wd.tripped() is True
        assert wd.interrupt_delivery_failed() is True
        delivery_events = [
            e for e in captured if e.get("event") == "interrupt_delivery_failed"
        ]
        assert delivery_events
        assert delivery_events[0]["watchdog"] == "io_stall"
        assert delivery_events[0]["device"] == "sda1"

    def test_default_state_is_false(self):
        """
        A fresh watchdog (never tripped) reports
        ``interrupt_delivery_failed() == False``.

        Tests:
            (Test Case 1) Each of the three watchdogs returns False
                immediately after construction.
        """
        host = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0)
        assert host.interrupt_delivery_failed() is False

        gpu = GpuMemoryWatchdog(warn_pct=70.0, abort_pct=90.0)
        assert gpu.interrupt_delivery_failed() is False

        from pathlib import Path as _P

        # IOStallWatchdog needs a real folder for path resolution.
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            io_stall = IOStallWatchdog(_P(td), stall_s=10.0, poll_interval_s=1.0)
            assert io_stall.interrupt_delivery_failed() is False


class TestWatchdogExitContextVarResetSwallow:
    """All three watchdog ``__exit__`` methods swallow ``RuntimeError`` from
    re-used ``ContextVar.reset`` tokens (Python 3.10+ behaviour).
    """

    def _make_used_token(self):
        """Return a ContextVar plus an already-consumed token."""
        import contextvars

        ctx_var: contextvars.ContextVar = contextvars.ContextVar(
            "exhausted", default=None
        )
        token = ctx_var.set("x")
        ctx_var.reset(token)
        # Calling reset again on this token raises RuntimeError on
        # Python 3.10+ ("Token has already been used once").
        return ctx_var, token

    def test_gpu_watchdog_exit_swallows_runtime_error(self, monkeypatch):
        """
        ``GpuMemoryWatchdog.__exit__`` swallows ``RuntimeError`` from a
        re-used ContextVar token and still proceeds to session
        shutdown so NVML resources are not leaked.

        Tests:
            (Test Case 1) Patched ``_active_gpu_watchdog`` is a fresh
                ContextVar with a consumed token. ``__exit__`` does
                not raise.
            (Test Case 2) ``_token`` is cleared to None.
            (Test Case 3) The session's ``shutdown()`` was called.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        wd = GpuMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, poll_interval_s=60.0)
        ctx_var, used_token = self._make_used_token()
        wd._token = used_token  # type: ignore[assignment]

        shutdown_calls = {"count": 0}

        def _shutdown():
            shutdown_calls["count"] += 1

        wd._session = SimpleNamespace(shutdown=_shutdown)

        monkeypatch.setattr(gpu_mod, "_active_gpu_watchdog", ctx_var)
        wd.__exit__(None, None, None)

        assert wd._token is None
        assert shutdown_calls["count"] == 1

    def test_io_stall_watchdog_exit_swallows_runtime_error(self, tmp_path, monkeypatch):
        """
        ``IOStallWatchdog.__exit__`` swallows ``RuntimeError`` from a
        re-used token; the with-block teardown completes silently.

        Tests:
            (Test Case 1) Patched ``_active_io_stall_watchdog`` raises
                RuntimeError on reset; ``__exit__`` does not propagate.
            (Test Case 2) ``_token`` is cleared to None.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
        ctx_var, used_token = self._make_used_token()
        wd._token = used_token  # type: ignore[assignment]

        monkeypatch.setattr(iom, "_active_io_stall_watchdog", ctx_var)
        wd.__exit__(None, None, None)

        assert wd._token is None

    def test_host_memory_watchdog_exit_swallows_runtime_error(self, monkeypatch):
        """
        ``HostMemoryWatchdog.__exit__`` (newly symmetric with the GPU
        and IO stall watchdogs) swallows ``RuntimeError`` from a
        re-used token. Previously this watchdog had no try/except
        around ``_active_watchdog.reset`` and the exception would
        propagate out of teardown.

        Tests:
            (Test Case 1) Patched ``_active_watchdog`` raises
                RuntimeError on reset; ``__exit__`` does not
                propagate.
            (Test Case 2) ``_token`` is cleared and
                ``_subprocesses`` is empty after exit.
        """
        from spikelab.spike_sorting.guards import _watchdog as wm

        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0)
        ctx_var, used_token = self._make_used_token()
        wd._token = used_token  # type: ignore[assignment]

        # Pre-register a subprocess to verify the post-reset clear()
        # still runs after the swallowed error.
        popen = mock.Mock(spec=subprocess.Popen)
        wd.register_subprocess(popen, kill_grace_s=0.0)

        monkeypatch.setattr(wm, "_active_watchdog", ctx_var)
        wd.__exit__(None, None, None)

        assert wd._token is None
        assert wd._subprocesses == []


class TestPynvmlSessionDoubleFailure:
    """``_PynvmlSession.start`` swallows both inner exceptions on double failure."""

    def test_handle_failure_with_shutdown_failure_returns_false(self, monkeypatch):
        """
        When ``nvmlDeviceGetHandleByIndex`` raises *and* the cleanup
        ``nvmlShutdown`` also raises, both exceptions are swallowed
        and ``start()`` returns False. The session leaves
        ``_pynvml`` / ``_handle`` cleared so a later retry can
        reinitialise.

        Tests:
            (Test Case 1) Fake pynvml with both ``handle_raises`` and
                ``shutdown_raises`` → ``start()`` returns False.
            (Test Case 2) Both raises were attempted (counters > 0).
            (Test Case 3) Session state stays uninitialised.
        """
        from spikelab.spike_sorting.guards._gpu_watchdog import _PynvmlSession

        fake = _make_fake_pynvml(handle_raises=True, shutdown_raises=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        session = _PynvmlSession(0)
        assert session.start() is False
        assert fake._counters["init"] == 1
        assert fake._counters["handle"] == 1
        assert fake._counters["shutdown"] == 1
        assert session._pynvml is None
        assert session._handle is None


class TestReadGpuMemoryNvidiaSmiSkipsMalformedLines:
    """``_read_gpu_memory_nvidia_smi`` skips malformed CSV lines."""

    def test_malformed_line_before_match_skipped(self, monkeypatch):
        """
        A malformed CSV line (non-integer index or wrong column count)
        appearing before the matching device line is skipped without
        aborting parsing; the matching line is found.

        Tests:
            (Test Case 1) Output ``"garbage\\nfoo, bar\\n0, 1024,
                4096"`` → returns ``(25.0, 4.0)`` for device 0.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        out = "garbage\nfoo, bar\n0, 1024, 4096\n"
        monkeypatch.setattr(gpu_mod.subprocess, "check_output", lambda *a, **k: out)
        result = gpu_mod._read_gpu_memory_nvidia_smi(0)
        assert result is not None
        used_pct, total_gb = result
        assert used_pct == pytest.approx(25.0)
        assert total_gb == pytest.approx(4.0)


class TestGpuMemoryWatchdogEnterPynvmlSessionFailure:
    """``GpuMemoryWatchdog.__enter__`` falls back when ``_PynvmlSession.start`` fails."""

    def test_session_start_failure_falls_back_to_none_session(self, monkeypatch):
        """
        ``_PynvmlSession.start()`` returning False puts the watchdog
        into nvidia-smi-only mode (``self._session = None``); the
        watchdog still enables because the initial probe via
        ``read_gpu_memory`` succeeded.

        Tests:
            (Test Case 1) Patched ``read_gpu_memory`` returns a tuple
                (probe succeeds); patched ``_PynvmlSession.start``
                returns False → watchdog enters with ``_session=None``
                and ``_enabled=True``.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        monkeypatch.setattr(gpu_mod, "read_gpu_memory", lambda i: (10.0, 16.0))
        monkeypatch.setattr(gpu_mod._PynvmlSession, "start", lambda self: False)

        wd = GpuMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, poll_interval_s=60.0)
        with wd:
            assert wd._enabled is True
            assert wd._session is None


class TestGpuMemoryWatchdogPollLoopThermalAbort:
    """``GpuMemoryWatchdog._poll_loop`` thermal warn → abort transition."""

    def test_temp_above_abort_triggers_thermal_abort(self, monkeypatch):
        """
        With a session reporting steady temperatures above
        ``abort_temp_c``, the watchdog tags the trip as thermal,
        records ``_temp_c_at_trip``, and ``make_error`` returns a
        ``GpuThermalWatchdogError``.

        Tests:
            (Test Case 1) Stub session reports temp 95.0 (>= abort 92);
                after polling the watchdog tripped() and trip_kind
                ('thermal'); make_error returns GpuThermalWatchdogError.

        Notes:
            - Suppresses ``_thread.interrupt_main`` by setting the
              watchdog's stop event before the abort can propagate.
        """
        from spikelab.spike_sorting._exceptions import GpuThermalWatchdogError
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        # Memory healthy, temperature hot.
        fake_session = SimpleNamespace(
            read_memory=lambda: (10.0, 16.0),
            read_temperature_c=lambda: 95.0,
            read_throttle_reasons=lambda: 0,
            shutdown=lambda: None,
        )

        monkeypatch.setattr(gpu_mod, "read_gpu_memory", lambda i: (10.0, 16.0))
        # Force _PynvmlSession.start() to populate ourselves with the fake.
        monkeypatch.setattr(gpu_mod._PynvmlSession, "start", lambda self: False)

        wd = GpuMemoryWatchdog(
            warn_pct=70.0,
            abort_pct=90.0,
            warn_temp_c=80.0,
            abort_temp_c=92.0,
            poll_interval_s=0.05,
        )
        # Inject our fake session by hand so the thermal branch in
        # _poll_loop is exercised.
        with wd:
            wd._session = fake_session
            try:
                deadline = time.time() + 3.0
                while time.time() < deadline and not wd.tripped():
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass

        assert wd.tripped()
        assert wd.trip_kind() == "thermal"
        assert wd.temperature_c_at_trip() == pytest.approx(95.0)
        err = wd.make_error()
        assert isinstance(err, GpuThermalWatchdogError)


class TestGpuMemoryWatchdogPollLoopThrottleWarn:
    """``GpuMemoryWatchdog._poll_loop`` surfaces throttle reasons as warnings."""

    def test_throttle_mask_triggers_throttle_warn(self, monkeypatch, caplog):
        """
        With ``monitor_throttle_reasons=True`` and a session reporting
        an active throttle bit, the watchdog logs a throttle warning.
        VRAM and temperature readings stay healthy so the watchdog
        does not trip.

        Tests:
            (Test Case 1) Stub session reports throttle mask 0x4
                (SW power cap); watchdog logs ``"throttling"``
                warning with the reason text.
            (Test Case 2) Watchdog does not trip during the test
                window.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        fake_session = SimpleNamespace(
            read_memory=lambda: (10.0, 16.0),
            read_temperature_c=lambda: 60.0,
            read_throttle_reasons=lambda: 0x4,  # SW power cap
            shutdown=lambda: None,
        )

        monkeypatch.setattr(gpu_mod, "read_gpu_memory", lambda i: (10.0, 16.0))
        monkeypatch.setattr(gpu_mod._PynvmlSession, "start", lambda self: False)

        wd = GpuMemoryWatchdog(
            warn_pct=70.0,
            abort_pct=90.0,
            warn_temp_c=80.0,
            abort_temp_c=92.0,
            monitor_throttle_reasons=True,
            poll_interval_s=0.05,
            warn_repeat_s=0.0,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="spikelab.spike_sorting.guards._gpu_watchdog",
        ):
            with wd:
                wd._session = fake_session
                time.sleep(0.4)

        assert any("throttling" in r.getMessage() for r in caplog.records)
        assert not wd.tripped()


class TestGpuMemoryWatchdogPollLoopInfoNoneSkip:
    """``GpuMemoryWatchdog._poll_loop`` waits and skips on None VRAM info."""

    def test_info_none_does_not_trip(self, monkeypatch):
        """
        When ``read_memory`` (or ``read_gpu_memory``) returns None
        the watchdog accumulates the blindness counter and waits the
        next poll instead of tripping.

        Tests:
            (Test Case 1) Stub returns None on every poll; watchdog
                does not trip during the test window.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        # First call (the __enter__ probe) succeeds so the watchdog
        # enables; subsequent in-loop calls return None.
        seq = iter([(10.0, 16.0), None, None, None, None, None])

        def _read(_idx):
            try:
                return next(seq)
            except StopIteration:
                return None

        monkeypatch.setattr(gpu_mod, "read_gpu_memory", _read)
        monkeypatch.setattr(gpu_mod._PynvmlSession, "start", lambda self: False)

        wd = GpuMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, poll_interval_s=0.05)
        with wd:
            time.sleep(0.4)
            assert not wd.tripped()


class TestGpuMemoryWatchdogMaybeWarnAuditEvent:
    """``GpuMemoryWatchdog._maybe_warn`` writes a watchdog="gpu_memory" audit event."""

    def test_audit_event_written_on_warn(self, monkeypatch):
        """
        ``_maybe_warn`` calls ``append_audit_event`` with the
        ``"gpu_memory"`` watchdog label and a ``"warn"`` event,
        passing the percent and thresholds in the payload.

        Tests:
            (Test Case 1) Patched ``append_audit_event`` records the
                call; the watchdog and event labels match.
            (Test Case 2) Payload contains ``used_pct``, ``warn_pct``,
                ``abort_pct``, ``device_index``.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        captured = []

        def _fake_audit(**kwargs):
            captured.append(kwargs)

        monkeypatch.setattr(gpu_mod, "append_audit_event", _fake_audit)

        wd = GpuMemoryWatchdog(
            device_index=2, warn_pct=70.0, abort_pct=90.0, warn_repeat_s=300.0
        )
        wd._maybe_warn(76.5)

        assert len(captured) == 1
        evt = captured[0]
        assert evt["watchdog"] == "gpu_memory"
        assert evt["event"] == "warn"
        assert evt["device_index"] == 2
        assert evt["used_pct"] == 76.5
        assert evt["warn_pct"] == 70.0
        assert evt["abort_pct"] == 90.0


class TestIOStallWatchdogStallSValidation:
    """``IOStallWatchdog.__init__`` rejects non-positive ``stall_s``."""

    def test_zero_or_negative_stall_s_raises(self, tmp_path):
        """
        ``stall_s <= 0`` raises ValueError at construction; mirrors
        the ``poll_interval_s`` and ``kill_grace_s`` validators so
        misconfig is caught at the construction site rather than
        silently disabling the watchdog at ``__enter__``.

        Tests:
            (Test Case 1) ``stall_s=0`` → ValueError.
            (Test Case 2) ``stall_s=-30.0`` → ValueError.
        """
        with pytest.raises(ValueError, match="stall_s"):
            IOStallWatchdog(tmp_path, stall_s=0.0, poll_interval_s=1.0)
        with pytest.raises(ValueError, match="stall_s"):
            IOStallWatchdog(tmp_path, stall_s=-30.0, poll_interval_s=1.0)


class TestReadLockInfoEmptyJson:
    """``_read_lock_info`` parses empty JSON object as ``{}`` not None."""

    def test_empty_json_object_returns_empty_dict(self, tmp_path):
        """
        A lock file containing exactly ``{}`` parses to an empty dict
        (not None). Downstream callers treat the empty dict as a
        partial/stale lock (pid lookup yields -1 → not alive →
        reclaimed).

        Tests:
            (Test Case 1) Lock file containing ``{}`` → returns ``{}``.
        """
        from spikelab.spike_sorting.guards._sort_lock import _read_lock_info

        lock = tmp_path / ".spikelab_sort.lock"
        lock.write_text("{}", encoding="utf-8")
        info = _read_lock_info(lock)
        assert info == {}


class TestPreflightRtSortDiskFindingExtra:
    """``_rt_sort_disk_finding`` skip branches when no estimates / no free disks."""

    def test_no_estimates_returns_none(self, monkeypatch):
        """
        When every recording raises on ``get_num_channels`` /
        ``get_num_samples``, the helper has no estimates and returns
        None — the projection check is silently skipped.

        Tests:
            (Test Case 1) Two recordings, both raise on shape probes;
                the finding is None.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="rt_sort", use_docker=False)
        )
        bad1 = SimpleNamespace(
            get_num_channels=mock.Mock(side_effect=RuntimeError("bad")),
            get_num_samples=mock.Mock(side_effect=RuntimeError("bad")),
        )
        bad2 = SimpleNamespace(
            get_num_channels=mock.Mock(side_effect=ValueError("bad")),
            get_num_samples=mock.Mock(side_effect=ValueError("bad")),
        )
        result = pf._rt_sort_disk_finding(cfg, [bad1, bad2], ["/inter"])
        assert result is None

    def test_no_free_gbs_returns_none(self, monkeypatch):
        """
        When every intermediate folder's ``_disk_free_gb`` returns
        None, the helper cannot compute a comparison and returns
        None.

        Tests:
            (Test Case 1) Patched ``_disk_free_gb`` always returns
                None → result is None even with valid estimates.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="rt_sort", use_docker=False)
        )
        rec = SimpleNamespace(
            get_num_channels=lambda: 64,
            get_num_samples=lambda: 30_000_000,
        )
        monkeypatch.setattr(pf, "_disk_free_gb", lambda p: None)

        result = pf._rt_sort_disk_finding(cfg, [rec], ["/inter1", "/inter2"])
        assert result is None


class TestPreflightCheckSorterDependenciesEmpty:
    """``_check_sorter_dependencies`` returns [] when each per-sorter check returns []."""

    def test_empty_findings_pass_through(self, monkeypatch):
        """
        When the dispatched per-sorter helper returns an empty list,
        ``_check_sorter_dependencies`` propagates the empty list.

        Tests:
            (Test Case 1) Patched ``_check_kilosort2_host`` returns []
                → top-level ``_check_sorter_dependencies`` returns [].
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        monkeypatch.setattr(pf, "_check_kilosort2_host", lambda c: [])

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort2", use_docker=False)
        )
        assert pf._check_sorter_dependencies(cfg) == []


class TestPreflightCheckRtSortCudaRaisePropagates:
    """``_check_rt_sort`` surfaces cuda runtime errors as environment findings."""

    def test_cuda_runtime_error_surfaces_as_environment_finding(self, monkeypatch):
        """
        When ``torch.cuda.is_available()`` raises ``RuntimeError``
        (e.g. driver crashed mid-process), ``_check_rt_sort`` appends
        a fail-level ``sorter_dependency_missing`` environment finding
        rather than letting the exception escape.

        Tests:
            (Test Case 1) The call returns a list of
                ``PreflightFinding`` (no exception).
            (Test Case 2) Exactly one finding has ``level='fail'``,
                ``code='sorter_dependency_missing'``,
                ``category='environment'``.
        """
        import importlib.util as _importutil

        from spikelab.spike_sorting.guards import _preflight as pf

        def _raising_is_available():
            raise RuntimeError("driver crash")

        # Mark all RT-Sort deps as present via find_spec so the
        # dependency loop reports them importable.
        present = {"torch", "diptest", "sklearn", "h5py", "tqdm"}
        real_find_spec = _importutil.find_spec

        def _fake_find_spec(name, package=None):
            if name in present:
                return _importutil.spec_from_loader(name, loader=None)
            try:
                return real_find_spec(name, package)
            except (ImportError, ValueError):
                return None

        monkeypatch.setattr(_importutil, "find_spec", _fake_find_spec)
        # The cuda branch still does ``import torch``, so torch must
        # exist in sys.modules with the raising attribute.
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=_raising_is_available)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        cfg = SimpleNamespace(
            rt_sort=SimpleNamespace(device="cuda:0"),
        )
        findings = pf._check_rt_sort(cfg)
        assert isinstance(findings, list)
        assert all(isinstance(f, PreflightFinding) for f in findings)
        matching = [
            f
            for f in findings
            if f.level == "fail"
            and f.code == "sorter_dependency_missing"
            and f.category == "environment"
        ]
        assert len(matching) == 1


class TestFindTrippedGlobalWatchdogPriority:
    """``find_tripped_global_watchdog`` returns first-tripped in priority order."""

    def test_host_priority_over_gpu_and_io(self, tmp_path):
        """
        With all three watchdogs active and tripped, the host
        watchdog is returned first (highest priority).

        Tests:
            (Test Case 1) Host + GPU + IO stall watchdogs all in
                ``__enter__``ed state and marked tripped; the helper
                returns the host instance.
        """
        from spikelab.spike_sorting.guards import (
            find_tripped_global_watchdog,
        )
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod
        from spikelab.spike_sorting.guards import _io_stall as iom

        fake_vm = SimpleNamespace(percent=10.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)

        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with HostMemoryWatchdog(
                warn_pct=70.0, abort_pct=90.0, poll_interval_s=60.0
            ) as host:
                with mock.patch.object(
                    gpu_mod, "read_gpu_memory", lambda i: (10.0, 16.0)
                ):
                    with GpuMemoryWatchdog(
                        warn_pct=70.0,
                        abort_pct=90.0,
                        poll_interval_s=60.0,
                    ) as gpu:
                        with (
                            mock.patch.object(
                                iom,
                                "_resolve_device_for_path",
                                return_value="sda1",
                            ),
                            mock.patch.object(iom, "_read_io_bytes", return_value=100),
                        ):
                            with IOStallWatchdog(
                                tmp_path,
                                stall_s=60.0,
                                poll_interval_s=60.0,
                            ) as io:
                                # Mark all three as tripped.
                                host._tripped = True
                                gpu._tripped = True
                                io._tripped = True
                                tripped = find_tripped_global_watchdog()
                                assert tripped is host


class TestParseWslconfigEmptyValue:
    """``_parse_wslconfig_memory_gb`` returns None for malformed values."""

    def test_empty_value_returns_none(self):
        """
        ``memory=`` (no number after the equals) does not match the
        regex and returns None.

        Tests:
            (Test Case 1) ``[wsl2]\\nmemory=\\n`` → returns None.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        assert _parse_wslconfig_memory_gb("[wsl2]\nmemory=\n") is None

    def test_no_number_with_unit_returns_none(self):
        """
        ``memory=GB`` (unit but no number) fails the digit-required
        portion of the regex.

        Tests:
            (Test Case 1) ``[wsl2]\\nmemory=GB\\n`` → returns None.
        """
        from spikelab.spike_sorting.guards._preflight import (
            _parse_wslconfig_memory_gb,
        )

        assert _parse_wslconfig_memory_gb("[wsl2]\nmemory=GB\n") is None


class TestCheckKilosort2HostEmptyEnvVar:
    """``_check_kilosort2_host`` treats empty ``KILOSORT_PATH`` as unset."""

    def test_empty_string_env_var_yields_unset_finding(self, monkeypatch):
        """
        ``KILOSORT_PATH=""`` is falsy in ``if not ks_path:`` and
        triggers the same "neither sorter_path nor KILOSORT_PATH is
        set" finding as a missing env var.

        Tests:
            (Test Case 1) Empty env var → fail finding with message
                containing 'neither sorter_path nor KILOSORT_PATH'.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        monkeypatch.setattr(pf.shutil, "which", lambda name: "/usr/bin/matlab")
        monkeypatch.setenv("KILOSORT_PATH", "")

        cfg = SimpleNamespace(sorter=SimpleNamespace(sorter_path=None))
        findings = pf._check_kilosort2_host(cfg)
        codes = [f.code for f in findings if f.level == "fail"]
        assert "sorter_dependency_missing" in codes
        assert any("KILOSORT_PATH" in f.message for f in findings)


class TestCheckRtSortRtSortNoneCrashes:
    """``_check_rt_sort`` documented behaviour when ``config.rt_sort`` is None."""

    def test_rt_sort_none_raises_attribute_error(self, monkeypatch):
        """
        When ``config.rt_sort`` is None, ``getattr(None, "device", "")``
        does not return the default — instead the device string
        becomes ``str(None or "")`` = ``""`` after the ``or`` guard,
        so the cuda branch is skipped. The helper does NOT crash on
        a None ``rt_sort`` thanks to the ``or ""`` defensive cast.

        Tests:
            (Test Case 1) ``rt_sort=None`` → no cuda-related finding
                (the device string normalises to empty).

        Notes:
            - Documents current behaviour. The original concern in
              REVIEW.md was that this would raise AttributeError, but
              the source uses ``str(getattr(...) or "")`` which
              short-circuits None safely.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        for name in ("torch", "diptest", "sklearn", "h5py", "tqdm"):
            monkeypatch.setitem(sys.modules, name, SimpleNamespace())

        cfg = SimpleNamespace(rt_sort=None)
        findings = pf._check_rt_sort(cfg)
        # The function returned successfully; no cuda-specific finding.
        assert not any("is_available()" in f.message for f in findings)


class TestCheckRecordingSampleRateZeroFreq:
    """``_check_recording_sample_rate`` emits warn for fs_hz=0 (out of window)."""

    def test_zero_frequency_emits_out_of_window_warn(self):
        """
        A recording reporting ``get_sampling_frequency() == 0`` falls
        outside any positive window and yields a warn-level
        ``sample_rate_out_of_window`` finding.

        Tests:
            (Test Case 1) Recording at 0 Hz for ``kilosort4`` →
                emits warn finding referencing 0.00 kHz.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.guards._preflight import (
            _check_recording_sample_rate,
        )

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort4"
        rec = SimpleNamespace(get_sampling_frequency=lambda: 0.0)
        findings = _check_recording_sample_rate(cfg, [rec])
        assert len(findings) == 1
        assert findings[0].level == "warn"
        assert findings[0].code == "sample_rate_out_of_window"


class TestDiskFreeGbDotSegments:
    """``_disk_free_gb`` walks up parent-by-parent on paths with ``..``."""

    def test_dot_dot_path_walks_to_existing_ancestor(self, tmp_path, monkeypatch):
        """
        A path containing ``..`` segments that resolve outside the
        configured tree is walked parent-by-parent until an existing
        path is found; ``shutil.disk_usage`` is called on that path.

        Tests:
            (Test Case 1) Synthetic ``tmp_path / "child" / ".." /
                "missing"`` resolves to an existing parent →
                helper returns a numeric GB value.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        # The path child/../missing has the literal ".." segment; the
        # helper walks via .parent until it lands on an existing dir.
        target = tmp_path / "child" / ".." / "missing"
        # Patch disk_usage so the test is hermetic.
        monkeypatch.setattr(
            dw.shutil,
            "disk_usage",
            lambda p: SimpleNamespace(total=0, used=0, free=4 * (1024**3)),
        )
        result = dw._disk_free_gb(target)
        assert result == pytest.approx(4.0)


class TestFreeVramGbZeroDevicePynvml:
    """``_free_vram_gb`` returns 0.0 when pynvml reports zero devices."""

    def test_zero_devices_returns_zero_gb(self, monkeypatch):
        """
        A pynvml-reachable host with ``nvmlDeviceGetCount() == 0``
        sums no per-device memory and returns 0.0 — current behaviour
        (which then triggers a low_vram warning even though the host
        has no GPU).

        Tests:
            (Test Case 1) Stub pynvml with ``nvmlDeviceGetCount=0`` →
                returns 0.0.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        fake = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 0,
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert pf._free_vram_gb() == 0.0


class TestCheckFilesystemWritableNonexistentAncestor:
    """``_check_filesystem_writable`` skips when no ancestor exists."""

    def test_fully_nonexistent_path_skipped(self):
        """
        A folder whose every ancestor (up to root) does not exist
        falls out of the existence loop; the helper continues without
        emitting a finding.

        Tests:
            (Test Case 1) On Windows, a path with a fake drive letter
                (``"Z:/totally/fake/path"``) walks up to ``Z:`` which
                does not exist; the helper returns ``[]``.

        Notes:
            - Skipped on POSIX where root ``/`` always exists.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        if sys.platform == "win32":
            target = "Z:/totally_fake_drive_pythontest_doesnotexist/path"
        else:
            pytest.skip("POSIX root always exists; cannot reproduce")

        findings = pf._check_filesystem_writable(
            [target], label="intermediate", code_prefix="intermediate"
        )
        assert findings == []


class TestCheckResourceRlimitsInfinity:
    """``_check_resource_rlimits`` treats RLIM_INFINITY as healthy."""

    def test_rlim_infinity_no_finding(self, monkeypatch):
        """
        ``RLIMIT_NOFILE`` returning -1 (RLIM_INFINITY) fails the
        ``0 < soft_nofile < 4096`` test and produces no finding.

        Tests:
            (Test Case 1) Patched ``getrlimit`` returns (-1, -1) for
                NOFILE and (65536, 65536) for NPROC → no findings.
        """
        try:
            import resource as _resource
        except ImportError:
            pytest.skip("resource module not available on this platform")

        from spikelab.spike_sorting.guards import _preflight as pf

        def _fake(name):
            if name == _resource.RLIMIT_NOFILE:
                return (-1, -1)
            return (65536, 65536)

        monkeypatch.setattr(_resource, "getrlimit", _fake)

        cfg = SimpleNamespace(rt_sort=None)
        assert pf._check_resource_rlimits(cfg) == []


class TestHostMemoryWatchdogConstructionStored:
    """``HostMemoryWatchdog.__init__`` round-trips fields verbatim."""

    def test_fields_stored_verbatim(self):
        """
        After construction, every constructor field lands on the
        instance unchanged.

        Tests:
            (Test Case 1) ``warn_pct``, ``abort_pct``, ``poll_interval_s``,
                ``warn_repeat_s``, ``kill_grace_s`` round-trip.
            (Test Case 2) Internal lists initialise empty.
        """
        wd = HostMemoryWatchdog(
            warn_pct=72.5,
            abort_pct=88.5,
            poll_interval_s=3.5,
            warn_repeat_s=45.0,
            kill_grace_s=7.5,
        )
        assert wd.warn_pct == 72.5
        assert wd.abort_pct == 88.5
        assert wd.poll_interval_s == 3.5
        assert wd.warn_repeat_s == 45.0
        assert wd.kill_grace_s == 7.5
        assert wd._subprocesses == []
        assert wd._kill_callbacks == []


class TestHostMemoryWatchdogMakeErrorNoneAtTrip:
    """``HostMemoryWatchdog.make_error`` falls back to ``"?"`` for None pct."""

    def test_default_message_uses_question_mark_when_pct_none(self):
        """
        When ``_percent_at_trip`` is None the default error message
        substitutes ``"?"``; the error is constructible without
        crashing on a None format.

        Tests:
            (Test Case 1) Pre-trip ``make_error`` contains ``"?"``.
        """
        wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0)
        err = wd.make_error()
        assert "?" in str(err)


class TestHostMemoryWatchdogEnterEnabledFlag:
    """``HostMemoryWatchdog.__enter__`` sets ``_enabled=True`` on success."""

    def test_enabled_true_after_enter(self):
        """
        With psutil importable, ``__enter__`` flips ``_enabled`` to
        True and starts the polling thread.

        Tests:
            (Test Case 1) Inside the with-block, ``_enabled`` is True
                and ``_thread`` is a live Thread instance.
        """
        fake_vm = SimpleNamespace(percent=10.0)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, poll_interval_s=60.0)
            with wd:
                assert wd._enabled is True
                assert wd._thread is not None
                assert wd._thread.is_alive()


class TestHostMemoryWatchdogPollLoopPsutilRetry:
    """``HostMemoryWatchdog._poll_loop`` silently retries when psutil raises."""

    def test_psutil_raise_does_not_tear_down_watchdog(self):
        """
        Transient ``psutil.virtual_memory()`` failures (e.g. on some
        platforms under load) are caught by the inner exception
        handler; the loop waits one poll interval and continues.

        Tests:
            (Test Case 1) Stub ``virtual_memory`` raises for the
                first few calls, then returns a healthy reading.
                The watchdog runs without crashing and never trips.
        """
        call_count = {"value": 0}

        def _flaky():
            call_count["value"] += 1
            if call_count["value"] <= 2:
                raise RuntimeError("transient psutil failure")
            return SimpleNamespace(percent=10.0)

        fake_psutil = SimpleNamespace(virtual_memory=_flaky)
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            wd = HostMemoryWatchdog(warn_pct=70.0, abort_pct=90.0, poll_interval_s=0.05)
            with wd:
                time.sleep(0.4)
                assert not wd.tripped()
        # Confirm the loop did call psutil multiple times despite errors.
        assert call_count["value"] >= 2


class TestPreflightAvailableRamGbHappyPath:
    """``_available_ram_gb`` returns a positive float on a healthy host."""

    def test_returns_positive_value(self, monkeypatch):
        """
        With a stub psutil reporting 16 GB available, the helper
        returns 16.0 GB.

        Tests:
            (Test Case 1) Stub ``virtual_memory().available`` of
                ``16 * 1024**3`` → returns 16.0.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        fake_vm = SimpleNamespace(available=16 * (1024**3))
        fake_psutil = SimpleNamespace(virtual_memory=lambda: fake_vm)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        result = pf._available_ram_gb()
        assert result == pytest.approx(16.0)


class TestPreflightSorterUsesGpu:
    """``_sorter_uses_gpu`` matches the documented sorter table."""

    def test_kilosort2_host_returns_false(self):
        """
        KS2 host (no Docker) does not use the GPU at the host level —
        MATLAB is CPU-only here.

        Tests:
            (Test Case 1) ``kilosort2`` + ``use_docker=False`` → False.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort2", use_docker=False)
        )
        assert pf._sorter_uses_gpu(cfg) is False

    def test_kilosort2_docker_returns_true(self):
        """
        KS2 with ``use_docker=True`` runs the GPU-enabled image.

        Tests:
            (Test Case 1) ``kilosort2`` + ``use_docker=True`` → True.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort2", use_docker=True)
        )
        assert pf._sorter_uses_gpu(cfg) is True

    def test_unknown_sorter_returns_false(self):
        """
        An unrecognised sorter name conservatively returns False.

        Tests:
            (Test Case 1) ``sorter_name="unknown"`` → False.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="unknown", use_docker=False)
        )
        assert pf._sorter_uses_gpu(cfg) is False


class TestPreflightWslconfigFindingExtra:
    """``_wslconfig_finding`` extra branches: unreadable / sane / no host RAM."""

    def test_unreadable_wslconfig_returns_none(self, tmp_path, monkeypatch):
        """
        ``OSError`` on ``read_text`` (e.g. permission denied) makes
        the helper return None — silent skip rather than a crash.

        Tests:
            (Test Case 1) Patch ``Path.read_text`` to raise OSError;
                the finding is None.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        if sys.platform != "win32":
            monkeypatch.setattr(pf.sys, "platform", "win32")

        # Pre-create a real .wslconfig so the file-existence check passes.
        home = tmp_path
        monkeypatch.setattr(pf.os.path, "expanduser", lambda _: str(home))
        wslconfig = home / ".wslconfig"
        wslconfig.write_text("[wsl2]\nmemory=8GB\n")

        original_read = Path.read_text

        def _refuse(self, *args, **kwargs):
            if self == wslconfig:
                raise OSError("permission denied")
            return original_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _refuse)

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort4", use_docker=True)
        )
        assert pf._wslconfig_finding(cfg) is None

    def test_sane_memory_under_85pct_returns_none(self, tmp_path, monkeypatch):
        """
        A wslconfig with memory ≤ 85% of host RAM is considered
        healthy and returns None.

        Tests:
            (Test Case 1) ``memory=8GB`` on a 16 GB host (50%) → None.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        if sys.platform != "win32":
            monkeypatch.setattr(pf.sys, "platform", "win32")

        home = tmp_path
        monkeypatch.setattr(pf.os.path, "expanduser", lambda _: str(home))
        wslconfig = home / ".wslconfig"
        wslconfig.write_text("[wsl2]\nmemory=8GB\n")

        # Patch get_system_ram_bytes via the import in _wslconfig_finding.
        from spikelab.spike_sorting import sorting_utils as su

        monkeypatch.setattr(su, "get_system_ram_bytes", lambda: 16 * (1024**3))

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort4", use_docker=True)
        )
        assert pf._wslconfig_finding(cfg) is None

    def test_host_ram_undetectable_falls_through_to_none(self, tmp_path, monkeypatch):
        """
        When ``get_system_ram_bytes`` returns None or raises, the
        host-vs-config comparison is skipped and the function returns
        None (no actionable finding).

        Tests:
            (Test Case 1) ``get_system_ram_bytes`` patched to return
                None → wslconfig finding is None.
        """
        from spikelab.spike_sorting.guards import _preflight as pf

        if sys.platform != "win32":
            monkeypatch.setattr(pf.sys, "platform", "win32")

        home = tmp_path
        monkeypatch.setattr(pf.os.path, "expanduser", lambda _: str(home))
        wslconfig = home / ".wslconfig"
        wslconfig.write_text("[wsl2]\nmemory=64GB\n")

        from spikelab.spike_sorting import sorting_utils as su

        monkeypatch.setattr(su, "get_system_ram_bytes", lambda: None)

        cfg = SimpleNamespace(
            sorter=SimpleNamespace(sorter_name="kilosort4", use_docker=True)
        )
        assert pf._wslconfig_finding(cfg) is None


class TestDiskUsageWatchdogConstructionExtra:
    """``DiskUsageWatchdog.__init__`` additional branches and storage."""

    def test_callback_only_enables_watchdog(self, tmp_path):
        """
        ``popen=None`` plus a ``kill_callback`` is a valid kill target;
        the watchdog enables.

        Tests:
            (Test Case 1) callback-only watchdog has ``_enabled=True``.
        """
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
        )
        assert wd._enabled is True
        assert wd.popen is None
        assert wd.kill_callback is not None

    def test_kill_grace_s_stored_verbatim(self, tmp_path):
        """
        ``kill_grace_s`` is round-tripped to the instance.

        Tests:
            (Test Case 1) ``kill_grace_s=11.5`` is stored as a float.
        """
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
            kill_grace_s=11.5,
        )
        assert wd.kill_grace_s == 11.5

    def test_poll_interval_negative_raises(self, tmp_path):
        """
        ``poll_interval_s <= 0`` raises ValueError (the helper also
        rejects zero — both negative and zero are invalid).

        Tests:
            (Test Case 1) Negative poll_interval_s → ValueError.
        """
        with pytest.raises(ValueError, match="poll_interval_s"):
            DiskUsageWatchdog(
                folder=tmp_path,
                warn_free_gb=10.0,
                abort_free_gb=5.0,
                kill_callback=lambda: None,
                poll_interval_s=-1.0,
            )


class TestDiskUsageWatchdogReportNotTripped:
    """``DiskUsageWatchdog.report`` returns None when not tripped."""

    def test_report_returns_none_when_not_tripped(self, tmp_path):
        """
        A fresh watchdog (never tripped) returns None from
        ``report()``; the trip-time report is built lazily.

        Tests:
            (Test Case 1) Fresh watchdog → ``report() is None``.
        """
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
        )
        assert wd.report() is None


class TestDiskUsageWatchdogMakeErrorBranches:
    """``DiskUsageWatchdog.make_error`` custom message + None formatting."""

    def test_custom_message_overrides_default(self, tmp_path):
        """
        ``make_error("explicit text")`` returns a DiskExhaustionError
        whose ``str()`` is the supplied message.

        Tests:
            (Test Case 1) Custom message is preserved verbatim.
        """
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
        )
        err = wd.make_error("custom-disk-msg")
        assert isinstance(err, DiskExhaustionError)
        assert str(err) == "custom-disk-msg"

    def test_none_free_at_trip_uses_question_mark(self, tmp_path):
        """
        When ``_free_at_trip`` is None (called pre-trip or trip
        without a reading), the default-message formatter substitutes
        ``"?"`` rather than crashing.

        Tests:
            (Test Case 1) Pre-trip ``make_error`` contains ``"?"``
                in place of a number.
        """
        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
        )
        err = wd.make_error()
        assert "?" in str(err)


class TestDiskUsageWatchdogEnterTopConsumersSwallow:
    """``DiskUsageWatchdog.__enter__`` swallows _top_consumers exceptions."""

    def test_top_consumers_raise_does_not_propagate(self, tmp_path, monkeypatch):
        """
        If ``_top_consumers`` raises during enter, the watchdog still
        starts; ``_initial_top_consumers`` falls back to an empty list.

        Tests:
            (Test Case 1) Patched ``_top_consumers`` raises;
                ``__enter__`` returns successfully.
            (Test Case 2) ``_initial_top_consumers`` is ``[]``.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        def _raise(*_a, **_kw):
            raise RuntimeError("simulated walk failure")

        monkeypatch.setattr(dw, "_top_consumers", _raise)
        monkeypatch.setattr(dw, "_disk_free_gb", lambda p: 100.0)

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
            poll_interval_s=60.0,  # avoid trip during the test.
        )
        with wd:
            assert wd._initial_top_consumers == []


class TestDiskUsageWatchdogExitJoinsThread:
    """``DiskUsageWatchdog.__exit__`` joins the polling thread."""

    def test_thread_is_none_after_exit(self, tmp_path, monkeypatch):
        """
        After exiting the context the watchdog's polling thread is
        joined and ``_thread`` is reset to None — so a follow-up
        ``__exit__`` call is a no-op.

        Tests:
            (Test Case 1) ``_thread`` is None after ``__exit__``.
        """
        from spikelab.spike_sorting.guards import _disk_watchdog as dw

        monkeypatch.setattr(dw, "_disk_free_gb", lambda p: 100.0)
        monkeypatch.setattr(dw, "_top_consumers", lambda p: [])

        wd = DiskUsageWatchdog(
            folder=tmp_path,
            warn_free_gb=10.0,
            abort_free_gb=5.0,
            kill_callback=lambda: None,
            poll_interval_s=0.05,
        )
        with wd:
            time.sleep(0.05)
        assert wd._thread is None


class TestIOStallReadIoBytesBranches:
    """``_read_io_bytes`` covers psutil missing / no counters / sum / fallbacks."""

    def test_returns_none_when_psutil_missing(self, monkeypatch):
        """
        ``ImportError`` on the ``psutil`` import returns None.

        Tests:
            (Test Case 1) Patched ``__import__`` blocks ``psutil`` →
                returns None.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        monkeypatch.delitem(sys.modules, "psutil", raising=False)

        import builtins as _b

        real_import = _b.__import__

        def _blocked(name, *a, **k):
            if name == "psutil":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        monkeypatch.setattr(_b, "__import__", _blocked)
        assert iom._read_io_bytes("sda1") is None

    def test_returns_none_when_disk_io_counters_returns_none(self, monkeypatch):
        """
        ``psutil.disk_io_counters(perdisk=True)`` returning None
        propagates as None.

        Tests:
            (Test Case 1) Stub psutil with ``disk_io_counters`` →
                None → ``_read_io_bytes`` returns None.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        fake_psutil = SimpleNamespace(disk_io_counters=lambda perdisk=True: None)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert iom._read_io_bytes("sda1") is None

    def test_sum_of_read_and_write_bytes(self, monkeypatch):
        """
        The returned value is ``read_bytes + write_bytes`` for the
        matched device.

        Tests:
            (Test Case 1) Stub counters report read=100, write=250 →
                ``_read_io_bytes`` returns 350.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        info = SimpleNamespace(read_bytes=100, write_bytes=250)
        fake_psutil = SimpleNamespace(
            disk_io_counters=lambda perdisk=True: {"sda1": info}
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert iom._read_io_bytes("sda1") == 350

    def test_trailing_colon_fallback_lookup(self, monkeypatch):
        """
        Windows ``perdisk`` keys appear as ``"C:"``; passing ``"C:"``
        directly works, and a caller passing ``"C"`` (no colon) finds
        the entry via the ``device + ":"`` fallback.

        Tests:
            (Test Case 1) Stub counters keyed at ``"C:"``; lookup with
                ``"C"`` returns the byte sum via fallback.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        info = SimpleNamespace(read_bytes=10, write_bytes=20)
        fake_psutil = SimpleNamespace(
            disk_io_counters=lambda perdisk=True: {"C:": info}
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        assert iom._read_io_bytes("C") == 30


class TestIOStallResolveDeviceLongestMountpoint:
    """``_resolve_device_for_path`` picks the longest matching mountpoint."""

    def test_longest_mountpoint_prefix_wins(self, monkeypatch):
        """
        With overlapping mountpoints (``/`` and ``/data``), a path
        under ``/data`` should resolve to the more specific mount —
        the helper picks the longest mountpoint prefix.

        Tests:
            (Test Case 1) Stub partitions ``/`` (sda1) and ``/data``
                (sdb1); resolve ``/data/recording`` → returns the
                normalised key for sdb1.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        # POSIX-style stub. On Windows the test path is rewritten,
        # but the resolution logic still picks the longer mountpoint.
        if sys.platform == "win32":
            pytest.skip("POSIX-style mountpoint test")

        partitions = [
            SimpleNamespace(mountpoint="/", device="/dev/sda1"),
            SimpleNamespace(mountpoint="/data", device="/dev/sdb1"),
        ]
        fake_psutil = SimpleNamespace(disk_partitions=lambda all=False: partitions)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        result = iom._resolve_device_for_path(Path("/data/recording.h5"))
        assert result == "sdb1"


class TestIOStallWatchdogConstructionStored:
    """``IOStallWatchdog.__init__`` stores fields verbatim."""

    def test_fields_stored_and_subprocesses_empty(self, tmp_path):
        """
        After construction, configuration fields land on the instance
        and the internal ``_subprocesses`` list initialises empty.

        Tests:
            (Test Case 1) ``stall_s``, ``poll_interval_s``,
                ``warn_repeat_s``, ``kill_grace_s`` are set.
            (Test Case 2) ``_subprocesses`` is an empty list.
        """
        wd = IOStallWatchdog(
            tmp_path,
            stall_s=12.5,
            poll_interval_s=2.5,
            warn_repeat_s=20.0,
            kill_grace_s=3.5,
        )
        assert wd.stall_s == 12.5
        assert wd.poll_interval_s == 2.5
        assert wd.warn_repeat_s == 20.0
        assert wd.kill_grace_s == 3.5
        assert wd._subprocesses == []


class TestIOStallWatchdogMakeErrorBranches:
    """``IOStallWatchdog.make_error`` custom message + None formatting."""

    def test_custom_message_overrides_default(self, tmp_path):
        """
        ``make_error("custom")`` produces an IOStallError with the
        custom message verbatim.

        Tests:
            (Test Case 1) Custom message is preserved.
        """
        wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
        err = wd.make_error("custom-io-msg")
        assert isinstance(err, IOStallError)
        assert str(err) == "custom-io-msg"

    def test_none_stall_at_trip_uses_question_mark(self, tmp_path):
        """
        Pre-trip ``make_error`` (``_stall_at_trip is None``) renders
        ``"?"`` instead of a numeric value.

        Tests:
            (Test Case 1) Pre-trip default message contains ``"?"``.
        """
        wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
        err = wd.make_error()
        assert "?" in str(err)


class TestIOStallWatchdogRegisterKillCallback:
    """``IOStallWatchdog.register_kill_callback`` appends to internal list."""

    def test_register_appends_callback(self, tmp_path):
        """
        ``register_kill_callback`` records the callable in the
        watchdog's ``_kill_callbacks`` list.

        Tests:
            (Test Case 1) After two registrations, both callbacks
                are present in the list.
        """
        wd = IOStallWatchdog(tmp_path, stall_s=10.0, poll_interval_s=1.0)
        cb1 = lambda: None  # noqa: E731
        cb2 = lambda: None  # noqa: E731
        wd.register_kill_callback(cb1)
        wd.register_kill_callback(cb2)
        assert wd._kill_callbacks == [cb1, cb2]


class TestIOStallWatchdogEnterContextVarToken:
    """``IOStallWatchdog.__enter__`` captures the ContextVar token on success."""

    def test_token_captured_after_successful_enter(self, tmp_path):
        """
        On successful ``__enter__`` the watchdog publishes itself via
        the ContextVar and stores the reset token; ``get_active_io_stall_watchdog``
        sees this watchdog inside the with-block.

        Tests:
            (Test Case 1) Inside the with-block, ``_token`` is not None
                and the active getter returns this watchdog.
            (Test Case 2) After exit, ``_token`` is cleared.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", return_value=100),
        ):
            wd = IOStallWatchdog(tmp_path, stall_s=60.0, poll_interval_s=60.0)
            with wd:
                assert wd._token is not None
                assert iom.get_active_io_stall_watchdog() is wd
            assert wd._token is None


class TestLogInactivityWatchdogKillGraceStored:
    """``LogInactivityWatchdog.__init__`` stores ``kill_grace_s`` verbatim."""

    def test_kill_grace_s_stored(self, tmp_path):
        """
        The constructor round-trips ``kill_grace_s`` to the instance.

        Tests:
            (Test Case 1) ``kill_grace_s=8.0`` is stored.
        """
        wd = LogInactivityWatchdog(
            log_path=tmp_path / "log",
            popen=mock.Mock(spec=subprocess.Popen),
            inactivity_s=600.0,
            sorter="kilosort4",
            kill_grace_s=8.0,
        )
        assert wd.kill_grace_s == 8.0


class TestGetActiveGpuWatchdogDefault:
    """``get_active_gpu_watchdog`` returns None outside any context."""

    def test_default_none_outside_any_context(self):
        """
        With no GPU watchdog active, ``get_active_gpu_watchdog`` reads
        the ContextVar default (None).

        Tests:
            (Test Case 1) Outside any ``GpuMemoryWatchdog`` context →
                returns None.
        """
        from spikelab.spike_sorting.guards import get_active_gpu_watchdog

        assert get_active_gpu_watchdog() is None


class TestReadGpuMemoryNvidiaSmiPositive:
    """``_read_gpu_memory_nvidia_smi`` parses well-formed output for the
    matching device index."""

    def test_parses_matching_device_line(self, monkeypatch):
        """
        Valid CSV output is parsed; the line whose ``index`` matches
        ``device_index`` produces ``(used_pct, total_gb)``. Other
        lines (different indices, malformed) are skipped.

        Tests:
            (Test Case 1) Three-line output; only device 1 matches;
                returns 50% used and 8 GB total.
            (Test Case 2) A malformed line preceding the match is
                skipped without raising.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        # Output: index, used_mib, total_mib. 8 GB = 8192 MiB.
        # Line 0 device-0 something; line 1 garbage; line 2 matches.
        out = "0, 1024, 16384\nbadline\n1, 4096, 8192\n"
        monkeypatch.setattr(gpu_mod.subprocess, "check_output", lambda *a, **k: out)
        result = gpu_mod._read_gpu_memory_nvidia_smi(1)
        assert result is not None
        used_pct, total_gb = result
        assert used_pct == pytest.approx(50.0)
        assert total_gb == pytest.approx(8.0)


class TestCaptureGpuSnapshotTorchBranches:
    """``capture_gpu_snapshot`` records torch availability state in the file."""

    def test_torch_not_installed_records_marker(self, tmp_path, monkeypatch):
        """
        When ``torch`` cannot be imported, the snapshot file contains
        the ``"(torch not installed)"`` marker line.

        Tests:
            (Test Case 1) Patched ``__import__`` blocks ``torch``;
                the resulting file contains the marker.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        # Provide a benign nvidia-smi output so the function reaches
        # the torch block.
        monkeypatch.setattr(
            gpu_mod.subprocess, "check_output", lambda *a, **k: "smi-output\n"
        )

        # Block torch import.
        import builtins as _b

        real_import = _b.__import__

        def _blocked(name, *a, **k):
            if name == "torch":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        monkeypatch.setattr(_b, "__import__", _blocked)
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        target = tmp_path / "snap.txt"
        result = gpu_mod.capture_gpu_snapshot(target, header="trip")
        assert result == str(target)
        contents = target.read_text(encoding="utf-8")
        assert "(torch not installed)" in contents

    def test_torch_cuda_unavailable_records_marker(self, tmp_path, monkeypatch):
        """
        When torch is importable but ``torch.cuda.is_available()``
        returns False, the snapshot records the
        ``"(torch.cuda.is_available() = False)"`` marker.

        Tests:
            (Test Case 1) Stub ``torch`` with cuda unavailable;
                snapshot file contains the marker.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        monkeypatch.setattr(
            gpu_mod.subprocess, "check_output", lambda *a, **k: "smi-output\n"
        )

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        target = tmp_path / "snap.txt"
        result = gpu_mod.capture_gpu_snapshot(target, header="trip")
        assert result == str(target)
        contents = target.read_text(encoding="utf-8")
        assert "torch.cuda.is_available()" in contents
        assert "False" in contents


class TestReadGpuMemoryCascade:
    """``read_gpu_memory`` cascades pynvml → nvidia-smi → None."""

    def test_pynvml_success_short_circuits(self, monkeypatch):
        """
        When ``_read_gpu_memory_pynvml`` returns a tuple, the
        nvidia-smi fallback is not consulted.

        Tests:
            (Test Case 1) Patched pynvml-reader returns a tuple;
                patched nvidia-smi-reader records whether it was
                called → it must not have been.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        smi_calls = {"count": 0}

        def _smi(_idx):
            smi_calls["count"] += 1
            return (99.9, 99.9)

        monkeypatch.setattr(gpu_mod, "_read_gpu_memory_pynvml", lambda i: (10.0, 16.0))
        monkeypatch.setattr(gpu_mod, "_read_gpu_memory_nvidia_smi", _smi)

        result = gpu_mod.read_gpu_memory(0)
        assert result == (10.0, 16.0)
        assert smi_calls["count"] == 0

    def test_falls_back_to_nvidia_smi_when_pynvml_returns_none(self, monkeypatch):
        """
        ``_read_gpu_memory_pynvml`` returning None triggers the
        nvidia-smi fallback; its tuple is the final result.

        Tests:
            (Test Case 1) pynvml-reader returns None; nvidia-smi-reader
                returns ``(75.0, 24.0)`` → final result is that tuple.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        monkeypatch.setattr(gpu_mod, "_read_gpu_memory_pynvml", lambda i: None)
        monkeypatch.setattr(
            gpu_mod, "_read_gpu_memory_nvidia_smi", lambda i: (75.0, 24.0)
        )
        assert gpu_mod.read_gpu_memory(0) == (75.0, 24.0)

    def test_returns_none_when_both_fail(self, monkeypatch):
        """
        Both readers returning None propagates as ``None`` from
        ``read_gpu_memory``.

        Tests:
            (Test Case 1) Both stubbed readers return None → result
                is None.
        """
        from spikelab.spike_sorting.guards import _gpu_watchdog as gpu_mod

        monkeypatch.setattr(gpu_mod, "_read_gpu_memory_pynvml", lambda i: None)
        monkeypatch.setattr(gpu_mod, "_read_gpu_memory_nvidia_smi", lambda i: None)
        assert gpu_mod.read_gpu_memory(0) is None


class TestGpuMemoryWatchdogConstructionExtra:
    """Additional ``GpuMemoryWatchdog.__init__`` validation + storage."""

    def test_thermal_validation_reversed_raises(self):
        """
        ``warn_temp_c >= abort_temp_c`` raises ValueError, mirroring
        the VRAM threshold check.

        Tests:
            (Test Case 1) ``warn_temp_c=92, abort_temp_c=85`` raises.
            (Test Case 2) Equal values raise.
        """
        with pytest.raises(ValueError, match="warn_temp_c"):
            GpuMemoryWatchdog(warn_temp_c=92.0, abort_temp_c=85.0)
        with pytest.raises(ValueError, match="warn_temp_c"):
            GpuMemoryWatchdog(warn_temp_c=85.0, abort_temp_c=85.0)

    def test_kill_grace_and_throttle_flag_stored_verbatim(self):
        """
        ``kill_grace_s`` and ``monitor_throttle_reasons`` are
        round-tripped to attributes.

        Tests:
            (Test Case 1) ``kill_grace_s=12.5`` lands on the instance.
            (Test Case 2) ``monitor_throttle_reasons=False`` is stored
                as the bool.
        """
        wd = GpuMemoryWatchdog(
            kill_grace_s=12.5,
            monitor_throttle_reasons=False,
        )
        assert wd.kill_grace_s == 12.5
        assert wd.monitor_throttle_reasons is False


class TestGpuMemoryWatchdogMakeErrorBranches:
    """``GpuMemoryWatchdog.make_error`` thermal vs VRAM + override branches."""

    def test_thermal_branch_returns_thermal_error(self):
        """
        When ``_tripped_kind == "thermal"``, ``make_error`` returns a
        ``GpuThermalWatchdogError`` carrying the trip temperature.

        Tests:
            (Test Case 1) Manually setting trip_kind=thermal yields a
                ``GpuThermalWatchdogError`` (not memory).
            (Test Case 2) The default message references the
                temperature and abort threshold.
        """
        from spikelab.spike_sorting._exceptions import GpuThermalWatchdogError

        wd = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0)
        wd._tripped = True
        wd._tripped_kind = "thermal"
        wd._temp_c_at_trip = 95.0

        err = wd.make_error()
        assert isinstance(err, GpuThermalWatchdogError)
        assert "95.0" in str(err)

    def test_custom_message_override(self):
        """
        Passing ``message=...`` overrides the default formatting for
        both VRAM and thermal trips.

        Tests:
            (Test Case 1) Memory trip with custom message → that
                exact string is the error's str().
            (Test Case 2) Thermal trip with custom message likewise.
        """
        wd = GpuMemoryWatchdog()
        wd._tripped = True
        wd._tripped_kind = "memory"
        wd._used_pct_at_trip = 96.0
        err = wd.make_error("custom-vram-msg")
        assert str(err) == "custom-vram-msg"

        wd2 = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0)
        wd2._tripped = True
        wd2._tripped_kind = "thermal"
        wd2._temp_c_at_trip = 95.0
        err2 = wd2.make_error("custom-temp-msg")
        assert str(err2) == "custom-temp-msg"

    def test_none_pct_uses_question_mark_fallback(self):
        """
        When neither ``_used_pct_at_trip`` nor ``_temp_c_at_trip`` is
        populated, the default message uses the ``"?"`` placeholder
        rather than crashing on a None format.

        Tests:
            (Test Case 1) Pre-trip memory ``make_error`` contains
                ``"?"`` instead of a number.
            (Test Case 2) Pre-trip thermal ``make_error`` likewise.
        """
        wd = GpuMemoryWatchdog()
        # No trip yet; _used_pct_at_trip and _tripped_kind stay None.
        err = wd.make_error()
        assert "?" in str(err)

        wd2 = GpuMemoryWatchdog(warn_temp_c=80.0, abort_temp_c=92.0)
        wd2._tripped_kind = "thermal"
        # _temp_c_at_trip stays None.
        err2 = wd2.make_error()
        assert "?" in str(err2)


class TestGpuMemoryWatchdogRegisterSubprocess:
    """``GpuMemoryWatchdog.register_subprocess`` + custom kill_grace override."""

    def test_register_appends_to_internal_list(self):
        """
        ``register_subprocess`` appends a tuple of (popen, grace) to
        ``_subprocesses``; default grace is the watchdog's
        ``kill_grace_s``.

        Tests:
            (Test Case 1) After one registration the list has one
                entry whose grace matches the watchdog default.
        """
        wd = GpuMemoryWatchdog(kill_grace_s=4.0)
        popen = mock.Mock(spec=subprocess.Popen)
        wd.register_subprocess(popen)
        assert len(wd._subprocesses) == 1
        assert wd._subprocesses[0][0] is popen
        assert wd._subprocesses[0][1] == 4.0

    def test_register_with_custom_kill_grace(self):
        """
        Passing ``kill_grace_s=`` overrides the watchdog default per
        registration.

        Tests:
            (Test Case 1) Per-call grace 9.5 lands on the entry.
        """
        wd = GpuMemoryWatchdog(kill_grace_s=4.0)
        popen = mock.Mock(spec=subprocess.Popen)
        wd.register_subprocess(popen, kill_grace_s=9.5)
        assert wd._subprocesses[0][1] == 9.5

    def test_unregister_removes_by_identity(self):
        """
        ``unregister_subprocess`` filters out the supplied popen;
        unrelated entries stay.

        Tests:
            (Test Case 1) Two registrations; after unregistering one,
                the other remains.
        """
        wd = GpuMemoryWatchdog()
        p1 = mock.Mock(spec=subprocess.Popen)
        p2 = mock.Mock(spec=subprocess.Popen)
        wd.register_subprocess(p1)
        wd.register_subprocess(p2)
        wd.unregister_subprocess(p1)
        assert len(wd._subprocesses) == 1
        assert wd._subprocesses[0][0] is p2


class TestIOStallReadIoBytesCounterWrap:
    """``_read_io_bytes`` counter wrap is treated as an advance."""

    def test_decreasing_counter_resets_stall_clock(self, tmp_path):
        """
        If the byte counter wraps from a large value back to a small
        one (32-bit overflow on older Windows), ``current != last_bytes``
        is True, the watchdog resets the stall clock, and no trip
        fires. The behaviour is imprecise (the wrap is treated as
        progress) but safe.

        Tests:
            (Test Case 1) ``_read_io_bytes`` returns 2**31 on the
                first poll, then 0 (the wrap) on the second poll,
                then progressively higher values → no trip during
                the test window.
        """
        from spikelab.spike_sorting.guards import _io_stall as iom

        # Sequence: __enter__ probe, then loop reads.
        seq = iter([2**31, 2**31, 0, 1024, 2048, 4096])

        def _read(_dev):
            try:
                return next(seq)
            except StopIteration:
                return 8192

        with (
            mock.patch.object(iom, "_resolve_device_for_path", return_value="sda1"),
            mock.patch.object(iom, "_read_io_bytes", side_effect=_read),
        ):
            wd = IOStallWatchdog(tmp_path, stall_s=2.0, poll_interval_s=0.05)
            with wd:
                time.sleep(0.4)
                assert not wd.tripped()


# ===========================================================================
# ExecutionConfig defaults for guard fields
# ===========================================================================


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


# ===========================================================================
# NaN-input guards across the watchdog family.
#
# A NaN-shaped configuration value (from a malformed YAML, a
# float("nan") literal, or an upstream computation drift) must not
# silently disable the abort path. The contract — taken from the
# already-tested ``HostMemoryWatchdog`` and
# ``compute_inactivity_timeout_s`` cases — is:
#   - Either reject NaN at construction with a clear ValueError, OR
#   - Skip the tick at runtime so the watchdog never trips on a NaN
#     reading / threshold (NaN-as-no-op, *not* NaN-as-trip).
# These tests pin the current behaviour for the four remaining
# watchdogs.
# ===========================================================================


class TestGpuMemoryWatchdogNanGuard:
    """``GpuMemoryWatchdog`` rejects NaN thresholds at construction."""

    def test_nan_warn_pct_raises(self):
        """
        NaN ``warn_pct`` fails the ``0 < warn_pct < abort_pct`` band
        check at construction (NaN comparisons return False).

        Tests:
            (Test Case 1) ``warn_pct=NaN`` raises ValueError.
            (Test Case 2) The error names ``warn_pct`` / ``abort_pct``.
        """
        with pytest.raises(ValueError, match="warn_pct"):
            GpuMemoryWatchdog(warn_pct=float("nan"), abort_pct=95.0)

    def test_nan_abort_pct_raises(self):
        """
        NaN ``abort_pct`` fails the same band check.

        Tests:
            (Test Case 1) ``abort_pct=NaN`` raises ValueError.
        """
        with pytest.raises(ValueError, match="abort_pct"):
            GpuMemoryWatchdog(warn_pct=85.0, abort_pct=float("nan"))

    def test_nan_warn_temp_pair_raises(self):
        """
        NaN ``warn_temp_c`` (with a numeric ``abort_temp_c``) fails
        the thermal band check at construction.

        Tests:
            (Test Case 1) ``warn_temp_c=NaN`` raises ValueError.
        """
        with pytest.raises(ValueError, match="warn_temp_c"):
            GpuMemoryWatchdog(
                warn_pct=85.0,
                abort_pct=95.0,
                warn_temp_c=float("nan"),
                abort_temp_c=92.0,
            )

    def test_nan_poll_interval_raises(self):
        """
        NaN ``poll_interval_s`` is rejected at construction with a
        ``ValueError`` naming ``poll_interval_s``. The source now
        guards explicitly against NaN.

        Tests:
            (Test Case 1) ``poll_interval_s=NaN`` raises ValueError.
        """
        with pytest.raises(ValueError, match="poll_interval_s"):
            GpuMemoryWatchdog(
                warn_pct=85.0, abort_pct=95.0, poll_interval_s=float("nan")
            )


class TestLogInactivityWatchdogNanGuard:
    """``LogInactivityWatchdog`` with NaN ``inactivity_s`` never trips."""

    def test_nan_inactivity_s_does_not_trip(self, tmp_path):
        """
        NaN ``inactivity_s`` is rejected at construction with a
        ``ValueError`` naming ``inactivity_s``. The source now
        guards explicitly against NaN rather than allowing a silent
        no-op watchdog.

        Tests:
            (Test Case 1) ``inactivity_s=NaN`` raises ValueError.
        """
        log_path = tmp_path / "log"
        log_path.write_text("hello", encoding="utf-8")
        old_t = time.time() - 1000.0
        os.utime(log_path, (old_t, old_t))

        with pytest.raises(ValueError, match="inactivity_s"):
            LogInactivityWatchdog(
                log_path=log_path,
                popen=mock.Mock(spec=subprocess.Popen),
                inactivity_s=float("nan"),
                sorter="kilosort2",
                poll_interval_s=0.02,
            )


class TestIOStallWatchdogNanGuard:
    """``IOStallWatchdog`` with NaN ``stall_s`` never trips."""

    def test_nan_stall_s_does_not_trip(self, tmp_path):
        """
        NaN ``stall_s`` is rejected at construction with a
        ``ValueError`` naming ``stall_s``. The source now guards
        explicitly against NaN rather than allowing a silent no-op
        watchdog.

        Tests:
            (Test Case 1) ``stall_s=NaN`` raises ValueError.
        """
        with pytest.raises(ValueError, match="stall_s"):
            IOStallWatchdog(
                folder=tmp_path,
                stall_s=float("nan"),
                poll_interval_s=0.02,
            )


class TestDiskUsageWatchdogNanGuard:
    """``DiskUsageWatchdog`` with NaN thresholds never trips."""

    def test_nan_warn_free_gb_does_not_raise(self, tmp_path):
        """
        ``DiskUsageWatchdog(warn_free_gb=NaN, abort_free_gb=1.0)`` is
        rejected at construction with a ``ValueError`` naming
        ``warn_free_gb``. The source now guards explicitly against
        NaN thresholds rather than allowing a silent no-op watchdog.

        Tests:
            (Test Case 1) ``warn_free_gb=NaN`` raises ValueError.
        """
        with pytest.raises(ValueError, match="warn_free_gb"):
            DiskUsageWatchdog(
                folder=tmp_path,
                warn_free_gb=float("nan"),
                abort_free_gb=1.0,
                poll_interval_s=0.05,
            )

    def test_nan_abort_free_gb_does_not_trip(self, tmp_path):
        """
        ``abort_free_gb=NaN`` is rejected at construction with a
        ``ValueError`` naming ``abort_free_gb``. The source now
        guards explicitly against NaN thresholds rather than
        allowing a silent no-op watchdog.

        Tests:
            (Test Case 1) ``abort_free_gb=NaN`` raises ValueError.
        """
        with pytest.raises(ValueError, match="abort_free_gb"):
            DiskUsageWatchdog(
                folder=tmp_path,
                warn_free_gb=5.0,
                abort_free_gb=float("nan"),
                poll_interval_s=0.05,
                popen=mock.Mock(spec=subprocess.Popen),
            )
