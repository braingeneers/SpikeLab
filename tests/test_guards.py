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
        """
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
        """
        monkeypatch.setattr(preflight_mod, "_check_sorter_dependencies", lambda c: [])
        monkeypatch.setattr(preflight_mod, "_check_gpu_device_present", lambda c: None)
        monkeypatch.setattr(
            preflight_mod, "_check_recording_sample_rate", lambda c, recs: []
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
        findings = run_preflight(cfg, [], ["/inter"], ["/results"])
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

    def test_no_findings_passes_silently(self, capsys):
        """
        Empty findings list returns without raising.

        Tests:
            (Test Case 1) report_findings([]) does not raise.
            (Test Case 2) Prints the "all checks passed" line.
        """
        report_findings([])
        out = capsys.readouterr().out
        assert "all checks passed" in out

    def test_warn_only_does_not_raise(self, capsys):
        """
        Warn-level findings print but do not raise in default mode.

        Tests:
            (Test Case 1) No exception raised.
            (Test Case 2) Output contains the WARN marker.
        """
        findings = [PreflightFinding(level="warn", code="low_ram", message="m")]
        report_findings(findings, strict=False)
        out = capsys.readouterr().out
        assert "WARN" in out

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

    def test_disabled_when_inactivity_nonpositive(self, tmp_path):
        """
        Zero / negative inactivity_s also disables the watchdog.

        Tests:
            (Test Case 1) inactivity_s=0 → disabled.
            (Test Case 2) inactivity_s=-5 → disabled.
        """
        for bad in (0, -5):
            wd = LogInactivityWatchdog(
                log_path=tmp_path / "log",
                popen=mock.Mock(spec=subprocess.Popen),
                inactivity_s=bad,
                sorter="x",
            )
            assert wd._enabled is False


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

        # Inject a fake psutil so the watchdog loop runs without the
        # real OS readings, and use a tiny poll interval for speed.
        wd = HostMemoryWatchdog(warn_pct=85.0, abort_pct=92.0, poll_interval_s=0.02)
        wd._psutil = _FakePsutil
        with wd:
            time.sleep(0.15)
        assert wd.tripped() is False


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
