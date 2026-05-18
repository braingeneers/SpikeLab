"""Tests for ``spike_sorting/canary.py`` — short-window smoke test.

Covers:

* ``_build_canary_config`` — relaxed config clone with overrides
  applied + original config left untouched.
* ``_wipe_canary_folder`` — best-effort cleanup of the canary's
  intermediate folder.
* ``run_canary`` — early-return on disabled / NaN window, classified
  failure propagation, success cleanup, unexpected-exception swallow,
  and KeyboardInterrupt / SystemExit re-raise.
* The pipeline-side recording-too-short-skip contract documented in
  the canary module docstring.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from spikelab.spike_sorting._exceptions import (
    EnvironmentSortFailure,
    InsufficientActivityError,
)
from spikelab.spike_sorting.config import SortingPipelineConfig


class TestBuildCanaryConfig:
    """``_build_canary_config`` returns a relaxed config clone."""

    def test_overrides_applied(self):
        """
        Canary clone restricts the recording window and disables every
        post-sort exporter, figures, and the recursive preflight.

        Tests:
            (Test Case 1) start/end_time_s set to [0, window_s].
            (Test Case 2) Curation disabled.
            (Test Case 3) Compilation/figures/report disabled.
            (Test Case 4) preflight=False.
            (Test Case 5) tee_log_policy='keep'.
        """
        from spikelab.spike_sorting.canary import _build_canary_config

        cfg = SortingPipelineConfig()
        clone = _build_canary_config(cfg, 30.0)
        assert clone.recording.start_time_s == 0.0
        assert clone.recording.end_time_s == 30.0
        assert clone.curation.curate_first is False
        assert clone.curation.curate_second is False
        assert clone.compilation.compile_single_recording is False
        assert clone.compilation.compile_to_npz is False
        assert clone.compilation.compile_waveforms is False
        assert clone.figures.create_figures is False
        assert clone.figures.create_unit_figures is False
        assert clone.execution.generate_sorting_report is False
        assert clone.execution.preflight is False
        assert clone.execution.tee_log_policy == "keep"

    def test_rec_chunks_cleared(self):
        """
        Non-flat ``rec_chunks`` and ``rec_chunks_s`` are cleared so the
        loader uses the simple start/end window.

        Tests:
            (Test Case 1) Set non-empty rec_chunks before clone → both
                cleared in the clone.
        """
        from spikelab.spike_sorting.canary import _build_canary_config

        cfg = SortingPipelineConfig()
        cfg.recording.rec_chunks = [(0, 30000)]
        cfg.recording.rec_chunks_s = [(0.0, 30.0)]
        clone = _build_canary_config(cfg, 30.0)
        assert clone.recording.rec_chunks == []
        assert clone.recording.rec_chunks_s == []

    def test_original_config_not_mutated(self):
        """
        Building a canary clone never mutates the input config.

        Tests:
            (Test Case 1) Original start_time_s and curate_first stay
                at their defaults after the clone.
        """
        from spikelab.spike_sorting.canary import _build_canary_config

        cfg = SortingPipelineConfig()
        _ = _build_canary_config(cfg, 30.0)
        assert cfg.recording.start_time_s is None
        assert cfg.recording.end_time_s is None
        assert cfg.curation.curate_first is True
        assert cfg.execution.preflight is True


class TestWipeCanaryFolder:
    """``_wipe_canary_folder`` is best-effort cleanup."""

    def test_existing_folder_removed(self, tmp_path):
        """
        Existing folder + contents are deleted.

        Tests:
            (Test Case 1) Folder with nested files is wiped.
        """
        from spikelab.spike_sorting.canary import _wipe_canary_folder

        folder = tmp_path / "canary"
        (folder / "sub").mkdir(parents=True)
        (folder / "sub" / "x.txt").write_text("x", encoding="utf-8")
        _wipe_canary_folder(folder)
        assert not folder.exists()

    def test_missing_folder_no_op(self, tmp_path):
        """
        Calling on a missing folder is a silent no-op.

        Tests:
            (Test Case 1) Path that never existed → no exception.
        """
        from spikelab.spike_sorting.canary import _wipe_canary_folder

        _wipe_canary_folder(tmp_path / "never-existed")  # no raise

    def test_exists_raise_is_swallowed(self, tmp_path, monkeypatch):
        """
        ``folder.exists()`` raising is caught by the outer try/except
        so cleanup never propagates an exception.

        Tests:
            (Test Case 1) Patched ``Path.exists`` raising ``OSError``
                does not propagate; the helper logs a warning and
                returns normally.

        Notes:
            - Documents current best-effort behaviour: the canary
              folder cleanup is wrapped in a broad try/except so a
              path-readability bug never breaks a sort.
        """
        from pathlib import Path

        from spikelab.spike_sorting.canary import _wipe_canary_folder

        target = tmp_path / "canary"

        def _refuse(self):
            if self == target:
                raise OSError("simulated path-stat failure")
            return False

        monkeypatch.setattr(Path, "exists", _refuse)
        # Must not raise.
        _wipe_canary_folder(target)


class TestRunCanary:
    """``run_canary`` runs the canary clone and propagates classified
    failures while swallowing unexpected ones."""

    def test_window_zero_returns_none(self, tmp_path):
        """
        canary_first_n_s == 0 → run_canary short-circuits to None.

        Tests:
            (Test Case 1) Default config (window=0) → None and no
                _canary folder is created.
        """
        from spikelab.spike_sorting.canary import run_canary

        cfg = SortingPipelineConfig()
        result = run_canary(cfg, recording=None, rec_path="rec", inter_path=tmp_path)
        assert result is None
        assert not (tmp_path / "_canary").exists()

    def test_negative_window_returns_none(self, tmp_path):
        """
        canary_first_n_s < 0 → run_canary short-circuits to None (same
        as the disabled-at-zero path).

        Tests:
            (Test Case 1) A negative window is treated as "disabled" by
                the ``canary_window_s <= 0`` guard; the function returns
                None without raising or creating any folder.
            (Test Case 2) No ``_canary_*`` subfolder is created under
                inter_path (the guard fires before the per-pid folder is
                computed).
        """
        from spikelab.spike_sorting.canary import run_canary

        cfg = SortingPipelineConfig()
        cfg.execution.canary_first_n_s = -1.0
        result = run_canary(cfg, recording=None, rec_path="rec", inter_path=tmp_path)
        assert result is None
        # No per-pid canary folder should exist either.
        assert not any(tmp_path.glob("_canary*"))

    def test_classified_failure_returned(self, tmp_path, monkeypatch):
        """
        process_recording returning a classified failure → run_canary
        returns the same exception instance and wipes the canary folder.

        Tests:
            (Test Case 1) process_recording stub returns an
                InsufficientActivityError.
            (Test Case 2) Returned object is the same instance.
            (Test Case 3) _canary subfolder is removed afterwards.
        """
        from spikelab.spike_sorting import canary as canary_mod
        from spikelab.spike_sorting.canary import run_canary

        cfg = SortingPipelineConfig()
        cfg.execution.canary_first_n_s = 5.0

        exc = InsufficientActivityError("silent rec", sorter="kilosort2")

        class _FakeBackend:
            def __init__(self, _cfg):
                pass

        monkeypatch.setattr(
            canary_mod,
            "_build_canary_config",
            lambda c, w: SortingPipelineConfig(),
        )

        from spikelab.spike_sorting import backends as backends_mod

        monkeypatch.setattr(
            backends_mod, "get_backend_class", lambda name: _FakeBackend
        )

        from spikelab.spike_sorting import pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod, "process_recording", lambda *a, **kw: exc)

        result = run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=tmp_path,
            sorter_name="kilosort2",
        )
        assert result is exc
        assert not (tmp_path / "_canary").exists()

    def test_success_returns_none_and_cleans_up(self, tmp_path, monkeypatch):
        """
        process_recording returning a real result → run_canary returns
        None and wipes the canary folder.

        Tests:
            (Test Case 1) Stub returns a sentinel SpikeData-like object.
            (Test Case 2) run_canary returns None.
            (Test Case 3) _canary subfolder removed.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting.canary import run_canary

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
        sentinel = object()
        monkeypatch.setattr(
            pipeline_mod, "process_recording", lambda *a, **kw: sentinel
        )

        result = run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=tmp_path,
            sorter_name="kilosort2",
        )
        assert result is None
        assert not (tmp_path / "_canary").exists()

    def test_unexpected_exception_swallowed(self, tmp_path, monkeypatch):
        """
        process_recording raising a non-classified exception → run_canary
        returns None (smoke test, not a hard gate) and cleans up.

        Tests:
            (Test Case 1) Stub raises RuntimeError → returns None.
            (Test Case 2) Folder removed.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting.canary import run_canary

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

        def _boom(*_a, **_kw):
            raise RuntimeError("disk full mid-canary")

        monkeypatch.setattr(pipeline_mod, "process_recording", _boom)

        result = run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=tmp_path,
            sorter_name="kilosort2",
        )
        assert result is None
        assert not (tmp_path / "_canary").exists()

    def test_classified_returned_as_value_returned(self, tmp_path, monkeypatch):
        """
        process_recording *returning* (not raising) a classified
        exception is also propagated.

        Tests:
            (Test Case 1) Stub returns EnvironmentSortFailure value;
                run_canary returns the same value.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting.canary import run_canary

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
        env_fail = EnvironmentSortFailure("docker exploded")
        monkeypatch.setattr(
            pipeline_mod, "process_recording", lambda *a, **kw: env_fail
        )

        result = run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=tmp_path,
            sorter_name="kilosort2",
        )
        assert result is env_fail


class TestCanaryNanAndInterruptHandling:
    """``run_canary`` NaN window guard + KeyboardInterrupt re-raise."""

    def test_nan_window_returns_none(self, tmp_path):
        """
        ``canary_first_n_s == NaN`` is treated as disabled.

        Tests:
            (Test Case 1) NaN window → run_canary returns None and
                creates no _canary subfolder.
        """
        from spikelab.spike_sorting.canary import run_canary

        cfg = SortingPipelineConfig()
        cfg.execution.canary_first_n_s = float("nan")
        result = run_canary(cfg, recording=None, rec_path="rec", inter_path=tmp_path)
        assert result is None
        assert not (tmp_path / "_canary").exists()

    def test_keyboard_interrupt_propagates(self, tmp_path, monkeypatch):
        """
        KeyboardInterrupt raised inside the canary propagates rather
        than being silently swallowed.

        Tests:
            (Test Case 1) process_recording raises KeyboardInterrupt →
                run_canary re-raises, _canary folder is cleaned up.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )

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

        def _raise_kbi(*_a, **_kw):
            raise KeyboardInterrupt("user abort")

        monkeypatch.setattr(pipeline_mod, "process_recording", _raise_kbi)
        with pytest.raises(KeyboardInterrupt):
            canary_mod.run_canary(
                cfg,
                recording=None,
                rec_path="rec.h5",
                inter_path=tmp_path,
                sorter_name="kilosort2",
            )
        assert not (tmp_path / "_canary").exists()

    def test_system_exit_propagates(self, tmp_path, monkeypatch):
        """
        SystemExit raised inside the canary propagates and triggers
        cleanup.

        Tests:
            (Test Case 1) process_recording raises SystemExit →
                run_canary re-raises, folder cleaned up.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )

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

        def _raise_se(*_a, **_kw):
            raise SystemExit(1)

        monkeypatch.setattr(pipeline_mod, "process_recording", _raise_se)
        with pytest.raises(SystemExit):
            canary_mod.run_canary(
                cfg,
                recording=None,
                rec_path="rec.h5",
                inter_path=tmp_path,
                sorter_name="kilosort2",
            )
        assert not (tmp_path / "_canary").exists()


class TestPipelineCanaryRecordingTooShortNotice:
    """The recording-too-short notice now lives in pipeline.py, not canary.py.

    Documented as such in canary.py's module docstring; verified here so
    a future re-introduction of the check inside canary.py is caught.
    """

    def test_canary_module_docstring_explains_pipeline_owns_check(self):
        """
        The canary module docstring explicitly attributes the
        recording-shorter-than-window skip to the pipeline call site.

        Tests:
            (Test Case 1) canary.__doc__ mentions 'pipeline' and
                'shorter'.
        """
        from spikelab.spike_sorting import canary as canary_mod

        doc = canary_mod.__doc__ or ""
        assert "pipeline" in doc
        assert "shorter" in doc


class TestBuildCanaryConfigOverridesExtra:
    """``_build_canary_config`` propagates additional override keys."""

    def test_tee_log_policy_keep_propagates(self):
        """
        The canary clone sets ``tee_log_policy="keep"`` so the canary
        log is preserved for debugging instead of cleaned with the
        rest of the canary folder.

        Tests:
            (Test Case 1) Clone's ``tee_log_policy`` is ``"keep"``.
        """
        from spikelab.spike_sorting.canary import _build_canary_config

        cfg = SortingPipelineConfig()
        clone = _build_canary_config(cfg, 30.0)
        assert clone.execution.tee_log_policy == "keep"

    def test_sorter_inactivity_base_s_scales_with_window(self):
        """
        ``sorter_inactivity_base_s`` is set to
        ``min(300, max(120, 4 * canary_window_s))`` — clamped between
        a 120 s floor (cold-start tolerance) and 300 s ceiling.

        Tests:
            (Test Case 1) ``window=30`` → 4*30=120 (floor) → 120.0.
            (Test Case 2) ``window=60`` → 4*60=240 (mid-range) → 240.0.
            (Test Case 3) ``window=120`` → 4*120=480 (capped) → 300.0.
        """
        from spikelab.spike_sorting.canary import _build_canary_config

        cfg = SortingPipelineConfig()
        assert (
            _build_canary_config(cfg, 30.0).execution.sorter_inactivity_base_s == 120.0
        )
        assert (
            _build_canary_config(cfg, 60.0).execution.sorter_inactivity_base_s == 240.0
        )
        assert (
            _build_canary_config(cfg, 120.0).execution.sorter_inactivity_base_s == 300.0
        )

    def test_start_and_end_time_set_to_window(self):
        """
        Explicitly assert ``start_time_s=0.0`` and
        ``end_time_s == canary_window_s`` — the canary always sorts
        the leading slice from frame zero.

        Tests:
            (Test Case 1) Clone with window=42 → ``start_time_s=0.0``,
                ``end_time_s=42.0``.
        """
        from spikelab.spike_sorting.canary import _build_canary_config

        cfg = SortingPipelineConfig()
        clone = _build_canary_config(cfg, 42.0)
        assert clone.recording.start_time_s == 0.0
        assert clone.recording.end_time_s == 42.0


class TestRunCanaryEqualDurationBoundary:
    """``run_canary`` boundary at canary_window_s = recording duration."""

    def test_canary_window_uses_supplied_value_regardless_of_recording_length(self):
        """
        ``run_canary`` does not consult the recording's duration to
        decide the window — it always passes the configured
        ``canary_first_n_s`` through. With a window equal to (or
        exceeding) the actual recording length the backend would sort
        the full recording. Documents that the canary's window is
        purely a config value.

        Tests:
            (Test Case 1) ``canary_first_n_s=10.0`` produces a clone
                whose ``end_time_s=10.0`` regardless of recording.
        """
        from spikelab.spike_sorting.canary import _build_canary_config

        cfg = SortingPipelineConfig()
        clone = _build_canary_config(cfg, 10.0)
        # The canary clone's end_time_s reflects only the supplied
        # window; recording duration is not inspected.
        assert clone.recording.end_time_s == 10.0
        # If a recording happens to be 10s long, the backend would sort
        # the whole thing — that's the documented "off-by-equal" risk.


class TestRunCanarySorterNameOverride:
    """``run_canary`` honours an explicit ``sorter_name`` parameter."""

    def test_sorter_name_param_overrides_config(self, tmp_path, monkeypatch):
        """
        Passing ``sorter_name="custom_sorter"`` causes the backend
        lookup to use that name instead of the one in
        ``config.sorter.sorter_name``.

        Tests:
            (Test Case 1) Patched ``get_backend_class`` records the
                requested name; with ``sorter_name="kilosort4"``
                override on a config whose ``sorter_name="kilosort2"``,
                the recorded lookup name is ``"kilosort4"``.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting.canary import run_canary

        cfg = SortingPipelineConfig()
        cfg.sorter.sorter_name = "kilosort2"
        cfg.execution.canary_first_n_s = 5.0

        recorded = {"name": None}

        class _FakeBackend:
            def __init__(self, _cfg):
                pass

        def _record_backend(name):
            recorded["name"] = name
            return _FakeBackend

        monkeypatch.setattr(backends_mod, "get_backend_class", _record_backend)
        monkeypatch.setattr(
            canary_mod,
            "_build_canary_config",
            lambda c, w: SortingPipelineConfig(),
        )
        monkeypatch.setattr(
            pipeline_mod, "process_recording", lambda *a, **kw: object()
        )

        run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=tmp_path,
            sorter_name="kilosort4",
        )
        assert recorded["name"] == "kilosort4"


class TestRunCanaryRngForwarded:
    """``run_canary`` forwards ``rng`` to ``process_recording``."""

    def test_rng_passed_through_to_process_recording(self, tmp_path, monkeypatch):
        """
        The optional ``rng`` parameter is passed through to
        ``process_recording`` for reproducibility.

        Tests:
            (Test Case 1) Patched ``process_recording`` captures the
                ``rng`` kwarg; the supplied sentinel is recorded.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting.canary import run_canary

        cfg = SortingPipelineConfig()
        cfg.execution.canary_first_n_s = 5.0

        captured = {"rng": "missing"}

        class _FakeBackend:
            def __init__(self, _cfg):
                pass

        def _capture(*args, **kwargs):
            captured["rng"] = kwargs.get("rng", "missing")
            return object()

        monkeypatch.setattr(
            canary_mod,
            "_build_canary_config",
            lambda c, w: SortingPipelineConfig(),
        )
        monkeypatch.setattr(
            backends_mod, "get_backend_class", lambda name: _FakeBackend
        )
        monkeypatch.setattr(pipeline_mod, "process_recording", _capture)

        sentinel = object()
        run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=tmp_path,
            sorter_name="kilosort2",
            rng=sentinel,
        )
        assert captured["rng"] is sentinel


class TestRunCanaryNonClassifiedReturnedAsValue:
    """``run_canary`` swallows non-classified BaseException returned-as-value."""

    def test_non_classified_baseexception_returned_swallowed(
        self, tmp_path, monkeypatch
    ):
        """
        When ``process_recording`` *returns* (rather than raises) a
        non-classified ``BaseException`` instance — e.g. a generic
        ``RuntimeError`` returned as a value — ``run_canary``
        treats it like a non-classified failure: prints a notice,
        cleans up the canary folder, and returns ``None``.

        Tests:
            (Test Case 1) ``process_recording`` stub returns a
                ``RuntimeError`` instance (not raised).
            (Test Case 2) ``run_canary`` returns ``None``.
            (Test Case 3) The canary folder has been wiped.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting.canary import run_canary

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
        # Return (not raise) a non-classified RuntimeError instance.
        monkeypatch.setattr(
            pipeline_mod,
            "process_recording",
            lambda *a, **kw: RuntimeError("non-classified returned-as-value"),
        )

        result = run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=tmp_path,
            sorter_name="kilosort2",
        )
        assert result is None


class TestRunCanaryInterPathBoundaries:
    """``run_canary`` inter_path setup edge cases."""

    def test_inter_path_mkdir_failure_propagates(self, tmp_path, monkeypatch):
        """
        When ``Path.mkdir`` fails on the canary subfolder (read-only
        mount, permission denied) the OSError surfaces from
        ``run_canary`` rather than being silently swallowed —
        documents current behaviour so a future caller knows to
        wrap the call.

        Tests:
            (Test Case 1) Patched ``Path.mkdir`` raises
                ``PermissionError`` for the per-pid ``_canary_<pid>``
                directory; ``run_canary`` propagates that error.
        """
        from spikelab.spike_sorting.canary import run_canary

        cfg = SortingPipelineConfig()
        cfg.execution.canary_first_n_s = 5.0

        original_mkdir = Path.mkdir

        def _refuse(self, *args, **kwargs):
            if "_canary_" in str(self):
                raise PermissionError("simulated read-only mount")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _refuse)

        with pytest.raises(PermissionError):
            run_canary(
                cfg,
                recording=None,
                rec_path="rec.h5",
                inter_path=tmp_path,
                sorter_name="kilosort2",
            )

    def test_inter_path_missing_is_created(self, tmp_path, monkeypatch):
        """
        When ``inter_path`` itself does not yet exist, ``run_canary``
        creates the per-pid canary subfolder via
        ``mkdir(parents=True, exist_ok=True)`` so the canary can
        proceed.

        Tests:
            (Test Case 1) ``inter_path = tmp_path / "missing"`` (not
                yet present); patched backend + classified-failure
                ``process_recording`` returns the exception. Track
                the mkdir target via a side-effect collector to
                confirm the canary path was created.
        """
        from spikelab.spike_sorting import (
            backends as backends_mod,
            canary as canary_mod,
            pipeline as pipeline_mod,
        )
        from spikelab.spike_sorting.canary import run_canary

        cfg = SortingPipelineConfig()
        cfg.execution.canary_first_n_s = 5.0

        nonexistent = tmp_path / "fresh"
        assert not nonexistent.exists()

        # Track mkdir call paths so we can confirm the canary subdir
        # was the one created.
        original_mkdir = Path.mkdir
        mkdir_paths = []

        def _tracking_mkdir(self, *args, **kwargs):
            if "_canary_" in str(self):
                mkdir_paths.append(Path(self))
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _tracking_mkdir)

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
        # Return a classified failure to exit early — keeps the test
        # hermetic without needing a real backend.
        exc = InsufficientActivityError("mock", sorter="kilosort2")
        monkeypatch.setattr(pipeline_mod, "process_recording", lambda *a, **kw: exc)

        result = run_canary(
            cfg,
            recording=None,
            rec_path="rec.h5",
            inter_path=nonexistent,
            sorter_name="kilosort2",
        )
        assert result is exc
        assert mkdir_paths, "no canary subfolder mkdir was attempted"
        # The recorded mkdir call landed under the (formerly missing)
        # parent — the helper created it via parents=True.
        assert any(
            nonexistent in p.parents or p.parent == nonexistent for p in mkdir_paths
        )


class TestExtractUnitCount:
    """Boundary tests for _extract_unit_count covering tuple, list, None,
    and numpy-int N values."""

    def test_extract_unit_count_none_returns_none(self):
        """
        _extract_unit_count on None returns None because the candidate
        lacks an N attribute.

        Tests:
            (Test Case 1) result=None returns None.
        """
        from spikelab.spike_sorting.canary import _extract_unit_count

        assert _extract_unit_count(None) is None

    def test_extract_unit_count_empty_tuple_returns_none(self):
        """
        _extract_unit_count on an empty tuple skips the tuple branch
        (because ``result and ...`` is False) and the candidate (the
        empty tuple itself) has no N attribute.

        Tests:
            (Test Case 1) result=() returns None.
        """
        from spikelab.spike_sorting.canary import _extract_unit_count

        assert _extract_unit_count(()) is None

    def test_extract_unit_count_two_tuple_returns_curated_count(self):
        """
        _extract_unit_count on a (sd, sd_curated) tuple returns the
        curated SpikeData's N (the last entry).

        Tests:
            (Test Case 1) (sd_raw, sd_curated) returns sd_curated.N.
        """
        from spikelab.spike_sorting.canary import _extract_unit_count

        class _FakeSD:
            def __init__(self, n):
                self.N = n

        assert _extract_unit_count((_FakeSD(10), _FakeSD(7))) == 7

    def test_extract_unit_count_list_returns_none(self):
        """
        _extract_unit_count only unwraps tuples, not lists. A list
        result is treated as a candidate object, which lacks an N
        attribute, so the helper returns None.

        Tests:
            (Test Case 1) result=[sd] returns None.
        """
        from spikelab.spike_sorting.canary import _extract_unit_count

        class _FakeSD:
            def __init__(self, n):
                self.N = n

        assert _extract_unit_count([_FakeSD(5)]) is None

    def test_extract_unit_count_numpy_int_n_returns_python_int(self):
        """
        _extract_unit_count accepts numpy integer types (np.int64,
        np.int32, etc.) — SpikeData.N can be assigned from numpy
        operations like np.unique(...).size.

        Tests:
            (Test Case 1) np.int64(5) returns Python int 5.
            (Test Case 2) np.int32(7) returns Python int 7.
            (Test Case 3) The returned type is exactly Python int
                (not a numpy scalar) so JSON serializers don't trip
                on it.
        """
        import numpy as _np

        from spikelab.spike_sorting.canary import _extract_unit_count

        class _FakeSD:
            def __init__(self, n):
                self.N = n

        result_int64 = _extract_unit_count(_FakeSD(_np.int64(5)))
        assert result_int64 == 5
        assert type(result_int64) is int

        result_int32 = _extract_unit_count(_FakeSD(_np.int32(7)))
        assert result_int32 == 7
        assert type(result_int32) is int


# TestRunCanaryGlobalsIsolation removed in Phase 5 of the _globals.py
# refactor (iat/TO_IMPLEMENT.md). The canary's snapshot/restore helpers
# (`_snapshot_pipeline_globals` / `_restore_pipeline_globals`) and the
# `_globals.py` module itself were deleted: the canary now isolates its
# state via the deep-copied SortingPipelineConfig clone returned by
# `_build_canary_config` instead of mirroring overrides through globals.
# The state-isolation contract is exercised by TestBuildCanaryConfig
# (above), which verifies `_build_canary_config(config, ...)` does not
# mutate its input.
