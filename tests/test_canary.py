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
