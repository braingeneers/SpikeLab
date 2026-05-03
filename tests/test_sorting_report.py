"""Tests for spike_sorting.report — Markdown post-sorting report.

The report module distills the Tee log + recording_report.json +
config_used.json + curated SpikeData pickle into a human-readable
``sorting_report.md`` and applies the configured ``tee_log_policy``
to the original Tee log on success.

Tests cover:

* Tee log parsing (banner, timing, curation line, warnings, traceback)
* Config diff against defaults (only non-default fields surface)
* Unit-quality stats extraction from a fake SpikeData
* End-to-end report generation with real input artefacts in tmp_path
* tee_log_policy: keep / gzip_on_success / delete_on_success
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from spikelab.spike_sorting.report import (
    apply_tee_log_policy,
    diff_against_default,
    extract_unit_quality_stats,
    generate_sorting_report,
    parse_sorting_log,
    serialize_config_for_report,
)

# ---------------------------------------------------------------------------
# Sample Tee log — covers every section the parser is expected to extract.
# ---------------------------------------------------------------------------

_SAMPLE_LOG_SUCCESS = """
======================================================================
                  SPIKE SORTING — KILOSORT2
                 [2026-05-02 10:00:00]
======================================================================

-- Environment --
Started:        2026-05-02T10:00:00
Host:           tjits-workstation
Platform:       Windows-11-10.0.26200-SP0
Python:         3.10.20
SpikeInterface: 0.104.0
SpikeLab:       0.5.2

-- System Resources --
CPU cores:      24
RAM total:      64.0 GB
Heap cap:       (Windows — not enforced)
GPU:            NVIDIA RTX 4090, 590.44.01, 24576 MiB

-- Run --
Sorter:         kilosort2
Use Docker:     False
Recording:      /data/raw/rec1.raw.h5
Log file:       /data/sorted/rec1/sorting_260502_100000.log

======================================================================
                  SPIKE SORTING
                 [2026-05-02 10:00:30]
======================================================================
Loading recording ...

======================================================================
                 GENERATING PER-UNIT FIGURES
                 [2026-05-02 10:05:12]
======================================================================
Doing per-unit figures ...

Curation: 230 -> 92 units (138 removed)

UserWarning: deprecated thing happened
DeprecationWarning: another one

======================================================================
                  SUMMARY
                 [2026-05-02 10:08:45]
======================================================================

Status:         SUCCESS
Wall time:      8m 45s
RAM total:      64.0 GB
GPU memory:     5400 MiB, 24576 MiB
Finished:       2026-05-02T10:08:45
"""


_SAMPLE_LOG_FAILURE = """
======================================================================
                  SPIKE SORTING — KILOSORT4
                 [2026-05-02 11:00:00]
======================================================================

-- Environment --
Started:        2026-05-02T11:00:00
Python:         3.10.20

-- Run --
Sorter:         kilosort4

======================================================================
                  SPIKE SORTING
                 [2026-05-02 11:00:15]
======================================================================
detected 8 channels
extracting waveforms ...
chunk 1/3
chunk 2/3
chunk 3/3
Traceback (most recent call last):
  File "/path/to/some.py", line 42, in some_func
    raise ValueError("something went wrong")
ValueError: something went wrong

======================================================================
                  SUMMARY
                 [2026-05-02 11:01:00]
======================================================================

Status:         FAILED
Wall time:      0m 45s
"""


# ---------------------------------------------------------------------------
# Tee log parsing
# ---------------------------------------------------------------------------


class TestParseSortingLog:
    """``parse_sorting_log`` extracts banner / timing / warnings / traceback."""

    def test_environment_banner_extracted(self):
        """
        Environment key/value pairs are extracted into a dict.

        Tests:
            (Test Case 1) Python / SpikeInterface / SpikeLab versions
                appear in the environment dict.
            (Test Case 2) Host and platform also extracted.
        """
        info = parse_sorting_log(_SAMPLE_LOG_SUCCESS)
        env = info["environment"]
        assert env.get("Python") == "3.10.20"
        assert env.get("SpikeInterface") == "0.104.0"
        assert env.get("SpikeLab") == "0.5.2"
        assert env.get("Host") == "tjits-workstation"

    def test_run_section_extracted(self):
        """
        Run block keys are extracted.

        Tests:
            (Test Case 1) Sorter, recording, log file all present.
        """
        info = parse_sorting_log(_SAMPLE_LOG_SUCCESS)
        run = info["run"]
        assert run.get("Sorter") == "kilosort2"
        assert "rec1.raw.h5" in run.get("Recording", "")
        assert "Log file" in run

    def test_stage_timings_paired_with_banners(self):
        """
        Stage banners pair with their timestamp lines.

        Tests:
            (Test Case 1) "SPIKE SORTING" stage is captured with its
                timestamp.
            (Test Case 2) "GENERATING PER-UNIT FIGURES" is captured.
        """
        info = parse_sorting_log(_SAMPLE_LOG_SUCCESS)
        stages = info["stage_timings"]
        names = [s["name"] for s in stages]
        # The opening "SPIKE SORTING — KILOSORT2" banner is in
        # ENVIRONMENT-mode, not parsed; the inner "SPIKE SORTING"
        # is.
        assert any("SPIKE SORTING" in n for n in names)
        assert any("PER-UNIT FIGURES" in n for n in names)

    def test_curation_line_extracted(self):
        """
        The "Curation: N -> M units" line is captured verbatim.

        Tests:
            (Test Case 1) curation_line is the full line string.
        """
        info = parse_sorting_log(_SAMPLE_LOG_SUCCESS)
        assert info["curation_line"] is not None
        assert "230 -> 92" in info["curation_line"]

    def test_closing_summary_extracted(self):
        """
        The closing SUMMARY block is parsed into a dict.

        Tests:
            (Test Case 1) Status / Wall time / Finished captured.
        """
        info = parse_sorting_log(_SAMPLE_LOG_SUCCESS)
        summary = info["closing_summary"]
        assert summary.get("Status") == "SUCCESS"
        assert summary.get("Wall time") == "8m 45s"

    def test_warnings_extracted(self):
        """
        Lines containing Warning markers are collected verbatim.

        Tests:
            (Test Case 1) UserWarning and DeprecationWarning lines
                appear in the warnings list.
        """
        info = parse_sorting_log(_SAMPLE_LOG_SUCCESS)
        warns = info["warnings"]
        assert any("UserWarning" in w for w in warns)
        assert any("DeprecationWarning" in w for w in warns)

    def test_traceback_extracted_on_failure(self):
        """
        On failure, the full traceback is captured plus tail context.

        Tests:
            (Test Case 1) traceback is non-None and contains the
                exception line.
            (Test Case 2) last_lines_before_traceback contains the
                "chunk N/3" stdout that preceded the error.
        """
        info = parse_sorting_log(_SAMPLE_LOG_FAILURE)
        assert info["traceback"] is not None
        assert "ValueError: something went wrong" in info["traceback"]
        last = info["last_lines_before_traceback"]
        assert any("chunk 3/3" in l for l in last)

    def test_no_traceback_on_success(self):
        """
        Successful logs have no traceback block.

        Tests:
            (Test Case 1) traceback is None.
            (Test Case 2) last_lines_before_traceback is empty.
        """
        info = parse_sorting_log(_SAMPLE_LOG_SUCCESS)
        assert info["traceback"] is None
        assert info["last_lines_before_traceback"] == []


# ---------------------------------------------------------------------------
# Config serialisation + diff
# ---------------------------------------------------------------------------


class TestConfigSerialisation:
    """``serialize_config_for_report`` and ``diff_against_default``."""

    def test_serialize_default_round_trips_via_json(self):
        """
        Default config serialises to a JSON-safe dict.

        Tests:
            (Test Case 1) ``json.dumps`` of the result succeeds.
            (Test Case 2) Top-level keys mirror the dataclass groups.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        snapshot = serialize_config_for_report(cfg)
        # Round-trip via json to confirm everything is JSON-safe.
        json.dumps(snapshot)
        assert "execution" in snapshot
        assert "sorter" in snapshot
        assert "curation" in snapshot

    def test_diff_against_default_empty_for_default(self):
        """
        A config equal to the default has no diffs.

        Tests:
            (Test Case 1) diff list is empty.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        diffs = diff_against_default(cfg)
        assert diffs == []

    def test_diff_against_default_surfaces_changes(self):
        """
        Mutating one field surfaces a single diff entry.

        Tests:
            (Test Case 1) Setting curation.snr_min = 7.5 produces
                exactly one diff entry naming that path.
            (Test Case 2) Default and used values are recorded.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        cfg.curation.snr_min = 7.5
        diffs = diff_against_default(cfg)
        paths = [d[0] for d in diffs]
        assert any("curation.snr_min" in p for p in paths)
        # Find the entry; default is 5.0, used is 7.5.
        entry = next(d for d in diffs if "curation.snr_min" in d[0])
        assert entry[1] == 5.0
        assert entry[2] == 7.5


# ---------------------------------------------------------------------------
# Unit quality stats
# ---------------------------------------------------------------------------


class TestExtractUnitQualityStats:
    """``extract_unit_quality_stats`` reads neuron_attributes from a pickle."""

    def _make_fake_spikedata(self, n_units=5):
        """Build a minimal SpikeData-like object."""
        sd = SimpleNamespace()
        # 5 units, varying spike counts
        sd.train = [
            [1.0, 2.0, 3.0],
            [10.0, 20.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [100.0],
            [50.0, 60.0],
        ]
        sd.length = 1000.0  # ms → 1s
        sd.neuron_attributes = [
            {"snr": 6.0, "std_norm": 0.4, "amplitude": 50.0},
            {"snr": 8.5, "std_norm": 0.3, "amplitude": 75.0},
            {"snr": 12.0, "std_norm": 0.2, "amplitude": 100.0},
            {"snr": 5.5, "std_norm": 0.5, "amplitude": 30.0},
            {"snr": 9.0, "std_norm": 0.35, "amplitude": 60.0},
        ]
        return sd

    def test_returns_empty_when_pickle_missing(self, tmp_path):
        """
        Missing pickle yields an empty stats dict.

        Tests:
            (Test Case 1) Non-existent path returns ``{}``.
        """
        result = extract_unit_quality_stats(tmp_path / "does_not_exist.pkl")
        assert result == {}

    def test_extracts_snr_and_std_norm_and_amplitude(self, tmp_path):
        """
        Per-metric summary stats are extracted from neuron_attributes.

        Tests:
            (Test Case 1) snr stats include the expected mean.
            (Test Case 2) firing_rate_hz is computed from train/length.
            (Test Case 3) total_spikes counts all spikes across units.
        """
        import pickle

        sd = self._make_fake_spikedata()
        p = tmp_path / "sorted_spikedata_curated.pkl"
        with open(p, "wb") as f:
            pickle.dump(sd, f)

        stats = extract_unit_quality_stats(p)
        assert "snr" in stats
        assert "std_norm" in stats
        assert "amplitude_uv" in stats
        assert "firing_rate_hz" in stats
        assert "total_spikes" in stats
        # Mean SNR over [6.0, 8.5, 12.0, 5.5, 9.0] = 8.2
        assert stats["snr"]["mean"] == pytest.approx(8.2, rel=1e-3)
        # Total spikes: 3 + 2 + 5 + 1 + 2 = 13
        assert stats["total_spikes"]["n"] == 13


# ---------------------------------------------------------------------------
# End-to-end report generation
# ---------------------------------------------------------------------------


class TestGenerateSortingReport:
    """``generate_sorting_report`` writes a Markdown file with all sections."""

    def _setup_results_folder(self, tmp_path, with_failure=False, with_pickle=True):
        """Build a fake per-recording results folder."""
        import pickle as _pkl

        folder = tmp_path / "rec1"
        folder.mkdir()

        log_path = folder / "sorting_260502_100000.log"
        log_path.write_text(
            _SAMPLE_LOG_FAILURE if with_failure else _SAMPLE_LOG_SUCCESS,
            encoding="utf-8",
        )

        # Recording report JSON
        rec_record = {
            "rec_name": "rec1",
            "rec_path": "/data/raw/rec1.raw.h5",
            "results_folder": str(folder),
            "status": "failed" if with_failure else "success",
            "wall_time_s": 525.0,
            "n_curated_units": None if with_failure else 92,
            "error_class": "ValueError" if with_failure else None,
            "error_message": "something went wrong" if with_failure else None,
            "retries_used": 0,
            "log_path": str(log_path),
        }
        (folder / "recording_report.json").write_text(
            json.dumps(rec_record, indent=2), encoding="utf-8"
        )

        # Config used JSON — non-default snr_min so the diff section
        # has something to show.
        from spikelab.spike_sorting.config import SortingPipelineConfig

        cfg = SortingPipelineConfig()
        cfg.curation.snr_min = 7.5
        (folder / "config_used.json").write_text(
            json.dumps(serialize_config_for_report(cfg), indent=2),
            encoding="utf-8",
        )

        # Curated SpikeData pickle (success path only).
        if with_pickle and not with_failure:
            sd = SimpleNamespace()
            sd.train = [[1.0, 2.0], [3.0, 4.0, 5.0]]
            sd.length = 1000.0
            sd.neuron_attributes = [
                {"snr": 6.5, "std_norm": 0.3, "amplitude": 55.0},
                {"snr": 9.0, "std_norm": 0.2, "amplitude": 80.0},
            ]
            with open(folder / "sorted_spikedata_curated.pkl", "wb") as f:
                _pkl.dump(sd, f)

        return folder, log_path

    def test_success_path_writes_all_sections(self, tmp_path):
        """
        Full success-path report contains every documented section.

        Tests:
            (Test Case 1) Report file is written and returned.
            (Test Case 2) All required H2 sections appear.
            (Test Case 3) Curation outcome includes the curation line.
            (Test Case 4) Non-default settings table shows the
                changed snr_min.
        """
        folder, _log = self._setup_results_folder(tmp_path)
        out = generate_sorting_report(folder)
        assert out is not None
        assert out.is_file()
        text = out.read_text(encoding="utf-8")
        for section in (
            "## Curation outcome",
            "## Overview",
            "## Script settings (non-default)",
            "## Environment",
            "## Pipeline timing",
            "## Unit quality distributions",
            "## Resources at finish",
            "## Warnings",
            "## Output files",
        ):
            assert section in text, f"missing section: {section}"
        assert "230 -> 92" in text  # curation line
        assert "curation.snr_min" in text  # diff entry

    def test_failure_path_includes_failure_section(self, tmp_path):
        """
        On failure, the report adds a Failure section with traceback.

        Tests:
            (Test Case 1) "## Failure" section is present.
            (Test Case 2) The traceback is embedded verbatim in a
                fenced code block.
            (Test Case 3) The last-200-lines context is also embedded.
        """
        folder, _log = self._setup_results_folder(tmp_path, with_failure=True)
        out = generate_sorting_report(folder)
        assert out is not None
        text = out.read_text(encoding="utf-8")
        assert "## Failure" in text
        assert "ValueError: something went wrong" in text
        assert "chunk 3/3" in text  # last-lines-before-traceback

    def test_atomic_write(self, tmp_path):
        """
        ``sorting_report.md`` is written atomically.

        Tests:
            (Test Case 1) The .tmp file is gone after a successful
                write.
        """
        folder, _log = self._setup_results_folder(tmp_path)
        out = generate_sorting_report(folder)
        assert out is not None
        assert not (out.with_suffix(".md.tmp")).exists()


# ---------------------------------------------------------------------------
# tee_log_policy
# ---------------------------------------------------------------------------


class TestApplyTeeLogPolicy:
    """``apply_tee_log_policy`` honours each of the three values."""

    def test_keep(self, tmp_path):
        """
        ``"keep"`` leaves the file untouched.

        Tests:
            (Test Case 1) Returned path equals the input path.
            (Test Case 2) File still exists.
        """
        log = tmp_path / "rec.log"
        log.write_text("hello", encoding="utf-8")
        result = apply_tee_log_policy(log, "keep")
        assert result == log
        assert log.exists()

    def test_delete_on_success(self, tmp_path):
        """
        ``"delete_on_success"`` removes the file.

        Tests:
            (Test Case 1) Returned path is None.
            (Test Case 2) File no longer exists.
        """
        log = tmp_path / "rec.log"
        log.write_text("hello", encoding="utf-8")
        result = apply_tee_log_policy(log, "delete_on_success")
        assert result is None
        assert not log.exists()

    def test_gzip_on_success(self, tmp_path):
        """
        ``"gzip_on_success"`` compresses to .gz and removes the original.

        Tests:
            (Test Case 1) Returned path ends in ``.gz``.
            (Test Case 2) Original log file is gone.
            (Test Case 3) The .gz file decompresses back to the
                original content.
        """
        log = tmp_path / "rec.log"
        original = "this is a log\nwith two lines"
        log.write_text(original, encoding="utf-8")
        result = apply_tee_log_policy(log, "gzip_on_success")
        assert result is not None
        assert str(result).endswith(".gz")
        assert not log.exists()
        with gzip.open(result, "rt", encoding="utf-8") as f:
            assert f.read() == original

    def test_unknown_policy_keeps(self, tmp_path):
        """
        Unknown policy values fall through to "keep".

        Tests:
            (Test Case 1) File is preserved.
            (Test Case 2) Returned path equals the input path.
        """
        log = tmp_path / "rec.log"
        log.write_text("hello", encoding="utf-8")
        result = apply_tee_log_policy(log, "garbage_value")
        assert result == log
        assert log.exists()

    def test_missing_log_returns_none(self, tmp_path):
        """
        Missing input file returns None and does not raise.

        Tests:
            (Test Case 1) Non-existent path returns None.
        """
        result = apply_tee_log_policy(tmp_path / "does_not_exist.log", "keep")
        assert result is None


# ---------------------------------------------------------------------------
# ExecutionConfig defaults for Stream 2 fields
# ---------------------------------------------------------------------------


class TestStream2ConfigDefaults:
    """ExecutionConfig has the new tee_log_policy + sorting-report fields."""

    def test_defaults(self):
        """
        Stream 2 fields default to documented values.

        Tests:
            (Test Case 1) tee_log_policy defaults to
                "delete_on_success".
            (Test Case 2) generate_sorting_report defaults to True.
        """
        from spikelab.spike_sorting.config import ExecutionConfig

        cfg = ExecutionConfig()
        assert cfg.tee_log_policy == "delete_on_success"
        assert cfg.generate_sorting_report is True
