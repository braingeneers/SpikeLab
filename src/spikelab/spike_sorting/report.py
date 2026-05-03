"""Post-sorting Markdown report generator.

After every sorted recording, ``sort_recording`` calls
:func:`generate_sorting_report` to produce a human-readable
``sorting_report.md`` next to the per-recording results. The report
distills the verbose Tee log + the structured ``recording_report.json``
+ the curated SpikeData pickle into a single Markdown file with:

* **Curation outcome** at the top — raw vs curated unit count, total
  spikes, mean firing rate, mean SNR.
* **Overview** — sorter, status, wall time, log path, retry count.
* **Script settings** — non-default sorter parameters from the
  serialised ``config_used.json``.
* **Environment** — Python / SpikeInterface / SpikeLab versions, host,
  RAM, GPU, heap-cap state — all parsed from the Tee log banner.
* **Pipeline timing** — table of stage banners with ISO timestamps
  parsed from ``[YYYY-MM-DD HH:MM:SS]`` markers in the Tee log.
* **Unit quality distributions** — summary stats (mean / median / std
  / min / max) for SNR, firing rate, ISI%, std_norm, amplitude, drawn
  from the curated SpikeData's ``neuron_attributes``.
* **Resources at finish** — the closing summary banner from the Tee
  log.
* **Output files** — recursive listing of the results folder with
  per-file size in MB.
* **Warnings** — any ``WARN`` / ``Warning`` lines extracted from the
  Tee log (always shown).
* **Failure section** (only on failure) — the full Python traceback
  + last 200 lines of stdout before the error, both verbatim, so the
  Tee log can be safely deleted under ``tee_log_policy``.

The function is deliberately tolerant — missing inputs become
"(unavailable)" sections rather than aborting the report. The
caller (``sort_recording``) treats a successful report write as the
gate for applying ``tee_log_policy`` (delete / gzip the Tee log).
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tee log parsing
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
_BANNER_LINE_RE = re.compile(r"^=+$")
_TRACEBACK_START_RE = re.compile(r"^Traceback \(most recent call last\):")
_TRACEBACK_END_RE = re.compile(r"^[A-Z][\w\.]*(?:Error|Exception|Interrupt)\b.*")
_WARNING_RE = re.compile(r"(?i)warning|warn")
_BANNER_TEXT_RE = re.compile(
    r"^\s+([A-Z][A-Z0-9 \-_/().,]{2,})\s*$"  # centered uppercase banner text
)


def parse_sorting_log(log_text: str) -> Dict[str, Any]:
    """Extract structured fields from a Tee-mirrored sorting log.

    The sort_recording pipeline writes per-recording stdout to a
    ``sorting_<timestamp>.log`` via ``Tee``. That log includes a
    structured banner block, ISO-stamped stage banners, the
    "Curation: N -> M units" line, a closing summary, and any
    Python traceback on failure. This function pulls those pieces
    out into a dict suitable for templating into Markdown.

    Parameters:
        log_text (str): Full text of the Tee log file.

    Returns:
        info (dict): Keys include ``environment`` (dict),
            ``run`` (dict), ``stage_timings`` (list of
            ``{name, timestamp}`` dicts), ``curation_line`` (str or
            None), ``closing_summary`` (dict), ``warnings``
            (list[str]), ``traceback`` (str or None),
            ``last_lines_before_traceback`` (list[str]).
    """
    lines = log_text.splitlines()

    environment: Dict[str, str] = {}
    run: Dict[str, str] = {}
    closing_summary: Dict[str, str] = {}
    stage_timings: List[Dict[str, str]] = []
    warnings: List[str] = []
    curation_line: Optional[str] = None
    traceback_block: Optional[str] = None
    last_lines: List[str] = []

    # The banner block has the structure
    #   -- Environment --
    #   key:           value
    #   ...
    #   -- System Resources --
    #   ...
    #   -- Run --
    #   ...
    # We walk the file once and collect into the right bucket.
    current_section: Optional[str] = None
    summary_started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-- Environment --"):
            current_section = "environment"
            continue
        if stripped.startswith("-- System Resources --"):
            current_section = "system_resources"
            continue
        if stripped.startswith("-- Run --"):
            current_section = "run"
            continue
        if "SUMMARY" in stripped and _BANNER_TEXT_RE.match(line):
            current_section = "summary"
            summary_started = True
            continue

        # Within an active section, consume key:value lines into the
        # right bucket. Two reset rules apply:
        #   * environment / system_resources / run end at an empty
        #     line or a `===` boundary — they're tightly packed
        #     dashed-marker blocks.
        #   * summary's key:value pairs come AFTER a `===` /
        #     ``[timestamp]`` banner pair, so we skip those silently
        #     and only reset when the next *real* section marker
        #     appears (handled at the top of the loop).
        if current_section in ("environment", "system_resources", "run", "summary"):
            m = re.match(r"^([A-Za-z][\w \-/]*?):\s*(.+)$", stripped)
            if m is not None:
                key = m.group(1).strip()
                value = m.group(2).strip()
                if current_section == "summary":
                    closing_summary[key] = value
                elif current_section in ("environment", "system_resources"):
                    environment[key] = value
                elif current_section == "run":
                    run[key] = value
                continue
            if current_section in ("environment", "system_resources", "run"):
                if stripped == "" or stripped.startswith("="):
                    current_section = None

        # Stage banners — print_stage produces a centered uppercase
        # message followed (after a couple lines) by an
        # ``[YYYY-MM-DD HH:MM:SS]`` line. We pair them.
        m_ts = _TIMESTAMP_RE.search(line)
        if m_ts is not None:
            # Look back up to 3 lines for the centered banner text.
            ts = m_ts.group(1)
            lookback_idx = lines.index(line) if line in lines else -1
            if lookback_idx > 0:
                for j in range(max(0, lookback_idx - 3), lookback_idx):
                    cand = _BANNER_TEXT_RE.match(lines[j])
                    if cand:
                        name = cand.group(1).strip()
                        if name and name not in ("ENVIRONMENT", "SYSTEM RESOURCES"):
                            stage_timings.append({"name": name, "timestamp": ts})
                            break

        # Curation line — emitted by process_recording.
        if curation_line is None and stripped.startswith("Curation: "):
            curation_line = stripped

        # Warnings.
        if _WARNING_RE.search(stripped) and stripped:
            # Filter out the matplotlib UserWarning's "context" lines
            # that lack a recognisable warning identifier.
            if any(
                tag in stripped
                for tag in (
                    "Warning",
                    "WARN",
                    "warning",
                )
            ):
                warnings.append(stripped)

    # Traceback extraction. Find the start, capture until end.
    tb_start = None
    for i, line in enumerate(lines):
        if _TRACEBACK_START_RE.match(line):
            tb_start = i
            break
    if tb_start is not None:
        tb_end = tb_start
        for i in range(tb_start + 1, len(lines)):
            if _TRACEBACK_END_RE.match(lines[i].strip()):
                tb_end = i
                break
        traceback_block = "\n".join(lines[tb_start : tb_end + 1])
        # Last 200 stdout lines before the traceback for context.
        ctx_start = max(0, tb_start - 200)
        last_lines = lines[ctx_start:tb_start]

    return {
        "environment": environment,
        "run": run,
        "stage_timings": stage_timings,
        "curation_line": curation_line,
        "closing_summary": closing_summary,
        "warnings": warnings,
        "traceback": traceback_block,
        "last_lines_before_traceback": last_lines,
    }


# ---------------------------------------------------------------------------
# Config diff
# ---------------------------------------------------------------------------


def serialize_config_for_report(config: Any) -> Dict[str, Any]:
    """Convert a SortingPipelineConfig into a JSON-safe dict.

    Used to write ``config_used.json`` per recording so the Markdown
    report can list non-default settings. Path / Tuple values are
    coerced to plain strings so the result is fully JSON-friendly.

    Parameters:
        config: ``SortingPipelineConfig`` instance.

    Returns:
        snapshot (dict): Nested dict mirroring the dataclass
            structure.
    """
    raw = asdict(config)
    return _jsonify(raw)


def _jsonify(obj: Any) -> Any:
    """Coerce Path / Tuple / non-JSON-friendly values to JSON-safe forms."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def diff_against_default(
    config: Any, default_config: Optional[Any] = None
) -> List[Tuple[str, Any, Any]]:
    """Return ``(field_path, default, actual)`` tuples for non-default values.

    Used by the report to show only the parameters the user actually
    changed, rather than dumping the full default-laden config.

    Parameters:
        config: The config used during the sort.
        default_config: The reference default. ``None`` constructs
            a fresh ``SortingPipelineConfig()``.

    Returns:
        diffs (list): List of ``(dotted_path, default_value, actual_value)``
            triples for fields where the value diverged.
    """
    if default_config is None:
        from .config import SortingPipelineConfig

        default_config = SortingPipelineConfig()
    actual = _jsonify(asdict(config))
    default = _jsonify(asdict(default_config))
    diffs: List[Tuple[str, Any, Any]] = []
    _walk_diff("", default, actual, diffs)
    return diffs


def _walk_diff(prefix: str, default: Any, actual: Any, out: List) -> None:
    """Recurse two parallel dicts; record diverging leaf values."""
    if isinstance(default, dict) and isinstance(actual, dict):
        for key in actual.keys() | default.keys():
            sub_prefix = f"{prefix}.{key}" if prefix else key
            _walk_diff(sub_prefix, default.get(key), actual.get(key), out)
        return
    if default != actual:
        out.append((prefix, default, actual))


# ---------------------------------------------------------------------------
# Unit quality stats
# ---------------------------------------------------------------------------


def _summary_stats(values: List[float]) -> Dict[str, float]:
    """Return mean / median / std / min / max of a numeric list.

    Skips non-finite values. Returns ``{}`` when the input is empty
    or all values are non-finite.
    """
    finite = [
        v
        for v in values
        if isinstance(v, (int, float))
        and v == v
        and v not in (float("inf"), float("-inf"))
    ]
    if not finite:
        return {}
    n = len(finite)
    mean = sum(finite) / n
    sorted_vals = sorted(finite)
    median = (
        sorted_vals[n // 2]
        if n % 2 == 1
        else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    )
    var = sum((v - mean) ** 2 for v in finite) / n
    std = var**0.5
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "n": n,
    }


def extract_unit_quality_stats(curated_pkl_path: Path) -> Dict[str, Dict[str, float]]:
    """Read the curated SpikeData pickle and return per-metric summary stats.

    Reads attributes from ``sd.neuron_attributes`` for SNR, std_norm,
    amplitude. Computes firing rate from ``sd.train`` lengths and
    ``sd.length``. Returns ``{}`` when the pickle cannot be loaded
    or is empty.

    Parameters:
        curated_pkl_path (Path): Path to ``sorted_spikedata_curated.pkl``.

    Returns:
        stats (dict): Dict of metric name → summary stats dict.
    """
    p = Path(curated_pkl_path)
    if not p.is_file():
        return {}
    try:
        with open(p, "rb") as f:
            sd = pickle.load(f)
    except Exception:
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    attrs = getattr(sd, "neuron_attributes", None) or []

    snr_vals: List[float] = []
    std_norm_vals: List[float] = []
    amp_vals: List[float] = []
    for a in attrs:
        try:
            snr_vals.append(float(a.get("snr")))
        except (TypeError, ValueError):
            pass
        try:
            std_norm_vals.append(float(a.get("std_norm")))
        except (TypeError, ValueError):
            pass
        try:
            amp_vals.append(float(a.get("amplitude")))
        except (TypeError, ValueError):
            pass

    if snr_vals:
        stats["snr"] = _summary_stats(snr_vals)
    if std_norm_vals:
        stats["std_norm"] = _summary_stats(std_norm_vals)
    if amp_vals:
        stats["amplitude_uv"] = _summary_stats(amp_vals)

    # Firing rates: spikes per second for each unit.
    train = getattr(sd, "train", None)
    length = getattr(sd, "length", None)
    if train is not None and length and length > 0:
        # length is in ms; convert to seconds.
        length_s = float(length) / 1000.0
        fr_vals = [float(len(t)) / length_s for t in train]
        stats["firing_rate_hz"] = _summary_stats(fr_vals)
        n_total_spikes = sum(len(t) for t in train)
        stats["total_spikes"] = {"n": n_total_spikes}
    return stats


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _md_kv_table(rows: Dict[str, Any]) -> str:
    """Render a 2-column key/value Markdown table."""
    if not rows:
        return "_(unavailable)_\n"
    lines = ["| Field | Value |", "|---|---|"]
    for k, v in rows.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def _md_stage_table(stages: List[Dict[str, str]]) -> str:
    """Render the pipeline-timing table with deltas between stages."""
    if not stages:
        return "_(no stage banners parsed)_\n"
    lines = ["| Stage | Timestamp | Δ from previous |", "|---|---|---|"]
    prev_dt: Optional[datetime] = None
    for entry in stages:
        ts = entry["timestamp"]
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            dt = None
        delta = ""
        if dt is not None and prev_dt is not None:
            secs = (dt - prev_dt).total_seconds()
            delta = f"{secs:.0f}s" if secs < 60 else f"{secs/60:.1f}m"
        lines.append(f"| {entry['name']} | {ts} | {delta} |")
        if dt is not None:
            prev_dt = dt
    return "\n".join(lines) + "\n"


def _md_quality_section(stats: Dict[str, Dict[str, float]]) -> str:
    """Render the unit quality summary table."""
    if not stats:
        return "_(no curated SpikeData loaded)_\n"
    lines = [
        "| Metric | n | mean | median | std | min | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for metric, s in stats.items():
        if metric == "total_spikes":
            continue
        if not s:
            continue
        lines.append(
            f"| {metric} | {s.get('n', '')} | "
            f"{s.get('mean', float('nan')):.3g} | "
            f"{s.get('median', float('nan')):.3g} | "
            f"{s.get('std', float('nan')):.3g} | "
            f"{s.get('min', float('nan')):.3g} | "
            f"{s.get('max', float('nan')):.3g} |"
        )
    if "total_spikes" in stats:
        lines.append("")
        lines.append(f"**Total spikes (curated):** {stats['total_spikes']['n']:,}")
    return "\n".join(lines) + "\n"


def _md_settings_section(diffs: List[Tuple[str, Any, Any]]) -> str:
    """Render the non-default-settings table."""
    if not diffs:
        return "_(all defaults)_\n"
    lines = ["| Setting | Default | Used |", "|---|---|---|"]
    for path, default, used in diffs:
        lines.append(f"| `{path}` | `{default}` | `{used}` |")
    return "\n".join(lines) + "\n"


def _md_files_section(folder: Path) -> str:
    """Render the output-files listing with per-file MB sizes."""
    if not folder.exists():
        return "_(results folder missing)_\n"
    lines = ["| File | Size (MB) |", "|---|---|"]
    rows: List[Tuple[str, float]] = []
    base = folder
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            p = Path(dirpath) / name
            try:
                size = p.stat().st_size / (1024 * 1024)
            except OSError:
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            rows.append((rel, size))
    rows.sort(key=lambda x: -x[1])
    for rel, size in rows[:50]:
        lines.append(f"| `{rel}` | {size:.2f} |")
    if len(rows) > 50:
        lines.append("")
        lines.append(
            f"_({len(rows) - 50} additional files omitted; sorted by size descending)_"
        )
    return "\n".join(lines) + "\n"


def _md_curation_outcome(
    parsed: Dict[str, Any], stats: Dict[str, Dict[str, float]]
) -> str:
    """Render the headline curation-outcome block."""
    line = parsed.get("curation_line") or "_(curation line not found in log)_"
    parts = [line, ""]
    snr = stats.get("snr") or {}
    fr = stats.get("firing_rate_hz") or {}
    total = stats.get("total_spikes") or {}
    if snr or fr or total:
        parts.append("**Quick stats:**")
        if total:
            parts.append(f"- Total curated spikes: {total['n']:,}")
        if fr:
            parts.append(
                f"- Mean firing rate: {fr.get('mean', float('nan')):.2f} Hz "
                f"(median {fr.get('median', float('nan')):.2f} Hz)"
            )
        if snr:
            parts.append(
                f"- Mean SNR: {snr.get('mean', float('nan')):.2f} "
                f"(median {snr.get('median', float('nan')):.2f})"
            )
    return "\n".join(parts) + "\n"


def _md_warnings_section(warnings: List[str]) -> str:
    """Render the warnings extracted from the Tee log."""
    if not warnings:
        return "_(none)_\n"
    return "\n".join(f"- `{w}`" for w in warnings) + "\n"


def _md_failure_section(parsed: Dict[str, Any]) -> str:
    """Render the failure section with traceback + tail context."""
    tb = parsed.get("traceback")
    last_lines = parsed.get("last_lines_before_traceback") or []
    if not tb:
        return ""
    parts = ["", "## Failure", ""]
    if last_lines:
        parts.append("### Last 200 stdout lines before failure")
        parts.append("")
        parts.append("```")
        parts.extend(last_lines)
        parts.append("```")
        parts.append("")
    parts.append("### Traceback")
    parts.append("")
    parts.append("```")
    parts.append(tb)
    parts.append("```")
    return "\n".join(parts) + "\n"


def generate_sorting_report(
    results_folder: Any,
    *,
    log_path: Any = None,
    recording_report_path: Any = None,
    curated_pkl_path: Any = None,
    config_used_path: Any = None,
    output_path: Any = None,
) -> Optional[Path]:
    """Generate a Markdown sorting report for a single recording.

    Reads the per-recording Tee log, ``recording_report.json``,
    ``config_used.json``, and the curated SpikeData pickle (each
    auto-detected from *results_folder* when its argument is
    ``None``), then writes a structured Markdown report describing
    the run.

    The report is the input the ``spikelab-spikesorter`` agent skill
    consumes — it replaces the manual report-writing instructions
    with a deterministic, testable artefact.

    Parameters:
        results_folder (path-like): The per-recording results
            directory. All other paths default to standard names
            inside this folder when their argument is ``None``.
        log_path (path-like or None): Path to the Tee log file
            (``sorting_<timestamp>.log``). ``None`` auto-picks the
            most recent matching file in *results_folder*.
        recording_report_path (path-like or None): Path to
            ``recording_report.json``. Default:
            ``<results_folder>/recording_report.json``.
        curated_pkl_path (path-like or None): Path to the curated
            SpikeData pickle. Default:
            ``<results_folder>/sorted_spikedata_curated.pkl``.
        config_used_path (path-like or None): Path to
            ``config_used.json``. Default:
            ``<results_folder>/config_used.json``.
        output_path (path-like or None): Where to write the report.
            Default: ``<results_folder>/sorting_report.md``.

    Returns:
        path (Path or None): The written file's path, or ``None`` on
            best-effort failure (the surrounding pipeline never lets
            a report failure abort the batch).
    """
    folder = Path(results_folder)
    if output_path is None:
        output_path = folder / "sorting_report.md"
    output_path = Path(output_path)

    # Auto-detect paths.
    if log_path is None:
        candidates = sorted(folder.glob("sorting_*.log"))
        log_path = candidates[-1] if candidates else None
    if recording_report_path is None:
        rec_report = folder / "recording_report.json"
        recording_report_path = rec_report if rec_report.is_file() else None
    if curated_pkl_path is None:
        cur = folder / "sorted_spikedata_curated.pkl"
        curated_pkl_path = cur if cur.is_file() else None
    if config_used_path is None:
        cfg = folder / "config_used.json"
        config_used_path = cfg if cfg.is_file() else None

    log_text = ""
    if log_path is not None and Path(log_path).is_file():
        try:
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            log_text = ""
    parsed = parse_sorting_log(log_text)

    rec_record: Dict[str, Any] = {}
    if recording_report_path is not None and Path(recording_report_path).is_file():
        try:
            rec_record = json.loads(
                Path(recording_report_path).read_text(encoding="utf-8")
            )
        except Exception:
            rec_record = {}

    stats: Dict[str, Dict[str, float]] = {}
    if curated_pkl_path is not None:
        stats = extract_unit_quality_stats(Path(curated_pkl_path))

    config_diffs: List[Tuple[str, Any, Any]] = []
    if config_used_path is not None and Path(config_used_path).is_file():
        try:
            used = json.loads(Path(config_used_path).read_text(encoding="utf-8"))
            from .config import SortingPipelineConfig

            default_dict = _jsonify(asdict(SortingPipelineConfig()))
            _walk_diff("", default_dict, used, config_diffs)
        except Exception:
            config_diffs = []

    rec_name = rec_record.get("rec_name") or folder.name
    md = _render_report_markdown(
        rec_name=rec_name,
        rec_record=rec_record,
        parsed=parsed,
        stats=stats,
        config_diffs=config_diffs,
        folder=folder,
        log_path=log_path,
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so an os._exit fired by the inactivity
        # watchdog mid-write cannot leave a corrupt report behind.
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(md)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        os.replace(tmp, output_path)
        return output_path
    except Exception as exc:
        print(f"[sorting report] failed to write {output_path}: {exc!r}")
        return None


def _render_report_markdown(
    *,
    rec_name: str,
    rec_record: Dict[str, Any],
    parsed: Dict[str, Any],
    stats: Dict[str, Dict[str, float]],
    config_diffs: List[Tuple[str, Any, Any]],
    folder: Path,
    log_path: Optional[Any],
) -> str:
    """Assemble the full Markdown body."""
    parts = [f"# Sorting report: {rec_name}", ""]

    parts.append("## Curation outcome")
    parts.append("")
    parts.append(_md_curation_outcome(parsed, stats))

    overview: Dict[str, Any] = {}
    if rec_record:
        overview["Status"] = rec_record.get("status", "unknown")
        if rec_record.get("error_class"):
            overview["Error class"] = rec_record["error_class"]
        if rec_record.get("error_message"):
            overview["Error message"] = rec_record["error_message"]
        overview["Wall time (s)"] = rec_record.get("wall_time_s", "")
        overview["Retries used"] = rec_record.get("retries_used", "")
        overview["Curated units"] = rec_record.get("n_curated_units", "")
        if rec_record.get("rec_path"):
            overview["Recording"] = rec_record["rec_path"]
        if rec_record.get("results_folder"):
            overview["Results folder"] = rec_record["results_folder"]
    if log_path is not None:
        overview["Log file"] = str(log_path)
    parts.append("## Overview")
    parts.append("")
    parts.append(_md_kv_table(overview))

    parts.append("## Script settings (non-default)")
    parts.append("")
    parts.append(_md_settings_section(config_diffs))

    parts.append("## Environment")
    parts.append("")
    parts.append(_md_kv_table(parsed.get("environment", {})))

    parts.append("## Run banner")
    parts.append("")
    parts.append(_md_kv_table(parsed.get("run", {})))

    parts.append("## Pipeline timing")
    parts.append("")
    parts.append(_md_stage_table(parsed.get("stage_timings", [])))

    parts.append("## Unit quality distributions")
    parts.append("")
    parts.append(_md_quality_section(stats))

    parts.append("## Resources at finish")
    parts.append("")
    parts.append(_md_kv_table(parsed.get("closing_summary", {})))

    parts.append("## Warnings")
    parts.append("")
    parts.append(_md_warnings_section(parsed.get("warnings", [])))

    parts.append("## Output files")
    parts.append("")
    parts.append(_md_files_section(folder))

    parts.append(_md_failure_section(parsed))

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Tee log lifecycle (delete / gzip on success)
# ---------------------------------------------------------------------------


def apply_tee_log_policy(log_path: Any, policy: str) -> Optional[Path]:
    """Delete or gzip the Tee log per *policy*; return the resulting path.

    Called by ``sort_recording`` AFTER ``generate_sorting_report``
    returns successfully — failures preserve the log automatically
    because ``generate_sorting_report`` returns ``None`` on report
    failure and the caller only invokes this function on a non-None
    return.

    Parameters:
        log_path (path-like): Path to the Tee log file.
        policy (str): One of ``"keep"``, ``"gzip_on_success"``,
            ``"delete_on_success"``. Anything else is treated as
            ``"keep"`` for safety.

    Returns:
        result (Path or None): Final path of the Tee log
            (``<log>.gz`` for gzip; ``None`` for delete; the
            original path for keep). ``None`` on any error.
    """
    p = Path(log_path)
    if not p.is_file():
        return None
    if policy == "keep":
        return p
    if policy == "gzip_on_success":
        try:
            import gzip

            target = p.with_suffix(p.suffix + ".gz")
            with open(p, "rb") as src, gzip.open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            p.unlink()
            return target
        except Exception as exc:
            print(f"[tee log policy] gzip failed for {p}: {exc!r}")
            return p
    if policy == "delete_on_success":
        try:
            p.unlink()
            return None
        except Exception as exc:
            print(f"[tee log policy] delete failed for {p}: {exc!r}")
            return p
    # Unknown policy → keep, with a warning.
    print(f"[tee log policy] unknown policy {policy!r}; keeping log untouched.")
    return p
