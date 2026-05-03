"""Best-effort sweep of leaked temp files at sort end.

When a sort crashes mid-flight (os._exit fired by an in-process
inactivity watchdog, MATLAB child killed, RT-Sort numba kernel
deadlocked) Python's normal ``tempfile`` cleanup hooks may not run
and intermediate files in ``$TMPDIR`` / ``%TEMP%`` are left behind.
Across many crashed sorts this can fill the temp volume.

This module records the temp-folder state at sort start and, on
exit, sweeps any new files matching well-known sorter prefixes.
Best-effort — failures (permission errors, files locked by other
processes) are swallowed.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Iterator, List, Set

# Filename prefixes / substrings that identify temp artefacts as
# belonging to this sort or its sorter children. Conservative list
# to avoid removing unrelated files.
_SORTER_TEMP_MARKERS = (
    "spikelab",
    "kilosort",
    "rt_sort",
    "rtsort",
    "spikeinterface",
    "matlab_temp",
    "mxdumper",
    "tmp_recording_",
)


def _list_marker_files(temp_dir: Path) -> Set[Path]:
    """Return the set of files in *temp_dir* whose names match a marker.

    Non-recursive scan — only top-level files. Sorter temp artefacts
    are typically created at the top of ``$TMPDIR``. Subdirectories
    are inspected only one level deep for ``spikelab*`` /
    ``kilosort*`` / ``rt_sort*`` parents (where mxdumper-style trees
    can land).
    """
    found: Set[Path] = set()
    if not temp_dir.exists():
        return found
    try:
        for entry in temp_dir.iterdir():
            name = entry.name.lower()
            if any(m in name for m in _SORTER_TEMP_MARKERS):
                if entry.is_file():
                    found.add(entry)
                elif entry.is_dir():
                    try:
                        for sub in entry.rglob("*"):
                            if sub.is_file():
                                found.add(sub)
                    except OSError:
                        continue
    except OSError:
        return found
    return found


@contextlib.contextmanager
def cleanup_temp_files(enabled: bool = True) -> Iterator[None]:
    """Sweep new sorter-marker temp files created during the context.

    Records the set of marker-matched temp files at entry; on exit,
    deletes any that appeared during the context. Sweeping happens
    only on **clean** ``__exit__`` (no exception propagating out)
    to avoid removing artefacts that the user may want to inspect
    after a failure.

    Best-effort: per-file errors are logged but do not propagate.

    Parameters:
        enabled (bool): When False, the context manager is a no-op.

    Notes:
        - The sweep is non-aggressive: it removes only files
          matching the well-known prefix list, to avoid clobbering
          unrelated workspace temp files.
        - A failed sort intentionally leaves its temp files behind
          so an operator can diagnose. The next successful sort
          will sweep them up.
    """
    if not enabled:
        yield
        return

    try:
        temp_dir = Path(tempfile.gettempdir())
    except Exception:
        yield
        return

    pre_existing = _list_marker_files(temp_dir)

    try:
        yield
    except Exception:
        # On failure, leave temp files alone so the operator can
        # inspect them.
        raise

    # Clean exit — sweep any new marker files.
    try:
        post = _list_marker_files(temp_dir)
        new_files = post - pre_existing
        if not new_files:
            return
        removed = 0
        failed = 0
        for f in new_files:
            try:
                f.unlink()
                removed += 1
            except Exception:
                failed += 1
        if removed > 0 or failed > 0:
            print(
                f"[temp cleanup] swept {removed} stale temp file(s) from "
                f"{temp_dir} ({failed} failed)."
            )
    except Exception as exc:
        print(f"[temp cleanup] sweep failed: {exc!r}")
