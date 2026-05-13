"""Unit tests for ``spikelab.mcp.progress`` — log-file discovery + tail helper.

The progress tool's heart is two pure pieces of logic:

- ``_LOG_FILENAME_RE`` regex — extracts sorter log paths from transcript text.
  Must match the four recognised filenames (``kilosort{2,4}.log``,
  ``rt_sort.log``, ``sorting_*.log``) and ignore anything else.
- ``_tail_lines(path, n)`` — reads the last ``n`` lines of a file via reverse
  chunked reads. Must handle small files, large files, partial last lines,
  files without trailing newlines, and missing files.

The transcript walker (``_scan_transcript_for_log_paths``) and the SDK-backed
``get_task_progress`` entrypoint are exercised by the integration tests; this
file stays pure-Python (no Claude credentials, no daemon).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spikelab.mcp.progress import _LOG_FILENAME_RE, _tail_lines


class TestLogFilenameRegex:
    @pytest.mark.parametrize(
        "filename",
        [
            "kilosort2.log",
            "kilosort4.log",
            "rt_sort.log",
            "sorting_0.log",
            "sorting_15.log",
            "sorting_run_42.log",
        ],
    )
    def test_matches_known_sorter_logs(self, filename):
        haystack = f"output written to /data/sorted/recA/{filename}"
        m = _LOG_FILENAME_RE.search(haystack)
        assert m is not None, f"failed to match {filename}"
        assert m.group("path").endswith(filename)

    def test_picks_full_path_not_just_filename(self):
        haystack = "Wrote /home/user/sorted/rec_b/kilosort4.log (success)"
        m = _LOG_FILENAME_RE.search(haystack)
        assert m is not None
        assert m.group("path") == "/home/user/sorted/rec_b/kilosort4.log"

    def test_matches_relative_path(self):
        haystack = "tail -f ./output/kilosort2.log"
        m = _LOG_FILENAME_RE.search(haystack)
        assert m is not None
        assert m.group("path") == "./output/kilosort2.log"

    def test_matches_tilde_path(self):
        haystack = "see ~/sorted/recA/rt_sort.log for details"
        m = _LOG_FILENAME_RE.search(haystack)
        assert m is not None
        assert m.group("path").endswith("/sorted/recA/rt_sort.log")

    def test_ignores_unrelated_log_filename(self):
        # Files like "mxwserver.log" or "agent.log" should NOT match —
        # we only recognise the four sorter-specific filenames.
        for haystack in [
            "see /var/log/syslog.log",
            "wrote /tmp/agent.log",
            "kilosort3.log doesn't exist",  # not a real sorter
            "sorting.log",  # generic — needs the trailing _<something>
        ]:
            assert (
                _LOG_FILENAME_RE.search(haystack) is None
            ), f"unexpectedly matched: {haystack!r}"

    def test_finds_multiple_in_one_string(self):
        haystack = (
            "First wrote /a/b/kilosort4.log, then later /c/d/sorting_3.log "
            "and another at /e/f/rt_sort.log."
        )
        paths = [m.group("path") for m in _LOG_FILENAME_RE.finditer(haystack)]
        assert paths == [
            "/a/b/kilosort4.log",
            "/c/d/sorting_3.log",
            "/e/f/rt_sort.log",
        ]

    def test_does_not_grab_trailing_quote(self):
        # Bash tool args often appear quoted in the transcript; the regex
        # must stop at the closing quote so the path is clean.
        haystack = "subprocess: 'python sort.py --log /data/kilosort4.log'"
        m = _LOG_FILENAME_RE.search(haystack)
        assert m is not None
        assert m.group("path") == "/data/kilosort4.log"


class TestTailLines:
    def test_returns_all_when_fewer_than_n(self, tmp_path):
        p = tmp_path / "small.log"
        p.write_text("line1\nline2\nline3\n")
        assert _tail_lines(p, 10) == "line1\nline2\nline3"

    def test_returns_last_n_when_more(self, tmp_path):
        p = tmp_path / "big.log"
        p.write_text("\n".join(f"line{i}" for i in range(1, 101)) + "\n")
        out = _tail_lines(p, 5)
        assert out == "line96\nline97\nline98\nline99\nline100"

    def test_handles_file_without_trailing_newline(self, tmp_path):
        p = tmp_path / "noeol.log"
        p.write_text("a\nb\nc")  # no trailing \n
        out = _tail_lines(p, 2)
        assert out == "b\nc"

    def test_handles_empty_file(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("")
        assert _tail_lines(p, 5) == ""

    def test_handles_single_line(self, tmp_path):
        p = tmp_path / "one.log"
        p.write_text("only line\n")
        assert _tail_lines(p, 10) == "only line"

    def test_handles_multi_chunk_read(self, tmp_path):
        # Force the reverse-chunked read path: write enough data that the
        # 64 KB chunk size matters. Each line is ~50 bytes, so 5000 lines
        # ≈ 250 KB → 4 chunks.
        p = tmp_path / "long.log"
        p.write_text(
            "\n".join(f"row {i:05d} payload " + "x" * 30 for i in range(5000)) + "\n"
        )
        out = _tail_lines(p, 3)
        lines = out.splitlines()
        assert len(lines) == 3
        assert lines[-1].startswith("row 04999")
        assert lines[0].startswith("row 04997")

    def test_returns_error_marker_for_missing_file(self, tmp_path):
        missing = tmp_path / "does_not_exist.log"
        out = _tail_lines(missing, 5)
        assert out.startswith("<could not read")
        assert "does_not_exist.log" in out

    def test_handles_binary_bytes_gracefully(self, tmp_path):
        # Kilosort logs are normally UTF-8, but a stray non-UTF-8 byte
        # shouldn't crash the tail. The decoder runs with errors='replace'.
        p = tmp_path / "binary.log"
        p.write_bytes(b"line a\nline b\n\xff\xfeweird bytes\nlast\n")
        out = _tail_lines(p, 2)
        # Last two lines should come through; binary bytes are replaced.
        lines = out.splitlines()
        assert len(lines) == 2
        assert lines[-1] == "last"
