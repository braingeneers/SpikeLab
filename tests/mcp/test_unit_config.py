"""Unit tests for ``spikelab.mcp.config`` — watchdog config + socket path.

Covers the read-side helpers that ``daemon.py`` and ``server.py`` call:

- ``idle_watchdog_seconds(kind)`` — reads ``system_params.json`` overrides,
  falls back to compiled-in defaults per kind. Must reject unknown kinds.
- ``daemon_socket_path()`` — prefers ``$XDG_RUNTIME_DIR``, falls back to
  ``/tmp/spikelab-daemon-<uid>.sock`` when XDG is unset / not a writable dir.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The package import requires OC_SPIKELAB_PROJECT_ROOT to point at a directory
# containing system_config.json — pytest must be invoked with that exported.
from spikelab.mcp import config as cfg_mod
from spikelab.mcp.config import daemon_socket_path, idle_watchdog_seconds


class TestIdleWatchdogDefaults:
    @pytest.fixture(autouse=True)
    def _clean_system_params(self, monkeypatch):
        # Force the defaults path by pointing SYSTEM_PARAMS at an empty dict.
        monkeypatch.setattr(cfg_mod, "SYSTEM_PARAMS", {})

    def test_educator_default(self):
        assert idle_watchdog_seconds("educator") == 1800

    def test_sorter_default(self):
        # Sorter watchdog is intentionally long (2 h) — Kilosort runs are
        # multi-minute and customers may step away mid-sort.
        assert idle_watchdog_seconds("sorter") == 7200

    def test_analyzer_default(self):
        assert idle_watchdog_seconds("analyzer") == 3600

    def test_developer_default(self):
        assert idle_watchdog_seconds("developer") == 3600

    def test_map_updater_default(self):
        assert idle_watchdog_seconds("map_updater") == 600

    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError):
            idle_watchdog_seconds("sorrter_typo")


class TestIdleWatchdogOverride:
    def test_per_kind_override(self, monkeypatch):
        monkeypatch.setattr(
            cfg_mod,
            "SYSTEM_PARAMS",
            {
                "ask_spikelab_sorter_idle_watchdog_seconds": 10800,
                "ask_spikelab_educator_idle_watchdog_seconds": 60,
                # other keys omitted — fall back to defaults
            },
        )
        assert idle_watchdog_seconds("sorter") == 10800
        assert idle_watchdog_seconds("educator") == 60
        assert idle_watchdog_seconds("analyzer") == 3600  # default
        assert idle_watchdog_seconds("developer") == 3600  # default
        assert idle_watchdog_seconds("map_updater") == 600  # default

    def test_override_coerces_str_to_int(self, monkeypatch):
        # system_params.json is JSON so values are usually ints, but
        # the helper should be defensive about string values too.
        monkeypatch.setattr(
            cfg_mod,
            "SYSTEM_PARAMS",
            {"ask_spikelab_educator_idle_watchdog_seconds": "900"},
        )
        assert idle_watchdog_seconds("educator") == 900
        assert isinstance(idle_watchdog_seconds("educator"), int)


class TestDaemonSocketPath:
    def test_xdg_runtime_dir_used_when_writable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        p = daemon_socket_path()
        assert p == tmp_path / "spikelab-daemon.sock"

    def test_falls_back_to_tmp_when_xdg_unset(self, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        p = daemon_socket_path()
        assert p.parent == Path("/tmp")
        assert p.name == f"spikelab-daemon-{os.getuid()}.sock"

    def test_falls_back_to_tmp_when_xdg_not_a_dir(self, monkeypatch, tmp_path):
        # Point XDG_RUNTIME_DIR at a regular file — not a directory.
        bogus = tmp_path / "not_a_dir"
        bogus.write_text("x")
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(bogus))
        p = daemon_socket_path()
        assert p.parent == Path("/tmp")

    def test_falls_back_to_tmp_when_xdg_not_writable(self, monkeypatch, tmp_path):
        # Make the directory read-only so os.access(..., W_OK) fails.
        readonly = tmp_path / "ro"
        readonly.mkdir()
        readonly.chmod(0o500)
        try:
            monkeypatch.setenv("XDG_RUNTIME_DIR", str(readonly))
            p = daemon_socket_path()
            assert p.parent == Path("/tmp")
        finally:
            readonly.chmod(0o700)  # let tmp_path cleanup work
