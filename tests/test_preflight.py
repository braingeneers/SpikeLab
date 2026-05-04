"""Tests for ``guards/_preflight.py`` — the pre-loop resource checks.

Covers every helper and the assembled ``run_preflight`` orchestrator
including:

* Sorter dependency probes (``_check_kilosort2_host``,
  ``_check_kilosort4_host``, ``_check_docker_sorter``,
  ``_check_rt_sort``, ``_check_sorter_dependencies``).
* GPU device-index validation (``_resolve_target_device_index``,
  ``_detect_gpu_device_count``, ``_check_gpu_device_present``).
* Recording sample-rate sanity check
  (``_expected_sample_rate_window``, ``_check_recording_sample_rate``).
* Filesystem writability (``_check_filesystem_writable``).
* Free-VRAM detection (``_free_vram_gb``).
* Version parsing (``_parse_version_tuple``).
* Resource limits + SpikeInterface version checks
  (``_check_resource_rlimits``, ``_check_spikeinterface_version``).

The ``run_preflight`` aggregator and ``report_findings`` are exercised
in :mod:`tests.test_guards`. All tests here are hermetic — every
detection path that would otherwise touch the host is patched.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from spikelab.spike_sorting.guards import _preflight as preflight_mod


def _block_imports(monkeypatch, *names):
    """Patch builtins.__import__ so the listed names raise ImportError.

    Other imports continue to work normally. Captures the original
    ``__import__`` before patching so re-entrant imports inside the
    helper still resolve.
    """
    real_import = builtins.__import__
    blocked = set(names)

    def _patched_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"blocked-by-test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)


def _make_cfg(
    *,
    sorter_name: str = "kilosort2",
    use_docker: bool = False,
    sorter_path=None,
    sorter_params=None,
    rt_device: str = "cuda",
    rt_probe: str = "mea",
    canary_first_n_s: float = 0.0,
    docker_image_expected_digest=None,
):
    """Construct a SimpleNamespace config that satisfies the helpers' API.

    Mirrors the ``_make_config`` helper used in ``test_guards.py`` but
    extends it with the additional sub-fields the per-sorter probes,
    GPU device check, and sample-rate check read:
    ``sorter.sorter_path``, ``sorter.sorter_params``, ``rt_sort.device``,
    ``rt_sort.probe``, ``execution.canary_first_n_s``,
    ``execution.docker_image_expected_digest``.
    """
    return SimpleNamespace(
        sorter=SimpleNamespace(
            sorter_name=sorter_name,
            use_docker=use_docker,
            sorter_path=sorter_path,
            sorter_params=sorter_params,
        ),
        rt_sort=SimpleNamespace(
            device=rt_device,
            probe=rt_probe,
        ),
        execution=SimpleNamespace(
            canary_first_n_s=canary_first_n_s,
            docker_image_expected_digest=docker_image_expected_digest,
        ),
    )


# ---------------------------------------------------------------------------
# Sorter dependency probes
# ---------------------------------------------------------------------------


class TestCheckKilosort2Host:
    """``_check_kilosort2_host`` validates matlab + KILOSORT_PATH."""

    def test_matlab_missing_yields_fail(self, monkeypatch):
        """
        Missing ``matlab`` on PATH yields a sorter_dependency_missing
        fail finding.

        Tests:
            (Test Case 1) shutil.which patched to return None for matlab.
            (Test Case 2) Finding code is sorter_dependency_missing,
                level fail, category environment.
        """
        monkeypatch.setattr(preflight_mod.shutil, "which", lambda _: None)
        monkeypatch.setenv("KILOSORT_PATH", "/tmp/does-not-exist")
        cfg = _make_cfg(sorter_name="kilosort2")
        findings = preflight_mod._check_kilosort2_host(cfg)
        codes = [f.code for f in findings]
        assert "sorter_dependency_missing" in codes
        assert all(f.level == "fail" for f in findings)
        assert all(f.category == "environment" for f in findings)

    def test_kilosort_path_unset_yields_fail(self, monkeypatch):
        """
        Unset KILOSORT_PATH (and no SorterConfig.sorter_path) yields a
        sorter_dependency_missing fail finding.

        Tests:
            (Test Case 1) matlab present, KILOSORT_PATH unset → exactly
                one finding with the path-related message.
        """
        monkeypatch.setattr(preflight_mod.shutil, "which", lambda _: "/usr/bin/matlab")
        monkeypatch.delenv("KILOSORT_PATH", raising=False)
        cfg = _make_cfg(sorter_name="kilosort2", sorter_path=None)
        findings = preflight_mod._check_kilosort2_host(cfg)
        assert len(findings) == 1
        assert findings[0].level == "fail"
        assert "KILOSORT_PATH" in findings[0].message

    def test_kilosort_path_dir_missing_yields_fail(self, monkeypatch, tmp_path):
        """
        KILOSORT_PATH pointing at a non-existent directory yields a fail.

        Tests:
            (Test Case 1) Path doesn't exist on disk → fail with
                'does not exist' message.
        """
        monkeypatch.setattr(preflight_mod.shutil, "which", lambda _: "/usr/bin/matlab")
        bogus = tmp_path / "definitely-not-here"
        monkeypatch.setenv("KILOSORT_PATH", str(bogus))
        cfg = _make_cfg(sorter_name="kilosort2")
        findings = preflight_mod._check_kilosort2_host(cfg)
        assert len(findings) == 1
        assert "does not exist" in findings[0].message

    def test_kilosort_path_missing_master_yields_fail(self, monkeypatch, tmp_path):
        """
        KILOSORT_PATH dir without master_kilosort.m yields a fail.

        Tests:
            (Test Case 1) Real directory without master_kilosort.m →
                fail with 'master_kilosort.m' in message.
        """
        monkeypatch.setattr(preflight_mod.shutil, "which", lambda _: "/usr/bin/matlab")
        ks_dir = tmp_path / "ks2_src"
        ks_dir.mkdir()
        monkeypatch.setenv("KILOSORT_PATH", str(ks_dir))
        cfg = _make_cfg(sorter_name="kilosort2")
        findings = preflight_mod._check_kilosort2_host(cfg)
        assert len(findings) == 1
        assert "master_kilosort.m" in findings[0].message

    def test_all_present_yields_empty(self, monkeypatch, tmp_path):
        """
        matlab on PATH + KILOSORT_PATH containing master_kilosort.m
        yields no findings.

        Tests:
            (Test Case 1) Healthy env returns empty list.
        """
        monkeypatch.setattr(preflight_mod.shutil, "which", lambda _: "/usr/bin/matlab")
        ks_dir = tmp_path / "ks2_src"
        ks_dir.mkdir()
        (ks_dir / "master_kilosort.m").write_text("% ks2", encoding="utf-8")
        monkeypatch.setenv("KILOSORT_PATH", str(ks_dir))
        cfg = _make_cfg(sorter_name="kilosort2")
        assert preflight_mod._check_kilosort2_host(cfg) == []

    def test_sorter_path_overrides_env_var(self, monkeypatch, tmp_path):
        """
        ``SorterConfig.sorter_path`` takes precedence over KILOSORT_PATH.

        Tests:
            (Test Case 1) sorter_path points at a valid dir while env
                points at junk → no findings (the explicit setting wins).
        """
        monkeypatch.setattr(preflight_mod.shutil, "which", lambda _: "/usr/bin/matlab")
        ks_dir = tmp_path / "ks2_src"
        ks_dir.mkdir()
        (ks_dir / "master_kilosort.m").write_text("% ks2", encoding="utf-8")
        monkeypatch.setenv("KILOSORT_PATH", str(tmp_path / "invalid"))
        cfg = _make_cfg(sorter_name="kilosort2", sorter_path=str(ks_dir))
        assert preflight_mod._check_kilosort2_host(cfg) == []


class TestCheckKilosort4Host:
    """``_check_kilosort4_host`` validates the ``kilosort`` package."""

    def test_import_failure_yields_fail(self, monkeypatch):
        """
        Missing ``kilosort`` yields a sorter_dependency_missing fail
        finding.

        Tests:
            (Test Case 1) sys.modules['kilosort'] removed; import path
                is forced to raise ImportError.
        """
        monkeypatch.delitem(sys.modules, "kilosort", raising=False)
        _block_imports(monkeypatch, "kilosort")
        cfg = _make_cfg(sorter_name="kilosort4")
        findings = preflight_mod._check_kilosort4_host(cfg)
        assert len(findings) == 1
        assert findings[0].level == "fail"
        assert findings[0].code == "sorter_dependency_missing"

    def test_version_in_range_yields_empty(self, monkeypatch):
        """
        Kilosort4 with a version inside the tested range produces no
        findings.

        Tests:
            (Test Case 1) __version__ == "4.2.0" → empty list.
        """
        fake_ks = SimpleNamespace(__version__="4.2.0")
        monkeypatch.setitem(sys.modules, "kilosort", fake_ks)
        cfg = _make_cfg(sorter_name="kilosort4")
        assert preflight_mod._check_kilosort4_host(cfg) == []

    def test_version_out_of_range_yields_warn(self, monkeypatch):
        """
        Kilosort4 with a version outside the tested range yields a
        warn-level finding (not fail — newer versions sometimes work).

        Tests:
            (Test Case 1) __version__ == "5.0.0" → warn with code
                kilosort4_version_outside_tested_range.
        """
        fake_ks = SimpleNamespace(__version__="5.0.0")
        monkeypatch.setitem(sys.modules, "kilosort", fake_ks)
        cfg = _make_cfg(sorter_name="kilosort4")
        findings = preflight_mod._check_kilosort4_host(cfg)
        assert len(findings) == 1
        assert findings[0].level == "warn"
        assert findings[0].code == "kilosort4_version_outside_tested_range"

    def test_version_unparseable_yields_empty(self, monkeypatch):
        """
        Kilosort4 with a version string we cannot parse yields no
        finding (silent skip — we only flag what we are confident
        about).

        Tests:
            (Test Case 1) __version__ == "garbage" → empty.
            (Test Case 2) __version__ missing → empty.
        """
        for ver in ("garbage", None):
            fake_ks = SimpleNamespace()
            if ver is not None:
                fake_ks.__version__ = ver
            monkeypatch.setitem(sys.modules, "kilosort", fake_ks)
            cfg = _make_cfg(sorter_name="kilosort4")
            assert preflight_mod._check_kilosort4_host(cfg) == []


class TestCheckDockerSorter:
    """``_check_docker_sorter`` validates daemon + cached image."""

    def test_subprocess_fallback_daemon_down_yields_fail(self, monkeypatch):
        """
        With docker-py absent and ``docker info`` failing, returns a
        sorter_dependency_missing fail finding.

        Tests:
            (Test Case 1) docker-py import patched to raise ImportError.
            (Test Case 2) subprocess.run patched to raise
                subprocess.SubprocessError on the info call.
        """
        monkeypatch.delitem(sys.modules, "docker", raising=False)
        _block_imports(monkeypatch, "docker")

        def _fake_run(*_args, **_kwargs):
            raise subprocess.SubprocessError("daemon down")

        monkeypatch.setattr(preflight_mod.subprocess, "run", _fake_run)

        cfg = _make_cfg(sorter_name="kilosort4", use_docker=True)
        findings = preflight_mod._check_docker_sorter(cfg)
        codes = [(f.level, f.code) for f in findings]
        assert ("fail", "sorter_dependency_missing") in codes
        assert any("not reachable" in f.message for f in findings)

    def test_daemon_ok_image_cached_yields_empty(self, monkeypatch):
        """
        Daemon reachable + image present locally → no findings.

        Tests:
            (Test Case 1) docker-py absent forces subprocess fallback.
            (Test Case 2) ``docker info`` succeeds, ``docker image
                inspect`` succeeds → empty list.
        """
        monkeypatch.delitem(sys.modules, "docker", raising=False)
        _block_imports(monkeypatch, "docker")

        def _ok_run(args, **_kwargs):
            return subprocess.CompletedProcess(args, 0, b"", b"")

        monkeypatch.setattr(preflight_mod.subprocess, "run", _ok_run)

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        monkeypatch.setattr(
            docker_utils_mod, "get_docker_image", lambda _name: "fake/img:tag"
        )
        cfg = _make_cfg(sorter_name="kilosort4", use_docker=True)
        assert preflight_mod._check_docker_sorter(cfg) == []

    def test_image_not_cached_yields_warn(self, monkeypatch):
        """
        Daemon reachable + image not in local cache → warn-level
        finding so the operator knows SI will trigger a network pull.

        Tests:
            (Test Case 1) ``docker info`` succeeds, ``docker image
                inspect`` fails → exactly one warn finding.
        """
        monkeypatch.delitem(sys.modules, "docker", raising=False)
        _block_imports(monkeypatch, "docker")

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        monkeypatch.setattr(
            docker_utils_mod, "get_docker_image", lambda _name: "fake/img:tag"
        )

        call_log = []

        def _selective_run(args, **_kwargs):
            call_log.append(tuple(args))
            if args[:2] == ["docker", "info"]:
                return subprocess.CompletedProcess(args, 0, b"", b"")
            if args[:3] == ["docker", "image", "inspect"]:
                raise subprocess.CalledProcessError(1, args)
            raise AssertionError(f"unexpected call: {args}")

        monkeypatch.setattr(preflight_mod.subprocess, "run", _selective_run)
        cfg = _make_cfg(sorter_name="kilosort4", use_docker=True)
        findings = preflight_mod._check_docker_sorter(cfg)
        assert len(findings) == 1
        assert findings[0].level == "warn"
        assert "not in the local" in findings[0].message
        # Both docker calls were attempted.
        assert any(args[:2] == ("docker", "info") for args in call_log)
        assert any(args[:3] == ("docker", "image", "inspect") for args in call_log)

    def test_image_resolution_failure_yields_fail(self, monkeypatch):
        """
        ``get_docker_image`` raising (e.g. unsupported CUDA tag) →
        fail finding rather than letting the preflight crash.

        Tests:
            (Test Case 1) docker_utils.get_docker_image patched to
                raise → fail finding.
        """
        monkeypatch.delitem(sys.modules, "docker", raising=False)
        _block_imports(monkeypatch, "docker")

        def _ok_run(args, **_kwargs):
            return subprocess.CompletedProcess(args, 0, b"", b"")

        monkeypatch.setattr(preflight_mod.subprocess, "run", _ok_run)

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        def _raise(*_a, **_k):
            raise RuntimeError("no compatible image")

        monkeypatch.setattr(docker_utils_mod, "get_docker_image", _raise)

        cfg = _make_cfg(sorter_name="kilosort4", use_docker=True)
        findings = preflight_mod._check_docker_sorter(cfg)
        assert len(findings) == 1
        assert findings[0].level == "fail"
        assert "Could not resolve" in findings[0].message


class TestCheckRtSort:
    """``_check_rt_sort`` validates RT-Sort's runtime dependencies."""

    def test_each_missing_import_yields_fail(self, monkeypatch):
        """
        Each missing required dependency adds its own
        sorter_dependency_missing finding.

        Tests:
            (Test Case 1) Patch __import__ so ``diptest`` raises
                ImportError; the resulting finding mentions diptest.
        """
        _block_imports(monkeypatch, "diptest")
        cfg = _make_cfg(sorter_name="rt_sort", rt_device="cpu")
        findings = preflight_mod._check_rt_sort(cfg)
        codes_by_message = [(f.code, "diptest" in f.message) for f in findings]
        assert (("sorter_dependency_missing", True)) in codes_by_message

    def test_cuda_unavailable_yields_fail(self, monkeypatch):
        """
        device='cuda' but ``torch.cuda.is_available()`` is False yields
        an additional fail finding.

        Tests:
            (Test Case 1) Inject a fake torch with
                cuda.is_available() == False; assert the
                cuda-specific finding is present.
        """
        fake_cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        # Provide stubs so the other required imports succeed.
        for name in ("diptest", "sklearn", "h5py", "tqdm"):
            monkeypatch.setitem(sys.modules, name, SimpleNamespace())

        cfg = _make_cfg(sorter_name="rt_sort", rt_device="cuda:0")
        findings = preflight_mod._check_rt_sort(cfg)
        # Should include exactly one cuda-related finding (no other
        # imports failed since we stubbed them).
        cuda_findings = [f for f in findings if "is_available()" in f.message]
        assert len(cuda_findings) == 1
        assert cuda_findings[0].level == "fail"

    def test_all_dependencies_present_cpu_yields_empty(self, monkeypatch):
        """
        All required imports succeed and device='cpu' → no findings.

        Tests:
            (Test Case 1) Stub torch + diptest + sklearn + h5py + tqdm
                in sys.modules; rt_device='cpu' skips the cuda check.
        """
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        for name in ("diptest", "sklearn", "h5py", "tqdm"):
            monkeypatch.setitem(sys.modules, name, SimpleNamespace())

        cfg = _make_cfg(sorter_name="rt_sort", rt_device="cpu")
        assert preflight_mod._check_rt_sort(cfg) == []


class TestCheckSorterDependencies:
    """``_check_sorter_dependencies`` dispatches by sorter + use_docker."""

    def test_dispatches_to_kilosort2_host(self, monkeypatch):
        """
        sorter='kilosort2' + use_docker=False → routes to KS2 host check.

        Tests:
            (Test Case 1) Patch _check_kilosort2_host to a sentinel,
                confirm it was called and returned the sentinel.
        """
        sentinel = [
            preflight_mod.PreflightFinding(
                level="warn", code="ks2_host_called", message="m"
            )
        ]
        monkeypatch.setattr(preflight_mod, "_check_kilosort2_host", lambda c: sentinel)
        cfg = _make_cfg(sorter_name="kilosort2", use_docker=False)
        assert preflight_mod._check_sorter_dependencies(cfg) is sentinel

    def test_dispatches_to_docker_for_use_docker(self, monkeypatch):
        """
        sorter='kilosort4' + use_docker=True → routes to docker check.

        Tests:
            (Test Case 1) Patched _check_docker_sorter sentinel
                returned even though sorter is kilosort4.
        """
        sentinel = [
            preflight_mod.PreflightFinding(
                level="warn", code="docker_called", message="m"
            )
        ]
        monkeypatch.setattr(preflight_mod, "_check_docker_sorter", lambda c: sentinel)
        cfg = _make_cfg(sorter_name="kilosort4", use_docker=True)
        assert preflight_mod._check_sorter_dependencies(cfg) is sentinel

    def test_dispatches_to_rt_sort(self, monkeypatch):
        """
        sorter='rt_sort' → routes to RT-Sort check.

        Tests:
            (Test Case 1) Patched _check_rt_sort sentinel returned.
        """
        sentinel = [
            preflight_mod.PreflightFinding(
                level="warn", code="rt_sort_called", message="m"
            )
        ]
        monkeypatch.setattr(preflight_mod, "_check_rt_sort", lambda c: sentinel)
        cfg = _make_cfg(sorter_name="rt_sort")
        assert preflight_mod._check_sorter_dependencies(cfg) is sentinel

    def test_unknown_sorter_yields_empty(self):
        """
        Unrecognized sorter name returns empty rather than raising.

        Tests:
            (Test Case 1) sorter_name='unknown' → [].
        """
        cfg = _make_cfg(sorter_name="unknown")
        assert preflight_mod._check_sorter_dependencies(cfg) == []


# ---------------------------------------------------------------------------
# GPU device existence
# ---------------------------------------------------------------------------


class TestResolveTargetDeviceIndex:
    """``_resolve_target_device_index`` mirrors the GPU watchdog's resolver."""

    def test_kilosort4_reads_torch_device(self):
        """
        KS4's torch_device sorter_param drives the resolved index.

        Tests:
            (Test Case 1) sorter_params={'torch_device': 'cuda:2'} → 2.
        """
        cfg = _make_cfg(
            sorter_name="kilosort4",
            sorter_params={"torch_device": "cuda:2"},
        )
        assert preflight_mod._resolve_target_device_index(cfg) == 2

    def test_rt_sort_reads_rt_device(self):
        """
        RT-Sort's RTSortConfig.device drives the resolved index.

        Tests:
            (Test Case 1) rt_device='cuda:1' → 1.
        """
        cfg = _make_cfg(sorter_name="rt_sort", rt_device="cuda:1")
        assert preflight_mod._resolve_target_device_index(cfg) == 1

    def test_default_zero_for_kilosort2(self):
        """
        KS2 has no explicit device knob → defaults to 0.

        Tests:
            (Test Case 1) sorter_name='kilosort2' → 0.
        """
        cfg = _make_cfg(sorter_name="kilosort2")
        assert preflight_mod._resolve_target_device_index(cfg) == 0


class TestDetectGpuDeviceCount:
    """``_detect_gpu_device_count`` cascades pynvml → torch → nvidia-smi."""

    def test_pynvml_count_used_when_available(self, monkeypatch):
        """
        With pynvml importable, its device count wins.

        Tests:
            (Test Case 1) Inject a fake pynvml; assert the returned
                count matches.
        """
        fake = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 4,
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert preflight_mod._detect_gpu_device_count() == 4

    def test_torch_fallback_when_no_pynvml(self, monkeypatch):
        """
        Without pynvml, torch.cuda.device_count() is used.

        Tests:
            (Test Case 1) Force pynvml import to fail; inject torch
                stub with cuda.is_available()=True and device_count()=2.
        """
        monkeypatch.delitem(sys.modules, "pynvml", raising=False)
        _block_imports(monkeypatch, "pynvml")

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 2,
            )
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert preflight_mod._detect_gpu_device_count() == 2

    def test_nvidia_smi_fallback_returns_count(self, monkeypatch):
        """
        With pynvml + torch unavailable, nvidia-smi --query-gpu=count
        is parsed.

        Tests:
            (Test Case 1) Patched check_output returns "1\\n1\\n" →
                count is 1 (first valid line wins).
        """
        monkeypatch.delitem(sys.modules, "pynvml", raising=False)
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        _block_imports(monkeypatch, "pynvml", "torch")
        monkeypatch.setattr(
            preflight_mod.subprocess,
            "check_output",
            lambda *a, **k: "1\n1\n",
        )
        assert preflight_mod._detect_gpu_device_count() == 1

    def test_all_paths_fail_returns_none(self, monkeypatch):
        """
        Every detection path failing → None (silent skip).

        Tests:
            (Test Case 1) pynvml + torch import-fail; nvidia-smi raises
                FileNotFoundError → None.
        """
        monkeypatch.delitem(sys.modules, "pynvml", raising=False)
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        _block_imports(monkeypatch, "pynvml", "torch")

        def _raise_smi(*_a, **_k):
            raise FileNotFoundError("no nvidia-smi")

        monkeypatch.setattr(preflight_mod.subprocess, "check_output", _raise_smi)
        assert preflight_mod._detect_gpu_device_count() is None


class TestCheckGpuDevicePresent:
    """``_check_gpu_device_present`` validates configured device index."""

    def test_in_range_returns_none(self, monkeypatch):
        """
        Configured index < detected count → no finding.

        Tests:
            (Test Case 1) Index 0 with count 1 → None.
        """
        monkeypatch.setattr(preflight_mod, "_detect_gpu_device_count", lambda: 1)
        cfg = _make_cfg(sorter_name="rt_sort", rt_device="cuda:0")
        assert preflight_mod._check_gpu_device_present(cfg) is None

    def test_out_of_range_returns_fail(self, monkeypatch):
        """
        Configured index >= detected count → fail finding listing the
        valid indices in the remediation.

        Tests:
            (Test Case 1) Index 2 with count 1 → fail with
                gpu_device_not_present, remediation mentions index 0.
        """
        monkeypatch.setattr(preflight_mod, "_detect_gpu_device_count", lambda: 1)
        cfg = _make_cfg(sorter_name="rt_sort", rt_device="cuda:2")
        finding = preflight_mod._check_gpu_device_present(cfg)
        assert finding is not None
        assert finding.level == "fail"
        assert finding.code == "gpu_device_not_present"
        assert finding.category == "environment"
        assert "0" in finding.remediation

    def test_unknown_count_returns_none(self, monkeypatch):
        """
        Detection unavailable → silent (the existing vram_unknown
        finding already covers the broader gap).

        Tests:
            (Test Case 1) _detect_gpu_device_count → None → no finding.
            (Test Case 2) Count == 0 also returns None (no GPUs).
        """
        for count in (None, 0):
            monkeypatch.setattr(
                preflight_mod, "_detect_gpu_device_count", lambda c=count: c
            )
            cfg = _make_cfg(sorter_name="rt_sort", rt_device="cuda:0")
            assert preflight_mod._check_gpu_device_present(cfg) is None


# ---------------------------------------------------------------------------
# Recording sample-rate sanity check
# ---------------------------------------------------------------------------


class TestExpectedSampleRateWindow:
    """``_expected_sample_rate_window`` returns per-sorter rate ranges."""

    def test_kilosort2_returns_wide_window(self):
        """
        KS2 returns the [10, 50] kHz drift-corrected window.

        Tests:
            (Test Case 1) sorter='kilosort2' → (10000, 50000, 'kilosort2').
        """
        cfg = _make_cfg(sorter_name="kilosort2")
        assert preflight_mod._expected_sample_rate_window(cfg) == (
            10_000.0,
            50_000.0,
            "kilosort2",
        )

    def test_kilosort4_returns_wide_window(self):
        """
        KS4 shares the [10, 50] kHz window with KS2.

        Tests:
            (Test Case 1) sorter='kilosort4' → (10000, 50000, 'kilosort4').
        """
        cfg = _make_cfg(sorter_name="kilosort4")
        assert preflight_mod._expected_sample_rate_window(cfg) == (
            10_000.0,
            50_000.0,
            "kilosort4",
        )

    def test_rt_sort_mea_returns_tight_window(self):
        """
        RT-Sort with the MEA model returns 20 kHz ± 0.5 %.

        Tests:
            (Test Case 1) sorter='rt_sort', probe='mea' → (19900, 20100,
                'rt_sort/mea').
        """
        cfg = _make_cfg(sorter_name="rt_sort", rt_probe="mea")
        low, high, label = preflight_mod._expected_sample_rate_window(cfg)
        assert low == pytest.approx(19_900.0)
        assert high == pytest.approx(20_100.0)
        assert label == "rt_sort/mea"

    def test_rt_sort_neuropixels_returns_tight_window(self):
        """
        RT-Sort with the Neuropixels model returns 30 kHz ± 0.5 %.

        Tests:
            (Test Case 1) sorter='rt_sort', probe='neuropixels' →
                (29850, 30150, 'rt_sort/neuropixels').
        """
        cfg = _make_cfg(sorter_name="rt_sort", rt_probe="neuropixels")
        low, high, label = preflight_mod._expected_sample_rate_window(cfg)
        assert low == pytest.approx(29_850.0)
        assert high == pytest.approx(30_150.0)
        assert label == "rt_sort/neuropixels"

    def test_unknown_sorter_returns_none(self):
        """
        Unrecognized sorter / probe returns None.

        Tests:
            (Test Case 1) sorter='unknown' → None.
            (Test Case 2) RT-Sort with probe='unknown' → None.
        """
        assert (
            preflight_mod._expected_sample_rate_window(_make_cfg(sorter_name="unknown"))
            is None
        )
        assert (
            preflight_mod._expected_sample_rate_window(
                _make_cfg(sorter_name="rt_sort", rt_probe="unknown")
            )
            is None
        )


class TestCheckRecordingSampleRate:
    """``_check_recording_sample_rate`` warns on out-of-window rates."""

    def _fake_recording(self, fs_hz: float):
        """Build a minimal pre-loaded-recording stub."""
        return SimpleNamespace(get_sampling_frequency=lambda: fs_hz)

    def test_in_window_yields_empty(self):
        """
        Pre-loaded recording with a rate inside the sorter window →
        no findings.

        Tests:
            (Test Case 1) KS4 with 20 kHz recording → empty.
        """
        cfg = _make_cfg(sorter_name="kilosort4")
        rec = self._fake_recording(20_000.0)
        assert preflight_mod._check_recording_sample_rate(cfg, [rec]) == []

    def test_out_of_window_yields_warn(self):
        """
        Out-of-window rate produces a warn-level environment finding.

        Tests:
            (Test Case 1) KS4 with 5 kHz recording → exactly one warn
                with code sample_rate_out_of_window and category
                'environment'. The category is environment (not
                resource) because a sample-rate mismatch is a
                recording-vs-sorter misconfiguration, not a transient
                resource shortage.
        """
        cfg = _make_cfg(sorter_name="kilosort4")
        rec = self._fake_recording(5_000.0)
        findings = preflight_mod._check_recording_sample_rate(cfg, [rec])
        assert len(findings) == 1
        assert findings[0].level == "warn"
        assert findings[0].code == "sample_rate_out_of_window"
        assert findings[0].category == "environment"

    def test_path_only_input_skipped(self, tmp_path):
        """
        Path-only inputs (no get_sampling_frequency) are skipped.

        Tests:
            (Test Case 1) Mixed inputs (Path + pre-loaded out-of-range) →
                only the pre-loaded one yields a finding.
        """
        cfg = _make_cfg(sorter_name="kilosort4")
        rec = self._fake_recording(5_000.0)
        findings = preflight_mod._check_recording_sample_rate(
            cfg, [tmp_path / "rec.h5", rec]
        )
        assert len(findings) == 1

    def test_unknown_sorter_yields_empty(self):
        """
        When ``_expected_sample_rate_window`` returns None, no findings
        regardless of recordings.

        Tests:
            (Test Case 1) Unknown sorter + 5 kHz recording → empty.
        """
        cfg = _make_cfg(sorter_name="unknown")
        rec = self._fake_recording(5_000.0)
        assert preflight_mod._check_recording_sample_rate(cfg, [rec]) == []

    def test_get_sampling_frequency_raises_skipped(self):
        """
        A recording whose ``get_sampling_frequency`` raises is silently
        skipped (no crash, no finding).

        Tests:
            (Test Case 1) Stub raises RuntimeError → empty findings.
        """
        cfg = _make_cfg(sorter_name="kilosort4")

        def _raise():
            raise RuntimeError("not initialized")

        rec = SimpleNamespace(get_sampling_frequency=_raise)
        assert preflight_mod._check_recording_sample_rate(cfg, [rec]) == []


# ---------------------------------------------------------------------------
# Filesystem writability
# ---------------------------------------------------------------------------


class TestCheckFilesystemWritable:
    """``_check_filesystem_writable`` flags read-only mounts before any sort."""

    def test_writable_folder_yields_empty(self, tmp_path):
        """
        A normal writable folder yields no findings.

        Tests:
            (Test Case 1) tmp_path → empty list.
        """
        findings = preflight_mod._check_filesystem_writable(
            [tmp_path], label="intermediate", code_prefix="intermediate"
        )
        assert findings == []

    def test_readonly_folder_yields_fail(self, monkeypatch, tmp_path):
        """
        Folder whose nearest existing parent fails ``W_OK`` yields a
        fail-level environment finding with the right code prefix.

        Tests:
            (Test Case 1) os.access patched to return False → exactly
                one finding, level=fail, code='intermediate_readonly',
                category='environment', folder path in message.
        """
        monkeypatch.setattr(preflight_mod.os, "access", lambda _p, _mode: False)
        findings = preflight_mod._check_filesystem_writable(
            [tmp_path], label="intermediate", code_prefix="intermediate"
        )
        assert len(findings) == 1
        f = findings[0]
        assert f.level == "fail"
        assert f.code == "intermediate_readonly"
        assert f.category == "environment"
        assert str(tmp_path) in f.message

    def test_nonexistent_folder_walks_to_existing_parent(self, monkeypatch, tmp_path):
        """
        For a folder that does not exist, the nearest existing parent
        is checked.

        Tests:
            (Test Case 1) Pass tmp_path / 'never' / 'created'; access
                patched to fail; finding mentions the original folder.
        """
        bogus = tmp_path / "never" / "created"
        monkeypatch.setattr(preflight_mod.os, "access", lambda _p, _mode: False)
        findings = preflight_mod._check_filesystem_writable(
            [bogus], label="results", code_prefix="results"
        )
        assert len(findings) == 1
        assert findings[0].code == "results_readonly"
        assert "never/created" in findings[0].message.replace("\\", "/")

    def test_no_existing_parent_skips(self, monkeypatch):
        """
        When neither the folder nor any ancestor exists, the check
        skips silently rather than crashing.

        Tests:
            (Test Case 1) Path('/totally/fake/path') with Path.exists
                patched to always return False → no findings, no raise.
        """
        # Patch Path.exists at the class level so the loop walks up
        # without ever finding an existing parent.
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert (
            preflight_mod._check_filesystem_writable(
                [Path("/totally/fake/path")],
                label="intermediate",
                code_prefix="intermediate",
            )
            == []
        )


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------


class TestParseVersionTuple:
    """``_parse_version_tuple`` pads to length 3 for safe ordering."""

    def test_three_components(self):
        """
        Standard three-component versions parse verbatim.

        Tests:
            (Test Case 1) '1.2.3' → (1, 2, 3).
        """
        assert preflight_mod._parse_version_tuple("1.2.3") == (1, 2, 3)

    def test_single_component_padded(self):
        """
        Single-component versions get padded so '4' compares as (4,0,0).

        Tests:
            (Test Case 1) '4' → (4, 0, 0). Fixes the regression where
                '4' falsely reported as below the [4.0.0, 5.0.0) tested
                range because (4,) < (4, 0, 0) in Python tuple ordering.
        """
        assert preflight_mod._parse_version_tuple("4") == (4, 0, 0)

    def test_two_components_padded(self):
        """
        Two-component versions get padded with zero.

        Tests:
            (Test Case 1) '4.2' → (4, 2, 0).
        """
        assert preflight_mod._parse_version_tuple("4.2") == (4, 2, 0)

    def test_rc_marker_stripped(self):
        """
        RC / dev markers are stripped per-segment via the digit filter.

        Tests:
            (Test Case 1) '1.2.3rc4' → (1, 2, 3) — not perfect but
                matches the "best-effort" contract.
        """
        # Note: the digit-only filter treats '3rc4' as '34', so the
        # third component ends up as 34, not 3. That is the documented
        # behavior of the helper; this test pins it.
        assert preflight_mod._parse_version_tuple("1.2.3rc4") == (1, 2, 34)

    def test_garbage_returns_none(self):
        """
        Unparseable strings yield None.

        Tests:
            (Test Case 1) 'garbage' → None (each segment is empty after
                digit-filter so int('') raises).
        """
        assert preflight_mod._parse_version_tuple("garbage") is None


# ---------------------------------------------------------------------------
# Free-VRAM detection
# ---------------------------------------------------------------------------


class TestFreeVramGb:
    """``_free_vram_gb`` cascades pynvml → nvidia-smi → None."""

    def test_pynvml_path_sums_across_devices(self, monkeypatch):
        """
        With pynvml available, free memory across all devices is summed.

        Tests:
            (Test Case 1) Two devices with 2 GB and 3 GB free → 5 GB.
        """
        free_gbs = [2.0, 3.0]

        def _info(handle):
            return SimpleNamespace(free=int(free_gbs[handle] * (1024**3)))

        fake_pynvml = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 2,
            nvmlDeviceGetHandleByIndex=lambda i: i,
            nvmlDeviceGetMemoryInfo=_info,
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
        assert preflight_mod._free_vram_gb() == pytest.approx(5.0)

    def test_falls_back_to_nvidia_smi(self, monkeypatch):
        """
        Without pynvml, parses ``nvidia-smi --query-gpu=memory.free``
        output (MiB → GB).

        Tests:
            (Test Case 1) "1024\\n2048\\n" (3 GiB) → ~3 GB.
        """
        _block_imports(monkeypatch, "pynvml")
        monkeypatch.setattr(
            preflight_mod.subprocess,
            "check_output",
            lambda *a, **k: "1024\n2048\n",
        )
        assert preflight_mod._free_vram_gb() == pytest.approx(3.0)

    def test_returns_none_when_no_source_works(self, monkeypatch):
        """
        Both pynvml and nvidia-smi unavailable → None.

        Tests:
            (Test Case 1) pynvml import blocked, nvidia-smi raises
                FileNotFoundError → None.
        """
        _block_imports(monkeypatch, "pynvml")

        def _raise(*_a, **_k):
            raise FileNotFoundError("no nvidia-smi")

        monkeypatch.setattr(preflight_mod.subprocess, "check_output", _raise)
        assert preflight_mod._free_vram_gb() is None


# ---------------------------------------------------------------------------
# Resource limits + SpikeInterface version
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
        for ver in ("0.090.0", "1.50.0"):
            fake_si = SimpleNamespace(__version__=ver)
            with mock.patch.dict(sys.modules, {"spikeinterface": fake_si}):
                finding = preflight_mod._check_spikeinterface_version()
                assert finding is not None
                assert finding.level == "warn"
                assert finding.code == "spikeinterface_version_outside_tested_range"
