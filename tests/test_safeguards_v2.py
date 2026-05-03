"""Tests for the second safeguarding round — FEAT-001..005.

Covered modules:

* ``guards/_preflight`` — sorter dependency probes
  (``_check_kilosort2_host``, ``_check_kilosort4_host``,
  ``_check_docker_sorter``, ``_check_rt_sort``,
  ``_check_sorter_dependencies``); GPU device-index existence
  (``_check_gpu_device_present`` + ``_resolve_target_device_index``
  + ``_detect_gpu_device_count``); recording sample-rate sanity
  check (``_check_recording_sample_rate`` +
  ``_expected_sample_rate_window``).
* ``canary`` — short-window smoke test
  (``_build_canary_config``, ``_wipe_canary_folder``,
  ``run_canary``).
* ``docker_utils.get_local_image_digest`` — local Docker image
  digest lookup with docker-py and subprocess fallback.
* ``ExecutionConfig`` — defaults + flat-map round-trip for the new
  ``canary_first_n_s`` and ``docker_image_expected_digest`` fields.

All tests are hermetic: every detection path that would otherwise
touch the host (matlab on PATH, kilosort import, docker daemon,
nvidia-smi, real recordings) is patched.
"""

from __future__ import annotations

import builtins
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


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


from spikelab.spike_sorting._exceptions import (
    BiologicalSortFailure,
    EnvironmentSortFailure,
    InsufficientActivityError,
    ResourceSortFailure,
)
from spikelab.spike_sorting.config import (
    ExecutionConfig,
    SortingPipelineConfig,
)
from spikelab.spike_sorting.guards import _preflight as preflight_mod

# ---------------------------------------------------------------------------
# Shared lightweight config builder
# ---------------------------------------------------------------------------


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
    extends it with the additional sub-fields that FEAT-001..005 read:
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
# FEAT-001 — Sorter dependency probes
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
# FEAT-002 — GPU device existence preflight
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
# FEAT-003 — Recording sample-rate sanity check
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
        Out-of-window rate produces a warn-level resource finding.

        Tests:
            (Test Case 1) KS4 with 5 kHz recording → exactly one warn
                with code sample_rate_out_of_window and category
                'resource'.
        """
        cfg = _make_cfg(sorter_name="kilosort4")
        rec = self._fake_recording(5_000.0)
        findings = preflight_mod._check_recording_sample_rate(cfg, [rec])
        assert len(findings) == 1
        assert findings[0].level == "warn"
        assert findings[0].code == "sample_rate_out_of_window"
        assert findings[0].category == "resource"

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
# FEAT-004 — Pipeline canary
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FEAT-005 — Docker image digest pinning
# ---------------------------------------------------------------------------


class TestGetLocalImageDigest:
    """``docker_utils.get_local_image_digest`` returns sha256 or None."""

    def test_empty_tag_returns_none(self):
        """
        Empty / falsy tag short-circuits to None.

        Tests:
            (Test Case 1) tag='' → None.
        """
        from spikelab.spike_sorting.docker_utils import get_local_image_digest

        assert get_local_image_digest("") is None

    def test_docker_py_path_used_when_available(self, monkeypatch):
        """
        Python ``docker`` client returns the digest via images.get().id.

        Tests:
            (Test Case 1) Inject a fake docker module with
                from_env().images.get(tag).id == 'sha256:abc'.
        """
        fake_image = SimpleNamespace(id="sha256:abc")
        fake_client = SimpleNamespace(
            images=SimpleNamespace(get=lambda tag: fake_image)
        )
        fake_docker = SimpleNamespace(from_env=lambda: fake_client)
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        from spikelab.spike_sorting.docker_utils import get_local_image_digest

        assert get_local_image_digest("foo:bar") == "sha256:abc"

    def test_subprocess_fallback_when_docker_py_missing(self, monkeypatch):
        """
        Without docker-py, ``docker inspect --format={{.Id}}`` is used.

        Tests:
            (Test Case 1) docker-py import patched to raise; subprocess
                stub returns 'sha256:def\\n' → trimmed to 'sha256:def'.
        """
        monkeypatch.delitem(sys.modules, "docker", raising=False)
        _block_imports(monkeypatch, "docker")

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        def _fake_run(args, **_kwargs):
            return subprocess.CompletedProcess(args, 0, "sha256:def\n", "")

        monkeypatch.setattr(docker_utils_mod.subprocess, "run", _fake_run)
        assert docker_utils_mod.get_local_image_digest("foo:bar") == "sha256:def"

    def test_both_paths_fail_returns_none(self, monkeypatch):
        """
        docker-py absent and ``docker inspect`` failing → None.

        Tests:
            (Test Case 1) Import-fail + subprocess raises → None.
        """
        monkeypatch.delitem(sys.modules, "docker", raising=False)
        _block_imports(monkeypatch, "docker")

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        def _fake_run(*_a, **_k):
            raise FileNotFoundError("no docker cli")

        monkeypatch.setattr(docker_utils_mod.subprocess, "run", _fake_run)
        assert docker_utils_mod.get_local_image_digest("foo:bar") is None

    def test_docker_py_get_failure_falls_back_to_subprocess(self, monkeypatch):
        """
        When docker-py is importable but ``images.get(tag)`` raises,
        the function falls back to the ``docker inspect`` subprocess
        path rather than returning None outright.

        Tests:
            (Test Case 1) docker.from_env().images.get raises → CLI
                fallback returns sha256:cli.
        """
        fake_client = SimpleNamespace(
            images=SimpleNamespace(get=mock.Mock(side_effect=RuntimeError("not found")))
        )
        fake_docker = SimpleNamespace(from_env=lambda: fake_client)
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        from spikelab.spike_sorting import docker_utils as docker_utils_mod

        def _fake_run(args, **_kwargs):
            return subprocess.CompletedProcess(args, 0, "sha256:cli\n", "")

        monkeypatch.setattr(docker_utils_mod.subprocess, "run", _fake_run)
        assert docker_utils_mod.get_local_image_digest("foo:bar") == "sha256:cli"


# ---------------------------------------------------------------------------
# Config field defaults + flat-map round-trip
# ---------------------------------------------------------------------------


class TestNewExecutionConfigFields:
    """``ExecutionConfig`` exposes the new canary + digest fields."""

    def test_defaults(self):
        """
        New fields default to disabled / unset.

        Tests:
            (Test Case 1) canary_first_n_s defaults to 0.0.
            (Test Case 2) docker_image_expected_digest defaults to None.
        """
        cfg = ExecutionConfig()
        assert cfg.canary_first_n_s == 0.0
        assert cfg.docker_image_expected_digest is None

    def test_flat_map_round_trip(self):
        """
        Both fields can be set via SortingPipelineConfig.from_kwargs()
        and survive ``override``.

        Tests:
            (Test Case 1) from_kwargs accepts canary_first_n_s and
                docker_image_expected_digest.
            (Test Case 2) override re-sets them.
        """
        cfg = SortingPipelineConfig.from_kwargs(
            canary_first_n_s=12.5,
            docker_image_expected_digest="sha256:abc",
        )
        assert cfg.execution.canary_first_n_s == 12.5
        assert cfg.execution.docker_image_expected_digest == "sha256:abc"
        cfg2 = cfg.override(canary_first_n_s=0.0)
        assert cfg2.execution.canary_first_n_s == 0.0
        # Other field preserved.
        assert cfg2.execution.docker_image_expected_digest == "sha256:abc"


# ---------------------------------------------------------------------------
# FEAT-005 — _print_pipeline_banner Docker lines
# ---------------------------------------------------------------------------


class TestPrintPipelineBannerDockerLines:
    """``_print_pipeline_banner`` surfaces Docker image + digest."""

    def test_lines_emitted_when_use_docker(self, capsys, tmp_path):
        """
        With use_docker=True and both kwargs supplied, the banner
        prints 'Docker image:' and 'Docker image digest:' lines under
        the Environment section so the report parser picks them up.

        Tests:
            (Test Case 1) Both lines present in stdout.
            (Test Case 2) Image tag + digest values appear verbatim.
        """
        from spikelab.spike_sorting.config import SorterConfig
        from spikelab.spike_sorting.pipeline import _print_pipeline_banner

        cfg = SortingPipelineConfig(
            sorter=SorterConfig(sorter_name="kilosort4", use_docker=True)
        )
        _print_pipeline_banner(
            "kilosort4",
            "/data/rec.h5",
            cfg,
            log_path=tmp_path / "log.log",
            recording=None,
            docker_image_tag="spikeinterface/kilosort4-base:py311-si0.104",
            docker_image_digest="sha256:deadbeef",
        )
        out = capsys.readouterr().out
        assert "Docker image:" in out
        assert "spikeinterface/kilosort4-base:py311-si0.104" in out
        assert "Docker image digest: sha256:deadbeef" in out

    def test_lines_suppressed_when_use_docker_false(self, capsys, tmp_path):
        """
        Without use_docker the Docker lines are not printed even when
        the kwargs are supplied.

        Tests:
            (Test Case 1) use_docker=False suppresses both lines.
        """
        from spikelab.spike_sorting.config import SorterConfig
        from spikelab.spike_sorting.pipeline import _print_pipeline_banner

        cfg = SortingPipelineConfig(
            sorter=SorterConfig(sorter_name="kilosort4", use_docker=False)
        )
        _print_pipeline_banner(
            "kilosort4",
            "/data/rec.h5",
            cfg,
            log_path=tmp_path / "log.log",
            recording=None,
            docker_image_tag="spikeinterface/kilosort4-base:py311-si0.104",
            docker_image_digest="sha256:deadbeef",
        )
        out = capsys.readouterr().out
        assert "Docker image:" not in out
        assert "Docker image digest:" not in out
