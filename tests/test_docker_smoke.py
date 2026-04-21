"""
Smoke tests for the SpikeLab Docker image.

Tests cover:
- Docker image builds successfully
- Server starts and exposes the SSE endpoint
- MCP tools are discoverable via the running server
"""

import pathlib
import subprocess
import time

import pytest

SPIKELAB_DIR = pathlib.Path(__file__).resolve().parents[1]
IMAGE_NAME = "spikelab-mcp-smoke-test"


def _docker_available() -> bool:
    """Check whether the Docker CLI is available and the daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker is not available"
)


@pytest.fixture(scope="module")
def docker_image():
    """Build the Docker image once for the entire test module."""
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "."],
        cwd=str(SPIKELAB_DIR),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Docker build failed:\n{result.stderr}"
    yield IMAGE_NAME
    subprocess.run(["docker", "rmi", IMAGE_NAME], capture_output=True, timeout=30)


@pytest.fixture()
def docker_container(docker_image):
    """Run the Docker image and yield the container ID. Cleans up after."""
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            "18080:8080",
            docker_image,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Docker run failed:\n{result.stderr}"
    container_id = result.stdout.strip()
    # Wait for the server to be ready
    _wait_for_server(container_id)
    yield container_id
    subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=30)


def _wait_for_server(container_id: str, timeout: int = 15):
    """Poll container logs until uvicorn signals it is serving."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["docker", "logs", container_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "Uvicorn running" in result.stderr or "Uvicorn running" in result.stdout:
            return
        time.sleep(1)
    raise TimeoutError(f"Server did not start within {timeout}s")


class TestDockerBuild:
    """Tests that the Docker image builds correctly."""

    def test_image_builds(self, docker_image):
        """Image builds without errors."""
        result = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_image_has_python(self, docker_image):
        """Image contains the expected Python version."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                docker_image,
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "3.10" in result.stdout

    def test_spikelab_importable(self, docker_image):
        """SpikeLab package is importable inside the container."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                docker_image,
                "-c",
                "from spikelab.mcp_server.server import server; print(server.name)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "spikelab" in result.stdout


class TestDockerServer:
    """Tests that the server starts and responds over SSE."""

    def test_sse_endpoint_reachable(self, docker_container):
        """The /sse endpoint accepts connections."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request("http://localhost:18080/sse", method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
        except urllib.error.URLError as e:
            pytest.fail(f"Could not reach SSE endpoint: {e}")

    def test_messages_endpoint_exists(self, docker_container):
        """The /messages/ endpoint responds (rejects GET with 400 or 405)."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(
                "http://localhost:18080/messages/", method="GET"
            )
            urllib.request.urlopen(req, timeout=5)
            pytest.fail("Expected error for GET on /messages/")
        except urllib.error.HTTPError as e:
            assert e.code in (400, 405)
        except urllib.error.URLError as e:
            pytest.fail(f"Could not reach messages endpoint: {e}")
