#!/bin/bash
# Run Kilosort2 spike sorting with a system-enforced memory limit.
#
# Uses systemd-run to cap the entire pipeline (host Python + Docker) at
# 80% of system RAM.  If the pipeline exceeds the limit, the kernel
# kills the sorting process — not the desktop or remote sessions.
#
# Requirements:
#   - Linux with systemd (user session)
#   - NVIDIA GPU with Docker GPU support
#   - conda environment with spikelab installed
#
# Usage:
#   bash run_kilosort2_docker.sh <python_script> [results_dir]
#
# Arguments:
#   python_script   Path to a Python script that calls sort_with_kilosort2()
#   results_dir     Directory for log files (default: directory of python_script)
#
# Example:
#   bash docker/run_kilosort2_docker.sh scripts/sort_my_recording.py results/

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script> [results_dir]"
    echo "  python_script   Python script calling sort_with_kilosort2()"
    echo "  results_dir     Directory for log files (default: script directory)"
    exit 1
fi

PYTHON_SCRIPT="$(realpath "$1")"
RESULTS_DIR="${2:-$(dirname "$PYTHON_SCRIPT")}"
mkdir -p "$RESULTS_DIR"
LOGFILE="${RESULTS_DIR}/sorting_$(date +%y%m%d_%H%M%S).log"

# Detect conda environment name (default: spikelab)
CONDA_ENV="${SPIKELAB_CONDA_ENV:-spikelab}"

# 80% of total RAM
TOTAL_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
LIMIT_MB=$(( TOTAL_KB * 80 / 100 / 1024 ))

{
    echo "========================================"
    echo "  Kilosort2 Docker Spike Sorting"
    echo "========================================"
    echo ""
    echo "-- Environment --"
    echo "Started:       $(date)"
    echo "Host:          $(hostname)"
    echo "Conda env:     ${CONDA_ENV}"
    echo "Python:        $(conda run -n "$CONDA_ENV" python --version 2>&1)"
    echo "SI version:    $(conda run -n "$CONDA_ENV" python -c 'import spikeinterface; print(spikeinterface.__version__)' 2>/dev/null)"
    echo "SpikeLab:      $(conda run -n "$CONDA_ENV" python -c 'import spikelab; print(spikelab.__version__)' 2>/dev/null)"
    echo "Docker image:  $(docker images --format '{{.Repository}}:{{.Tag}}  ({{.Size}})' | grep kilosort2.*py310 || echo 'not found — build with docker/Dockerfile.kilosort2')"
    echo ""
    echo "-- System Resources --"
    echo "CPU cores:     $(nproc)"
    echo "RAM total:     $(( TOTAL_KB / 1024 ))M"
    echo "RAM available: $(grep MemAvailable /proc/meminfo | awk '{printf "%dM", $2/1024}')"
    echo "Memory limit:  ${LIMIT_MB}M (80% of total, swap disabled)"
    echo "GPU:           $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo 'unknown')"
    DISK_AVAIL=$(df -h "$RESULTS_DIR" --output=avail | tail -1 | xargs)
    DISK_TOTAL=$(df -h "$RESULTS_DIR" --output=size | tail -1 | xargs)
    echo "Disk:          ${DISK_AVAIL} available / ${DISK_TOTAL} total"
    echo ""
    echo "-- Input --"
    echo "Script:        ${PYTHON_SCRIPT}"
    echo "Results dir:   ${RESULTS_DIR}"
    echo "Log file:      ${LOGFILE}"
    echo ""
    echo "========================================"
    echo ""

    START_SECONDS=$SECONDS

    systemd-run --user --scope \
        -p MemoryMax="${LIMIT_MB}M" \
        -p MemorySwapMax=0 \
        conda run --no-capture-output -n "$CONDA_ENV" python -u \
            "$PYTHON_SCRIPT"

    EXIT_CODE=$?

    ELAPSED=$(( SECONDS - START_SECONDS ))
    ELAPSED_MIN=$(( ELAPSED / 60 ))
    ELAPSED_SEC=$(( ELAPSED % 60 ))

    echo ""
    echo "========================================"
    echo "  Summary"
    echo "========================================"
    echo ""

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status:        COMPLETED SUCCESSFULLY"
        echo ""
        echo "-- Output Files --"
        ls -lh "$RESULTS_DIR"/*.pkl "$RESULTS_DIR"/*.npz 2>/dev/null | awk '{print $5, $9}' || echo "(none found)"
    elif [ $EXIT_CODE -eq 137 ]; then
        echo "Status:        KILLED (OOM)"
        echo "Reason:        Exceeded memory limit (${LIMIT_MB}M)"
        echo ""
        echo "Only the sorting process was killed. The system is unaffected."
        echo "Try: reduce n_jobs, use first_n_mins, or increase memory limit."
    else
        echo "Status:        FAILED (exit code $EXIT_CODE)"
    fi

    echo ""
    echo "-- Resources at Finish --"
    echo "RAM available: $(grep MemAvailable /proc/meminfo | awk '{printf "%dM", $2/1024}')"
    echo "GPU memory:    $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
    echo "Disk avail:    $(df -h "$RESULTS_DIR" --output=avail | tail -1 | xargs)"
    echo ""
    echo "Wall time:     ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo "Finished:      $(date)"
    echo "========================================"
} 2>&1 | tee "$LOGFILE"
