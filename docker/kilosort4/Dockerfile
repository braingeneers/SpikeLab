FROM spikeinterface/kilosort4-base:latest

# Upgrade PyTorch to a cu126 build so the image works on hosts with
# NVIDIA driver >= 550 (cu126 hosts) and >= 560 (cu130 hosts) — cu126
# wheels include sm_120 (Blackwell consumer, e.g. RTX 5090) and run
# natively on cu130 drivers via NVIDIA forward compatibility.
# Must uninstall first since the base image pins cu118 packages.
RUN pip uninstall -y torch torchvision torchaudio \
    nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cuda-cupti-cu11 \
    nvidia-cudnn-cu11 nvidia-cublas-cu11 nvidia-cufft-cu11 nvidia-curand-cu11 \
    nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-nccl-cu11 nvidia-nvtx-cu11 2>/dev/null; \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Pre-install SI 0.104.0 and dependencies
RUN pip install --no-cache-dir \
    "spikeinterface[full]==0.104.0" \
    kilosort \
    neo \
    h5py

# Verify
RUN python -c "\
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); \
import spikeinterface; print('SI:', spikeinterface.__version__); \
import kilosort; print('KS4:', kilosort.__version__)" && \
    python --version
