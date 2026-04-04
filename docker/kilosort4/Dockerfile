FROM spikeinterface/kilosort4-base:latest

# Force upgrade PyTorch to 2.9+ with CUDA 13.0 for RTX 5090 (compute capability 12.0)
# Must uninstall first since the base image pins cu118 packages
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
