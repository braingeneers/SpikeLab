FROM spikeinterface/kilosort2-compiled-base:latest

# Upgrade conda env to Python 3.10 and pre-install SI 0.104.0
# The compiled MATLAB binary (ks2_compiled) is unaffected by Python version.
RUN /home/miniconda3/bin/conda install -y python=3.10 -n base && \
    /home/miniconda3/bin/conda clean -afy

RUN /home/miniconda3/bin/pip install --no-cache-dir \
    "spikeinterface[full]==0.104.0" \
    neo \
    h5py

# Verify
RUN python -c "import spikeinterface; print('SI', spikeinterface.__version__)" && \
    python --version
