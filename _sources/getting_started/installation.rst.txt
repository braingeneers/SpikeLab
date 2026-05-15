============
Installation
============

Basic Install
-------------

Install SpikeLab from source using pip:

.. code-block:: bash

   pip install oc-spikelab

Optional Extras
---------------

SpikeLab keeps most dependencies optional so the core library stays lightweight.
Install only what you need:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Extra
     - Command
     - Includes
   * - ``io``
     - ``pip install oc-spikelab[io]``
     - pandas (KiloSort cluster info)
   * - ``ml``
     - ``pip install oc-spikelab[ml]``
     - scikit-learn, umap-learn, networkx, python-louvain
   * - ``neo``
     - ``pip install oc-spikelab[neo]``
     - neo, quantities, pynwb
   * - ``ibl``
     - ``pip install oc-spikelab[ibl]`` (+ ``pip install git+https://github.com/int-brain-lab/paper-brain-wide-map.git``)
     - ONE-api (query and load IBL Brain-Wide Map datasets); ``brainwidemap`` is not on PyPI and must be installed separately from its git repo
   * - ``numba``
     - ``pip install oc-spikelab[numba]``
     - numba (JIT-compiled analysis functions)
   * - ``s3``
     - ``pip install oc-spikelab[s3]``
     - boto3 (Amazon S3 access)
   * - ``gplvm``
     - ``pip install oc-spikelab[gplvm]``
     - poor-man-gplvm, jax, jaxlib, jaxopt, optax
   * - ``spike-sorting``
     - ``pip install oc-spikelab[spike-sorting]``
     - spikeinterface, natsort, six, pandas
   * - ``kilosort4``
     - ``pip install oc-spikelab[kilosort4]``
     - kilosort (PyTorch + CUDA must be installed separately)
   * - ``batch-jobs``
     - ``pip install oc-spikelab[batch-jobs]``
     - pydantic, PyYAML, Jinja2, kubernetes
   * - ``mcp``
     - ``pip install oc-spikelab[mcp]``
     - mcp (Model Context Protocol server)
   * - ``all``
     - ``pip install oc-spikelab[all]``
     - Everything above plus dev and docs dependencies

Development Install
-------------------

To install from source in editable mode (for contributing or local development):

.. code-block:: bash

   git clone https://github.com/braingeneers/spikelab.git
   cd spikelab/SpikeLab
   pip install -e ".[dev]"

This installs the package in development mode along with testing tools
(pytest, pytest-asyncio, black).

