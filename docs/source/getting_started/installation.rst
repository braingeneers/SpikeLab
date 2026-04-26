============
Installation
============

Basic Install
-------------

Install SpikeLab from source using pip:

.. code-block:: bash

   pip install spikelab

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
     - ``pip install spikelab[io]``
     - pandas (KiloSort cluster info)
   * - ``ml``
     - ``pip install spikelab[ml]``
     - scikit-learn, umap-learn, networkx, python-louvain
   * - ``neo``
     - ``pip install spikelab[neo]``
     - neo, quantities, pynwb
   * - ``ibl``
     - ``pip install spikelab[ibl]`` (+ ``pip install git+https://github.com/int-brain-lab/paper-brain-wide-map.git``)
     - ONE-api (query and load IBL Brain-Wide Map datasets); ``brainwidemap`` is not on PyPI and must be installed separately from its git repo
   * - ``numba``
     - ``pip install spikelab[numba]``
     - numba (JIT-compiled analysis functions)
   * - ``s3``
     - ``pip install spikelab[s3]``
     - boto3 (Amazon S3 access)
   * - ``gplvm``
     - ``pip install spikelab[gplvm]``
     - poor-man-gplvm, jax, jaxlib, jaxopt, optax
   * - ``spike-sorting``
     - ``pip install spikelab[spike-sorting]``
     - spikeinterface, natsort, six, pandas
   * - ``kilosort4``
     - ``pip install spikelab[kilosort4]``
     - kilosort (PyTorch + CUDA must be installed separately)
   * - ``batch-jobs``
     - ``pip install spikelab[batch-jobs]``
     - pydantic, PyYAML, Jinja2, kubernetes
   * - ``mcp``
     - ``pip install spikelab[mcp]``
     - mcp (Model Context Protocol server)
   * - ``all``
     - ``pip install spikelab[all]``
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

