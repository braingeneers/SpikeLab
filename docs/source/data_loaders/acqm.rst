ACQM Files
==========

load_spikedata_from_acqm
-------------------------

Load spike train data from ACQM files (NPZ format containing spike trains and metadata). Supports both local files and S3 URIs with automatic caching.

Parameters
^^^^^^^^^^

* **filepath** (``str``): Path to ACQM file (local or S3 URI). S3 URIs should start with ``s3://``.
* **cache_dir** (``str | None``): Directory for caching S3 downloads. If None, uses temporary directory. Default: None
* **s3_endpoint_url** (``str | None``): S3 endpoint URL. Default: ``'https://s3-west.nrp-nautilus.io'`` (Nautilus)
* **length_ms** (``float | None``): Recording duration in milliseconds. If None, inferred from last spike time.

Returns
^^^^^^^

* **SpikeData**: Spike trains in milliseconds with neuron attributes and metadata.

Raises
^^^^^^

* **ValueError**: If ACQM file structure is invalid or missing required fields.
* **ImportError**: If ``boto3`` is unavailable (only for S3 URIs).
* **FileNotFoundError**: If local file doesn't exist.

ACQM File Format
^^^^^^^^^^^^^^^^

ACQM files are NPZ archives with the following structure:

* ``train``: Dict mapping unit IDs → spike time arrays (in samples)
* ``neuron_data``: Dict mapping unit IDs → metadata dicts (cluster_id, channel, position, etc.)
* ``fs``: Sampling frequency in Hz
* ``config``: Optional recording configuration dict
* ``redundant_pairs``: Optional array of redundant unit pairs

Basic Usage
^^^^^^^^^^^

Loading from Local File
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from data_loaders import load_spikedata_from_acqm
   
   # Load from local file
   sd = load_spikedata_from_acqm("recording.acqm.zip")
   
   # Access data
   print(f"Neurons: {sd.N}")
   print(f"Duration: {sd.length} ms")
   print(f"Sampling rate: {sd.metadata['fs_Hz']} Hz")
   
   # Access neuron attributes
   if sd.neuron_attributes:
       df = sd.neuron_attributes.to_dataframe()
       print(df[['cluster_id', 'channel', 'position']])

S3 Support
^^^^^^^^^^

Loading from S3
~~~~~~~~~~~~~~~

.. code-block:: python

   # Load directly from S3 (automatically downloads and caches)
   sd = load_spikedata_from_acqm("s3://my-bucket/data/recording.acqm.zip")
   
   # With custom cache directory
   sd = load_spikedata_from_acqm(
       "s3://my-bucket/data/recording_acqm.zip",
       cache_dir="/path/to/cache"
   )
   
   # With custom S3 endpoint
   sd = load_spikedata_from_acqm(
       "s3://my-bucket/data/recording_acqm.zip",
       s3_endpoint_url="https://s3.amazonaws.com"
   )

Caching Behavior
~~~~~~~~~~~~~~~~

S3 files are automatically cached:

.. code-block:: python

   # First load - downloads the file
   sd = load_spikedata_from_acqm(
       "s3://my-bucket/data/recording_acqm.zip",
       cache_dir="/tmp/my_cache"
   )
   
   # Subsequent loads - uses cached file (instant!)
   sd2 = load_spikedata_from_acqm(
       "s3://my-bucket/data/recording_acqm.zip",
       cache_dir="/tmp/my_cache"
   )  # No download, reads from /tmp/my_cache

The cache directory structure::

   cache_dir/
     └── my-bucket/
         └── data/
             └── recording.acqm.zip

Installation for S3
^^^^^^^^^^^^^^^^^^^

S3 functionality requires boto3:

.. code-block:: bash

   pip install boto3

If boto3 is not installed:

* Local file loading still works
* S3 URIs will raise an informative ImportError

AWS Credentials
~~~~~~~~~~~~~~~

For S3 access, configure AWS credentials using one of:

1. **Environment variables:**

   .. code-block:: bash

      export AWS_ACCESS_KEY_ID=your_key
      export AWS_SECRET_ACCESS_KEY=your_secret

2. **AWS credentials file** (``~/.aws/credentials``):

   .. code-block:: ini

      [default]
      aws_access_key_id = your_key
      aws_secret_access_key = your_secret

3. **IAM role** (for EC2/ECS instances)

Advanced Examples
^^^^^^^^^^^^^^^^^

Complete Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from data_loaders import load_spikedata_from_acqm
   import numpy as np
   
   # Load from S3 with caching
   sd = load_spikedata_from_acqm(
       "s3://my-bucket/experiments/recording_001.acqm.zip",
       cache_dir="./data_cache"
   )
   
   # Compute firing rates
   sd.compute_firing_rates(unit='Hz')
   
   # Access neuron attributes
   if sd.neuron_attributes:
       df = sd.neuron_attributes.to_dataframe()
       print(f"\nChannels: {df['channel'].unique()}")
       print(f"Positions: {df['position'].unique()}")
   
   # Compute ISI statistics
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Filter high-quality neurons
   violations = sd.neuron_attributes.get_attribute('refractory_violations')
   good_neurons = np.where(violations == 0)[0]
   sd_clean = sd.subset(good_neurons)
   
   # Export to HDF5
   sd_clean.to_hdf5('processed_data.h5', style='ragged')

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from data_loaders import load_spikedata_from_acqm
   
   # List of S3 files
   s3_files = [
       "s3://bucket/exp1.acqm.zip",
       "s3://bucket/exp2.acqm.zip",
       "s3://bucket/exp3.acqm.zip"
   ]
   
   # Process all with shared cache
   results = []
   for s3_path in s3_files:
       sd = load_spikedata_from_acqm(
           s3_path,
           cache_dir="./batch_cache"
       )
       
       # Compute metrics
       sd.compute_firing_rates(unit='Hz')
       mean_rate = sd.neuron_attributes.get_attribute('firing_rate_hz').mean()
       
       results.append({
           'file': s3_path,
           'n_neurons': sd.N,
           'duration_s': sd.length / 1000,
           'mean_firing_rate': mean_rate
       })
   
   import pandas as pd
   results_df = pd.DataFrame(results)
   print(results_df)

Metadata Access
^^^^^^^^^^^^^^^

ACQM files contain rich metadata:

.. code-block:: python

   sd = load_spikedata_from_acqm("recording_acqm.zip")
   
   # Sampling frequency
   fs_Hz = sd.metadata['fs_Hz']
   print(f"Sampling rate: {fs_Hz} Hz")
   
   # Recording configuration (if present)
   if 'config' in sd.metadata:
       config = sd.metadata['config']
       print(f"Config: {config}")
   
   # Redundant pairs (if present)
   if 'redundant_pairs' in sd.metadata:
       pairs = sd.metadata['redundant_pairs']
       print(f"Redundant pairs: {pairs}")
   
   # Source file path
   source = sd.metadata['source_file']
   print(f"Loaded from: {source}")

Neuron Attributes
^^^^^^^^^^^^^^^^^

ACQM files typically include per-neuron metadata:

.. code-block:: python

   sd = load_spikedata_from_acqm("recording.acqm.zip")
   
   if sd.neuron_attributes:
       df = sd.neuron_attributes.to_dataframe()
       
       # Common attributes:
       # - cluster_id: Cluster assignment
       # - channel: Recording channel
       # - position: Electrode position or neuron location
       # - quality: Quality metric from spike sorting
       
       # Filter by channel
       channel_10_neurons = sd.subset([10], by='channel')
       
       # Group by cluster
       for cluster_id in df['cluster_id'].unique():
           cluster_neurons = sd.subset([cluster_id], by='cluster_id')
           print(f"Cluster {cluster_id}: {cluster_neurons.N} neurons")


Best Practices
^^^^^^^^^^^^^^

1. **Use caching for S3**: Always specify ``cache_dir`` for S3 files to avoid repeated downloads

2. **Check metadata**: Verify sampling frequency and configuration after loading

3. **Validate attributes**: Check that neuron attributes exist and contain expected fields

4. **Handle missing data**: Some ACQM files may not include all optional fields (config, redundant_pairs)

Example:

.. code-block:: python

   sd = load_spikedata_from_acqm("recording.acqm.zip")
   
   # Validate
   assert 'fs_Hz' in sd.metadata, "Missing sampling frequency"
   assert sd.N > 0, "No neurons found"
   
   if sd.neuron_attributes:
       df = sd.neuron_attributes.to_dataframe()
       assert 'cluster_id' in df.columns, "Missing cluster IDs"

Downloading Files Manually
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For advanced use cases, you can manually download S3 files:

download_s3_to_local
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from data_loaders import download_s3_to_local
   
   # Download a file from S3
   download_s3_to_local(
       's3://my-bucket/data/recording.acqm.zip',
       '/tmp/recording.acqm.zip',
       endpoint_url='https://s3-west.nrp-nautilus.io'
   )
   
   # Then load the local file
   sd = load_spikedata_from_acqm('/tmp/recording.acqm.zip')

Parameters:

* ``src`` (str): S3 URI (e.g., ``s3://bucket/key/file.ext``)
* ``dst`` (str): Local destination path
* ``endpoint_url`` (str): S3 endpoint URL (default: Nautilus)
* ``**s3_client_kwargs``: Additional boto3 client arguments

Troubleshooting
^^^^^^^^^^^^^^^

**ImportError: boto3 required for S3**

Install boto3:

.. code-block:: bash

   pip install boto3

**FileNotFoundError**

Check that the file path is correct and the file exists.

**Access Denied (S3)**

Verify AWS credentials are configured and have read access to the bucket.

**Invalid ACQM structure**

Ensure the NPZ file contains required fields: ``train``, ``neuron_data``, ``fs``.

**Cache growing large**

Manually clean cache directory:

.. code-block:: bash

   rm -rf /path/to/cache

Or use a temporary directory that clears automatically:

.. code-block:: python

   import tempfile
   with tempfile.TemporaryDirectory() as tmpdir:
       sd = load_spikedata_from_acqm(s3_uri, cache_dir=tmpdir)
       # Use sd...
   # tmpdir deleted automatically

