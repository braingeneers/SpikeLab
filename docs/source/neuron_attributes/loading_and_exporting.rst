Loading and Exporting
=====================

Neuron attributes are automatically extracted when loading from various formats and preserved when exporting.

Loading from Files
------------------

Neuron attributes are automatically extracted when loading from supported file formats:

From NWB Files
^^^^^^^^^^^^^^

.. code-block:: python

   from data_loaders import load_spikedata_from_nwb
   
   sd = load_spikedata_from_nwb('recording.nwb')
   
   # All columns from the Units table are loaded
   if sd.neuron_attributes is not None:
       print(sd.neuron_attributes.to_dataframe())
       print(sd.neuron_attributes.to_dataframe().columns)

NWB files store neuron metadata in the Units table. All columns are automatically loaded as neuron attributes.

From KiloSort
^^^^^^^^^^^^^

.. code-block:: python

   from data_loaders import load_spikedata_from_kilosort
   
   sd = load_spikedata_from_kilosort(
       'kilosort_output/',
       fs_Hz=30000,
       cluster_info_tsv='cluster_info.tsv'
   )
   
   # Cluster info is automatically loaded into neuron_attributes
   cluster_ids = sd.get_neuron_attribute('cluster_id')

If a ``cluster_info.tsv`` file is present, all columns are loaded as neuron attributes.

From HDF5
^^^^^^^^^

.. code-block:: python

   from data_loaders import load_spikedata_from_hdf5
   
   sd = load_spikedata_from_hdf5('data.h5', ...)
   
   # Attributes are loaded from /neuron_attributes group if present
   if sd.neuron_attributes is not None:
       df = sd.neuron_attributes.to_dataframe()

HDF5 files can store neuron attributes in a ``/neuron_attributes`` group. Each dataset in the group becomes a column.

From ACQM Files
^^^^^^^^^^^^^^^

.. code-block:: python

   from data_loaders import load_spikedata_from_acqm
   
   # Load from local file
   sd = load_spikedata_from_acqm("recording.acqm.zip")
   
   # Or load from S3
   sd = load_spikedata_from_acqm("s3://bucket/recording.acqm.zip")
   
   # ACQM files include neuron metadata
   if sd.neuron_attributes is not None:
       df = sd.neuron_attributes.to_dataframe()
       print(df[['cluster_id', 'channel', 'position']])

ACQM files (NPZ format) contain a ``neuron_data`` dictionary with cluster IDs, channels, positions, and other metadata that are automatically extracted.

From SpikeInterface
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from data_loaders import load_spikedata_from_spikeinterface
   
   # Load from a SpikeInterface sorting object
   sd = load_spikedata_from_spikeinterface(sorting)
   
   # All unit properties are automatically extracted
   if sd.neuron_attributes is not None:
       df = sd.neuron_attributes.to_dataframe()

SpikeInterface unit properties are automatically converted to neuron attributes.

Exporting with Neuron Attributes
---------------------------------

Neuron attributes are automatically saved when exporting to supported formats:

To HDF5
^^^^^^^

.. code-block:: python

   # Attributes saved to /neuron_attributes group
   sd.to_hdf5('output.h5', style='ragged')
   
   # Each attribute becomes a dataset
   # Can be loaded back with load_spikedata_from_hdf5()

The neuron_attributes DataFrame is saved to an HDF5 group with each column as a separate dataset.

To NWB
^^^^^^

.. code-block:: python

   # Attributes saved as additional columns in Units table
   sd.to_nwb('output.nwb')
   
   # Standard NWB columns (spike_times) plus all custom attributes

All neuron attributes are added as columns to the NWB Units table alongside spike times.

To KiloSort
^^^^^^^^^^^

.. code-block:: python

   # Attributes saved as cluster_info.tsv
   sd.to_kilosort('kilosort_output/', fs_Hz=30000)
   
   # Creates spike_times.npy, spike_clusters.npy, and cluster_info.tsv

Neuron attributes are saved as a tab-separated file (``cluster_info.tsv``) that can be loaded by Phy and other tools.

Round-Trip Examples
-------------------

HDF5 Round-Trip
^^^^^^^^^^^^^^^

.. code-block:: python

   from data_loaders import load_spikedata_from_hdf5
   
   # Add some attributes
   sd.set_neuron_attribute('quality', ['good', 'mua', 'good'])
   sd.compute_firing_rates(unit='Hz')
   sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Export
   sd.to_hdf5('test.h5', style='ragged')
   
   # Load back
   sd_loaded = load_spikedata_from_hdf5(
       'test.h5',
       spike_times_dataset='spike_times',
       spike_times_index_dataset='spike_times_index'
   )
   
   # All attributes preserved
   print(sd_loaded.neuron_attributes.to_dataframe())

NWB Round-Trip
^^^^^^^^^^^^^^

.. code-block:: python

   from data_loaders import load_spikedata_from_nwb
   
   # Add attributes
   sd.set_neuron_attribute('cluster_id', [1, 2, 3])
   sd.set_neuron_attribute('quality', ['good', 'good', 'mua'])
   
   # Export
   sd.to_nwb('test.nwb')
   
   # Load back
   sd_loaded = load_spikedata_from_nwb('test.nwb')
   
   # All attributes in Units table
   print(sd_loaded.neuron_attributes.to_dataframe())

Format Compatibility
--------------------

Some formats have limitations on what data types can be stored:

**HDF5:**
- Supports all numpy dtypes
- Arrays, strings, numbers all work
- Most flexible format

**NWB:**
- Supports most common types
- Arrays must be 1D per neuron
- Use ragged arrays for variable-length data

**KiloSort (TSV):**
- Text-based, so arrays become strings
- Best for simple attributes (IDs, quality labels, scalars)
- Not ideal for waveforms or large arrays

**Recommendation:** Use HDF5 for maximum flexibility, especially when storing computed metrics like waveforms or complex analyses.

Best Practices
--------------

1. **Compute Before Export**: Calculate all desired metrics before exporting so they're included in the saved file.

2. **Verify Round-Trip**: Test that your attributes survive a save/load cycle for your chosen format.

3. **Document Custom Columns**: Add metadata about custom attribute meanings in your project documentation.

4. **Use Standard Names**: Stick to standard column names when possible for interoperability.

5. **Check After Loading**: Always verify that neuron_attributes exist and contain expected columns after loading:

.. code-block:: python

   # Always check after loading
   if sd.neuron_attributes is None:
       print("Warning: No neuron attributes found")
   else:
       df = sd.neuron_attributes.to_dataframe()
       print(f"Loaded {len(df.columns)} attributes for {len(df)} neurons")
       print(f"Columns: {df.columns.tolist()}")

