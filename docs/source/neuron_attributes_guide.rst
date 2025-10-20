Neuron Attributes Guide
=======================

Overview
--------

The ``NeuronAttributes`` class provides a DataFrame-based system for managing neuron-level metadata in SpikeData objects. This allows you to store and manipulate information about individual neurons such as cluster IDs, electrode positions, firing rates, and quality metrics.

Standard Column Names
---------------------

While you can use any column names, the following standard names are recommended for consistency:

**Core Attributes:**

* ``unit_id``: Unique identifier for each neuron
* ``cluster_id``: Cluster assignment (multiple neurons can share a cluster_id)
* ``electrode_id``: Physical electrode identifier
* ``channel``: Recording channel number
* ``firing_rate_hz``: Mean firing rate in Hz

**Quality Metrics:**

* ``snr``: Signal-to-noise ratio
* ``amplitude``: Spike amplitude (μV or arbitrary units)
* ``isolation_distance``: Cluster isolation quality metric
* ``isi_violations``: Inter-spike interval violation rate

**Spatial Location:**

* ``unit_x``, ``unit_y``, ``unit_z``: Unit position coordinates (μm)
* ``electrode_x``, ``electrode_y``, ``electrode_z``: Electrode coordinates (μm)

Note: Store spatial coordinates as separate x, y, z columns. Combine using ``np.column_stack()`` when needed.

Creating SpikeData with Neuron Attributes
------------------------------------------

From a Dictionary
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spikedata import SpikeData
    
    # Create spike trains
    trains = [...]  # Your spike time arrays
    
    # Create with neuron attributes from dict
    sd = SpikeData(
        trains,
        neuron_attributes={
            'unit_id': [101, 102, 103],
            'cluster_id': [1, 1, 2],
            'electrode_id': [0, 1, 1],
            'firing_rate_hz': [5.2, 8.1, 3.4],
            'snr': [4.2, 6.8, 3.1]
        }
    )

From a DataFrame
~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from spikedata import SpikeData
    
    # Create a DataFrame with neuron metadata
    attrs_df = pd.DataFrame({
        'unit_id': [101, 102, 103],
        'cluster_id': [1, 1, 2],
        'quality': ['good', 'good', 'mua']
    })
    
    sd = SpikeData(trains, neuron_attributes=attrs_df)

Using NeuronAttributes Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spikedata import SpikeData, NeuronAttributes
    
    attrs = NeuronAttributes.from_dict({
        'unit_id': [1, 2, 3],
        'cluster_id': [10, 20, 30]
    }, n_neurons=3)
    
    sd = SpikeData(trains, neuron_attributes=attrs)

Working with Neuron Attributes
-------------------------------

Setting Attributes
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set a new attribute
    sd.set_neuron_attribute('quality', ['good', 'good', 'mua'])
    
    # Set quality metrics
    sd.set_neuron_attribute('snr', [4.2, 5.1, 3.8])

Getting Attributes
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get a specific attribute as numpy array
    cluster_ids = sd.get_neuron_attribute('cluster_id')
    
    # Access the full DataFrame
    df = sd.neuron_attributes.to_dataframe()
    print(df)

Computing and Storing Firing Rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compute firing rates and store in neuron_attributes
    rates = sd.compute_firing_rates(unit='Hz')
    
    # Access stored rates later
    rates = sd.get_neuron_attribute('firing_rate_hz')

Working with Spatial Locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Store as separate x, y, z coordinates (recommended)
    sd.set_neuron_attribute('unit_x', [100.5, 150.2, 200.8])
    sd.set_neuron_attribute('unit_y', [50.3, 75.1, 100.0])
    sd.set_neuron_attribute('unit_z', [0.0, 0.0, 50.0])
    
    # Access spatial data
    x_coords = sd.get_neuron_attribute('unit_x')
    
    # Combine coordinates for analysis
    import numpy as np
    df = sd.neuron_attributes.to_dataframe()
    positions = np.column_stack([df['unit_x'], df['unit_y'], df['unit_z']])
    # positions is now a (N, 3) array of coordinates
    
    # Same approach for electrode locations
    sd.set_neuron_attribute('electrode_x', [100.0, 150.0, 200.0])
    sd.set_neuron_attribute('electrode_y', [50.0, 75.0, 100.0])
    sd.set_neuron_attribute('electrode_z', [0.0, 0.0, 50.0])

Operations with Neuron Attributes
----------------------------------

Subsetting
~~~~~~~~~~

When you subset neurons, the attributes are automatically subsetted:

.. code-block:: python

    # Subset by indices
    sd_subset = sd.subset([0, 2, 4])
    
    # Subset by attribute value
    good_neurons = sd.subset(['good'], by='quality')
    
    # The neuron_attributes are automatically filtered
    print(sd_subset.neuron_attributes.to_dataframe())

Concatenation
~~~~~~~~~~~~~

When concatenating SpikeData objects, attributes are concatenated:

.. code-block:: python

    # Time concatenation (append) - keeps same neurons
    sd_combined = sd1.append(sd2)
    
    # Neuron concatenation - adds new neurons
    sd1.concatenate_spike_data(sd2)
    # Attributes from both are combined

Loading from Files
------------------

Neuron attributes are automatically extracted when loading from various formats:

From NWB Files
~~~~~~~~~~~~~~

.. code-block:: python

    from data_loaders import load_spikedata_from_nwb
    
    sd = load_spikedata_from_nwb('recording.nwb')
    
    # All columns from the Units table are loaded
    if sd.neuron_attributes is not None:
        print(sd.neuron_attributes.to_dataframe())

From KiloSort
~~~~~~~~~~~~~

.. code-block:: python

    from data_loaders import load_spikedata_from_kilosort
    
    sd = load_spikedata_from_kilosort(
        'kilosort_output/',
        fs_Hz=30000,
        cluster_info_tsv='cluster_info.tsv'
    )
    
    # Cluster info is automatically loaded into neuron_attributes
    cluster_ids = sd.get_neuron_attribute('cluster_id')

From HDF5
~~~~~~~~~

.. code-block:: python

    from data_loaders import load_spikedata_from_hdf5
    
    sd = load_spikedata_from_hdf5('data.h5', ...)
    
    # Attributes are loaded from /neuron_attributes group if present

From SpikeInterface
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from data_loaders import load_spikedata_from_spikeinterface
    
    # Load from a SpikeInterface sorting object
    sd = load_spikedata_from_spikeinterface(sorting)
    
    # All unit properties are automatically extracted

Exporting with Neuron Attributes
---------------------------------

Neuron attributes are automatically saved when exporting:

To HDF5
~~~~~~~

.. code-block:: python

    # Attributes saved to /neuron_attributes group
    sd.to_hdf5('output.h5', style='ragged')

To NWB
~~~~~~

.. code-block:: python

    # Attributes saved as additional columns in Units table
    sd.to_nwb('output.nwb')

To KiloSort
~~~~~~~~~~~

.. code-block:: python

    # Attributes saved as cluster_info.tsv
    sd.to_kilosort('kilosort_output/', fs_Hz=30000)

Example Workflow
----------------

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

    from spikedata import SpikeData
    from data_loaders import load_spikedata_from_kilosort
    
    # Load data with attributes
    sd = load_spikedata_from_kilosort(
        'kilosort_data/',
        fs_Hz=30000,
        cluster_info_tsv='cluster_info.tsv'
    )
    
    # Compute and store firing rates
    sd.compute_firing_rates(unit='Hz')
    
    # Add custom quality assessment
    snr_values = [...]  # Your SNR computation
    sd.set_neuron_attribute('snr', snr_values)
    
    # Filter to high-quality neurons
    high_snr = sd.neuron_attributes.get_attribute('snr') > 5.0
    good_indices = [i for i, keep in enumerate(high_snr) if keep]
    sd_filtered = sd.subset(good_indices)
    
    # Export with all attributes preserved
    sd_filtered.to_hdf5('filtered_data.h5', style='ragged')
    
    # Verify round-trip
    from data_loaders import load_spikedata_from_hdf5
    sd_loaded = load_spikedata_from_hdf5(
        'filtered_data.h5',
        spike_times_dataset='spike_times',
        spike_times_index_dataset='spike_times_index'
    )
    
    print(sd_loaded.neuron_attributes.to_dataframe())

Spatial Filtering Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spikedata import SpikeData
    import numpy as np
    
    # Create SpikeData with spatial information
    sd = SpikeData(
        trains,
        neuron_attributes={
            'unit_id': [1, 2, 3, 4, 5],
            'unit_x': [100.0, 150.0, 200.0, 250.0, 300.0],
            'unit_y': [50.0, 75.0, 100.0, 125.0, 150.0],
            'unit_z': [0.0, 10.0, 20.0, 30.0, 40.0],
            'electrode_id': [1, 1, 2, 2, 3]
        }
    )
    
    # Filter neurons within a spatial region
    df = sd.neuron_attributes.to_dataframe()
    
    # Find neurons in a specific X range (100-200 μm)
    in_region = (df['unit_x'] >= 100) & (df['unit_x'] <= 200)
    region_indices = df[in_region].index.tolist()
    sd_region = sd.subset(region_indices)
    
    # Find neurons near a specific electrode
    electrode_neurons = sd.subset([1], by='electrode_id')
    
    # Calculate distances from a reference point
    ref_point = np.array([150.0, 75.0, 10.0])
    positions = np.column_stack([
        df['unit_x'].values,
        df['unit_y'].values,
        df['unit_z'].values
    ])
    distances = np.linalg.norm(positions - ref_point, axis=1)
    
    # Store distances as an attribute
    sd.set_neuron_attribute('distance_from_ref', distances)
    
    # Filter to neurons within 100 μm of reference
    nearby = distances < 100.0
    nearby_indices = [i for i, keep in enumerate(nearby) if keep]
    sd_nearby = sd.subset(nearby_indices)

Advanced Usage
--------------

Direct DataFrame Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get the DataFrame for advanced operations
    df = sd.neuron_attributes.to_dataframe()
    
    # Perform pandas operations
    df['log_firing_rate'] = np.log10(df['firing_rate_hz'] + 1)
    
    # Create new NeuronAttributes from modified DataFrame
    from spikedata import NeuronAttributes
    sd.neuron_attributes = NeuronAttributes.from_dataframe(df, n_neurons=sd.N)

Validation and Warnings
~~~~~~~~~~~~~~~~~~~~~~~~

The system validates column names and warns about potential typos:

.. code-block:: python

    # This will trigger a warning about case mismatch
    sd.set_neuron_attribute('Firing_Rate_Hz', rates)  # Should be 'firing_rate_hz'

Migration from Old System
-------------------------

If you were using the old list-based ``neuron_attributes`` system, here's how to migrate:

Old Way (Removed)
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # OLD: List of dataclass objects
    from dataclasses import dataclass
    
    @dataclass
    class NeuronInfo:
        unit_id: int
        cluster_id: int
    
    neuron_attrs = [
        NeuronInfo(unit_id=1, cluster_id=10),
        NeuronInfo(unit_id=2, cluster_id=20),
    ]
    
    sd = SpikeData(trains, neuron_attributes=neuron_attrs)
    
    # Access
    unit_id = neuron_attrs[0].unit_id

New Way (Current)
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # NEW: Dictionary or DataFrame
    sd = SpikeData(
        trains,
        neuron_attributes={
            'unit_id': [1, 2],
            'cluster_id': [10, 20]
        }
    )
    
    # Access
    unit_ids = sd.get_neuron_attribute('unit_id')
    unit_id = unit_ids[0]
    
    # Or access as DataFrame
    df = sd.neuron_attributes.to_dataframe()
    unit_id = df.loc[0, 'unit_id']

Best Practices
--------------

1. **Use Standard Column Names**: Stick to the recommended standard column names for consistency across projects.

2. **Store Computed Metrics**: Use ``neuron_attributes`` to store computed quality metrics for future reference.

3. **Document Custom Columns**: If using custom column names, document them in your project.

4. **Validate Before Export**: Check that important attributes are present before exporting.

5. **Use Proper Units**: Store firing rates in Hz, amplitudes in μV, times in ms, etc.

6. **Round-trip Testing**: Verify that attributes survive load/save cycles for your file format.


