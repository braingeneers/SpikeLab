Advanced Usage
==============

This section covers advanced techniques for working with neuron attributes.

Direct DataFrame Manipulation
------------------------------

For complex operations, you can work directly with the underlying DataFrame:

.. code-block:: python

   # Get the DataFrame for advanced operations
   df = sd.neuron_attributes.to_dataframe()
   
   # Perform pandas operations
   df['log_firing_rate'] = np.log10(df['firing_rate_hz'] + 1)
   
   # Complex computed columns
   df['burst_score'] = df['burst_index'] * (1.0 / (df['cv_isi'] + 0.1))
   
   # Create new NeuronAttributes from modified DataFrame
   from spikedata import NeuronAttributes
   sd.neuron_attributes = NeuronAttributes.from_dataframe(df, n_neurons=sd.N)

Conditional Attribute Setting
------------------------------

Set attributes conditionally based on existing values:

.. code-block:: python

   import numpy as np
   
   df = sd.neuron_attributes.to_dataframe()
   
   # Create categorical attribute from continuous
   quality = np.full(sd.N, 'poor', dtype=object)
   quality[df['snr'] > 3.0] = 'fair'
   quality[df['snr'] > 5.0] = 'good'
   quality[df['snr'] > 10.0] = 'excellent'
   
   sd.set_neuron_attribute('quality_category', quality)
   
   # Combine multiple criteria
   is_high_quality = (
       (df['refractory_violations'] == 0) &
       (df['snr'] > 5.0) &
       (df['isolation_distance'] > 20)
   )
   sd.set_neuron_attribute('is_high_quality', is_high_quality.values)

Merging Attributes from Multiple Sources
-----------------------------------------

Combine attributes from different analyses:

.. code-block:: python

   import pandas as pd
   
   # Get current attributes
   df1 = sd.neuron_attributes.to_dataframe()
   
   # Create new attributes from external source
   external_attrs = pd.DataFrame({
       'morphology_type': ['pyramidal', 'interneuron', 'pyramidal'],
       'layer': [2, 3, 2]
   }, index=df1.index)
   
   # Merge
   df_merged = pd.concat([df1, external_attrs], axis=1)
   
   # Update neuron_attributes
   from spikedata import NeuronAttributes
   sd.neuron_attributes = NeuronAttributes.from_dataframe(df_merged, n_neurons=sd.N)

Validation and Warnings
------------------------

The system validates column names and warns about potential typos:

.. code-block:: python

   # This will trigger a warning about case mismatch
   sd.set_neuron_attribute('Firing_Rate_Hz', rates)  # Should be 'firing_rate_hz'
   
   # Check for typos programmatically
   from spikedata.neuron_attributes import NeuronAttributes
   
   valid_columns = NeuronAttributes.STANDARD_COLUMNS
   df = sd.neuron_attributes.to_dataframe()
   
   for col in df.columns:
       if col not in valid_columns:
           print(f"Warning: '{col}' is not a standard column name")

Custom Validation
-----------------

Add your own validation logic:

.. code-block:: python

   def validate_attributes(sd):
       """Validate neuron attributes for consistency."""
       
       if sd.neuron_attributes is None:
           raise ValueError("No neuron attributes found")
       
       df = sd.neuron_attributes.to_dataframe()
       
       # Check required columns
       required = ['unit_id', 'electrode_id']
       missing = [col for col in required if col not in df.columns]
       if missing:
           raise ValueError(f"Missing required columns: {missing}")
       
       # Check firing rates are positive
       if 'firing_rate_hz' in df.columns:
           if (df['firing_rate_hz'] < 0).any():
               raise ValueError("Negative firing rates found")
       
       # Check spatial coordinates match
       spatial_cols = ['unit_x', 'unit_y', 'unit_z']
       spatial_present = [col in df.columns for col in spatial_cols]
       if any(spatial_present) and not all(spatial_present):
           raise ValueError("Incomplete spatial coordinates (need all of x, y, z)")
       
       return True
   
   # Use validation
   validate_attributes(sd)

Handling Missing Values
-----------------------

Work with missing or incomplete attribute data:

.. code-block:: python

   import numpy as np
   import pandas as pd
   
   df = sd.neuron_attributes.to_dataframe()
   
   # Check for missing values
   print(df.isna().sum())
   
   # Fill missing values
   df['snr'].fillna(df['snr'].median(), inplace=True)
   
   # Filter to complete cases
   complete = df.dropna()
   sd_complete = sd.subset(complete.index.tolist())
   
   # Or filter to neurons with specific attributes present
   has_spatial = df[['unit_x', 'unit_y', 'unit_z']].notna().all(axis=1)
   sd_spatial = sd.subset(has_spatial[has_spatial].index.tolist())

Performance Optimization
-------------------------

For large datasets, optimize attribute operations:

.. code-block:: python

   # Use vectorized operations instead of loops
   df = sd.neuron_attributes.to_dataframe()
   
   # SLOW: Loop over neurons
   # distances = [compute_distance(x, y, z) for x, y, z in zip(...)]
   
   # FAST: Vectorized
   import numpy as np
   positions = np.column_stack([df['unit_x'], df['unit_y'], df['unit_z']])
   ref_point = np.array([0, 0, 0])
   distances = np.linalg.norm(positions - ref_point, axis=1)
   
   # For very large attribute sets, work with views
   # (only if you know what you're doing)
   df_view = df[['unit_id', 'firing_rate_hz']]  # Smaller working set

Migration from Old System
--------------------------

If you were using the old list-based ``neuron_attributes`` system, here's how to migrate:

Old Way (Removed)
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # OLD: List of dataclass objects
   from dataclasses import dataclass
   
   @dataclass
   class NeuronInfo:
       unit_id: int
       electrode_id: int
   
   neuron_attrs = [
       NeuronInfo(unit_id=1, electrode_id=10),
       NeuronInfo(unit_id=2, electrode_id=20),
   ]
   
   sd = SpikeData(trains, neuron_attributes=neuron_attrs)
   
   # Access
   unit_id = neuron_attrs[0].unit_id

New Way (Current)
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # NEW: Dictionary or DataFrame
   sd = SpikeData(
       trains,
       neuron_attributes={
           'unit_id': [1, 2],
           'electrode_id': [10, 20]
       }
   )
   
   # Access
   unit_ids = sd.get_neuron_attribute('unit_id')
   unit_id = unit_ids[0]
   
   # Or access as DataFrame
   df = sd.neuron_attributes.to_dataframe()
   unit_id = df.loc[0, 'unit_id']

Migration Helper
^^^^^^^^^^^^^^^^

Convert old-style attribute lists:

.. code-block:: python

   def migrate_old_attributes(old_attrs):
       """Convert old list-of-objects to new dict format."""
       
       if not old_attrs:
           return None
       
       # Get all field names from first object
       if hasattr(old_attrs[0], '__dataclass_fields__'):
           fields = old_attrs[0].__dataclass_fields__.keys()
       else:
           fields = vars(old_attrs[0]).keys()
       
       # Build dict
       new_attrs = {}
       for field in fields:
           new_attrs[field] = [getattr(obj, field) for obj in old_attrs]
       
       return new_attrs
   
   # Use it
   old_attrs = [...]  # Your old attribute objects
   new_attrs = migrate_old_attributes(old_attrs)
   sd = SpikeData(trains, neuron_attributes=new_attrs)

Best Practices
--------------

1. **Use Standard Column Names**

   Stick to the recommended standard column names for consistency across projects and easier collaboration.

2. **Store Computed Metrics**

   Use ``neuron_attributes`` to store computed quality metrics, analysis results, and classifications for future reference.

3. **Document Custom Columns**

   If using custom column names, document their meaning and units in your project README or analysis notebooks.

4. **Validate Before Export**

   Check that important attributes are present and valid before exporting to ensure downstream analyses have needed data.

5. **Use Proper Units**

   - Firing rates in Hz
   - Amplitudes in μV
   - Times in ms
   - Distances in μm
   - Document if using different units

6. **Round-trip Testing**

   Verify that attributes survive load/save cycles for your file format:

   .. code-block:: python

      # Save
      sd.to_hdf5('test.h5', style='ragged')
      
      # Load
      from data_loaders import load_spikedata_from_hdf5
      sd_loaded = load_spikedata_from_hdf5(
          'test.h5',
          spike_times_dataset='spike_times',
          spike_times_index_dataset='spike_times_index'
      )
      
      # Verify
      assert sd_loaded.neuron_attributes is not None
      df_orig = sd.neuron_attributes.to_dataframe()
      df_loaded = sd_loaded.neuron_attributes.to_dataframe()
      pd.testing.assert_frame_equal(df_orig, df_loaded)

7. **Clear Caches Appropriately**

   When using analysis caching methods, clear caches after subsetting or before exporting:

   .. code-block:: python

      # After subsetting
      sd_subset = sd.subset([0, 1, 2])
      sd_subset.clear_sttc_cache()  # Old cache no longer valid
      
      # Before export to reduce file size
      sd.clear_sttc_cache()
      sd.to_hdf5('output.h5', style='ragged')

8. **Use auto_save Wisely**

   The ``auto_save=True`` default is convenient but increases memory usage. For exploratory analysis or temporary calculations, consider ``auto_save=False``:

   .. code-block:: python

      # For permanent storage
      isi_stats = sd.neuron_attributes.compute_isi_statistics(sd, auto_save=True)
      
      # For temporary exploration
      isi_stats = sd.neuron_attributes.compute_isi_statistics(sd, auto_save=False)
      # Use isi_stats['cv_isi'] directly without storing

