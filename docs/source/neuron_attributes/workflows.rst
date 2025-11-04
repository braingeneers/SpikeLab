Example Workflows
=================

This page shows complete workflows using neuron attributes for common analysis tasks.

Complete Analysis Pipeline
---------------------------

.. code-block:: python

   from spikedata import SpikeData
   from data_loaders import load_spikedata_from_kilosort
   import numpy as np
   
   # 1. Load data with attributes
   sd = load_spikedata_from_kilosort(
       'kilosort_data/',
       fs_Hz=30000,
       cluster_info_tsv='cluster_info.tsv'
   )
   
   # 2. Compute and store firing rates
   sd.compute_firing_rates(unit='Hz')
   
   # 3. Compute ISI statistics for quality control
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # 4. Add custom quality assessment
   snr_values = compute_snr(sd)  # Your SNR computation
   sd.set_neuron_attribute('snr', snr_values)
   
   # 5. Filter to high-quality neurons
   df = sd.neuron_attributes.to_dataframe()
   high_quality = df[
       (df['refractory_violations'] == 0) &
       (df['snr'] > 5.0) &
       (df['firing_rate_hz'] > 1.0) &
       (df['firing_rate_hz'] < 50.0)
   ]
   good_indices = high_quality.index.tolist()
   sd_filtered = sd.subset(good_indices)
   
   # 6. Export with all attributes preserved
   sd_filtered.to_hdf5('filtered_data.h5', style='ragged')
   
   # 7. Verify round-trip
   from data_loaders import load_spikedata_from_hdf5
   sd_loaded = load_spikedata_from_hdf5(
       'filtered_data.h5',
       spike_times_dataset='spike_times',
       spike_times_index_dataset='spike_times_index'
   )
   
   print(sd_loaded.neuron_attributes.to_dataframe())

Spatial Analysis Workflow
--------------------------

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
   
   # Analyze distance-dependent properties
   df = sd.neuron_attributes.to_dataframe()
   sd.compute_firing_rates(unit='Hz')
   
   import matplotlib.pyplot as plt
   plt.scatter(df['distance_from_ref'], df['firing_rate_hz'])
   plt.xlabel('Distance from reference (μm)')
   plt.ylabel('Firing rate (Hz)')

Cell Classification Workflow
-----------------------------

.. code-block:: python

   from spikedata import SpikeData
   import numpy as np
   
   # Load data
   sd = ...  # Your SpikeData object
   
   # Compute multiple metrics
   sd.compute_firing_rates(unit='Hz')
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Access all attributes as DataFrame
   attrs = sd.neuron_attributes.to_dataframe()
   
   # Classify by firing pattern
   regular = attrs[attrs['cv_isi'] < 0.5].index.tolist()
   bursting = attrs[attrs['burst_index'] > 0.2].index.tolist()
   fast_spiking = attrs[attrs['firing_rate_hz'] > 20].index.tolist()
   
   # Store classifications
   cell_type = np.full(sd.N, 'other', dtype=object)
   cell_type[regular] = 'regular'
   cell_type[bursting] = 'bursting'
   cell_type[fast_spiking] = 'fast_spiking'
   sd.set_neuron_attribute('cell_type', cell_type)
   
   # Analyze each cell type
   for ctype in ['regular', 'bursting', 'fast_spiking', 'other']:
       sd_type = sd.subset([ctype], by='cell_type')
       print(f"{ctype}: {sd_type.N} neurons")
       
       # Type-specific analysis
       if sd_type.N > 0:
           rates = sd_type.binned_meanrate(bin_size=100.0)
           print(f"  Mean rate: {rates.mean():.2f} Hz")

Network Burst Analysis
----------------------

.. code-block:: python

   from spikedata import SpikeData
   import numpy as np
   
   # Load data
   sd = ...  # Your SpikeData object
   
   # 1. Detect network bursts (using your preferred method)
   burst_edges = detect_network_bursts(sd)  # Returns (n_bursts, 2) array
   
   # 2. Compute burst participation
   burst_stats = sd.neuron_attributes.compute_burst_participation(
       sd,
       burst_edges=burst_edges,
       min_spikes=5,
       backbone_threshold=0.6
   )
   
   # 3. Analyze backbone neurons
   backbone_indices = burst_stats['backbone_indices']
   sd_backbone = sd.subset(backbone_indices)
   
   print(f"Found {len(backbone_indices)} backbone neurons")
   
   # 4. Compute correlations among backbone units
   sttc = sd_backbone.get_sttc_matrix(delt=20.0)
   
   # 5. Find highly correlated pairs
   high_corr_pairs = np.where(sttc > 0.5)
   
   # 6. Combine with other metrics
   df = sd.neuron_attributes.to_dataframe()
   backbone_df = df.iloc[backbone_indices]
   
   print("Backbone neuron properties:")
   print(backbone_df[['firing_rate_hz', 'cv_isi', 'burst_participation']])

Functional Connectivity Analysis
---------------------------------

.. code-block:: python

   from spikedata import SpikeData
   import numpy as np
   
   # Load data
   sd = ...  # Your SpikeData object
   
   # 1. Compute ISI statistics to find regular firing neurons
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # 2. Find potential pacemaker (regular, high firing rate)
   sd.compute_firing_rates(unit='Hz')
   df = sd.neuron_attributes.to_dataframe()
   
   # Pacemaker criteria: low CV, high firing rate
   pacemaker_candidates = df[
       (df['cv_isi'] < 0.3) &
       (df['firing_rate_hz'] > 5.0)
   ]
   
   if len(pacemaker_candidates) > 0:
       # Choose highest firing rate regular neuron
       pacemaker_idx = pacemaker_candidates['firing_rate_hz'].idxmax()
       
       # 3. Compute latencies relative to pacemaker
       lat_stats = sd.neuron_attributes.compute_latency_statistics(
           sd,
           reference_neuron=pacemaker_idx,
           window_ms=100.0
       )
       
       # 4. Find followers (positive latency, low jitter)
       df = sd.neuron_attributes.to_dataframe()
       followers = df[
           (df['mean_latency_ms'] > 0) &
           (df['latency_jitter_ms'] < 5.0)
       ]
       
       print(f"Pacemaker: neuron {pacemaker_idx}")
       print(f"Followers: {len(followers)} neurons")
       print(followers[['mean_latency_ms', 'latency_jitter_ms']])
       
       # 5. Visualize latency distribution
       import matplotlib.pyplot as plt
       plt.hist(df['mean_latency_ms'].dropna(), bins=20)
       plt.xlabel('Latency to pacemaker (ms)')
       plt.ylabel('Count')
       plt.title(f'Latency distribution relative to neuron {pacemaker_idx}')

Multi-Session Comparison
-------------------------

.. code-block:: python

   from data_loaders import load_spikedata_from_hdf5
   import pandas as pd
   
   # Load multiple sessions
   sessions = ['session1.h5', 'session2.h5', 'session3.h5']
   
   all_attrs = []
   for session_file in sessions:
       sd = load_spikedata_from_hdf5(
           session_file,
           spike_times_dataset='spike_times',
           spike_times_index_dataset='spike_times_index'
       )
       
       # Compute metrics
       sd.compute_firing_rates(unit='Hz')
       sd.neuron_attributes.compute_isi_statistics(sd)
       
       # Get attributes
       df = sd.neuron_attributes.to_dataframe()
       df['session'] = session_file
       all_attrs.append(df)
   
   # Combine all sessions
   combined = pd.concat(all_attrs, ignore_index=True)
   
   # Compare across sessions
   print(combined.groupby('session')['firing_rate_hz'].mean())
   print(combined.groupby('session')['cv_isi'].mean())
   
   # Find stable properties
   import matplotlib.pyplot as plt
   combined.boxplot(column='firing_rate_hz', by='session')
   plt.ylabel('Firing rate (Hz)')
   plt.title('Firing rate distribution across sessions')

Quality Control Pipeline
-------------------------

.. code-block:: python

   from spikedata import SpikeData
   from data_loaders import load_spikedata_from_kilosort
   import numpy as np
   
   # Load raw sorted data
   sd = load_spikedata_from_kilosort(
       'kilosort_output/',
       fs_Hz=30000,
       cluster_info_tsv='cluster_info.tsv'
   )
   
   print(f"Initial: {sd.N} neurons")
   
   # 1. Compute ISI statistics
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # 2. Compute firing rates
   sd.compute_firing_rates(unit='Hz')
   
   # 3. Define quality criteria
   df = sd.neuron_attributes.to_dataframe()
   
   good_units = df[
       (df['refractory_violations'] == 0) &          # No refractory violations
       (df['firing_rate_hz'] > 0.5) &                # Active neurons
       (df['firing_rate_hz'] < 100.0) &              # Not unrealistically high
       (df['cv_isi'] < 3.0) &                        # Not too irregular
       (df.get('isolation_distance', 100) > 20)      # Well isolated (if available)
   ]
   
   # 4. Apply filter
   good_indices = good_units.index.tolist()
   sd_clean = sd.subset(good_indices)
   
   print(f"After QC: {sd_clean.N} neurons ({sd_clean.N/sd.N*100:.1f}%)")
   
   # 5. Store QC metadata
   sd_clean.metadata['qc_applied'] = True
   sd_clean.metadata['qc_criteria'] = {
       'refractory_violations': 0,
       'min_firing_rate': 0.5,
       'max_firing_rate': 100.0,
       'max_cv_isi': 3.0
   }
   
   # 6. Export clean data
   sd_clean.to_hdf5('clean_data.h5', style='ragged')
   
   # 7. Generate QC report
   print("\nQC Report:")
   print(f"Neurons removed: {sd.N - sd_clean.N}")
   print(f"Mean firing rate: {df.loc[good_indices, 'firing_rate_hz'].mean():.2f} Hz")
   print(f"Mean CV: {df.loc[good_indices, 'cv_isi'].mean():.2f}")

