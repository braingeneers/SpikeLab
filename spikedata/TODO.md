# TODO List for SpikeData Project

- [ ] Remove deprecated or problematic code sections:
    - [ ] Remove the following functions/classes:
        - NestIDNeuronAttributes
        - from_nest
        - from_mbt_neurons
        - isi_skewness
        - isi_log_histogram
        - isi_threshold_cma
        - burstiness_index
        - avalanches
        - avalanche_duration_size
        - deviation_from_criticality
        - randomized (does not work well, will be replaced)
        - randomize_raster
        - _okun_swap
        - randomize_raster_okun
        - randomize_raster_greedy
        - best_effort_sample
        - population_firing_rate (will be replaced)
        - _p_and_alpha
        - fano_factors
        - pearson
        - cumulative_moving_average
        - burst_detection (will be replaced)

- [ ] Clean up and organize spike time tiling code:
    - [ ] Collect all spike time tiling related methods and helper functions into a dedicated section of the file
    - [ ] Remove redundant or duplicate spike time tiling code
    - [ ] Ensure only one implementation per helper/function

- [ ] Implement new data loaders:
    - [ ] Add NWB loader for SpikeData
    - [ ] Add S3 loader to load Kilosort2 outputs directly from S3 into SpikeData
    - [ ] Add SpikeInterface loader (specifically from BaseSorting: https://spikeinterface.readthedocs.io/en/stable/modules/core.html)

- [ ] Add and handle waveform template attribute in SpikeData class:
    - [ ] Update SpikeData class to include waveform template attribute
    - [ ] Ensure all loaders check for and load waveform template data if available

- [ ] Create and maintain a progress tracking sheet on Drive:
    - [ ] Organize all tasks in the sheet
    - [ ] Keep track of progress and update regularly
