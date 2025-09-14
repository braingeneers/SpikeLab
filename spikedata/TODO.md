# TODO List for SpikeData Project

- [ ] Remove deprecated or problematic code sections:
    - [ ] Remove the following functions/classes:
        - [x] NestIDNeuronAttributes
        - [x] from_nest
        - [x] from_mbt_neurons
        - [x] isi_skewness
        - [x] isi_log_histogram
        - [x] isi_threshold_cma
        - [x] burstiness_index
        - [x] avalanches
        - [x] avalanche_duration_size
        - [x] deviation_from_criticality
        - [x] randomized (does not work well, will be replaced)
        - [x] randomize_raster
        - [x] _okun_swap
        - [x] randomize_raster_okun
        - [x] randomize_raster_greedy
        - [x] best_effort_sample
        - [x] population_firing_rate (will be replaced)
        - [x] _p_and_alpha
        - [x] fano_factors
        - [x] pearson
        - [x] cumulative_moving_average
        - [x] burst_detection (will be replaced)

- [ ] Clean up and organize spike time tiling code:
    - [x] Collect all spike time tiling related methods and helper functions into a dedicated section of the file
    - [x] Remove redundant or duplicate spike time tiling code
    - [x] Ensure only one implementation per helper/function

## Completed (2025-09-14)

- [x] Update README with migration notes for SpikeData refactor
- [x] Add/refine docstrings and inline comments in `spikedata/spikedata.py`
- [x] Update tests to remove deprecated APIs and ensure Python 3.9/SciPy compatibility

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
