# SpikeData Project: TODO & Progress Tracker

## Deprecated/Problematic Code Removal

- [x] Remove the following deprecated functions/classes (DONE):
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

## Spike Time Tiling Code Organization

- [x] Consolidate all spike time tiling (STTC) related methods and helpers into a single section (DONE)
- [x] Remove redundant/duplicate STTC code (DONE)
- [x] Ensure only one implementation per STTC helper/function (DONE)

---

## Completed Tasks (as of 2025-09-14)

- [x] Update README with migration notes for SpikeData refactor
- [x] Add/refine docstrings and inline comments in `spikedata/spikedata.py`
- [x] Update tests to remove deprecated APIs and ensure Python 3.9/SciPy compatibility

---

## Outstanding Tasks

### Data Loader Enhancements

- [ ] Implement new data loaders for SpikeData:
    - [ ] NWB loader
    - [ ] S3 loader for direct Kilosort2 output import
    - [ ] SpikeInterface loader (from BaseSorting: https://spikeinterface.readthedocs.io/en/stable/modules/core.html)

### Waveform Template Support

- [ ] Add waveform template attribute to SpikeData class:
    - [ ] Update class definition to include waveform template attribute
    - [ ] Update all loaders to check for and load waveform template data if available

---
