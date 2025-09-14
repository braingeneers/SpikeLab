# IntegratedAnalysisTools
A monorepo for a bunch of analysis tools for automated closed loop experimentation and analysis

## SpikeData module refactor (2025-09)

The `spikedata/spikedata.py` module was streamlined to focus on core spike train
representation, binning, and spike time tiling (STTC). Several legacy or niche APIs
were removed and will be replaced by focused utilities.

Removed items:
- Nest/NEST features: `NestIDNeuronAttributes`, `SpikeData.from_nest`
- MuscleBeachTools: `SpikeData.from_mbt_neurons`
- ISI analytics: `SpikeData.isi_skewness`, `SpikeData.isi_log_histogram`,
  `SpikeData.isi_threshold_cma`
- Burst/avalanche/DCC: `SpikeData.burstiness_index`, `SpikeData.avalanches`,
  `SpikeData.avalanche_duration_size`, `SpikeData.deviation_from_criticality`,
  `DCCResult`, `_p_and_alpha`
- Randomization: `SpikeData.randomized`, `randomize_raster`, `randomize_raster_greedy`,
  `randomize_raster_okun`, `_okun_swap`, `best_effort_sample`
- Rates/correlations/hist utils: `population_firing_rate` (function and method),
  `fano_factors`, `pearson`, `cumulative_moving_average`, `burst_detection`

Reorganization:
- STTC helpers (`_sttc_ta`, `_sttc_na`) are colocated with `spike_time_tiling` for
  clarity. Behavior unchanged.

Migration tips:
- Population rate: use `SpikeData.binned(bin_size)` then smooth externally, e.g.:
  ```python
  bins = sd.binned(10)
  smoothed = np.convolve(bins / 10, np.ones(5), 'same') / 5
  ```
- Pairwise correlations: compute on `sd.raster()` using NumPy/SciPy:
  ```python
  r = sd.raster(1.0)
  corr = np.corrcoef(r)
  ```
- Burst/avalanche/DCC: replacement modules will provide dedicated implementations.
