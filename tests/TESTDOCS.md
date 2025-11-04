## Test Documentation

This document explains what each test does in detail and why it matters. It is intended to help maintainers and contributors understand the coverage and the design guarantees provided by the test suite.

---

## SpikeData tests (`tests/test_spikedata.py`)

### Helpers
- **assertSpikeDataEqual(sda, sdb)**
  - Ensures two `SpikeData` instances have identical per-neuron spike trains (length and values within tolerance).
  - Used across tests to assert semantic equality rather than object identity.

- **assertSpikeDataSubtime(sd, sdsub, tmin, tmax)**
  - Verifies a time-sliced `SpikeData` (`sdsub = sd.subtime(tmin, tmax)`) has the expected duration, contains only spikes within the target window, and preserves per-unit alignment.
  - Ensures consistent slicing semantics including edge handling when `tmin <= 0`.

- **assertAll(bools)** and **assertClose(a, b, ...)**
  - Convenience wrappers around `numpy` assertions for readability and consistent error messages.

### Tests
- **test_sd_from_counts**
  - Purpose: Validates that a helper `sd_from_counts` produces a `SpikeData` whose `binned(1)` exactly matches the provided per-bin spike counts.
  - Why: Confirms the correctness of binning rules at unit bin size and the integrity of spike placement in bins.
  - Key checks: Equality between `sd.binned(1)` and the generated counts array.

- **test_neo_conversion** (skipped if `neo`/`quantities` are not installed)
  - Purpose: Round-trip conversion between `SpikeData` and `neo.core.SpikeTrain` objects preserves spike times and recording length.
  - Why: Ensures interoperability with the Neo ecosystem, a common interchange format in neuroscience.
  - Key checks: Equality of the original and reconstructed `SpikeData` via `assertSpikeDataEqual`.

- **test_spike_data**
  - Purpose: Comprehensive coverage of core constructors and basic methods.
  - Why: Ensures consistency of various constructors and core APIs that other features depend on.
  - Verifies:
    - `from_idces_times` yields expected `times` (sorted) and `events` round-trip.
    - `from_events`, base constructor, and `from_idces_times(*sd.idces_times())` are all consistent with each other.
    - `from_raster` produces rasters consistent with the input after re-binning.
    - `subset` preserves train identities and ordering.
    - `subtime` is idempotent over full range and extracts correct spikes for ranges (including negative arguments and ellipses `...`).
    - `frames` equivalence with `subtime` for consecutive non-overlapping and overlapping windows.

- **test_raster**
  - Purpose: Validates rasterization logic and shape consistency.
  - Why: Rasters underpin many analyses; correctness ensures downstream metrics are valid.
  - Verifies:
    - Total spike count preserved in dense and sparse rasters.
    - Raster shape consistency across datasets with equal length.
    - Inclusion/exclusion rules at bin edges (0 included; other bins lower-open/upper-closed).
    - Consistency between `raster` and `binned` outputs.

- **test_rates**
  - Purpose: Checks rate calculations in default units and conversion to Hz.
  - Why: Accurate rate metrics are foundational for common analyses.
  - Verifies default unit returns counts, `Hz` conversion factor, and error on invalid unit string.

- **test_interspike_intervals**
  - Purpose: Validates ISI computation for simple, multi-train, and random cases.
  - Why: ISIs are widely used statistics; correctness is crucial.
  - Verifies constant-spacing trains yield uniform ISIs, multi-train handling, and random case equivalence to finite differences.

- **test_spike_time_tiling_ta** and **test_spike_time_tiling_na**
  - Purpose: Unit tests for internal STTC helpers `_sttc_ta` (total available time) and `_sttc_na` (spike counts in window).
  - Why: These low-level functions are easy to break and difficult to debug indirectly.
  - Verifies trivial, boundary, overlap, and interval-closure behavior.

- **test_spike_time_tiling_coefficient**
  - Purpose: Validates `spike_time_tiling` and `spike_time_tilings` end-to-end.
  - Why: Ensures the STTC implementation behaves as expected across symmetrical, identical, random, anti-correlated, and empty cases.
  - Verifies symmetry, diagonal elements at 1 for identical trains, expected limits for alternating trains, and range bounds [-1, 1].

- **test_binning_doesnt_lose_spikes**
  - Purpose: Asserts that binning preserves total spike counts for a Poisson spike train.
  - Why: Guarantees conservation, preventing silent data loss in binning-based pipelines.

- **test_binning**
  - Purpose: Checks bin assignment for a fixed set of spike times at bin size 4.
  - Why: Regression against off-by-one binning errors.
  - Verifies exact expected counts per bin.

- **test_metadata**
  - Purpose: Validates metadata and `neuron_attributes` propagation and copy semantics through `subset` and `subtime`.
  - Why: Prevents subtle bugs where metadata mutates unexpectedly or attributes misalign after subsetting.
  - Verifies:
    - Constructor errors on malformed attributes.
    - `subset` preserves metadata dict and subsets attributes.
    - `subtime` copies metadata (not shared reference) and preserves attributes.

- **test_raw_data**
  - Purpose: Validates handling of optional `raw_data` and `raw_time` in `SpikeData`.
  - Why: Ensures consistent API for raw traces alongside spikes, including automatic timebase generation and slicing.
  - Verifies constructor errors for mismatched presence/lengths, generation of time vector from scalar `raw_time`, and correct slicing with `subtime`.

- **test_isi_rate**
  - Purpose: Tests ISI-based rate estimation (`resampled_isi` and `_resampled_isi`).
  - Why: Validates a core analysis measure, including limits and variation with spike intervals.
  - Verifies constant-rate neuron returns expected rate and varied intervals produce expected rate trends.

- **test_latencies**
  - Purpose: Validates latency calculations relative to reference times.
  - Why: Ensures latency analyses behave predictably for shifted trains and edge cases.
  - Verifies correct positive/negative latencies and empty results for too-small windows.

- **test_randomize_preserves_marginals**
  - Purpose: Tests `spikedata.randomize` preserves row/column sums and total spikes.
  - Why: Guarantees the randomization keeps degree distributions intact for null-model analyses.
  - Verifies shape, binarity, and equality of marginals.

- **test_get_pop_rate_square_only_matches_convolution**
  - Purpose: With square window only, `get_pop_rate` matches direct convolution of summed spike train.
  - Why: Validates correctness against a simple analytical reference.

- **test_get_pop_rate_gaussian_only_impulse**
  - Purpose: With Gaussian-only smoothing and a single impulse, output forms a normalized, symmetric Gaussian.
  - Why: Confirms kernel application and normalization.

- **test_get_bursts_detects_simple_peaks**
  - Purpose: Detects simple, well-separated bursts with amplitude-scaled edges.
  - Why: Ensures basic burst detection sanity: peak times, edges, and amplitudes.

---

## Data loader tests (`tests/test_dataloaders.py`)

### HDF5 loaders (`TestHDF5Loaders`)
- Uses temporary HDF5 files; skipped entirely if `h5py` is unavailable.
- Includes teardown to clean up created temp files.

- **test_hdf5_raster**
  - Purpose: Loads a 2D raster (units × time) and verifies rasterization and unit count.
  - Why: Confirms the raster-style HDF5 input path.

- **test_hdf5_raster_not_2d_raises**
  - Purpose: Asserts a non-2D raster dataset raises `ValueError`.
  - Why: Guards against malformed inputs.

- **test_hdf5_multiple_styles_raises**
  - Purpose: Providing more than one input style raises `ValueError`.
  - Why: Enforces API contract of mutually exclusive style selection.

- **test_hdf5_idces_times_ms**
  - Purpose: Paired indices/times in milliseconds are loaded losslessly.
  - Why: Validates paired-arrays style and unit handling.

- **test_hdf5_group_per_unit_seconds**
  - Purpose: Group-per-unit datasets with seconds are converted to ms correctly.
  - Why: Confirms per-dataset unit conversion and ordering across child datasets.

- **test_hdf5_group_per_unit_empty_units**
  - Purpose: Handles empty per-unit datasets with correct `N`, `length`, and empty trains.
  - Why: Robustness for sparse or placeholder datasets.

- **test_hdf5_flat_ragged_spike_times**
  - Purpose: Flat ragged arrays plus end indices are split into per-unit trains and converted to ms.
  - Why: Validates NWB-like ragged representation loading path.

- **test_hdf5_idces_times_samples_with_fs**
  - Purpose: Times in samples are converted to ms with specified `fs_Hz`.
  - Why: Ensures sample-based unit conversions are correct.

- **test_hdf5_raw_attachment_seconds_and_samples**
  - Purpose: Optional `raw_data` attachment and `raw_time` conversion from seconds and from samples (with `fs_Hz`).
  - Why: Confirms auxiliary raw arrays are ingested and timebase conversion is correct.

- **test_hdf5_invalid_style_error**
  - Purpose: Empty file with no recognizable inputs raises `ValueError`.
  - Why: Prevents silent success on invalid inputs.

- **test_hdf5_samples_without_fs_error**
  - Purpose: Using `'samples'` time unit without `fs_Hz` raises `ValueError`.
  - Why: Enforces required arguments for unit conversion.

- **test_hdf5_raw_thresholded**
  - Purpose: Threshold detection on a raw dataset finds events in supra-threshold segments.
  - Why: Validates `load_spikedata_from_hdf5_raw_thresholded` end-to-end.

### NWB loader (`TestNWBLoader`)
- Skipped if `h5py` is unavailable.

- **test_nwb_units_via_h5py**
  - Purpose: Minimal NWB-like Units table via `h5py` is parsed and converted to ms.
  - Why: Provides fallback path validation when `pynwb` is not used.

- **test_nwb_missing_units_raises**
  - Purpose: Missing `/units` group raises `ValueError`.
  - Why: Ensures informative failure on malformed NWB files.

- **test_nwb_alt_names_with_endswith**
  - Purpose: Accepts datasets whose names end with `spike_times` and `spike_times_index`.
  - Why: Robustness to minor naming variations seen in some NWB exports.

### ACQM and S3 Support (`TestACQMLoader`)
- **test_acqm_basic**
  - Purpose: Loads ACQM file (NPZ with spike trains, metadata, neuron data) and validates conversion from samples to milliseconds.
  - Why: Ensures ACQM format handling, sampling frequency application, and neuron attributes extraction.
  - Verifies neuron count, duration, spike time conversion, and metadata preservation (fs_Hz, config, redundant_pairs).

- **test_acqm_neuron_attributes**
  - Purpose: Validates extraction of neuron attributes from ACQM neuron_data structure.
  - Why: ACQM files contain rich per-neuron metadata (cluster_id, channel, position) that must be preserved.
  - Verifies presence of standard columns and filtering of large arrays (waveforms, templates).

- **test_acqm_s3_mock** (if boto3 available)
  - Purpose: Mocks S3 download to test S3 URI handling and caching logic without actual S3 access.
  - Why: Validates S3 URI parsing, cache directory creation, and download integration.
  - Verifies caching behavior: first access downloads, subsequent uses cache.

- **test_download_s3_to_local** (if boto3 available)
  - Purpose: Tests direct S3 download function with mocked boto3 client.
  - Why: Ensures S3 download infrastructure works correctly for both ACQM and HDF5 loaders.
  - Verifies bucket/key parsing, endpoint URL handling, and file writing.

- **test_hdf5_s3_support**
  - Purpose: Tests HDF5 loader with S3 URI and caching.
  - Why: Extends HDF5 loader functionality to cloud storage for large datasets.
  - Verifies automatic S3 resolution and cache reuse.

### KiloSort and SpikeInterface (`TestKiloSortAndSpikeInterface`)
- **test_kilosort_basic**
  - Purpose: Loads two clusters, verifies per-cluster times in ms and metadata `cluster_ids` alignment with train order.
  - Why: Validates basic KiloSort flow and deterministic ordering by cluster id.

- **test_spikeinterface_mock**
  - Purpose: Converts a mock `SortingExtractor` to `SpikeData` with sample-to-ms conversion using sorting `fs`.
  - Why: Ensures compatibility with SpikeInterface without requiring the library.

- **test_spikeinterface_base_recording_thresholding**
  - Purpose: Thresholds a mock `BaseRecording`, detects spikes on a supra-threshold burst, and auto-orients time × channels input.
  - Why: Validates detection and orientation heuristics for recordings.

- **test_spikeinterface_subset_and_override_fs**
  - Purpose: Subsets to specific unit IDs and overrides sampling frequency when sorting lacks it.
  - Why: Flexibility for partial analyses and incomplete metadata.

- **test_spikeinterface_invalid_object_raises**
  - Purpose: Invalid sorting-like objects raise `TypeError` early.
  - Why: Clear error messaging and API safety.

- **test_kilosort_empty_arrays**
  - Purpose: Empty KiloSort arrays yield zero units and zero length without errors.
  - Why: Edge-case robustness when no spikes were detected.

- **test_kilosort_metadata_cluster_ids_alignment**
  - Purpose: `cluster_ids` metadata is sorted and matches train order (ascending by cluster id).
  - Why: Ensures metadata integrity for downstream labeling.

- **test_kilosort_tsv_missing_columns_keeps_all**
  - Purpose: Missing expected columns in TSV triggers "keep all clusters" behavior.
  - Why: Graceful degradation with partial TSV metadata.

---

## Data exporter tests (`tests/test_dataexporters.py`)

### Base test class (`BaseExportTest`)
- **make_sd()**
  - Creates standardized test data: 3 units (3 spikes, 2 spikes, empty), 25ms length, with metadata.
  - Why: Consistent test data across all export formats ensures comparable validation and includes edge cases like empty units.

### HDF5 exporters (`TestHDF5Exporters`)
- Skipped if `h5py` is unavailable.
- Uses temporary HDF5 files with automatic cleanup.

- **test_export_hdf5_ragged_roundtrip**
  - Purpose: Tests ragged array export (flat spike times + cumulative indices) with time unit conversion to seconds, then round-trip import.
  - Why: Ragged arrays are the most storage-efficient format for sparse spike data and are used by NWB. Validates both export logic and time conversion accuracy.
  - Key checks: All spike trains match original after round-trip within floating-point precision.

- **test_export_hdf5_group_roundtrip_samples**
  - Purpose: Tests group-per-unit export style with conversion to sample indices at 1000 Hz sampling rate.
  - Why: Group style enables easy per-unit access without parsing index arrays. Sample units preserve exact timing relationships with original recordings.
  - Key checks: Each unit gets its own dataset; times converted correctly from milliseconds to samples; round-trip preserves data.

- **test_export_hdf5_paired_roundtrip_ms**
  - Purpose: Tests paired arrays format (separate unit indices and spike times arrays) keeping original millisecond timing.
  - Why: Paired format is intuitive and matches how many analysis pipelines represent spike data internally. Avoiding time conversion prevents precision loss.
  - Key checks: Unit indices and spike times are properly paired; empty units handled correctly; round-trip preserves data.

- **test_export_hdf5_raster**
  - Purpose: Tests raster export for binned spike count analysis with 5ms bins.
  - Why: Raster format enables analyses requiring fixed-size inputs (neural decoders, population dynamics). Essential for rate-based analyses.
  - Key checks: Exported raster exactly matches SpikeData's own raster() method output.

- **test_export_hdf5_with_raw**
  - Purpose: Tests export of raw data arrays alongside spike data with proper time unit conversion.
  - Why: Many analyses require both spike times and underlying continuous data stored together with consistent time bases.
  - Key checks: Raw time array is correctly converted from milliseconds to seconds; raw data preserved.

### NWB exporters (`TestNWBExporters`)
- Skipped if `h5py` is unavailable.
- Uses temporary NWB files with automatic cleanup.

- **test_export_nwb_roundtrip**
  - Purpose: Tests NWB format export and round-trip import compatibility.
  - Why: NWB is becoming the standard for sharing neurophysiology data. Ensures exports create valid NWB files maintaining data integrity for use with other NWB tools.
  - Key checks: Uses ragged array format internally; times converted to seconds (NWB standard); data organized in /units group; round-trip preserves all spike trains.

### KiloSort exporters (`TestKiloSortExporters`)
- **test_export_kilosort_roundtrip_samples**
  - Purpose: Tests KiloSort format export (spike_times.npy, spike_clusters.npy) and round-trip import with sample-based timing.
  - Why: KiloSort format is widely used in spike sorting community. Ensures compatibility with KiloSort, Phy, and other tools using this simple but effective format.
  - Key checks: Unit indices map to cluster IDs; spike times converted to sample indices; round-trip through KiloSort loader preserves data; loader sorts by cluster ID matching export order.

- **test_export_kilosort_custom_cluster_ids**
  - Purpose: Tests KiloSort export with custom cluster ID assignment instead of default unit index mapping.
  - Why: Preserves original cluster IDs from spike sorting results or enables specific numbering schemes. Important for data provenance.
  - Key checks: Custom cluster IDs [10, 5, 7] correctly assigned; spike counts match expected values; empty units handled properly (don't contribute spikes to output).

---

## NeuronAttributes tests (`tests/test_neuron_attributes.py`)

### ISI Statistics Tests
- **test_compute_isi_statistics**
  - Purpose: Validates computation of seven ISI-based metrics (mean, median, CV, skewness, burst index, pause ratio, refractory violations).
  - Why: ISI statistics characterize firing patterns and are essential for cell classification and quality control.
  - Verifies correct calculation for regular, bursting, and irregular spike trains.

- **test_compute_isi_statistics_auto_save**
  - Purpose: Tests auto_save=True default behavior stores all metrics in neuron_attributes.
  - Why: Ensures computed metrics persist for later use without manual storage.
  - Verifies all seven columns present after computation.

- **test_isi_statistics_empty_neurons**
  - Purpose: Handles neurons with no spikes (returns NaN for metrics).
  - Why: Robustness for sparse datasets with inactive neurons.
  - Verifies NaN returned for empty spike trains, not errors.

### Latency Statistics Tests
- **test_compute_latency_statistics**
  - Purpose: Validates latency computation relative to reference neuron (mean, median, jitter).
  - Why: Latency analysis identifies functional connectivity and temporal coordination.
  - Verifies positive/negative latencies for followers/leaders and window size handling.

- **test_latency_statistics_auto_save**
  - Purpose: Tests auto_save=True stores latency metrics in neuron_attributes.
  - Why: Persistence of computed timing relationships.
  - Verifies three latency columns saved.

- **test_latency_statistics_window**
  - Purpose: Validates window_ms parameter correctly filters latencies.
  - Why: Window size affects which spike pairs are considered relevant.
  - Verifies NaN for spikes outside window, values for spikes within.

### Burst Participation Tests
- **test_compute_burst_participation**
  - Purpose: Validates burst participation calculation and backbone unit identification.
  - Why: Identifies neurons driving network activity versus bystanders.
  - Verifies participation fractions, backbone classification, and metadata storage.

- **test_burst_participation_auto_save**
  - Purpose: Tests auto_save=True stores burst metrics in neuron_attributes.
  - Why: Persistence of burst analysis results.
  - Verifies burst_participation and is_backbone_unit columns.

- **test_burst_participation_metadata**
  - Purpose: Validates storage of burst analysis metadata in spikedata.metadata.
  - Why: Preserves analysis parameters for reproducibility.
  - Verifies burst_edges, parameters, and counts stored correctly.

- **test_get_frac_active**
  - Purpose: Tests low-level burst participation calculation (SpikeData.get_frac_active).
  - Why: Direct validation of burst detection logic before neuron_attributes integration.
  - Verifies per-unit and per-burst fractions, backbone classification.

### STTC Caching Tests
- **test_get_sttc_matrix_caching**
  - Purpose: Validates STTC matrix caching mechanism provides identical results.
  - Why: STTC is expensive (O(N²)); caching provides major speedup.
  - Verifies first call computes, second call returns same result from cache.

- **test_get_sttc_matrix_use_cache_false**
  - Purpose: Tests use_cache=False forces recomputation.
  - Why: Allows bypassing cache when needed for testing or validation.
  - Verifies new computation occurs and result matches.

- **test_sttc_cache_delt_separation**
  - Purpose: Validates different delt values cached separately.
  - Why: Multi-scale analysis requires caching multiple parameter values.
  - Verifies delt=20.0 and delt=40.0 cached independently.

- **test_clear_sttc_cache**
  - Purpose: Tests cache clearing for memory management.
  - Why: Large datasets need cache clearing to free memory.
  - Verifies clear_sttc_cache() removes cached matrices, both specific and all.

- **test_sttc_cache_performance**
  - Purpose: Benchmarks caching speedup (typically 100-1000x).
  - Why: Validates performance benefit of caching.
  - Verifies cached access much faster than initial computation.

---

## How to use this document
- When modifying code, identify which tests validate the behavior you are touching and run those first.
- For new features, mirror the style of existing tests and add explanations here to keep this document in sync.
