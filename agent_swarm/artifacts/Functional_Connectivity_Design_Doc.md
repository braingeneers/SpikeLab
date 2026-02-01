# Design Document for Functional Connectivity Metric Implementation

## Overview
This design document provides a detailed plan for implementing the functional connectivity metric described in the paper "Statistical connection between networks: Granger causality inference from multivariate time series data" available at PubMed (PMID: 29024669). The objective is to integrate a functional connectivity analysis tool into the existing Python-based `IntegratedAnalysisTools`, leveraging `numpy`, `scipy`, and `torch`.

## Existing Structures
The primary data structures involved are:
- `SpikeData`: Stores spiking activity data from neural recordings.
- `RateData`: Processes and provides binned firing rate data.

## Functional Connectivity Metric
The paper proposes a functional connectivity metric based on Granger causality concepts, which can be briefly summarized as follows:

### Core Concept
Granger causality is a statistical hypothesis test for determining whether one time series can predict another. If the past information of time series X provides statistically significant information about future values of time series Y, X is said to "Granger-cause" Y.

### Key Steps for Implementation:
1. **Preprocessing Data**
   - Use the `SpikeData` or `RateData` class to extract time series data.
   - Ensure the data is appropriately binned and formatted.

2. **Granger Causality Analysis**
   - Utilize the `scipy` library for applying vector autoregressive models (VAR) to assess temporal relationships.
   - Calculate test statistics to evaluate if one time series Granger-causes another.
   
3. **Significance Testing**
   - Use statistical methods to ascertain the significance of the Granger causality tests.
   - Employ bootstrapping methods if necessary to ensure robustness.

4. **Integration with Library**
   - Design and implement functions within `IntegratedAnalysisTools` using Python with `numpy`, `scipy`, and potentially `torch` for enhanced computational performance.
   - Provide clear API documentation for the new additions.

5. **Validation**
   - Design unit tests to validate the correctness of implementations against synthetic data where the ground truth is known.

## Algorithm Design
- **Input**: Time series data extracted from `SpikeData` or `RateData`.
- **Output**: A matrix detailing Granger-causal links and their statistical significance.

### Function Specification
- `def compute_granger_causality(time_series_x, time_series_y, max_lag):`
  - Computes whether `time_series_x` Granger-causes `time_series_y` using VAR and significance testing.
  - **Parameters**:
    - `time_series_x`: Array containing time series data of neuron X.
    - `time_series_y`: Array containing time series data of neuron Y.
    - `max_lag`: Maximum number of time lags to consider in the VAR model.
  - **Returns**: A tuple indicating statistical significance and test statistic.

## Testing Plan
- Create synthetic datasets with known causal relationships to ensure algorithm reliability.
- Validate implementation on empirical datasets to check for consistency with documented scientific findings.

## Appendices
### References
- Source Paper: [Functional Connectivity Paper](https://pubmed.ncbi.nlm.nih.gov/29024669/)
- Additional Reading: Granger Causality documentation, VAR processes, signal processing recommendations.

