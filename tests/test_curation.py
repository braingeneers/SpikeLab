"""Tests for spikelab.spikedata.curation module."""

import numpy as np
import pytest

from spikelab.spikedata import SpikeData
from spikelab.spikedata.curation import (
    build_curation_history,
    compute_pairwise_similarity,
    compute_waveform_metrics,
    curate,
    curate_by_firing_rate,
    curate_by_isi_violations,
    curate_by_merge_duplicates,
    curate_by_min_spikes,
    curate_by_snr,
    curate_by_std_norm,
    filter_by_cosine_sim,
    filter_pairs_by_isi_violations,
    find_nearby_unit_pairs,
    merge_redundant_units,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_sd(n_units=5, spikes_per_unit=20, length=1000.0, **kwargs):
    """Build a simple SpikeData for curation tests."""
    rng = np.random.default_rng(42)
    trains = [
        np.sort(rng.uniform(0, length, size=spikes_per_unit)) for _ in range(n_units)
    ]
    return SpikeData(trains, length=length, **kwargs)


def _make_sd_varied():
    """Build a SpikeData with deliberately varied spike counts.

    Returns a SpikeData with 4 units:
        unit 0: 50 spikes
        unit 1: 5 spikes
        unit 2: 100 spikes
        unit 3: 2 spikes
    """
    rng = np.random.default_rng(99)
    trains = [np.sort(rng.uniform(0, 1000, size=n)) for n in [50, 5, 100, 2]]
    return SpikeData(
        trains,
        length=1000.0,
        neuron_attributes=[
            {"unit_id": 10},
            {"unit_id": 20},
            {"unit_id": 30},
            {"unit_id": 40},
        ],
    )


def _make_sd_with_raw(n_units=3, length_ms=100.0, fs_kHz=30.0):
    """Build a SpikeData with raw_data attached for waveform tests.

    Creates synthetic raw data with clear spikes (large negative
    deflections) so that SNR is measurable.
    """
    rng = np.random.default_rng(7)
    n_channels = n_units
    n_samples = int(length_ms * fs_kHz)

    # Base noise
    raw = rng.normal(0, 1.0, size=(n_channels, n_samples))

    # Inject spikes as large negative deflections
    trains = []
    for u in range(n_units):
        spike_times = np.array([20.0, 50.0, 80.0])
        trains.append(spike_times)
        for t_ms in spike_times:
            sample = int(t_ms * fs_kHz)
            if sample < n_samples:
                raw[u, max(0, sample - 5) : sample + 5] -= 20.0

    return SpikeData(
        trains,
        length=length_ms,
        raw_data=raw,
        raw_time=fs_kHz,
        neuron_attributes=[{"unit_id": i, "channel": i} for i in range(n_units)],
    )


# ---------------------------------------------------------------------------
# curate_by_min_spikes
# ---------------------------------------------------------------------------


class TestCurateByMinSpikes:
    def test_basic_filtering(self):
        """
        Units below the spike count threshold are removed.

        Tests:
            (Test Case 1) Units with fewer than min_spikes are excluded.
            (Test Case 2) Returned metric contains spike counts for all
                original units.
            (Test Case 3) Passed array is boolean with correct shape.
        """
        sd = _make_sd_varied()
        sd_out, res = curate_by_min_spikes(sd, min_spikes=10)

        assert sd_out.N == 2  # units 0 (50) and 2 (100)
        assert res["metric"].shape == (4,)
        assert res["passed"].dtype == bool
        assert res["passed"].shape == (4,)
        np.testing.assert_array_equal(res["metric"], [50, 5, 100, 2])
        np.testing.assert_array_equal(res["passed"], [True, False, True, False])

    def test_all_pass(self):
        """
        All units pass when threshold is 1.

        Tests:
            (Test Case 1) No units are removed when all exceed threshold.
        """
        sd = _make_sd_varied()
        sd_out, res = curate_by_min_spikes(sd, min_spikes=1)
        assert sd_out.N == sd.N
        assert np.all(res["passed"])

    def test_none_pass(self):
        """
        All units fail when threshold exceeds all spike counts.

        Tests:
            (Test Case 1) Empty SpikeData returned when no units pass.
        """
        sd = _make_sd_varied()
        sd_out, res = curate_by_min_spikes(sd, min_spikes=200)
        assert sd_out.N == 0
        assert not np.any(res["passed"])

    def test_empty_spike_train(self):
        """
        Units with zero spikes are correctly handled.

        Tests:
            (Test Case 1) A unit with an empty train has metric 0 and
                fails any positive threshold.
        """
        sd = SpikeData([np.array([]), np.array([10.0, 20.0])], length=100.0)
        sd_out, res = curate_by_min_spikes(sd, min_spikes=1)
        assert sd_out.N == 1
        assert res["metric"][0] == 0.0
        assert not res["passed"][0]

    def test_neuron_attributes_preserved(self):
        """
        Neuron attributes are carried through to curated output.

        Tests:
            (Test Case 1) Curated SpikeData retains neuron_attributes
                of passing units.
        """
        sd = _make_sd_varied()
        sd_out, _ = curate_by_min_spikes(sd, min_spikes=10)
        assert sd_out.neuron_attributes is not None
        ids = [a["unit_id"] for a in sd_out.neuron_attributes]
        assert ids == [10, 30]


# ---------------------------------------------------------------------------
# curate_by_firing_rate
# ---------------------------------------------------------------------------


class TestCurateByFiringRate:
    def test_basic_filtering(self):
        """
        Units below the firing rate threshold are removed.

        Tests:
            (Test Case 1) Metric values are firing rates in Hz.
            (Test Case 2) Only units above min_rate_hz pass.
        """
        sd = _make_sd_varied()  # length=1000 ms = 1 s
        sd_out, res = curate_by_firing_rate(sd, min_rate_hz=10.0)

        expected_rates = np.array([50.0, 5.0, 100.0, 2.0])
        np.testing.assert_allclose(res["metric"], expected_rates)
        # Units with rate >= 10 Hz: unit 0 (50), unit 2 (100)
        assert sd_out.N == 2
        np.testing.assert_array_equal(res["passed"], [True, False, True, False])

    def test_zero_length_recording(self):
        """
        Zero-length recording produces zero firing rates without error.

        Tests:
            (Test Case 1) All metrics are zero and no units pass a
                positive threshold.
        """
        sd = SpikeData([np.array([])], length=0.0)
        sd_out, res = curate_by_firing_rate(sd, min_rate_hz=0.01)
        assert res["metric"][0] == 0.0
        assert sd_out.N == 0


# ---------------------------------------------------------------------------
# curate_by_isi_violations
# ---------------------------------------------------------------------------


class TestCurateByIsiViolations:
    def test_percent_method(self):
        """
        ISI violation percentage is computed correctly.

        Tests:
            (Test Case 1) Unit with tightly spaced spikes has high
                violation percentage.
            (Test Case 2) Unit with well-separated spikes has zero
                violations.
        """
        # Unit 0: spikes at 1ms apart (all violate 1.5ms threshold)
        # Unit 1: spikes at 10ms apart (no violations)
        sd = SpikeData(
            [np.array([10.0, 11.0, 12.0, 13.0]), np.array([10.0, 20.0, 30.0, 40.0])],
            length=100.0,
        )
        sd_out, res = curate_by_isi_violations(
            sd, max_violation=50.0, threshold_ms=1.5, method="percent"
        )
        # Unit 0: 4 spikes, 3 ISIs all < 1.5ms → 3/4 * 100 = 75%
        assert res["metric"][0] == pytest.approx(75.0)
        # Unit 1: 3 ISIs, none < 1.5ms → 0%
        assert res["metric"][1] == pytest.approx(0.0)

    def test_hill_method(self):
        """
        Hill method ISI violation ratio is computed without error.

        Tests:
            (Test Case 1) Hill method produces a non-negative metric.
            (Test Case 2) Clean unit has zero Hill metric.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0, 30.0, 40.0])],
            length=100.0,
        )
        sd_out, res = curate_by_isi_violations(
            sd, max_violation=1.0, threshold_ms=1.5, method="hill"
        )
        assert res["metric"][0] == pytest.approx(0.0)
        assert sd_out.N == 1

    def test_invalid_method_raises(self):
        """
        Invalid method string raises ValueError.

        Tests:
            (Test Case 1) method='invalid' raises ValueError.
        """
        sd = _make_sd(n_units=2)
        with pytest.raises(ValueError, match="method must be"):
            curate_by_isi_violations(sd, method="invalid")

    def test_single_spike_unit(self):
        """
        Unit with a single spike has zero ISI violations.

        Tests:
            (Test Case 1) A unit with one spike cannot have ISI violations.
        """
        sd = SpikeData([np.array([50.0])], length=100.0)
        sd_out, res = curate_by_isi_violations(sd, max_violation=1.0)
        assert res["metric"][0] == 0.0
        assert sd_out.N == 1

    def test_empty_train(self):
        """
        Unit with no spikes has zero ISI violations.

        Tests:
            (Test Case 1) Empty train produces metric 0.
        """
        sd = SpikeData([np.array([])], length=100.0)
        _, res = curate_by_isi_violations(sd, max_violation=1.0)
        assert res["metric"][0] == 0.0


# ---------------------------------------------------------------------------
# curate_by_snr
# ---------------------------------------------------------------------------


class TestCurateBySnr:
    def test_from_neuron_attributes(self):
        """
        SNR is read from precomputed neuron_attributes when available.

        Tests:
            (Test Case 1) Units with precomputed snr are filtered
                without needing raw_data.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0]), np.array([30.0, 40.0])],
            length=100.0,
            neuron_attributes=[{"snr": 10.0}, {"snr": 3.0}],
        )
        sd_out, res = curate_by_snr(sd, min_snr=5.0)
        assert sd_out.N == 1
        np.testing.assert_array_equal(res["metric"], [10.0, 3.0])
        np.testing.assert_array_equal(res["passed"], [True, False])

    def test_from_raw_data(self):
        """
        SNR is computed from raw_data when neuron_attributes lacks it.

        Tests:
            (Test Case 1) SNR is computed and units are filtered.
            (Test Case 2) Computed SNR values are positive.
        """
        sd = _make_sd_with_raw()
        sd_out, res = curate_by_snr(sd, min_snr=1.0)
        assert res["metric"].shape == (3,)
        assert np.all(res["metric"] > 0)

    def test_missing_both_raises(self):
        """
        ValueError raised when neither neuron_attributes nor raw_data
        provides SNR.

        Tests:
            (Test Case 1) Error message suggests compute_waveform_metrics.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0])],
            length=100.0,
            neuron_attributes=[{}],
        )
        with pytest.raises(ValueError, match="compute_waveform_metrics"):
            curate_by_snr(sd, min_snr=5.0)


# ---------------------------------------------------------------------------
# curate_by_std_norm
# ---------------------------------------------------------------------------


class TestCurateByStdNorm:
    def test_from_neuron_attributes(self):
        """
        Normalized STD is read from precomputed neuron_attributes.

        Tests:
            (Test Case 1) Units with precomputed std_norm are filtered
                correctly.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0]), np.array([30.0, 40.0])],
            length=100.0,
            neuron_attributes=[{"std_norm": 0.5}, {"std_norm": 1.5}],
        )
        sd_out, res = curate_by_std_norm(sd, max_std_norm=1.0)
        assert sd_out.N == 1
        np.testing.assert_array_equal(res["passed"], [True, False])

    def test_from_raw_data(self):
        """
        Normalized STD is computed from raw_data when precomputed values
        are not available.

        Tests:
            (Test Case 1) std_norm is computed and units are filtered.
        """
        sd = _make_sd_with_raw()
        sd_out, res = curate_by_std_norm(sd, max_std_norm=5.0)
        assert res["metric"].shape == (3,)

    def test_missing_both_raises(self):
        """
        ValueError raised when neither neuron_attributes nor raw_data
        provides std_norm.

        Tests:
            (Test Case 1) Error message suggests compute_waveform_metrics.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0])],
            length=100.0,
            neuron_attributes=[{}],
        )
        with pytest.raises(ValueError, match="compute_waveform_metrics"):
            curate_by_std_norm(sd, max_std_norm=1.0)


# ---------------------------------------------------------------------------
# compute_waveform_metrics
# ---------------------------------------------------------------------------


class TestComputeWaveformMetrics:
    def test_stores_in_neuron_attributes(self):
        """
        compute_waveform_metrics stores snr and std_norm in
        neuron_attributes.

        Tests:
            (Test Case 1) snr key is set for every unit.
            (Test Case 2) std_norm key is set for every unit.
            (Test Case 3) Returned metric arrays have correct shape.
        """
        sd = _make_sd_with_raw()
        sd_out, metrics = compute_waveform_metrics(sd)

        assert sd_out is sd  # modified in place
        assert metrics["snr"].shape == (3,)
        assert metrics["std_norm"].shape == (3,)
        for attrs in sd.neuron_attributes:
            assert "snr" in attrs
            assert "std_norm" in attrs

    def test_snr_positive_for_clear_spikes(self):
        """
        SNR is positive for units with injected spike deflections.

        Tests:
            (Test Case 1) All units have SNR > 1 given strong spike
                injection.
        """
        sd = _make_sd_with_raw()
        _, metrics = compute_waveform_metrics(sd)
        assert np.all(metrics["snr"] > 1.0)

    def test_no_raw_data_raises(self):
        """
        ValueError raised when raw_data is empty.

        Tests:
            (Test Case 1) Error message mentions raw voltage traces.
        """
        sd = _make_sd(n_units=2)
        with pytest.raises(ValueError, match="raw_data is empty"):
            compute_waveform_metrics(sd)

    def test_initializes_neuron_attributes(self):
        """
        neuron_attributes is created if None before computation.

        Tests:
            (Test Case 1) SpikeData with neuron_attributes=None gets
                attributes initialized.
        """
        rng = np.random.default_rng(0)
        raw = rng.normal(0, 1, size=(1, 3000))
        raw[0, 500:510] -= 20.0
        sd = SpikeData(
            [np.array([16.0, 50.0])],
            length=100.0,
            raw_data=raw,
            raw_time=30.0,
        )
        assert sd.neuron_attributes is None
        compute_waveform_metrics(sd)
        assert sd.neuron_attributes is not None
        assert len(sd.neuron_attributes) == 1


# ---------------------------------------------------------------------------
# curate (combined wrapper)
# ---------------------------------------------------------------------------


class TestCurate:
    def test_multiple_criteria(self):
        """
        Combined curation applies multiple criteria in sequence.

        Tests:
            (Test Case 1) Only units passing all criteria survive.
            (Test Case 2) Results dict contains one entry per requested
                criterion.
        """
        sd = _make_sd_varied()
        sd_out, results = curate(sd, min_spikes=10, min_rate_hz=20.0)

        # min_spikes=10 keeps units 0(50), 2(100)
        # min_rate_hz=20 on those: 50Hz, 100Hz → both pass
        assert sd_out.N == 2
        assert "spike_count" in results
        assert "firing_rate" in results

    def test_no_criteria_returns_unchanged(self):
        """
        Calling curate with no thresholds returns the original SpikeData.

        Tests:
            (Test Case 1) No criteria applied means all units survive.
        """
        sd = _make_sd_varied()
        sd_out, results = curate(sd)
        assert sd_out.N == sd.N
        assert len(results) == 0

    def test_only_requested_criteria_included(self):
        """
        Results dict only contains keys for criteria that were requested.

        Tests:
            (Test Case 1) Only min_spikes is present when only that
                threshold is specified.
        """
        sd = _make_sd_varied()
        _, results = curate(sd, min_spikes=3)
        assert list(results.keys()) == ["spike_count"]

    def test_sequential_filtering(self):
        """
        Criteria are applied sequentially — later criteria see only
        units that passed earlier ones.

        Tests:
            (Test Case 1) Firing rate metric array length equals the
                number of units that passed spike count filtering.
        """
        sd = _make_sd_varied()  # units: 50, 5, 100, 2 spikes
        _, results = curate(sd, min_spikes=10, min_rate_hz=1.0)

        # spike_count runs on all 4 units
        assert results["spike_count"]["metric"].shape == (4,)
        # firing_rate runs on 2 survivors
        assert results["firing_rate"]["metric"].shape == (2,)

    def test_with_snr_from_attributes(self):
        """
        Combined curation can include SNR when precomputed in
        neuron_attributes.

        Tests:
            (Test Case 1) SNR criterion is applied from neuron_attributes.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0, 30.0]), np.array([40.0, 50.0, 60.0])],
            length=100.0,
            neuron_attributes=[{"snr": 10.0}, {"snr": 2.0}],
        )
        sd_out, results = curate(sd, min_spikes=1, min_snr=5.0)
        assert sd_out.N == 1
        assert "snr" in results


# ---------------------------------------------------------------------------
# build_curation_history
# ---------------------------------------------------------------------------


class TestBuildCurationHistory:
    def test_basic_structure(self):
        """
        History dict has all required top-level keys.

        Tests:
            (Test Case 1) All expected keys are present.
            (Test Case 2) initial contains all original unit IDs.
            (Test Case 3) curated_final contains only surviving unit IDs.
        """
        sd = _make_sd_varied()
        sd_out, results = curate(sd, min_spikes=10)
        history = build_curation_history(sd, sd_out, results)

        assert set(history.keys()) == {
            "curation_parameters",
            "initial",
            "curations",
            "curated",
            "failed",
            "metrics",
            "curated_final",
        }
        assert history["initial"] == [10, 20, 30, 40]
        assert history["curated_final"] == [10, 30]
        assert history["curations"] == ["spike_count"]

    def test_curated_and_failed_partition(self):
        """
        Curated and failed lists partition the input units for each
        criterion.

        Tests:
            (Test Case 1) Union of curated and failed equals the input
                units for that stage.
        """
        sd = _make_sd_varied()
        sd_out, results = curate(sd, min_spikes=10)
        history = build_curation_history(sd, sd_out, results)

        c = set(history["curated"]["spike_count"])
        f = set(history["failed"]["spike_count"])
        assert c | f == set(history["initial"])
        assert c & f == set()

    def test_metrics_per_unit(self):
        """
        Metrics dict maps unit IDs to float metric values.

        Tests:
            (Test Case 1) Every unit in the stage input has a metric entry.
        """
        sd = _make_sd_varied()
        sd_out, results = curate(sd, min_spikes=10)
        history = build_curation_history(sd, sd_out, results)

        m = history["metrics"]["spike_count"]
        # All 4 original units should have metrics (stage input = all)
        assert len(m) == 4

    def test_parameters_stored(self):
        """
        Custom parameters dict is stored in the history.

        Tests:
            (Test Case 1) Parameters dict is preserved as-is.
        """
        sd = _make_sd_varied()
        sd_out, results = curate(sd, min_spikes=10)
        params = {"min_spikes": 10, "source": "test"}
        history = build_curation_history(sd, sd_out, results, parameters=params)
        assert history["curation_parameters"] == params

    def test_multi_stage_history(self):
        """
        Multi-criterion curation produces correct per-stage history.

        Tests:
            (Test Case 1) Second stage metrics only cover units surviving
                the first stage.
            (Test Case 2) Curations list has entries in order.
        """
        sd = _make_sd_varied()  # units: 50, 5, 100, 2 spikes
        sd_out, results = curate(sd, min_spikes=10, min_rate_hz=60.0)
        history = build_curation_history(sd, sd_out, results)

        assert history["curations"] == ["spike_count", "firing_rate"]
        # spike_count: 4 units → 2 pass (50, 100)
        assert len(history["metrics"]["spike_count"]) == 4
        # firing_rate: 2 survivors → only 2 have metrics
        assert len(history["metrics"]["firing_rate"]) == 2

    def test_fallback_to_positional_indices(self):
        """
        When neuron_attributes has no unit_id, positional indices are
        used.

        Tests:
            (Test Case 1) initial contains [0, 1, 2, ...] when no
                unit_id attribute exists.
        """
        sd = _make_sd(n_units=3)
        sd_out, results = curate(sd, min_spikes=1)
        history = build_curation_history(sd, sd_out, results)
        assert history["initial"] == [0, 1, 2]

    def test_spikedata_static_method(self):
        """
        build_curation_history is accessible as SpikeData static method.

        Tests:
            (Test Case 1) SpikeData.build_curation_history returns the
                same result as the standalone function.
        """
        sd = _make_sd_varied()
        sd_out, results = curate(sd, min_spikes=10)
        history = SpikeData.build_curation_history(sd, sd_out, results)
        assert "curated_final" in history


# ---------------------------------------------------------------------------
# SpikeData method bindings
# ---------------------------------------------------------------------------


class TestSpikeDataCurationMethods:
    def test_curate_by_min_spikes_method(self):
        """
        SpikeData.curate_by_min_spikes delegates to the curation module.

        Tests:
            (Test Case 1) Method returns same result as standalone function.
        """
        sd = _make_sd_varied()
        sd_out, res = sd.curate_by_min_spikes(min_spikes=10)
        assert sd_out.N == 2
        assert "metric" in res and "passed" in res

    def test_curate_by_firing_rate_method(self):
        """
        SpikeData.curate_by_firing_rate delegates to the curation module.

        Tests:
            (Test Case 1) Method produces correct firing rate metrics.
        """
        sd = _make_sd_varied()
        sd_out, res = sd.curate_by_firing_rate(min_rate_hz=10.0)
        assert sd_out.N == 2

    def test_curate_by_isi_violations_method(self):
        """
        SpikeData.curate_by_isi_violations delegates to the curation
        module.

        Tests:
            (Test Case 1) Method produces ISI violation metrics.
        """
        sd = SpikeData(
            [np.array([10.0, 11.0, 12.0]), np.array([10.0, 100.0, 200.0])],
            length=300.0,
        )
        sd_out, res = sd.curate_by_isi_violations(max_violation=50.0)
        assert res["metric"].shape == (2,)

    def test_curate_method(self):
        """
        SpikeData.curate delegates to the combined wrapper.

        Tests:
            (Test Case 1) Combined method applies multiple criteria.
        """
        sd = _make_sd_varied()
        sd_out, results = sd.curate(min_spikes=10, min_rate_hz=20.0)
        assert sd_out.N == 2
        assert "spike_count" in results
        assert "firing_rate" in results

    def test_curate_by_snr_method(self):
        """
        SpikeData.curate_by_snr delegates to the curation module.

        Tests:
            (Test Case 1) Method reads SNR from neuron_attributes.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0])],
            length=100.0,
            neuron_attributes=[{"snr": 10.0}],
        )
        sd_out, res = sd.curate_by_snr(min_snr=5.0)
        assert sd_out.N == 1

    def test_curate_by_std_norm_method(self):
        """
        SpikeData.curate_by_std_norm delegates to the curation module.

        Tests:
            (Test Case 1) Method reads std_norm from neuron_attributes.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0])],
            length=100.0,
            neuron_attributes=[{"std_norm": 0.5}],
        )
        sd_out, res = sd.curate_by_std_norm(max_std_norm=1.0)
        assert sd_out.N == 1

    def test_compute_waveform_metrics_method(self):
        """
        SpikeData.compute_waveform_metrics delegates to the curation
        module.

        Tests:
            (Test Case 1) Method computes and stores metrics.
        """
        sd = _make_sd_with_raw()
        sd_out, metrics = sd.compute_waveform_metrics()
        assert "snr" in metrics
        assert "std_norm" in metrics


# ---------------------------------------------------------------------------
# split_epochs
# ---------------------------------------------------------------------------


def _make_concatenated_sd():
    """Build a SpikeData simulating two concatenated recordings.

    Epoch 0: 0–500 ms, Epoch 1: 500–1000 ms.
    Two units with spikes in both epochs and per-epoch templates.
    """
    sd = SpikeData(
        [
            np.array([100.0, 200.0, 600.0, 700.0]),
            np.array([150.0, 550.0, 800.0]),
        ],
        length=1000.0,
        neuron_attributes=[
            {
                "unit_id": 0,
                "template": np.ones(10),
                "epoch_templates": [np.ones(10) * 1.0, np.ones(10) * 2.0],
            },
            {
                "unit_id": 1,
                "template": np.ones(10),
                "epoch_templates": [np.ones(10) * 3.0, np.ones(10) * 4.0],
            },
        ],
        metadata={
            "rec_chunks_ms": [(0.0, 500.0), (500.0, 1000.0)],
            "rec_chunk_names": ["rec_a.raw.h5", "rec_b.raw.h5"],
            "source_format": "Kilosort2",
        },
    )
    return sd


class TestSplitEpochs:
    def test_basic_split(self):
        """
        split_epochs produces one SpikeData per epoch with correct spikes.

        Tests:
            (Test Case 1) Two epochs produce two SpikeData objects.
            (Test Case 2) Each epoch contains only spikes from its time
                range, shifted to start at t=0.
        """
        sd = _make_concatenated_sd()
        epochs = sd.split_epochs()

        assert len(epochs) == 2
        # Epoch 0: spikes at 100, 200 for unit 0; 150 for unit 1
        assert len(epochs[0].train[0]) == 2
        assert len(epochs[0].train[1]) == 1
        # Epoch 1: spikes at 600→100, 700→200 for unit 0; 550→50, 800→300 for unit 1
        assert len(epochs[1].train[0]) == 2
        assert len(epochs[1].train[1]) == 2

    def test_epoch_templates_assigned(self):
        """
        Each epoch SpikeData receives its corresponding epoch template.

        Tests:
            (Test Case 1) Epoch 0 gets epoch_templates[0] as its template.
            (Test Case 2) Epoch 1 gets epoch_templates[1] as its template.
        """
        sd = _make_concatenated_sd()
        epochs = sd.split_epochs()

        assert epochs[0].neuron_attributes[0]["template"].mean() == 1.0
        assert epochs[1].neuron_attributes[0]["template"].mean() == 2.0
        assert epochs[0].neuron_attributes[1]["template"].mean() == 3.0
        assert epochs[1].neuron_attributes[1]["template"].mean() == 4.0

    def test_epoch_templates_list_removed(self):
        """
        The epoch_templates list is removed from individual epoch
        SpikeData objects.

        Tests:
            (Test Case 1) No epoch has epoch_templates in neuron_attributes.
        """
        sd = _make_concatenated_sd()
        epochs = sd.split_epochs()

        for ep in epochs:
            for attrs in ep.neuron_attributes:
                assert "epoch_templates" not in attrs

    def test_source_file_labels(self):
        """
        Each epoch SpikeData is labeled with its source file name.

        Tests:
            (Test Case 1) metadata["source_file"] matches the chunk name.
            (Test Case 2) metadata["epoch_index"] is set correctly.
        """
        sd = _make_concatenated_sd()
        epochs = sd.split_epochs()

        assert epochs[0].metadata["source_file"] == "rec_a.raw.h5"
        assert epochs[1].metadata["source_file"] == "rec_b.raw.h5"
        assert epochs[0].metadata["epoch_index"] == 0
        assert epochs[1].metadata["epoch_index"] == 1

    def test_concatenation_metadata_removed(self):
        """
        Concatenation-specific metadata is removed from epoch SpikeData.

        Tests:
            (Test Case 1) rec_chunks_ms, rec_chunks_frames, and
                rec_chunk_names are not present in epoch metadata.
        """
        sd = _make_concatenated_sd()
        epochs = sd.split_epochs()

        for ep in epochs:
            assert "rec_chunks_ms" not in ep.metadata
            assert "rec_chunks_frames" not in ep.metadata
            assert "rec_chunk_names" not in ep.metadata

    def test_original_unchanged(self):
        """
        Splitting does not modify the original SpikeData.

        Tests:
            (Test Case 1) Original neuron_attributes still contain
                epoch_templates.
            (Test Case 2) Original metadata still has rec_chunks_ms.
        """
        sd = _make_concatenated_sd()
        sd.split_epochs()

        assert "epoch_templates" in sd.neuron_attributes[0]
        assert "rec_chunks_ms" in sd.metadata

    def test_independent_attributes(self):
        """
        Epoch SpikeData objects have independent neuron_attributes
        (modifying one does not affect others).

        Tests:
            (Test Case 1) Changing epoch 0's template does not affect
                epoch 1's template.
        """
        sd = _make_concatenated_sd()
        epochs = sd.split_epochs()

        epochs[0].neuron_attributes[0]["template"] = np.zeros(10)
        assert epochs[1].neuron_attributes[0]["template"].mean() == 2.0

    def test_no_epochs_raises(self):
        """
        ValueError raised when SpikeData has no epoch boundaries.

        Tests:
            (Test Case 1) SpikeData without rec_chunks_ms raises.
        """
        sd = _make_sd(n_units=2)
        with pytest.raises(ValueError, match="No epoch boundaries"):
            sd.split_epochs()

    def test_preserved_metadata(self):
        """
        Non-concatenation metadata is preserved in epoch SpikeData.

        Tests:
            (Test Case 1) source_format is carried through.
        """
        sd = _make_concatenated_sd()
        epochs = sd.split_epochs()

        for ep in epochs:
            assert ep.metadata["source_format"] == "Kilosort2"


# ---------------------------------------------------------------------------
# Merge-based deduplication: helpers
# ---------------------------------------------------------------------------

_WF_SAMPLES = 30
_WF_CHANNELS = [0, 1]


def _make_waveform(peak_sample=15, amplitude=-5.0):
    """Return a (2, 30) avg_waveform with a peak at peak_sample."""
    wf = np.zeros((2, _WF_SAMPLES))
    wf[0, peak_sample] = amplitude
    wf[1, peak_sample] = amplitude * 0.6
    return wf


def _make_sd_with_positions(positions, spike_counts=None, length=1000.0, seed=0):
    """SpikeData with 'location' in neuron_attributes.

    positions : list of [x, y] pairs, one per unit.
    spike_counts : list of int (defaults to 50 per unit).
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    counts = spike_counts or [50] * n
    trains = [np.sort(rng.uniform(10.0, length - 10.0, c)) for c in counts]
    attrs = [{"unit_id": i, "location": list(positions[i])} for i in range(n)]
    return SpikeData(trains, length=length, neuron_attributes=attrs)


def _make_sd_with_waveforms(
    positions, waveforms, spike_counts=None, length=1000.0, seed=0
):
    """SpikeData with 'location', 'avg_waveform', and 'traces_meta' populated.

    positions  : list of [x, y] pairs.
    waveforms  : list of (2, 30) arrays, one per unit.
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    counts = spike_counts or [50] * n
    trains = [np.sort(rng.uniform(10.0, length - 10.0, c)) for c in counts]
    attrs = [
        {
            "unit_id": i,
            "location": list(positions[i]),
            "avg_waveform": waveforms[i],
            "traces_meta": {"channels": _WF_CHANNELS},
        }
        for i in range(n)
    ]
    return SpikeData(trains, length=length, neuron_attributes=attrs)


def _make_duplicate_pair_sd():
    """SpikeData with two near-identical units and one unrelated unit.

    Units 0 and 1: positions 5 µm apart, identical waveforms, spike trains
    with jitter < 0.4 ms (they are genuine duplicates).
    Unit 2: 500 µm away, orthogonal waveform, independent train.
    """
    length = 1000.0
    # Duplicate base train at 10ms ISI (well above 1.5ms threshold)
    base = np.arange(10.0, length, 10.0)
    train0 = base.copy()
    train1 = base + 0.15  # 0.15 ms jitter — within delta_ms=0.4

    # Unrelated unit
    rng = np.random.default_rng(77)
    train2 = np.sort(rng.uniform(10, length - 10, 50))

    wf_dup = _make_waveform(peak_sample=15, amplitude=-5.0)
    wf_other = np.zeros((2, _WF_SAMPLES))
    wf_other[1, 5] = 5.0  # orthogonal to wf_dup

    attrs = [
        {
            "unit_id": 0,
            "location": [0.0, 0.0],
            "avg_waveform": wf_dup.copy(),
            "traces_meta": {"channels": _WF_CHANNELS},
        },
        {
            "unit_id": 1,
            "location": [5.0, 0.0],
            "avg_waveform": wf_dup.copy(),
            "traces_meta": {"channels": _WF_CHANNELS},
        },
        {
            "unit_id": 2,
            "location": [500.0, 0.0],
            "avg_waveform": wf_other,
            "traces_meta": {"channels": _WF_CHANNELS},
        },
    ]
    return SpikeData(
        [train0, train1, train2],
        length=length,
        neuron_attributes=attrs,
    )


# ---------------------------------------------------------------------------
# find_nearby_unit_pairs
# ---------------------------------------------------------------------------


class TestFindNearbyUnitPairs:
    def test_nearby_pairs_detected(self):
        """
        Unit pairs within dist_um are returned; distant pairs are not.

        Tests:
            (Test Case 1) Two units 10 µm apart are included in pairs.
            (Test Case 2) A unit 500 µm away is not paired with others.
            (Test Case 3) Pairs are (i, j) with i < j.
        """
        sd = _make_sd_with_positions([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0]])
        pairs = find_nearby_unit_pairs(sd, dist_um=24.8)

        assert (0, 1) in pairs
        assert (0, 2) not in pairs
        assert (1, 2) not in pairs
        for i, j in pairs:
            assert i < j

    def test_all_nearby(self):
        """
        All pairs returned when all units are within dist_um.

        Tests:
            (Test Case 1) Three mutually nearby units yield all three pairs.
        """
        sd = _make_sd_with_positions([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        pairs = find_nearby_unit_pairs(sd, dist_um=24.8)
        assert {(0, 1), (0, 2), (1, 2)} == pairs

    def test_no_pairs_when_all_far(self):
        """
        Empty set returned when all units are beyond dist_um.

        Tests:
            (Test Case 1) Units 100 µm apart produce no pairs at 24.8 µm threshold.
        """
        sd = _make_sd_with_positions([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
        pairs = find_nearby_unit_pairs(sd, dist_um=24.8)
        assert pairs == set()

    def test_no_locations_raises(self):
        """
        ValueError raised when sd.locations is None.

        Tests:
            (Test Case 1) SpikeData without position data raises ValueError.
        """
        sd = _make_sd(n_units=2)
        with pytest.raises(ValueError, match="unit_locations is None"):
            find_nearby_unit_pairs(sd, dist_um=24.8)

    def test_boundary_distance(self):
        """
        Pair at exactly dist_um is included; pair just beyond is not.

        Tests:
            (Test Case 1) Distance == dist_um is included (<=).
            (Test Case 2) Distance == dist_um + epsilon is excluded.
        """
        d = 24.8
        sd_on = _make_sd_with_positions([[0.0, 0.0], [d, 0.0]])
        sd_over = _make_sd_with_positions([[0.0, 0.0], [d + 0.01, 0.0]])

        assert (0, 1) in find_nearby_unit_pairs(sd_on, dist_um=d)
        assert (0, 1) not in find_nearby_unit_pairs(sd_over, dist_um=d)

    def test_2d_positions_only(self):
        """
        Only x and y coordinates are used for distance; z is ignored.

        Tests:
            (Test Case 1) Units at same xy but different z are still paired.
        """
        trains = [np.array([10.0, 20.0, 30.0])] * 2
        attrs = [
            {"location": [0.0, 0.0, 0.0]},
            {"location": [5.0, 0.0, 999.0]},
        ]
        sd = SpikeData(trains, length=100.0, neuron_attributes=attrs)
        pairs = find_nearby_unit_pairs(sd, dist_um=24.8)
        assert (0, 1) in pairs


# ---------------------------------------------------------------------------
# filter_pairs_by_isi_violations
# ---------------------------------------------------------------------------


class TestFilterPairsByIsiViolations:
    def test_filters_violating_units(self):
        """
        Pairs containing a high-violation unit are removed.

        Tests:
            (Test Case 1) Pair where one unit has ISI violations > threshold
                is excluded.
            (Test Case 2) Pair where both units are clean is retained.
            (Test Case 3) violation_rates dict is returned for all units in
                the input pairs.
        """
        # Unit 0: spikes 1 ms apart — all violate 1.5 ms threshold
        # Unit 1: spikes 10 ms apart — no violations
        # Unit 2: spikes 10 ms apart — no violations
        sd = SpikeData(
            [
                np.arange(10.0, 200.0, 1.0),
                np.arange(10.0, 200.0, 10.0),
                np.arange(15.0, 200.0, 10.0),
            ],
            length=200.0,
        )
        pairs = {(0, 1), (1, 2)}
        filtered, rates = filter_pairs_by_isi_violations(
            sd, pairs, max_violation_rate=0.04, threshold_ms=1.5
        )

        assert (0, 1) not in filtered  # unit 0 violates
        assert (1, 2) in filtered  # both clean
        assert set(rates.keys()) == {0, 1, 2}
        assert rates[0] > 0.04
        assert rates[1] == pytest.approx(0.0)

    def test_both_must_pass(self):
        """
        Both units in a pair must pass the threshold.

        Tests:
            (Test Case 1) Pair (0, 1) is excluded when unit 0 violates even
                if unit 1 is clean.
            (Test Case 2) Pair (0, 1) is excluded when unit 1 violates even
                if unit 0 is clean.
        """
        clean = np.arange(10.0, 500.0, 10.0)
        dirty = np.arange(10.0, 500.0, 1.0)
        sd_a = SpikeData([dirty, clean], length=500.0)
        sd_b = SpikeData([clean, dirty], length=500.0)

        for sd in (sd_a, sd_b):
            filtered, _ = filter_pairs_by_isi_violations(
                sd, {(0, 1)}, max_violation_rate=0.04
            )
            assert (0, 1) not in filtered


# ---------------------------------------------------------------------------
# compute_pairwise_similarity
# ---------------------------------------------------------------------------


class TestComputePairwiseSimilarity:
    def test_identical_waveforms_give_similarity_one(self):
        """
        Identical waveforms produce cosine similarity of 1.0.

        Tests:
            (Test Case 1) sim_mat[i, j] == 1.0 for units with the same
                avg_waveform.
        """
        wf = _make_waveform()
        sd = _make_sd_with_waveforms(
            positions=[[0.0, 0.0], [5.0, 0.0]],
            waveforms=[wf.copy(), wf.copy()],
        )
        sim_mat, lag_mat, _ = compute_pairwise_similarity(sd, {(0, 1)}, max_lag=10)

        assert sim_mat[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert sim_mat[1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_waveforms_give_low_similarity(self):
        """
        Orthogonal waveforms produce near-zero cosine similarity.

        Tests:
            (Test Case 1) sim_mat[i, j] is close to 0 for orthogonal waveforms.
        """
        wf_a = np.zeros((2, _WF_SAMPLES))
        wf_a[0, 10] = 1.0
        wf_b = np.zeros((2, _WF_SAMPLES))
        wf_b[1, 20] = 1.0

        sd = _make_sd_with_waveforms(
            positions=[[0.0, 0.0], [5.0, 0.0]],
            waveforms=[wf_a, wf_b],
        )
        sim_mat, _, _ = compute_pairwise_similarity(sd, {(0, 1)}, max_lag=0)
        assert abs(sim_mat[0, 1]) < 0.1

    def test_output_shapes(self):
        """
        Similarity and lag matrices are (N, N) with correct diagonals.

        Tests:
            (Test Case 1) sim_mat shape is (N, N).
            (Test Case 2) Diagonal of sim_mat is 1.0.
            (Test Case 3) Diagonal of lag_mat is 0.0.
            (Test Case 4) Unevaluated entries are NaN.
        """
        wf = _make_waveform()
        sd = _make_sd_with_waveforms(
            positions=[[0.0, 0.0], [5.0, 0.0], [500.0, 0.0]],
            waveforms=[wf.copy(), wf.copy(), wf.copy()],
        )
        sim_mat, lag_mat, unit_ids = compute_pairwise_similarity(
            sd, {(0, 1)}, max_lag=10
        )

        assert sim_mat.shape == (3, 3)
        assert lag_mat.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(sim_mat), [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(np.diag(lag_mat), [0.0, 0.0, 0.0])
        # Pair (0,2) not in pairs → NaN
        assert np.isnan(sim_mat[0, 2])
        assert len(unit_ids) == 3

    def test_no_avg_waveform_raises(self):
        """
        ValueError raised when no units have avg_waveform.

        Tests:
            (Test Case 1) Error message mentions avg_waveform.
        """
        sd = _make_sd_with_positions([[0.0, 0.0], [5.0, 0.0]])
        with pytest.raises(ValueError, match="avg_waveform"):
            compute_pairwise_similarity(sd, {(0, 1)})

    def test_no_neuron_attributes_raises(self):
        """
        ValueError raised when neuron_attributes is None.

        Tests:
            (Test Case 1) SpikeData without neuron_attributes raises ValueError.
        """
        sd = _make_sd(n_units=2)
        with pytest.raises(ValueError, match="neuron_attributes is None"):
            compute_pairwise_similarity(sd, {(0, 1)})

    def test_lag_boundary_sets_sim_to_zero(self):
        """
        Similarity is set to 0 when best lag hits the max_lag boundary.

        Tests:
            (Test Case 1) sim_mat[i, j] == 0.0 when max_lag=0 forces a
                boundary result for shifted waveforms.
        """
        wf_a = np.zeros((1, _WF_SAMPLES))
        wf_a[0, 5] = 1.0
        wf_b = np.zeros((1, _WF_SAMPLES))
        wf_b[0, 25] = 1.0  # shifted far from wf_a

        attrs = [
            {
                "unit_id": 0,
                "avg_waveform": wf_a,
                "traces_meta": {"channels": [0]},
            },
            {
                "unit_id": 1,
                "avg_waveform": wf_b,
                "traces_meta": {"channels": [0]},
            },
        ]
        trains = [np.array([10.0, 20.0, 30.0])] * 2
        sd = SpikeData(trains, length=100.0, neuron_attributes=attrs)
        # max_lag=0 means no shifting allowed — best_lag will == max_lag
        sim_mat, _, _ = compute_pairwise_similarity(sd, {(0, 1)}, max_lag=0)
        assert sim_mat[0, 1] == pytest.approx(0.0)

# ---------------------------------------------------------------------------
# filter_by_cosine_sim
# ---------------------------------------------------------------------------


class TestFilterByCosineSim:
    def test_threshold_filtering(self):
        """
        Pairs below the cosine threshold are excluded.

        Tests:
            (Test Case 1) Pair with similarity >= threshold is retained.
            (Test Case 2) Pair with similarity < threshold is excluded.
        """
        sim_mat = np.full((3, 3), np.nan)
        sim_mat[0, 1] = sim_mat[1, 0] = 0.95
        sim_mat[0, 2] = sim_mat[2, 0] = 0.70

        pairs = {(0, 1), (0, 2)}
        filtered = filter_by_cosine_sim(pairs, sim_mat, threshold=0.9)

        assert (0, 1) in filtered
        assert (0, 2) not in filtered

    def test_nan_excluded(self):
        """
        Pairs with NaN similarity are excluded.

        Tests:
            (Test Case 1) A NaN entry in sim_mat causes the pair to be
                dropped regardless of threshold.
        """
        sim_mat = np.full((2, 2), np.nan)
        filtered = filter_by_cosine_sim({(0, 1)}, sim_mat, threshold=0.0)
        assert filtered == set()

# ---------------------------------------------------------------------------
# merge_redundant_units
# ---------------------------------------------------------------------------


class TestMergeRedundantUnits:
    def _make_merge_inputs(self):
        """Return (sd, pairs, sim_mat) for a clear duplicate pair."""
        sd = _make_duplicate_pair_sd()
        sim_mat = np.full((3, 3), np.nan)
        np.fill_diagonal(sim_mat, 1.0)
        sim_mat[0, 1] = sim_mat[1, 0] = 1.0  # identical waveforms
        return sd, {(0, 1)}, sim_mat

    def test_merges_duplicate_pair(self):
        """
        Two duplicate units are merged into one; total units decreases.

        Tests:
            (Test Case 1) Output SpikeData has N = original N - 1.
            (Test Case 2) merged_pairs list contains the accepted pair.
            (Test Case 3) n_removed == 1.
        """
        sd, pairs, sim_mat = self._make_merge_inputs()
        sd_out, result = merge_redundant_units(sd, pairs, sim_mat)

        assert sd_out.N == sd.N - 1
        assert len(result["merged_pairs"]) == 1
        assert result["n_removed"] == 1

    def test_merged_train_deduplicates_spikes(self):
        """
        Merged train removes near-coincident spikes within delta_ms.

        Tests:
            (Test Case 1) Merged train has fewer spikes than the naive
                concatenation of the two trains.
            (Test Case 2) Merged train is sorted.
        """
        sd, pairs, sim_mat = self._make_merge_inputs()
        sd_out, _ = merge_redundant_units(sd, pairs, sim_mat, delta_ms=0.4)

        naive_count = len(sd.train[0]) + len(sd.train[1])
        merged_count = len(sd_out.train[0])
        assert merged_count < naive_count
        assert np.all(np.diff(sd_out.train[0]) > 0)

    def test_preserves_start_time(self):
        """
        Output SpikeData preserves start_time from the source.

        Tests:
            (Test Case 1) sd_out.start_time == sd.start_time.
        """
        sd, pairs, sim_mat = self._make_merge_inputs()
        sd_out, _ = merge_redundant_units(sd, pairs, sim_mat)
        assert sd_out.start_time == sd.start_time

    def test_merged_from_attribute_set(self):
        """
        merged_from neuron attribute records which units were combined.

        Tests:
            (Test Case 1) Primary unit's merged_from contains both
                original unit IDs.
        """
        sd, pairs, sim_mat = self._make_merge_inputs()
        sd_out, _ = merge_redundant_units(sd, pairs, sim_mat)

        primary_new_idx = 0  # sorted output — primary was unit 0 or 1
        merged_from = sd_out.neuron_attributes[primary_new_idx]["merged_from"]
        assert len(merged_from) == 2

    def test_isi_guard_rejects_bad_merge(self):
        """
        Merge is rejected when it would excessively increase ISI violations.

        Tests:
            (Test Case 1) No pairs accepted when merged train would violate
                max_isi_increase.
            (Test Case 2) n_removed == 0 and output N == input N.
        """
        # Two units at same location, waveforms identical, but trains that
        # interleave at sub-threshold ISI after merging.
        # Train 0: spikes at every 1.0 ms — already violates 1.5 ms.
        # Train 1: same spikes + 0.5 ms shift — merged result still very high ISI.
        # Use max_isi_increase=0.0 to guarantee rejection.
        length = 200.0
        train0 = np.arange(10.0, length, 1.0)
        train1 = np.arange(10.5, length, 1.0)
        wf = _make_waveform()
        attrs = [
            {
                "unit_id": 0,
                "location": [0.0, 0.0],
                "avg_waveform": wf.copy(),
                "traces_meta": {"channels": _WF_CHANNELS},
            },
            {
                "unit_id": 1,
                "location": [5.0, 0.0],
                "avg_waveform": wf.copy(),
                "traces_meta": {"channels": _WF_CHANNELS},
            },
        ]
        sd = SpikeData([train0, train1], length=length, neuron_attributes=attrs)
        sim_mat = np.array([[1.0, 1.0], [1.0, 1.0]])

        sd_out, result = merge_redundant_units(
            sd, {(0, 1)}, sim_mat, max_isi_increase=0.0
        )
        assert result["n_removed"] == 0
        assert sd_out.N == sd.N

    def test_empty_pairs_raises(self):
        """
        ValueError raised when pairs is empty.

        Tests:
            (Test Case 1) Empty pairs set raises ValueError.
        """
        sd = _make_duplicate_pair_sd()
        sim_mat = np.eye(sd.N)
        with pytest.raises(ValueError):
            merge_redundant_units(sd, set(), sim_mat)

    def test_unrelated_unit_unchanged(self):
        """
        Units not involved in any merge are passed through unchanged.

        Tests:
            (Test Case 1) Train of the unrelated third unit is preserved
                exactly in the output.
        """
        sd, pairs, sim_mat = self._make_merge_inputs()
        sd_out, _ = merge_redundant_units(sd, pairs, sim_mat)

        # Unit 2 (index 2 in input) should survive as the last unit in output
        # (sorted by original index). Its train must match exactly.
        np.testing.assert_array_equal(sd_out.train[-1], sd.train[2])


# ---------------------------------------------------------------------------
# curate_by_merge_duplicates (full pipeline)
# ---------------------------------------------------------------------------


class TestCurateByMergeDuplicates:
    def test_merges_duplicate_pair(self):
        """
        Full pipeline merges a genuine duplicate pair.

        Tests:
            (Test Case 1) Output has fewer units than input.
            (Test Case 2) result_dict has 'metric' and 'passed' arrays.
            (Test Case 3) 'passed' is False for the absorbed unit.
        """
        sd = _make_duplicate_pair_sd()
        sd_out, result = curate_by_merge_duplicates(
            sd, dist_um=24.8, cosine_threshold=0.9
        )

        assert sd_out.N < sd.N
        assert result["metric"].shape == (sd.N,)
        assert result["passed"].shape == (sd.N,)
        assert result["passed"].dtype == bool
        assert not np.all(result["passed"])

    def test_no_nearby_pairs_returns_unchanged(self):
        """
        Returns the original SpikeData unchanged when no pairs are nearby.

        Tests:
            (Test Case 1) N is unchanged.
            (Test Case 2) All units pass (passed is all True).
            (Test Case 3) All metrics are 0.0.
        """
        sd = _make_sd_with_waveforms(
            positions=[[0.0, 0.0], [500.0, 0.0], [1000.0, 0.0]],
            waveforms=[_make_waveform()] * 3,
        )
        sd_out, result = curate_by_merge_duplicates(sd, dist_um=24.8)

        assert sd_out.N == sd.N
        assert np.all(result["passed"])
        assert np.all(result["metric"] == 0.0)

    def test_result_dict_structure(self):
        """
        result_dict always has 'metric' and 'passed' with shape (N,).

        Tests:
            (Test Case 1) Keys 'metric' and 'passed' are present.
            (Test Case 2) Arrays have length equal to input N.
        """
        sd = _make_duplicate_pair_sd()
        _, result = curate_by_merge_duplicates(sd)

        assert set(result.keys()) == {"metric", "passed"}
        assert len(result["metric"]) == sd.N
        assert len(result["passed"]) == sd.N

    def test_metric_reflects_similarity(self):
        """
        metric[i] reflects cosine similarity of the accepted merge.

        Tests:
            (Test Case 1) Absorbed unit has metric > 0.
            (Test Case 2) Unrelated unit has metric == 0.
        """
        sd = _make_duplicate_pair_sd()
        _, result = curate_by_merge_duplicates(sd, dist_um=24.8, cosine_threshold=0.9)

        # At least one unit should have metric > 0 (was merged)
        assert np.any(result["metric"] > 0)
        # The unrelated unit (index 2, far away) has metric 0
        assert result["metric"][2] == pytest.approx(0.0)

    def test_spikedata_method_binding(self):
        """
        curate_by_merge_duplicates is accessible as sd.curate_by_merge_duplicates().

        Tests:
            (Test Case 1) Method call returns same result structure as
                standalone function.
        """
        sd = _make_duplicate_pair_sd()
        sd_out, result = sd.curate_by_merge_duplicates(dist_um=24.8)

        assert isinstance(sd_out, SpikeData)
        assert "metric" in result
        assert "passed" in result

    def test_no_locations_raises(self):
        """
        ValueError propagates when SpikeData has no location data.

        Tests:
            (Test Case 1) SpikeData without positions raises ValueError
                from find_nearby_unit_pairs.
        """
        sd = _make_sd(n_units=3)
        with pytest.raises(ValueError, match="unit_locations is None"):
            curate_by_merge_duplicates(sd)
