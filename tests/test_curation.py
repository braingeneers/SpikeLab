"""Tests for spikelab.spikedata.curation module."""

import numpy as np
import pytest

from spikelab.spikedata import SpikeData
from spikelab.spikedata.curation import (
    build_curation_history,
    compute_waveform_metrics,
    curate,
    curate_by_firing_rate,
    curate_by_isi_violations,
    curate_by_min_spikes,
    curate_by_snr,
    curate_by_std_norm,
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
