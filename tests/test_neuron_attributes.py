"""
Tests for the NeuronAttributes module.
"""

import numpy as np
import pandas as pd
import pytest

from spikedata import NeuronAttributes, SpikeData


class TestNeuronAttributesCreation:
    """Test creation and initialization of NeuronAttributes."""

    def test_from_dict(self):
        """Test creating NeuronAttributes from dictionary."""
        data = {
            "unit_id": [101, 102, 103],
            "firing_rate_hz": [5.2, 8.1, 3.4],
        }
        attrs = NeuronAttributes.from_dict(data, n_neurons=3)
        assert len(attrs) == 3
        assert attrs.n_neurons == 3
        np.testing.assert_array_equal(attrs.get_attribute("unit_id"), [101, 102, 103])

    def test_from_dataframe(self):
        """Test creating NeuronAttributes from pandas DataFrame."""
        df = pd.DataFrame(
            {
                "unit_id": [1, 2, 3],
            }
        )
        attrs = NeuronAttributes.from_dataframe(df)
        assert len(attrs) == 3
        np.testing.assert_array_equal(attrs.get_attribute("unit_id"), [1, 2, 3])

    def test_validation_n_neurons_mismatch(self):
        """Test that validation catches neuron count mismatch."""
        data = {"unit_id": [1, 2]}
        with pytest.raises(ValueError, match="2 rows but expected 3"):
            NeuronAttributes.from_dict(data, n_neurons=3)

    def test_empty_attributes(self):
        """Test creating empty NeuronAttributes."""
        df = pd.DataFrame()
        attrs = NeuronAttributes.from_dataframe(df, n_neurons=0)
        assert len(attrs) == 0


class TestNeuronAttributesOperations:
    """Test operations on NeuronAttributes."""

    def test_set_and_get_attribute(self):
        """Test setting and getting attributes."""
        attrs = NeuronAttributes.from_dict({"unit_id": [1, 2, 3]}, n_neurons=3)
        attrs.set_attribute("quality", ["good", "mua", "good"])
        quality = attrs.get_attribute("quality")
        assert list(quality) == ["good", "mua", "good"]

    def test_get_nonexistent_attribute(self):
        """Test getting attribute that doesn't exist."""
        attrs = NeuronAttributes.from_dict({"unit_id": [1, 2]}, n_neurons=2)
        with pytest.raises(KeyError, match="'nonexistent'"):
            attrs.get_attribute("nonexistent")

    def test_set_attribute_wrong_length(self):
        """Test that setting attribute with wrong length raises error."""
        attrs = NeuronAttributes.from_dict({"unit_id": [1, 2, 3]}, n_neurons=3)
        with pytest.raises(ValueError, match="Values length"):
            attrs.set_attribute(
                "quality", ["good", "mua"]
            )  # Only 2 values for 3 neurons

    def test_subset(self):
        """Test subsetting neurons."""
        attrs = NeuronAttributes.from_dict(
            {"unit_id": [1, 2, 3, 4]}, n_neurons=4
        )
        subset_attrs = attrs.subset([0, 2])
        assert len(subset_attrs) == 2
        np.testing.assert_array_equal(subset_attrs.get_attribute("unit_id"), [1, 3])

    def test_concat(self):
        """Test concatenating NeuronAttributes."""
        attrs1 = NeuronAttributes.from_dict({"unit_id": [1, 2]}, n_neurons=2)
        attrs2 = NeuronAttributes.from_dict({"unit_id": [3, 4]}, n_neurons=2)
        combined = attrs1.concat(attrs2)
        assert len(combined) == 4
        np.testing.assert_array_equal(combined.get_attribute("unit_id"), [1, 2, 3, 4])

    def test_concat_different_columns(self):
        """Test concatenating with different columns fills with NaN."""
        attrs1 = NeuronAttributes.from_dict(
            {"unit_id": [1, 2], "snr": [4.0, 5.0]}, n_neurons=2
        )
        attrs2 = NeuronAttributes.from_dict(
            {"unit_id": [3, 4], "amplitude": [100.0, 150.0]}, n_neurons=2
        )
        combined = attrs1.concat(attrs2)
        assert len(combined) == 4
        df = combined.to_dataframe()
        assert "snr" in df.columns
        assert "amplitude" in df.columns
        assert pd.isna(df.loc[2, "snr"])  # Should be NaN for neurons from attrs2


class TestNeuronAttributesValidation:
    """Test validation features."""

    def test_column_name_typo_warning(self):
        """Test that similar column names trigger warning."""
        data = {"firing_rate_Hz": [1.0, 2.0]}  # Wrong case
        with pytest.warns(UserWarning, match="firing_rate_hz"):
            NeuronAttributes.from_dict(data, n_neurons=2)

    def test_standard_columns_no_warning(self):
        """Test that standard column names don't trigger warnings."""
        data = {
            "unit_id": [1, 2],
            "firing_rate_hz": [5.0, 10.0],
            "snr": [3.5, 4.2],
        }
        # Should not raise any warnings
        attrs = NeuronAttributes.from_dict(data, n_neurons=2)
        assert len(attrs) == 2


class TestNeuronAttributesDataFrame:
    """Test DataFrame conversion."""

    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        data = {
            "unit_id": [1, 2, 3],
            "firing_rate_hz": [5.2, 8.1, 3.4],
        }
        attrs = NeuronAttributes.from_dict(data, n_neurons=3)
        df = attrs.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["unit_id", "firing_rate_hz"]

    def test_dataframe_round_trip(self):
        """Test that DataFrame round-trip preserves data."""
        original_df = pd.DataFrame(
            {
                "unit_id": [1, 2, 3],
                "quality": ["good", "mua", "good"],
                "snr": [4.2, 3.1, 5.5],
            }
        )
        attrs = NeuronAttributes.from_dataframe(original_df, n_neurons=3)
        result_df = attrs.to_dataframe()
        pd.testing.assert_frame_equal(result_df, original_df, check_dtype=False)


class TestNeuronAttributesRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        attrs = NeuronAttributes.from_dict(
            {"unit_id": [1, 2]}, n_neurons=2
        )
        repr_str = repr(attrs)
        assert "NeuronAttributes" in repr_str
        assert "2 neurons" in repr_str
        assert "1 attribute" in repr_str

    def test_len(self):
        """Test __len__ method."""
        attrs = NeuronAttributes.from_dict({"unit_id": [1, 2, 3]}, n_neurons=3)
        assert len(attrs) == 3


class TestISIStatistics:
    """Test ISI statistics computation methods."""

    def test_compute_isi_statistics_basic(self):
        """Test basic ISI statistics computation."""
        # Create regular firing neuron (constant ISI)
        regular_train = [np.arange(0, 1000, 100)]  # Fires every 100ms
        sd = SpikeData(regular_train, length=1000)
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        # Compute ISI stats
        isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)

        # Check that stats were computed
        assert "mean_isi_ms" in isi_stats
        assert "cv_isi" in isi_stats
        assert "burst_index" in isi_stats

        # Regular firing should have CV near 0
        assert isi_stats["cv_isi"][0] < 0.1
        # Mean ISI should be 100ms
        assert np.isclose(isi_stats["mean_isi_ms"][0], 100.0)
        # Should have no bursts
        assert isi_stats["burst_index"][0] < 0.01
        # Should have no refractory violations
        assert isi_stats["refractory_violations"][0] == 0

    def test_compute_isi_statistics_auto_save(self):
        """Test that ISI statistics are auto-saved to neuron_attributes."""
        train = [np.array([0, 100, 200, 300])]
        sd = SpikeData(train, length=400)
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        # Compute with auto_save=True (default)
        sd.neuron_attributes.compute_isi_statistics(sd)

        # Check that attributes were saved
        assert sd.neuron_attributes.get_attribute("mean_isi_ms") is not None
        assert sd.neuron_attributes.get_attribute("cv_isi") is not None
        assert sd.neuron_attributes.get_attribute("burst_index") is not None

    def test_compute_isi_statistics_no_save(self):
        """Test ISI statistics computation without saving."""
        train = [np.array([0, 100, 200])]
        sd = SpikeData(train, length=300)
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        # Compute with auto_save=False
        isi_stats = sd.neuron_attributes.compute_isi_statistics(sd, auto_save=False)

        # Results should be returned
        assert "mean_isi_ms" in isi_stats

        # But not saved to attributes
        with pytest.raises(KeyError):
            sd.neuron_attributes.get_attribute("mean_isi_ms")

    def test_compute_isi_statistics_bursting(self):
        """Test ISI statistics for bursting neuron."""
        # Create bursting pattern: clusters of spikes with short ISIs
        bursting_train = [np.array([0, 2, 4, 6, 100, 102, 104, 200, 202, 204])]
        sd = SpikeData(bursting_train, length=300)
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)

        # Bursting neuron should have high CV
        assert isi_stats["cv_isi"][0] > 1.0
        # Should have high burst index (many short ISIs)
        assert isi_stats["burst_index"][0] > 0.5

    def test_compute_isi_statistics_empty_train(self):
        """Test ISI statistics for neuron with no spikes."""
        train = [np.array([])]
        sd = SpikeData(train, length=1000, N=1)
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)

        # Should return NaN for neuron with no spikes
        assert np.isnan(isi_stats["mean_isi_ms"][0])
        assert np.isnan(isi_stats["cv_isi"][0])

    def test_compute_isi_statistics_multiple_neurons(self):
        """Test ISI statistics for multiple neurons."""
        # Regular, bursting, and sparse neurons
        train = [
            np.arange(0, 1000, 100),  # Regular
            np.array([0, 2, 4, 100, 102, 104, 200, 202, 204]),  # Bursting
            np.array([0, 500]),  # Sparse
        ]
        sd = SpikeData(train, length=1000)
        sd.neuron_attributes = NeuronAttributes.from_dict(
            {"unit_id": [0, 1, 2]}, n_neurons=3
        )

        isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)

        # Check shapes
        assert len(isi_stats["cv_isi"]) == 3
        assert len(isi_stats["burst_index"]) == 3

        # Regular neuron should have low CV
        assert isi_stats["cv_isi"][0] < 0.5
        # Bursting neuron should have high burst index
        assert isi_stats["burst_index"][1] > 0.5


class TestLatencyStatistics:
    """Test latency statistics computation methods."""

    def test_compute_latency_statistics_basic(self):
        """Test basic latency statistics computation."""
        # Two neurons with consistent latency relationship
        train = [
            np.array([0, 100, 200, 300]),  # Reference neuron
            np.array([10, 110, 210, 310]),  # Follows with 10ms latency
        ]
        sd = SpikeData(train, length=400)
        sd.neuron_attributes = NeuronAttributes.from_dict(
            {"unit_id": [0, 1]}, n_neurons=2
        )

        # Compute latencies relative to neuron 0
        lat_stats = sd.neuron_attributes.compute_latency_statistics(
            sd, reference_neuron=0, window_ms=50.0
        )

        # Reference neuron should have 0 latency to itself
        assert np.isclose(lat_stats["mean_latency_ms"][0], 0.0)

        # Neuron 1 should have positive latency (follows)
        assert lat_stats["mean_latency_ms"][1] > 0
        assert np.isclose(lat_stats["mean_latency_ms"][1], 10.0, atol=1.0)

        # Should have low jitter (consistent timing)
        assert lat_stats["latency_jitter_ms"][1] < 5.0

    def test_compute_latency_statistics_auto_save(self):
        """Test that latency statistics are auto-saved."""
        train = [np.array([0, 100]), np.array([10, 110])]
        sd = SpikeData(train, length=200)
        sd.neuron_attributes = NeuronAttributes.from_dict(
            {"unit_id": [0, 1]}, n_neurons=2
        )

        sd.neuron_attributes.compute_latency_statistics(sd, reference_neuron=0)

        # Check saved attributes
        assert sd.neuron_attributes.get_attribute("mean_latency_ms") is not None
        assert sd.neuron_attributes.get_attribute("latency_jitter_ms") is not None

    def test_compute_latency_statistics_no_spikes_in_window(self):
        """Test latency statistics when no spikes are within window."""
        train = [
            np.array([0, 1000]),  # Reference
            np.array([500]),  # Far from reference spikes
        ]
        sd = SpikeData(train, length=1500)
        sd.neuron_attributes = NeuronAttributes.from_dict(
            {"unit_id": [0, 1]}, n_neurons=2
        )

        # Use very small window
        lat_stats = sd.neuron_attributes.compute_latency_statistics(
            sd, reference_neuron=0, window_ms=10.0
        )

        # Should have NaN for neuron with no spikes in window
        assert np.isnan(lat_stats["mean_latency_ms"][1])


class TestBurstParticipation:
    """Test burst participation metrics computation."""

    def test_compute_burst_participation_basic(self):
        """Test basic burst participation computation."""
        # Create neurons with different burst participation
        train = [
            np.array([50, 51, 52, 550, 551, 552]),  # Active in both bursts
            np.array([50, 51, 52, 53, 54]),  # Active in first burst only
            np.array([550, 551, 552, 553, 554]),  # Active in second burst only
        ]
        sd = SpikeData(train, length=1000)
        sd.neuron_attributes = NeuronAttributes.from_dict(
            {"unit_id": [0, 1, 2]}, n_neurons=3
        )

        # Define burst edges
        burst_edges = np.array(
            [
                [40, 100],
                [540, 600],
            ]
        )

        burst_stats = sd.neuron_attributes.compute_burst_participation(
            sd, burst_edges=burst_edges, min_spikes=3, backbone_threshold=0.5
        )

        # Neuron 0 should participate in both bursts (2/2 = 1.0)
        assert np.isclose(burst_stats["burst_participation"][0], 1.0)

        # Neurons 1 and 2 should participate in 1/2 bursts (0.5)
        assert np.isclose(burst_stats["burst_participation"][1], 0.5)
        assert np.isclose(burst_stats["burst_participation"][2], 0.5)

        # Only neuron 0 should be backbone (>= 0.5 threshold)
        assert burst_stats["is_backbone_unit"][0] == True
        assert 0 in burst_stats["backbone_indices"]

    def test_compute_burst_participation_auto_save(self):
        """Test that burst participation is auto-saved."""
        train = [np.array([50, 51, 52, 53, 54])]
        sd = SpikeData(train, length=200)
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        burst_edges = np.array([[40, 100]])

        sd.neuron_attributes.compute_burst_participation(sd, burst_edges=burst_edges)

        # Check saved attributes
        assert sd.neuron_attributes.get_attribute("burst_participation") is not None
        assert sd.neuron_attributes.get_attribute("is_backbone_unit") is not None

        # Check metadata
        assert "burst_analysis" in sd.metadata
        assert sd.metadata["burst_analysis"]["n_bursts"] == 1

    def test_compute_burst_participation_min_spikes(self):
        """Test that min_spikes parameter works correctly."""
        train = [
            np.array([50, 51]),  # Only 2 spikes in burst
            np.array([50, 51, 52, 53, 54]),  # 5 spikes in burst
        ]
        sd = SpikeData(train, length=200)
        sd.neuron_attributes = NeuronAttributes.from_dict(
            {"unit_id": [0, 1]}, n_neurons=2
        )

        burst_edges = np.array([[40, 100]])

        # Require at least 3 spikes
        burst_stats = sd.neuron_attributes.compute_burst_participation(
            sd, burst_edges=burst_edges, min_spikes=3
        )

        # Neuron 0 should not be active (only 2 spikes)
        assert burst_stats["burst_participation"][0] == 0.0

        # Neuron 1 should be active (5 spikes >= 3)
        assert burst_stats["burst_participation"][1] == 1.0


class TestMeanWaveforms:
    """Test mean waveform calculation methods."""

    def test_calculate_mean_waveforms_basic(self):
        """Test basic mean waveform calculation."""
        # Create synthetic raw data with 2 channels
        n_channels = 2
        n_samples = 10000
        fs_khz = 20.0  # 20 kHz sampling rate

        # Generate raw data with some structure
        np.random.seed(42)
        raw_data = np.random.randn(n_channels, n_samples) * 0.1

        # Add some spike-like waveforms at specific times
        spike_times_ms = np.array([50, 150, 250])  # in ms
        spike_samples = (spike_times_ms * fs_khz).astype(int)

        # Add spike templates (larger on channel 0)
        for spike_sample in spike_samples:
            if spike_sample < n_samples - 100:
                # Channel 0: large spike
                raw_data[0, spike_sample - 20 : spike_sample + 20] += (
                    np.sin(np.linspace(0, 2 * np.pi, 40)) * 5.0
                )
                # Channel 1: smaller spike
                raw_data[1, spike_sample - 20 : spike_sample + 20] += (
                    np.sin(np.linspace(0, 2 * np.pi, 40)) * 2.0
                )

        # Create SpikeData with spike times and raw data
        train = [spike_times_ms]
        sd = SpikeData(
            train, length=n_samples / fs_khz, raw_data=raw_data, raw_time=fs_khz
        )
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        # Calculate mean waveforms
        wf_data = sd.neuron_attributes.calculate_mean_waveforms(
            sd, ms_before=1.0, ms_after=1.0
        )

        # Verify output structure
        assert "mean_waveforms" in wf_data
        assert "std_waveforms" in wf_data
        assert "best_channels" in wf_data
        assert "n_spikes_used" in wf_data
        assert "time_ms" in wf_data

        # Check shapes
        assert wf_data["mean_waveforms"].shape[0] == 1  # One neuron
        assert (
            wf_data["best_channels"][0] == 0
        )  # Should pick channel 0 (larger amplitude)
        assert wf_data["n_spikes_used"][0] == 3  # Used all 3 spikes

    def test_calculate_mean_waveforms_auto_save(self):
        """Test that mean waveforms are auto-saved to neuron_attributes."""
        # Create simple synthetic data
        n_channels = 1
        n_samples = 5000
        fs_khz = 20.0

        raw_data = np.random.randn(n_channels, n_samples) * 0.1
        spike_times_ms = np.array([50, 150])

        train = [spike_times_ms]
        sd = SpikeData(
            train, length=n_samples / fs_khz, raw_data=raw_data, raw_time=fs_khz
        )
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        # Calculate with auto_save=True (default)
        sd.neuron_attributes.calculate_mean_waveforms(sd)

        # Verify attributes were saved
        assert sd.neuron_attributes.get_attribute("mean_waveform") is not None
        assert sd.neuron_attributes.get_attribute("waveform_channel") is not None
        assert sd.neuron_attributes.get_attribute("waveform_n_spikes") is not None

        # Verify saved values
        saved_waveform = sd.neuron_attributes.get_attribute("mean_waveform")[0]
        assert isinstance(saved_waveform, np.ndarray)
        assert len(saved_waveform) > 0

    def test_calculate_mean_waveforms_no_save(self):
        """Test mean waveform calculation without auto-save."""
        n_channels = 1
        n_samples = 5000
        fs_khz = 20.0

        raw_data = np.random.randn(n_channels, n_samples) * 0.1
        spike_times_ms = np.array([50, 150])

        train = [spike_times_ms]
        sd = SpikeData(
            train, length=n_samples / fs_khz, raw_data=raw_data, raw_time=fs_khz
        )
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        # Calculate with auto_save=False
        wf_data = sd.neuron_attributes.calculate_mean_waveforms(sd, auto_save=False)

        # Results should be returned
        assert "mean_waveforms" in wf_data

        # But not saved to attributes
        with pytest.raises(KeyError):
            sd.neuron_attributes.get_attribute("mean_waveform")

    def test_calculate_mean_waveforms_no_raw_data(self):
        """Test that error is raised when no raw data is available."""
        # Create SpikeData without raw data
        train = [np.array([50, 150, 250])]
        sd = SpikeData(train, length=500)
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        # Should raise ValueError
        with pytest.raises(ValueError, match="raw_data"):
            sd.neuron_attributes.calculate_mean_waveforms(sd)

    def test_calculate_mean_waveforms_empty_spike_train(self):
        """Test mean waveform calculation for neuron with no spikes."""
        n_channels = 1
        n_samples = 5000
        fs_khz = 20.0

        raw_data = np.random.randn(n_channels, n_samples) * 0.1

        # Empty spike train
        train = [np.array([])]
        sd = SpikeData(
            train, length=n_samples / fs_khz, N=1, raw_data=raw_data, raw_time=fs_khz
        )
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        wf_data = sd.neuron_attributes.calculate_mean_waveforms(sd)

        # Should have 0 spikes used
        assert wf_data["n_spikes_used"][0] == 0
        # Waveform should be all zeros
        assert np.all(wf_data["mean_waveforms"][0] == 0)

    def test_calculate_mean_waveforms_multiple_neurons(self):
        """Test mean waveform calculation for multiple neurons."""
        n_channels = 3
        n_samples = 10000
        fs_khz = 20.0

        raw_data = np.random.randn(n_channels, n_samples) * 0.1

        # Different spike times for different neurons
        train = [
            np.array([50, 150]),  # Neuron 0
            np.array([100, 200]),  # Neuron 1
            np.array([75, 175]),  # Neuron 2
        ]

        sd = SpikeData(
            train, length=n_samples / fs_khz, raw_data=raw_data, raw_time=fs_khz
        )
        sd.neuron_attributes = NeuronAttributes.from_dict(
            {"unit_id": [0, 1, 2]}, n_neurons=3
        )

        wf_data = sd.neuron_attributes.calculate_mean_waveforms(sd)

        # Check shapes
        assert wf_data["mean_waveforms"].shape[0] == 3
        assert len(wf_data["best_channels"]) == 3
        assert len(wf_data["n_spikes_used"]) == 3

        # Each neuron should have used 2 spikes
        assert wf_data["n_spikes_used"][0] == 2
        assert wf_data["n_spikes_used"][1] == 2
        assert wf_data["n_spikes_used"][2] == 2

    def test_calculate_mean_waveforms_max_spikes(self):
        """Test that max_spikes parameter limits spike usage."""
        n_channels = 1
        n_samples = 20000
        fs_khz = 20.0

        raw_data = np.random.randn(n_channels, n_samples) * 0.1

        # Create many spike times
        spike_times_ms = np.arange(50, 500, 10)  # 45 spikes
        train = [spike_times_ms]

        sd = SpikeData(
            train, length=n_samples / fs_khz, raw_data=raw_data, raw_time=fs_khz
        )
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        # Limit to 20 spikes
        wf_data = sd.neuron_attributes.calculate_mean_waveforms(sd, max_spikes=20)

        # Should use at most 20 spikes
        assert wf_data["n_spikes_used"][0] <= 20

    def test_calculate_mean_waveforms_best_channel_selection(self):
        """Test that the channel with largest amplitude is selected."""
        n_channels = 3
        n_samples = 5000
        fs_khz = 20.0

        # Create raw data where channel 2 has largest spikes
        raw_data = np.random.randn(n_channels, n_samples) * 0.1

        spike_times_ms = np.array([50, 150])
        spike_samples = (spike_times_ms * fs_khz).astype(int)

        # Add spikes with different amplitudes per channel
        for spike_sample in spike_samples:
            if spike_sample < n_samples - 50:
                raw_data[0, spike_sample - 10 : spike_sample + 10] += (
                    np.sin(np.linspace(0, np.pi, 20)) * 1.0
                )
                raw_data[1, spike_sample - 10 : spike_sample + 10] += (
                    np.sin(np.linspace(0, np.pi, 20)) * 2.0
                )
                raw_data[2, spike_sample - 10 : spike_sample + 10] += (
                    np.sin(np.linspace(0, np.pi, 20)) * 5.0
                )  # Largest

        train = [spike_times_ms]
        sd = SpikeData(
            train, length=n_samples / fs_khz, raw_data=raw_data, raw_time=fs_khz
        )
        sd.neuron_attributes = NeuronAttributes.from_dict({"unit_id": [0]}, n_neurons=1)

        wf_data = sd.neuron_attributes.calculate_mean_waveforms(
            sd, ms_before=0.5, ms_after=0.5
        )

        # Should select channel 2 (index 2)
        assert wf_data["best_channels"][0] == 2


class TestAnalysisCachingIntegration:
    """Integration tests for analysis caching features."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline with all caching features."""
        # Create diverse firing patterns
        np.random.seed(42)
        train = []
        for i in range(5):
            if i < 2:  # Regular
                train.append(np.arange(0, 1000, 100))
            else:  # Bursting
                spikes = []
                for burst_start in [0, 200, 400, 600, 800]:
                    spikes.extend([burst_start + j * 2 for j in range(5)])
                train.append(np.array(spikes))

        sd = SpikeData(train, length=1000)
        sd.neuron_attributes = NeuronAttributes.from_dict(
            {"unit_id": list(range(5))}, n_neurons=5
        )

        # Run all analyses
        isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
        lat_stats = sd.neuron_attributes.compute_latency_statistics(
            sd, reference_neuron=0
        )

        burst_edges = np.array([[0, 50], [200, 250], [400, 450]])
        burst_stats = sd.neuron_attributes.compute_burst_participation(
            sd, burst_edges=burst_edges, min_spikes=3
        )

        # Verify all attributes were saved
        df = sd.neuron_attributes.to_dataframe()
        assert "cv_isi" in df.columns
        assert "mean_latency_ms" in df.columns
        assert "burst_participation" in df.columns

        # Verify we can classify neurons
        assert len(df) == 5
        assert df["cv_isi"][0] < 1.0  # Regular neurons
        assert df["cv_isi"][2] > 1.0  # Bursting neurons
