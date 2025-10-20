"""
Tests for the NeuronAttributes module.
"""

import numpy as np
import pandas as pd
import pytest

from spikedata import NeuronAttributes


class TestNeuronAttributesCreation:
    """Test creation and initialization of NeuronAttributes."""

    def test_from_dict(self):
        """Test creating NeuronAttributes from dictionary."""
        data = {
            'unit_id': [101, 102, 103],
            'cluster_id': [1, 1, 2],
            'firing_rate_hz': [5.2, 8.1, 3.4]
        }
        attrs = NeuronAttributes.from_dict(data, n_neurons=3)
        assert len(attrs) == 3
        assert attrs.n_neurons == 3
        np.testing.assert_array_equal(attrs.get_attribute('unit_id'), [101, 102, 103])

    def test_from_dataframe(self):
        """Test creating NeuronAttributes from pandas DataFrame."""
        df = pd.DataFrame({
            'unit_id': [1, 2, 3],
            'cluster_id': [10, 20, 30],
        })
        attrs = NeuronAttributes.from_dataframe(df)
        assert len(attrs) == 3
        np.testing.assert_array_equal(attrs.get_attribute('cluster_id'), [10, 20, 30])

    def test_validation_n_neurons_mismatch(self):
        """Test that validation catches neuron count mismatch."""
        data = {'unit_id': [1, 2]}
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
        attrs = NeuronAttributes.from_dict({'unit_id': [1, 2, 3]}, n_neurons=3)
        attrs.set_attribute('quality', ['good', 'mua', 'good'])
        quality = attrs.get_attribute('quality')
        assert list(quality) == ['good', 'mua', 'good']

    def test_get_nonexistent_attribute(self):
        """Test getting attribute that doesn't exist."""
        attrs = NeuronAttributes.from_dict({'unit_id': [1, 2]}, n_neurons=2)
        with pytest.raises(KeyError, match="'nonexistent'"):
            attrs.get_attribute('nonexistent')

    def test_set_attribute_wrong_length(self):
        """Test that setting attribute with wrong length raises error."""
        attrs = NeuronAttributes.from_dict({'unit_id': [1, 2, 3]}, n_neurons=3)
        with pytest.raises(ValueError, match="Values length"):
            attrs.set_attribute('quality', ['good', 'mua'])  # Only 2 values for 3 neurons

    def test_subset(self):
        """Test subsetting neurons."""
        attrs = NeuronAttributes.from_dict({
            'unit_id': [1, 2, 3, 4],
            'cluster_id': [10, 20, 30, 40]
        }, n_neurons=4)
        subset_attrs = attrs.subset([0, 2])
        assert len(subset_attrs) == 2
        np.testing.assert_array_equal(subset_attrs.get_attribute('unit_id'), [1, 3])
        np.testing.assert_array_equal(subset_attrs.get_attribute('cluster_id'), [10, 30])

    def test_concat(self):
        """Test concatenating NeuronAttributes."""
        attrs1 = NeuronAttributes.from_dict({'unit_id': [1, 2]}, n_neurons=2)
        attrs2 = NeuronAttributes.from_dict({'unit_id': [3, 4]}, n_neurons=2)
        combined = attrs1.concat(attrs2)
        assert len(combined) == 4
        np.testing.assert_array_equal(combined.get_attribute('unit_id'), [1, 2, 3, 4])

    def test_concat_different_columns(self):
        """Test concatenating with different columns fills with NaN."""
        attrs1 = NeuronAttributes.from_dict({'unit_id': [1, 2], 'snr': [4.0, 5.0]}, n_neurons=2)
        attrs2 = NeuronAttributes.from_dict({'unit_id': [3, 4], 'amplitude': [100.0, 150.0]}, n_neurons=2)
        combined = attrs1.concat(attrs2)
        assert len(combined) == 4
        df = combined.to_dataframe()
        assert 'snr' in df.columns
        assert 'amplitude' in df.columns
        assert pd.isna(df.loc[2, 'snr'])  # Should be NaN for neurons from attrs2


class TestNeuronAttributesValidation:
    """Test validation features."""

    def test_column_name_typo_warning(self):
        """Test that similar column names trigger warning."""
        data = {'firing_rate_Hz': [1.0, 2.0]}  # Wrong case
        with pytest.warns(UserWarning, match="firing_rate_hz"):
            NeuronAttributes.from_dict(data, n_neurons=2)

    def test_standard_columns_no_warning(self):
        """Test that standard column names don't trigger warnings."""
        data = {
            'unit_id': [1, 2],
            'cluster_id': [10, 20],
            'firing_rate_hz': [5.0, 10.0],
            'snr': [3.5, 4.2]
        }
        # Should not raise any warnings
        attrs = NeuronAttributes.from_dict(data, n_neurons=2)
        assert len(attrs) == 2


class TestNeuronAttributesDataFrame:
    """Test DataFrame conversion."""

    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        data = {
            'unit_id': [1, 2, 3],
            'cluster_id': [10, 20, 30],
            'firing_rate_hz': [5.2, 8.1, 3.4]
        }
        attrs = NeuronAttributes.from_dict(data, n_neurons=3)
        df = attrs.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['unit_id', 'cluster_id', 'firing_rate_hz']

    def test_dataframe_round_trip(self):
        """Test that DataFrame round-trip preserves data."""
        original_df = pd.DataFrame({
            'unit_id': [1, 2, 3],
            'quality': ['good', 'mua', 'good'],
            'snr': [4.2, 3.1, 5.5]
        })
        attrs = NeuronAttributes.from_dataframe(original_df, n_neurons=3)
        result_df = attrs.to_dataframe()
        pd.testing.assert_frame_equal(result_df, original_df, check_dtype=False)


class TestNeuronAttributesRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        attrs = NeuronAttributes.from_dict({
            'unit_id': [1, 2],
            'cluster_id': [10, 20]
        }, n_neurons=2)
        repr_str = repr(attrs)
        assert 'NeuronAttributes' in repr_str
        assert '2 neurons' in repr_str
        assert '2 attributes' in repr_str

    def test_len(self):
        """Test __len__ method."""
        attrs = NeuronAttributes.from_dict({'unit_id': [1, 2, 3]}, n_neurons=3)
        assert len(attrs) == 3


