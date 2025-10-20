"""
NeuronAttributes module for managing neuron metadata in SpikeData.

This module provides a DataFrame-based system for storing and manipulating
neuron-level attributes such as cluster IDs, electrode positions, firing rates,
and quality metrics.
"""

from typing import Optional, Union, List, Dict, Any
import warnings

import numpy as np
import pandas as pd


class NeuronAttributes:
    """
    Wrapper class for managing neuron attributes as a pandas DataFrame.

    This class provides a structured way to store and manipulate metadata
    for individual neurons in a SpikeData object. The DataFrame is indexed
    by neuron position (0 to N-1), matching the order in SpikeData.train.

    Standard column names (optional but recommended):
        Core attributes:
            - unit_id: Unique identifier for each neuron
            - cluster_id: Cluster assignment (multiple neurons can share)
            - electrode_id: Physical electrode identifier
            - channel: Recording channel number
            - firing_rate_hz: Mean firing rate in Hz

        Quality metrics:
            - snr: Signal-to-noise ratio
            - amplitude: Spike amplitude (μV or arbitrary units)
            - isolation_distance: Cluster isolation quality
            - isi_violations: ISI violation rate

        Spatial location:
            - unit_x, unit_y, unit_z: Unit position coordinates (μm)
            - electrode_x, electrode_y, electrode_z: Electrode coordinates (μm)
            
        Note: Store spatial coordinates as separate x, y, z columns for best
        compatibility with pandas operations. Combine using np.column_stack()
        when needed for analysis.

    Additional custom columns are supported and encouraged for analysis-specific
    metadata.

    Attributes
    ----------
    _df : pd.DataFrame
        The underlying DataFrame storing neuron attributes
    n_neurons : int
        Number of neurons (must match length of DataFrame)

    Examples
    --------
    >>> # Create from dictionary
    >>> attrs = NeuronAttributes.from_dict({
    ...     'unit_id': [101, 102, 103],
    ...     'cluster_id': [1, 1, 2],
    ...     'firing_rate_hz': [5.2, 8.1, 3.4]
    ... }, n_neurons=3)
    >>> 
    >>> # Access as DataFrame
    >>> df = attrs.to_dataframe()
    >>> 
    >>> # Get specific attribute
    >>> rates = attrs.get_attribute('firing_rate_hz')
    >>> 
    >>> # Set new attribute
    >>> attrs.set_attribute('quality', ['good', 'good', 'mua'])
    """

    # Standard column names for validation and documentation
    STANDARD_COLUMNS = {
        # Core attributes
        'unit_id', 'cluster_id', 'electrode_id', 'channel', 'firing_rate_hz',
        # Quality metrics
        'snr', 'amplitude', 'isolation_distance', 'isi_violations',
        # Spatial location (separate coordinate columns)
        'unit_x', 'unit_y', 'unit_z',
        'electrode_x', 'electrode_y', 'electrode_z'
    }

    def __init__(
        self,
        df: pd.DataFrame,
        n_neurons: int,
        validate: bool = True
    ):
        """
        Initialize NeuronAttributes from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing neuron attributes. Index should be
            integer-based (0 to n_neurons-1).
        n_neurons : int
            Expected number of neurons. Must match DataFrame length.
        validate : bool, optional
            If True, validate the DataFrame structure and column names.
            Default is True.

        Raises
        ------
        ValueError
            If n_neurons doesn't match DataFrame length or if validation fails.
        """
        self.n_neurons = n_neurons
        self._df = df.copy()

        # Ensure integer index starting from 0
        if len(self._df) > 0:
            self._df.index = pd.RangeIndex(len(self._df))

        if validate:
            self.validate_n_neurons(n_neurons)
            self.validate_columns()

    def validate_n_neurons(self, expected_n: int) -> None:
        """
        Validate that the DataFrame has the expected number of rows.

        Parameters
        ----------
        expected_n : int
            Expected number of neurons.

        Raises
        ------
        ValueError
            If the number of rows doesn't match expected_n.
        """
        if len(self._df) != expected_n:
            raise ValueError(
                f"NeuronAttributes has {len(self._df)} rows but "
                f"expected {expected_n} neurons."
            )

    def validate_columns(self) -> None:
        """
        Validate column names and warn about potential typos.

        Checks for columns that are similar to standard names but don't
        match exactly, which might indicate typos.
        """
        for col in self._df.columns:
            if col not in self.STANDARD_COLUMNS:
                # Check for potential typos in standard columns
                col_lower = col.lower().replace('_', '').replace('-', '')
                for std_col in self.STANDARD_COLUMNS:
                    std_lower = std_col.lower().replace('_', '').replace('-', '')
                    if col_lower == std_lower and col != std_col:
                        warnings.warn(
                            f"Column '{col}' looks like standard column '{std_col}'. "
                            f"Consider using the standard name for consistency.",
                            UserWarning
                        )

    def set_attribute(self, column: str, values: Union[np.ndarray, List, pd.Series]) -> None:
        """
        Set or update a neuron attribute column.

        Parameters
        ----------
        column : str
            Name of the attribute column.
        values : array-like
            Values for the attribute. Must have length equal to n_neurons.

        Raises
        ------
        ValueError
            If values length doesn't match n_neurons.
        """
        values = np.asarray(values)
        if len(values) != self.n_neurons:
            raise ValueError(
                f"Values length ({len(values)}) doesn't match "
                f"n_neurons ({self.n_neurons})."
            )
        self._df[column] = values

    def get_attribute(self, column: str) -> np.ndarray:
        """
        Get values for a specific attribute column.

        Parameters
        ----------
        column : str
            Name of the attribute column.

        Returns
        -------
        np.ndarray
            Array of attribute values.

        Raises
        ------
        KeyError
            If column doesn't exist.
        """
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found in neuron attributes.")
        return self._df[column].values

    def subset(self, indices: Union[List[int], np.ndarray]) -> 'NeuronAttributes':
        """
        Select a subset of neurons by their indices.

        Parameters
        ----------
        indices : array-like of int
            Indices of neurons to select.

        Returns
        -------
        NeuronAttributes
            New NeuronAttributes object with selected neurons.
        """
        indices = np.asarray(indices)
        subset_df = self._df.iloc[indices].copy()
        subset_df.index = pd.RangeIndex(len(subset_df))
        return NeuronAttributes(subset_df, n_neurons=len(indices), validate=False)

    def concat(self, other: 'NeuronAttributes') -> 'NeuronAttributes':
        """
        Concatenate with another NeuronAttributes object.

        Columns that exist in one but not the other will be filled with NaN.

        Parameters
        ----------
        other : NeuronAttributes
            Another NeuronAttributes object to concatenate.

        Returns
        -------
        NeuronAttributes
            New NeuronAttributes with concatenated data.
        """
        combined_df = pd.concat([self._df, other._df], ignore_index=True)
        new_n = self.n_neurons + other.n_neurons
        return NeuronAttributes(combined_df, n_neurons=new_n, validate=False)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get the underlying DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of the underlying DataFrame.
        """
        return self._df.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], n_neurons: int) -> 'NeuronAttributes':
        """
        Create NeuronAttributes from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary mapping column names to values.
        n_neurons : int
            Expected number of neurons.

        Returns
        -------
        NeuronAttributes
            New NeuronAttributes object.
        """
        df = pd.DataFrame(data)
        return cls(df, n_neurons=n_neurons, validate=True)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, n_neurons: Optional[int] = None) -> 'NeuronAttributes':
        """
        Create NeuronAttributes from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing neuron attributes.
        n_neurons : int, optional
            Expected number of neurons. If None, inferred from DataFrame length.

        Returns
        -------
        NeuronAttributes
            New NeuronAttributes object.
        """
        if n_neurons is None:
            n_neurons = len(df)
        return cls(df, n_neurons=n_neurons, validate=True)

    def __len__(self) -> int:
        """Return the number of neurons."""
        return self.n_neurons

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NeuronAttributes({self.n_neurons} neurons, {len(self._df.columns)} attributes)\n{self._df}"

    def __str__(self) -> str:
        """Return string representation."""
        return self.__repr__()


