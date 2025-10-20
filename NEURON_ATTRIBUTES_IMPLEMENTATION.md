# Neuron Attributes Implementation Summary

This document summarizes the implementation of DataFrame-based neuron attributes for the IntegratedAnalysisTools project.

## Overview

A new `NeuronAttributes` class has been implemented to manage neuron-level metadata using pandas DataFrames. This replaces the previous list-based system and provides a more flexible, powerful way to store and manipulate neuron metadata.

## Key Features

### 1. NeuronAttributes Class (`spikedata/neuron_attributes.py`)

A dedicated module for managing neuron metadata:

- **Standard column names** for consistency:
  - Core: `unit_id`, `cluster_id`, `electrode_id`, `channel`, `firing_rate_hz`
  - Quality: `snr`, `amplitude`, `isolation_distance`, `isi_violations`
  - Spatial: `unit_x`, `unit_y`, `unit_z`, `electrode_x`, `electrode_y`, `electrode_z`
  
- **Flexible storage**: Allows arbitrary additional columns beyond standard names

- **Key methods**:
  - `from_dict()`, `from_dataframe()` - Create from various inputs
  - `set_attribute()`, `get_attribute()` - Access individual attributes
  - `subset()` - Select specific neurons
  - `concat()` - Concatenate attributes from multiple objects
  - `to_dataframe()` - Export to pandas DataFrame
  - `validate_n_neurons()`, `validate_columns()` - Data validation

### 2. SpikeData Integration

Updated `SpikeData` class to work seamlessly with `NeuronAttributes`:

- **Constructor** accepts `NeuronAttributes`, DataFrame, dict, or None
- **Automatic type conversion** when initializing
- **All operations updated**:
  - `subset()` - Automatically subsets attributes
  - `subtime()` - Propagates attributes unchanged
  - `append()` - Warns if attributes differ
  - `concatenate_spike_data()` - Concatenates attributes

- **New convenience methods**:
  - `set_neuron_attribute(column, values)` - Set/update an attribute
  - `get_neuron_attribute(column)` - Get attribute values
  - `compute_firing_rates(unit)` - Compute and store firing rates

### 3. Data Loaders Updated

All data loaders now automatically extract neuron attributes:

#### NWB Loader (`load_spikedata_from_nwb`)
- Extracts all columns from Units table
- Works with both pynwb and h5py fallback
- Excludes spike_times column, keeps all other metadata

#### KiloSort Loader (`load_spikedata_from_kilosort`)
- Parses `cluster_info.tsv` or similar files
- Auto-detects common TSV files if not specified
- Extracts cluster metadata (group, channel, quality metrics)

#### HDF5 Loader (`load_spikedata_from_hdf5`)
- Looks for `/neuron_attributes` group
- Loads each dataset in the group as a column
- Works with all HDF5 styles (raster, ragged, group, paired)

#### SpikeInterface Loader (`load_spikedata_from_spikeinterface`)
- Extracts all unit properties via `get_property_keys()` and `get_property()`
- Maps SpikeInterface properties to standard columns where applicable

### 4. Data Exporters Updated

All exporters now save neuron attributes:

#### HDF5 Exporter (`export_spikedata_to_hdf5`)
- Creates `/neuron_attributes` group
- Saves each DataFrame column as a separate dataset
- Preserves data types

#### NWB Exporter (`export_spikedata_to_nwb`)
- Writes attributes as additional columns in the Units table
- Follows NWB conventions for standard columns
- Includes custom columns

#### KiloSort Exporter (`export_spikedata_to_kilosort`)
- Generates `cluster_info.tsv` from neuron_attributes
- Maps to KiloSort standard columns (cluster_id, ch, group)
- Includes all custom attributes

### 5. Dependencies

Added `pandas>=1.3` to project dependencies in `pyproject.toml`.

## Files Created

1. **`spikedata/neuron_attributes.py`** - New module (358 lines)
2. **`tests/test_neuron_attributes.py`** - Comprehensive test suite (177 lines)
3. **`docs/source/neuron_attributes_guide.rst`** - User guide with examples
4. **`docs/source/api/neuron_attributes.rst`** - API documentation

## Files Modified

1. **`spikedata/__init__.py`** - Export NeuronAttributes
2. **`spikedata/spikedata.py`** - Integrated NeuronAttributes (updated 9 methods)
3. **`data_loaders/data_loaders.py`** - Updated all loaders (4 functions + helper)
4. **`data_loaders/data_exporters.py`** - Updated all exporters (3 functions + helper)
5. **`pyproject.toml`** - Added pandas dependency
6. **`README.md`** - Updated example to show neuron_attributes usage

## Testing

All basic functionality has been verified:

✓ Creating NeuronAttributes from dict and DataFrame  
✓ Creating SpikeData with neuron_attributes  
✓ Setting and getting attributes  
✓ Subset operation preserves attributes  
✓ Computing and storing firing rates  
✓ DataFrame conversion  

Comprehensive test suite created with:
- Creation and initialization tests
- Operations tests (set, get, subset, concat)
- Validation tests
- DataFrame conversion tests
- String representation tests

## Usage Example

```python
from spikedata import SpikeData
from data_loaders import load_spikedata_from_kilosort

# Load with automatic attribute extraction
sd = load_spikedata_from_kilosort(
    'kilosort_output/',
    fs_Hz=30000,
    cluster_info_tsv='cluster_info.tsv'
)

# Work with attributes
sd.compute_firing_rates(unit='Hz')
sd.set_neuron_attribute('quality', ['good', 'mua', 'good'])

# Subset automatically handles attributes
good_neurons = sd.subset(['good'], by='quality')

# Export with attributes preserved
good_neurons.to_hdf5('filtered.h5', style='ragged')

# Verify round-trip
from data_loaders import load_spikedata_from_hdf5
sd_loaded = load_spikedata_from_hdf5(
    'filtered.h5',
    spike_times_dataset='spike_times',
    spike_times_index_dataset='spike_times_index'
)
print(sd_loaded.neuron_attributes.to_dataframe())
```

## Standard Column Examples

```python
# After loading from KiloSort:
#     unit_id  cluster_id  channel  group  firing_rate_hz
# 0        0           0       12   good            5.23
# 1        1           1       15   good            8.91
# 2        2           2       18    mua           15.67

# Access specific columns
cluster_ids = sd.get_neuron_attribute('cluster_id')
channels = sd.get_neuron_attribute('channel')

# Filter by quality
good_mask = sd.get_neuron_attribute('group') == 'good'
good_indices = [i for i, m in enumerate(good_mask) if m]
sd_good = sd.subset(good_indices)
```

## Spatial Location Examples

```python
# Add spatial location information
sd.set_neuron_attribute('unit_x', [100.0, 150.0, 200.0])
sd.set_neuron_attribute('unit_y', [50.0, 75.0, 100.0])
sd.set_neuron_attribute('unit_z', [0.0, 10.0, 20.0])

# Combine coordinates for distance calculations
df = sd.neuron_attributes.to_dataframe()
positions = np.column_stack([df['unit_x'], df['unit_y'], df['unit_z']])

# Spatial filtering - find neurons in a region
import numpy as np
df = sd.neuron_attributes.to_dataframe()
in_region = (df['unit_x'] >= 100) & (df['unit_x'] <= 200)
region_neurons = sd.subset(df[in_region].index.tolist())

# Calculate distances from reference point
ref = np.array([150.0, 75.0, 10.0])
positions = np.column_stack([df['unit_x'], df['unit_y'], df['unit_z']])
distances = np.linalg.norm(positions - ref, axis=1)
sd.set_neuron_attribute('distance_from_ref', distances)
```

## Backward Compatibility

The old list-based `neuron_attributes` system has been completely removed. Users need to migrate to the new DataFrame-based system:

**Old:**
```python
neuron_attrs = [dataclass_obj1, dataclass_obj2, ...]
value = neuron_attrs[0].field_name
```

**New:**
```python
neuron_attrs = {'field_name': [val1, val2, ...]}
value = sd.get_neuron_attribute('field_name')[0]
```

## Benefits

1. **More Flexible**: Can add/remove columns dynamically
2. **More Powerful**: Leverage pandas DataFrame operations
3. **Better Integration**: Automatic extraction from standard file formats
4. **Type Flexible**: Support for any data type (numeric, string, etc.)
5. **Validation**: Built-in validation and warnings for common mistakes
6. **Round-trip**: Full preservation through load/save cycles

## Future Enhancements

Possible future additions:

1. Support for waveform templates (mentioned in TODO but excluded from this implementation)
2. Additional validation for standard columns (e.g., unit_id uniqueness)
3. Automatic computation of common quality metrics
4. Integration with visualization tools
5. Support for hierarchical attributes (e.g., per-electrode metadata)

## Notes

- All spike times remain in milliseconds (internal SpikeData convention)
- Firing rates in neuron_attributes are stored in Hz for consistency
- Empty/missing attributes are handled gracefully (None stored in SpikeData)
- Column name validation warns about potential typos in standard names
- Integer indexing (0 to N-1) always matches position in SpikeData.train


