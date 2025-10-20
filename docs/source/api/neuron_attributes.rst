NeuronAttributes API
====================

.. currentmodule:: spikedata

NeuronAttributes Class
----------------------

.. autoclass:: NeuronAttributes
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__

   .. rubric:: Class Methods

   .. automethod:: from_dict
   .. automethod:: from_dataframe

   .. rubric:: Attribute Access

   .. automethod:: set_attribute
   .. automethod:: get_attribute
   .. automethod:: to_dataframe

   .. rubric:: Operations

   .. automethod:: subset
   .. automethod:: concat

   .. rubric:: Validation

   .. automethod:: validate_n_neurons
   .. automethod:: validate_columns

   .. rubric:: Special Methods

   .. automethod:: __len__
   .. automethod:: __repr__
   .. automethod:: __str__

Standard Column Names
---------------------

The following column names are recommended for consistency:

Core Attributes
~~~~~~~~~~~~~~~

* ``unit_id`` - Unique identifier for each neuron
* ``cluster_id`` - Cluster assignment
* ``electrode_id`` - Physical electrode identifier  
* ``channel`` - Recording channel number
* ``firing_rate_hz`` - Mean firing rate in Hz

Quality Metrics
~~~~~~~~~~~~~~~

* ``snr`` - Signal-to-noise ratio
* ``amplitude`` - Spike amplitude
* ``isolation_distance`` - Cluster isolation quality
* ``isi_violations`` - ISI violation rate

Spatial Location
~~~~~~~~~~~~~~~~

* ``unit_x``, ``unit_y``, ``unit_z`` - Unit position coordinates (μm)
* ``electrode_x``, ``electrode_y``, ``electrode_z`` - Electrode coordinates (μm)

.. note::
   Store spatial coordinates as separate x, y, z columns for best compatibility with pandas.
   Use ``np.column_stack()`` to combine them when needed for analysis.

Usage Examples
--------------

Creating NeuronAttributes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spikedata import NeuronAttributes
    
    # From dictionary
    attrs = NeuronAttributes.from_dict({
        'unit_id': [1, 2, 3],
        'cluster_id': [10, 20, 30],
        'firing_rate_hz': [5.2, 8.1, 3.4]
    }, n_neurons=3)
    
    # From DataFrame
    import pandas as pd
    df = pd.DataFrame({'unit_id': [1, 2, 3]})
    attrs = NeuronAttributes.from_dataframe(df)

Setting and Getting Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set a new attribute
    attrs.set_attribute('quality', ['good', 'mua', 'good'])
    
    # Get attribute values
    qualities = attrs.get_attribute('quality')
    
    # Convert to DataFrame for advanced operations
    df = attrs.to_dataframe()

Operations
~~~~~~~~~~

.. code-block:: python

    # Subset neurons
    subset_attrs = attrs.subset([0, 2])  # Select neurons 0 and 2
    
    # Concatenate
    combined = attrs1.concat(attrs2)
    
    # Check length
    n_neurons = len(attrs)

Integration with SpikeData
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spikedata import SpikeData
    
    # Create SpikeData with attributes
    sd = SpikeData(
        trains,
        neuron_attributes={
            'unit_id': [1, 2, 3],
            'cluster_id': [10, 20, 30]
        }
    )
    
    # Access via SpikeData methods
    sd.set_neuron_attribute('quality', ['good', 'good', 'mua'])
    quality = sd.get_neuron_attribute('quality')
    
    # Attributes are preserved during operations
    sd_subset = sd.subset([0, 2])
    print(sd_subset.neuron_attributes.to_dataframe())


