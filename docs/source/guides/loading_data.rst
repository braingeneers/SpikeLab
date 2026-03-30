============
Loading Data
============

SpikeLab supports loading spike train data from a variety of common
electrophysiology formats. All loaders return a
:class:`~spikelab.spikedata.spike_data.SpikeData` object and convert spike
times to **milliseconds** where possible, since all analyses assume this unit.

The main loader functions live in
``spikelab.data_loaders.data_loaders``. In addition,
:class:`~spikelab.spikedata.spike_data.SpikeData` provides static constructors
for building spike data directly from arrays or raw traces.


From pickle files
-----------------

If you have a previously saved ``SpikeData`` object in a pickle file, you can
load it with the standard library or with the SpikeLab convenience loader.

Using the standard library:

.. code-block:: python

   import pickle

   with open("my_data.pkl", "rb") as f:
       sd = pickle.load(f)

Using the SpikeLab loader (which also supports S3 URLs):

.. code-block:: python

   from spikelab.data_loaders.data_loaders import load_spikedata_from_pickle

   sd = load_spikedata_from_pickle("my_data.pkl")

   # Load from S3
   sd = load_spikedata_from_pickle(
       "s3://my-bucket/my_data.pkl",
       aws_access_key_id="...",
       aws_secret_access_key="...",
   )


From HDF5 files
----------------

HDF5 is the most flexible format. SpikeLab supports four different storage
styles within an HDF5 file:

- **Raster** -- a ``(N, T)`` spike count matrix stored as a single dataset.
- **Ragged** -- a flat array of spike times plus an index array that marks the
  boundaries of each unit (NWB-like layout).
- **Group** -- one HDF5 group per unit, each containing a 1-D array of spike
  times.
- **Paired** -- two parallel 1-D arrays: unit indices and spike times.

You choose which style to load by setting the corresponding parameters.
Exactly one style must be specified per call.

.. code-block:: python

   from spikelab.data_loaders.data_loaders import load_spikedata_from_hdf5

   # Raster style
   sd = load_spikedata_from_hdf5(
       "recording.h5",
       raster_dataset="raster",
       raster_bin_size_ms=1.0,
   )

   # Ragged style (spike_times + spike_times_index)
   sd = load_spikedata_from_hdf5(
       "recording.h5",
       spike_times_dataset="spike_times",
       spike_times_index_dataset="spike_times_index",
       spike_times_unit="s",       # times in the file are in seconds
   )

   # Group-per-unit style
   sd = load_spikedata_from_hdf5(
       "recording.h5",
       group_per_unit="units",
       group_time_unit="s",
   )

   # Paired style (idces + times)
   sd = load_spikedata_from_hdf5(
       "recording.h5",
       idces_dataset="idces",
       times_dataset="times",
       times_unit="ms",
   )

The ``spike_times_unit`` parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the ragged style, the ``spike_times_unit`` parameter (default ``'s'``)
tells the loader what unit the times in the file are stored in. The loader
converts them to milliseconds automatically. Set this to ``'ms'`` if your file
already stores times in milliseconds.

Analogous parameters exist for the other styles: ``group_time_unit`` for group
style and ``times_unit`` for paired style.


From NWB files
--------------

SpikeLab can load spike trains from NWB (Neurodata Without Borders) files. The
loader reads the ``/units`` group and populates ``neuron_attributes`` with
``unit_id``, ``electrode``, and ``location`` when available.

.. code-block:: python

   from spikelab.data_loaders.data_loaders import load_spikedata_from_nwb

   sd = load_spikedata_from_nwb("session.nwb")

   # Optionally control the backend and recording length
   sd = load_spikedata_from_nwb(
       "session.nwb",
       prefer_pynwb=True,    # try pynwb first, fall back to h5py
       length_ms=600000.0,   # truncate to first 10 minutes
   )

The NWB loader tries ``pynwb`` first by default. If ``pynwb`` is not installed
it falls back to reading the HDF5 file directly with ``h5py``. Both
``pynwb`` and ``h5py`` are optional dependencies.


From KiloSort/Phy output
-------------------------

KiloSort and Phy produce a folder of ``.npy`` files. SpikeLab reads
``spike_times.npy`` and ``spike_clusters.npy`` and optionally filters clusters
using the TSV cluster info file produced by Phy (keeping ``good`` and ``mua``
labels by default).

The ``fs_Hz`` parameter is required -- it specifies the sampling frequency so
that sample indices can be converted to milliseconds.

.. code-block:: python

   from spikelab.data_loaders.data_loaders import load_spikedata_from_kilosort

   sd = load_spikedata_from_kilosort(
       "path/to/kilosort_output/",
       fs_Hz=30000,
   )

   # With cluster filtering and custom file names
   sd = load_spikedata_from_kilosort(
       "path/to/kilosort_output/",
       fs_Hz=30000,
       cluster_info_tsv="cluster_info.tsv",
       include_noise=False,
       spike_times_file="spike_times.npy",
       spike_clusters_file="spike_clusters.npy",
   )

The loader populates ``neuron_attributes`` with ``unit_id``, ``electrode``,
and ``location`` when the corresponding information is available in the
KiloSort output.


From SpikeInterface
-------------------

If you are using `SpikeInterface <https://spikeinterface.readthedocs.io/>`_
for spike sorting, you can convert a ``SortingExtractor`` directly into a
:class:`~spikelab.spikedata.spike_data.SpikeData`:

.. code-block:: python

   from spikelab.data_loaders.data_loaders import load_spikedata_from_spikeinterface

   # sorting is any SpikeInterface SortingExtractor-like object
   sd = load_spikedata_from_spikeinterface(
       sorting,
       sampling_frequency=30000,   # override if not set on the object
       unit_ids=None,              # load all units
       segment_index=0,
   )

SpikeLab also supports loading from a SpikeInterface ``BaseRecording`` object,
which applies threshold detection to the raw traces:

.. code-block:: python

   from spikelab.data_loaders.data_loaders import load_spikedata_from_spikeinterface_recording

   sd = load_spikedata_from_spikeinterface_recording(
       recording,
       segment_index=0,
       threshold_sigma=5.0,
       filter=False,
       hysteresis=True,
       direction="both",
   )


From raw data
-------------

:class:`~spikelab.spikedata.spike_data.SpikeData` provides two static
constructors for building spike data from arrays that do not require any
external file format.

From threshold detection
^^^^^^^^^^^^^^^^^^^^^^^^

If you have raw voltage traces as a NumPy array of shape ``(channels, time)``,
you can detect spikes using a threshold crossing method:

.. code-block:: python

   from spikelab.spikedata.spike_data import SpikeData
   import numpy as np

   raw_traces = np.random.randn(64, 600000)  # 64 channels, 30 s at 20 kHz

   sd = SpikeData.from_thresholding(
       raw_traces,
       fs_Hz=20000,
       threshold_sigma=5.0,
       filter=True,           # 300-6000 Hz Butterworth bandpass
       hysteresis=True,
       direction="both",      # detect both positive and negative crossings
   )

The resulting ``SpikeData`` object has the original traces attached as
``raw_data`` and ``raw_time`` attributes.

From a spike count raster
^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have a ``(N, T)`` spike count raster, you can convert it
directly:

.. code-block:: python

   from spikelab.spikedata.spike_data import SpikeData
   import numpy as np

   raster = np.random.poisson(0.01, size=(32, 100000))

   sd = SpikeData.from_raster(raster, bin_size_ms=1.0)

Spikes are placed evenly within each bin.
