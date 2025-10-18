Time Windowing
==============

subtime(start, stop)
--------------------

Return a new SpikeData object with spikes restricted to a time window.

**Parameters:**

* ``start`` (float): Start time (ms).
* ``stop`` (float): Stop time (ms).

**Returns:**

* ``SpikeData``: Time-sliced object.

----

frames(frame_size, overlap=0)
------------------------------

Yield SpikeData objects for consecutive time frames.

**Parameters:**

* ``frame_size`` (float): Frame duration (ms).
* ``overlap`` (float, optional): Overlap between frames (ms).

**Yields:**

* ``SpikeData``: For each frame.

