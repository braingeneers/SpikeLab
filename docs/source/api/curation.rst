========
Curation
========

Unit-curation helpers for :class:`~spikelab.SpikeData` objects. Each function
takes a ``SpikeData`` as its first argument and returns a tuple
``(SpikeData, result_dict)`` where ``result_dict`` contains the per-unit metric
and a boolean mask of units that passed the criterion. These functions are also
bound as methods on :class:`~spikelab.SpikeData` (e.g. ``sd.curate_by_snr(...)``)
and can be applied in combination via :meth:`~spikelab.SpikeData.curate`.

.. automodule:: spikelab.spikedata.curation
   :members:
   :show-inheritance:
