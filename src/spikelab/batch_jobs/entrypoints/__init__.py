"""Container entrypoints for SpikeLab batch jobs.

These modules are invoked inside job containers via
``python -m spikelab.batch_jobs.entrypoints.workspace`` or
``python -m spikelab.batch_jobs.entrypoints.sorting``.
They are pre-installed in the container image as part of the
``spikelab`` package.
"""
