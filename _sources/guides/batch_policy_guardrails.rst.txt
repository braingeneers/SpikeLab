========================
Batch Policy Guardrails
========================

The batch job runner performs a preflight policy check before submission.
It reports:

- ``PASS`` when checks are compliant
- ``WARN`` when settings are risky but allowed
- ``BLOCK`` when patterns should not be submitted by default

Policy thresholds (GPU limits, runtime caps, sleep detection) are
configurable per-profile via the ``policy`` section in the cluster profile
YAML.

Current checks
--------------

- Detect disallowed batch placeholders such as ``sleep infinity``
- Ensure GPU request/limit consistency
- Warn when request/limit tuning is likely inefficient
- Warn when runtimes exceed the configured maximum

Override behavior
-----------------

The default behavior avoids policy-risk submissions. If an expert user
explicitly wants to proceed, they can pass:

.. code-block:: bash

   spikelab-batch-jobs deploy-job --job-config configs/job.yaml --allow-policy-risk

Use override flags only when the user understands and accepts the trade-offs.

Operational guidance
--------------------

- Keep requests close to real usage.
- Remove completed jobs.
- Avoid idle long-running pods unless intentionally using the Deployment model.
