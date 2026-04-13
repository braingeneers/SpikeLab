===================
NRP Jobs Quickstart
===================

This guide shows how to submit and monitor SpikeLab analysis jobs on NRP using
the ``spikelab-nrp-jobs`` CLI.

Prerequisites
-------------

- Install optional dependencies:

  .. code-block:: bash

     pip install -e ".[nrp-jobs,s3,neo]"

- Ensure Kubernetes access is configured:

  .. code-block:: bash

     kubectl version --client
     kubectl config current-context

- Configure AWS-compatible credentials in your shell (or use your normal
  credentials chain).

Minimal job config
------------------

Create a config file such as ``configs/job.yaml``:

.. code-block:: yaml

   name_prefix: analysis-job
   namespace: default
   labels:
     analysis: spikelab
   container:
     image_pull_policy: IfNotPresent
     command: ["python"]
     args: ["-m", "my_analysis.entrypoint"]
     env:
       OUTPUT_PREFIX: s3://braingeneers/ephys-analysis/
   resources:
     requests_cpu: "2"
     requests_memory: "8Gi"
     limits_cpu: "2"
     limits_memory: "8Gi"
     requests_gpu: 1
     limits_gpu: 1
     node_selector: {}
   volumes: []

Dry-run render
--------------

Always render first:

.. code-block:: bash

   spikelab-nrp-jobs render-job --profile nrp --job-config configs/job.yaml --image-profile cpu --output-manifest ./rendered-job.yaml

Deploy and monitor
------------------

Submit:

.. code-block:: bash

   spikelab-nrp-jobs deploy-job --profile nrp --job-config configs/job.yaml --image-profile gpu --wait --max-wait-seconds 3600

The deploy command prints a stable machine-readable line:

.. code-block:: text

   JOB_NAME=<generated-job-name>

Use that value with status/log commands:

.. code-block:: bash

   spikelab-nrp-jobs job-status <generated-job-name>
   spikelab-nrp-jobs job-logs <generated-job-name> --follow
   spikelab-nrp-jobs job-delete <generated-job-name>

S3 destination
--------------

Default artifact prefix is:

``s3://braingeneers/ephys-analysis/``

You can override this in your selected profile or job-level environment variables.
