=======================
Batch Jobs Quickstart
=======================

This guide shows how to submit and monitor SpikeLab analysis jobs on a
Kubernetes cluster using the ``spikelab-batch-jobs`` CLI.

Prerequisites
-------------

- Install optional dependencies:

  .. code-block:: bash

     pip install -e ".[batch-jobs,s3,neo]"

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
       OUTPUT_PREFIX: s3://YOUR-BUCKET/YOUR-PREFIX/
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

   spikelab-batch-jobs render-job --job-config configs/job.yaml --image-profile cpu --output-manifest ./rendered-job.yaml

Deploy and monitor
------------------

Submit:

.. code-block:: bash

   spikelab-batch-jobs deploy-job --job-config configs/job.yaml --image-profile gpu --wait --max-wait-seconds 3600

The deploy command prints a stable machine-readable line:

.. code-block:: text

   JOB_NAME=<generated-job-name>

Use that value with status/log commands:

.. code-block:: bash

   spikelab-batch-jobs job-status <generated-job-name>
   spikelab-batch-jobs job-logs <generated-job-name> --follow
   spikelab-batch-jobs job-delete <generated-job-name>

Using a cluster profile
-----------------------

Pass ``--profile nrp`` to use the built-in NRP profile, or point to a
custom profile YAML with ``--profile-file /path/to/profile.yaml``.

The profile controls the default namespace, S3 prefix, Docker images,
credential mounts, and policy thresholds.

S3 destination
--------------

The artifact S3 prefix is set by the active profile's ``default_s3_prefix``
field or overridden via the ``OUTPUT_PREFIX`` environment variable in your
job config.
