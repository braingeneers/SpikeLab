=======================
Batch Docker Workflow
=======================

Use this guide to build temporary analysis containers and deploy them as
batch jobs on a Kubernetes cluster.

Base images
-----------

Build reusable base images first:

.. code-block:: bash

   docker build -f docker/analysis-base/Dockerfile.cpu -t spikelab/analysis-base:cpu .
   docker build -f docker/analysis-base/Dockerfile.gpu -t spikelab/analysis-base:gpu .

Temporary image workflow
------------------------

Build and push a temporary image for one run:

.. code-block:: bash

   bash scripts/build_temp_image.sh gpu ghcr.io/<org>/spikelab-analysis-temp:<tag>
   bash scripts/push_temp_image.sh ghcr.io/<org>/spikelab-analysis-temp:<tag>

Generate a job config from that tag:

.. code-block:: bash

   python scripts/generate_job_config.py \
     --image ghcr.io/<org>/spikelab-analysis-temp:<tag> \
     --profile gpu \
     --output configs/batch-temp-job.yaml

Smoke-check render and deploy
-----------------------------

Render first:

.. code-block:: bash

   spikelab-batch-jobs render-job \
     --job-config configs/batch-temp-job.yaml \
     --image-profile gpu \
     --output-manifest /tmp/batch-temp.yaml

Deploy after validation:

.. code-block:: bash

   spikelab-batch-jobs deploy-job \
     --job-config configs/batch-temp-job.yaml \
     --image-profile gpu \
     --wait \
     --max-wait-seconds 3600

Then monitor:

.. code-block:: bash

   spikelab-batch-jobs job-status <job-name>
   spikelab-batch-jobs job-logs <job-name> --follow

Artifacts
---------

Use ``OUTPUT_PREFIX`` in the job config environment to direct results to
object storage. The prefix is set by the active cluster profile or
overridden per-job.
