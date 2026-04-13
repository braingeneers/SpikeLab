===================
NRP Docker Workflow
===================

Use this guide to build temporary analysis containers and deploy them as NRP jobs.

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

   bash scripts/nrp_build_temp_image.sh gpu ghcr.io/<org>/spikelab-analysis-temp:<tag>
   bash scripts/nrp_push_temp_image.sh ghcr.io/<org>/spikelab-analysis-temp:<tag>

Generate an NRP job config from that tag:

.. code-block:: bash

   python scripts/nrp_generate_job_config.py \
     --image ghcr.io/<org>/spikelab-analysis-temp:<tag> \
     --profile gpu \
     --output configs/nrp-temp-job.yaml

Smoke-check render and deploy
-----------------------------

Render first:

.. code-block:: bash

   spikelab-nrp-jobs render-job \
     --profile nrp \
     --job-config configs/nrp-temp-job.yaml \
     --image-profile gpu \
     --output-manifest /tmp/nrp-temp.yaml

Deploy after validation:

.. code-block:: bash

   spikelab-nrp-jobs deploy-job \
     --profile nrp \
     --job-config configs/nrp-temp-job.yaml \
     --image-profile gpu \
     --wait \
     --max-wait-seconds 3600

Then monitor:

.. code-block:: bash

   spikelab-nrp-jobs job-status <job-name>
   spikelab-nrp-jobs job-logs <job-name> --follow

Artifacts
---------

Use `OUTPUT_PREFIX` to direct results to object storage. Default prefix:

``s3://braingeneers/ephys-analysis/``
