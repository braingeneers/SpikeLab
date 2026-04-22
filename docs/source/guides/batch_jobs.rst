==========
Batch Jobs
==========

SpikeLab can submit analysis and spike sorting jobs to a Kubernetes cluster.
Both workflows follow the same pattern: bundle inputs, submit a job, and
retrieve results as an :class:`~spikelab.workspace.workspace.AnalysisWorkspace`.

.. contents:: On this page
   :local:
   :depth: 2


Prerequisites
-------------

Install the batch jobs optional dependencies:

.. code-block:: bash

   pip install -e ".[batch-jobs,s3]"

Ensure Kubernetes access is configured:

.. code-block:: bash

   kubectl version --client
   kubectl config current-context

Configure AWS-compatible credentials in your shell (or use your normal
credentials chain).


Setting Up a Session
--------------------

All batch operations go through a
:class:`~spikelab.batch_jobs.session.RunSession`. Create one from a cluster
profile:

.. code-block:: python

   from spikelab.batch_jobs import RunSession, load_cluster_profile

   profile = load_cluster_profile("nrp")
   session = RunSession.from_profile(profile)

The profile controls the default namespace, S3 prefix, Docker images,
credential mounts, and policy thresholds. You can also load a custom profile
from a YAML file with ``load_profile_from_name``.


Analysis Jobs
-------------

Analysis jobs run a user-provided script against an
:class:`~spikelab.workspace.workspace.AnalysisWorkspace`. The workspace is
uploaded to the cluster, the script modifies it, and the updated workspace is
returned.

Submitting
^^^^^^^^^^

.. code-block:: python

   from spikelab.batch_jobs import RunSession, JobSpec

   result = session.submit_workspace_job(
       workspace=ws,                    # AnalysisWorkspace or path to saved workspace
       script="my_analysis.py",         # analysis script to run
       job_spec=JobSpec(
           name_prefix="my-analysis",
           container=ContainerSpec(image="spikelab/analysis:latest"),
           resources=ResourceSpec(
               requests_cpu="2",
               requests_memory="8Gi",
           ),
       ),
   )

   print(result.job_name)
   print(result.run_id)

The session saves the workspace to HDF5, bundles it with the analysis script,
uploads the bundle to S3, and submits a Kubernetes job. Inside the container,
the workspace is loaded and made available to the script as a global
``workspace`` variable. The script reads from and writes to the workspace
directly.

Writing an analysis script
^^^^^^^^^^^^^^^^^^^^^^^^^^

The analysis script runs inside the container with the workspace already
loaded. Use it like a normal Python script:

.. code-block:: python

   # my_analysis.py
   # 'workspace' is available as a global variable

   sd = workspace.get("recording_01", "spikedata")

   rates = sd.rates(unit="Hz")
   workspace.store("recording_01", "firing_rates_hz", rates)

   pop_rate = sd.get_pop_rate(square_width=20, gauss_sigma=100)
   workspace.store("recording_01", "pop_rate", pop_rate)

After the script finishes, the workspace is saved and uploaded automatically.

Retrieving results
^^^^^^^^^^^^^^^^^^

Once the job completes, retrieve the updated workspace:

.. code-block:: python

   ws_updated = session.retrieve_result(result, local_dir="./results")

This downloads the workspace from S3 and returns an
:class:`~spikelab.workspace.workspace.AnalysisWorkspace` with all the results
your script stored.


Sorting Jobs
------------

Sorting jobs run the SpikeLab spike sorting pipeline on raw recording files
and return the sorted results as a workspace.

Submitting
^^^^^^^^^^

.. code-block:: python

   result = session.submit_sorting_job(
       recording_paths=["session1.raw.h5", "session2.raw.h5"],
       config="kilosort4",             # preset name, SortingPipelineConfig, or None
       config_overrides={"fr_min": 0.1, "snr_min": 6.0},
       job_spec=JobSpec(
           name_prefix="sorting-run",
           container=ContainerSpec(image="spikelab/sorting:latest"),
           resources=ResourceSpec(
               requests_cpu="4",
               requests_memory="16Gi",
               requests_gpu=1,
           ),
       ),
   )

The session bundles the recording files and sorting configuration, uploads
them to S3, and submits a job that runs ``sort_recording`` inside the
container.

Retrieving results
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ws_sorted = session.retrieve_result(result, local_dir="./sorted")

The returned workspace contains one namespace per recording. Each namespace
holds:

- ``"spikedata"`` — the curated :class:`~spikelab.SpikeData` object
- ``"sorting_metadata"`` — sorting parameters, curation history, and unit
  counts per stage

Any QC figures generated during sorting are downloaded to the local directory
alongside the workspace.

You can then continue directly with analysis:

.. code-block:: python

   sd = ws_sorted.get("session1", "spikedata")
   print(f"{sd.N} curated units, {sd.length / 1000:.1f} seconds")


Policy Guardrails
-----------------

Before submission, a preflight policy check runs automatically. It reports:

- ``PASS`` — checks are compliant
- ``WARN`` — settings are risky but allowed
- ``BLOCK`` — submission is blocked by default

Policy thresholds (GPU limits, runtime caps, sleep detection) are configured
per-profile via the ``policy`` section in the cluster profile YAML.

Current checks:

- Detect disallowed batch placeholders such as ``sleep infinity``
- Ensure GPU request/limit consistency
- Warn when request/limit tuning is likely inefficient
- Warn when runtimes exceed the configured maximum

To override a blocked submission when you understand and accept the
trade-offs:

.. code-block:: python

   result = session.submit_workspace_job(
       workspace=ws,
       script="my_analysis.py",
       job_spec=job_spec,
       allow_policy_risk=True,
   )


Docker Images
-------------

Base images
^^^^^^^^^^^

Build reusable base images for CPU and GPU workloads:

.. code-block:: bash

   docker build -f docker/analysis-base/Dockerfile.cpu -t spikelab/analysis-base:cpu .
   docker build -f docker/analysis-base/Dockerfile.gpu -t spikelab/analysis-base:gpu .

Temporary images
^^^^^^^^^^^^^^^^

Build and push a temporary image for a single run:

.. code-block:: bash

   bash scripts/build_temp_image.sh gpu ghcr.io/<org>/spikelab-analysis-temp:<tag>
   bash scripts/push_temp_image.sh ghcr.io/<org>/spikelab-analysis-temp:<tag>

Reference this tag in the ``ContainerSpec`` when creating your ``JobSpec``.


Monitoring Jobs
---------------

Use the ``spikelab-batch-jobs`` CLI to check job status and stream logs:

.. code-block:: bash

   spikelab-batch-jobs job-status <job-name>
   spikelab-batch-jobs job-logs <job-name> --follow
   spikelab-batch-jobs job-delete <job-name>

The job name is available in ``result.job_name`` after submission.
