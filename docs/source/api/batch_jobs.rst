==========
Batch Jobs
==========

The ``spikelab.batch_jobs`` sub-package provides Kubernetes batch-job
launching helpers for submitting SpikeLab analysis workloads to a cluster.
It requires the ``batch-jobs`` optional dependency group:

.. code-block:: bash

   pip install spikelab[batch-jobs]

See the :doc:`/guides/batch_jobs` guide for usage examples.


Models
------

Pydantic models defining job specifications, cluster profiles, and related
configuration.

.. automodule:: spikelab.batch_jobs.models
   :members:
   :show-inheritance:


Policy Engine
-------------

Cluster policy preflight checks using profile-driven thresholds.

.. automodule:: spikelab.batch_jobs.policy
   :members:
   :show-inheritance:


Profile Loading
---------------

.. automodule:: spikelab.batch_jobs.profiles
   :members:
   :show-inheritance:


Run Session
-----------

High-level orchestration for packaging, uploading, and job submission.

.. automodule:: spikelab.batch_jobs.session
   :members:
   :show-inheritance:


Kubernetes Backend
------------------

.. automodule:: spikelab.batch_jobs.backend_k8s
   :members:
   :show-inheritance:


S3 Storage
----------

.. automodule:: spikelab.batch_jobs.storage_s3
   :members:
   :show-inheritance:


Artifact Packager
-----------------

.. automodule:: spikelab.batch_jobs.artifact_packager
   :members:
   :show-inheritance:


Credentials
-----------

.. automodule:: spikelab.batch_jobs.credentials
   :members:
   :show-inheritance:


Validation
----------

.. automodule:: spikelab.batch_jobs.validation
   :members:
   :show-inheritance:
