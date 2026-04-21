====================
Artifact Packaging
====================

SpikeLab batch job sessions package analysis artifacts into a zip bundle
before uploading.

Flow
----

1. Generate output files in ``pickle`` and/or ``nwb`` format.
2. Copy selected files into a temporary bundle directory.
3. Generate a ``manifest.json`` with checksums and metadata.
4. Create ``<run_id>.zip``.
5. Upload to S3 using the profile-configured prefix.

Python API example
------------------

.. code-block:: python

   from spikelab.batch_jobs.artifact_packager import package_analysis_bundle

   zip_path = package_analysis_bundle(
       input_paths=["./outputs/run123.pkl", "./outputs/run123.nwb"],
       run_id="run123",
       output_dir="./dist",
       output_format="both",
       metadata={"workspace_id": "example-workspace"},
   )

The resulting zip is suitable for batch job handoff and traceability.
