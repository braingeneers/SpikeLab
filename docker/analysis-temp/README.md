# Temporary NRP Job Image

This folder provides a temporary image layer that can be built quickly and submitted
through `spikelab-batch-jobs`.

## Build

```bash
docker build \
  -f docker/analysis-temp/Dockerfile.temp \
  --build-arg BASE_IMAGE=spikelab/analysis-base:cpu \
  -t ghcr.io/<org>/spikelab-analysis-temp:<tag> \
  .
```

For GPU workflows, set `BASE_IMAGE` to your GPU base image tag.

## Push

```bash
docker push ghcr.io/<org>/spikelab-analysis-temp:<tag>
```

## Generate NRP job config

```bash
python scripts/nrp_generate_job_config.py \
  --image ghcr.io/<org>/spikelab-analysis-temp:<tag> \
  --profile gpu \
  --output configs/nrp-temp-job.yaml
```

## Deploy

```bash
spikelab-batch-jobs render-job --profile nrp --job-config configs/nrp-temp-job.yaml --output-manifest /tmp/nrp-job.yaml
spikelab-batch-jobs deploy-job --profile nrp --job-config configs/nrp-temp-job.yaml --wait --max-wait-seconds 3600
```
