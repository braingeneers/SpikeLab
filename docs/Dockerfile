# SpikeLab documentation server.
#
# Single-stage image: nginx serves Sphinx-built HTML, with a cron job inside
# the container that pulls the SpikeLab repo every hour and rebuilds the docs
# in place. Doc updates merged to the configured branch propagate to the live
# site without redeploying the container.
#
# Build (from SpikeLab/):
#   docker build -f docs/Dockerfile -t braingeneers/spikelab-docs:latest ..
# Run:
#   docker run -p 8080:80 braingeneers/spikelab-docs:latest
#
# Build args (override-able):
#   --build-arg REPO_URL=https://github.com/braingeneers/SpikeLab.git
#   --build-arg BRANCH=main

FROM python:3.11-slim

ARG REPO_URL=https://github.com/braingeneers/SpikeLab.git
ARG BRANCH=main

ENV SPIKELAB_REPO_URL=${REPO_URL} \
    SPIKELAB_BRANCH=${BRANCH} \
    SPIKELAB_REPO_DIR=/opt/spikelab \
    SPIKELAB_SERVE_DIR=/srv/spikelab-docs \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        git \
        cron \
        supervisor \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Clone SpikeLab and install with docs extras. Editable install means future
# git pulls update the importable package without reinstalling, except when
# pyproject.toml changes (handled by the rebuild script).
RUN git clone --branch "${BRANCH}" "${REPO_URL}" "${SPIKELAB_REPO_DIR}"
WORKDIR ${SPIKELAB_REPO_DIR}
RUN pip install --no-cache-dir -e ".[docs]"

# Initial baseline build into a versioned dir; "current" symlink points to it.
# nginx serves through the symlink so atomic swaps require no nginx reload.
RUN mkdir -p "${SPIKELAB_SERVE_DIR}" && \
    python -m sphinx -b html docs/source "${SPIKELAB_SERVE_DIR}/build-initial" && \
    ln -sfn "${SPIKELAB_SERVE_DIR}/build-initial" "${SPIKELAB_SERVE_DIR}/current"

# Nginx vhost, supervisord config, cron job, rebuild script.
COPY docs/nginx.conf /etc/nginx/sites-available/default
COPY docs/supervisord.conf /etc/supervisor/conf.d/spikelab-docs.conf
COPY docs/crontab /etc/cron.d/spikelab-docs
COPY docs/rebuild-docs.sh /usr/local/bin/rebuild-docs

RUN chmod +x /usr/local/bin/rebuild-docs && \
    chmod 0644 /etc/cron.d/spikelab-docs && \
    crontab /etc/cron.d/spikelab-docs && \
    touch /var/log/spikelab-docs.log

EXPOSE 80

# tini handles PID 1 and zombie reaping; supervisord runs nginx and cron.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
