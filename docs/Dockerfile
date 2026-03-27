# Multi-stage build: compile docs in a builder image, then publish a single
# lightweight nginx image that serves the static HTML artifacts.
# Usage:
#   docker build -f docs/Dockerfile -t spikelab-docs .
#   docker run -p 8080:80 spikelab-docs

# --- Stage 1: Build the documentation ---
FROM python:3.11-slim AS docs-builder

WORKDIR /build

# Install the package and docs dependencies
COPY pyproject.toml .
COPY README.md .
COPY src/ src/
RUN pip install --no-cache-dir ".[docs]"

# Copy docs source and build
COPY docs/ docs/
RUN python -m sphinx -b html docs/source docs/build/html

# --- Stage 2: Serve with nginx ---
FROM nginx:alpine AS docs-server

# Copy built HTML into nginx default serve directory
COPY --from=docs-builder /build/docs/build/html /usr/share/nginx/html

EXPOSE 80
