# Multi-stage build: build docs with Sphinx, then serve with nginx
# Usage:
#   docker build -t spikelab-docs docs/
#   docker run -p 8080:80 spikelab-docs

# --- Stage 1: Build the documentation ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install the package and docs dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[docs]"

# Copy docs source and build
COPY docs/ docs/
RUN python -m sphinx -b html docs/source docs/build/html

# --- Stage 2: Serve with nginx ---
FROM nginx:alpine

# Copy built HTML into nginx default serve directory
COPY --from=builder /build/docs/build/html /usr/share/nginx/html

EXPOSE 80
