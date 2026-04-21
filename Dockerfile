FROM python:3.10-slim

WORKDIR /app

# Copy the package source and install it with MCP + SSE extras
COPY . /app/
RUN pip install --no-cache-dir -e ".[mcp,sse,s3]"

EXPOSE 8080

ENTRYPOINT ["python", "-m", "spikelab.mcp_server", "--transport", "sse"]
CMD ["--port", "8080", "--host", "0.0.0.0"]
