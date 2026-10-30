# Minimal image for the Docker sandbox backend (verifiers run with --network none).
FROM python:3.12-slim
RUN pip install --no-cache-dir "pytest>=8,<10" "pyyaml>=6,<7" && useradd -m runner
USER runner
WORKDIR /work
