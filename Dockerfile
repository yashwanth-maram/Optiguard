FROM python:3.12-slim

WORKDIR /app

# Copy only what is needed
COPY pyproject.toml ./
COPY src/ ./src/
COPY apps/ ./apps/
COPY configs/ ./configs/
COPY evidence/ ./evidence/

# Install package + runtime deps (no GPU, no torch)
RUN pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# Expose console port
EXPOSE 8000

# Default: live inference on localhost. Override CMD for replay.
CMD ["python", "apps/serve.py", "--host", "0.0.0.0", "--port", "8000"]
