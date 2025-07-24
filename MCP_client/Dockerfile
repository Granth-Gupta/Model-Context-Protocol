# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Runtime stage
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy dependencies from builder
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["uv", "run", "main.py"]
