FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy requirements and install Python dependencies with uv
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# Copy application code
COPY . .

# Expose port 8001 (leaving server port)
EXPOSE 8001

# Use uv run instead of uvicorn directly
CMD ["uv", "run", "./main.py"]
