FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --upgrade pip && pip install uv

# Copy project configuration files
COPY pyproject.toml* ./
COPY uv.lock* ./
COPY requirements.txt* ./

# Install dependencies
RUN if [ -f "pyproject.toml" ]; then \
        uv sync --frozen; \
    elif [ -f "requirements.toml" ]; then \
        uv pip install --no-cache -r requirements.txt; \
    else \
        echo "Installing basic dependencies..."; \
        uv pip install fastapi uvicorn pydantic; \
    fi

# Copy source code
COPY . .

# Expose port 8001 (different port to avoid conflicts)
EXPOSE 8001

# Run with uv
CMD ["uv", "run", "main.py"]
