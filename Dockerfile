FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and platform
COPY src/ /app/src/
COPY main.py /app/main.py
COPY tests/ /app/tests/

# Expose port for FastAPI explorer
EXPOSE 8000

# Default command: launch the FastAPI service
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]

