FROM python:3.10-slim

# Install system dependencies for OpenCV and general operation
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV EPILEPSY_TASK=task1
ENV PORT=5000

# Install the package and all dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# Expose the default port
EXPOSE 5000

# Start the server using the root server.py
CMD ["python", "server.py"]
