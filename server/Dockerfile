FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files from root
COPY . .

# PYTHONPATH so 'models' and 'server' are importable
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default task (overridable via docker run -e EPILEPSY_TASK=task2)
ENV EPILEPSY_TASK=task1

# Install Python dependencies from root (new pyproject.toml will be here)
# Since we are using pyproject.toml now, let's use pip install .
RUN pip install --no-cache-dir .

# Hugging Face Spaces uses port 7860 by default;
# we keep 5000 for local use and set app_port in README metadata.
EXPOSE 5000

CMD ["python", "server/app.py"]
