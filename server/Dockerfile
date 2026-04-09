FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV EPILEPSY_TASK=task1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir .

EXPOSE 5000

CMD ["python", "-m", "server.app"]
