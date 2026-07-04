# Production Dockerfile for Music Analysis API
# Build: docker build -t musiclab-api -f deploy/Dockerfile .
# Run:   docker run -p 8777:8777 musiclab-api

FROM python:3.11-slim

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

# Install Python deps (librosa + essentia are heavy, pin versions)
COPY deploy/requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY deploy/server.py /app/

# Create tmp dir for audio files (volume mount target)
RUN mkdir -p /tmp/analysis && chmod 777 /tmp/analysis

EXPOSE 8777

# Health check: verify API is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8777/docs')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8777", "--workers", "1"]
