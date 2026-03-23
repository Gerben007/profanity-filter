FROM python:3.11-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper model at build time so the container starts offline.
# Override the model size with --build-arg WHISPER_MODEL=small (or medium, large-v2, etc.)
ARG WHISPER_MODEL=base
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

# Copy application source
COPY . .

# Runtime environment defaults (all overridable in docker-compose)
ENV WHISPER_MODEL=base \
    WHISPER_DEVICE=cpu \
    WATCH_FOLDER=/media \
    DB_PATH=/data/jobs.db \
    BADWORDS_PATH=/app/badwords.txt \
    FFMPEG_BIN=ffmpeg \
    MUTE_PADDING=0.1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
