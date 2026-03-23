FROM python:3.11-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Runtime environment defaults (all overridable in docker-compose)
# HF_HOME points to the persistent volume so the Whisper model is downloaded
# once on first start and cached across container restarts/recreates.
ENV WHISPER_MODEL=base \
    WHISPER_DEVICE=cpu \
    WATCH_FOLDER=/media \
    DB_PATH=/data/jobs.db \
    BADWORDS_PATH=/app/badwords.txt \
    FFMPEG_BIN=ffmpeg \
    MUTE_PADDING=0.1 \
    HF_HOME=/data/models

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run entrypoint as root so it can fix /data ownership, then drops to uid 1000
ENTRYPOINT ["/entrypoint.sh"]
