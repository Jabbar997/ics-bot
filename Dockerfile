# ICS Telegram bot — portable image that runs identically on Railway / Render.
# The bot uses Telegram long-polling, so run EXACTLY ONE instance (replicas = 1).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code (config.yaml is copied too; .env / *.db are excluded by .dockerignore).
COPY . .

# Persistent SQLite lives on a mounted volume at /data.
# Set DATABASE_URL=sqlite:////data/ics.db and mount a volume at /data,
# OR point DATABASE_URL at a managed Postgres instead.
RUN mkdir -p /data
ENV DATABASE_URL=sqlite:////data/ics.db

CMD ["python", "-m", "app.main", "bot"]
