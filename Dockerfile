FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/ ./src/

USER appuser

HEALTHCHECK --interval=5m --timeout=10s --start-period=2m --retries=3 \
    CMD find /app/data/heartbeat.txt -mmin -45 | grep -q . || exit 1

CMD ["python", "-u", "-m", "src.researcher"]
