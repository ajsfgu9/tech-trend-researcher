FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

HEALTHCHECK --interval=5m --timeout=10s --start-period=2m --retries=3 \
    CMD find /app/data/heartbeat.txt -mmin -45 | grep -q . || exit 1

CMD ["python", "-u", "-m", "src.researcher"]
