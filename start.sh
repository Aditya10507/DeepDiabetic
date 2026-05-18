#!/usr/bin/env bash
set -o errexit

PORT="${PORT:-8000}"

echo "Running Django migrations before startup..."
timeout 60s python manage.py migrate --no-input || echo "Migration step failed or timed out; continuing so the web service can start."

echo "Starting Gunicorn on port ${PORT}..."
python -m gunicorn Diabetic.wsgi:application --bind "0.0.0.0:${PORT}" --workers 1 --timeout 120 --access-logfile - --error-logfile - --capture-output
