FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    DEBUG=false \
    APP_DATA_DIR=/tmp/deepdiabetic \
    ALLOWED_HOSTS=.hf.space,localhost,127.0.0.1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p "$APP_DATA_DIR" \
    && python manage.py collectstatic --noinput

EXPOSE 7860

CMD python manage.py migrate --noinput && python manage.py ensure_default_user && gunicorn Diabetic.wsgi:application --bind 0.0.0.0:${PORT} --workers 1 --timeout 180
