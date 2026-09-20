#!/bin/sh
# Applies migrations before starting, so a clean `docker compose up` works with
# no manual step. Only the `backend` service uses this entrypoint; `worker`
# waits for backend to become healthy instead, so migrations never race.
set -e

echo "[entrypoint] applying migrations…"
python manage.py migrate --noinput

exec "$@"
