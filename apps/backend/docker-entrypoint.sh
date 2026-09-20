#!/bin/sh
# Applies migrations before starting, so a clean `docker compose up` works with
# no manual step. Only the `backend` service uses this entrypoint; `worker`
# waits for backend to become healthy instead, so migrations never race.
set -e

# Both go to stderr on purpose. They are progress logs, and stdout has to stay
# clean for `docker compose run` commands whose output is piped to a file —
# `nx run backend:openapi` redirects stdout straight into openapi.json.
echo "[entrypoint] applying migrations…" >&2
python manage.py migrate --noinput >&2

exec "$@"
