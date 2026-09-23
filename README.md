# plate-vision

License plate **detection** and **recognition** demo. Upload an image or a video,
watch it process, see the plates.

Nx monorepo · Next.js frontend · Django backend · FastAPI AI service · Docker.

> The AI service supports YOLO detection and Iranian LPRNet recognition, plus
> explicit stub backends for demos. Train and evaluate models before enabling
> real inference. See [the model workflow](docs/model-workflow.md).

---

## Quickstart

Prerequisites: **Docker Desktop** (running) and nothing else. Node and Python are
only needed if you want to run Nx commands outside Docker.

The example configuration uses **demo stubs**: their boxes and plate strings are
generated placeholders. For actual plate readings, use the trained deployment
in [the model workflow](docs/model-workflow.md). Keep an existing `.env` when
restarting an already configured installation.

```bash
cp -n .env.example .env
docker compose up
```

First run builds four images and takes a few minutes. Then:

| URL | What |
|---|---|
| http://localhost:3000 | The app — upload here |
| http://localhost:3000/jobs | Recent jobs |
| http://localhost:8000/api/docs | **Swagger UI — backend API** |
| http://localhost:8000/api/redoc | ReDoc — same schema, reference layout |
| http://localhost:8000/api/schema | Raw OpenAPI 3.1 (YAML; add `?format=json`) |
| http://localhost:8000/admin | Django admin (`nx run backend:shell` to make a user) |
| http://localhost:8100/docs | **Swagger UI — AI service** |
| http://localhost:8100/redoc | ReDoc — same schema, reference layout |
| http://localhost:8100/openapi.json | Raw OpenAPI 3.1 |
| http://localhost:8100/health | AI service liveness |

The backend's docs are also reachable through the Next proxy at
http://localhost:3000/api/docs, since it forwards all of `/api`. The AI service
is internal — only the worker calls it — so it has no proxied route; reach its
docs on :8100 directly.

Migrations are applied automatically when `backend` starts.

## How it works

```
browser :3000
   |  /api/*  /media/*      (same-origin; the Next server proxies)
   v
frontend (Next.js)
   |  docker network
backend (Django :8000) ------> db (postgres)
   |          |
   |          +--> redis --> worker (Celery)
   |                            |  POST /infer
   |                            v
   +---------------------> ai-service (FastAPI :8100)

shared volume `media` -> /app/media in backend, worker, ai-service
```

Uploads are never processed in the request. Django writes the file to the shared
volume, creates a `Job`, and queues a Celery task. The worker calls the AI service
over HTTP, writes `Detection` rows, and flips the job to `done`. The browser polls
`GET /api/jobs/{id}` every 2s until the job reaches a terminal state.

## Commands

Everything runs in Docker; Nx wraps it.

```bash
pnpm dev                      # docker compose up
pnpm dev:build                # rebuild images and start
pnpm down                     # stop
pnpm reset                    # stop AND delete the database + uploaded media

pnpm nx run backend:test      # pytest
pnpm nx run backend:lint      # ruff
pnpm nx run backend:migrate
pnpm nx run backend:makemigrations
pnpm nx run backend:shell

pnpm nx run ai-service:test
pnpm nx run ai-service:lint

pnpm nx run frontend:build    # also type-checks
pnpm nx run frontend:lint
pnpm nx run frontend:test

pnpm nx run api-types:generate  # re-export OpenAPI -> regenerate TS types
pnpm nx run ai-service:openapi  # export the AI service schema to libs/api-types/ai-openapi.json
pnpm nx run-many -t lint test   # everything
```

## API types are generated

The frontend does not hand-write API types. `drf-spectacular` exports the schema
and `openapi-typescript` turns it into `libs/api-types`. **A backend field rename
breaks the frontend build instead of production:**

```bash
pnpm nx run api-types:generate
```

`libs/api-types/src/schema.ts` is generated and committed; never edit it by hand.

## Real Iranian plate models

See [the model workflow](docs/model-workflow.md) for dataset preparation, YOLO26
and LPRNet training, the image upload endpoints, native MPS/NVIDIA deployment,
checkpoint export, and accuracy-preserving runtime selection. The existing
Django/Celery upload flow uses those same models through `/infer`.

## Configuration

All of it lives in `.env` (copy from `.env.example`). The ones worth knowing:

| Variable | Default | Notes |
|---|---|---|
| `MAX_UPLOAD_MB` | 200 | Enforced server-side and mirrored in the browser |
| `AI_SERVICE_URL` | `http://ai-service:8100` | Point at the host to bypass Docker |
| `AI_REQUEST_TIMEOUT_SECONDS` | 600 | Worker gives up after this |
| `AI_RETRY_BACKOFF_SECONDS` | 5 | Doubles per retry (5s, 10s), 3 attempts total |
| `LPD_BACKEND` | `stub` | Detection stage: `stub`, or `model` for `app/lpd/model.py` |
| `LPR_BACKEND` | `stub` | Recognition stage: `stub`, or `model` for `app/lpr/model.py` |
| `LPD_WEIGHTS` / `LPR_WEIGHTS` | — | Paths inside the container, e.g. `/app/weights/lpd/best.pt` |
| `VIDEO_SAMPLE_FPS` | 2 | Target analyzed frames per second across the whole video; replaces the retired `MAX_FRAMES` cap |
| `STUB_DELAY_MS` | 0 | Fake latency so the polling UI is visible. Stub only |

## Troubleshooting

**A URL 404s or redirects oddly after a code change.** Browsers cache 301s
permanently. Hard-reload (⇧⌘R) before debugging anything else.

**The frontend re-installs packages on every start.** Its `node_modules` lives in
an anonymous volume, and compose preserves those across `--force-recreate`. After
changing frontend dependencies:

```bash
docker compose up -d --force-recreate --renew-anon-volumes frontend
```

**`uv sync` hangs during an image build.** Docker Desktop's build network is much
slower than the host's. The Dockerfiles already set `UV_HTTP_TIMEOUT=300` and
`UV_CONCURRENT_DOWNLOADS=2`; if it still stalls, retry — the BuildKit cache mount
resumes rather than restarting.

**Jobs stay `queued` forever.** The worker is not running or cannot reach Redis:
`docker compose logs worker`.

**Jobs fail with "could not reach AI service".** `docker compose ps ai-service` —
the worker retries twice with backoff before giving up.

## Repository layout

```
apps/
  frontend/     Next.js (App Router, Tailwind)
  backend/      Django + DRF + Celery
  ai-service/   FastAPI — the stub to be replaced
libs/
  api-types/    TypeScript types generated from the OpenAPI schema
docs/
  PLAN.md         phased build plan, with what actually happened
  ai-contract.md  the frozen AI service contract
```

## Scope

Deliberately not included: authentication, object storage, a production compose
file, CI, and full annotated-video re-encoding. This is a local demo.
