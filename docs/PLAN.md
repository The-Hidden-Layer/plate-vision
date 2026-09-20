# plate-vision — Implementation Plan

License plate **detection** + **recognition** demo.
Nx monorepo · Next.js frontend · Django backend · FastAPI AI service · all in Docker on one laptop.

> The AI model is developed elsewhere and lands later. This repo ships a FastAPI
> **stub** behind a frozen HTTP contract so the whole pipeline is demoable today
> and the real model is a drop-in replacement.

---

## 1. Decisions (locked)

| Area | Decision |
|---|---|
| Monorepo | Nx; TS apps native, Python apps via hand-written `project.json` + `nx:run-commands` |
| Frontend | Next.js (App Router) + Tailwind |
| Backend | Django 5 + DRF + drf-spectacular |
| Async | Celery worker + Redis broker; Django never blocks on inference |
| AI boundary | HTTP. Worker → `POST http://ai-service:8100/infer` |
| Input | **image or video** — one Job model, `media_type` discriminator |
| Storage | Shared Docker named volume `media` at `/app/media` (backend, worker, ai-service) |
| Database | Postgres 16 container, `pgdata` volume |
| Result delivery | Frontend HTTP polling of `GET /api/jobs/{id}/` every 2s |
| API routing | Next.js `rewrites` proxy — browser is same-origin on `:3000`, no CORS |
| Types | drf-spectacular → `openapi.json` → `openapi-typescript` → `libs/api-types` |
| Python deps | `uv` + `pyproject.toml` + `uv.lock` |
| Dev mode | Bind-mounted source, hot reload in all three app containers |
| UI output | Detection table with plate crops + annotated preview frames |

**Out of scope:** auth/users, S3/MinIO, production compose, CI, full annotated video re-encode, GPU passthrough.

---

## 2. Target topology

```
browser :3000
   |  /api/*  /media/*      (same-origin; Next server proxies)
   v
frontend (Next.js dev server)
   |  docker network
backend (Django :8000) ------> db (postgres:16)
   |          |
   |          +--> redis :6379 --> worker (Celery; same image as backend)
   |                                  |  POST /infer
   |                                  v
   +-----------------------------> ai-service (FastAPI :8100)

shared volume `media` -> /app/media in backend, worker, ai-service
```

Six services. `worker` + `redis` are the cost of non-blocking uploads; `ai-service`
is the seam the model team plugs into.

---

## 3. Repo layout (target)

```
plate-vision/
├── apps/
│   ├── frontend/            Next.js app (Nx-native targets)
│   │   ├── app/
│   │   │   ├── page.tsx                 upload
│   │   │   ├── jobs/page.tsx            recent jobs
│   │   │   └── jobs/[id]/page.tsx       polling + results
│   │   ├── next.config.js               rewrites -> backend
│   │   └── Dockerfile
│   ├── backend/             Django project
│   │   ├── config/          settings, urls, celery.py
│   │   ├── jobs/            models, serializers, views, tasks, ai_client
│   │   ├── pyproject.toml   uv
│   │   ├── Dockerfile
│   │   └── project.json     nx run-commands
│   └── ai-service/          FastAPI stub
│       ├── app/main.py      /infer, /health
│       ├── app/stub.py      real decode + fake detections
│       ├── pyproject.toml
│       ├── Dockerfile
│       └── project.json
├── libs/
│   └── api-types/           generated TS types + openapi.json
├── docs/
│   ├── PLAN.md              this file
│   └── ai-contract.md       the frozen AI service contract
├── docker-compose.yml
├── .env.example
├── nx.json
└── README.md
```

---

## 4. Contracts

### 4.1 Data model

`Job`
- `id` UUID pk
- `media_type` — `image` | `video`
- `source_filename` — original upload name
- `media_path` — path relative to MEDIA_ROOT
- `status` — `queued` | `processing` | `done` | `failed`
- `error` — text, null unless failed
- `result_raw` — JSON, verbatim AI response (debugging while contract settles)
- `annotated_frames` — JSON list of paths
- `frame_count` — int, 1 for images
- `created_at` / `started_at` / `finished_at`

`Detection` (FK → Job, related_name `detections`)
- `plate_text`, `confidence` (float)
- `bbox` — JSON `[x1, y1, x2, y2]`, pixel coords
- `frame_index` — int, `0` for images
- `timestamp_ms` — int, null for images
- `crop_path` — path to cropped plate JPEG

### 4.2 REST API

| Method | Path | Notes |
|---|---|---|
| POST | `/api/jobs/` | multipart `file`; sniffs MIME → `media_type`; size cap; creates Job, dispatches task; `201` |
| GET | `/api/jobs/{id}/` | Job + nested detections + frame URLs — **the polling target** |
| GET | `/api/jobs/` | recent jobs, newest first |
| GET | `/api/schema/` | OpenAPI (drf-spectacular) |
| GET | `/api/docs/` | Swagger UI |
| GET | `/media/...` | dev-only static serve of the media volume |

### 4.3 AI service contract — FROZEN

```
POST /infer
  { "job_id": "<uuid>", "media_path": "uploads/<uuid>.mp4", "media_type": "video" }

200 { "media_type": "video",
      "frame_count": 300,
      "detections": [
        { "plate_text": "34ABC123", "confidence": 0.91,
          "bbox": [120, 340, 260, 392],
          "frame_index": 12, "timestamp_ms": 480,
          "crop_path": "jobs/<uuid>/crops/0012_0.jpg" }
      ],
      "annotated_frames": ["jobs/<uuid>/frames/0012.jpg"] }

GET /health -> { "status": "ok", "model": "stub" }
```

All paths are **relative to `/app/media`**, which both sides mount.
Errors: non-2xx with `{ "detail": "..." }`; the worker records it on `Job.error`.

This contract is what the model team builds against. Written to `docs/ai-contract.md`.

---

## Phases

Each phase has a **done-when** that is checkable by running something.

---

### Phase 0 — Workspace skeleton
**Goal:** an Nx workspace exists and `nx graph` runs.

- [x] `git init` at repo root
- [x] Create Nx workspace (`apps/` + `libs/` layout, pnpm)
- [x] Generate `apps/frontend` (Next.js, App Router, Tailwind)
- [x] `.gitignore` — node_modules, `.next`, `__pycache__`, `.venv`, `media/`, `.env`, `dist`, `.nx`
- [x] `.env.example` — `POSTGRES_*`, `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `REDIS_URL`, `AI_SERVICE_URL`, `MEDIA_ROOT`, `MAX_UPLOAD_MB`
- [x] `docs/ai-contract.md` (§4.3 verbatim) — unblocks the AI team immediately

**Done when:** `pnpm nx graph --file=/tmp/g.json` succeeds and `frontend` appears.

**Status: complete (Nx 23.2.1, Next.js 16.1, React 19, Tailwind v4).**
Deviations from plan, all deliberate:
- The `apps` preset now scaffolds a `packages/` dir; replaced with `apps/` + `libs/`
  and updated `pnpm-workspace.yaml`.
- pnpm 12 blocks postinstall scripts; approved builds for `@swc/core`, `nx`,
  `@parcel/watcher`, `sharp` (recorded in `pnpm-workspace.yaml` under `allowBuilds`).
- `@nx/next` no longer wires Tailwind; added Tailwind v4 via `@tailwindcss/postcss`.
- Deleted the generated `src/app/api/hello/route.ts` — a Next route handler at
  `/api/*` takes precedence over rewrites and would have shadowed the Django proxy.

---

### Phase 1 — Compose topology
**Goal:** all six services defined and the infra ones actually boot.

- [x] `docker-compose.yml`: `db`, `redis`, `backend`, `worker`, `ai-service`, `frontend`
- [x] Named volumes `pgdata`, `media`; `media` → `/app/media` in backend + worker + ai-service
- [x] Healthchecks on `db` and `redis`; `depends_on: condition: service_healthy`
- [x] Bind-mount source into all three app containers for hot reload
- [x] Port map: 3000 frontend, 8000 backend, 8100 ai-service, 5432 db (host debug)

**Done when:** `docker compose up db redis` reports both healthy.
**Requires:** Docker Desktop running.

**Status: complete.** `docker compose up db redis` -> both healthy; verified
Postgres 16.15 accepts the `.env` credentials, Redis answers `PONG`, and a file
written to the `media` volume by one container is readable by another.

Added beyond plan:
- `.dockerignore` at the repo root — the frontend build context is the whole
  monorepo, so without it every image build would ship `node_modules` and `.git`.
- `.env` created from `.env.example` (gitignored).
- Healthchecks also declared on `backend` (`/api/health/`) and `ai-service`
  (`/health`), since `worker` gates on `ai-service: service_healthy`.
  **Phase 2 owes Django an `/api/health/` endpoint** or `backend` never turns healthy.

Two bind-mount hazards, documented in the compose header and binding on later phases:
- Python venvs must live at `/opt/venv` (`UV_PROJECT_ENVIRONMENT`), not `/app` —
  the source bind mount would hide an in-project `.venv`.
- The frontend keeps `node_modules` and `.next` in anonymous volumes so the host
  mount cannot shadow the image's pnpm install.

---

### Phase 2 — Django backend
**Goal:** upload an image or video, get a `queued` Job back, poll it.

- [x] `pyproject.toml` (uv): django, djangorestframework, drf-spectacular, celery, redis, psycopg[binary], python-dotenv, gunicorn
- [x] Dockerfile: `uv sync --frozen`, non-root user, dev command `runserver 0.0.0.0:8000`
- [x] `config/settings.py` — env-driven, Postgres, `MEDIA_ROOT=/app/media`, Celery config
- [x] `config/celery.py` + app autodiscovery
- [x] `jobs` app: `Job` + `Detection` models, initial migration
- [x] Serializers: `JobCreateSerializer` (validates MIME + size, derives `media_type`), `JobDetailSerializer` (nested detections, absolute media URLs)
- [x] `JobViewSet` — create / retrieve / list
- [x] `GET /api/health/` — required by the `backend` compose healthcheck (Phase 1)
- [x] drf-spectacular at `/api/schema/` + `/api/docs/`
- [x] Dev media serving at `/media/`
- [x] pytest + pytest-django; tests for image upload, video upload, rejected file type, oversized file

**Done when:**
```
curl -F file=@sample.jpg http://localhost:8000/api/jobs/     # 201, status=queued
curl http://localhost:8000/api/jobs/<id>/                     # 200
```

**Status: complete.** 9 pytest tests pass; live API verified against the running
stack — `POST /api/jobs/` with a JPEG returns `201 queued`, `GET /api/jobs/{id}/`
returns `200`, video classified as `video`, unsupported type rejected `400`,
uploaded media served back over `/media/`, `/api/docs/` and `/api/schema/` render,
and the Celery app imports with the Redis broker configured.
`backend` reaches compose health via `/api/health/`.

Confirmed the Phase 1 hazards were handled:
- Non-root uid 1000 can write to the shared `media` volume — the volume inherited
  `app:app` from the image's `/app/media`. (Required deleting the root-owned volume
  left behind by the Phase 1 probe; `docker volume rm plate-vision_media`.)
- The venv at `/opt/venv` survives the `/app` bind mount.
- `makemigrations` inside the container writes through to the host tree.

Docker build networking needed tuning, kept in the Dockerfile:
- `uv sync` stalled indefinitely on the 9.5 MB ruff wheel. The host pulls the same
  file in 16s, so it is Docker Desktop's build network, not the lockfile or PyPI.
  Fixed with `UV_HTTP_TIMEOUT=300`, `UV_CONCURRENT_DOWNLOADS=2` (default parallelism
  saturates the VM's NAT) and a BuildKit cache mount so retries resume.
  Expect the same in Phase 3 — the ai-service image reuses these settings.

Deferred deliberately:
- Task dispatch from `JobViewSet.create` is Phase 4; the comment marking the spot
  is in `jobs/views.py`.
- `apps/backend/project.json` is Phase 7, so commands are run via
  `docker compose run --rm backend ...` for now.

---

### Phase 3 — AI service stub
**Goal:** a real, working `/infer` that produces genuine crops and annotated frames.

- [x] `pyproject.toml` (uv): fastapi, uvicorn, pydantic, pillow, opencv-python-headless
- [x] Dockerfile; dev command `uvicorn app.main:app --reload --host 0.0.0.0 --port 8100`
- [x] Pydantic request/response models matching §4.3 exactly
- [x] `stub.py`: **actually decodes** the media (Pillow for images, OpenCV for video),
      samples up to N frames, invents a plausible bbox + plate string per frame,
      writes real cropped JPEGs and boxed annotated JPEGs to `/app/media/jobs/<id>/`
- [x] Configurable fake latency (`STUB_DELAY_MS`) so the polling UI is observable
- [x] `/health`
- [x] Tests: image path, video path, missing file → 4xx

**Why a decoding stub and not hardcoded JSON:** the UI renders real images, frame
indices and timings are real, and the replacement only swaps the detector — the
I/O, paths and response shape are already exercised.

**Done when:** `curl -X POST ai-service:8100/infer -d '{...}'` returns detections
and the referenced JPEGs exist on the shared volume.

**Status: complete.** 9 contract tests pass. Live done-when: a real 60-frame MP4
uploaded through Django, then `POST localhost:8100/infer` -> `200` in 1.6s with
`frame_count=60`, 8 detections and 7 annotated frames. All 14 referenced files
were confirmed present **from inside the backend container** and served over HTTP
at `/media/...` — the shared volume works in both directions. Image path verified
separately: `frame_count=1`, `timestamp_ms=null`. Annotated JPEGs inspected
visually: green box plus plate label, legible at UI size.

Notes:
- Boxes are at random positions, not on the actual plate. That is the point of a
  stub — it decodes and writes real files, it does not detect.
- Output is deterministic per `job_id` (RNG seeded from it), so re-running a job
  gives identical results. There is a test for this.
- A video's plates recur across sampled frames (one or two "vehicles" per clip)
  rather than being random per frame, so the UI has realistic data to group.
- Added beyond plan: path-traversal rejection on `media_path`, and tests for
  determinism, relative-path compliance and `media_type` validation.
- `libglib2.0-0` is required in the image; `opencv-python-headless` still needs it.
- The Phase 2 uv network tuning was carried over and the image built first try,
  despite OpenCV being far larger than anything in the backend.

Error codes match the contract: missing file `404`, undecodable `422`, traversal
`400` — all 4xx, so the Phase 4 worker must not retry them.

Also fixed here: ruff now excludes `*/migrations/*` (Django-generated, not
hand-maintained). Both Python apps are lint-clean.

---

### Phase 4 — Wire the pipeline
**Goal:** upload → Celery → AI → DB → `done`, with no frontend involved.

- [ ] `jobs/ai_client.py` — httpx client, `AI_SERVICE_URL`, explicit timeout
- [ ] `jobs/tasks.py::process_job` — `processing` → call AI → persist `Detection` rows,
      `annotated_frames`, `frame_count`, `result_raw` → `done`
- [ ] Failure path: connection error / non-2xx / timeout → `failed` + readable `Job.error`
- [ ] Retries: 2 attempts with backoff on connection errors only, never on 4xx
- [ ] Dispatch the task from `JobViewSet.create` (via `transaction.on_commit`)
- [ ] Integration test with the AI service mocked

**Done when:** after `POST /api/jobs/`, polling flips `queued → processing → done`
and the detail response carries populated detections.

---

### Phase 5 — Frontend
**Goal:** the demo a human can actually drive.

- [ ] `next.config.js` rewrites: `/api/:p*` and `/media/:p*` → `http://backend:8000`
- [ ] Dockerfile; dev command `next dev`
- [ ] `/` — drag-and-drop accepting `image/*` and `video/*`, client-side size check,
      inline error display, POST then route to the job page
- [ ] `/jobs/[id]` — client component polling every 2s while `queued|processing`;
      stops on `done|failed`; status pill; error banner on failure
- [ ] Results: **detection table** (plate · confidence · frame/timestamp · crop `<img>`)
      and **annotated frames gallery**
- [ ] `/jobs` — recent jobs list
- [ ] Empty / loading / failed states for each page

**Done when:** drop a clip at `localhost:3000`, watch it progress, see plates and frames.

---

### Phase 6 — Type generation
**Goal:** the API contract cannot silently drift.

- [ ] `nx run backend:openapi` → `libs/api-types/openapi.json` (spectacular export)
- [ ] `libs/api-types` with `openapi-typescript` → `src/index.ts`, exported via tsconfig path
- [ ] Frontend fetch layer typed from the generated `Job` / `Detection`
- [ ] `generate` depends on `openapi` in Nx so one command refreshes both

**Done when:** renaming a serializer field and regenerating produces a frontend type error.

---

### Phase 7 — Nx targets, docs, smoke test
**Goal:** one obvious command per task; a fresh clone works.

- [ ] `apps/backend/project.json`: `serve`, `migrate`, `makemigrations`, `shell`, `test`, `lint`, `openapi`
- [ ] `apps/ai-service/project.json`: `serve`, `test`, `lint`
- [ ] Root `dev` target = `docker compose up`; `cache: false` on all serve targets;
      declare inputs on `test`/`lint` so caching is real
- [ ] Ruff for both Python apps
- [ ] `README.md` — prerequisites, `cp .env.example .env`, `docker compose up`, the URLs,
      common commands, and **how to swap the stub for the real model**
- [ ] End-to-end smoke test with a sample JPEG and a short MP4, from a clean
      `docker compose down -v`

**Done when:** `docker compose down -v && docker compose up` on a clean checkout
gets to a working upload with no manual steps beyond copying `.env`.

---

## 5. Known constraints / risks

- ~~Docker daemon not running~~ — started during Phase 1; `db` and `redis` verified healthy.
- **No GPU in Docker on Apple Silicon.** When the real model arrives it runs CPU-only
  inside the container. `AI_SERVICE_URL` is env-driven precisely so `ai-service` can be
  run natively on the host and pointed at via `host.docker.internal:8100` if inference
  is too slow. Plan for this; don't design around it yet.
- **Large uploads.** `MAX_UPLOAD_MB` is enforced in the serializer and checked client-side.
  Keep demo clips short — the stub samples frames, but the real model will not be fast.
- **Media volume grows unbounded.** No cleanup job in scope; `docker compose down -v` resets it.
- **Nx caching over Docker commands is shallow.** `serve` targets are uncached by design;
  only `test`/`lint` get meaningful cache hits.

## 6. Model hand-off checklist

What the AI team needs, and nothing more:
1. `docs/ai-contract.md` — request/response shapes, path conventions, error format.
2. They implement `POST /infer` + `GET /health`. Paths in the response are relative to `/app/media`.
3. They replace `apps/ai-service/app/stub.py`; `main.py` and the Pydantic models stay.
4. Verification: the existing ai-service tests must pass against the real implementation.
