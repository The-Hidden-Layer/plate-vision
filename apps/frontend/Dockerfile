# syntax=docker/dockerfile:1
FROM node:22-slim

RUN corepack enable

WORKDIR /app

# Workspace manifests first so the dependency layer caches independently.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/frontend/package.json apps/frontend/package.json
# Workspace dependency of the frontend; its manifest must exist before install.
COPY libs/api-types/package.json libs/api-types/package.json
RUN --mount=type=cache,target=/root/.cache/pnpm pnpm install --frozen-lockfile

COPY . .

WORKDIR /app/apps/frontend
EXPOSE 3000
CMD ["pnpm", "exec", "next", "dev", "--port", "3000", "--hostname", "0.0.0.0"]
