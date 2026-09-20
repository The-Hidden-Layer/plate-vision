# @org/api-types

TypeScript types generated from the backend's OpenAPI schema. The frontend
imports these so an API change breaks the build instead of production.

```bash
pnpm nx run api-types:generate
```

That runs `backend:openapi` first (drf-spectacular exports `openapi.json` from
the running backend container), then `openapi-typescript` regenerates
`src/schema.ts`.

- `src/schema.ts` is **generated** — never edit it.
- `src/index.ts` is hand-written and maps the raw schema names to friendly
  aliases. It is types only, so nothing is emitted at runtime.
- `openapi.json` is a build artifact and is gitignored; `src/schema.ts` is
  committed so a fresh clone type-checks without Docker running.
