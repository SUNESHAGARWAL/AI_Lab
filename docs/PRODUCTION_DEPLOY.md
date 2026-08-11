# Production deployment

Topology: **Railway** (FastAPI backend, Docker), **Vercel** (Next.js frontend), **Neon**
(Postgres + pgvector), **Upstash** (Redis). Everything here is config, commands, and
verification steps — no secret value is ever written to this file or committed anywhere.
You paste real values into each platform's own dashboard.

## Deploy order

Neon → Upstash → Railway → Vercel. Railway needs a live `DATABASE_URL`/`REDIS_URL` at
boot; Vercel needs Railway's live public URL for `NEXT_PUBLIC_API_URL`. Doing it in this
order means each stage's prerequisites already exist when you reach it.

---

## Stage 1 — Railway (backend)

### Build config

`railway.json` (repo root) already points Railway at `apps/api/Dockerfile` with
`healthcheckPath: /health`. **Set the service's build context/root directory to the repo
root**, not `apps/api` — the Dockerfile is a uv workspace build and needs the sibling
`packages/*` directories visible (see the Dockerfile's own top comment).

### Environment variables (set in Railway's dashboard)

| Variable | Required | Value |
|---|---|---|
| `DATABASE_URL` | yes | Neon **pooled** connection string (has `-pooler` in the host) |
| `MIGRATIONS_DATABASE_URL` | yes | Neon **direct** connection string (no `-pooler`) — migrations run over this, not `DATABASE_URL`; running them over the pooled connection concurrently with the pool's own startup crashes uvicorn outright (reproduced directly — silent crash, only under uvicorn + a pooled connection) |
| `REDIS_URL` | yes | Upstash Redis connection string |
| `DEEPSEEK_API_KEY` | yes | your DeepSeek key |
| `GROQ_API_KEY` | yes | your Groq key |
| `FRONTEND_ORIGIN` | yes | your Vercel domain, e.g. `https://your-app.vercel.app` (comma-separate if you also want to allow a preview domain) |
| `PORT` | yes | `8000`. Railway's healthcheck probes the port named by `PORT` and reports `service unavailable` against anything else — without it every probe fails and the deploy never goes live even though the app is serving correctly. Railway injects only `RAILWAY_*` vars, never `PORT`, so it must be set by hand; the Dockerfile binds `${PORT:-8000}` to match. |
| `APP_ENV` | recommended | `production` |
| `MAX_QUERY_LENGTH` | optional | defaults to `2000` |
| `LIVE_QUERY_RATE_LIMIT_PER_HOUR` | optional | defaults to `5` |
| `GEMINI_API_KEY` | optional | unused while Gemini is disabled in `packages/llm/src/llm/registry.py` |

Any `LLM_*`-prefixed override (`packages/llm/src/llm/config.py`'s `GatewaySettings`) is
optional — every field there already has a production-reasonable default.

### Verify

```bash
curl https://<railway-domain>/health
# {"status":"ok","env":"production"}

curl -X POST https://<railway-domain>/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a data protection impact assessment under GDPR?"}'
# real SSE frames (id:/event:/data: lines), not a connection error
```

Check Railway's deploy logs for a clean `api.startup` log line and no missing-config
crash (the app fails loudly at import time if `DATABASE_URL`/`REDIS_URL` are missing —
you'll see it immediately in the logs, not as a mysterious later failure).

---

## Stage 2 — Neon (database)

No code changes needed — `packages/retrieval/migrations/versions/0001_create_chunks.py`
already runs `CREATE EXTENSION IF NOT EXISTS vector` idempotently, and the app
self-migrates on every boot regardless (`apps/api/src/api/main.py`'s lifespan). This
stage is about seeding the corpus, run **locally**, pointed at Neon's **direct**
(non-pooled) connection string — one-off DDL/bulk-write work should go through the
direct endpoint, not the pooler; the app's own runtime traffic is what pooling is for.

```bash
export DATABASE_URL="<neon-direct-connection-string>"   # no "-pooler" in the host
uv run --package ingest python -m ingest.cli ingest-corpus
```

This single command runs the Alembic migration (idempotent) and then ingests both the
EU AI Act and GDPR (idempotent — `INSERT ... ON CONFLICT (chunk_id) DO UPDATE`, safe to
re-run any time, e.g. after a source document change). Expect:

- **~1,437 chunks** total (793 EU AI Act: 113 articles, 180 recitals; 644 GDPR: 99
  articles, 173 recitals)
- ~6s model load + ~25s CPU encode time, but **a few minutes wall time overall** — the
  1,437 individual upsert round-trips to Neon over the network dominate, not the local
  embedding step
- **$0 cost** (local CPU compute, Neon free-tier writes)

### Verify

In the Neon SQL console (or `psql "<direct-or-pooled-url>"`):

```sql
SELECT count(*) FROM chunks;
-- ~1437
```

Remember: `DATABASE_URL` you export locally for this command is the **direct** string;
Railway's `DATABASE_URL` env var (Stage 1) is the **pooled** string — they're different
values from the same Neon project.

---

## Stage 3 — Vercel (frontend)

### Project settings

- **Root Directory**: `apps/web` (it has no `workspace:*` deps — it's the pnpm
  workspace's only member — and builds standalone from here)
- **Framework Preset**: Next.js (auto-detected)
- **Build/Install commands**: leave as Vercel's defaults; `apps/web/package.json` now
  pins `"packageManager": "pnpm@9.15.0"` so corepack picks the right version
  automatically even though Vercel's root directory won't see the repo-root
  `package.json`'s own pin

### Environment variables

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | your Railway backend's public URL, e.g. `https://your-api.up.railway.app` (no trailing slash) |

### Verify

1. Load the deployed Vercel URL. Click a cached example question — this works **even if
   Railway is paused or down**, since `lib/replay-client.ts`/`lib/example-fixtures.json`
   are fully self-contained (no network call at all). Worth actually testing this by
   pausing the Railway service once, to confirm the headline demo never depends on
   backend uptime.
2. Type a free-form question — this needs Railway up. Confirm in the browser's Network
   tab that the SSE request goes to the Railway domain, not `localhost`.
3. Open the browser console: no errors.

---

## Stage 4 — cross-cutting checks

### CORS

```bash
# From an allowed origin (simulated) — should succeed / return the CORS header:
curl -i -X OPTIONS https://<railway-domain>/query/stream \
  -H "Origin: https://your-app.vercel.app" \
  -H "Access-Control-Request-Method: POST" | grep -i access-control-allow-origin

# From a disallowed origin — should NOT return the header at all:
curl -i -X OPTIONS https://<railway-domain>/query/stream \
  -H "Origin: https://evil.example" \
  -H "Access-Control-Request-Method: POST" | grep -i access-control-allow-origin
# (no output = correctly rejected)
```

### Secrets

- `git log -p -- .env` should show nothing (the file is gitignored, never committed).
  `.github/workflows/gitleaks.yml` scans every push/PR for accidentally-committed
  secrets going forward.
- Re-confirm no real key/URL value appears anywhere in this repo outside your own local
  `.env`/`.env.local` files (both gitignored) — `.env.example` and
  `apps/web/.env.example` document variable *names* only.

### Rate limiting / budget ceiling (already built and tested — see the demo-hardening
pass's tests in `apps/api/tests/test_ratelimit.py`/`test_stream.py`)

Hammer the live-query path past `LIVE_QUERY_RATE_LIMIT_PER_HOUR` and confirm the
friendly in-band message appears (not a raw error), with a working link back to the
free example questions. Cached examples keep working throughout — they're exempt by
design (see `apps/api/src/api/ratelimit.py`'s module docstring).
