# Production deployment

Topology: **Google Cloud Run** (FastAPI backend, Docker), **Vercel** (Next.js frontend),
**Neon** (Postgres + pgvector), **Upstash** (Redis). Everything here is config, commands,
and verification steps — no secret value is ever written to this file or committed
anywhere. You paste real values into each platform's own dashboard or secret store.

## Deploy order

Neon → Upstash → Cloud Run → Vercel. Cloud Run needs a live `DATABASE_URL`/`REDIS_URL` at
boot; Vercel needs the Cloud Run service URL for `NEXT_PUBLIC_API_URL`.

---

## Stage 1 — Cloud Run (backend)

### Why Cloud Run, and what keeps it at $0

The API image loads torch and `bge-small` in process: peak RSS is ~590 MB, so every
512 MB free tier (Render, Koyeb, Back4app) is out. Cloud Run's always-free tier —
2M requests, 180k GiB-seconds, 360k vCPU-seconds a month — at 1 GiB with request-based
billing is ~50 active hours a month, far above a demo's traffic. It needs a billing
account with a card, so three guardrails are part of the config, not optional:

- `--max-instances 1` — the ceiling on concurrent spend, whatever the traffic.
- `--min-instances 0` — scale to zero; idle time is free. Cost: a ~10–20 s cold start.
- A **$1 budget alert** on the billing account.

The image lives on **public GHCR**, not Artifact Registry: Cloud Run pulls public GHCR
images directly, and the 2.6 GB image would exceed Artifact Registry's 0.5 GB free
storage. The image holds no secrets.

### Image

`.github/workflows/publish-api.yml` builds `apps/api/Dockerfile` from the repo root on
every push to `main` that touches the API's dependency closure, and pushes
`ghcr.io/suneshagarwal/ai-lab-api:{latest,<sha>}`. After the first run, set the package's
visibility to **public** (GitHub → Packages → ai-lab-api → Package settings). Deploy by
`:<sha>` so a rollback is one command with an older sha.

### Secrets

Secret Manager, one secret per credential: `DATABASE_URL`, `MIGRATIONS_DATABASE_URL`,
`REDIS_URL`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY` — five active versions, inside the free
six. Grant the service's runtime service account
`roles/secretmanager.secretAccessor`. When rotating, destroy the old version so the
count stays under six.

### Deploy

```bash
gcloud run deploy ai-lab-api \
  --image ghcr.io/suneshagarwal/ai-lab-api:<sha> \
  --region europe-west1 --allow-unauthenticated \
  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 1 \
  --concurrency 20 --timeout 300 \
  --startup-probe httpGet.path=/health,initialDelaySeconds=0,timeoutSeconds=5,periodSeconds=10,failureThreshold=12 \
  --set-env-vars APP_ENV=production,FRONTEND_ORIGIN=https://<vercel-domain>,CLIENT_IP_HEADER=x-forwarded-for,TRUSTED_PROXY_HOPS=<n> \
  --set-secrets DATABASE_URL=DATABASE_URL:latest,MIGRATIONS_DATABASE_URL=MIGRATIONS_DATABASE_URL:latest,REDIS_URL=REDIS_URL:latest,DEEPSEEK_API_KEY=DEEPSEEK_API_KEY:latest,GROQ_API_KEY=GROQ_API_KEY:latest
```

| Setting | Why |
|---|---|
| `europe-west1` | Any tier-1 region works (Tier-1 pricing, where the free allowance goes furthest); pick the one nearest the Neon and Upstash regions — every query makes several DB and Redis round trips. |
| `--timeout 300` | A live query streams over SSE for well under a minute; 300 s leaves room for a cold Neon plus provider retries. |
| startup probe on `/health` | Up to 120 s for the model load and self-migration. `/health` touches no dependency — see below. The Dockerfile's `HEALTHCHECK` is ignored by Cloud Run. |
| `PORT` | Not set — Cloud Run injects `PORT=8080`, and the Dockerfile binds `${PORT:-8000}`. |
| `DATABASE_URL` | Neon **pooled** string (`-pooler` in the host). |
| `MIGRATIONS_DATABASE_URL` | Neon **direct** string. Migrations over the pooled connection, concurrent with the pool's own startup, crash uvicorn outright (reproduced — silent crash, only under uvicorn + a pooled connection). |
| `MAX_QUERY_LENGTH`, `LIVE_QUERY_RATE_LIMIT_PER_HOUR` | Optional; defaults `2000`, `5`. |
| `LLM_*` | Optional; `packages/llm/src/llm/config.py`'s `GatewaySettings` defaults are production-reasonable. |

### Client IP for the rate limiter

The socket address is Google's front end, so the per-IP limit reads a header. Cloud Run
does **not** set `X-Real-IP` — trusting it there would let any caller pick their own
bucket. It *appends* to `X-Forwarded-For`, so the real client sits a fixed number of
entries from the right; `TRUSTED_PROXY_HOPS` is that number (see
`api.routes.stream.client_key_for`).

Pin the value empirically rather than from memory — Google has changed the XFF shape
before. Deploy with `TRUSTED_PROXY_HOPS=1`, send one live query with a spoofed header,
and read the resulting key in Upstash's data browser (`api:ratelimit:<ip>:<hour>`):

```bash
curl -N -X POST https://<service-url>/query/stream \
  -H "Content-Type: application/json" -H "X-Forwarded-For: 1.2.3.4" \
  -d '{"query":"What is a data protection impact assessment under GDPR?"}'
```

- key has your real IP → `1` is right.
- key has a Google address (`35.191.*`, `130.211.*`, …) → Google appended two entries;
  redeploy with `2` and repeat.
- key has `1.2.3.4` → never happens with the right value; the spoof reached the key.

### Verify

```bash
curl https://<service-url>/health
# {"status":"ok","env":"production"}

curl https://<service-url>/ready
# {"status":"ok","checks":{"postgres":"ok","redis":"ok"}}
# 503 + {"status":"degraded",...} names whichever dependency is unreachable.

curl -N -X POST https://<service-url>/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a data protection impact assessment under GDPR?"}'
# real SSE frames (id:/event:/data: lines), not a connection error
```

Check the service's logs (Cloud Run → Logs) for a clean `api.startup` line and no
missing-config crash (the app fails loudly at import time if `DATABASE_URL`/`REDIS_URL`
are missing).

**`/health` vs `/ready`.** `/health` is liveness — is the process serving? — and
deliberately touches no dependency, because the platform kills an instance whose probe
fails, and a suspended Neon compute would otherwise trigger a restart loop over a
database that is merely asleep. `/ready` is the one that answers "can this instance
actually serve a query", and it is the endpoint to check when the frontend reports
errors but `/health` looks fine. That combination — `/health` green while every single
query failed — is exactly what made a dead backend look like a frontend bug once, when
the checkpointer was holding a Postgres connection Neon had already severed.

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
Cloud Run's `DATABASE_URL` secret (Stage 1) is the **pooled** string — they're different
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
| `NEXT_PUBLIC_API_URL` | the Cloud Run service URL, e.g. `https://ai-lab-api-xxxxx.europe-west1.run.app` (no trailing slash) |

### Verify

1. Load the deployed Vercel URL. Click a cached example question — this works **even if
   the API is down**, since `lib/replay-client.ts`/`lib/example-fixtures.json`
   are fully self-contained (no network call at all). Worth actually testing this once
   with the API unreachable, to confirm the headline demo never depends on
   backend uptime.
2. Type a free-form question — this needs the API up. Confirm in the browser's Network
   tab that the SSE request goes to the `run.app` URL, not `localhost`.
3. Open the browser console: no errors.

---

## Stage 4 — cross-cutting checks

### CORS

```bash
# From an allowed origin (simulated) — should succeed / return the CORS header:
curl -i -X OPTIONS https://<service-url>/query/stream \
  -H "Origin: https://your-app.vercel.app" \
  -H "Access-Control-Request-Method: POST" | grep -i access-control-allow-origin

# From a disallowed origin — should NOT return the header at all:
curl -i -X OPTIONS https://<service-url>/query/stream \
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
