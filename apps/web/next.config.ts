import type { NextConfig } from "next";

// No rewrite proxy — sse-client.ts talks directly to NEXT_PUBLIC_API_URL (the FastAPI
// backend's own origin) in both dev and production, and apps/api's CORSMiddleware
// (FRONTEND_ORIGIN env var) is what makes that cross-origin call work. See
// docs/PRODUCTION_DEPLOY.md for the local-dev env var setup this requires.
const nextConfig: NextConfig = {};

export default nextConfig;
