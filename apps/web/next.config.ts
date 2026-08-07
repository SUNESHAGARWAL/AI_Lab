import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // Dev-only proxy to the FastAPI backend — avoids adding CORS middleware to a
    // public-facing API for what's currently a same-machine dev convenience.
    return [{ source: "/api/:path*", destination: "http://localhost:8000/:path*" }];
  },
};

export default nextConfig;
