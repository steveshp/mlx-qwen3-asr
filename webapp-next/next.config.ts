import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/health", destination: "http://localhost:8000/health" },
      {
        source: "/transcribe",
        destination: "http://localhost:8000/transcribe",
      },
      {
        source: "/transcribe/batch",
        destination: "http://localhost:8000/transcribe/batch",
      },
    ];
  },
  turbopack: {},
};

export default nextConfig;
