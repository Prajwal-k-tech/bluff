import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  ...(process.env.BLUFF_STATIC_EXPORT === "1"
    ? { output: "export" as const, trailingSlash: true }
    : {}),
};

export default nextConfig;
