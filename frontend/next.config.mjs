import { fileURLToPath } from "url";
import { dirname } from "path";

const currentDir = dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  turbopack: {
    root: currentDir
  }
};

export default nextConfig;
