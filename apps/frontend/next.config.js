//@ts-check

// Everything the browser needs is same-origin on :3000. The Next server
// proxies /api and /media to Django inside the docker network, which avoids
// CORS entirely and keeps the backend origin out of client code.
const backendOrigin = process.env.BACKEND_ORIGIN || 'http://localhost:8000';

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: '/api/:path*', destination: `${backendOrigin}/api/:path*` },
      { source: '/media/:path*', destination: `${backendOrigin}/media/:path*` },
    ];
  },
};

module.exports = nextConfig;
