//@ts-check

// Everything the browser needs is same-origin on :3000. The Next server
// proxies /api and /media to Django inside the docker network, which avoids
// CORS entirely and keeps the backend origin out of client code.
const backendOrigin = process.env.BACKEND_ORIGIN || 'http://localhost:8000';
const maxUploadMb = Number(
  process.env.MAX_UPLOAD_MB ?? process.env.NEXT_PUBLIC_MAX_UPLOAD_MB ?? '200',
);
if (!Number.isFinite(maxUploadMb) || maxUploadMb <= 0) {
  throw new Error('MAX_UPLOAD_MB must be a positive number');
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Next otherwise truncates rewritten request bodies at 10 MB. Allow the
    // advertised file size plus multipart headers; Django checks the file limit.
    proxyClientMaxBodySize: Math.ceil((maxUploadMb + 1) * 1024 * 1024),
    proxyTimeout: 600_000,
  },
  async rewrites() {
    return [
      { source: '/api/:path*', destination: `${backendOrigin}/api/:path*` },
      { source: '/media/:path*', destination: `${backendOrigin}/media/:path*` },
    ];
  },
};

module.exports = nextConfig;
