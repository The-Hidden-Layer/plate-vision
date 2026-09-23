"""Bounds uploads before multipart parsing, including bodies without Content-Length."""

from fastapi import HTTPException
from starlette.responses import JSONResponse


class UploadLimitsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in {
            "/v1/plates/detect",
            "/v1/plates/recognize",
        }:
            return await self.app(scope, receive, send)
        runtime = scope["app"].state.runtime
        if not runtime.upload_capacity.acquire(blocking=False):
            return await JSONResponse(
                {"detail": "image request capacity is full"}, 503, headers={"Retry-After": "1"}
            )(scope, receive, send)
        limit = runtime.settings.max_upload_bytes + 64 * 1024  # bounded multipart overhead
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > limit:
                raise HTTPException(413, "multipart upload exceeds the request limit")
            return message

        try:
            headers = dict(scope["headers"])
            try:
                content_length = int(headers.get(b"content-length", b"0"))
            except ValueError:
                return await JSONResponse({"detail": "invalid Content-Length"}, 400)(
                    scope, receive, send
                )
            if content_length > limit:
                return await JSONResponse(
                    {"detail": "multipart upload exceeds the request limit"}, 413
                )(scope, receive, send)
            return await self.app(scope, bounded_receive, send)
        finally:
            runtime.upload_capacity.release()
