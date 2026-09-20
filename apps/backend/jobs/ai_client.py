"""HTTP client for the AI service (docs/ai-contract.md).

The distinction that matters here is retryable vs not. Per the contract, 4xx
means the input is permanently bad and retrying would just process it three
times; connection errors, timeouts and 5xx are worth another attempt.
"""

from __future__ import annotations

import httpx
from django.conf import settings


class AIServiceError(Exception):
    """Base class. Carries a message suitable for showing to the user."""


class AIServiceUnavailable(AIServiceError):
    """Transient: connection refused, timeout, or 5xx. Safe to retry."""


class AIServiceRejected(AIServiceError):
    """Permanent: 4xx or a malformed response. Retrying changes nothing."""


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500].strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:500]


def infer(*, job_id: str, media_path: str, media_type: str) -> dict:
    url = f"{settings.AI_SERVICE_URL.rstrip('/')}/infer"
    request = {"job_id": job_id, "media_path": media_path, "media_type": media_type}

    try:
        response = httpx.post(url, json=request, timeout=settings.AI_REQUEST_TIMEOUT_SECONDS)
    except httpx.TimeoutException as exc:
        raise AIServiceUnavailable(
            f"AI service did not respond within {settings.AI_REQUEST_TIMEOUT_SECONDS}s"
        ) from exc
    except httpx.RequestError as exc:
        raise AIServiceUnavailable(f"could not reach AI service at {url}: {exc}") from exc

    if response.status_code >= 500:
        raise AIServiceUnavailable(f"AI service error: {_detail(response)}")
    if response.status_code >= 400:
        raise AIServiceRejected(_detail(response))

    try:
        payload = response.json()
    except ValueError as exc:
        raise AIServiceRejected("AI service returned a non-JSON response") from exc

    if not isinstance(payload, dict):
        raise AIServiceRejected("AI service returned an unexpected response shape")
    return payload
