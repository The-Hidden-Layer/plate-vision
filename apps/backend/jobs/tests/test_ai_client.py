"""The retryable/permanent split is the whole point of ai_client."""

from unittest.mock import patch

import httpx
import pytest

from jobs import ai_client


def response(status: int, payload=None, text: str = "") -> httpx.Response:
    kwargs = {"json": payload} if payload is not None else {"text": text}
    return httpx.Response(status, request=httpx.Request("POST", "http://ai/infer"), **kwargs)


def call():
    return ai_client.infer(job_id="j", media_path="uploads/a.mp4", media_type="video")


def test_success_returns_payload():
    with patch("httpx.post", return_value=response(200, {"frame_count": 1})):
        assert call() == {"frame_count": 1}


@pytest.mark.parametrize("status", [400, 404, 422])
def test_4xx_is_permanent(status):
    with patch("httpx.post", return_value=response(status, {"detail": "bad input"})):
        with pytest.raises(ai_client.AIServiceRejected, match="bad input"):
            call()


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_is_retryable(status):
    with patch("httpx.post", return_value=response(status, {"detail": "boom"})):
        with pytest.raises(ai_client.AIServiceUnavailable):
            call()


def test_connection_error_is_retryable():
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(ai_client.AIServiceUnavailable, match="could not reach"):
            call()


def test_timeout_is_retryable():
    with patch("httpx.post", side_effect=httpx.ReadTimeout("slow")):
        with pytest.raises(ai_client.AIServiceUnavailable, match="did not respond within"):
            call()


def test_non_json_200_is_permanent():
    with patch("httpx.post", return_value=response(200, text="<html>oops</html>")):
        with pytest.raises(ai_client.AIServiceRejected, match="non-JSON"):
            call()


def test_error_detail_falls_back_to_body_text():
    with patch("httpx.post", return_value=response(503, text="upstream down")):
        with pytest.raises(ai_client.AIServiceUnavailable, match="upstream down"):
            call()
