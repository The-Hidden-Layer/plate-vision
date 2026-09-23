import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from PIL import Image

from app import pipeline
from app.config import get_settings
from app.lpd import PlateDetector
from app.lpr.stub import StubRecognizer
from app.runtime import InferenceBusy, InferenceRuntime


def test_single_model_thread_bounded_queue_and_failure_releases_capacity(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    thread_ids = []

    class Detector(PlateDetector):
        def detect(self, image, **kwargs):
            thread_ids.append(threading.get_ident())
            entered.set()
            assert release.wait(5)
            if kwargs["job_id"] == "fail":
                raise RuntimeError("model failure")
            return []

    monkeypatch.setattr(pipeline, "get_models", lambda _: (Detector(), StubRecognizer()))
    runtime = InferenceRuntime(replace(get_settings(), queue_size=0))
    image = Image.new("RGB", (10, 10))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(runtime.process, image, job_id="fail")
            assert entered.wait(5)
            with pytest.raises(InferenceBusy):
                runtime.process(image, job_id="second")
            release.set()
            with pytest.raises(RuntimeError, match="model failure"):
                first.result()
        assert runtime.process(image, job_id="third").plates == []
        assert len(set(thread_ids)) == 1
    finally:
        release.set()
        runtime.close()
