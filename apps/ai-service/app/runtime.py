"""One model-owning thread and a bounded FIFO of individual frames per process."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .frame import process_frame


class InferenceBusy(Exception):
    pass


class InferenceRuntime:
    def __init__(self, settings):
        self.settings = settings
        self._capacity = threading.BoundedSemaphore(settings.queue_size + 1)
        # Also bound uploads/decoding/encoding, before multipart parsing starts.
        self.upload_capacity = threading.BoundedSemaphore(settings.queue_size + 1)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plate-inference")
        self.closed = False
        try:
            self.detector, self.recognizer = self._executor.submit(self._load).result()
        except BaseException:
            self.close()
            raise

    def _load(self):
        from .pipeline import get_models

        detector, recognizer = get_models(self.settings)
        detector.warmup()
        recognizer.warmup()
        return detector, recognizer

    @property
    def name(self):
        return f"lpd:{self.detector.name}+lpr:{self.recognizer.name}"

    def process(self, image, *, job_id, frame_index=0, recognize=True):
        if self.closed or not self._capacity.acquire(blocking=False):
            raise InferenceBusy("inference capacity is full; retry a newer frame")
        enqueued = time.perf_counter()

        def work():
            started = time.perf_counter()
            result = process_frame(
                image,
                self.detector,
                self.recognizer,
                job_id=job_id,
                frame_index=frame_index,
                recognize=recognize,
            )
            result.timings_ms["queue"] = (started - enqueued) * 1000
            return result

        try:
            future = self._executor.submit(work)
        except BaseException:
            self._capacity.release()
            raise
        future.add_done_callback(lambda _: self._capacity.release())
        return future.result()

    def close(self):
        self.closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
