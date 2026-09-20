import uuid

from django.db import models


class MediaType(models.TextChoices):
    IMAGE = "image", "Image"
    VIDEO = "video", "Video"


class JobStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    PROCESSING = "processing", "Processing"
    DONE = "done", "Done"
    FAILED = "failed", "Failed"


class Job(models.Model):
    """One uploaded image or video and the result of running it through the AI service."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    media_type = models.CharField(max_length=8, choices=MediaType.choices)
    source_filename = models.CharField(max_length=255)
    # Relative to MEDIA_ROOT, e.g. "uploads/<id>.mp4". The AI service resolves
    # the same path against its own mount of the shared volume.
    media_path = models.CharField(max_length=512)

    status = models.CharField(
        max_length=12, choices=JobStatus.choices, default=JobStatus.QUEUED, db_index=True
    )
    error = models.TextField(blank=True, default="")

    # Verbatim AI response, kept for debugging while the contract settles.
    result_raw = models.JSONField(null=True, blank=True)
    annotated_frames = models.JSONField(default=list, blank=True)
    frame_count = models.PositiveIntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.media_type} {self.id} ({self.status})"


class Detection(models.Model):
    """A single recognized plate, in one frame."""

    job = models.ForeignKey(Job, related_name="detections", on_delete=models.CASCADE)
    plate_text = models.CharField(max_length=32, blank=True)
    confidence = models.FloatField()
    bbox = models.JSONField(help_text="[x1, y1, x2, y2] pixel coords, top-left origin")
    frame_index = models.PositiveIntegerField(default=0)
    timestamp_ms = models.PositiveIntegerField(null=True, blank=True)
    crop_path = models.CharField(max_length=512)

    class Meta:
        ordering = ["frame_index", "id"]

    def __str__(self) -> str:
        return self.plate_text or "<unreadable>"
