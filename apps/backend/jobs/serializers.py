import uuid
from pathlib import Path

from django.conf import settings
from rest_framework import serializers

from .models import Detection, Job, MediaType

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def resolve_media_type(content_type: str | None, filename: str) -> str | None:
    """Classify an upload as image or video, or None if unsupported.

    The browser-supplied content type is checked first and the extension is used
    as a fallback, since some browsers send application/octet-stream for .mkv
    and similar.
    """
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in IMAGE_CONTENT_TYPES:
        return MediaType.IMAGE
    if ct in VIDEO_CONTENT_TYPES:
        return MediaType.VIDEO

    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    if ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    return None


def _media_url(relative_path: str | None) -> str | None:
    """Root-relative URL, e.g. /media/jobs/<id>/frames/0012.jpg.

    Deliberately NOT absolute. The browser reaches this API through the Next.js
    rewrites proxy, so Django sees Host: backend:8000 and build_absolute_uri
    would hand the browser a hostname it cannot resolve. A root-relative URL is
    correct both through the proxy and when calling Django directly.
    """
    if not relative_path:
        return None
    return f"{settings.MEDIA_URL}{relative_path.lstrip('/')}"


class DetectionSerializer(serializers.ModelSerializer):
    crop_url = serializers.SerializerMethodField()
    # Declared explicitly so the OpenAPI schema says number[4] rather than the
    # shapeless "unknown" a plain JSONField produces.
    bbox = serializers.ListField(
        child=serializers.IntegerField(),
        min_length=4,
        max_length=4,
        read_only=True,
        help_text="[x1, y1, x2, y2] pixel coords, top-left origin",
    )

    class Meta:
        model = Detection
        fields = [
            "id",
            "plate_text",
            "confidence",
            "bbox",
            "frame_index",
            "timestamp_ms",
            "crop_url",
        ]
        read_only_fields = fields

    def get_crop_url(self, obj: Detection) -> str | None:
        return _media_url(obj.crop_path)


class JobSerializer(serializers.ModelSerializer):
    """Read shape returned by create, retrieve and list. The polling target."""

    detections = DetectionSerializer(many=True, read_only=True)
    media_url = serializers.SerializerMethodField()
    annotated_frame_urls = serializers.SerializerMethodField()

    class Meta:
        model = Job
        fields = [
            "id",
            "status",
            "media_type",
            "source_filename",
            "media_url",
            "error",
            "frame_count",
            "annotated_frame_urls",
            "detections",
            "created_at",
            "started_at",
            "finished_at",
        ]
        # This serializer is never used for writes. Declaring that makes the
        # generated OpenAPI schema mark every field as always-present, instead
        # of optional-because-the-model-has-a-default.
        read_only_fields = fields

    def get_media_url(self, obj: Job) -> str | None:
        return _media_url(obj.media_path)

    def get_annotated_frame_urls(self, obj: Job) -> list[str]:
        return [_media_url(p) for p in (obj.annotated_frames or [])]


class JobCreateSerializer(serializers.Serializer):
    """Accepts the multipart upload and persists it to the shared media volume."""

    file = serializers.FileField(write_only=True)

    def validate_file(self, upload):
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if upload.size > max_bytes:
            raise serializers.ValidationError(
                f"File is {upload.size / 1024 / 1024:.1f} MB; the limit is "
                f"{settings.MAX_UPLOAD_MB} MB."
            )
        if resolve_media_type(upload.content_type, upload.name) is None:
            raise serializers.ValidationError(
                "Unsupported file type. Upload an image "
                f"({', '.join(sorted(IMAGE_EXTENSIONS))}) or a video "
                f"({', '.join(sorted(VIDEO_EXTENSIONS))})."
            )
        return upload

    def create(self, validated_data) -> Job:
        upload = validated_data["file"]
        media_type = resolve_media_type(upload.content_type, upload.name)

        job_id = uuid.uuid4()
        extension = Path(upload.name).suffix.lower()
        relative_path = f"uploads/{job_id}{extension}"
        destination = Path(settings.MEDIA_ROOT) / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as handle:
            for chunk in upload.chunks():
                handle.write(chunk)

        return Job.objects.create(
            id=job_id,
            media_type=media_type,
            source_filename=upload.name,
            media_path=relative_path,
        )


class HealthSerializer(serializers.Serializer):
    status = serializers.CharField()


class ValidationErrorSerializer(serializers.Serializer):
    """Shape of DRF's 400 body, so the schema documents the failure case too.

    DRF keys validation errors by field name; the only writable field here is
    `file`, so that is the only key an upload can fail on.
    """

    file = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="Reasons the upload was rejected: too large, or an unsupported type.",
    )
