from django.db import transaction
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import (
    OpenApiExample,
    OpenApiParameter,
    extend_schema,
    extend_schema_view,
)
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import api_view
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import Job
from .serializers import (
    HealthSerializer,
    JobCreateSerializer,
    JobSerializer,
    ValidationErrorSerializer,
)
from .tasks import process_job


@extend_schema(
    tags=["health"],
    summary="Liveness probe",
    responses=HealthSerializer,
    description=(
        "Used by the compose healthcheck. Answers as soon as Django is serving; "
        "it does not touch the database, Redis or the AI service."
    ),
    examples=[OpenApiExample("ok", value={"status": "ok"}, response_only=True)],
)
@api_view(["GET"])
def health(_request):
    return Response({"status": "ok"})


@extend_schema(tags=["jobs"])
@extend_schema_view(
    list=extend_schema(
        summary="List jobs",
        description="Newest first, paginated 20 per page via the `page` query parameter.",
    ),
    retrieve=extend_schema(
        summary="Retrieve a job",
        description=(
            "The polling target. Re-request until `status` is `done` or `failed`; "
            "`detections` and `annotated_frame_urls` are only populated once `done`."
        ),
        parameters=[
            OpenApiParameter(
                "id",
                type=OpenApiTypes.UUID,
                location=OpenApiParameter.PATH,
                description="Job UUID returned by the upload.",
            )
        ],
    ),
)
class JobViewSet(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    """Upload media, then poll the job until it is done or failed."""

    queryset = Job.objects.prefetch_related("detections")
    parser_classes = [MultiPartParser, FormParser]

    def get_serializer_class(self):
        return JobCreateSerializer if self.action == "create" else JobSerializer

    @extend_schema(
        summary="Upload media and queue a job",
        description=(
            "`multipart/form-data` with a single `file` part — an image "
            "(.jpg/.jpeg/.png/.webp/.bmp) or a video (.mp4/.mov/.avi/.mkv/.webm). "
            "Returns the job in `queued` state; poll `GET /api/jobs/{id}` for the result."
        ),
        request=JobCreateSerializer,
        responses={201: JobSerializer, 400: ValidationErrorSerializer},
    )
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save()
        # Only queue once the row is actually visible to the worker.
        transaction.on_commit(lambda: process_job.delay(str(job.id)))
        return Response(
            JobSerializer(job, context=self.get_serializer_context()).data,
            status=status.HTTP_201_CREATED,
        )
