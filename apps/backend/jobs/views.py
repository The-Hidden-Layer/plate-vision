from drf_spectacular.utils import extend_schema
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import api_view
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import Job
from .serializers import HealthSerializer, JobCreateSerializer, JobSerializer


@extend_schema(responses=HealthSerializer, description="Liveness probe for the compose healthcheck.")
@api_view(["GET"])
def health(_request):
    return Response({"status": "ok"})


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

    @extend_schema(request=JobCreateSerializer, responses={201: JobSerializer})
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save()
        # Phase 4 dispatches the Celery task here.
        return Response(
            JobSerializer(job, context=self.get_serializer_context()).data,
            status=status.HTTP_201_CREATED,
        )
