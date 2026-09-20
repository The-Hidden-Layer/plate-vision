from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)
from rest_framework.routers import DefaultRouter

from jobs.views import JobViewSet, health

# Slash-free to match Next.js. Next strips trailing slashes when proxying
# (/api/jobs/ reaches Django as /api/jobs), so DRF's default trailing slash
# would bounce off APPEND_SLASH in a redirect loop. Aligning the API with the
# proxy's convention removes the mismatch instead of patching around it.
router = DefaultRouter(trailing_slash=False)
router.register("jobs", JobViewSet, basename="job")

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/health", health, name="health"),
    path("api/schema", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs", SpectacularSwaggerView.as_view(url_name="schema"), name="docs"),
    path("api/redoc", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
    path("api/", include(router.urls)),
]

if settings.DEBUG:
    # Dev-only: serve the shared media volume. Behind the Next.js rewrites
    # proxy this is reachable from the browser at /media/...
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
