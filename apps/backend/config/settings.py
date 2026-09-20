"""Django settings for plate-vision. Everything environment-driven; see .env.example."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Only used when running outside Docker; in compose the vars come from env_file.
load_dotenv(BASE_DIR.parent.parent / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(int(default))).strip().lower() in {"1", "true", "yes", "on"}


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-insecure-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,backend").split(",")
    if h.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "drf_spectacular",
    "jobs",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "platevision"),
        "USER": os.environ.get("POSTGRES_USER", "platevision"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "platevision"),
        "HOST": os.environ.get("POSTGRES_HOST", "db"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }
}

AUTH_PASSWORD_VALIDATORS = []
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# Shared docker volume, identical mount point in backend, worker and ai-service.
MEDIA_URL = "/media/"
MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", BASE_DIR / "media"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- uploads -------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
# Anything above this streams to a temp file instead of being held in memory,
# which matters for video. Files are not covered by DATA_UPLOAD_MAX_MEMORY_SIZE,
# so the real size ceiling is enforced in JobCreateSerializer.
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024

# --- DRF -----------------------------------------------------------------
REST_FRAMEWORK = {
    # No auth in this demo. Explicitly empty so a Django admin session in the
    # same browser cannot trigger SessionAuthentication's CSRF check on uploads.
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

SPECTACULAR_SETTINGS = {
    "TITLE": "plate-vision API",
    "DESCRIPTION": "License plate detection and recognition demo.",
    "VERSION": "0.1.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
}

# --- Celery --------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# --- AI service (consumed in Phase 4) ------------------------------------
AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://ai-service:8100")
AI_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "600"))
# Base delay for retrying transient AI failures; doubles each attempt.
AI_RETRY_BACKOFF_SECONDS = int(os.environ.get("AI_RETRY_BACKOFF_SECONDS", "5"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
