"""Native contract tests; deployment and Docker tests continue to use PostgreSQL."""

from .settings import *  # noqa: F403

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
