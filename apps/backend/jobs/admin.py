from django.contrib import admin

from .models import Detection, Job


class DetectionInline(admin.TabularInline):
    model = Detection
    extra = 0


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("id", "media_type", "status", "source_filename", "created_at")
    list_filter = ("status", "media_type")
    inlines = [DetectionInline]
