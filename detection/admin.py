from django.contrib import admin
from .models import Detection

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('date', 'time', 'helmet_status', 'license_plate_text', 'video_source')
    list_filter = ('helmet_status', 'date')
    search_fields = ('license_plate_text', 'video_source')