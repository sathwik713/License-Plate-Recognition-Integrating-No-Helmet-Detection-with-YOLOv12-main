from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_video, name='upload'),
    path('live/', views.live_camera, name='live'),
    path('report/', views.report, name='report'),
    path('training/', views.training_results, name='training'),
    path('process_video/', views.process_video, name='process_video'),
    path('generate_report/', views.generate_report, name='generate_report'),
    path('stream/<str:filename>/', views.stream_uploaded_video, name='stream_video'),
]