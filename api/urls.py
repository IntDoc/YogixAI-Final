from django.urls import path
from . import views

urlpatterns = [
    path('',                          views.index,               name='index'),
    path('demo/',                     views.demo,                name='demo'),
    path('api/analyze-image/',        views.analyze_image_view,  name='analyze_image'),
    path('api/analyze-video-frame/',  views.analyze_frame_view,  name='analyze_frame'),
    path('api/generate-report/',      views.generate_report_view,name='generate_report'),
    path('api/suggest-next-pose/',    views.suggest_pose_view,   name='suggest_pose'),
]
