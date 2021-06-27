from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url


urlpatterns = [
    path('', views.index_fr, name='index_fr'),
    path('create', views.create_face, name='create_face'),
    path('caller/<str:id>/', views.caller_fr, name='caller_fr'),
    path('import', views.import_face, name="import_face"),
    path('image', views.image_face, name="image_face"),
    path('train', views.train_face, name='train_face'),
    path('test', views.test_face, name="test_face"),
    # AJAX
    url(r'^training-ongoing/$', views.training_ongoing, name="training_ongoing"),

    # VIDEO STREAM
    # access the laptop camera
    path('video_feed', views.video_feed, name='video_feed'),
    # access the phone camera
    path('webcam_feed', views.webcam_feed, name='webcam_feed'),
]
