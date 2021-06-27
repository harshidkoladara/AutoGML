from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url


urlpatterns = [
    path('', views.index_obj, name="index_obj"),
    path('create', views.create_obj, name='create_obj'),
    path('caller/<str:id>/', views.caller_od, name='caller_od'),
    path('import', views.import_obj, name="import_obj"),
    path('train', views.train_obj, name='train_obj'),
    path('test', views.test_obj, name="test_obj"),

    # AJAX
    url(r'^training-ongoing/$', views.training_ongoing,
        name="training_ongoing"),
    url(r'^detect/$', views.detect, name="detect"),
]
