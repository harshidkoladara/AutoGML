from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url

urlpatterns = [
    path('', views.index_clf, name='index_clf'),
    path('new', views.createProject_clf, name='createProject_clf'),
    path('model', views.classifier_clf, name='classifier_clf'),
    path('caller/<str:id>/', views.caller_clf, name='caller_clf'),
    path('import', views.import_clf, name='import_clf'),
    path('image', views.image_clf, name='image_clf'),
    path('train', views.train_clf, name="train_clf"),
    path('test', views.test_clf, name="test_clf"),
    # AJAX
    url(r'^training-ongoing/$', views.training_ongoing, name="training_ongoing"),
    url(r'^detect/$', views.detect, name="detect"),


]
