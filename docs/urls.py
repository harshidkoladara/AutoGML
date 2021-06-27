from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url

urlpatterns = [
    path('fr', views.index_doc_fr, name='index_fr_doc'),
    path('img', views.index_doc_img, name='index_img_doc'),
    path('od', views.index_doc_od, name='index_od_doc'),
    path('table', views.index_doc_table, name='index_doc_table'),
]
