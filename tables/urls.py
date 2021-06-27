from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url

urlpatterns = [
    path('', views.index, name='index_table'),
    path('new', views.createProject, name='createProject'),
    path('model_x', views.table, name='table'),
    path('caller/<str:id>/', views.caller, name='caller'),
    path('import', views.import_table, name='import_table'),
    path('datasets', views.imported_table, name='imported_table'),
    path('schema', views.schema_table, name='schema_table'),
    path('analyze', views.analyze_table, name='analyze_table'),
    path('visualize/<str:id>', views.visualize_csv_table, name='visualize_table'),
    path('train', views.train_table, name='train_table'),
    path('test', views.test_table, name='test_table'),
    path('result', views.result_table, name='result_table'),
    # AJAX
    url(r'^trainingProcessStart/$', views.training_process_start_table,
        name='training_process_start'),
    url(r'^singlePredict/$', views.single_predict, name='single_predict'),
    url(r'^batch-prediction/$', views.batch_predict, name='batch_predict'),
]
