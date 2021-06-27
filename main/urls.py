from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signin', views.login, name='login'),
    path('signup', views.signup, name='signup'),
    path('pswdreset', views.forgotpwd, name='forgotpwd'),
    path('confirmation', views.confirmation, name='confirmation'),
    path('dashboard', views.allProject, name='allProject'),
    path('logout', views.logout, name='logout'),
]
