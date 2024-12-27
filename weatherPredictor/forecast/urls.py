from django.urls import include, path
from debug_toolbar.toolbar import debug_toolbar_urls
from . import views

urlpatterns = [
    path("", views.weather_view, name="Weather View"),
] + debug_toolbar_urls()