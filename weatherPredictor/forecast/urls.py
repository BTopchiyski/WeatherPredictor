from django.urls import include, path
from debug_toolbar.toolbar import debug_toolbar_urls
from . import views
from .views import get_location

urlpatterns = [
    path("", views.weather_view, name="Weather View"),
    path('get_location/', get_location, name='get_location'),
] + debug_toolbar_urls()