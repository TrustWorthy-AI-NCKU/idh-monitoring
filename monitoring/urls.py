"""
URL configuration for the monitoring app.
"""
from django.urls import path, include
from . import views

app_name = 'monitoring'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('reset/', views.reset_view, name='reset'),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]
