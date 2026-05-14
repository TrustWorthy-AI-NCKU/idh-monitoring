"""
URL configuration for the monitoring app.
"""
from django.urls import path, include
from . import views

app_name = 'monitoring'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('reset/', views.reset_view, name='reset'),
    path('retrain/', views.retrain_view, name='retrain'),
    path('retrain/download/<str:model_key>/', views.retrain_download_view, name='retrain_download'),
    path('retrain/download_data/<str:data_key>/', views.retrain_data_download_view, name='retrain_data_download'),
    path('retrain/reanalyze/', views.retrain_reanalyze_view, name='retrain_reanalyze'),
    path('retrain/best_worst/', views.retrain_best_worst_view, name='retrain_best_worst'),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]
