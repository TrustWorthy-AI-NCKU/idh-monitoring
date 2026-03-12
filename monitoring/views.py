"""
Views for the monitoring app.
"""
import os
import json
import tempfile
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.cache import cache
import plotly.utils

from .forms import MonitoringForm
from .services import load_and_process, build_dashboard_figure, generate_alerts


CACHE_KEY = 'monitoring_data_default'
CACHE_TIMEOUT = 3600  # 1 hour


def dashboard_view(request):
    """Main dashboard view - handles file upload and renders the dashboard."""
    form = MonitoringForm()
    has_data = cache.get(CACHE_KEY) is not None

    if request.method == 'POST':
        form = MonitoringForm(request.POST, request.FILES)
        if form.is_valid():
            # Save uploaded files to temp directory
            data_file = request.FILES['data_file']
            model_file = request.FILES['model_file']
            train_file = request.FILES.get('train_file')

            temp_dir = tempfile.mkdtemp()

            try:
                # Save data file
                data_path = os.path.join(temp_dir, data_file.name)
                with open(data_path, 'wb') as f:
                    for chunk in data_file.chunks():
                        f.write(chunk)

                # Save model file
                model_path = os.path.join(temp_dir, model_file.name)
                with open(model_path, 'wb') as f:
                    for chunk in model_file.chunks():
                        f.write(chunk)

                # Save train file (optional)
                train_path = None
                if train_file:
                    train_path = os.path.join(temp_dir, train_file.name)
                    with open(train_path, 'wb') as f:
                        for chunk in train_file.chunks():
                            f.write(chunk)

                # Run processing
                result = load_and_process(
                    data_path=data_path,
                    model_path=model_path,
                    feature_str=form.cleaned_data['features'],
                    target_col=form.cleaned_data['target_col'],
                    train_path=train_path,
                )

                if result['success']:
                    # Store results in cache for later use
                    cache.set(CACHE_KEY, {
                        'windows': result['windows'],
                        'baseline': result['baseline'],
                        'dates': result['dates'],
                        'model': result['model'],
                        'features': result['features'],
                        'target_col': result['target_col'],
                        'trend_window': form.cleaned_data.get('trend_window', 5),
                        'alert_mode': form.cleaned_data.get('alert_mode', 'strict'),
                    }, CACHE_TIMEOUT)

                    # Save filenames to session for display
                    request.session['uploaded_files'] = {
                        'data_file': data_file.name,
                        'model_file': model_file.name,
                        'train_file': train_file.name if train_file else None,
                        'features': form.cleaned_data['features'],
                        'target_col': form.cleaned_data['target_col'],
                        'trend_window': form.cleaned_data.get('trend_window', 5),
                        'alert_mode': form.cleaned_data.get('alert_mode', 'strict'),
                    }

                    messages.success(request, result['message'])
                    has_data = True
                else:
                    messages.error(request, result['message'])

            except Exception as e:
                messages.error(request, f'處理錯誤: {str(e)}')

            finally:
                # Clean up temp files
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

            return redirect('monitoring:dashboard')

    # Pre-fill form with previously used values from session
    uploaded_files = request.session.get('uploaded_files', {})
    if request.method == 'GET' and uploaded_files:
        form = MonitoringForm(initial={
            'features': uploaded_files.get('features', ''),
            'target_col': uploaded_files.get('target_col', 'Nadir90/100'),
            'trend_window': uploaded_files.get('trend_window', 5),
            'alert_mode': uploaded_files.get('alert_mode', 'strict'),
        })

    # Build figure JSON and alerts if data is available
    figure_json = None
    figure_error = None
    alerts = []
    analysis_info = {}
    if has_data:
        cached_data = cache.get(CACHE_KEY)
        if cached_data:
            try:
                fig, msg = build_dashboard_figure(
                    windows=cached_data['windows'],
                    baseline_data=cached_data['baseline'],
                    dates=cached_data['dates'],
                    model=cached_data['model'],
                    final_feats=cached_data['features'],
                    target_col=cached_data['target_col'],
                    metrics_list=['AUPRC', 'AUROC', 'F1 score', 'JS divergence'],
                )
                figure_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            except Exception as e:
                figure_error = str(e)

            # Generate alerts
            try:
                alerts, analysis_info = generate_alerts(
                    windows=cached_data['windows'],
                    baseline_data=cached_data['baseline'],
                    dates=cached_data['dates'],
                    model=cached_data['model'],
                    final_feats=cached_data['features'],
                    target_col=cached_data['target_col'],
                    trend_window=cached_data.get('trend_window', 5),
                    alert_mode=cached_data.get('alert_mode', 'strict'),
                )
            except Exception as e:
                alerts = [{'level': 'warning', 'title': '警告分析失敗',
                           'detail': str(e), 'metric': None, 'value': None}]
                analysis_info = {}

    context = {
        'form': form,
        'has_data': has_data,
        'uploaded_files': uploaded_files,
        'figure_json': figure_json,
        'figure_error': figure_error,
        'alerts': alerts,
        'analysis_info': analysis_info,
    }
    return render(request, 'monitoring/dashboard.html', context)
