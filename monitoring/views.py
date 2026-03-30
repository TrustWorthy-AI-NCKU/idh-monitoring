"""
Views for the monitoring app.
"""
import os
import json
import logging
import tempfile
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.cache import cache
import plotly.utils

from .forms import MonitoringForm
from .services import load_and_process, build_dashboard_figure, generate_alerts, generate_monthly_report
from .llm_service import generate_llm_summary
from .moe_service import MoEModel

logger = logging.getLogger(__name__)


CACHE_KEY = 'monitoring_data_default'
CACHE_TIMEOUT = 3600  # 1 hour


def _save_uploaded_file(uploaded_file, temp_dir):
    """Save an uploaded file to temp_dir and return the path."""
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return path


def dashboard_view(request):
    """Main dashboard view - handles file upload and renders the dashboard."""
    form = MonitoringForm()
    has_data = cache.get(CACHE_KEY) is not None

    if request.method == 'POST':
        # Determine which tab and model type was submitted
        enable_comparison = request.POST.get('enable_comparison') == 'on'
        single_model_type = request.POST.get('single_model_type', 'ebm')  # 'ebm' or 'moe'

        data_file = request.FILES.get('data_file')
        train_file = request.FILES.get('train_file')

        # Get features and target from POST
        features = request.POST.get('features', '').strip()
        target_col = request.POST.get('target_col', 'Nadir90/100').strip()
        trend_window_raw = request.POST.get('trend_window', '5')
        try:
            trend_window = int(trend_window_raw)
        except (ValueError, TypeError):
            trend_window = 5

        if not data_file:
            messages.error(request, '請上傳 Dataset (.csv) 檔案。')
            return redirect('monitoring:dashboard')

        if not features:
            messages.error(request, '請填入 Features 欄位。')
            return redirect('monitoring:dashboard')

        temp_dir = tempfile.mkdtemp()

        try:
            data_path = _save_uploaded_file(data_file, temp_dir)
            train_path = _save_uploaded_file(train_file, temp_dir) if train_file else None

            # ======================================================
            # SINGLE MODEL MODE
            # ======================================================
            if not enable_comparison:

                if single_model_type == 'moe':
                    # --- Single MoE ---
                    meta_file = request.FILES.get('single_meta_learner')
                    expert_files = request.FILES.getlist('single_expert_files')
                    expert_files = [ef for ef in expert_files if ef.size > 0]

                    if not meta_file:
                        messages.error(request, 'MoE 模式：請上傳 Gating Network (.joblib)。')
                        return redirect('monitoring:dashboard')
                    if len(expert_files) < 2:
                        messages.error(request, 'MoE 模式：請上傳至少 2 個 Expert 模型。')
                        return redirect('monitoring:dashboard')

                    meta_path = _save_uploaded_file(meta_file, temp_dir)
                    expert_paths = {}
                    for ef in expert_files:
                        name = os.path.splitext(ef.name)[0]
                        expert_paths[name] = _save_uploaded_file(ef, temp_dir)

                    try:
                        moe = MoEModel(expert_paths, meta_path)
                    except Exception as e:
                        messages.error(request, f'MoE 模型載入失敗: {str(e)}')
                        return redirect('monitoring:dashboard')

                    # Use MoE as the "model"; wrap it so services.py can call predict/predict_proba
                    from .moe_service import MoESklearnWrapper
                    wrapped_model = MoESklearnWrapper(moe)

                    result = load_and_process(
                        data_path=data_path,
                        model_path=None,            # no single joblib file
                        feature_str=features,
                        target_col=target_col,
                        train_path=train_path,
                        preloaded_model=wrapped_model,
                    )

                else:
                    # --- Single EBM (default) ---
                    model_file = request.FILES.get('model_file')
                    if not model_file:
                        messages.error(request, '請上傳 Model (.joblib) 檔案。')
                        return redirect('monitoring:dashboard')

                    model_path = _save_uploaded_file(model_file, temp_dir)
                    result = load_and_process(
                        data_path=data_path,
                        model_path=model_path,
                        feature_str=features,
                        target_col=target_col,
                        train_path=train_path,
                    )

                if result['success']:
                    cache_data = {
                        'windows': result['windows'],
                        'baseline': result['baseline'],
                        'dates': result['dates'],
                        'model': result['model'],
                        'features': result['features'],
                        'target_col': result['target_col'],
                        'trend_window': trend_window,
                        'alert_mode': request.POST.get('alert_mode', 'strict'),
                        'model_type': single_model_type,
                    }
                    cache.set(CACHE_KEY, cache_data, CACHE_TIMEOUT)

                    request.session['uploaded_files'] = {
                        'data_file': data_file.name,
                        'model_file': (request.FILES.get('model_file') or
                                       request.FILES.get('single_meta_learner') or data_file).name,
                        'train_file': train_file.name if train_file else None,
                        'features': features,
                        'target_col': target_col,
                        'trend_window': trend_window,
                        'model_type': single_model_type,
                    }
                    messages.success(request, result['message'])
                else:
                    logger.error(f"[dashboard] load_and_process failed: {result['message']}")
                    messages.error(request, result['message'])

            # ======================================================
            # COMPARISON MODE
            # ======================================================
            else:
                # Model 1 (required: model_file)
                model_file = request.FILES.get('model_file')
                if not model_file:
                    messages.error(request, '比較模式：請上傳 Model 1 (.joblib) 檔案。')
                    return redirect('monitoring:dashboard')

                model_path = _save_uploaded_file(model_file, temp_dir)
                result = load_and_process(
                    data_path=data_path,
                    model_path=model_path,
                    feature_str=features,
                    target_col=target_col,
                    train_path=train_path,
                )

                if not result['success']:
                    messages.error(request, result['message'])
                    return redirect('monitoring:dashboard')

                cache_data = {
                    'windows': result['windows'],
                    'baseline': result['baseline'],
                    'dates': result['dates'],
                    'model': result['model'],
                    'features': result['features'],
                    'target_col': result['target_col'],
                    'trend_window': trend_window,
                    'alert_mode': request.POST.get('alert_mode', 'strict'),
                }

                # Model 2 comparison
                cmp_type = request.POST.get('comparison_model_type', 'ebm')
                if cmp_type == 'moe':
                    meta_file = request.FILES.get('moe_meta_learner')
                    expert_files = request.FILES.getlist('moe_expert_files')
                    expert_files = [ef for ef in expert_files if ef.size > 0]
                    if meta_file and expert_files:
                        meta_path = _save_uploaded_file(meta_file, temp_dir)
                        expert_paths = {}
                        expert_names = []
                        for ef in expert_files:
                            epath = _save_uploaded_file(ef, temp_dir)
                            name = os.path.splitext(ef.name)[0]
                            expert_paths[name] = epath
                            expert_names.append(ef.name)
                        try:
                            moe = MoEModel(expert_paths, meta_path)
                            cache_data['moe_model'] = moe
                            cache_data['comparison_type'] = 'moe'
                            cache_data['comparison_info'] = {
                                'type': 'MoE (Optuna)',
                                'meta_learner': meta_file.name,
                                'experts': expert_names,
                            }
                        except Exception as e:
                            messages.warning(request, f'MoE 模型載入失敗: {str(e)}')
                    else:
                        messages.warning(request, '請上傳 Meta-Learner 和至少一個 Expert。')
                elif cmp_type == 'ebm':
                    cmp_model_file = request.FILES.get('comparison_ebm_model')
                    if cmp_model_file:
                        import joblib
                        cmp_path = _save_uploaded_file(cmp_model_file, temp_dir)
                        try:
                            cmp_model = joblib.load(cmp_path)
                            cache_data['comparison_model'] = cmp_model
                            cache_data['comparison_type'] = 'ebm'
                            cache_data['comparison_info'] = {
                                'type': 'EBM',
                                'model_file': cmp_model_file.name,
                            }
                        except Exception as e:
                            messages.warning(request, f'比較 EBM 模型載入失敗: {str(e)}')
                    else:
                        messages.warning(request, '請上傳比較用的 EBM 模型。')

                cache.set(CACHE_KEY, cache_data, CACHE_TIMEOUT)
                request.session['uploaded_files'] = {
                    'data_file': data_file.name,
                    'model_file': model_file.name,
                    'train_file': train_file.name if train_file else None,
                    'features': features,
                    'target_col': target_col,
                    'trend_window': trend_window,
                }
                if cache_data.get('comparison_info'):
                    request.session['uploaded_files']['comparison'] = cache_data['comparison_info']
                messages.success(request, result['message'])

        except Exception as e:
            logger.exception(f"[dashboard] Unhandled error during POST: {e}")
            messages.error(request, f'處理錯誤: {str(e)}')

        finally:
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
    monthly_report = None
    llm_summary = None
    moe_metrics = None
    comparison_info = None

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

            # Generate monthly report
            try:
                monthly_report = generate_monthly_report(
                    windows=cached_data['windows'],
                    baseline_data=cached_data['baseline'],
                    dates=cached_data['dates'],
                    model=cached_data['model'],
                    final_feats=cached_data['features'],
                    target_col=cached_data['target_col'],
                    model_version=uploaded_files.get('model_file', '未知'),
                    trend_window=cached_data.get('trend_window', 5),
                )
                llm_summary = generate_llm_summary(monthly_report)
            except Exception as e:
                llm_summary = f"報告生成失敗（{str(e)}）"

            # --- MoE comparison metrics ---
            comparison_info = cached_data.get('comparison_info')
            if cached_data.get('comparison_type') == 'moe':
                moe_model = cached_data.get('moe_model')
                if moe_model:
                    try:
                        import pandas as pd
                        import numpy as np
                        # Aggregate all window data for MoE evaluation
                        all_dfs = []
                        for w_df in cached_data['windows']:
                            all_dfs.append(w_df)
                        combined = pd.concat(all_dfs, ignore_index=True)

                        feats_with_date = ['Session_Date'] + cached_data['features']
                        available = [f for f in feats_with_date if f in combined.columns]
                        X_moe = combined[available].copy()
                        y_moe = combined[cached_data['target_col']].copy()

                        moe_metrics, _, _ = moe_model.evaluate(X_moe, y_moe)
                    except Exception as e:
                        moe_metrics = {'error': str(e)}

    context = {
        'form': form,
        'has_data': has_data,
        'uploaded_files': uploaded_files,
        'figure_json': figure_json,
        'figure_error': figure_error,
        'alerts': alerts,
        'analysis_info': analysis_info,
        'monthly_report': monthly_report,
        'llm_summary': llm_summary,
        'moe_metrics': moe_metrics,
        'comparison_info': comparison_info,
    }
    return render(request, 'monitoring/dashboard.html', context)
