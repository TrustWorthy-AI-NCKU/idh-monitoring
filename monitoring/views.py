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
from .services import (
    load_and_process, build_dashboard_figure, build_comparison_figure,
    generate_alerts, generate_monthly_report,
    compute_model_windows_metrics, compute_comparison_summary, compute_longevity_metrics,
    sliding_windows_exact, compute_adaptive_thresholds,
    compute_single_model_longevity, build_combined_llm_prompt,
)
from .llm_service import generate_llm_summary
from .moe_service import MoEModel, create_moe_model

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


def reset_view(request):
    """Clear all cached data and session, redirect to fresh dashboard."""
    cache.delete(CACHE_KEY)
    if 'uploaded_files' in request.session:
        del request.session['uploaded_files']
    messages.success(request, '已清除所有資料與分析結果。')
    return redirect('monitoring:dashboard')


def dashboard_view(request):
    """Main dashboard view - handles file upload and renders the dashboard."""
    form = MonitoringForm()
    has_data = cache.get(CACHE_KEY) is not None

    if request.method == 'POST':
        # Clear stale cache before processing new upload
        cache.delete(CACHE_KEY)
        has_data = False

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

        # Advanced window settings
        try:
            window_size_days = int(request.POST.get('window_size_days', '90'))
            window_size_days = max(7, window_size_days)  # minimum 7 days
        except (ValueError, TypeError):
            window_size_days = 90
        try:
            stride_days = int(request.POST.get('stride_days', '30'))
            stride_days = max(7, stride_days)  # minimum 7 days (one week)
        except (ValueError, TypeError):
            stride_days = 30
        try:
            auprc_threshold = float(request.POST.get('auprc_threshold', '0.40'))
        except (ValueError, TypeError):
            auprc_threshold = 0.40

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
                    moe_subtype = request.POST.get('single_moe_subtype', 'optuna')

                    if moe_subtype == 'custom':
                        # Custom: single pre-packaged .joblib
                        custom_file = request.FILES.get('single_custom_model')
                        if not custom_file:
                            messages.error(request, 'Custom MoE：請上傳預打包的模型 (.joblib)。')
                            return redirect('monitoring:dashboard')
                        custom_path = _save_uploaded_file(custom_file, temp_dir)
                        try:
                            wrapped_model = create_moe_model({}, custom_path, moe_subtype='custom')
                        except Exception as e:
                            messages.error(request, f'Custom MoE 載入失敗: {str(e)}')
                            return redirect('monitoring:dashboard')
                    else:
                        # Optuna or Standard: need router/gating + experts
                        meta_file = request.FILES.get('single_meta_learner')
                        expert_files = request.FILES.getlist('single_expert_files')
                        expert_files = [ef for ef in expert_files if ef.size > 0]

                        if not meta_file:
                            label = 'Gating Network' if moe_subtype == 'optuna' else 'Router'
                            messages.error(request, f'MoE 模式：請上傳 {label} (.joblib)。')
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
                            wrapped_model = create_moe_model(expert_paths, meta_path, moe_subtype=moe_subtype)
                        except Exception as e:
                            messages.error(request, f'MoE 模型載入失敗: {str(e)}')
                            return redirect('monitoring:dashboard')

                    result = load_and_process(
                        data_path=data_path,
                        model_path=None,
                        feature_str=features,
                        target_col=target_col,
                        train_path=train_path,
                        preloaded_model=wrapped_model,
                        window_size_days=window_size_days,
                        stride_days=stride_days,
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
                        window_size_days=window_size_days,
                        stride_days=stride_days,
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
                        'df_clean': result.get('df_clean'),
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
                        'auprc_threshold': auprc_threshold,
                        'window_size_days': window_size_days,
                        'stride_days': stride_days,
                    }
                    messages.success(request, result['message'])
                else:
                    logger.error(f"[dashboard] load_and_process failed: {result['message']}")
                    messages.error(request, result['message'])

            # ======================================================
            # COMPARISON MODE
            # ======================================================
            else:
                m1_type = request.POST.get('m1_type', 'ebm')

                # --- Model 1 ---
                if m1_type == 'moe':
                    m1_moe_subtype = request.POST.get('m1_moe_subtype', 'optuna')
                    if m1_moe_subtype == 'custom':
                        m1_custom = request.FILES.get('m1_custom_model')
                        if not m1_custom:
                            messages.error(request, 'M1 Custom MoE：請上傳預打包模型。')
                            return redirect('monitoring:dashboard')
                        m1_custom_path = _save_uploaded_file(m1_custom, temp_dir)
                        try:
                            m1_model = create_moe_model({}, m1_custom_path, moe_subtype='custom')
                        except Exception as e:
                            messages.error(request, f'M1 Custom MoE 載入失敗: {str(e)}')
                            return redirect('monitoring:dashboard')
                    else:
                        m1_meta = request.FILES.get('m1_meta_learner')
                        m1_experts = [ef for ef in request.FILES.getlist('m1_expert_files') if ef.size > 0]
                        if not m1_meta or len(m1_experts) < 2:
                            messages.error(request, 'M1 MoE：請上傳 Router/Gating 和至少 2 個 Expert。')
                            return redirect('monitoring:dashboard')
                        m1_meta_path = _save_uploaded_file(m1_meta, temp_dir)
                        m1_expert_paths = {os.path.splitext(ef.name)[0]: _save_uploaded_file(ef, temp_dir) for ef in m1_experts}
                        try:
                            m1_model = create_moe_model(m1_expert_paths, m1_meta_path, moe_subtype=m1_moe_subtype)
                        except Exception as e:
                            messages.error(request, f'M1 MoE 載入失敗: {str(e)}')
                            return redirect('monitoring:dashboard')
                    result = load_and_process(
                        data_path=data_path, model_path=None,
                        feature_str=features, target_col=target_col,
                        train_path=train_path, preloaded_model=m1_model,
                        window_size_days=window_size_days, stride_days=stride_days,
                    )
                else:
                    # M1 EBM
                    model_file = request.FILES.get('model_file')
                    if not model_file:
                        messages.error(request, '比較模式：請上傳 Model 1 (.joblib) 檔案。')
                        return redirect('monitoring:dashboard')
                    model_path = _save_uploaded_file(model_file, temp_dir)
                    result = load_and_process(
                        data_path=data_path, model_path=model_path,
                        feature_str=features, target_col=target_col,
                        train_path=train_path,
                        window_size_days=window_size_days, stride_days=stride_days,
                    )

                if not result['success']:
                    logger.error(f"[dashboard] compare M1 failed: {result['message']}")
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
                    'df_clean': result.get('df_clean'),
                }

                # --- Model 2 (comparison) ---
                m2_type = request.POST.get('m2_type', 'ebm')
                if m2_type == 'moe':
                    m2_moe_subtype = request.POST.get('m2_moe_subtype', 'optuna')
                    if m2_moe_subtype == 'custom':
                        m2_custom = request.FILES.get('m2_custom_model')
                        if m2_custom:
                            m2_custom_path = _save_uploaded_file(m2_custom, temp_dir)
                            try:
                                m2_model = create_moe_model({}, m2_custom_path, moe_subtype='custom')
                                cache_data['comparison_model_wrapped'] = m2_model
                                cache_data['comparison_type'] = 'moe'
                                cache_data['comparison_info'] = {
                                    'type': 'MoE (Custom)',
                                    'model_file': m2_custom.name,
                                }
                            except Exception as e:
                                messages.warning(request, f'M2 Custom MoE 載入失敗: {str(e)}')
                        else:
                            messages.warning(request, 'M2 Custom MoE：請上傳預打包模型。')
                    else:
                        meta_file = request.FILES.get('m2_meta_learner')
                        expert_files = [ef for ef in request.FILES.getlist('m2_expert_files') if ef.size > 0]
                        if meta_file and expert_files:
                            meta_path = _save_uploaded_file(meta_file, temp_dir)
                            expert_paths = {}
                            expert_names = []
                            for ef in expert_files:
                                name = os.path.splitext(ef.name)[0]
                                expert_paths[name] = _save_uploaded_file(ef, temp_dir)
                                expert_names.append(ef.name)
                            try:
                                m2_wrapped = create_moe_model(expert_paths, meta_path, moe_subtype=m2_moe_subtype)
                                cache_data['comparison_model_wrapped'] = m2_wrapped
                                cache_data['comparison_type'] = 'moe'
                                subtype_label = 'Optuna' if m2_moe_subtype == 'optuna' else 'Standard'
                                cache_data['comparison_info'] = {
                                    'type': f'MoE ({subtype_label})',
                                    'meta_learner': meta_file.name,
                                    'experts': expert_names,
                                }
                            except Exception as e:
                                messages.warning(request, f'M2 MoE 載入失敗: {str(e)}')
                        else:
                            messages.warning(request, 'M2 MoE：請上傳 Router/Gating 和至少一個 Expert。')
                else:
                    # M2 EBM
                    cmp_model_file = request.FILES.get('m2_ebm_model')
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
                            messages.warning(request, f'M2 EBM 載入失敗: {str(e)}')
                    else:
                        messages.warning(request, '請上傳 Model 2 EBM (.joblib)。')

                cache.set(CACHE_KEY, cache_data, CACHE_TIMEOUT)
                m1_name = (request.FILES.get('model_file') or request.FILES.get('m1_meta_learner') or data_file).name
                request.session['uploaded_files'] = {
                    'data_file': data_file.name,
                    'model_file': m1_name,
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
    figure_json_a = None   # Config A: 90-day windows
    figure_json_b = None   # Config B: 30-day windows
    figure_error = None
    alerts = []
    analysis_info = {}
    monthly_report_a = None
    monthly_report_b = None
    longevity_a = None
    longevity_b = None
    llm_summary = None
    moe_metrics = None
    comparison_info = None
    comparison_summary = None
    longevity_metrics = None
    comparison_figure_json = None
    comparison_warning_json = None
    monthly_report_m2 = None

    if has_data:
        cached_data = cache.get(CACHE_KEY)
        if cached_data:
            model = cached_data['model']
            feats = cached_data['features']
            target = cached_data['target_col']
            baseline = cached_data['baseline']
            tw = cached_data.get('trend_window', 5)
            model_ver = uploaded_files.get('model_file', '未知')
            has_baseline = baseline is not None
            df_clean = cached_data.get('df_clean')

            # =============================================
            # Config A: 90-day window / 30-day stride
            # =============================================
            windows_a = cached_data['windows']
            dates_a = cached_data['dates']

            # Config A chart
            try:
                fig_a, _ = build_dashboard_figure(
                    windows=windows_a, baseline_data=baseline,
                    dates=dates_a, model=model,
                    final_feats=feats, target_col=target,
                    metrics_list=['AUPRC', 'AUROC', 'F1 score', 'JS divergence'],
                    has_real_baseline=has_baseline,
                )
                figure_json_a = json.dumps(fig_a, cls=plotly.utils.PlotlyJSONEncoder)
            except Exception as e:
                figure_error = str(e)

            # Config A report
            try:
                monthly_report_a = generate_monthly_report(
                    windows=windows_a, baseline_data=baseline,
                    dates=dates_a, model=model,
                    final_feats=feats, target_col=target,
                    model_version=model_ver, trend_window=tw,
                )
            except Exception as e:
                logger.error(f"[dashboard] Config A report failed: {e}")

            # =============================================
            # Config B: 30-day window / 15-day stride
            # =============================================
            windows_b = []
            dates_b = []
            if df_clean is not None:
                try:
                    windows_b, dates_b = sliding_windows_exact(
                        df_clean, window_size_days=30, stride_days=15
                    )
                except Exception as e:
                    logger.error(f"[dashboard] Config B sliding_windows failed: {e}")

            if windows_b:
                # Config B chart
                try:
                    fig_b, _ = build_dashboard_figure(
                        windows=windows_b, baseline_data=baseline,
                        dates=dates_b, model=model,
                        final_feats=feats, target_col=target,
                        metrics_list=['AUPRC', 'AUROC', 'F1 score', 'JS divergence'],
                        has_real_baseline=has_baseline,
                    )
                    figure_json_b = json.dumps(fig_b, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    logger.error(f"[dashboard] Config B chart failed: {e}")

                # Config B report
                try:
                    monthly_report_b = generate_monthly_report(
                        windows=windows_b, baseline_data=baseline,
                        dates=dates_b, model=model,
                        final_feats=feats, target_col=target,
                        model_version=model_ver, trend_window=tw,
                    )
                except Exception as e:
                    logger.error(f"[dashboard] Config B report failed: {e}")

            # =============================================
            # Adaptive Thresholds & Longevity
            # =============================================
            adaptive_thresh = None
            if monthly_report_a and not monthly_report_a.get('basic_info', {}).get('error'):
                try:
                    adaptive_thresh = compute_adaptive_thresholds(monthly_report_a)
                except Exception as e:
                    logger.error(f"[dashboard] adaptive thresholds failed: {e}")

            # Longevity A (90-day)
            if windows_a and len(windows_a) >= 2:
                try:
                    longevity_a = compute_single_model_longevity(
                        windows=windows_a, baseline_data=baseline,
                        dates=dates_a, model=model,
                        final_feats=feats, target_col=target,
                        adaptive_thresholds=adaptive_thresh,
                    )
                except Exception as e:
                    logger.error(f"[dashboard] longevity A failed: {e}")

            # Longevity B (30-day)
            if windows_b and len(windows_b) >= 2:
                try:
                    longevity_b = compute_single_model_longevity(
                        windows=windows_b, baseline_data=baseline,
                        dates=dates_b, model=model,
                        final_feats=feats, target_col=target,
                        adaptive_thresholds=adaptive_thresh,
                    )
                except Exception as e:
                    logger.error(f"[dashboard] longevity B failed: {e}")

            # =============================================
            # Alerts (use Config A windows)
            # =============================================
            try:
                alerts, analysis_info = generate_alerts(
                    windows=windows_a, baseline_data=baseline,
                    dates=dates_a, model=model,
                    final_feats=feats, target_col=target,
                    trend_window=tw,
                    alert_mode=cached_data.get('alert_mode', 'strict'),
                )
            except Exception as e:
                alerts = [{'level': 'warning', 'title': '警告分析失敗',
                           'detail': str(e), 'metric': None, 'value': None}]
                analysis_info = {}

            # =============================================
            # LLM Summary (single call, combined prompt)
            # =============================================
            try:
                if monthly_report_a and monthly_report_b:
                    combined_prompt = build_combined_llm_prompt(
                        monthly_report_a, monthly_report_b,
                        label_a='90天窗格（季度觀察）',
                        label_b='30天窗格（月度觀察）',
                    )
                    llm_summary = generate_llm_summary(
                        monthly_report_a, override_prompt=combined_prompt
                    )
                elif monthly_report_a:
                    llm_summary = generate_llm_summary(monthly_report_a)
                else:
                    llm_summary = None
            except Exception as e:
                llm_summary = f"報告生成失敗（{str(e)}）"

            # =============================================
            # Comparison Model (unchanged logic)
            # =============================================
            comparison_info = cached_data.get('comparison_info')
            m2_model = cached_data.get('comparison_model_wrapped') or cached_data.get('comparison_model')
            if m2_model:
                try:
                    m1_name = uploaded_files.get('model_file', 'Model 1')
                    m2_name = comparison_info.get('type', 'Model 2') if comparison_info else 'Model 2'
                    auprc_threshold = float(uploaded_files.get('auprc_threshold', 0.40))
                    comp_figs, _ = build_comparison_figure(
                        windows=windows_a, dates=dates_a,
                        model_1=model, model_2=m2_model,
                        final_feats=feats, target_col=target,
                        baseline=baseline,
                        model_1_name=m1_name, model_2_name=m2_name,
                        auprc_threshold=auprc_threshold,
                    )
                    comparison_figure_json = json.dumps(comp_figs['main'], cls=plotly.utils.PlotlyJSONEncoder)
                    comparison_warning_json = json.dumps(comp_figs['warning'], cls=plotly.utils.PlotlyJSONEncoder)

                    df_m1 = compute_model_windows_metrics(
                        windows_a, dates_a, model, feats, target, baseline
                    )
                    df_m2 = compute_model_windows_metrics(
                        windows_a, dates_a, m2_model, feats, target, baseline
                    )
                    comparison_summary = compute_comparison_summary(df_m1, df_m2, m1_name, m2_name)
                    longevity_metrics = compute_longevity_metrics(df_m1, df_m2, m1_name, m2_name)

                    try:
                        monthly_report_m2 = generate_monthly_report(
                            windows=windows_a, baseline_data=baseline,
                            dates=dates_a, model=m2_model,
                            final_feats=feats, target_col=target,
                            model_version=m2_name, trend_window=tw,
                        )
                    except Exception as e:
                        logger.error(f"[dashboard] M2 monthly report failed: {e}")

                except Exception as e:
                    logger.error(f"[dashboard] comparison figure failed: {e}")
                    moe_metrics = {'error': str(e)}

    context = {
        'form': form,
        'has_data': has_data,
        'uploaded_files': uploaded_files,
        # Dual window config
        'figure_json_a': figure_json_a,
        'figure_json_b': figure_json_b,
        'figure_json': figure_json_a,     # backward compat
        'figure_error': figure_error,
        'alerts': alerts,
        'analysis_info': analysis_info,
        'monthly_report': monthly_report_a,
        'monthly_report_a': monthly_report_a,
        'monthly_report_b': monthly_report_b,
        'longevity_a': longevity_a,
        'longevity_b': longevity_b,
        # LLM (shared)
        'llm_summary': llm_summary,
        # Comparison (unchanged)
        'monthly_report_m2': monthly_report_m2,
        'moe_metrics': moe_metrics,
        'comparison_info': comparison_info,
        'comparison_figure_json': comparison_figure_json,
        'comparison_warning_json': comparison_warning_json,
        'comparison_summary': comparison_summary,
        'longevity_metrics': longevity_metrics,
    }
    return render(request, 'monitoring/dashboard.html', context)
