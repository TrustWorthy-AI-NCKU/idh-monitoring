"""
Views for the monitoring app.
"""
import os
import json
import logging
import tempfile
import shutil
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.cache import cache
from django.http import HttpResponse, Http404
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

    # =============================================
    # Overview Indicators (traffic-light summary)
    # =============================================
    overview_indicators = []
    if monthly_report_a and not monthly_report_a.get('basic_info', {}).get('error'):
        # 1. AUPRC — from safety_gates
        for g in monthly_report_a.get('safety_gates', []):
            if g['metric'] == 'AUPRC':
                auprc_val = g['value']
                if auprc_val >= 0.50:
                    ov_status = 'green'
                elif auprc_val >= 0.35:
                    ov_status = 'yellow'
                else:
                    ov_status = 'red'
                overview_indicators.append({
                    'key': 'auprc', 'label': '表現',
                    'metric': 'AUPRC', 'value': auprc_val,
                    'status': ov_status,
                    'desc': f'模型整體預測效能。越高越好（門檻 {g["threshold"]}）。',
                })
                break

        # 2. Sensitivity — from safety_gates
        for g in monthly_report_a.get('safety_gates', []):
            if g['metric'] == 'Sensitivity':
                sens_val = g['value']
                if sens_val >= 0.75:
                    ov_status = 'green'
                elif sens_val >= 0.60:
                    ov_status = 'yellow'
                else:
                    ov_status = 'red'
                overview_indicators.append({
                    'key': 'sensitivity', 'label': '正確召回率',
                    'metric': 'Sensitivity', 'value': sens_val,
                    'status': ov_status,
                    'desc': f'偵測正例（IDH 發生）的能力（門檻 {g["threshold"]}）。',
                })
                break

        # 3. JS Divergence — from additional
        js_val = monthly_report_a.get('additional', {}).get('JS_divergence')
        if js_val is not None:
            if js_val < 0.10:
                ov_status = 'green'
            elif js_val < 0.25:
                ov_status = 'yellow'
            else:
                ov_status = 'red'
            overview_indicators.append({
                'key': 'js_div', 'label': '老化-資料飄移',
                'metric': 'JS Divergence', 'value': js_val,
                'status': ov_status,
                'desc': '輸入資料與訓練基準的分布差異。越低越穩定。',
            })

        # 4. ECE — from additional
        ece_val = monthly_report_a.get('additional', {}).get('ECE')
        if ece_val is not None:
            if ece_val < 0.10:
                ov_status = 'green'
            elif ece_val < 0.20:
                ov_status = 'yellow'
            else:
                ov_status = 'red'
            overview_indicators.append({
                'key': 'ece', 'label': '老化-預測信心',
                'metric': 'ECE', 'value': ece_val,
                'status': ov_status,
                'desc': '模型預測機率與實際結果的校準誤差。越低越可靠。',
            })

        # 5. Health Score — from longevity_a
        if longevity_a and longevity_a.get('health_score') is not None:
            hs = longevity_a['health_score']
            if hs >= 70:
                ov_status = 'green'
            elif hs >= 40:
                ov_status = 'yellow'
            else:
                ov_status = 'red'
            overview_indicators.append({
                'key': 'health', 'label': '模型健康分數',
                'metric': '健康分數', 'value': hs,
                'status': ov_status,
                'desc': '綜合達標率、連續達標比和趨勢穩定度的整體評分（0~100）。',
            })

    context = {
        'form': form,
        'has_data': has_data,
        'uploaded_files': uploaded_files,
        # Overview indicators
        'overview_indicators': overview_indicators,
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


RETRAIN_CACHE_KEY = 'retrain_pipeline_result'
RETRAIN_CACHE_TIMEOUT = 7200  # 2 hours


def retrain_view(request):
    """Model retrain pipeline view."""
    has_result = cache.get(RETRAIN_CACHE_KEY) is not None

    if request.method == 'POST':
        cache.delete(RETRAIN_CACHE_KEY)
        has_result = False

        data_file = request.FILES.get('data_file')
        features_str = request.POST.get('features', '').strip()
        target_col = request.POST.get('target_col', 'Nadir90/100').strip()

        new_data_start_date = request.POST.get('new_data_start_date', '').strip()
        if not new_data_start_date:
            new_data_start_date = None

        ver1_model_file = request.FILES.get('ver1_model')

        if not data_file:
            messages.error(request, '請上傳 Dataset (.csv) 檔案。')
            return redirect('monitoring:retrain')

        if not features_str:
            messages.error(request, '請填入 Features 欄位。')
            return redirect('monitoring:retrain')

        features = [f.strip() for f in features_str.split(',') if f.strip()]

        temp_dir = tempfile.mkdtemp()

        try:
            data_path = os.path.join(temp_dir, data_file.name)
            with open(data_path, 'wb') as f:
                for chunk in data_file.chunks():
                    f.write(chunk)

            from .retrain_service import (
                run_retrain_pipeline,
                build_drift_chart,
                build_shape_comparison_figures,
                build_scan_chart,
            )

            ver1_model = None
            model_name = 'Unknown'
            if ver1_model_file:
                import joblib
                model_name = ver1_model_file.name
                ver1_path = os.path.join(temp_dir, ver1_model_file.name)
                with open(ver1_path, 'wb') as f:
                    for chunk in ver1_model_file.chunks():
                        f.write(chunk)
                try:
                    ver1_model = joblib.load(ver1_path)
                except Exception as e:
                    messages.error(request, f'無法載入模型檔案: {e}')
                    return redirect('monitoring:retrain')
            else:
                model_name = '重新訓練 (From Scratch)'

            data_name = data_file.name
            drift_threshold_mode = request.POST.get('drift_threshold_mode', 'fixed')
            scan_mode = request.POST.get('scan_mode') == 'on'

            result = run_retrain_pipeline(
                csv_path=data_path,
                features=features,
                target_col=target_col,
                ver1_model=ver1_model,
                ver1_end_date=new_data_start_date,
                model_name=model_name,
                data_name=data_name,
                drift_threshold_mode=drift_threshold_mode,
                scan_mode=scan_mode,
            )

            if result['success']:
                charts = {}

                # Scan chart (AUPRC timeline with T0/T1 highlights)
                if result.get('scan_results') and result.get('segments'):
                    scan_fig = build_scan_chart(result['scan_results'], result['segments'])
                    if scan_fig:
                        charts['scan'] = json.dumps(scan_fig, cls=plotly.utils.PlotlyJSONEncoder)

                # Drift chart
                drift_fig = build_drift_chart(
                    result.get('data_drift', {}).get('per_feature', {}),
                    adaptive_thresholds=result.get('drift_thresholds'),
                    threshold_mode=result.get('drift_threshold_mode', 'fixed'),
                )
                if drift_fig:
                    charts['drift'] = json.dumps(drift_fig, cls=plotly.utils.PlotlyJSONEncoder)

                # Shape function charts
                shape_figs = build_shape_comparison_figures(
                    result.get('shape_functions', []),
                )
                shape_jsons = []
                for sf in shape_figs:
                    shape_jsons.append({
                        'name': sf['name'],
                        'json': json.dumps(sf['figure'], cls=plotly.utils.PlotlyJSONEncoder),
                    })
                charts['shapes'] = shape_jsons

                # Save models for download (using joblib in temp)
                import joblib
                model_dir = os.path.join(temp_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)

                if result.get('_ver1'):
                    ver1_path = os.path.join(model_dir, 'ver1.joblib')
                    joblib.dump(result['_ver1'], ver1_path)
                if result.get('_ver2'):
                    ver2_path = os.path.join(model_dir, 'ver2.joblib')
                    joblib.dump(result['_ver2'], ver2_path)

                # Remove internal model objects before caching (not serializable)
                cache_result = {k: v for k, v in result.items() if not k.startswith('_')}
                cache_result['charts'] = charts
                cache_result['model_dir'] = model_dir
                cache_result['_temp_dir'] = temp_dir  # Keep temp dir alive

                cache.set(RETRAIN_CACHE_KEY, cache_result, RETRAIN_CACHE_TIMEOUT)
                messages.success(request, f'Pipeline 完成！耗時 {result["elapsed_s"]}s')
            else:
                messages.error(request, f'Pipeline 失敗：{result.get("error", "未知錯誤")}')
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            logger.exception(f"[retrain] Unhandled error: {e}")
            messages.error(request, f'系統錯誤：{str(e)}')
            shutil.rmtree(temp_dir, ignore_errors=True)

        return redirect('monitoring:retrain')

    # GET: Render page
    cached = cache.get(RETRAIN_CACHE_KEY)

    context = {
        'has_result': cached is not None,
        'result': cached,
    }
    return render(request, 'monitoring/retrain.html', context)


def retrain_download_view(request, model_key):
    """Download a trained model file."""
    cached = cache.get(RETRAIN_CACHE_KEY)
    if not cached:
        raise Http404('No retrain results found.')

    model_dir = cached.get('model_dir')
    if not model_dir:
        raise Http404('Model directory not found.')

    valid_keys = {'ver1': 'ver1.joblib', 'ver2': 'ver2.joblib', 'ver1_edited': 'ver1_edited.joblib'}
    if model_key not in valid_keys:
        raise Http404('Invalid model key.')

    filepath = os.path.join(model_dir, valid_keys[model_key])
    if not os.path.exists(filepath):
        raise Http404('Model file not found.')

    with open(filepath, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename="{valid_keys[model_key]}"'
        return response


def retrain_data_download_view(request, data_key):
    """Download a saved data segment (T0 or T1)."""
    cached = cache.get(RETRAIN_CACHE_KEY)
    if not cached:
        raise Http404('No retrain results found.')

    saved_segments = cached.get('saved_segments')
    if not saved_segments:
        raise Http404('No saved segments found.')

    valid_keys = {'t0': 't0_path', 't1': 't1_path'}
    if data_key not in valid_keys:
        raise Http404('Invalid data key.')

    filepath = saved_segments.get(valid_keys[data_key])
    if not filepath or not os.path.exists(filepath):
        raise Http404('Data file not found.')

    with open(filepath, 'rb') as f:
        response = HttpResponse(f.read(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{os.path.basename(filepath)}"'
        return response
