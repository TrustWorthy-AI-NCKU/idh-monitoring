"""
Core computation logic for Model Monitoring.
Extracted from model_monitoring_UI_251217.py — no UI dependencies.
"""

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    accuracy_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import warnings

# Try to import EBM (required for some models)
try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:
    pass

warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Preprocessing
# ==========================================

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    if 'Session_Date' in df.columns:
        df['Session_Date'] = pd.to_datetime(df['Session_Date'], errors='coerce')
        df = df.dropna(subset=['Session_Date']).sort_values('Session_Date')

    if 'Nadir90/100' in df.columns:
        c1_map = {1: 1.0, 0: 0.0, True: 1.0, False: 0.0, '1': 1.0, '0': 0.0}
        if df['Nadir90/100'].dtype == 'object' or df['Nadir90/100'].dtype == 'bool':
            df['Nadir90/100'] = df['Nadir90/100'].map(c1_map)

    if 'Sex' in df.columns:
        sex_map = {1: 1, 0: 0, '男': 1, '女': 0, '1': 1, '0': 0}
        if df['Sex'].dtype == 'object':
            df['Sex'] = df['Sex'].replace(sex_map)

    return df


# ==========================================
# 2. Sliding Window
# ==========================================

def sliding_windows_exact(df, window_size_days=90, stride_days=30):
    """Slice data into sliding time windows."""
    windows = []
    window_metrics_dates = []

    if 'Session_Date' not in df.columns or df.empty:
        return [], []

    start_date = df['Session_Date'].min()
    end_date = df['Session_Date'].max()
    current_start = start_date

    max_steps = 200
    step = 0

    while current_start < end_date and step < max_steps:
        current_end = current_start + pd.Timedelta(days=window_size_days)
        mask = (df['Session_Date'] >= current_start) & (df['Session_Date'] < current_end)
        window_df = df.loc[mask]

        if len(window_df) > 5:
            windows.append(window_df)
            window_metrics_dates.append(current_end)

        current_start += pd.Timedelta(days=stride_days)
        step += 1

    return windows, window_metrics_dates


# ==========================================
# 3. Model Metrics
# ==========================================

def get_metrics(model, X, y):
    """Compute AUROC, AUPRC, F1 for a given model and data."""
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None, "樣本數不足或標籤單一"

    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X).astype(np.float32)

        try:
            roc = roc_auc_score(y, y_pred_proba)
        except Exception:
            roc = 0.5

        try:
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            auprc = auc(recall, precision)
        except Exception:
            auprc = 0.0

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, pos_label=1.0)

        return {'AUROC': roc, 'AUPRC': auprc, 'F1 score': f1}, None
    except Exception as e:
        return None, str(e)


# ==========================================
# 4. JS Divergence (Feature Drift)
# ==========================================

def js_divergence_continuous(d1, d2, bins=20):
    """Compute JS divergence for continuous features."""
    d1 = d1.dropna()
    d2 = d2.dropna()
    d1, d2 = d1[np.isfinite(d1)], d2[np.isfinite(d2)]

    if len(d1) == 0 or len(d2) == 0:
        return 0.0

    min_val = min(d1.min(), d2.min())
    max_val = max(d1.max(), d2.max())
    if min_val == max_val:
        return 0.0

    p, _ = np.histogram(d1, bins=bins, range=(min_val, max_val), density=True)
    q, _ = np.histogram(d2, bins=bins, range=(min_val, max_val), density=True)

    p = p + 1e-10
    q = q + 1e-10

    return jensenshannon(p, q, base=2)


def js_divergence_categorical(data1, data2):
    """Compute JS divergence for categorical features."""
    categories = pd.concat([data1, data2]).dropna().astype(str).unique()
    if len(categories) == 0:
        return np.nan

    p_counts = data1.dropna().astype(str).value_counts(normalize=True)
    q_counts = data2.dropna().astype(str).value_counts(normalize=True)

    p = np.array([p_counts.get(cat, 0) for cat in categories]) + 1e-10
    q = np.array([q_counts.get(cat, 0) for cat in categories]) + 1e-10

    return jensenshannon(p, q, base=2)


def compute_features_drift(df_base, df_curr, features):
    """Compute per-feature JS divergence drift scores."""
    scores = {}
    js_vals = []
    for f in features:
        if f not in df_base.columns or f not in df_curr.columns:
            scores[f] = 0.0
            continue

        is_cat = (f == 'Sex') or (df_base[f].nunique() < 10)
        val = 0.0
        if is_cat:
            val = js_divergence_categorical(df_base[f], df_curr[f])
        else:
            val = js_divergence_continuous(df_base[f], df_curr[f])

        if np.isnan(val):
            val = 0.0
        scores[f] = val
        js_vals.append(val)

    avg_js = np.mean(js_vals) if js_vals else 0.0
    return avg_js, scores


# ==========================================
# 4b. New Drift & Calibration Metrics
# ==========================================

def calculate_psi(base: pd.Series, new: pd.Series, bins: int = 10) -> float:
    """Population Stability Index — industry standard for data drift.
    PSI < 0.10: stable, 0.10~0.25: moderate shift, > 0.25: significant.
    """
    base = base.dropna()
    new = new.dropna()
    if len(base) == 0 or len(new) == 0:
        return 0.0

    base_bins = np.histogram(base, bins=bins)[1]
    base_counts = np.histogram(base, bins=base_bins)[0]
    new_counts = np.histogram(new, bins=base_bins)[0]

    base_dist = base_counts / len(base)
    new_dist = new_counts / len(new)

    base_dist = np.where(base_dist == 0, 0.0001, base_dist)
    new_dist = np.where(new_dist == 0, 0.0001, new_dist)

    return float(np.sum((new_dist - base_dist) * np.log(new_dist / base_dist)))


def calculate_average_entropy(probs: np.ndarray) -> float:
    """Average prediction entropy — measures model uncertainty.
    Higher entropy = model is less confident = may be seeing unfamiliar data.
    """
    individual_entropy = entropy(probs.T, base=2)
    return float(np.mean(individual_entropy))


def calculate_robust_z_score(value: float, history: list) -> float:
    """Robust Z-score using median + MAD (resistant to outliers).
    |Z| > 2: unusual, |Z| > 3: highly anomalous.
    """
    if len(history) < 2:
        return 0.0
    history_series = pd.Series(history)
    median = history_series.median()
    mad = (history_series - median).abs().median()
    if mad == 0:
        return 0.0
    return float((value - median) / (mad * 1.4826))


def calculate_ece(y_true, y_prob, n_bins=10) -> float:
    """Expected Calibration Error — measures how well predicted probabilities
    match actual frequencies. Lower is better.
    """
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        bin_totals = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
        non_empty_bins = bin_totals > 0
        bin_weights = bin_totals / len(y_prob)
        bin_weights = bin_weights[non_empty_bins]
        return float(np.sum(bin_weights * np.abs(prob_true - prob_pred)))
    except Exception:
        return 0.0


def calculate_brier(y_true, y_prob) -> float:
    """Brier Score — combines calibration and accuracy. 0=perfect, 0.25=random."""
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return 0.25


def compute_avg_psi(df_base, df_curr, features) -> float:
    """Average PSI across all features."""
    psi_vals = []
    for f in features:
        if f not in df_base.columns or f not in df_curr.columns:
            continue
        psi_vals.append(calculate_psi(df_base[f], df_curr[f]))
    return float(np.mean(psi_vals)) if psi_vals else 0.0


# ==========================================
# 5. Pipeline: Load, Process, Compute
# ==========================================

def load_and_process(data_path, model_path, feature_str, target_col, train_path=None):
    """
    Full pipeline: load data, model, slice windows, compute metrics + drift.
    Returns a dict with all results needed for plotting.
    
    Bug fix: consistent return structure (always returns a dict).
    """
    result = {
        'success': False,
        'message': '',
        'windows': None,
        'baseline': None,
        'dates': None,
        'model': None,
        'features': None,
        'target_col': target_col,
    }

    if not data_path or not model_path:
        result['message'] = "請上傳資料檔與模型檔。"
        return result

    try:
        # Load data
        df = pd.read_csv(data_path)
        model = joblib.load(model_path)

        # Determine features
        model_feats = getattr(model, "feature_names_in_", getattr(model, "feature_names", []))
        user_feats = [x.strip() for x in feature_str.split(',') if x.strip()]
        final_feats = list(model_feats) if len(model_feats) > 0 else [f for f in user_feats if f in df.columns]

        if not final_feats:
            result['message'] = "Error: 找不到有效特徵。"
            return result

        if target_col not in df.columns and 'Nadir90_100' in df.columns:
            target_col = 'Nadir90_100'

        # Preprocess and slice
        df_clean = preprocess_data(df)
        windows, dates = sliding_windows_exact(df_clean)

        # Baseline
        df_baseline = None
        if train_path:
            try:
                df_train = pd.read_csv(train_path)
                df_baseline = preprocess_data(df_train)
            except Exception as e:
                result['message'] = f"處理 Baseline 資料錯誤: {str(e)}"
                return result

        if not windows:
            result['message'] = "資料切分失敗，無有效窗口。"
            return result

        result.update({
            'success': True,
            'message': f"資料處理完成！共建立 {len(windows)} 個時間窗口。",
            'windows': windows,
            'baseline': df_baseline,
            'dates': dates,
            'model': model,
            'features': final_feats,
            'target_col': target_col,
        })
        return result

    except Exception as e:
        result['message'] = f"系統錯誤：{str(e)}"
        return result


# ==========================================
# 6. Build Dashboard Figure
# ==========================================

def build_dashboard_figure(windows, baseline_data, dates, model, final_feats, target_col, metrics_list):
    """
    Build the combined Plotly figure: performance line chart + feature drift heatmap.
    
    Bug fix: handles baseline_data=None safely.
    """
    if not windows:
        return go.Figure(), "請先上傳並處理資料。"

    # Determine baseline (bug fix: no unconditional .copy())
    if baseline_data is not None:
        baseline = baseline_data.copy()
    else:
        baseline = windows[0]

    # Storage
    res_perf = {'Date': [], 'AUROC': [], 'AUPRC': [], 'F1 score': [], 'JS divergence': []}
    heatmap_raw_list = []

    total = len(windows)

    for i in range(total):
        try:
            w_df = windows[i]
            mets, _ = get_metrics(model, w_df[final_feats], w_df[target_col])
            avg_js, scores = compute_features_drift(baseline, w_df, final_feats)

            if mets:
                res_perf['Date'].append(dates[i])
                res_perf['AUROC'].append(mets['AUROC'])
                res_perf['AUPRC'].append(mets['AUPRC'])
                res_perf['F1 score'].append(mets['F1 score'])
                res_perf['JS divergence'].append(avg_js)
                heatmap_raw_list.append(scores)
        except Exception:
            continue

    if not res_perf['Date']:
        return go.Figure(), "計算失敗，無有效結果。"

    # Heatmap z-data
    z_data = []
    for feat in final_feats:
        row_vals = [rec.get(feat, 0.0) for rec in heatmap_raw_list]
        z_data.append(row_vals)

    # Build combined subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.6],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    df_res = pd.DataFrame(res_perf)
    colors = {
        'AUROC': '#636EFA',
        'AUPRC': '#00CC96',
        'F1 score': '#EF553B',
        'JS divergence': '#AB63FA'
    }

    # Line traces (top)
    for metric in metrics_list:
        if metric in df_res.columns:
            is_secondary = (metric == 'JS divergence')
            dash_style = 'dot' if is_secondary else 'solid'

            fig.add_trace(
                go.Scatter(
                    x=df_res['Date'], y=df_res[metric], name=metric,
                    mode='lines+markers',
                    marker=dict(color=colors.get(metric, 'grey')),
                    line=dict(width=2, dash=dash_style)
                ),
                row=1, col=1,
                secondary_y=is_secondary
            )

    # Heatmap (bottom)
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=df_res['Date'],
            y=final_feats,
            colorscale='RdBu_r',
            zmin=0, zmax=1.0,
            colorbar=dict(title="JS Div", y=0.3, len=0.6),
            hovertemplate='<b>Feature</b>: %{y}<br><b>Date</b>: %{x}<br><b>JS</b>: %{z:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title="Model Performance & Feature Drift Analysis",
        height=800,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='#fafafa',
        plot_bgcolor='#ffffff',
    )

    fig.update_yaxes(title_text="Performance", secondary_y=False, range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Avg Drift (JS)", secondary_y=True, range=[0, 1.0], showgrid=False, row=1, col=1)
    fig.update_yaxes(title="Features", automargin=True, row=2, col=1)

    fig.update_xaxes(
        title="Time",
        rangeslider=dict(visible=True, thickness=0.05),
        type="date",
        row=2, col=1
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    return fig, "Dashboard 更新成功。"


# ==========================================
# 7. Alert Generation (Interpretability)
# ==========================================

# Thresholds for Strict mode (General Monitoring)
ALERT_THRESHOLDS_STRICT = {
    'JS divergence': {
        'warning': 0.10,      # Moderate data drift
        'critical': 0.25,     # Severe data drift
    },
    'AUPRC': {
        'warning_below': 0.45,   # AUPRC falling below this is concerning
        'critical_below': 0.30,  # AUPRC below this is critical
        'warning_drop': 0.15,    # >15% drop from peak is warning
        'critical_drop': 0.30,   # >30% drop from peak is critical
    },
    'AUROC': {
        'warning_below': 0.65,
        'critical_below': 0.55,
        'warning_drop': 0.10,
        'critical_drop': 0.20,
    },
    'F1 score': {
        'warning_below': 0.35,
        'critical_below': 0.20,
        'warning_drop': 0.20,
        'critical_drop': 0.35,
    },
    'PSI': {
        'warning': 0.10,       # Siddiqi 2005 industry standard
        'critical': 0.25,
    },
    'ECE': {
        'warning': 0.15,       # Guo et al. 2017; well-calibrated < 0.05~0.10
        'critical': 0.25,
    },
    'Brier': {
        'warning': 0.25,       # Random guess = 0.25
        'critical': 0.35,
    },
    'Entropy': {
        'z_warning': 2.0,      # 2σ deviation (robust)
        'z_critical': 3.0,     # 3σ deviation (robust)
    },
}

# Thresholds for Loose mode (Anomaly Diagnosis - to find the root cause)
ALERT_THRESHOLDS_LOOSE = {
    'JS divergence': {
        'warning': 0.20,      # Higher threshold for drift
        'critical': 0.40,
    },
    'AUPRC': {
        'warning_below': 0.30,   # Only warn if it's really bad
        'critical_below': 0.15,
        'warning_drop': 0.25,    # Only warn on huge drops
        'critical_drop': 0.45,
    },
    'AUROC': {
        'warning_below': 0.55,
        'critical_below': 0.50,  # Random guess level
        'warning_drop': 0.15,
        'critical_drop': 0.25,
    },
    'F1 score': {
        'warning_below': 0.20,
        'critical_below': 0.10,
        'warning_drop': 0.30,
        'critical_drop': 0.50,
    },
    'PSI': {
        'warning': 0.20,       # Looser PSI standard
        'critical': 0.40,
    },
    'ECE': {
        'warning': 0.25,       # High miscalibration
        'critical': 0.35,
    },
    'Brier': {
        'warning': 0.35,       # Worse than random
        'critical': 0.50,
    },
    'Entropy': {
        'z_warning': 3.0,      # 3σ deviation (rare)
        'z_critical': 5.0,     # 5σ deviation (extreme)
    },
}

DEFAULT_TREND_WINDOW = 5


def generate_alerts(windows, baseline_data, dates, model, final_feats, target_col,
                    trend_window=None, alert_mode='strict'):
    """
    Analyze recent model metrics and data drift to generate human-readable alerts.

    Args:
        trend_window: Number of recent windows to analyze for trends.
                      If None, uses DEFAULT_TREND_WINDOW.
        alert_mode: 'strict' for general monitoring, 'loose' for anomaly diagnosis.


    Returns:
        (alerts, analysis_info) tuple.
        alerts: list of dicts with 'level', 'title', 'detail', 'metric', 'value'.
    """
    if trend_window is None:
        trend_window = DEFAULT_TREND_WINDOW

    if not windows or len(windows) < 2:
        return ([{'level': 'info', 'title': '資料不足',
                 'detail': '至少需要 2 個時間窗口才能進行趨勢分析。',
                 'metric': None, 'value': None}],
                {'total_windows': 0, 'trend_windows': 0,
                 'start_date': None, 'end_date': None})

    # Determine baseline
    if baseline_data is not None:
        baseline = baseline_data.copy()
    else:
        baseline = windows[0]

    # Compute metrics for all windows (including new ones)
    all_metrics = []
    entropy_history = []  # Track entropy across windows for Z-score

    for i, w_df in enumerate(windows):
        try:
            X = w_df[final_feats]
            y = w_df[target_col]

            mets, _ = get_metrics(model, X, y)
            avg_js, _ = compute_features_drift(baseline, w_df, final_feats)
            if mets:
                # Existing metrics
                mets['JS divergence'] = avg_js
                mets['_date'] = dates[i]

                # --- New metrics (computed but not plotted) ---
                try:
                    y_prob = model.predict_proba(X)[:, 1]
                    y_prob_all = model.predict_proba(X)

                    mets['PSI'] = compute_avg_psi(baseline, w_df, final_feats)
                    mets['ECE'] = calculate_ece(y, y_prob)
                    mets['Brier'] = calculate_brier(y, y_prob)

                    ent = calculate_average_entropy(y_prob_all)
                    mets['Entropy'] = ent
                    entropy_history.append(ent)

                    # Compute entropy Z-score using history so far
                    mets['Entropy_z'] = calculate_robust_z_score(
                        ent, entropy_history[:-1]  # exclude current from history
                    )
                except Exception:
                    mets['PSI'] = 0.0
                    mets['ECE'] = 0.0
                    mets['Brier'] = 0.25
                    mets['Entropy'] = 0.0
                    mets['Entropy_z'] = 0.0

                all_metrics.append(mets)
        except Exception:
            continue

    if len(all_metrics) < 2:
        return ([{'level': 'info', 'title': '資料不足',
                 'detail': '有效的指標計算結果不足，無法分析趨勢。',
                 'metric': None, 'value': None}],
                {'total_windows': len(all_metrics), 'trend_windows': 0,
                 'start_date': None, 'end_date': None})

    alerts = []
    recent = all_metrics[-trend_window:] if len(all_metrics) >= trend_window else all_metrics
    latest = all_metrics[-1]

    # Select threshold dictionary based on mode
    thresholds = ALERT_THRESHOLDS_STRICT if alert_mode == 'strict' else ALERT_THRESHOLDS_LOOSE

    # ---------- JS Divergence (Data Drift) ----------
    js_val = latest.get('JS divergence', 0)
    js_thresh = thresholds['JS divergence']

    if js_val >= js_thresh['critical']:
        alerts.append({
            'level': 'critical',
            'title': '🚨 模型老化警告！最新資料分布嚴重偏移',
            'detail': (
                f'最新 JS Divergence = {js_val:.3f}（門檻值 {js_thresh["critical"]:.2f}），'
                f'表示目前輸入資料的分布已與 baseline 顯著不同。'
                f'\n建議立即：1) 檢查新資料的來源是否改變、2) 重新訓練模型。'
            ),
            'metric': '最新 JS divergence', 'value': js_val,
            'category': '最新資料偏移',
        })
    elif js_val >= js_thresh['warning']:
        alerts.append({
            'level': 'warning',
            'title': '⚠️ 最新資料分布偏移偵測',
            'detail': (
                f'最新 JS Divergence = {js_val:.3f}（門檻值 {js_thresh["warning"]:.2f}），'
                f'資料分布開始偏離 baseline。'
                f'\n建議：持續觀察後續幾個時間窗口的趨勢。'
            ),
            'metric': '最新 JS divergence', 'value': js_val,
            'category': '最新資料偏移',
        })

    # JS Divergence trend: consecutive increases
    if len(recent) >= 3:
        js_recent = [m.get('JS divergence', 0) for m in recent]
        consecutive_up = sum(
            1 for j in range(1, len(js_recent))
            if js_recent[j] > js_recent[j - 1]
        )
        if consecutive_up >= 3:
            alerts.append({
                'level': 'warning',
                'title': '⚠️ JS Divergence 持續上升趨勢',
                'detail': (
                    f'在最近 {len(recent)} 個窗口中，JS Divergence 有 {consecutive_up} 次連續上升'
                    f'（{js_recent[0]:.3f} → {js_recent[-1]:.3f}），表示資料偏移正在加劇。'
                ),
                'metric': 'JS divergence', 'value': js_val,
                'category': '資料偏移',
            })

    # ---------- Performance Metrics (AUPRC, AUROC, F1) ----------
    for metric in ['AUPRC', 'AUROC', 'F1 score']:
        thresh = thresholds.get(metric)
        if not thresh:
            continue

        val = latest.get(metric)
        if val is None:
            continue

        # Historical peak
        all_vals = [m.get(metric, 0) for m in all_metrics if m.get(metric) is not None]
        peak = max(all_vals) if all_vals else val

        metric_label = metric

        # Absolute threshold check
        if val < thresh.get('critical_below', 0):
            alerts.append({
                'level': 'critical',
                'title': f'🚨 {metric_label} 嚴重下降！',
                'detail': (
                    f'最新 {metric_label} = {val:.3f}，已低於警戒值 {thresh["critical_below"]:.2f}。'
                    f'\n模型在此時間窗口的表現已嚴重退化，建議立即介入檢查。'
                ),
                'metric': metric, 'value': val,
                'category': '預測表現',
            })
        elif val < thresh.get('warning_below', 0):
            alerts.append({
                'level': 'warning',
                'title': f'⚠️ {metric_label} 低於預期',
                'detail': (
                    f'最新 {metric_label} = {val:.3f}，低於預期門檻 {thresh["warning_below"]:.2f}。'
                    f'\n建議關注近期資料品質或模型是否需要更新。'
                ),
                'metric': metric, 'value': val,
                'category': '預測表現',
            })

        # Drop from peak
        if peak > 0:
            drop_pct = (peak - val) / peak
            if drop_pct >= thresh.get('critical_drop', 999):
                alerts.append({
                    'level': 'critical',
                    'title': f'🚨 {metric_label} 大幅衰退！',
                    'detail': (
                        f'{metric_label} 從歷史最高 {peak:.3f} 下降至 {val:.3f}，'
                        f'降幅 {drop_pct:.0%}，超過警戒降幅 {thresh["critical_drop"]:.0%}。'
                        f'\n建議：檢查資料品質、重新訓練模型。'
                    ),
                    'metric': metric, 'value': val,
                    'category': '預測表現',
                })
            elif drop_pct >= thresh.get('warning_drop', 999):
                alerts.append({
                    'level': 'warning',
                    'title': f'⚠️ {metric_label} 下滑趨勢',
                    'detail': (
                        f'{metric_label} 從歷史最高 {peak:.3f} 下降至 {val:.3f}，'
                        f'降幅 {drop_pct:.0%}。建議持續觀察。'
                    ),
                    'metric': metric, 'value': val,
                    'category': '預測表現',
                })

        # Consecutive decline trend
        if len(recent) >= 3:
            metric_recent = [m.get(metric, 0) for m in recent]
            consecutive_down = sum(
                1 for j in range(1, len(metric_recent))
                if metric_recent[j] < metric_recent[j - 1]
            )
            if consecutive_down >= 3:
                alerts.append({
                    'level': 'warning',
                    'title': f'⚠️ {metric_label} 持續下降中',
                    'detail': (
                        f'在最近 {len(recent)} 個窗口中，{metric_label} 有 {consecutive_down} 次連續下降'
                        f'（{metric_recent[0]:.3f} → {metric_recent[-1]:.3f}）。'
                    ),
                    'metric': metric, 'value': val,
                    'category': '預測表現',
                })

    # ---------- New Metrics: PSI ----------
    psi_val = latest.get('PSI', 0)
    psi_thresh = thresholds['PSI']
    if psi_val >= psi_thresh['critical']:
        alerts.append({
            'level': 'critical',
            'title': '🚨 PSI 嚴重偏移！',
            'detail': (
                f'最新 Average PSI = {psi_val:.4f}（門檻值 {psi_thresh["critical"]:.2f}），'
                f'資料族群已顯著改變。'
                f'\n建議：檢查資料來源是否改變，考慮重新訓練模型。'
            ),
            'metric': 'PSI', 'value': psi_val,
            'category': '資料偏移',
        })
    elif psi_val >= psi_thresh['warning']:
        alerts.append({
            'level': 'warning',
            'title': '⚠️ PSI 偵測到資料族群偏移',
            'detail': (
                f'最新 Average PSI = {psi_val:.4f}（門檻值 {psi_thresh["warning"]:.2f}），'
                f'資料分佈有中等程度的改變。'
            ),
            'metric': 'PSI', 'value': psi_val,
            'category': '資料偏移',
        })

    # ---------- New Metrics: ECE ----------
    ece_val = latest.get('ECE', 0)
    ece_thresh = thresholds['ECE']
    if ece_val >= ece_thresh['critical']:
        alerts.append({
            'level': 'critical',
            'title': '🚨 ECE 校準度嚴重偏差！',
            'detail': (
                f'最新 ECE = {ece_val:.4f}（門檻值 {ece_thresh["critical"]:.2f}），'
                f'模型預測信心與實際正確率嚴重不一致。'
                f'\n建議：模型可能需要重新校準 (Platt scaling / Isotonic regression)。'
            ),
            'metric': 'ECE', 'value': ece_val,
            'category': '校準度',
        })
    elif ece_val >= ece_thresh['warning']:
        alerts.append({
            'level': 'warning',
            'title': '⚠️ ECE 校準度偏差',
            'detail': (
                f'最新 ECE = {ece_val:.4f}（門檻值 {ece_thresh["warning"]:.2f}），'
                f'預測機率的可信度有所下降。'
            ),
            'metric': 'ECE', 'value': ece_val,
            'category': '校準度',
        })

    # ---------- New Metrics: Brier Score ----------
    brier_val = latest.get('Brier', 0)
    brier_thresh = thresholds['Brier']
    if brier_val >= brier_thresh['critical']:
        alerts.append({
            'level': 'critical',
            'title': '🚨 Brier Score 嚴重偏高！',
            'detail': (
                f'最新 Brier Score = {brier_val:.4f}（門檻值 {brier_thresh["critical"]:.2f}），'
                f'模型預測品質已低於隨機猜測水準。'
            ),
            'metric': 'Brier', 'value': brier_val,
            'category': '校準度',
        })
    elif brier_val >= brier_thresh['warning']:
        alerts.append({
            'level': 'warning',
            'title': '⚠️ Brier Score 偏高',
            'detail': (
                f'最新 Brier Score = {brier_val:.4f}（門檻值 {brier_thresh["warning"]:.2f}），'
                f'模型預測品質接近隨機猜測。'
            ),
            'metric': 'Brier', 'value': brier_val,
            'category': '校準度',
        })

    # ---------- New Metrics: Entropy Z-Score ----------
    ent_z = abs(latest.get('Entropy_z', 0))
    ent_thresh = thresholds['Entropy']
    if ent_z >= ent_thresh['z_critical']:
        alerts.append({
            'level': 'critical',
            'title': '🚨 模型不確定性異常升高！',
            'detail': (
                f'預測熵的 Robust Z-Score = {ent_z:.2f}（門檻 {ent_thresh["z_critical"]:.1f}σ），'
                f'最新 Entropy = {latest.get("Entropy", 0):.4f}。'
                f'\n模型對目前資料的信心顯著下降，可能遇到訓練時未見過的數據模式。'
            ),
            'metric': 'Entropy', 'value': latest.get('Entropy', 0),
            'category': '不確定性',
        })
    elif ent_z >= ent_thresh['z_warning']:
        alerts.append({
            'level': 'warning',
            'title': '⚠️ 模型不確定性升高',
            'detail': (
                f'預測熵的 Robust Z-Score = {ent_z:.2f}（門檻 {ent_thresh["z_warning"]:.1f}σ），'
                f'最新 Entropy = {latest.get("Entropy", 0):.4f}。'
            ),
            'metric': 'Entropy', 'value': latest.get('Entropy', 0),
            'category': '不確定性',
        })

    # ---------- Combined: Drift + Performance drop ----------
    has_drift = js_val >= js_thresh['warning'] or psi_val >= psi_thresh['warning']
    perf_drop = any(
        latest.get(m, 999) < thresholds[m].get('warning_below', 0)
        for m in ['AUPRC', 'AUROC', 'F1 score']
    )

    if has_drift and perf_drop:
        alerts.append({
            'level': 'critical',
            'title': '🔴 資料偏移 + 模型效能同時惡化！',
            'detail': (
                '同時偵測到資料分布偏移與模型效能下降，'
                '高度建議：1) 收集新的標註資料、2) 以最新資料重新訓練模型、'
                '3) 檢查是否有外部因素影響資料品質。'
            ),
            'metric': 'combined', 'value': None,
            'category': '綜合',
        })

    # If no alerts, everything looks good
    if not alerts:
        alerts.append({
            'level': 'info',
            'title': '✅ 系統狀態正常',
            'detail': (
                f'所有指標在正常範圍內。\n'
                f'AUPRC={latest.get("AUPRC", 0):.3f}、'
                f'AUROC={latest.get("AUROC", 0):.3f}、'
                f'F1={latest.get("F1 score", 0):.3f}、'
                f'JS Div={js_val:.3f}\n'
                f'PSI={psi_val:.4f}、ECE={ece_val:.4f}、'
                f'Brier={brier_val:.4f}、Entropy={latest.get("Entropy", 0):.4f}'
            ),
            'metric': None, 'value': None,
            'category': '總覽',
        })

    # Sort: critical first, then warning, then info
    level_order = {'critical': 0, 'warning': 1, 'info': 2}
    alerts.sort(key=lambda a: level_order.get(a['level'], 99))

    # Build analysis info for display
    trend_start = recent[0].get('_date')
    trend_end = recent[-1].get('_date')
    analysis_info = {
        'total_windows': len(all_metrics),
        'trend_windows': len(recent),
        'start_date': trend_start.strftime('%Y-%m-%d') if hasattr(trend_start, 'strftime') else str(trend_start)[:10],
        'end_date': trend_end.strftime('%Y-%m-%d') if hasattr(trend_end, 'strftime') else str(trend_end)[:10],
    }

    return alerts, analysis_info
