"""
Core computation logic for Model Monitoring.
Extracted from model_monitoring_UI_251217.py — no UI dependencies.
"""
import logging
logger = logging.getLogger(__name__)

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

    # Dynamic min samples: smaller windows need lower threshold
    min_samples = max(3, window_size_days // 15)

    start_date = df['Session_Date'].min()
    end_date = df['Session_Date'].max()
    current_start = start_date

    max_steps = 500
    step = 0

    while current_start < end_date and step < max_steps:
        current_end = current_start + pd.Timedelta(days=window_size_days)
        mask = (df['Session_Date'] >= current_start) & (df['Session_Date'] < current_end)
        window_df = df.loc[mask]

        if len(window_df) > min_samples:
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
    # Handle NaN in target
    if hasattr(y, 'isna'):
        valid_mask = ~y.isna()
        y = y[valid_mask]
        if hasattr(X, 'loc'):
            X = X.loc[valid_mask]
        else:
            X = X[valid_mask.values]

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
        logger.error(f"[get_metrics] predict failed: {type(e).__name__}: {e}")
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

def load_and_process(data_path, model_path, feature_str, target_col,
                     train_path=None, preloaded_model=None,
                     window_size_days=90, stride_days=30):
    """
    Full pipeline: load data, model, slice windows, compute metrics + drift.
    Returns a dict with all results needed for plotting.

    Args:
        preloaded_model: If provided, skip joblib.load(model_path) and use this directly.
                         Useful for MoE models assembled from multiple files.
        window_size_days: Size of each time window in days (default: 90)
        stride_days: Step size in days between windows (default: 30)
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

    if not data_path:
        result['message'] = "請上傳資料檔。"
        return result

    try:
        # Load data
        df = pd.read_csv(data_path)

        # Load or use preloaded model
        if preloaded_model is not None:
            model = preloaded_model
        elif model_path:
            model = joblib.load(model_path)
        else:
            result['message'] = "未提供模型檔案。"
            return result

        # Determine features
        model_feats = getattr(model, "feature_names_in_", getattr(model, "feature_names", []))
        user_feats = [x.strip() for x in feature_str.split(',') if x.strip()]
        final_feats = list(model_feats) if len(model_feats) > 0 else [f for f in user_feats if f in df.columns]

        # If model has no stored feature names, use user-provided features
        if not final_feats:
            final_feats = [f for f in user_feats if f in df.columns]

        if not final_feats:
            result['message'] = "Error: 找不到有效特徵，請確認 Features 欄位與資料欄位名稱一致。"
            return result

        # Resolve target column name (handle Nadir90/100 vs Nadir90_100 mismatch)
        target_aliases = [target_col, 'Nadir90_100', 'Nadir90/100', 'Nadir90/100.y']
        resolved_target = None
        for alias in target_aliases:
            if alias in df.columns:
                resolved_target = alias
                break
        if resolved_target is None:
            result['message'] = f"Error: 找不到目標欄位 '{target_col}'，資料中可用的欄位: {[c for c in df.columns if 'nadir' in c.lower() or 'target' in c.lower()]}"
            return result
        # Standardize: rename to user-specified target_col so all downstream code uses one name
        if resolved_target != target_col:
            df = df.rename(columns={resolved_target: target_col})

        # Preprocess and slice
        df_clean = preprocess_data(df)
        windows, dates = sliding_windows_exact(df_clean, window_size_days, stride_days)

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
            'df_clean': df_clean,  # For re-slicing with different window configs
        })
        return result

    except Exception as e:
        result['message'] = f"系統錯誤：{str(e)}"
        return result



# ==========================================
# 6. Build Dashboard Figure
# ==========================================

def build_dashboard_figure(windows, baseline_data, dates, model, final_feats,
                           target_col, metrics_list=None, has_real_baseline=True):
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

    # Skip drift if no real baseline (e.g. MoE without training data)
    compute_drift = has_real_baseline

    # Storage
    res_perf = {'Date': [], 'AUROC': [], 'AUPRC': [], 'F1 score': []}
    if compute_drift:
        res_perf['JS divergence'] = []
    heatmap_raw_list = []

    total = len(windows)

    for i in range(total):
        try:
            w_df = windows[i]
            X_infer = w_df.drop(columns=[target_col], errors='ignore') if getattr(model, 'is_moe', False) else w_df[final_feats]
            mets, _ = get_metrics(model, X_infer, w_df[target_col])

            if mets:
                res_perf['Date'].append(dates[i])
                res_perf['AUROC'].append(mets['AUROC'])
                res_perf['AUPRC'].append(mets['AUPRC'])
                res_perf['F1 score'].append(mets['F1 score'])

                if compute_drift:
                    avg_js, scores = compute_features_drift(baseline, w_df, final_feats)
                    res_perf['JS divergence'].append(avg_js)
                    heatmap_raw_list.append(scores)
        except Exception as e:
            logger.error(f"[build_dashboard_figure] window {i} failed: {type(e).__name__}: {e}")
            continue

    if not res_perf['Date']:
        return go.Figure(), "計算失敗，無有效結果。"

    # Heatmap z-data
    z_data = []
    for feat in final_feats:
        row_vals = [rec.get(feat, 0.0) for rec in heatmap_raw_list]
        z_data.append(row_vals)

    # Build subplot layout
    if compute_drift:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.4, 0.6],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
    else:
        # No drift: single row, no heatmap
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": False}]]
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

    # Heatmap (bottom) — only when drift data exists
    if compute_drift and heatmap_raw_list:
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
    chart_title = "Model Performance Analysis" if not compute_drift else "Model Performance & Feature Drift Analysis"
    chart_height = 450 if not compute_drift else 800
    fig.update_layout(
        title=chart_title,
        height=chart_height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='#fafafa',
        plot_bgcolor='#ffffff',
    )

    fig.update_yaxes(title_text="Performance", secondary_y=False, range=[0, 1.05], row=1, col=1)
    if compute_drift:
        fig.update_yaxes(title_text="Avg Drift (JS)", secondary_y=True, range=[0, 1.0], showgrid=False, row=1, col=1)
        fig.update_yaxes(title="Features", automargin=True, row=2, col=1)
        fig.update_xaxes(
            title="Time",
            rangeslider=dict(visible=True, thickness=0.05),
            type="date",
            row=2, col=1
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
    else:
        fig.update_xaxes(
            title="Time",
            rangeslider=dict(visible=True, thickness=0.05),
            type="date",
            row=1, col=1
        )

    return fig, "Dashboard 更新成功。"


# ==========================================
# 6b. Helper: Compute per-window metrics for a model
# ==========================================

def compute_model_windows_metrics(windows, dates, model, final_feats, target_col, baseline=None):
    """
    Compute AUROC, AUPRC, F1, JS divergence, ECE, Entropy, PSI for each time window.
    Returns a DataFrame with all metrics per window.
    """
    if baseline is None and windows:
        baseline = windows[0]

    res = {
        'Date': [], 'AUROC': [], 'AUPRC': [], 'F1 score': [],
        'JS divergence': [], 'ECE': [], 'Entropy': [], 'PSI': [],
    }

    for i, w_df in enumerate(windows):
        try:
            X_infer = w_df.drop(columns=[target_col], errors='ignore') if getattr(model, 'is_moe', False) else w_df[final_feats]
            y = w_df[target_col]

            # Drop NaN targets
            valid_mask = ~y.isna()
            y = y[valid_mask]
            X_infer = X_infer.loc[valid_mask] if hasattr(X_infer, 'loc') else X_infer[valid_mask.values]

            mets, _ = get_metrics(model, X_infer, y)
            avg_js, _ = compute_features_drift(baseline, w_df, final_feats)

            if mets:
                # Core metrics
                res['Date'].append(dates[i])
                res['AUROC'].append(mets['AUROC'])
                res['AUPRC'].append(mets['AUPRC'])
                res['F1 score'].append(mets['F1 score'])
                res['JS divergence'].append(avg_js)

                # Early warning metrics
                try:
                    y_pred_proba = model.predict_proba(X_infer)
                    res['ECE'].append(calculate_ece(y, y_pred_proba[:, 1]))
                    res['Entropy'].append(calculate_average_entropy(y_pred_proba))
                except Exception:
                    res['ECE'].append(0.0)
                    res['Entropy'].append(0.0)

                # PSI (data stability)
                try:
                    res['PSI'].append(compute_avg_psi(baseline, w_df, final_feats) if baseline is not None else 0.0)
                except Exception:
                    res['PSI'].append(0.0)
        except Exception as e:
            logger.error(f"[compute_model_windows_metrics] window {i} failed: {type(e).__name__}: {e}")
            continue

    return pd.DataFrame(res) if res['Date'] else pd.DataFrame()


# ==========================================
# 6c. Comparison Summary & Longevity Metrics
# ==========================================

def compute_comparison_summary(df_m1, df_m2, model_1_name='Model 1', model_2_name='Model 2'):
    """
    Compute summary statistics for two models' performance comparison.
    Focused on model longevity: which model maintains better performance over time.

    Returns:
        dict with 'metrics_table' (list of dicts per metric),
                   'overall_winner', 'longevity' (dict per model)
    """
    if df_m1.empty or df_m2.empty:
        return None

    metrics = ['AUROC', 'AUPRC', 'F1 score']
    table = []

    m1_wins = 0
    m2_wins = 0

    for metric in metrics:
        if metric not in df_m1.columns or metric not in df_m2.columns:
            continue

        v1 = df_m1[metric].dropna()
        v2 = df_m2[metric].dropna()

        if len(v1) == 0 or len(v2) == 0:
            continue

        avg1 = v1.mean()
        avg2 = v2.mean()
        std1 = v1.std()
        std2 = v2.std()
        min1 = v1.min()
        min2 = v2.min()

        diff = avg1 - avg2
        winner = model_1_name if diff > 0 else model_2_name if diff < 0 else '—'
        if diff > 0:
            m1_wins += 1
        elif diff < 0:
            m2_wins += 1

        table.append({
            'metric': metric,
            'm1_avg': round(avg1, 4),
            'm2_avg': round(avg2, 4),
            'm1_std': round(std1, 4),
            'm2_std': round(std2, 4),
            'm1_min': round(min1, 4),
            'm2_min': round(min2, 4),
            'diff': round(diff, 4),
            'winner': winner,
        })

    overall_winner = model_1_name if m1_wins > m2_wins else (
        model_2_name if m2_wins > m1_wins else '持平'
    )

    return {
        'metrics_table': table,
        'overall_winner': overall_winner,
        'm1_name': model_1_name,
        'm2_name': model_2_name,
        'm1_wins': m1_wins,
        'm2_wins': m2_wins,
    }


def compute_longevity_metrics(df_m1, df_m2, model_1_name='Model 1', model_2_name='Model 2'):
    """
    Compute model longevity comparison metrics.
    - consecutive_above: max consecutive windows where AUPRC >= threshold
    - trend_slope: linear regression slope of AUPRC over time (negative = degrading)
    - first_degradation: first window index where AUPRC drops below threshold

    Returns:
        dict with per-model longevity stats
    """
    AUPRC_THRESHOLD = 0.40  # Minimum acceptable AUPRC

    def _compute_for_model(df, name):
        if df.empty or 'AUPRC' not in df.columns:
            return {'name': name, 'n_windows': 0, 'max_consecutive': 0,
                    'trend_slope': 0.0, 'first_degradation': None,
                    'windows_above': 0, 'pct_above': 0.0}

        auprc = df['AUPRC'].values
        n = len(auprc)

        # Max consecutive windows above threshold
        max_consec = 0
        current_consec = 0
        for v in auprc:
            if v >= AUPRC_THRESHOLD:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        # Count total windows above threshold
        windows_above = int(np.sum(auprc >= AUPRC_THRESHOLD))
        pct_above = windows_above / n * 100 if n > 0 else 0.0

        # Linear regression slope
        if n >= 3:
            x = np.arange(n)
            slope = np.polyfit(x, auprc, 1)[0]
        else:
            slope = 0.0

        # First degradation window
        first_deg = None
        for i, v in enumerate(auprc):
            if v < AUPRC_THRESHOLD:
                first_deg = i + 1  # 1-indexed
                break

        return {
            'name': name,
            'n_windows': n,
            'max_consecutive': max_consec,
            'trend_slope': round(slope, 6),
            'first_degradation': first_deg,
            'windows_above': windows_above,
            'pct_above': round(pct_above, 1),
        }

    m1_stats = _compute_for_model(df_m1, model_1_name)
    m2_stats = _compute_for_model(df_m2, model_2_name)

    # Determine longevity winner
    if m1_stats['max_consecutive'] > m2_stats['max_consecutive']:
        longevity_winner = model_1_name
    elif m2_stats['max_consecutive'] > m1_stats['max_consecutive']:
        longevity_winner = model_2_name
    else:
        # Tiebreaker: better trend slope (less negative = more stable)
        longevity_winner = model_1_name if m1_stats['trend_slope'] > m2_stats['trend_slope'] else model_2_name

    return {
        'm1': m1_stats,
        'm2': m2_stats,
        'longevity_winner': longevity_winner,
        'auprc_threshold': AUPRC_THRESHOLD,
    }


# ==========================================
# 6d. Comparison Figure: Dual-Model Overlay + Δ Performance
# ==========================================

def build_comparison_figure(windows, dates, model_1, model_2, final_feats, target_col,
                            baseline=None, model_1_name='Model 1', model_2_name='Model 2',
                            auprc_threshold=0.40):
    """
    Build comparison Plotly figures with white background.
    Color scheme: M1 = orange-red family, M2 = blue-green family.
    Each trace is individually toggleable (no legendgroup).
    """
    if not windows:
        return {'main': go.Figure(), 'warning': go.Figure()}, "無可用的時間窗口資料。"

    df_m1 = compute_model_windows_metrics(windows, dates, model_1, final_feats, target_col, baseline)
    df_m2 = compute_model_windows_metrics(windows, dates, model_2, final_feats, target_col, baseline)

    if df_m1.empty and df_m2.empty:
        return {'main': go.Figure(), 'warning': go.Figure()}, "兩個模型都無法計算有效指標。"

    # =============================================
    # Color Families — same model = same color family
    # M1 (e.g. EBM) = Orange-Red | M2 (e.g. MoE) = Blue-Green
    # =============================================
    M1_COLORS = {
        'AUPRC': '#E65100',    # deep orange
        'AUROC': '#FF8F00',    # amber
        'F1 score': '#D84315', # red-orange
    }
    M2_COLORS = {
        'AUPRC': '#00897B',    # teal
        'AUROC': '#0277BD',    # blue
        'F1 score': '#00695C', # dark teal
    }

    # =============================================
    # CHART 1: Main overlay (Survival + Delta)
    # =============================================
    fig_main = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
    )

    # --- M1 traces (solid lines) ---
    for metric in ['AUPRC', 'AUROC', 'F1 score']:
        if not df_m1.empty and metric in df_m1.columns:
            fig_main.add_trace(
                go.Scatter(
                    x=df_m1['Date'], y=df_m1[metric],
                    name=f'{model_1_name} {metric}',
                    mode='lines+markers',
                    marker=dict(color=M1_COLORS[metric], size=6),
                    line=dict(width=2.5, dash='solid'),
                    visible=True if metric == 'AUPRC' else 'legendonly',
                ),
                secondary_y=False,
            )

    # --- M2 traces (dashed lines, diamond markers) ---
    for metric in ['AUPRC', 'AUROC', 'F1 score']:
        if not df_m2.empty and metric in df_m2.columns:
            fig_main.add_trace(
                go.Scatter(
                    x=df_m2['Date'], y=df_m2[metric],
                    name=f'{model_2_name} {metric}',
                    mode='lines+markers',
                    marker=dict(color=M2_COLORS[metric], size=6, symbol='diamond'),
                    line=dict(width=2.5, dash='dash'),
                    visible=True if metric == 'AUPRC' else 'legendonly',
                ),
                secondary_y=False,
            )

    # Threshold line
    all_dates = df_m1['Date'].tolist() if not df_m1.empty else (df_m2['Date'].tolist() if not df_m2.empty else [])
    if all_dates:
        fig_main.add_trace(
            go.Scatter(
                x=all_dates,
                y=[auprc_threshold] * len(all_dates),
                name=f'AUPRC 閥值 ({auprc_threshold})',
                mode='lines',
                line=dict(width=2, dash='dot', color='#B71C1C'),
            ),
            secondary_y=False,
        )

    # --- Delta Performance bars ---
    if not df_m1.empty and not df_m2.empty:
        merged = pd.merge(
            df_m1[['Date', 'AUPRC']],
            df_m2[['Date', 'AUPRC']],
            on='Date', suffixes=('_m1', '_m2'),
            how='inner'
        )
        if not merged.empty:
            delta = merged['AUPRC_m1'] - merged['AUPRC_m2']
            bar_colors = ['#2ECC71' if d >= 0 else '#E74C3C' for d in delta]
            fig_main.add_trace(
                go.Bar(
                    x=merged['Date'], y=delta,
                    name=f'Δ AUPRC ({model_1_name} − {model_2_name})',
                    marker_color=bar_colors,
                    opacity=0.45,
                ),
                secondary_y=True,
            )

    # --- JS Divergence (if baseline exists) ---
    if not df_m1.empty and 'JS divergence' in df_m1.columns:
        fig_main.add_trace(
            go.Scatter(
                x=df_m1['Date'], y=df_m1['JS divergence'],
                name='JS Divergence',
                mode='lines',
                line=dict(width=1.5, dash='dot', color='#7B1FA2'),
                visible='legendonly',
            ),
            secondary_y=True,
        )

    # Layout — white background
    fig_main.update_layout(
        title=dict(
            text=f"⚔️ {model_1_name} vs {model_2_name} — 壽命軌跡 × Δ Performance",
            font=dict(size=15, color='#333'),
        ),
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color='#333'),
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        bargap=0.3,
    )
    fig_main.update_yaxes(
        title_text="Performance",
        secondary_y=False, range=[0, 1.05],
        gridcolor='rgba(0,0,0,0.08)',
        zerolinecolor='rgba(0,0,0,0.1)',
        tickfont=dict(color='#555'),
    )
    fig_main.update_yaxes(
        title_text="Δ AUPRC / Drift",
        secondary_y=True,
        showgrid=False,
        zeroline=True, zerolinecolor='rgba(0,0,0,0.15)',
        tickfont=dict(color='#555'),
    )
    fig_main.update_xaxes(
        title="Time",
        type="date",
        rangeslider=dict(visible=True, thickness=0.05),
        gridcolor='rgba(0,0,0,0.08)',
        tickfont=dict(color='#555'),
    )

    # =============================================
    # CHART 2: Early Warning (ECE + Entropy)
    # Same color families: M1=orange-red, M2=blue-green
    # Both ECE and Entropy are "lower is better"
    # =============================================
    fig_warning = go.Figure()

    # M1 colors (orange-red family)
    m1_entropy_color = '#EF6C00'  # orange
    m1_ece_color = '#BF360C'      # dark red-orange

    # M2 colors (blue-green family)
    m2_entropy_color = '#00838F'  # teal-cyan
    m2_ece_color = '#01579B'      # dark blue

    # Entropy traces
    if not df_m1.empty and 'Entropy' in df_m1.columns:
        fig_warning.add_trace(
            go.Scatter(
                x=df_m1['Date'], y=df_m1['Entropy'],
                name=f'{model_1_name} Entropy',
                mode='lines+markers',
                marker=dict(color=m1_entropy_color, size=7),
                line=dict(width=2.5, dash='solid'),
            ),
        )
    if not df_m2.empty and 'Entropy' in df_m2.columns:
        fig_warning.add_trace(
            go.Scatter(
                x=df_m2['Date'], y=df_m2['Entropy'],
                name=f'{model_2_name} Entropy',
                mode='lines+markers',
                marker=dict(color=m2_entropy_color, size=7, symbol='diamond'),
                line=dict(width=2.5, dash='dash'),
            ),
        )

    # ECE traces
    if not df_m1.empty and 'ECE' in df_m1.columns:
        fig_warning.add_trace(
            go.Scatter(
                x=df_m1['Date'], y=df_m1['ECE'],
                name=f'{model_1_name} ECE',
                mode='lines+markers',
                marker=dict(color=m1_ece_color, size=5),
                line=dict(width=1.5, dash='dot'),
            ),
        )
    if not df_m2.empty and 'ECE' in df_m2.columns:
        fig_warning.add_trace(
            go.Scatter(
                x=df_m2['Date'], y=df_m2['ECE'],
                name=f'{model_2_name} ECE',
                mode='lines+markers',
                marker=dict(color=m2_ece_color, size=5, symbol='diamond'),
                line=dict(width=1.5, dash='dot'),
            ),
        )

    fig_warning.update_layout(
        title=dict(
            text="🚨 模型信心預警 — Entropy + ECE（兩者皆越低越好 ↓）",
            font=dict(size=15, color='#333'),
        ),
        height=400,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color='#333'),
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        yaxis=dict(
            title="Entropy / ECE（↓ 越低越好）",
            gridcolor='rgba(0,0,0,0.08)',
            tickfont=dict(color='#555'),
        ),
        xaxis=dict(
            title="Time",
            type="date",
            rangeslider=dict(visible=True, thickness=0.05),
            gridcolor='rgba(0,0,0,0.08)',
            tickfont=dict(color='#555'),
        ),
        # Annotation explaining interpretation
        annotations=[
            dict(
                text="ECE = 校準誤差（預測信心 vs 實際準確度的差距）<br>Entropy = 預測不確定性（模型對結果的猶豫程度）<br>兩者上升 = 模型正在退化的早期訊號",
                xref="paper", yref="paper",
                x=0.01, y=-0.22,
                showarrow=False,
                font=dict(size=10, color='#888'),
                align="left",
            )
        ],
    )

    return {'main': fig_main, 'warning': fig_warning}, "比較圖表產生成功。"



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
            X_infer = w_df.drop(columns=[target_col], errors='ignore') if getattr(model, 'is_moe', False) else w_df[final_feats]
            y = w_df[target_col]

            # Drop NaN targets
            valid_mask = ~y.isna()
            if valid_mask.sum() < 2:
                continue
            y_clean = y[valid_mask]
            X_clean = X_infer.loc[valid_mask] if hasattr(X_infer, 'loc') else X_infer[valid_mask.values]

            mets, _ = get_metrics(model, X_clean, y_clean)
            avg_js, _ = compute_features_drift(baseline, w_df, final_feats)
            if mets:
                # Existing metrics
                mets['JS divergence'] = avg_js
                mets['_date'] = dates[i]

                # --- New metrics (computed but not plotted) ---
                try:
                    y_prob = model.predict_proba(X_clean)[:, 1]
                    y_prob_all = model.predict_proba(X_clean)

                    mets['PSI'] = compute_avg_psi(baseline, w_df, final_feats)
                    mets['ECE'] = calculate_ece(y_clean, y_prob)
                    mets['Brier'] = calculate_brier(y_clean, y_prob)

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


# ==========================================
# 8. Monthly Report Generation
# ==========================================

SAFETY_GATES = {
    'Sensitivity': {'min': 0.70, 'label': 'Sensitivity', 'aging_aspect': '臨床敏感度', 'aging_desc': '偵測正例的能力'},
    'FPR':         {'max': 0.35, 'label': 'FPR', 'aging_aspect': '誤報控制', 'aging_desc': '假陽性控制能力'},
    'AUPRC':       {'min': 0.40, 'label': 'AUPRC', 'aging_aspect': '模型表現', 'aging_desc': '整體預測效能'},
    'AUROC':       {'min': 0.70, 'label': 'AUROC', 'aging_aspect': '分類能力', 'aging_desc': '區分正負例能力'},
}


def generate_monthly_report(windows, baseline_data, dates, model,
                            final_feats, target_col, model_version='未知',
                            trend_window=None):
    """
    Generate a structured monthly monitoring report with per-sample accuracy.

    Returns:
        dict with keys:
            'basic_info': model version, date range, sample counts, accuracy
            'safety_gates': list of dicts with metric/value/threshold/passed
            'additional': ECE, Brier, F1 values
            'all_passed': bool — True if ALL safety gates passed
            'llm_prompt': pre-built prompt string ready for LLM
    """
    import sklearn.metrics as skm

    report = {
        'basic_info': {},
        'safety_gates': [],
        'additional': {},
        'all_passed': True,
        'llm_prompt': '',
    }

    if not windows or len(windows) < 1:
        report['basic_info'] = {'error': '無可用資料'}
        return report

    # Apply trend_window: only use the last N windows
    if trend_window and trend_window < len(windows):
        analysis_windows = windows[-trend_window:]
        analysis_dates = dates[-trend_window:] if dates else []
    else:
        analysis_windows = windows
        analysis_dates = dates if dates else []

    # --- Aggregate all predictions across selected windows ---
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for w_df in analysis_windows:
        try:
            X_infer = w_df.drop(columns=[target_col], errors='ignore') if getattr(model, 'is_moe', False) else w_df[final_feats]
            y = w_df[target_col]

            # Drop NaN targets
            valid_mask = ~y.isna()
            if valid_mask.sum() < 2:
                continue
            y = y[valid_mask]
            X_valid = X_infer.loc[valid_mask] if hasattr(X_infer, 'loc') else X_infer[valid_mask.values]

            y_pred = model.predict(X_valid).astype(float)
            y_prob = model.predict_proba(X_valid)[:, 1]

            all_y_true.extend(y.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())
        except Exception:
            continue

    if len(all_y_true) < 10:
        report['basic_info'] = {'error': '有效樣本不足'}
        return report

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)

    # Final NaN cleanup
    valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_prob))
    y_true, y_pred, y_prob = y_true[valid], y_pred[valid], y_prob[valid]

    total = len(y_true)
    positive = int(y_true.sum())
    negative = total - positive
    prevalence = positive / total * 100

    # Confusion matrix
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    correct = tp + tn
    incorrect = fp + fn
    accuracy = correct / total * 100

    # Date range (based on selected analysis windows)
    start_date = analysis_dates[0] if analysis_dates else None
    end_date = analysis_dates[-1] if analysis_dates else None

    report['basic_info'] = {
        'model_version': model_version,
        'start_date': start_date.strftime('%Y/%m/%d') if hasattr(start_date, 'strftime') else str(start_date)[:10] if start_date else '?',
        'end_date': end_date.strftime('%Y/%m/%d') if hasattr(end_date, 'strftime') else str(end_date)[:10] if end_date else '?',
        'total': total,
        'positive': positive,
        'negative': negative,
        'prevalence': round(prevalence, 2),
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': round(accuracy, 2),
        'error_rate': round(100 - accuracy, 2),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }

    # --- Safety Gate Metrics ---
    # Sensitivity = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5
    # AUPRC
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(rec_curve, prec_curve)
    except Exception:
        auprc = 0.0

    gate_values = {
        'Sensitivity': round(sensitivity, 4),
        'FPR': round(fpr, 4),
        'AUPRC': round(auprc, 4),
        'AUROC': round(auroc, 4),
    }

    for metric_name, gate in SAFETY_GATES.items():
        val = gate_values[metric_name]
        if 'min' in gate:
            passed = val >= gate['min']
            threshold_str = f">= {gate['min']}"
            # Gauge: value as % of 1.0 scale
            gauge_pct = min(val * 100, 100)
            threshold_pct = gate['min'] * 100
        else:
            passed = val <= gate['max']
            threshold_str = f"<= {gate['max']}"
            # For FPR (lower is better): show value on 0-1 scale
            gauge_pct = min(val * 100, 100)
            threshold_pct = gate['max'] * 100

        if not passed:
            report['all_passed'] = False

        report['safety_gates'].append({
            'metric': metric_name,
            'label': gate['label'],
            'value': val,
            'threshold': threshold_str,
            'passed': passed,
            'gauge_pct': round(gauge_pct, 1),
            'threshold_pct': round(threshold_pct, 1),
            'is_lower_better': 'max' in gate,
            'aging_aspect': gate.get('aging_aspect', ''),
            'aging_desc': gate.get('aging_desc', ''),
        })

    # --- Additional Metrics ---
    f1 = f1_score(y_true, y_pred, pos_label=1.0)
    ece_val = calculate_ece(y_true, y_prob)
    brier_val = calculate_brier(y_true, y_prob)

    # JS Divergence for aging context
    if baseline_data is not None:
        baseline = baseline_data.copy()
    elif windows:
        baseline = windows[0]
    else:
        baseline = None
    latest_window = analysis_windows[-1] if analysis_windows else None
    if baseline is not None and latest_window is not None:
        avg_js, _ = compute_features_drift(baseline, latest_window, final_feats)
    else:
        avg_js = 0.0

    report['additional'] = {
        'F1': round(f1, 4),
        'ECE': round(ece_val, 4),
        'Brier': round(brier_val, 4),
        'JS_divergence': round(avg_js, 4),
    }

    # --- Aging Indicators (structured for display) ---
    report['aging_indicators'] = [
        {
            'icon': '📊',
            'label': '資料飄移',
            'metric': 'JS Divergence',
            'value': round(avg_js, 4),
            'status': 'critical' if avg_js >= 0.25 else ('warning' if avg_js >= 0.10 else 'good'),
            'desc': '輸入資料與訓練基準的分布差異。越高表示資料環境變化越大。',
        },
        {
            'icon': '🎯',
            'label': '預測信心',
            'metric': 'ECE',
            'value': round(ece_val, 4),
            'status': 'critical' if ece_val >= 0.25 else ('warning' if ece_val >= 0.15 else 'good'),
            'desc': '模型預測機率與實際結果的校準誤差。越高表示模型信心越不可靠。',
        },
        {
            'icon': '📈',
            'label': '模型表現',
            'metric': 'AUPRC',
            'value': round(auprc, 4),
            'status': 'critical' if auprc < 0.30 else ('warning' if auprc < 0.45 else 'good'),
            'desc': '在類別不平衡場景下的整體預測品質。越高代表表現越好。',
        },
    ]

    # --- Build LLM prompt (structured bullet-point format) ---
    info = report['basic_info']
    gates_lines = []
    for g in report['safety_gates']:
        status = '達標' if g['passed'] else '未達標'
        gates_lines.append(
            f"  {g['metric']}（{g.get('aging_aspect', '')}）: {g['value']} (門檻{g['threshold']}) [{status}]"
        )

    aging_lines = []
    for ai in report.get('aging_indicators', []):
        status_map = {'good': '正常', 'warning': '注意', 'critical': '警告'}
        aging_lines.append(
            f"  {ai['label']}（{ai['metric']}）= {ai['value']} [{status_map.get(ai['status'], '未知')}]"
        )

    report['llm_prompt'] = (
        f"你是醫療AI模型監控專家。根據以下IDH（血液透析中低血壓）預測模型的監控指標，寫一段結構化的分析報告。\n\n"
        f"格式要求（非常重要，務必嚴格遵守）：\n"
        f"1. 用以下四個段落標題分段，每段標題前加上對應符號：\n"
        f"   [現況] 模型本期表現概述（1~2句話總結）\n"
        f"   [老化] 模型老化跡象分析（根據三大老化指標判斷）\n"
        f"   [原因] 效能變化的可能根本原因（推測具體原因）\n"
        f"   [建議] 具體改善行動方案（可執行的建議）\n"
        f"2. 每段內用「・」開頭條列重點，每條不超過30字\n"
        f"3. 全文不超過300字\n"
        f"4. 不使用emoji、不使用markdown格式\n"
        f"5. 語氣專業但易懂\n\n"
        f"--- 監控數據 ---\n"
        f"模型版本：{info['model_version']}\n"
        f"監測期間：{info['start_date']}~{info['end_date']}\n"
        f"總樣本：{info['total']}筆，正例比例：{info['prevalence']}%\n"
        f"預測正確：{info['correct']}筆（{info['accuracy']}%）\n"
        f"預測錯誤：{info['incorrect']}筆（{round(100 - info['accuracy'], 2)}%）\n\n"
        f"安全閘門：\n{chr(10).join(gates_lines)}\n\n"
        f"老化指標：\n{chr(10).join(aging_lines)}\n\n"
        f"附加指標：\n"
        f"  F1: {report['additional']['F1']}\n"
        f"  ECE: {report['additional']['ECE']}\n"
        f"  Brier: {report['additional']['Brier']}\n"
        f"  JS Divergence: {report['additional']['JS_divergence']}\n"
    )

    return report


# ==========================================
# 9. Adaptive Thresholds
# ==========================================

# Absolute floor/ceiling values — never let adaptive thresholds go beyond these
_ADAPTIVE_FLOORS = {
    'Sensitivity': {'min': 0.15},
    'FPR':         {'max': 0.50},
    'AUPRC':       {'min': 0.20},
    'AUROC':       {'min': 0.50},
}


def compute_adaptive_thresholds(baseline_report):
    """
    Compute safety gate thresholds based on baseline model performance.

    Strategy:
    - Performance metrics (Sensitivity, AUPRC, AUROC): threshold = baseline × decay_rate
    - FPR (lower is better): threshold = baseline × inflation_rate
    - All values clamped by absolute floor/ceiling

    Args:
        baseline_report: dict from generate_monthly_report (needs 'safety_gates')

    Returns:
        dict keyed by metric name with 'min' or 'max' threshold
    """
    if not baseline_report or not baseline_report.get('safety_gates'):
        return None

    # Build baseline value lookup
    baseline_vals = {}
    for gate in baseline_report['safety_gates']:
        baseline_vals[gate['metric']] = gate['value']

    adaptive = {}

    for metric_name, floor in _ADAPTIVE_FLOORS.items():
        bval = baseline_vals.get(metric_name)
        if bval is None:
            continue

        gate_info = SAFETY_GATES.get(metric_name, {})

        if 'min' in gate_info:
            # Higher is better — allow 15% degradation
            thresh = round(bval * 0.85, 4)
            absolute_floor = floor.get('min', 0.0)
            thresh = max(thresh, absolute_floor)
            adaptive[metric_name] = {
                **gate_info,
                'min': thresh,
            }
        elif 'max' in gate_info:
            # Lower is better (FPR) — allow 50% increase
            thresh = round(bval * 1.50, 4)
            absolute_ceiling = floor.get('max', 1.0)
            thresh = min(thresh, absolute_ceiling)
            adaptive[metric_name] = {
                **gate_info,
                'max': thresh,
            }

    return adaptive if adaptive else None


# ==========================================
# 10. Single-Model Longevity Analysis
# ==========================================

def compute_single_model_longevity(windows, baseline_data, dates, model,
                                    final_feats, target_col,
                                    adaptive_thresholds=None):
    """
    Analyze single-model lifespan using safety gate pass/fail patterns over time.

    Instead of inventing new thresholds, reuses safety gates to determine
    each window's health status, then looks at the temporal pattern.

    Args:
        windows: list of DataFrames (time windows)
        baseline_data: baseline DataFrame or None
        dates: list of window end dates
        model: model object
        final_feats: list of feature names
        target_col: target column name
        adaptive_thresholds: dict from compute_adaptive_thresholds (or None for defaults)

    Returns:
        dict with longevity analysis results
    """
    if not windows or len(windows) < 2:
        return {
            'level': '資料不足',
            'level_code': 'unknown',
            'health_score': 0,
            'timeline': [],
            'n_windows': 0,
            'max_consecutive_pass': 0,
            'first_degradation': None,
            'trend_slope': 0.0,
        }

    import sklearn.metrics as skm

    # Use adaptive thresholds if available, otherwise defaults
    gates_config = adaptive_thresholds if adaptive_thresholds else SAFETY_GATES

    timeline = []
    auprc_values = []

    for i, w_df in enumerate(windows):
        try:
            X_infer = (w_df.drop(columns=[target_col], errors='ignore')
                       if getattr(model, 'is_moe', False)
                       else w_df[final_feats])
            y = w_df[target_col]

            valid_mask = ~y.isna()
            if valid_mask.sum() < 2:
                continue

            y_clean = y[valid_mask]
            X_clean = X_infer.loc[valid_mask] if hasattr(X_infer, 'loc') else X_infer[valid_mask.values]

            y_pred = model.predict(X_clean).astype(float)
            y_prob = model.predict_proba(X_clean)[:, 1]

            y_arr = np.array(y_clean)
            y_pred_arr = np.array(y_pred)

            tp = int(((y_pred_arr == 1) & (y_arr == 1)).sum())
            tn = int(((y_pred_arr == 0) & (y_arr == 0)).sum())
            fp = int(((y_pred_arr == 1) & (y_arr == 0)).sum())
            fn = int(((y_pred_arr == 0) & (y_arr == 1)).sum())

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            try:
                auroc = roc_auc_score(y_arr, y_prob)
            except Exception:
                auroc = 0.5
            try:
                prec_c, rec_c, _ = precision_recall_curve(y_arr, y_prob)
                auprc = auc(rec_c, prec_c)
            except Exception:
                auprc = 0.0

            gate_values = {
                'Sensitivity': round(sensitivity, 4),
                'FPR': round(fpr, 4),
                'AUPRC': round(auprc, 4),
                'AUROC': round(auroc, 4),
            }

            # Check each gate
            n_pass = 0
            n_total = 0
            for metric_name, gate in gates_config.items():
                val = gate_values.get(metric_name)
                if val is None:
                    continue
                n_total += 1
                if 'min' in gate and val >= gate['min']:
                    n_pass += 1
                elif 'max' in gate and val <= gate['max']:
                    n_pass += 1

            pass_rate = n_pass / n_total if n_total > 0 else 0.0
            all_pass = (n_pass == n_total)

            auprc_values.append(auprc)
            timeline.append({
                'window': i + 1,
                'date': dates[i].strftime('%Y/%m/%d') if hasattr(dates[i], 'strftime') else str(dates[i])[:10],
                'auprc': round(auprc, 4),
                'pass_rate': round(pass_rate, 2),
                'all_pass': all_pass,
                'n_pass': n_pass,
                'n_total': n_total,
            })
        except Exception:
            continue

    if not timeline:
        return {
            'level': '資料不足',
            'level_code': 'unknown',
            'health_score': 0,
            'timeline': [],
            'n_windows': 0,
            'max_consecutive_pass': 0,
            'first_degradation': None,
            'trend_slope': 0.0,
        }

    n_windows = len(timeline)

    # Max consecutive all-pass windows
    max_consecutive = 0
    current_consec = 0
    for t in timeline:
        if t['all_pass']:
            current_consec += 1
            max_consecutive = max(max_consecutive, current_consec)
        else:
            current_consec = 0

    # First degradation (first window that didn't all-pass)
    first_degradation = None
    for t in timeline:
        if not t['all_pass']:
            first_degradation = t['window']
            break

    # AUPRC trend slope (weighted — recent windows matter more)
    trend_slope = 0.0
    if len(auprc_values) >= 3:
        x = np.arange(len(auprc_values))
        # Exponential weights: more recent = heavier
        weights = np.exp(np.linspace(0, 2, len(auprc_values)))
        weights /= weights.sum()
        # Weighted least squares
        x_mean = np.average(x, weights=weights)
        y_mean = np.average(auprc_values, weights=weights)
        num = np.sum(weights * (x - x_mean) * (np.array(auprc_values) - y_mean))
        den = np.sum(weights * (x - x_mean) ** 2)
        trend_slope = float(num / den) if den != 0 else 0.0

    # === Determine level using voting (no new magic numbers) ===
    recent_n = min(5, n_windows)
    recent = timeline[-recent_n:]

    if all(t['all_pass'] for t in recent):
        level = '穩定'
        level_code = 'stable'
    elif sum(t['all_pass'] for t in recent) >= (recent_n * 0.6):
        level = '觀察'
        level_code = 'watch'
    elif any(t['all_pass'] for t in recent):
        level = '警示'
        level_code = 'warning'
    else:
        level = '失效'
        level_code = 'critical'

    # Health score (0-100, equal weights)
    in_band_ratio = sum(1 for t in timeline if t['all_pass']) / n_windows
    recent_pass_ratio = sum(1 for t in recent if t['all_pass']) / recent_n
    consecutive_ratio = max_consecutive / n_windows if n_windows > 0 else 0
    slope_stability = max(0, 1 + trend_slope * 10)  # slope ~ -0.04 → 0.6, slope ~ 0 → 1.0

    health_score = int(round(
        (in_band_ratio + recent_pass_ratio + consecutive_ratio + min(slope_stability, 1.0)) / 4.0 * 100
    ))
    health_score = max(0, min(100, health_score))

    return {
        'level': level,
        'level_code': level_code,
        'health_score': health_score,
        'timeline': timeline,
        'n_windows': n_windows,
        'max_consecutive_pass': max_consecutive,
        'first_degradation': first_degradation,
        'trend_slope': round(trend_slope, 6),
    }


# ==========================================
# 11. Combined LLM Prompt (Dual Window Configs)
# ==========================================

def build_combined_llm_prompt(report_a, report_b, label_a='90天窗格', label_b='30天窗格'):
    """
    Merge two window-config reports into a single LLM prompt.
    The LLM is asked to output tagged sections for each config.

    Args:
        report_a: dict from generate_monthly_report (config A, e.g., 90-day)
        report_b: dict from generate_monthly_report (config B, e.g., 30-day)

    Returns:
        str: combined prompt string
    """
    def _format_report_data(report, label):
        info = report.get('basic_info', {})
        if info.get('error'):
            return f"--- {label} ---\n資料不足，無法分析\n"

        gates_lines = []
        for g in report.get('safety_gates', []):
            status = '達標' if g['passed'] else '未達標'
            gates_lines.append(
                f"  {g['metric']}: {g['value']} (門檻{g['threshold']}) [{status}]"
            )

        aging_lines = []
        for ai in report.get('aging_indicators', []):
            status_map = {'good': '正常', 'warning': '注意', 'critical': '警告'}
            aging_lines.append(
                f"  {ai['label']}（{ai['metric']}）= {ai['value']} [{status_map.get(ai['status'], '未知')}]"
            )

        return (
            f"--- {label} ---\n"
            f"監測期間：{info['start_date']}~{info['end_date']}\n"
            f"總樣本：{info['total']}筆，正例比例：{info['prevalence']}%\n"
            f"預測正確：{info['correct']}筆（{info['accuracy']}%）\n"
            f"安全閘門：\n{chr(10).join(gates_lines)}\n"
            f"老化指標：\n{chr(10).join(aging_lines)}\n"
            f"F1: {report['additional']['F1']}  ECE: {report['additional']['ECE']}  "
            f"Brier: {report['additional']['Brier']}  JS Div: {report['additional'].get('JS_divergence', 'N/A')}\n"
        )

    data_a = _format_report_data(report_a, label_a)
    data_b = _format_report_data(report_b, label_b)

    info_a = report_a.get('basic_info', {})
    model_ver = info_a.get('model_version', '未知')

    prompt = (
        f"你是醫療AI模型監控專家。以下是IDH（血液透析中低血壓）預測模型 {model_ver} 在兩種不同時間粒度下的監控結果。\n\n"
        f"格式要求（非常重要，務必嚴格遵守）：\n"
        f"1. 用以下六個段落標題分段，每段標題前必須加方括號標記：\n"
        f"   [90天-現況] 90天窗格觀察到的模型本期表現概述\n"
        f"   [30天-現況] 30天窗格觀察到的模型本期表現概述\n"
        f"   [90天-老化] 90天窗格觀察到的老化跡象\n"
        f"   [30天-老化] 30天窗格觀察到的老化跡象\n"
        f"   [原因] 效能變化的可能根本原因（綜合兩組觀察）\n"
        f"   [建議] 具體改善行動方案（綜合兩組觀察）\n"
        f"2. 每段內用「・」開頭條列重點，每條不超過30字\n"
        f"3. 全文不超過500字\n"
        f"4. 不使用emoji、不使用markdown格式\n"
        f"5. 如果兩組結論有差異，在對應段落中特別指出\n\n"
        f"{data_a}\n"
        f"{data_b}\n"
    )

    return prompt

