"""
Retrain Pipeline Service — Core logic for Model Maintenance
=============================================================
Implements the 3-level pipeline:
  Level 0: Monitor AUPRC
  Level 1: Feature Pruning (zero out low-contribution terms)
  Level 2: Full Retraining (train ver2 with new data)
  + Data Drift & Concept Drift diagnosis
"""

import copy
import logging
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ==========================================
# Helpers
# ==========================================

def _compute_auprc(y, p):
    """Compute AUPRC with NaN handling."""
    y, p = np.array(y, dtype=float), np.array(p, dtype=float)
    v = ~(np.isnan(y) | np.isnan(p))
    y, p = y[v], p[v]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    pr, rc, _ = precision_recall_curve(y, p)
    return round(float(auc(rc, pr)), 4)


def _eval_model(model, X, y):
    """Evaluate a model and return metrics dict."""
    y = np.array(y, dtype=float)
    v = ~np.isnan(y)
    X2 = X.iloc[v.nonzero()[0]] if hasattr(X, 'iloc') else X[v]
    y2 = y[v]
    if len(y2) < 2 or len(np.unique(y2)) < 2:
        return None
    prob = model.predict_proba(X2)[:, 1]
    pred = model.predict(X2).astype(float)
    pr, rc, _ = precision_recall_curve(y2, prob)
    return {
        'AUROC': round(float(roc_auc_score(y2, prob)), 4),
        'AUPRC': round(float(auc(rc, pr)), 4),
        'F1': round(float(f1_score(y2, pred, pos_label=1.0)), 4),
    }


def _bootstrap_ci(y, p1, p2, n=1000, seed=42):
    """Bootstrap CI for AUPRC difference (p2 - p1)."""
    rng = np.random.RandomState(seed)
    N = len(y)
    ds = []
    for _ in range(n):
        i = rng.choice(N, N, replace=True)
        yb = y[i]
        if len(np.unique(yb)) < 2:
            continue
        a1 = _compute_auprc(yb, p1[i])
        a2 = _compute_auprc(yb, p2[i])
        if a1 and a2:
            ds.append(a2 - a1)
    ds = np.array(ds)
    if len(ds) == 0:
        return {'mean': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0, 'significant': False}
    return {
        'mean': round(float(np.mean(ds)), 4),
        'ci_lo': round(float(np.percentile(ds, 2.5)), 4),
        'ci_hi': round(float(np.percentile(ds, 97.5)), 4),
        'significant': bool(np.percentile(ds, 2.5) > 0),
    }


def _js_div(d1, d2, bins=20):
    """JS Divergence between two series."""
    d1 = pd.Series(d1).dropna()
    d2 = pd.Series(d2).dropna()
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0
    lo, hi = min(d1.min(), d2.min()), max(d1.max(), d2.max())
    if lo == hi:
        return 0.0
    p, _ = np.histogram(d1, bins=bins, range=(lo, hi), density=True)
    q, _ = np.histogram(d2, bins=bins, range=(lo, hi), density=True)
    return float(jensenshannon(p + 1e-10, q + 1e-10, base=2))


def _get_feature_importances(model):
    """Extract EBM feature importances as dict."""
    importances = model.term_importances()
    names = model.term_names_
    result = {}
    for i, name in enumerate(names):
        if i < len(importances):
            result[name] = float(importances[i])
    return result


def _edit_model_prune(model, features_to_zero):
    """Zero out shape functions for specified features."""
    edited = copy.deepcopy(model)
    for i, name in enumerate(edited.term_names_):
        if name in features_to_zero:
            edited.term_scores_[i] = np.zeros_like(edited.term_scores_[i])
    return edited


# ==========================================
# AUPRC Scan & Segment Detection
# ==========================================

def scan_auprc_windows(df, features, target_col, window_days=90, stride_days=30):
    """Scan all data with sliding windows and compute AUPRC per window.
    
    Returns list of dicts: [{date_start, date_end, auprc, n_samples, prevalence}, ...]
    """
    df = df.sort_values('Session_Date')
    start = df['Session_Date'].min()
    end = df['Session_Date'].max()
    current = start

    # Train a temporary model on first 6 months for scoring
    six_mo = start + pd.DateOffset(months=6)
    df_init = df[df['Session_Date'] <= six_mo]
    X_init = df_init[features]
    y_init = df_init[target_col]
    m = ~y_init.isna()
    if m.sum() < 50:
        return []
    tmp_model = ExplainableBoostingClassifier(random_state=RANDOM_STATE)
    tmp_model.fit(X_init[m], y_init[m])

    results = []
    while current < end:
        w_end = current + pd.Timedelta(days=window_days)
        mask = (df['Session_Date'] >= current) & (df['Session_Date'] < w_end)
        w_df = df.loc[mask]
        if len(w_df) > 10:
            y = w_df[target_col].values
            valid = ~np.isnan(y.astype(float))
            if valid.sum() >= 5 and len(np.unique(y[valid])) >= 2:
                prob = tmp_model.predict_proba(w_df[features].iloc[valid.nonzero()[0]])[:, 1]
                auprc = _compute_auprc(y[valid], prob)
                if auprc is not None:
                    prev = float(np.nanmean(y[valid]))
                    results.append({
                        'date_start': str(current.date()),
                        'date_end': str(w_end.date()),
                        'date_mid': str((current + (w_end - current) / 2).date()),
                        'auprc': auprc,
                        'n_samples': int(valid.sum()),
                        'prevalence': round(prev, 4),
                    })
        current += pd.Timedelta(days=stride_days)

    return results


def detect_auprc_segments(scan_results, min_t0_months=6, min_t1_months=3):
    """Detect T0 (high AUPRC) and T1 (drop) segments from scan results.
    
    Three-layer detection (from services.py ALERT_THRESHOLDS_STRICT):
      Layer 1: Relative drop > 15% from peak → 'transition', > 30% → 'drop'
      Layer 2: Absolute AUPRC < 0.45 → 'transition', < 0.30 → 'drop'
      Layer 3: Trend analysis
    
    Returns dict with t0_start, t0_end, t1_start, t1_end, labels, etc.
    """
    if not scan_results or len(scan_results) < 4:
        return None

    auprc_vals = [r['auprc'] for r in scan_results]
    peak = max(auprc_vals)  # global peak — used for peak_auprc in return value

    # Label each window using rolling LOCAL peak (±6 windows ≈ ±6 months)
    local_half = 6  # look ±6 windows on each side
    labels = []
    for i, val in enumerate(auprc_vals):
        lo = max(0, i - local_half)
        hi = min(len(auprc_vals), i + local_half + 1)
        local_peak = max(auprc_vals[lo:hi])
        pct_drop = (local_peak - val) / local_peak if local_peak > 0 else 0
        if pct_drop > 0.30 or val < 0.30:
            labels.append('drop')
        elif pct_drop > 0.15 or val < 0.45:
            labels.append('transition')
        else:
            labels.append('high')

    # Find longest consecutive 'high' run for T0
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, l in enumerate(labels):
        if l == 'high':
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0

    # If no 'high' windows, use top 50% by AUPRC
    if best_len == 0:
        median_auprc = np.median(auprc_vals)
        for i, val in enumerate(auprc_vals):
            labels[i] = 'high' if val >= median_auprc else 'drop'
        # Re-find longest high run
        best_start, best_len = 0, 0
        cur_start, cur_len = 0, 0
        for i, l in enumerate(labels):
            if l == 'high':
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
            else:
                cur_len = 0

    # Ensure T0 is at least min_t0_months worth of windows (~6 windows for 90d/30d)
    min_t0_windows = max(2, int(min_t0_months * 30 / 30))  # ~1 window per month
    if best_len < min_t0_windows:
        # Extend T0 by including adjacent 'transition' windows
        while best_len < min_t0_windows and best_start + best_len < len(labels):
            best_len += 1
        while best_len < min_t0_windows and best_start > 0:
            best_start -= 1
            best_len += 1

    t0_end_idx = best_start + best_len - 1

    # Find T1: first 'drop' or 'transition' run after T0
    t1_start_idx = t0_end_idx + 1
    if t1_start_idx >= len(scan_results):
        t1_start_idx = max(0, len(scan_results) - 4)

    min_t1_windows = max(1, int(min_t1_months * 30 / 30))
    t1_end_idx = min(len(scan_results) - 1, t1_start_idx + min_t1_windows - 1)
    # Extend T1 to include consecutive drop/transition
    while t1_end_idx + 1 < len(scan_results) and labels[t1_end_idx + 1] in ('drop', 'transition'):
        t1_end_idx += 1

    # Ensure minimum T1 length
    if t1_end_idx - t1_start_idx + 1 < min_t1_windows:
        t1_end_idx = min(len(scan_results) - 1, t1_start_idx + min_t1_windows - 1)

    # T0 ends at the START of the first non-high window (to avoid including drop data)
    # Because windows are 90d wide, using date_end of the last high window would
    # extend T0 into the drop period.
    if t0_end_idx + 1 < len(scan_results):
        t0_date_end = scan_results[t0_end_idx + 1]['date_start']
    else:
        t0_date_end = scan_results[t0_end_idx]['date_end']

    # T1 starts exactly where T0 ends
    t1_date_start = t0_date_end
    t1_date_end = scan_results[t1_end_idx]['date_end']
    
    # if T1 end is somehow before or equal to T1 start due to windowing, push it forward
    if pd.Timestamp(t1_date_end) <= pd.Timestamp(t1_date_start):
        t1_date_end = str((pd.Timestamp(t1_date_start) + pd.DateOffset(months=min_t1_months)).date())

    return {
        'labels': labels,
        't0_start_idx': best_start,
        't0_end_idx': t0_end_idx,
        't1_start_idx': t1_start_idx,
        't1_end_idx': t1_end_idx,
        't0_date_start': scan_results[best_start]['date_start'],
        't0_date_end': t0_date_end,
        't1_date_start': t1_date_start,
        't1_date_end': t1_date_end,
        't0_windows': best_len,
        't1_windows': t1_end_idx - t1_start_idx + 1,
        'peak_auprc': peak,
        'drop_criteria': {
            'warning_drop_pct': 0.15,
            'critical_drop_pct': 0.30,
            'warning_below': 0.45,
            'critical_below': 0.30,
        },
    }


def detect_best_worst_segments(scan_results, segment_months=6):
    """Find the globally best and worst AUPRC segments for drift comparison.
    
    Selects two non-overlapping consecutive runs:
      - T0 (Best):  the consecutive block with the highest mean AUPRC
      - T1 (Worst): the consecutive block with the lowest mean AUPRC
    
    Args:
        scan_results: list of window dicts from scan_auprc_windows
        segment_months: target length of each segment in months (~N windows with 30d stride)
    
    Returns:
        dict with t0/t1 dates and labels, or None if insufficient data.
    """
    if not scan_results or len(scan_results) < 4:
        return None

    auprc_vals = [r['auprc'] for r in scan_results]
    n = len(auprc_vals)
    seg_len = max(2, segment_months)  # ~1 window per month with 30d stride

    if seg_len > n // 2:
        seg_len = max(2, n // 3)

    # Sliding window mean AUPRC
    best_start, best_mean = 0, -1
    worst_start, worst_mean = 0, float('inf')

    for i in range(n - seg_len + 1):
        window = auprc_vals[i:i + seg_len]
        m = sum(window) / len(window)
        if m > best_mean:
            best_mean = m
            best_start = i
        if m < worst_mean:
            worst_mean = m
            worst_start = i

    best_end = best_start + seg_len - 1
    worst_end = worst_start + seg_len - 1

    # Ensure non-overlapping: if they overlap, shrink the one selected second
    if not (best_end < worst_start or worst_end < best_start):
        # They overlap — keep the best, re-find worst outside best range
        worst_mean = float('inf')
        worst_start = 0
        for i in range(n - seg_len + 1):
            if i + seg_len - 1 < best_start or i > best_end:
                window = auprc_vals[i:i + seg_len]
                m = sum(window) / len(window)
                if m < worst_mean:
                    worst_mean = m
                    worst_start = i
        worst_end = worst_start + seg_len - 1

    # Build labels for chart coloring
    labels = []
    for i in range(n):
        if best_start <= i <= best_end:
            labels.append('high')
        elif worst_start <= i <= worst_end:
            labels.append('drop')
        else:
            labels.append('transition')

    return {
        'labels': labels,
        't0_start_idx': best_start,
        't0_end_idx': best_end,
        't1_start_idx': worst_start,
        't1_end_idx': worst_end,
        't0_date_start': scan_results[best_start]['date_start'],
        't0_date_end': scan_results[best_end]['date_end'],
        't1_date_start': scan_results[worst_start]['date_start'],
        't1_date_end': scan_results[worst_end]['date_end'],
        't0_windows': seg_len,
        't1_windows': seg_len,
        'peak_auprc': max(auprc_vals),
        'best_mean_auprc': round(best_mean, 4),
        'worst_mean_auprc': round(worst_mean, 4),
        'mode': 'best_vs_worst',
        'drop_criteria': {
            'warning_drop_pct': 0.15,
            'critical_drop_pct': 0.30,
            'warning_below': 0.45,
            'critical_below': 0.30,
        },
    }

def save_segment_data(df, segments, save_dir):
    """Save T0 and T1 data segments as CSV files."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    t0_start = pd.Timestamp(segments['t0_date_start'])
    t0_end = pd.Timestamp(segments['t0_date_end'])
    t1_start = pd.Timestamp(segments['t1_date_start'])
    t1_end = pd.Timestamp(segments['t1_date_end'])

    df_t0 = df[(df['Session_Date'] >= t0_start) & (df['Session_Date'] < t0_end)].copy()
    df_t1 = df[(df['Session_Date'] >= t1_start) & (df['Session_Date'] < t1_end)].copy()

    t0_path = os.path.join(save_dir, 'T0_data.csv')
    t1_path = os.path.join(save_dir, 'T1_data.csv')
    df_t0.to_csv(t0_path, index=False)
    df_t1.to_csv(t1_path, index=False)

    return {'t0_path': t0_path, 't1_path': t1_path,
            't0_rows': len(df_t0), 't1_rows': len(df_t1)}


def build_scan_chart(scan_results, segments, baseline_auprc=None,
                     drift_thresholds=None, threshold_mode='fixed'):
    """Build AUPRC timeline chart with T0/T1 regions highlighted."""
    if not scan_results:
        return None

    dates = [
        r.get('date_mid') or str(
            (pd.Timestamp(r['date_start']) + (pd.Timestamp(r['date_end']) - pd.Timestamp(r['date_start'])) / 2).date()
        )
        for r in scan_results
    ]
    auprcs = [r['auprc'] for r in scan_results]

    # Use segment labels if available and length matches, otherwise derive from baseline or AUPRC values
    raw_labels = segments['labels'] if segments else []
    if len(raw_labels) == len(dates):
        labels = raw_labels
    elif baseline_auprc and baseline_auprc > 0:
        # Manual mode: color relative to baseline
        drop_thresh = baseline_auprc * 0.85
        warn_thresh = baseline_auprc * 0.95
        labels = []
        for a in auprcs:
            if a >= warn_thresh:
                labels.append('high')
            elif a >= drop_thresh:
                labels.append('transition')
            else:
                labels.append('drop')
    else:
        # Fallback: color by absolute AUPRC
        labels = []
        for a in auprcs:
            if a >= 0.55:
                labels.append('high')
            elif a >= 0.45:
                labels.append('transition')
            else:
                labels.append('drop')

    colors = {'high': '#10b981', 'transition': '#f59e0b', 'drop': '#ef4444'}
    marker_colors = [colors.get(l, '#94a3b8') for l in labels]

    fig = go.Figure()

    # AUPRC line
    fig.add_trace(go.Scatter(
        x=dates, y=auprcs, mode='lines+markers',
        name='AUPRC',
        line=dict(color='#6366f1', width=2),
        marker=dict(color=marker_colors, size=8, line=dict(width=1, color='#fff')),
    ))

    # Threshold lines — use adaptive global thresholds when available
    if threshold_mode == 'adaptive' and drift_thresholds and '__global__' in drift_thresholds:
        gl = drift_thresholds['__global__']
        warn_val = gl['warning']
        crit_val = gl['critical']
        warn_label = f'Warning (adaptive p95={warn_val:.3f})'
        crit_label = f'Critical (adaptive p99={crit_val:.3f})'
    else:
        warn_val = 0.45
        crit_val = 0.30
        warn_label = 'Warning (0.45)'
        crit_label = 'Critical (0.30)'

    fig.add_hline(y=warn_val, line_dash='dash', line_color='#f59e0b',
                  annotation_text=warn_label, annotation_position='top left')
    fig.add_hline(y=crit_val, line_dash='dash', line_color='#ef4444',
                  annotation_text=crit_label, annotation_position='top left')

    # Baseline AUPRC line (manual mode)
    if baseline_auprc and baseline_auprc > 0:
        fig.add_hline(y=baseline_auprc, line_dash='dot', line_color='#a5b4fc',
                      annotation_text=f'Baseline AUPRC ({baseline_auprc:.3f})',
                      annotation_position='bottom right',
                      annotation_font_color='#a5b4fc')

    if segments:
        is_bw = segments.get('mode') == 'best_vs_worst'
        t0_label = 'Best (高表現期)' if is_bw else 'T0 (訓練期)'
        t1_label = 'Worst (低表現期)' if is_bw else 'T1 (推論期/Drop)'

        # T0 / Best region
        fig.add_vrect(
            x0=segments['t0_date_start'], x1=segments['t0_date_end'],
            fillcolor='rgba(16,185,129,0.12)', line_width=2,
            line_color='rgba(16,185,129,0.5)',
            annotation_text=t0_label, annotation_position='top left',
            annotation_font_color='#10b981',
        )
        # T1 / Worst region
        fig.add_vrect(
            x0=segments['t1_date_start'], x1=segments['t1_date_end'],
            fillcolor='rgba(239,68,68,0.12)', line_width=2,
            line_color='rgba(239,68,68,0.5)',
            annotation_text=t1_label, annotation_position='top right',
            annotation_font_color='#ef4444',
        )

    fig.update_layout(
        title='AUPRC 時序掃描 — 滑動視窗',
        xaxis_title='Time', yaxis_title='AUPRC',
        height=400, yaxis=dict(range=[0, 1.05], gridcolor='rgba(99,102,241,0.08)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    return fig


# ==========================================
# Main Pipeline
# ==========================================

def run_retrain_pipeline(csv_path, features, target_col,
                         ver1_model=None, ver1_start_date=None, ver1_end_date=None,
                         model_name='Unknown', data_name='Unknown',
                         drift_threshold_mode='fixed',
                         scan_mode=False,
                         test_csv_path=None,
                         progress_callback=None,
                         manual_t0_start=None, manual_t0_end=None,
                         cached_scan_results=None):
    """
    Run the full retrain pipeline.

    Args:
        csv_path: Path to the CSV data file.
        features: List of feature column names.
        target_col: Target column name.
        ver1_model: Pre-trained EBM model (ver1). If None, trains from data.
        ver1_end_date: Cutoff date string (YYYY-MM-DD).
        model_name: Display name of the uploaded model.
        data_name: Display name of the uploaded dataset.
        drift_threshold_mode: 'fixed' or 'adaptive'.
        scan_mode: If True, auto-detect T0/T1 via AUPRC scan instead of using ver1_end_date.
        test_csv_path: Optional path to test CSV for baseline evaluation.
        progress_callback: Optional callable(step, total, message).
        manual_t0_start: Optional manual T0 start date string (YYYY-MM-DD). Overrides auto-detection.
        manual_t0_end: Optional manual T0 end date string (YYYY-MM-DD). Overrides auto-detection.
        cached_scan_results: Optional pre-computed scan results (for re-analysis without re-scanning).

    Returns:
        dict with all results.
    """
    result = {
        'success': False,
        'error': None,
        'elapsed_s': 0,
    }
    t0 = time.time()

    def _progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        logger.info(f"[retrain] Step {step}/{total}: {msg}")

    try:
        # ========== STEP 0: Load and prepare data ==========
        _progress(1, 8, '載入資料中...')

        df = pd.read_csv(csv_path, low_memory=False)
        df['Session_Date'] = pd.to_datetime(df['Session_Date'], errors='coerce')
        df = df.dropna(subset=['Session_Date']).sort_values('Session_Date')
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        data_start = df['Session_Date'].min()
        data_end = df['Session_Date'].max()
        cutoff_start = data_start  # default; overridden by manual mode

        # ========== SCAN MODE: Auto-detect or Manual T0/T1 ==========
        if scan_mode:
            # If manual T0 dates are specified, skip scanning and use them directly
            if manual_t0_start and manual_t0_end:
                _progress(1, 7, '使用手動 T0 範圍...')

                # Re-use cached scan results if available, otherwise re-scan
                if cached_scan_results:
                    scan_results = cached_scan_results
                else:
                    scan_results = scan_auprc_windows(df, features, target_col)
                    if not scan_results:
                        result['error'] = 'AUPRC 掃描失敗：無法產生有效窗格。'
                        return result

                # Re-detect segments with auto-detect for labels/chart
                segments = detect_auprc_segments(scan_results)

                # Override T0 dates with manual values
                manual_t0_s = pd.Timestamp(manual_t0_start)
                manual_t0_e = pd.Timestamp(manual_t0_end)

                if segments:
                    segments['t0_date_start'] = str(manual_t0_s.date())
                    segments['t0_date_end'] = str(manual_t0_e.date())
                    # T1 starts right after manual T0 end
                    segments['t1_date_start'] = str((manual_t0_e + pd.Timedelta(days=1)).date())
                else:
                    # Build minimal segments from manual dates
                    last_date = df['Session_Date'].max()
                    t1_end = last_date
                    segments = {
                        't0_date_start': str(manual_t0_s.date()),
                        't0_date_end': str(manual_t0_e.date()),
                        't1_date_start': str((manual_t0_e + pd.Timedelta(days=1)).date()),
                        't1_date_end': str(t1_end.date()),
                        'labels': [],
                        'peak_auprc': max(r['auprc'] for r in scan_results) if scan_results else 0,
                    }

                result['scan_results'] = scan_results
                result['segments'] = segments
                result['manual_t0'] = True

                cutoff = manual_t0_e
                ver2_end = pd.Timestamp(segments['t1_date_end'])
            else:
                _progress(1, 7, 'AUPRC 滑動視窗掃描中（大資料集可能需要1-2分鐘）...')
                scan_results = scan_auprc_windows(df, features, target_col)
                if not scan_results:
                    result['error'] = 'AUPRC 掃描失敗：無法產生有效窗格。'
                    return result

                _progress(2, 7, '偵測 AUPRC Drop 區段中...')
                segments = detect_auprc_segments(scan_results)
                if not segments:
                    result['error'] = 'AUPRC 區段偵測失敗：資料窗格不足。'
                    return result

                result['scan_results'] = scan_results
                result['segments'] = segments

                # Save segment data
                import os
                save_dir = os.path.join(os.path.dirname(csv_path), 'segments')
                saved = save_segment_data(df, segments, save_dir)
                result['saved_segments'] = saved

                # Use detected segments as cutoff
                cutoff = pd.Timestamp(segments['t0_date_end'])
                ver2_end = pd.Timestamp(segments['t1_date_end'])
                cutoff_start = data_start
        else:
            # Manual cutoff mode
            _progress(1, 7, '設定手動訓練區間與基準驗證集...')
            cutoff_start = pd.Timestamp(ver1_start_date) if ver1_start_date else data_start
            cutoff_end = pd.Timestamp(ver1_end_date) if ver1_end_date else (data_end - pd.DateOffset(months=3))
            
            df_v1_all = df[(df['Session_Date'] >= cutoff_start) & (df['Session_Date'] <= cutoff_end)].copy()
            if 'Patient_ID' not in df.columns or len(df_v1_all) < 50:
                result['error'] = '資料不足或缺少 Patient_ID 欄位以進行 Validation Baseline 計算。'
                return result
                
            splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
            try:
                v1_train_idx, v1_val_idx = next(splitter_val.split(df_v1_all, groups=df_v1_all['Patient_ID']))
            except Exception as e:
                result['error'] = f'Train/Val 切分失敗: {str(e)}'
                return result
                
            df_v1_train_tmp = df_v1_all.iloc[v1_train_idx]
            df_v1_val_tmp = df_v1_all.iloc[v1_val_idx]
            
            _progress(2, 7, '訓練 T0 基準模型 (Baseline)...')
            from sklearn.ensemble import HistGradientBoostingClassifier
            tmp_model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
            X1_tmp = df_v1_train_tmp[features]
            y1_tmp = df_v1_train_tmp[target_col]
            m1_tmp = ~y1_tmp.isna()
            tmp_model.fit(X1_tmp[m1_tmp], y1_tmp[m1_tmp])
            
            y1_val = df_v1_val_tmp[target_col].values.astype(float)
            m1_val = ~np.isnan(y1_val)
            if m1_val.sum() >= 5:
                prob_val = tmp_model.predict_proba(df_v1_val_tmp[features].iloc[m1_val.nonzero()[0]])[:, 1]
                baseline_auprc = _compute_auprc(y1_val[m1_val], prob_val)
            else:
                baseline_auprc = 0.5
            if baseline_auprc is None: baseline_auprc = 0.5
            
            _progress(3, 7, f'基準 AUPRC={baseline_auprc:.4f}，掃描推論期 Drop...')
            drop_threshold = baseline_auprc * 0.85
            df_inf = df[df['Session_Date'] > cutoff_end]
            
            ver2_end = data_end
            current = cutoff_end
            inf_end = df_inf['Session_Date'].max() if not df_inf.empty else cutoff_end
            
            scan_results = []
            drop_found = False
            while current < inf_end:
                w_end = current + pd.Timedelta(days=30)
                w_df = df_inf[(df_inf['Session_Date'] >= current) & (df_inf['Session_Date'] < w_end)]
                if len(w_df) > 10:
                    y_w = w_df[target_col].values
                    m_w = ~np.isnan(y_w.astype(float))
                    if m_w.sum() >= 5 and len(np.unique(y_w[m_w])) >= 2:
                        prob_w = tmp_model.predict_proba(w_df[features].iloc[m_w.nonzero()[0]])[:, 1]
                        auprc_w = _compute_auprc(y_w[m_w], prob_w)
                        if auprc_w is not None:
                            w_mid = current + (w_end - current) / 2
                            scan_results.append({
                                'date_start': str(current.date()),
                                'date_end': str(w_end.date()),
                                'date_mid': str(w_mid.date()),
                                'auprc': auprc_w,
                                'n_samples': int(m_w.sum())
                            })
                            # Record the first drop point for T1 but keep scanning
                            if not drop_found and auprc_w < drop_threshold:
                                ver2_end = w_end
                                drop_found = True
                current += pd.Timedelta(days=30)
                
            cutoff = cutoff_end
            result['scan_results'] = scan_results
            result['baseline_auprc'] = baseline_auprc
            result['manual_t0'] = True
            
            segments = {
                't0_date_start': str(cutoff_start.date()),
                't0_date_end': str(cutoff_end.date()),
                't1_date_start': str((cutoff_end + pd.Timedelta(days=1)).date()),
                't1_date_end': str(ver2_end.date()),
                'labels': [],
                'peak_auprc': baseline_auprc,
            }
            result['segments'] = segments

        df_v1_data = df[(df['Session_Date'] >= cutoff_start) & (df['Session_Date'] <= cutoff)].copy()
        df_new_data_all = df[(df['Session_Date'] > cutoff) & (df['Session_Date'] <= ver2_end)].copy()

        result['data_info'] = {
            'total_rows': len(df),
            'model_name': model_name,
            'data_name': data_name,
            'data_start': str(data_start.date()),
            'data_end': str(data_end.date()),
            'cutoff_date': str(cutoff.date()),
            'ver2_end_date': str(ver2_end.date()),
            'v1_rows': len(df_v1_data),
            'new_rows': len(df_new_data_all),
            'ver1_period': f"{cutoff_start.date()} ~ {cutoff.date()}",
            'ver2_period': f"{cutoff_start.date()} ~ {ver2_end.date()}",
            'new_data_period': f"{cutoff.date()} ~ {ver2_end.date()}",
            'scan_mode': scan_mode,
        }

        if 'segments' in result:
            seg = result['segments']
            result['data_info']['t0_period'] = f"{seg['t0_date_start']} ~ {seg['t0_date_end']}"
            result['data_info']['t1_period'] = f"{seg['t1_date_start']} ~ {seg['t1_date_end']}"
            result['data_info']['t0_rows'] = len(df_v1_data)
            result['data_info']['t1_rows'] = len(df_new_data_all)
            result['data_info']['peak_auprc'] = seg['peak_auprc']

        # ========== STEP 1: Patient-level split ==========
        step_offset = 3 if scan_mode else 4
        total_steps = 7 if scan_mode else 7
        _progress(step_offset, total_steps, 'Patient-level split...')

        if 'Patient_ID' not in df.columns:
            result['error'] = '資料中缺少 Patient_ID 欄位。'
            return result

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(
            splitter.split(df, groups=df['Patient_ID'])
        )
        df_train_all = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        df_v1_train = df_train_all[(df_train_all['Session_Date'] >= cutoff_start) & (df_train_all['Session_Date'] <= cutoff)].copy()
        df_new_data = df_train_all[(df_train_all['Session_Date'] > cutoff) & (df_train_all['Session_Date'] <= ver2_end)].copy()
        df_v2_train = df_new_data.copy()  # Train ver2 ONLY on T1 data

        n_train_patients = df_train_all['Patient_ID'].nunique()
        n_test_patients = df_test['Patient_ID'].nunique()

        result['split_info'] = {
            'train_patients': n_train_patients,
            'test_patients': n_test_patients,
            'v1_train_sessions': len(df_v1_train),
            'v2_train_sessions': len(df_v2_train),
            'test_sessions': len(df_test),
            'new_data_sessions': len(df_new_data),
        }

        # ========== STEP 2: Load or train ver1 ==========
        if ver1_model is not None:
            _progress(step_offset + 1, total_steps, '載入 ver1 模型...')
            ver1 = ver1_model
        else:
            _progress(step_offset + 1, total_steps, '訓練 ver1 模型...')
            X1 = df_v1_train[features]
            y1 = df_v1_train[target_col]
            m1 = ~y1.isna()
            ver1 = ExplainableBoostingClassifier(random_state=RANDOM_STATE)
            ver1.fit(X1[m1], y1[m1])

        # Evaluate ver1 on test
        Xt = df_test[features]
        yt = df_test[target_col].values.astype(float)
        v = ~np.isnan(yt)
        Xtv = Xt.iloc[v.nonzero()[0]]
        ytv = yt[v]

        ver1_metrics = _eval_model(ver1, Xt, yt)
        result['ver1_metrics'] = ver1_metrics

        # ========== STEP: Full Retrain ==========
        _progress(step_offset + 2, total_steps, '訓練 ver2 模型...')

        X2 = df_v2_train[features]
        y2 = df_v2_train[target_col]
        m2 = ~y2.isna()
        ver2 = ExplainableBoostingClassifier(random_state=RANDOM_STATE)
        ver2.fit(X2[m2], y2[m2])

        ver2_metrics = _eval_model(ver2, Xt, yt)
        delta_retrain = round(ver2_metrics['AUPRC'] - ver1_metrics['AUPRC'], 4)

        prob_v1 = ver1.predict_proba(Xtv)[:, 1]
        prob_v2 = ver2.predict_proba(Xtv)[:, 1]
        bs_retrain = _bootstrap_ci(ytv, prob_v1, prob_v2)

        result['ver2_metrics'] = ver2_metrics
        result['delta_retrain'] = delta_retrain
        result['bootstrap_retrain'] = bs_retrain

        # ========== STEP: Decision (deferred if scan_mode) ==========
        _progress(step_offset + 3, total_steps, '決策分析中...')

        # In scan mode, decision is based on drift diagnosis (computed below).
        # Store metrics for now; final decision after drift analysis.
        if not scan_mode:
            if bs_retrain['significant'] and delta_retrain > 0:
                decision = 'RETRAIN'
                decision_desc = f'ver2 顯著優於 ver1（Δ={delta_retrain:+.4f}），建議更新為 ver2。'
            elif delta_retrain > 0:
                decision = 'MONITOR'
                decision_desc = f'ver2 微幅改善但不顯著（Δ={delta_retrain:+.4f}），建議持續觀察。'
            else:
                decision = 'INVESTIGATE'
                decision_desc = 'Retrain 未改善表現，需深度調查資料品質或模型架構。'
            result['decision'] = decision
            result['decision_desc'] = decision_desc

        # ========== STEP: Data Drift ==========
        _progress(step_offset + 4, total_steps, 'Drift 分析中...')

        drift_report = {}
        for f in features:
            js = round(_js_div(df_v1_train[f], df_new_data[f]), 4)
            drift_report[f] = js
        avg_js = round(np.mean(list(drift_report.values())), 4)

        result['data_drift'] = {
            'avg_js': avg_js,
            'per_feature': drift_report,
        }

        # Adaptive drift thresholds (bootstrap)
        result['drift_threshold_mode'] = drift_threshold_mode
        if drift_threshold_mode == 'adaptive':
            _progress(7, 9, 'Bootstrap 自適應閥値計算中...')
            result['drift_thresholds'] = _calibrate_js_thresholds(
                df_v1_train, features, n_bootstrap=400, seed=RANDOM_STATE
            )
        else:
            result['drift_thresholds'] = None

        # ========== STEP: Concept Drift ==========
        imp1 = _get_feature_importances(ver1)
        imp2 = _get_feature_importances(ver2)

        # Feature importance table
        max_imp = max(imp1.values()) if imp1 else 1.0
        sorted_feats = sorted(imp1.items(), key=lambda x: -x[1])
        result['feature_importances'] = [
            {'name': name, 'importance': round(val, 4),
             'pct_of_max': round(val / max_imp * 100, 1)}
            for name, val in sorted_feats
        ]

        # Spearman rank correlation
        common = sorted(set(imp1.keys()) & set(imp2.keys()))
        if len(common) >= 3:
            r1 = [imp1[f] for f in common]
            r2 = [imp2[f] for f in common]
            rho, pval = spearmanr(r1, r2)
            rank_corr = {
                'rho': round(float(rho), 4),
                'p_value': round(float(pval), 6),
            }
        else:
            rank_corr = {'rho': None, 'p_value': None}

        # Top-5 comparison
        top5_v1 = [f for f, _ in sorted(imp1.items(), key=lambda x: -x[1])[:5]]
        top5_v2 = [f for f, _ in sorted(imp2.items(), key=lambda x: -x[1])[:5]]
        overlap = len(set(top5_v1) & set(top5_v2))

        # Concept stability judgment
        if rank_corr['rho'] is not None:
            if rank_corr['rho'] > 0.8:
                concept_status = 'stable'
                concept_desc = 'Concept 穩定：特徵重要性結構相似'
            elif rank_corr['rho'] > 0.5:
                concept_status = 'shifting'
                concept_desc = 'Concept 正在偏移：重要性結構有中度變化'
            else:
                concept_status = 'changed'
                concept_desc = 'Concept 已改變：重要性結構顯著變化'
        else:
            concept_status = 'unknown'
            concept_desc = '無法計算（共用特徵不足）'

        result['concept_drift'] = {
            'rank_correlation': rank_corr,
            'top5_v1': top5_v1,
            'top5_v2': top5_v2,
            'top5_overlap': overlap,
            'status': concept_status,
            'description': concept_desc,
        }

        # Shape function data for comparison charts
        result['shape_functions'] = _extract_shape_functions(ver1, ver2, features)

        # ========== Scan Mode Decision (based on drift diagnosis) ==========
        if scan_mode:
            data_drift_high = avg_js > 0.05
            concept_changed = (rank_corr.get('rho') is not None
                               and rank_corr['rho'] < 0.8)

            if data_drift_high and concept_changed:
                decision = 'RETRAIN'
                decision_desc = (
                    f'Data Drift 顯著（平均 JS={avg_js:.4f}）且 Concept 已偏移'
                    f'（Spearman ρ={rank_corr["rho"]:.4f}），'
                    f'資料分佈與特徵重要性結構均發生變化，建議以新資料重新訓練模型。'
                )
            elif data_drift_high and not concept_changed:
                decision = 'RECALIBRATE'
                decision_desc = (
                    f'Data Drift 顯著（平均 JS={avg_js:.4f}）但 Concept 穩定'
                    f'（Spearman ρ={rank_corr["rho"]:.4f}），'
                    f'特徵與目標的關係未改變，表現下降可能來自資料分佈偏移。'
                    f'建議校正輸入資料或增量更新模型。'
                )
            elif not data_drift_high and concept_changed:
                decision = 'INVESTIGATE'
                decision_desc = (
                    f'Data Drift 不明顯（平均 JS={avg_js:.4f}）但 Concept 已偏移'
                    f'（Spearman ρ={rank_corr["rho"]:.4f}），'
                    f'資料分佈沒變但特徵與目標的關係改變了，'
                    f'需調查 Target 定義或標註品質是否有變動。'
                )
            else:
                decision = 'STABLE'
                decision_desc = (
                    f'Data Drift 低（平均 JS={avg_js:.4f}）且 Concept 穩定'
                    f'（Spearman ρ={rank_corr["rho"]:.4f}），'
                    f'模型表現下降可能為隨機波動或外部因素，'
                    f'建議持續監控。'
                )

            result['decision'] = decision
            result['decision_desc'] = decision_desc

        # ========== DONE ==========
        _progress(total_steps, total_steps, '完成！')

        result['success'] = True
        result['elapsed_s'] = round(time.time() - t0, 1)

        # Store models for potential download
        result['_ver1'] = ver1
        result['_ver2'] = ver2

        return result

    except Exception as e:
        logger.exception(f"[retrain] Pipeline failed: {e}")
        result['error'] = str(e)
        result['elapsed_s'] = round(time.time() - t0, 1)
        return result


# ==========================================
# Shape Function Extraction
# ==========================================

def _extract_shape_functions(ver1, ver2, features, max_terms=8):
    """Extract shape function data for top features (skip interactions)."""
    imp1 = ver1.term_importances()
    terms = ver1.term_names_

    sorted_idx = np.argsort(imp1)[::-1]
    shape_data = []
    count = 0

    for idx in sorted_idx:
        if count >= max_terms:
            break
        term_name = terms[idx]

        # Skip interaction terms
        if ' x ' in term_name:
            continue

        # Find in ver2
        if term_name not in ver2.term_names_:
            continue
        idx2 = ver2.term_names_.index(term_name)

        try:
            bins1 = ver1.bins_[idx][0]
            scores1 = ver1.term_scores_[idx]
            bins2 = ver2.bins_[idx2][0]
            scores2 = ver2.term_scores_[idx2]

            # Handle categorical vs continuous
            if isinstance(bins1, dict):
                # Categorical — skip for now
                continue

            bins1 = np.array(bins1, dtype=float)
            scores1 = np.array(scores1, dtype=float)
            bins2 = np.array(bins2, dtype=float)
            scores2 = np.array(scores2, dtype=float)

            n1 = min(len(bins1), len(scores1))
            n2 = min(len(bins2), len(scores2))

            # Compute Spearman correlation between shape functions
            # Interpolate both to common bins for fair comparison
            from scipy.stats import spearmanr as _spearmanr
            common_bins = np.union1d(bins1[:n1], bins2[:n2])
            interp1 = np.interp(common_bins, bins1[:n1], scores1[:n1])
            interp2 = np.interp(common_bins, bins2[:n2], scores2[:n2])
            if len(common_bins) >= 3:
                rho, p_val = _spearmanr(interp1, interp2)
            else:
                rho, p_val = float('nan'), float('nan')

            shape_data.append({
                'name': term_name,
                'importance_v1': float(imp1[idx]),
                'bins_v1': bins1[:n1].tolist(),
                'scores_v1': scores1[:n1].tolist(),
                'bins_v2': bins2[:n2].tolist(),
                'scores_v2': scores2[:n2].tolist(),
                'spearman_rho': round(float(rho), 4) if not np.isnan(rho) else None,
                'spearman_p': round(float(p_val), 6) if not np.isnan(p_val) else None,
            })
            count += 1

        except Exception as e:
            logger.warning(f"[shape] Failed for {term_name}: {e}")
            continue

    return shape_data


# ==========================================
# Chart Builders
# ==========================================

def build_pruning_chart(pruning_results, ver1_auprc):
    """Build a Plotly bar chart showing pruning sweep results."""
    if not pruning_results:
        return None

    labels = [f"<{r['threshold_pct']}% max\n({r['n_pruned']} terms)" for r in pruning_results]
    auprcs = [r['auprc'] for r in pruning_results]
    deltas = [r['delta'] for r in pruning_results]
    colors = ['#10b981' if d > 0 else '#ef4444' for d in deltas]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels, y=auprcs,
        marker_color=colors,
        text=[f"{a:.4f}\n({d:+.4f})" for a, d in zip(auprcs, deltas)],
        textposition='outside',
        name='Pruned AUPRC',
    ))

    fig.add_hline(
        y=ver1_auprc, line_dash='dash', line_color='#6366f1',
        annotation_text=f'ver1 baseline: {ver1_auprc:.4f}',
        annotation_position='top left',
    )

    fig.update_layout(
        title='Feature Pruning 閾值掃描',
        xaxis_title='Pruning 閾值',
        yaxis_title='AUPRC',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
    )

    return fig


def _calibrate_js_thresholds(df_train, features, n_bootstrap=400, seed=42):
    """
    Bootstrap calibration of per-feature JS Divergence thresholds.
    Splits df_train randomly n times and measures JS between halves.
    Returns p95 (warning) and p99 (critical) per feature.
    """
    rng = np.random.RandomState(seed)
    n = len(df_train)
    null_js = {f: [] for f in features}

    for _ in range(n_bootstrap):
        idx = rng.permutation(n)
        half = n // 2
        df_a = df_train.iloc[idx[:half]]
        df_b = df_train.iloc[idx[half:]]
        for f in features:
            if f in df_a.columns and f in df_b.columns:
                null_js[f].append(_js_div(df_a[f], df_b[f]))

    thresholds = {}
    for f in features:
        vals = np.array(null_js[f])
        if len(vals) > 0:
            thresholds[f] = {
                'warning': round(float(np.percentile(vals, 95)), 4),
                'critical': round(float(np.percentile(vals, 99)), 4),
            }
        else:
            thresholds[f] = {'warning': 0.05, 'critical': 0.10}

    # Also compute global (across all features)
    all_vals = [v for vlist in null_js.values() for v in vlist]
    thresholds['__global__'] = {
        'warning': round(float(np.percentile(all_vals, 95)), 4) if all_vals else 0.05,
        'critical': round(float(np.percentile(all_vals, 99)), 4) if all_vals else 0.10,
    }
    logger.info(f"[bootstrap] global JS p95={thresholds['__global__']['warning']:.4f} "
                f"p99={thresholds['__global__']['critical']:.4f}")
    return thresholds


def build_drift_chart(drift_report, adaptive_thresholds=None, threshold_mode='fixed'):
    """Build a horizontal bar chart for per-feature JS Divergence.
    
    Args:
        drift_report: dict of {feature: js_value}
        adaptive_thresholds: dict from _calibrate_js_thresholds (or None for fixed)
        threshold_mode: 'fixed' or 'adaptive'
    """
    if not drift_report:
        return None

    sorted_items = sorted(drift_report.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    # Determine global thresholds for bar colouring
    if threshold_mode == 'adaptive' and adaptive_thresholds:
        glb = adaptive_thresholds.get('__global__', {})
        warn_g = glb.get('warning', 0.05)
        crit_g = glb.get('critical', 0.10)
    else:
        warn_g, crit_g = 0.05, 0.10

    colors = ['#ef4444' if v > crit_g else '#f59e0b' if v > warn_g else '#10b981' for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names, x=values,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in values],
        textposition='outside',
    ))

    if threshold_mode == 'adaptive' and adaptive_thresholds:
        # Per-feature adaptive warning lines
        warn_vals = [adaptive_thresholds.get(n, {}).get('warning', warn_g) for n in names]
        crit_vals = [adaptive_thresholds.get(n, {}).get('critical', crit_g) for n in names]
        glb = adaptive_thresholds.get('__global__', {})
        warn_g = glb.get('warning', 0.05)
        crit_g = glb.get('critical', 0.10)
        # Draw global lines with annotation
        fig.add_vline(
            x=warn_g, line_dash='dash', line_color='#f59e0b',
            annotation_text=f'Warning P95 ({warn_g:.3f})',
        )
        fig.add_vline(
            x=crit_g, line_dash='dash', line_color='#ef4444',
            annotation_text=f'Critical P99 ({crit_g:.3f})',
        )
        title_suffix = ' 《自適應閥値 Bootstrap P95/P99》'
    else:
        fig.add_vline(
            x=0.05, line_dash='dash', line_color='#f59e0b',
            annotation_text='Warning (0.05)',
        )
        fig.add_vline(
            x=0.10, line_dash='dash', line_color='#ef4444',
            annotation_text='Alert (0.10)',
        )
        title_suffix = ' 《固定閥値》'

    fig.update_layout(
        title=f'Data Drift — JS Divergence per Feature{title_suffix}',
        xaxis_title='JS Divergence',
        height=max(300, len(names) * 35 + 100),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
        yaxis=dict(autorange='reversed'),
    )

    return fig


def build_shape_comparison_figures(shape_functions):
    """Build Plotly figures comparing ver1 vs ver2 shape functions."""
    if not shape_functions:
        return []

    figures = []

    for sf in shape_functions:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sf['bins_v1'], y=sf['scores_v1'],
            mode='lines',
            name='ver1 (舊模型)',
            line=dict(color='#6366f1', width=2.5),
        ))

        fig.add_trace(go.Scatter(
            x=sf['bins_v2'], y=sf['scores_v2'],
            mode='lines',
            name='ver2 (新模型)',
            line=dict(color='#f59e0b', width=2.5),
        ))

        fig.add_hline(y=0, line_dash='dot', line_color='rgba(148,163,184,0.3)')

        fig.update_layout(
            title=f'Shape Function: {sf["name"]}',
            xaxis_title=sf['name'],
            yaxis_title='Log-Odds 貢獻',
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
            yaxis=dict(gridcolor='rgba(99,102,241,0.08)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(t=50, b=40, l=50, r=20),
        )

        figures.append({
            'name': sf['name'],
            'figure': fig,
        })

    return figures


def build_timeline_chart(data_info):
    """Build a simple Plotly line chart representing data periods."""
    if not data_info:
        return None

    data_start = data_info.get('data_start')
    cutoff_date = data_info.get('cutoff_date')
    ver2_end_date = data_info.get('ver2_end_date') or data_info.get('data_end')
    
    if not (data_start and ver2_end_date and cutoff_date):
        return None

    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()

        # Line 1: ver1 Training
        fig.add_trace(go.Scatter(
            x=[data_start, cutoff_date],
            y=['ver1 訓練集', 'ver1 訓練集'],
            mode='lines+markers+text',
            name='ver1',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=8, color='#6366f1'),
            text=[f" {data_start}", f"{cutoff_date} "],
            textposition=['top right', 'top left'],
            textfont=dict(color='#94a3b8', size=11)
        ))

        # Line 2: ver2 Training
        fig.add_trace(go.Scatter(
            x=[data_start, ver2_end_date],
            y=['ver2 訓練集', 'ver2 訓練集'],
            mode='lines+markers+text',
            name='ver2',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=8, color='#f59e0b'),
            text=[f" {data_start}", f"{ver2_end_date} "],
            textposition=['bottom right', 'bottom left'],
            textfont=dict(color='#94a3b8', size=11)
        ))

        # Line 3: New Data
        fig.add_trace(go.Scatter(
            x=[cutoff_date, ver2_end_date],
            y=['新資料集', '新資料集'],
            mode='lines+markers+text',
            name='New Data',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8, color='#10b981'),
            text=[f" {cutoff_date}", f"{ver2_end_date} "],
            textposition=['top right', 'top left'],
            textfont=dict(color='#94a3b8', size=11)
        ))

        fig.update_layout(
            title='資料切分時間軸 (Data Split Timeline)',
            height=200,
            margin=dict(t=40, b=20, l=100, r=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, autorange='reversed'),
            showlegend=False,
            hovermode='closest'
        )
        return fig
    except Exception as e:
        logger.error(f"[timeline] Failed to build timeline chart: {e}")
        return None
