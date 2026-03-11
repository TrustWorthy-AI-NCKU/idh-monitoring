"""
Dash application for the Model Monitoring Dashboard.
Uses django-plotly-dash's DjangoDash instead of standard Dash.
"""
import logging
from dash import dcc, html, Input, Output
from django_plotly_dash import DjangoDash
from django.core.cache import cache

from .services import build_dashboard_figure

logger = logging.getLogger(__name__)

CACHE_KEY = 'monitoring_data_default'

# Create the DjangoDash application
app = DjangoDash(
    'monitoring-dashboard',
    add_bootstrap_links=True,
    suppress_callback_exceptions=True,
    serve_locally=True
)

# Layout
app.layout = html.Div([
    # Controls
    html.Div([
        html.Label('選擇要顯示的指標：', style={
            'fontWeight': '600',
            'fontSize': '14px',
            'color': '#334155',
            'marginBottom': '8px',
            'display': 'block',
        }),
        dcc.Checklist(
            id='metrics-checklist',
            options=[
                {'label': ' AUPRC', 'value': 'AUPRC'},
                {'label': ' AUROC', 'value': 'AUROC'},
                {'label': ' F1 Score', 'value': 'F1 score'},
                {'label': ' JS Divergence', 'value': 'JS divergence'},
            ],
            value=['AUPRC', 'JS divergence'],
            inline=True,
            style={'marginBottom': '16px'},
            inputStyle={'marginRight': '6px'},
            labelStyle={
                'marginRight': '20px',
                'fontSize': '14px',
                'color': '#475569',
                'cursor': 'pointer',
            },
        ),
    ], style={
        'padding': '16px 20px',
        'backgroundColor': '#f8fafc',
        'borderRadius': '8px',
        'border': '1px solid #e2e8f0',
        'marginBottom': '16px',
    }),

    # Status message
    html.Div(id='status-message', style={
        'padding': '12px 16px',
        'borderRadius': '8px',
        'marginBottom': '16px',
        'fontSize': '14px',
    }),

    # Loading wrapper + Graph
    dcc.Loading(
        id='loading-graph',
        type='circle',
        children=[
            dcc.Graph(
                id='dashboard-graph',
                style={'height': '800px'},
                config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'responsive': True,
                }
            ),
        ],
    ),
], style={'width': '100%'})


@app.callback(
    [Output('dashboard-graph', 'figure'),
     Output('status-message', 'children'),
     Output('status-message', 'style')],
    [Input('metrics-checklist', 'value')],
)
def update_dashboard(metrics_list):
    """Update the dashboard when metrics selection changes."""
    base_style = {
        'padding': '12px 16px',
        'borderRadius': '8px',
        'marginBottom': '16px',
        'fontSize': '14px',
    }

    # Retrieve processed data from cache
    try:
        cached_data = cache.get(CACHE_KEY)
        logger.info(f"[Dash Callback] Cache key={CACHE_KEY}, has_data={cached_data is not None}")
    except Exception as e:
        logger.error(f"[Dash Callback] Cache read error: {e}")
        cached_data = None

    if not cached_data:
        empty_fig = {
            'data': [],
            'layout': {
                'title': '請先上傳資料並點擊「開始分析」',
                'height': 800,
                'paper_bgcolor': '#fafafa',
                'plot_bgcolor': '#ffffff',
                'annotations': [{
                    'text': '⬅ 請在左側上傳 CSV 和模型檔案',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 20, 'color': '#94a3b8'},
                }]
            }
        }
        return (
            empty_fig,
            '尚未上傳資料，或快取已過期。請重新上傳檔案並點擊「開始分析」。',
            {**base_style, 'backgroundColor': '#fef3c7', 'color': '#92400e', 'border': '1px solid #fcd34d'},
        )

    # Build figure
    try:
        logger.info(f"[Dash Callback] Building figure with {len(cached_data['windows'])} windows, metrics={metrics_list}")
        fig, msg = build_dashboard_figure(
            windows=cached_data['windows'],
            baseline_data=cached_data['baseline'],
            dates=cached_data['dates'],
            model=cached_data['model'],
            final_feats=cached_data['features'],
            target_col=cached_data['target_col'],
            metrics_list=metrics_list,
        )
        logger.info(f"[Dash Callback] Figure built successfully: {msg}")
    except Exception as e:
        logger.error(f"[Dash Callback] build_dashboard_figure error: {e}", exc_info=True)
        error_fig = {
            'data': [],
            'layout': {
                'title': f'繪圖錯誤: {str(e)}',
                'height': 800,
            }
        }
        return (
            error_fig,
            f'繪圖錯誤: {str(e)}',
            {**base_style, 'backgroundColor': '#fee2e2', 'color': '#991b1b', 'border': '1px solid #fca5a5'},
        )

    return (
        fig,
        msg,
        {**base_style, 'backgroundColor': '#d1fae5', 'color': '#065f46', 'border': '1px solid #6ee7b7'},
    )
