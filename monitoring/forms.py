"""
Django forms for the monitoring app.
"""
from django import forms


class MonitoringForm(forms.Form):
    """Form for uploading monitoring data files."""
    
    data_file = forms.FileField(
        label='Dataset (CSV)',
        help_text='上傳要監控的資料集 (.csv)',
        widget=forms.ClearableFileInput(attrs={
            'accept': '.csv',
            'class': 'form-control',
        })
    )
    
    train_file = forms.FileField(
        label='Baseline Data (CSV)',
        required=False,
        help_text='上傳訓練資料作為 JS Divergence 的 baseline。'
                  '若未上傳，將以第一個時間窗口作為 baseline，'
                  '可能影響特徵偏移 (Feature Drift) 檢測的準確性。',
        widget=forms.ClearableFileInput(attrs={
            'accept': '.csv',
            'class': 'form-control',
        })
    )
    
    model_file = forms.FileField(
        label='Model (Joblib)',
        help_text='上傳模型檔案 (.joblib)',
        widget=forms.ClearableFileInput(attrs={
            'accept': '.joblib',
            'class': 'form-control',
        })
    )
    
    features = forms.CharField(
        label='Features',
        initial='Sex, Age, Pre_HD_SBP, Start_DBP, Heart_Rate, Respiratory_Rate, '
                'Blood_Flow_Rate, Dialysate_Temperature, Dialysate_Flow_Rate, '
                'Pre_HD_Weight, UF_BW_Perc, Body_Temperature, Dry_Weight, '
                'Target_UF_Volume, IDH_N_28D, IDH_N_7D',
        widget=forms.Textarea(attrs={
            'rows': 3,
            'class': 'form-control',
            'placeholder': '以逗號分隔的特徵名稱',
        })
    )
    
    target_col = forms.CharField(
        label='Target Column',
        initial='Nadir90/100',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
        })
    )

    trend_window = forms.IntegerField(
        label='分析窗格數',
        initial=5,
        min_value=2,
        max_value=50,
        help_text='取最近 N 個時間窗口進行趨勢分析（預設 5）',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
        })
    )

    alert_mode = forms.ChoiceField(
        label='alert模式',
        choices=[
            ('strict', '一般監控模式 (嚴格 - 及早預警)'),
            ('loose', '異常診斷模式 (寬鬆 - 減少干擾)'),
        ],
        initial='strict',
        help_text='嚴格模式適用於日常監控；若是已知模型異常，可切換至寬鬆模式以專注於最嚴重的偏差。',
        widget=forms.Select(attrs={
            'class': 'form-control',
        })
    )
