import os
import sys
import django

# Setup Django
sys.path.append('c:/Users/Q56144018/Documents/IDH/code/idh_monitoring')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'idh_monitoring.settings')
django.setup()

from monitoring.retrain_service import run_retrain_pipeline

csv_path = 'c:/Users/Q56144018/Documents/IDH/code/web_model_performance/TN_train_mice.csv'
features = [
    'Sex', 'Age', 'Pre_HD_SBP', 'Start_DBP', 'Heart_Rate', 'Respiratory_Rate', 
    'Blood_Flow_Rate', 'Dialysate_Temperature', 'Dialysate_Flow_Rate', 
    'Pre_HD_Weight', 'UF_BW_Perc', 'Body_Temperature', 'Dry_Weight', 
    'Target_UF_Volume', 'IDH_N_28D', 'IDH_N_7D'
]
target_col = 'Nadir90/100'

def _cb(step, total, msg):
    print(f"[{step}/{total}] {msg}")

print("Running pipeline in SCAN MODE...")
res = run_retrain_pipeline(
    csv_path=csv_path,
    features=features,
    target_col=target_col,
    ver1_model=None, # will train from scratch
    scan_mode=True,
    progress_callback=_cb
)

if res['success']:
    print(f"\nSUCCESS! Elapsed: {res['elapsed_s']}s")
    print("\nData Info:", res['data_info'])
    print("\nDecision:", res['decision'], "-", res['decision_desc'])
    if 'segments' in res:
        seg = res['segments']
        print(f"\nT0: {seg['t0_date_start']} ~ {seg['t0_date_end']} ({seg['t0_windows']} windows)")
        print(f"T1: {seg['t1_date_start']} ~ {seg['t1_date_end']} ({seg['t1_windows']} windows)")
        print(f"Peak AUPRC: {seg['peak_auprc']:.4f}")
else:
    print("\nFAILED:", res['error'])
