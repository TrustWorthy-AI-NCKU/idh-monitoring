"""Quick integration test for moe_service.py"""
import sys
sys.path.insert(0, r'c:\Users\Q56144018\Documents\IDH\code\idh_monitoring')

import pandas as pd
import numpy as np

# Import the service
from monitoring.moe_service import MoEModel

# Paths
BASE = r'c:\Users\Q56144018\Documents\IDH\code\web_model_performance'
MOE = f'{BASE}/optuna_moe/expert_models'

expert_paths = {
    'TN_1': f'{MOE}/TN_2026_01_07_11_03_1.Rate_only_rs2024_ebm_final.joblib',
    'CY_1': f'{MOE}/CY_2026_01_07_12_47_1.Rate_only_rs8888_ebm_final.joblib',
    'TNCY_cas2': f'{MOE}/TNCY_cas2_EBM.joblib',
}
meta_path = f'{MOE}/moe_gb_meta_learner.joblib'

# Load MoE
print("Loading MoE model...")
moe = MoEModel(expert_paths, meta_path)

# Load a small subset of test data
print("Loading test data (first 500 rows)...")
df = pd.read_csv(f'{BASE}/D6_test_mice.csv', nrows=500, low_memory=False)

# Prepare features (same as dataloading.py training_mode=0)
feature_cols = [
    'Session_Date', 'Sex', 'Age', 'Pre_HD_SBP', 'Start_DBP',
    'Heart_Rate', 'Respiratory_Rate', 'Blood_Flow_Rate',
    'Dialysate_Temperature', 'Dialysate_Flow_Rate', 'Pre_HD_Weight',
    'UF_BW_Perc', 'Body_Temperature', 'Dry_Weight', 'Target_UF_Volume',
    'IDH_N_28D', 'IDH_N_7D'
]

# Map target and Sex first
C1_map = {1: 1.0, 0: 0.0, True: 1.0, False: 0.0}
sex_map = {1: 1, 0: 0, '男': 1, '女': 0}

if 'Nadir90/100' in df.columns:
    df['Nadir90/100'] = df['Nadir90/100'].map(C1_map)
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map(sex_map)

X = df[feature_cols].copy()
y = df['Nadir90/100'].copy()

# Clean numeric issues
X.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in X.columns:
    if X[col].dtype in ['float64', 'int64', 'float32']:
        X[col].fillna(0, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)
y.fillna(0, inplace=True)

# Evaluate
print("Running evaluation...")
metrics, probs, expert_probs = moe.evaluate(X, y)

print("\n=== MoE Results (500 samples) ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

print(f"\n  Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
print(f"  Prob mean:  {probs.mean():.4f}")
print("\nDone!")
