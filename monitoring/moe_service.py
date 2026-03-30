"""
MoE Service: 載入 MoE 模型（Experts + GB Meta-Learner），執行推論並計算指標。

架構:
  Final = w_gb × GB_Meta(enhanced_features) + w_expert × (w_tn×TN + w_cy×CY + w_cas2×(1-TNCY))

使用方式:
  moe = MoEModel(expert_paths, meta_learner_path)
  probs = moe.predict_proba(X)
  metrics = moe.evaluate(X, y)
"""
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, auc, confusion_matrix,
)

logger = logging.getLogger(__name__)


def create_enhanced_features(X, probs_tn, probs_cy, probs_tncy,
                             sbp_threshold=115, risk_params=None):
    """
    Create advanced meta-features for MoE routing.
    Mirrors create_advanced_features_with_sbp from model_utils.py.
    """
    df = X.copy()

    # Expert predictions
    df['prob_tn'] = probs_tn
    df['prob_cy'] = probs_cy
    df['prob_tncy'] = probs_tncy

    # Risk indicators
    if risk_params is None:
        risk_params = {}

    idh28_thresh = risk_params.get('idh28_thresh', 2)
    uf_thresh = risk_params.get('uf_thresh', 0.04)

    df['high_idh_history'] = (df.get('IDH_N_28D', 0) > idh28_thresh).astype(float)
    df['recent_idh'] = (df.get('IDH_N_7D', 0) > 0).astype(float)
    df['high_uf'] = (df.get('UF_BW_Perc', 0) > uf_thresh).astype(float)
    df['low_pre_sbp'] = (df.get('Pre_HD_SBP', 999) < sbp_threshold).astype(float)

    # Risk score
    w_idh28 = risk_params.get('w_idh28', 0.3)
    w_idh7 = risk_params.get('w_idh7', 0.3)
    w_uf = risk_params.get('w_uf', 0.2)
    w_sbp = risk_params.get('w_sbp', 0.2)

    norm_idh28 = risk_params.get('norm_idh28', 12)
    norm_idh7 = risk_params.get('norm_idh7', 3)
    norm_uf = risk_params.get('norm_uf', 0.1)

    sbp_values = df.get('Pre_HD_SBP', 140)
    sbp_risk = np.clip((180 - sbp_values) / 90, 0, 1)

    df['risk_score'] = (
        np.clip(df.get('IDH_N_28D', 0) / norm_idh28, 0, 1) * w_idh28 +
        np.clip(df.get('IDH_N_7D', 0) / norm_idh7, 0, 1) * w_idh7 +
        np.clip(df.get('UF_BW_Perc', 0) / norm_uf, 0, 1) * w_uf +
        sbp_risk * w_sbp
    )

    # Expert agreement
    df['max_tn_cy'] = np.maximum(probs_tn, probs_cy)
    df['min_tn_cy'] = np.minimum(probs_tn, probs_cy)
    df['mean_tn_cy'] = (probs_tn + probs_cy) / 2
    df['diff_tn_cy'] = np.abs(probs_tn - probs_cy)

    # TNCY signal
    df['inv_tncy'] = 1 - probs_tncy
    df['tncy_high'] = (probs_tncy > 0.5).astype(float)
    df['tncy_very_high'] = (probs_tncy > 0.7).astype(float)

    # Confidence
    df['tn_uncertain'] = ((probs_tn > 0.3) & (probs_tn < 0.7)).astype(float)
    df['cy_uncertain'] = ((probs_cy > 0.3) & (probs_cy < 0.7)).astype(float)
    df['both_uncertain'] = df['tn_uncertain'] * df['cy_uncertain']

    # Ensemble stats
    df['pred_std'] = np.column_stack([probs_tn, probs_cy, probs_tncy]).std(axis=1)
    df['pred_max'] = np.column_stack([probs_tn, probs_cy, probs_tncy]).max(axis=1)
    df['pred_min'] = np.column_stack([probs_tn, probs_cy, probs_tncy]).min(axis=1)

    # Interactions
    df['risk_x_tncy'] = df['risk_score'] * probs_tncy
    df['high_risk_tncy'] = df['high_idh_history'] * df['tncy_high']

    return df


class MoEModel:
    """
    MoE 推論模型。載入 3 個 Expert + GB Meta-Learner，執行融合推論。
    """

    def __init__(self, expert_paths, meta_learner_path):
        """
        Args:
            expert_paths: dict like {'TN_1': path, 'CY_1': path, 'TNCY_cas2': path}
            meta_learner_path: path to moe_gb_meta_learner.joblib
        """
        self.experts = {}
        self.expert_names = ['TN_1', 'CY_1', 'TNCY_cas2']

        # Load experts
        for name, path in expert_paths.items():
            self.experts[name] = joblib.load(path)
            logger.info(f"[MoE] Loaded expert: {name}")

        # Load meta-learner bundle
        bundle = joblib.load(meta_learner_path)
        self.gb = bundle['gb']
        self.scaler = bundle['scaler']
        self.feature_cols = bundle['feature_cols']
        self.config = bundle['config']

        # Extract weights from config
        self.w_gb = self.config['w_gb']
        self.w_expert = self.config['w_expert']
        self.expert_weights = self.config['expert_weights']
        self.risk_params = self.config.get('risk_params', {})
        self.sbp_threshold = self.config.get('sbp_threshold', 115)

        logger.info(f"[MoE] Loaded GB Meta-Learner "
                    f"(w_gb={self.w_gb}, w_expert={self.w_expert}, "
                    f"{len(self.feature_cols)} features)")

    def predict_proba(self, X):
        """
        執行 MoE 推論，回傳最終機率值。

        Args:
            X: DataFrame with features (must include Session_Date + 16 features)

        Returns:
            final_probs: np.array of probabilities
            expert_probs: dict of {name: np.array} for each expert
        """
        X = X.copy()

        # Convert Session_Date to numeric
        if 'Session_Date' in X.columns:
            X['Session_Date'] = pd.to_datetime(
                X['Session_Date']).astype('int64') // 10**9

        # Get expert predictions
        expert_probs = {}
        for name, model in self.experts.items():
            expert_probs[name] = model.predict_proba(X)[:, 1]

        # Create enhanced features
        X_enhanced = create_enhanced_features(
            X, expert_probs['TN_1'], expert_probs['CY_1'],
            expert_probs['TNCY_cas2'],
            sbp_threshold=self.sbp_threshold,
            risk_params=self.risk_params,
        )

        # Prepare for GB
        X_meta = X_enhanced[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X_meta)
        gb_probs = self.gb.predict_proba(X_scaled)[:, 1]

        # Expert ensemble (fixed weights)
        w = self.expert_weights
        expert_ensemble = (
            w['TN_1'] * expert_probs['TN_1'] +
            w['CY_1'] * expert_probs['CY_1'] +
            w['inv_TNCY'] * (1 - expert_probs['TNCY_cas2'])
        )

        # Final fusion
        final_probs = self.w_gb * gb_probs + self.w_expert * expert_ensemble

        return final_probs, expert_probs

    def evaluate(self, X, y_true, threshold=0.5):
        """
        執行推論並計算所有監控指標。

        Args:
            X: Feature DataFrame
            y_true: Ground truth labels (0/1)
            threshold: Classification threshold

        Returns:
            dict with keys: auprc, auroc, f1, sensitivity, fpr,
                            precision, recall, accuracy, correct, incorrect,
                            tp, tn, fp, fn, total
        """
        y_true = np.array(y_true).astype(float)
        final_probs, expert_probs = self.predict_proba(X)
        y_pred = (final_probs >= threshold).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total = len(y_true)

        # Metrics
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, final_probs)
        auprc = auc(recall_arr, precision_arr)

        metrics = {
            'auprc': round(auprc, 4),
            'auroc': round(roc_auc_score(y_true, final_probs), 4),
            'f1': round(f1_score(y_true, y_pred, pos_label=1.0), 4),
            'sensitivity': round(tp / (tp + fn + 1e-9), 4),
            'fpr': round(fp / (fp + tn + 1e-9), 4),
            'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'accuracy': round((tp + tn) / total * 100, 2),
            'total': total,
            'correct': int(tp + tn),
            'incorrect': int(fp + fn),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        }

        return metrics, final_probs, expert_probs


def load_moe_from_uploads(expert_files, meta_learner_file):
    """
    從上傳的檔案路徑載入 MoE 模型（供 views.py 呼叫）。

    Args:
        expert_files: list of (name, path) tuples,
                      e.g. [('TN_1', '/tmp/tn.joblib'), ...]
        meta_learner_file: path to moe_gb_meta_learner.joblib

    Returns:
        MoEModel instance
    """
    expert_paths = {name: path for name, path in expert_files}
    return MoEModel(expert_paths, meta_learner_file)


class MoESklearnWrapper:
    """
    Wraps MoEModel so it looks like a sklearn classifier.
    Provides predict() and predict_proba() compatible with services.py.
    """

    def __init__(self, moe_model: MoEModel):
        self.moe = moe_model
        # Expose feature_names_in_ so load_and_process can auto-detect features
        self.feature_names_in_ = []

    def predict_proba(self, X):
        """Returns (N, 2) array like sklearn, column 0 = neg, column 1 = pos."""
        final_probs, _ = self.moe.predict_proba(X)
        return np.column_stack([1 - final_probs, final_probs])

    def predict(self, X, threshold=0.5):
        """Returns binary predictions (0/1)."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
