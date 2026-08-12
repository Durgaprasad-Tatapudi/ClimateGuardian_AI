import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, auc, roc_curve, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
import shap
from scipy.stats import chi2 as chi2_dist

# Set aesthetic styling for IEEE quality
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

run_id = os.environ.get("EXPERIMENT_RUN_ID", "latest")
models_dir = Path(f"results/experiment_runs/{run_id}/models")
results_dir = Path(f"results/experiment_runs/{run_id}/metrics")
figures_dir = Path(f"results/experiment_runs/{run_id}/figures")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
print("Loading data...")
features = pd.read_csv("03_Features/master_features.csv")
features['date'] = pd.to_datetime(features['date'])
flood_lbl = pd.read_csv("04_Labels/flood_labels.csv")
flood_lbl['date'] = pd.to_datetime(flood_lbl['date'])
heatwave_lbl = pd.read_csv("04_Labels/heatwave_labels.csv")
heatwave_lbl['date'] = pd.to_datetime(heatwave_lbl['date'])
compound_lbl = pd.read_csv("04_Labels/compound_labels.csv")
compound_lbl['date'] = pd.to_datetime(compound_lbl['date'])

df = features.merge(flood_lbl[['date', 'flood_target']], on='date')
df = df.merge(heatwave_lbl[['date', 'heatwave_target']], on='date')
df = df.merge(compound_lbl[['date', 'compound_target']], on='date')
df = df.sort_values('date').reset_index(drop=True)

train_mask = (df['date'].dt.year >= 2000) & (df['date'].dt.year <= 2012)
val_mask = (df['date'].dt.year >= 2013) & (df['date'].dt.year <= 2015)
test_mask = (df['date'].dt.year >= 2016) & (df['date'].dt.year <= 2018)

X = df.drop(columns=['date', 'flood_target', 'heatwave_target', 'compound_target'])
X = X.ffill() 

X_train = X[train_mask].copy()
X_val = X[val_mask].copy()
X_test = X[test_mask].copy()

imputation_means = X_train.mean()
X_train.fillna(imputation_means, inplace=True)
X_val.fillna(imputation_means, inplace=True)
X_test.fillna(imputation_means, inplace=True)
targets = ['flood_target', 'heatwave_target', 'compound_target']

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

def dl_predict(model_path, rnn_type, X_s):
    class RNNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, rnn_type='LSTM'):
            super(RNNModel, self).__init__()
            if rnn_type == 'LSTM':
                self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            else:
                self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)
    
    seq_len = 14
    model = RNNModel(X_s.shape[1], 32, rnn_type)
    if not os.path.exists(model_path): return np.zeros(len(X_s))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    Xs = []
    for i in range(len(X_s) - seq_len):
        Xs.append(X_s[i:(i + seq_len)])
    
    if len(Xs) == 0:
        return np.zeros(len(X_s))

    with torch.no_grad():
        tensor_Xs = torch.tensor(np.array(Xs), dtype=torch.float32)
        logits = model(tensor_Xs).squeeze()
        probs = torch.sigmoid(logits).numpy()
    
    probs = np.pad(probs, (seq_len, 0), mode='constant', constant_values=0)
    return probs

def compute_all_metrics(y_true, probs):
    preds = (probs > 0.5).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
    
    return {
        'Accuracy': accuracy_score(y_true, preds),
        'Precision': precision_score(y_true, preds, zero_division=0),
        'Recall': recall_score(y_true, preds, zero_division=0),
        'F1-Score': f1_score(y_true, preds, zero_division=0),
        'Balanced Accuracy': balanced_accuracy_score(y_true, preds),
        'Specificity': spec,
        'Sensitivity': sens,
        'ROC-AUC': roc_auc_score(y_true, probs) if len(np.unique(y_true))>1 else 0,
        'PR-AUC': auc(recall_curve, precision_curve),
        'MCC': matthews_corrcoef(y_true, preds),
        'Brier Score': brier_score_loss(y_true, probs)
    }

# Ensure base models are available
model_names = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 'LSTM', 'GRU']

print("Ready for evaluation phases.")

# ---------------------------------------------------------
# Leakage-Free Stacking
# ---------------------------------------------------------
def train_leakage_free_stack(target, X_tv, y_tv, X_test):
    print(f"Training Stack for {target}...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Base models to use for stack
    xgb = XGBClassifier(scale_pos_weight=(len(y_tv)-y_tv.sum())/(y_tv.sum()+1e-5), random_state=42, eval_metric='logloss')
    lgbm = LGBMClassifier(scale_pos_weight=(len(y_tv)-y_tv.sum())/(y_tv.sum()+1e-5), random_state=42, verbose=-1)
    
    meta_features_train = np.zeros((len(y_tv), 2))
    
    for train_index, val_index in tscv.split(X_tv):
        X_fold_train, X_fold_val = X_tv[train_index], X_tv[val_index]
        y_fold_train, y_fold_val = y_tv.iloc[train_index], y_tv.iloc[val_index]
        
        xgb.fit(X_fold_train, y_fold_train)
        meta_features_train[val_index, 0] = xgb.predict_proba(X_fold_val)[:, 1]
        
        lgbm.fit(X_fold_train, y_fold_train)
        meta_features_train[val_index, 1] = lgbm.predict_proba(X_fold_val)[:, 1]
        
    # Drop rows where meta_features are 0 (the first fold train set)
    valid_idx = np.where(meta_features_train.sum(axis=1) > 0)[0]
    
    meta_learner = LogisticRegression(class_weight='balanced')
    if len(np.unique(y_tv.iloc[valid_idx])) > 1:
        meta_learner.fit(meta_features_train[valid_idx], y_tv.iloc[valid_idx])
    else:
        meta_learner.fit(meta_features_train, y_tv) # Fallback
        
    # Retrain base models on full Train+Val for final test predictions
    xgb.fit(X_tv, y_tv)
    lgbm.fit(X_tv, y_tv)
    
    meta_features_test = np.column_stack([
        xgb.predict_proba(X_test)[:, 1],
        lgbm.predict_proba(X_test)[:, 1]
    ])
    
    stack_probs = meta_learner.predict_proba(meta_features_test)[:, 1]
    return stack_probs, meta_learner, xgb, lgbm

# ---------------------------------------------------------
# Calibration
# ---------------------------------------------------------
def evaluate_calibration(target, y_val, probs_val, y_test, probs_test):
    # Uncalibrated
    brier_uncal = brier_score_loss(y_test, probs_test)
    
    # Platt (Sigmoid)
    platt = LogisticRegression()
    platt.fit(probs_val.reshape(-1, 1), y_val)
    probs_platt = platt.predict_proba(probs_test.reshape(-1, 1))[:, 1]
    brier_platt = brier_score_loss(y_test, probs_platt)
    
    # Isotonic
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probs_val, y_val)
    probs_iso = iso.predict(probs_test)
    brier_iso = brier_score_loss(y_test, probs_iso)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    fop_uncal, mpv_uncal = calibration_curve(y_test, probs_test, n_bins=10)
    fop_platt, mpv_platt = calibration_curve(y_test, probs_platt, n_bins=10)
    fop_iso, mpv_iso = calibration_curve(y_test, probs_iso, n_bins=10)
    
    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax.plot(mpv_uncal, fop_uncal, 's-', label=f'Uncalibrated (Brier={brier_uncal:.3f})')
    ax.plot(mpv_platt, fop_platt, 'o-', label=f'Platt (Brier={brier_platt:.3f})')
    ax.plot(mpv_iso, fop_iso, '^-', label=f'Isotonic (Brier={brier_iso:.3f})')
    
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_title(f"Calibration Curves - {target}")
    ax.legend(loc="lower right")
    fig.savefig(figures_dir / f"14_Calibration_{target}.png")
    plt.close(fig)
    
    return [
        {'Target': target, 'Method': 'Uncalibrated', 'Brier': brier_uncal},
        {'Target': target, 'Method': 'Platt Scaling', 'Brier': brier_platt},
        {'Target': target, 'Method': 'Isotonic Regression', 'Brier': brier_iso}
    ]

# ---------------------------------------------------------
# Lead-Time Evaluation
# ---------------------------------------------------------
def evaluate_lead_time(target, base_model, df, X_cols_t, scaler, col_indices):
    horizons = [1, 3, 5, 7]
    lead_time_results = []
    
    for h in horizons:
        # Shift target BACKWARDS to align features at t with target at t+h
        # wait, if we want to predict t+h, target series needs to be shifted by -h
        shifted_target = df[target].shift(-h)
        mask = df['date'].dt.year >= 2016
        mask &= df['date'].dt.year <= 2018
        mask &= shifted_target.notna()
        
        y_test_h = shifted_target[mask]
        X_test_h = df.loc[mask, X_cols_t]
        X_test_h_s = scaler.transform(df.loc[mask, df.drop(columns=['date', 'flood_target', 'heatwave_target', 'compound_target']).columns])
        
        if col_indices is not None:
            X_test_h_s = X_test_h_s[:, col_indices]
            
        probs = base_model.predict_proba(X_test_h_s)[:, 1]
        metrics = compute_all_metrics(y_test_h, probs)
        metrics['Target'] = target
        metrics['Horizon'] = f"{h}-day"
        lead_time_results.append(metrics)
        
    return lead_time_results

# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
all_metrics = []
calib_results = []
lead_time_res = []

X_cols = df.drop(columns=['date', 'flood_target', 'heatwave_target', 'compound_target']).columns

# Prepare train+val for stacking
train_val_mask = (df['date'].dt.year >= 2000) & (df['date'].dt.year <= 2015)
X_train_val = X[train_val_mask].copy()
X_train_val.fillna(imputation_means, inplace=True)
X_train_val_s = scaler.transform(X_train_val)

for target in targets:
    y_test = df[test_mask][target].values
    y_val = df[val_mask][target].values
    y_train_val = df[train_val_mask][target]
    
    print(f"--- Evaluating {target} ---")
    
    model_probs = {}
    
    fig_cm, axes_cm = plt.subplots(2, 4, figsize=(20, 10))
    fig_cm.suptitle(f'Confusion Matrices: {target}', fontsize=16)
    axes_flat = axes_cm.flatten()
    
    fig_cm_norm, axes_cm_norm = plt.subplots(2, 4, figsize=(20, 10))
    fig_cm_norm.suptitle(f'Normalized Confusion Matrices: {target}', fontsize=16)
    axes_flat_norm = axes_cm_norm.flatten()
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 8))
    
    plot_idx = 0
    
    # 1. Base Models
    for name in ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']:
        path = models_dir / f"{name}_{target}.joblib"
        if name == 'LogisticRegression':
            path = models_dir / f"LR_{target}.joblib"
        elif name == 'RandomForest':
            path = models_dir / f"RF_{target}.joblib"
        elif name == 'XGBoost':
            path = models_dir / f"XGB_{target}.joblib"
        elif name == 'LightGBM':
            path = models_dir / f"LGBM_{target}.joblib"
            
        if path.exists():
            if target == 'heatwave_target':
                restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
                col_indices = [X.columns.get_loc(c) for c in restricted_cols]
                X_test_t = X_test_s[:, col_indices]
            else:
                X_test_t = X_test_s
                
            model = joblib.load(path)
            probs = model.predict_proba(X_test_t)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_t)
            model_probs[name] = probs
            
            m = compute_all_metrics(y_test, probs)
            m['Target'] = target
            m['Model'] = name
            all_metrics.append(m)
            
            # ROC & PR
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {m["ROC-AUC"]:.3f})')
            prec, rec, _ = precision_recall_curve(y_test, probs)
            ax_pr.plot(rec, prec, label=f'{name} (PR-AUC = {m["PR-AUC"]:.3f})')
            
            # Confusion Matrix
            preds = (probs > 0.5).astype(int)
            cm = confusion_matrix(y_test, preds, labels=[0,1])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[plot_idx], cbar=False)
            axes_flat[plot_idx].set_title(name)
            cm_norm = confusion_matrix(y_test, preds, labels=[0,1], normalize='true')
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes_flat_norm[plot_idx], cbar=False)
            axes_flat_norm[plot_idx].set_title(name)
            plot_idx += 1
            
    for name in ['LSTM', 'GRU']:
        path = models_dir / f"{name}_{target}.pt"
        if path.exists():
            if target == 'heatwave_target':
                restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
                col_indices = [X.columns.get_loc(c) for c in restricted_cols]
                X_test_t = X_test_s[:, col_indices]
            else:
                X_test_t = X_test_s
                
            probs = dl_predict(path, name, X_test_t)
            model_probs[name] = probs
            
            m = compute_all_metrics(y_test, probs)
            m['Target'] = target
            m['Model'] = name
            all_metrics.append(m)
            
            # ROC & PR
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, probs)
                ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {m["ROC-AUC"]:.3f})')
                prec, rec, _ = precision_recall_curve(y_test, probs)
                ax_pr.plot(rec, prec, label=f'{name} (PR-AUC = {m["PR-AUC"]:.3f})')
                
            # Confusion Matrix
            preds = (probs > 0.5).astype(int)
            cm = confusion_matrix(y_test, preds, labels=[0,1])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[plot_idx], cbar=False)
            axes_flat[plot_idx].set_title(name)
            cm_norm = confusion_matrix(y_test, preds, labels=[0,1], normalize='true')
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes_flat_norm[plot_idx], cbar=False)
            axes_flat_norm[plot_idx].set_title(name)
            plot_idx += 1
            
    # 2. Stacking
    if y_train_val.sum() > 0:
        if target == 'heatwave_target':
            restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
            col_indices = [X.columns.get_loc(c) for c in restricted_cols]
            X_train_val_t = X_train_val_s[:, col_indices]
            X_test_t = X_test_s[:, col_indices]
        else:
            X_train_val_t = X_train_val_s
            X_test_t = X_test_s
            
        stack_probs, meta, xgb_base, lgbm_base = train_leakage_free_stack(target, X_train_val_t, y_train_val, X_test_t)
        model_probs['StackingEnsemble'] = stack_probs
        
        m = compute_all_metrics(y_test, stack_probs)
        m['Target'] = target
        m['Model'] = 'StackingEnsemble'
        all_metrics.append(m)
        
        # ROC & PR
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, stack_probs)
            ax_roc.plot(fpr, tpr, label=f'StackingEnsemble (AUC = {m["ROC-AUC"]:.3f})')
            prec, rec, _ = precision_recall_curve(y_test, stack_probs)
            ax_pr.plot(rec, prec, label=f'StackingEnsemble (PR-AUC = {m["PR-AUC"]:.3f})')
            
        # Confusion Matrix
        preds = (stack_probs > 0.5).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0,1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[plot_idx], cbar=False)
        axes_flat[plot_idx].set_title('StackingEnsemble')
        cm_norm = confusion_matrix(y_test, preds, labels=[0,1], normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes_flat_norm[plot_idx], cbar=False)
        axes_flat_norm[plot_idx].set_title('StackingEnsemble')
        plot_idx += 1
        
        # Finalize plots
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curves: {target}')
        ax_roc.legend()
        fig_roc.savefig(figures_dir / f'12_ROC_Curves_{target}.png', dpi=300)
        plt.close(fig_roc)
        
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'Precision-Recall Curves: {target}')
        ax_pr.legend()
        fig_pr.savefig(figures_dir / f'13_PR_Curves_{target}.png', dpi=300)
        plt.close(fig_pr)
        
        fig_cm.tight_layout()
        fig_cm.savefig(figures_dir / f'11_Confusion_Matrices_{target}.png', dpi=300)
        plt.close(fig_cm)
        
        fig_cm_norm.tight_layout()
        fig_cm_norm.savefig(figures_dir / f'11_Confusion_Matrices_Norm_{target}.png', dpi=300)
        plt.close(fig_cm_norm)
        
        # 3. Calibration on the Stack
        # We need validation predictions.
        if target == 'heatwave_target':
            restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
            col_indices = [X.columns.get_loc(c) for c in restricted_cols]
            X_val_t = X_val_s[:, col_indices]
        else:
            X_val_t = X_val_s
            
        meta_feat_val = np.column_stack([
            xgb_base.predict_proba(X_val_t)[:, 1],
            lgbm_base.predict_proba(X_val_t)[:, 1]
        ])
        probs_val = meta.predict_proba(meta_feat_val)[:, 1]
        c_res = evaluate_calibration(target, y_val, probs_val, y_test, stack_probs)
        calib_results.extend(c_res)
        
        # 4. Lead-Time on the best model (using XGB base for simplicity of single model interface)
        if target == 'heatwave_target':
            restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
            col_indices = [X.columns.get_loc(c) for c in restricted_cols]
            X_cols_t = restricted_cols
            X_test_t = X_test_s[:, col_indices]
        else:
            col_indices = None
            X_cols_t = X_cols
            X_test_t = X_test_s
            
        lt_res = evaluate_lead_time(target, xgb_base, df, X_cols_t, scaler, col_indices)
        lead_time_res.extend(lt_res)
        
        # 5. Explainability (SHAP)
        print("Generating SHAP for XGBoost...")
        explainer = shap.TreeExplainer(xgb_base)
        # Explain on a subset of test to save time
        X_test_df = pd.DataFrame(X_test_t[:100], columns=X_cols_t)
        shap_values = explainer.shap_values(X_test_df)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.title(f"SHAP Summary - {target}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"16_SHAP_Summary_{target}.png")
        plt.close()

# Save tables
pd.DataFrame(all_metrics).to_csv(results_dir / "MODEL_RESULTS_ADVANCED.csv", index=False)
pd.DataFrame(calib_results).to_csv(results_dir / "CALIBRATION_RESULTS.csv", index=False)
if lead_time_res:
    pd.DataFrame(lead_time_res).to_csv(results_dir / "LEAD_TIME_RESULTS.csv", index=False)

print("Main evaluation logic complete.")

# ---------------------------------------------------------
# Ablation Study
# ---------------------------------------------------------
print("Starting Ablation Study...")
ablation_results = []
climate_cols = [c for c in X_cols if any(x in c for x in ['temperature', 'dewpoint', 'pressure', 'wind', 'evaporation', 'u_component', 'v_component']) and 'lag' not in c]
hydro_cols = [c for c in X_cols if any(x in c for x in ['rainfall', 'runoff', 'soil_moisture']) and 'lag' not in c]
temporal_cols = ['month', 'day_of_year', 'season', 'monsoon_indicator']
lag_cols = [c for c in X_cols if 'lag' in c]

feature_sets = {
    'A_Climate_Only': climate_cols,
    'B_Hydro_Only': hydro_cols,
    'C_Climate_Hydro': climate_cols + hydro_cols,
    'D_Climate_Hydro_Temporal_Lag': climate_cols + hydro_cols + temporal_cols + lag_cols,
    'E_Full_Features': list(X_cols)
}

for target in targets:
    y_test = df[test_mask][target].values
    y_train_val = df[train_val_mask][target].values
    if y_train_val.sum() == 0: continue
    
    for fs_name, fs_cols in feature_sets.items():
        if target == 'heatwave_target':
            fs_cols = [c for c in fs_cols if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
            if len(fs_cols) == 0: continue
            
        X_tv_fs = df.loc[train_val_mask, fs_cols].copy()
        X_test_fs = df.loc[test_mask, fs_cols].copy()
        
        X_tv_fs.fillna(X_tv_fs.mean(), inplace=True)
        X_test_fs.fillna(X_tv_fs.mean(), inplace=True)
        
        sc = StandardScaler()
        X_tv_fs_s = sc.fit_transform(X_tv_fs)
        X_test_fs_s = sc.transform(X_test_fs)
        
        xgb = XGBClassifier(scale_pos_weight=(len(y_train_val)-y_train_val.sum())/(y_train_val.sum()+1e-5), random_state=42, eval_metric='logloss')
        xgb.fit(X_tv_fs_s, y_train_val)
        probs = xgb.predict_proba(X_test_fs_s)[:, 1]
        
        m = compute_all_metrics(y_test, probs)
        m['Target'] = target
        m['Experiment'] = fs_name
        ablation_results.append(m)

pd.DataFrame(ablation_results).to_csv(results_dir / "ABLATION_RESULTS.csv", index=False)


# ---------------------------------------------------------
# Required Figures Plotting (11, 12, 13, 17, 18)
# ---------------------------------------------------------
print("Generating required figures...")
# Load all_metrics again or use the dictionaries from the main loop
res_df = pd.DataFrame(all_metrics)

for target in targets:
    # Get model probabilities from the dictionaries created earlier (we'd need them stored, or we can just plot metrics)
    target_res = res_df[res_df['Target'] == target]
    if target_res.empty: continue
    
    # 10. Model Comparison
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    metrics_to_plot = ['F1-Score', 'ROC-AUC', 'PR-AUC']
    for i, m in enumerate(metrics_to_plot):
        sns.barplot(data=target_res, x='Model', y=m, ax=axes[i])
        axes[i].set_title(f"{m} Comparison - {target}")
        axes[i].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figures_dir / f"10_Model_Comparison_{target}.png")
    plt.close()

# 17. Lead-Time Performance Graph
if lead_time_res:
    lt_df = pd.DataFrame(lead_time_res)
    for target in targets:
        target_lt = lt_df[lt_df['Target'] == target]
        if target_lt.empty: continue
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(target_lt['Horizon'], target_lt['F1-Score'], marker='o', label='F1-Score')
        ax.plot(target_lt['Horizon'], target_lt['ROC-AUC'], marker='s', label='ROC-AUC')
        ax.plot(target_lt['Horizon'], target_lt['PR-AUC'], marker='^', label='PR-AUC')
        ax.set_title(f"Lead-Time Performance Decay - {target} (XGBoost)")
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"17_Lead_Time_Performance_{target}.png")
        plt.close()

# 18. Ablation Study
ab_df = pd.DataFrame(ablation_results)
for target in targets:
    target_ab = ab_df[ab_df['Target'] == target]
    if target_ab.empty: continue
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=target_ab, y='Experiment', x='F1-Score', ax=ax, orient='h')
    ax.set_title(f"Ablation Study (F1-Score) - {target}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"18_Ablation_Study_{target}.png")
    plt.close()

# 19 & 20. Probability Timeline and Temporal Prediction Timeline
for target in targets:
    y_test = df[test_mask][target].values
    if len(y_test) == 0: continue
    path = models_dir / f"XGB_{target}.joblib"
    if path.exists():
        if target == 'heatwave_target':
            restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
            col_indices = [X.columns.get_loc(c) for c in restricted_cols]
            X_test_t = X_test_s[:, col_indices]
        else:
            X_test_t = X_test_s
            
        xgb = joblib.load(path)
        probs = xgb.predict_proba(X_test_t)[:, 1]
        
        # 19. Probability Timeline
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df.loc[test_mask, 'date'], probs, label='Predicted Probability', alpha=0.7)
        actuals_idx = df.loc[test_mask, 'date'][y_test == 1]
        ax.scatter(actuals_idx, [1]*len(actuals_idx), color='red', label='Actual Event', zorder=5)
        ax.set_title(f"Probability Timeline - {target} (Test Set 2016-2018)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"19_Probability_Timeline_{target}.png")
        plt.close()
        
        # 20. Temporal prediction timeline (Zoomed in on the first event)
        fig, ax = plt.subplots(figsize=(14, 4))
        if len(actuals_idx) > 0:
            first_event = actuals_idx.iloc[0]
            start_date = first_event - pd.Timedelta(days=30)
            end_date = first_event + pd.Timedelta(days=30)
            
            mask_zoom = (df.loc[test_mask, 'date'] >= start_date) & (df.loc[test_mask, 'date'] <= end_date)
            ax.plot(df.loc[test_mask, 'date'][mask_zoom], probs[mask_zoom], label='Predicted Probability', alpha=0.7)
            ax.scatter(df.loc[test_mask, 'date'][(y_test == 1) & mask_zoom], 
                       [1]*sum((y_test == 1) & mask_zoom), color='red', label='Actual Event', zorder=5)
            ax.set_title(f"Temporal Prediction Timeline (Zoomed) - {target}")
        else:
            ax.plot(df.loc[test_mask, 'date'], probs, label='Predicted Probability')
            ax.set_title(f"Temporal Prediction Timeline (No Events) - {target}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"20_Temporal_Prediction_Timeline_{target}.png")
        plt.close()

# ---------------------------------------------------------
# Statistical Comparison & Final Selection
# ---------------------------------------------------------
print("Running Statistical Comparisons...")
statistical_results = []
res_df = pd.DataFrame(all_metrics)

def mcnemar_pvalue(y_true, p1, p2):
    pr1 = (p1 > 0.5).astype(int)
    pr2 = (p2 > 0.5).astype(int)
    b = np.sum((pr1 == y_true) & (pr2 != y_true))
    c = np.sum((pr1 != y_true) & (pr2 == y_true))
    if b + c == 0: return 1.0
    chi2 = ((abs(b - c) - 1.0)**2) / (b + c)
    return 1.0 - chi2_dist.cdf(chi2, 1)

final_selection_md = ["# Final Model Selection\n"]

for target in targets:
    target_res = res_df[res_df['Target'] == target].copy()
    if target_res.empty: continue
    
    # Sort models by a combination of F1 and ROC-AUC
    target_res['Score'] = (target_res['F1-Score'] + target_res['ROC-AUC']) / 2
    target_res = target_res.sort_values(by='Score', ascending=False)
    
    best_model = target_res.iloc[0]['Model']
    second_best = target_res.iloc[1]['Model'] if len(target_res) > 1 else None
    
    final_selection_md.append(f"## Target: {target}")
    final_selection_md.append(f"**Selected Model:** {best_model}")
    final_selection_md.append(f"\n### Justification:")
    final_selection_md.append(f"- Highest combined F1 and ROC-AUC score.")
    final_selection_md.append(f"- F1-Score: {target_res.iloc[0]['F1-Score']:.4f}")
    final_selection_md.append(f"- ROC-AUC: {target_res.iloc[0]['ROC-AUC']:.4f}")
    final_selection_md.append(f"- PR-AUC: {target_res.iloc[0]['PR-AUC']:.4f}")
    final_selection_md.append(f"- Recall: {target_res.iloc[0]['Recall']:.4f}")
    
    if second_best:
        y_test = df[test_mask][target].values
        # Need predictions. In a real scenario we'd have them stored.
        # Since this script runs fast, we will load them.
        def get_preds(m_name, t):
            if m_name == 'StackingEnsemble': return None # Skip stack vs stack
            path = models_dir / f"{m_name.replace('LogisticRegression', 'LR').replace('RandomForest', 'RF').replace('XGBoost', 'XGB').replace('LightGBM', 'LGBM')}_{t}"
            
            if t == 'heatwave_target':
                restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
                col_indices = [X.columns.get_loc(c) for c in restricted_cols]
                X_test_t = X_test_s[:, col_indices]
            else:
                X_test_t = X_test_s
                
            if m_name in ['LSTM', 'GRU']:
                return dl_predict(path.with_suffix('.pt'), m_name, X_test_t)
            else:
                if not path.with_suffix('.joblib').exists(): return None
                return joblib.load(path.with_suffix('.joblib')).predict_proba(X_test_t)[:, 1]
                
        p1 = get_preds(best_model, target)
        p2 = get_preds(second_best, target)
        
        if p1 is not None and p2 is not None:
            pval = mcnemar_pvalue(y_test, p1, p2)
            statistical_results.append({
                'Target': target,
                'Model_A': best_model,
                'Model_B': second_best,
                'McNemar_p-value': pval
            })
            final_selection_md.append(f"\n### Statistical Significance:")
            final_selection_md.append(f"McNemar's test between {best_model} and {second_best} yielded a p-value of {pval:.4f}.")
            if pval < 0.05:
                final_selection_md.append("This difference is statistically significant (p < 0.05).")
            else:
                final_selection_md.append("This difference is NOT statistically significant (p >= 0.05).")
    final_selection_md.append("\n---\n")

pd.DataFrame(statistical_results).to_csv(results_dir / "STATISTICAL_RESULTS.csv", index=False)
with open(results_dir / "FINAL_MODEL_SELECTION.md", "w") as f:
    f.write("\n".join(final_selection_md))

print("All advanced evaluation tasks completed.")

