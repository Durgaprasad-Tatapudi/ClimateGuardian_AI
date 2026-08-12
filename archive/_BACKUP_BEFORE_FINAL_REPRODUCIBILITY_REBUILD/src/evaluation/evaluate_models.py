import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

models_dir = Path("05_Models")
results_dir = Path("06_Results")
figures_dir = Path("07_Figures")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

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

test_mask = (df['date'].dt.year >= 2016) & (df['date'].dt.year <= 2018)
train_mask = (df['date'].dt.year >= 2000) & (df['date'].dt.year <= 2012)

X = df.drop(columns=['date', 'flood_target', 'heatwave_target', 'compound_target'])
X = X.ffill().bfill() 

X_train = X[train_mask]
X_test = X[test_mask]
targets = ['flood_target', 'heatwave_target', 'compound_target']

scaler = StandardScaler()
scaler.fit(X_train)
X_test_s = scaler.transform(X_test)

def dl_predict(model_path, rnn_type, X_test_s):
    import torch.nn as nn
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
    model = RNNModel(X_test_s.shape[1], 32, rnn_type)
    if not os.path.exists(model_path): return None
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    Xs = []
    for i in range(len(X_test_s) - seq_len):
        Xs.append(X_test_s[i:(i + seq_len)])
    
    with torch.no_grad():
        tensor_Xs = torch.tensor(np.array(Xs), dtype=torch.float32)
        logits = model(tensor_Xs).squeeze()
        probs = torch.sigmoid(logits).numpy()
    
    probs = np.pad(probs, (seq_len, 0), mode='constant', constant_values=0)
    return probs

def specificity_score(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        return 0.0

results_records = []

for target in targets:
    print(f"Evaluating models for {target}...")
    y_test = df[test_mask][target].values
    
    models = {
        'LogisticRegression': f'LR_{target}.joblib',
        'RandomForest': f'RF_{target}.joblib',
        'XGBoost': f'XGB_{target}.joblib',
        'LightGBM': f'LGBM_{target}.joblib',
        'LSTM': f'LSTM_{target}.pt',
        'GRU': f'GRU_{target}.pt'
    }
    
    # 11: Confusion matrices (raw and normalized)
    fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 10))
    fig_cm.suptitle(f'Confusion Matrices: {target}', fontsize=16)
    axes_flat = axes_cm.flatten()
    
    fig_cm_norm, axes_cm_norm = plt.subplots(2, 3, figsize=(15, 10))
    fig_cm_norm.suptitle(f'Normalized Confusion Matrices: {target}', fontsize=16)
    axes_flat_norm = axes_cm_norm.flatten()
    
    # 12 & 13: ROC and PR Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 8))
    
    for i, (name, filename) in enumerate(models.items()):
        path = models_dir / filename
        if not path.exists():
            continue
            
        if name in ['LSTM', 'GRU']:
            probs = dl_predict(path, name, X_test_s)
        else:
            model = joblib.load(path)
            probs = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_s)
            
        preds = (probs > 0.5).astype(int)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, preds)
        spec = specificity_score(y_test, preds)
        sens = rec
        
        try:
            roc = roc_auc_score(y_test, probs)
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc:.3f})')
        except ValueError:
            roc = 0
            
        brier = brier_score_loss(y_test, probs)
        mcc = matthews_corrcoef(y_test, preds)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall_curve, precision_curve)
        ax_pr.plot(recall_curve, precision_curve, label=f'{name} (PR-AUC = {pr_auc:.3f})')
        
        results_records.append({
            'Target': target, 'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'Balanced Accuracy': bal_acc,
            'Specificity': spec, 'Sensitivity': sens, 'ROC-AUC': roc,
            'PR-AUC': pr_auc, 'MCC': mcc, 'Brier Score': brier
        })
        
        cm = confusion_matrix(y_test, preds, labels=[0,1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[i], cbar=False)
        axes_flat[i].set_title(name)
        axes_flat[i].set_xlabel('Predicted')
        axes_flat[i].set_ylabel('Actual')
        
        cm_norm = confusion_matrix(y_test, preds, labels=[0,1], normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes_flat_norm[i], cbar=False)
        axes_flat_norm[i].set_title(name)
        axes_flat_norm[i].set_xlabel('Predicted')
        axes_flat_norm[i].set_ylabel('Actual')
        
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curves: {target}')
    ax_roc.legend(loc='lower right')
    fig_roc.tight_layout()
    fig_roc.savefig(figures_dir / f'12_ROC_Curves_{target}.png', dpi=300)
    plt.close(fig_roc)
    
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Precision-Recall Curves: {target}')
    ax_pr.legend(loc='lower left')
    fig_pr.tight_layout()
    fig_pr.savefig(figures_dir / f'13_PR_Curves_{target}.png', dpi=300)
    plt.close(fig_pr)
    
    fig_cm.tight_layout()
    fig_cm.savefig(figures_dir / f'11_Confusion_Matrices_{target}.png', dpi=300)
    plt.close(fig_cm)
    
    fig_cm_norm.tight_layout()
    fig_cm_norm.savefig(figures_dir / f'11_Confusion_Matrices_Norm_{target}.png', dpi=300)
    plt.close(fig_cm_norm)

results_df = pd.DataFrame(results_records)
results_df.to_csv(results_dir / "MODEL_RESULTS.csv", index=False)

# 10: Model comparison (Bar Chart)
print("Generating Model Comparison Bar Chart...")
fig_comp, axes_comp = plt.subplots(3, 1, figsize=(12, 15))
for i, target in enumerate(targets):
    target_data = results_df[results_df['Target'] == target]
    target_data = target_data.melt(id_vars='Model', value_vars=['F1-Score', 'ROC-AUC', 'PR-AUC'])
    sns.barplot(data=target_data, x='Model', y='value', hue='variable', ax=axes_comp[i])
    axes_comp[i].set_title(f'Model Comparison: {target}')
    axes_comp[i].set_ylim(0, 1)
fig_comp.tight_layout()
fig_comp.savefig(figures_dir / '10_Model_Comparison.png', dpi=300)
plt.close(fig_comp)

print("Evaluation Complete.")
