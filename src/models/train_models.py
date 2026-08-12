import os
import shutil
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, roc_auc_score, average_precision_score, 
    matthews_corrcoef, brier_score_loss, confusion_matrix
)
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Deterministic seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def evaluate_model(model, X_test, y_test, name, target):
    # Predict classes and probabilities
    preds = model.predict(X_test)
    
    # Handle neural network probabilities correctly
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "forward"):
        # For our custom RNN, we should handle this differently but we'll assume the caller passes preds and probs directly
        # or we just skip this part for RNNs and handle it externally. 
        # Actually, let's keep the model-agnostic code here if possible.
        pass
    else:
        probs = preds

    return calculate_metrics(y_test, preds, probs, name, target)

def calculate_metrics(y_test, preds, probs, name, target):
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, preds)
    
    cm = confusion_matrix(y_test, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        specificity = 0
        sensitivity = rec
        
    roc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0
    pr_auc = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0
    mcc = matthews_corrcoef(y_test, preds)
    brier = brier_score_loss(y_test, probs)

    print("\n============================================================")
    print("CLIMATEGUARDIAN AI MODEL EVALUATION")
    print("============================================================")
    print(f"Target: {target}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print("============================================================\n")

    return {
        "Target": target,
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Balanced Accuracy": bal_acc,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "ROC-AUC": roc,
        "PR-AUC": pr_auc,
        "MCC": mcc,
        "Brier Score": brier
    }

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
        out = self.fc(out)
        return out

def create_sequences(X, y, seq_length=14):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y.iloc[i + seq_length])
    return torch.tensor(np.array(Xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)

def main():
    run_id = os.environ.get("EXPERIMENT_RUN_ID", "latest")
    results_dir = Path(f"results/experiment_runs/{run_id}")
    models_dir = Path(f"results/experiment_runs/{run_id}/models")
    reports_dir = Path("08_Reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading Features and Labels...")
    try:
        features = pd.read_csv("03_Features/master_features.csv")
        features['date'] = pd.to_datetime(features['date'])

        flood_lbl = pd.read_csv("04_Labels/flood_labels.csv")
        flood_lbl['date'] = pd.to_datetime(flood_lbl['date'])

        heatwave_lbl = pd.read_csv("04_Labels/heatwave_labels.csv")
        heatwave_lbl['date'] = pd.to_datetime(heatwave_lbl['date'])

        compound_lbl = pd.read_csv("04_Labels/compound_labels.csv")
        compound_lbl['date'] = pd.to_datetime(compound_lbl['date'])
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    df = features.merge(flood_lbl[['date', 'flood_target']], on='date')
    df = df.merge(heatwave_lbl[['date', 'heatwave_target']], on='date')
    df = df.merge(compound_lbl[['date', 'compound_target']], on='date')
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Dataset Shape: {df.shape}")

    # Chronological Split
    train_mask = (df['date'].dt.year >= 2000) & (df['date'].dt.year <= 2012)
    val_mask = (df['date'].dt.year >= 2013) & (df['date'].dt.year <= 2015)
    test_mask = (df['date'].dt.year >= 2016) & (df['date'].dt.year <= 2018)

    X = df.drop(columns=['date', 'flood_target', 'heatwave_target', 'compound_target'])
    targets = ['flood_target', 'heatwave_target', 'compound_target']

    X = X.ffill() 

    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    X_test = X[test_mask].copy()

    imputation_means = X_train.mean()
    X_train.fillna(imputation_means, inplace=True)
    X_val.fillna(imputation_means, inplace=True)
    X_test.fillna(imputation_means, inplace=True)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    joblib.dump(imputation_means, models_dir / "imputation_means.joblib")
    joblib.dump(X_train.columns.tolist(), models_dir / "feature_cols.joblib")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, models_dir / "scaler.joblib")

    results = []
    param_dist_rf = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    param_dist_xgb = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    param_dist_lgb = {'n_estimators': [50, 100], 'num_leaves': [15, 31], 'learning_rate': [0.01, 0.1]}
    tscv = TimeSeriesSplit(n_splits=3)

    for target in targets:
        y_train = df[train_mask][target]
        y_val = df[val_mask][target]
        y_test = df[test_mask][target]
        
        if y_train.sum() == 0:
            print(f"Skipping ML {target}, 0 positive samples in Train.")
            continue
            
        cw = 'balanced'
        scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)

        print(f"\nTraining Traditional ML for {target}...")

        if target == 'heatwave_target':
            print(f"Applying RESTRICTED feature set for {target} to avoid deterministic target encoding.")
            restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
            col_indices = [X.columns.get_loc(c) for c in restricted_cols]
            X_train_t = X_train_s[:, col_indices]
            X_test_t = X_test_s[:, col_indices]
        else:
            X_train_t = X_train_s
            X_test_t = X_test_s

        try:
            # Logistic Regression
            lr = LogisticRegression(class_weight=cw, max_iter=1000)
            lr.fit(X_train_t, y_train)
            results.append(evaluate_model(lr, X_test_t, y_test, "LogisticRegression", target))
            joblib.dump(lr, models_dir / f"LR_{target}.joblib")

            # Random Forest
            rf = RandomForestClassifier(class_weight=cw, random_state=42)
            search_rf = RandomizedSearchCV(rf, param_dist_rf, n_iter=2, cv=tscv, scoring='f1', n_jobs=1, random_state=42)
            search_rf.fit(X_train_t, y_train)
            results.append(evaluate_model(search_rf.best_estimator_, X_test_t, y_test, "RandomForest", target))
            joblib.dump(search_rf.best_estimator_, models_dir / f"RF_{target}.joblib")

            # XGBoost
            xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
            search_xgb = RandomizedSearchCV(xgb, param_dist_xgb, n_iter=2, cv=tscv, scoring='f1', n_jobs=1, random_state=42)
            search_xgb.fit(X_train_t, y_train)
            results.append(evaluate_model(search_xgb.best_estimator_, X_test_t, y_test, "XGBoost", target))
            joblib.dump(search_xgb.best_estimator_, models_dir / f"XGB_{target}.joblib")

            # LightGBM
            lgb = LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1)
            search_lgb = RandomizedSearchCV(lgb, param_dist_lgb, n_iter=2, cv=tscv, scoring='f1', n_jobs=1, random_state=42)
            search_lgb.fit(X_train_t, y_train)
            results.append(evaluate_model(search_lgb.best_estimator_, X_test_t, y_test, "LightGBM", target))
            joblib.dump(search_lgb.best_estimator_, models_dir / f"LGBM_{target}.joblib")
            
            # Compound Stacking Ensemble
            if target == 'compound_target':
                print(f"Training Compound Stacking Ensemble for {target}...")
                estimators = [
                    ('xgb', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', max_depth=3, n_estimators=50)),
                    ('lgb', LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1, num_leaves=15, n_estimators=50))
                ]
                # Note: TimeSeriesSplit fails cross_val_predict with imbalanced compound target.
                # cv=5 generates out-of-fold base-model probabilities for the meta-learner.
                # Final evaluation is strictly on the held-out 2016-2018 test set.
                stack = StackingClassifier(
                    estimators=estimators, 
                    final_estimator=LogisticRegression(class_weight='balanced'), 
                    cv=5
                )
                stack.fit(X_train_t, y_train)
                results.append(evaluate_model(stack, X_test_t, y_test, "Stacking Ensemble", target))
                joblib.dump(stack, models_dir / f"Stacking_{target}.joblib")

        except Exception as e:
            print(f"\nTarget: {target}")
            print(f"Model: Traditional ML / Stacking")
            print(f"ERROR: {e}\n")

    # Deep Learning Models (LSTM & GRU)
    print("\nTraining Deep Learning Models...")
    seq_len = 14
    hidden_dim = 32

    for target in targets:
        y_train = df[train_mask][target]
        y_val = df[val_mask][target]
        y_test = df[test_mask][target]
        
        if y_train.sum() == 0:
            continue

        if target == 'heatwave_target':
            restricted_cols = [c for c in X.columns if not ('temperature_max' in c or 'hw_rolling' in c or 'hw_exceed' in c)]
            col_indices = [X.columns.get_loc(c) for c in restricted_cols]
            X_train_t = X_train_s[:, col_indices]
            X_val_t = X_val_s[:, col_indices]
            X_test_t = X_test_s[:, col_indices]
            input_dim = len(col_indices)
        else:
            X_train_t = X_train_s
            X_val_t = X_val_s
            X_test_t = X_test_s
            input_dim = X_train_s.shape[1]
            
        X_train_seq, y_train_seq = create_sequences(X_train_t, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val_t, y_val, seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test_t, y_test, seq_len)
        
        train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=32, shuffle=False)
        
        for rnn_type in ['LSTM', 'GRU']:
            print(f"Training {rnn_type} for {target}...")
            try:
                model = RNNModel(input_dim, hidden_dim, rnn_type)
                pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)])
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                
                best_val_loss = float('inf')
                patience = 5
                patience_counter = 0
                
                for epoch in range(15):
                    model.train()
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        out = model(batch_x).squeeze()
                        loss = criterion(out, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                    model.eval()
                    with torch.no_grad():
                        val_out = model(X_val_seq).squeeze()
                        val_loss = criterion(val_out, y_val_seq).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), models_dir / f"{rnn_type}_{target}.pt")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
                
                model.load_state_dict(torch.load(models_dir / f"{rnn_type}_{target}.pt", weights_only=True))
                model.eval()
                with torch.no_grad():
                    test_logits = model(X_test_seq).squeeze()
                    test_probs = torch.sigmoid(test_logits).numpy()
                    test_preds = (test_probs > 0.5).astype(int)
                    
                    res = calculate_metrics(y_test_seq.numpy(), test_preds, test_probs, rnn_type, target)
                    results.append(res)
            except Exception as e:
                print(f"\nTarget: {target}")
                print(f"Model: {rnn_type}")
                print(f"ERROR: {e}\n")

    res_df = pd.DataFrame(results)
    csv_path = results_dir / "MODEL_RESULTS.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nFinal model evaluation results saved to {csv_path}")
    
    # Resolve the paper evidence path missing file error if directory exists
    evidence_dir = Path("IEEE_PAPER_EVIDENCE/04_METRICS")
    if evidence_dir.exists():
        evidence_csv_path = evidence_dir / "MODEL_RESULTS.csv"
        shutil.copy(csv_path, evidence_csv_path)
        print(f"Copied final results to canonical evidence path: {evidence_csv_path}")

if __name__ == "__main__":
    main()
