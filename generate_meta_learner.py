import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path

models_dir = Path("05_Models_corrected")

# Load Data
features = pd.read_csv("03_Features/master_features.csv")
features['date'] = pd.to_datetime(features['date'])
compound_lbl = pd.read_csv("04_Labels/compound_labels.csv")
compound_lbl['date'] = pd.to_datetime(compound_lbl['date'])

df = features.merge(compound_lbl[['date', 'compound_target']], on='date')
df = df.sort_values('date').reset_index(drop=True)

# Splits
train_val_mask = (df['date'].dt.year >= 2000) & (df['date'].dt.year <= 2015)

X = df.drop(columns=['date', 'compound_target'])
X = X.ffill()

X_train_val = X[train_val_mask].copy()

# The imputation means should come from the train set (2000-2012)
train_mask = (df['date'].dt.year >= 2000) & (df['date'].dt.year <= 2012)
X_train = X[train_mask].copy()
imputation_means = X_train.mean()

X_train_val.fillna(imputation_means, inplace=True)

scaler = StandardScaler()
# Scaler is fitted on train set
X_train.fillna(imputation_means, inplace=True)
scaler.fit(X_train)

X_train_val_s = scaler.transform(X_train_val)
y_train_val = df[train_val_mask]['compound_target']

print("Training Stack for compound_target...")
tscv = TimeSeriesSplit(n_splits=3)

# Base models to use for stack
xgb = XGBClassifier(scale_pos_weight=(len(y_train_val)-y_train_val.sum())/(y_train_val.sum()+1e-5), random_state=42, eval_metric='logloss')
lgbm = LGBMClassifier(scale_pos_weight=(len(y_train_val)-y_train_val.sum())/(y_train_val.sum()+1e-5), random_state=42, verbose=-1)

meta_features_train = np.zeros((len(y_train_val), 2))

for train_index, val_index in tscv.split(X_train_val_s):
    X_fold_train, X_fold_val = X_train_val_s[train_index], X_train_val_s[val_index]
    y_fold_train, y_fold_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
    
    xgb.fit(X_fold_train, y_fold_train)
    meta_features_train[val_index, 0] = xgb.predict_proba(X_fold_val)[:, 1]
    
    lgbm.fit(X_fold_train, y_fold_train)
    meta_features_train[val_index, 1] = lgbm.predict_proba(X_fold_val)[:, 1]

# Drop rows where meta_features are 0 (the first fold train set)
valid_idx = np.where(meta_features_train.sum(axis=1) > 0)[0]

meta_learner = LogisticRegression(class_weight='balanced')
if len(np.unique(y_train_val.iloc[valid_idx])) > 1:
    meta_learner.fit(meta_features_train[valid_idx], y_train_val.iloc[valid_idx])
else:
    meta_learner.fit(meta_features_train, y_train_val) # Fallback

# Verify shape
print("Meta-learner n_features_in_:", meta_learner.n_features_in_)

# Dump artifact
out_path = models_dir / "LR_meta_compound_target.joblib"
joblib.dump(meta_learner, out_path)
print(f"Saved meta-learner to {out_path}")
