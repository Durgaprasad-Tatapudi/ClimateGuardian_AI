# Model Configurations

## Flood (RandomForest)
- n_estimators=100
- max_depth=5
- class_weight=balanced

## Heatwave (LightGBM)
- 41 input features (proxies removed)
- learning_rate=0.1
- max_depth=-1
- scale_pos_weight derived

## Compound Operational (Stacking)
- Base 1: XGBoost (lr=0.01, max_depth=3)
- Base 2: LightGBM (lr=0.1, max_depth=-1)
- Meta: LogisticRegression (C=1.0, class_weight='balanced')
- Meta input shape: 2

## Compound Offline Benchmark (GRU)
- 14-day sequence, 45 features
- Used strictly as offline comparison.
